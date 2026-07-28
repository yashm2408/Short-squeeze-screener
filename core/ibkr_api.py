"""
IBKR data, two channels:

1. Borrow data (CTB + shortable shares). Two sources, tried in order:
     a. iborrowdesk.com over HTTPS — a public mirror of IBKR's own
        ftp3.interactivebrokers.com shortstock file (that FTP host is
        blocked on this network — confirmed via direct connection test).
        Gives BOTH cost-to-borrow and shortable shares when it's up.
     b. IB Gateway, two free calls per ticker, no market data subscription:
          * CTB       -- reqHistoricalData(whatToShow="FEE_RATE"). This is a
                         DIFFERENT permission path than the borrow-fee market
                         data TICK (which does need a paid subscription and
                         returns Error 10089), which is why it works here.
                         IBKR returns it as a decimal fraction: 0.0025 = 0.25%.
                         Verified against iborrowdesk's last known AAPL
                         reading of 0.2513% -- IBKR gives 0.0025 for AAPL.
          * Shortable -- generic tick 236.
   (a) is tried first since it is a single request for both fields; (b) is
   the fallback, and as of this writing (a) has been down for many days
   straight (its backend crashes on every ticker lookup), so (b) is in
   practice the live path for both values.

2. Historical daily bars for the TTM Squeeze — IB Gateway on port 4001
   via ib_insync (asyncio loop in a daemon thread).

All public getters are non-blocking cache reads; fetches happen in the
controller's background thread.
"""
import asyncio
import math
import threading
import time
from concurrent import futures

import requests
from ib_insync import IB, Stock

_borrow_cache = {}            # ticker -> {"ctb", "shortable", "_ts"}
_bars_cache = {}              # ticker -> {"bars", "_ts"}
_contract_cache = {}          # ticker -> qualified Contract (qualifying is slow)

# --- IBKR historical-data pacing budget ---------------------------------
# IBKR cancels requests (Error 162 / "query cancelled") past roughly 60
# historical-data requests in any rolling 10 minutes. BOTH the CTB fee-rate
# call and the TTM bars call go through reqHistoricalData, so they share one
# budget: ~2 requests per ticker. At 40+ tickers a naive burst is ~80
# requests and blows the limit outright, which cancels EVERYTHING — that is
# what wiped out CTB, Shortable and TTM at once. A token bucket below meters
# every historical request through one shared limiter.
_HIST_WINDOW = 600.0          # IBKR's pacing window: ~60 requests per 10 min
# FEE_RATE (CTB) and TRADES (TTM bars) get SEPARATE budgets, not one shared
# pool, because they are not equally valuable and not equally reliable:
#   * TRADES bars exist for essentially every real listed ticker and drive a
#     visible column (TTM Squeeze). They must never be starved.
#   * FEE_RATE has no stock-loan feed at all for most of this screener's
#     thin/OTC candidates -- IBKR simply never answers those requests.
# With ONE shared bucket, the borrow pass (which runs first and covers every
# ticker) drained the whole budget on requests that return nothing, leaving
# bars waiting ~11s per token for a refill. fetch_bars gave up at its own
# timeout, cached a failure, and TTM stayed blank forever. Split budgets make
# that impossible: the fee lane can only ever exhaust its own share.
# Burst capacity vs sustained rate (see _HistLimiter):
#   * BURST covers a cold start -- one screener pool's worth of tickers in a
#     single wave. Bars measured 36/36 in 18s when the burst covers the pool,
#     but 125s when it didn't and requests queued on refills. TTM is the
#     visible column, so its burst must comfortably cover a full pool.
#   * SUSTAINED is the long-run ceiling. In steady state bars re-fetch only
#     every 4 h and fees every 30 min, so actual ongoing draw is far below
#     this; 20 + 30 = 50 per 10 min keeps a real margin under IBKR's ~60.
# Verified empirically: two full runs at ~72 requests apiece with these
# concurrency caps produced zero pacing errors (162/420/100).
_BARS_BURST = 45              # TTM bars -- covers a full candidate pool at once
_BARS_BUDGET = 20             # ...sustained
_FEE_BURST = 35               # CTB
_FEE_BUDGET = 30              # ...sustained (the recurring consumer)

# Separate concurrency lanes for the same reason (a pile of hung FEE_RATE
# requests must not occupy the slots bars need).
_HIST_MAX_CONCURRENT_FEE = 2
_HIST_MAX_CONCURRENT_BARS = 3

# A real FEE_RATE quote measured under 1s round-trip when the data exists, so
# prolonged silence means "no data", not "still coming" -- fail fast to free
# the lane. TRADES bars legitimately take a bit longer, so they keep a longer
# leash.
_FEE_RATE_TIMEOUT = 8
_BARS_TIMEOUT = 25

# Once IBKR has shown it has no lending feed for a ticker, re-asking every
# 30 min just re-burns the scarce fee budget on a guaranteed miss. A name
# without a stock-loan quote today will not grow one today.
# But IBKR is also plain flaky: liquid names (MSFT/TSLA/GME) answered on one
# run and timed out on the next. Benching those for hours over a single blip
# would be worse than the wasted budget, so a ticker is only written off after
# repeated misses; any success clears the record.
_FEE_DEAD_TTL = 21600         # 6 h
_FEE_DEAD_STRIKES = 2         # consecutive misses before we stop asking
_fee_dead = {}                # ticker -> ts of the strike that benched it
_fee_strikes = {}             # ticker -> consecutive miss count

# Smaller chunk = tighter worst-case timeout bound (see fetch_bars_batch): at
# concurrency 3, chunk 6 bounds a chunk to 2 waves instead of 4.
_BARS_BATCH_SIZE = 6

# Daily bars only change once a day, so re-pulling them every 30 min was
# pure waste of a scarce budget. 4h keeps them fresh enough while freeing
# most of the budget for the fee-rate calls, which do move intraday.
_BORROW_TTL = 1800            # 30 min
_BARS_TTL = 14400             # 4 h
# A failed fetch must NOT be cached as long as a good one, or one pacing
# blip freezes a ticker's column at "—" for the whole TTL.
_FAIL_RETRY_TTL = 240         # 4 min

_IBORROW_URL = "https://iborrowdesk.com/api/ticker/{ticker}"
# Thin/illiquid tickers (which is most of this screener's candidate pool)
# take noticeably longer for tick 236 to arrive than large caps do, and a
# bigger simultaneous batch slows every tick in it down further — measured
# empirically: wait=2.0s/batch=25 -> ~65% coverage, wait=4.0s/batch=15 -> ~67%
# coverage in under a fifth of the time. Some names never report a shortable
# tick at all regardless of wait (no lending-desk data flowing for them via
# this free channel) — that's a real data-availability gap, not a bug.
_SHORTABLE_TICK_WAIT = 4.0

_ib = IB()
_loop = asyncio.new_event_loop()
_loop_started = threading.Event()


def _run_loop():
    asyncio.set_event_loop(_loop)
    _loop_started.set()
    _loop.run_forever()


threading.Thread(target=_run_loop, daemon=True, name="ibkr-loop").start()


class _HistLimiter:
    """Pacing gate for reqHistoricalData calls. One instance per request kind.

    Token bucket: starts full so a cold start can burst up to its budget
    immediately, then refills at budget/window. That gives fast warm-up
    without ever crossing the line that makes IBKR cancel requests.
    """

    def __init__(self, budget, window=_HIST_WINDOW, name="", capacity=None):
        # capacity (burst) is deliberately separable from budget (sustained
        # refill). A cold start wants one big burst — every ticker at once —
        # while the long-run average must stay under IBKR's ceiling. Tying
        # the two together forced a choice between a slow cold start and an
        # unsafe sustained rate; splitting them gives both.
        self.name = name
        capacity = float(capacity if capacity is not None else budget)
        self._tokens = capacity
        self._budget = capacity               # bucket ceiling
        self._rate = budget / window          # tokens per second (sustained)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()
        self.waits = 0                        # diagnostics

    async def acquire(self):
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(self._budget,
                                   self._tokens + (now - self._updated) * self._rate)
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                need = (1.0 - self._tokens) / self._rate
            self.waits += 1
            await asyncio.sleep(min(need, 5.0))

    def snapshot(self):
        return {"tokens_left": round(self._tokens, 1), "throttled_waits": self.waits}


_bars_limiter = _HistLimiter(_BARS_BUDGET, name="bars", capacity=_BARS_BURST)
_fee_limiter = _HistLimiter(_FEE_BUDGET, name="fee", capacity=_FEE_BURST)
_slots_fee = asyncio.Semaphore(_HIST_MAX_CONCURRENT_FEE)
_slots_bars = asyncio.Semaphore(_HIST_MAX_CONCURRENT_BARS)


async def _paced_hist(contract, *, limiter, slots, timeout, **kwargs):
    """Every historical request in this module goes through here.

    IBKR occasionally never answers a historical-data request for thin/OTC
    tickers -- no error callback, no data, it just never completes. Since
    that await sits inside a concurrency-slot semaphore, one hung ticker
    would otherwise permanently occupy a slot; enough of those over time
    wedges the pool and stalls every later request of that same kind.
    wait_for bounds the wait so the slot always gets released. FEE_RATE and
    TRADES use separate budgets AND separate slot pools (see constants above)
    so one kind can never starve the other.
    """
    await limiter.acquire()
    async with slots:
        try:
            return await asyncio.wait_for(
                _ib.reqHistoricalDataAsync(contract, **kwargs),
                timeout=timeout)
        except asyncio.TimeoutError:
            what = kwargs.get("whatToShow", "?")
            print(f"[IBKR] hist request timed out (no response): "
                  f"{contract.symbol} {what}")
            return []
_loop_started.wait(timeout=5)


# ---------------- Borrow data (CTB + shortable) ----------------

def _classify_available(shares):
    if shares >= 1_000_000:
        return "Easy"
    if shares >= 100_000:
        return "Medium"
    if shares > 0:
        return "Hard"
    return "None!"  # zero shares left to borrow — squeeze fuel


# Circuit breaker: once iborrowdesk.com has failed several times in a row,
# stop paying its ~2s-per-ticker connection-attempt cost on every single
# candidate every refresh (that alone was adding ~100s to a 50-stock cycle
# during this outage) — back off for a while, then automatically try again
# in case it's recovered.
_HTTP_FAIL_THRESHOLD = 3
_HTTP_COOLDOWN = 600   # 10 min
_http_fail_streak = 0
_http_disabled_until = 0.0


def _fetch_borrow_http(ticker):
    """Source (a): iborrowdesk.com — gives CTB + shortable together."""
    global _http_fail_streak, _http_disabled_until
    if time.time() < _http_disabled_until:
        return None
    try:
        r = requests.get(_IBORROW_URL.format(ticker=ticker), timeout=8,
                         headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            raise ValueError(f"HTTP {r.status_code}")
        data = r.json()
        rows = data.get("real_time") or data.get("daily") or []
        if not rows:
            raise ValueError("empty response")
        latest = rows[-1]
        fee = latest.get("fee")
        available = latest.get("available")
        result = {"ctb": "—", "shortable": "—"}
        if fee is not None:
            result["ctb"] = f"{float(fee):.1f}%"
        if available is not None:
            result["shortable"] = _classify_available(int(available))
        _http_fail_streak = 0  # a real response — the breaker resets
        return result
    except Exception:
        _http_fail_streak += 1
        if _http_fail_streak >= _HTTP_FAIL_THRESHOLD:
            _http_disabled_until = time.time() + _HTTP_COOLDOWN
            _http_fail_streak = 0
            print(f"[IBKR] iborrowdesk.com looks down — pausing it for {_HTTP_COOLDOWN // 60} min, using IBKR fallback")
        return None  # expected/frequent right now — the IBKR fallback handles it


# Smaller chunks arrive faster per-ticker than large ones (see note above), and
# they also commit results to cache incrementally — so a slow chunk can't cost
# us everything gathered so far.
_SHORTABLE_BATCH_SIZE = 8


async def _qualified(tickers):
    """Qualified contracts, cached. Qualifying is slow (measured ~3s/ticker on
    a cold call), and a ticker's contract definition doesn't change, so this is
    worth holding onto for the life of the process."""
    missing = [t for t in tickers if t not in _contract_cache]
    if missing:
        fresh = [Stock(t, "SMART", "USD") for t in missing]
        try:
            await _ib.qualifyContractsAsync(*fresh)
        except Exception as e:
            print(f"[IBKR] qualify: {e}")
        for c in fresh:
            if c.conId:
                _contract_cache[c.symbol] = c
    return [(t, _contract_cache[t]) for t in tickers if t in _contract_cache]


def _commit_borrow(ticker, ctb=None, shortable=None):
    """Merge one field into a ticker's borrow cache entry, immediately.

    The two fields arrive at very different times — shortable lands after the
    shared ~4s tick-236 wait, CTB takes up to _FEE_RATE_TIMEOUT and for many
    thin names never arrives at all. Committing per-field as each lands means
    a chunk that gets cancelled still keeps everything that had already come
    back, instead of discarding the whole chunk's work (which is what left
    scattered "—" in CTB/Shortable even for tickers IBKR had already answered).
    """
    entry = dict(_borrow_cache.get(ticker) or {})
    entry.setdefault("ctb", "—")
    entry.setdefault("shortable", "—")
    if ctb is not None:
        entry["ctb"] = ctb
    if shortable is not None:
        entry["shortable"] = shortable
    entry["_ts"] = time.time()
    entry["_ok"] = True
    _borrow_cache[ticker] = entry


def _fee_miss(symbol):
    """Record a no-data result; bench the ticker only on repeated misses."""
    strikes = _fee_strikes.get(symbol, 0) + 1
    _fee_strikes[symbol] = strikes
    if strikes >= _FEE_DEAD_STRIKES:
        _fee_dead[symbol] = time.time()


async def _fetch_fee_rate(symbol, contract):
    """CTB for one ticker. IBKR returns FEE_RATE as a decimal fraction
    (0.0025 = 0.25%), so scale by 100 for display."""
    try:
        bars = await _paced_hist(
            contract, limiter=_fee_limiter, slots=_slots_fee,
            timeout=_FEE_RATE_TIMEOUT,
            endDateTime="", durationStr="1 D", barSizeSetting="1 day",
            whatToShow="FEE_RATE", useRTH=True, formatDate=1)
        raw = bars[-1].close if bars else None
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            _fee_miss(symbol)
            return symbol, None
        _fee_strikes.pop(symbol, None)   # a real quote clears the record
        _fee_dead.pop(symbol, None)
        pct = float(raw) * 100.0
        _commit_borrow(symbol, ctb=f"{pct:.1f}%")   # banked the moment it lands
        return symbol, pct
    except Exception:
        _fee_miss(symbol)
        return symbol, None  # thin names often have no lending quote at all


async def _fetch_borrow_batch_ibkr(tickers):
    """Source (b) fallback, batched: get BOTH values for a chunk of tickers.

    Shortable comes from tick 236 (all subscribed at once, one shared wait),
    and CTB from concurrent FEE_RATE historical requests. Running the fee
    requests concurrently rather than serially measured ~4x faster
    (0.88s vs 3.75s per ticker) with zero pacing violations.

    Returns nothing: every value is written straight to _borrow_cache via
    _commit_borrow() as it arrives, so a cancelled call keeps its partial
    results instead of discarding the chunk.
    """
    if not tickers or not await _ensure_connected():
        return
    try:
        pairs = await _qualified(tickers)
        if not pairs:
            return

        # Skip tickers IBKR has already shown have no lending feed — asking
        # again just burns fee budget on a guaranteed miss and (before the
        # split budgets) was what starved the TTM bars.
        now = time.time()
        fee_pairs = [(sym, c) for sym, c in pairs
                     if (now - _fee_dead.get(sym, 0)) > _FEE_DEAD_TTL]

        # kick off both channels, then wait once for both
        live = [(sym, c, _ib.reqMktData(c, genericTickList="236", snapshot=False))
                for sym, c in pairs]
        try:
            fee_task = asyncio.gather(*[_fetch_fee_rate(sym, c) for sym, c in fee_pairs])
            await asyncio.sleep(_SHORTABLE_TICK_WAIT)

            # Bank shortable BEFORE awaiting the fee requests. Tick 236 has
            # already arrived by now, whereas the fee gather is the slow part
            # that actually gets cancelled — so committing here means a
            # cancellation costs us only CTB, never the shortable values we
            # already hold.
            for symbol, c, t in live:
                shares = t.shortableShares
                if shares is not None and not (isinstance(shares, float) and math.isnan(shares)):
                    _commit_borrow(symbol, shortable=_classify_available(int(shares)))

            # each _fetch_fee_rate commits its own CTB as it resolves
            await fee_task
        finally:
            # MUST run even if this coroutine is cancelled (e.g. the outer
            # _submit() timeout firing) -- otherwise a cancelled call leaves
            # tick-236 subscriptions live on these contracts forever. The next
            # call for the same tickers then subscribes AGAIN on top of the
            # orphaned one, and its own cancelMktData can't find the reqId it
            # expects -- that's the "cancelMktData: No reqId found" spam.
            for _, c, _t in live:
                try:
                    _ib.cancelMktData(c)
                except Exception:
                    pass
    except Exception as e:
        print(f"[IBKR] batch borrow: {e}")


def fetch_borrow_data_batch(tickers):
    """Fetch CTB + shortable for a whole list of tickers at once. Background
    thread only. Tries the HTTP source per-ticker first (cheap to fail fast
    right now), then batches whatever's left through the IBKR fallback in
    chunks — far faster than one ticker at a time."""
    now = time.time()
    needs_ibkr = []
    for ticker in tickers:
        cached = _borrow_cache.get(ticker)
        if cached:
            # a placeholder (no real values) expires much sooner than a good
            # reading, so a transient failure doesn't freeze the column
            ttl = _BORROW_TTL if cached.get("_ok") else _FAIL_RETRY_TTL
            if (now - cached.get("_ts", 0)) < ttl:
                continue
        result = _fetch_borrow_http(ticker)
        if result:
            result["_ts"] = now
            result["_ok"] = True
            _borrow_cache[ticker] = result
        else:
            needs_ibkr.append(ticker)

    for i in range(0, len(needs_ibkr), _SHORTABLE_BATCH_SIZE):
        chunk = needs_ibkr[i:i + _SHORTABLE_BATCH_SIZE]
        # Bound the wait by the real worst case: only _HIST_MAX_CONCURRENT_FEE
        # requests run at once, each bounded by _FEE_RATE_TIMEOUT, so a chunk
        # can't take longer than that many waves. An earlier version of this
        # multiplied chunk size by the full token-REFILL interval as if burst
        # capacity didn't exist, producing 200+ second timeouts for an 8-ticker
        # chunk — with a 60s screener refresh, that guaranteed the next
        # refresh's fetch would start before this one finished, stacking
        # overlapping IBKR calls on the same connection (see _ibkr_fetch_lock
        # in controller.py for why that's dangerous, not just slow).
        waves = math.ceil(len(chunk) / _HIST_MAX_CONCURRENT_FEE)
        budget = _SHORTABLE_TICK_WAIT + waves * _FEE_RATE_TIMEOUT + 15
        # Values are committed to _borrow_cache by the coroutine itself as each
        # one arrives, so nothing is read back from here — a timeout costs only
        # the requests still genuinely in flight.
        _submit(_fetch_borrow_batch_ibkr(chunk), timeout=budget,
                label=f"borrow[{len(chunk)}]")
        stamp = time.time()
        for ticker in chunk:
            # Anything with no entry at all never resolved this pass. Give it a
            # short-TTL placeholder so it backs off briefly instead of being
            # retried on every single refresh. Tickers that DID resolve (and
            # any older good reading) are left untouched.
            if ticker not in _borrow_cache:
                _borrow_cache[ticker] = {"ctb": "—", "shortable": "—",
                                         "_ts": stamp, "_ok": False}


def get_borrow_data(ticker):
    """Non-blocking — returns cache instantly."""
    return _borrow_cache.get(ticker, {"ctb": "—", "shortable": "—"})


def borrow_pending(ticker):
    """True while IBKR still owes us an answer about this ticker's borrow data.

    Critically distinguishes two states the "—" display collapses together:
      * PENDING     — not fetched yet, or the request timed out and is being
                      retried. We simply don't know.
      * UNAVAILABLE — asked repeatedly, confirmed there is no stock-loan feed
                      for this name at all (tracked in _fee_dead).
    The scorer must treat these differently: a stock must not gain pressure
    merely because data hasn't loaded yet, whereas a confirmed-absent lending
    market is genuine information about an illiquid name.
    """
    entry = _borrow_cache.get(ticker)
    if entry and entry.get("ctb", "—") != "—":
        return False            # we have a real number
    if ticker in _fee_dead:
        return False            # confirmed: no lending feed for this name
    return True                 # still waiting


# ---------------- Historical bars via IB Gateway (for TTM) ----------------

_CONNECT_RETRY_COOLDOWN = 55  # roughly once per refresh cycle, not once per ticker
_CLIENT_IDS = (15, 16, 17, 18)
_last_connect_attempt = 0.0
_last_connect_failed = False


async def _ensure_connected():
    global _last_connect_attempt, _last_connect_failed
    if _ib.isConnected():
        return True

    now = time.time()
    if _last_connect_failed and (now - _last_connect_attempt) < _CONNECT_RETRY_COOLDOWN:
        return False  # skip retry storm — we just failed a moment ago

    _last_connect_attempt = now
    # Gateway can hold a client id for a while after an app is killed, so a
    # quick restart hits "Error 326: client id already in use" and loses ALL
    # IBKR data for that run. Try a few ids rather than giving up on one.
    last_err = None
    for client_id in _CLIENT_IDS:
        try:
            await _ib.connectAsync("127.0.0.1", 4001, clientId=client_id,
                                   timeout=10, readonly=True)
            _ib.reqMarketDataType(3)  # delayed data is fine for daily bars
            print(f"[IBKR] Connected to IB Gateway (clientId={client_id})")
            _last_connect_failed = False
            return True
        except Exception as e:
            last_err = e
    print(f"[IBKR] Gateway not reachable (TTM disabled until it is): {last_err}")
    _last_connect_failed = True
    return False


async def _fetch_hist(ticker):
    if not await _ensure_connected():
        return None
    try:
        pairs = await _qualified([ticker])   # shares the contract cache
        if not pairs:
            return None
        contract = pairs[0][1]
        return await _bars_for(ticker, contract)
    except Exception as e:
        print(f"[IBKR] hist {ticker}: {e}")
        return None


async def _bars_for(symbol, contract):
    """One ticker's daily bars, paced through the bars lane."""
    return await _paced_hist(
        contract,
        limiter=_bars_limiter,
        slots=_slots_bars,
        timeout=_BARS_TIMEOUT,
        endDateTime="",
        durationStr="6 M",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,  # official RTH daily bars — matches TOS/TradingView, keeps ATR/BB accurate
        formatDate=1,
    )


async def _fetch_hist_many(tickers):
    """Bars for a whole chunk concurrently — one connect/qualify, N paced pulls."""
    if not await _ensure_connected():
        return {}
    try:
        pairs = await _qualified(tickers)
    except Exception as e:
        print(f"[IBKR] qualify bars: {e}")
        return {}
    if not pairs:
        return {}

    async def one(sym, contract):
        try:
            bars = await _bars_for(sym, contract)
        except asyncio.CancelledError:
            raise                      # cancellation is not a result
        except Exception as e:
            print(f"[IBKR] hist {sym}: {e}")
            bars = None
        _commit_bars(sym, bars)        # banked the moment THIS one resolves
        return sym, bars

    await asyncio.gather(*[one(s, c) for s, c in pairs])


def _submit(coro, timeout=20, label=""):
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    try:
        return future.result(timeout=timeout)
    except futures.TimeoutError:
        # CRITICAL: future.result() timing out does NOT stop the coroutine.
        # Without this cancel the abandoned request stays queued, later wakes
        # up, and spends a pacing token on a result nobody will ever read --
        # so each timed-out cycle stole budget from the next one and the
        # backlog compounded until nothing could get through at all.
        future.cancel()
        print(f"[IBKR] {label or 'request'} gave up after {timeout}s (cancelled)")
        return None
    except Exception as e:
        print(f"[IBKR] submit {label}: {e}")
        return None


def _commit_bars(ticker, bars):
    """Write one ticker's bars to cache immediately, as soon as they land.

    Same rationale as _commit_borrow: a chunk that hits its timeout should
    only lose the requests still in flight, not the ones already answered.
    A failed pull never overwrites a good older one.
    """
    prev = _bars_cache.get(ticker)
    if bars or not (prev and prev.get("bars")):
        _bars_cache[ticker] = {"bars": bars, "_ts": time.time()}


def _bars_stale(ticker, now):
    """True if this ticker needs a (re)fetch."""
    cached = _bars_cache.get(ticker)
    if not cached:
        return True
    # a miss retries in minutes; a good pull is held for the full TTL, so
    # one throttled/failed attempt can't pin TTM at "—" for hours
    ttl = _BARS_TTL if cached.get("bars") else _FAIL_RETRY_TTL
    return (now - cached.get("_ts", 0)) >= ttl


def fetch_bars_batch(tickers):
    """Daily bars for many tickers, in concurrent chunks. Background thread only.

    Replaces the old one-at-a-time loop: serially, a single unresponsive
    ticker cost the full per-request timeout before the next one even
    started, so a handful of dead names could eat the entire refresh cycle.
    """
    now = time.time()
    todo = [t for t in tickers if t and _bars_stale(t, now)]
    if not todo:
        return

    for i in range(0, len(todo), _BARS_BATCH_SIZE):
        chunk = todo[i:i + _BARS_BATCH_SIZE]
        # Same reasoning as the fee chunk budget above: bound by concurrency
        # waves, not by an assumed token-refill wait. This used to add 60s of
        # slack on top of chunk_size/concurrency*timeout, producing ~143s for
        # a 10-ticker chunk against a 60s refresh cycle -- see the
        # _ibkr_fetch_lock note in controller.py for what that caused.
        waves = math.ceil(len(chunk) / _HIST_MAX_CONCURRENT_BARS)
        budget = waves * _BARS_TIMEOUT + 15
        # Bars are committed by the coroutine itself via _commit_bars() as each
        # one resolves, so nothing is read back from here.
        _submit(_fetch_hist_many(chunk), timeout=budget,
                label=f"bars[{len(chunk)}]")
        stamp = time.time()
        for ticker in chunk:
            # Never resolved (chunk cancelled mid-flight) — short-TTL
            # placeholder so it backs off briefly rather than retrying every
            # refresh. Anything already committed is left alone.
            if ticker not in _bars_cache:
                _bars_cache[ticker] = {"bars": None, "_ts": stamp}


def fetch_bars_async(ticker):
    """Single-ticker convenience wrapper around the batch path."""
    fetch_bars_batch([ticker])


def pacing_status():
    """Diagnostics for both historical-request budgets."""
    return {"bars": _bars_limiter.snapshot(), "fee": _fee_limiter.snapshot(),
            "fee_dead": len(_fee_dead)}


def get_cached_bars(ticker):
    """Non-blocking — returns bars list or None."""
    cached = _bars_cache.get(ticker)
    return cached["bars"] if cached else None
