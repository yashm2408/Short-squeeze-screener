import csv
import sys
import os
import threading
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
from core.sentiment import train_or_load_model, classify_headlines
from core.finviz_api import fetch_all_finviz_api_news
from core.ibkr_api import (get_borrow_data, fetch_borrow_data_batch,
                           fetch_bars_batch, borrow_pending)
from core.short_volume import ensure_loaded as load_short_volume, get_short_volume_pct
from core.si_estimate import ensure_history_loaded as load_si_history, estimate_si_pct
from core.squeeze_score import compute_squeeze_score
from core.setup_classifier import evaluate as evaluate_setup, PRIME_PRESSURE, PRIME_IGNITION
import webbrowser
from core.filters import get_candidate_stocks, legacy_classify, prepare_finviz_df, LEGACY_DEFAULTS


def _parse_pct(value):
    try:
        return float(str(value).replace("%", "").replace("*", ""))
    except Exception:
        return None


class Controller:

    def log_prime_ticker(ticker_data):
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "prime_log.csv")
        fields = ["timestamp", "ticker", "price", "target", "stop_loss", "sentiment"]
        new_entry = {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker_data.get("Ticker"),
            "price": ticker_data.get("Price"),
            "target": ticker_data.get("Target"),
            "stop_loss": ticker_data.get("StopLoss"),
            "sentiment": ticker_data.get("Sentiment", "")
        }
        if os.path.exists(log_path):
            with open(log_path, "r", newline="", encoding="utf-8") as f:
                today = datetime.now().date()
                for row in csv.DictReader(f):
                    row_date = datetime.fromisoformat(row["timestamp"]).date()
                    if row["ticker"] == new_entry["ticker"] and row_date == today:
                        return
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(new_entry)

    NEWS_CACHE_TTL = 60

    def __init__(self):
        # Load the sentiment model in the background so the window opens instantly
        self.model, self.vectorizer = None, None
        self._news_cache = []
        self._news_cache_time = 0
        self._last_correlation_data = []

        # Adjustable thresholds — same criteria, user-tunable cutoffs.
        self.my_thresholds = {"pressure": PRIME_PRESSURE, "ignition": PRIME_IGNITION}
        self.legacy_thresholds = dict(LEGACY_DEFAULTS)

        # Cached raw data from the last Finviz fetch, so changing a threshold
        # can reclassify instantly without re-fetching or re-enriching.
        self._last_df = None
        self._last_candidates = []
        self._enrich_cache = {}  # ticker -> (row, signals, corr_row); cleared each real fetch
        self._ibkr_fetch_triggered = set()  # tickers already asked of IBKR this session
        # A full IBKR pass (bars + borrow) can legitimately run longer than one
        # 60s screener refresh. Without this lock, a slow pass gets a second
        # (and third, and fourth...) overlapping pass stacked on top of it every
        # refresh — all racing on the SAME ib_insync connection and the SAME
        # contracts' market-data subscriptions at once. That's what produced
        # "cancelMktData: No reqId found" (two coroutines fighting over one
        # subscription) and ever-growing wait times (more threads competing for
        # the same 2-3 IBKR concurrency slots each cycle). One pass at a time;
        # a refresh that finds one already running just skips and reuses cache.
        self._ibkr_fetch_lock = threading.Lock()

        threading.Thread(target=self._load_model_bg, daemon=True).start()

    def _load_model_bg(self):
        try:
            self.model, self.vectorizer = train_or_load_model()
            print("[Sentiment] Model ready")
        except Exception as e:
            print(f"[Sentiment] Model load failed: {e}")

    @property
    def news_cache(self):
        import time
        if time.time() - self._news_cache_time > self.NEWS_CACHE_TTL:
            fresh = fetch_all_finviz_api_news()
            if fresh:
                self._news_cache = fresh
                self._news_cache_time = time.time()
        return self._news_cache

    def _fetch_ibkr_background(self, tickers):
        # Borrow data (CTB + shortable) is fetched as ONE batched call — the
        # IBKR fallback subscribes tick 236 for a whole chunk of tickers at
        # once and waits only once per chunk, instead of ~2-4s per ticker
        # one at a time (which made a 50-stock cold start take minutes).
        # fetch_borrow_data_batch / fetch_bars_batch are cache-TTL gated, so
        # calling this repeatedly with overlapping ticker sets (e.g. Stage-1
        # candidates ∪ legacy picks) costs nothing extra for warm tickers.
        #
        # Only one pass at a time (see _ibkr_fetch_lock comment at __init__).
        # blocking=False: if a previous pass is still running, skip this one
        # entirely rather than queuing behind it — the next 60s refresh will
        # try again, and cached values from the last completed pass are still
        # served in the meantime.
        if not self._ibkr_fetch_lock.acquire(blocking=False):
            return
        try:
            # No manual sleep between calls: core.ibkr_api meters every
            # historical request through one shared pacing limiter, which is
            # both more precise and safer than a fixed delay here (a fixed
            # delay can't know how much of IBKR's budget is actually left).
            # Bars FIRST. They drive a visible column (TTM Squeeze), they
            # exist for essentially every listed ticker, and they're cached
            # for hours — so they're both the most valuable and the cheapest
            # thing to fetch. The borrow pass runs second because CTB is
            # best-effort (most thin names have no lending feed at all). The
            # two use separate pacing budgets in core.ibkr_api, so this
            # ordering can't starve either one; it just gets the column the
            # user actually watches populated soonest.
            try:
                fetch_bars_batch(tickers)  # daily bars for TTM Squeeze
            except Exception as e:
                print(f"[IBKR] bars batch: {e}")

            try:
                fetch_borrow_data_batch(tickers)
            except Exception as e:
                print(f"[IBKR] borrow batch: {e}")
        finally:
            self._ibkr_fetch_lock.release()

    # ---------------- threshold controls ----------------

    def get_thresholds(self):
        return {"my": dict(self.my_thresholds), "legacy": dict(self.legacy_thresholds)}

    def set_my_thresholds(self, pressure, ignition):
        self.my_thresholds = {"pressure": float(pressure), "ignition": float(ignition)}

    def set_legacy_thresholds(self, price_min, price_max, change_min, relvol_min, si_min):
        self.legacy_thresholds = {
            "price_min": float(price_min), "price_max": float(price_max),
            "change_min": float(change_min), "relvol_min": float(relvol_min),
            "si_min": float(si_min),
        }

    # ---------------- shared enrichment (both systems use this) ----------------

    def _enrich(self, stock):
        """SI-Live, CTB/Shortable, Short Vol%, Squeeze Score/TTM for one stock.
        Memoized per-refresh so a stock picked by BOTH systems (common — e.g.
        a name that is both loaded-and-firing and clears the old 4 checks)
        is only computed once. Returns (row, signals, corr_row):
          row       -- the 13-column display row, identical shape for both systems
          signals   -- raw values needed by setup_classifier.evaluate() ("My" only)
          corr_row  -- (ticker, si_official, si_live, change_pct, rel_vol) for the 3D plot
        """
        ticker = stock.get("Ticker", "")
        if ticker in self._enrich_cache:
            return self._enrich_cache[ticker]

        si_pct     = stock.get("SI_Pct", "—")
        dtc        = stock.get("DTC", "—")
        rel_volume = stock.get("RelVolume", "—")
        change_pct = stock.get("ChangePercent", "—")

        ibkr      = get_borrow_data(ticker)
        ctb       = ibkr.get("ctb", "—")
        shortable = ibkr.get("shortable", "—")

        short_vol = get_short_volume_pct(ticker)

        si_est_str, si_est_val = estimate_si_pct(
            ticker, si_pct,
            stock.get("FloatShares"), rel_volume, stock.get("AvgVolume"),
            ctb, shortable,
        )

        score_si = si_est_val if si_est_val is not None else si_pct
        squeeze  = compute_squeeze_score(ticker, score_si, dtc, rel_volume, change_pct, ctb, short_vol)
        ttm      = squeeze.get("ttm", {})

        row = [
            ticker,
            stock.get("Price", "?"),
            stock.get("Float", "?"),
            rel_volume,
            change_pct,
            si_pct,
            si_est_str,
            dtc,
            short_vol,
            ctb,
            shortable,
            squeeze["ttm_signal"],
            squeeze["squeeze_score"],
        ]
        signals = {
            "si_est_val": si_est_val, "dtc": dtc, "ctb": ctb, "shortable": shortable,
            "short_vol": short_vol, "ttm": ttm,
            # "—" in the CTB column is ambiguous: not-yet-fetched vs no lending
            # market exists. Only the former should withhold credit from the
            # pressure score — see pressure_score(borrow_pending=...).
            "borrow_pending": borrow_pending(ticker),
        }
        corr_row = (ticker, _parse_pct(si_pct), si_est_val, _parse_pct(change_pct), stock.get("RelVol_num"))

        result = (row, signals, corr_row)
        self._enrich_cache[ticker] = result
        return result

    # ---------------- My Prime Setup (Pressure / Ignition) ----------------

    def _classify_my(self, candidates):
        """Enrich + classify candidates against self.my_thresholds. No network
        calls beyond cache reads — safe to call repeatedly on threshold changes."""
        corr_rows = []
        prime, subprime = [], []

        for stock in candidates:
            row, sig, corr_row = self._enrich(stock)
            corr_rows.append(corr_row)

            si_for_setup = sig["si_est_val"] if sig["si_est_val"] is not None else stock.get("SI_official_num")
            verdict = evaluate_setup(
                si_for_setup, sig["dtc"], sig["ctb"], sig["shortable"], sig["short_vol"],
                stock.get("RelVol_num"), stock.get("Change_num"),
                sig["ttm"].get("signal"), sig["ttm"].get("mom_up"),
                stock.get("FloatShares"),
                prime_pressure=self.my_thresholds["pressure"],
                prime_ignition=self.my_thresholds["ignition"],
                borrow_pending=sig["borrow_pending"],
            )

            if verdict["setup"] is None:
                continue

            entry = (verdict["pressure"] + verdict["ignition"], row)
            if verdict["setup"] == "prime":
                prime.append(entry)
                Controller.log_prime_ticker(stock)
            else:
                subprime.append(entry)

        prime.sort(key=lambda e: e[0], reverse=True)
        subprime.sort(key=lambda e: e[0], reverse=True)
        return [r for _, r in prime], [r for _, r in subprime], corr_rows

    # ---------------- Previous Prime Setup (original 4-condition rule) ----------------

    def _build_legacy_rows(self, prime_stocks, subprime_stocks):
        """Same enrichment/columns as 'My' — only the selection rule differs."""
        def build(stocks):
            entries = []
            for stock in stocks:
                row, _, _ = self._enrich(stock)
                entries.append((stock.get("Change_num", 0), row))
            entries.sort(key=lambda e: e[0], reverse=True)
            return [r for _, r in entries]
        return build(prime_stocks), build(subprime_stocks)

    def get_screener_results(self):
        # Bulk, non-per-ticker loads. Each internally no-ops for up to an hour
        # once loaded, so blocking here costs a few seconds on the first call
        # (or the first call after the hourly TTL expires) and is a no-op on
        # every refresh in between. This has to run BEFORE classification —
        # previously it only ran inside the background thread below, so every
        # refresh that started before that thread finished (in practice, the
        # very first refresh after launch, every time) formatted SI %(Live)
        # and Short Vol % from empty caches: estimate_si_pct silently fell
        # back to the raw SI% with excess_shares=0, making "Live" identical
        # to the official number instead of genuinely differing.
        try:
            load_short_volume()
        except Exception as e:
            print(f"[ShortVol] {e}")
        try:
            load_si_history()
        except Exception as e:
            print(f"[SI Est] {e}")

        df = prepare_finviz_df()
        candidates = get_candidate_stocks(df)
        legacy_prime_stocks, legacy_subprime_stocks = legacy_classify(df, **self.legacy_thresholds)

        self._last_df = df
        self._last_candidates = candidates
        self._last_legacy_prime_stocks = legacy_prime_stocks
        self._last_legacy_subprime_stocks = legacy_subprime_stocks
        self._enrich_cache = {}  # fresh data this cycle — start a clean memo cache

        # Union: legacy picks aren't always a subset of the Stage-1 candidate
        # pool (a name can clear the old 5% SI bar without clearing Stage-1's
        # stricter 8% "loaded" gate), so make sure THOSE tickers get borrow
        # data fetched too, not just the candidates.
        all_stocks = {s["Ticker"]: s for s in candidates}
        for s in legacy_prime_stocks + legacy_subprime_stocks:
            all_stocks.setdefault(s["Ticker"], s)
        all_tickers = [t for t in all_stocks if t]
        self._ibkr_fetch_triggered = set(all_tickers)
        threading.Thread(
            target=self._fetch_ibkr_background,
            args=(all_tickers,),
            daemon=True
        ).start()

        my_prime, my_subprime, corr_rows = self._classify_my(candidates)
        self._last_correlation_data = corr_rows
        legacy_prime, legacy_subprime = self._build_legacy_rows(legacy_prime_stocks, legacy_subprime_stocks)

        return {
            "my_prime": my_prime, "my_subprime": my_subprime,
            "legacy_prime": legacy_prime, "legacy_subprime": legacy_subprime,
        }

    def reclassify(self):
        """Re-run classification on the last-fetched data with current
        thresholds — no Finviz re-fetch, no network calls (enrichment is
        memoized, so this is fast even though it re-scans the full universe
        for the legacy rule). Used by the threshold 'Apply' button."""
        if self._last_df is None:
            return self.get_screener_results()

        my_prime, my_subprime, corr_rows = self._classify_my(self._last_candidates)
        self._last_correlation_data = corr_rows

        legacy_prime_stocks, legacy_subprime_stocks = legacy_classify(self._last_df, **self.legacy_thresholds)
        legacy_prime, legacy_subprime = self._build_legacy_rows(legacy_prime_stocks, legacy_subprime_stocks)

        # Any newly-selected legacy stock IBKR has never been asked about
        # this session gets its borrow data fetched in the background —
        # checked against the trigger set, not _enrich_cache (which is
        # already fully populated by _build_legacy_rows above by this point).
        new_tickers = [s["Ticker"] for s in legacy_prime_stocks + legacy_subprime_stocks
                      if s["Ticker"] not in self._ibkr_fetch_triggered]
        if new_tickers:
            self._ibkr_fetch_triggered.update(new_tickers)
            threading.Thread(target=self._fetch_ibkr_background, args=(new_tickers,), daemon=True).start()

        return {
            "my_prime": my_prime, "my_subprime": my_subprime,
            "legacy_prime": legacy_prime, "legacy_subprime": legacy_subprime,
        }

    def get_correlation_data(self):
        """Non-blocking — returns (ticker, si_official, si_live, change_pct, rel_vol) tuples from the last refresh."""
        return list(self._last_correlation_data)

    def get_positive_news(self):
        if not self.model:
            return []
        news = self.news_cache
        headlines = [item["headline"] for item in news]
        df = classify_headlines(headlines, self.model, self.vectorizer)
        positive = []
        for i, row in df.iterrows():
            if "Positive" in row["prediction"] and row["confidence_score"] >= 0.6:
                positive.append({
                    "headline": row["headline"],
                    "confidence_score": row["confidence_score"],
                    "tickers": news[i].get("tickers", []),
                    "url": news[i].get("url", "")
                })
        return positive

    def open_url(self, url):
        webbrowser.open_new_tab(url)
