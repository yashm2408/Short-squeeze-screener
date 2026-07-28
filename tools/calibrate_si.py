"""
Calibrate the live SI% estimator against FINRA's actual bi-monthly outcomes.

The estimator in core/si_estimate.py predicts the CHANGE in short interest
between official settlements as:

    predicted_delta_shares = k * excess_short_volume_shares

where excess = sum over window days of (ShortVol_d - baseline * TotalVol_d)
from FINRA's daily short-sale files, and baseline is the stock's own normal
short-volume ratio. Until now k = 0.25 was a hand-picked guess.

This tool backtests that model against reality:

  1. Pulls actual short interest per settlement from FINRA's free
     consolidatedShortInterest dataset (api.finra.org, no key needed).
  2. Builds a universe of shorted, liquid-to-illiquid listed stocks.
  3. Pulls current float shares per symbol from yfinance (used only as the
     per-day turnover-cap threshold below — a coarse guard, not part of the
     core SI-delta relationship, so using today's float as a proxy for the
     float at each historical settlement is an acceptable approximation).
  4. For each settlement->settlement window, computes the excess-short-volume
     signal exactly the way si_estimate.py does, including its per-day
     turnover cap (a single day's volume capped at 3x float before entering
     the sum — past that point, extra volume is overwhelmingly the same
     shares churning intraday, not new distinct short positions; this
     matters most for the micro-float names this screener actually targets).
  5. Fits k by grid search (min median absolute error, robust to outliers),
     pooled and per liquidity bucket.
  6. Writes data/si_calibration.json, which si_estimate.py loads at runtime.

Run (from ScreenerProject root):
    venv/Scripts/python.exe tools/calibrate_si.py --windows 8 --universe 120
"""
import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta

import requests
import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core.si_estimate import _get_day_file, _parse, _cap_day_turnover  # reuse daily-file cache + turnover cap

_API = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
_PARTITIONS = "https://api.finra.org/partitions/group/otcMarket/name/consolidatedShortInterest"
_H = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json", "Accept": "application/json"}

_OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "si_calibration.json")

BASELINE_TRADING_DAYS = 20     # ~4 weeks, matches si_estimate.py
MIN_WINDOW_COVERAGE = 0.8      # need daily files for >=80% of window days
ADV_BUCKETS = [(1_000_000, "small"), (10_000_000, "mid"), (None, "large")]


# ---------------- FINRA consolidated SI API ----------------

def api_settlements():
    r = requests.get(_PARTITIONS, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    parts = [p["partitions"][0] for p in r.json()["availablePartitions"]]
    return sorted(parts, reverse=True)  # newest first


def api_rows(settlement, symbols=None, limit=5000, offset=0):
    body = {
        "limit": limit,
        "offset": offset,
        "compareFilters": [{"fieldName": "settlementDate", "compareType": "EQUAL",
                            "fieldValue": settlement}],
    }
    if symbols:
        body["domainFilters"] = [{"fieldName": "symbolCode", "values": list(symbols)}]
    r = requests.post(_API, json=body, headers=_H, timeout=60)
    if r.status_code == 204:
        return []
    r.raise_for_status()
    return r.json()


def fetch_si(settlement, symbols):
    """{symbol: {si, prev_si, adv}} for the given settlement date."""
    out = {}
    symbols = list(symbols)
    for i in range(0, len(symbols), 50):  # keep domain filter lists modest
        chunk = symbols[i:i + 50]
        for row in api_rows(settlement, symbols=chunk):
            out[row["symbolCode"]] = {
                "si": row.get("currentShortPositionQuantity"),
                "prev_si": row.get("previousShortPositionQuantity"),
                "adv": row.get("averageDailyVolumeQuantity"),
            }
        time.sleep(0.4)  # be polite to the free endpoint
    return out


def build_universe(settlement, size):
    """Most-shorted listed symbols, balanced across liquidity buckets."""
    rows, offset = [], 0
    while True:
        page = api_rows(settlement, limit=5000, offset=offset)
        rows.extend(page)
        if len(page) < 5000:
            break
        offset += 5000
        time.sleep(0.4)

    listed = [r for r in rows
              if r.get("marketClassCode") in ("NYSE", "NNM", "NQGM", "NQGS", "NQCM", "AMEX", "ARCA")
              and r.get("symbolCode") and r["symbolCode"].isalpha()
              and (r.get("currentShortPositionQuantity") or 0) > 500_000
              and (r.get("averageDailyVolumeQuantity") or 0) > 50_000]

    per_bucket = size // len(ADV_BUCKETS)
    universe = []
    for cap, _name in ADV_BUCKETS:
        lo = 0 if cap == 1_000_000 else (1_000_000 if cap == 10_000_000 else 10_000_000)
        hi = cap or float("inf")
        bucket = [r for r in listed if lo <= (r.get("averageDailyVolumeQuantity") or 0) < hi]
        bucket.sort(key=lambda r: r.get("daysToCoverQuantity") or 0, reverse=True)
        universe.extend(r["symbolCode"] for r in bucket[:per_bucket])
    return universe


# ---------------- daily short-volume windows ----------------

def trading_days_between(start, end):
    """Business days in (start, end] — matches how the estimator accumulates."""
    days, d = [], start + timedelta(days=1)
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def load_day(d, cache):
    ymd = d.strftime("%Y%m%d")
    if ymd not in cache:
        text = _get_day_file(d)
        cache[ymd] = _parse(text) if text else None
    return cache[ymd]


def window_signal(symbol, s_prev, s_curr, cache, float_shares):
    """(excess_shares, coverage) for one stock over one settlement window.
    Applies the same per-day turnover cap as estimate_si_pct() at runtime,
    so the fitted k stays matched to what the estimator actually computes."""
    base_days = []
    d = s_prev
    while len(base_days) < BASELINE_TRADING_DAYS and (s_prev - d).days < 60:
        if d.weekday() < 5:
            day = load_day(d, cache)
            if day and symbol in day:
                base_days.append(day[symbol])
        d -= timedelta(days=1)
    base_rows = [(sv, tv) for sv, tv in base_days if tv > 0]
    if len(base_rows) < 5:
        return None, 0.0
    baseline = sum(sv for sv, _ in base_rows) / sum(tv for _, tv in base_rows)

    win_days = trading_days_between(s_prev, s_curr)
    got, excess = 0, 0.0
    for d in win_days:
        day = load_day(d, cache)
        if day is None:
            continue  # holiday or unpublished file
        rec = day.get(symbol)
        if not rec:
            continue
        sv, tv = rec
        if tv > 0:
            csv, ctv = _cap_day_turnover(sv, tv, float_shares)
            excess += csv - baseline * ctv
            got += 1
    coverage = got / max(1, len([d for d in win_days if load_day(d, cache) is not None]))
    return excess, coverage


_float_cache = {}


def get_float(symbol):
    """Current float shares from yfinance, used only as the turnover-cap
    threshold (a coarse guard, not the core SI-delta relationship) — cached
    across all windows for a run since float rarely moves week to week."""
    if symbol in _float_cache:
        return _float_cache[symbol]
    flt = None
    try:
        flt = yf.Ticker(symbol).info.get("floatShares")
    except Exception:
        pass
    _float_cache[symbol] = flt
    return flt


# ---------------- fitting ----------------

def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def eval_k(k, obs):
    """Median absolute error of predicted vs actual delta, relative to prior SI."""
    return median([abs(k * x - y) / si for x, y, si, _ in obs])


def fit_k(obs, k_grid=None):
    if k_grid is None:
        k_grid = [i / 100 for i in range(0, 151)]
    best = min(k_grid, key=lambda k: eval_k(k, obs))
    return best, eval_k(best, obs)


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    vx = sum((a - mx) ** 2 for a in xs) ** 0.5
    vy = sum((b - my) ** 2 for b in ys) ** 0.5
    return cov / (vx * vy) if vx > 0 and vy > 0 else 0.0


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=8, help="settlement windows to backtest")
    ap.add_argument("--universe", type=int, default=120, help="symbols in calibration universe")
    args = ap.parse_args()

    print("[1/6] Settlement partitions…")
    settlements = api_settlements()
    needed = settlements[:args.windows + 1]          # newest first
    needed_dates = [datetime.strptime(s, "%Y-%m-%d").date() for s in needed]
    print(f"      Using {needed[-1]} … {needed[0]} ({args.windows} windows)")

    print("[2/6] Building universe from", needed[0])
    universe = build_universe(needed[0], args.universe)
    print(f"      {len(universe)} symbols")

    print("[3/6] Fetching official SI for each settlement…")
    si_by_settlement = {}
    for s in needed:
        si_by_settlement[s] = fetch_si(s, universe)
        print(f"      {s}: {len(si_by_settlement[s])} symbols")

    print("[4/6] Fetching float shares per symbol (turnover-cap threshold, yfinance)…")
    floats = {}
    for i, sym in enumerate(universe):
        floats[sym] = get_float(sym)
        if (i + 1) % 30 == 0:
            print(f"      {i + 1}/{len(universe)}…")
    have_float = sum(1 for v in floats.values() if v)
    print(f"      {have_float}/{len(universe)} symbols have a float value; "
          f"the rest fall back to no cap for this run")

    print("[5/6] Computing excess-short-volume signal per window (daily files download+cache on first run)…")
    cache, obs = {}, []
    for i in range(len(needed) - 1):
        s_curr_str, s_prev_str = needed[i], needed[i + 1]
        s_curr, s_prev = needed_dates[i], needed_dates[i + 1]
        cur_si, prev_si = si_by_settlement[s_curr_str], si_by_settlement[s_prev_str]
        n_win = 0
        for sym in universe:
            a, b = cur_si.get(sym), prev_si.get(sym)
            if not a or not b or not a.get("si") or not b.get("si"):
                continue
            excess, coverage = window_signal(sym, s_prev, s_curr, cache, floats.get(sym))
            if excess is None or coverage < MIN_WINDOW_COVERAGE:
                continue
            actual_delta = a["si"] - b["si"]
            adv = a.get("adv") or 0
            obs.append((excess, actual_delta, b["si"], adv))
            n_win += 1
        print(f"      window {s_prev_str} -> {s_curr_str}: {n_win} usable observations")

    if len(obs) < 50:
        print(f"ABORT: only {len(obs)} observations — not enough to fit. Check network/data.")
        sys.exit(1)

    print(f"[6/6] Fitting k on {len(obs)} observations…")
    xs = [o[0] for o in obs]
    ys = [o[1] for o in obs]
    corr = pearson(xs, ys)

    mae_naive = eval_k(0.0, obs)          # carry official number forward
    mae_current = eval_k(0.25, obs)       # today's hand-picked constant
    k_pooled, mae_pooled = fit_k(obs)

    buckets_out = []
    for cap, name in ADV_BUCKETS:
        lo = 0 if cap == 1_000_000 else (1_000_000 if cap == 10_000_000 else 10_000_000)
        hi = cap or float("inf")
        sub = [o for o in obs if lo <= o[3] < hi]
        if len(sub) >= 30:
            kb, maeb = fit_k(sub)
            buckets_out.append({"max_adv": cap, "k": kb, "n": len(sub), "mae": round(maeb, 4)})
            print(f"      {name:6s} (ADV<{cap or 'inf'}): k={kb:.2f}  medAE={maeb:.4f}  n={len(sub)}")
        else:
            buckets_out.append({"max_adv": cap, "k": k_pooled, "n": len(sub), "mae": None})
            print(f"      {name:6s}: only {len(sub)} obs — falling back to pooled k")

    print()
    print(f"      signal correlation (excess vs actual dSI): r = {corr:.3f}")
    print(f"      median |error| / prior SI:")
    print(f"        naive (k=0, official carried forward): {mae_naive:.4f}")
    print(f"        current hand-picked k=0.25:            {mae_current:.4f}")
    print(f"        fitted pooled k={k_pooled:.2f}:                 {mae_pooled:.4f}")

    from core.si_estimate import _MAX_DAILY_TURNOVER

    result = {
        "fitted_at": date.today().isoformat(),
        "windows": args.windows,
        "settlements": [needed[-1], needed[0]],
        "n_obs": len(obs),
        "signal_correlation": round(corr, 4),
        "k_pooled": k_pooled,
        "max_daily_turnover_cap": _MAX_DAILY_TURNOVER,
        "symbols_with_float": have_float,
        "buckets": buckets_out,
        "metrics": {
            "median_relative_error_naive": round(mae_naive, 4),
            "median_relative_error_k025": round(mae_current, 4),
            "median_relative_error_fitted": round(mae_pooled, 4),
        },
    }
    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {os.path.normpath(_OUT_PATH)}")
    if k_pooled == 0.0:
        print("NOTE: fitted k = 0 — the daily-short-volume signal did not add accuracy "
              "over carrying the official number forward. The estimator will behave "
              "nearly like the official number, which is the honest answer.")


if __name__ == "__main__":
    main()
