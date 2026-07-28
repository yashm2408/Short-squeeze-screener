"""
Live estimated short interest (SI% Est) — recalculated every screen refresh.



    SI_est = SI_official + delta_since_settlement + delta_today

    delta_since_settlement = k * sum_over_days[ (SVR_d - B) * V_d ] / Float * 100
    delta_today            = k * (SVR_latest - B) * V_today / Float * 100

    SVR_d   = short volume ratio on day d (from the nightly FINRA file)
    B       = the stock's own baseline short volume ratio (its mean SVR
              over the window; 0.45 market average if too little history)
    V_d     = total consolidated volume on day d
    V_today = RelVol x AvgVol from Finviz (refreshes every minute)
    k       = dampening — most short volume is market-maker flow that is
              covered the same day and never becomes short interest.
              Calibrated per liquidity bucket by tools/calibrate_si.py
              against actual FINRA settlement outcomes
              (data/si_calibration.json); falls back to 0.25 if absent.
    Float   = shares float

The delta is then scaled by borrow pressure from IBKR lending data
(fees spiking / shares unavailable means shorts really are accumulating)
and clamped to a sane band around the official number. Estimates are
marked with * in the UI.
"""
import json
import os
import threading
import time
from calendar import monthrange
from datetime import date, timedelta

import requests

_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ymd}.txt"
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "shortvol_cache")
_CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "si_calibration.json")

_K = 0.25                 # fallback dampening if no calibration file exists
_MARKET_BASELINE = 0.45   # typical market-wide short volume ratio

# On micro-float names, daily volume can be several multiples of the entire
# float (the same shares changing hands repeatedly intraday). Past this
# point extra volume is overwhelmingly repeat churn, not new distinct short
# positions accumulating, so its contribution to the excess-volume signal is
# capped at this many float-turnovers per day (ratio preserved, magnitude
# bounded). Without this a single mega-volume day on a thin float can
# single-handedly blow the estimate out to the sanity-band ceiling.
# tools/calibrate_si.py applies the identical cap when fitting k, so k stays
# matched to what estimate_si_pct actually feeds it at runtime.
_MAX_DAILY_TURNOVER = 3.0


def _cap_day_turnover(sv, tv, float_shares):
    if not float_shares or float_shares <= 0 or tv <= 0:
        return sv, tv
    cap = _MAX_DAILY_TURNOVER * float_shares
    if tv > cap:
        scale = cap / tv
        return sv * scale, cap
    return sv, tv

# k calibrated by tools/calibrate_si.py against actual FINRA settlement
# outcomes (see data/si_calibration.json for fit metrics). Buckets by ADV.
_calibration = None
try:
    with open(_CALIBRATION_PATH, encoding="utf-8") as _f:
        _calibration = json.load(_f)
    print(f"[SI Est] Calibration loaded (fitted {_calibration.get('fitted_at')}, "
          f"n={_calibration.get('n_obs')}, k_pooled={_calibration.get('k_pooled')})")
except Exception:
    pass


def _k_for(avg_vol_shares):
    """Calibrated k for this stock's liquidity bucket; falls back to pooled, then 0.25."""
    if not _calibration:
        return _K
    try:
        adv = float(avg_vol_shares or 0)
        for b in _calibration.get("buckets", []):
            cap = b.get("max_adv")
            if cap is None or adv < cap:
                return float(b["k"])
    except Exception:
        pass
    return float(_calibration.get("k_pooled", _K))

_lock = threading.Lock()
_history = {}             # since settlement: {"YYYYMMDD": {sym: (short_vol, total_vol)}}
_baseline_hist = {}       # ~4 weeks BEFORE settlement — the stock's normal shorting level
_settlement_ymd = ""      # official SI base date the estimate builds on
_loaded = False
_last_attempt = 0.0


# ---------------- FINRA settlement calendar ----------------

def _prev_business_day(d):
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _business_days_after(d, n):
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def _latest_published_settlement(today):
    """Most recent FINRA SI settlement date whose report is already public
    (settlements are the 15th and month-end; published ~9 business days later)."""
    candidates = []
    for months_back in range(3):
        y, m = today.year, today.month - months_back
        while m <= 0:
            m += 12
            y -= 1
        candidates.append(_prev_business_day(date(y, m, 15)))
        candidates.append(_prev_business_day(date(y, m, monthrange(y, m)[1])))
    published = [d for d in candidates if _business_days_after(d, 9) <= today]
    return max(published)


# ---------------- Daily file download (disk-cached) ----------------

def _get_day_file(d):
    ymd = d.strftime("%Y%m%d")
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"CNMSshvol{ymd}.txt")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    try:
        r = requests.get(_URL.format(ymd=ymd), timeout=20,
                         headers={"User-Agent": "Mozilla/5.0"})
    except Exception:
        return None
    if r.status_code != 200 or not r.text.startswith("Date|"):
        return None  # weekend / holiday / not yet published
    with open(path, "w", encoding="utf-8") as f:
        f.write(r.text)
    return r.text


def _parse(text):
    day = {}
    for line in text.splitlines()[1:]:
        p = line.split("|")
        if len(p) < 5:
            continue
        try:
            day[p[1]] = (float(p[2]), float(p[4]))
        except ValueError:
            continue
    return day


def ensure_history_loaded():
    """Download daily short volume files since the last FINRA settlement.
    Call from a background thread only. Safe to call repeatedly."""
    global _history, _baseline_hist, _settlement_ymd, _loaded, _last_attempt
    now = time.time()
    with _lock:
        if _loaded and now - _last_attempt < 3600:
            return
        _last_attempt = now

    today = date.today()
    settlement = _latest_published_settlement(today)

    history = {}
    d = settlement + timedelta(days=1)
    while d < today:
        text = _get_day_file(d)
        if text:
            history[d.strftime("%Y%m%d")] = _parse(text)
        d += timedelta(days=1)

    # baseline window: the ~4 weeks BEFORE settlement — what "normal"
    # shorting looks like for each stock, so post-settlement deviations
    # measure a genuine shift rather than netting out to zero
    baseline_hist = {}
    d = settlement - timedelta(days=28)
    while d <= settlement:
        text = _get_day_file(d)
        if text:
            baseline_hist[d.strftime("%Y%m%d")] = _parse(text)
        d += timedelta(days=1)

    with _lock:
        _history = history
        _baseline_hist = baseline_hist
        _settlement_ymd = settlement.strftime("%Y%m%d")
        _loaded = True
    print(f"[SI Est] Loaded {len(history)} days since settlement {_settlement_ymd} "
          f"(+{len(baseline_hist)} baseline days)")


# ---------------- The estimate ----------------

def _borrow_multiplier(ctb_str, shortable):
    try:
        ctb = float(str(ctb_str).replace("%", ""))
    except Exception:
        ctb = None
    if shortable in ("Hard", "None!") or (ctb is not None and ctb >= 20):
        return 1.25
    if shortable == "Easy" and ctb is not None and ctb < 1:
        return 0.75
    return 1.0


def estimate_si_pct(ticker, base_si_str, float_shares, rel_vol, avg_vol_shares,
                    ctb_str="—", shortable="—"):
    """Returns (display_str, numeric_value). Falls back to the official
    number when the estimate can't be computed."""
    try:
        base = float(str(base_si_str).replace("%", ""))
    except Exception:
        return "—", None
    if not float_shares or float_shares <= 0:
        return f"{base:.1f}%", base

    with _lock:
        days = [(ymd, day.get(ticker)) for ymd, day in sorted(_history.items())]
        base_days = [day.get(ticker) for day in _baseline_hist.values()]
    rows = [(ymd, sv, tv) for ymd, r in days if r for sv, tv in [r] if tv > 0]

    # stock's normal shorting level = volume-weighted short ratio in the
    # weeks BEFORE the official settlement date
    base_rows = [(sv, tv) for r in base_days if r for sv, tv in [r] if tv > 0]
    if len(base_rows) >= 5:
        baseline = sum(sv for sv, _ in base_rows) / sum(tv for _, tv in base_rows)
    else:
        baseline = _MARKET_BASELINE

    # accumulated excess shorting since the official settlement date
    # (each day's volume capped at _MAX_DAILY_TURNOVER x float — see above)
    excess_shares = sum(
        (csv - baseline * ctv)
        for _, sv, tv in rows
        for csv, ctv in [_cap_day_turnover(sv, tv, float_shares)]
    )

    # intraday component: assume today's shorting runs at the latest known
    # ratio, applied to today's live volume (RelVol x AvgVol, minute-fresh)
    try:
        v_today = float(rel_vol) * float(avg_vol_shares)
        if rows and v_today > 0:
            _, sv, tv = rows[-1]
            svr_latest = sv / tv
            _, v_today_capped = _cap_day_turnover(svr_latest * v_today, v_today, float_shares)
            excess_shares += (svr_latest - baseline) * v_today_capped
    except Exception:
        pass

    delta = _k_for(avg_vol_shares) * excess_shares / float_shares * 100
    delta *= _borrow_multiplier(ctb_str, shortable)

    est = base + delta
    est = max(0.0, max(0.3 * base, min(2.5 * base, est)))  # sanity band
    return f"{est:.1f}%*", round(est, 1)
