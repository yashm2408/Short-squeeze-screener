"""
Squeeze Score (0-100) — weighted blend of the classic squeeze factors:

    Score = SI%      x 0.25   (fuel: % of float sold short — FINRA, bi-monthly)
          + DTC      x 0.20   (exit door: days for shorts to cover at avg volume)
          + CTB      x 0.15   (real-time pain: annualized borrow fee, IBKR ~15 min)
          + ShortVol x 0.10   (daily pressure: short sales / total volume, nightly)
          + RelVol   x 0.15   (spark: unusual volume today)
          + Momentum x 0.15   (ignition: price change + TTM Squeeze signal)

Tiers: 80-100 PRIME, 60-79 SUBPRIME, 40-59 WATCH, below 40 IGNORE.
"""
from core.ttm_squeeze import compute_ttm


def _ttm_squeeze(ticker):
    try:
        from core.ibkr_api import get_cached_bars
        bars = get_cached_bars(ticker)
        if bars:
            return compute_ttm(bars)
    except Exception:
        pass
    return {"signal": "—", "display": "—", "compression": None,
            "mom_up": None, "mom_rising": None, "fired_bars_ago": None}


def _score_si(si_pct_str):
    try:
        val = float(str(si_pct_str).replace("%", ""))
    except Exception:
        return 0
    if val < 10: return 0
    if val < 20: return 20
    if val < 30: return 50
    if val < 50: return 80
    return 100


def _score_dtc(dtc_str):
    try:
        val = float(str(dtc_str))
    except Exception:
        return 0
    if val < 1:  return 0
    if val < 3:  return 30
    if val < 5:  return 60
    if val < 10: return 85
    return 100


def _score_relvol(relvol_str):
    try:
        val = float(str(relvol_str))
    except Exception:
        return 0
    if val < 1: return 0
    if val < 2: return 20
    if val < 3: return 50
    if val < 5: return 80
    return 100


def _score_momentum(change_pct_str, ttm):
    score = 0
    try:
        change = float(str(change_pct_str).replace("%", ""))
        if change > 5:   score += 40
        elif change > 2: score += 20
        elif change > 0: score += 10
    except Exception:
        pass

    signal = ttm.get("signal")
    if signal == "Fired":
        # a release leaning upward is the ignition we screen for; a fresh
        # fire counts more than one from several days back
        if ttm.get("mom_up"):
            score += 60 if (ttm.get("fired_bars_ago") or 9) <= 2 else 45
        else:
            score += 15  # fired downward — not a short-squeeze setup
    elif signal == "Squeezed":
        score += 30
        if ttm.get("compression") == "HIGH":
            score += 15   # tightest coil
        if ttm.get("mom_up") and ttm.get("mom_rising"):
            score += 10   # already leaning up while still coiled
    return min(score, 100)


def _score_ctb(ctb_str):
    try:
        val = float(str(ctb_str).replace("%", ""))
    except Exception:
        return 0
    if val < 1:  return 0
    if val < 5:  return 20
    if val < 20: return 50
    if val < 50: return 80
    return 100


def _score_short_vol(short_vol_str):
    # ~40-50% is normal market-making noise; sustained 60%+ is real pressure
    try:
        val = float(str(short_vol_str).replace("%", ""))
    except Exception:
        return 0
    if val < 40: return 0
    if val < 50: return 30
    if val < 60: return 60
    if val < 70: return 80
    return 100


def compute_squeeze_score(ticker, si_pct, dtc, rel_volume, change_pct, ctb="—", short_vol="—"):
    """
    Score = (SI% x 0.25) + (DTC x 0.20) + (CTB x 0.15) + (ShortVol x 0.10)
          + (RelVol x 0.15) + (Momentum x 0.15)
    """
    ttm   = _ttm_squeeze(ticker)
    score = round(
        _score_si(si_pct)                * 0.25 +
        _score_dtc(dtc)                  * 0.20 +
        _score_ctb(ctb)                  * 0.15 +
        _score_short_vol(short_vol)      * 0.10 +
        _score_relvol(rel_volume)        * 0.15 +
        _score_momentum(change_pct, ttm) * 0.15
    )
    return {"squeeze_score": score, "ttm_signal": ttm.get("display", "—"), "ttm": ttm}
