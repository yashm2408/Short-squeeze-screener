"""
Prime / Subprime setup classification — a two-axis squeeze model.

Real short squeezes need two independent things to be true at once, and
professional squeeze traders judge them separately:

  PRESSURE  — the structural fuel: how violently could this squeeze if it
              got a spark?  Built from SI% (Live), Days-to-Cover,
              Cost-to-Borrow, Shortability and Short Volume %.

  IGNITION  — the spark: is it actually firing right now?  Built from
              Relative Volume, today's Price Change % (up only — squeezes
              go up), and the TTM Squeeze signal.

  PRIME    = loaded AND firing   (high pressure + real ignition)
  SUBPRIME = loaded but early    (high pressure, little ignition), OR
             firing on lighter fuel (moderate pressure + strong ignition)
  (anything else is dropped and not shown)

Every threshold below is a named constant so the strictness is easy to
tune in one place. Each axis is a 0-100 weighted blend.

Missing inputs are handled in two distinct ways, and the difference matters:
  * CONFIRMED ABSENT (no stock-loan market exists for this name; historical
    borrow data that was never recorded) — the component is dropped and the
    remaining weights renormalized, so a name isn't punished for a data
    source that cannot exist.
  * NOT YET LOADED (IBKR hasn't answered this refresh) — the component scores
    0 with its weight retained, so nothing is ever promoted on the strength
    of data that simply hasn't arrived. See pressure_score(borrow_pending=).
"""

# ---- how strict Prime / Subprime are (the main knobs, adjustable from the UI) ----
# Prime is the pair a user actually tunes; every Subprime boundary below is
# derived from that pair at the SAME ratios these defaults hold today, so the
# whole model scales consistently instead of exposing 7 confusing knobs.
PRIME_PRESSURE   = 55   # "loaded"  — needs real short-interest fuel
PRIME_IGNITION   = 50   # "firing"  — needs a real move underway
SUB_PRESSURE_HI  = 55   # loaded but not yet firing …
SUB_IGNITION_LO  = 25   #   … still shows early signs of a spark
SUB_PRESSURE_LO  = 35   # lighter fuel but …
SUB_IGNITION_HI  = 50   #   … clearly firing right now
SUB_LOADED_WATCH = 65   # so heavily shorted it's worth watching even while quiet

# Extreme pressure is Prime-grade on its own, with no ignition required.
#
# Rationale (tools/backtest_setups.py, run against real historical data):
# requiring ignition means Prime can only fire mid-squeeze, never before one
# — the day BEFORE a squeeze ignites, volume and price are quiet by
# definition. GME on 2021-01-12 scored Pressure 91 / Ignition 10 and then ran
# +1642% over the next 10 sessions; GME on 2021-01-21 scored 87/44 and ran
# +708%. Both were Subprime, never Prime.
#
# The bar is safe because extreme pressure simply does not occur in ordinary
# stocks: across 12 controls on quiet dates the HIGHEST pressure observed was
# 16, versus 87-91 for those setups — a 5x margin at this threshold.
#
# Honest caveat: only 3 historical cases in that sample cleared 80, and one of
# them (BBBY) went on to a far more modest +17.9%. This is a structurally
# motivated rule validated on a small sample, not a statistically fitted one.
PRIME_EXTREME_PRESSURE = 80

# The extreme path additionally requires genuinely heavy short interest.
# Without this guard it fires on the wrong stocks: when the borrow feed is
# down, pressure_score renormalizes over the components it still has, so a
# name with only moderate SI but a maxed DTC and short-volume ratio can reach
# 80+ on partial data alone. Observed live: WKHS (27.7% SI) and NTHI (25.3%)
# both hit 85 that way — not remotely the GME/BBBY (84%/82% SI) caliber of
# setup this rule exists to catch. Requiring the SI sub-score to be at least
# 85 (i.e. SI >= 30% of float) keeps the genuinely-loaded names and drops the
# renormalization artifacts.
_EXTREME_MIN_SI_SCORE = 85

# ratios locked to the defaults above — used to derive Subprime bounds
# whenever a caller supplies custom Prime thresholds
_SUB_IGNITION_LO_RATIO  = SUB_IGNITION_LO / PRIME_IGNITION           # 0.50
_SUB_PRESSURE_LO_RATIO  = SUB_PRESSURE_LO / PRIME_PRESSURE           # ~0.636
_SUB_LOADED_WATCH_DELTA = SUB_LOADED_WATCH - PRIME_PRESSURE          # +10
_EXTREME_PRESSURE_RATIO = PRIME_EXTREME_PRESSURE / PRIME_PRESSURE    # ~1.455


def _num(value):
    """Parse a number out of things like '20.5%', '4.2', 12.3, or None."""
    if value is None:
        return None
    try:
        return float(str(value).replace("%", "").replace("*", "").strip())
    except (TypeError, ValueError):
        return None


# ---------------- PRESSURE (structural fuel) ----------------

def _p_si(si):
    if si is None:   return 0
    if si < 10:      return 0
    if si < 20:      return 40
    if si < 30:      return 65
    if si < 50:      return 85
    return 100

def _p_dtc(dtc):
    if dtc is None:  return 0
    if dtc < 1:      return 0
    if dtc < 2:      return 30
    if dtc < 3:      return 50
    if dtc < 5:      return 75
    return 100

def _p_ctb(ctb):
    if ctb is None:  return 0
    if ctb < 5:      return 0
    if ctb < 10:     return 30
    if ctb < 20:     return 55
    if ctb < 50:     return 80
    return 100

def _p_shortable(shortable):
    return {"Easy": 0, "Medium": 40, "Hard": 75, "None!": 100}.get(shortable, 0)

def _p_shortvol(sv):
    if sv is None:   return 0
    if sv < 45:      return 0
    if sv < 55:      return 30
    if sv < 65:      return 60
    if sv < 75:      return 85
    return 100

def _p_float_mult(float_shares):
    """A tight float doesn't ADD fuel — it AMPLIFIES it. The same short
    interest is far more explosive on 2M shares than on 18M. So float acts
    as a multiplier on the whole pressure score, not another additive term
    (which would wrongly let a big-float low-SI name gain pressure for free)."""
    try:
        m = float(float_shares) / 1_000_000  # shares -> millions
    except (TypeError, ValueError):
        return 1.0
    if m <= 0:  return 1.0
    if m < 2:   return 1.15   # micro-float — biggest springs
    if m < 5:   return 1.08
    if m < 10:  return 1.00   # neutral
    if m < 15:  return 0.92
    return 0.85               # 15-20M — same SI is harder to squeeze

def pressure_score(si_live, dtc, ctb, shortable, short_vol, float_shares=None,
                   borrow_pending=False):
    """0-100. SI is the dominant fuel; borrow stress and DTC confirm it; a
    tight float amplifies the whole thing.

    Components with NO data available are excluded and the remaining weights
    renormalized — so a stock is never penalized just because a data source is
    genuinely unavailable (historical borrow data doesn't exist at all, which
    is exactly the case tools/backtest_setups.py runs under). Present-but-low
    values (e.g. 'Easy' to borrow) still count as real evidence of low
    pressure and are kept in.

    borrow_pending=True means the borrow feed simply hasn't answered YET, as
    opposed to having confirmed there's no lending market. Those components
    then score 0 with their weight RETAINED, so a stock can never be promoted
    on the strength of data that hasn't loaded. This matters because
    renormalization is not direction-neutral: dropping a weak-but-real CTB
    raises the average, which was pushing cheap-to-borrow names (JACK at
    0.8% CTB: 58 -> 68) over the Subprime bar purely on missing data. Once
    the value lands the score settles, and it settles DOWNWARD only, so a
    setup is never advertised before the evidence supports it."""
    si  = _num(si_live)
    dtc = _num(dtc)
    ctb = _num(ctb)
    sv  = _num(short_vol)
    sh  = _p_shortable(shortable) if shortable in ("Easy", "Medium", "Hard", "None!") else None

    # unknown-yet counts as zero evidence (weight kept); confirmed-absent is
    # excluded entirely (weight renormalized away)
    missing = 0 if borrow_pending else None
    ctb_score = _p_ctb(ctb) if ctb is not None else missing
    sh_score  = sh          if sh  is not None else missing

    components = [
        (0.45, _p_si(si)       if si  is not None else None),
        (0.20, _p_dtc(dtc)     if dtc is not None else None),
        (0.15, ctb_score),
        (0.10, sh_score),
        (0.10, _p_shortvol(sv) if sv  is not None else None),
    ]
    avail = [(w, s) for w, s in components if s is not None]
    if not avail:
        return 0.0
    base = sum(w * s for w, s in avail) / sum(w for w, _ in avail)  # available-case weighting
    return min(100.0, base * _p_float_mult(float_shares))


# ---------------- IGNITION (the spark, now) ----------------

def _i_relvol(rv):
    if rv is None:   return 0
    if rv < 1.5:     return 0
    if rv < 3:       return 35
    if rv < 5:       return 60
    if rv < 10:      return 85
    return 100

def _i_change(ch):
    if ch is None or ch <= 0:  return 0   # squeezes move UP; down moves score 0
    if ch < 5:       return 25
    if ch < 10:      return 50
    if ch < 20:      return 80
    return 100

def _i_ttm(signal, mom_up):
    if signal == "Fired":
        return 100 if mom_up else 20      # fired downward isn't a squeeze
    if signal == "Squeezed":
        return 60                         # coiled, ready
    return 0

def ignition_score(rel_volume, change_pct, ttm_signal, ttm_mom_up):
    """0-100. Unusual volume + an up move + a firing/coiled TTM."""
    return (
        _i_relvol(_num(rel_volume)) * 0.40 +
        _i_change(_num(change_pct)) * 0.40 +
        _i_ttm(ttm_signal, ttm_mom_up) * 0.20
    )


# ---------------- classification ----------------

def si_qualifies_extreme(si_live):
    """Is short interest itself heavy enough to allow the extreme-pressure
    Prime path? Guards against that path firing on a score that only reached
    the bar through renormalization of partial data (see note above)."""
    return _p_si(_num(si_live)) >= _EXTREME_MIN_SI_SCORE


def classify(pressure, ignition, prime_pressure=PRIME_PRESSURE, prime_ignition=PRIME_IGNITION,
             extreme_eligible=False):
    """Return 'prime', 'subprime', or None (drop). Subprime bounds are derived
    from the given Prime thresholds at the module's default ratios, so
    retuning Prime keeps the same *shape* of criteria, just stricter/looser."""
    sub_pressure_hi  = prime_pressure
    sub_ignition_lo  = prime_ignition * _SUB_IGNITION_LO_RATIO
    sub_pressure_lo  = prime_pressure * _SUB_PRESSURE_LO_RATIO
    sub_ignition_hi  = prime_ignition
    sub_loaded_watch = prime_pressure + _SUB_LOADED_WATCH_DELTA
    extreme_pressure = prime_pressure * _EXTREME_PRESSURE_RATIO

    if pressure >= prime_pressure and ignition >= prime_ignition:
        return "prime"                      # loaded AND firing
    if extreme_eligible and pressure >= extreme_pressure:
        return "prime"                      # so loaded it qualifies before firing
    if pressure >= sub_pressure_hi and ignition >= sub_ignition_lo:
        return "subprime"                   # loaded, early spark
    if pressure >= sub_pressure_lo and ignition >= sub_ignition_hi:
        return "subprime"                   # lighter fuel, clearly firing
    if pressure >= sub_loaded_watch:
        return "subprime"                   # so heavily shorted it's worth watching
    return None


def evaluate(si_live, dtc, ctb, shortable, short_vol,
             rel_volume, change_pct, ttm_signal, ttm_mom_up, float_shares=None,
             prime_pressure=PRIME_PRESSURE, prime_ignition=PRIME_IGNITION,
             borrow_pending=False):
    """Convenience: compute both axes and the label in one call."""
    p = pressure_score(si_live, dtc, ctb, shortable, short_vol, float_shares,
                       borrow_pending=borrow_pending)
    i = ignition_score(rel_volume, change_pct, ttm_signal, ttm_mom_up)
    return {"pressure": round(p), "ignition": round(i),
            "setup": classify(p, i, prime_pressure, prime_ignition,
                              extreme_eligible=si_qualifies_extreme(si_live))}
