"""
Tests for the SI% (Live) estimator's per-day turnover cap (core/si_estimate.py).

Run:  python tests/test_si_estimate.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.si_estimate import _cap_day_turnover, _MAX_DAILY_TURNOVER
import core.si_estimate as si


def test_no_cap_when_under_threshold():
    sv, tv = 40_000, 100_000
    csv, ctv = _cap_day_turnover(sv, tv, float_shares=1_000_000)  # 0.1x turnover
    assert (csv, ctv) == (sv, tv), (csv, ctv)
    print("PASS: under-threshold day passes through unchanged")


def test_cap_preserves_ratio():
    sv, tv, float_shares = 4_492_690, 7_913_219, 490_000  # real BNRG day, ~16x turnover
    csv, ctv = _cap_day_turnover(sv, tv, float_shares)
    assert ctv == _MAX_DAILY_TURNOVER * float_shares, ctv
    assert abs(csv / ctv - sv / tv) < 1e-9, "capping must preserve that day's short-volume ratio"
    assert ctv < tv and csv < sv, "capped day must be strictly smaller in magnitude"
    print(f"PASS: 16x-turnover day capped to {_MAX_DAILY_TURNOVER}x, ratio preserved "
          f"({csv/ctv:.3f} == {sv/tv:.3f})")


def test_no_float_no_cap():
    sv, tv = 5_000_000, 8_000_000
    for bad in (None, 0, -1):
        csv, ctv = _cap_day_turnover(sv, tv, bad)
        assert (csv, ctv) == (sv, tv), (bad, csv, ctv)
    print("PASS: missing/invalid float leaves the day uncapped (no false safety)")


def test_extreme_microfloat_raw_signal_is_damped():
    """A synthetic single-day mega-volume spike on a tiny float should
    produce a much smaller raw (pre-sanity-clamp) excess-shares signal when
    capped vs uncapped -- checked below the final estimate_si_pct() sanity
    band (0.3x-2.5x base), because for a sufficiently extreme case BOTH the
    capped and uncapped paths can independently saturate that band and
    collide on the same displayed number (this is real, observed behavior
    for very extreme names like BNRG -- see the conversation). That outer
    clamp is a deliberate, separate, pre-existing safety net; asserting
    through it here would make this test flaky on exactly the inputs it's
    meant to cover, and wouldn't actually verify the cap's own effect."""
    si.ensure_history_loaded()
    ticker = "__TEST_MICROFLOAT__"
    float_shares = 500_000
    huge_day = {"20990101": {ticker: (9_000_000, 15_000_000)}}  # 15M vol vs 500K float = 30x
    baseline_day = {f"2098010{i}": {ticker: (45_000, 100_000)} for i in range(1, 7)}  # normal SVR ~0.45

    with si._lock:
        orig_hist, orig_base = si._history, si._baseline_hist
        si._history = huge_day
        si._baseline_hist = baseline_day
    try:
        with si._lock:
            days = [(ymd, day.get(ticker)) for ymd, day in sorted(si._history.items())]
        rows = [(ymd, sv, tv) for ymd, r in days if r for sv, tv in [r] if tv > 0]
        baseline = 0.45

        capped_excess = sum(
            (csv - baseline * ctv)
            for _, sv, tv in rows
            for csv, ctv in [_cap_day_turnover(sv, tv, float_shares)]
        )
        uncapped_excess = sum(sv - baseline * tv for _, sv, tv in rows)
    finally:
        with si._lock:
            si._history, si._baseline_hist = orig_hist, orig_base

    assert uncapped_excess == 9_000_000 - baseline * 15_000_000, uncapped_excess
    assert capped_excess < uncapped_excess, (capped_excess, uncapped_excess)
    reduction = (1 - capped_excess / uncapped_excess) * 100
    assert reduction > 50, f"expected the 30x-turnover day to be damped by well over half, got {reduction:.0f}%"
    print(f"PASS: 30x-turnover synthetic day -> raw excess shares reduced {reduction:.0f}% "
          f"by the cap (uncapped={uncapped_excess:,.0f}, capped={capped_excess:,.0f})")


def test_calibration_uses_identical_cap_function():
    """tools/calibrate_si.py must import the SAME _cap_day_turnover used at
    runtime, not a reimplementation that can drift out of sync."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "calibrate_si.py")
    spec = importlib.util.spec_from_file_location("calibrate_si", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._cap_day_turnover is _cap_day_turnover, \
        "calibrate_si.py must import core.si_estimate._cap_day_turnover directly, not redefine it"
    print("PASS: calibration script shares the exact runtime cap function (no drift risk)")


if __name__ == "__main__":
    test_no_cap_when_under_threshold()
    test_cap_preserves_ratio()
    test_no_float_no_cap()
    test_extreme_microfloat_raw_signal_is_damped()
    test_calibration_uses_identical_cap_function()
    print("\nAll SI estimate tests passed.")
