"""
Tests for core/ttm_squeeze.py using synthetic bars — no IB Gateway needed.

Run:  python tests/test_ttm_squeeze.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.ttm_squeeze import compute_ttm, MIN_BARS


class Bar:
    def __init__(self, high, low, close):
        self.high, self.low, self.close = high, low, close


def flat_bars(n, price=100.0, wick=0.05):
    """Dead-calm tape: tiny range -> BB collapse inside KC -> squeeze ON."""
    return [Bar(price + wick, price - wick, price) for _ in range(n)]


def trending_bars(n, price=100.0, step=2.0):
    """Steady trend: rolling stdev of closes inflates with the trend while
    per-bar true range stays ~step, so BB expand outside KC -> squeeze OFF.
    (This is exactly why the indicator fires on breakouts.)"""
    bars = []
    p = price
    for _ in range(n):
        p += step
        bars.append(Bar(p + step / 2, p - step / 2, p))
    return bars


def test_too_few_bars():
    out = compute_ttm(flat_bars(MIN_BARS - 1))
    assert out["signal"] == "—", out
    print("PASS: too few bars -> '—'")


def test_squeeze_on():
    out = compute_ttm(flat_bars(60))
    assert out["signal"] == "Squeezed", out
    assert out["compression"] in ("HIGH", "MID"), out
    assert out["bars_in_squeeze"] > 0, out
    assert "SQZ" in out["display"], out
    print(f"PASS: flat tape -> {out['display']}")


def test_squeeze_off():
    out = compute_ttm(trending_bars(60))
    assert out["signal"] == "Off", out
    print(f"PASS: trending tape -> {out['display']}")


def test_fire_up():
    """Long calm coil, then a hard 3-day breakout up -> Fired with mom_up."""
    bars = flat_bars(70, price=100.0)
    p = 100.0
    for _ in range(3):
        p *= 1.18  # +18% per day pops the BBs wide open
        bars.append(Bar(p * 1.01, p * 0.97, p))
    out = compute_ttm(bars)
    assert out["signal"] == "Fired", out
    assert out["fired_bars_ago"] is not None and out["fired_bars_ago"] <= 3, out
    assert out["mom_up"] is True, out
    # display carries the signed momentum % (it replaced the old ↑/↓ arrows)
    assert "FIRE" in out["display"] and "+" in out["display"], out
    print(f"PASS: coil + breakout -> {out['display']}")


def test_fire_down():
    """Coil then a crash down -> Fired with mom_up False."""
    bars = flat_bars(70, price=100.0)
    p = 100.0
    for _ in range(3):
        p *= 0.82
        bars.append(Bar(p * 1.03, p * 0.99, p))
    out = compute_ttm(bars)
    assert out["signal"] == "Fired", out
    assert out["mom_up"] is False, out
    assert "-" in out["display"], out
    print(f"PASS: coil + breakdown -> {out['display']}")


def test_momentum_sign_tracks_trend():
    """Steady uptrend -> positive momentum; steady downtrend -> negative."""
    up = [Bar(p + 0.5, p - 0.5, p) for p in [100 + 0.8 * i for i in range(60)]]
    down = [Bar(p + 0.5, p - 0.5, p) for p in [160 - 0.8 * i for i in range(60)]]
    mo_up = compute_ttm(up)["momentum"]
    mo_down = compute_ttm(down)["momentum"]
    assert mo_up is not None and mo_up > 0, mo_up
    assert mo_down is not None and mo_down < 0, mo_down
    print(f"PASS: momentum sign (up={mo_up:.2f}, down={mo_down:.2f})")


def test_score_integration():
    """squeeze_score consumes the new dict and still returns the old keys."""
    from core.squeeze_score import compute_squeeze_score
    out = compute_squeeze_score("FAKE_TICKER_NO_BARS", "35%", "6.2", "3.1", "4.5%",
                                ctb="25%", short_vol="62%")
    assert "squeeze_score" in out and "ttm_signal" in out, out
    assert isinstance(out["squeeze_score"], int), out
    assert out["ttm_signal"] == "—", out  # no cached bars for a fake ticker
    print(f"PASS: score integration -> {out}")


if __name__ == "__main__":
    test_too_few_bars()
    test_squeeze_on()
    test_squeeze_off()
    test_fire_up()
    test_fire_down()
    test_momentum_sign_tracks_trend()
    test_score_integration()
    print("\nAll TTM Squeeze tests passed.")
