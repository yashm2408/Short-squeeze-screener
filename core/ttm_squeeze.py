"""
TTM Squeeze (John Carter) — full implementation with momentum histogram.

Formula (matches the standard open-source version of the indicator, i.e.
the same math TradingView's widely-used replica runs, which is what most
people compare against):

    Bollinger Bands:  SMA(close, 20) +/- 2.0 * stdev(close, 20)   [population]
    Keltner Channels: SMA(close, 20) +/- mult * SMA(TrueRange, 20)

    Squeeze ON  when both BBs are inside the KCs  (volatility coiled)
    Squeeze OFF when both BBs are outside the KCs (volatility released)

    Momentum = linear-regression endpoint over 20 bars of:
        close - midline,  where
        midline = avg( avg(highest(high,20), lowest(low,20)), SMA(close,20) )

    Momentum > 0 and rising  -> pressure building upward (squeeze fuel bullish)
    Momentum < 0 and falling -> leaning downward

Compression tiers (TTM Squeeze Pro-style upgrade): the tightest Keltner
multiplier that still contains the Bollinger Bands.

    HIGH compression: BB inside KC(1.0)  — tightest coil, biggest springs
    MID  compression: BB inside KC(1.5)  — the classic squeeze
    LOW  compression: BB inside KC(2.0)  — early warning only

The classic 1.5x KC is what defines "in squeeze" / "fired"; tiers only
add texture on top of it.
"""
import numpy as np
import pandas as pd

LENGTH = 20          # shared period for BB, KC, ATR and momentum
BB_MULT = 2.0
KC_MULTS = {"HIGH": 1.0, "MID": 1.5, "LOW": 2.0}
FIRE_LOOKBACK = 5    # a fire older than this many bars is stale -> "Off"
MIN_BARS = LENGTH * 2  # need indicator warm-up + a momentum window


def _linreg_endpoint(series, length):
    """Rolling endpoint of the OLS regression line (TradingView linreg(..., 0))."""
    x = np.arange(length, dtype=float)

    def endpoint(window):
        slope, intercept = np.polyfit(x, window, 1)
        return intercept + slope * (length - 1)

    return series.rolling(length).apply(endpoint, raw=True)


def compute_ttm(bars):
    """
    bars: iterable of objects with .high/.low/.close (ib_insync BarData works).
    Returns a dict:
        signal          'Squeezed' | 'Fired' | 'Off' | '—'
        compression     'HIGH' | 'MID' | 'LOW' | None   (while squeezed)
        bars_in_squeeze consecutive bars squeezed (while squeezed)
        fired_bars_ago  bars since squeeze release (while fired)
        momentum        latest histogram value (float) or None
        momentum_pct    momentum as a % of price (comparable across tickers)
        mom_up          momentum > 0
        mom_rising      histogram increasing vs previous bar
        display         compact string for the UI column — always numeric
    """
    empty = {"signal": "—", "compression": None, "bars_in_squeeze": 0,
             "fired_bars_ago": None, "momentum": None, "momentum_pct": None,
             "mom_up": None, "mom_rising": None, "display": "—"}
    if not bars or len(bars) < MIN_BARS:
        return empty

    df = pd.DataFrame([{"high": b.high, "low": b.low, "close": b.close} for b in bars])
    close, high, low = df["close"], df["high"], df["low"]

    basis = close.rolling(LENGTH).mean()
    dev = BB_MULT * close.rolling(LENGTH).std(ddof=0)  # population stdev, like the reference
    bb_upper, bb_lower = basis + dev, basis - dev

    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(LENGTH).mean()

    sq_on = {}
    for tier, mult in KC_MULTS.items():
        kc_upper = basis + mult * atr
        kc_lower = basis - mult * atr
        sq_on[tier] = (bb_upper < kc_upper) & (bb_lower > kc_lower)

    on = sq_on["MID"]  # classic 1.5x defines the squeeze state

    midline = (pd.concat([high.rolling(LENGTH).max(),
                          low.rolling(LENGTH).min()], axis=1).mean(axis=1) + basis) / 2
    mom = _linreg_endpoint(close - midline, LENGTH)

    momentum = mom.iloc[-1]
    momentum = None if pd.isna(momentum) else float(momentum)
    mom_prev = mom.iloc[-2] if len(mom) >= 2 and not pd.isna(mom.iloc[-2]) else None
    # neutral when the histogram is a hair from zero (dead-flat tape)
    _eps = max(1e-9, 1e-5 * abs(close.iloc[-1]))
    mom_up = (momentum > 0) if momentum is not None and abs(momentum) > _eps else None
    mom_rising = (momentum > mom_prev) if momentum is not None and mom_prev is not None else None

    # momentum normalized as a % of price — comparable across any ticker,
    # unlike the raw histogram value which scales with the stock's own price
    last_close = close.iloc[-1]
    momentum_pct = (momentum / last_close * 100) if (momentum is not None and last_close) else None
    mom_str = f"{momentum_pct:+.1f}%" if momentum_pct is not None else "n/a"

    # --- currently squeezed ---
    if bool(on.iloc[-1]):
        run = 0
        for v in reversed(on.tolist()):
            if v:
                run += 1
            else:
                break
        tier = "MID"
        if bool(sq_on["HIGH"].iloc[-1]):
            tier = "HIGH"
        result = {"signal": "Squeezed", "compression": tier, "bars_in_squeeze": run,
                  "fired_bars_ago": None, "momentum": momentum, "momentum_pct": momentum_pct,
                  "mom_up": mom_up, "mom_rising": mom_rising}
        label = "SQZ·HI" if tier == "HIGH" else "SQZ"
        result["display"] = f"{label} {run}d {mom_str}"
        return result

    # --- recently fired? (was on, released within lookback) ---
    on_list = on.tolist()
    for k in range(1, FIRE_LOOKBACK + 1):
        if len(on_list) > k and on_list[-1 - k]:
            # bar -1-k was squeezed; make sure every bar after it is released
            if not any(on_list[-k:]):
                result = {"signal": "Fired", "compression": None, "bars_in_squeeze": 0,
                          "fired_bars_ago": k, "momentum": momentum, "momentum_pct": momentum_pct,
                          "mom_up": mom_up, "mom_rising": mom_rising}
                result["display"] = f"FIRE {k}d {mom_str}"
                return result
            break

    return {"signal": "Off", "compression": None, "bars_in_squeeze": 0,
            "fired_bars_ago": None, "momentum": momentum, "momentum_pct": momentum_pct,
            "mom_up": mom_up, "mom_rising": mom_rising, "display": f"Off {mom_str}"}
