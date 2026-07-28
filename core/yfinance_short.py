import time
import yfinance as yf

_cache = {}
_CACHE_TTL = 3600  # 1 hour — Yahoo/FINRA updates weekly, no point hitting more often


def get_short_data(ticker):
    """Non-blocking — returns cache instantly."""
    return _cache.get(ticker, {"si_pct": "—", "dtc": "—"})


def fetch_short_data_async(ticker):
    """
    Fetches SI% and Days to Cover from yfinance (Yahoo Finance / FINRA).
    Data updates weekly. Cache for 1 hour to avoid hammering Yahoo.
    Call from background thread only.
    """
    now = time.time()
    cached = _cache.get(ticker)
    if cached and (now - cached.get("_ts", 0)) < _CACHE_TTL:
        return

    result = {"si_pct": "—", "dtc": "—", "_ts": now}
    try:
        info = yf.Ticker(ticker).info

        si = info.get("shortPercentOfFloat")
        if si is not None:
            result["si_pct"] = f"{si * 100:.1f}%"

        dtc = info.get("shortRatio")
        if dtc is not None:
            result["dtc"] = f"{dtc:.1f}"

    except Exception as e:
        print(f"[YF Short] {ticker}: {e}")

    _cache[ticker] = result
