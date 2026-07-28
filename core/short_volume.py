"""
Daily short sale volume (the data CBOE sells on DataShop, from the free
FINRA consolidated feed that covers all exchanges including CBOE's four).

File format (pipe-delimited), published nightly ~6pm ET:
    Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market

Short Volume % = ShortVolume / TotalVolume * 100 for the most recent
trading day. A ratio persistently above ~60% signals heavy intraday
shorting pressure — the best free daily proxy for rising short interest
between FINRA's bi-monthly SI reports.
"""
import threading
import time
from datetime import date, timedelta

import requests

_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ymd}.txt"

_lock = threading.Lock()
_data = {}          # {"AAPL": 42.1, ...}
_data_date = ""     # YYYYMMDD of the loaded file
_last_attempt = 0.0
_RETRY_SECONDS = 3600  # re-check hourly so the nightly file gets picked up


def _download_latest():
    """Try today then walk back a few days (weekends/holidays/not yet published)."""
    for days_back in range(6):
        d = date.today() - timedelta(days=days_back)
        ymd = d.strftime("%Y%m%d")
        try:
            r = requests.get(_URL.format(ymd=ymd), timeout=20,
                             headers={"User-Agent": "Mozilla/5.0"})
        except Exception as e:
            print(f"[ShortVol] download error: {e}")
            return None, ""
        if r.status_code != 200 or not r.text.startswith("Date|"):
            continue
        parsed = {}
        for line in r.text.splitlines()[1:]:
            parts = line.split("|")
            if len(parts) < 5:
                continue
            try:
                short_vol = float(parts[2])
                total_vol = float(parts[4])
                if total_vol > 0:
                    parsed[parts[1]] = round(short_vol / total_vol * 100, 1)
            except ValueError:
                continue
        print(f"[ShortVol] Loaded {len(parsed)} symbols for {ymd}")
        return parsed, ymd
    print("[ShortVol] No file found in the last 6 days")
    return None, ""


def ensure_loaded():
    """Download/refresh the daily file. Call from a background thread only."""
    global _data, _data_date, _last_attempt
    now = time.time()
    with _lock:
        if _data and now - _last_attempt < _RETRY_SECONDS:
            return
        _last_attempt = now
    parsed, ymd = _download_latest()
    if parsed:
        with _lock:
            if ymd >= _data_date:
                _data, _data_date = parsed, ymd


def get_short_volume_pct(ticker):
    """Non-blocking cache read. Returns e.g. '63.4%' or '—'."""
    with _lock:
        val = _data.get(ticker)
    return f"{val}%" if val is not None else "—"


def get_data_date():
    return _data_date
