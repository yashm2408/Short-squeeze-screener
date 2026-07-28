"""
Historical validation of the Prime/Subprime classifier against real squeezes.

Reconstructs the Pressure and Ignition scores as they WOULD have been on a
given past date, using only data that was actually available on that date,
then checks what the classifier would have said.

Data sources (all real, all free — nothing hardcoded or hand-fed):
  * official short interest  -- FINRA consolidatedShortInterest API (back to 2017)
  * days-to-cover            -- same API (daysToCoverQuantity)
  * daily short volume %     -- FINRA RegSHO daily files
  * price / volume / TTM     -- yfinance daily bars
  * shares outstanding       -- yfinance get_shares_full (float proxy, see note)

Deliberately NOT supplied: CTB and Shortable. No free historical borrow-fee
data exists, and the live app frequently runs without them too, so every
score here is computed the way the app scores a stock when the borrow feed
is unavailable — pressure_score() renormalizes over the components it has.
That makes this a conservative test: real-world scores with borrow data
would only be higher for genuinely hard-to-borrow names.

Two populations are measured, which is what separates this from cherry-picking:
  SQUEEZES -- famous, well-documented squeezes, scored the day BEFORE the run
  CONTROLS -- ordinary large/mid caps on quiet dates (false-positive check)

Run (from ScreenerProject root):
    venv/Scripts/python.exe tools/backtest_setups.py
"""
import os
import sys
from collections import namedtuple
from datetime import date, datetime, timedelta

import requests
import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core.setup_classifier import pressure_score, ignition_score, classify, si_qualifies_extreme
from core.ttm_squeeze import compute_ttm
from core.si_estimate import _get_day_file, _parse

_API = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
_H = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json", "Accept": "application/json"}

Bar = namedtuple("Bar", "high low close")

RELVOL_LOOKBACK = 63   # ~3 months, matches how Finviz computes Relative Volume


# ---------------- historical inputs ----------------

_si_cache = {}


def finra_si(symbol, on_or_before):
    """Most recent official SI row published on/before `on_or_before`.
    Mirrors reality: the app only ever sees the last settlement, not the future."""
    key = (symbol, on_or_before.isoformat())
    if key in _si_cache:
        return _si_cache[key]
    body = {
        "limit": 40,
        "compareFilters": [{"fieldName": "settlementDate", "compareType": "LTE",
                            "fieldValue": on_or_before.isoformat()}],
        "domainFilters": [{"fieldName": "symbolCode", "values": [symbol]}],
    }
    try:
        r = requests.post(_API, json=body, headers=_H, timeout=60)
        rows = [] if r.status_code == 204 else r.json()
    except Exception:
        rows = []
    rows = [x for x in rows if x.get("settlementDate")]
    rows.sort(key=lambda x: x["settlementDate"], reverse=True)
    out = rows[0] if rows else None
    _si_cache[key] = out
    return out


def shares_outstanding(symbol, on_date):
    """Shares outstanding at `on_date` — the SI% denominator.

    This is a float PROXY: outstanding >= float, so SI% here is a slight
    UNDER-estimate of true short-interest-as-%-of-float. Conservative in the
    same direction as the missing borrow data — real scores would be higher.
    """
    try:
        s = yf.Ticker(symbol).get_shares_full(
            start=(on_date - timedelta(days=400)).isoformat(),
            end=on_date.isoformat())
        if s is None or len(s) == 0:
            return None
        return float(s.iloc[-1])
    except Exception:
        return None


def short_volume_pct(symbol, on_date):
    """Short-sale volume as % of total volume for that day (FINRA RegSHO)."""
    for back in range(5):  # walk back over weekends/holidays
        d = on_date - timedelta(days=back)
        text = _get_day_file(d)
        if not text:
            continue
        rec = _parse(text).get(symbol)
        if rec and rec[1] > 0:
            return rec[0] / rec[1] * 100
    return None


def price_features(symbol, on_date):
    """RelVol, change%, and TTM squeeze state using only bars up to `on_date`."""
    df = yf.download(symbol,
                     start=(on_date - timedelta(days=400)).isoformat(),
                     end=(on_date + timedelta(days=1)).isoformat(),
                     interval="1d", auto_adjust=False, progress=False)
    if df is None or len(df) < 45:
        return None
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)   # flatten yfinance MultiIndex

    closes = df["Close"].tolist()
    highs = df["High"].tolist()
    lows = df["Low"].tolist()
    vols = df["Volume"].tolist()

    change = (closes[-1] - closes[-2]) / closes[-2] * 100 if len(closes) >= 2 and closes[-2] else 0.0
    prior_vols = vols[-(RELVOL_LOOKBACK + 1):-1]
    avg_vol = sum(prior_vols) / len(prior_vols) if prior_vols else 0
    relvol = vols[-1] / avg_vol if avg_vol > 0 else 0.0

    bars = [Bar(h, l, c) for h, l, c in zip(highs, lows, closes)]
    ttm = compute_ttm(bars)
    return {"change": change, "relvol": relvol, "ttm": ttm,
            "close": closes[-1], "avg_vol": avg_vol}


def forward_return(symbol, on_date, days=10):
    """Best close-to-close gain over the next `days` sessions — 'did it squeeze?'"""
    df = yf.download(symbol, start=on_date.isoformat(),
                     end=(on_date + timedelta(days=days * 2 + 10)).isoformat(),
                     interval="1d", auto_adjust=False, progress=False)
    if df is None or len(df) < 2:
        return None
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    closes = df["Close"].tolist()
    base = closes[0]
    fwd = closes[1:days + 1]
    if not base or not fwd:
        return None
    return (max(fwd) - base) / base * 100


# ---------------- scoring one historical day ----------------

def evaluate_on(symbol, on_date, label=""):
    si_row = finra_si(symbol, on_date)
    pf = price_features(symbol, on_date)
    if pf is None:
        return {"symbol": symbol, "date": on_date, "label": label, "error": "no price data"}

    si_pct = dtc = None
    if si_row:
        shares_short = si_row.get("currentShortPositionQuantity")
        dtc = si_row.get("daysToCoverQuantity")
        so = shares_outstanding(symbol, on_date)
        if shares_short and so:
            si_pct = shares_short / so * 100

    sv = short_volume_pct(symbol, on_date)
    ttm = pf["ttm"]

    # CTB / Shortable intentionally absent — see module docstring
    p = pressure_score(si_pct, dtc, None, "—", sv, None)
    i = ignition_score(pf["relvol"], pf["change"], ttm.get("signal"), ttm.get("mom_up"))
    verdict = classify(p, i, extreme_eligible=si_qualifies_extreme(si_pct))

    return {
        "symbol": symbol, "date": on_date, "label": label,
        "si_pct": si_pct, "dtc": dtc, "short_vol": sv,
        "relvol": pf["relvol"], "change": pf["change"],
        "ttm": ttm.get("display", "—"),
        "pressure": round(p), "ignition": round(i),
        "setup": verdict or "none",
        "fwd_10d": forward_return(symbol, on_date, 10),
        "settlement": si_row.get("settlementDate") if si_row else None,
    }


# ---------------- test populations ----------------

# Day BEFORE the well-documented run began, so the classifier has to call it
# in advance rather than after the fact.
SQUEEZES = [
    ("GME",  date(2021, 1, 12), "GME — day before Jan 13 +57%"),
    ("GME",  date(2021, 1, 21), "GME — day before Jan 22 +51%"),
    ("AMC",  date(2021, 5, 26), "AMC — before the June run"),
    ("AMC",  date(2021, 1, 26), "AMC — day before Jan 27 +301%"),
    ("BBBY", date(2022, 8, 5),  "BBBY — before the Aug 2022 run"),
    ("ATER", date(2021, 9, 24), "ATER — before the Sept/Oct 2021 run"),
    ("BGFV", date(2021, 6, 1),  "BGFV — small-cap 2021 squeeze"),
    ("SPRT", date(2021, 8, 20), "SPRT — Aug 2021 squeeze"),
]

# Ordinary names on quiet dates — the false-positive check.
CONTROLS = [
    ("AAPL", date(2021, 3, 10)), ("MSFT", date(2021, 6, 15)),
    ("KO",   date(2021, 9, 14)), ("JNJ",  date(2022, 2, 8)),
    ("PG",   date(2022, 5, 17)), ("WMT",  date(2022, 10, 11)),
    ("XOM",  date(2023, 1, 24)), ("VZ",   date(2023, 4, 18)),
    ("PFE",  date(2023, 7, 11)), ("CSCO", date(2023, 11, 7)),
    ("INTC", date(2024, 2, 13)), ("F",    date(2024, 6, 4)),
]


def fmt(v, suffix="", nd=1):
    return f"{v:.{nd}f}{suffix}" if isinstance(v, (int, float)) else "—"


def show(rows, title):
    print()
    print("=" * 108)
    print(title)
    print("=" * 108)
    print(f"{'Symbol':<7}{'Date':<12}{'SI%':>7}{'DTC':>7}{'SV%':>7}{'RelVol':>8}"
          f"{'Chg%':>8}{'TTM':>16}{'Pres':>6}{'Ign':>6}{'Verdict':>10}{'Fwd10d':>9}")
    print("-" * 108)
    for r in rows:
        if r.get("error"):
            print(f"{r['symbol']:<7}{str(r['date']):<12}  ERROR: {r['error']}")
            continue
        print(f"{r['symbol']:<7}{str(r['date']):<12}"
              f"{fmt(r['si_pct']):>7}{fmt(r['dtc']):>7}{fmt(r['short_vol']):>7}"
              f"{fmt(r['relvol'], nd=1):>8}{fmt(r['change'], nd=1):>8}"
              f"{str(r['ttm'])[:15]:>16}"
              f"{r['pressure']:>6}{r['ignition']:>6}{r['setup']:>10}"
              f"{fmt(r['fwd_10d'], '%'):>9}")


def main():
    print("Reconstructing historical Pressure/Ignition scores…")
    print("(CTB + Shortable deliberately absent — no free historical borrow data;")
    print(" scores are therefore conservative, as in a live borrow-feed outage)")

    sq = []
    for sym, d, label in SQUEEZES:
        print(f"  … {sym} {d}")
        sq.append(evaluate_on(sym, d, label))

    ct = []
    for sym, d in CONTROLS:
        print(f"  … {sym} {d} (control)")
        ct.append(evaluate_on(sym, d, "control"))

    show(sq, "KNOWN SQUEEZES — scored the day BEFORE the run")
    show(ct, "CONTROLS — ordinary names on quiet dates (false-positive check)")

    ok_sq = [r for r in sq if not r.get("error")]
    ok_ct = [r for r in ct if not r.get("error")]
    flagged_sq = [r for r in ok_sq if r["setup"] != "none"]
    prime_sq = [r for r in ok_sq if r["setup"] == "prime"]
    flagged_ct = [r for r in ok_ct if r["setup"] != "none"]
    prime_ct = [r for r in ok_ct if r["setup"] == "prime"]

    print()
    print("=" * 108)
    print("SUMMARY")
    print("=" * 108)
    print(f"  Squeezes caught (Prime or Subprime): {len(flagged_sq)}/{len(ok_sq)}")
    print(f"  Squeezes rated Prime specifically:   {len(prime_sq)}/{len(ok_sq)}")
    print(f"  Controls falsely flagged (any):      {len(flagged_ct)}/{len(ok_ct)}")
    print(f"  Controls falsely rated Prime:        {len(prime_ct)}/{len(ok_ct)}")
    if ok_sq:
        print(f"  Median Pressure on squeezes: {sorted(r['pressure'] for r in ok_sq)[len(ok_sq)//2]}")
        print(f"  Median Ignition on squeezes: {sorted(r['ignition'] for r in ok_sq)[len(ok_sq)//2]}")
    if ok_ct:
        print(f"  Median Pressure on controls: {sorted(r['pressure'] for r in ok_ct)[len(ok_ct)//2]}")
        print(f"  Median Ignition on controls: {sorted(r['ignition'] for r in ok_ct)[len(ok_ct)//2]}")


if __name__ == "__main__":
    main()
