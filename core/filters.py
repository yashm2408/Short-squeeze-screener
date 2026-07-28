import pandas as pd
from core.finviz_api import fetch_finviz_data

# Converts a percentage string like '12.3%' to float 12.3
def clean_percent(value):
    try:
        return float(value.strip('%'))
    except:
        return None

# Converts a float-in-millions value (e.g. 5.1 or 0.58) to an actual share count
def clean_float(value):
    try:
        return float(value) * 1_000_000
    except:
        return None

# Calculates percentage change from previous close
def change_from_close(price, prev_close):
    if pd.isna(price) or pd.isna(prev_close) or price == 0 or prev_close == 0:
        return 0
    return ((price - prev_close) / prev_close) * 100


# Applies all filters to Finviz data to identify high-potential stocks
# (kept for tests/test_filters.py — not used by the live app)
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Previous Close"] = pd.to_numeric(df["Prev Close"], errors="coerce")
    df["Change%"] = df.apply(lambda row: change_from_close(row["Price"], row["Previous Close"]), axis=1)
    df["Rel Volume"] = pd.to_numeric(df["Relative Volume"], errors="coerce")
    df["Float"] = df["Shares Float"].apply(clean_float)

    if "Short Float" in df.columns:
        df["Short Float"] = df["Short Float"].apply(clean_percent)

    filtered_df = df[
        (df["Price"] >= 2) &
        (df["Price"] <= 20) &
        (df["Change%"] >= 10) &
        (df["Rel Volume"] >= 5)
    ]

    if "Short Float" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Short Float"] >= 5]

    return filtered_df.reset_index(drop=True)


def prepare_finviz_df():
    """One shared Finviz fetch + column prep, reused by both the Stage-1
    candidate pre-filter (My Prime Setup) and the legacy classifier
    (Previous Prime Setup) so a refresh only ever hits Finviz once."""
    df = fetch_finviz_data()
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Previous Close"] = pd.to_numeric(df["Prev Close"], errors="coerce")
    df["Change%"] = df.apply(lambda r: change_from_close(r["Price"], r["Previous Close"]), axis=1)
    df["Rel Volume"] = pd.to_numeric(df["Relative Volume"], errors="coerce")
    df["Float"] = df["Shares Float"].apply(clean_float)
    df["SI_num"] = df["Short Float"].apply(clean_percent) if "Short Float" in df.columns else 0
    return df.dropna(subset=["Float", "Price"])


def _row_to_stock_dict(row, si_num, change, relvol):
    """One shared per-row builder — used for BOTH systems' picks so a stock
    looks identical (same fields, same later enrichment) no matter which
    classifier selected it. The only thing that differs between the two
    systems is which stocks get selected, never what data is shown for them."""
    try:
        vol_w = float(str(row.get("Volatility (Week)", 0)).replace("%", ""))
    except Exception:
        vol_w = 0
    try:
        rsi = float(row.get("Relative Strength Index (14)", 50))
    except Exception:
        rsi = 50
    try:
        avg_vol_shares = float(row.get("Average Volume", 0)) * 1000  # Finviz exports in thousands
    except Exception:
        avg_vol_shares = 0

    raw_dtc = row.get("Short Ratio", "")
    dtc_num = None
    try:
        dtc_num = float(raw_dtc)
    except Exception:
        pass

    return {
        "Ticker": row.get("Ticker", ""),
        "Price": row.get("Price", ""),
        "Float": row.get("Shares Float", ""),
        "FloatShares": row.get("Float"),
        "AvgVolume": avg_vol_shares,
        "RelVolume": row.get("Relative Volume", ""),
        "RelVol_num": relvol,
        "ChangePercent": row.get("Change", ""),
        "Change_num": change,
        "SI_Pct": f"{si_num}%" if si_num else "—",
        "SI_official_num": si_num,
        "DTC": str(round(dtc_num, 1)) if dtc_num is not None else "—",
        "DTC_num": dtc_num,
        "Target": round(0.7 * vol_w + 0.03 * (70 - rsi), 2),
        "StopLoss": round((0.3 * vol_w - 0.02 * (rsi - 50)) * -1, 2),
    }


# ---- Stage 1: cheap Finviz-only pre-filter (feeds "My Prime Setup") ----
# Narrows the whole float<20M / price<20 universe (~1,500 names) down to the
# handful worth the expensive enrichment (IBKR borrow + bars, SI-Live). A name
# is worth a closer look if it is already heavily shorted OR is actively moving
# today; the final Prime/Subprime call is made later, after enrichment, by
# core.setup_classifier using the full signal set.
CANDIDATE_MIN_PRICE   = 1.0    # drop sub-$1 penny junk (unreliable to borrow/short)
CANDIDATE_MIN_SI      = 8.0    # "already shorted" gate (official SI %)
CANDIDATE_MIN_RELVOL  = 2.0    # "moving" gate: unusual volume …
CANDIDATE_MIN_CHANGE  = 3.0    #   … and an up move today
CANDIDATE_CAP         = 50     # most we enrich per refresh (keeps it fast)


def get_candidate_stocks(df=None):
    """Return a flat list of enrichment-worthy candidate dicts (Stage 1)."""
    try:
        if df is None:
            df = prepare_finviz_df()

        candidates = []
        for _, row in df.iterrows():
            price   = row["Price"]
            change  = row["Change%"] if pd.notna(row["Change%"]) else 0
            relvol  = row["Rel Volume"] if pd.notna(row["Rel Volume"]) else 0
            si_num  = row["SI_num"] if pd.notna(row["SI_num"]) else 0

            if price < CANDIDATE_MIN_PRICE:
                continue
            loaded = si_num >= CANDIDATE_MIN_SI
            moving = relvol >= CANDIDATE_MIN_RELVOL and change >= CANDIDATE_MIN_CHANGE
            if not (loaded or moving):
                continue

            stock = _row_to_stock_dict(row, si_num, change, relvol)
            # priority for the cap: surface both loaded and moving names
            stock["_priority"] = si_num + max(change, 0) * 0.5 + relvol
            candidates.append(stock)

        candidates.sort(key=lambda s: s["_priority"], reverse=True)
        return candidates[:CANDIDATE_CAP]
    except Exception as e:
        print(f"[Finviz] Error building candidates: {e}")
        return []


# ---- "Previous Prime Setup" — the original 4-condition rule ----
# Kept intentionally simple, exactly as it was originally: a flat point count
# over 4 independent checks (price/change/relvol/SI% only). Thresholds are
# parameters (not fixed constants) so the UI can let a user retune them
# without changing the logic. Returns stock DICTS (same shape as
# get_candidate_stocks) rather than pre-formatted rows — the controller
# enriches and displays them with the exact same columns as "My Prime Setup",
# so the only real difference between the two systems is the selection rule.
LEGACY_DEFAULTS = {
    "price_min": 2.0, "price_max": 20.0,
    "change_min": 10.0, "relvol_min": 5.0, "si_min": 5.0,
}


def legacy_classify(df=None, price_min=2.0, price_max=20.0,
                     change_min=10.0, relvol_min=5.0, si_min=5.0):
    """Original Prime/Subprime rule: pass all 4 checks -> Prime, exactly 3 -> Subprime."""
    try:
        if df is None:
            df = prepare_finviz_df()

        prime, subprime = [], []
        for _, row in df.iterrows():
            price  = row["Price"]
            change = row["Change%"] if pd.notna(row["Change%"]) else 0
            relvol = row["Rel Volume"] if pd.notna(row["Rel Volume"]) else 0
            si     = row["SI_num"] if pd.notna(row["SI_num"]) else 0

            score = 0
            if price_min <= price <= price_max: score += 1
            if change >= change_min:            score += 1
            if relvol >= relvol_min:            score += 1
            if si >= si_min:                    score += 1
            if score < 3:
                continue

            stock = _row_to_stock_dict(row, si, change, relvol)
            (prime if score == 4 else subprime).append(stock)

        prime.sort(key=lambda s: s["Change_num"], reverse=True)
        subprime.sort(key=lambda s: s["Change_num"], reverse=True)
        return prime, subprime
    except Exception as e:
        print(f"[Finviz] Error in legacy classification: {e}")
        return [], []
