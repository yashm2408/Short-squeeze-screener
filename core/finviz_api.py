import os
import pandas as pd
import io
import csv
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

FINVIZ_API_KEY = os.getenv("FINVIZ_API_KEY", "")
_aspxauth = os.getenv("FINVIZ_ASPXAUTH", "")


def _get(url, use_cookie=True):
    from curl_cffi import requests as req
    if use_cookie and _aspxauth:
        headers = {
            "Cookie": f".ASPXAUTH={_aspxauth}",
            "Referer": "https://elite.finviz.com/",
        }
        r = req.get(url, impersonate="chrome", headers=headers, timeout=15)
    else:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = req.get(url, impersonate="chrome", headers=headers, timeout=15)

    if r.status_code != 200:
        raise Exception(f"HTTP {r.status_code}")
    if r.text.lstrip().startswith("<!DOCTYPE") or "<html" in r.text[:200].lower():
        raise Exception("Auth expired (got login page instead of data)")
    return r.text


def fetch_finviz_data():
    # Try cookie auth first (stable), fall back to API token
    try:
        url = (
            "https://elite.finviz.com/export/screener"
            "?v=152&f=sh_float_u20,sh_price_u20"
            "&c=1,25,26,30,31,84,42,43,49,50,52,53,55,56,57,59,60,61,63,64,81,86,87,65,66"
        )
        return pd.read_csv(io.StringIO(_get(url, use_cookie=True)))
    except Exception:
        url = (
            "https://elite.finviz.com/export/screener"
            "?v=152&f=sh_float_u20,sh_price_u20"
            f"&c=1,25,26,30,31,84,42,43,49,50,52,53,55,56,57,59,60,61,63,64,81,86,87,65,66"
            f"&auth={FINVIZ_API_KEY}"
        )
        return pd.read_csv(io.StringIO(_get(url, use_cookie=False)))


def fetch_all_finviz_api_news():
    try:
        url = "https://elite.finviz.com/news_export?v=3"
        text = _get(url, use_cookie=True)
    except Exception:
        url = f"https://elite.finviz.com/news_export?v=3&auth={FINVIZ_API_KEY}"
        try:
            text = _get(url, use_cookie=False)
        except Exception as e:
            print(f"[Finviz] News fetch failed: {e}")
            return []

    try:
        headlines = []
        for row in csv.DictReader(io.StringIO(text)):
            tickers = row.get("Ticker", "").strip()
            ticker_list = [t.strip() for t in tickers.split(",")] if tickers else []
            headlines.append({
                "headline": row.get("Title", "No title"),
                "timestamp": row.get("Date", ""),
                "url": row.get("Url", ""),
                "tickers": ticker_list,
            })
        return headlines
    except Exception as e:
        print(f"[Finviz] News parse failed: {e}")
        return []
