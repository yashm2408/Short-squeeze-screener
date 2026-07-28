# Short Squeeze Screener

A live short squeeze screener that scans stocks every 60 seconds, scores them on a two-axis Pressure/Ignition model, and displays results in a color-coded Tkinter desktop app.

---

## What It Does

- Pulls ~1,500 stocks from Finviz Elite and narrows them to ~50 candidates using a Stage-1 pre-filter (float, price, short interest, relative volume)
- Fetches real-time Cost-to-Borrow and shortability data from Interactive Brokers
- Computes a live SI% estimate by layering daily FINRA short volume data on top of the official FINRA settlement number
- Scores every candidate on two independent axes:
  - **Pressure** (SI%, Days-to-Cover, Cost-to-Borrow, Shortability, Short Volume%) -- how explosive could this be?
  - **Ignition** (Relative Volume, Price Change%, TTM Squeeze signal) -- is it firing right now?
- Classifies each stock as **Prime** (loaded + firing), **Subprime** (loaded or firing), or drops it
- Runs news sentiment classification using a TF-IDF + RandomForest model trained on labeled headlines
- Displays a 3D correlation scatter plot (Official SI% vs Live SI% vs Relative Volume)
- Logs all Prime setups to `data/prime_log.csv`
- Lets you export the full screener table to CSV

---

## Project Structure

```
ScreenerProject/
├── core/
│   ├── finviz_api.py       # Finviz Elite data fetch
│   ├── filters.py          # Stage-1 candidate pre-filter
│   ├── ibkr_api.py         # IBKR borrow data + TTM bars (rate-limited)
│   ├── setup_classifier.py # Pressure/Ignition scoring + Prime/Subprime classification
│   ├── si_estimate.py      # Live SI% estimator using FINRA short volume
│   ├── ttm_squeeze.py      # TTM Squeeze indicator (Bollinger vs Keltner)
│   ├── sentiment.py        # News headline sentiment classifier
│   ├── short_volume.py     # FINRA short volume loader
│   ├── squeeze_score.py    # Composite squeeze score
│   └── yfinance_short.py   # yfinance fallback for short data
├── controller/
│   └── controller.py       # Orchestrates fetching, enrichment, classification
├── ui/
│   └── view.py             # Tkinter GUI (screener table, charts, news tab)
├── tools/
│   ├── calibrate_si.py     # Fits the SI% dampening constant against FINRA history
│   └── backtest_setups.py  # Backtests Prime/Subprime classification on historical squeezes
├── tests/
│   ├── test_filters.py
│   ├── test_si_estimate.py
│   └── test_ttm_squeeze.py
├── data/
│   ├── labeled_data.csv    # Training data for sentiment model
│   └── si_calibration.json # Fitted dampening constants (output of calibrate_si.py)
├── model/                  # Saved sentiment model (auto-generated on first run)
├── main.py                 # Entry point
├── requirements.txt
└── .env                    # API keys (not committed -- see .env.example)
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- **Interactive Brokers TWS or IB Gateway** running locally on port 7497 (paper trading account works)
- **Finviz Elite** subscription

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Tkinter is included with most Python installations. If you are on Linux and it is missing:

```bash
sudo apt-get install python3-tk
```

### 3. Configure API keys

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

### 4. Run

```bash
python main.py
```

The app connects to IBKR and begins scanning. The screener table populates within the first 60-second refresh cycle. IBKR borrow data and TTM Squeeze values may take 1-2 additional cycles to fully populate.

---

## How the Scoring Works

### Pressure Score (0-100)

Measures structural short squeeze fuel. Weighted blend of:

- SI% (45%) -- dominant fuel
- Days-to-Cover (20%)
- Cost-to-Borrow (15%)
- Shortability (10%)
- Short Volume% (10%)

Float size applies a multiplier: tight float amplifies pressure, large float dampens it.

### Ignition Score (0-100)

Measures whether a squeeze is actively firing:

- Relative Volume (40%)
- Price Change% -- up moves only (40%)
- TTM Squeeze signal (20%)

### Classification

- **Prime**: Pressure >= 55 AND Ignition >= 50
- **Subprime**: Several paths -- loaded with early spark, lighter fuel but clearly firing, or extremely heavy short interest

---

## Notes

- The `.env` file contains your API keys and is gitignored -- never commit it
- IBKR must be running before launching the app; if not connected, borrow data and TTM columns show `--`
- The sentiment model is auto-trained and saved on first run if `model/` is empty