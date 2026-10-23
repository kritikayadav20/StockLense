"""Project settings, the Indian stock list, and user-facing data notes."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Friendly names for the built-in NSE list. Add another row to extend the menu.
# Any other NSE (.NS) or BSE (.BO) Yahoo Finance ticker can also be typed in the sidebar.
STOCKS: dict[str, dict[str, str]] = {
    "RELIANCE.NS": {"name": "Reliance Industries", "exchange": "NSE"},
    "TCS.NS": {"name": "Tata Consultancy Services", "exchange": "NSE"},
    "INFY.NS": {"name": "Infosys", "exchange": "NSE"},
    "HDFCBANK.NS": {"name": "HDFC Bank", "exchange": "NSE"},
    "ICICIBANK.NS": {"name": "ICICI Bank", "exchange": "NSE"},
    "SBIN.NS": {"name": "State Bank of India", "exchange": "NSE"},
    "TATAMOTORS.NS": {"name": "Tata Motors", "exchange": "NSE"},
    "ITC.NS": {"name": "ITC", "exchange": "NSE"},
    "BHARTIARTL.NS": {"name": "Bharti Airtel", "exchange": "NSE"},
    "LT.NS": {"name": "Larsen & Toubro", "exchange": "NSE"},
    "HINDUNILVR.NS": {"name": "Hindustan Unilever", "exchange": "NSE"},
    "WIPRO.NS": {"name": "Wipro", "exchange": "NSE"},
    "AXISBANK.NS": {"name": "Axis Bank", "exchange": "NSE"},
    "KOTAKBANK.NS": {"name": "Kotak Mahindra Bank", "exchange": "NSE"},
    "HCLTECH.NS": {"name": "HCL Technologies", "exchange": "NSE"},
    "SUNPHARMA.NS": {"name": "Sun Pharmaceutical", "exchange": "NSE"},
    "BAJFINANCE.NS": {"name": "Bajaj Finance", "exchange": "NSE"},
    "MARUTI.NS": {"name": "Maruti Suzuki", "exchange": "NSE"},
    "ASIANPAINT.NS": {"name": "Asian Paints", "exchange": "NSE"},
    "M&M.NS": {"name": "Mahindra & Mahindra", "exchange": "NSE"},
}

DEFAULT_SYMBOL = "RELIANCE.NS"

# Labels shown in the app map to yfinance period codes.
PERIODS: dict[str, str] = {
    "6 months": "6mo",
    "1 year": "1y",
    "2 years": "2y",
    "5 years": "5y",
    "Maximum available": "max",
}
DEFAULT_PERIOD = "1 year"
HORIZONS = (1, 5, 10)
DEFAULT_HORIZON = 5

# Round-trip assumptions for the hypothetical simulation only.
# These are teaching assumptions, not a broker's live fee schedule.
TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005

CACHE_TTL_HOURS = 6
TRADING_DAYS_PER_YEAR = 252
MIN_LABELED_ROWS = 50
MIN_TRAIN_ROWS = 30
MIN_VAL_ROWS = 10
MIN_TEST_ROWS = 10
# Below this holdout size, simulated returns are withheld so a tiny sample cannot look like a track record.
MIN_BACKTEST_TEST_ROWS = 30

DATA_SOURCE_NOTE = (
    "Historical prices come from Yahoo Finance through the yfinance package, which is an unofficial interface. "
    "Availability, delay, completeness, revisions, and permitted use are decided by the provider. "
    "The latest close is the last historical bar the provider returned. It is not a live quote unless a real-time feed is connected."
)

DISCLAIMER = (
    "StockLens is an educational analysis tool. It is not investment advice, not a guaranteed prediction system, "
    "and not an automated trading platform. Past prices and model scores do not determine future returns."
)


def display_name(symbol: str) -> str:
    """Return the company name for a known ticker, or the ticker itself."""
    meta = STOCKS.get(symbol)
    if not meta:
        return symbol
    return meta["name"]


def exchange_for(symbol: str) -> str:
    meta = STOCKS.get(symbol)
    if meta:
        return meta["exchange"]
    if symbol.endswith(".BO"):
        return "BSE"
    if symbol.endswith(".NS"):
        return "NSE"
    return "Unknown"


def get_database_path() -> Path:
    raw = os.getenv("DATABASE_PATH", "data/stocklens.db")
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_cache_dir() -> Path:
    raw = os.getenv("STOCKLENS_CACHE_DIR", "data/cache")
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_dir() -> Path:
    path = ROOT / "models" / "saved"
    path.mkdir(parents=True, exist_ok=True)
    return path
