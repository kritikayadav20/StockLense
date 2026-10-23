"""Download, cache, and validate Indian equity history from Yahoo Finance."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from config import CACHE_TTL_HOURS, DATA_SOURCE_NOTE, PERIODS, get_cache_dir
from src.data_cleaning import clean_ohlcv
from src.database import DatabaseError, log_download
from src.errors import DataAccessError, DataQualityError, DataValidationError

SOURCE_NAME = "Yahoo Finance via yfinance (unofficial)"
_SYMBOL_PATTERN = r"^[A-Z0-9&\-]+\.(NS|BO)$"


def validate_symbol(symbol: str) -> str:
    """Accept NSE (.NS) and BSE (.BO) Yahoo Finance tickers only."""
    import re

    cleaned = (symbol or "").strip().upper()
    if not re.fullmatch(_SYMBOL_PATTERN, cleaned):
        raise DataValidationError(
            "Use an NSE or BSE Yahoo Finance ticker, such as RELIANCE.NS or RELIANCE.BO. "
            "StockLens is built for Indian listings and does not treat other symbols as supported."
        )
    return cleaned


def validate_period(period_label: str) -> str:
    if period_label not in PERIODS:
        raise DataValidationError(
            "Choose one of these history lengths: " + ", ".join(PERIODS) + "."
        )
    return PERIODS[period_label]


def _cache_paths(symbol: str, yf_period: str):
    token = symbol.replace("&", "AND").replace(".", "_")
    directory = get_cache_dir()
    stem = directory / f"{token}_{yf_period}"
    return stem.with_suffix(".csv"), stem.with_suffix(".meta.json")


def _read_cache(csv_path, meta_path):
    frame = pd.read_csv(csv_path, parse_dates=["Date"])
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return frame, meta


def _write_cache(frame: pd.DataFrame, meta: dict, csv_path, meta_path) -> None:
    frame.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _is_fresh(meta: dict) -> bool:
    fetched = meta.get("fetched_at")
    if not fetched:
        return False
    fetched_at = datetime.fromisoformat(fetched)
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - fetched_at < timedelta(hours=CACHE_TTL_HOURS)


def download_raw(symbol: str, yf_period: str) -> pd.DataFrame:
    """Download daily bars. The caller validates and cleans the result."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataAccessError(
            "yfinance is not installed. From the project folder run: pip install -r requirements.txt"
        ) from exc
    try:
        ticker = yf.Ticker(symbol)
        try:
            frame = ticker.history(period=yf_period, interval="1d", auto_adjust=False, actions=False, timeout=30)
        except TypeError:
            frame = ticker.history(period=yf_period, interval="1d", auto_adjust=False, actions=False)
    except Exception as exc:  # noqa: BLE001 - provider errors vary by yfinance version.
        detail = str(exc).strip().replace("\n", " ")
        if len(detail) > 180:
            detail = detail[:180] + "..."
        raise DataAccessError(
            f"Could not download historical prices for {symbol}. Check the ticker and your internet connection, then refresh. "
            f"{DATA_SOURCE_NOTE} Provider message: {detail or exc.__class__.__name__}."
        ) from exc
    if frame is None or frame.empty:
        raise DataAccessError(
            f"No historical prices were returned for {symbol}. The ticker may be incorrect, delisted, or temporarily unavailable. "
            "Confirm the .NS or .BO symbol and refresh."
        )
    return frame


def _package(frame: pd.DataFrame, meta: dict) -> dict:
    return {"frame": frame, "meta": meta}


def load_history(symbol: str, period_label: str, force_refresh: bool = False) -> dict:
    """Return cleaned daily history and metadata. Use cache when it is still fresh.

    A failed download falls back to an existing cache and marks it stale.
    If there is no cache, the error explains how to retry. No prices are invented.
    """
    symbol = validate_symbol(symbol)
    yf_period = validate_period(period_label)
    csv_path, meta_path = _cache_paths(symbol, yf_period)
    cached = None
    if csv_path.exists() and meta_path.exists():
        try:
            cached = _read_cache(csv_path, meta_path)
        except (OSError, json.JSONDecodeError, ValueError):
            cached = None

    if cached is not None and not force_refresh and _is_fresh(cached[1]):
        frame, meta = cached
        meta = dict(meta)
        meta["from_cache"] = True
        meta["stale"] = False
        return _package(frame, meta)

    try:
        raw = download_raw(symbol, yf_period)
        frame, report = clean_ohlcv(raw)
    except (DataAccessError, DataQualityError) as exc:
        if cached is not None:
            frame, meta = cached
            meta = dict(meta)
            meta["from_cache"] = True
            meta["stale"] = True
            meta["stale_reason"] = (
                "The latest download failed, so StockLens is showing the last cached historical bars. "
                + str(exc)
            )
            return _package(frame, meta)
        raise

    fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    meta = {
        "symbol": symbol,
        "period_label": period_label,
        "yf_period": yf_period,
        "fetched_at": fetched_at,
        "rows": int(len(frame)),
        "first_date": str(pd.Timestamp(frame["Date"].iloc[0]).date()),
        "last_date": str(pd.Timestamp(frame["Date"].iloc[-1]).date()),
        "source": SOURCE_NAME,
        "source_note": DATA_SOURCE_NOTE,
        "clean_report": report,
        "from_cache": False,
        "stale": False,
        "price_label": "Latest available close from the downloaded history. This is not a live quote.",
    }
    try:
        _write_cache(frame, meta, csv_path, meta_path)
    except OSError:
        meta["cache_warning"] = "The download succeeded, but StockLens could not write the local cache."
    try:
        log_download(
            symbol,
            period_label,
            fetched_at,
            int(len(frame)),
            meta["first_date"],
            meta["last_date"],
            SOURCE_NAME,
        )
    except DatabaseError as exc:
        meta["db_warning"] = str(exc) + " The price table is still available."
    return _package(frame, meta)
