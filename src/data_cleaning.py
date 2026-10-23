"""Validate and clean OHLCV history without inventing prices."""

from __future__ import annotations

import pandas as pd

from src.errors import DataQualityError

PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
OUTPUT_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

_COLUMN_NAMES = {
    "date": "Date",
    "datetime": "Date",
    "index": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj close": "Adj Close",
    "adjclose": "Adj Close",
    "adjusted close": "Adj Close",
    "volume": "Volume",
}


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Flatten a yfinance table and rename columns to the StockLens schema."""
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        level_zero = [str(value).strip().lower() for value in out.columns.get_level_values(0)]
        if "open" in level_zero or "close" in level_zero:
            out.columns = out.columns.get_level_values(0)
        else:
            out.columns = out.columns.get_level_values(-1)
    if "Date" not in [str(column) for column in out.columns] and "Datetime" not in [str(column) for column in out.columns]:
        out = out.reset_index()
    rename = {}
    for column in out.columns:
        key = str(column).strip().lower()
        if key in _COLUMN_NAMES:
            rename[column] = _COLUMN_NAMES[key]
    return out.rename(columns=rename)


def normalise_dates(series: pd.Series) -> pd.Series:
    """Convert provider timestamps to a timezone-naive Indian market date."""
    parsed = pd.to_datetime(series, errors="coerce")
    timezone = getattr(parsed.dt, "tz", None)
    if timezone is not None:
        parsed = parsed.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return parsed.dt.normalize()


def clean_ohlcv(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Sort, de-duplicate, and drop rows that are not usable prices.

    Missing closes are removed. They are not replaced with zero.
    """
    if frame is None or len(frame) == 0:
        raise DataQualityError(
            "The provider returned no historical rows. Try another period, confirm the ticker, or refresh the download."
        )

    working = normalise_columns(frame)
    missing = [column for column in ["Date", "Open", "High", "Low", "Close", "Volume"] if column not in working.columns]
    if missing:
        raise DataQualityError(
            "Downloaded data is missing required columns: "
            + ", ".join(missing)
            + ". Refresh the download. StockLens will not fill these fields in."
        )

    rows_in = int(len(working))
    working["Date"] = normalise_dates(working["Date"])
    undated = int(working["Date"].isna().sum())
    working = working.dropna(subset=["Date"]).sort_values("Date")
    duplicates_removed = int(working["Date"].duplicated().sum())
    working = working.drop_duplicates(subset=["Date"], keep="last")

    for column in PRICE_COLUMNS + ["Volume", "Adj Close"]:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    missing_close_removed = int(working["Close"].isna().sum())
    working = working.dropna(subset=["Close"])

    warnings: list[str] = []
    if "Adj Close" not in working.columns:
        working["Adj Close"] = pd.NA
        warnings.append("Adjusted close was not included by the provider. Charts use the unadjusted close.")

    # A tiny tolerance avoids dropping a valid bar because of floating-point noise.
    tolerance = 1e-6
    invalid = (
        working[PRICE_COLUMNS].isna().any(axis=1)
        | (working[PRICE_COLUMNS] <= 0).any(axis=1)
        | (working["High"] + tolerance < working["Low"])
        | (working["High"] + tolerance < working["Open"])
        | (working["High"] + tolerance < working["Close"])
        | (working["Low"] - tolerance > working["Open"])
        | (working["Low"] - tolerance > working["Close"])
    )
    invalid_rows_removed = int(invalid.sum())
    working = working.loc[~invalid].copy()

    negative_volume = working["Volume"].notna() & (working["Volume"] < 0)
    if int(negative_volume.sum()):
        warnings.append("Negative volume values were left blank rather than converted to zero.")
        working.loc[negative_volume, "Volume"] = pd.NA

    working = working[OUTPUT_COLUMNS].reset_index(drop=True)
    if working.empty:
        raise DataQualityError(
            "No valid historical rows remained after cleaning. The download may be incomplete. Refresh and try again."
        )

    report = {
        "rows_in": rows_in,
        "rows_out": int(len(working)),
        "undated_removed": undated,
        "duplicates_removed": duplicates_removed,
        "missing_close_removed": missing_close_removed,
        "invalid_rows_removed": invalid_rows_removed,
        "warnings": warnings,
    }
    return working, report
