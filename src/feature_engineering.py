"""Model features and the future-direction target.

Leakage rule: a feature on a date may use the close, range, and volume from that
date and earlier dates only. The target looks forward by the selected horizon and
is never included in FEATURE_COLUMNS. The final horizon rows have no known target
and are kept out of training and evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.errors import DataQualityError
from src.technical_analysis import add_indicator_columns

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "sma_5",
    "sma_20",
    "sma_50",
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_position",
    "volatility_20",
    "volume_change",
    "price_to_sma_5",
    "price_to_sma_20",
    "price_to_sma_50",
    "high_low_range",
]

FEATURE_LABELS = {
    "return_1d": "1-day return",
    "return_5d": "5-day return",
    "return_10d": "10-day return",
    "return_20d": "20-day return",
    "sma_5": "SMA 5",
    "sma_20": "SMA 20",
    "sma_50": "SMA 50",
    "rsi_14": "RSI (14)",
    "macd": "MACD",
    "macd_signal": "MACD signal",
    "bb_position": "Bollinger position",
    "volatility_20": "20-day return volatility",
    "volume_change": "Volume change",
    "price_to_sma_5": "Price relative to SMA 5",
    "price_to_sma_20": "Price relative to SMA 20",
    "price_to_sma_50": "Price relative to SMA 50",
    "high_low_range": "High-low range",
}

# Raw prices and the forward-looking columns must never be offered to the model.
FORBIDDEN_FEATURE_COLUMNS = {
    "target",
    "future_close",
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
}


def assert_feature_columns_safe(columns=None) -> None:
    overlap = FORBIDDEN_FEATURE_COLUMNS.intersection(columns or FEATURE_COLUMNS)
    if overlap:
        raise DataQualityError(
            "The feature list includes columns that would leak price level or future information: "
            + ", ".join(sorted(overlap))
        )


def build_feature_frame(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return indicator features plus the forward target for this horizon."""
    if horizon not in (1, 5, 10):
        raise DataQualityError("Prediction horizon must be 1, 5, or 10 trading days.")
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise DataQualityError("Feature construction needs these columns: " + ", ".join(sorted(missing)))
    assert_feature_columns_safe()

    out = add_indicator_columns(frame)
    if not out["Date"].is_monotonic_increasing:
        out = out.sort_values("Date").reset_index(drop=True)

    # shift(-horizon) is the future close. It is stored only as the label and the
    # backtest outcome. It is not part of FEATURE_COLUMNS.
    future_close = out["Close"].shift(-horizon)
    out["future_close"] = future_close
    target = pd.Series(float("nan"), index=out.index, dtype=float)
    known = future_close.notna()
    target.loc[known] = (future_close.loc[known] > out.loc[known, "Close"]).astype(float)
    out["target"] = target
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def labeled_frame(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Rows that have a full feature set and a known future outcome."""
    featured = build_feature_frame(frame, horizon)
    labeled = featured.dropna(subset=FEATURE_COLUMNS + ["target"]).copy()
    labeled["target"] = labeled["target"].astype(int)
    return labeled
