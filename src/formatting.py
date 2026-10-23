"""Display helpers for rupees, percentages, and missing values."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        return False
    return False


def format_inr(value, decimals: int = 2) -> str:
    if is_missing(value):
        return "N/A"
    return f"₹{float(value):,.{decimals}f}"


def format_percent(value, decimals: int = 2, already_percent: bool = False) -> str:
    """Format a fraction as a percentage. Pass already_percent=True for RSI-style 0-100 values."""
    if is_missing(value):
        return "N/A"
    number = float(value) if already_percent else float(value) * 100
    return f"{number:.{decimals}f}%"


def format_number(value, decimals: int = 2) -> str:
    if is_missing(value):
        return "N/A"
    return f"{float(value):,.{decimals}f}"


def format_large_inr(value) -> str:
    """Format a rupee amount in crore when it is large enough to read that way."""
    if is_missing(value):
        return "N/A"
    amount = float(value)
    crore = amount / 1e7
    if abs(crore) >= 1:
        return f"₹{crore:,.1f} crore"
    return format_inr(amount)


def tone_for(value) -> str:
    if is_missing(value):
        return ""
    if float(value) > 0:
        return "positive"
    if float(value) < 0:
        return "negative"
    return ""


def as_date_text(value) -> str:
    if is_missing(value):
        return "N/A"
    if isinstance(value, datetime):
        return value.strftime("%d %b %Y")
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%d %b %Y")


def to_builtin(value):
    """Convert NumPy and pandas values so results can be stored as JSON."""
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if value is pd.NaT:
        return None
    return value
