"""Technical indicators calculated with Pandas and NumPy.

pandas-ta is intentionally not required. Its Numba builds are often incompatible
with current Python installs, and these formulas are the standard definitions
used by the charts, the technical summary, and the model features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import TRADING_DAYS_PER_YEAR
from src.formatting import is_missing

SUPPORT_WINDOW = 20
SUMMARY_RULE = (
    "The summary counts five documented readings: price against the 20-day average, "
    "price against the 50-day average, price against the 200-day average when that average exists, "
    "RSI(14) outside 30-70, and MACD against its signal line. "
    "The stance is Bullish or Bearish only when one side leads by at least two readings. "
    "Otherwise it is Neutral. None of these readings is a buy or sell instruction."
)


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-style RSI. A period with no losses is 100. A flat period is left blank."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    average_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    average_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    result = pd.Series(np.nan, index=close.index, dtype=float)
    valid = average_gain.notna() & average_loss.notna()
    loss_zero = valid & (average_loss == 0) & (average_gain > 0)
    normal = valid & (average_loss > 0)
    result.loc[loss_zero] = 100.0
    result.loc[normal] = 100 - (100 / (1 + average_gain.loc[normal] / average_loss.loc[normal]))
    return result


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line is the 12-period EMA minus the 26-period EMA. The signal is a 9-period EMA of that line."""
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_fast - ema_slow
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = line - signal_line
    return line, signal_line, histogram


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    """Bollinger position is 0 at the lower band and 1 at the upper band."""
    middle = sma(close, window)
    deviation = close.rolling(window, min_periods=window).std()
    upper = middle + num_std * deviation
    lower = middle - num_std * deviation
    width = (upper - lower).where((upper - lower) != 0)
    position = (close - lower) / width
    return middle, upper, lower, position


def add_indicator_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add indicator columns. Every value at a row uses only that row and earlier rows."""
    out = frame.copy()
    close = out["Close"].astype(float)
    out["return_1d"] = close.pct_change(1)
    out["return_5d"] = close.pct_change(5)
    out["return_10d"] = close.pct_change(10)
    out["return_20d"] = close.pct_change(20)
    out["sma_5"] = sma(close, 5)
    out["sma_20"] = sma(close, 20)
    out["sma_50"] = sma(close, 50)
    out["sma_200"] = sma(close, 200)
    out["rsi_14"] = rsi(close, 14)
    macd_line, signal_line, histogram = macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_histogram"] = histogram
    middle, upper, lower, position = bollinger(close)
    out["bb_middle"] = middle
    out["bb_upper"] = upper
    out["bb_lower"] = lower
    out["bb_position"] = position
    out["volatility_20"] = out["return_1d"].rolling(20, min_periods=20).std()
    out["volume_change"] = out["Volume"].pct_change()
    out["price_to_sma_5"] = close / out["sma_5"] - 1
    out["price_to_sma_20"] = close / out["sma_20"] - 1
    out["price_to_sma_50"] = close / out["sma_50"] - 1
    out["high_low_range"] = (out["High"] - out["Low"]) / close.where(close != 0)
    out["support_20"] = out["Low"].rolling(SUPPORT_WINDOW, min_periods=SUPPORT_WINDOW).min()
    out["resistance_20"] = out["High"].rolling(SUPPORT_WINDOW, min_periods=SUPPORT_WINDOW).max()
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def _number(value):
    if is_missing(value):
        return None
    return float(value)


def _trend_signal(label: str, close: float | None, average: float | None, average_name: str) -> dict:
    if close is None or average is None:
        return {
            "indicator": label,
            "observation": f"{average_name} is not available for this bar.",
            "reading": "Neutral",
            "meaning": "This average needs more history than the current window contains.",
        }
    if close > average:
        reading = "Bullish"
        relation = "above"
    elif close < average:
        reading = "Bearish"
        relation = "below"
    else:
        reading = "Neutral"
        relation = "equal to"
    return {
        "indicator": label,
        "observation": f"Close is {relation} {average_name} ({average:,.2f}).",
        "reading": reading,
        "meaning": f"Price {relation} this average is a trend description, not a trading instruction.",
    }


def build_signals(latest: pd.Series) -> list[dict]:
    close = _number(latest.get("Close"))
    signals = [
        _trend_signal("Short-term trend", close, _number(latest.get("sma_20")), "SMA 20"),
        _trend_signal("Medium-term trend", close, _number(latest.get("sma_50")), "SMA 50"),
        _trend_signal("Long-term trend", close, _number(latest.get("sma_200")), "SMA 200"),
    ]
    rsi_value = _number(latest.get("rsi_14"))
    if rsi_value is None:
        signals.append(
            {
                "indicator": "RSI (14)",
                "observation": "RSI is not available for this bar.",
                "reading": "Neutral",
                "meaning": "RSI needs enough earlier price changes before it can be calculated.",
            }
        )
    elif rsi_value > 70:
        signals.append(
            {
                "indicator": "RSI (14)",
                "observation": f"RSI is {rsi_value:.1f}, above 70.",
                "reading": "Bearish",
                "meaning": "A reading above 70 is a commonly used overbought condition. It is not, by itself, a sell signal.",
            }
        )
    elif rsi_value < 30:
        signals.append(
            {
                "indicator": "RSI (14)",
                "observation": f"RSI is {rsi_value:.1f}, below 30.",
                "reading": "Bullish",
                "meaning": "A reading below 30 is a commonly used oversold condition. It is not, by itself, a buy signal.",
            }
        )
    else:
        signals.append(
            {
                "indicator": "RSI (14)",
                "observation": f"RSI is {rsi_value:.1f}, between 30 and 70.",
                "reading": "Neutral",
                "meaning": "RSI is outside the usual overbought and oversold thresholds.",
            }
        )

    macd_value = _number(latest.get("macd"))
    signal_value = _number(latest.get("macd_signal"))
    if macd_value is None or signal_value is None:
        signals.append(
            {
                "indicator": "MACD",
                "observation": "MACD or its signal line is not available for this bar.",
                "reading": "Neutral",
                "meaning": "MACD needs a longer price history before both lines exist.",
            }
        )
    elif macd_value > signal_value:
        signals.append(
            {
                "indicator": "MACD",
                "observation": f"MACD ({macd_value:.2f}) is above its signal line ({signal_value:.2f}).",
                "reading": "Bullish",
                "meaning": "MACD above the signal line is a positive momentum reading, not a guaranteed advance.",
            }
        )
    elif macd_value < signal_value:
        signals.append(
            {
                "indicator": "MACD",
                "observation": f"MACD ({macd_value:.2f}) is below its signal line ({signal_value:.2f}).",
                "reading": "Bearish",
                "meaning": "MACD below the signal line is a negative momentum reading, not a guaranteed decline.",
            }
        )
    else:
        signals.append(
            {
                "indicator": "MACD",
                "observation": "MACD is equal to its signal line.",
                "reading": "Neutral",
                "meaning": "The two lines do not currently show a momentum gap.",
            }
        )
    return signals


def summary_stance(signals: list[dict]) -> str:
    bullish = sum(1 for signal in signals if signal["reading"] == "Bullish")
    bearish = sum(1 for signal in signals if signal["reading"] == "Bearish")
    if bullish >= bearish + 2:
        return "Bullish"
    if bearish >= bullish + 2:
        return "Bearish"
    return "Neutral"


def technical_snapshot(frame: pd.DataFrame) -> dict:
    """Summarise the latest bar. Indicator values stay separate from model predictions."""
    if frame is None or frame.empty:
        raise ValueError("Technical analysis needs a historical price table.")
    enriched = add_indicator_columns(frame)
    latest = enriched.iloc[-1]
    previous_close = float(enriched["Close"].iloc[-2]) if len(enriched) > 1 else None
    last_close = float(latest["Close"])
    daily_change = None if previous_close in (None, 0) else last_close / previous_close - 1
    first_close = float(enriched["Close"].iloc[0])
    period_return = None if first_close == 0 else last_close / first_close - 1
    returns = enriched["return_1d"].dropna()
    annualized_volatility = None
    volatility_note = None
    if len(returns) >= 20:
        annualized_volatility = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    else:
        volatility_note = "Fewer than 20 daily returns are available, so annualised volatility is not shown."

    volume = enriched["Volume"].dropna()
    average_volume = float(volume.mean()) if not volume.empty else None
    signals = build_signals(latest)
    return {
        "last_date": latest["Date"],
        "close": last_close,
        "previous_close": previous_close,
        "daily_change": daily_change,
        "period_return": period_return,
        "rsi_14": _number(latest.get("rsi_14")),
        "macd": _number(latest.get("macd")),
        "macd_signal": _number(latest.get("macd_signal")),
        "macd_histogram": _number(latest.get("macd_histogram")),
        "sma_20": _number(latest.get("sma_20")),
        "sma_50": _number(latest.get("sma_50")),
        "sma_200": _number(latest.get("sma_200")),
        "bb_upper": _number(latest.get("bb_upper")),
        "bb_middle": _number(latest.get("bb_middle")),
        "bb_lower": _number(latest.get("bb_lower")),
        "bb_position": _number(latest.get("bb_position")),
        "annualized_volatility": annualized_volatility,
        "volatility_note": volatility_note,
        "average_volume": average_volume,
        "volume_change": _number(latest.get("volume_change")),
        "support": _number(latest.get("support_20")),
        "resistance": _number(latest.get("resistance_20")),
        "support_method": (
            f"Support is the lowest low, and resistance is the highest high, "
            f"over the last {SUPPORT_WINDOW} trading days including the latest bar. "
            "This is a rolling-window range estimate, not a confirmed floor or ceiling."
        ),
        "signals": signals,
        "summary_stance": summary_stance(signals),
        "summary_rule": SUMMARY_RULE,
        "rows": int(len(enriched)),
        "enriched": enriched,
    }


INDICATOR_HELP = {
    "RSI (14)": "RSI compares the size of recent gains with recent losses on a 0 to 100 scale. Readings above 70 are often called overbought, and readings below 30 oversold.",
    "MACD": "MACD compares a shorter and a longer exponential moving average. The signal line smooths MACD. A gap between them describes momentum.",
    "SMA 20 / 50 / 200": "A simple moving average is the mean close over that many trading days. A close above the average describes a rising window, and a close below it a falling window.",
    "Bollinger Bands": "The bands sit two standard deviations above and below the 20-day average. Position shows where the close sits between the lower and upper band.",
    "Volatility": "Annualised historical volatility is the sample standard deviation of daily returns multiplied by the square root of 252. It describes how widely the price has moved, not the direction.",
    "Support and resistance": "The 20-day low and 20-day high are a simple recent range. They are estimates, and price can move through them.",
}
