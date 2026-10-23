"""Side-by-side comparison helpers. Rows are not sorted by model probability."""

from __future__ import annotations

import pandas as pd

from src.formatting import is_missing


def moving_average_trend(close, sma_50) -> str:
    if is_missing(close) or is_missing(sma_50):
        return "N/A"
    if float(close) > float(sma_50):
        return "Above SMA 50"
    if float(close) < float(sma_50):
        return "Below SMA 50"
    return "At SMA 50"


def indexed_closes(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rebase overlapping closes to 100 at the first shared date.

    The result is an index for shape comparison. It is not a price in rupees.
    """
    series = {}
    for symbol, frame in frames.items():
        column = frame.set_index("Date")["Close"].sort_index()
        column = column[~column.index.duplicated(keep="last")]
        series[symbol] = column
    if not series:
        return pd.DataFrame()
    combined = pd.DataFrame(series).dropna(how="any")
    if combined.empty:
        return combined
    return combined / combined.iloc[0] * 100


def comparison_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a display table in the order the user selected."""
    display_rows = []
    for row in rows:
        display_rows.append(
            {
                "Symbol": row.get("symbol"),
                "Company": row.get("company"),
                "Latest available close": row.get("latest_close_display"),
                "As of": row.get("as_of"),
                "Period return": row.get("period_return_display"),
                "Annualised volatility": row.get("volatility_display"),
                "RSI (14)": row.get("rsi_display"),
                "Moving-average trend": row.get("moving_average_trend"),
                "Upward score": row.get("probability_display"),
                "Holdout accuracy": row.get("accuracy_display"),
                "Holdout days": "N/A" if row.get("model_test_size") in (None, "N/A") else str(row.get("model_test_size")),
                "Baseline accuracy": row.get("baseline_display"),
                "News sentiment": row.get("news_sentiment") or "Not loaded",
                "P/E": row.get("pe_display") or "N/A",
                "Market cap": row.get("market_cap_display") or "N/A",
                "Revenue growth": row.get("revenue_growth_display") or "N/A",
            }
        )
    return pd.DataFrame(display_rows)


def reliability_note(rows: list[dict] | None = None) -> str:
    del rows  # The warning applies to every comparison, including an empty one.
    return (
        "Predictions for different stocks can have different holdout sizes and baseline accuracy. "
        "A higher upward-movement score is not a ranking and is not a recommendation. "
        "Read the evaluation sample and the majority-class baseline beside any score, and remember that companies can differ by sector and accounting."
    )
