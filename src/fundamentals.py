"""Company fundamentals from yfinance, with missing values left as N/A."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.formatting import is_missing

REVENUE_NAMES = ("total revenue", "operating revenue")
INCOME_NAMES = ("net income", "net income common stockholders")


def _blank_number(value):
    if is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _date_text(value):
    if is_missing(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp = timestamp / 1000
        if timestamp > 1_000_000_000:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return pd.Timestamp(parsed).date().isoformat()


def _find_row(frame: pd.DataFrame, candidates: tuple[str, ...]):
    if frame is None or frame.empty:
        return None
    lookup = {str(index).strip().lower(): index for index in frame.index}
    for candidate in candidates:
        if candidate in lookup:
            return frame.loc[lookup[candidate]]
    for index in frame.index:
        name = str(index).strip().lower()
        if any(candidate in name for candidate in candidates):
            return frame.loc[index]
    return None


def _series_from_row(row) -> list[dict]:
    if row is None:
        return []
    if isinstance(row, pd.DataFrame):
        # A duplicated label can return a frame. Use the first matching row only.
        row = row.iloc[0]
    points = []
    for period, value in row.items():
        number = _blank_number(value)
        if number is None:
            continue
        points.append({"period": _date_text(period) or str(period), "value": number})
    points.sort(key=lambda item: item["period"])
    return points


def extract_statement_series(frame: pd.DataFrame | None) -> dict:
    """Pull revenue and net income from a provider statement without filling gaps."""
    if frame is None or getattr(frame, "empty", True):
        return {
            "revenue": [],
            "net_income": [],
            "note": "No income statement was returned by the provider.",
        }
    return {
        "revenue": _series_from_row(_find_row(frame, REVENUE_NAMES)),
        "net_income": _series_from_row(_find_row(frame, INCOME_NAMES)),
        "note": None,
    }


def _yield_display(info: dict):
    """Prefer dividend rate divided by price.

    Yahoo Finance has reported dividendYield both as a fraction and as a percent.
    A value of 0.5 can mean 0.5% or 50%, so the raw field is not rescaled.
    """
    rate = _blank_number(info.get("dividendRate"))
    price = _blank_number(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))
    raw = _blank_number(info.get("dividendYield"))
    if rate is not None and price not in (None, 0):
        fraction = rate / price
        return (
            fraction,
            f"{fraction * 100:.2f}%",
            "Calculated from the provider's dividend rate and share price. The raw dividend-yield field is not rescaled.",
        )
    if raw is None:
        return None, "N/A", "The provider did not return a dividend yield or dividend rate."
    return raw, f"{raw:g}", "Shown as returned by the provider. It was not converted, because this field does not have one reliable scale."


def _fraction_display(value, label: str):
    number = _blank_number(value)
    if number is None:
        return None, "N/A", f"The provider did not return {label}."
    if abs(number) <= 2.5:
        return number, f"{number * 100:.2f}%", "Displayed as a percentage. Yahoo Finance usually returns this item as a fraction."
    return number, f"{number:.2f}", "Shown as returned by the provider because the value is outside the usual fraction range."


def _growth_sentence(series: list[dict], label: str) -> str:
    if len(series) < 2:
        return f"{label} growth is not calculated because fewer than two reporting periods were available."
    older = series[0]["value"]
    newer = series[-1]["value"]
    if older in (None, 0) or newer is None:
        return f"{label} growth is not calculated because a period value is missing or the starting value is zero."
    change = (newer - older) / abs(older)
    if change > 0:
        verb = "increased"
    elif change < 0:
        verb = "decreased"
    else:
        verb = "was unchanged"
    return (
        f"{label} {verb} from {series[0]['period']} to {series[-1]['period']} "
        f"({change * 100:.1f}% across the available periods, which may cover more than one year)."
    )


def metrics_from_info(info: dict | None, statements: dict | None = None) -> dict:
    """Map a provider info dictionary to display rows. Missing keys stay None."""
    info = info or {}
    statements = statements or {"revenue": [], "net_income": [], "note": "No income statement was returned by the provider."}
    currency = info.get("financialCurrency") or info.get("currency") or None
    dividend_value, dividend_display, dividend_note = _yield_display(info)
    roe_value, roe_display, roe_note = _fraction_display(info.get("returnOnEquity"), "return on equity")
    margin_value, margin_display, margin_note = _fraction_display(info.get("profitMargins"), "profit margin")
    revenue_growth_value, revenue_growth_display, revenue_growth_note = _fraction_display(
        info.get("revenueGrowth"), "revenue growth"
    )
    earnings_growth_value, earnings_growth_display, earnings_growth_note = _fraction_display(
        info.get("earningsGrowth"), "earnings growth"
    )
    market_cap = _blank_number(info.get("marketCap"))
    trailing_pe = _blank_number(info.get("trailingPE"))
    eps = _blank_number(info.get("trailingEps"))
    debt = _blank_number(info.get("debtToEquity"))

    rows = [
        {"metric": "Market capitalization", "value": market_cap, "display": _money(market_cap, currency), "note": "Provider market cap."},
        {"metric": "P/E ratio", "value": trailing_pe, "display": "N/A" if trailing_pe is None else f"{trailing_pe:.2f}", "note": "Trailing P/E as returned by the provider."},
        {"metric": "EPS", "value": eps, "display": _money(eps, currency, small=True), "note": "Trailing earnings per share."},
        {"metric": "Dividend yield", "value": dividend_value, "display": dividend_display, "note": dividend_note},
        {"metric": "Return on equity", "value": roe_value, "display": roe_display, "note": roe_note},
        {"metric": "Profit margin", "value": margin_value, "display": margin_display, "note": margin_note},
        {
            "metric": "Debt to equity",
            "value": debt,
            "display": "N/A" if debt is None else f"{debt:.2f}",
            "note": "Shown as returned by the provider. Yahoo Finance sometimes reports this ratio in percentage points, so compare it within a sector rather than to a fixed cutoff.",
        },
        {"metric": "Revenue growth", "value": revenue_growth_value, "display": revenue_growth_display, "note": revenue_growth_note},
        {"metric": "Earnings growth", "value": earnings_growth_value, "display": earnings_growth_display, "note": earnings_growth_note},
    ]
    interpretations = [
        _growth_sentence(statements.get("revenue", []), "Revenue"),
        _growth_sentence(statements.get("net_income", []), "Net income"),
        "A single ratio is not a conclusion about financial strength. Sector, accounting choices, and the reporting date all matter.",
    ]
    available = any(row["value"] is not None for row in rows) or bool(statements.get("revenue") or statements.get("net_income"))
    return {
        "currency": currency,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "long_name": info.get("longName") or info.get("shortName"),
        "as_of": _date_text(info.get("mostRecentQuarter") or info.get("lastFiscalYearEnd")),
        "rows": rows,
        "revenue": statements.get("revenue", []),
        "net_income": statements.get("net_income", []),
        "statement_note": statements.get("note"),
        "interpretations": interpretations,
        "available": available,
        "currency_note": None
        if currency in (None, "INR")
        else f"The provider reported currency {currency}. StockLens does not convert it with an assumed exchange rate.",
    }


def _money(value, currency, small: bool = False) -> str:
    if value is None:
        return "N/A"
    if currency == "INR" and not small:
        return f"₹{value / 1e7:,.1f} crore"
    if currency == "INR":
        return f"₹{value:,.2f}"
    if currency:
        return f"{currency} {value:,.2f}" if small else f"{currency} {value:,.0f}"
    return f"{value:,.2f}"


def fetch_fundamentals(symbol: str) -> dict:
    """Download provider fundamentals. Failures stay visible and are not replaced with zeros."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed. Run pip install -r requirements.txt.") from exc
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        statement = getattr(ticker, "income_stmt", None)
        if statement is None or getattr(statement, "empty", True):
            statement = getattr(ticker, "financials", None)
        statements = extract_statement_series(statement)
    except Exception as exc:  # noqa: BLE001 - yfinance raises several provider-specific errors.
        detail = str(exc).strip().replace("\n", " ")
        if len(detail) > 180:
            detail = detail[:180] + "..."
        return {
            "available": False,
            "rows": [],
            "revenue": [],
            "net_income": [],
            "interpretations": [],
            "error": (
                "Fundamental data could not be downloaded. The rest of StockLens can still use price history. "
                f"Provider message: {detail or exc.__class__.__name__}."
            ),
        }
    result = metrics_from_info(info if isinstance(info, dict) else {}, statements)
    if not result["available"]:
        result["error"] = (
            "The provider did not return fundamental metrics or an income statement for this symbol. "
            "Missing items are left as N/A."
        )
    return result
