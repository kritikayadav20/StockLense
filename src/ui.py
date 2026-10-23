"""Shared Streamlit layout: theme, sidebar, and small display helpers."""

from __future__ import annotations

import html

import streamlit as st

from config import (
    DATA_SOURCE_NOTE,
    DEFAULT_HORIZON,
    DEFAULT_PERIOD,
    DEFAULT_SYMBOL,
    DISCLAIMER,
    HORIZONS,
    PERIODS,
    STOCKS,
    display_name,
    exchange_for,
)
from src.data_loader import load_history
from src.errors import StockLensError
from src.formatting import as_date_text

CSS = """
<style>
[data-testid="stAppViewContainer"] { background: #f4f7fb; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }
[data-testid="stHeader"] { background: transparent; }
h1, h2, h3 { color: #0b1f3a; letter-spacing: -0.02em; }
.sl-title { color: #0b1f3a; font-size: 1.85rem; font-weight: 700; margin: 0 0 0.2rem 0; }
.sl-sub { color: #334155; font-size: 1.02rem; margin: 0 0 0.8rem 0; line-height: 1.45; }
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 14px 16px 12px 16px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    min-height: 112px;
}
.metric-label { color: #475569; font-size: 0.8rem; }
.metric-value { color: #0b1f3a; font-size: 1.45rem; font-weight: 700; margin-top: 4px; }
.metric-value.positive { color: #0f766e; }
.metric-value.negative { color: #b91c1c; }
.metric-note { color: #64748b; font-size: 0.78rem; margin-top: 4px; }
.sl-brand { color: #0b1f3a; font-weight: 700; font-size: 1.25rem; margin-bottom: 0; }
.sl-brand-note { color: #64748b; font-size: 0.82rem; margin-top: 0; }
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 10px 12px;
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def bootstrap() -> dict:
    """Sidebar controls shared by every page. The returned settings drive the analysis."""
    inject_css()
    if "active_symbol" not in st.session_state:
        st.session_state.active_symbol = DEFAULT_SYMBOL
        st.session_state.last_choice = DEFAULT_SYMBOL
    if "refresh_token" not in st.session_state:
        st.session_state.refresh_token = 0

    st.sidebar.markdown("<p class='sl-brand'>StockLens</p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p class='sl-brand-note'>Indian equity analysis for learning</p>", unsafe_allow_html=True)
    choice = st.sidebar.selectbox(
        "Indian stock",
        list(STOCKS),
        format_func=lambda symbol: f"{STOCKS[symbol]['name']} ({symbol})",
        key="stock_choice",
    )
    if choice != st.session_state.get("last_choice"):
        st.session_state.last_choice = choice
        st.session_state.active_symbol = choice

    custom = st.sidebar.text_input("Other NSE or BSE ticker", placeholder="RELIANCE.BO")
    if st.sidebar.button("Use custom ticker", width="stretch"):
        st.session_state.pending_custom_ticker = custom.strip()

    period = st.sidebar.selectbox("Historical period", list(PERIODS), index=list(PERIODS).index(DEFAULT_PERIOD))
    horizon = st.sidebar.selectbox(
        "Prediction horizon (trading days)",
        list(HORIZONS),
        index=list(HORIZONS).index(DEFAULT_HORIZON),
    )
    if st.sidebar.button("Refresh market data", width="stretch"):
        st.session_state.refresh_token += 1
        st.session_state.force_next_load = True

    symbol = st.session_state.active_symbol
    st.sidebar.caption(f"Active symbol: {display_name(symbol)} ({symbol}, {exchange_for(symbol)})")
    st.sidebar.caption("Prices are shown in rupees from the latest downloaded bar, not as a live quote.")
    return {
        "symbol": symbol,
        "period": period,
        "horizon": int(horizon),
        "refresh_token": int(st.session_state.refresh_token),
        "force_refresh": bool(st.session_state.pop("force_next_load", False)),
    }


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def cached_history(symbol: str, period: str, refresh_token: int, force_refresh: bool):
    del refresh_token  # Part of the cache key so a refresh click bypasses the previous result.
    return load_history(symbol, period, force_refresh=force_refresh)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f"<p class='sl-title'>{html.escape(title)}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='sl-sub'>{html.escape(subtitle)}</p>", unsafe_allow_html=True)


def page_footer() -> None:
    st.divider()
    st.caption(DISCLAIMER)
    st.caption(DATA_SOURCE_NOTE)


def show_markdown(text: str) -> None:
    st.markdown((text or "").replace("$", "\\$"))


def metric_row(items: list[dict]) -> None:
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        tone = item.get("tone") or ""
        with column:
            st.markdown(
                (
                    "<div class='metric-card'>"
                    f"<div class='metric-label'>{html.escape(str(item.get('label', '')))}</div>"
                    f"<div class='metric-value {html.escape(tone)}'>{html.escape(str(item.get('value', 'N/A')))}</div>"
                    f"<div class='metric-note'>{html.escape(str(item.get('note', '')))}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def load_selected_history(settings: dict):
    """Load the sidebar stock. Returns None after showing an error, so the page can stop."""
    pending = st.session_state.pop("pending_custom_ticker", None)
    symbol = settings["symbol"]
    if pending:
        from src.data_loader import validate_symbol

        try:
            symbol = validate_symbol(pending)
        except StockLensError as exc:
            st.error(str(exc))
            return None
        if symbol != st.session_state.get("active_symbol"):
            st.session_state.active_symbol = symbol
            st.rerun()

    try:
        with st.spinner("Loading historical prices..."):
            package = cached_history(symbol, settings["period"], settings["refresh_token"], settings["force_refresh"])
    except StockLensError as exc:
        st.error(str(exc))
        return None
    except Exception as exc:  # noqa: BLE001
        st.error(
            "Historical prices could not be loaded. Check the ticker and your connection, then use Refresh market data. "
            f"Details: {exc.__class__.__name__}."
        )
        with st.expander("Technical detail"):
            st.exception(exc)
        return None

    package = {"frame": package["frame"].copy(), "meta": dict(package["meta"]), "symbol": symbol}
    meta = package["meta"]
    if meta.get("stale"):
        st.warning(meta.get("stale_reason") or "Showing cached history because the latest download failed.")
    if meta.get("db_warning"):
        st.warning(meta["db_warning"])
    if meta.get("cache_warning"):
        st.warning(meta["cache_warning"])
    return package


def show_data_line(meta: dict) -> None:
    fetched = as_date_text(meta.get("fetched_at"))
    st.caption(
        f"History: {as_date_text(meta.get('first_date'))} to {as_date_text(meta.get('last_date'))} "
        f"({meta.get('rows', 'N/A')} bars). Downloaded {fetched}. Source: {meta.get('source', 'Yahoo Finance via yfinance')}."
    )
    st.caption(meta.get("price_label") or "The latest close is the last available historical bar, not a live quote.")


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def cached_fundamentals(symbol: str):
    from src.fundamentals import fetch_fundamentals

    return fetch_fundamentals(symbol)


def show_frame(frame, height: int = 280) -> None:
    st.dataframe(frame, width="stretch", hide_index=True, height=height)
