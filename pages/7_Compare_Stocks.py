import pandas as pd
import streamlit as st

from config import STOCKS, display_name
from src.ai_explainer import explain
from src.charts import indexed_chart
from src.errors import ModelTrainingError, StockLensError
from src.formatting import format_inr, format_number, format_percent, is_missing
from src.stock_comparison import comparison_frame, indexed_closes, moving_average_trend, reliability_note
from src.stock_prediction import refresh_saved_prediction, train_and_evaluate
from src.technical_analysis import technical_snapshot
from src.ui import (
    bootstrap,
    cached_fundamentals,
    cached_history,
    page_footer,
    page_header,
    show_frame,
    show_markdown,
)

settings = bootstrap()
page_header(
    "Compare stocks",
    "Up to three Indian listings, shown side by side. Scores are not sorted into a ranking.",
)

options = list(STOCKS)
if settings["symbol"] not in options:
    options = [settings["symbol"]] + options
default = [settings["symbol"]]
if len(default) < 2 and "TCS.NS" in options and settings["symbol"] != "TCS.NS":
    default.append("TCS.NS")

selected = st.multiselect(
    "Stocks",
    options,
    default=default,
    max_selections=3,
    format_func=lambda symbol: f"{display_name(symbol)} ({symbol})",
)
include_models = st.checkbox("Train any missing models before comparing", value=False)
st.caption("Training is optional and can take a minute for three stocks. Existing saved models are reused either way.")

if st.button("Run comparison", type="primary"):
    if not selected:
        st.error("Select at least one stock.")
    else:
        rows = []
        frames = {}
        progress = st.progress(0)
        for index, symbol in enumerate(selected, start=1):
            try:
                package = cached_history(symbol, settings["period"], settings["refresh_token"], settings["force_refresh"])
            except StockLensError as exc:
                st.error(f"{symbol}: {exc}")
                progress.progress(index / len(selected))
                continue
            frame = package["frame"]
            frames[symbol] = frame
            snapshot = technical_snapshot(frame)
            fundamentals = cached_fundamentals(symbol)
            prediction = refresh_saved_prediction(frame, symbol, settings["period"], settings["horizon"])
            if prediction is None and include_models:
                try:
                    prediction = train_and_evaluate(frame, settings["horizon"], symbol, settings["period"])
                except (ModelTrainingError, StockLensError) as exc:
                    st.warning(f"{symbol} was not trained: {exc}")
            sentiment = st.session_state.get(f"sentiment::{symbol}") or {}
            metric_lookup = {row["metric"]: row for row in fundamentals.get("rows") or []}
            probability = None if prediction is None else prediction.get("upward_probability")
            accuracy = None if prediction is None else (prediction.get("metrics") or {}).get("accuracy")
            baseline = None if prediction is None else (prediction.get("baseline_majority") or {}).get("accuracy")
            test_size = None if prediction is None else (prediction.get("metrics") or {}).get("test_size")
            rows.append(
                {
                    "symbol": symbol,
                    "company": display_name(symbol),
                    "horizon": settings["horizon"],
                    "latest_close": snapshot["close"],
                    "latest_close_display": format_inr(snapshot["close"]),
                    "as_of": str(pd.Timestamp(snapshot["last_date"]).date()),
                    "period_return": snapshot["period_return"],
                    "period_return_display": format_percent(snapshot["period_return"]),
                    "annualized_volatility": snapshot["annualized_volatility"],
                    "volatility_display": format_percent(snapshot["annualized_volatility"]),
                    "rsi_14": snapshot["rsi_14"],
                    "rsi_display": format_number(snapshot["rsi_14"], 1),
                    "moving_average_trend": moving_average_trend(snapshot["close"], snapshot["sma_50"]),
                    "upward_probability": probability,
                    "probability_display": "Not trained" if is_missing(probability) else format_percent(probability, 1),
                    "model_accuracy": accuracy,
                    "accuracy_display": format_percent(accuracy),
                    "model_test_size": test_size,
                    "baseline_display": format_percent(baseline),
                    "news_sentiment": sentiment.get("overall_label"),
                    "pe_display": (metric_lookup.get("P/E ratio") or {}).get("display"),
                    "market_cap_display": (metric_lookup.get("Market capitalization") or {}).get("display"),
                    "revenue_growth_display": (metric_lookup.get("Revenue growth") or {}).get("display"),
                }
            )
            progress.progress(index / len(selected))
        st.session_state["comparison_rows"] = rows
        st.session_state["comparison_frames"] = frames
        st.session_state["comparison_period"] = settings["period"]
        st.session_state["comparison_horizon"] = settings["horizon"]

rows = st.session_state.get("comparison_rows") or []
frames = st.session_state.get("comparison_frames") or {}
if rows:
    if st.session_state.get("comparison_period") != settings["period"] or st.session_state.get("comparison_horizon") != settings["horizon"]:
        st.warning("These rows were built for a different period or horizon. Run the comparison again before relying on them.")
    st.caption(reliability_note(rows))
    show_frame(comparison_frame(rows), height=220)
    indexed = indexed_closes({row["symbol"]: frames[row["symbol"]] for row in rows if row["symbol"] in frames})
    if indexed.empty or len(indexed) < 20:
        st.info("There is not a long enough overlapping date range to draw a rebased price chart.")
    else:
        st.plotly_chart(indexed_chart(indexed), width="stretch")
        st.caption("Each line is rebased to 100 on the first shared date. It compares shape, not the rupee price level.")
    if st.button("Write comparison summary"):
        summary = explain({"comparison": rows, "symbol": "comparison", "name": "Comparison", "period": settings["period"], "horizon": settings["horizon"]}, "Compare the selected stocks.")
        st.session_state["comparison_summary"] = summary
    summary = st.session_state.get("comparison_summary")
    if summary:
        for warning in summary.get("warnings") or []:
            st.info(warning)
        show_markdown(summary["text"])
else:
    st.info("Run the comparison to load prices and any saved model scores. Nothing is ranked until you read the table yourself.")

page_footer()
