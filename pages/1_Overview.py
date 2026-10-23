from config import display_name
from src.charts import price_chart
from src.formatting import format_inr, format_number, format_percent, tone_for
from src.stock_prediction import refresh_saved_prediction
from src.technical_analysis import technical_snapshot
from src.ui import bootstrap, load_selected_history, metric_row, page_footer, page_header, show_data_line

import streamlit as st

settings = bootstrap()
page_header(
    "Overview",
    "A single view of the selected Indian stock: the latest available close, recent price behaviour, technical readings, and a saved model estimate.",
)
history = load_selected_history(settings)
if history is None:
    page_footer()
    st.stop()

symbol = history["symbol"]
frame = history["frame"]
meta = history["meta"]
snapshot = technical_snapshot(frame)
prediction = refresh_saved_prediction(frame, symbol, settings["period"], settings["horizon"])

st.caption(f"{display_name(symbol)} ({symbol})")
metric_row(
    [
        {
            "label": "Latest available close",
            "value": format_inr(snapshot["close"]),
            "note": "Last downloaded bar, not a live quote",
        },
        {
            "label": "Change vs previous close",
            "value": format_percent(snapshot["daily_change"]),
            "note": "Previous bar in this history",
            "tone": tone_for(snapshot["daily_change"]),
        },
        {
            "label": "Selected-period return",
            "value": format_percent(snapshot["period_return"]),
            "note": settings["period"],
            "tone": tone_for(snapshot["period_return"]),
        },
        {
            "label": "Annualised volatility",
            "value": format_percent(snapshot["annualized_volatility"]),
            "note": snapshot["volatility_note"] or "Sample daily volatility, sqrt(252)",
        },
    ]
)

chart_column, summary_column = st.columns([1.45, 1])
with chart_column:
    st.plotly_chart(
        price_chart(snapshot["enriched"], f"{display_name(symbol)} close and moving averages"),
        width="stretch",
    )
with summary_column:
    st.subheader("Technical summary")
    st.write(f"Rule-count stance: **{snapshot['summary_stance']}**")
    st.caption(snapshot["summary_rule"])
    for signal in snapshot["signals"]:
        st.write(f"{signal['indicator']}: {signal['reading']}. {signal['observation']}")

    st.subheader("Prediction summary")
    if prediction is None:
        st.info(
            f"No saved model for {symbol}, {settings['period']}, and a {settings['horizon']}-trading-day horizon. "
            "Open ML Prediction and choose Train and evaluate. A probability is not shown until that model exists."
        )
    else:
        metric_row(
            [
                {
                    "label": f"{settings['horizon']}-day direction",
                    "value": prediction["direction"],
                    "note": "Upward only if the model score is at least 50%",
                },
                {
                    "label": "Upward-movement score",
                    "value": format_percent(prediction["upward_probability"], 1),
                    "note": "Model estimate, not a calibrated probability",
                },
            ]
        )
        st.caption(prediction.get("probability_note", ""))
        st.caption(prediction.get("freshness_note", ""))
        accuracy = (prediction.get("metrics") or {}).get("accuracy")
        baseline = (prediction.get("baseline_majority") or {}).get("accuracy")
        st.write(
            f"Holdout accuracy {format_percent(accuracy)} versus majority-class baseline {format_percent(baseline)}."
        )

show_data_line(meta)
clean = (meta.get("clean_report") or {})
if clean:
    st.caption(
        f"Cleaning kept {clean.get('rows_out', 'N/A')} of {clean.get('rows_in', 'N/A')} rows. "
        f"Duplicate dates removed: {clean.get('duplicates_removed', 0)}. "
        f"Missing closes removed: {clean.get('missing_close_removed', 0)}. "
        f"Invalid price rows removed: {clean.get('invalid_rows_removed', 0)}."
    )
for warning in clean.get("warnings") or []:
    st.caption(warning)

with st.expander("Latest indicator values"):
    st.write(
        {
            "RSI (14)": format_number(snapshot["rsi_14"], 1),
            "MACD": format_number(snapshot["macd"], 2),
            "MACD signal": format_number(snapshot["macd_signal"], 2),
            "SMA 20": format_inr(snapshot["sma_20"]),
            "SMA 50": format_inr(snapshot["sma_50"]),
            "SMA 200": format_inr(snapshot["sma_200"]),
            "Support (20-day low)": format_inr(snapshot["support"]),
            "Resistance (20-day high)": format_inr(snapshot["resistance"]),
        }
    )

page_footer()
