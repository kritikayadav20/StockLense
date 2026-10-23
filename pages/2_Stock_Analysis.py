import pandas as pd
import streamlit as st

from config import display_name
from src.charts import macd_chart, price_chart, rsi_chart, volume_chart
from src.formatting import format_inr, format_number, format_percent
from src.technical_analysis import INDICATOR_HELP, technical_snapshot
from src.ui import bootstrap, load_selected_history, metric_row, page_footer, page_header, show_data_line, show_frame

settings = bootstrap()
page_header(
    "Stock Analysis",
    "Price history and technical indicators calculated from the downloaded bars. These readings are not the machine-learning prediction.",
)
history = load_selected_history(settings)
if history is None:
    page_footer()
    st.stop()

symbol = history["symbol"]
snapshot = technical_snapshot(history["frame"])
enriched = snapshot["enriched"]
show_adjusted = st.checkbox("Show adjusted close when the provider supplied it", value=False)

st.plotly_chart(
    price_chart(
        enriched,
        f"{display_name(symbol)} ({symbol})",
        show_adjusted=show_adjusted,
        candlestick=True,
    ),
    width="stretch",
)
left, right = st.columns(2)
with left:
    st.plotly_chart(rsi_chart(enriched), width="stretch")
with right:
    st.plotly_chart(macd_chart(enriched), width="stretch")
st.plotly_chart(volume_chart(enriched), width="stretch")

metric_row(
    [
        {"label": "RSI (14)", "value": format_number(snapshot["rsi_14"], 1), "note": "Above 70 overbought, below 30 oversold"},
        {"label": "MACD", "value": format_number(snapshot["macd"], 2), "note": f"Signal {format_number(snapshot['macd_signal'], 2)}"},
        {"label": "SMA 50", "value": format_inr(snapshot["sma_50"]), "note": f"Close {format_inr(snapshot['close'])}"},
        {"label": "Annualised volatility", "value": format_percent(snapshot["annualized_volatility"]), "note": "Historical, not a forecast"},
    ]
)

st.subheader("Technical interpretation")
st.write(f"Summary stance: **{snapshot['summary_stance']}**")
st.caption(snapshot["summary_rule"])
show_frame(pd.DataFrame(snapshot["signals"]))
st.caption(snapshot["support_method"])
st.write(
    f"Support estimate {format_inr(snapshot['support'])}. "
    f"Resistance estimate {format_inr(snapshot['resistance'])}. "
    f"Average volume {format_number(snapshot['average_volume'], 0)}. "
    f"Latest volume change {format_percent(snapshot['volume_change'])}."
)

with st.expander("What each indicator means"):
    for name, explanation in INDICATOR_HELP.items():
        st.write(f"**{name}.** {explanation}")

show_data_line(history["meta"])
page_footer()
