import pandas as pd
import streamlit as st

from config import display_name
from src.charts import statement_chart
from src.ui import (
    bootstrap,
    cached_fundamentals,
    load_selected_history,
    page_footer,
    page_header,
    show_frame,
)

settings = bootstrap()
page_header(
    "Fundamentals",
    "Provider-reported company figures. Missing metrics stay N/A. One ratio is not a verdict on financial strength.",
)
history = load_selected_history(settings)
if history is None:
    page_footer()
    st.stop()

symbol = history["symbol"]
with st.spinner("Loading fundamental data..."):
    fundamentals = cached_fundamentals(symbol)

st.write(f"**{fundamentals.get('long_name') or display_name(symbol)}** ({symbol})")
profile = []
if fundamentals.get("sector"):
    profile.append(f"Sector: {fundamentals['sector']}")
if fundamentals.get("industry"):
    profile.append(f"Industry: {fundamentals['industry']}")
if fundamentals.get("as_of"):
    profile.append(f"Provider reporting marker: {fundamentals['as_of']}")
if fundamentals.get("currency"):
    profile.append(f"Reported currency: {fundamentals['currency']}")
if profile:
    st.caption(" | ".join(profile))
if fundamentals.get("currency_note"):
    st.warning(fundamentals["currency_note"])
if fundamentals.get("error"):
    st.warning(fundamentals["error"])

rows = fundamentals.get("rows") or []
if rows:
    show_frame(pd.DataFrame(rows)[["metric", "display", "note"]].rename(
        columns={"metric": "Metric", "display": "Value", "note": "Note"}
    ))
else:
    st.info("No fundamental metric table is available for this symbol.")

left, right = st.columns(2)
revenue = fundamentals.get("revenue") or []
income = fundamentals.get("net_income") or []
with left:
    if len(revenue) >= 1:
        st.plotly_chart(statement_chart(revenue, "Revenue", fundamentals.get("currency")), width="stretch")
    else:
        st.info("Revenue was not found in the income statement returned by the provider.")
with right:
    if len(income) >= 1:
        st.plotly_chart(statement_chart(income, "Net income", fundamentals.get("currency")), width="stretch")
    else:
        st.info("Net income was not found in the income statement returned by the provider.")

if fundamentals.get("statement_note"):
    st.caption(fundamentals["statement_note"])

st.subheader("Reading the available figures")
for line in fundamentals.get("interpretations") or []:
    st.write(f"- {line}")
st.caption(
    "Avoid comparing these figures with another company without accounting for sector and accounting differences. "
    "StockLens does not fill missing statement lines with zero."
)
page_footer()
