import pandas as pd
import streamlit as st

from config import display_name
from src.ai_explainer import build_conflicts, build_risks, explain
from src.database import DatabaseError, save_analysis
from src.formatting import to_builtin
from src.stock_prediction import refresh_saved_prediction
from src.technical_analysis import INDICATOR_HELP, technical_snapshot
from src.ui import (
    bootstrap,
    cached_fundamentals,
    load_selected_history,
    page_footer,
    page_header,
    show_markdown,
)

SUGGESTIONS = [
    "Why does this stock have this outlook?",
    "What are the main risks?",
    "What do the RSI and MACD values mean?",
    "Summarise the latest analysed news.",
    "Explain the company's available fundamentals.",
    "Compare the selected stocks.",
]


def build_context(settings, history, snapshot, fundamentals, prediction, sentiment) -> dict:
    as_of = str(pd.Timestamp(snapshot["last_date"]).date())
    technical = {key: value for key, value in snapshot.items() if key != "enriched"}
    technical["last_date"] = as_of
    technical["help"] = INDICATOR_HELP
    context = {
        "symbol": history["symbol"],
        "name": display_name(history["symbol"]),
        "period": settings["period"],
        "horizon": settings["horizon"],
        "price": {
            "close": snapshot["close"],
            "as_of": as_of,
            "period_return": snapshot["period_return"],
            "daily_change": snapshot["daily_change"],
        },
        "technical": technical,
        "fundamentals": fundamentals,
        "prediction": prediction,
        "sentiment": sentiment or {"status": "missing", "message": "Headlines have not been analysed in this session."},
        "comparison": st.session_state.get("comparison_rows") or [],
    }
    return to_builtin(context)


settings = bootstrap()
page_header(
    "AI Insights",
    "An explanation built from the calculated price, model, technical, fundamental, and sentiment results. Missing inputs stay missing.",
)
history = load_selected_history(settings)
if history is None:
    page_footer()
    st.stop()

symbol = history["symbol"]
snapshot = technical_snapshot(history["frame"])
with st.spinner("Loading fundamentals for the explanation..."):
    fundamentals = cached_fundamentals(symbol)
prediction = refresh_saved_prediction(history["frame"], symbol, settings["period"], settings["horizon"])
sentiment = st.session_state.get(f"sentiment::{symbol}")
context = build_context(settings, history, snapshot, fundamentals, prediction, sentiment)

st.subheader("Evidence in this explanation")
left, right = st.columns(2)
with left:
    st.markdown("**Supporting or descriptive points**")
    if prediction:
        st.write(f"Model direction over {settings['horizon']} trading days: {prediction['direction']}.")
    else:
        st.write("No stored model for this selection.")
    st.write(f"Technical summary: {snapshot['summary_stance']}.")
    if sentiment and sentiment.get("status") == "ok":
        st.write(f"Headline sentiment: {sentiment['overall_label']} across {sentiment['article_count']} articles.")
    else:
        st.write("Headline sentiment is not loaded.")
with right:
    st.markdown("**Conflicts**")
    for item in build_conflicts(context):
        st.write(f"- {item}")
    st.markdown("**Risks and gaps**")
    for item in build_risks(context):
        st.write(f"- {item}")

if st.button("Build explanation", type="primary"):
    result = explain(context)
    st.session_state["ai_result"] = result
    try:
        save_analysis(symbol, "explanation", result["text"])
    except DatabaseError as exc:
        st.caption(str(exc))

result = st.session_state.get("ai_result")
if result:
    for warning in result.get("warnings") or []:
        st.info(warning)
    st.caption("Source: language model" if result["source"] == "llm" else "Source: template based on calculated results")
    show_markdown(result["text"])

st.subheader("Ask a question")
st.caption("Answers use the same calculated context. A question does not create new market data.")
question = st.text_input("Question", key="ai_question", placeholder="Why does this stock have this outlook?")
clicked = None
columns = st.columns(3)
for index, suggestion in enumerate(SUGGESTIONS):
    if columns[index % 3].button(suggestion, key=f"suggest_{index}"):
        clicked = suggestion

if st.button("Answer question") or clicked:
    answer = explain(context, clicked or question)
    st.session_state["ai_answer"] = answer

answer = st.session_state.get("ai_answer")
if answer:
    for warning in answer.get("warnings") or []:
        st.info(warning)
    show_markdown(answer["text"])

page_footer()
