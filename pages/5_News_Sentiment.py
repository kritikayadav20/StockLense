import pandas as pd
import streamlit as st

from config import display_name
from src.charts import sentiment_chart
from src.errors import SentimentModelError
from src.formatting import format_percent
from src.news_collector import fetch_company_news
from src.sentiment_analysis import aggregate_sentiments, analyse_text, analyse_texts
from src.ui import bootstrap, metric_row, page_footer, page_header, show_frame

settings = bootstrap()
page_header(
    "News and sentiment",
    "FinBERT scores real headlines when a news key is configured, and it can score a headline you paste. Scores describe the text, not the chance of a price rise.",
)

symbol = settings["symbol"]
company = display_name(symbol)
st.write(f"Selected company: **{company}** ({symbol})")

if st.button("Fetch headlines", type="primary"):
    with st.spinner("Requesting headlines..."):
        news = fetch_company_news(company, symbol)
    st.session_state[f"news_raw::{symbol}"] = news
    st.session_state.pop(f"sentiment::{symbol}", None)
else:
    news = st.session_state.get(f"news_raw::{symbol}")

if news is None:
    st.info("Headlines are not loaded yet. Fetch them after NEWS_API_KEY is set, or paste a headline below. The rest of StockLens works without news.")
elif news["status"] != "ok":
    st.warning(news["message"])
else:
    st.caption(news["message"])
    if st.button("Analyse headlines with FinBERT"):
        try:
            with st.spinner("Loading FinBERT and scoring headlines. The first run downloads the model and needs memory and an internet connection."):
                scored = analyse_texts([article["title"] for article in news["articles"]])
        except SentimentModelError as exc:
            st.error(str(exc))
            scored = None
        if scored:
            combined = []
            for article, score in zip(news["articles"], scored):
                combined.append(
                    {
                        "title": article["title"],
                        "source": article["source"],
                        "published_at": article["published_at"],
                        "url": article["url"],
                        "label": score["label"],
                        "confidence": score["confidence"],
                        "probabilities": score["probabilities"],
                    }
                )
            summary = aggregate_sentiments(scored)
            summary["status"] = "ok"
            summary["headlines"] = combined
            summary["message"] = summary["method"]
            st.session_state[f"sentiment::{symbol}"] = summary

sentiment = st.session_state.get(f"sentiment::{symbol}")
if sentiment and sentiment.get("status") == "ok":
    means = sentiment["mean_probabilities"]
    metric_row(
        [
            {"label": "Headlines analysed", "value": sentiment["article_count"], "note": "Duplicates removed"},
            {"label": "Positive", "value": sentiment["positive_count"], "note": format_percent(means["positive"], 1) + " average"},
            {"label": "Neutral", "value": sentiment["neutral_count"], "note": format_percent(means["neutral"], 1) + " average"},
            {"label": "Negative", "value": sentiment["negative_count"], "note": format_percent(means["negative"], 1) + " average"},
        ]
    )
    st.write(f"Overall label from average probabilities: **{sentiment['overall_label']}**")
    st.caption(sentiment["method"])
    st.plotly_chart(sentiment_chart(means), width="stretch")
    table = pd.DataFrame(
        [
            {
                "Published": item.get("published_at"),
                "Source": item.get("source"),
                "Headline": item.get("title"),
                "Sentiment": item.get("label"),
                "Confidence": format_percent(item.get("confidence"), 1),
            }
            for item in sentiment["headlines"]
        ]
    )
    show_frame(table, height=360)
elif news and news.get("status") == "ok":
    show_frame(
        pd.DataFrame(news["articles"])[["published_at", "source", "title"]].rename(
            columns={"published_at": "Published", "source": "Source", "title": "Headline"}
        ),
        height=320,
    )
    st.caption("These headlines are not scored yet. Run FinBERT to add sentiment. No score has been guessed.")

st.subheader("Analyse a headline you provide")
st.caption("This box is for text you supply. It is not a news feed, and it is not added to the company sentiment average.")
user_text = st.text_area("Headline", placeholder="Paste a financial headline you already have", height=100)
if st.button("Analyse this text"):
    if not user_text.strip():
        st.warning("Enter a headline first.")
    else:
        try:
            with st.spinner("Scoring the text you entered..."):
                user_score = analyse_text(user_text)
            st.session_state["user_headline_score"] = user_score
        except SentimentModelError as exc:
            st.error(str(exc))

user_score = st.session_state.get("user_headline_score")
if user_score:
    st.write(f"User-provided text: {user_score['text']}")
    st.write(
        f"Label: **{user_score['label']}** (confidence {format_percent(user_score['confidence'], 1)}). "
        f"Positive {format_percent(user_score['probabilities']['positive'], 1)}, "
        f"neutral {format_percent(user_score['probabilities']['neutral'], 1)}, "
        f"negative {format_percent(user_score['probabilities']['negative'], 1)}."
    )

page_footer()
