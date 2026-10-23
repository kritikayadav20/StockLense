import streamlit as st

from config import DATA_SOURCE_NOTE, DISCLAIMER, HORIZONS, PERIODS, STOCKS
from src.technical_analysis import SUMMARY_RULE
from src.ui import bootstrap, page_header

bootstrap()
page_header(
    "Methodology",
    "What StockLens calculates, which sources it uses, and what the results do not mean.",
)

st.subheader("Data source")
st.write(DATA_SOURCE_NOTE)
st.write(
    "Daily bars are requested with yfinance for NSE tickers ending in .NS and BSE tickers ending in .BO. "
    "The app sorts by date, drops duplicate dates, removes rows with a missing or invalid price, and does not replace a missing close with zero. "
    "A fresh download is cached for six hours. If a new download fails and a cache exists, the cache is shown and labelled as stale. "
    "If there is no cache, the page shows an error and does not invent prices."
)
st.write("Built-in menu: " + ", ".join(f"{meta['name']} ({symbol})" for symbol, meta in STOCKS.items()) + ".")
st.write("Add another menu entry in config.py, or type any other .NS or .BO ticker in the sidebar.")

st.subheader("Prediction task")
st.write(
    "The model is an XGBoost classifier. For a horizon of 1, 5, or 10 trading days, the label is 1 when the close that many trading days later "
    "is higher than the close on the prediction date, and 0 otherwise. The available horizons are "
    + ", ".join(str(item) for item in HORIZONS)
    + f". The default history length is 1 year. Choices: {', '.join(PERIODS)}."
)
st.write(
    "Features at a date use returns, moving averages, RSI, MACD, Bollinger position, rolling volatility, volume change, "
    "the close relative to its moving averages, and the high-low range. Those calculations look backward. "
    "The future close is stored only as the label and as the outcome in the holdout simulation. It is not a model input."
)
st.write(
    "Rows are split in date order into training, validation, and test segments. They are not shuffled. "
    "The logistic baseline scales features with a scaler fit on the training rows only. "
    "Validation is reported and is not used to search for a better model, so the test segment remains the holdout. "
    "The direction shown for today uses the latest complete feature row. Its future close is unknown."
)
st.write(
    "The majority-class baseline always predicts the more common training outcome. "
    "The logistic baseline is a linear comparison. If XGBoost accuracy is close to the majority baseline, the page says so."
)
st.write(
    "Predicted probabilities are the model's class scores. They are not automatically calibrated probabilities of a real market event."
)

st.subheader("Hypothetical simulation")
st.write(
    "When the holdout is long enough, StockLens simulates a long-only rule: take the realised horizon return when the model predicted an upward move, "
    "then skip the overlapping days, and otherwise stay in cash. Each entry deducts an assumed 0.10% transaction cost and 0.05% slippage. "
    "Those are teaching assumptions. The result is labelled as a simulation. It is withheld when the holdout has fewer than 30 labelled days. "
    "A positive simulated number is not a claim of future profit."
)

st.subheader("Technical indicators")
st.write(SUMMARY_RULE)
st.write(
    "RSI uses Wilder's smoothing over 14 periods. MACD is the 12-period EMA minus the 26-period EMA, with a 9-period signal line. "
    "Bollinger Bands use a 20-day average and two standard deviations. Annualised volatility is the sample standard deviation of daily returns "
    "multiplied by the square root of 252. Support and resistance are the 20-day low and 20-day high, a rolling range rather than a confirmed pivot."
)

st.subheader("Fundamentals and news")
st.write(
    "Fundamental fields come from the Yahoo Finance data exposed by yfinance. If a field or statement line is missing, the page shows N/A. "
    "Revenue and net-income changes are described only across the periods the provider returned. "
    "News comes from NewsAPI when NEWS_API_KEY is set. Headlines keep the provider's date and source. Duplicate titles are removed. "
    "FinBERT (ProsusAI/finbert) scores each headline. The page averages the positive, negative, and neutral probabilities. "
    "That average is a description of the headlines, not a probability that the share will rise. "
    "If the key or the model is missing, the news page explains the gap and the other pages continue to work."
)

st.subheader("Explanations")
st.write(
    "The AI Insights page sends a structured summary of numbers StockLens has already calculated. "
    "Without LLM_API_KEY, a Python template fills those numbers into an explanation. "
    "With a key, an OpenAI-compatible chat endpoint can rewrite that brief. If the request fails, the template is shown. "
    "The model is instructed not to invent prices, results, news, or metrics."
)

st.subheader("Limits")
st.write(
    "StockLens does not place orders, does not know a personal financial situation, and does not cover derivatives, mutual funds, or markets outside India. "
    "Daily bars hide intraday moves. A single historical window can flatter or punish a model. "
    "Company accounts are not adjusted into a common template, so cross-company comparisons need sector context."
)
st.warning(DISCLAIMER)
