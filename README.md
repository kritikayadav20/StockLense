# StockLens

StockLens is an educational analysis app for Indian stocks listed on the NSE and BSE. It puts historical prices, technical indicators, an XGBoost direction model, company fundamentals, FinBERT news sentiment, and a plain-English explanation in one Streamlit dashboard.

It does not place trades, and it does not promise that a price will rise or fall.

## Features

- Download daily NSE (`.NS`) and BSE (`.BO`) history with `yfinance`, then clean and cache it.
- Chart the close, moving averages, RSI, MACD, and volume.
- Train an XGBoost classifier for a 1, 5, or 10 trading-day horizon: will the future close be higher than today's close?
- Compare that model with a majority-class baseline and a logistic regression baseline on a chronological holdout.
- Show a hypothetical holdout simulation with assumed costs, and withhold it when the sample is too small.
- Display fundamental metrics and statement trends only when the provider returns them.
- Score real headlines with ProsusAI/finbert when a news API key is configured.
- Explain the calculated results with a template, or with an optional language-model API.
- Compare up to three Indian stocks without ranking them by model score alone.
- Save download logs, prediction records, and explanation notes in SQLite.

## Technologies

- Python, Streamlit, and Plotly for the dashboard
- yfinance, Pandas, and NumPy for market data
- scikit-learn, XGBoost, and Joblib for the model
- Transformers and PyTorch for FinBERT
- Requests for the optional news and language-model APIs
- SQLite for local history

Technical indicators are calculated in this project with Pandas and NumPy. `pandas-ta` was not used because its native builds are often incompatible with current Python environments. The formulas are documented on the Methodology page.

## Project structure

```text
StockLens/
├── app.py                      # Streamlit entry point
├── config.py                   # Stock list, periods, paths, disclaimers
├── requirements.txt
├── .env.example
├── pages/                      # Overview, analysis, model, fundamentals, news, AI, compare, methodology
├── src/                        # Data, indicators, model, news, sentiment, explanations, database
├── tests/                      # Unit tests, using synthetic prices only
├── data/                       # Cache and SQLite files are created here at runtime
└── models/saved/               # Trained Joblib files are created here at runtime
```

## Python version

Use Python 3.10 or newer. This project was built and tested with Python 3.11 on Windows.

## Setup on Windows PowerShell

From the project folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Windows blocks the activation script, run this once in the same window and then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Copy the environment file and edit it only if you want news or language-model features:

```powershell
Copy-Item .env.example .env
```

## Optional API keys

Create `.env` from `.env.example`. Leave a value blank if you do not have that integration.

| Variable | Purpose |
| --- | --- |
| `NEWS_API_KEY` | NewsAPI.org key for recent headlines. [newsapi.org](https://newsapi.org) |
| `NEWS_API_BASE_URL` | Defaults to `https://newsapi.org/v2` |
| `LLM_API_KEY` | Key for an OpenAI-compatible chat endpoint |
| `LLM_BASE_URL` | Defaults to `https://api.openai.com/v1` |
| `LLM_MODEL` | Defaults to `gpt-4o-mini` |
| `FINBERT_MODEL` | Defaults to `ProsusAI/finbert` |
| `DATABASE_PATH` | Defaults to `data/stocklens.db` |

The app never prints these keys. Historical prices, indicators, fundamentals, and the model do not need them.

Without `NEWS_API_KEY`, the News page explains how to add one. You can still paste your own headline for FinBERT. Without `LLM_API_KEY`, AI Insights uses a Python template filled with the numbers StockLens calculated.

## Run the app

```powershell
streamlit run app.py
```

Use the sidebar to pick a stock such as `RELIANCE.NS`, a history length, and a prediction horizon. The default history is one year. Open **ML Prediction** and click **Train and evaluate** when you want a model. Training does not run on every page refresh.

To add a stock to the menu, add a row to `STOCKS` in `config.py`. Any other Yahoo Finance `.NS` or `.BO` ticker can also be typed into **Other NSE or BSE ticker**.

## Run the tests

```powershell
pytest
```

The tests use synthetic prices so they can check targets, leakage, and validation. Those fixtures are not market data and are not shown in the dashboard.

## How stock prediction works

For the horizon you choose, each historical day gets a label:

- `1` if the close that many trading days later is higher than the close on that day
- `0` otherwise

The features on that day are returns, moving averages, RSI, MACD, Bollinger position, rolling volatility, volume change, price relative to the moving averages, and the day's high-low range. They use the current bar and earlier bars only.

The rows are split in time order into train, validation, and test. They are not shuffled. XGBoost is fit on the training rows. Accuracy, precision, recall, F1, and ROC-AUC (when both classes exist) are reported on the later holdout, next to a majority-class baseline and a logistic baseline. The logistic scaler is fit on the training rows only.

The number on the dashboard is the model's estimated probability of an upward close over that horizon. It is a model score, not a calibrated real-world probability, and it is not a recommendation. The app does not claim the model is profitable. When the holdout is long enough, a separate section shows a hypothetical long-only simulation with a 0.10% transaction-cost assumption and a 0.05% slippage assumption.

## How FinBERT sentiment works

Headlines come from NewsAPI when a key is configured. Titles are normalised and duplicates are removed. Each remaining headline is classified by `ProsusAI/finbert` into positive, negative, and neutral. The page averages those class probabilities and also counts each headline's top class.

That summary describes the headlines. It is not the probability that the stock will rise.

The first analysis downloads the model, so it needs an internet connection and enough memory for PyTorch. If the download or the model fails, StockLens shows the error and does not invent sentiment scores. A text box on the same page can score a headline you paste. That text is labelled as user-provided and is not mixed into the company average.

## Data source limitations

`yfinance` is an unofficial interface to Yahoo Finance. Bars can be delayed, missing, revised, or temporarily unavailable, and permitted use is set by the provider. The latest close in StockLens is the last historical bar in the download. It is not a live quote.

If the download fails and a cache exists, the cache is labelled as stale. If nothing is cached, the page shows an error. StockLens does not generate fake prices, fake news, or fake financial metrics. Missing fundamentals are shown as N/A.

## Known limitations

- The label is only an up or not-up close over a few trading days. It is not a price target.
- One chronological holdout can make a model look better or worse than it will on another period.
- Daily bars hide intraday moves, and the cost assumptions are not a broker's live schedule.
- Model scores for two companies are not directly a ranking. Their samples and baselines can differ.
- Fundamental fields differ by company and can be missing. They are not adjusted onto one accounting template.
- News coverage depends on the news plan and the query. No coverage means no sentiment, not a neutral score.
- The optional language model only receives numbers already calculated by StockLens. It can still phrase them poorly; the template remains available when the API call fails.

## Educational disclaimer

StockLens is an educational analysis tool. It is not investment advice, not a guaranteed prediction system, and not an automated trading platform. Past prices and model scores do not determine future returns.
