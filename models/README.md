# Models directory

Training from the ML Prediction page saves an XGBoost model and its evaluation metadata with Joblib under `models/saved/`.

Those files are created on your machine from the history you download. They are git-ignored. Delete `models/saved/` if you want the app to train again from scratch.

FinBERT is not stored in this folder. Hugging Face downloads `ProsusAI/finbert` into the local Hugging Face cache the first time you analyse a headline.
