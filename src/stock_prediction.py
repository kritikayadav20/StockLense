"""Train an XGBoost direction model and score the latest available bar.

The fit uses a chronological split. Preprocessing for the linear baseline is fit
on the training rows only. The latest bar can be scored because its features end
on that date; its future close is unknown and is not required for the estimate.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config import get_model_dir
from src.backtesting import run_backtest
from src.database import save_prediction, utc_now
from src.errors import DatabaseError, ModelTrainingError
from src.feature_engineering import FEATURE_COLUMNS, FEATURE_LABELS, build_feature_frame, labeled_frame
from src.formatting import to_builtin
from src.model_evaluation import (
    chronological_split,
    compute_classification_metrics,
    confusion_and_report,
    fit_logistic_baseline,
    fit_majority_baseline,
    predict_upward_probability,
)

LIMITATIONS = [
    "The label only records whether the close after the chosen number of trading days is higher than the close on the prediction date. It is not a price target.",
    "Features on a date use that day's bar and earlier bars. The future close is used only as the training label and the holdout outcome.",
    "Rows are split in date order and are not shuffled. The reported accuracy is the later holdout, not a randomly mixed score.",
    "Predicted probabilities are model estimates. They should not be treated as calibrated real-world probabilities.",
    "A result for one stock and horizon is not a ranking against another stock. Sample size and baseline accuracy can differ.",
    "Nothing here is a claim that the model is profitable. The simulation, when shown, is a historical illustration with assumed costs.",
]


def _artifact_path(symbol: str, period: str, horizon: int, model_dir: Path | None = None) -> Path:
    directory = Path(model_dir) if model_dir is not None else get_model_dir()
    directory.mkdir(parents=True, exist_ok=True)
    token = re.sub(r"[^A-Za-z0-9]+", "_", f"{symbol}_{period}_{horizon}d").strip("_")
    return directory / f"{token}.joblib"


def _direction(probability: float) -> str:
    return "Upward" if probability >= 0.5 else "Downward/Not Upward"


def _score_row(model, row: pd.Series) -> float:
    features = pd.DataFrame([row[FEATURE_COLUMNS].astype(float).to_dict()])
    return float(predict_upward_probability(model, features)[0])


def train_and_evaluate(
    frame: pd.DataFrame,
    horizon: int,
    symbol: str,
    period: str,
    model_dir: Path | None = None,
    persist: bool = True,
) -> dict:
    """Train on earlier history, evaluate on later history, and save the model."""
    labeled = labeled_frame(frame, horizon)
    if labeled.empty:
        raise ModelTrainingError(
            "Not enough history to calculate the indicators and a known future close. "
            "The 50-day average removes the earliest rows, and the prediction horizon removes the latest rows. "
            "Choose a longer period and train again."
        )
    if labeled["target"].nunique() < 2:
        raise ModelTrainingError(
            "This historical window contains only one outcome (always upward, or never upward), so a classifier cannot be trained. "
            "Choose a longer period or a different stock."
        )

    train, validation, test = chronological_split(labeled)
    x_train = train[FEATURE_COLUMNS]
    y_train = train["target"]
    positive = int((y_train == 1).sum())
    negative = int((y_train == 0).sum())
    scale_pos_weight = float(negative / positive) if positive else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        scale_pos_weight=scale_pos_weight,
    )
    # Validation is not used to pick a model. It is scored only so the middle period is visible.
    model.fit(x_train, y_train, verbose=False)

    def evaluate(split):
        probabilities = predict_upward_probability(model, split[FEATURE_COLUMNS])
        predictions = (probabilities >= 0.5).astype(int)
        metrics = compute_classification_metrics(split["target"], predictions, probabilities)
        return probabilities, predictions, metrics

    validation_proba, validation_pred, validation_metrics = evaluate(validation)
    test_proba, test_pred, test_metrics = evaluate(test)
    matrix, report = confusion_and_report(test["target"], test_pred)
    del validation_proba, validation_pred

    majority = fit_majority_baseline(x_train, y_train)
    majority_proba = predict_upward_probability(majority, test[FEATURE_COLUMNS])
    majority_pred = majority.predict(test[FEATURE_COLUMNS])
    majority_metrics = compute_classification_metrics(test["target"], majority_pred, majority_proba)

    logistic_metrics = {"status": "ok"}
    try:
        logistic = fit_logistic_baseline(x_train, y_train)
        logistic_proba = predict_upward_probability(logistic, test[FEATURE_COLUMNS])
        logistic_pred = (logistic_proba >= 0.5).astype(int)
        logistic_metrics.update(compute_classification_metrics(test["target"], logistic_pred, logistic_proba))
    except Exception as exc:  # noqa: BLE001 - show a usable baseline failure without losing XGBoost.
        logistic_metrics = {"status": "unavailable", "reason": f"The logistic baseline could not be fit ({exc.__class__.__name__})."}

    featured = build_feature_frame(frame, horizon)
    complete = featured.dropna(subset=FEATURE_COLUMNS)
    if complete.empty:
        raise ModelTrainingError("Indicators could not be calculated for any row, so there is nothing to score.")
    latest = complete.iloc[-1]
    probability = _score_row(model, latest)
    warnings = []
    if pd.Timestamp(latest["Date"]) != pd.Timestamp(featured.iloc[-1]["Date"]):
        warnings.append("The latest bar is missing an indicator input, so the estimate uses the last complete bar.")
    positive_rate = float(y_train.mean())
    minority = min(positive_rate, 1 - positive_rate)
    if minority < 0.2:
        warnings.append(
            f"The training window is imbalanced: {positive_rate:.0%} of training rows are upward moves. "
            "Accuracy can look strong while a model mostly repeats the common class. Compare precision, recall, and the baselines."
        )
    if abs(test_metrics["accuracy"] - majority_metrics["accuracy"]) < 0.02:
        warnings.append(
            "XGBoost accuracy on the holdout is close to the majority-class baseline. "
            "This sample does not show a clear accuracy advantage."
        )
    if test_metrics["test_size"] < 30:
        warnings.append(
            f"The holdout has only {test_metrics['test_size']} labelled days. "
            "Read the metrics as a description of that short window, not as evidence of forecasting skill."
        )
    if test_metrics["roc_auc"] is None:
        warnings.append("ROC-AUC is unavailable because the holdout contains only one actual class.")

    importance = (
        pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
        .sort_values(ascending=False)
    )
    importance_rows = [
        {"feature": feature, "label": FEATURE_LABELS.get(feature, feature), "importance": float(value)}
        for feature, value in importance.items()
    ]
    backtest = run_backtest(test, test_pred, horizon)
    trained_at = utc_now()
    result = {
        "symbol": symbol,
        "period": period,
        "horizon_trading_days": horizon,
        "trained_at": trained_at,
        "training_last_date": str(pd.Timestamp(labeled["Date"].iloc[-1]).date()),
        "as_of_date": str(pd.Timestamp(latest["Date"]).date()),
        "latest_available_close": float(latest["Close"]),
        "price_label": "Latest available close used for this estimate. This is not a live quote.",
        "direction": _direction(probability),
        "upward_probability": probability,
        "decision_threshold": 0.5,
        "probability_note": (
            "This percentage is the XGBoost estimate for an upward close over the selected horizon. "
            "It is not automatically a calibrated real-world probability, and it is not a recommendation."
        ),
        "metrics": test_metrics,
        "validation_metrics": validation_metrics,
        "baseline_majority": majority_metrics,
        "baseline_logistic": logistic_metrics,
        "confusion_matrix": matrix,
        "confusion_labels": ["Not upward", "Upward"],
        "classification_report": report,
        "feature_importance": importance_rows,
        "backtest": backtest,
        "warnings": warnings,
        "limitations": LIMITATIONS,
        "split": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
            "train_end": str(pd.Timestamp(train["Date"].iloc[-1]).date()),
            "validation_start": str(pd.Timestamp(validation["Date"].iloc[0]).date()),
            "validation_end": str(pd.Timestamp(validation["Date"].iloc[-1]).date()),
            "test_start": str(pd.Timestamp(test["Date"].iloc[0]).date()),
            "test_end": str(pd.Timestamp(test["Date"].iloc[-1]).date()),
        },
        "hyperparameters": {
            "model": "XGBClassifier",
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 3,
            "objective": "binary:logistic",
            "random_state": 42,
            "scale_pos_weight": scale_pos_weight,
        },
        "positive_rate_train": positive_rate,
    }
    result = to_builtin(result)

    artifact = {
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS),
        "symbol": symbol,
        "period": period,
        "horizon": horizon,
        "result": result,
    }
    joblib.dump(artifact, _artifact_path(symbol, period, horizon, model_dir))

    if persist:
        try:
            save_prediction(
                {
                    "symbol": symbol,
                    "period": period,
                    "horizon": horizon,
                    "created_at": trained_at,
                    "as_of_date": result["as_of_date"],
                    "direction": result["direction"],
                    "upward_probability": result["upward_probability"],
                    "test_accuracy": test_metrics["accuracy"],
                    "baseline_accuracy": majority_metrics["accuracy"],
                    "metrics_json": json.dumps(to_builtin({"metrics": test_metrics, "warnings": warnings})),
                }
            )
        except DatabaseError as exc:
            result["warnings"].append(str(exc) + " The model file was still saved.")
    return result


def load_artifact(symbol: str, period: str, horizon: int, model_dir: Path | None = None):
    path = _artifact_path(symbol, period, horizon, model_dir)
    if not path.exists():
        return None
    try:
        artifact = joblib.load(path)
    except Exception:
        return None
    if artifact.get("symbol") != symbol or artifact.get("period") != period or artifact.get("horizon") != horizon:
        return None
    if list(artifact.get("feature_columns", [])) != list(FEATURE_COLUMNS):
        return None
    return artifact


def refresh_saved_prediction(frame: pd.DataFrame, symbol: str, period: str, horizon: int, model_dir: Path | None = None):
    """Apply a saved model to the current latest bar without retraining."""
    artifact = load_artifact(symbol, period, horizon, model_dir)
    if artifact is None:
        return None
    featured = build_feature_frame(frame, horizon)
    complete = featured.dropna(subset=FEATURE_COLUMNS)
    if complete.empty:
        return None
    latest = complete.iloc[-1]
    probability = _score_row(artifact["model"], latest)
    result = dict(artifact["result"])
    result["as_of_date"] = str(pd.Timestamp(latest["Date"]).date())
    result["latest_available_close"] = float(latest["Close"])
    result["upward_probability"] = probability
    result["direction"] = _direction(probability)
    warnings = list(result.get("warnings") or [])
    training_last = result.get("training_last_date")
    current_last = str(pd.Timestamp(frame["Date"].iloc[-1]).date())
    freshness = (
        f"Evaluation metrics were calculated on the training run through {training_last}. "
        f"The direction above uses indicator values as of {result['as_of_date']}."
    )
    if training_last and current_last > str(training_last):
        freshness += " Newer price history is available. Train again if you want that history included in the evaluation."
    result["freshness_note"] = freshness
    result["warnings"] = warnings
    return to_builtin(result)
