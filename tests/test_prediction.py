"""Training output, leakage-sensitive preprocessing, and the hypothetical simulation."""

import numpy as np
import pandas as pd
import pytest

from config import SLIPPAGE, TRANSACTION_COST
from src.ai_explainer import render_template
from src.backtesting import run_backtest
from src.errors import ModelTrainingError
from src.model_evaluation import fit_logistic_baseline
from src.stock_prediction import refresh_saved_prediction, train_and_evaluate
from tests.synthetic import make_ohlcv


def _context(probability=0.42):
    return {
        "symbol": "TEST.NS",
        "name": "Test Company",
        "period": "1 year",
        "horizon": 5,
        "price": {"close": 100, "as_of": "2024-01-01", "period_return": 0.1, "daily_change": 0.01},
        "technical": {
            "summary_stance": "Neutral",
            "rsi_14": 55,
            "macd": 1.2,
            "macd_signal": 0.4,
            "signals": [],
            "annualized_volatility": 0.2,
            "help": {},
        },
        "fundamentals": {
            "available": True,
            "rows": [{"metric": "P/E ratio", "display": "22.00"}],
            "interpretations": ["Revenue increased over the available reporting periods."],
        },
        "prediction": {
            "direction": "Upward",
            "upward_probability": probability,
            "probability_note": "This is a model score.",
            "metrics": {"accuracy": 0.55, "test_size": 40},
            "baseline_majority": {"accuracy": 0.5},
            "split": {"test_start": "2024-01-01", "test_end": "2024-06-01"},
            "warnings": [],
        },
        "sentiment": {"status": "missing", "message": "No headlines were analysed."},
    }


def test_scaler_uses_training_rows_only():
    rng = np.random.default_rng(0)
    x_train = pd.DataFrame({"a": rng.normal(0, 1, 40), "b": rng.normal(5, 1, 40)})
    y_train = np.array([0, 1] * 20)
    pipeline = fit_logistic_baseline(x_train, y_train)
    np.testing.assert_allclose(pipeline.named_steps["scaler"].mean_, x_train.mean().to_numpy())
    x_test = pd.DataFrame({"a": rng.normal(50, 1, 10), "b": rng.normal(-20, 1, 10)})
    combined_mean = pd.concat([x_train, x_test]).mean().to_numpy()
    assert not np.allclose(pipeline.named_steps["scaler"].mean_, combined_mean)


def test_insufficient_history_is_rejected():
    with pytest.raises(ModelTrainingError):
        train_and_evaluate(make_ohlcv(30), 5, "TEST.NS", "6 months", persist=False)


def test_prediction_output_and_saved_model(tmp_path):
    result = train_and_evaluate(
        make_ohlcv(280, seed=15),
        5,
        "TEST.NS",
        "2 years",
        model_dir=tmp_path,
        persist=False,
    )
    assert result["horizon_trading_days"] == 5
    assert result["direction"] in {"Upward", "Downward/Not Upward"}
    assert 0 <= result["upward_probability"] <= 1
    assert "future_close" not in {row["feature"] for row in result["feature_importance"]}
    assert result["split"]["train_end"] < result["split"]["test_start"]
    assert set(result["metrics"]) >= {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert len(result["confusion_matrix"]) == 2
    loaded = refresh_saved_prediction(make_ohlcv(280, seed=15), "TEST.NS", "2 years", 5, model_dir=tmp_path)
    assert loaded["direction"] in {"Upward", "Downward/Not Upward"}
    assert "probability" in loaded["probability_note"].lower() or "model" in loaded["probability_note"].lower()


def test_backtest_withholds_small_samples_and_deducts_costs():
    small = make_ohlcv(12)
    small["future_close"] = small["Close"].shift(-1)
    withheld = run_backtest(small.dropna(subset=["future_close"]), np.ones(11, dtype=int), horizon=1)
    assert withheld["status"] == "withheld"
    assert "cumulative_net_return" not in withheld

    frame = make_ohlcv(40, seed=2)
    frame["future_close"] = frame["Close"].shift(-5)
    usable = frame.dropna(subset=["future_close"]).reset_index(drop=True)
    signals = np.zeros(len(usable), dtype=int)
    signals[0] = 1
    result = run_backtest(usable, signals, horizon=5)
    gross = usable["future_close"].iloc[0] / usable["Close"].iloc[0] - 1
    assert result["status"] == "ok"
    assert result["trades"] == 1
    assert result["average_net_return"] == pytest.approx(gross - TRANSACTION_COST - SLIPPAGE)

    all_up = np.ones(len(usable), dtype=int)
    many = run_backtest(usable, all_up, horizon=5)
    assert many["trades"] < len(usable)


def test_template_uses_calculated_probability_only():
    text = render_template(_context(0.42))
    assert "42.0%" in text
    assert "Test Company" in text
    assert "not a probability that the share price will rise" in text.lower() or "not a guaranteed chance" in text.lower()
    missing = _context()
    missing["prediction"] = None
    without_model = render_template(missing)
    assert "42.0%" not in without_model
    assert "No trained model" in without_model
