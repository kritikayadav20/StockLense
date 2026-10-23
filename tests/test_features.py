"""Feature calculations, targets, and leakage checks. Prices here are synthetic."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import FEATURE_COLUMNS, FORBIDDEN_FEATURE_COLUMNS, build_feature_frame
from src.model_evaluation import chronological_split
from tests.synthetic import make_ohlcv


def test_feature_list_excludes_future_and_raw_prices():
    assert "target" not in FEATURE_COLUMNS
    assert "future_close" not in FEATURE_COLUMNS
    assert FORBIDDEN_FEATURE_COLUMNS.isdisjoint(FEATURE_COLUMNS)


def test_known_return_range_and_target():
    frame = make_ohlcv(80)
    frame.loc[frame.index[-2], "Close"] = 100
    frame.loc[frame.index[-1], "Close"] = 110
    frame.loc[frame.index[-1], "High"] = 112
    frame.loc[frame.index[-1], "Low"] = 108
    featured = build_feature_frame(frame, horizon=1)
    assert featured["return_1d"].iloc[-1] == pytest.approx(0.10)
    assert featured["high_low_range"].iloc[-1] == pytest.approx(4 / 110)
    assert np.isnan(featured["target"].iloc[-1])
    assert featured["target"].iloc[-2] == 1


def test_target_matches_future_close_direction():
    frame = make_ohlcv(100, seed=3)
    horizon = 5
    featured = build_feature_frame(frame, horizon)
    index = 60
    expected = int(frame["Close"].iloc[index + horizon] > frame["Close"].iloc[index])
    assert featured["target"].iloc[index] == expected
    assert featured["future_close"].iloc[index] == pytest.approx(frame["Close"].iloc[index + horizon])


def test_future_close_does_not_change_past_features():
    frame = make_ohlcv(300, seed=11)
    horizon = 5
    index = len(frame) - 1 - horizon
    original = build_feature_frame(frame, horizon)
    changed = frame.copy()
    current = float(changed["Close"].iloc[index])
    future = float(changed["Close"].iloc[index + horizon])
    changed.loc[changed.index[-1], "Close"] = current * (0.5 if future > current else 1.5)
    altered = build_feature_frame(changed, horizon)
    pd.testing.assert_series_equal(
        original.loc[index, FEATURE_COLUMNS].astype(float),
        altered.loc[index, FEATURE_COLUMNS].astype(float),
        check_names=False,
    )
    assert original.loc[index, "target"] != altered.loc[index, "target"]


def test_chronological_split_does_not_overlap_or_shuffle():
    featured = build_feature_frame(make_ohlcv(300, seed=4), horizon=5)
    labeled = featured.dropna(subset=FEATURE_COLUMNS + ["target"]).copy()
    labeled["target"] = labeled["target"].astype(int)
    train, validation, test = chronological_split(labeled)
    assert train["Date"].max() < validation["Date"].min()
    assert validation["Date"].max() < test["Date"].min()
    assert list(train["Date"]) == sorted(train["Date"])
    assert len(train) + len(validation) + len(test) == len(labeled)


def test_indicator_rows_are_finite_after_warmup():
    featured = build_feature_frame(make_ohlcv(160, seed=8), horizon=5)
    complete = featured.dropna(subset=FEATURE_COLUMNS)
    assert len(complete) > 50
    assert np.isfinite(complete[FEATURE_COLUMNS].to_numpy()).all()
