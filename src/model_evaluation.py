"""Chronological evaluation and the two baseline classifiers."""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import MIN_LABELED_ROWS, MIN_TEST_ROWS, MIN_TRAIN_ROWS, MIN_VAL_ROWS
from src.errors import ModelTrainingError


def chronological_split(
    frame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
):
    """Split a date-sorted frame into past train, middle validation, and later test rows.

    The slices do not overlap and are not shuffled. Validation is reported so the
    three-way split is visible. This version does not search hyperparameters on it,
    so the test segment stays the untouched holdout.
    """
    ordered = frame.sort_values("Date").reset_index(drop=True)
    needed = MIN_TRAIN_ROWS + MIN_VAL_ROWS + MIN_TEST_ROWS
    n_rows = len(ordered)
    if n_rows < needed:
        raise ModelTrainingError(
            f"Only {n_rows} labelled trading days are available after indicators and the future target are calculated. "
            f"A chronological train, validation, and test split needs at least {needed}. "
            "Choose 2 years, 5 years, or the maximum available history, then train again."
        )
    if n_rows < MIN_LABELED_ROWS:
        raise ModelTrainingError(
            f"Only {n_rows} labelled rows are available. At least {MIN_LABELED_ROWS} are required before training."
        )

    n_train = int(n_rows * train_ratio)
    n_val = int(n_rows * val_ratio)
    n_test = n_rows - n_train - n_val
    if n_test < MIN_TEST_ROWS:
        n_train -= MIN_TEST_ROWS - n_test
        n_test = MIN_TEST_ROWS
    if n_val < MIN_VAL_ROWS:
        n_train -= MIN_VAL_ROWS - n_val
        n_val = MIN_VAL_ROWS
    if n_train < MIN_TRAIN_ROWS:
        raise ModelTrainingError(
            "The history is too short to keep separate train, validation, and test periods. Select a longer period."
        )

    train = ordered.iloc[:n_train].copy()
    validation = ordered.iloc[n_train : n_train + n_val].copy()
    test = ordered.iloc[n_train + n_val :].copy()
    if train["Date"].max() >= validation["Date"].min() or validation["Date"].max() >= test["Date"].min():
        raise ModelTrainingError("The time split overlapped. Training was stopped so future rows could not enter the fit.")
    return train, validation, test


def compute_classification_metrics(y_true, y_pred, y_proba) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": None,
        "test_size": int(len(y_true)),
        "upward_count": int((y_true == 1).sum()),
        "not_upward_count": int((y_true == 0).sum()),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return metrics


def confusion_and_report(y_true, y_pred) -> tuple[list, dict]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Not upward", "Upward"],
        output_dict=True,
        zero_division=0,
    )
    return matrix, report


def fit_majority_baseline(x_train, y_train):
    model = DummyClassifier(strategy="most_frequent")
    model.fit(x_train, y_train)
    return model


def fit_logistic_baseline(x_train, y_train):
    """Scale inside the pipeline so the scaler is fit on training rows only."""
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    return model


def positive_class_index(model) -> int:
    classes = list(model.classes_)
    if 1 not in classes:
        raise ModelTrainingError("The trained model does not contain the upward class, so a probability cannot be read.")
    return classes.index(1)


def predict_upward_probability(model, features) -> np.ndarray:
    index = positive_class_index(model)
    return model.predict_proba(features)[:, index]
