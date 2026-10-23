"""Validation, missing fundamentals, cache fallback, and the local database."""

import json

import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import clean_ohlcv
from src.data_loader import load_history
from src.database import list_predictions, save_prediction
from src.errors import DataAccessError, DataQualityError, DataValidationError
from src.fundamentals import _growth_sentence, extract_statement_series, metrics_from_info
from tests.synthetic import make_ohlcv


def test_missing_close_is_not_replaced_with_zero():
    frame = make_ohlcv(10)
    frame.loc[3, "Close"] = np.nan
    cleaned, report = clean_ohlcv(frame)
    assert len(cleaned) == 9
    assert report["missing_close_removed"] == 1
    assert (cleaned["Close"] == 0).sum() == 0


def test_duplicate_and_unsorted_dates_are_cleaned():
    frame = make_ohlcv(6)
    scrambled = pd.concat([frame.iloc[3:], frame.iloc[:3], frame.iloc[[2]]], ignore_index=True)
    cleaned, report = clean_ohlcv(scrambled)
    assert cleaned["Date"].is_monotonic_increasing
    assert cleaned["Date"].duplicated().sum() == 0
    assert report["duplicates_removed"] == 1
    assert len(cleaned) == 6


def test_invalid_price_row_is_removed():
    frame = make_ohlcv(8)
    frame.loc[1, "High"] = 1
    frame.loc[1, "Low"] = 50
    cleaned, report = clean_ohlcv(frame)
    assert len(cleaned) == 7
    assert report["invalid_rows_removed"] == 1


def test_invalid_symbol_and_period_are_rejected():
    with pytest.raises(DataValidationError):
        load_history("AAPL", "1 year")
    with pytest.raises(DataValidationError):
        load_history("RELIANCE.NS", "10 years")


def test_cache_is_reused_and_stale_cache_survives_a_failed_download(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCKLENS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "stocklens.db"))
    calls = {"n": 0}

    def fake_download(symbol, period):
        calls["n"] += 1
        frame = make_ohlcv(40)
        return frame.set_index("Date")

    monkeypatch.setattr("src.data_loader.download_raw", fake_download)
    first = load_history("INFY.NS", "1 year", force_refresh=True)
    second = load_history("INFY.NS", "1 year", force_refresh=False)
    assert calls["n"] == 1
    assert len(second["frame"]) == len(first["frame"])
    assert second["meta"]["from_cache"] is True

    meta_path = next((tmp_path / "cache").glob("*.meta.json"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["fetched_at"] = "2020-01-01T00:00:00+00:00"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    def fail_download(symbol, period):
        raise DataAccessError("offline")

    monkeypatch.setattr("src.data_loader.download_raw", fail_download)
    stale = load_history("INFY.NS", "1 year", force_refresh=False)
    assert stale["meta"]["stale"] is True
    assert len(stale["frame"]) == len(first["frame"])


def test_download_failure_without_cache_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCKLENS_CACHE_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "stocklens.db"))

    def fail_download(symbol, period):
        raise DataAccessError("offline")

    monkeypatch.setattr("src.data_loader.download_raw", fail_download)
    with pytest.raises(DataAccessError):
        load_history("TCS.NS", "1 year", force_refresh=True)


def test_dividend_yield_uses_rate_and_price_instead_of_ambiguous_field():
    result = metrics_from_info({"dividendYield": 0.5, "dividendRate": 6.0, "currentPrice": 1200.0})
    values = {row["metric"]: row for row in result["rows"]}
    assert values["Dividend yield"]["display"] == "0.50%"
    assert values["Dividend yield"]["value"] == pytest.approx(0.005)


def test_missing_fundamental_metrics_stay_missing():
    result = metrics_from_info({})
    values = {row["metric"]: row for row in result["rows"]}
    assert values["P/E ratio"]["value"] is None
    assert values["P/E ratio"]["display"] == "N/A"
    assert values["EPS"]["value"] is None
    assert values["Market capitalization"]["value"] != 0
    assert result["available"] is False


def test_income_statement_skips_missing_lines():
    statement = pd.DataFrame(
        {"2023-03-31": [100.0, np.nan], "2024-03-31": [120.0, 10.0]},
        index=["Total Revenue", "Other item"],
    )
    extracted = extract_statement_series(statement)
    assert [point["value"] for point in extracted["revenue"]] == [100.0, 120.0]
    assert extracted["net_income"] == []
    assert "increased" in _growth_sentence(extracted["revenue"], "Revenue")


def test_prediction_record_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "records.db"))
    save_prediction(
        {
            "symbol": "TCS.NS",
            "period": "1 year",
            "horizon": 5,
            "created_at": "2024-01-01T00:00:00+00:00",
            "as_of_date": "2023-12-29",
            "direction": "Upward",
            "upward_probability": 0.61,
            "test_accuracy": 0.54,
            "baseline_accuracy": 0.51,
            "metrics_json": "{}",
        }
    )
    rows = list_predictions("TCS.NS")
    assert rows[0]["symbol"] == "TCS.NS"
    assert rows[0]["horizon"] == 5
    assert rows[0]["upward_probability"] == pytest.approx(0.61)


def test_empty_provider_frame_raises():
    with pytest.raises(DataQualityError):
        clean_ohlcv(pd.DataFrame())
