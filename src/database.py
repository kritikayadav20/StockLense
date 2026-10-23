"""SQLite persistence for download logs, predictions, and short analysis notes."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from config import get_database_path
from src.errors import DatabaseError

SCHEMA = """
CREATE TABLE IF NOT EXISTS download_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    period TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    row_count INTEGER,
    first_bar TEXT,
    last_bar TEXT,
    source TEXT
);

CREATE TABLE IF NOT EXISTS prediction_record (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    period TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    as_of_date TEXT,
    direction TEXT,
    upward_probability REAL,
    test_accuracy REAL,
    baseline_accuracy REAL,
    metrics_json TEXT
);

CREATE TABLE IF NOT EXISTS analysis_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    created_at TEXT NOT NULL,
    analysis_type TEXT NOT NULL,
    summary TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    try:
        path = get_database_path()
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        connection.executescript(SCHEMA)
        return connection
    except sqlite3.Error as exc:
        raise DatabaseError(
            "StockLens could not open its local SQLite database. "
            "Check DATABASE_PATH and that the data folder is writable."
        ) from exc


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def log_download(symbol: str, period: str, fetched_at: str, row_count: int, first_bar: str, last_bar: str, source: str) -> None:
    try:
        with _connect() as connection:
            connection.execute(
                """
                INSERT INTO download_log (symbol, period, fetched_at, row_count, first_bar, last_bar, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (symbol, period, fetched_at, row_count, first_bar, last_bar, source),
            )
    except sqlite3.Error as exc:
        raise DatabaseError("The download log could not be saved.") from exc


def save_prediction(record: dict) -> None:
    try:
        with _connect() as connection:
            connection.execute(
                """
                INSERT INTO prediction_record (
                    symbol, period, horizon, created_at, as_of_date, direction,
                    upward_probability, test_accuracy, baseline_accuracy, metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.get("symbol"),
                    record.get("period"),
                    record.get("horizon"),
                    record.get("created_at"),
                    record.get("as_of_date"),
                    record.get("direction"),
                    record.get("upward_probability"),
                    record.get("test_accuracy"),
                    record.get("baseline_accuracy"),
                    record.get("metrics_json"),
                ),
            )
    except sqlite3.Error as exc:
        raise DatabaseError("The prediction record could not be saved.") from exc


def list_predictions(symbol: str | None = None, limit: int = 8) -> list[dict]:
    try:
        with _connect() as connection:
            if symbol:
                rows = connection.execute(
                    """
                    SELECT symbol, period, horizon, created_at, as_of_date, direction,
                           upward_probability, test_accuracy, baseline_accuracy
                    FROM prediction_record
                    WHERE symbol = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (symbol, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT symbol, period, horizon, created_at, as_of_date, direction,
                           upward_probability, test_accuracy, baseline_accuracy
                    FROM prediction_record
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as exc:
        raise DatabaseError("Saved predictions could not be read.") from exc


def save_analysis(symbol: str, analysis_type: str, summary: str) -> None:
    try:
        with _connect() as connection:
            connection.execute(
                """
                INSERT INTO analysis_history (symbol, created_at, analysis_type, summary)
                VALUES (?, ?, ?, ?)
                """,
                (symbol, utc_now(), analysis_type, summary[:4000]),
            )
    except sqlite3.Error as exc:
        raise DatabaseError("The analysis note could not be saved.") from exc


def list_analysis(symbol: str | None = None, limit: int = 5) -> list[dict]:
    try:
        with _connect() as connection:
            if symbol:
                rows = connection.execute(
                    """
                    SELECT symbol, created_at, analysis_type, summary
                    FROM analysis_history
                    WHERE symbol = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (symbol, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT symbol, created_at, analysis_type, summary
                    FROM analysis_history
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as exc:
        raise DatabaseError("Saved analysis notes could not be read.") from exc
