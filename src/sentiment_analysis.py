"""FinBERT sentiment. The model is loaded only when a headline is analysed."""

from __future__ import annotations

import os

import numpy as np

from src.errors import SentimentModelError

_PIPELINE = None
_LABELS = ("positive", "negative", "neutral")
AGGREGATION_METHOD = (
    "Each headline is classified by ProsusAI/finbert. "
    "The overall distribution is the average of the article-level positive, negative, and neutral probabilities. "
    "The overall label is the class with the highest average probability. "
    "The counts use each article's highest-probability class. "
    "This describes the analysed headlines. It is not a probability that the stock price will rise."
)


def parse_pipeline_output(raw) -> dict:
    """Turn a transformers pipeline response into positive, negative, and neutral probabilities."""
    if not raw:
        raise SentimentModelError("FinBERT returned no scores. No sentiment was substituted.")
    rows = raw[0] if isinstance(raw[0], list) else raw
    probabilities = {}
    for row in rows:
        if not isinstance(row, dict) or "label" not in row or "score" not in row:
            raise SentimentModelError("FinBERT returned a score in an unexpected format. No sentiment was substituted.")
        probabilities[str(row["label"]).strip().lower()] = float(row["score"])
    missing = [label for label in _LABELS if label not in probabilities]
    if missing:
        raise SentimentModelError(
            "FinBERT did not return positive, negative, and neutral scores "
            f"(missing: {', '.join(missing)}). No sentiment was substituted."
        )
    label = max(_LABELS, key=lambda name: probabilities[name])
    return {
        "label": label,
        "confidence": float(probabilities[label]),
        "probabilities": {name: float(probabilities[name]) for name in _LABELS},
    }


def aggregate_sentiments(results: list[dict]) -> dict:
    """Average FinBERT probabilities. An empty collection stays empty."""
    if not results:
        return {
            "article_count": 0,
            "positive_count": 0,
            "neutral_count": 0,
            "negative_count": 0,
            "mean_probabilities": None,
            "overall_label": None,
            "method": "No articles were available, so there is no sentiment distribution.",
        }
    means = {
        label: float(np.mean([item["probabilities"][label] for item in results]))
        for label in _LABELS
    }
    counts = {label: int(sum(item["label"] == label for item in results)) for label in _LABELS}
    overall = max(_LABELS, key=lambda label: means[label])
    return {
        "article_count": len(results),
        "positive_count": counts["positive"],
        "neutral_count": counts["neutral"],
        "negative_count": counts["negative"],
        "mean_probabilities": means,
        "overall_label": overall,
        "method": AGGREGATION_METHOD,
    }


def load_finbert():
    """Load ProsusAI/finbert once. Failure is reported; scores are never randomised."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    except ImportError as exc:
        raise SentimentModelError(
            "FinBERT's libraries are not installed. Run pip install -r requirements.txt, including transformers and torch, then try again."
        ) from exc

    model_name = os.getenv("FINBERT_MODEL", "ProsusAI/finbert").strip() or "ProsusAI/finbert"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _PIPELINE = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=None,
            device=-1,
        )
    except Exception as exc:  # noqa: BLE001 - download, memory, and runtime errors should stay visible.
        raise SentimentModelError(
            f"FinBERT could not be loaded ({model_name}). The first run needs an internet connection and enough memory "
            f"for PyTorch. No sentiment scores were invented. Details: {exc.__class__.__name__}: {exc}"
        ) from exc
    return _PIPELINE


def analyse_texts(texts: list[str]) -> list[dict]:
    """Score each non-empty string. The caller decides whether the text is news or user-supplied."""
    cleaned = [" ".join(text.split()) for text in texts]
    if not cleaned or any(not text for text in cleaned):
        raise SentimentModelError("There is no text to analyse.")
    pipe = load_finbert()
    try:
        raw_results = pipe([text[:2000] for text in cleaned], batch_size=8, truncation=True)
    except Exception as exc:  # noqa: BLE001
        raise SentimentModelError(
            f"FinBERT failed while scoring text ({exc.__class__.__name__}). No sentiment scores were invented."
        ) from exc
    parsed = []
    for text, raw in zip(cleaned, raw_results):
        item = parse_pipeline_output(raw)
        item["text"] = text
        parsed.append(item)
    return parsed


def analyse_text(text: str) -> dict:
    return analyse_texts([text])[0]
