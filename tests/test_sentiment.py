"""FinBERT parsing and empty news collections. No model download is required."""

import pytest

from src.errors import SentimentModelError
from src.news_collector import fetch_company_news, normalize_articles
from src.sentiment_analysis import AGGREGATION_METHOD, aggregate_sentiments, parse_pipeline_output


def test_finbert_output_parsing_and_label_case():
    parsed = parse_pipeline_output(
        [
            [
                {"label": "Positive", "score": 0.70},
                {"label": "Negative", "score": 0.10},
                {"label": "Neutral", "score": 0.20},
            ]
        ]
    )
    assert parsed["label"] == "positive"
    assert parsed["confidence"] == pytest.approx(0.70)
    assert set(parsed["probabilities"]) == {"positive", "negative", "neutral"}


def test_unexpected_finbert_labels_are_rejected():
    with pytest.raises(SentimentModelError):
        parse_pipeline_output([{"label": "LABEL_0", "score": 0.9}])


def test_aggregate_averages_probabilities_and_counts_labels():
    results = [
        {"label": "positive", "probabilities": {"positive": 0.8, "negative": 0.1, "neutral": 0.1}},
        {"label": "negative", "probabilities": {"positive": 0.2, "negative": 0.6, "neutral": 0.2}},
    ]
    summary = aggregate_sentiments(results)
    assert summary["article_count"] == 2
    assert summary["positive_count"] == 1
    assert summary["negative_count"] == 1
    assert summary["neutral_count"] == 0
    assert summary["mean_probabilities"]["positive"] == pytest.approx(0.5)
    assert summary["overall_label"] == "positive"
    assert "not a probability that the stock price will rise" in summary["method"]
    assert summary["method"] == AGGREGATION_METHOD


def test_empty_sentiment_collection():
    summary = aggregate_sentiments([])
    assert summary["article_count"] == 0
    assert summary["mean_probabilities"] is None
    assert summary["overall_label"] is None


def test_empty_and_duplicate_headlines_are_removed():
    articles = normalize_articles(
        [
            {"title": "Infosys raises guidance", "source": {"name": "Example"}, "publishedAt": "2024-01-02"},
            {"title": "  Infosys   raises guidance ", "source": {"name": "Other"}, "publishedAt": "2024-01-03"},
            {"title": "[Removed]", "source": {"name": "Wire"}, "publishedAt": "2024-01-04"},
            {"title": "", "source": {"name": "Wire"}},
        ]
    )
    assert len(articles) == 1
    assert articles[0]["source"] == "Example"
    assert articles[0]["published_at"] == "2024-01-02"


def test_missing_news_key_returns_no_articles(monkeypatch):
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    result = fetch_company_news("Infosys", "INFY.NS")
    assert result["status"] == "missing_key"
    assert result["articles"] == []
    assert "NEWS_API_KEY" in result["message"]


def test_news_error_does_not_echo_the_key(monkeypatch):
    monkeypatch.setenv("NEWS_API_KEY", "test-key-not-real")

    class FakeResponse:
        status_code = 429

        def json(self):
            return {}

    monkeypatch.setattr("src.news_collector.requests.get", lambda *args, **kwargs: FakeResponse())
    result = fetch_company_news("Infosys", "INFY.NS")
    assert result["status"] == "error"
    assert result["articles"] == []
    assert "test-key-not-real" not in result["message"]
