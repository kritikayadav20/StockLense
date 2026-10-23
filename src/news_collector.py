"""Collect recent company headlines from NewsAPI. No headlines are invented."""

from __future__ import annotations

import os

import requests

def news_api_key() -> str:
    return os.getenv("NEWS_API_KEY", "").strip()


def normalize_articles(items: list[dict] | None) -> list[dict]:
    """Keep real provider fields and drop duplicate or removed headlines."""
    articles = []
    seen = set()
    for item in items or []:
        title = " ".join(str(item.get("title") or "").split())
        if not title or title.lower() == "[removed]":
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        source = item.get("source") or {}
        source_name = source.get("name") if isinstance(source, dict) else None
        articles.append(
            {
                "title": title,
                "source": source_name or "Unknown source",
                "published_at": item.get("publishedAt") or item.get("published_at"),
                "url": item.get("url"),
                "description": item.get("description"),
            }
        )
    return articles


def missing_key_result() -> dict:
    return {
        "status": "missing_key",
        "articles": [],
        "message": (
            "No news API key is configured, so StockLens has not loaded headlines. "
            "Create a key at https://newsapi.org, put it in .env as NEWS_API_KEY, and restart the app. "
            "Price analysis, technical indicators, fundamentals, and the model do not need this key. "
            "You can still analyse a headline you paste yourself on this page."
        ),
    }


def fetch_company_news(company_name: str, symbol: str) -> dict:
    """Request recent English headlines. The API key is sent as a header and is never returned."""
    key = news_api_key()
    if not key:
        return missing_key_result()

    base = os.getenv("NEWS_API_BASE_URL", "https://newsapi.org/v2").rstrip("/")
    url = f"{base}/everything"
    params = {
        "q": f"\"{company_name}\" OR \"{symbol}\"",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 20,
        "searchIn": "title,description",
    }
    headers = {"X-Api-Key": key}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=20)
    except requests.RequestException:
        return {
            "status": "error",
            "articles": [],
            "message": "The news request failed. Check your internet connection and try again. The API key is not shown.",
        }

    if response.status_code in (401, 403):
        return {
            "status": "error",
            "articles": [],
            "message": "The news provider rejected the API key. Check NEWS_API_KEY in the .env file and restart StockLens. The key is not shown.",
        }
    if response.status_code == 429:
        return {
            "status": "error",
            "articles": [],
            "message": "The news provider rate limit was reached. Wait a little, then fetch headlines again.",
        }
    if response.status_code >= 400:
        return {
            "status": "error",
            "articles": [],
            "message": f"The news provider returned HTTP {response.status_code}. Try again later. The API key is not shown.",
        }
    try:
        payload = response.json()
    except ValueError:
        return {
            "status": "error",
            "articles": [],
            "message": "The news provider returned a response that could not be read. Try again later.",
        }
    if payload.get("status") not in (None, "ok"):
        detail = str(payload.get("message") or "The news provider reported an error.")
        if key and key in detail:
            detail = "The news provider reported an error."
        return {"status": "error", "articles": [], "message": detail, "articles_raw_count": 0}

    articles = normalize_articles(payload.get("articles"))
    if not articles:
        return {
            "status": "empty",
            "articles": [],
            "message": (
                f"The news provider returned no usable headlines for {company_name}. "
                "This can happen when coverage is thin or the free plan does not include this query. Nothing has been invented."
            ),
        }
    return {
        "status": "ok",
        "articles": articles,
        "message": f"{len(articles)} unique headlines were returned. Dates and sources are the provider's values.",
    }
