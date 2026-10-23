"""Turn calculated StockLens results into plain-English explanations.

Templates use only the context dictionary. The optional language model receives
that same dictionary and is instructed not to invent figures. If it is not
configured, or the request fails, the template is the answer.
"""

from __future__ import annotations

import json
import os

import requests

from src.errors import LLMError
from src.formatting import format_inr, format_number, format_percent, is_missing, to_builtin
from src.stock_comparison import reliability_note


def llm_configured() -> bool:
    return bool(os.getenv("LLM_API_KEY", "").strip())


def build_risks(context: dict) -> list[str]:
    risks = []
    technical = context.get("technical") or {}
    volatility = technical.get("annualized_volatility")
    if not is_missing(volatility) and float(volatility) >= 0.35:
        risks.append(
            f"Annualised historical volatility over the selected period is {format_percent(volatility)}. "
            "That describes a wide range of past daily moves."
        )
    rsi_value = technical.get("rsi_14")
    if not is_missing(rsi_value) and float(rsi_value) > 70:
        risks.append(f"RSI (14) is {float(rsi_value):.1f}, inside the usual overbought zone.")
    if not is_missing(rsi_value) and float(rsi_value) < 30:
        risks.append(f"RSI (14) is {float(rsi_value):.1f}, inside the usual oversold zone.")

    prediction = context.get("prediction") or {}
    for warning in prediction.get("warnings") or []:
        risks.append(str(warning))

    fundamentals = context.get("fundamentals") or {}
    if fundamentals.get("error") or fundamentals.get("available") is False:
        risks.append("Some or all fundamental metrics were not returned by the data provider.")
    elif not fundamentals:
        risks.append("Fundamental metrics were not loaded for this explanation.")

    sentiment = context.get("sentiment") or {}
    if sentiment.get("status") != "ok":
        risks.append("Recent news sentiment is not included because headlines were not analysed in this session.")
    elif sentiment.get("overall_label") == "negative":
        risks.append("The average FinBERT score of the analysed headlines is negative. That describes those headlines, not a future price.")

    risks.append("Past prices, indicator readings, and model scores do not determine future returns.")
    return risks


def build_conflicts(context: dict) -> list[str]:
    conflicts = []
    stance = (context.get("technical") or {}).get("summary_stance")
    direction = (context.get("prediction") or {}).get("direction")
    sentiment = (context.get("sentiment") or {}).get("overall_label")
    if stance == "Bullish" and direction == "Downward/Not Upward":
        conflicts.append("The technical summary is bullish while the model direction is not upward.")
    if stance == "Bearish" and direction == "Upward":
        conflicts.append("The technical summary is bearish while the model direction is upward.")
    if sentiment == "negative" and direction == "Upward":
        conflicts.append("The analysed headlines lean negative while the model direction is upward.")
    if sentiment == "positive" and direction == "Downward/Not Upward":
        conflicts.append("The analysed headlines lean positive while the model direction is not upward.")
    if not conflicts:
        conflicts.append("No disagreement rule was triggered. The readings can still be uncertain, and a missing input is not an agreement.")
    return conflicts


def _metric_phrase(fundamentals: dict) -> str:
    if not fundamentals or fundamentals.get("available") is False:
        return "Available fundamental metrics were not returned for this symbol."
    shown = []
    for row in fundamentals.get("rows") or []:
        if row.get("display") and row.get("display") != "N/A":
            shown.append(f"{row['metric']} {row['display']}")
    if not shown:
        phrase = "The provider response did not include the requested ratios."
    else:
        phrase = "Available figures include " + "; ".join(shown[:6]) + "."
    extra = " ".join(fundamentals.get("interpretations") or [])
    return f"{phrase} {extra}".strip()


def _sentiment_phrase(sentiment: dict) -> str:
    if not sentiment or sentiment.get("status") != "ok":
        reason = (sentiment or {}).get("message") or "Headlines have not been analysed in this session."
        return f"Recent financial news sentiment is unavailable. {reason}"
    means = sentiment.get("mean_probabilities") or {}
    return (
        f"Recent financial news sentiment, averaged across {sentiment.get('article_count')} analysed headlines, "
        f"is {sentiment.get('overall_label')} "
        f"(positive {format_percent(means.get('positive'), 1)}, "
        f"neutral {format_percent(means.get('neutral'), 1)}, "
        f"negative {format_percent(means.get('negative'), 1)}). "
        "That distribution describes the headlines. It is not a probability that the share price will rise."
    )


def _prediction_phrase(context: dict) -> str:
    prediction = context.get("prediction")
    horizon = context.get("horizon")
    if not prediction:
        return (
            "No trained model is stored for this stock, period, and horizon, so no upward-movement probability is shown. "
            "Open ML Prediction and choose Train and evaluate."
        )
    probability = prediction.get("upward_probability")
    metrics = prediction.get("metrics") or {}
    baseline = prediction.get("baseline_majority") or {}
    split = prediction.get("split") or {}
    sentence = (
        f"Based on the available historical data, the model estimates a {horizon}-trading-day upward-movement probability of "
        f"{format_percent(probability, 1)}. The direction label is {prediction.get('direction')} "
        "when that estimate is at least 50%. This is a model score, not a guaranteed chance of profit."
    )
    if not is_missing(metrics.get("accuracy")):
        sentence += (
            f" On the chronological holdout ({metrics.get('test_size')} labelled days, "
            f"{split.get('test_start')} to {split.get('test_end')}), XGBoost accuracy was {format_percent(metrics.get('accuracy'))}"
        )
        if not is_missing(baseline.get("accuracy")):
            sentence += f" and the majority-class baseline accuracy was {format_percent(baseline.get('accuracy'))}"
        sentence += "."
    return sentence


def _technical_phrase(technical: dict) -> str:
    if not technical:
        return "Technical indicators are unavailable because price history was not loaded."
    readings = []
    for signal in technical.get("signals") or []:
        readings.append(f"{signal.get('indicator')} is {str(signal.get('reading', '')).lower()}: {signal.get('observation')}")
    detail = " ".join(readings)
    return (
        f"Technical indicators show a {technical.get('summary_stance', 'Neutral')} summary from the documented rule count. "
        f"RSI (14) is {format_number(technical.get('rsi_14'), 1)}. "
        f"MACD is {format_number(technical.get('macd'), 2)} and its signal line is {format_number(technical.get('macd_signal'), 2)}. "
        f"{detail}"
    ).strip()


def _price_phrase(context: dict) -> str:
    price = context.get("price") or {}
    name = context.get("name") or context.get("symbol")
    if is_missing(price.get("close")):
        return f"Historical prices for {name} are not available in this explanation."
    return (
        f"{name} ({context.get('symbol')}) has a latest available close of {format_inr(price.get('close'))} "
        f"on {price.get('as_of', 'the last downloaded bar')}. "
        f"That is the last historical bar, not a live quote. "
        f"The change over the selected {context.get('period')} is {format_percent(price.get('period_return'))}, "
        f"and the last daily change is {format_percent(price.get('daily_change'))}."
    )


def full_brief(context: dict) -> str:
    risks = build_risks(context)
    conflicts = build_conflicts(context)
    paragraphs = [
        _price_phrase(context),
        _prediction_phrase(context),
        _technical_phrase(context.get("technical") or {}),
        _sentiment_phrase(context.get("sentiment") or {}),
        f"Available fundamental metrics indicate the following. {_metric_phrase(context.get('fundamentals') or {})}",
        "Points of disagreement: " + " ".join(conflicts),
        "Important risks and limits include: " + " ".join(risks),
        "These signals can disagree, and the prediction is uncertain.",
    ]
    return "\n\n".join(paragraphs)


def _indicator_answer(context: dict) -> str:
    technical = context.get("technical") or {}
    help_text = technical.get("help") or {}
    lines = [
        _technical_phrase(technical),
        f"RSI: {help_text.get('RSI (14)', 'RSI compares recent gains and losses.')}",
        f"MACD: {help_text.get('MACD', 'MACD compares two moving averages of the close.')}",
        "These are descriptions of past prices. They are calculated indicators, not the XGBoost prediction.",
    ]
    return "\n\n".join(lines)


def _news_answer(context: dict) -> str:
    sentiment = context.get("sentiment") or {}
    lines = [_sentiment_phrase(sentiment)]
    headlines = sentiment.get("headlines") or []
    if headlines:
        lines.append("Analysed headlines:")
        for item in headlines[:10]:
            lines.append(
                f"- {item.get('published_at') or 'Date unavailable'} | {item.get('source') or 'Unknown source'} | "
                f"{item.get('title')} | {item.get('label')} ({format_percent(item.get('confidence'), 1)} confidence)"
            )
    else:
        lines.append("No analysed headline list is stored. Fetch news and run FinBERT, or paste a headline on the News page.")
    return "\n\n".join(lines)


def _comparison_answer(context: dict) -> str:
    rows = context.get("comparison") or []
    if not rows:
        return (
            "A stock comparison is not loaded. Open Compare Stocks, select up to three Indian listings, and run the comparison. "
            "The summary will then use those calculated rows."
        )
    lines = [
        "This comparison places the available figures side by side. It does not rank the stocks by predicted probability.",
    ]
    for row in rows:
        probability = "not trained" if is_missing(row.get("upward_probability")) else format_percent(row.get("upward_probability"), 1)
        lines.append(
            f"{row.get('company')} ({row.get('symbol')}): latest available close {format_inr(row.get('latest_close'))}, "
            f"selected-period return {format_percent(row.get('period_return'))}, "
            f"annualised volatility {format_percent(row.get('annualized_volatility'))}, "
            f"RSI {format_number(row.get('rsi_14'), 1)}, moving-average trend {row.get('moving_average_trend')}, "
            f"{row.get('horizon')}-trading-day upward score {probability}, "
            f"holdout accuracy {format_percent(row.get('model_accuracy'))} on {row.get('model_test_size') or 'N/A'} test days, "
            f"news sentiment {row.get('news_sentiment') or 'not loaded'}, "
            f"P/E {row.get('pe_display') or 'N/A'}."
        )
    lines.append(reliability_note(rows))
    return "\n\n".join(lines)


def render_template(context: dict, question: str | None = None) -> str:
    """Answer from calculated context. A question selects which section to emphasise."""
    asked = (question or "").strip().lower()
    if not asked:
        return full_brief(context)
    if "risk" in asked:
        return "Main risks from the available results:\n\n" + "\n\n".join(f"- {risk}" for risk in build_risks(context))
    if any(word in asked for word in ("rsi", "macd", "moving average", "bollinger", "volatility", "support", "resistance")):
        return _indicator_answer(context)
    if any(word in asked for word in ("news", "sentiment", "headline")):
        return _news_answer(context)
    if any(word in asked for word in ("fundamental", "revenue", "profit", "valuation", "p/e", "balance sheet", "debt")):
        return "Fundamentals from the provider response:\n\n" + _metric_phrase(context.get("fundamentals") or {})
    if "compar" in asked:
        return _comparison_answer(context)
    return "The answer below uses the calculated StockLens results for this question.\n\n" + full_brief(context)


def _llm_context(context: dict) -> dict:
    payload = to_builtin(context)
    prediction = payload.get("prediction")
    if isinstance(prediction, dict):
        prediction.pop("classification_report", None)
        prediction.pop("feature_importance", None)
        backtest = prediction.get("backtest")
        if isinstance(backtest, dict):
            backtest.pop("equity_curve", None)
    # Drop anything that could carry a secret if a caller attached one by mistake.
    return {
        key: value
        for key, value in payload.items()
        if "key" not in key.lower() and "token" not in key.lower() and "secret" not in key.lower()
    }


def call_llm(context: dict, question: str | None) -> str:
    key = os.getenv("LLM_API_KEY", "").strip()
    if not key:
        raise LLMError("No language-model key is configured.")
    base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    brief = render_template(context, question)
    messages = [
        {
            "role": "system",
            "content": (
                "You are StockLens, an educational assistant for Indian equity analysis. "
                "Use only the JSON context and the factual brief. Do not invent prices, indicators, company results, "
                "news, dates, or model metrics. If a value is missing, say it is unavailable. "
                "Do not give a buy, sell, or hold instruction. Do not claim the model is profitable. "
                "Predicted probabilities are model estimates, not guaranteed outcomes. Explain in simple English."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": question or "Explain the current stock analysis.",
                    "context": _llm_context(context),
                    "factual_brief": brief,
                },
                default=str,
            ),
        },
    ]
    try:
        response = requests.post(
            f"{base}/chat/completions",
            json={"model": model, "temperature": 0.2, "messages": messages},
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=45,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise LLMError(f"The language model request failed with HTTP {status}.") from exc
    except requests.RequestException as exc:
        raise LLMError("The language model request could not be completed.") from exc
    except ValueError as exc:
        raise LLMError("The language model returned a response that could not be read.") from exc

    choices = payload.get("choices") or []
    content = choices[0].get("message", {}).get("content") if choices else None
    if not content:
        raise LLMError("The language model returned no text.")
    return str(content).strip()


def explain(context: dict, question: str | None = None) -> dict:
    """Prefer the configured LLM, and always keep the template if that call fails."""
    template = render_template(context, question)
    if not llm_configured():
        return {
            "text": template,
            "source": "template",
            "warnings": ["No LLM_API_KEY is configured. This explanation is a template filled with calculated results."],
        }
    try:
        text = call_llm(context, question)
    except LLMError as exc:
        return {
            "text": template,
            "source": "template",
            "warnings": [
                f"{exc} A template explanation based on the calculated results is shown instead. The API key is not displayed."
            ],
        }
    return {"text": text, "source": "llm", "warnings": []}
