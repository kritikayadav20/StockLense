"""Plotly charts for prices, indicators, model diagnostics, and comparisons."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

NAVY = "#1e3a5f"
TEAL = "#0f766e"
RED = "#b91c1c"
AMBER = "#b45309"
SLATE = "#64748b"
PURPLE = "#6d28d9"


def _style(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "#0b1f3a", "family": "Arial"},
        margin={"l": 48, "r": 24, "t": 56, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e2e8f0")
    return fig


def price_chart(frame: pd.DataFrame, title: str, show_adjusted: bool = False, candlestick: bool = False) -> go.Figure:
    fig = go.Figure()
    if candlestick:
        fig.add_trace(
            go.Candlestick(
                x=frame["Date"],
                open=frame["Open"],
                high=frame["High"],
                low=frame["Low"],
                close=frame["Close"],
                name="OHLC",
                increasing_line_color=TEAL,
                decreasing_line_color=RED,
            )
        )
    else:
        fig.add_trace(go.Scatter(x=frame["Date"], y=frame["Close"], name="Close", line={"color": NAVY, "width": 2}))
    for column, label, color in (
        ("sma_20", "SMA 20", TEAL),
        ("sma_50", "SMA 50", AMBER),
        ("sma_200", "SMA 200", PURPLE),
    ):
        if column in frame.columns and frame[column].notna().any():
            fig.add_trace(go.Scatter(x=frame["Date"], y=frame[column], name=label, line={"color": color, "width": 1.4}))
    if show_adjusted and "Adj Close" in frame.columns and frame["Adj Close"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=frame["Date"],
                y=frame["Adj Close"],
                name="Adjusted close",
                line={"color": SLATE, "width": 1.2, "dash": "dot"},
            )
        )
    fig.update_layout(xaxis_rangeslider_visible=False, yaxis_title="Rupees")
    return _style(fig, title, 460)


def rsi_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["Date"], y=frame["rsi_14"], name="RSI 14", line={"color": NAVY, "width": 2}))
    fig.add_hline(y=70, line_dash="dash", line_color=RED, annotation_text="70")
    fig.add_hline(y=30, line_dash="dash", line_color=TEAL, annotation_text="30")
    fig.update_yaxes(range=[0, 100], title="RSI")
    return _style(fig, "RSI (14)", 280)


def macd_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = [TEAL if value >= 0 else RED for value in frame["macd_histogram"].fillna(0)]
    fig.add_trace(go.Bar(x=frame["Date"], y=frame["macd_histogram"], name="Histogram", marker_color=colors))
    fig.add_trace(go.Scatter(x=frame["Date"], y=frame["macd"], name="MACD", line={"color": NAVY, "width": 1.6}))
    fig.add_trace(go.Scatter(x=frame["Date"], y=frame["macd_signal"], name="Signal", line={"color": AMBER, "width": 1.6}))
    return _style(fig, "MACD", 280)


def volume_chart(frame: pd.DataFrame) -> go.Figure:
    change = frame["Close"].diff()
    colors = [TEAL if value >= 0 else RED for value in change.fillna(0)]
    fig = go.Figure(go.Bar(x=frame["Date"], y=frame["Volume"], name="Volume", marker_color=colors))
    fig.update_yaxes(title="Shares")
    return _style(fig, "Volume", 260)


def importance_chart(rows: list[dict]) -> go.Figure:
    ordered = list(reversed(rows[:12]))
    fig = go.Figure(
        go.Bar(
            x=[row["importance"] for row in ordered],
            y=[row["label"] for row in ordered],
            orientation="h",
            marker_color=NAVY,
            name="Importance",
        )
    )
    fig.update_xaxes(title="Relative contribution")
    return _style(fig, "Feature importance", 420)


def confusion_chart(matrix: list, labels: list[str]) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=[f"Predicted {label}" for label in labels],
            y=[f"Actual {label}" for label in labels],
            text=matrix,
            texttemplate="%{text}",
            colorscale=[[0, "#f8fafc"], [1, "#0f766e"]],
            showscale=False,
        )
    )
    return _style(fig, "Holdout confusion matrix", 360)


def statement_chart(series: list[dict], title: str, currency: str | None) -> go.Figure:
    values = [point["value"] / 1e7 if currency == "INR" else point["value"] for point in series]
    fig = go.Figure(go.Bar(x=[point["period"] for point in series], y=values, marker_color=NAVY, name=title))
    fig.update_yaxes(title="₹ crore" if currency == "INR" else (currency or "Reported units"))
    return _style(fig, title, 320)


def indexed_chart(indexed: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = [NAVY, TEAL, AMBER]
    for index, column in enumerate(indexed.columns):
        fig.add_trace(
            go.Scatter(
                x=indexed.index,
                y=indexed[column],
                name=column,
                line={"color": colors[index % len(colors)], "width": 2},
            )
        )
    fig.update_yaxes(title="Indexed to 100")
    return _style(fig, "Overlapping closes, rebased to 100", 420)


def sentiment_chart(means: dict) -> go.Figure:
    labels = ["Positive", "Neutral", "Negative"]
    keys = ["positive", "neutral", "negative"]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=[means.get(key) or 0 for key in keys],
            marker_color=[TEAL, SLATE, RED],
            name="Average probability",
        )
    )
    fig.update_yaxes(title="Average FinBERT probability", range=[0, 1])
    return _style(fig, "Average headline sentiment", 320)


def equity_chart(curve: list[dict]) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=[point["date"] for point in curve],
            y=[point["equity"] for point in curve],
            name="Simulated equity",
            line={"color": NAVY, "width": 2},
        )
    )
    fig.update_yaxes(title="Starting value = 1")
    return _style(fig, "Hypothetical equity of non-overlapping trades", 320)
