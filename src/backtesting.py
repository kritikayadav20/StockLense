"""Hypothetical evaluation of out-of-sample directional calls.

The signal at each test date comes from the model trained on earlier dates.
The return is the realised close-to-close change over the selected horizon.
That future close is the outcome being scored. It is not a model input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import MIN_BACKTEST_TEST_ROWS, SLIPPAGE, TRANSACTION_COST


def assumption_text(horizon: int) -> str:
    return (
        "This is a hypothetical long-only simulation on the chronological holdout, not a forecast of profit. "
        f"When the model predicts a higher close {horizon} trading days ahead, the simulation takes the actual "
        "close-to-close return over that horizon and then skips the overlapping days. Otherwise it stays in cash. "
        f"Each simulated entry deducts {TRANSACTION_COST:.2%} transaction costs and {SLIPPAGE:.2%} slippage. "
        "Those two figures are round-trip teaching assumptions, not a live brokerage quote. "
        "A positive simulated result is not evidence that the model will make money in the future."
    )


def run_backtest(test: pd.DataFrame, predicted_up: np.ndarray, horizon: int) -> dict:
    """Simulate non-overlapping long trades from holdout predictions."""
    test_size = int(len(test))
    assumptions = assumption_text(horizon)
    if test_size < MIN_BACKTEST_TEST_ROWS:
        return {
            "status": "withheld",
            "reason": (
                f"The out-of-sample window has {test_size} labelled trading days. "
                f"Simulated returns are withheld below {MIN_BACKTEST_TEST_ROWS} so a short sample is not presented as performance."
            ),
            "test_size": test_size,
            "horizon_trading_days": horizon,
            "transaction_cost": TRANSACTION_COST,
            "slippage": SLIPPAGE,
            "assumptions": assumptions,
        }

    predictions = np.asarray(predicted_up).astype(int)
    equity = 1.0
    curve = [{"date": str(pd.Timestamp(test["Date"].iloc[0]).date()), "equity": 1.0}]
    trades = []
    index = 0
    while index < test_size:
        if predictions[index] != 1:
            index += 1
            continue
        close = float(test["future_close"].iloc[index])
        entry = float(test["Close"].iloc[index])
        if entry == 0 or np.isnan(close) or np.isnan(entry):
            index += horizon
            continue
        gross = close / entry - 1
        net = gross - TRANSACTION_COST - SLIPPAGE
        equity *= 1 + net
        trade_date = str(pd.Timestamp(test["Date"].iloc[index]).date())
        trades.append({"date": trade_date, "gross_return": gross, "net_return": net})
        curve.append({"date": trade_date, "equity": equity})
        index += horizon

    start = float(test["Close"].iloc[0])
    end = float(test["Close"].iloc[-1])
    window_change = None if start == 0 else end / start - 1
    result = {
        "status": "ok" if trades else "no_trades",
        "label": "Hypothetical simulation",
        "assumptions": assumptions,
        "horizon_trading_days": horizon,
        "test_size": test_size,
        "trades": len(trades),
        "hit_rate_gross": None,
        "average_gross_return": None,
        "average_net_return": None,
        "cumulative_net_return": equity - 1 if trades else 0.0,
        "test_window_price_change": window_change,
        "transaction_cost": TRANSACTION_COST,
        "slippage": SLIPPAGE,
        "equity_curve": curve,
        "note": (
            "The model did not predict an upward move on a trade date in this non-overlapping simulation, so it stayed in cash."
            if not trades
            else "Cumulative net return compounds the simulated trades only. Cash periods do not earn a return in this illustration."
        ),
    }
    if trades:
        gross_values = [trade["gross_return"] for trade in trades]
        net_values = [trade["net_return"] for trade in trades]
        result["hit_rate_gross"] = float(np.mean([value > 0 for value in gross_values]))
        result["average_gross_return"] = float(np.mean(gross_values))
        result["average_net_return"] = float(np.mean(net_values))
    return result
