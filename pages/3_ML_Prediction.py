import pandas as pd
import streamlit as st

from config import display_name
from src.charts import confusion_chart, equity_chart, importance_chart
from src.database import DatabaseError, list_predictions
from src.errors import ModelTrainingError, StockLensError
from src.formatting import format_inr, format_percent
from src.stock_prediction import refresh_saved_prediction, train_and_evaluate
from src.ui import bootstrap, load_selected_history, metric_row, page_footer, page_header, show_frame

settings = bootstrap()
page_header(
    "ML Prediction",
    "XGBoost estimates whether the close will be higher after the selected number of trading days. The model trains only when you ask it to.",
)
history = load_selected_history(settings)
if history is None:
    page_footer()
    st.stop()

symbol = history["symbol"]
st.write(
    f"Stock: **{display_name(symbol)} ({symbol})**. History: **{settings['period']}**. "
    f"Horizon: **{settings['horizon']} trading days**. "
    "The label is 1 when the future close is higher than the close on the prediction date."
)

if st.button("Train and evaluate", type="primary"):
    try:
        with st.spinner("Training on earlier bars and evaluating on the later holdout..."):
            result = train_and_evaluate(history["frame"], settings["horizon"], symbol, settings["period"])
        st.success("Training finished. The figures below are from this run.")
    except (ModelTrainingError, StockLensError) as exc:
        st.error(str(exc))
        result = None
else:
    result = refresh_saved_prediction(history["frame"], symbol, settings["period"], settings["horizon"])
    if result is None:
        st.info("No saved model matches this stock, period, and horizon. Choose Train and evaluate to create one.")

if result:
    for warning in result.get("warnings") or []:
        st.warning(warning)
    metric_row(
        [
            {"label": f"{result['horizon_trading_days']}-day direction", "value": result["direction"], "note": result["as_of_date"]},
            {"label": "Upward-movement score", "value": format_percent(result["upward_probability"], 1), "note": "Not a calibrated probability"},
            {"label": "Latest available close", "value": format_inr(result["latest_available_close"]), "note": "Not a live quote"},
            {
                "label": "Holdout accuracy",
                "value": format_percent((result.get("metrics") or {}).get("accuracy")),
                "note": f"Baseline {format_percent((result.get('baseline_majority') or {}).get('accuracy'))}",
            },
        ]
    )
    st.caption(result.get("probability_note", ""))
    st.caption(result.get("freshness_note", ""))

    metrics = result["metrics"]
    validation = result["validation_metrics"]
    logistic = result["baseline_logistic"]
    metric_table = pd.DataFrame(
        [
            {
                "Model": "XGBoost holdout",
                "Accuracy": format_percent(metrics.get("accuracy")),
                "Precision": format_percent(metrics.get("precision")),
                "Recall": format_percent(metrics.get("recall")),
                "F1": format_percent(metrics.get("f1")),
                "ROC-AUC": "N/A" if metrics.get("roc_auc") is None else f"{metrics['roc_auc']:.3f}",
            },
            {
                "Model": "Majority-class baseline",
                "Accuracy": format_percent(result["baseline_majority"].get("accuracy")),
                "Precision": format_percent(result["baseline_majority"].get("precision")),
                "Recall": format_percent(result["baseline_majority"].get("recall")),
                "F1": format_percent(result["baseline_majority"].get("f1")),
                "ROC-AUC": "N/A" if result["baseline_majority"].get("roc_auc") is None else f"{result['baseline_majority']['roc_auc']:.3f}",
            },
            {
                "Model": "Logistic baseline",
                "Accuracy": format_percent(logistic.get("accuracy")) if logistic.get("status") == "ok" else "N/A",
                "Precision": format_percent(logistic.get("precision")) if logistic.get("status") == "ok" else "N/A",
                "Recall": format_percent(logistic.get("recall")) if logistic.get("status") == "ok" else "N/A",
                "F1": format_percent(logistic.get("f1")) if logistic.get("status") == "ok" else "N/A",
                "ROC-AUC": "N/A" if logistic.get("roc_auc") is None else f"{logistic['roc_auc']:.3f}",
            },
            {
                "Model": "XGBoost validation period",
                "Accuracy": format_percent(validation.get("accuracy")),
                "Precision": format_percent(validation.get("precision")),
                "Recall": format_percent(validation.get("recall")),
                "F1": format_percent(validation.get("f1")),
                "ROC-AUC": "N/A" if validation.get("roc_auc") is None else f"{validation['roc_auc']:.3f}",
            },
        ]
    )
    st.subheader("Evaluation")
    split = result["split"]
    st.caption(
        f"Train {split['train']} rows through {split['train_end']}. "
        f"Validation {split['validation']} rows from {split['validation_start']} to {split['validation_end']}. "
        f"Test {split['test']} rows from {split['test_start']} to {split['test_end']}. "
        "Validation was not used to choose hyperparameters. The holdout row is the one to read."
    )
    show_frame(metric_table, height=220)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(confusion_chart(result["confusion_matrix"], result["confusion_labels"]), width="stretch")
    with right:
        st.plotly_chart(importance_chart(result["feature_importance"]), width="stretch")
    st.caption("Feature importance is the relative contribution inside this XGBoost model. It is not a causal weight.")

    with st.expander("Classification report"):
        show_frame(pd.DataFrame(result["classification_report"]).T.reset_index(), height=240)
    with st.expander("Hyperparameters"):
        st.json(result["hyperparameters"])

    st.subheader("Hypothetical historical simulation")
    backtest = result["backtest"]
    st.caption(backtest.get("assumptions", ""))
    if backtest.get("status") == "withheld":
        st.warning(backtest["reason"])
    else:
        st.write(backtest.get("note", ""))
        metric_row(
            [
                {"label": "Simulated trades", "value": backtest.get("trades"), "note": "Non-overlapping"},
                {"label": "Gross hit rate", "value": format_percent(backtest.get("hit_rate_gross")), "note": "Before costs"},
                {"label": "Cumulative net return", "value": format_percent(backtest.get("cumulative_net_return")), "note": "Simulation only"},
                {"label": "Test-window price change", "value": format_percent(backtest.get("test_window_price_change")), "note": "Reference, not a strategy"},
            ]
        )
        if backtest.get("equity_curve"):
            st.plotly_chart(equity_chart(backtest["equity_curve"]), width="stretch")

    st.subheader("Limitations")
    for item in result["limitations"]:
        st.write(f"- {item}")

try:
    records = list_predictions(symbol)
except DatabaseError as exc:
    st.caption(str(exc))
    records = []
if records:
    st.subheader("Saved prediction records")
    show_frame(pd.DataFrame(records), height=240)

page_footer()
