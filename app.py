"""Streamlit dashboard for the synthetic KKBox subscription analytics project."""
from pathlib import Path
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="KKBox Subscription Analytics", layout="wide")
st.title("KKBox Subscription Analytics")
st.caption(
    "Synthetic demonstration data only. These figures are not actual KKBox customer "
    "or campaign results and should not be used for real customer decisions."
)


@st.cache_data
def load_data():
    train = pd.read_csv(ROOT / "train_v2.csv")
    members = pd.read_csv(ROOT / "members_v3.csv")
    transactions = pd.read_csv(ROOT / "transactions_v2.csv")
    logs = pd.read_csv(ROOT / "user_logs_v2.csv")
    return train, members, transactions, logs


train, members, transactions, logs = load_data()
left, middle, right = st.columns(3)
left.metric("Labelled subscribers", f"{len(train):,}")
middle.metric("Observed churn rate", f"{train.is_churn.mean():.1%}")
last_log_date = logs["date"].max()
active_ids = set(logs.loc[logs["date"] == last_log_date, "msno"])
right.metric("Active on latest log date", f"{members.msno.isin(active_ids).mean():.1%}")

metrics_path = ROOT / "results" / "churn" / "evaluation_metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    st.subheader("Out-of-time churn model evaluation")
    st.caption(
        f"Chronological hold-out begins {metrics['test_start_registration_date']}; "
        "all learned preprocessing is fitted inside the training pipeline."
    )
    metric_cols = st.columns(6)
    for column, label, key in zip(
        metric_cols,
        ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1", "Log-loss"],
        ["roc_auc", "accuracy", "precision", "recall", "f1", "log_loss"],
    ):
        column.metric(label, f"{metrics[key]:.3f}")
    cm_path = ROOT / "results" / "churn" / "confusion_matrix.csv"
    if cm_path.exists():
        st.dataframe(pd.read_csv(cm_path, index_col=0), use_container_width=True)
else:
    st.info("Run the leakage-safe training script to create verified model metrics.")

st.subheader("Churn by registration channel")
channel_data = train.merge(members[["msno", "registered_via"]], on="msno", how="left")
by_channel = (
    channel_data.groupby("registered_via")["is_churn"]
    .agg(subscribers="size", churn_rate="mean")
    .reset_index()
)
st.bar_chart(by_channel.set_index("registered_via")["churn_rate"])
st.dataframe(by_channel, use_container_width=True)

st.subheader("Observed revenue by registration channel")
revenue = transactions.groupby("msno", as_index=False)["actual_amount_paid"].sum()
member_value = members[["msno", "registered_via"]].merge(revenue, on="msno", how="left")
by_value = (
    member_value.groupby("registered_via")["actual_amount_paid"]
    .agg(subscribers="size", observed_revenue="sum",
         observed_revenue_per_subscriber="mean")
    .reset_index()
)
st.bar_chart(by_value.set_index("registered_via")["observed_revenue_per_subscriber"])
st.dataframe(by_value, use_container_width=True)
st.caption("Observed synthetic revenue per subscriber is not a forward-looking LTV forecast.")

st.subheader("Listening activity snapshot")
activity = logs.groupby("date").agg(
    listening_seconds=("total_secs", "sum"),
    active_subscribers=("msno", "nunique"),
)
st.line_chart(activity[["active_subscribers"]])
st.caption("This is activity recorded on each synthetic log date, not a matured retention curve.")

ltv_path = ROOT / "results" / "ltv" / "out_of_time_validation.json"
if ltv_path.exists():
    ltv = json.loads(ltv_path.read_text(encoding="utf-8"))
    st.subheader("Six-month LTV out-of-time validation")
    st.write(
        f"Earlier cohorts: {ltv['train_signup_months'][0]}–{ltv['train_signup_months'][-1]}; "
        f"hold-out cohorts: {ltv['holdout_start_month']}–{ltv['holdout_end_month']}."
    )
    st.metric("Hold-out WAPE", f"{ltv['wape']:.1%}" if ltv["wape"] is not None else "N/A")
    st.caption("Historical channel-mean baseline; synthetic-data six-month validation only.")
