"""Streamlit overview for the synthetic KKBox subscription analytics project."""
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="KKBox Subscription Analytics", layout="wide")
st.title("KKBox Subscription Analytics")
st.caption("Synthetic demonstration data — not actual KKBox customer or campaign results.")

@st.cache_data
def load_data():
    train = pd.read_csv(ROOT / "train_v2.csv")
    members = pd.read_csv(ROOT / "members_v3.csv")
    tx = pd.read_csv(ROOT / "transactions_v2.csv")
    logs = pd.read_csv(ROOT / "user_logs_v2.csv")
    return train, members, tx, logs

train, members, tx, logs = load_data()
left, middle, right = st.columns(3)
left.metric("Labelled subscribers", f"{len(train):,}")
middle.metric("Observed churn rate", f"{train.is_churn.mean():.1%}")
last_log_date = logs["date"].max()
active_ids = set(logs.loc[logs["date"] == last_log_date, "msno"])
active_share = members.assign(active_on_latest_log=members.msno.isin(active_ids))
right.metric("Active on latest log date", f"{active_share.active_on_latest_log.mean():.1%}")

st.subheader("Churn by registration channel")
seg = train.merge(members[["msno", "registered_via"]], on="msno", how="left")
by_channel = (seg.groupby("registered_via").is_churn
              .agg(subscribers="size", churn_rate="mean").reset_index())
st.bar_chart(by_channel.set_index("registered_via")["churn_rate"])

st.subheader("Activity snapshot by registration channel")
activity_channel = (active_share.groupby("registered_via").active_on_latest_log
                    .mean().rename("active_share").reset_index())
st.bar_chart(activity_channel.set_index("registered_via")["active_share"])
st.caption("Activity snapshot = subscribers with a listening record on the latest synthetic log date / all members in that registration channel. It is not a matured cohort-retention curve.")

st.subheader("Observed revenue by registration channel")
member_channel = members[["msno", "registered_via"]]
revenue = tx.groupby("msno", as_index=False).actual_amount_paid.sum()
ltv = member_channel.merge(revenue, on="msno", how="left")
by_ltv = (ltv.groupby("registered_via").actual_amount_paid
          .agg(subscribers="size", revenue="sum", revenue_per_subscriber="mean")
          .reset_index())
st.dataframe(by_ltv, use_container_width=True)
st.bar_chart(by_ltv.set_index("registered_via")["revenue_per_subscriber"])

st.subheader("Listening activity over time")
activity = logs.groupby("date").agg(
    listening_seconds=("total_secs", "sum"),
    active_subscribers=("msno", "nunique")
)
st.line_chart(activity[["active_subscribers"]])
st.caption("Channel-level revenue per subscriber is observed synthetic revenue, not a forward-looking LTV forecast.")
