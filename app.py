"""Streamlit dashboard for the synthetic KKBox subscription analytics project.

Community Cloud safe-start measures:
- Resolve files relative to this script, never the process working directory.
- Read only the CSV columns used by the dashboard.
- Validate required inputs and print useful startup diagnostics to Cloud logs.
- Render the Python traceback in the app if loading or rendering fails.
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
REQUIRED_FILES = (
    "train_v2.csv",
    "members_v3.csv",
    "transactions_v2.csv",
    "user_logs_v2.csv",
)
REQUIRED_COLUMNS = {
    "train_v2.csv": {"msno", "is_churn"},
    "members_v3.csv": {"msno", "registered_via"},
    "transactions_v2.csv": {"msno", "actual_amount_paid"},
    "user_logs_v2.csv": {"msno", "date", "total_secs"},
}

st.set_page_config(page_title="KKBox Subscription Analytics", layout="wide")


def log_startup(message: str) -> None:
    print(f"[kkbox-dashboard] {message}", flush=True)


def validate_source_files() -> None:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing required CSV file(s) in app directory {ROOT}: {', '.join(missing)}"
        )
    for filename, required_columns in REQUIRED_COLUMNS.items():
        # Read only the CSV header for schema validation.
        header = pd.read_csv(ROOT / filename, nrows=0).columns
        missing_columns = required_columns - set(header)
        if missing_columns:
            raise ValueError(
                f"{filename} is missing required column(s): {sorted(missing_columns)}"
            )


@st.cache_data(ttl=3600, max_entries=1, show_spinner="Loading dashboard data...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load only columns used by this dashboard to reduce memory consumption."""
    validate_source_files()
    train = pd.read_csv(
        ROOT / "train_v2.csv",
        usecols=["msno", "is_churn"],
        dtype={"msno": "string"},
    )
    members = pd.read_csv(
        ROOT / "members_v3.csv",
        usecols=["msno", "registered_via"],
        dtype={"msno": "string"},
    )
    transactions = pd.read_csv(
        ROOT / "transactions_v2.csv",
        usecols=["msno", "actual_amount_paid"],
        dtype={"msno": "string"},
    )
    logs = pd.read_csv(
        ROOT / "user_logs_v2.csv",
        usecols=["msno", "date", "total_secs"],
        dtype={"msno": "string"},
    )

    for name, frame in (
        ("train_v2.csv", train),
        ("members_v3.csv", members),
        ("transactions_v2.csv", transactions),
        ("user_logs_v2.csv", logs),
    ):
        if frame.empty:
            raise ValueError(f"{name} contains no data rows.")

    train["is_churn"] = pd.to_numeric(train["is_churn"], errors="coerce")
    transactions["actual_amount_paid"] = pd.to_numeric(
        transactions["actual_amount_paid"], errors="coerce"
    )
    logs["total_secs"] = pd.to_numeric(logs["total_secs"], errors="coerce").fillna(0)

    if train["is_churn"].dropna().empty:
        raise ValueError("train_v2.csv has no numeric is_churn labels.")
    if transactions["actual_amount_paid"].dropna().empty:
        raise ValueError("transactions_v2.csv has no numeric actual_amount_paid values.")
    if logs["date"].dropna().empty:
        raise ValueError("user_logs_v2.csv has no valid dates.")

    log_startup(
        "loaded rows: "
        f"train={len(train):,}, members={len(members):,}, "
        f"transactions={len(transactions):,}, logs={len(logs):,}"
    )
    return train, members, transactions, logs


def show_diagnostics() -> None:
    with st.expander("Runtime and data diagnostics"):
        st.write(
            {
                "Python": platform.python_version(),
                "Streamlit": st.__version__,
                "pandas": pd.__version__,
                "app_directory": str(ROOT),
            }
        )
        rows = []
        for name in REQUIRED_FILES:
            path = ROOT / name
            rows.append({
                "file": name,
                "exists": path.is_file(),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2)
                if path.is_file() else None,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_dashboard() -> None:
    st.title("KKBox Subscription Analytics")
    st.caption(
        "Synthetic demonstration data only. These figures are not actual KKBox customer "
        "or campaign results and should not be used for real customer decisions."
    )

    try:
        train, members, transactions, logs = load_data()
    except Exception as exc:
        log_startup(f"DATA LOAD ERROR: {type(exc).__name__}: {exc}")
        st.error("The dashboard could not load its source CSV files.")
        st.exception(exc)
        show_diagnostics()
        st.stop()

    show_diagnostics()

    left, middle, right = st.columns(3)
    left.metric("Labelled subscribers", f"{len(train):,}")
    middle.metric("Observed churn rate", f"{train['is_churn'].mean():.1%}")

    last_log_date = logs["date"].max()
    active_ids = set(logs.loc[logs["date"] == last_log_date, "msno"].dropna())
    active_rate = members["msno"].isin(active_ids).mean()
    right.metric("Active on latest log date", f"{active_rate:.1%}")

    metrics_path = ROOT / "results" / "churn" / "evaluation_metrics.json"
    if metrics_path.is_file():
        try:
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
                column.metric(label, f"{float(metrics[key]):.3f}")
            cm_path = ROOT / "results" / "churn" / "confusion_matrix.csv"
            if cm_path.is_file():
                st.dataframe(pd.read_csv(cm_path, index_col=0), use_container_width=True)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log_startup(f"CHURN METRICS WARNING: {type(exc).__name__}: {exc}")
            st.warning(f"Churn evaluation output could not be displayed: {exc}")
    else:
        st.info("Churn evaluation outputs are not present yet.")

    st.subheader("Churn by registration channel")
    channel_data = train.merge(
        members[["msno", "registered_via"]],
        on="msno",
        how="left",
        validate="many_to_one",
    )
    by_channel = (
        channel_data.groupby("registered_via", dropna=False)
        .agg(subscribers=("is_churn", "size"), churn_rate=("is_churn", "mean"))
        .reset_index()
    )
    by_channel["registered_via"] = by_channel["registered_via"].astype("string").fillna("Unknown")
    st.bar_chart(by_channel.set_index("registered_via")["churn_rate"])
    st.dataframe(by_channel, use_container_width=True)

    st.subheader("Observed revenue by registration channel")
    revenue = (
        transactions.groupby("msno", as_index=False)["actual_amount_paid"]
        .sum()
    )
    member_value = members[["msno", "registered_via"]].merge(
        revenue, on="msno", how="left", validate="one_to_one"
    )
    by_value = (
        member_value.groupby("registered_via", dropna=False)
        .agg(
            subscribers=("msno", "size"),
            observed_revenue=("actual_amount_paid", "sum"),
            observed_revenue_per_subscriber=("actual_amount_paid", "mean"),
        )
        .reset_index()
    )
    by_value["registered_via"] = by_value["registered_via"].astype("string").fillna("Unknown")
    st.bar_chart(by_value.set_index("registered_via")["observed_revenue_per_subscriber"])
    st.dataframe(by_value, use_container_width=True)
    st.caption("Observed synthetic revenue per subscriber is not a forward-looking LTV forecast.")

    st.subheader("Listening activity snapshot")
    activity = (
        logs.groupby("date")
        .agg(active_subscribers=("msno", "nunique"))
        .sort_index()
    )
    st.line_chart(activity)
    st.caption("Activity recorded on each synthetic log date; not a matured retention curve.")

    output_sections = (
        (
            "Six-month LTV out-of-time validation",
            ROOT / "results" / "ltv" / "out_of_time_validation.json",
        ),
        (
            "24-month LTV rolling out-of-time validation",
            ROOT / "results" / "ltv" / "out_of_time_24m_validation.json",
        ),
    )
    for heading, path in output_sections:
        if not path.is_file():
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            st.subheader(heading)
            if "baseline_wape" in result:
                l1, l2, l3, l4 = st.columns(4)
                l1.metric("Decay-model WAPE", f"{result['wape']:.1%}")
                l2.metric("Historical-mean baseline WAPE", f"{result['baseline_wape']:.1%}")
                l3.metric("MAE", f"{result['mae']:,.0f}")
                l4.metric("Rolling folds", str(result["fold_count"]))
                st.write(
                    "Decay model WAPE improvement vs baseline: "
                    f"{result['decay_model_wape_improvement_vs_baseline']:.1%} "
                    "(negative means the decay model is worse)."
                )
                st.caption(
                    "Synthetic data only. This 24-month backtest has few matured cohorts "
                    "and is not a production forecast or contribution-margin LTV."
                )
            else:
                st.metric(
                    "Hold-out WAPE",
                    f"{result['wape']:.1%}" if result["wape"] is not None else "N/A",
                )
                st.caption("Historical channel-mean baseline; synthetic six-month validation only.")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log_startup(f"LTV OUTPUT WARNING ({path.name}): {type(exc).__name__}: {exc}")
            st.warning(f"{path.name} could not be displayed: {exc}")

    importance_path = ROOT / "results" / "churn" / "grouped_permutation_importance.csv"
    if importance_path.is_file():
        try:
            importance = pd.read_csv(importance_path).head(15)
            st.subheader("Correlation-aware permutation importance")
            st.caption(
                "Strongly associated numeric, categorical and mixed features are grouped "
                "using training data and jointly shuffled on the chronological hold-out. "
                "This measures model reliance, not a causal effect."
            )
            st.bar_chart(importance.set_index("feature_group")["mean_roc_auc_drop"])
            st.dataframe(
                importance[[
                    "feature_group", "feature_count", "mean_roc_auc_drop",
                    "permutation_p2_5", "permutation_p97_5",
                ]],
                use_container_width=True,
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log_startup(f"FEATURE IMPORTANCE WARNING: {type(exc).__name__}: {exc}")
            st.warning(f"Feature-importance output could not be displayed: {exc}")


def main() -> None:
    try:
        render_dashboard()
    except Exception as exc:
        log_startup(f"UNHANDLED APP ERROR: {type(exc).__name__}: {exc}")
        st.error("The dashboard encountered an unexpected error. Technical details are shown below.")
        st.exception(exc)
        show_diagnostics()


main()
