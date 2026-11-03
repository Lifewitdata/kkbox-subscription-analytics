"""Out-of-time validation of 6-month revenue LTV on fully matured signup cohorts.

Forecasts for later signup-month cohorts use only earlier signup-month cohorts.
This is a transparent historical channel-mean baseline, not a validated 24-month
decay projection and not a production acquisition-spend recommendation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "ltv"


def parse_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.astype("string").str.replace(r"\.0$", "", regex=True),
                           format="%Y%m%d", errors="coerce")


def run_validation() -> dict:
    members = pd.read_csv(ROOT / "members_v3.csv")
    transactions = pd.read_csv(ROOT / "transactions_v2.csv")
    members["signup_date"] = parse_date(members["registration_init_time"])
    transactions["transaction_dt"] = parse_date(transactions["transaction_date"])
    cutoff = transactions["transaction_dt"].max()
    if pd.isna(cutoff):
        raise ValueError("No valid transaction dates available for LTV validation.")

    members = members.dropna(subset=["signup_date"]).copy()
    # Keep only cohorts with a complete 180-day observation window by the data cutoff.
    members = members[members["signup_date"] + pd.Timedelta(days=180) <= cutoff].copy()
    if members.empty:
        raise ValueError("No fully matured six-month signup cohorts exist in the current data.")

    members["channel"] = members["registered_via"].astype("string").fillna("unknown")
    members["cohort_month"] = members["signup_date"].dt.to_period("M").astype(str)
    tx = transactions.dropna(subset=["transaction_dt"]).merge(
        members[["msno", "signup_date", "channel", "cohort_month"]],
        on="msno", how="inner", validate="many_to_one",
    )
    tx = tx[
        (tx["transaction_dt"] >= tx["signup_date"])
        & (tx["transaction_dt"] < tx["signup_date"] + pd.Timedelta(days=180))
    ].copy()
    revenue = tx.groupby("msno")["actual_amount_paid"].sum().rename("observed_6m_revenue")
    users = members[["msno", "channel", "cohort_month"]].merge(
        revenue, left_on="msno", right_index=True, how="left"
    )
    users["observed_6m_revenue"] = users["observed_6m_revenue"].fillna(0.0)

    cohorts = sorted(users["cohort_month"].unique())
    if len(cohorts) < 4:
        raise ValueError(
            f"Need at least four fully matured signup months for chronological LTV validation; found {len(cohorts)}."
        )
    split_index = max(1, min(int(np.floor(len(cohorts) * 0.70)), len(cohorts) - 1))
    train_cohorts, test_cohorts = cohorts[:split_index], cohorts[split_index:]
    historical = users[users["cohort_month"].isin(train_cohorts)]
    holdout = users[users["cohort_month"].isin(test_cohorts)].copy()

    channel_forecast = historical.groupby("channel")["observed_6m_revenue"].mean()
    global_forecast = float(historical["observed_6m_revenue"].mean())
    holdout["forecast_6m_ltv"] = holdout["channel"].map(channel_forecast).fillna(global_forecast)
    holdout["absolute_error"] = (
        holdout["observed_6m_revenue"] - holdout["forecast_6m_ltv"]
    ).abs()
    error = holdout["forecast_6m_ltv"] - holdout["observed_6m_revenue"]
    actual_sum = float(holdout["observed_6m_revenue"].sum())
    overall = {
        "train_signup_months": train_cohorts,
        "holdout_signup_months": test_cohorts,
        "holdout_start_month": test_cohorts[0],
        "holdout_end_month": test_cohorts[-1],
        "data_cutoff": cutoff.date().isoformat(),
        "complete_window_days": 180,
        "train_users": int(len(historical)),
        "holdout_users": int(len(holdout)),
        "mae": float(holdout["absolute_error"].mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "wape": float(holdout["absolute_error"].sum() / actual_sum) if actual_sum else None,
        "mean_forecast_bias": float(error.mean()),
        "forecast_method": "earlier-cohort channel mean 6-month revenue per subscriber",
        "scope": "synthetic data; six-month observed revenue only; not 24-month decay validation",
    }
    by_cohort = (holdout.groupby(["channel", "cohort_month"])
                 .agg(subscribers=("msno", "size"),
                      actual_6m_ltv=("observed_6m_revenue", "mean"),
                      forecast_6m_ltv=("forecast_6m_ltv", "mean"),
                      mae=("absolute_error", "mean"))
                 .reset_index())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "out_of_time_validation.json").write_text(
        json.dumps(overall, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    by_cohort.to_csv(OUTPUT / "out_of_time_by_cohort.csv", index=False)
    holdout.to_csv(OUTPUT / "out_of_time_subscriber_predictions.csv", index=False)

    # Explicit sensitivity of observed revenue to contribution-margin assumptions.
    sensitivity = pd.DataFrame([
        {"contribution_margin_assumption": margin,
         "holdout_contribution_ltv_mean": float(holdout["observed_6m_revenue"].mean() * margin),
         "interpretation": "scenario only; margin is not present in the source data"}
        for margin in (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
    ])
    sensitivity.to_csv(OUTPUT / "contribution_margin_sensitivity.csv", index=False)
    return overall


if __name__ == "__main__":
    print(json.dumps(run_validation(), indent=2))
