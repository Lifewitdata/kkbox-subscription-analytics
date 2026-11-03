"""Rolling out-of-time backtest of the 24-month decay-based LTV scenario.

Only fully matured monthly signup cohorts are eligible. Each fold estimates
retention decay and active-user ARPU from earlier cohorts, then predicts the
next matured signup month. The 24-month outcome is observed transaction revenue
per acquired subscriber. This is validation of a synthetic-data scenario, not a
production forecast or margin-adjusted LTV.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "ltv"
HORIZON_MONTHS = 24
FIT_MONTHS = 6
MIN_CHANNEL_TRAIN = 25
RANDOM_STATE = 20261103


def parse_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(
        values.astype("string").str.replace(r"\.0$", "", regex=True),
        format="%Y%m%d",
        errors="coerce",
    )


def load_source() -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    members = pd.read_csv(ROOT / "members_v3.csv")
    tx = pd.read_csv(ROOT / "transactions_v2.csv")
    members["signup_date"] = parse_date(members["registration_init_time"])
    tx["transaction_dt"] = parse_date(tx["transaction_date"])
    tx["expiry_dt"] = parse_date(tx["membership_expire_date"])
    cutoff = tx["transaction_dt"].max()
    if pd.isna(cutoff):
        raise ValueError("No valid transaction date exists in transactions_v2.csv.")
    members = members.dropna(subset=["signup_date"]).copy()
    tx = tx.dropna(subset=["transaction_dt", "expiry_dt"]).copy()
    members["channel"] = members["registered_via"].astype("string").fillna("unknown")
    members["cohort_month"] = members["signup_date"].dt.to_period("M").astype(str)
    return members, tx, cutoff


def get_transactions(
    users: pd.DataFrame, tx: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    joined = tx.merge(
        users[["msno", "signup_date", "channel", "cohort_month"]],
        on="msno",
        how="inner",
        validate="many_to_one",
    )
    joined["horizon_end"] = joined["signup_date"] + pd.DateOffset(months=horizon)
    joined = joined[
        (joined["transaction_dt"] >= joined["signup_date"])
        & (joined["transaction_dt"] < joined["horizon_end"])
    ].copy()
    joined["age_month"] = (
        (joined["transaction_dt"].dt.year - joined["signup_date"].dt.year) * 12
        + joined["transaction_dt"].dt.month
        - joined["signup_date"].dt.month
    )
    return joined


def active_rate_by_month(
    users: pd.DataFrame, tx: pd.DataFrame, months: int
) -> dict[int, float]:
    tx_by_user = {
        user_id: group[["transaction_dt", "expiry_dt"]].to_numpy()
        for user_id, group in tx.groupby("msno", sort=False)
    }
    rates: dict[int, float] = {}
    for age in range(1, months + 1):
        active = 0
        for row in users[["msno", "signup_date"]].itertuples(index=False):
            events = tx_by_user.get(row.msno)
            if events is None:
                continue
            point = row.signup_date + pd.DateOffset(months=age)
            if any(start <= point < expiry for start, expiry in events):
                active += 1
        rates[age] = active / len(users) if len(users) else 0.0
    return rates


def estimate_channel(
    training_users: pd.DataFrame, training_tx: pd.DataFrame
) -> dict:
    """Fit retention decay and steady active-user ARPU using earlier cohorts only."""
    active_rates = active_rate_by_month(training_users, training_tx, FIT_MONTHS)
    ages = np.array(list(active_rates), dtype=float)
    ret = np.array([active_rates[int(age)] for age in ages], dtype=float)
    valid = ret > 0
    if valid.sum() < 3:
        raise ValueError("Insufficient non-zero monthly retention observations to fit decay.")

    slope, intercept = np.polyfit(ages[valid], np.log(ret[valid]), deg=1)
    raw_decay = float(np.exp(slope))
    monthly_decay = float(np.clip(raw_decay, 0.50, 0.999))

    six_month_tx = get_transactions(training_users, training_tx, FIT_MONTHS)
    monthly_revenue = six_month_tx[
        six_month_tx["age_month"].between(0, FIT_MONTHS - 1)
    ].groupby("age_month")["actual_amount_paid"].sum()
    monthly_active = {
        month: active_rates[month] * len(training_users)
        for month in range(1, FIT_MONTHS + 1)
    }
    active_arpu_values = []
    for age in range(1, FIT_MONTHS + 1):
        revenue = float(monthly_revenue.get(age - 1, 0.0))
        denominator = max(monthly_active[age], 1.0)
        active_arpu_values.append(revenue / denominator)
    steady_arpu = float(np.mean(active_arpu_values))

    observed_6m = (
        six_month_tx.groupby("msno")["actual_amount_paid"].sum()
        .reindex(training_users["msno"], fill_value=0.0)
    )
    base_ltv_6m = float(observed_6m.mean())
    survival_at_6 = float(active_rates[FIT_MONTHS])
    forecast_24m = base_ltv_6m
    survival = survival_at_6
    for age in range(FIT_MONTHS, HORIZON_MONTHS):
        if age > FIT_MONTHS:
            survival *= monthly_decay
        forecast_24m += survival * steady_arpu

    return {
        "monthly_decay_raw": raw_decay,
        "monthly_decay_used": monthly_decay,
        "steady_arpu_per_active_user": steady_arpu,
        "active_rate_month_6": survival_at_6,
        "historical_mean_ltv_6m": base_ltv_6m,
        "forecast_ltv_24m": float(forecast_24m),
        "active_rates_fit_window": active_rates,
    }


def score_fold(
    members: pd.DataFrame,
    tx: pd.DataFrame,
    train_months: list[str],
    holdout_month: str,
    fold_number: int,
) -> tuple[list[dict], pd.DataFrame]:
    train = members[members["cohort_month"].isin(train_months)].copy()
    holdout = members[members["cohort_month"] == holdout_month].copy()
    if train.empty or holdout.empty:
        return [], pd.DataFrame()

    predicted_by_channel = {}
    params_by_channel = {}
    global_params = estimate_channel(train, tx)
    global_forecast = global_params["forecast_ltv_24m"]
    training_channels = sorted(train["channel"].dropna().unique())

    for channel in training_channels:
        channel_users = train[train["channel"] == channel]
        channel_tx = tx[tx["msno"].isin(channel_users["msno"])]
        try:
            params = (
                estimate_channel(channel_users, channel_tx)
                if len(channel_users) >= MIN_CHANNEL_TRAIN
                else global_params
            )
        except ValueError:
            params = global_params
        predicted_by_channel[channel] = params["forecast_ltv_24m"]
        params_by_channel[channel] = params

    # Channel codes not present in the earlier cohorts use the global baseline.
    holdout = holdout.copy()
    holdout["predicted_ltv_24m"] = holdout["channel"].map(predicted_by_channel).fillna(global_forecast)
    holdout_tx = get_transactions(holdout, tx, HORIZON_MONTHS)
    actual = holdout_tx.groupby("msno")["actual_amount_paid"].sum()
    holdout["actual_ltv_24m"] = holdout["msno"].map(actual).fillna(0.0)
    holdout["absolute_error"] = (holdout["actual_ltv_24m"] - holdout["predicted_ltv_24m"]).abs()
    holdout["fold"] = fold_number
    holdout["training_cohorts"] = ",".join(train_months)
    holdout["holdout_cohort"] = holdout_month
    holdout["absolute_percentage_error"] = (
        holdout["absolute_error"] / holdout["actual_ltv_24m"].replace(0, np.nan)
    )

    rows = []
    for channel, group in holdout.groupby("channel"):
        p = params_by_channel.get(channel, global_params)
        actual_mean = float(group["actual_ltv_24m"].mean())
        forecast_mean = float(group["predicted_ltv_24m"].mean())
        rows.append({
            "fold": fold_number,
            "training_cohorts": ",".join(train_months),
            "holdout_cohort": holdout_month,
            "channel": channel,
            "train_subscribers": int((train["channel"] == channel).sum()),
            "holdout_subscribers": int(len(group)),
            "monthly_decay_raw": float(p["monthly_decay_raw"]),
            "monthly_decay_used": float(p["monthly_decay_used"]),
            "steady_arpu_per_active_user": float(p["steady_arpu_per_active_user"]),
            "active_rate_month_6": float(p["active_rate_month_6"]),
            "historical_mean_ltv_6m": float(p["historical_mean_ltv_6m"]),
            "forecast_ltv_24m": forecast_mean,
            "actual_ltv_24m": actual_mean,
            "mae": float(group["absolute_error"].mean()),
            "wape": float(group["absolute_error"].sum() / max(group["actual_ltv_24m"].sum(), 1e-9)),
            "mean_bias_forecast_minus_actual": forecast_mean - actual_mean,
        })
    return rows, holdout


def run_validation() -> dict:
    members, tx, cutoff = load_source()
    # Full monthly cohorts only: the final calendar day must have a complete
    # 24-calendar-month outcome window by the transaction data cutoff.
    month_last_dates = (
        members.groupby("cohort_month")["signup_date"].max()
        .rename("last_signup_in_month")
        .reset_index()
    )
    month_last_dates["horizon_end"] = month_last_dates["last_signup_in_month"] + pd.DateOffset(months=24)
    matured_months = sorted(
        month_last_dates.loc[month_last_dates["horizon_end"] <= cutoff, "cohort_month"].tolist()
    )
    if len(matured_months) < 3:
        raise ValueError(
            "A rolling 24-month backtest requires at least three fully matured monthly cohorts; "
            f"only found {matured_months} by {cutoff.date()}."
        )

    # Sequential one-month-ahead folds: train on all earlier fully matured
    # cohorts and validate on the next fully matured cohort.
    folds = []
    predictions = []
    for index in range(1, len(matured_months)):
        train_months = matured_months[:index]
        holdout_month = matured_months[index]
        rows, prediction_frame = score_fold(
            members, tx, train_months, holdout_month, fold_number=index
        )
        folds.extend(rows)
        if not prediction_frame.empty:
            predictions.append(prediction_frame)

    if not predictions:
        raise ValueError("No valid rolling 24-month hold-out predictions were produced.")
    predictions_df = pd.concat(predictions, ignore_index=True)
    fold_df = pd.DataFrame(folds)
    abs_error = predictions_df["absolute_error"].to_numpy()
    actual = predictions_df["actual_ltv_24m"].to_numpy()
    predicted = predictions_df["predicted_ltv_24m"].to_numpy()
    total_actual = float(actual.sum())
    summary = {
        "dataset": "synthetic KKBox-schema demonstration data",
        "validation_design": "rolling out-of-time monthly cohort backtest",
        "target_horizon": "24 calendar months",
        "cutoff_date": cutoff.date().isoformat(),
        "matured_signup_months": matured_months,
        "fold_count": int(predictions_df["fold"].nunique()),
        "training_and_holdout_subscribers": int(len(predictions_df)),
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "wape": float(np.sum(abs_error) / max(total_actual, 1e-9)),
        "mean_forecast_bias": float(np.mean(predicted - actual)),
        "mean_actual_ltv_24m": float(np.mean(actual)),
        "mean_predicted_ltv_24m": float(np.mean(predicted)),
        "scope": (
            "24-month transaction revenue scenario validation on matured synthetic cohorts; "
            "not contribution-margin LTV, real KKBox performance or production forecasting"
        ),
    }

    # Subscriber bootstrap measures sampling variation in held-out users only;
    # it does not include synthetic-data, cohort, model-form or decay uncertainty.
    rng = np.random.default_rng(RANDOM_STATE)
    boot = np.empty(1000)
    for b in range(len(boot)):
        idx = rng.integers(0, len(predictions_df), len(predictions_df))
        boot[b] = (
            predictions_df["absolute_error"].to_numpy()[idx].sum()
            / max(predictions_df["actual_ltv_24m"].to_numpy()[idx].sum(), 1e-9)
        )
    summary["subscriber_bootstrap_wape_95_low"] = float(np.quantile(boot, 0.025))
    summary["subscriber_bootstrap_wape_95_high"] = float(np.quantile(boot, 0.975))
    summary["bootstrap_note"] = (
        "Subscriber-sampling interval only; it does not include model, decay, cohort or "
        "synthetic-data uncertainty."
    )

    # Fold/channel decay sensitivity: recompute the post-month-6 tail while
    # holding each fold's observed six-month LTV and active-user ARPU fixed.
    sensitivity_rows = []
    for row in fold_df.itertuples(index=False):
        for multiplier in (0.90, 0.95, 1.00, 1.05, 1.10):
            decay = float(np.clip(row.monthly_decay_used * multiplier, 0.50, 0.999))
            survival = float(row.active_rate_month_6)
            forecast = float(row.historical_mean_ltv_6m)
            for age in range(FIT_MONTHS, HORIZON_MONTHS):
                if age > FIT_MONTHS:
                    survival *= decay
                forecast += survival * float(row.steady_arpu_per_active_user)
            sensitivity_rows.append({
                "fold": int(row.fold),
                "holdout_cohort": row.holdout_cohort,
                "channel": row.channel,
                "decay_multiplier": multiplier,
                "scenario_monthly_decay": decay,
                "scenario_forecast_ltv_24m": forecast,
                "actual_ltv_24m": float(row.actual_ltv_24m),
                "absolute_error": abs(forecast - float(row.actual_ltv_24m)),
                "interpretation": "decay assumption sensitivity; not a statistical confidence interval",
            })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "out_of_time_24m_validation.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    fold_df.to_csv(OUTPUT / "out_of_time_24m_by_fold_channel.csv", index=False)
    predictions_df.to_csv(OUTPUT / "out_of_time_24m_subscriber_predictions.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(OUTPUT / "out_of_time_24m_decay_sensitivity.csv", index=False)
    return summary


if __name__ == "__main__":
    print(json.dumps(run_validation(), indent=2))
