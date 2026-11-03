from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_churn_model import build_features, chronological_split
from evaluate_intervention import evaluate


def test_required_synthetic_inputs_exist():
    for name in ("members_v3.csv", "transactions_v2.csv", "user_logs_v2.csv", "train_v2.csv"):
        assert (ROOT / name).is_file()


def test_train_labels_are_binary_and_unique():
    train = pd.read_csv(ROOT / "train_v2.csv")
    assert train["msno"].is_unique
    assert set(train["is_churn"].dropna().unique()).issubset({0, 1})


def test_user_logs_schema_and_values():
    logs = pd.read_csv(ROOT / "user_logs_v2.csv")
    required = {"msno", "date", "num_25", "num_50", "num_75", "num_985",
                "num_100", "num_unq", "total_secs"}
    assert required.issubset(logs.columns)
    assert logs["total_secs"].ge(0).all()


def test_member_ids_not_null_and_unique():
    members = pd.read_csv(ROOT / "members_v3.csv", usecols=["msno"])
    assert members["msno"].notna().all()
    assert members["msno"].is_unique


def test_churn_split_is_strictly_chronological_and_has_both_classes():
    X, y, dates = build_features()
    X_train, X_test, y_train, y_test, test_start = chronological_split(X, y, dates)
    assert pd.to_datetime(dates.loc[X_train.index]).max() < test_start
    assert pd.to_datetime(dates.loc[X_test.index]).min() >= test_start
    assert y_train.nunique() == 2
    assert y_test.nunique() == 2


def test_intervention_evaluator_reports_itt_and_uncertainty():
    frame = pd.DataFrame({
        "assigned_treatment": [0] * 100 + [1] * 100,
        "retained_30d": [0, 1] * 50 + [1] * 60 + [0] * 40,
    })
    result = evaluate(frame)
    assert result["analysis"] == "intention_to_treat"
    assert result["treatment_n"] == 100
    assert result["control_n"] == 100
    assert np.isclose(result["absolute_uplift_percentage_points"], 10.0)
    assert result["ci_95_low_percentage_points"] < result["ci_95_high_percentage_points"]


def test_sql_outputs_and_executed_results_exist():
    output = ROOT / "results" / "sql"
    expected = [
        "01_cohort_retention.csv", "02_churn_by_segment.csv",
        "03_ltv_by_channel.csv", "04_data_quality_checks.csv",
        "execution_summary.json",
    ]
    for name in expected:
        assert (output / name).is_file()
    summary = json.loads((output / "execution_summary.json").read_text())
    assert set(summary) == {
        "01_cohort_retention.sql", "02_churn_by_segment.sql",
        "03_ltv_by_channel.sql", "04_data_quality_checks.sql",
    }
    assert summary["02_churn_by_segment.sql"]["result_rows"] > 0


def test_churn_model_and_metrics_artifacts_exist_and_are_valid():
    model_path = ROOT / "models" / "kkbox_churn_pipeline.joblib"
    metrics_path = ROOT / "results" / "churn" / "evaluation_metrics.json"
    assert model_path.is_file()
    metrics = json.loads(metrics_path.read_text())
    for key in ("roc_auc", "accuracy", "precision", "recall", "f1", "log_loss"):
        assert np.isfinite(metrics[key])
    assert len(metrics["confusion_matrix"]) == 2


def test_ltv_validation_is_out_of_time_not_random():
    path = ROOT / "results" / "ltv" / "out_of_time_validation.json"
    assert path.is_file()
    result = json.loads(path.read_text())
    assert result["train_signup_months"]
    assert result["holdout_signup_months"]
    assert result["train_signup_months"][-1] < result["holdout_signup_months"][0]
    assert result["holdout_users"] > 0


def test_24_month_ltv_rolling_backtest_outputs_exist_and_are_chronological():
    path = ROOT / "results" / "ltv" / "out_of_time_24m_validation.json"
    assert path.is_file()
    result = json.loads(path.read_text())
    assert result["target_horizon"] == "24 calendar months"
    assert result["fold_count"] >= 2
    assert result["training_and_holdout_subscribers"] > 0
    assert result["matured_signup_months"] == sorted(result["matured_signup_months"])
    assert np.isfinite(result["wape"])
    fold_results = pd.read_csv(ROOT / "results" / "ltv" / "out_of_time_24m_by_fold_channel.csv")
    for row in fold_results.itertuples(index=False):
        last_train_month = row.training_cohorts.split(",")[-1]
        assert last_train_month < row.holdout_cohort


def test_grouped_permutation_importance_outputs_exist():
    path = ROOT / "results" / "churn" / "grouped_permutation_importance.csv"
    assert path.is_file()
    importance = pd.read_csv(path)
    assert not importance.empty
    assert importance["feature_count"].ge(1).all()
    assert importance["permutation_repeats"].eq(30).all()
    assert importance["interpretation"].str.contains("not a causal effect").all()


def test_retention_assignment_is_reproducible_and_balanced():
    from prepare_retention_experiment import assign

    eligible = pd.DataFrame({
        "msno": [f"subscriber_{i:03d}" for i in range(101)],
        "stratum": ["high"] * 51 + ["standard"] * 50,
    })
    first = assign(eligible, "test-secret")
    second = assign(eligible, "test-secret")
    pd.testing.assert_frame_equal(first, second)
    counts = first.groupby(["stratum", "assigned_treatment"]).size().unstack(fill_value=0)
    assert (counts.max(axis=1) - counts.min(axis=1)).max() <= 1


def test_grouped_importance_groups_correlated_columns():
    from analyze_feature_importance import correlated_groups

    frame = pd.DataFrame({
        "numeric_a": np.arange(100, dtype=float),
        "numeric_b": np.arange(100, dtype=float) * 2 + 1,
        "channel": ["A", "B"] * 50,
        "channel_copy": ["A", "B"] * 50,
        "independent": np.random.default_rng(4).normal(size=100),
    })
    groups = correlated_groups(frame, threshold=0.85)
    flattened = [column for group in groups for column in group]
    assert sorted(flattened) == sorted(frame.columns.tolist())
    assert any({"numeric_a", "numeric_b"}.issubset(set(group)) for group in groups)
    assert any({"channel", "channel_copy"}.issubset(set(group)) for group in groups)


def test_24_month_validation_records_small_sample_and_bootstrap_limits():
    result = json.loads(
        (ROOT / "results" / "ltv" / "out_of_time_24m_validation.json").read_text()
    )
    assert result["fold_count"] == 2
    assert result["training_and_holdout_subscribers"] == 1509
    assert result["target_horizon"] == "24 calendar months"
    assert result["subscriber_bootstrap_wape_95_low"] < result["subscriber_bootstrap_wape_95_high"]
    assert np.isfinite(result["baseline_wape"])
    assert np.isfinite(result["decay_model_wape_improvement_vs_baseline"])
    assert "does not include model" in result["bootstrap_note"]


def test_sample_size_planner_returns_balanced_two_arm_design():
    from plan_retention_experiment import calculate_sample_size

    result = calculate_sample_size(0.80, 0.03, alpha=0.05, power=0.80, nonresponse_rate=0.05)
    assert result["required_total_assignments"] == 2 * result["required_assignments_per_arm"]
    assert result["required_assignments_per_arm"] >= result["required_evaluable_per_arm"]
