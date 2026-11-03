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
