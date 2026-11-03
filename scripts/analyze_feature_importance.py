"""Correlation-aware held-out permutation importance for the churn pipeline.

Raw columns are permuted on the untouched chronological hold-out. Highly
correlated numeric predictors are grouped and shuffled together so redundant
signals are not interpreted as independent evidence. This is model reliance,
not causality.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from train_churn_model import (
    MODEL_PATH, RESULTS, build_features, chronological_split,
)

ROOT = Path(__file__).resolve().parents[1]
REPEATS = 30
CORRELATION_THRESHOLD = 0.85
RANDOM_STATE = 20261103


def correlated_groups(X_train: pd.DataFrame, threshold: float = CORRELATION_THRESHOLD) -> list[list[str]]:
    numeric = X_train.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median()).dropna(axis=1, how="all")
    if numeric.shape[1] == 0:
        return [[column] for column in X_train.columns]

    corr = numeric.corr(method="spearman").abs().fillna(0.0)
    parent = {column: column for column in numeric.columns}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    columns = list(numeric.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            if corr.loc[left, right] >= threshold:
                union(left, right)

    grouped_numeric: dict[str, list[str]] = {}
    for column in numeric.columns:
        grouped_numeric.setdefault(find(column), []).append(column)

    groups: list[list[str]] = []
    consumed = set()
    for group in grouped_numeric.values():
        groups.append(group)
        consumed.update(group)
    groups.extend([[column] for column in X_train.columns if column not in consumed])
    return groups


def permutation_importance_grouped(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    groups: list[list[str]],
    repeats: int = REPEATS,
) -> pd.DataFrame:
    baseline = float(roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]))
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for group in groups:
        drops = np.empty(repeats)
        for repeat in range(repeats):
            permuted = X_test.copy()
            permutation = rng.permutation(len(permuted))
            # Apply the same row permutation to the whole group. This retains
            # correlations internal to the group but breaks their target link.
            for column in group:
                permuted[column] = X_test[column].to_numpy()[permutation]
            score = roc_auc_score(y_test, pipeline.predict_proba(permuted)[:, 1])
            drops[repeat] = baseline - score
        rows.append({
            "feature_group": " + ".join(group),
            "features": json.dumps(group),
            "feature_count": len(group),
            "baseline_roc_auc": baseline,
            "mean_roc_auc_drop": float(drops.mean()),
            "std_roc_auc_drop": float(drops.std(ddof=1)) if repeats > 1 else 0.0,
            "permutation_p2_5": float(np.quantile(drops, 0.025)),
            "permutation_p97_5": float(np.quantile(drops, 0.975)),
            "permutation_repeats": repeats,
            "correlation_threshold": CORRELATION_THRESHOLD,
            "interpretation": (
                "held-out model reliance; permutation variability range only; "
                "not a causal effect or population confidence interval"
            ),
        })
    return pd.DataFrame(rows).sort_values("mean_roc_auc_drop", ascending=False)


def run() -> pd.DataFrame:
    X, y, dates = build_features()
    X_train, X_test, y_train, y_test, boundary = chronological_split(X, y, dates)
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    groups = correlated_groups(X_train)
    result = permutation_importance_grouped(pipeline, X_test, y_test, groups)

    RESULTS.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULTS / "grouped_permutation_importance.csv", index=False)
    metadata = {
        "evaluation_design": "same chronological held-out sample as churn evaluation",
        "holdout_start_registration_date": boundary.date().isoformat(),
        "holdout_rows": int(len(X_test)),
        "baseline_roc_auc": float(roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])),
        "correlation_method": "Spearman absolute correlation computed on training data only",
        "correlation_threshold": CORRELATION_THRESHOLD,
        "groups": groups,
        "permutation_repeats": REPEATS,
        "limitations": [
            "Permutation importance measures predictive reliance, not causal effect.",
            "Jointly permuted correlated groups reduce redundant-signal masking but can hide within-group allocation.",
            "The repeat quantiles describe permutation variability, not a population confidence interval.",
            "All results apply only to this synthetic dataset and one chronological hold-out.",
        ],
    }
    (RESULTS / "grouped_permutation_importance.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(run().head(20).to_string(index=False))
