"""Train and evaluate a leakage-safe churn model on the committed synthetic data."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "kkbox_churn_pipeline.joblib"
RESULTS = ROOT / "results" / "churn"
RANDOM_STATE = 42


def parse_yyyymmdd(values: pd.Series) -> pd.Series:
    """Parse integer/string YYYYMMDD columns without treating missing values as 1970."""
    cleaned = values.astype("string").str.replace(r"\.0$", "", regex=True)
    return pd.to_datetime(cleaned, format="%Y%m%d", errors="coerce")


def build_features() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    train = pd.read_csv(ROOT / "train_v2.csv")
    members = pd.read_csv(ROOT / "members_v3.csv")
    transactions = pd.read_csv(ROOT / "transactions_v2.csv")
    logs = pd.read_csv(ROOT / "user_logs_v2.csv")

    for frame, col in (
        (members, "registration_init_time"),
        (transactions, "transaction_date"),
        (transactions, "membership_expire_date"),
        (logs, "date"),
    ):
        frame[col + "_parsed"] = parse_yyyymmdd(frame[col])

    # Feature construction is deterministic aggregation. No target information is
    # used here; all learned imputation/encoding is contained in the sklearn Pipeline.
    member_features = members[[
        "msno", "city", "registered_via", "registration_init_time_parsed"
    ]].copy()
    member_features["registration_init_time_parsed"] = pd.to_datetime(
        member_features["registration_init_time_parsed"], errors="coerce"
    )
    for categorical_column in ("city", "registered_via"):
        member_features[categorical_column] = member_features[categorical_column].astype("string")

    transactions = transactions.dropna(subset=["transaction_date_parsed"]).copy()
    transactions = transactions.sort_values(["msno", "transaction_date_parsed"])
    tx_agg = transactions.groupby("msno").agg(
        transaction_count=("transaction_date_parsed", "size"),
        transaction_total_paid=("actual_amount_paid", "sum"),
        transaction_mean_paid=("actual_amount_paid", "mean"),
        transaction_max_paid=("actual_amount_paid", "max"),
    )
    tx_last = transactions.drop_duplicates("msno", keep="last").set_index("msno")
    tx_agg["last_payment_method_id"] = tx_last["payment_method_id"]
    tx_agg["last_payment_plan_days"] = tx_last["payment_plan_days"]
    tx_agg["last_plan_list_price"] = tx_last["plan_list_price"]
    tx_agg["last_is_auto_renew"] = tx_last["is_auto_renew"]
    tx_agg["last_is_cancel"] = tx_last["is_cancel"]
    tx_agg["last_transaction_date"] = tx_last["transaction_date_parsed"]
    for categorical_column in (
        "last_payment_method_id", "last_payment_plan_days",
        "last_is_auto_renew", "last_is_cancel",
    ):
        tx_agg[categorical_column] = tx_agg[categorical_column].astype("string")

    logs = logs.dropna(subset=["date_parsed"]).copy()
    log_agg = logs.groupby("msno").agg(
        log_days=("date_parsed", "nunique"),
        total_secs=("total_secs", "sum"),
        mean_secs_per_log=("total_secs", "mean"),
        max_secs_per_log=("total_secs", "max"),
        total_num_25=("num_25", "sum"),
        total_num_50=("num_50", "sum"),
        total_num_75=("num_75", "sum"),
        total_num_985=("num_985", "sum"),
        total_num_100=("num_100", "sum"),
        total_unique_tracks=("num_unq", "sum"),
        last_log_date=("date_parsed", "max"),
    )

    cutoff_candidates = [
        transactions["transaction_date_parsed"].max(),
        logs["date_parsed"].max(),
    ]
    feature_cutoff = max(d for d in cutoff_candidates if pd.notna(d))

    dataset = (
        train[["msno", "is_churn"]]
        .merge(member_features, on="msno", how="left", validate="one_to_one")
        .merge(tx_agg, left_on="msno", right_index=True, how="left")
        .merge(log_agg, left_on="msno", right_index=True, how="left")
    )
    dataset["registration_recency_days"] = (
        feature_cutoff - dataset["registration_init_time_parsed"]
    ).dt.days
    dataset["days_since_last_transaction"] = (
        feature_cutoff - dataset["last_transaction_date"]
    ).dt.days
    dataset["days_since_last_log"] = (
        feature_cutoff - dataset["last_log_date"]
    ).dt.days

    # Dates are used only for the chronological split, not as model inputs.
    split_dates = dataset["registration_init_time_parsed"]
    usable = split_dates.notna() & dataset["is_churn"].isin([0, 1])
    dataset = dataset.loc[usable].copy()
    split_dates = split_dates.loc[usable]
    y = dataset["is_churn"].astype(int)
    excluded = {
        "msno", "is_churn", "registration_init_time_parsed",
        "last_transaction_date", "last_log_date",
    }
    X = dataset.drop(columns=[c for c in excluded if c in dataset.columns])
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y, split_dates


def chronological_split(
    X: pd.DataFrame, y: pd.Series, dates: pd.Series, test_fraction: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Timestamp]:
    unique_dates = np.sort(pd.to_datetime(dates).dropna().unique())
    if len(unique_dates) < 2:
        raise ValueError("At least two distinct signup dates are required for an out-of-time split.")
    boundary_index = min(max(int(np.floor(len(unique_dates) * (1 - test_fraction))), 1),
                         len(unique_dates) - 1)
    boundary = pd.Timestamp(unique_dates[boundary_index])
    test_mask = pd.to_datetime(dates) >= boundary
    X_train, X_test = X.loc[~test_mask], X.loc[test_mask]
    y_train, y_test = y.loc[~test_mask], y.loc[test_mask]
    if X_train.empty or X_test.empty or y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "Chronological split must contain both churn classes in train and test. "
            "Inspect label coverage by registration date; do not fall back to a random split."
        )
    return X_train, X_test, y_train, y_test, boundary


def main() -> None:
    X, y, dates = build_features()
    X_train, X_test, y_train, y_test, test_start = chronological_split(X, y, dates)

    categorical = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric = [column for column in X_train.columns if column not in categorical]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
            n_jobs=1,
        )),
    ])

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions={
            "classifier__max_depth": [None, 12, 20],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", 0.8],
        },
        n_iter=5,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_

    probability = best_pipeline.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.50).astype(int)
    cm = confusion_matrix(y_test, prediction, labels=[0, 1])
    fpr, tpr, thresholds = roc_curve(y_test, probability)
    metrics = {
        "dataset": "synthetic KKBox-schema demonstration data",
        "evaluation_design": "chronological hold-out by registration date",
        "test_start_registration_date": test_start.date().isoformat(),
        "threshold": 0.50,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
        "cv_best_roc_auc": float(search.best_score_),
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "log_loss": float(log_loss(y_test, probability, labels=[0, 1])),
        "confusion_matrix_labels": ["retained_0", "churn_1"],
        "confusion_matrix": cm.tolist(),
        "best_params": search.best_params_,
        "feature_count_before_encoding": int(X_train.shape[1]),
        "feature_count_after_encoding": int(
            len(best_pipeline.named_steps["preprocessor"].get_feature_names_out())
        ),
        "random_state": RANDOM_STATE,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    (RESULTS / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(cm, index=["actual_retained", "actual_churn"],
                 columns=["predicted_retained", "predicted_churn"]).to_csv(
        RESULTS / "confusion_matrix.csv", index=True
    )
    pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr,
                  "threshold": thresholds}).to_csv(RESULTS / "roc_curve.csv", index=False)

    feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = best_pipeline.named_steps["classifier"].feature_importances_
    importance_frame = pd.DataFrame({
        "model_input_feature": feature_names,
        "importance": importances,
        "interpretation": "native_random_forest_impurity_importance_not_causal",
    }).sort_values("importance", ascending=False)
    importance_frame.to_csv(RESULTS / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Out-of-time churn ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "roc_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["Retained", "Churn"])
    ax.set_yticks([0, 1], labels=["Retained", "Churn"])
    ax.set(xlabel="Predicted", ylabel="Actual", title="Out-of-time confusion matrix")
    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row, str(value), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    joblib.dump({
        "pipeline": best_pipeline,
        "threshold": 0.50,
        "target": "is_churn",
        "model_type": "RandomForestClassifier",
        "evaluation": metrics,
        "input_features": list(X_train.columns),
        "created_for": "synthetic demonstration data only",
    }, MODEL_PATH, compress=3)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved evaluation outputs: {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
