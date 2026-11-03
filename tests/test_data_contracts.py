from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

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
