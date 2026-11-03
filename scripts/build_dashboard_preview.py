"""Build a static, reproducible dashboard preview from the synthetic CSV inputs."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dashboard_screenshots" / "overview.png"


def main() -> None:
    train = pd.read_csv(ROOT / "train_v2.csv")
    members = pd.read_csv(ROOT / "members_v3.csv")
    transactions = pd.read_csv(ROOT / "transactions_v2.csv")
    logs = pd.read_csv(ROOT / "user_logs_v2.csv")
    channel = train.merge(members[["msno", "registered_via"]], on="msno", how="left")
    churn = channel.groupby("registered_via")["is_churn"].mean().sort_index()
    revenue = transactions.groupby("msno")["actual_amount_paid"].sum()
    by_user = members[["msno", "registered_via"]].merge(
        revenue.rename("revenue"), left_on="msno", right_index=True, how="left"
    )
    ltv = by_user.groupby("registered_via")["revenue"].mean().sort_index()
    activity = logs.groupby("date")["msno"].nunique().sort_index()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("KKBox Subscription Analytics | Synthetic-data preview", fontsize=16)
    axes[0, 0].bar(["Labelled users", "Members"], [len(train), len(members)])
    axes[0, 0].set_title("Subscriber coverage")
    axes[0, 0].set_ylabel("Rows")
    axes[0, 1].bar(churn.index.astype(str), churn.values)
    axes[0, 1].set_title("Observed churn rate by registration channel")
    axes[0, 1].set_xlabel("Registration channel code")
    axes[0, 1].set_ylabel("Churn rate")
    axes[0, 1].tick_params(axis="x", rotation=30)
    axes[1, 0].bar(ltv.index.astype(str), ltv.values)
    axes[1, 0].set_title("Observed revenue per member by channel")
    axes[1, 0].set_xlabel("Registration channel code")
    axes[1, 0].set_ylabel("Revenue (source units)")
    axes[1, 0].tick_params(axis="x", rotation=30)
    axes[1, 1].plot(activity.index.astype(str), activity.values, marker="o")
    axes[1, 1].set_title("Listening activity snapshot")
    axes[1, 1].set_xlabel("Synthetic log date")
    axes[1, 1].set_ylabel("Active subscribers")
    axes[1, 1].tick_params(axis="x", rotation=30)
    fig.text(0.5, 0.015,
             "Static preview generated from synthetic data; descriptive metrics are not causal or production claims.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
