"""Create reproducible, stratified 50/50 assignments for a prospective retention test.

Requires an authorised eligible-subscriber CSV with unique msno and a pre-defined
stratum column. The script only creates assignment; it does not send an offer,
contact subscribers, or fabricate outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import os
from pathlib import Path

import pandas as pd


def assign(frame: pd.DataFrame, secret: str) -> pd.DataFrame:
    required = {"msno", "stratum"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input requires columns: {sorted(required)}")
    data = frame[["msno", "stratum"]].copy()
    if data["msno"].isna().any() or data["stratum"].isna().any():
        raise ValueError("Subscriber IDs and pre-defined strata cannot be missing.")
    if data["msno"].duplicated().any():
        raise ValueError("Input must contain one row per eligible subscriber.")

    secret_bytes = secret.encode("utf-8")
    rows = []
    for stratum, group in data.groupby("stratum", sort=True):
        ranked = []
        for msno in group["msno"].astype(str):
            message = f"{stratum}|{msno}".encode("utf-8")
            digest = hmac.new(secret_bytes, message, hashlib.sha256).hexdigest()
            ranked.append((digest, msno))
        ranked.sort()
        for index, (_, msno) in enumerate(ranked):
            rows.append({
                "msno": msno,
                "stratum": stratum,
                "assigned_treatment": index % 2,
                "assignment_method": "stratified_deterministic_hmac_50_50",
            })
    result = pd.DataFrame(rows).sort_values(["stratum", "msno"]).reset_index(drop=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Eligible subscriber CSV with msno and stratum.")
    parser.add_argument("--output", required=True, help="Path for the assignment CSV.")
    parser.add_argument(
        "--secret-env",
        default="RETENTION_EXPERIMENT_SEED",
        help="Environment variable containing the private randomisation key.",
    )
    args = parser.parse_args()
    secret = os.environ.get(args.secret_env)
    if not secret:
        raise SystemExit(f"Set {args.secret_env} in the environment; do not commit the key.")
    result = assign(pd.read_csv(args.input), secret)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    balance = result.groupby(["stratum", "assigned_treatment"]).size().unstack(fill_value=0)
    print(f"Assigned {len(result):,} eligible subscribers.")
    print(balance.to_string())


if __name__ == "__main__":
    main()
