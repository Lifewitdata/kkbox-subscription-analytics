"""Evaluate an assigned retention offer using intention-to-treat.

Input must contain one row per eligible subscriber with:
  assigned_treatment: 1=assigned offer, 0=control (assignment, not redemption)
  retained_30d:       1=retained and paying after 30 days, 0=otherwise

No actual intervention data are included in this repository. This script deliberately
does not infer uplift from descriptive churn segments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]


def evaluate(frame: pd.DataFrame) -> dict:
    required = {"assigned_treatment", "retained_30d"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required experiment columns: {sorted(missing)}")
    data = frame.dropna(subset=list(required)).copy()
    for column in required:
        if not set(data[column].unique()).issubset({0, 1}):
            raise ValueError(f"{column} must contain only 0/1 values.")
    treatment = data[data["assigned_treatment"] == 1]["retained_30d"].astype(float)
    control = data[data["assigned_treatment"] == 0]["retained_30d"].astype(float)
    if treatment.empty or control.empty:
        raise ValueError("Both randomized groups must contain at least one observation.")
    p_t, p_c = float(treatment.mean()), float(control.mean())
    difference = p_t - p_c
    se = float(np.sqrt(p_t * (1 - p_t) / len(treatment)
                       + p_c * (1 - p_c) / len(control)))
    z = difference / se if se else 0.0
    return {
        "analysis": "intention_to_treat",
        "outcome": "30-day retained-and-paying indicator",
        "treatment_n": int(len(treatment)),
        "control_n": int(len(control)),
        "treatment_retention_rate": p_t,
        "control_retention_rate": p_c,
        "absolute_uplift_percentage_points": 100 * difference,
        "relative_uplift_percent": (100 * difference / p_c) if p_c else None,
        "ci_95_low_percentage_points": 100 * (difference - 1.96 * se),
        "ci_95_high_percentage_points": 100 * (difference + 1.96 * se),
        "two_sided_p_value_normal_approximation": float(2 * norm.sf(abs(z))),
        "limitations": [
            "Requires randomized assignment and complete outcome follow-up.",
            "Does not adjust for multiple tests or non-compliance; report assignment-based ITT.",
            "An observational treatment flag is not a substitute for randomized assignment.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV containing assigned_treatment and retained_30d")
    parser.add_argument("--output", default="results/intervention/experiment_itt.json")
    args = parser.parse_args()
    result = evaluate(pd.read_csv(args.input))
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
