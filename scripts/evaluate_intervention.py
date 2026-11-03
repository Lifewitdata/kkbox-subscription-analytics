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
    result = {
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

    # Optional experiment guardrails. These are still evaluated by original
    # assignment (intention-to-treat), not by offer redemption or delivery.
    if "unsubscribed_30d" in data.columns:
        unsub_t = data.loc[data["assigned_treatment"] == 1, "unsubscribed_30d"].astype(float)
        unsub_c = data.loc[data["assigned_treatment"] == 0, "unsubscribed_30d"].astype(float)
        if set(pd.concat([unsub_t, unsub_c]).dropna().unique()).issubset({0.0, 1.0}):
            pt, pc = float(unsub_t.mean()), float(unsub_c.mean())
            se_u = float(np.sqrt(pt * (1 - pt) / len(unsub_t) + pc * (1 - pc) / len(unsub_c)))
            diff = pt - pc
            result["unsubscribe_guardrail"] = {
                "treatment_rate": pt,
                "control_rate": pc,
                "difference_percentage_points": 100 * diff,
                "ci_95_low_percentage_points": 100 * (diff - 1.96 * se_u),
                "ci_95_high_percentage_points": 100 * (diff + 1.96 * se_u),
                "interpretation": "intention-to-treat safety guardrail",
            }

    if "contribution_margin_30d" in data.columns:
        mt = pd.to_numeric(
            data.loc[data["assigned_treatment"] == 1, "contribution_margin_30d"], errors="coerce"
        ).dropna()
        mc = pd.to_numeric(
            data.loc[data["assigned_treatment"] == 0, "contribution_margin_30d"], errors="coerce"
        ).dropna()
        if len(mt) > 1 and len(mc) > 1:
            difference = float(mt.mean() - mc.mean())
            se_m = float(np.sqrt(mt.var(ddof=1) / len(mt) + mc.var(ddof=1) / len(mc)))
            result["contribution_margin_guardrail"] = {
                "treatment_mean": float(mt.mean()),
                "control_mean": float(mc.mean()),
                "difference_treatment_minus_control": difference,
                "ci_95_low": difference - 1.96 * se_m,
                "ci_95_high": difference + 1.96 * se_m,
                "interpretation": "intention-to-treat incremental contribution margin",
            }

    if "offer_sent" in data.columns:
        result["offer_sent_rate_by_assignment"] = (
            data.groupby("assigned_treatment")["offer_sent"].mean().to_dict()
        )
    if "offer_redeemed" in data.columns:
        result["offer_redemption_rate_by_assignment"] = (
            data.groupby("assigned_treatment")["offer_redeemed"].mean().to_dict()
        )
    return result


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
