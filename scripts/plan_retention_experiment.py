"""Approximate sample-size calculator for a two-arm retention-rate experiment."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy.stats import norm


def calculate_sample_size(
    baseline_retention: float,
    minimum_detectable_uplift: float,
    alpha: float = 0.05,
    power: float = 0.80,
    nonresponse_rate: float = 0.0,
) -> dict:
    if not 0 < baseline_retention < 1:
        raise ValueError("baseline_retention must be between 0 and 1.")
    if not 0 < minimum_detectable_uplift < 1:
        raise ValueError("minimum_detectable_uplift must be between 0 and 1.")
    if baseline_retention + minimum_detectable_uplift >= 1:
        raise ValueError("Baseline plus detectable uplift must be below 1.")
    if not 0 < alpha < 0.5 or not 0.5 < power < 1:
        raise ValueError("Use 0 < alpha < 0.5 and 0.5 < power < 1.")
    if not 0 <= nonresponse_rate < 1:
        raise ValueError("nonresponse_rate must be in [0, 1).")

    p0 = baseline_retention
    p1 = p0 + minimum_detectable_uplift
    pbar = (p0 + p1) / 2
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    numerator = (
        z_alpha * math.sqrt(2 * pbar * (1 - pbar))
        + z_power * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    ) ** 2
    evaluable_per_arm = math.ceil(numerator / minimum_detectable_uplift ** 2)
    assigned_per_arm = math.ceil(evaluable_per_arm / (1 - nonresponse_rate))
    return {
        "design": "two-arm individual randomization, equal allocation",
        "baseline_retention": p0,
        "target_treatment_retention": p1,
        "minimum_detectable_absolute_uplift": minimum_detectable_uplift,
        "alpha_two_sided": alpha,
        "power": power,
        "assumed_nonresponse_or_missing_outcome_rate": nonresponse_rate,
        "required_evaluable_per_arm": evaluable_per_arm,
        "required_assignments_per_arm": assigned_per_arm,
        "required_total_assignments": 2 * assigned_per_arm,
        "method": "normal approximation for two independent proportions",
        "limitations": [
            "Approximate planning calculation; use a validated power tool before launch.",
            "Assumes individual independent assignment and a binary retention outcome.",
            "Does not account for clustering, repeated testing, non-compliance or covariate adjustment.",
            "Baseline retention and minimum detectable effect must be justified from real business data.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-retention", type=float, required=True)
    parser.add_argument("--minimum-detectable-uplift", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--nonresponse-rate", type=float, default=0.0)
    parser.add_argument("--output", default="results/intervention/sample_size_plan.json")
    args = parser.parse_args()
    result = calculate_sample_size(
        args.baseline_retention,
        args.minimum_detectable_uplift,
        args.alpha,
        args.power,
        args.nonresponse_rate,
    )
    path = Path(__file__).resolve().parents[1] / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
