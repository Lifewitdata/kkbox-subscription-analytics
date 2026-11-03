"""Negative-control design check; not an offer or uplift experiment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_intervention import evaluate

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    labels = pd.read_csv(ROOT / "train_v2.csv")
    rng = np.random.default_rng(20261103)
    frame = pd.DataFrame({
        "assigned_treatment": rng.integers(0, 2, size=len(labels)),
        # Historical label used only as a placebo outcome; nobody received an offer.
        "retained_30d": 1 - labels["is_churn"].astype(int),
    })
    result = evaluate(frame)
    result["dataset"] = "synthetic historical labels with randomized placebo assignment"
    result["actual_offer_delivered"] = False
    result["causal_intervention_effect_claimed"] = False
    result["interpretation"] = (
        "Null-effect randomization/design check only. This does not evaluate a retention "
        "offer, estimate real uplift, or establish that an intervention reduces churn."
    )
    output = ROOT / "results" / "intervention" / "placebo_design_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
