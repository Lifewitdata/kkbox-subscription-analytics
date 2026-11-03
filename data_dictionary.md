# Data dictionary

All committed CSV files are synthetic demonstration data. They do not contain actual KKBox customer records.

| File | Grain | Key fields | Description |
|---|---|---|---|
| members_v3.csv | One row per subscriber | msno | City code, age field, gender, registration channel and registration date. The churn model excludes gender and date-of-birth fields. |
| transactions_v2.csv | One row per transaction | msno | Payment method, plan duration, list price, paid amount, auto-renew, transaction and expiry dates, cancellation flag. |
| user_logs_v2.csv | One row per subscriber-day | msno, date | Listening counts by completion bucket, unique tracks and total listening seconds. |
| train_v2.csv | One row per labelled subscriber | msno | Binary churn outcome (is_churn: 1 = churn, 0 = retained). |
| sample_submission_v2.csv | One row per prediction subscriber | msno | Subscriber IDs for a submission-style scoring file. |

## Metric definitions

- **Churn rate:** churned labelled subscribers divided by all labelled subscribers in the reporting group.
- **Retention rate:** active subscribers in a cohort-age bucket divided by eligible cohort members who have had time to reach that age.
- **ROC-AUC:** probability-ranking metric based on the area under the receiver operating characteristic curve; it does not depend on a fixed classification threshold.
- **Accuracy:** fraction of held-out records classified correctly at the selected threshold.
- **Precision:** true positives divided by predicted positives.
- **Recall:** true positives divided by actual positives.
- **F1:** harmonic mean of precision and recall.
- **Log-loss:** negative log likelihood of the predicted probability assigned to the observed class.
- **LTV:** cumulative revenue per acquired subscriber over a stated horizon. The out-of-time script validates six-month observed revenue only.
- **Break-even CAC:** estimated LTV at the chosen horizon; excludes overhead and contribution-margin adjustments unless explicitly modelled.
- **Target CAC (3:1):** LTV divided by three; a scenario ceiling, not observed acquisition spend.
- **Intervention uplift:** intention-to-treat difference in the pre-defined outcome rate between randomized offer assignment and control. Descriptive segment churn rates are not uplift.

## Model feature timing

The leakage-safe churn training script uses subscriber, transaction and listening aggregates from the synthetic inputs. Signup date is used to create the chronological hold-out and is not a predictor. The current dataset does not contain a true event-time snapshot contract, so the aggregated features must not be represented as production point-in-time features.

## Caveats

City, registered_via, payment plan and payment method are coded categorical fields. bd contains invalid and outlier values and is not treated as clean age. The synthetic listening log generator uses a fixed seed. LTV extrapolation depends on decay, attribution and margin assumptions and should be presented as a scenario rather than a production forecast with statistical certainty.

A real intervention evaluation requires randomized assignment, a complete follow-up outcome and documented eligibility/experiment rules. This repository contains no delivered retention offer or real experiment outcome.
