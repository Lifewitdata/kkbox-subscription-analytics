# Data dictionary

All committed CSV files are synthetic demonstration data. They do not contain actual KKBox customer records.

| File | Grain | Key fields | Description |
|---|---|---|---|
| `members_v3.csv` | One row per subscriber | `msno` | City code, age field, gender, registration channel and registration date. |
| `transactions_v2.csv` | One row per transaction | `msno` | Payment method, plan duration, list price, paid amount, auto-renew, transaction and expiry dates, cancellation flag. |
| `user_logs_v2.csv` | One row per subscriber-day | `msno`, `date` | Listening counts by completion bucket, unique tracks and total listening seconds. |
| `train_v2.csv` | One row per labelled subscriber | `msno` | Binary churn outcome (`is_churn`: 1 churn, 0 retained). |
| `sample_submission_v2.csv` | One row per prediction subscriber | `msno` | Subscriber IDs for a submission-style scoring file. |

## Metric definitions

- **Churn rate:** churned labelled subscribers divided by all labelled subscribers in the reporting group.
- **Retention rate:** active subscribers in a cohort-age bucket divided by eligible cohort members who have had time to reach that age.
- **ROC-AUC:** probability ranking metric based on the area under the receiver operating characteristic curve; it does not depend on a fixed classification threshold.
- **Precision:** true positives divided by predicted positives.
- **Recall:** true positives divided by actual positives.
- **F1:** harmonic mean of precision and recall.
- **LTV:** cumulative revenue per acquired subscriber over a stated horizon.
- **Break-even CAC:** estimated LTV at the chosen horizon; excludes overhead and contribution-margin adjustments unless explicitly modelled.
- **Target CAC (3:1):** LTV divided by three; a scenario ceiling, not observed acquisition spend.

## Caveats

`city`, `registered_via`, payment plan and payment method are coded categorical fields. `bd` contains invalid and outlier values and should not be treated as clean age without validation. The synthetic listening log generator uses a fixed seed. LTV extrapolation depends on the decay assumption and should be presented as a scenario, not a forecast with statistical certainty.
