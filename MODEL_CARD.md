# Model card — KKBox churn demonstration

## Intended use

Educational portfolio demonstration using synthetic, KKBox-schema-shaped data. The model is not intended for production use, real customer targeting, real retention decisions or external performance claims.

## Data and target

- Inputs: members_v3.csv, transactions_v2.csv and user_logs_v2.csv.
- Label: train_v2.csv field is_churn.
- All data are synthetic. The data generator and source schema do not represent actual KKBox customer records.
- Gender and date-of-birth fields are excluded from model features. City and registration/payment codes are categorical fields, not ordered measurements.

## Feature construction and leakage controls

scripts/train_churn_model.py aggregates transaction and listening records to subscriber-level features. It excludes the subscriber ID, raw event dates and target from the predictor matrix. Signup date is retained solely to construct the chronological hold-out split.

All learned transformations are in a scikit-learn Pipeline:
- numeric median imputation;
- categorical most-frequent imputation;
- one-hot encoding with unknown-category handling;
- Random Forest classification and hyperparameter search inside training data.

There is no target-mean encoding, PCA, global feature selection, or imputer/scaler fit before the split. The newest registration-date block is held out; the model and feature preprocessing are fitted using earlier signup cohorts only.

**Timing limitation:** the source transaction and listening files are aggregated to the data cutoff for each user. For a production churn model, replace these aggregates with point-in-time features as of a defined prediction timestamp and a label window strictly after that timestamp. The synthetic exercise does not provide a production-grade event-time contract.

## Evaluation and artifacts

GitHub Actions executes the current committed CSVs and, after tests pass, persists:
- models/kkbox_churn_pipeline.joblib — complete preprocessing and fitted classifier;
- results/churn/evaluation_metrics.json — ROC-AUC, accuracy, precision, recall, F1, log-loss, confusion matrix, sample sizes and split boundary;
- results/churn/roc_curve.csv and roc_curve.png;
- results/churn/feature_importance.csv — native Random Forest impurity importance by transformed model input;
- results/churn/grouped_permutation_importance.csv and grouped_permutation_importance.json — grouped permutation importance on the chronological hold-out.

The metrics are for the synthetic dataset and a single chronological hold-out. They do not estimate performance on real KKBox users or on a future production population.

## Feature importance and correlated predictors

Native Random Forest impurity importance is retained as a model diagnostic, but it is not a causal explanation and can favour variables with many possible split points.

The companion analysis in scripts/analyze_feature_importance.py:
- computes feature associations on training data only;
- uses Spearman correlation for numeric/numeric pairs, bias-corrected Cramer's V for categorical/categorical pairs, and correlation ratio for numeric/categorical pairs;
- groups strongly associated columns before jointly permuting the group on the held-out sample;
- reports the fall in ROC-AUC over 30 row permutations.

This estimates predictive model reliance under a grouped permutation scheme. It is not a causal effect, an intervention response, or a claim that changing a feature will change churn. Grouped shuffling preserves associations within each feature group but breaks its link to the target; it can conceal which member of a group carries predictive signal. The reported 2.5th–97.5th percentiles are permutation-repeat variability, not population confidence intervals.

## Limitations and risks

- Synthetic data generation can create relationships that do not exist in real subscriber behaviour.
- Aggregated events are not yet a point-in-time production feature store.
- A single chronological split can be sensitive to cohort composition and does not establish long-term stability.
- A 0.50 threshold is a demonstration default, not a cost-optimised operating threshold.
- Segment churn rates are descriptive only; they are not causal estimates or treatment effects.
- Model importance does not establish that changing a feature will change churn.
- The model is not calibrated or fairness-certified for operational use.

## Required work before production use

1. Replace synthetic inputs with authorised, privacy-reviewed, point-in-time customer data.
2. Define an explicit feature snapshot date and a future churn observation window.
3. Run repeated out-of-time validation across several cohort cutoffs.
4. Calibrate probabilities and choose an operating threshold from retention economics.
5. Review feature stability, calibration and performance across relevant segments.
6. Run a randomized retention experiment and estimate intention-to-treat lift with guardrails and contribution margin.
