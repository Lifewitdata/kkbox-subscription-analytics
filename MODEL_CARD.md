# Model card — KKBox churn demonstration

## Intended use

Educational portfolio demonstration of binary subscriber churn classification using synthetic, KKBox-schema-shaped data. Not intended for production decisions or real customer targeting.

## Data

`train_v2.csv` provides the binary label. Subscriber attributes, transaction history and listening activity are joined by `msno`. All data in this repository are synthetic.

## Model

The notebook compares baseline classifiers and tunes an XGBoost classifier. The legacy workflow uses feature engineering, imputation, target-mean encoding, scaling, feature selection and PCA.

## Evaluation

The documented split is random 70/30 with `random_state=0`. Metrics in the notebook include ROC-AUC, accuracy, precision, recall, F1, log-loss and a confusion matrix. The old preprocessing workflow fits some transformations before splitting, which can leak information. Treat results as exploratory and do not compare them as unbiased generalisation estimates.

## Limitations and risks

- Synthetic data do not represent real subscriber behaviour or prevalence.
- Random split may not reflect future-period performance.
- City and plan segment churn rates are descriptive, not causal.
- Threshold 0.50 is a default, not a cost-optimised decision threshold.
- Feature importance mapped from PCA components is approximate and not causal.
- The notebook artifact is not a complete production inference pipeline.

## Recommended next validation

Move all learned preprocessing into a scikit-learn Pipeline/ColumnTransformer fitted only on training folds, evaluate on a time-based hold-out, calibrate probabilities, select threshold using business costs, and review fairness across segments before any operational use.
