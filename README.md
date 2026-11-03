# KKBox Subscription Analytics

Churn prediction, cohort retention, customer segmentation and subscription LTV/CAC analysis using **synthetic KKBox-schema demonstration data**. No actual KKBox customer or campaign records are included.

## Projects

1. **Churn prediction** — feature engineering from member, transaction and listening activity; classifier comparison and XGBoost tuning; held-out classification metrics; exploratory feature attribution; segment churn diagnostics.
2. **Cohort LTV and CAC** — retention by acquisition channel and signup month, observed revenue per subscriber, and decay-based LTV/CAC scenarios.
3. **SQL analytics** — cohort retention, churn by plan/auto-renew/city, observed revenue by channel, and data-quality checks.
4. **Streamlit dashboard** — headline churn, churn by registration channel, observed revenue per subscriber, and listening activity.

## Data files and naming contract

All notebooks, scripts, SQL documentation and dashboard use these canonical input names:

| File | Grain | Purpose |
|---|---|---|
| `members_v3.csv` | Subscriber | City, demographic fields, registration channel and signup date |
| `transactions_v2.csv` | Transaction | Plan, payments, auto-renew, expiry and cancellation |
| `user_logs_v2.csv` | Subscriber-day | Listening activity and seconds streamed |
| `train_v2.csv` | Labelled subscriber | `is_churn` target (1=churn, 0=retained) |
| `sample_submission_v2.csv` | Prediction subscriber | Submission-style ID list used by the original notebook |

All CSVs are stored in the repository root. `user_logs_v2.csv` is synthetic and can be regenerated deterministically from `members_v3.csv`:

```bash
python scripts/generate_user_logs.py
```

See [data_dictionary.md](data_dictionary.md) for columns, grain and metric definitions.

## Quick start

Python 3.11 is the pinned environment.

```bash
git clone https://github.com/Lifewitdata/kkbox-subscription-analytics.git
cd kkbox-subscription-analytics
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Open `kkbox_churn_modeling.ipynb` or `kkbox_ltv_cac_analysis.ipynb`.

Run the dashboard:

```bash
streamlit run app.py
```

## Churn model evaluation

The notebook uses a 70/30 random train/test split (`test_size=0.30`, `random_state=0`) and reports ROC-AUC, accuracy, precision, recall, F1, log-loss, a confusion matrix and a ROC curve after execution. The metric code is committed, but no newly generated metric values are published as verified results until the notebook has been run end-to-end against the current synthetic CSVs.

**Important modelling caveat:** the legacy notebook performs some learned preprocessing (including target-mean encoding, imputation, scaling, feature selection and PCA) before the split. This creates information leakage. Its evaluation is exploratory and should not be presented as an unbiased production estimate. The [model card](MODEL_CARD.md) describes limitations and the recommended leakage-safe redesign.

The feature-importance view maps PCA component importances back to original features using squared loadings; it is approximate and not causal. Segment churn rates are descriptive, not intervention effects.

## SQL analytics

SQL files are in [sql/](sql/). They use SQLite-compatible CTEs and window functions. Load the CSV files into SQLite tables named after their file stems (without `.csv`) before running the queries. The SQL LTV query calculates observed revenue per subscriber; it is not the notebook's forward-looking LTV projection.

## Reproducibility and quality

- Pinned dependencies: [requirements.txt](requirements.txt)
- MIT license: [LICENSE](LICENSE)
- Ignore rules: [.gitignore](.gitignore)
- Data dictionary and metric definitions: [data_dictionary.md](data_dictionary.md)
- Model card: [MODEL_CARD.md](MODEL_CARD.md)
- Automated data-contract tests: [tests/](tests/)
- GitHub Actions CI: [.github/workflows/ci.yml](.github/workflows/ci.yml)

Run tests locally with `pytest -q`.

## LTV interpretation and stakeholder use

LTV and break-even CAC are scenario estimates based on synthetic revenue and an assumed retention-decay curve. Do not recommend real acquisition spend or claim a winning channel from this synthetic dataset. The notebook's decay assumption, sample eligibility and data cut-off materially affect projected LTV; validate on a hold-out cohort and sensitivity analysis before using the output for budget decisions.

## Repository structure

```
kkbox-subscription-analytics/
├── .github/workflows/ci.yml
├── .gitignore
├── LICENSE
├── MODEL_CARD.md
├── README.md
├── app.py
├── data_dictionary.md
├── kkbox_churn_modeling.ipynb
├── kkbox_ltv_cac_analysis.ipynb
├── members_v3.csv
├── requirements.txt
├── sample_submission_v2.csv
├── scripts/generate_user_logs.py
├── sql/
│   ├── 01_cohort_retention.sql
│   ├── 02_churn_by_segment.sql
│   ├── 03_ltv_by_channel.sql
│   └── 04_data_quality_checks.sql
├── tests/test_data_contracts.py
├── train_v2.csv
├── transactions_v2.csv
└── user_logs_v2.csv
```
