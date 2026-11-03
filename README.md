# KKBox Subscription Analytics

An end-to-end subscription analytics portfolio project covering churn prediction, cohort retention, six-month revenue LTV, SQL reporting and a Streamlit dashboard. **Every committed data file is synthetic and KKBox-schema-shaped; this repository contains no actual KKBox customer or campaign records.**

## What is included

1. **Churn prediction:** subscriber-level feature engineering from member, transaction and listening records; train-only imputation and one-hot encoding inside a scikit-learn Pipeline; chronological signup-date hold-out; Random Forest model selection; ROC-AUC, accuracy, precision, recall, F1, log-loss, confusion matrix, ROC curve and native model feature importance.
2. **Cohort LTV:** observed revenue and a historical channel baseline evaluated on later, fully matured signup cohorts. Validation is explicitly six-month and out-of-time; it does not claim to validate the 24-month decay extrapolation.
3. **SQL analytics:** SQLite cohort retention, churn by plan/auto-renew/city, observed revenue by channel and data-quality checks. Executed result tables are written to results/sql/.
4. **Intervention measurement:** an intention-to-treat evaluator for a properly randomized retention experiment. No treatment-assignment or 30-day outcome data are present, so the repository does not claim an actual offer uplift. A placebo design check is generated from historical labels only to demonstrate the calculation without pretending an offer was delivered.
5. **Dashboard:** Streamlit interactive dashboard plus a reproducible static preview image.

## Data contract

All input CSVs are committed in the repository root and use these canonical names:

| File | Grain | Purpose |
|---|---|---|
| members_v3.csv | Subscriber | City, registration channel and signup date |
| transactions_v2.csv | Transaction | Payment method, plan, amount, auto-renew, transaction and expiry dates |
| user_logs_v2.csv | Subscriber-day | Listening counts, unique tracks and seconds streamed |
| train_v2.csv | Labelled subscriber | Binary is_churn target (1 = churn, 0 = retained) |
| sample_submission_v2.csv | Prediction subscriber | Submission-style subscriber ID list from the original exercise |

The synthetic listening log can be regenerated deterministically:

~~~bash
python scripts/generate_user_logs.py
~~~

See data_dictionary.md for field-level definitions.

## Reproducible setup

Python 3.11 is the pinned runtime.

~~~bash
git clone https://github.com/Lifewitdata/kkbox-subscription-analytics.git
cd kkbox-subscription-analytics
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
~~~

Run the complete reproducible build locally:

~~~bash
python scripts/run_sql_analytics.py
python scripts/train_churn_model.py
python scripts/validate_ltv.py
python scripts/run_intervention_design_check.py
python scripts/build_dashboard_preview.py
pytest -q
~~~

The GitHub Actions workflow runs the same sequence on pushes and pull requests. It installs development requirements, runs Streamlit's headless AppTest against app.py, and persists generated model, SQL result tables, evaluation outputs and dashboard preview after the tests pass. The workflow also uploads a 30-day build artifact.

## Churn model: leakage-safe evaluation

The canonical model implementation is scripts/train_churn_model.py. The old notebook's target-mean encoding, imputation, scaling, feature selection and PCA-before-split path has been retired from the modelling workflow.

- The labelled population is split chronologically by registration_init_time; the newest signup-date block is held out.
- Median imputation and categorical imputation/one-hot encoding are fitted only inside the sklearn pipeline during training/CV.
- Target-derived mean encodings and PCA are not used.
- The complete preprocessing-plus-classifier pipeline is saved as models/kkbox_churn_pipeline.joblib.
- results/churn/evaluation_metrics.json stores the held-out ROC-AUC, accuracy, precision, recall, F1 and log-loss, along with split details and the confusion matrix.
- results/churn/roc_curve.csv and roc_curve.png contain the ROC data and plot.
- results/churn/feature_importance.csv contains native Random Forest impurity importance for the actual encoded model inputs. This is model attribution, not causal explanation; correlated features can share or distort importance.

Metric values should be quoted only after the current workflow has completed successfully. They describe this synthetic dataset and this particular chronological split, not real KKBox performance.

### Latest verified synthetic-data build

GitHub Actions completed the full build and test suite on the current feature branch. The committed files under models/ and results/ are generated from the current CSV inputs.

**Chronological churn hold-out** — 3,337 training subscribers; 1,644 hold-out subscribers; hold-out signup dates begin 2016-09-01; classification threshold 0.50.

| Metric | Verified hold-out result |
|---|---:|
| ROC-AUC | **0.9980** |
| Accuracy | **99.15%** |
| Precision | **93.02%** |
| Recall | **96.00%** |
| F1 | **94.49%** |
| Log-loss | **0.1249** |

Confusion matrix (rows = actual, columns = predicted):

|  | Predicted retained | Predicted churn |
|---|---:|---:|
| Actual retained | 1,510 | 9 |
| Actual churn | 5 | 120 |

These are verified outputs for the committed synthetic dataset and this one signup-date hold-out. The unusually high scores may reflect relationships built into the synthetic label-generation process. They must not be presented as real KKBox, production or future-population performance.

**Six-month LTV out-of-time hold-out** — training cohorts 2015-01 to 2016-03; later matured hold-out cohorts 2016-04 to 2016-10; 4,846 hold-out subscribers; source data cutoff 2017-03-31.

| Metric | Verified result |
|---|---:|
| MAE | 383.57 source amount units |
| RMSE | 477.37 source amount units |
| WAPE | **36.68%** |
| Mean forecast bias (forecast − actual) | −6.67 source amount units |

The forecast is a historical channel-mean baseline for six-month observed revenue. This does not validate the notebook's 24-month exponential-decay scenario.

**Executed SQL outputs:** cohort-retention query 25 rows; churn-by-segment query 27 rows; observed-revenue-by-channel query 5 rows. The six data-quality-check statements returned zero exception rows. Result CSVs and execution_summary.json are committed in results/sql/.

**Retention experiment design check:** randomized placebo assignment over historical labels returned +0.084 percentage points, 95% CI −1.466 to +1.634 pp, p = 0.916. No retention offer was delivered; this is a null-effect design check, not an estimate of intervention uplift.

## SQL execution and tests

SQL files live in sql/. Execute them with:

~~~bash
python scripts/run_sql_analytics.py
~~~

The runner loads the root CSV files into a local SQLite database and persists:
- results/sql/01_cohort_retention.csv
- results/sql/02_churn_by_segment.csv
- results/sql/03_ltv_by_channel.csv
- results/sql/04_data_quality_checks.csv
- results/sql/execution_summary.json

The automated tests check data contracts, temporal separation, both-class coverage, model metrics/artifact, SQL execution outputs, chronological LTV validation and the intervention evaluator.

## Six-month LTV: out-of-time validation

Run scripts/validate_ltv.py. It calculates six-month revenue per subscriber only for signup cohorts with a complete 180-day observation window. Earlier signup-month cohorts form the training period; later matured cohorts are held out chronologically. A channel-level historical mean is estimated from earlier cohorts and scored against later cohorts with MAE, RMSE, WAPE and forecast bias.

This is a transparent baseline validation of **six-month observed revenue LTV**. It is complemented by the 24-month rolling backtest below; the two horizons are reported separately. Contribution-margin sensitivity is shown separately because margin is not provided in the source data.

## 24-month decay scenario: rolling out-of-time validation

Run scripts/validate_ltv_24m.py. The script retains only fully matured monthly signup cohorts with a complete 24-calendar-month outcome window by the transaction data cutoff. It runs sequential one-month-ahead folds: fit on earlier matured cohorts, estimate a six-month retention-decay curve and active-user ARPU, then forecast the next matured cohort's 24-month transaction revenue per subscriber.

It writes:
- results/ltv/out_of_time_24m_validation.json — overall MAE, RMSE, WAPE, bias, fold and sample-size details;
- results/ltv/out_of_time_24m_by_fold_channel.csv — forecast versus actual by fold and acquisition channel;
- results/ltv/out_of_time_24m_subscriber_predictions.csv — subscriber-level predictions and actual 24-month revenue;
- results/ltv/out_of_time_24m_decay_sensitivity.csv — ±5%/±10% decay-rate scenario sensitivity.

A subscriber bootstrap interval is also reported, but it captures subscriber-sampling variation only. It does not capture uncertainty from synthetic data generation, cohort shifts, attribution, revenue margin, model form or decay assumptions. This is a historical backtest on the small number of fully matured synthetic cohorts; do not call it a production forecast or contribution-margin LTV.

## Retention experiment evaluation

For a real randomized retention offer, provide a CSV with one row per eligible subscriber and these columns:

| Column | Meaning |
|---|---|
| assigned_treatment | 1 = randomly assigned offer, 0 = control |
| retained_30d | 1 = retained and paying after 30 days, 0 = otherwise |

Then run:

~~~bash
python scripts/evaluate_intervention.py --input path/to/randomized_experiment.csv
~~~

The evaluator reports intention-to-treat retention rates, absolute and relative uplift, a 95% confidence interval and a two-sided p-value. Analyze assignment, not offer redemption, to preserve the randomized comparison. This repository has no real experiment dataset; results/intervention/placebo_design_check.json is explicitly a null-effect design check over historical labels, not evidence that an offer reduces churn. Do not describe segment churn differences as uplift or causal effects.

## Streamlit dashboard

Run locally:

~~~bash
streamlit run app.py
~~~

The dashboard shows synthetic-data coverage, churn by registration channel, observed revenue, listening activity, and model/LTV evaluation outputs when they exist.

### Static dashboard preview

![Static dashboard preview built from the synthetic dataset](dashboard_screenshots/overview.png)

This image is a reproducible static preview generated by scripts/build_dashboard_preview.py; it is not a screenshot of a hosted Streamlit instance.

### Deploy to Streamlit Community Cloud

Runtime setup is included in the repository: requirements.txt is intentionally limited to Streamlit, pandas and NumPy; requirements-dev.txt holds notebook, modelling and test packages. The repository pins the Streamlit runtime and contains .streamlit/config.toml. GitHub Actions also runs Streamlit's headless AppTest against app.py so app startup and rendering are exercised on every validation run.

[**Open the Streamlit Community Cloud deployment flow**](https://share.streamlit.io/deploy?repository=Lifewitdata%2Fkkbox-subscription-analytics&branch=main&mainModule=app.py)

In the deployment form choose:
- Repository: Lifewitdata/kkbox-subscription-analytics
- Branch: main
- Main file: app.py
- Python: 3.12 (Community Cloud default)

Sign in to Streamlit Community Cloud with the GitHub account that can access the repository, then click Deploy. Streamlit assigns the final streamlit.app URL after deployment. The URL cannot be truthfully published before that account-level action completes. [Streamlit's deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) explain repository, branch and entrypoint selection.

**Hosting status:** the code and deployment configuration are committed and the app has an automated headless smoke test; a live hosted URL still requires the account-level deploy action. Do not mistake the static screenshot above for a hosted deployment.

## Model card, stakeholder use and limits

- MODEL_CARD.md documents intended use, feature timing, validation and limitations.
- STAKEHOLDER_BRIEF.md explains which decisions the synthetic data cannot support.
- data_dictionary.md defines the dataset and metrics.

Do not use the synthetic model for real customer targeting, choose a real acquisition channel, set a real CAC budget or claim a retention intervention works. Production use requires authorised first-party data, a point-in-time feature contract, privacy/fairness review, calibration and monitoring, mature cohort validation, contribution margin and spend attribution, and a properly randomized retention experiment.

## Repository structure

~~~text
kkbox-subscription-analytics/
├── .github/workflows/ci.yml
├── app.py
├── data_dictionary.md
├── kkbox_churn_modeling.ipynb
├── kkbox_ltv_cac_analysis.ipynb
├── members_v3.csv
├── transactions_v2.csv
├── user_logs_v2.csv
├── train_v2.csv
├── sample_submission_v2.csv
├── scripts/
│   ├── build_dashboard_preview.py
│   ├── evaluate_intervention.py
│   ├── generate_user_logs.py
│   ├── load_sqlite.py
│   ├── run_intervention_design_check.py
│   ├── run_sql_analytics.py
│   ├── train_churn_model.py
│   └── validate_ltv.py
├── sql/
│   ├── 01_cohort_retention.sql
│   ├── 02_churn_by_segment.sql
│   ├── 03_ltv_by_channel.sql
│   └── 04_data_quality_checks.sql
├── tests/
│   └── test_data_contracts.py
├── models/                         # generated fitted pipeline
├── results/
│   ├── churn/                      # metrics, ROC, confusion matrix, importance
│   ├── intervention/               # placebo design check
│   ├── ltv/                        # chronological hold-out results
│   └── sql/                        # executed query result tables
└── dashboard_screenshots/
    └── overview.png
~~~
