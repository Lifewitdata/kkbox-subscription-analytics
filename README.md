<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2563eb,100:9333ea&height=190&section=header&text=KKBox%20Subscription%20Analytics&fontSize=40&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Churn%20Prediction%20%7C%20Subscription%20LTV%20%26%20CAC&descAlignY=55&descSize=17" width="100%"/>

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-EA4C89?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

</div>

---

## About

This repository contains two subscription-analytics projects built around the data structure of the **KKBox Churn Prediction Challenge (WSDM Cup 2018)**.

| Project | Business question |
|---|---|
| **Churn Prediction** | Which subscribers are more likely to churn? |
| **Cohort LTV & Break-Even CAC** | How does subscriber value vary by acquisition channel, and what acquisition cost could that value support? |

**Data notice:** The committed CSVs are synthetic project data, not actual KKBox customer records or marketing-spend data. Results should be treated as demonstrations of analytical methods, not as measured KKBox business performance.

## Project 1 — Churn Prediction

The churn notebook joins member, transaction, listening-activity and training-label data, creates user-level features, applies feature selection and PCA, compares classifiers using 10-fold cross-validation with negative log-loss, and tunes an XGBoost model.

### Saved notebook evaluation snapshot

The following values are taken from the notebook's saved evaluation output. Cross-validation uses scikit-learn's \`neg_log_loss\` scoring, so values are negative and a score closer to zero is better.

| Classifier | 10-fold CV negative log-loss |
|---|---:|
| K-Nearest Neighbours | -1.1965 |
| Logistic Regression | -0.1637 |
| Linear Discriminant Analysis | -0.2061 |
| Quadratic Discriminant Analysis | -0.2453 |
| Decision Tree | -3.2362 |
| Gaussian Naive Bayes | -0.2918 |
| Bagging | -0.1642 |
| AdaBoost | -0.4236 |
| Gradient Boosting | -0.1598 |
| Random Forest | -0.1774 |
| Extra Trees | -0.1594 |
| XGBoost | -0.1416 |

The saved tuning output reports a best 10-fold CV negative log-loss of **-0.1398**. The held-out test output reports a positive log-loss of **0.1223**. The saved train/test split contains **3,486 training rows** and **1,495 test rows**; the training split churn rate is **8.4%**.

These are saved synthetic-data notebook outputs. Re-run the notebook to regenerate the results. The notebook reports log-loss; it does **not** calculate ROC-AUC.

**Notebook:** [kkbox_churn_modeling.ipynb](./kkbox_churn_modeling.ipynb)

**Input limitation:** The notebook also expects \`user_logs_v2.csv\`, but that file is not currently committed in this repository. The other churn inputs (\`train_v2.csv\`, \`sample_submission_v2.csv\`, \`transactions_v2.csv\` and \`members_v3.csv\`) are in the repository root. Add the matching synthetic or authorised source file \`user_logs_v2.csv\` to the root directory before running the churn notebook from start to finish.

## Project 2 — Cohort LTV & Break-Even CAC

The LTV notebook builds a transaction-level revenue ledger, calculates retention by acquisition channel, produces a registration-cohort retention heatmap, estimates cumulative LTV, and derives break-even and 3:1 target CAC ceilings at 6-, 12- and 24-month horizons.

The model uses an exponential-decay assumption to extrapolate retention beyond the reliable observed window. Its CAC values are **modelled ceilings derived from synthetic revenue data**, not actual acquisition costs. No marketing-spend dataset is included, so the notebook does not calculate actual channel-level LTV:CAC performance.

**Notebook:** [kkbox_ltv_cac_analysis.ipynb](./kkbox_ltv_cac_analysis.ipynb)

**Inputs:** \`members_v3.csv\`, \`transactions_v2.csv\` and \`train_v2.csv\`. These files are committed in the repository root, and the notebook has been updated to read those exact filenames.

## Repository structure

\`\`\`
kkbox-subscription-analytics/
├── README.md
├── kkbox_churn_modeling.ipynb
├── kkbox_ltv_cac_analysis.ipynb
├── members_v3.csv
├── sample_submission_v2.csv
├── train_v2.csv
└── transactions_v2.csv
\`\`\`

There are no separate data-generator scripts, no SQL files or SQL implementation, and no \`input/\` subdirectory in the current repository.

## Getting started

Clone the repository and install the Python dependencies:

\`\`\`bash
git clone https://github.com/Lifewitdata/kkbox-subscription-analytics.git
cd kkbox-subscription-analytics

pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
jupyter notebook
\`\`\`

Open \`kkbox_ltv_cac_analysis.ipynb\` to run the LTV analysis with the CSV files included in the repository. To run \`kkbox_churn_modeling.ipynb\` fully, first add the missing \`user_logs_v2.csv\` input described above.

## Tech stack

Python, pandas, NumPy, Matplotlib, seaborn, scikit-learn, XGBoost and Jupyter Notebook.

