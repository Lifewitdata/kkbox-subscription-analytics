<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2563eb,100:9333ea&height=200&section=header&text=KKBox%20Subscription%20Analytics&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Churn%20Prediction%20%7C%20LTV%20%26%20Break-Even%20CAC&descAlignY=55&descSize=18" width="100%"/>

<a href="https://github.com/YOUR_USERNAME/YOUR_REPO">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=800&color=2563EB&center=true&vCenter=true&width=650&lines=Predicting+churn+for+subscription+users+%F0%9F%8E%A7;Modeling+LTV+and+break-even+CAC+per+channel+%F0%9F%92%B0;Built+end-to-end+in+Python+%2B+scikit-learn+%2B+XGBoost" alt="Typing SVG" />
</a>

<br/>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-EA4C89?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge)

</div>

---

## 📌 About

Two connected analyses on subscription-business behavior, built on the **KKBox Churn
Prediction Challenge** (WSDM Cup 2018) schema:

| # | Project | Question it answers |
|---|---|---|
| 1️⃣ | **Churn Prediction Model** | *Which users are about to cancel?* |
| 2️⃣ | **LTV & Break-Even CAC** | *How much can we afford to pay to acquire them?* |

> ⚠️ **On the data:** the real Kaggle competition closed in 2018, and new accounts can
> no longer accept its rules to download the data. Both notebooks run on a **synthetic
> dataset that exactly matches the real schema** (`transactions`, `members`,
> `user_logs`, `train`) with realistic subscription statistics — churn rates, plan
> pricing, auto-renew behavior, and channel quality all calibrated to published figures
> for the real dataset. Drop the real CSVs in with the same filenames and every cell
> runs unchanged.

---

## 🗂️ Table of Contents

- [Project 1 — Churn Prediction Model](#-project-1--churn-prediction-model)
- [Project 2 — LTV & Break-Even CAC](#-project-2--ltv--break-even-cac)
- [Repo Structure](#-repo-structure)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)

---

## 🔮 Project 1 — Churn Prediction Model

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:16a34a,100:2563eb&height=3&section=header" width="100%"/>

Predicts whether a user cancels within 30 days of their subscription expiring, using
transaction history, listening behavior, and membership data.

**Pipeline:** merge 5 tables → engineer recency/usage features → impute & cap outliers
→ target-encode categoricals → scale + select features (χ² / ANOVA) → PCA → benchmark
**12 classifiers** via 10-fold CV → tune the winner with `GridSearchCV`.

<div align="center">

| Model | CV Log-Loss | | Model | CV Log-Loss |
|---|---|---|---|---|
| KNN | -1.176 | | Random Forest | -0.160 |
| LDA | -0.206 | | Logistic Regression | -0.164 |
| Naive Bayes | -0.291 | | Bagging | -0.164 |
| AdaBoost | -0.429 | | Extra Trees | -0.153 |
| Decision Tree | -2.915 | | GBM | -0.147 |
| QDA | *(singular matrix)* | | **XGBoost 🏆** | **-0.124** |

**Final tuned XGBoost — held-out test log-loss: `0.117`**

</div>

📓 [`kkbox_churn_modeling_v2.ipynb`](./kkbox_churn_modeling_v2.ipynb)

---

## 💰 Project 2 — LTV & Break-Even CAC

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:9333ea,100:ea580c&height=3&section=header" width="100%"/>

Builds cohort retention curves, projects 24-month LTV per acquisition channel, and
derives the maximum CAC each channel can bear at 6/12/24-month payback horizons.

**Pipeline:** build per-user revenue ledger → correct for cohort right-censoring →
retention curves by channel → cohort heatmap → LTV curves (careful to keep numerator
and denominator on the same eligible population — a common source of silently inflated
tail estimates) → exponential-decay extrapolation → break-even CAC table.

<div align="center">

| Channel | 12-mo LTV | Break-Even CAC | Target CAC (3:1) |
|---|---|---|---|
| Partner Bundle 🥇 | ~1,750 NTD | 1,750 NTD | 585 NTD |
| Organic/Web | ~1,710 NTD | 1,710 NTD | 570 NTD |
| App Store | ~1,555 NTD | 1,555 NTD | 520 NTD |
| Paid Social | ~1,395 NTD | 1,395 NTD | 465 NTD |
| Affiliate/Promo | ~1,255 NTD | 1,255 NTD | 420 NTD |

</div>

📓 [`kkbox_ltv_cac_analysis.ipynb`](./kkbox_ltv_cac_analysis.ipynb)

---

## 📁 Repo Structure

```
├── kkbox_churn_modeling_v2.ipynb      # Project 1: churn prediction
├── kkbox_ltv_cac_analysis.ipynb       # Project 2: LTV & CAC
├── generate_v2_data.py                # synthetic data generator (churn schema)
├── generate_synthetic_data.py         # synthetic data generator (LTV schema)
├── input/
│   ├── members_v3.csv
│   ├── transactions_v2.csv
│   ├── train_v2.csv
│   ├── user_logs_v2.csv
│   └── sample_submission_v2.csv
└── README.md
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter

# regenerate synthetic data (optional - CSVs are already included)
python generate_v2_data.py

jupyter notebook kkbox_churn_modeling_v2.ipynb
```

> 💡 Have real Kaggle access? Drop `transactions_v2.csv`, `members_v3.csv`,
> `user_logs_v2.csv`, `train_v2.csv`, and `sample_submission_v2.csv` into `input/` with
> the same filenames — no code changes needed.

---

## 🛠️ Tech Stack

<div align="center">

![pandas](https://img.shields.io/badge/-pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/-XGBoost-EA4C89?style=flat-square)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557c?style=flat-square)

</div>

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:9333ea,100:2563eb&height=120&section=footer" width="100%"/>
</div>
