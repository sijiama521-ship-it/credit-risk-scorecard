# Credit Risk Scorecard + IFRS 9 ECL Engine

An end-to-end credit risk modelling project built in Python, replicating a bank-grade workflow: data cleaning, PD modelling, IFRS 9 stage classification, and expected credit loss calculation.

---

## Project Overview

**Business problem:** Banks must estimate the probability that each borrower will default, classify them under IFRS 9, and hold sufficient loss provisions. This project replicates that workflow using the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset (150,000 retail borrowers).

| Component | Description |
|-----------|-------------|
| **EDA** | Exploratory analysis — default rate, income patterns, missing values |
| **PD Model** | Logistic Regression to estimate probability of default |
| **IFRS 9 Engine** | Stage classification (Stage 1/2/3) based on PD |
| **ECL Calculator** | Expected Credit Loss = PD × LGD × EAD × Horizon |

---

## Key Results

| Metric | Value |
|--------|-------|
| Test set AUC-ROC | **0.83** |
| Overall default rate | 6.7% (class imbalance addressed with `class_weight="balanced"`) |
| Stage 1 avg ECL | $3,806 (12-month provision) |
| Stage 2 avg ECL | $26,348 (lifetime provision) |
| Stage 3 avg ECL | $62,534 (lifetime provision) |
| Stage 3 vs Stage 1 provisioning | **16× higher** |

**Key insight:** Stage 3 borrowers require 16× more provisioning than Stage 1. This demonstrates why accurate PD modelling matters — misclassifying a high-risk borrower as low-risk causes the bank to under-provision by 16×, creating hidden balance sheet risk.

---

## Methodology

### 1. Data Cleaning
- **Missing values:** `MonthlyIncome` (19.8% missing) and `NumberOfDependents` (2.6% missing) filled with median. Median chosen over mean because income is right-skewed — extreme high earners would inflate the mean.
- **Outliers:** `RevolvingUtilizationOfUnsecuredLines` clipped to [0, 1] (values above 100% are data errors). Borrowers aged < 18 removed (banks do not lend to minors).

### 2. Feature Engineering
Combined three delinquency columns into a single `TotalDelinquencies` feature:
```
TotalDelinquencies = 30-59 days past due + 60-89 days past due + 90+ days late
```
Delinquency history is the strongest predictor of default. One aggregated signal gives the model a clearer input than three correlated columns.

### 3. PD Model
- **Algorithm:** Logistic Regression with `class_weight="balanced"`
- **Why Logistic Regression:** High interpretability — each coefficient can be explained to regulators. Trade-off against XGBoost (higher accuracy but black-box).
- **Why `class_weight="balanced"`:** Default rate is only 6.7%. Without balancing, the model would predict "no default" for all borrowers and achieve 93% accuracy while being useless for risk management.
- **Scaling:** `StandardScaler` applied to training data; same scale applied to test data (test set does not re-fit the scaler — that would constitute data leakage).
- **Train/test split:** 80% train, 20% test. AUC evaluated on the held-out test set only.

### 4. IFRS 9 Stage Classification

| Stage | PD Threshold | ECL Horizon | Rationale |
|-------|-------------|-------------|-----------|
| Stage 1 | PD < 10% | 12 months | Low risk — short-term provision sufficient |
| Stage 2 | 10% ≤ PD < 50% | Lifetime (3 yr) | Credit quality has deteriorated |
| Stage 3 | PD ≥ 50% | Lifetime (3 yr) | High risk / in default |

### 5. ECL Calculation
```
ECL = PD × LGD × EAD × Horizon
```
| Parameter | Value | Assumption |
|-----------|-------|------------|
| PD | Model output | Per borrower |
| LGD | 45% | Basel II standard for unsecured retail |
| EAD | Monthly income × 12 | Proxy — actual loan balances not in dataset |
| Horizon | 1 year (S1) / 3 years (S2/S3) | Per IFRS 9 |

---

## Limitations

- `class_weight="balanced"` causes PD scores to be systematically overstated. In production, a calibration step (e.g. Platt scaling) would align predicted PDs with observed default rates.
- LGD and EAD use simplified assumptions. Real banks estimate LGD from historical recovery data and EAD from actual contract balances.
- Stage thresholds are PD-based only. IFRS 9 also uses a 30-days-past-due backstop not implemented here.
- Model is not validated on out-of-time data.

---

## Repository Structure

```
credit-risk-scorecard/
├── notebooks/
│   ├── notebook_01_eda.py          # EDA: default rate, income analysis, missing values
│   ├── notebook_02_pd_model.py     # PD model: cleaning, feature engineering, AUC = 0.83
│   └── notebook_03_ecl_engine.py   # IFRS 9 stage classification + ECL calculation
├── outputs/
│   ├── ecl_results.csv             # Per-borrower PD, Stage, EAD, ECL
│   └── ecl_stage_summary.csv       # ECL summary by IFRS 9 Stage
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/sijiama521-ship-it/credit-risk-scorecard.git
cd credit-risk-scorecard
pip install pandas scikit-learn matplotlib

# Download cs-training.csv from Kaggle and place in /data/
# https://www.kaggle.com/c/GiveMeSomeCredit/data

python notebooks/notebook_01_eda.py
python notebooks/notebook_02_pd_model.py
python notebooks/notebook_03_ecl_engine.py
```

---

## Tech Stack

- **Python** — pandas, scikit-learn, matplotlib
- **Modelling** — Logistic Regression, StandardScaler, train_test_split
- **Evaluation** — AUC-ROC

---

## Data Source

**Give Me Some Credit** — Kaggle Competition
[https://www.kaggle.com/c/GiveMeSomeCredit](https://www.kaggle.com/c/GiveMeSomeCredit)

