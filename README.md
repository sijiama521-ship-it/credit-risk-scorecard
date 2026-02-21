# 🏦 Credit Risk Scorecard + IFRS 9 ECL Engine

A full end-to-end credit risk modelling project built in Python, covering probability of default (PD) estimation, IFRS 9 stage classification, and expected credit loss (ECL) calculation across multiple economic scenarios.

---

## 📌 Project Overview

This project replicates a bank-grade credit risk workflow using the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset (150,000 retail borrowers). It is structured around three core deliverables:

| Component | Description |
|-----------|-------------|
| **PD Model** | Logistic Regression model to estimate probability of default |
| **ECL Engine** | IFRS 9-style expected credit loss calculator |
| **Model Governance** | Assumptions, limitations, and monitoring framework |

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.8611** |
| KS Statistic | **0.5610** |
| Overall Default Rate | ~6.7% |
| Weighted ECL Coverage | See `ecl_stage_summary.csv` |

---

## 🗂️ Repository Structure

```
credit-risk-scorecard/
│
├── data/
│   ├── cs-training.csv           # Raw data (from Kaggle)
│   └── cs-training-clean.csv     # Cleaned data (output of Notebook 1)
│
├── notebooks/
│   ├── notebook_01_eda.ipynb           # Exploratory Data Analysis & Cleaning
│   ├── notebook_02_pd_model.ipynb      # PD Model Training & Evaluation
│   └── notebook_03_ecl_engine.ipynb    # IFRS 9 ECL Calculator
│
├── report/
│   ├── plot_01_target_distribution.png
│   ├── plot_06_roc_curve.png
│   ├── plot_07_confusion_matrix.png
│   ├── plot_08_calibration_curve.png
│   ├── plot_09_feature_importance.png
│   ├── plot_11_ifrs9_stages.png
│   ├── plot_12_scenario_ecl.png
│   └── plot_13_pd_by_stage.png
│
├── outputs/
│   ├── pd_predictions.csv        # PD scores for test set
│   ├── ecl_results.csv           # Full ECL results per borrower
│   └── ecl_stage_summary.csv     # ECL summary by IFRS 9 Stage
│
└── README.md
```

---

## 🔧 Tech Stack

- **Python** — pandas, numpy, scikit-learn, matplotlib, seaborn
- **Modelling** — Logistic Regression (sklearn), StandardScaler
- **Reporting** — Jupyter Notebook, joblib

---

## 🚀 How to Run

### Option 1: Google Colab (recommended)
1. Open each notebook in [Google Colab](https://colab.research.google.com)
2. Upload the required CSV files when prompted
3. Run all cells: `Runtime → Run all`

### Option 2: Local
```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-scorecard.git
cd credit-risk-scorecard
pip install pandas numpy scikit-learn matplotlib seaborn joblib
jupyter notebook
```
Run notebooks in order: 01 → 02 → 03

---

## 📐 Methodology

### 1. Data Cleaning (Notebook 1)
- 150,000 borrower records from Kaggle
- Missing value imputation: median for `monthly_income` and `dependents`
- Outlier treatment: age filter (18–100), utilisation rate clipping, delinquency code removal

### 2. PD Model (Notebook 2)
- **Model:** Logistic Regression with `class_weight='balanced'`
- **Features:** 12 variables including engineered features (total delinquency, log-income, income per dependent)
- **Evaluation:** AUC-ROC = 0.8611, KS = 0.5610, calibration curve

### 3. IFRS 9 ECL Engine (Notebook 3)

**ECL Formula:**
$$ECL = PD \times LGD \times EAD \times DF$$

| Parameter | Assumption |
|-----------|-----------|
| PD | Model output (per borrower) |
| LGD | 45% (Basel II unsecured retail standard) |
| EAD | 12× monthly income proxy |
| Discount Factor | 1 / (1 + 5%) |

**Stage Classification:**
| Stage | Criteria | ECL Horizon |
|-------|----------|-------------|
| Stage 1 | PD < 10% | 12-month ECL |
| Stage 2 | 10% ≤ PD < 50% | Lifetime ECL (3 years) |
| Stage 3 | PD ≥ 50% or defaulted | Lifetime ECL (3 years) |

**Scenario Weighting:**
| Scenario | Weight | PD Adjustment |
|----------|--------|---------------|
| Optimistic | 25% | −30% |
| Base | 50% | No change |
| Pessimistic | 25% | +50% |

---

## ⚠️ Limitations & Assumptions

- LGD and EAD are fixed assumptions, not estimated from data
- EAD is proxied from income, not actual loan balances
- The model is not validated on out-of-time data
- Stage thresholds are based on PD only (not 30 DPD backstop)
- Scenario multipliers are illustrative, not macro-model-derived

---

## 📄 Data Source

**Give Me Some Credit** — Kaggle Competition  
[https://www.kaggle.com/c/GiveMeSomeCredit](https://www.kaggle.com/c/GiveMeSomeCredit)

---

## 👤 Author

**Your Name**  
[LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)
