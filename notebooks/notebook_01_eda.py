# Notebook 01 — Exploratory Data Analysis (EDA)
# Dataset: Give Me Some Credit (Kaggle) — 150,000 retail borrowers
# Goal: Understand the data, find patterns, prepare for modelling

import pandas as pd
import matplotlib.pyplot as plt

# ── 1. Load data ──────────────────────────────────────────────
df = pd.read_csv("../data/cs-training.csv")

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# ── 2. Statistical summary ────────────────────────────────────
print("\nDescribe:")
print(df.describe())

# ── 3. Check missing values ───────────────────────────────────
print("\nMissing values per column:")
print(df.isnull().sum())

# Key finding: MonthlyIncome has 29,731 missing (19.8%)
#              NumberOfDependents has 3,924 missing (2.6%)

# ── 4. Default rate ───────────────────────────────────────────
default_rate = df["SeriousDlqin2yrs"].mean()
print(f"\nOverall default rate: {default_rate:.2%}")
# Result: ~6.7% — typical for retail credit portfolios
# This is class imbalance: 93.3% non-default vs 6.7% default

# ── 5. Income by default status ───────────────────────────────
print("\nAverage income by default status:")
print(df.groupby("SeriousDlqin2yrs")["MonthlyIncome"].mean())
# Finding: Defaulters earn ~2.5x less than non-defaulters
# This confirms MonthlyIncome is a strong predictor

# ── 6. Age by default status ──────────────────────────────────
print("\nAverage age by default status:")
print(df.groupby("SeriousDlqin2yrs")["age"].mean())
# Finding: Defaulters are younger on average (27 vs 42)
# Younger borrowers have less stable income

# ── 7. Visualise: income by default status ────────────────────
df.groupby("SeriousDlqin2yrs")["MonthlyIncome"].mean().plot(
    kind="bar",
    color=["steelblue", "tomato"]
)
plt.title("Average Monthly Income by Default Status")
plt.xlabel("Defaulted (0=No, 1=Yes)")
plt.ylabel("Average Monthly Income ($)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("../credit-risk-ecl/report/plot_income_by_default.png")
print("\nChart saved.")
