# Notebook 02 — PD Model (Probability of Default)
# Model: Logistic Regression
# Evaluation: AUC-ROC on held-out test set (20%)

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ── 1. Load data ──────────────────────────────────────────────
df = pd.read_csv("../data/cs-training.csv")

# ── 2. Data cleaning ──────────────────────────────────────────

# 2a. Fill missing values with median
# Reason: income is right-skewed; median is more representative than mean
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
df["NumberOfDependents"] = df["NumberOfDependents"].fillna(df["NumberOfDependents"].median())

# 2b. Clip outliers
# RevolvingUtilization should be 0-1 (0%-100%); values above 1 are data errors
df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(0, 1)

# 2c. Remove invalid ages
# Banks do not lend to minors; age < 18 is a data entry error
df = df[df["age"] > 18]

# ── 3. Feature engineering ────────────────────────────────────
# Combine three delinquency columns into one total delinquency count
# Rationale: delinquency history is the strongest predictor of default;
# a single aggregated signal is more direct than three separate columns
df["TotalDelinquencies"] = (
    df["NumberOfTime30-59DaysPastDueNotWorse"] +
    df["NumberOfTime60-89DaysPastDueNotWorse"] +
    df["NumberOfTimes90DaysLate"]
)

# ── 4. Train / test split ─────────────────────────────────────
features = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "MonthlyIncome",
    "DebtRatio",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
    "NumberOfDependents",
    "TotalDelinquencies"
]

X = df[features]
y = df["SeriousDlqin2yrs"]

# 80% train, 20% test — model is evaluated on data it has never seen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train):,} rows")
print(f"Test set:     {len(X_test):,} rows")

# ── 5. Scale features ─────────────────────────────────────────
# StandardScaler normalises all features to the same range
# so the model treats each feature fairly regardless of unit size
# Important: fit only on training data; apply same scale to test data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 6. Train model ────────────────────────────────────────────
# class_weight="balanced" addresses class imbalance (6.7% default rate)
# Without it, the model would predict "no default" for everyone and
# achieve 93% accuracy while being completely useless for risk management
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_scaled, y_train)

# ── 7. Evaluate ───────────────────────────────────────────────
pred = model.predict_proba(X_test_scaled)[:, 1]
auc  = roc_auc_score(y_test, pred)
print(f"\nTest set AUC: {auc:.4f}")
# AUC = 0.83 — exceeds the 0.80 threshold considered acceptable in banking

# AUC interpretation:
# If we randomly pick one defaulter and one non-defaulter,
# the model assigns the defaulter a higher risk score 83% of the time

# ── 8. Known limitations ──────────────────────────────────────
# class_weight="balanced" causes PD scores to be systematically overstated.
# In production, a calibration step (e.g. Platt scaling) would be applied
# to align predicted PDs with observed default rates.
# LGD and EAD in the ECL engine use simplified assumptions, not contract data.
