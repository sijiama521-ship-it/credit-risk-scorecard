# Notebook 03 — IFRS 9 ECL Engine
# Implements: Stage classification + Expected Credit Loss calculation
# Formula: ECL = PD x LGD x EAD x Horizon

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── 1. Load + clean + engineer (same as notebook_02) ─────────
df = pd.read_csv("../data/cs-training.csv")

df["MonthlyIncome"]     = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
df["NumberOfDependents"]= df["NumberOfDependents"].fillna(df["NumberOfDependents"].median())
df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(0, 1)
df = df[df["age"] > 18]

df["TotalDelinquencies"] = (
    df["NumberOfTime30-59DaysPastDueNotWorse"] +
    df["NumberOfTime60-89DaysPastDueNotWorse"] +
    df["NumberOfTimes90DaysLate"]
)

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_scaled, y_train)

# Score all borrowers
X_all_scaled = scaler.transform(df[features])
df["PD"] = model.predict_proba(X_all_scaled)[:, 1]

# ── 2. IFRS 9 Stage classification ───────────────────────────
# Stage 1: PD < 10%  — low risk,    12-month ECL
# Stage 2: PD 10-50% — medium risk, lifetime ECL
# Stage 3: PD >= 50% — high risk,   lifetime ECL
def assign_stage(pd_score):
    if pd_score < 0.10:
        return 1
    elif pd_score < 0.50:
        return 2
    else:
        return 3

df["Stage"] = df["PD"].apply(assign_stage)

print("Stage distribution:")
print(df["Stage"].value_counts().sort_index())

print("\nAverage PD by Stage:")
print(df.groupby("Stage")["PD"].mean().round(4))

# ── 3. ECL calculation ────────────────────────────────────────
# ECL = PD x LGD x EAD x Horizon
#
# LGD = 45%: Basel II standard assumption for unsecured retail loans.
#   Borrowers who default do not lose everything — banks recover ~55%
#   through collections. LGD = 1 - recovery rate.
#
# EAD = MonthlyIncome x 12: proxy for loan balance (actual contract
#   balances not available in this dataset).
#
# Horizon:
#   Stage 1 = 1 year  (short-term provision; risk is low)
#   Stage 2 = 3 years (lifetime provision; credit quality has deteriorated)
#   Stage 3 = 3 years (lifetime provision; borrower is high risk / in default)

LGD = 0.45

df["EAD"] = df["MonthlyIncome"] * 12

def ecl_horizon(stage):
    return 1 if stage == 1 else 3

df["Horizon"] = df["Stage"].apply(ecl_horizon)
df["ECL"]     = df["PD"] * LGD * df["EAD"] * df["Horizon"]

# ── 4. Results summary ────────────────────────────────────────
print("\nAverage ECL by Stage:")
print(df.groupby("Stage")["ECL"].mean().round(2))

total_ecl   = df["ECL"].sum()
avg_ecl     = df["ECL"].mean()
stage3_avg  = df[df["Stage"] == 3]["ECL"].mean()
stage1_avg  = df[df["Stage"] == 1]["ECL"].mean()
multiplier  = stage3_avg / stage1_avg

print(f"\nTotal portfolio ECL:       ${total_ecl:,.0f}")
print(f"Average ECL per borrower:  ${avg_ecl:,.0f}")
print(f"Stage 3 requires {multiplier:.0f}x more provisioning than Stage 1")

# Key insight:
# Stage 3 borrowers require ~16x more provisioning than Stage 1.
# This demonstrates why accurate PD modelling and Stage classification
# are critical — misclassifying a Stage 3 borrower as Stage 1
# would cause the bank to under-provision by 16x.

# ── 5. Save outputs ───────────────────────────────────────────
summary = df.groupby("Stage").agg(
    count=("ECL", "count"),
    avg_pd=("PD", "mean"),
    avg_ecl=("ECL", "mean"),
    total_ecl=("ECL", "sum")
).round(2)

summary.to_csv("../outputs/ecl_stage_summary.csv")
df[["PD", "Stage", "EAD", "ECL"]].to_csv("../outputs/ecl_results.csv", index=False)
print("\nOutputs saved to /outputs/")
