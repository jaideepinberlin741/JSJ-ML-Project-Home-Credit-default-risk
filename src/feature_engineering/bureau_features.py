import pandas as pd
import numpy as np

# Bureau Data — External Credit History

# Load dataset
bureau = pd.read_csv("data/raw/bureau.csv")

# Aggregate numeric features to applicant level
bureau_agg = bureau.groupby("SK_ID_CURR").agg(
    BUREAU_CREDIT_COUNT=("SK_ID_BUREAU", "count"),
    BUREAU_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
    BUREAU_CREDIT_SUM_SUM=("AMT_CREDIT_SUM", "sum"),
    BUREAU_CREDIT_SUM_MAX=("AMT_CREDIT_SUM", "max"),
    BUREAU_CREDIT_SUM_MIN=("AMT_CREDIT_SUM", "min"),
    BUREAU_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
    BUREAU_DEBT_MEAN=("AMT_CREDIT_SUM_DEBT", "mean"),
    BUREAU_DEBT_MAX=("AMT_CREDIT_SUM_DEBT", "max"),
    BUREAU_OVERDUE_MAX=("AMT_CREDIT_MAX_OVERDUE", "max"),
    BUREAU_OVERDUE_MEAN=("AMT_CREDIT_MAX_OVERDUE", "mean"),
    BUREAU_DAYS_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
    BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
    BUREAU_DAYS_CREDIT_MIN=("DAYS_CREDIT", "min"),
    BUREAU_DAYS_CREDIT_MAX=("DAYS_CREDIT", "max"),
    BUREAU_PROLONG_SUM=("CNT_CREDIT_PROLONG", "sum"),
    BUREAU_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
    BUREAU_ANNUITY_SUM=("AMT_ANNUITY", "sum"),
).reset_index()

# Aggregate credit status (Active/Closed counts)
active_counts = bureau.groupby(["SK_ID_CURR", "CREDIT_ACTIVE"]).size().unstack(fill_value=0)
active_counts.columns = ["BUREAU_" + col.upper().replace(" ", "_") for col in active_counts.columns]
active_counts = active_counts.reset_index()

# Merge active counts
bureau_agg = bureau_agg.merge(active_counts, on="SK_ID_CURR", how="left")

# Create derived features
bureau_agg["BUREAU_DEBT_CREDIT_RATIO"] = bureau_agg["BUREAU_DEBT_SUM"] / bureau_agg["BUREAU_CREDIT_SUM_SUM"]
bureau_agg["BUREAU_ACTIVE_RATIO"] = bureau_agg.get("BUREAU_ACTIVE", 0) / bureau_agg["BUREAU_CREDIT_COUNT"]

# Replace inf with NaN
bureau_agg = bureau_agg.replace([np.inf, -np.inf], np.nan)

# Verify unique applicants
assert bureau_agg["SK_ID_CURR"].is_unique

# Save Output
bureau_agg.to_csv("data/processed/bureau_features.csv", index=False)

print(f"Bureau features saved: {bureau_agg.shape}")

## Bureau Feature Engineering
#
# External credit history from other financial institutions was aggregated
# to applicant-level features. Credit counts, amounts, debt levels, and
# overdue statistics provide insight into the applicant's credit behavior
# outside of Home Credit. The debt-to-credit ratio and active credit ratio
# are derived features that capture overall credit utilization patterns.
