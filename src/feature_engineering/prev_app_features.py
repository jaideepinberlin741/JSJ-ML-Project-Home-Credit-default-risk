import pandas as pd
import numpy as np

# Previous Application Data — Home Credit Loan History

# Load dataset
prev_app = pd.read_csv("data/raw/previous_application.csv")

# Aggregate numeric features to applicant level
prev_agg = prev_app.groupby("SK_ID_CURR").agg(
    PREV_APP_COUNT=("SK_ID_PREV", "count"),
    PREV_CREDIT_MEAN=("AMT_CREDIT", "mean"),
    PREV_CREDIT_SUM=("AMT_CREDIT", "sum"),
    PREV_CREDIT_MAX=("AMT_CREDIT", "max"),
    PREV_CREDIT_MIN=("AMT_CREDIT", "min"),
    PREV_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
    PREV_APPLICATION_SUM=("AMT_APPLICATION", "sum"),
    PREV_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
    PREV_ANNUITY_MAX=("AMT_ANNUITY", "max"),
    PREV_DOWN_PAYMENT_MEAN=("AMT_DOWN_PAYMENT", "mean"),
    PREV_DOWN_PAYMENT_MAX=("AMT_DOWN_PAYMENT", "max"),
    PREV_CNT_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),
    PREV_CNT_PAYMENT_SUM=("CNT_PAYMENT", "sum"),
).reset_index()

# Aggregate contract status (Approved/Refused counts)
status_counts = prev_app.groupby(["SK_ID_CURR", "NAME_CONTRACT_STATUS"]).size().unstack(fill_value=0)
status_counts.columns = ["PREV_" + col.upper().replace(" ", "_") for col in status_counts.columns]
status_counts = status_counts.reset_index()

# Merge status counts
prev_agg = prev_agg.merge(status_counts, on="SK_ID_CURR", how="left")

# Aggregate contract types
type_counts = prev_app.groupby(["SK_ID_CURR", "NAME_CONTRACT_TYPE"]).size().unstack(fill_value=0)
type_counts.columns = ["PREV_TYPE_" + col.upper().replace(" ", "_") for col in type_counts.columns]
type_counts = type_counts.reset_index()

# Merge type counts
prev_agg = prev_agg.merge(type_counts, on="SK_ID_CURR", how="left")

# Create derived features
if "PREV_APPROVED" in prev_agg.columns:
    prev_agg["PREV_APPROVAL_RATIO"] = prev_agg["PREV_APPROVED"] / prev_agg["PREV_APP_COUNT"]
if "PREV_REFUSED" in prev_agg.columns:
    prev_agg["PREV_REFUSED_RATIO"] = prev_agg["PREV_REFUSED"] / prev_agg["PREV_APP_COUNT"]
prev_agg["PREV_CREDIT_APPLICATION_RATIO"] = prev_agg["PREV_CREDIT_SUM"] / prev_agg["PREV_APPLICATION_SUM"]

# Replace inf with NaN
prev_agg = prev_agg.replace([np.inf, -np.inf], np.nan)

# Verify unique applicants
assert prev_agg["SK_ID_CURR"].is_unique

# Save Output
prev_agg.to_csv("data/processed/prev_app_features.csv", index=False)

print(f"Previous application features saved: {prev_agg.shape}")

## Previous Application Feature Engineering
#
# Historical loan applications at Home Credit were aggregated to applicant-level
# features. Application counts, credit amounts, and approval/rejection patterns
# provide insight into the applicant's history with Home Credit specifically.
# The approval ratio indicates how often the applicant was approved in the past,
# while the credit-to-application ratio shows how much they received vs requested.
