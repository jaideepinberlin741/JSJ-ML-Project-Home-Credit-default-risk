import pandas as pd
import numpy as np

# Installments Payments — Repayment Behavior

# Load dataset
inst = pd.read_csv("data/raw/installments_payments.csv")

# Create payment delay feature
inst["PAYMENT_DELAY"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
inst["LATE_PAYMENT"] = (inst["PAYMENT_DELAY"] > 0).astype(int)

# Aggregate to applicant level
inst_agg = inst.groupby("SK_ID_CURR").agg(
    INST_COUNT=("SK_ID_PREV", "count"),
    LATE_PAYMENT_RATE=("LATE_PAYMENT", "mean"),
    AVG_PAYMENT_DELAY=("PAYMENT_DELAY", "mean"),
    MAX_PAYMENT_DELAY=("PAYMENT_DELAY", "max"),
    TOTAL_PAID=("AMT_PAYMENT", "sum")
).reset_index()

# POS_CASH_balance — Loan Status Patterns

# Load & aggregate POS_CASH_balance data
pos = pd.read_csv("data/raw/POS_CASH_balance.csv")
pos_agg = pos.groupby("SK_ID_CURR").agg(
    POS_LOAN_COUNT=("SK_ID_PREV", "nunique"),
    POS_AVG_DPD=("SK_DPD", "mean"),
    POS_MAX_DPD=("SK_DPD", "max"),
    POS_AVG_DPD_DEF=("SK_DPD_DEF", "mean")
).reset_index()

# Credit Card Balance — Usage & Risk Signals

# Load & aggregate credit card balance data
cc = pd.read_csv("data/raw/credit_card_balance.csv")
cc_agg = cc.groupby("SK_ID_CURR").agg(
    CC_LOAN_COUNT=("SK_ID_PREV", "nunique"),
    CC_AVG_BALANCE=("AMT_BALANCE", "mean"),
    CC_MAX_BALANCE=("AMT_BALANCE", "max"),
    CC_AVG_DPD=("SK_DPD", "mean"),
    CC_MAX_DPD=("SK_DPD", "max")
).reset_index()

# Merge All Payment & Balance Features
payment_features = inst_agg.merge(pos_agg, on="SK_ID_CURR", how="left")
payment_features = payment_features.merge(cc_agg, on="SK_ID_CURR", how="left")

assert payment_features["SK_ID_CURR"].is_unique

# Save Output
payment_features.to_csv("data/processed/payment_balance_features.csv", index=False)

## Payment & Balance Feature Engineering

# Historical repayment behavior was aggregated to applicant-level features.
# Late payment frequency, delay severity, and days past due were extracted
# as these signals are strong predictors of credit default risk.
# POS_CASH_balance data provided patterns of loan status over time,
# indicating how well clients manage their credit obligations.
# Credit card balance data revealed usage patterns and risk signals,
# helping identify clients who may struggle with debt repayment.
# These features collectively provide a comprehensive view of a client's
# repayment behavior and financial stability.

## Conclusion

# The payment and balance features are now available in the processed data directory.
# These features will be used to train the machine learning model.