import pandas as pd

# Main table
app = pd.read_csv("data/processed/cleaned_application_train.csv")

# Engineered feature tables
bureau = pd.read_csv("data/processed/bureau_features.csv")
prev_app = pd.read_csv("data/processed/prev_app_features.csv")
payment = pd.read_csv("data/processed/payment_balance_features.csv")

# Merge step by step on SK_ID_CURR
df = app.merge(bureau, on="SK_ID_CURR", how="left")
df = df.merge(prev_app, on="SK_ID_CURR", how="left")
df = df.merge(payment, on="SK_ID_CURR", how="left")

# Check row count matches original train set
assert df.shape[0] == app.shape[0], "Row count mismatch!"

# Check no duplicate SK_ID_CURR
assert df["SK_ID_CURR"].is_unique, "Duplicate SK_ID_CURR found!"

# Check for missing TARGET
assert "TARGET" in df.columns, "TARGET missing!"


df.head()
df.describe().T.head(10)

df.to_csv("data/processed/final_train_features.csv", index=False)

## Final Feature Dataset Assembly

# - Merged all engineered feature sets:
#     - Bureau & bureau balance
#     - Previous applications
#     - Installments, POS, and credit card payment features
# - Ensured one row per applicant (`SK_ID_CURR`)
# - Preserved TARGET variable
# - Dataset saved as `final_train_features.csv` in processed folder
