# ==================================================
# FINAL NON-HANGING PIPELINE (macOS SAFE)
# XGBoost + LightGBM + SHAP
# ==================================================

import os
import pickle
import numpy as np
import pandas as pd

# ---- IMPORTANT: Non-interactive backend ----
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# --------------------------------------------------
# 1️⃣ Setup
# --------------------------------------------------
os.makedirs("models", exist_ok=True)

print("Loading dataset...")
df = pd.read_csv("data/processed/final_train_features.csv")

X = df.drop(columns=["TARGET", "SK_ID_CURR"]).select_dtypes(include=["int64", "float64"])
y = df["TARGET"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# --------------------------------------------------
# 2️⃣ Initial XGBoost training
# --------------------------------------------------
print("\nTraining initial XGBoost model...")

xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    learning_rate=0.05,
    n_estimators=500,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=1
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=30,
    verbose=50
)

y_pred = xgb_model.predict_proba(X_val)[:, 1]
print("Initial XGBoost ROC-AUC:", round(roc_auc_score(y_val, y_pred), 4))

# --------------------------------------------------
# 3️⃣ SHAP-based feature pruning (SAFE)
# --------------------------------------------------
print("\nComputing SHAP for feature pruning...")

shap_sample = X_train.sample(n=min(5000, len(X_train)), random_state=42)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(shap_sample)

shap_importance = pd.DataFrame({
    "feature": shap_sample.columns,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values(by="mean_abs_shap", ascending=False)

threshold = shap_importance["mean_abs_shap"].quantile(0.3)
low_impact_features = shap_importance[
    shap_importance["mean_abs_shap"] < threshold
]["feature"].tolist()

print(f"Dropping {len(low_impact_features)} weak features")

X_reduced = X.drop(columns=low_impact_features)

X_train_red, X_val_red, y_train_red, y_val_red = train_test_split(
    X_reduced, y, test_size=0.2, stratify=y, random_state=42
)

# --------------------------------------------------
# 4️⃣ Final XGBoost model
# --------------------------------------------------
print("\nTraining final XGBoost model...")

final_xgb = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    learning_rate=0.05,
    n_estimators=500,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=1
)

final_xgb.fit(
    X_train_red, y_train_red,
    eval_set=[(X_val_red, y_val_red)],
    early_stopping_rounds=30,
    verbose=50
)

y_pred_xgb = final_xgb.predict_proba(X_val_red)[:, 1]
print("Final XGBoost ROC-AUC:", round(roc_auc_score(y_val_red, y_pred_xgb), 4))

# --------------------------------------------------
# 5️⃣ Final LightGBM model (CORRECT API)
# --------------------------------------------------
print("\nTraining final LightGBM model...")

lgb_train = lgb.Dataset(X_train_red, label=y_train_red)
lgb_val = lgb.Dataset(X_val_red, label=y_val_red, reference=lgb_train)

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "is_unbalance": True,
    "verbose": -1,
    "seed": 42
}

final_lgb = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_train, lgb_val],
    valid_names=["train", "valid"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=30),
        lgb.log_evaluation(period=50)
    ]
)

y_pred_lgb = final_lgb.predict(X_val_red)
print("Final LightGBM ROC-AUC:", round(roc_auc_score(y_val_red, y_pred_lgb), 4))

# --------------------------------------------------
# 6️⃣ SHAP explainability (NON-BLOCKING)
# --------------------------------------------------
print("\nGenerating SHAP plots...")

shap_sample_final = X_reduced.sample(n=min(5000, len(X_reduced)), random_state=42)

# XGBoost SHAP
xgb_explainer = shap.TreeExplainer(final_xgb)
xgb_shap_values = xgb_explainer.shap_values(shap_sample_final)

shap.summary_plot(xgb_shap_values, shap_sample_final, plot_type="bar", show=False)
plt.savefig("artifacts/xgb_shap_bar.png", dpi=200)
plt.close()

shap.summary_plot(xgb_shap_values, shap_sample_final, show=False)
plt.savefig("artifacts/xgb_shap_beeswarm.png", dpi=200)
plt.close()

# LightGBM SHAP
lgb_explainer = shap.TreeExplainer(final_lgb)
lgb_shap_values = lgb_explainer.shap_values(shap_sample_final)

shap.summary_plot(lgb_shap_values, shap_sample_final, plot_type="bar", show=False)
plt.savefig("artifacts/lgb_shap_bar.png", dpi=200)
plt.close()

shap.summary_plot(lgb_shap_values, shap_sample_final, show=False)
plt.savefig("artifacts/lgb_shap_beeswarm.png", dpi=200)
plt.close()

# --------------------------------------------------
# 7️⃣ Save models
# --------------------------------------------------
print("\nSaving models and artifacts...")

with open("models/xgboost_final_model.pkl", "wb") as f:
    pickle.dump(final_xgb, f)

final_lgb.save_model("models/lightgbm_final_model.txt")

np.save("models/xgb_shap_values.npy", xgb_shap_values)
np.save("models/lgb_shap_values.npy", lgb_shap_values)

pd.DataFrame({"feature": X_reduced.columns}).to_csv(
    "artifacts/final_features_list.csv", index=False
)

print("\n✅ PIPELINE COMPLETED SUCCESSFULLY")
