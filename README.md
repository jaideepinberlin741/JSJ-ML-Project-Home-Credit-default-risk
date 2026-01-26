# Home Credit Default Risk — Machine Learning Project
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-success?logo=leaflet)
![Explainability](https://img.shields.io/badge/Explainability-SHAP-purple)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue?logo=kaggle)

**Project Description 🚧  
A complete end‑to‑end machine learning pipeline for predicting loan default risk using the Kaggle **Home Credit Default Risk** dataset.  
The project combines data engineering, modeling, explainability, and business reasoning to support safer and more inclusive lending decisions.

---

## 1. Problem Statement

> How can we accurately predict a loan applicant’s repayment ability using alternative data sources, in order to broaden financial inclusion while managing lending risk?

Home Credit aims to serve clients with limited or no credit history. Traditional credit scoring often fails these applicants, leading to unnecessary rejections and missed opportunities.  
This project builds a predictive model that estimates default probability using rich behavioral and financial data.

---

## 2. Success Metrics

### ML Metric — ROC‑AUC
Chosen because it:
- Handles **class imbalance** effectively  
- Measures **ranking ability**, not just classification  
- Reflects how well the model separates risky vs safe applicants  

**Success Criterion:**  
A model is considered successful if it achieves **ROC‑AUC ≥ 0.75** on the holdout test set.

### Business Metrics
Two error types matter:

- **False Positive (Predict Repaid → Actually Default)**  
  → Most costly: leads to financial loss.

- **False Negative (Predict Default → Actually Repaid)**  
  → Missed opportunity: reduces revenue and harms financial inclusion.

A good model minimizes both, with the final threshold chosen based on business trade‑offs.

---

## 3. Dataset Overview

Source: Kaggle — *Home Credit Default Risk*  
The dataset includes demographic, financial, and behavioral data across multiple relational tables.

### Main Table
`application_train.csv` / `application_test.csv`  
- One row per loan application  
- Contains demographics, income, credit amounts, loan details  
- `TARGET`:  
  - `1` → Default  
  - `0` → Repaid  

### Auxiliary Tables
Used for feature engineering (one‑to‑many relationships):

- `bureau.csv`, `bureau_balance.csv` — external credit history  
- `previous_application.csv` — past Home Credit loans  
- `POS_CASH_balance.csv` — POS/cash loan history  
- `credit_card_balance.csv` — credit card usage  
- `installments_payments.csv` — repayment behavior  

### Key Challenges
- Highly imbalanced dataset  
- Heavy feature engineering required  
- Multiple relational tables requiring aggregation  

---

## 4. Project Structure

```
JSJ-ML-Project-Home-Credit-default-risk/
│
├── data/
│   ├── raw/          # Raw Kaggle data (ignored in Git)
│   └── processed/    # Preprocessed datasets
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_lightgbm_model.ipynb
│   ├── 05_shap_explainability.ipynb
│   ├── 06_threshold_analysis.ipynb
│   └── 07_submission.ipynb
│
├── models/           # Saved model files (e.g., lightgbm_model.pkl)
├── submissions/      # Kaggle submission files
├── src/              # Optional scripts
├── requirements.txt
└── README.md
```

---

## 5. Getting Started

### Prerequisites
- Python 3.11.3  
- `pip`
- `pyenv` (optional but recommended)

---

### Installation

#### macOS
```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Windows (PowerShell)
```powershell
pyenv local 3.11.3
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6. Data Setup

⚠️ **Important:** The dataset is too large for GitHub.  
You must download it manually.

### Option A — Raw Kaggle Data (Full Pipeline)
Download from Kaggle and place all `.csv` files into:

```
data/raw/
```

### Option B — Preprocessed Data (Recommended)
Download from Google Drive:  
https://drive.google.com/drive/folders/1sF8oaBiNfejXVH303rNFqUEFYP85arG0?usp=drive_link

Place files into:

```
data/processed/
```

---

## 7. Usage

Run the notebooks in the following order:

1. `01_eda.ipynb` — Exploratory data analysis  
2. `02_feature_engineering.ipynb` — Aggregations and feature creation  
3. `03_preprocessing.ipynb` — Cleaning, encoding, and preparing the dataset  
4. `04_lightgbm_model.ipynb` — Training the LightGBM model  
5. `05_shap_explainability.ipynb` — SHAP global and local interpretability  
6. `06_threshold_analysis.ipynb` — Business-driven threshold selection  
7. `07_submission.ipynb` — Generate final predictions for Kaggle submission  

Model outputs are saved in:

```
models/lightgbm_model.pkl
```

---

## 8. Roadmap

### Phase 1 — Problem Framing
Define business context, personas, metrics, constraints.

### Phase 2 — Data Understanding
EDA, missing values, correlations, class imbalance.

### Phase 3 — Feature Engineering
Aggregations, ratios, domain‑inspired features.

### Phase 4 — Modeling
Baseline → LightGBM → tuning → evaluation.

### Phase 5 — Explainability
SHAP analysis, risk threshold exploration.

### Phase 6 — Delivery
Save model, generate submission, finalize documentation.

---

## 9. Future Improvements
- Hyperparameter tuning  
- Threshold optimization  
- Model deployment pipeline  
- Fairness analysis  
- Automated reporting  
```

## Future Improvements
...

## Contributors
This project was developed as part of a team assignment for the Data Science program.  
Contributions were made by our team across data exploration, modeling, and documentation.

- **Jaideep** – https://github.com/jaideepinberlin741/
- **Sumit** – https://github.com/summyhug
- **Jolanda** – https://github.com/joolanda

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

