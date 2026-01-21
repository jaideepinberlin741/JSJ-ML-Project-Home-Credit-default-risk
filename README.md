# Home Credit Default Risk: Prediction Model

**Project Status:** In Progress 🚧

This repository contains the work for building a machine learning model to predict the probability of a client defaulting on a loan, based on the Kaggle competition ["Home Credit Default Risk"](https://www.kaggle.com/c/home-credit-default-risk).

## 1. Problem Statement

> How can we accurately predict a loan applicant's repayment ability using alternative data sources, in order to broaden financial inclusion for the unbanked and ensure they have a positive, safe borrowing experience?

### Business Value
A successful model will enable Home Credit to provide loans more safely and effectively to a broader population, including those with limited or non-existent credit histories. This helps unlock financial services for underserved communities while managing business risk.

## 2. Success Metrics

### ML Metric: ROC-AUC
The primary evaluation metric for this project is the Area Under the Receiver Operating Characteristic Curve (ROC-AUC).

* **Justification**: This metric is ideal for this problem for two key reasons:

1. **Handles Class Imbalance**: The dataset is highly imbalanced, with far more repaid loans than defaulted ones. Metrics like accuracy can be misleading, but ROC-AUC provides a reliable measure of the model's ability to distinguish between the two classes regardless of their distribution.
2. **Measures Discriminatory Power**: It evaluates how well the model ranks predictions, giving a higher probability to a default than to a repayment. This is crucial for the business goal of identifying risky applicants.

* **Success Definition**: The model will be considered successful if it achieves a ROC-AUC score of 0.75 or higher on the holdout test set. This target represents a strong predictive model that provides significant business value.

### Product & Business Metrics
From a business perspective, the model's value is determined by its impact on lending decisions. This requires balancing the risks of two types of errors:

* False Positives (Predict Repaid, Actual Default): This is the most costly error. It means the bank approves a loan for an applicant who ultimately defaults, resulting in a direct financial loss.
* False Negatives (Predict Default, Actual Repaid): This represents a missed business opportunity. The bank denies a loan to a capable applicant, losing potential revenue and failing in the mission of financial inclusion.

A successful model will therefore be one that reduces the number of false positives (to minimize financial loss) while also minimizing false negatives (to increase business and serve the unbanked). The specific threshold for approving loans will be determined by balancing these two competing costs.


### ML Metric
The primary evaluation metric for this project is **Area Under the Receiver Operating Characteristic Curve (ROC-AUC)**. This metric is well-suited for this binary classification problem as it measures the model's ability to correctly distinguish between clients who will repay and those who will default, independent of the class imbalance.

### Product Metric
Success will be measured by the model's ability to **increase the number of approved loans for capable applicants** (i.e., reduce false negatives) while keeping the default rate (false positives) at an acceptable level.

## 3. Dataset

### Source
This project uses the Home Credit Default Risk dataset from Kaggle. The data is provided by Home Credit, a financial services company, and contains real, anonymized information about their loan applicants.

### Overview
The objective is to use historical loan application data to predict the probability that a client will default on a loan. The dataset is unique because it includes not only demographic and financial information from the loan application itself but also extensive alternative data sources, such as transactional history and credit history from other institutions.

### Data Structure
The data is spread across several tables, linked by a common identifier (SK_ID_CURR).

**Main Table (application_train/test.csv):**

Contains one row per loan application.
Includes applicant demographics, financial information, loan details, and external credit scores.
This table contains the **TARGET** variable for training:
1 → The client defaulted on the loan.
0 → The loan was repaid.
**Auxiliary Tables:** These tables provide historical data and have a **one-to-many relationship** with the main application data. They are crucial for feature engineering.

* bureau.csv & bureau_balance.csv: Data from other credit bureaus.
* previous_application.csv: History of the client's previous loans with Home Credit.
* POS_CASH_balance.csv: Monthly balance snapshots of previous point-of-sale or cash loans.
* credit_card_balance.csv: Monthly balance snapshots for credit cards.
* installments_payments.csv: Repayment history for previous loans.

### Key Characteristics & Challenges
* **High Imbalance:** The dataset is highly imbalanced, with a small percentage of loans resulting in a default. This is a key challenge that will influence model training and evaluation.
* **Feature Engineering:** The rich, multi-table structure means that significant feature engineering—primarily through the aggregation of historical data—is required to create a useful feature set for the model.

This project uses the **Home Credit Default Risk** dataset from Kaggle, which contains anonymized information about loan applicants and their credit history.
The objective is to predict whether a client will **default on a loan**.


### Main Table

* **application_train.csv / application_test.csv**
* One row per loan application
* Contains applicant demographics, financial information, loan details, and external credit scores
* `TARGET` (train only):

  * `1` → Default
  * `0` → Repaid

### Auxiliary Tables (Historical Data)

* **bureau.csv / bureau_balance.csv**
  External credit history and monthly loan status from other financial institutions
* **previous_application.csv**
  Previous loan applications with Home Credit
* **POS_CASH_balance.csv**
  Monthly history of POS and cash loans
* **credit_card_balance.csv**
  Monthly credit card usage history
* **installments_payments.csv**
  Detailed repayment and payment delay information

### Relationships

* `SK_ID_CURR` links all tables to the main application data
* One-to-many relationships require aggregation to applicant-level features

### Key Notes

* Dataset is highly imbalanced (few defaults)
* Historical repayment behavior is a strong predictor of default risk
* Feature engineering through aggregation is critical
* ROC-AUC is used as the primary evaluation metric

**Source:** [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data) – Home Credit Default Risk


## 4. Project Structure
├── data/ │ ├── raw/ # Raw, immutable data from Kaggle │ └── processed/ # Cleaned and preprocessed data ├── notebooks/ # Jupyter notebooks for EDA and experimentation ├── scripts/ │ ├── data_processing.py │ └── train_model.py ├── src/ # Source code for the project ├── README.md # This file! └── requirements.txt # Project dependencies


## 5. Getting Started

### Prerequisites
*   Python 3.8+ (Less,, like our previous notebooks)
*   Poetry (or pip) for dependency management

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [YOUR_REPOSITORY_URL]
    cd [YOUR_REPOSITORY_NAME]
    ```

2.  **Set up the environment and install dependencies:**

The added [requirements file](requirements.txt) contains all libraries and dependencies we need to execute the notebooks.

### **`macOS`** type the following commands : 

- Install the virtual environment and the required packages by following commands:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-Bash` CLI :
    ```
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

*Note: If there are errors during environment setup, try removing the versions from the failing packages in the requirements file.*


3.  **Download the data:**
    Place the raw data files from Kaggle into the `data/raw/` directory.

## 6. Usage

*(This section will be updated with instructions on how to run the data processing and model training scripts.)*
