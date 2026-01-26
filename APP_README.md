# Loan Approval Demo Web App

A Streamlit-based web application for the Home Credit Default Risk ML project that provides an interactive loan approval interface.

## Features

- **Interactive Form**: Single-page form with 10 input fields for loan application details
- **ML Prediction**: Uses trained LightGBM model to predict default probability
- **Approval Decision**: Automatic approval/denial based on 0.20 risk threshold
- **Visual Indicators**: Color-coded risk scores (green/yellow/red) and decision display
- **Sample Applicant**: Quick-load button to populate form with example values
- **Bank-Style Design**: Professional banking dashboard aesthetic

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

**Important:** If you encounter a `numpy._core` error, upgrade NumPy:
```bash
pip install --upgrade 'numpy>=1.24.0'
```

2. Ensure the following files exist:
   - `models/lightgbm_model.pkl` - Trained LightGBM model
   - `data/processed/final_train_features.csv` - Training data with feature names

## Troubleshooting

### NumPy/Joblib Version Compatibility Errors

If you see errors like:
- `No module named 'numpy._core'`
- `module 'numpy._globals' has no attribute '_signature_descriptor'`

These occur when the model was saved with different NumPy/Joblib versions than your current environment.

**Quick Fix - Try in order:**

1. **Upgrade to latest versions (recommended):**
   ```bash
   pip install --upgrade 'numpy>=1.24.0' 'joblib>=1.3.0'
   ```

2. **Or try specific compatible versions:**
   ```bash
   pip install numpy==1.26.4 joblib==1.3.2
   ```

3. **If that doesn't work, try downgrading:**
   ```bash
   pip install 'numpy<1.24' 'joblib<1.3'
   ```

4. **Run diagnostic script:**
   ```bash
   python fix_model_loading.py
   ```

**Important:** After changing versions, **completely restart** the Streamlit app (stop and start again).

## Running the App

Start the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## User Input Fields

1. **Credit Score** (300-850) → Maps to `EXT_SOURCE_2` and `EXT_SOURCE_3` (normalized)
2. **Loan Amount Requested** ($) → `AMT_CREDIT`
3. **Monthly Payment** ($) → `AMT_ANNUITY`
4. **Age** (years) → `DAYS_BIRTH` (converted to negative days)
5. **Years at Current Job** → `DAYS_EMPLOYED` (converted to negative days)
6. **Employer Type** (dropdown) → `ORGANIZATION_TYPE`
7. **Education Level** (dropdown) → `NAME_EDUCATION_TYPE`
8. **Gender** (M/F) → `CODE_GENDER`
9. **Existing Debt Ratio** (0-100%) → `BUREAU_DEBT_CREDIT_RATIO`
10. **Late Payment History** (0-100%) → `LATE_PAYMENT_RATE`

## How It Works

1. User fills in the application form
2. The app loads the trained LightGBM model
3. User inputs are mapped to model features
4. Missing features are filled with median (numeric) or mode (categorical) values from training data
5. Model predicts default probability
6. Decision is made: **APPROVED** if probability < 0.20, **DENIED** otherwise
7. Results display includes:
   - Risk score as percentage
   - Visual risk indicator (low/medium/high)
   - Approval/denial decision
   - Key factor explanations

## Technical Details

- **Model**: LightGBM binary classifier
- **Threshold**: 0.20 (20% default probability)
- **Features**: 132 total features (10 from user input, 122 from training data defaults)
- **Framework**: Streamlit
- **Backend**: Python with pandas, numpy, lightgbm

## Notes

- The model expects categorical columns to be of type 'category' (handled automatically)
- All features not provided by the user are filled with median/mode values from training data
- The app dynamically extracts categorical value options from the training data for accurate mapping
