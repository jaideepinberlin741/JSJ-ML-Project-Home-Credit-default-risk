import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import lightgbm as lgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Home Credit Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for bank-style design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: #e0e0e0;
        margin: 0.5rem 0 0 0;
    }
    .form-container {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .result-container {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-top: 2rem;
    }
    .approved {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
    }
    .denied {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
    }
    .risk-score {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
    }
    .risk-low {
        color: #11998e;
    }
    .risk-medium {
        color: #f39c12;
    }
    .risk-high {
        color: #e74c3c;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 5px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2a5298 0%, #3d6bb3 100%);
    }
    .sample-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
</style>
""", unsafe_allow_html=True)

def _load_with_pickle(model_path):
    """Fallback: Load using pickle directly"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_model():
    """Load the trained LightGBM model with multiple fallback strategies"""
    # Display current versions
    numpy_version = np.__version__
    joblib_version = joblib.__version__
    
    model_path = 'models/lightgbm_model.pkl'
    
    # Check if file exists
    import os
    if not os.path.exists(model_path):
        st.error(f"**Model file not found:** {model_path}")
        st.info("Ensure the model file exists in the project directory.")
        return None
    
    # Try multiple loading strategies
    strategies = [
        ("joblib.load (default)", lambda: joblib.load(model_path)),
        ("joblib.load (with mmap_mode)", lambda: joblib.load(model_path, mmap_mode='r')),
        ("pickle.load (compatibility)", lambda: _load_with_pickle(model_path)),
    ]
    
    last_error = None
    for strategy_name, load_func in strategies:
        try:
            model = load_func()
            if model is not None:
                # Only show success message if not the first (default) strategy
                if strategy_name != strategies[0][0]:
                    st.success(f"✅ Model loaded using: {strategy_name}")
                return model
        except Exception as e:
            last_error = e
            # Continue to next strategy
            continue
    
    # If all strategies failed, show detailed error
    if last_error:
        error_msg = str(last_error)
        
        if '_signature_descriptor' in error_msg or '_globals' in error_msg or 'numpy._core' in error_msg:
            st.error("""
            ## ❌ NumPy/Joblib Version Compatibility Issue
            
            The model was saved with a different version combination of NumPy and Joblib.
            
            **Current versions:**
            - NumPy: """ + numpy_version + """
            - Joblib: """ + joblib_version + """
            
            **Try these solutions (in order):**
            
            1. **Upgrade both packages (recommended):**
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
            
            **Important:** After installing, **restart the Streamlit app** completely.
            """)
        else:
            st.error(f"**Error loading model:** {last_error}")
            st.info("Try: `pip install --upgrade numpy joblib`")
    
    return None

@st.cache_data
def load_feature_data(model):
    """Get feature names from model and create feature structure without CSV"""
    try:
        # Get feature names from the LightGBM model
        # LightGBM models store feature names in different ways
        feature_names = None
        
        if hasattr(model, 'feature_name'):
            try:
                feature_names = model.feature_name()
            except:
                pass
        
        if not feature_names and hasattr(model, 'feature_name_'):
            feature_names = model.feature_name_
        
        if not feature_names and hasattr(model, 'model') and hasattr(model.model, 'feature_name'):
            try:
                feature_names = model.model.feature_name()
            except:
                pass
        
        # Fallback: use the feature names we know from the model file
        if not feature_names:
            feature_names = [
                'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
                'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE',
                'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
                'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION',
                'DAYS_ID_PUBLISH', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
                'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
                'REGION_RATING_CLIENT_W_CITY', 'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START',
                'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
                'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
                'ORGANIZATION_TYPE', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'OBS_30_CNT_SOCIAL_CIRCLE',
                'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
                'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4',
                'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8',
                'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
                'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
                'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
                'FLAG_DOCUMENT_21', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
                'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
                'AMT_REQ_CREDIT_BUREAU_YEAR', 'BUREAU_CREDIT_COUNT', 'BUREAU_CREDIT_SUM_MEAN',
                'BUREAU_CREDIT_SUM_SUM', 'BUREAU_CREDIT_SUM_MAX', 'BUREAU_CREDIT_SUM_MIN',
                'BUREAU_DEBT_SUM', 'BUREAU_DEBT_MEAN', 'BUREAU_DEBT_MAX', 'BUREAU_OVERDUE_MAX',
                'BUREAU_OVERDUE_MEAN', 'BUREAU_DAYS_OVERDUE_MAX', 'BUREAU_DAYS_CREDIT_MEAN',
                'BUREAU_DAYS_CREDIT_MIN', 'BUREAU_DAYS_CREDIT_MAX', 'BUREAU_PROLONG_SUM',
                'BUREAU_ANNUITY_MEAN', 'BUREAU_ANNUITY_SUM', 'BUREAU_ACTIVE', 'BUREAU_BAD_DEBT',
                'BUREAU_CLOSED', 'BUREAU_SOLD', 'BUREAU_DEBT_CREDIT_RATIO', 'BUREAU_ACTIVE_RATIO',
                'PREV_APP_COUNT', 'PREV_CREDIT_MEAN', 'PREV_CREDIT_SUM', 'PREV_CREDIT_MAX',
                'PREV_CREDIT_MIN', 'PREV_APPLICATION_MEAN', 'PREV_APPLICATION_SUM', 'PREV_ANNUITY_MEAN',
                'PREV_ANNUITY_MAX', 'PREV_DOWN_PAYMENT_MEAN', 'PREV_DOWN_PAYMENT_MAX',
                'PREV_CNT_PAYMENT_MEAN', 'PREV_CNT_PAYMENT_SUM', 'PREV_APPROVED', 'PREV_CANCELED',
                'PREV_REFUSED', 'PREV_UNUSED_OFFER', 'PREV_TYPE_CASH_LOANS', 'PREV_TYPE_CONSUMER_LOANS',
                'PREV_TYPE_REVOLVING_LOANS', 'PREV_TYPE_XNA', 'PREV_APPROVAL_RATIO', 'PREV_REFUSED_RATIO',
                'PREV_CREDIT_APPLICATION_RATIO', 'INST_COUNT', 'LATE_PAYMENT_RATE', 'AVG_PAYMENT_DELAY',
                'MAX_PAYMENT_DELAY', 'TOTAL_PAID', 'POS_LOAN_COUNT', 'POS_AVG_DPD', 'POS_MAX_DPD',
                'POS_AVG_DPD_DEF', 'CC_LOAN_COUNT', 'CC_AVG_BALANCE', 'CC_MAX_BALANCE', 'CC_AVG_DPD', 'CC_MAX_DPD'
            ]
        
        if not feature_names:
            st.error("Could not extract feature names from model")
            return None, None, None
        
        # Create a DataFrame structure with feature names
        feature_df = pd.DataFrame(columns=feature_names)
        
        # Define known categorical columns (from the model structure)
        known_categorical = [
            'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
            'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
            'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
            'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE'
        ]
        
        # Get categorical values (common values for dropdowns)
        categorical_values = {
            'NAME_EDUCATION_TYPE': [
                'Higher education',
                'Secondary / secondary special',
                'Incomplete higher',
                'Lower secondary',
                'Academic degree'
            ],
            'ORGANIZATION_TYPE': [
                'Business Entity Type 3',
                'Self-employed',
                'Government',
                'Other',
                'Business Entity Type 2',
                'Business Entity Type 1'
            ],
            'CODE_GENDER': ['M', 'F'],
            'NAME_CONTRACT_TYPE': ['Cash loans', 'Revolving loans'],
            'FLAG_OWN_CAR': ['Y', 'N'],
            'FLAG_OWN_REALTY': ['Y', 'N']
        }
        
        # Add any other categorical columns with empty lists (will use mode later)
        for col in known_categorical:
            if col not in categorical_values and col in feature_names:
                categorical_values[col] = []
        
        return feature_df, None, categorical_values
    except Exception as e:
        st.error(f"Error setting up feature data: {e}")
        return None, None, None

def compute_default_values(feature_df):
    """Compute default values for features (without CSV, use reasonable defaults)"""
    defaults = {}
    
    # Known categorical columns
    categorical_cols = [
        'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
        'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
        'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
        'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE'
    ]
    
    # Set defaults for all features
    for col in feature_df.columns:
        if col in categorical_cols:
            # Use common default values for categorical
            if col == 'CODE_GENDER':
                defaults[col] = 'M'
            elif col == 'NAME_EDUCATION_TYPE':
                defaults[col] = 'Higher education'
            elif col == 'ORGANIZATION_TYPE':
                defaults[col] = 'Business Entity Type 3'
            elif col == 'NAME_CONTRACT_TYPE':
                defaults[col] = 'Cash loans'
            elif col in ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
                defaults[col] = 'N'
            else:
                defaults[col] = 'XNA'  # Common missing value indicator
        else:
            # Use reasonable defaults for numeric features
            if 'AMT' in col:
                defaults[col] = 0.0
            elif 'DAYS' in col:
                defaults[col] = -1000.0  # Typical negative days value
            elif 'CNT' in col or 'COUNT' in col:
                defaults[col] = 0.0
            elif 'RATIO' in col or 'RATE' in col:
                defaults[col] = 0.0
            elif 'EXT_SOURCE' in col:
                defaults[col] = 0.5  # Normalized score, middle value
            elif col.startswith('FLAG_'):
                defaults[col] = 0.0
            else:
                defaults[col] = 0.0  # Default to 0 for unknown numeric
    
    return defaults

def create_feature_vector(user_inputs, feature_df, defaults, categorical_values):
    """Create feature vector from user inputs"""
    # Start with default values
    feature_vector = pd.DataFrame([defaults])
    
    # Map user inputs to features
    # 1. External Credit Score 2 -> EXT_SOURCE_2 (normalized 0-1)
    ext_source_2_score = user_inputs['ext_source_2']
    normalized_score_2 = (ext_source_2_score - 300) / (850 - 300)  # Normalize to 0-1
    feature_vector['EXT_SOURCE_2'] = normalized_score_2
    
    # 2. External Credit Score 3 -> EXT_SOURCE_3 (normalized 0-1)
    ext_source_3_score = user_inputs['ext_source_3']
    normalized_score_3 = (ext_source_3_score - 300) / (850 - 300)  # Normalize to 0-1
    feature_vector['EXT_SOURCE_3'] = normalized_score_3
    
    # 3. Loan Amount -> AMT_CREDIT
    feature_vector['AMT_CREDIT'] = user_inputs['loan_amount']
    
    # 4. Monthly Payment -> AMT_ANNUITY
    feature_vector['AMT_ANNUITY'] = user_inputs['monthly_payment']
    
    # 5. Age -> DAYS_BIRTH (negative days)
    age = user_inputs['age']
    birth_date = datetime.now() - timedelta(days=age*365.25)
    days_birth = -(datetime.now() - birth_date).days
    feature_vector['DAYS_BIRTH'] = days_birth
    
    # 6. Years at Current Job -> DAYS_EMPLOYED (negative days)
    years_employed = user_inputs['years_employed']
    if years_employed > 0:
        days_employed = -int(years_employed * 365.25)
    else:
        days_employed = 0
    feature_vector['DAYS_EMPLOYED'] = days_employed
    
    # 7. Employer Type -> ORGANIZATION_TYPE
    # Map to closest matching value from training data
    org_options = categorical_values.get('ORGANIZATION_TYPE', [])
    org_mapping = {
        'Business Entity': [x for x in org_options if 'Business' in x or 'Entity' in x],
        'Government': [x for x in org_options if 'Government' in x or 'gov' in x.lower()],
        'Self-employed': [x for x in org_options if 'Self' in x or 'self' in x.lower()],
        'Other': org_options
    }
    # Use first match or fallback to most common
    matched_orgs = org_mapping.get(user_inputs['employer_type'], org_options)
    feature_vector['ORGANIZATION_TYPE'] = matched_orgs[0] if matched_orgs else org_options[0] if org_options else 'Other'
    
    # 8. Education -> NAME_EDUCATION_TYPE
    # Map to exact match from training data
    edu_options = categorical_values.get('NAME_EDUCATION_TYPE', [])
    edu_input = user_inputs['education']
    # Try exact match first, then partial match
    if edu_input in edu_options:
        feature_vector['NAME_EDUCATION_TYPE'] = edu_input
    else:
        # Try to find closest match
        matched_edu = [x for x in edu_options if any(word in x for word in edu_input.split())]
        feature_vector['NAME_EDUCATION_TYPE'] = matched_edu[0] if matched_edu else edu_options[0] if edu_options else 'Higher education'
    
    # 9. Gender -> CODE_GENDER
    feature_vector['CODE_GENDER'] = user_inputs['gender']
    
    # 10. Existing Debt Ratio -> BUREAU_DEBT_CREDIT_RATIO
    feature_vector['BUREAU_DEBT_CREDIT_RATIO'] = user_inputs['debt_ratio'] / 100.0
    
    # 11. Late Payment History -> LATE_PAYMENT_RATE
    feature_vector['LATE_PAYMENT_RATE'] = user_inputs['late_payment_rate'] / 100.0
    
    # Ensure all columns from training data are present
    for col in feature_df.columns:
        if col not in feature_vector.columns:
            feature_vector[col] = defaults[col]
    
    # Reorder columns to match training data
    feature_vector = feature_vector[feature_df.columns]
    
    # Convert categorical columns to category dtype (required by LightGBM)
    categorical_cols = feature_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in feature_vector.columns:
            feature_vector[col] = feature_vector[col].astype('category')
    
    return feature_vector

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Home Credit Loan Approval System</h1>
        <p>Advanced Machine Learning Risk Assessment Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for threshold configuration
    with st.sidebar:
        st.header("⚙️ Approval Settings")
        threshold = st.slider(
            "Approval Threshold (%)",
            min_value=5,
            max_value=50,
            value=20,
            step=1,
            help="Applications with default probability below this threshold will be approved"
        )
        threshold_decimal = threshold / 100.0
        st.info(f"**Current threshold:** {threshold}%\n\nApplications with risk score **< {threshold}%** will be **APPROVED**")
    
    # Load model first
    model = load_model()
    if model is None:
        st.error("Failed to load model. Please check file paths.")
        return
    
    # Load feature data (now uses model to get feature names)
    feature_df, full_df, categorical_values = load_feature_data(model)
    
    if feature_df is None or categorical_values is None:
        st.error("Failed to set up feature data.")
        return
    
    # Compute default values
    defaults = compute_default_values(feature_df)
    
    # Initialize sample values in session state
    if 'sample_loaded' not in st.session_state:
        st.session_state.sample_loaded = False
    
    # Get education options from training data
    edu_options = categorical_values.get('NAME_EDUCATION_TYPE', [
        'Higher education', 
        'Secondary / secondary special', 
        'Incomplete higher', 
        'Lower secondary'
    ])
    
    # Sample applicant button (before form)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📝 Load Sample Applicant", use_container_width=True):
            st.session_state.sample_loaded = True
            st.session_state.sample_ext_source_2 = 720
            st.session_state.sample_ext_source_3 = 750
            st.session_state.sample_loan_amount = 150000.0
            st.session_state.sample_monthly_payment = 4000.0
            st.session_state.sample_age = 42
            st.session_state.sample_years_employed = 8.0
            st.session_state.sample_employer_type = 'Business Entity'
            st.session_state.sample_education = edu_options[0] if edu_options else 'Higher education'
            st.session_state.sample_gender = 'M'
            st.session_state.sample_debt_ratio = 25
            st.session_state.sample_late_payment_rate = 3
            # Force widget refresh by updating a counter
            st.session_state.widget_refresh = st.session_state.get('widget_refresh', 0) + 1
            st.rerun()
    
    # Show sample info if loaded
    if st.session_state.sample_loaded:
        st.success("✅ Sample applicant data loaded! Form values have been updated below.")
        with st.expander("📋 View Sample Applicant Profile"):
            st.markdown("""
            **Sample Applicant Profile:**
            - **External Credit Score 2:** 720
            - **External Credit Score 3:** 750
            - **Loan Amount:** $150,000
            - **Monthly Payment:** $4,000
            - **Age:** 42 years
            - **Years Employed:** 8 years
            - **Employer:** Business Entity
            - **Education:** Higher education
            - **Gender:** Male
            - **Debt Ratio:** 25%
            - **Late Payment Rate:** 3%
            """)
    
    # Form container
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Applicant Information")
        
        # Use sample values if loaded, otherwise use defaults
        # Use widget keys that include refresh counter to force update
        refresh_key = st.session_state.get('widget_refresh', 0)
        
        # External Credit Score 2
        default_ext_2 = st.session_state.get('sample_ext_source_2', 650) if st.session_state.sample_loaded else 650
        ext_source_2 = st.slider(
            "External Credit Score 2",
            min_value=300,
            max_value=850,
            value=default_ext_2,
            step=10,
            help="External credit score from bureau #2 (300-850)",
            key=f'ext_source_2_{refresh_key}'
        )
        
        # External Credit Score 3
        default_ext_3 = st.session_state.get('sample_ext_source_3', 650) if st.session_state.sample_loaded else 650
        ext_source_3 = st.slider(
            "External Credit Score 3",
            min_value=300,
            max_value=850,
            value=default_ext_3,
            step=10,
            help="External credit score from bureau #3 (300-850)",
            key=f'ext_source_3_{refresh_key}'
        )
        
        default_loan = st.session_state.get('sample_loan_amount', 100000.0) if st.session_state.sample_loaded else 100000.0
        loan_amount = st.number_input(
            "Loan Amount Requested ($)",
            min_value=0.0,
            value=default_loan,
            step=1000.0,
            format="%.0f",
            key=f'loan_amount_{refresh_key}'
        )
        
        default_payment = st.session_state.get('sample_monthly_payment', 3000.0) if st.session_state.sample_loaded else 3000.0
        monthly_payment = st.number_input(
            "Monthly Payment ($)",
            min_value=0.0,
            value=default_payment,
            step=100.0,
            format="%.0f",
            key=f'monthly_payment_{refresh_key}'
        )
        
        default_age = st.session_state.get('sample_age', 35) if st.session_state.sample_loaded else 35
        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=default_age,
            step=1,
            key=f'age_{refresh_key}'
        )
        
        default_years = st.session_state.get('sample_years_employed', 5.0) if st.session_state.sample_loaded else 5.0
        years_employed = st.number_input(
            "Years at Current Job",
            min_value=0.0,
            max_value=50.0,
            value=default_years,
            step=0.5,
            format="%.1f",
            key=f'years_employed_{refresh_key}'
        )
    
    with col2:
        st.subheader("📊 Additional Details")
        
        # Get organization types (simplified for user, will map internally)
        employer_options = ['Business Entity', 'Government', 'Self-employed', 'Other']
        default_employer_idx = employer_options.index(st.session_state.get('sample_employer_type', 'Business Entity')) if st.session_state.sample_loaded and st.session_state.get('sample_employer_type') in employer_options else 0
        employer_type = st.selectbox(
            "Employer Type",
            options=employer_options,
            index=default_employer_idx,
            key=f'employer_type_{refresh_key}'
        )
        
        # Use first 4 options for simplicity
        default_edu = st.session_state.get('sample_education', edu_options[0]) if st.session_state.sample_loaded else edu_options[0]
        edu_display_options = edu_options[:4] if len(edu_options) >= 4 else edu_options
        default_edu_idx = edu_display_options.index(default_edu) if default_edu in edu_display_options else 0
        education = st.selectbox(
            "Education Level",
            options=edu_display_options,
            index=min(default_edu_idx, len(edu_display_options) - 1),
            key=f'education_{refresh_key}'
        )
        
        gender_options = ['M', 'F']
        default_gender_idx = gender_options.index(st.session_state.get('sample_gender', 'M')) if st.session_state.sample_loaded and st.session_state.get('sample_gender') in gender_options else 0
        gender = st.selectbox(
            "Gender",
            options=gender_options,
            index=default_gender_idx,
            key=f'gender_{refresh_key}'
        )
        
        default_debt = st.session_state.get('sample_debt_ratio', 30) if st.session_state.sample_loaded else 30
        debt_ratio = st.slider(
            "Existing Debt Ratio (%)",
            min_value=0,
            max_value=100,
            value=default_debt,
            step=1,
            help="Percentage of income used for debt payments",
            key=f'debt_ratio_{refresh_key}'
        )
        
        default_late = st.session_state.get('sample_late_payment_rate', 5) if st.session_state.sample_loaded else 5
        late_payment_rate = st.slider(
            "Late Payment History (%)",
            min_value=0,
            max_value=100,
            value=default_late,
            step=1,
            help="Percentage of payments that were late",
            key=f'late_payment_rate_{refresh_key}'
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    submit_button = st.button("🔍 Evaluate Application", use_container_width=True, type="primary")
    
    if submit_button:
        # Collect user inputs
        user_inputs = {
            'ext_source_2': ext_source_2,
            'ext_source_3': ext_source_3,
            'loan_amount': loan_amount,
            'monthly_payment': monthly_payment,
            'age': age,
            'years_employed': years_employed,
            'employer_type': employer_type,
            'education': education,
            'gender': gender,
            'debt_ratio': debt_ratio,
            'late_payment_rate': late_payment_rate
        }
        
        # Create feature vector
        with st.spinner("Processing application..."):
            feature_vector = create_feature_vector(user_inputs, feature_df, defaults, categorical_values)
            
            # Make prediction
            try:
                prediction_proba = model.predict(feature_vector, num_iteration=model.best_iteration)[0]
                risk_score = prediction_proba * 100
                
                # Decision threshold (from sidebar)
                approved = prediction_proba < threshold_decimal
                
                # Display results
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                
                # Risk Score
                if risk_score < 20:
                    risk_class = "risk-low"
                    risk_label = "Low Risk"
                elif risk_score < 40:
                    risk_class = "risk-medium"
                    risk_label = "Medium Risk"
                else:
                    risk_class = "risk-high"
                    risk_label = "High Risk"
                
                st.markdown(f"""
                <div class="risk-score {risk_class}">
                    Risk Score: {risk_score:.1f}%
                    <div style="font-size: 1rem; margin-top: 0.5rem;">{risk_label}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show threshold info
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: #f0f0f0; border-radius: 5px; margin: 1rem 0;">
                    <strong>Approval Threshold:</strong> {threshold}%<br>
                    <small>Risk Score: {risk_score:.1f}% | Threshold: {threshold}%</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Decision
                if approved:
                    st.markdown('<div class="approved">✅ APPROVED</div>', unsafe_allow_html=True)
                    st.info(f"✅ **Approved** because risk score ({risk_score:.1f}%) is below the threshold ({threshold}%)")
                else:
                    st.markdown('<div class="denied">❌ DENIED</div>', unsafe_allow_html=True)
                    st.warning(f"❌ **Denied** because risk score ({risk_score:.1f}%) exceeds the threshold ({threshold}%)")
                
                # Explanation
                st.markdown("---")
                st.subheader("📊 Key Factors")
                
                factors = []
                # External Credit Score 2
                if ext_source_2 < 600:
                    factors.append(f"⚠️ **Low External Score 2** ({ext_source_2}): Below average creditworthiness from bureau #2")
                elif ext_source_2 >= 750:
                    factors.append(f"✅ **Excellent External Score 2** ({ext_source_2}): Strong credit history from bureau #2")
                else:
                    factors.append(f"📊 **External Score 2** ({ext_source_2}): Moderate credit profile from bureau #2")
                
                # External Credit Score 3
                if ext_source_3 < 600:
                    factors.append(f"⚠️ **Low External Score 3** ({ext_source_3}): Below average creditworthiness from bureau #3")
                elif ext_source_3 >= 750:
                    factors.append(f"✅ **Excellent External Score 3** ({ext_source_3}): Strong credit history from bureau #3")
                else:
                    factors.append(f"📊 **External Score 3** ({ext_source_3}): Moderate credit profile from bureau #3")
                
                if debt_ratio > 40:
                    factors.append(f"⚠️ **High Debt Ratio** ({debt_ratio}%): Significant existing debt burden")
                elif debt_ratio < 20:
                    factors.append(f"✅ **Low Debt Ratio** ({debt_ratio}%): Minimal existing debt")
                else:
                    factors.append(f"📊 **Debt Ratio** ({debt_ratio}%): Moderate debt level")
                
                if late_payment_rate > 20:
                    factors.append(f"⚠️ **Poor Payment History** ({late_payment_rate}%): Frequent late payments")
                elif late_payment_rate < 5:
                    factors.append(f"✅ **Good Payment History** ({late_payment_rate}%): Reliable payment behavior")
                else:
                    factors.append(f"📊 **Payment History** ({late_payment_rate}%): Some late payments")
                
                for factor in factors:
                    st.markdown(factor)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.exception(e)

if __name__ == "__main__":
    main()
