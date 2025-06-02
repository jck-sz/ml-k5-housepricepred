import streamlit as st
import pandas as pd
import joblib
import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.features.build_features import engineer_features
from src.data_preprocessing.preprocess import clean_data

# Define the list of features we'll accept from the user
# Expanded to include more important features
REQUIRED_FEATURES = [
    "OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", 
    "FullBath", "YearBuilt", "Neighborhood", "HouseStyle"
]

ADDITIONAL_FEATURES = [
    "OverallCond", "LotArea", "BedroomAbvGr", "YearRemodAdd",
    "1stFlrSF", "2ndFlrSF", "BsmtFullBath", "HalfBath",
    "Fireplaces", "WoodDeckSF", "OpenPorchSF", "GarageArea"
]

# Load the trained model
@st.cache_resource
def load_model():
    model_path = "model/house_price_model.pkl"
    if not os.path.exists(model_path):
        st.error("Trained model not found. Please train the model first.")
        return None
    return joblib.load(model_path)

# Load model metadata
@st.cache_data
def load_metadata():
    metadata_path = "model/model_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None

# Load model features from training set
@st.cache_data
def get_model_features():
    df = pd.read_csv("datasets/processed/ames-train-featured.csv")
    X = pd.get_dummies(df.drop(columns=["SalePrice"]))
    return X.columns.tolist()

# Get unique values for categorical features
@st.cache_data
def get_categorical_values():
    df = pd.read_csv("datasets/ames-train.csv")
    return {
        'Neighborhood': sorted(df['Neighborhood'].dropna().unique()),
        'HouseStyle': sorted(df['HouseStyle'].dropna().unique())
    }

# Prepare user input for prediction
def prepare_input_for_prediction(user_input, model_features):
    # Create DataFrame from user input
    df = pd.DataFrame([user_input])
    
    # Fill missing numerical features with reasonable defaults
    numerical_defaults = {
        'LotFrontage': 65,
        'MasVnrArea': 0,
        'BsmtFinSF1': 0,
        'BsmtFinSF2': 0,
        'BsmtUnfSF': 0,
        'BsmtHalfBath': 0,
        'KitchenAbvGr': 1,
        'TotRmsAbvGrd': 7,
        'GarageYrBlt': user_input.get('YearBuilt', 2000),
        'EnclosedPorch': 0,
        '3SsnPorch': 0,
        'ScreenPorch': 0,
        'PoolArea': 0,
        'MiscVal': 0,
        'LowQualFinSF': 0
    }
    
    # Fill missing values
    for col, default_val in numerical_defaults.items():
        if col not in df.columns:
            df[col] = default_val
    
    # Add required categorical defaults
    categorical_defaults = {
        'MSZoning': 'RL',
        'Street': 'Pave',
        'LotShape': 'Reg',
        'LandContour': 'Lvl',
        'Utilities': 'AllPub',
        'LotConfig': 'Inside',
        'LandSlope': 'Gtl',
        'Condition1': 'Norm',
        'Condition2': 'Norm',
        'BldgType': '1Fam',
        'RoofStyle': 'Gable',
        'RoofMatl': 'CompShg',
        'ExterQual': 'TA',
        'ExterCond': 'TA',
        'Foundation': 'PConc',
        'BsmtQual': 'TA',
        'BsmtCond': 'TA',
        'BsmtExposure': 'No',
        'BsmtFinType1': 'Unf',
        'BsmtFinType2': 'Unf',
        'Heating': 'GasA',
        'HeatingQC': 'Ex',
        'CentralAir': 'Y',
        'Electrical': 'SBrkr',
        'KitchenQual': 'TA',
        'Functional': 'Typ',
        'GarageType': 'Attchd',
        'GarageFinish': 'Unf',
        'GarageQual': 'TA',
        'GarageCond': 'TA',
        'PavedDrive': 'Y',
        'SaleType': 'WD',
        'SaleCondition': 'Normal'
    }
    
    for col, default_val in categorical_defaults.items():
        if col not in df.columns:
            df[col] = default_val
    
    # Add YrSold and MoSold (current date)
    df['YrSold'] = 2010  # Default sale year
    df['MoSold'] = 6     # Default sale month
    
    # Clean data (handle any remaining missing values)
    df = clean_data(df)
    
    # Engineer features
    df = engineer_features(df, is_training=False)
    
    # One-hot encode
    df = pd.get_dummies(df)
    
    # Align with model features
    missing_cols = set(model_features) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    
    df = df[model_features]
    
    return df

# Streamlit app
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

st.title("🏠 House Price Prediction App")
st.markdown("*Powered by Machine Learning with Feature Engineering*")

# Load model and metadata
model = load_model()
metadata = load_metadata()
cat_values = get_categorical_values()

if model and metadata:
    # Display model info in sidebar
    st.sidebar.header("📊 Model Information")
    st.sidebar.info(f"**Model Type:** {metadata['model_type']}")
    st.sidebar.info(f"**Training Date:** {metadata['training_date']}")
    st.sidebar.info(f"**Validation RMSE:** ${metadata['metrics']['validation']['rmse']:,.2f}")
    st.sidebar.info(f"**R² Score:** {metadata['metrics']['validation']['r2']:.3f}")
    
    # Show top features
    st.sidebar.header("🔝 Top 5 Important Features")
    for i, feat in enumerate(metadata['top_features'][:5]):
        st.sidebar.text(f"{i+1}. {feat['feature']}: {feat['importance']:.3f}")

# Main content
st.write("Enter the house features below to predict its sale price:")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Primary Features")
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 7, 
                            help="Rates the overall material and finish of the house")
    gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", 
                                 min_value=300, max_value=6000, value=1500, step=50)
    garage_cars = st.selectbox("Garage Capacity (cars)", 
                              options=[0, 1, 2, 3, 4], index=2)
    total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", 
                                   min_value=0, max_value=3000, value=1000, step=50)
    full_bath = st.selectbox("Full Bathrooms", 
                           options=[0, 1, 2, 3, 4], index=2)
    year_built = st.number_input("Year Built", 
                               min_value=1870, max_value=2025, value=1990)
    neighborhood = st.selectbox("Neighborhood", 
                              options=cat_values['Neighborhood'],
                              index=cat_values['Neighborhood'].index('NAmes') if 'NAmes' in cat_values['Neighborhood'] else 0)
    house_style = st.selectbox("House Style", 
                             options=cat_values['HouseStyle'],
                             index=cat_values['HouseStyle'].index('1Story') if '1Story' in cat_values['HouseStyle'] else 0)

with col2:
    st.subheader("Additional Features")
    overall_cond = st.slider("Overall Condition (1-10)", 1, 10, 5,
                           help="Rates the overall condition of the house")
    lot_area = st.number_input("Lot Area (sq ft)", 
                             min_value=1000, max_value=50000, value=10000, step=500)
    bedroom_abvgr = st.selectbox("Bedrooms Above Ground", 
                               options=[0, 1, 2, 3, 4, 5, 6], index=3)
    year_remod_add = st.number_input("Year Remodeled", 
                                   min_value=1870, max_value=2025, value=year_built)
    first_flr_sf = st.number_input("First Floor Area (sq ft)", 
                                 min_value=300, max_value=3000, value=1000, step=50)
    second_flr_sf = st.number_input("Second Floor Area (sq ft)", 
                                  min_value=0, max_value=2000, value=0, step=50)
    
    # Expandable section for more features
    with st.expander("More Features (Optional)"):
        bsmt_full_bath = st.selectbox("Basement Full Bathrooms", options=[0, 1, 2], index=0)
        half_bath = st.selectbox("Half Bathrooms", options=[0, 1, 2], index=0)
        fireplaces = st.selectbox("Number of Fireplaces", options=[0, 1, 2, 3], index=0)
        wood_deck_sf = st.number_input("Wood Deck Area (sq ft)", min_value=0, max_value=1000, value=0)
        open_porch_sf = st.number_input("Open Porch Area (sq ft)", min_value=0, max_value=500, value=0)
        garage_area = st.number_input("Garage Area (sq ft)", min_value=0, max_value=1500, value=500)

# Add some spacing
st.markdown("---")

# Prediction button
if st.button("🎯 Predict House Price", type="primary"):
    if model:
        # Prepare input data
        input_data = {
            # Primary features
            "OverallQual": overall_qual,
            "GrLivArea": gr_liv_area,
            "GarageCars": garage_cars,
            "TotalBsmtSF": total_bsmt_sf,
            "FullBath": full_bath,
            "YearBuilt": year_built,
            "Neighborhood": neighborhood,
            "HouseStyle": house_style,
            # Additional features
            "OverallCond": overall_cond,
            "LotArea": lot_area,
            "BedroomAbvGr": bedroom_abvgr,
            "YearRemodAdd": year_remod_add,
            "1stFlrSF": first_flr_sf,
            "2ndFlrSF": second_flr_sf,
            "BsmtFullBath": bsmt_full_bath,
            "HalfBath": half_bath,
            "Fireplaces": fireplaces,
            "WoodDeckSF": wood_deck_sf,
            "OpenPorchSF": open_porch_sf,
            "GarageArea": garage_area,
            # Add calculated MSSubClass based on house style
            "MSSubClass": 60 if '2Story' in house_style else 20,
            "KitchenAbvGr": 1,
            "TotRmsAbvGrd": 7
        }
        
        # Get model features
        model_features = get_model_features()
        
        # Prepare input
        prepared_input = prepare_input_for_prediction(input_data, model_features)
        
        # Make prediction
        prediction = model.predict(prepared_input)[0]
        
        # Display results
        st.success("✅ Prediction Complete!")
        
        # Create three columns for results
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.metric(label="💰 Predicted Price", value=f"${prediction:,.0f}")
        
        with res_col2:
            price_per_sqft = prediction / gr_liv_area
            st.metric(label="📏 Price per Sq Ft", value=f"${price_per_sqft:.2f}")
        
        with res_col3:
            # Confidence interval (rough estimate based on model RMSE)
            if metadata:
                rmse = metadata['metrics']['validation']['rmse']
                lower_bound = max(0, prediction - rmse)
                upper_bound = prediction + rmse
                st.metric(label="📊 Confidence Range", 
                         value=f"${lower_bound:,.0f} - ${upper_bound:,.0f}")
        
        # Feature summary
        st.markdown("### 📋 House Summary")
        col_sum1, col_sum2 = st.columns(2)
        
        with col_sum1:
            st.write(f"**Total Square Footage:** {first_flr_sf + second_flr_sf + total_bsmt_sf:,} sq ft")
            st.write(f"**House Age:** {2010 - year_built} years")
            st.write(f"**Quality Score:** {overall_qual * overall_cond}")
        
        with col_sum2:
            st.write(f"**Total Bathrooms:** {full_bath + 0.5 * half_bath + bsmt_full_bath}")
            st.write(f"**Garage Space:** {garage_cars} cars ({garage_area} sq ft)")
            st.write(f"**Outdoor Space:** {wood_deck_sf + open_porch_sf} sq ft")

# Footer
st.markdown("---")
st.markdown("*Note: This prediction is based on historical data and machine learning models. "
           "Actual prices may vary based on market conditions and other factors.*")