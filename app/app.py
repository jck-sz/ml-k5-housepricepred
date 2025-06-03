import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.features.build_features import engineer_features
from src.data_preprocessing.preprocess import clean_data

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
        'Neighborhood': sorted(df['Neighborhood'].dropna().unique())
    }

# Prepare user input for prediction
def prepare_input_for_prediction(user_input, model_features):
    # Create DataFrame from user input
    df = pd.DataFrame([user_input])
    
    # Add any missing columns with reasonable defaults
    # These are columns needed for feature engineering but not shown in UI
    if 'YrSold' not in df.columns:
        df['YrSold'] = 2010  # Default sale year
    if 'MoSold' not in df.columns:
        df['MoSold'] = 6     # Default sale month
    
    # Clean data (handle any remaining missing values)
    df = clean_data(df)
    
    # Engineer features (this adds our 10 new features)
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
    st.sidebar.header("🔝 Top Important Features")
    if 'top_features' in metadata:
        for i, feat in enumerate(metadata['top_features'][:5]):
            st.sidebar.text(f"{i+1}. {feat['feature']}: {feat['importance']:.3f}")

# Main content
st.write("Enter the house characteristics below to predict its sale price:")
st.write("Due to the dataset used, the prediction is most accurate for houses in the Ames, Iowa area.")
st.write("Unfortunately, the surface area is measured in Freedom Units, commonly known as Square Feet, so you might have to use a calculator.")

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
                              index=0)

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
        bsmt_half_bath = st.selectbox("Basement Half Bathrooms", options=[0, 1], index=0)
        fireplaces = st.selectbox("Number of Fireplaces", options=[0, 1, 2, 3], index=0)
        wood_deck_sf = st.number_input("Wood Deck Area (sq ft)", min_value=0, max_value=1000, value=0)
        open_porch_sf = st.number_input("Open Porch Area (sq ft)", min_value=0, max_value=500, value=0)
        garage_area = st.number_input("Garage Area (sq ft)", min_value=0, max_value=1500, value=500)

# Add some spacing
st.markdown("---")

# Show calculated features
with st.expander("🔧 See Calculated Features"):
    total_sf = total_bsmt_sf + first_flr_sf + second_flr_sf
    total_bathrooms = full_bath + 0.5 * half_bath + bsmt_full_bath + 0.5 * bsmt_half_bath
    house_age = 2010 - year_built  # Assuming 2010 sale year
    quality_score = overall_qual * overall_cond
    garage_capacity = garage_cars + (garage_area / 200)
    recent_remodel = 1 if (2010 - year_remod_add) < 10 else 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Total Square Footage:** {total_sf:,} sq ft")
        st.write(f"**Total Bathrooms:** {total_bathrooms}")
        st.write(f"**House Age:** {house_age} years")
        st.write(f"**Quality Score:** {quality_score}")
        st.write(f"**Quality × Living Area:** {overall_qual * gr_liv_area:,}")
    with col2:
        st.write(f"**Garage Capacity Score:** {garage_capacity:.2f}")
        st.write(f"**Has Basement:** {'Yes' if total_bsmt_sf > 0 else 'No'}")
        st.write(f"**Has Second Floor:** {'Yes' if second_flr_sf > 0 else 'No'}")
        st.write(f"**Recently Remodeled:** {'Yes' if recent_remodel else 'No'}")
        st.write(f"**Lot Area (log):** {np.log1p(lot_area):.2f}")

# Prediction button
if st.button("🎯 Predict House Price", type="primary"):
    if model:
        # Prepare input data - include all the fields needed for feature engineering
        input_data = {
            # Size features
            'TotalBsmtSF': total_bsmt_sf,
            '1stFlrSF': first_flr_sf,
            '2ndFlrSF': second_flr_sf,
            'GrLivArea': gr_liv_area,
            'LotArea': lot_area,
            
            # Quality features
            'OverallQual': overall_qual,
            'OverallCond': overall_cond,
            
            # Year features
            'YearBuilt': year_built,
            'YearRemodAdd': year_remod_add,
            'YrSold': 2010,  # Default sale year
            
            # Bathroom features
            'FullBath': full_bath,
            'HalfBath': half_bath,
            'BsmtFullBath': bsmt_full_bath,
            'BsmtHalfBath': bsmt_half_bath,
            
            # Garage features
            'GarageCars': garage_cars,
            'GarageArea': garage_area,
            
            # Categorical features
            'Neighborhood': neighborhood,
            
            # Additional features that might be needed
            'BedroomAbvGr': bedroom_abvgr,
            'Fireplaces': fireplaces,
            'WoodDeckSF': wood_deck_sf,
            'OpenPorchSF': open_porch_sf,
            
            # Default values for other common categorical features
            'MSZoning': 'RL',
            'Street': 'Pave',
            'LotShape': 'Reg',
            'LandContour': 'Lvl',
            'LotConfig': 'Inside',
            'LandSlope': 'Gtl',
            'BldgType': '1Fam',
            'HouseStyle': '1Story',
            'RoofStyle': 'Gable',
            'ExterQual': 'TA',
            'ExterCond': 'TA',
            'Foundation': 'PConc',
            'BsmtQual': 'TA' if total_bsmt_sf > 0 else 'NA',
            'BsmtCond': 'TA' if total_bsmt_sf > 0 else 'NA',
            'BsmtExposure': 'No' if total_bsmt_sf > 0 else 'NA',
            'BsmtFinType1': 'Unf' if total_bsmt_sf > 0 else 'NA',
            'Heating': 'GasA',
            'HeatingQC': 'Ex',
            'CentralAir': 'Y',
            'Electrical': 'SBrkr',
            'KitchenQual': 'TA',
            'Functional': 'Typ',
            'GarageType': 'Attchd' if garage_cars > 0 else 'NA',
            'GarageFinish': 'Unf' if garage_cars > 0 else 'NA',
            'GarageQual': 'TA' if garage_cars > 0 else 'NA',
            'GarageCond': 'TA' if garage_cars > 0 else 'NA',
            'PavedDrive': 'Y',
            'SaleType': 'WD',
            'SaleCondition': 'Normal',
            'MSSubClass': 20,  # Default to 1-story
            'Utilities': 'AllPub',
            'Condition1': 'Norm',
            'Condition2': 'Norm',
            'RoofMatl': 'CompShg',
            'Exterior1st': 'VinylSd',
            'Exterior2nd': 'VinylSd',
            'MasVnrType': 'None',
            'MasVnrArea': 0,
            'BsmtFinSF1': 0,
            'BsmtFinType2': 'Unf' if total_bsmt_sf > 0 else 'NA',
            'BsmtFinSF2': 0,
            'BsmtUnfSF': total_bsmt_sf,
            'LowQualFinSF': 0,
            'KitchenAbvGr': 1,
            'TotRmsAbvGrd': 7,
            'GarageYrBlt': year_built,
            'EnclosedPorch': 0,
            '3SsnPorch': 0,
            'ScreenPorch': 0,
            'PoolArea': 0,
            'PoolQC': 'NA',
            'Fence': 'NA',
            'MiscFeature': 'NA',
            'MiscVal': 0,
            'MoSold': 6,  # Default to June
            'Alley': 'NA',
            'LotFrontage': 65,  # Default value
            'FireplaceQu': 'Gd' if fireplaces > 0 else 'NA'
        }
        
        # Get model features
        model_features = get_model_features()
        
        # Prepare input
        with st.spinner('Calculating prediction...'):
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

# Footer
st.markdown("---")
st.markdown(
    "*Disclaimer of Grand Prognostication: The aforementioned predictive output, herein referred to as the 'Estimation Artifact,' is derived via arcane computational rituals involving historical datasets and algorithmic sorcery commonly labeled 'machine learning.'*\n\n"
    "*By proceeding to view or otherwise engage with said Estimation Artifact, the user (henceforth 'You, the Valued Price Enthusiast') acknowledges, accepts, and spiritually aligns with the cosmic truth that actual housing prices may diverge wildly, chaotically, and with reckless abandon from any numbers herein conjured—due to, but not limited to, market whimsy, acts of bureaucracy, lunar phases, interest rate voodoo, and the capricious decisions of financial deities.*"
)