import streamlit as st
import pandas as pd
import joblib
import os

# Define the list of features we'll accept from the user
FEATURES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "Neighborhood",
    "HouseStyle"
]

# Load the trained model
@st.cache_resource
def load_model():
    model_path = "model/house_price_model.pkl"
    if not os.path.exists(model_path):
        st.error("Trained model not found. Please train the model first.")
        return None
    return joblib.load(model_path)

# Load model features from training set
@st.cache_data
def get_model_features():
    df = pd.read_csv("datasets/processed/ames-train-clean.csv")
    X = pd.get_dummies(df.drop(columns=["SalePrice"]))
    return X.columns.tolist()

# Prepare user input for prediction
def prepare_input(user_input, model_features):
    df = pd.DataFrame([user_input])
    df = pd.get_dummies(df)

    # Add missing columns with 0
    missing_cols = []
    for col in model_features:
        if col not in df.columns:
            missing_cols.append(col)

    # Create a DataFrame with 0s for all missing columns
    missing_df = pd.DataFrame([[0] * len(missing_cols)], columns=missing_cols)

    # Combine input with missing columns
    df = pd.concat([df, missing_df], axis=1)

    # Reorder columns to match model
    df = df[model_features]


    # Reorder columns to match model input
    df = df[model_features]
    return df

# Streamlit app
st.title("🏠 House Price Prediction App")

st.write("Enter the house features below to predict its sale price:")

# Input fields
overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (GrLivArea)", min_value=100, max_value=5000, value=1500)
garage_cars = st.selectbox("Garage Capacity (GarageCars)", options=[0, 1, 2, 3, 4], index=2)
total_bsmt_sf = st.number_input("Total Basement SF (TotalBsmtSF)", min_value=0, max_value=3000, value=800)
full_bath = st.selectbox("Number of Full Bathrooms (FullBath)", options=[0, 1, 2, 3], index=1)
year_built = st.number_input("Year Built", min_value=1870, max_value=2025, value=1990)

neighborhood = st.selectbox("Neighborhood", options=[
    'CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel',
    'Somerst', 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer',
    'NridgHt', 'NAmes', 'SawyerW', 'IDOTRR', 'MeadowV',
    'Edwards', 'Timber', 'Gilbert', 'StoneBr', 'ClearCr',
    'NPkVill', 'Blmngtn', 'BrDale', 'SWISU', 'Blueste'])
house_style = st.selectbox("House Style", options=[
    '2Story', '1Story', '1.5Fin', '1.5Unf', 'SFoyer',
    'SLvl', '2.5Unf', '2.5Fin'])

# Submit button
if st.button("Predict Price"):
    input_data = {
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "GarageCars": garage_cars,
        "TotalBsmtSF": total_bsmt_sf,
        "FullBath": full_bath,
        "YearBuilt": year_built,
        "Neighborhood": neighborhood,
        "HouseStyle": house_style
    }

    model = load_model()
    model_features = get_model_features()
    prepared_input = prepare_input(input_data, model_features)

    if model:
        prediction = model.predict(prepared_input)[0]
        st.success(f"💰 Estimated Sale Price: ${prediction:,.2f}")
