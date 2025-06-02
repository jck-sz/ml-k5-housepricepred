import streamlit as st
import pandas as pd
import joblib
import os

# Rozszerzona lista cech do predykcji
FEATURES = [
    "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
    "GrLivArea", "TotalBsmtSF", "BsmtFinSF1", "GarageCars",
    "GarageArea", "FullBath", "HalfBath", "Bedroom",
    "TotRmsAbvGrd", "Fireplaces", "KitchenQual", "ExterQual",
    "HeatingQC", "Neighborhood", "HouseStyle", "GarageFinish",
    "MasVnrArea", "LotArea", "PavedDrive"
]

# Wczytaj model
@st.cache_resource
def load_model():
    model_path = "model/house_price_model.pkl"
    if not os.path.exists(model_path):
        st.error("Trained model not found. Please train the model first.")
        return None
    return joblib.load(model_path)

# Wczytaj cechy z modelu
@st.cache_data
def get_model_features():
    df = pd.read_csv("datasets/processed/ames-train-clean.csv")
    X = pd.get_dummies(df.drop(columns=["SalePrice"]))
    return X.columns.tolist()

# Przygotuj dane wejściowe
def prepare_input(user_input, model_features):
    df = pd.DataFrame([user_input])
    df = pd.get_dummies(df)
    missing_cols = [col for col in model_features if col not in df.columns]
    missing_df = pd.DataFrame([[0]*len(missing_cols)], columns=missing_cols)
    df = pd.concat([df, missing_df], axis=1)
    df = df[model_features]
    return df

# Streamlit app
# UI aplikacji
st.title("🏠 House Price Prediction App")

st.write("Wprowadź dane nieruchomości, by oszacować cenę:")

# Pola formularza
overall_qual = st.slider("Jakość wykończenia (1-10)", 1, 10, 5)
overall_cond = st.slider("Stan techniczny (1-10)", 1, 10, 5)
year_built = st.number_input("Rok budowy", 1870, 2025, 1990)
year_remod = st.number_input("Rok remontu", 1870, 2025, 1995)
gr_liv_area = st.number_input("Powierzchnia mieszkalna (GrLivArea)", 100, 5000, 1500)
total_bsmt_sf = st.number_input("Powierzchnia piwnicy", 0, 3000, 800)
bsmt_fin_sf1 = st.number_input("Wykończona część piwnicy", 0, 3000, 500)
garage_cars = st.selectbox("Liczba miejsc w garażu", [0, 1, 2, 3, 4], index=2)
garage_area = st.number_input("Powierzchnia garażu", 0, 1500, 500)
full_bath = st.selectbox("Pełne łazienki", [0, 1, 2, 3], index=1)
half_bath = st.selectbox("Toalety", [0, 1, 2], index=0)
bedrooms = st.number_input("Liczba sypialni", 0, 10, 3)
rooms_total = st.number_input("Liczba pokoi (bez łazienek)", 1, 20, 7)
fireplaces = st.selectbox("Kominki", [0, 1, 2, 3], index=1)
kitchen_qual = st.selectbox("Jakość kuchni", ["Ex", "Gd", "TA", "Fa", "Po"], index=2)
exterior_qual = st.selectbox("Jakość elewacji", ["Ex", "Gd", "TA", "Fa", "Po"], index=2)
heating_qc = st.selectbox("Jakość ogrzewania", ["Ex", "Gd", "TA", "Fa", "Po"], index=2)
neighborhood = st.selectbox("Dzielnica", [
    'CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel',
    'Somerst', 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer',
    'NridgHt', 'NAmes', 'SawyerW', 'IDOTRR', 'MeadowV',
    'Edwards', 'Timber', 'Gilbert', 'StoneBr', 'ClearCr',
    'NPkVill', 'Blmngtn', 'BrDale', 'SWISU', 'Blueste'
])
house_style = st.selectbox("Styl domu", [
    '2Story', '1Story', '1.5Fin', '1.5Unf', 'SFoyer',
    'SLvl', '2.5Unf', '2.5Fin'
])
garage_finish = st.selectbox("Wykończenie garażu", ["Fin", "RFn", "Unf", "NA"], index=2)
mas_vnr_area = st.number_input("Powierzchnia elewacji z kamienia/cegły", 0, 1000, 0)
lot_area = st.number_input("Powierzchnia działki", 1000, 100000, 8000)
paved_drive = st.selectbox("Utwardzony podjazd", ["Y", "P", "N"], index=0)

# Przycisk predykcji
if st.button("🔮 Oblicz cenę"):
    input_data = {
        "OverallQual": overall_qual,
        "OverallCond": overall_cond,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remod,
        "GrLivArea": gr_liv_area,
        "TotalBsmtSF": total_bsmt_sf,
        "BsmtFinSF1": bsmt_fin_sf1,
        "GarageCars": garage_cars,
        "GarageArea": garage_area,
        "FullBath": full_bath,
        "HalfBath": half_bath,
        "Bedroom": bedrooms,
        "TotRmsAbvGrd": rooms_total,
        "Fireplaces": fireplaces,
        "KitchenQual": kitchen_qual,
        "ExterQual": exterior_qual,
        "HeatingQC": heating_qc,
        "Neighborhood": neighborhood,
        "HouseStyle": house_style,
        "GarageFinish": garage_finish,
        "MasVnrArea": mas_vnr_area,
        "LotArea": lot_area,
        "PavedDrive": paved_drive
    }

    model = load_model()
    model_features = get_model_features()
    prepared_input = prepare_input(input_data, model_features)

    if model:
        prediction = model.predict(prepared_input)[0]
        st.success(f"💰 Szacowana cena sprzedaży: ${prediction:,.2f}")