import sys
import os  

# This makes sure we can import modules from the src folder (we are nested 2 levels inside the root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from src.utils.logger import get_logger
from src.features.build_features import engineer_features

logger = get_logger(__name__, log_file="logs/preprocess.log")


def load_data(path):
    """Load data from CSV file."""
    logger.info(f"Loading data from {path}")
    """Load data from CSV file."""
    logger.info(f"Loading data from {path}")
    return pd.read_csv(path)


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes target outliers from the dataset.
    """
    ACCEPTED_RANGE_WIDTH: float = 2.0

    iqr: float = df["SalePrice"].quantile(0.75) - df["SalePrice"].quantile(0.25)
    lower_bound: float = df["SalePrice"].quantile(0.25) - ACCEPTED_RANGE_WIDTH * iqr
    if lower_bound < 0.0:
        lower_bound = 0.0
    upper_bound: float = df["SalePrice"].quantile(0.75) + ACCEPTED_RANGE_WIDTH * iqr
    return df[(df["SalePrice"] <= upper_bound) & (df["SalePrice"] >= lower_bound)]


def clean_data(df: pd.DataFrame, do_remove_outliers:bool = True):
    """Clean data by imputing missing values & removing outliers."""
    logger.info("Starting data cleaning process")

    # Remove redundant 

    # Remove outliers (IQR)
    if do_remove_outliers:
        df = remove_outliers(df)
    
    # Input missing values
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include="object").columns

    # Log numeric columns with missing values
    for col in num_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            print(
                f"Filling {missing} missing values in numeric column '{col}' using most frequent value."
            )

    df.loc[:, num_cols] = num_imputer.fit_transform(df[num_cols])

    # Log categorical columns with missing values
    for col in cat_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            print(
                f"Filling {missing} missing values in categorical column '{col}' using constant NOT_PRESENT."
            )

    df.loc[:, cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    
    print("Data cleaning complete")
    
    print("Data cleaning complete")
    return df


def encode_hierarchical_categories(df: pd.DataFrame) -> pd.DataFrame:
    quality_encoder: OrdinalEncoder = OrdinalEncoder(categories=[['NOT_PRESENT', 'Po', 'Fa', 'TA', 'Gd', 'Ex']])
    quality_categories: tuple = (
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", 
        "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", 
        "GarageCond", "PoolQC"
    )

    street_encoder: OrdinalEncoder = OrdinalEncoder(categories=[['NOT_PRESENT', 'Grvl', 'Pave']])
    street_categories: tuple = ('Street', 'Alley')

    shape_encoder: OrdinalEncoder = OrdinalEncoder(categories=[["IR3", "IR2", "IR1", "Reg"]])
    shape_categories: tuple = ("LotShape", )

    utilities_encoder: OrdinalEncoder = OrdinalEncoder(categories=[['AllPub', 'NoSewr', 'NoSeWa', 'ELO']])
    utilities_categories: tuple = ("Utilities", )

    slope_encoder: OrdinalEncoder = OrdinalEncoder(categories=[["Gtl", "Mod", "Sev"]])
    slope_categories: tuple = ("LandSlope", )

    basement_exposure_encoder: OrdinalEncoder = OrdinalEncoder(
        categories=[["NOT_PRESENT", "No", "Mn", "Av", "Gd"]]
    )
    basement_exposure_categories: tuple = ("BsmtExposure",)

    basement_living_quatres_encoder: OrdinalEncoder = OrdinalEncoder(
        categories=[["NOT_PRESENT", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]]
    )
    basement_living_quatres_categories: tuple = ("BsmtFinType1", "BsmtFinType2")

    functionality_encoder: OrdinalEncoder = OrdinalEncoder(
        categories=[["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"]]
    )
    functionality_categories: tuple = ("Functional", )

    finish_encoder: OrdinalEncoder = OrdinalEncoder(categories=[["NOT_PRESENT", "Unf", "RFn", "Fin"]])
    finish_categories: tuple = ("GarageFinish", )

    paved_encoder: OrdinalEncoder = OrdinalEncoder(categories=[["N", "P", "Y"]])
    paved_categories: tuple = ("PavedDrive", )

    for column in paved_categories:
        df[column] = paved_encoder.fit_transform(df[[column]])
    for column in finish_categories:
        df[column] = finish_encoder.fit_transform(df[[column]])
    for column in functionality_categories:
        df[column] = functionality_encoder.fit_transform(df[[column]])
    for column in quality_categories:
        df[column] = quality_encoder.fit_transform(df[[column]])
    for column in street_categories:
        df[column] = street_encoder.fit_transform(df[[column]])
    for column in shape_categories:
        df[column] = shape_encoder.fit_transform(df[[column]])
    for column in utilities_categories:
        df[column] = utilities_encoder.fit_transform(df[[column]])
    for column in slope_categories:
        df[column] = slope_encoder.fit_transform(df[[column]])
    for column in basement_exposure_categories:
        df[column] = basement_exposure_encoder.fit_transform(df[[column]])
    for column in basement_living_quatres_categories:
        df[column] = basement_living_quatres_encoder.fit_transform(df[[column]])
    
    return df


def preprocess_pipeline(input_path, output_path, is_training=True):
    """
    Complete preprocessing pipeline: load -> clean -> engineer features -> encode -> save
    
    Parameters:
    input_path (str): Path to input CSV file
    output_path (str): Path to save processed CSV file
    is_training (bool): Whether this is training data (has SalePrice)
    """
    # Load data
    df = load_data(input_path)
    logger.info(f"Loaded data shape: {df.shape}")
    
    # Clean data
    df_cleaned = clean_data(df)
    logger.info(f"Cleaned data shape: {df_cleaned.shape}")
    
    # Engineer features
    logger.info("Starting feature engineering")
    df_featured = engineer_features(df_cleaned, is_training=is_training)
    logger.info(f"Featured data shape: {df_featured.shape}")
    
    # Encode categories that represent some kind of hierarchy
    df_featured = encode_hierarchical_categories(df_featured)

    # Convert categorical features to numeric using one-hot encoding
    df_featured = pd.get_dummies(df_featured)

    # Save processed data
    df_featured.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")
    
    return df_featured


if __name__ == "__main__":
    # Process training data
    train_featured = preprocess_pipeline(
        input_path="datasets/ames-train.csv",
        output_path="datasets/processed/ames-train-featured.csv",
        is_training=True
    )
    logger.info("Training data processed successfully")