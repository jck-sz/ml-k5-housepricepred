import sys
import os  

# This makes sure we can import logger from the src folder (we are nested 2 levels inside the root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import pandas as pd
from sklearn.impute import SimpleImputer
from src.utils.logger import get_logger
from src.features.build_features import engineer_features


logger = get_logger(__name__, log_file="logs/preprocess.log")


def load_data(path):
    """Load data from CSV file."""
    logger.info(f"Loading data from {path}")
    return pd.read_csv(path)


def clean_data(df):
    """Clean data by imputing missing values."""
    logger.info("Starting data cleaning process")
    
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include="object").columns

    # Log numeric columns with missing values
    for col in num_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            logger.info(
                f"Filling {missing} missing values in numeric column '{col}' using median."
            )

    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Log categorical columns with missing values
    for col in cat_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            logger.info(
                f"Filling {missing} missing values in categorical column '{col}' using most frequent."
            )

    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    
    logger.info("Data cleaning complete")
    return df


def preprocess_pipeline(input_path, output_path, is_training=True):
    """
    Complete preprocessing pipeline: load -> clean -> engineer features -> save
    
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