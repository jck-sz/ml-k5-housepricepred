import sys
import os  

# This makes sure we can import logger from the src folder (we are nested 2 levels inside the root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import pandas as pd
from sklearn.impute import SimpleImputer
from src.utils.logger import get_logger


logger = get_logger(__name__, log_file="logs/preprocess.log")


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
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

    return df


if __name__ == "__main__":
    df = load_data("datasets/ames-train.csv")
    cleaned = clean_data(df)
    cleaned.to_csv("datasets/processed/ames-train-clean.csv", index=False)
