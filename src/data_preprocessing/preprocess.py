import pandas as pd
from sklearn.impute import SimpleImputer

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include='object').columns

    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df

if __name__ == "__main__":
    df = load_data("datasets/ames-train.csv")
    cleaned = clean_data(df)
    cleaned.to_csv("datasets/processed/ames-train-clean.csv", index=False)
