import pandas as pd
import numpy as np
import os
import joblib

# Import machine learning tools from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt


def load_data(path: str) -> pd.DataFrame:
    """
    Load a CSV file and return it as a pandas DataFrame.

    Parameters:
    path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded dataset.
    """
    return pd.read_csv(path)


def train_model(
    df: pd.DataFrame, target_column: str = "SalePrice"
) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor model on the given DataFrame.

    Parameters:
    df (pd.DataFrame): The input dataset with features and target.
    target_column (str): The name of the column to predict. Defaults to 'SalePrice'.

    Returns:
    RandomForestRegressor: The trained model.
    """
    # Separate features (X) and target variable (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Convert categorical features to numeric using one-hot encoding
    X = pd.get_dummies(X)

    # Split into training and validation sets (80% training, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=2137
    )

    # Create and train the RandomForest model
    model = RandomForestRegressor(random_state=2137)
    model.fit(X_train, y_train)

    # Make predictions on validation set
    y_pred = model.predict(X_val)

    # Calculate RMSE (Root Mean Squared Error)
    rmse = sqrt(mean_squared_error(y_val, y_pred))
    print(f"Validation RMSE: {rmse:.2f}")

    return model


if __name__ == "__main__":
    # Load cleaned training data
    data_path = "datasets/processed/ames-train-clean.csv"
    df = load_data(data_path)

    # Ensure there are no missing values in the target
    df = df.dropna(subset=["SalePrice"])

    # Train the model
    model = train_model(df)

    # Save the model to the 'model' folder
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/house_price_model.pkl")
    print("Model saved to model/house_price_model.pkl")
