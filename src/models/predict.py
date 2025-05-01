import pandas as pd
import joblib
import os


def load_model(model_path: str):
    """
    Load a trained model from the specified file.

    Parameters:
    model_path (str): Path to the saved model file.

    Returns:
    Trained model object.
    """
    return joblib.load(model_path)


def prepare_input(input_data: dict, model_features: list) -> pd.DataFrame:
    """
    Convert user input into a pandas DataFrame and align it with model's expected features.

    Parameters:
    input_data (dict): A dictionary of input values (e.g., from a form or API).
    model_features (list): List of features the model was trained on.

    Returns:
    pd.DataFrame: Input data aligned with model features.
    """
    # Convert the dictionary to a DataFrame with a single row
    df = pd.DataFrame([input_data])

    # Convert categorical fields to one-hot encoded format
    df = pd.get_dummies(df)

    # Find out which features the model expects that are missing in the input
    missing_cols = [col for col in model_features if col not in df.columns]

    # Create a DataFrame with those missing columns and fill them with 0
    # This avoids fragmentation warnings and ensures correct shape
    missing_df = pd.DataFrame([[0] * len(missing_cols)], columns=missing_cols)

    # Combine the user input with the missing columns
    df = pd.concat([df, missing_df], axis=1)

    # Reorder columns to match the order the model was trained on
    df = df[model_features]

    return df


if __name__ == "__main__":
    # Path to the trained model file
    model_path = "model/house_price_model.pkl"

    # Check if model file exists before trying to load it
    if not os.path.exists(model_path):
        print("Model file not found. Please train the model first.")
        exit()

    # Load the trained model
    model = load_model(model_path)

    # Example input to test the prediction logic (replace with your own values)
    example_input = {
        "OverallQual": 7,
        "GrLivArea": 2110,
        "GarageCars": 5,
        "TotalBsmtSF": 856,
        "FullBath": 3,
        "YearBuilt": 2020,
        "Neighborhood": "CollgCr",
        "HouseStyle": "2Story",
    }

    # Load the cleaned training data to get model's expected feature columns
    train_df = pd.read_csv("datasets/processed/ames-train-clean.csv")
    X_train = pd.get_dummies(train_df.drop(columns=["SalePrice"]))
    model_features = X_train.columns.tolist()

    # Prepare the example input to match the model's feature structure
    prepared_input = prepare_input(example_input, model_features)

    # Run prediction and show the result
    prediction = model.predict(prepared_input)[0]
    print(f"Predicted house price: ${prediction:,.2f}")
