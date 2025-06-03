import pandas as pd
import numpy as np
import joblib
import os
import sys
import json

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.features.build_features import engineer_features
from src.data_preprocessing.preprocess import clean_data


def load_model(model_path: str):
    """
    Load a trained model from the specified file.

    Parameters:
    model_path (str): Path to the saved model file.

    Returns:
    Trained model object.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)


def load_model_metadata(metadata_path: str):
    """
    Load model metadata from JSON file.
    
    Parameters:
    metadata_path (str): Path to the metadata JSON file.
    
    Returns:
    dict: Model metadata
    """
    if not os.path.exists(metadata_path):
        return None
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


def get_model_features(featured_data_path: str = "datasets/processed/ames-train-featured.csv"):
    """
    Get the list of features expected by the model.
    
    Parameters:
    featured_data_path (str): Path to the featured training data
    
    Returns:
    list: List of feature names after one-hot encoding
    """
    if not os.path.exists(featured_data_path):
        raise FileNotFoundError(
            f"Featured training data not found at {featured_data_path}. "
            "Please run preprocessing first."
        )
    
    # Load just the first few rows to get column structure (faster)
    train_df = pd.read_csv(featured_data_path, nrows=5)
    X_train = pd.get_dummies(train_df.drop(columns=["SalePrice"]))
    return X_train.columns.tolist()


def prepare_single_input(input_data: dict, model_features: list) -> pd.DataFrame:
    """
    Prepare a single input for prediction.
    
    Parameters:
    input_data (dict): Dictionary of input features
    model_features (list): List of features expected by the model
    
    Returns:
    pd.DataFrame: Prepared input data
    """
    # Convert to DataFrame
    df = pd.DataFrame([input_data])
    
    # Clean and engineer features
    df = clean_data(df)
    df = engineer_features(df, is_training=False)
    
    # One-hot encode
    df = pd.get_dummies(df)
    
    # Align with model features
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    
    # Keep only model features in correct order
    df = df[model_features]
    
    return df


def predict_batch(
    test_file_path: str, 
    model_path: str = "model/house_price_model.pkl",
    output_path: str = None,
    featured_data_path: str = "datasets/processed/ames-train-featured.csv"
):
    """
    Make predictions on a test dataset.
    
    Parameters:
    test_file_path (str): Path to test CSV file
    model_path (str): Path to saved model
    output_path (str): Path to save predictions (optional)
    featured_data_path (str): Path to featured training data
    
    Returns:
    pd.DataFrame: DataFrame with predictions
    """
    print(f"Loading test data from {test_file_path}")
    
    # Check if test file exists
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"Test file not found at {test_file_path}")
    
    # Load test data
    test_df = pd.read_csv(test_file_path)
    print(f"Loaded {len(test_df)} test samples")
    
    # Store the Id column if it exists
    ids = test_df['Id'].copy() if 'Id' in test_df.columns else None
    
    # Remove Id column before processing
    if 'Id' in test_df.columns:
        test_df = test_df.drop(columns=['Id'])
    
    # Clean and engineer features
    print("Cleaning data...")
    test_df = clean_data(test_df)
    
    print("Engineering features...")
    test_df = engineer_features(test_df, is_training=False)
    
    # Load model
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    # Get model features from training data
    print("Aligning features with training data...")
    model_features = get_model_features(featured_data_path)
    
    # One-hot encode the test data
    print("One-hot encoding categorical features...")
    test_encoded = pd.get_dummies(test_df)
    
    # Align test data with model features
    missing_cols = set(model_features) - set(test_encoded.columns)
    if missing_cols:
        print(f"Adding {len(missing_cols)} missing columns with zeros")
        for col in missing_cols:
            test_encoded[col] = 0
    
    # Remove extra columns not in model
    extra_cols = set(test_encoded.columns) - set(model_features)
    if extra_cols:
        print(f"Removing {len(extra_cols)} extra columns not in training data")
    
    # Select only model features in correct order
    test_encoded = test_encoded[model_features]
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(test_encoded)
    
    # Create results dataframe
    if ids is not None:
        results = pd.DataFrame({
            'Id': ids,
            'SalePrice': predictions
        })
    else:
        results = pd.DataFrame({
            'SalePrice': predictions
        })
    
    # Save predictions if output path provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    
    # Display summary statistics
    print(f"\nPrediction Summary:")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Min price: ${predictions.min():,.2f}")
    print(f"Max price: ${predictions.max():,.2f}")
    print(f"Mean price: ${predictions.mean():,.2f}")
    print(f"Median price: ${pd.Series(predictions).median():,.2f}")
    print(f"Std deviation: ${predictions.std():,.2f}")
    
    return results


def main():
    """Main function to run predictions."""
    # Default paths
    model_path = "model/house_price_model.pkl"
    metadata_path = "model/model_metadata.json"
    test_file_path = "datasets/ames-test.csv"
    output_path = "predictions/test_predictions.csv"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("Error: Model file not found. Please train the model first.")
        print("Run: python src/models/train_model.py")
        return
    
    # Load and display model metadata
    metadata = load_model_metadata(metadata_path)
    if metadata:
        print("="*50)
        print("MODEL INFORMATION")
        print("="*50)
        print(f"Model type: {metadata['model_type']}")
        print(f"Training date: {metadata['training_date']}")
        print(f"Validation RMSE: ${metadata['metrics']['validation']['rmse']:,.2f}")
        print(f"Validation R²: {metadata['metrics']['validation']['r2']:.4f}")
        print("="*50)
    
    # Check if test file exists
    if not os.path.exists(test_file_path):
        print(f"\nTest file not found at {test_file_path}")
        print("Please ensure the test dataset is in the correct location.")
        return
    
    # Make batch predictions
    try:
        print("\nStarting batch predictions...")
        results = predict_batch(
            test_file_path=test_file_path,
            model_path=model_path,
            output_path=output_path
        )
        
        print("\n" + "="*50)
        print("PREDICTIONS COMPLETE")
        print("="*50)
        print(f"Output saved to: {output_path}")
        
        # Show first few predictions
        print("\nFirst 10 predictions:")
        print(results.head(10).to_string(index=False))
        
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        print("Please check that all required files are in place and try again.")
        raise


if __name__ == "__main__":
    main()