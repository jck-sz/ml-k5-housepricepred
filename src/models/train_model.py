import pandas as pd
import numpy as np
import os
import joblib
import json
from datetime import datetime

# Import machine learning tools from scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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


def evaluate_model(model, X, y, cv=5):
    """
    Evaluate model using cross-validation.
    
    Parameters:
    model: The model to evaluate
    X: Features
    y: Target
    cv: Number of cross-validation folds
    
    Returns:
    dict: Dictionary containing evaluation metrics
    """
    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    
    # R2 scores
    cv_r2 = cross_val_score(model, X, y, cv=cv, scoring='r2')
    
    return {
        'cv_rmse_mean': cv_rmse.mean(),
        'cv_rmse_std': cv_rmse.std(),
        'cv_r2_mean': cv_r2.mean(),
        'cv_r2_std': cv_r2.std()
    }


def get_datasets(
    estimators: pd.DataFrame, 
    targets: pd.DataFrame, 
    target_column: str, 
    validation_percentage: float = 0.2,
    dump: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into a training & validation datasets.

    Parameters:
    - estimators (pd.DataFrame): Dataset's columns to be split.
    - targets (pd.DataFrame): Dataset's labels to be split.
    - target_column (str): The name of the column to predict.
    - validation_percentage (float): How many % of dataset should be dedicated to validation.
    Should be float between 0.0 and 1.0. Defaults to 0.2 (20 %).
    - dump (bool): Whether to save created datasets to files for future processing.

    Returns:
    Dataset split to training and validation parts:
    - [0]: Estimators (inputs) for training dataset.
    - [1]: Targets (labels) for training dataset.
    - [2]: Estimators (inputs) for validation dataset.
    - [3]: Targets (labels) for validation dataset.
    """
    DEST_PATH: str = "datasets/used"
    TRAINING_DATASET_PATH: str = f"{DEST_PATH}/training.csv"
    VALIDATION_DATASET_PATH: str = f"{DEST_PATH}/validation.csv"
    
    training_estimators, validation_estimators, training_targets, validation_targets = train_test_split(
        estimators, targets, test_size=validation_percentage, random_state=2137
    )

    if dump:
        training: pd.DataFrame = training_estimators.copy()
        training[target_column] = training_targets
        training.to_csv(TRAINING_DATASET_PATH, index=False)
        print(f"Saved training dataset to: {TRAINING_DATASET_PATH}")

        validation: pd.DataFrame = validation_estimators.copy()
        validation[target_column] = validation_targets
        validation.to_csv(VALIDATION_DATASET_PATH, index=False)
        print(f"Saved validation dataset to: {VALIDATION_DATASET_PATH}")

    return training_estimators, validation_estimators, training_targets, validation_targets


def train_model_with_tuning(
    dataset: pd.DataFrame, 
    target_column: str = "SalePrice",
    tune_hyperparameters: bool = True
) -> tuple:
    """
    Train a RandomForestRegressor model with optional hyperparameter tuning.

    Parameters:
    df (pd.DataFrame): The input dataset with features and target.
    target_column (str): The name of the column to predict. Defaults to 'SalePrice'.
    tune_hyperparameters (bool): Whether to perform GridSearchCV for hyperparameter tuning.

    Returns:
    tuple: (trained_model, best_params, evaluation_metrics, feature_importance)
    """
    print("Starting model training...")

    # Separate features (X) and target variable (y)
    estimators: pd.DataFrame = dataset.drop(columns=[target_column])
    targets: pd.Series = dataset[target_column]
    
    # Update feature names after one-hot encoding
    encoded_feature_names = estimators.columns.tolist()
    
    # Split into training and validation sets (80% training, 20% validation)
    X_train, X_val, y_train, y_val = get_datasets(estimators, targets, target_column, dump=True)  
    
    if tune_hyperparameters:
        print("Performing hyperparameter tuning with GridSearchCV...")
        print("This may take several minutes...")
        
        # Define parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [2, 3, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        # Create base model
        rf_base = RandomForestRegressor(random_state=2137, n_jobs=-1)
        
        # Perform GridSearchCV
        grid_search = GridSearchCV(
            estimator=rf_base,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"\nBest parameters found: {best_params}")
        print(f"Best CV score (RMSE): {np.sqrt(-grid_search.best_score_):.2f}")
        
    else:
        print("Training model without hyperparameter tuning...")
        # Use default parameters
        model = RandomForestRegressor(random_state=2137, n_jobs=-1)
        model.fit(X_train, y_train)
        best_params = model.get_params()
    
    # Make predictions on validation set
    y_pred = model.predict(X_val)
    
    # Save validation predictions with IDs
    print("\nSaving validation predictions...")
    
    # Get the validation indices
    val_indices = X_val.index
    
    # Try to get original IDs
    # First check if we have the original raw data with IDs
    original_path = "datasets/ames-train.csv"
    if os.path.exists(original_path):
        original_raw = pd.read_csv(original_path)
        if 'Id' in original_raw.columns:
            # Get IDs based on index
            val_ids = original_raw.loc[val_indices, 'Id'].values
        else:
            # Use index as ID if no Id column
            val_ids = val_indices + 1  # Adding 1 to make IDs start from 1
    else:
        # Fallback: use index as ID
        val_ids = val_indices + 1
    
    # Create predictions dataframe with just Id and Predicted price
    val_predictions = pd.DataFrame({
        'Id': val_ids,
        'SalePrice': y_pred
    })
    
    # Save predictions
    os.makedirs("evaluation", exist_ok=True)
    val_predictions.to_csv("evaluation/validation_predictions.csv", index=False)
    print(f"   [OK] Validation predictions saved to evaluation/validation_predictions.csv")
    
    # Also save the actual prices for comparison (with IDs)
    val_actual = pd.DataFrame({
        'Id': val_ids,
        'SalePrice': y_val.values
    })
    val_actual.to_csv("evaluation/validation_actual.csv", index=False)
    print(f"   [OK] Validation actual prices saved to evaluation/validation_actual.csv")
    
    # Calculate metrics
    rmse = sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
    
    print(f"\nValidation Set Performance:")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"R-squared Score: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Get cross-validation scores
    cv_metrics = evaluate_model(model, X_train, y_train)
    print(f"\nCross-Validation Performance (5-fold):")
    print(f"CV RMSE: ${cv_metrics['cv_rmse_mean']:,.2f} (+/- ${cv_metrics['cv_rmse_std']:,.2f})")
    print(f"CV R-squared: {cv_metrics['cv_r2_mean']:.4f} (+/- {cv_metrics['cv_r2_std']:.4f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': encoded_feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Print top 20 features
    print("\nTop 20 Most Important Features:")
    print(feature_importance.head(20).to_string(index=False))
    
    # Compile evaluation metrics
    evaluation_metrics = {
        'validation': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        },
        'cross_validation': cv_metrics,
        'n_samples_train': len(X_train),
        'n_samples_val': len(X_val),
        'n_features': estimators.shape[1]
    }
    
    return model, best_params, evaluation_metrics, feature_importance


def save_model_and_metadata(model, params, metrics, feature_importance: pd.DataFrame, model_path="model/"):
    """
    Save the trained model and associated metadata.
    
    Parameters:
    model: Trained model
    params: Model parameters
    metrics: Evaluation metrics
    feature_importance: Feature importance dataframe
    model_path: Directory to save model and metadata
    """
    os.makedirs(model_path, exist_ok=True)
    
    # Save model
    model_file = os.path.join(model_path, "house_price_model.pkl")
    joblib.dump(model, model_file)
    print(f"\nModel saved to {model_file}")
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': 'RandomForestRegressor',
        'parameters': params,
        'metrics': metrics,
        'top_features': feature_importance.head(20).to_dict('records')
    }
    
    metadata_file = os.path.join(model_path, "model_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Model metadata saved to {metadata_file}")
    
    # Save feature importance
    feature_file = os.path.join(model_path, "feature_importance.csv")
    feature_importance.to_csv(feature_file, index=False)
    print(f"Feature importance saved to {feature_file}")


if __name__ == "__main__":
    # Load featured training data
    data_path = "datasets/processed/ames-train-featured.csv"
    
    # Check if featured data exists
    if not os.path.exists(data_path):
        print(f"Featured data not found at {data_path}")
        print("Please run preprocess.py first to generate featured data.")
        exit(1)
    
    dataset = load_data(data_path)
    
    # Ensure there are no missing values in the target
    dataset = dataset.dropna(subset=["SalePrice"])
    
    print(f"Loaded featured dataset with shape: {dataset.shape}")
    
    # Train the model with hyperparameter tuning
    model, best_params, metrics, feature_importance = train_model_with_tuning(
        dataset, 
        tune_hyperparameters=True  # Set to False to skip GridSearchCV
    )
    
    # Save everything
    save_model_and_metadata(model, best_params, metrics, feature_importance)