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


def train_model_with_tuning(
    df: pd.DataFrame, 
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
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Store feature names before one-hot encoding
    feature_names = X.columns.tolist()
    
    # Convert categorical features to numeric using one-hot encoding
    X = pd.get_dummies(X)
    
    # Update feature names after one-hot encoding
    encoded_feature_names = X.columns.tolist()
    
    # Split into training and validation sets (80% training, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=2137
    )
    
    if tune_hyperparameters:
        print("Performing hyperparameter tuning with GridSearchCV...")
        print("This may take several minutes...")
        
        # Define parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
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
    
    # Save validation data and predictions for evaluation
    print("\nSaving validation data for evaluation...")
    val_results = pd.DataFrame({
        'Actual': y_val,
        'Predicted': y_pred,
        'Error': y_pred - y_val,
        'PercentError': ((y_pred - y_val) / y_val) * 100
    })
    
    # Add the original features for analysis
    val_results = pd.concat([val_results, X_val.reset_index(drop=True)], axis=1)
    
    # Save to CSV
    os.makedirs("evaluation", exist_ok=True)
    val_results.to_csv("evaluation/validation_predictions.csv", index=False)
    print(f"   ✓ Validation results saved to evaluation/validation_predictions.csv")
    
    # Also save just the validation features and actual prices separately
    val_data = X_val.copy()
    val_data['SalePrice'] = y_val
    val_data.to_csv("evaluation/validation_set.csv", index=False)
    print(f"   ✓ Validation set saved to evaluation/validation_set.csv")
    
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
        'n_features': X.shape[1]
    }
    
    return model, best_params, evaluation_metrics, feature_importance


def save_model_and_metadata(model, params, metrics, feature_importance, model_path="model/"):
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
    
    df = load_data(data_path)
    
    # Ensure there are no missing values in the target
    df = df.dropna(subset=["SalePrice"])
    
    print(f"Loaded featured dataset with shape: {df.shape}")
    
    # Train the model with hyperparameter tuning
    model, best_params, metrics, feature_importance = train_model_with_tuning(
        df, 
        tune_hyperparameters=True  # Set to False to skip GridSearchCV
    )
    
    # Save everything
    save_model_and_metadata(model, best_params, metrics, feature_importance)