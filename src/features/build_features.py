import pandas as pd
import numpy as np


def engineer_features(df, is_training=True):
    """
    Feature engineering with 10 new features.
    Parameters:
    df (pd.DataFrame): Input dataframe with house data
    is_training (bool): Whether this is training data (has SalePrice)
    
    Returns:
    pd.DataFrame: Dataframe with 10 engineered features added
    """
    # Create a copy to avoid modifying the original
    df_featured = df.copy()
    
    # 1. TotalSF: Total Square Footage (basement + 1st floor + 2nd floor)
    # This captures the overall size of the house
    df_featured['TotalSF'] = (
        df_featured['TotalBsmtSF'].fillna(0) + 
        df_featured['1stFlrSF'].fillna(0) + 
        df_featured['2ndFlrSF'].fillna(0)
    )
    

    # 2. OverallQualityScore: Quality * Condition
    # Combines material quality and condition into one metric
    df_featured['OverallQualityScore'] = (
        df_featured['OverallQual'] * df_featured['OverallCond']
    )
    
    # 3. HouseAge: How old was the house when sold
    # Newer houses are typically more expensive

    df_featured['HouseAge'] = df_featured['YrSold'] - df_featured['YearBuilt']
    
    # 4. TotalBathrooms: All bathrooms combined (full + 0.5 * half)
    # More bathrooms = higher value, half baths count as 0.5
    df_featured['TotalBathrooms'] = (
        df_featured['FullBath'].fillna(0) + 
        0.5 * df_featured['HalfBath'].fillna(0) + 
        df_featured['BsmtFullBath'].fillna(0) + 
        0.5 * df_featured['BsmtHalfBath'].fillna(0)
    )
    
    # 5. GarageCapacity: Combined garage metric
    # Normalizes garage area to car units and adds to car capacity
    df_featured['GarageCapacity'] = (
        df_featured['GarageCars'].fillna(0) + 
        (df_featured['GarageArea'].fillna(0) / 200)  # ~200 sq ft per car
    )
    
    # 6. HasBasement: Binary indicator for basement presence
    # Simple but important - basements add significant value
    df_featured['HasBasement'] = (
        (df_featured['TotalBsmtSF'] > 0).astype(int)
    )
    
    # 7. HasSecondFloor: Binary indicator for multi-story
    # Distinguishes between single and multi-story homes
    df_featured['HasSecondFloor'] = (
        (df_featured['2ndFlrSF'] > 0).astype(int)
    )
    
    # 8. LotAreaLog: Natural log of lot size
    # Reduces impact of outliers and captures diminishing returns
    df_featured['LotAreaLog'] = np.log1p(df_featured['LotArea'])
    

    # 9. QualityPriceInteraction: Quality * Living Area
    # High quality matters more in larger homes
    df_featured['QualityPriceInteraction'] = (
        df_featured['OverallQual'] * df_featured['GrLivArea']
    )
    
    # 10. RecentRemodel: Was house remodeled in last 10 years?
    # Recent updates command premium prices
    df_featured['RecentRemodel'] = (
        (df_featured['YrSold'] - df_featured['YearRemodAdd']) < 10
    ).astype(int)
    
    # Print summary
    print(f"Feature engineering complete!")
    print(f"Original features: {df.shape[1]}")
    print(f"Total features after engineering: {df_featured.shape[1]}")
    print(f"New features created: 10")
    
    # List the new features for verification
    new_features = [
        'TotalSF', 'OverallQualityScore', 'HouseAge', 'TotalBathrooms',
        'GarageCapacity', 'HasBasement', 'HasSecondFloor', 'LotAreaLog',
        'QualityPriceInteraction', 'RecentRemodel'
    ]
    print(f"New features: {', '.join(new_features)}")
    
    return df_featured


# Example usage for testing
if __name__ == "__main__":
    # This allows you to test the feature engineering independently
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    
    # Load the cleaned data
    train_df = pd.read_csv("datasets/processed/ames-train-clean.csv")
    
    # Engineer features
    train_featured = engineer_features(train_df, is_training=True)
    
    # Display the new features
    print("\nFirst 5 rows of new features:")
    print(train_featured[['TotalSF', 'OverallQualityScore', 'HouseAge', 
                         'TotalBathrooms', 'GarageCapacity']].head())
    
    print("\nBasic statistics for new features:")
    new_feature_cols = ['TotalSF', 'OverallQualityScore', 'HouseAge', 'TotalBathrooms',
                       'GarageCapacity', 'HasBasement', 'HasSecondFloor', 'LotAreaLog',
                       'QualityPriceInteraction', 'RecentRemodel']
    print(train_featured[new_feature_cols].describe())
    
    # Save the featured dataset
    train_featured.to_csv("datasets/processed/ames-train-featured.csv", index=False)
    print("\nFeatured dataset saved to datasets/processed/ames-train-featured.csv")