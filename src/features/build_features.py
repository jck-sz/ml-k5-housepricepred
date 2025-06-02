def create_polynomial_features(df):
    """
    Create polynomial and interaction features for key numeric variables.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with house data
    
    Returns:
    pd.DataFrame: Dataframe with polynomial features
    """
    # GrLivArea_Squared: Square of above-ground living area
    # Example: GrLimport pandas as pd
import numpy as np
from datetime import datetime


import pandas as pd
import numpy as np
from datetime import datetime


def create_age_features(df):
    """
    Create features related to property age and remodeling.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with house data
    
    Returns:
    pd.DataFrame: Dataframe with new age-related features
    """
    # Assuming current year for age calculations (you might want to use YrSold instead)
    current_year = datetime.now().year
    
    # HouseAge: How old was the house when it was sold?
    # Example: If YearBuilt=1990 and YrSold=2010, then HouseAge=20 years
    # Rationale: Newer houses often command higher prices, but relationship isn't always linear
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    
    # YearsSinceRemodel: How long since the last renovation?
    # Example: If YearRemodAdd=2005 and YrSold=2010, then YearsSinceRemodel=5 years
    # Rationale: Recently remodeled homes typically sell for more
    df['YearsSinceRemodel'] = df['YrSold'] - df['YearRemodAdd']
    
    # WasRemodeled: Binary indicator - has the house ever been remodeled?
    # Example: If YearBuilt=1990 and YearRemodAdd=2005, then WasRemodeled=1 (True)
    # Rationale: Any remodeling (vs. none) can significantly impact value
    df['WasRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    
    # AgeCategory: Categorical grouping of house age
    # Example: HouseAge=3 → 'Very New', HouseAge=15 → 'Moderate'
    # Rationale: Captures non-linear age effects (e.g., 0-5 years and 40+ years might have different price patterns)
    df['AgeCategory'] = pd.cut(df['HouseAge'], 
                               bins=[0, 5, 10, 20, 40, 100], 
                               labels=['Very New', 'New', 'Moderate', 'Old', 'Very Old'])
    
    # GarageAge: Age of the garage structure
    # Example: If GarageYrBlt=1995 and YrSold=2010, then GarageAge=15 years
    # Rationale: Newer garages (better condition) add more value
    # Note: -1 indicates no garage exists
    if 'GarageYrBlt' in df.columns:
        df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']
        # Handle missing garage years
        df['GarageAge'] = df['GarageAge'].fillna(-1)  # -1 indicates no garage
    
    return df

def create_area_features(df):
    """
    Create features related to property size and area calculations.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with house data
    
    Returns:
    pd.DataFrame: Dataframe with new area-related features
    """
    # TotalSF: Combined square footage of all living areas (basement + 1st floor + 2nd floor)
    # Example: TotalBsmtSF=800 + 1stFlrSF=1200 + 2ndFlrSF=500 = TotalSF=2500 sq ft
    # Rationale: Total living space is often more predictive than individual floor areas
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
    # TotalBathrooms: Combined count of all bathrooms (full bathrooms + 0.5 * half bathrooms)
    # Example: 2 FullBath + 1 HalfBath + 1 BsmtFullBath + 0 BsmtHalfBath = 3.5 total bathrooms
    # Rationale: More bathrooms = higher value, half baths count as 0.5
    df['TotalBathrooms'] = (df['FullBath'] + 
                           0.5 * df['HalfBath'] + 
                           df['BsmtFullBath'] + 
                           0.5 * df['BsmtHalfBath'])
    
    # TotalPorchSF: Combined area of all outdoor living spaces
    # Example: WoodDeck=200 + OpenPorch=50 + ScreenPorch=100 = TotalPorchSF=350 sq ft
    # Rationale: Outdoor living space adds value, especially in good climates
    porch_cols = ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    df['TotalPorchSF'] = df[porch_cols].sum(axis=1)
    
    # AvgRoomSize: Average size per room (GrLivArea / TotRmsAbvGrd)
    # Example: GrLivArea=1500 sq ft / TotRmsAbvGrd=6 rooms = 250 sq ft per room
    # Rationale: Larger rooms often indicate higher-end properties
    df['AvgRoomSize'] = df['GrLivArea'] / df['TotRmsAbvGrd']
    # Handle division by zero
    df['AvgRoomSize'] = df['AvgRoomSize'].replace([np.inf, -np.inf], 0)
    
    # FinishedBsmtRatio: What proportion of the basement is finished?
    # Example: BsmtFinSF1=500 + BsmtFinSF2=0, TotalBsmtSF=1000 → Ratio=0.5 (50% finished)
    # Rationale: Finished basements add significant value vs. unfinished
    df['FinishedBsmtRatio'] = np.where(df['TotalBsmtSF'] > 0, 
                                       (df['BsmtFinSF1'] + df['BsmtFinSF2']) / df['TotalBsmtSF'], 
                                       0)
    
    # LivingAreaRatio: Proportion of above-ground space to total space
    # Example: GrLivArea=1500, TotalSF=2500 → Ratio=0.6 (60% is above ground)
    # Rationale: Houses with more above-ground space often valued higher than basement-heavy homes
    df['LivingAreaRatio'] = df['GrLivArea'] / (df['TotalSF'] + 1)  # +1 to avoid division by zero
    
    # GarageRatio: Size of garage relative to house size
    # Example: GarageArea=400, GrLivArea=2000 → Ratio=0.2 (garage is 20% of living area)
    # Rationale: Larger garages relative to house size can indicate car enthusiast homes or storage needs
    df['GarageRatio'] = df['GarageArea'] / df['GrLivArea']
    
    return df

def create_quality_features(df):
    """
    Create features related to quality and condition ratings.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with house data
    
    Returns:
    pd.DataFrame: Dataframe with new quality-related features
    """
    # OverallScore: Multiplicative interaction between overall quality and condition
    # Example: OverallQual=7 * OverallCond=8 = OverallScore=56
    # Rationale: High quality + good condition = premium value (multiplicative effect)
    df['OverallScore'] = df['OverallQual'] * df['OverallCond']
    
    # Quality mapping: Convert categorical quality ratings to numeric scale
    # Ex: 'Excellent', Gd: 'Good', TA: 'Typical/Average', Fa: 'Fair', Po: 'Poor'
    # Mapping: Ex=5, Gd=4, TA=3, Fa=2, Po=1
    quality_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    
    # ExterQualNum & ExterCondNum: Numeric versions of exterior quality/condition
    # Example: ExterQual='Gd' → ExterQualNum=4
    # Rationale: Allows mathematical operations on quality ratings
    if 'ExterQual' in df.columns:
        df['ExterQualNum'] = df['ExterQual'].map(quality_mapping).fillna(3)
    
    if 'ExterCond' in df.columns:
        df['ExterCondNum'] = df['ExterCond'].map(quality_mapping).fillna(3)
        
    # ExterScore: Combined exterior quality score
    # Example: ExterQual='Gd'(4) * ExterCond='TA'(3) = ExterScore=12
    # Rationale: Exterior appearance strongly influences curb appeal and value
    if 'ExterQual' in df.columns and 'ExterCond' in df.columns:
        df['ExterScore'] = df['ExterQualNum'] * df['ExterCondNum']
    
    # KitchenScore: Kitchen quality weighted by number of kitchens
    # Example: KitchenQual='Ex'(5) * KitchenAbvGr=2 = KitchenScore=10
    # Rationale: High-quality kitchens are major selling points; multiple kitchens add value
    if 'KitchenQual' in df.columns:
        df['KitchenQualNum'] = df['KitchenQual'].map(quality_mapping).fillna(3)
        df['KitchenScore'] = df['KitchenQualNum'] * df['KitchenAbvGr']
    
    # BsmtQualNum: Numeric basement quality (0 = no basement)
    # Example: BsmtQual='Gd' → BsmtQualNum=4; BsmtQual=NA → BsmtQualNum=0
    # Rationale: Quality basements add significant value
    if 'BsmtQual' in df.columns:
        df['BsmtQualNum'] = df['BsmtQual'].map(quality_mapping).fillna(0)  # 0 for no basement
        
    # GarageQualNum: Numeric garage quality (0 = no garage)
    # Example: GarageQual='TA' → GarageQualNum=3; GarageQual=NA → GarageQualNum=0
    # Rationale: Garage quality impacts value, especially in cold climates
    if 'GarageQual' in df.columns:
        df['GarageQualNum'] = df['GarageQual'].map(quality_mapping).fillna(0)  # 0 for no garage
    
    # AvgQuality: Mean quality across all quality measures
    # Example: OverallQual=7, ExterQual='Gd'(4), KitchenQual='Ex'(5) → AvgQuality=5.33
    # Rationale: Overall property quality indicator across multiple dimensions
    quality_cols = ['OverallQual']
    if 'ExterQualNum' in df.columns:
        quality_cols.append('ExterQualNum')
    if 'KitchenQualNum' in df.columns:
        quality_cols.append('KitchenQualNum')
    if 'BsmtQualNum' in df.columns:
        quality_cols.append('BsmtQualNum')
    
    df['AvgQuality'] = df[quality_cols].mean(axis=1)
    
    return df

def create_binary_features(df):
    """
    Create binary indicator features.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with house data
    
    Returns:
    pd.DataFrame: Dataframe with new binary features
    """
    # HasPool: Does the property have a pool?
    # Example: PoolArea=500 → HasPool=1; PoolArea=0 → HasPool=0
    # Rationale: Pools are luxury features that can significantly impact price
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    
    # HasFireplace: Does the property have any fireplaces?
    # Example: Fireplaces=2 → HasFireplace=1; Fireplaces=0 → HasFireplace=0
    # Rationale: Fireplaces add ambiance and value, especially in colder climates
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    
    # HasGarage: Does the property have a garage?
    # Example: GarageArea=400 → HasGarage=1; GarageArea=0 → HasGarage=0
    # Rationale: Garage presence is crucial in many markets
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    
    # HasBasement: Does the property have a basement?
    # Example: TotalBsmtSF=1000 → HasBasement=1; TotalBsmtSF=0 → HasBasement=0
    # Rationale: Basements add significant living/storage space
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
    
    # Has2ndFloor: Is this a multi-story home?
    # Example: 2ndFlrSF=600 → Has2ndFloor=1; 2ndFlrSF=0 → Has2ndFloor=0
    # Rationale: Multi-story homes often command different prices than single-story
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    
    # HasWoodDeck: Does the property have a wood deck?
    # Example: WoodDeckSF=200 → HasWoodDeck=1; WoodDeckSF=0 → HasWoodDeck=0
    # Rationale: Decks provide outdoor entertainment space
    df['HasWoodDeck'] = (df['WoodDeckSF'] > 0).astype(int)
    
    # HasPorch: Does the property have any type of porch?
    # Calculated after TotalPorchSF is created in create_area_features()
    # Rationale: Porches add curb appeal and outdoor living space
    df['HasPorch'] = (df['TotalPorchSF'] > 0).astype(int) if 'TotalPorchSF' in df.columns else 0
    
    # HasMultipleFloors: Does the house have both first and second floors with living space?
    # Example: 1stFlrSF=1000 AND 2ndFlrSF=800 → HasMultipleFloors=1
    # Rationale: True multi-story homes (not just finished attics) are often valued differently
    df['HasMultipleFloors'] = ((df['1stFlrSF'] > 0) & (df['2ndFlrSF'] > 0)).astype(int)
    
    # HasGoodFence: Does the property have a high-quality fence?
    # Example: Fence='GdPrv' (Good Privacy) → HasGoodFence=1
    # Rationale: Quality fencing adds privacy and security value
    if 'Fence' in df.columns:
        df['HasGoodFence'] = df['Fence'].isin(['GdPrv', 'GdWo']).astype(int)
    
    # HasCentralAir: Does the property have central air conditioning?
    # Example: CentralAir='Y' → HasCentralAir=1
    # Rationale: Central air is expected in many markets and adds significant value
    if 'CentralAir' in df.columns:
        df['HasCentralAir'] = (df['CentralAir'] == 'Y').astype(int)
    
    # HasPavedDrive: Is the driveway paved?
    # Example: PavedDrive='Y' → HasPavedDrive=1; PavedDrive='N' (gravel) → HasPavedDrive=0
    # Rationale: Paved driveways are more desirable than gravel/dirt
    if 'PavedDrive' in df.columns:
        df['HasPavedDrive'] = (df['PavedDrive'] == 'Y').astype(int)
    
    return df

def create_neighborhood_features(df):
    """
    Create features related to neighborhood statistics.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with house data
    
    Returns:
    pd.DataFrame: Dataframe with new neighborhood-related features
    """
    # Note: Neighborhood-based pricing features are only calculated for training data
    # Test data will use neighborhood statistics from training data
    
    if 'SalePrice' in df.columns:
        # NeighborhoodMedianPrice: Median sale price in this neighborhood
        # Example: If NAmes neighborhood has prices [150k, 160k, 170k], median = 160k
        # Rationale: Helps capture location-based pricing patterns
        
        # NeighborhoodMeanPrice: Average sale price in this neighborhood
        # Example: Same neighborhood, mean = 160k
        # Rationale: Alternative measure of neighborhood value
        
        # NeighborhoodPriceStd: Standard deviation of prices in neighborhood
        # Example: High std = diverse housing stock, Low std = uniform housing
        # Rationale: Price variability indicates neighborhood homogeneity
        neighborhood_stats = df.groupby('Neighborhood')['SalePrice'].agg(['median', 'mean', 'std'])
        neighborhood_stats.columns = ['NeighborhoodMedianPrice', 'NeighborhoodMeanPrice', 'NeighborhoodPriceStd']
        df = df.merge(neighborhood_stats, left_on='Neighborhood', right_index=True, how='left')
        
        # PriceDeviationFromNeighborhood: How does this house compare to neighborhood median?
        # Example: SalePrice=180k, NeighborhoodMedian=160k → Deviation=+20k
        # Rationale: Identifies houses priced above/below neighborhood norms
        df['PriceDeviationFromNeighborhood'] = df['SalePrice'] - df['NeighborhoodMedianPrice']
    
    # NeighborhoodSize: Number of houses in this neighborhood (in the dataset)
    # Example: 'NAmes' appears 50 times → NeighborhoodSize=50
    # Rationale: Larger neighborhoods may have more stable pricing
    neighborhood_counts = df['Neighborhood'].value_counts()
    df['NeighborhoodSize'] = df['Neighborhood'].map(neighborhood_counts)
    
    # NeighborhoodAvgQuality: Average overall quality rating in the neighborhood
    # Example: If CollgCr has houses with OverallQual [7,8,9], avg = 8
    # Rationale: Indicates whether this is a high/low quality neighborhood
    neighborhood_quality = df.groupby('Neighborhood')['OverallQual'].mean()
    df['NeighborhoodAvgQuality'] = df['Neighborhood'].map(neighborhood_quality)
    
    return df

def create_polynomial_features(df):
    """
    Create polynomial and interaction features for key numeric variables.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with house data
    
    Returns:
    pd.DataFrame: Dataframe with polynomial features
    """
    # GrLivArea_Squared: Square of above-ground living area
    # Example: GrLivArea=1500 → GrLivArea_Squared=2,250,000
    # Rationale: Captures non-linear relationship between size and price (e.g., diminishing returns for very large houses)
    df['GrLivArea_Squared'] = df['GrLivArea'] ** 2
    
    # TotalSF_Squared: Square of total square footage
    # Example: TotalSF=2000 → TotalSF_Squared=4,000,000
    # Rationale: Similar to above, captures non-linear size effects
    df['TotalSF_Squared'] = df['TotalSF'] ** 2 if 'TotalSF' in df.columns else 0
    
    # Qual_SF_Interaction: Quality rating multiplied by living area
    # Example: OverallQual=8 * GrLivArea=1500 = 12,000
    # Rationale: High quality matters more in larger homes (interaction effect)
    df['Qual_SF_Interaction'] = df['OverallQual'] * df['GrLivArea']
    
    # Qual_Age_Interaction: Quality rating multiplied by house age
    # Example: OverallQual=8 * HouseAge=10 = 80
    # Rationale: Quality ratings may have different impacts on new vs. old homes
    df['Qual_Age_Interaction'] = df['OverallQual'] * df['HouseAge'] if 'HouseAge' in df.columns else 0
    
    # Bath_Bedroom_Ratio: Bathrooms per bedroom
    # Example: TotalBathrooms=2.5 / BedroomAbvGr=3 = 0.83 bathrooms per bedroom
    # Rationale: Higher ratios indicate luxury (e.g., ensuite bathrooms)
    # Note: Add 1 to denominator to avoid division by zero for studio apartments
    df['Bath_Bedroom_Ratio'] = df['TotalBathrooms'] / (df['BedroomAbvGr'] + 1) if 'TotalBathrooms' in df.columns else 0
    
    return df

def create_temporal_features(df):
    """
    Create features related to time of sale.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with house data
    
    Returns:
    pd.DataFrame: Dataframe with temporal features
    """
    # SaleSeason: Season when the house was sold
    # Example: MoSold=6 (June) → SaleSeason='Summer'
    # Rationale: Real estate markets often have seasonal patterns (e.g., more sales in summer)
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    df['SaleSeason'] = df['MoSold'].map(season_map)
    
    # IsSummerSale: Binary indicator for summer sales (June, July, August)
    # Example: MoSold=7 → IsSummerSale=1; MoSold=12 → IsSummerSale=0
    # Rationale: Summer is typically the busiest real estate season with higher prices
    df['IsSummerSale'] = df['MoSold'].isin([6, 7, 8]).astype(int)
    
    # YearsSince2006: Years since the 2006 housing market peak (before 2008 crisis)
    # Example: YrSold=2010 → YearsSince2006=4
    # Rationale: Captures market recovery timeline after the housing crisis
    df['YearsSince2006'] = df['YrSold'] - 2006
    
    return df

def engineer_features(df, is_training=True):
    """
    Main function to engineer all features.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with house data
    is_training (bool): Whether this is training data (has SalePrice)
    
    Returns:
    pd.DataFrame: Dataframe with all engineered features
    """
    # Create a copy to avoid modifying the original
    df_featured = df.copy()
    
    # Apply all feature engineering functions
    df_featured = create_age_features(df_featured)
    df_featured = create_area_features(df_featured)
    df_featured = create_quality_features(df_featured)
    df_featured = create_binary_features(df_featured)
    df_featured = create_neighborhood_features(df_featured)
    df_featured = create_polynomial_features(df_featured)
    df_featured = create_temporal_features(df_featured)
    
    # Log transformation for skewed features
    # Log transforms help normalize right-skewed distributions
    # Example: LotArea=10,000 → LotArea_Log=log(10,001)≈9.21
    # Rationale: Many real estate features (area, price) are log-normally distributed
    skewed_features = ['LotArea', 'GrLivArea', 'TotalSF']
    for feature in skewed_features:
        if feature in df_featured.columns and (df_featured[feature] > 0).all():
            df_featured[f'{feature}_Log'] = np.log1p(df_featured[feature])  # log1p = log(1+x) to handle 0 values
    
    # Drop temporary columns used for calculations
    # These numeric versions were only needed to create other features
    temp_cols = ['ExterQualNum', 'ExterCondNum', 'KitchenQualNum', 'BsmtQualNum', 'GarageQualNum']
    df_featured = df_featured.drop(columns=[col for col in temp_cols if col in df_featured.columns])
    
    print(f"Feature engineering complete!")
    print(f"Original features: {df.shape[1]}")
    print(f"Total features after engineering: {df_featured.shape[1]}")
    print(f"New features created: {df_featured.shape[1] - df.shape[1]}")
    
    return df_featured

# Example usage
if __name__ == "__main__":
    # This allows you to test the feature engineering independently
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    
    # Load the cleaned data
    train_df = pd.read_csv("datasets/processed/ames-train-clean.csv")
    
    # Engineer features
    train_featured = engineer_features(train_df, is_training=True)
    
    # Save the featured dataset
    train_featured.to_csv("datasets/processed/ames-train-featured.csv", index=False)
    print("\nFeatured dataset saved to datasets/processed/ames-train-featured.csv")