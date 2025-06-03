import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def load_validation_results(file_path="evaluation/validation_predictions.csv"):
    """Load the validation results from CSV."""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please run train_model.py first.")
        return None
    return pd.read_csv(file_path)

def calculate_metrics(df):
    """Calculate evaluation metrics."""
    actual = df['Actual']
    predicted = df['Predicted']
    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
        'MAE': mean_absolute_error(actual, predicted),
        'R2': r2_score(actual, predicted),
        'MAPE': np.mean(np.abs((actual - predicted) / actual)) * 100,
        'Mean_Error': np.mean(predicted - actual),
        'Std_Error': np.std(predicted - actual)
    }
    
    return metrics

def create_visualizations(df, save_path="evaluation/plots"):
    """Create and save evaluation plots."""
    os.makedirs(save_path, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Actual vs Predicted Scatter Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(df['Actual'], df['Predicted'], alpha=0.5)
    
    # Add perfect prediction line
    min_price = min(df['Actual'].min(), df['Predicted'].min())
    max_price = max(df['Actual'].max(), df['Predicted'].max())
    plt.plot([min_price, max_price], [min_price, max_price], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Price ($)', fontsize=12)
    plt.ylabel('Predicted Price ($)', fontsize=12)
    plt.title('Actual vs Predicted House Prices', fontsize=14, fontweight='bold')
    plt.legend()
    
    # Add R² to the plot
    r2 = r2_score(df['Actual'], df['Predicted'])
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/actual_vs_predicted.png", dpi=300)
    plt.show()
    
    # 2. Residual Plot
    plt.figure(figsize=(10, 6))
    residuals = df['Error']
    plt.scatter(df['Predicted'], residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    
    # Add ±1 standard deviation lines
    std_residual = residuals.std()
    plt.axhline(y=std_residual, color='orange', linestyle='--', alpha=0.7, label=f'±1 SD (${std_residual:,.0f})')
    plt.axhline(y=-std_residual, color='orange', linestyle='--', alpha=0.7)
    
    plt.xlabel('Predicted Price ($)', fontsize=12)
    plt.ylabel('Residual (Predicted - Actual) ($)', fontsize=12)
    plt.title('Residual Plot', fontsize=14, fontweight='bold')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/residual_plot.png", dpi=300)
    plt.show()
    
    # 3. Distribution of Errors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Absolute errors
    ax1.hist(np.abs(df['Error']), bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(np.abs(df['Error']).mean(), color='r', linestyle='--', 
                label=f'Mean: ${np.abs(df["Error"]).mean():,.0f}')
    ax1.set_xlabel('Absolute Error ($)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Absolute Errors', fontsize=14)
    ax1.legend()
    
    # Percentage errors
    ax2.hist(df['PercentError'], bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(df['PercentError'].mean(), color='r', linestyle='--', 
                label=f'Mean: {df["PercentError"].mean():.1f}%')
    ax2.set_xlabel('Percentage Error (%)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Percentage Errors', fontsize=14)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/error_distributions.png", dpi=300)
    plt.show()
    
    # 4. Prediction Performance by Price Range
    plt.figure(figsize=(10, 6))
    
    # Create price bins
    price_bins = [0, 100000, 150000, 200000, 250000, 300000, 1000000]
    price_labels = ['<$100k', '$100-150k', '$150-200k', '$200-250k', '$250-300k', '>$300k']
    df['PriceRange'] = pd.cut(df['Actual'], bins=price_bins, labels=price_labels)
    
    # Calculate MAPE for each price range
    mape_by_range = df.groupby('PriceRange').apply(
        lambda x: np.mean(np.abs((x['Error']) / x['Actual'])) * 100
    )
    
    mape_by_range.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xlabel('Price Range', fontsize=12)
    plt.ylabel('Mean Absolute Percentage Error (%)', fontsize=12)
    plt.title('Model Performance by Price Range', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(mape_by_range):
        plt.text(i, v + 0.1, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/performance_by_price_range.png", dpi=300)
    plt.show()

def create_summary_report(df, metrics, save_path="evaluation"):
    """Create a summary report of the evaluation."""
    report = f"""
# MODEL EVALUATION REPORT
========================

## Overall Performance Metrics
- **RMSE**: ${metrics['RMSE']:,.2f}
- **MAE**: ${metrics['MAE']:,.2f}
- **R² Score**: {metrics['R2']:.4f}
- **MAPE**: {metrics['MAPE']:.2f}%
- **Mean Error**: ${metrics['Mean_Error']:,.2f}
- **Std Error**: ${metrics['Std_Error']:,.2f}

## Validation Set Summary
- **Number of Houses**: {len(df)}
- **Actual Price Range**: ${df['Actual'].min():,.0f} - ${df['Actual'].max():,.0f}
- **Predicted Price Range**: ${df['Predicted'].min():,.0f} - ${df['Predicted'].max():,.0f}

## Error Analysis
- **Houses within 5% error**: {len(df[df['PercentError'].abs() <= 5])} ({len(df[df['PercentError'].abs() <= 5])/len(df)*100:.1f}%)
- **Houses within 10% error**: {len(df[df['PercentError'].abs() <= 10])} ({len(df[df['PercentError'].abs() <= 10])/len(df)*100:.1f}%)
- **Houses within 15% error**: {len(df[df['PercentError'].abs() <= 15])} ({len(df[df['PercentError'].abs() <= 15])/len(df)*100:.1f}%)

## Top 5 Best Predictions
{df.nsmallest(5, 'PercentError', keep='all')[['Actual', 'Predicted', 'Error', 'PercentError']].to_string()}

## Top 5 Worst Predictions
{df.nlargest(5, 'PercentError', keep='all')[['Actual', 'Predicted', 'Error', 'PercentError']].to_string()}
"""
    
    # Save report
    with open(f"{save_path}/evaluation_report.txt", 'w') as f:
        f.write(report)
    
    print(report)

def main():
    """Run the complete evaluation."""
    print("="*60)
    print("HOUSE PRICE MODEL EVALUATION")
    print("="*60)
    
    # Load validation results
    print("\n1. Loading validation results...")
    df = load_validation_results()
    if df is None:
        return
    print(f"   ✓ Loaded {len(df)} validation samples")
    
    # Calculate metrics
    print("\n2. Calculating metrics...")
    metrics = calculate_metrics(df)
    print("   ✓ Metrics calculated")
    
    # Create visualizations
    print("\n3. Creating visualizations...")
    create_visualizations(df)
    print("   ✓ Plots saved to evaluation/plots/")
    
    # Create summary report
    print("\n4. Creating summary report...")
    create_summary_report(df, metrics)
    print("   ✓ Report saved to evaluation/evaluation_report.txt")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("\nCheck the 'evaluation' folder for:")
    print("- validation_predictions.csv (raw results)")
    print("- evaluation_report.txt (summary report)")
    print("- plots/ (visualization images)")

if __name__ == "__main__":
    main()