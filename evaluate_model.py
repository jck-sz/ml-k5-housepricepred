import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def load_validation_data():
    """Load the validation predictions and actual values."""
    pred_path = "evaluation/validation_predictions.csv"
    actual_path = "evaluation/validation_actual.csv"
    
    # Check if files exist
    if not os.path.exists(pred_path):
        print(f"Error: {pred_path} not found. Please run train_model.py first.")
        return None
    if not os.path.exists(actual_path):
        print(f"Error: {actual_path} not found. Please run train_model.py first.")
        return None
    
    # Load data
    predictions = pd.read_csv(pred_path)
    actuals = pd.read_csv(actual_path)
    
    # Merge on Id
    df = pd.merge(actuals, predictions, on='Id', suffixes=('_actual', '_predicted'))
    
    # Rename columns for clarity
    df = df.rename(columns={
        'SalePrice_actual': 'Actual',
        'SalePrice_predicted': 'Predicted'
    })
    
    # Calculate errors
    df['Error'] = df['Predicted'] - df['Actual']
    df['AbsError'] = np.abs(df['Error'])
    df['PercentError'] = (df['Error'] / df['Actual']) * 100
    df['AbsPercentError'] = np.abs(df['PercentError'])
    
    return df

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
        'Std_Error': np.std(predicted - actual),
        'Max_Error': np.max(np.abs(predicted - actual)),
        'Min_Error': np.min(np.abs(predicted - actual))
    }
    
    return metrics

def create_visualizations(df, save_path="evaluation/plots"):
    """Create and save evaluation plots."""
    os.makedirs(save_path, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Actual vs Predicted Scatter Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(df['Actual'], df['Predicted'], alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # Add perfect prediction line
    min_price = min(df['Actual'].min(), df['Predicted'].min())
    max_price = max(df['Actual'].max(), df['Predicted'].max())
    plt.plot([min_price, max_price], [min_price, max_price], 'r--', lw=2, label='Perfect Prediction')
    
    # Add trend line
    z = np.polyfit(df['Actual'], df['Predicted'], 1)
    p = np.poly1d(z)
    plt.plot(df['Actual'], p(df['Actual']), "g-", alpha=0.8, label='Trend Line')
    
    plt.xlabel('Actual Price ($)', fontsize=12)
    plt.ylabel('Predicted Price ($)', fontsize=12)
    plt.title('Actual vs Predicted House Prices', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add R-squared to the plot
    r2 = r2_score(df['Actual'], df['Predicted'])
    plt.text(0.05, 0.95, f'R-squared = {r2:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/actual_vs_predicted.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Residual Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Predicted'], df['Error'], alpha=0.6, edgecolors='k', linewidth=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    
    # Add ±1 standard deviation lines
    std_residual = df['Error'].std()
    plt.axhline(y=std_residual, color='orange', linestyle='--', alpha=0.7, label=f'±1 SD (${std_residual:,.0f})')
    plt.axhline(y=-std_residual, color='orange', linestyle='--', alpha=0.7)
    
    plt.xlabel('Predicted Price ($)', fontsize=12)
    plt.ylabel('Residual (Predicted - Actual) ($)', fontsize=12)
    plt.title('Residual Plot', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/residual_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Distribution of Errors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Absolute errors
    ax1.hist(df['AbsError'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.axvline(df['AbsError'].mean(), color='r', linestyle='--', linewidth=2,
                label=f'Mean: ${df["AbsError"].mean():,.0f}')
    ax1.axvline(df['AbsError'].median(), color='g', linestyle='--', linewidth=2,
                label=f'Median: ${df["AbsError"].median():,.0f}')
    ax1.set_xlabel('Absolute Error ($)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Absolute Errors', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Percentage errors
    ax2.hist(df['PercentError'], bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
    ax2.axvline(df['PercentError'].mean(), color='r', linestyle='--', linewidth=2,
                label=f'Mean: {df["PercentError"].mean():.1f}%')
    ax2.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.5)
    ax2.set_xlabel('Percentage Error (%)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Percentage Errors', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/error_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Prediction Performance by Price Range
    plt.figure(figsize=(12, 6))
    
    # Create price bins
    price_bins = [0, 100000, 150000, 200000, 250000, 300000, 1000000]
    price_labels = ['<$100k', '$100-150k', '$150-200k', '$200-250k', '$250-300k', '>$300k']
    df['PriceRange'] = pd.cut(df['Actual'], bins=price_bins, labels=price_labels)
    
    # Calculate metrics for each price range
    range_stats = df.groupby('PriceRange', observed=False).agg({
        'AbsPercentError': 'mean',
        'Id': 'count'
    }).rename(columns={'Id': 'Count'})
    
    # Create bar plot
    ax = range_stats['AbsPercentError'].plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.xlabel('Price Range', fontsize=12)
    plt.ylabel('Mean Absolute Percentage Error (%)', fontsize=12)
    plt.title('Model Performance by Price Range', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (v, count) in enumerate(zip(range_stats['AbsPercentError'], range_stats['Count'])):
        plt.text(i, v + 0.1, f'{v:.1f}%\n(n={count})', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/performance_by_price_range.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Q-Q Plot for residuals
    plt.figure(figsize=(8, 8))
    from scipy import stats
    stats.probplot(df['Error'], dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/qq_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] All plots saved to {save_path}/")

def create_summary_report(df, metrics, save_path="evaluation"):
    """Create a summary report of the evaluation."""
    
    # Calculate additional statistics
    within_5_pct = len(df[df['AbsPercentError'] <= 5])
    within_10_pct = len(df[df['AbsPercentError'] <= 10])
    within_15_pct = len(df[df['AbsPercentError'] <= 15])
    
    report = f"""MODEL EVALUATION REPORT
======================

Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL PERFORMANCE METRICS
---------------------------
RMSE (Root Mean Squared Error):     ${metrics['RMSE']:,.2f}
MAE (Mean Absolute Error):          ${metrics['MAE']:,.2f}
R-squared Score:                    {metrics['R2']:.4f}
MAPE (Mean Abs Percentage Error):   {metrics['MAPE']:.2f}%
Mean Error (Bias):                  ${metrics['Mean_Error']:,.2f}
Std Dev of Errors:                  ${metrics['Std_Error']:,.2f}
Maximum Absolute Error:             ${metrics['Max_Error']:,.2f}
Minimum Absolute Error:             ${metrics['Min_Error']:,.2f}

VALIDATION SET SUMMARY
----------------------
Number of Houses:         {len(df)}
Actual Price Range:       ${df['Actual'].min():,.0f} - ${df['Actual'].max():,.0f}
Predicted Price Range:    ${df['Predicted'].min():,.0f} - ${df['Predicted'].max():,.0f}
Mean Actual Price:        ${df['Actual'].mean():,.0f}
Mean Predicted Price:     ${df['Predicted'].mean():,.0f}

PREDICTION ACCURACY
-------------------
Houses within 5% error:   {within_5_pct} ({within_5_pct/len(df)*100:.1f}%)
Houses within 10% error:  {within_10_pct} ({within_10_pct/len(df)*100:.1f}%)
Houses within 15% error:  {within_15_pct} ({within_15_pct/len(df)*100:.1f}%)

TOP 5 BEST PREDICTIONS (Lowest % Error)
----------------------------------------
{df.nsmallest(5, 'AbsPercentError')[['Id', 'Actual', 'Predicted', 'AbsPercentError']].to_string(index=False)}

TOP 5 WORST PREDICTIONS (Highest % Error)
------------------------------------------
{df.nlargest(5, 'AbsPercentError')[['Id', 'Actual', 'Predicted', 'AbsPercentError']].to_string(index=False)}

PRICE RANGE ANALYSIS
--------------------
{df.groupby('PriceRange', observed=False)['AbsPercentError'].agg(['mean', 'count']).to_string()}

INTERPRETATION
--------------
- An R-squared of {metrics['R2']:.3f} means the model explains {metrics['R2']*100:.1f}% of the variance in house prices
- The average prediction error is ${metrics['MAE']:,.0f} or {metrics['MAPE']:.1f}% of the house price
- The model {'slightly overestimates' if metrics['Mean_Error'] > 0 else 'slightly underestimates'} prices on average by ${abs(metrics['Mean_Error']):,.0f}
"""
    
    # Save report
    with open(f"{save_path}/evaluation_report.txt", 'w') as f:
        f.write(report)
    
    # Also save the full results for Excel analysis
    df.to_csv(f"{save_path}/full_evaluation_results.csv", index=False)
    
    print(report)

def main():
    """Run the complete evaluation."""
    print("="*60)
    print("HOUSE PRICE MODEL EVALUATION")
    print("="*60)
    
    # Load validation results
    print("\n1. Loading validation data...")
    df = load_validation_data()
    if df is None:
        return
    print(f"   [OK] Loaded {len(df)} validation samples")
    
    # Calculate metrics
    print("\n2. Calculating metrics...")
    metrics = calculate_metrics(df)
    print("   [OK] Metrics calculated")
    
    # Create visualizations
    print("\n3. Creating visualizations...")
    create_visualizations(df)
    
    # Create summary report
    print("\n4. Creating summary report...")
    create_summary_report(df, metrics)
    print("   [OK] Report saved to evaluation/evaluation_report.txt")
    print("   [OK] Full results saved to evaluation/full_evaluation_results.csv")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("\nFiles created in 'evaluation' folder:")
    print("- evaluation_report.txt (text summary)")
    print("- full_evaluation_results.csv (detailed results for Excel)")
    print("- plots/ (5 visualization images)")
    print("  - actual_vs_predicted.png")
    print("  - residual_plot.png")
    print("  - error_distributions.png")
    print("  - performance_by_price_range.png")
    print("  - qq_plot.png")

if __name__ == "__main__":
    main()