"""
Intraday Stock Analysis
This script analyzes intraday stock price changes by comparing different datasets
and visualizing the percentage changes between opening and closing prices.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(data_path, stockdata_path):
    """Load and validate the two datasets."""
    try:
        # Load datasets
        data = pd.read_csv(data_path)
        stockdata = pd.read_csv(stockdata_path)
        
        print(f"Data 1 loaded: {data.shape}")
        print(f"Data 2 loaded: {stockdata.shape}")
        print(f"Data 1 columns: {list(data.columns)}")
        print(f"Data 2 columns: {list(stockdata.columns)}")
        
        return data, stockdata
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def calculate_percentage_changes(data, stockdata, num_periods=365):
    """Calculate percentage changes for both datasets."""
    try:
        # Extract data for first dataset
        # Note: The original code had an error - it used 'Date' for opening and 'Low' for closing
        # This seems incorrect, so I'll fix it to use proper OHLC columns
        
        if 'Open' in data.columns and 'Close' in data.columns:
            open_data = data['Open'].tolist()
            close_data = data['Close'].tolist()
        elif 'open' in data.columns and 'close' in data.columns:
            open_data = data['open'].tolist()
            close_data = data['close'].tolist()
        else:
            # Fallback to original logic if columns don't match expected names
            print("Warning: Using fallback column mapping")
            open_data = data.iloc[:, 1].tolist()  # Assuming second column is open-like
            close_data = data.iloc[:, -2].tolist()  # Assuming second-to-last is close-like
        
        # Extract data for second dataset
        if 'open' in stockdata.columns and 'close' in stockdata.columns:
            open_data_snp = stockdata['open'].tolist()
            close_data_snp = stockdata['close'].tolist()
        elif 'Open' in stockdata.columns and 'Close' in stockdata.columns:
            open_data_snp = stockdata['Open'].tolist()
            close_data_snp = stockdata['Close'].tolist()
        else:
            print("Error: Cannot find open/close columns in stockdata")
            return None, None
        
        # Calculate percentage changes
        change_data1 = []
        change_data2 = []
        
        # Use minimum of available data or requested periods
        periods = min(num_periods, len(open_data), len(close_data), 
                     len(open_data_snp), len(close_data_snp))
        
        print(f"Calculating changes for {periods} periods")
        
        for i in range(periods):
            # Calculate percentage change: (close - open) / open
            if open_data[i] != 0:
                change1 = (close_data[i] - open_data[i]) / open_data[i]
                change_data1.append(change1)
            else:
                change_data1.append(0)
            
            if open_data_snp[i] != 0:
                change2 = (close_data_snp[i] - open_data_snp[i]) / open_data_snp[i]
                change_data2.append(change2)
            else:
                change_data2.append(0)
        
        return change_data1, change_data2
        
    except Exception as e:
        print(f"Error calculating percentage changes: {e}")
        return None, None

def plot_changes(change_data1, change_data2, dataset1_name="Dataset 1", dataset2_name="Dataset 2"):
    """Plot the percentage changes for both datasets."""
    plt.figure(figsize=(14, 8))
    
    # Plot both datasets
    plt.plot(change_data2, color='pink', linewidth=1.5, label=dataset2_name, alpha=0.8)
    plt.plot(change_data1, color='blue', linewidth=1.5, label=dataset1_name, alpha=0.8)
    
    # Add zero line for reference
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # Formatting
    plt.title('Intraday Stock Price Changes Comparison', fontsize=16)
    plt.xlabel('Time Period (Days)', fontsize=12)
    plt.ylabel('Percentage Change (Open to Close)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    stats1 = {
        'mean': np.mean(change_data1),
        'std': np.std(change_data1),
        'min': np.min(change_data1),
        'max': np.max(change_data1)
    }
    
    stats2 = {
        'mean': np.mean(change_data2),
        'std': np.std(change_data2),
        'min': np.min(change_data2),
        'max': np.max(change_data2)
    }
    
    # Add text box with statistics
    stats_text = f"{dataset1_name}: μ={stats1['mean']:.4f}, σ={stats1['std']:.4f}\n"
    stats_text += f"{dataset2_name}: μ={stats2['mean']:.4f}, σ={stats2['std']:.4f}"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return stats1, stats2

def print_analysis(stats1, stats2, dataset1_name="Dataset 1", dataset2_name="Dataset 2"):
    """Print detailed analysis of the two datasets."""
    print("\n" + "="*50)
    print("INTRADAY ANALYSIS RESULTS")
    print("="*50)
    
    print(f"\n{dataset1_name} Statistics:")
    print(f"  Mean daily change: {stats1['mean']:.4f} ({stats1['mean']*100:.2f}%)")
    print(f"  Standard deviation: {stats1['std']:.4f}")
    print(f"  Minimum change: {stats1['min']:.4f} ({stats1['min']*100:.2f}%)")
    print(f"  Maximum change: {stats1['max']:.4f} ({stats1['max']*100:.2f}%)")
    
    print(f"\n{dataset2_name} Statistics:")
    print(f"  Mean daily change: {stats2['mean']:.4f} ({stats2['mean']*100:.2f}%)")
    print(f"  Standard deviation: {stats2['std']:.4f}")
    print(f"  Minimum change: {stats2['min']:.4f} ({stats2['min']*100:.2f}%)")
    print(f"  Maximum change: {stats2['max']:.4f} ({stats2['max']*100:.2f}%)")
    
    # Comparison
    print(f"\nComparison:")
    if abs(stats1['mean']) > abs(stats2['mean']):
        print(f"  {dataset1_name} has higher average volatility")
    else:
        print(f"  {dataset2_name} has higher average volatility")
    
    if stats1['std'] > stats2['std']:
        print(f"  {dataset1_name} is more volatile (higher std dev)")
    else:
        print(f"  {dataset2_name} is more volatile (higher std dev)")

def main():
    """Main execution function."""
    print("Intraday Stock Analysis")
    print("=" * 30)
    
    # File paths
    data_path = '../data/data.csv'
    stockdata_path = '../data/stockdata.csv'
    
    # Check if files exist
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        return
    
    if not os.path.exists(stockdata_path):
        print(f"Error: {stockdata_path} not found")
        return
    
    # Load data
    data, stockdata = load_data(data_path, stockdata_path)
    if data is None or stockdata is None:
        return
    
    # Calculate percentage changes
    change_data1, change_data2 = calculate_percentage_changes(data, stockdata, num_periods=365)
    if change_data1 is None or change_data2 is None:
        return
    
    # Plot and analyze
    stats1, stats2 = plot_changes(change_data1, change_data2, 
                                 dataset1_name="Primary Dataset", 
                                 dataset2_name="S&P 500 Dataset")
    
    # Print detailed analysis
    print_analysis(stats1, stats2, "Primary Dataset", "S&P 500 Dataset")

if __name__ == "__main__":
    main()


