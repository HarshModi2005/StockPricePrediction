"""
Stock Data Visualization Utility
This script provides utilities for plotting and visualizing stock price data.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import argparse

def clean_price_data(price_series):
    """Clean price data by removing dollar signs and converting to float."""
    cleaned_prices = []
    for price in price_series:
        if isinstance(price, str):
            # Remove dollar sign and convert to float
            cleaned_price = float(price.replace('$', ''))
        else:
            cleaned_price = float(price)
        cleaned_prices.append(cleaned_price)
    return np.array(cleaned_prices)

def load_and_clean_data(file_path):
    """Load stock data and clean price columns."""
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Clean price columns
        price_columns = ['Open', 'High', 'Low', 'Close/Last']
        for col in price_columns:
            if col in data.columns:
                if data[col].dtype == 'object':
                    data[col] = data[col].str.replace('$', '').astype(float)
        
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def plot_price_data(data, price_column='High', scale=True, title=None):
    """Plot stock price data with optional scaling."""
    if price_column not in data.columns:
        print(f"Column '{price_column}' not found in data")
        return
    
    prices = data[price_column].values
    
    if scale:
        scaler = MinMaxScaler()
        prices = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        ylabel = f'Scaled {price_column} Price'
    else:
        ylabel = f'{price_column} Price ($)'
    
    plt.figure(figsize=(12, 6))
    plt.plot(prices, linewidth=1.5, alpha=0.8)
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Stock {price_column} Price {"(Scaled)" if scale else ""}')
    
    plt.xlabel('Time Period')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_multiple_prices(data, columns=['Open', 'High', 'Low', 'Close/Last'], scale=True):
    """Plot multiple price columns on the same chart."""
    plt.figure(figsize=(14, 8))
    
    for col in columns:
        if col in data.columns:
            prices = data[col].values
            
            if scale:
                scaler = MinMaxScaler()
                prices = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
            
            plt.plot(prices, label=col, linewidth=1.5, alpha=0.8)
    
    plt.title(f'Stock Price Comparison {"(Scaled)" if scale else ""}')
    plt.xlabel('Time Period')
    plt.ylabel('Price' + (' (Scaled)' if scale else ' ($)'))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_volume(data):
    """Plot trading volume."""
    if 'Volume' not in data.columns:
        print("Volume column not found in data")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(data['Volume'], linewidth=1, alpha=0.7, color='orange')
    plt.title('Trading Volume')
    plt.xlabel('Time Period')
    plt.ylabel('Volume')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Plot stock price data')
    parser.add_argument('--file', default='../data/HistoricalData.csv', 
                       help='Path to CSV file')
    parser.add_argument('--column', default='High', 
                       help='Price column to plot (Open, High, Low, Close/Last)')
    parser.add_argument('--scale', action='store_true', 
                       help='Scale the data using MinMaxScaler')
    parser.add_argument('--multiple', action='store_true', 
                       help='Plot multiple price columns')
    parser.add_argument('--volume', action='store_true', 
                       help='Plot trading volume')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found")
        print("Please ensure the data file exists")
        return
    
    # Load data
    data = load_and_clean_data(args.file)
    if data is None:
        return
    
    print(f"Data preview:")
    print(data.head())
    print(f"\nData info:")
    print(data.info())
    
    # Plot based on arguments
    if args.volume:
        plot_volume(data)
    elif args.multiple:
        plot_multiple_prices(data, scale=args.scale)
    else:
        plot_price_data(data, args.column, scale=args.scale)

if __name__ == "__main__":
    # If run without arguments, show default high price plot
    if len(os.sys.argv) == 1:
        print("Stock Data Visualization")
        print("=" * 30)
        
        data_path = '../data/HistoricalData.csv'
        if os.path.exists(data_path):
            data = load_and_clean_data(data_path)
            if data is not None:
                plot_price_data(data, 'High', scale=True, 
                              title='Microsoft Stock - High Price (Scaled)')
        else:
            print(f"Default data file not found: {data_path}")
            print("Use --file argument to specify data file path")
    else:
        main()