"""
Real-time Stock Data Pipeline
This module provides real-time data fetching capabilities from multiple sources
including Yahoo Finance, Alpha Vantage, and other financial APIs.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class StockDataPipeline:
    """
    A comprehensive data pipeline for fetching real-time stock data
    from multiple sources with fallback mechanisms.
    """
    
    def __init__(self, alpha_vantage_key: Optional[str] = None):
        """
        Initialize the data pipeline.
        
        Args:
            alpha_vantage_key: API key for Alpha Vantage (optional)
        """
        self.alpha_vantage_key = alpha_vantage_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        self.session = requests.Session()
        
    def fetch_yahoo_finance(self, symbol: str, period: str = "2y", interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Fetch data from Yahoo Finance using yfinance.
        
        Args:
            symbol: Stock symbol (e.g., 'MSFT', 'AAPL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            print(f"Fetching {symbol} data from Yahoo Finance...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"No data returned for {symbol}")
                return None
            
            # Standardize column names
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            # Add derived features
            data['Date'] = data.index
            data['Close/Last'] = data['Close']  # For compatibility with existing scripts
            
            print(f"Successfully fetched {len(data)} records for {symbol}")
            return data.reset_index(drop=True)
            
        except Exception as e:
            print(f"Error fetching from Yahoo Finance: {e}")
            return None
    
    def fetch_alpha_vantage(self, symbol: str, function: str = "TIME_SERIES_DAILY") -> Optional[pd.DataFrame]:
        """
        Fetch data from Alpha Vantage API.
        
        Args:
            symbol: Stock symbol
            function: API function ('TIME_SERIES_DAILY', 'TIME_SERIES_WEEKLY', etc.)
        
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not self.alpha_vantage_key:
            print("Alpha Vantage API key not provided")
            return None
        
        try:
            print(f"Fetching {symbol} data from Alpha Vantage...")
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.alpha_vantage_key,
                'outputsize': 'full'
            }
            
            response = self.session.get(url, params=params)
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                print(f"Alpha Vantage Error: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                print(f"Alpha Vantage Note: {data['Note']}")
                return None
            
            # Extract time series data
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
            
            if not time_series_key:
                print("No time series data found")
                return None
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df_data = []
            for date, values in time_series.items():
                row = {
                    'Date': pd.to_datetime(date),
                    'Open': float(values.get('1. open', 0)),
                    'High': float(values.get('2. high', 0)),
                    'Low': float(values.get('3. low', 0)),
                    'Close': float(values.get('4. close', 0)),
                    'Volume': int(values.get('5. volume', 0))
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df = df.sort_values('Date').reset_index(drop=True)
            df['Close/Last'] = df['Close']  # For compatibility
            
            print(f"Successfully fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            print(f"Error fetching from Alpha Vantage: {e}")
            return None
    
    def fetch_real_time_quote(self, symbol: str) -> Optional[Dict]:
        """
        Fetch real-time quote data.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with current price info or None if failed
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            quote = {
                'symbol': symbol,
                'current_price': info.get('currentPrice', info.get('regularMarketPrice')),
                'previous_close': info.get('previousClose'),
                'open': info.get('open', info.get('regularMarketOpen')),
                'day_high': info.get('dayHigh', info.get('regularMarketDayHigh')),
                'day_low': info.get('dayLow', info.get('regularMarketDayLow')),
                'volume': info.get('volume', info.get('regularMarketVolume')),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'timestamp': datetime.now()
            }
            
            return quote
            
        except Exception as e:
            print(f"Error fetching real-time quote: {e}")
            return None
    
    def get_stock_data(self, symbol: str, period: str = "2y", source: str = "auto") -> Optional[pd.DataFrame]:
        """
        Get stock data with automatic fallback between sources.
        
        Args:
            symbol: Stock symbol
            period: Data period
            source: Data source ('yahoo', 'alpha_vantage', 'auto')
        
        Returns:
            DataFrame with stock data or None if all sources fail
        """
        if source == "auto" or source == "yahoo":
            data = self.fetch_yahoo_finance(symbol, period)
            if data is not None:
                return data
        
        if source == "auto" or source == "alpha_vantage":
            data = self.fetch_alpha_vantage(symbol)
            if data is not None:
                return data
        
        print(f"Failed to fetch data for {symbol} from all sources")
        return None
    
    def get_multiple_stocks(self, symbols: List[str], period: str = "2y") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks.
        
        Args:
            symbols: List of stock symbols
            period: Data period
        
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        results = {}
        
        for symbol in symbols:
            print(f"\nFetching data for {symbol}...")
            data = self.get_stock_data(symbol, period)
            if data is not None:
                results[symbol] = data
            else:
                print(f"Failed to fetch data for {symbol}")
            
            # Add delay to respect API limits
            time.sleep(0.1)
        
        return results
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the stock data.
        
        Args:
            data: DataFrame with OHLCV data
        
        Returns:
            DataFrame with additional technical indicators
        """
        df = data.copy()
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # Bollinger Bands
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = rolling_mean + (rolling_std * 2)
        df['BB_Lower'] = rolling_mean - (rolling_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Rate of Change
        df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_Abs'] = df['Close'].diff()
        
        return df
    
    def save_data(self, data: pd.DataFrame, symbol: str, directory: str = "../data") -> str:
        """
        Save fetched data to CSV file.
        
        Args:
            data: DataFrame to save
            symbol: Stock symbol
            directory: Directory to save in
        
        Returns:
            Path to saved file
        """
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_data_{timestamp}.csv"
        filepath = os.path.join(directory, filename)
        
        data.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        return filepath

def get_popular_symbols() -> Dict[str, List[str]]:
    """Get a dictionary of popular stock symbols by category."""
    return {
        'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX'],
        'finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK-B'],
        'healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO'],
        'indices': ['^GSPC', '^DJI', '^IXIC', '^RUT'],  # S&P 500, Dow, NASDAQ, Russell 2000
        'international': ['^NSEI', '^BSESN', '^N225', '^FTSE'],  # NIFTY, SENSEX, Nikkei, FTSE
        'crypto': ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD']
    }

def main():
    """Example usage of the data pipeline."""
    print("Stock Data Pipeline - Real-time Data Fetching")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = StockDataPipeline()
    
    # Example: Fetch Microsoft data
    symbol = "MSFT"
    print(f"\nFetching data for {symbol}...")
    
    data = pipeline.get_stock_data(symbol, period="1y")
    if data is not None:
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
        print("\nFirst few rows:")
        print(data.head())
        
        # Add technical indicators
        data_with_indicators = pipeline.calculate_technical_indicators(data)
        print(f"\nWith indicators shape: {data_with_indicators.shape}")
        print("Available columns:", list(data_with_indicators.columns))
        
        # Save data
        saved_path = pipeline.save_data(data_with_indicators, symbol)
        
        # Get real-time quote
        quote = pipeline.fetch_real_time_quote(symbol)
        if quote:
            print(f"\nReal-time quote for {symbol}:")
            print(f"Current Price: ${quote['current_price']:.2f}")
            print(f"Day Range: ${quote['day_low']:.2f} - ${quote['day_high']:.2f}")
            print(f"Volume: {quote['volume']:,}")
    
    # Example: Fetch multiple stocks
    symbols = ['AAPL', 'GOOGL', 'TSLA']
    print(f"\nFetching data for multiple stocks: {symbols}")
    
    multiple_data = pipeline.get_multiple_stocks(symbols, period="6mo")
    for symbol, data in multiple_data.items():
        print(f"{symbol}: {data.shape[0]} records")

if __name__ == "__main__":
    main() 