#!/usr/bin/env python3
"""
Market Indices Data Pipeline for Stock Prediction
================================================

This module provides comprehensive market indices data including:
- Major market indices (S&P 500, NASDAQ, Dow Jones, etc.)
- Volatility indices (VIX, VXN)
- Sector indices and ETFs
- International market indices
- Bond and commodity indices
- Technical indicators for market sentiment

Author: Stock Prediction Team
Date: 2024
"""

import os
import sys
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import time

import pandas as pd
import numpy as np
import yfinance as yf
import requests

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketIndicesAnalyzer:
    """
    Comprehensive market indices data analyzer for stock prediction enhancement
    """
    
    def __init__(self):
        """
        Initialize market indices analyzer
        """
        # Major US Market Indices
        self.us_indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones Industrial Average',
            '^IXIC': 'NASDAQ Composite',
            '^RUT': 'Russell 2000',
            '^NYA': 'NYSE Composite'
        }
        
        # Volatility Indices
        self.volatility_indices = {
            '^VIX': 'CBOE Volatility Index',
            '^VXN': 'NASDAQ Volatility Index',
            '^RVX': 'Russell 2000 Volatility Index'
        }
        
        # Sector ETFs
        self.sector_etfs = {
            'XLK': 'Technology Select Sector SPDR Fund',
            'XLF': 'Financial Select Sector SPDR Fund',
            'XLV': 'Health Care Select Sector SPDR Fund',
            'XLE': 'Energy Select Sector SPDR Fund',
            'XLI': 'Industrial Select Sector SPDR Fund',
            'XLY': 'Consumer Discretionary Select Sector SPDR Fund',
            'XLP': 'Consumer Staples Select Sector SPDR Fund',
            'XLU': 'Utilities Select Sector SPDR Fund',
            'XLRE': 'Real Estate Select Sector SPDR Fund',
            'XLB': 'Materials Select Sector SPDR Fund',
            'XLC': 'Communication Services Select Sector SPDR Fund'
        }
        
        # All indices combined
        self.all_indices = {
            **self.us_indices,
            **self.volatility_indices,
            **self.sector_etfs
        }
    
    def fetch_index_data(self, 
                        symbol: str, 
                        period: str = "1y",
                        interval: str = "1d") -> pd.DataFrame:
        """
        Fetch data for a single index
        
        Args:
            symbol: Index symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with index data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Add symbol column
            data['Symbol'] = symbol
            data['Name'] = self.all_indices.get(symbol, symbol)
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_indices(self, 
                             symbols: List[str] = None,
                             period: str = "1y",
                             interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple indices
        
        Args:
            symbols: List of symbols to fetch (default: major US indices)
            period: Data period
            interval: Data interval
            
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        if symbols is None:
            symbols = list(self.us_indices.keys())
        
        indices_data = {}
        
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}")
            data = self.fetch_index_data(symbol, period, interval)
            if not data.empty:
                indices_data[symbol] = data
            time.sleep(0.1)  # Rate limiting
        
        return indices_data
    
    def calculate_market_indicators(self, 
                                  indices_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate market-wide indicators from indices data
        
        Args:
            indices_data: Dictionary of indices data
            
        Returns:
            DataFrame with market indicators
        """
        if not indices_data:
            return pd.DataFrame()
        
        # Get the most recent date available across all indices
        latest_dates = []
        for symbol, data in indices_data.items():
            if not data.empty and 'Date' in data.columns:
                latest_dates.append(data['Date'].max())
        
        if not latest_dates:
            return pd.DataFrame()
        
        common_date = min(latest_dates)
        
        # Initialize market indicators DataFrame
        market_indicators = pd.DataFrame()
        
        # Process each index
        for symbol, data in indices_data.items():
            if data.empty:
                continue
            
            # Filter data up to common date
            filtered_data = data[data['Date'] <= common_date].copy()
            
            if len(filtered_data) < 20:  # Need at least 20 days for calculations
                continue
            
            # Calculate returns
            filtered_data['Daily_Return'] = filtered_data['Close'].pct_change()
            filtered_data['Weekly_Return'] = filtered_data['Close'].pct_change(periods=5)
            filtered_data['Monthly_Return'] = filtered_data['Close'].pct_change(periods=20)
            
            # Calculate volatility
            filtered_data['Volatility_5d'] = filtered_data['Daily_Return'].rolling(5).std()
            filtered_data['Volatility_20d'] = filtered_data['Daily_Return'].rolling(20).std()
            
            # Calculate moving averages
            filtered_data['SMA_5'] = filtered_data['Close'].rolling(5).mean()
            filtered_data['SMA_20'] = filtered_data['Close'].rolling(20).mean()
            filtered_data['SMA_50'] = filtered_data['Close'].rolling(50).mean()
            
            # Calculate relative strength
            filtered_data['RSI'] = self._calculate_rsi(filtered_data['Close'])
            
            # Calculate momentum indicators
            filtered_data['Price_Momentum_5d'] = (filtered_data['Close'] / filtered_data['Close'].shift(5) - 1) * 100
            filtered_data['Price_Momentum_20d'] = (filtered_data['Close'] / filtered_data['Close'].shift(20) - 1) * 100
            
            # Get latest values
            latest_row = filtered_data.iloc[-1]
            
            # Create indicator dictionary
            indicators = {
                'Date': latest_row['Date'],
                f'{symbol}_Close': latest_row['Close'],
                f'{symbol}_Daily_Return': latest_row['Daily_Return'],
                f'{symbol}_Weekly_Return': latest_row['Weekly_Return'],
                f'{symbol}_Monthly_Return': latest_row['Monthly_Return'],
                f'{symbol}_Volatility_5d': latest_row['Volatility_5d'],
                f'{symbol}_Volatility_20d': latest_row['Volatility_20d'],
                f'{symbol}_SMA_5': latest_row['SMA_5'],
                f'{symbol}_SMA_20': latest_row['SMA_20'],
                f'{symbol}_SMA_50': latest_row['SMA_50'],
                f'{symbol}_RSI': latest_row['RSI'],
                f'{symbol}_Price_Momentum_5d': latest_row['Price_Momentum_5d'],
                f'{symbol}_Price_Momentum_20d': latest_row['Price_Momentum_20d'],
                f'{symbol}_Volume': latest_row.get('Volume', 0)
            }
            
            # Add to market indicators
            if market_indicators.empty:
                market_indicators = pd.DataFrame([indicators])
            else:
                for key, value in indicators.items():
                    if key != 'Date':
                        market_indicators[key] = value
        
        return market_indicators
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices: Price series
            window: RSI calculation window
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_market_regime_indicators(self, 
                                   indices_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate market regime indicators
        
        Args:
            indices_data: Dictionary of indices data
            
        Returns:
            Dictionary with market regime indicators
        """
        regime_indicators = {}
        
        # VIX-based market regime
        if '^VIX' in indices_data and not indices_data['^VIX'].empty:
            vix_data = indices_data['^VIX']
            latest_vix = vix_data['Close'].iloc[-1]
            
            if latest_vix < 15:
                regime_indicators['vix_regime'] = 1  # Low volatility
            elif latest_vix < 25:
                regime_indicators['vix_regime'] = 2  # Normal volatility
            else:
                regime_indicators['vix_regime'] = 3  # High volatility
            
            regime_indicators['vix_level'] = latest_vix
        
        # Market trend indicators
        if '^GSPC' in indices_data and not indices_data['^GSPC'].empty:
            sp500_data = indices_data['^GSPC']
            
            # Calculate trend indicators
            sp500_data['SMA_50'] = sp500_data['Close'].rolling(50).mean()
            sp500_data['SMA_200'] = sp500_data['Close'].rolling(200).mean()
            
            latest_price = sp500_data['Close'].iloc[-1]
            latest_sma_50 = sp500_data['SMA_50'].iloc[-1]
            latest_sma_200 = sp500_data['SMA_200'].iloc[-1]
            
            # Bull/Bear market indicator
            if latest_price > latest_sma_50 > latest_sma_200:
                regime_indicators['market_trend'] = 1  # Bull market
            elif latest_price < latest_sma_50 < latest_sma_200:
                regime_indicators['market_trend'] = -1  # Bear market
            else:
                regime_indicators['market_trend'] = 0  # Sideways market
        
        # Sector rotation indicators
        sector_performance = {}
        for symbol in self.sector_etfs.keys():
            if symbol in indices_data and not indices_data[symbol].empty:
                sector_data = indices_data[symbol]
                monthly_return = sector_data['Close'].pct_change(periods=20).iloc[-1]
                sector_performance[symbol] = monthly_return
        
        if sector_performance:
            # Find best and worst performing sectors
            best_sector = max(sector_performance, key=sector_performance.get)
            worst_sector = min(sector_performance, key=sector_performance.get)
            
            regime_indicators['best_sector_return'] = sector_performance[best_sector]
            regime_indicators['worst_sector_return'] = sector_performance[worst_sector]
            regime_indicators['sector_dispersion'] = np.std(list(sector_performance.values()))
        
        return regime_indicators
    
    def get_market_features_for_stock(self, 
                                    stock_symbol: str,
                                    period: str = "1y") -> pd.DataFrame:
        """
        Get market features that can be used as inputs for stock prediction
        
        Args:
            stock_symbol: Target stock symbol
            period: Data period
            
        Returns:
            DataFrame with market features aligned with stock data
        """
        # Fetch stock data to get date range
        try:
            stock_ticker = yf.Ticker(stock_symbol)
            stock_data = stock_ticker.history(period=period)
            
            if stock_data.empty:
                logger.error(f"No data found for stock {stock_symbol}")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            return pd.DataFrame()
        
        # Fetch key market indices
        key_indices = ['^GSPC', '^IXIC', '^DJI', '^VIX']
        
        # Add sector ETF based on stock sector (simplified mapping)
        sector_mapping = {
            'AAPL': 'XLK', 'MSFT': 'XLK', 'GOOGL': 'XLK', 'AMZN': 'XLY',
            'TSLA': 'XLY', 'JPM': 'XLF', 'JNJ': 'XLV', 'XOM': 'XLE'
        }
        
        if stock_symbol in sector_mapping:
            key_indices.append(sector_mapping[stock_symbol])
        
        # Fetch indices data
        indices_data = self.fetch_multiple_indices(key_indices, period=period)
        
        # Calculate market indicators
        market_indicators = self.calculate_market_indicators(indices_data)
        
        # Get regime indicators
        regime_indicators = self.get_market_regime_indicators(indices_data)
        
        # Combine all features
        features_dict = {}
        
        if not market_indicators.empty:
            features_dict.update(market_indicators.iloc[0].to_dict())
        
        features_dict.update(regime_indicators)
        
        # Create features DataFrame
        market_features = pd.DataFrame([features_dict])
        
        # Align with stock data dates
        market_features['Date'] = stock_data.index[-1]
        
        return market_features
    
    def get_realtime_market_snapshot(self) -> Dict[str, float]:
        """
        Get real-time market snapshot for immediate use
        
        Returns:
            Dictionary with current market indicators
        """
        snapshot = {}
        
        # Key indices for real-time snapshot
        key_symbols = ['^GSPC', '^IXIC', '^DJI', '^VIX']
        
        for symbol in key_symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get current price and change
                current_price = info.get('regularMarketPrice', 0)
                previous_close = info.get('regularMarketPreviousClose', 0)
                
                if previous_close > 0:
                    daily_change = (current_price - previous_close) / previous_close
                else:
                    daily_change = 0
                
                snapshot[f'{symbol}_price'] = current_price
                snapshot[f'{symbol}_change'] = daily_change
                
            except Exception as e:
                logger.error(f"Error fetching real-time data for {symbol}: {e}")
                snapshot[f'{symbol}_price'] = 0
                snapshot[f'{symbol}_change'] = 0
        
        return snapshot

def main():
    """
    Main function for testing market indices analysis
    """
    # Initialize analyzer
    analyzer = MarketIndicesAnalyzer()
    
    # Test with major US indices
    print("Fetching market indices data...")
    indices_data = analyzer.fetch_multiple_indices(
        symbols=['^GSPC', '^IXIC', '^DJI', '^VIX'],
        period="1mo"
    )
    
    print(f"Fetched data for {len(indices_data)} indices")
    
    # Calculate market indicators
    print("\nCalculating market indicators...")
    market_indicators = analyzer.calculate_market_indicators(indices_data)
    
    if not market_indicators.empty:
        print("Market Indicators:")
        print("=" * 50)
        for col in market_indicators.columns:
            if col != 'Date':
                print(f"{col}: {market_indicators[col].iloc[0]:.4f}")
    
    # Get regime indicators
    print("\nMarket Regime Indicators:")
    print("=" * 50)
    regime_indicators = analyzer.get_market_regime_indicators(indices_data)
    for key, value in regime_indicators.items():
        print(f"{key}: {value}")
    
    # Get real-time snapshot
    print("\nReal-time Market Snapshot:")
    print("=" * 50)
    snapshot = analyzer.get_realtime_market_snapshot()
    for key, value in snapshot.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main() 