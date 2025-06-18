"""
Real-time Intraday Stock Trading Analysis
This script provides real-time intraday analysis with trading signals
using live market data and technical indicators.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import time
import argparse
import warnings
from data_pipeline import StockDataPipeline

warnings.filterwarnings('ignore')

class RealTimeIntradayAnalyzer:
    """
    Real-time intraday trading analyzer with live data feeds.
    """
    
    def __init__(self, symbol: str = "MSFT"):
        """
        Initialize the analyzer.
        
        Args:
            symbol: Stock symbol to analyze
        """
        self.symbol = symbol.upper()
        self.pipeline = StockDataPipeline()
        self.current_position = None  # 'long', 'short', or None
        self.entry_price = None
        self.trades = []
        
    def fetch_intraday_data(self, period: str = "1d", interval: str = "5m") -> pd.DataFrame:
        """
        Fetch real-time intraday data.
        
        Args:
            period: Data period ('1d', '5d', '1mo')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '1h')
            
        Returns:
            DataFrame with intraday data
        """
        print(f"📊 Fetching {interval} intraday data for {self.symbol}...")
        
        data = self.pipeline.fetch_yahoo_finance(self.symbol, period=period, interval=interval)
        if data is None:
            raise ValueError(f"Failed to fetch intraday data for {self.symbol}")
        
        # Add technical indicators for intraday
        data = self.calculate_intraday_indicators(data)
        
        print(f"✅ Fetched {len(data)} {interval} candles")
        return data
    
    def calculate_intraday_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate intraday-specific technical indicators.
        
        Args:
            data: Raw intraday data
            
        Returns:
            DataFrame with technical indicators
        """
        df = data.copy()
        
        # Short-term moving averages for intraday
        df['SMA_9'] = df['Close'].rolling(window=9).mean()
        df['SMA_21'] = df['Close'].rolling(window=21).mean()
        df['EMA_9'] = df['Close'].ewm(span=9).mean()
        df['EMA_21'] = df['Close'].ewm(span=21).mean()
        
        # VWAP (Volume Weighted Average Price)
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        # Bollinger Bands (shorter period for intraday)
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = rolling_mean + (rolling_std * 2)
        df['BB_Lower'] = rolling_mean - (rolling_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # RSI (shorter period)
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (faster settings for intraday)
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Price momentum
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Momentum'] = df['Close'].pct_change(periods=5)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Support and Resistance levels
        df['High_20'] = df['High'].rolling(window=20).max()
        df['Low_20'] = df['Low'].rolling(window=20).min()
        
        return df
    
    def generate_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators.
        
        Args:
            data: DataFrame with technical indicators
            
        Returns:
            DataFrame with trading signals
        """
        df = data.copy()
        
        # Initialize signal columns
        df['Signal'] = 0  # 0: hold, 1: buy, -1: sell
        df['Signal_Strength'] = 0  # Signal strength (0-100)
        df['Signal_Reason'] = ''
        
        for i in range(1, len(df)):
            signals = []
            strength = 0
            reasons = []
            
            # Moving Average Crossover
            if (df['EMA_9'].iloc[i] > df['EMA_21'].iloc[i] and 
                df['EMA_9'].iloc[i-1] <= df['EMA_21'].iloc[i-1]):
                signals.append(1)
                strength += 20
                reasons.append('EMA Bullish Cross')
            elif (df['EMA_9'].iloc[i] < df['EMA_21'].iloc[i] and 
                  df['EMA_9'].iloc[i-1] >= df['EMA_21'].iloc[i-1]):
                signals.append(-1)
                strength += 20
                reasons.append('EMA Bearish Cross')
            
            # VWAP Strategy
            if df['Close'].iloc[i] > df['VWAP'].iloc[i] and df['Volume_Ratio'].iloc[i] > 1.2:
                signals.append(1)
                strength += 15
                reasons.append('Above VWAP + High Volume')
            elif df['Close'].iloc[i] < df['VWAP'].iloc[i] and df['Volume_Ratio'].iloc[i] > 1.2:
                signals.append(-1)
                strength += 15
                reasons.append('Below VWAP + High Volume')
            
            # RSI Oversold/Overbought
            if df['RSI'].iloc[i] < 30 and df['RSI'].iloc[i-1] >= 30:
                signals.append(1)
                strength += 25
                reasons.append('RSI Oversold Recovery')
            elif df['RSI'].iloc[i] > 70 and df['RSI'].iloc[i-1] <= 70:
                signals.append(-1)
                strength += 25
                reasons.append('RSI Overbought')
            
            # Bollinger Bands
            if (df['Close'].iloc[i] > df['BB_Lower'].iloc[i] and 
                df['Close'].iloc[i-1] <= df['BB_Lower'].iloc[i-1]):
                signals.append(1)
                strength += 20
                reasons.append('BB Lower Band Bounce')
            elif (df['Close'].iloc[i] < df['BB_Upper'].iloc[i] and 
                  df['Close'].iloc[i-1] >= df['BB_Upper'].iloc[i-1]):
                signals.append(-1)
                strength += 20
                reasons.append('BB Upper Band Rejection')
            
            # MACD
            if (df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i] and 
                df['MACD'].iloc[i-1] <= df['MACD_Signal'].iloc[i-1]):
                signals.append(1)
                strength += 15
                reasons.append('MACD Bullish Cross')
            elif (df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i] and 
                  df['MACD'].iloc[i-1] >= df['MACD_Signal'].iloc[i-1]):
                signals.append(-1)
                strength += 15
                reasons.append('MACD Bearish Cross')
            
            # Aggregate signals
            if signals:
                signal_sum = sum(signals)
                if signal_sum > 0:
                    df.loc[i, 'Signal'] = 1
                elif signal_sum < 0:
                    df.loc[i, 'Signal'] = -1
                
                df.loc[i, 'Signal_Strength'] = min(strength, 100)
                df.loc[i, 'Signal_Reason'] = '; '.join(reasons)
        
        return df
    
    def get_current_market_status(self) -> dict:
        """
        Get current market status and live quote.
        
        Returns:
            Dictionary with current market information
        """
        quote = self.pipeline.fetch_real_time_quote(self.symbol)
        if not quote:
            return None
        
        # Get recent data for context
        recent_data = self.fetch_intraday_data(period="1d", interval="5m")
        if recent_data.empty:
            return quote
        
        latest = recent_data.iloc[-1]
        
        status = {
            'symbol': self.symbol,
            'current_price': quote['current_price'],
            'previous_close': quote['previous_close'],
            'day_change': quote['current_price'] - quote['previous_close'],
            'day_change_pct': ((quote['current_price'] - quote['previous_close']) / quote['previous_close']) * 100,
            'day_high': quote['day_high'],
            'day_low': quote['day_low'],
            'volume': quote['volume'],
            'timestamp': quote['timestamp'],
            'rsi': latest['RSI'] if 'RSI' in latest else None,
            'above_vwap': latest['Close'] > latest['VWAP'] if 'VWAP' in latest else None,
            'bb_position': latest['BB_Position'] if 'BB_Position' in latest else None
        }
        
        return status

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Real-time Intraday Trading Analyzer')
    parser.add_argument('--symbol', '-s', default='MSFT',
                       help='Stock symbol to analyze (default: MSFT)')
    parser.add_argument('--interval', '-i', default='5m',
                       choices=['1m', '2m', '5m', '15m', '30m', '1h'],
                       help='Data interval (default: 5m)')
    parser.add_argument('--period', '-p', default='1d',
                       choices=['1d', '5d', '1mo'],
                       help='Data period (default: 1d)')
    
    args = parser.parse_args()
    
    print(f"📊 Real-time Intraday Analyzer")
    print(f"🎯 Symbol: {args.symbol}")
    print(f"⏱️ Interval: {args.interval}")
    print("=" * 50)
    
    try:
        analyzer = RealTimeIntradayAnalyzer(args.symbol)
        
        # Analysis mode
        data = analyzer.fetch_intraday_data(period=args.period, interval=args.interval)
        data_with_signals = analyzer.generate_trading_signals(data)
        
        # Show current market status
        status = analyzer.get_current_market_status()
        if status:
            print(f"\n💰 Current Status:")
            print(f"   Price: ${status['current_price']:.2f}")
            print(f"   Change: {status['day_change_pct']:+.2f}%")
            print(f"   Volume: {status['volume']:,}")
        
        # Show recent signals
        recent_signals = data_with_signals[data_with_signals['Signal'] != 0].tail(5)
        if not recent_signals.empty:
            print(f"\n📊 Recent Trading Signals:")
            for _, signal in recent_signals.iterrows():
                signal_type = "BUY" if signal['Signal'] == 1 else "SELL"
                print(f"   {signal['Date']}: {signal_type} - {signal['Signal_Reason']} (Strength: {signal['Signal_Strength']})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have internet connection and required packages installed.")

if __name__ == "__main__":
    main() 