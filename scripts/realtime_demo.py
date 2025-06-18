"""
Real-time Stock Data Pipeline Demo
This script demonstrates all real-time capabilities including data fetching,
prediction, and live monitoring across different time frames.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import argparse
from data_pipeline import StockDataPipeline, get_popular_symbols
from realtime_predictor import RealTimeStockPredictor
from intraday_realtime import RealTimeIntradayAnalyzer
import warnings

warnings.filterwarnings('ignore')

class RealTimeStockDemo:
    """
    Comprehensive demo of real-time stock analysis capabilities.
    """
    
    def __init__(self):
        """Initialize the demo."""
        self.pipeline = StockDataPipeline()
        
    def demo_data_fetching(self, symbols: list = None):
        """
        Demonstrate real-time data fetching capabilities.
        
        Args:
            symbols: List of symbols to demonstrate
        """
        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        
        print("🚀 Real-time Data Fetching Demo")
        print("=" * 50)
        
        for symbol in symbols:
            print(f"\n📊 Fetching data for {symbol}...")
            
            try:
                # Get historical data
                data = self.pipeline.get_stock_data(symbol, period="1mo")
                if data is not None:
                    print(f"   ✅ Historical: {len(data)} records")
                    print(f"   📅 Range: {data['Date'].min()} to {data['Date'].max()}")
                
                # Get real-time quote
                quote = self.pipeline.fetch_real_time_quote(symbol)
                if quote:
                    print(f"   💰 Current Price: ${quote['current_price']:.2f}")
                    print(f"   📈 Day Change: {((quote['current_price'] - quote['previous_close']) / quote['previous_close'] * 100):+.2f}%")
                    print(f"   📊 Volume: {quote['volume']:,}")
                
                # Add technical indicators
                if data is not None:
                    data_with_indicators = self.pipeline.calculate_technical_indicators(data)
                    print(f"   🔧 With Indicators: {data_with_indicators.shape[1]} columns")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    def demo_prediction_models(self, symbol: str = "MSFT"):
        """
        Demonstrate different prediction models.
        
        Args:
            symbol: Symbol to predict
        """
        print(f"\n🤖 Prediction Models Demo - {symbol}")
        print("=" * 50)
        
        models = ['lstm', 'gru', 'dense']
        results = {}
        
        for model_type in models:
            print(f"\n🏗️ Training {model_type.upper()} model...")
            
            try:
                predictor = RealTimeStockPredictor(symbol, model_type)
                
                # Train with shorter period for demo
                training_results = predictor.train_model(
                    period="1y", 
                    epochs=20,  # Reduced for demo
                    feature_set="enhanced"
                )
                
                # Make predictions
                predictions = predictor.predict_future(days=3)
                
                results[model_type] = {
                    'training': training_results,
                    'predictions': predictions
                }
                
                print(f"   📈 R² Score: {training_results['r2']:.4f}")
                print(f"   📉 RMSE: {training_results['rmse']:.4f}")
                print(f"   🔮 Next 3 days: {[f'${p:.2f}' for p in predictions['predictions']]}")
                
            except Exception as e:
                print(f"   ❌ Error training {model_type}: {e}")
        
        # Compare models
        if results:
            print(f"\n📊 Model Comparison:")
            print(f"{'Model':<8} {'R² Score':<10} {'RMSE':<12} {'Next Day':<10}")
            print("-" * 45)
            
            for model_type, result in results.items():
                r2 = result['training']['r2']
                rmse = result['training']['rmse']
                next_day = result['predictions']['predictions'][0]
                print(f"{model_type.upper():<8} {r2:<10.4f} {rmse:<12.4f} ${next_day:<9.2f}")
    
    def demo_intraday_analysis(self, symbol: str = "AAPL"):
        """
        Demonstrate intraday analysis capabilities.
        
        Args:
            symbol: Symbol to analyze
        """
        print(f"\n📈 Intraday Analysis Demo - {symbol}")
        print("=" * 50)
        
        try:
            analyzer = RealTimeIntradayAnalyzer(symbol)
            
            # Fetch intraday data
            data = analyzer.fetch_intraday_data(period="1d", interval="15m")
            
            # Generate signals
            data_with_signals = analyzer.generate_trading_signals(data)
            
            # Show current status
            status = analyzer.get_current_market_status()
            if status:
                print(f"\n💰 Current Market Status:")
                print(f"   Price: ${status['current_price']:.2f}")
                print(f"   Change: {status['day_change_pct']:+.2f}%")
                print(f"   Volume: {status['volume']:,}")
                
                if status['rsi']:
                    rsi_status = "Overbought" if status['rsi'] > 70 else "Oversold" if status['rsi'] < 30 else "Neutral"
                    print(f"   RSI: {status['rsi']:.1f} ({rsi_status})")
            
            # Show recent signals
            signals = data_with_signals[data_with_signals['Signal'] != 0].tail(3)
            if not signals.empty:
                print(f"\n📊 Recent Trading Signals:")
                for _, signal in signals.iterrows():
                    signal_type = "🟢 BUY" if signal['Signal'] == 1 else "🔴 SELL"
                    print(f"   {signal['Date']}: {signal_type} - {signal['Signal_Reason']}")
                    print(f"      Strength: {signal['Signal_Strength']}/100")
            
            # Technical analysis summary
            latest = data_with_signals.iloc[-1]
            print(f"\n🔧 Technical Analysis Summary:")
            print(f"   Price vs VWAP: {'Above ✅' if latest['Close'] > latest['VWAP'] else 'Below ❌'}")
            print(f"   RSI: {latest['RSI']:.1f}")
            print(f"   MACD: {latest['MACD']:.4f}")
            print(f"   Volume Ratio: {latest['Volume_Ratio']:.2f}x")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    def demo_multi_symbol_monitoring(self, symbols: list = None):
        """
        Demonstrate monitoring multiple symbols simultaneously.
        
        Args:
            symbols: List of symbols to monitor
        """
        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
        
        print(f"\n👀 Multi-Symbol Real-time Monitoring")
        print("=" * 50)
        
        print(f"Monitoring {len(symbols)} symbols: {', '.join(symbols)}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Create header
        print(f"{'Symbol':<8} {'Price':<10} {'Change':<12} {'Volume':<15} {'RSI':<8} {'Status'}")
        print("-" * 70)
        
        for symbol in symbols:
            try:
                # Get quote
                quote = self.pipeline.fetch_real_time_quote(symbol)
                if not quote:
                    continue
                
                # Get technical data
                data = self.pipeline.get_stock_data(symbol, period="1mo")
                if data is not None:
                    data_with_indicators = self.pipeline.calculate_technical_indicators(data)
                    latest_rsi = data_with_indicators['RSI'].iloc[-1]
                else:
                    latest_rsi = None
                
                # Calculate change
                change_pct = ((quote['current_price'] - quote['previous_close']) / quote['previous_close']) * 100
                
                # Determine status
                status = "🟢" if change_pct > 0 else "🔴" if change_pct < 0 else "⚪"
                if latest_rsi:
                    if latest_rsi > 70:
                        status += " 🔥"  # Overbought
                    elif latest_rsi < 30:
                        status += " ❄️"   # Oversold
                
                # Format output
                price_str = f"${quote['current_price']:.2f}"
                change_str = f"{change_pct:+.2f}%"
                volume_str = f"{quote['volume']:,}" if quote['volume'] else "N/A"
                rsi_str = f"{latest_rsi:.1f}" if latest_rsi else "N/A"
                
                print(f"{symbol:<8} {price_str:<10} {change_str:<12} {volume_str:<15} {rsi_str:<8} {status}")
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"{symbol:<8} Error: {str(e)[:50]}")
    
    def demo_data_sources_comparison(self, symbol: str = "AAPL"):
        """
        Demonstrate different data sources and their capabilities.
        
        Args:
            symbol: Symbol to test
        """
        print(f"\n🔄 Data Sources Comparison - {symbol}")
        print("=" * 50)
        
        sources = [
            ("Yahoo Finance", "yahoo"),
            ("Alpha Vantage", "alpha_vantage")
        ]
        
        for source_name, source_code in sources:
            print(f"\n📡 Testing {source_name}...")
            
            try:
                start_time = time.time()
                
                if source_code == "yahoo":
                    data = self.pipeline.fetch_yahoo_finance(symbol, period="1mo")
                elif source_code == "alpha_vantage":
                    data = self.pipeline.fetch_alpha_vantage(symbol)
                
                end_time = time.time()
                
                if data is not None:
                    print(f"   ✅ Success: {len(data)} records")
                    print(f"   ⏱️ Fetch time: {end_time - start_time:.2f} seconds")
                    print(f"   📅 Date range: {data['Date'].min()} to {data['Date'].max()}")
                    print(f"   💰 Latest price: ${data['Close'].iloc[-1]:.2f}")
                else:
                    print(f"   ❌ Failed to fetch data")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    def run_comprehensive_demo(self):
        """Run a comprehensive demonstration of all capabilities."""
        print("🎯 Comprehensive Real-time Stock Analysis Demo")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. Data fetching demo
        self.demo_data_fetching(['AAPL', 'MSFT', 'TSLA'])
        
        # 2. Data sources comparison
        self.demo_data_sources_comparison('AAPL')
        
        # 3. Multi-symbol monitoring
        self.demo_multi_symbol_monitoring(['AAPL', 'MSFT', 'GOOGL', 'TSLA'])
        
        # 4. Intraday analysis
        self.demo_intraday_analysis('AAPL')
        
        # 5. Prediction models (reduced for demo)
        print(f"\n⚠️ Prediction demo skipped (takes ~5-10 minutes)")
        print(f"   Run with --full-demo to include model training")
        
        print(f"\n✅ Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Real-time Stock Data Pipeline Demo')
    parser.add_argument('--demo', '-d', default='comprehensive',
                       choices=['data', 'prediction', 'intraday', 'monitoring', 'sources', 'comprehensive'],
                       help='Demo type to run')
    parser.add_argument('--symbol', '-s', default='AAPL',
                       help='Stock symbol for demos (default: AAPL)')
    parser.add_argument('--symbols', nargs='+', 
                       help='Multiple symbols for monitoring demo')
    parser.add_argument('--full-demo', action='store_true',
                       help='Run full demo including model training (takes longer)')
    parser.add_argument('--list-symbols', action='store_true',
                       help='List popular symbols and exit')
    
    args = parser.parse_args()
    
    if args.list_symbols:
        symbols = get_popular_symbols()
        print("📊 Popular Stock Symbols by Category:")
        for category, symbol_list in symbols.items():
            print(f"\n{category.upper()}:")
            for symbol in symbol_list:
                print(f"  {symbol}")
        return
    
    demo = RealTimeStockDemo()
    
    try:
        if args.demo == 'data':
            symbols = args.symbols or [args.symbol]
            demo.demo_data_fetching(symbols)
            
        elif args.demo == 'prediction':
            demo.demo_prediction_models(args.symbol)
            
        elif args.demo == 'intraday':
            demo.demo_intraday_analysis(args.symbol)
            
        elif args.demo == 'monitoring':
            symbols = args.symbols or ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
            demo.demo_multi_symbol_monitoring(symbols)
            
        elif args.demo == 'sources':
            demo.demo_data_sources_comparison(args.symbol)
            
        elif args.demo == 'comprehensive':
            demo.run_comprehensive_demo()
            if args.full_demo:
                print(f"\n🤖 Running prediction models demo...")
                demo.demo_prediction_models('MSFT')
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("Make sure you have internet connection and required packages installed.")

if __name__ == "__main__":
    main() 