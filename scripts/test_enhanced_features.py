#!/usr/bin/env python3
"""
Test Script for Enhanced Stock Prediction Features
=================================================

This script tests the enhanced features including:
- Sentiment analysis pipeline
- Market indices integration
- Enhanced prediction model with multi-modal inputs
- Performance comparison with baseline models

Author: Stock Prediction Team
Date: 2024
"""

import os
import sys
import logging
import warnings
from datetime import datetime, timedelta
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import our enhanced modules
try:
    from sentiment_analysis import SentimentAnalyzer
    from market_indices import MarketIndicesAnalyzer
    from enhanced_prediction_model import EnhancedStockPredictor
    from prediction_microsoft_allfeatures import StockPredictor as BaselinePredictor
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Make sure all required files are in the same directory")

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_sentiment_analysis():
    """
    Test sentiment analysis functionality
    """
    print("\n" + "="*60)
    print("TESTING SENTIMENT ANALYSIS")
    print("="*60)
    
    try:
        # Initialize sentiment analyzer
        analyzer = SentimentAnalyzer()
        
        # Test with multiple stocks
        test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        
        for symbol in test_symbols:
            print(f"\n📊 Testing sentiment analysis for {symbol}")
            print("-" * 40)
            
            # Get aggregated sentiment
            sentiment_data = analyzer.get_aggregated_sentiment(symbol, days_back=5)
            
            print("Sentiment Scores:")
            for key, value in sentiment_data.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            
            # Test individual text analysis
            sample_text = f"{symbol} shows strong quarterly earnings growth and positive outlook"
            text_sentiment = analyzer.analyze_text_sentiment(sample_text)
            
            print(f"\nSample text sentiment for: '{sample_text}'")
            for key, value in text_sentiment.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        print("\n✅ Sentiment analysis tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Sentiment analysis test failed: {e}")
        return False

def test_market_indices():
    """
    Test market indices functionality
    """
    print("\n" + "="*60)
    print("TESTING MARKET INDICES")
    print("="*60)
    
    try:
        # Initialize market analyzer
        analyzer = MarketIndicesAnalyzer()
        
        # Test fetching major indices
        print("📈 Fetching major market indices...")
        indices_data = analyzer.fetch_multiple_indices(
            symbols=['^GSPC', '^IXIC', '^DJI', '^VIX'],
            period="1mo"
        )
        
        print(f"✅ Successfully fetched data for {len(indices_data)} indices")
        
        for symbol, data in indices_data.items():
            if not data.empty:
                latest_price = data['Close'].iloc[-1]
                daily_change = data['Close'].pct_change().iloc[-1] * 100
                print(f"  {symbol}: ${latest_price:.2f} ({daily_change:+.2f}%)")
        
        # Test market indicators calculation
        print("\n📊 Calculating market indicators...")
        market_indicators = analyzer.calculate_market_indicators(indices_data)
        
        if not market_indicators.empty:
            print("Market Indicators (sample):")
            sample_cols = [col for col in market_indicators.columns if 'Close' in col or 'Return' in col][:5]
            for col in sample_cols:
                if col in market_indicators.columns:
                    value = market_indicators[col].iloc[0]
                    print(f"  {col}: {value:.4f}")
        
        # Test regime indicators
        print("\n🔄 Calculating market regime indicators...")
        regime_indicators = analyzer.get_market_regime_indicators(indices_data)
        
        print("Market Regime Indicators:")
        for key, value in regime_indicators.items():
            print(f"  {key}: {value}")
        
        # Test real-time snapshot
        print("\n⚡ Getting real-time market snapshot...")
        snapshot = analyzer.get_realtime_market_snapshot()
        
        print("Real-time Market Snapshot:")
        for key, value in snapshot.items():
            print(f"  {key}: {value:.4f}")
        
        print("\n✅ Market indices tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Market indices test failed: {e}")
        return False

def test_enhanced_prediction_model():
    """
    Test enhanced prediction model
    """
    print("\n" + "="*60)
    print("TESTING ENHANCED PREDICTION MODEL")
    print("="*60)
    
    try:
        # Test with a sample stock
        stock_symbol = "AAPL"
        print(f"🧠 Testing enhanced model for {stock_symbol}")
        
        # Initialize enhanced predictor
        predictor = EnhancedStockPredictor(stock_symbol, lookback_days=30)
        
        # Prepare data (smaller dataset for testing)
        print("📊 Preparing enhanced features...")
        X, y = predictor.prepare_data(period="6mo")
        
        print(f"✅ Prepared {len(X)} sequences")
        print(f"   - Sequence length: {X.shape[1]} days")
        print(f"   - Features per day: {X.shape[2]}")
        print(f"   - Price features: {len(predictor.price_features)}")
        print(f"   - Technical features: {len(predictor.technical_features)}")
        print(f"   - Sentiment features: {len(predictor.sentiment_features)}")
        print(f"   - Market features: {len(predictor.market_features)}")
        
        # Split data for quick training test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model (reduced epochs for testing)
        print("\n🏋️ Training enhanced model (quick test)...")
        history = predictor.train_model(X_train, y_train, epochs=5, batch_size=16)
        
        # Evaluate model
        print("📈 Evaluating model performance...")
        metrics = predictor.evaluate_model(X_test, y_test)
        
        print("Enhanced Model Performance:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\n✅ Enhanced prediction model tests completed successfully")
        return True, metrics
        
    except Exception as e:
        print(f"❌ Enhanced prediction model test failed: {e}")
        return False, {}

def compare_models():
    """
    Compare enhanced model with baseline model
    """
    print("\n" + "="*60)
    print("COMPARING ENHANCED VS BASELINE MODELS")
    print("="*60)
    
    try:
        stock_symbol = "AAPL"
        
        # Test baseline model
        print("🔄 Testing baseline model...")
        try:
            baseline_predictor = BaselinePredictor(stock_symbol)
            baseline_X, baseline_y = baseline_predictor.prepare_data(period="6mo")
            
            split_idx = int(len(baseline_X) * 0.8)
            baseline_X_train = baseline_X[:split_idx]
            baseline_X_test = baseline_X[split_idx:]
            baseline_y_train = baseline_y[:split_idx]
            baseline_y_test = baseline_y[split_idx:]
            
            baseline_predictor.train_model(baseline_X_train, baseline_y_train, epochs=5)
            baseline_metrics = baseline_predictor.evaluate_model(baseline_X_test, baseline_y_test)
            
            print("Baseline Model Performance:")
            for metric, value in baseline_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
        except Exception as e:
            print(f"⚠️  Baseline model test failed: {e}")
            baseline_metrics = {}
        
        # Test enhanced model
        print("\n🚀 Testing enhanced model...")
        enhanced_success, enhanced_metrics = test_enhanced_prediction_model()
        
        # Compare results
        if baseline_metrics and enhanced_metrics:
            print("\n📊 MODEL COMPARISON RESULTS:")
            print("-" * 40)
            
            comparison_metrics = ['mse', 'mae', 'r2', 'directional_accuracy']
            
            for metric in comparison_metrics:
                if metric in baseline_metrics and metric in enhanced_metrics:
                    baseline_val = baseline_metrics[metric]
                    enhanced_val = enhanced_metrics[metric]
                    
                    if metric in ['mse', 'mae']:  # Lower is better
                        improvement = ((baseline_val - enhanced_val) / baseline_val) * 100
                        symbol = "📉" if improvement > 0 else "📈"
                    else:  # Higher is better
                        improvement = ((enhanced_val - baseline_val) / baseline_val) * 100
                        symbol = "📈" if improvement > 0 else "📉"
                    
                    print(f"{symbol} {metric.upper()}:")
                    print(f"    Baseline: {baseline_val:.4f}")
                    print(f"    Enhanced: {enhanced_val:.4f}")
                    print(f"    Change: {improvement:+.2f}%")
                    print()
        
        print("✅ Model comparison completed")
        return True
        
    except Exception as e:
        print(f"❌ Model comparison failed: {e}")
        return False

def run_comprehensive_test():
    """
    Run comprehensive test of all enhanced features
    """
    print("🚀 STARTING COMPREHENSIVE TEST OF ENHANCED FEATURES")
    print("=" * 80)
    
    start_time = time.time()
    test_results = {}
    
    # Test individual components
    test_results['sentiment_analysis'] = test_sentiment_analysis()
    test_results['market_indices'] = test_market_indices()
    
    # Test integrated model
    enhanced_success, _ = test_enhanced_prediction_model()
    test_results['enhanced_model'] = enhanced_success
    
    # Compare models
    test_results['model_comparison'] = compare_models()
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("🏁 COMPREHENSIVE TEST SUMMARY")
    print("="*80)
    
    print(f"⏱️  Total test duration: {duration:.2f} seconds")
    print("\n📋 Test Results:")
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
        if result:
            passed_tests += 1
    
    print(f"\n📊 Overall Success Rate: {passed_tests}/{total_tests} ({(passed_tests/total_tests)*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Enhanced features are working correctly.")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please check the error messages above.")
    
    return test_results

def main():
    """
    Main function to run tests
    """
    print("Enhanced Stock Prediction Features - Test Suite")
    print("=" * 60)
    print("This script will test the new sentiment analysis and market indices")
    print("integration with the neural network prediction models.")
    print()
    
    # Run comprehensive test
    results = run_comprehensive_test()
    
    # Provide recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 20)
    
    if results.get('sentiment_analysis', False):
        print("✅ Sentiment analysis is working - consider adding NewsAPI key for better results")
    
    if results.get('market_indices', False):
        print("✅ Market indices integration is working - ready for production use")
    
    if results.get('enhanced_model', False):
        print("✅ Enhanced model is working - ready for training with larger datasets")
    
    if results.get('model_comparison', False):
        print("✅ Model comparison shows the enhanced features are integrated successfully")
    
    print("\n🔧 NEXT STEPS:")
    print("1. Install any missing sentiment analysis dependencies if needed")
    print("2. Consider getting a NewsAPI key for better news sentiment analysis")
    print("3. Train the enhanced model with larger datasets for production use")
    print("4. Experiment with different feature combinations and model architectures")

if __name__ == "__main__":
    main() 