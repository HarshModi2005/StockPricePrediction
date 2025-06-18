#!/usr/bin/env python3
"""
Enhanced Stock Prediction Model with Sentiment Analysis and Market Indices
========================================================================

This module provides an advanced stock prediction model that combines:
- Traditional technical indicators
- Sentiment analysis from news and social media
- Market indices and regime indicators
- Multi-input neural network architecture
- Real-time prediction capabilities

Author: Stock Prediction Team
Date: 2024
"""

import os
import sys
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import yfinance as yf

# Import our custom modules
try:
    from sentiment_analysis import SentimentAnalyzer
    from market_indices import MarketIndicesAnalyzer
    from data_pipeline import DataPipeline
except ImportError as e:
    print(f"Warning: Could not import custom modules: {e}")
    print("Make sure sentiment_analysis.py and market_indices.py are in the same directory")

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedStockPredictor:
    """
    Enhanced stock prediction model with sentiment analysis and market indices
    """
    
    def __init__(self, 
                 stock_symbol: str,
                 lookback_days: int = 60,
                 prediction_days: int = 1,
                 newsapi_key: Optional[str] = None):
        """
        Initialize the enhanced stock predictor
        
        Args:
            stock_symbol: Stock symbol to predict
            lookback_days: Number of days to look back for features
            prediction_days: Number of days to predict ahead
            newsapi_key: NewsAPI key for sentiment analysis
        """
        self.stock_symbol = stock_symbol
        self.lookback_days = lookback_days
        self.prediction_days = prediction_days
        
        # Initialize analyzers
        self.sentiment_analyzer = SentimentAnalyzer(newsapi_key)
        self.market_analyzer = MarketIndicesAnalyzer()
        self.data_pipeline = DataPipeline()
        
        # Scalers for different feature types
        self.price_scaler = MinMaxScaler()
        self.technical_scaler = StandardScaler()
        self.sentiment_scaler = StandardScaler()
        self.market_scaler = StandardScaler()
        
        # Model
        self.model = None
        self.training_history = None
        
        # Feature names for tracking
        self.price_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.technical_features = []
        self.sentiment_features = []
        self.market_features = []
    
    def prepare_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare technical indicators
        
        Args:
            data: Stock price data
            
        Returns:
            DataFrame with technical indicators
        """
        df = data.copy()
        
        # Moving averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential moving averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / df['BB_width']
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['%D'] = df['%K'].rolling(window=3).mean()
        
        # Average True Range (ATR)
        df['TR1'] = df['High'] - df['Low']
        df['TR2'] = abs(df['High'] - df['Close'].shift())
        df['TR3'] = abs(df['Low'] - df['Close'].shift())
        df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # Price momentum
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
        df['Price_Change_10d'] = df['Close'].pct_change(periods=10)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Volatility
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()
        
        # Store technical feature names
        self.technical_features = [
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_12', 'EMA_26', 'MACD', 'MACD_signal', 'MACD_histogram',
            'RSI', 'BB_middle', 'BB_upper', 'BB_lower', 'BB_width', 'BB_position',
            '%K', '%D', 'ATR', 'Price_Change', 'Price_Change_5d', 'Price_Change_10d',
            'Volume_SMA', 'Volume_Ratio', 'Volatility'
        ]
        
        return df
    
    def prepare_sentiment_features(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Prepare sentiment analysis features
        
        Args:
            dates: Date range for sentiment analysis
            
        Returns:
            DataFrame with sentiment features
        """
        sentiment_data = []
        
        # Get sentiment data for the stock
        try:
            sentiment_scores = self.sentiment_analyzer.get_aggregated_sentiment(
                self.stock_symbol, 
                days_back=7
            )
            
            # Create sentiment features for each date
            for date in dates:
                sentiment_row = {
                    'Date': date,
                    'overall_sentiment': sentiment_scores.get('overall_sentiment', 0.0),
                    'sentiment_strength': sentiment_scores.get('sentiment_strength', 0.0),
                    'news_volume': sentiment_scores.get('news_volume', 0),
                    'vader_sentiment': sentiment_scores.get('vader_sentiment', 0.0),
                    'textblob_sentiment': sentiment_scores.get('textblob_sentiment', 0.0),
                    'finbert_sentiment': sentiment_scores.get('finbert_sentiment', 0.0),
                    'financial_sentiment': sentiment_scores.get('financial_sentiment', 0.0)
                }
                sentiment_data.append(sentiment_row)
                
        except Exception as e:
            logger.error(f"Error preparing sentiment features: {e}")
            # Create default sentiment features
            for date in dates:
                sentiment_row = {
                    'Date': date,
                    'overall_sentiment': 0.0,
                    'sentiment_strength': 0.0,
                    'news_volume': 0,
                    'vader_sentiment': 0.0,
                    'textblob_sentiment': 0.0,
                    'finbert_sentiment': 0.0,
                    'financial_sentiment': 0.0
                }
                sentiment_data.append(sentiment_row)
        
        sentiment_df = pd.DataFrame(sentiment_data)
        sentiment_df.set_index('Date', inplace=True)
        
        # Store sentiment feature names
        self.sentiment_features = [
            'overall_sentiment', 'sentiment_strength', 'news_volume',
            'vader_sentiment', 'textblob_sentiment', 'finbert_sentiment',
            'financial_sentiment'
        ]
        
        return sentiment_df
    
    def prepare_market_features(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Prepare market indices features
        
        Args:
            dates: Date range for market analysis
            
        Returns:
            DataFrame with market features
        """
        try:
            # Get market features for the stock
            market_features_df = self.market_analyzer.get_market_features_for_stock(
                self.stock_symbol, 
                period="1y"
            )
            
            if market_features_df.empty:
                # Create default market features
                market_data = []
                for date in dates:
                    market_row = {
                        'Date': date,
                        'sp500_return': 0.0,
                        'nasdaq_return': 0.0,
                        'vix_level': 20.0,
                        'market_trend': 0.0,
                        'sector_performance': 0.0
                    }
                    market_data.append(market_row)
                
                market_df = pd.DataFrame(market_data)
                market_df.set_index('Date', inplace=True)
                
                self.market_features = ['sp500_return', 'nasdaq_return', 'vix_level', 
                                      'market_trend', 'sector_performance']
                
            else:
                # Expand market features to all dates
                market_data = []
                latest_features = market_features_df.iloc[0].to_dict()
                
                for date in dates:
                    market_row = {'Date': date}
                    market_row.update(latest_features)
                    market_data.append(market_row)
                
                market_df = pd.DataFrame(market_data)
                market_df.set_index('Date', inplace=True)
                
                # Store market feature names (excluding Date)
                self.market_features = [col for col in market_df.columns if col != 'Date']
                
        except Exception as e:
            logger.error(f"Error preparing market features: {e}")
            # Create default market features
            market_data = []
            for date in dates:
                market_row = {
                    'Date': date,
                    'sp500_return': 0.0,
                    'nasdaq_return': 0.0,
                    'vix_level': 20.0,
                    'market_trend': 0.0,
                    'sector_performance': 0.0
                }
                market_data.append(market_row)
            
            market_df = pd.DataFrame(market_data)
            market_df.set_index('Date', inplace=True)
            
            self.market_features = ['sp500_return', 'nasdaq_return', 'vix_level', 
                                  'market_trend', 'sector_performance']
        
        return market_df
    
    def prepare_data(self, period: str = "2y") -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare all data for training
        
        Args:
            period: Data period to fetch
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        logger.info(f"Preparing data for {self.stock_symbol}")
        
        # Fetch stock data
        try:
            ticker = yf.Ticker(self.stock_symbol)
            stock_data = ticker.history(period=period)
            
            if stock_data.empty:
                raise ValueError(f"No data found for {self.stock_symbol}")
                
        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            raise
        
        # Prepare technical features
        stock_data = self.prepare_technical_features(stock_data)
        
        # Prepare sentiment features
        sentiment_df = self.prepare_sentiment_features(stock_data.index)
        
        # Prepare market features
        market_df = self.prepare_market_features(stock_data.index)
        
        # Combine all features
        combined_data = stock_data.copy()
        
        # Merge sentiment features
        combined_data = combined_data.join(sentiment_df, how='left')
        
        # Merge market features
        combined_data = combined_data.join(market_df, how='left')
        
        # Fill NaN values
        combined_data.fillna(method='ffill', inplace=True)
        combined_data.fillna(0, inplace=True)
        
        # Drop rows with insufficient data
        combined_data = combined_data.dropna()
        
        if len(combined_data) < self.lookback_days + self.prediction_days:
            raise ValueError(f"Insufficient data: {len(combined_data)} rows")
        
        # Prepare features and targets
        X, y = self._create_sequences(combined_data)
        
        logger.info(f"Prepared {len(X)} sequences with {X.shape[1]} timesteps and {X.shape[2]} features")
        
        return X, y
    
    def _create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction
        
        Args:
            data: Combined data with all features
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Select features
        price_cols = [col for col in self.price_features if col in data.columns]
        technical_cols = [col for col in self.technical_features if col in data.columns]
        sentiment_cols = [col for col in self.sentiment_features if col in data.columns]
        market_cols = [col for col in self.market_features if col in data.columns]
        
        all_feature_cols = price_cols + technical_cols + sentiment_cols + market_cols
        
        # Scale different feature types separately
        price_data = self.price_scaler.fit_transform(data[price_cols])
        technical_data = self.technical_scaler.fit_transform(data[technical_cols])
        sentiment_data = self.sentiment_scaler.fit_transform(data[sentiment_cols])
        market_data = self.market_scaler.fit_transform(data[market_cols])
        
        # Combine scaled features
        scaled_features = np.concatenate([
            price_data, technical_data, sentiment_data, market_data
        ], axis=1)
        
        # Target variable (next day's closing price)
        target_data = data['Close'].values
        
        # Create sequences
        X, y = [], []
        
        for i in range(len(scaled_features) - self.lookback_days - self.prediction_days + 1):
            # Features: lookback_days of all features
            X.append(scaled_features[i:(i + self.lookback_days)])
            
            # Target: closing price after prediction_days
            y.append(target_data[i + self.lookback_days + self.prediction_days - 1])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build the enhanced neural network model
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            
        Returns:
            Compiled Keras model
        """
        # Multi-input architecture
        input_layer = keras.Input(shape=input_shape, name='main_input')
        
        # LSTM layers for temporal patterns
        lstm1 = layers.LSTM(128, return_sequences=True, dropout=0.2)(input_layer)
        lstm2 = layers.LSTM(64, return_sequences=True, dropout=0.2)(lstm1)
        lstm3 = layers.LSTM(32, dropout=0.2)(lstm2)
        
        # Dense layers for feature combination
        dense1 = layers.Dense(64, activation='relu')(lstm3)
        dropout1 = layers.Dropout(0.3)(dense1)
        
        dense2 = layers.Dense(32, activation='relu')(dropout1)
        dropout2 = layers.Dropout(0.2)(dense2)
        
        dense3 = layers.Dense(16, activation='relu')(dropout2)
        
        # Output layer
        output = layers.Dense(1, activation='linear', name='price_output')(dense3)
        
        # Create model
        model = keras.Model(inputs=input_layer, outputs=output)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def train_model(self, 
                   X: np.ndarray, 
                   y: np.ndarray,
                   validation_split: float = 0.2,
                   epochs: int = 100,
                   batch_size: int = 32) -> keras.callbacks.History:
        """
        Train the model
        
        Args:
            X: Input features
            y: Target values
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        logger.info("Building and training model...")
        
        # Build model
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        # Train model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.training_history = history
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        # Calculate directional accuracy
        actual_direction = np.diff(y_test) > 0
        pred_direction = np.diff(predictions.flatten()) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'directional_accuracy': directional_accuracy
        }
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.training_history is None:
            logger.warning("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.training_history.history['loss'], label='Training Loss')
        ax1.plot(self.training_history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE plot
        ax2.plot(self.training_history.history['mae'], label='Training MAE')
        ax2.plot(self.training_history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Plot predictions vs actual values
        
        Args:
            X_test: Test features
            y_test: Test targets
        """
        predictions = self.predict(X_test)
        
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Actual', alpha=0.7)
        plt.plot(predictions, label='Predicted', alpha=0.7)
        plt.title(f'{self.stock_symbol} - Actual vs Predicted Prices')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def save_model(self, filepath: str):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")

def main():
    """
    Main function for testing the enhanced prediction model
    """
    # Initialize predictor
    stock_symbol = "AAPL"
    predictor = EnhancedStockPredictor(stock_symbol)
    
    try:
        # Prepare data
        X, y = predictor.prepare_data(period="1y")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Train model
        history = predictor.train_model(X_train, y_train, epochs=50)
        
        # Evaluate model
        metrics = predictor.evaluate_model(X_test, y_test)
        
        print("\nModel Performance:")
        print("=" * 40)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Plot results
        predictor.plot_training_history()
        predictor.plot_predictions(X_test, y_test)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 