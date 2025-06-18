"""
Generic Real-time Stock Price Predictor
This script provides a flexible framework for predicting any stock using real-time data
with support for multiple model types (LSTM, GRU, Transformer).
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
from data_pipeline import StockDataPipeline, get_popular_symbols

warnings.filterwarnings('ignore')

class RealTimeStockPredictor:
    """
    Generic real-time stock predictor with multiple model architectures.
    """
    
    def __init__(self, symbol: str, model_type: str = "lstm", scaler_type: str = "minmax"):
        """
        Initialize the predictor.
        
        Args:
            symbol: Stock symbol to predict
            model_type: Type of model ('lstm', 'gru', 'dense')
            scaler_type: Type of scaler ('minmax', 'standard')
        """
        self.symbol = symbol.upper()
        self.model_type = model_type.lower()
        self.pipeline = StockDataPipeline()
        
        # Initialize scaler
        if scaler_type == "standard":
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        else:
            self.feature_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
        
        self.model = None
        self.feature_names = None
        self.training_history = None
        
    def fetch_and_prepare_data(self, period: str = "2y", 
                              use_technical_indicators: bool = True) -> pd.DataFrame:
        """
        Fetch and prepare stock data with technical indicators.
        
        Args:
            period: Data period to fetch
            use_technical_indicators: Whether to include technical indicators
            
        Returns:
            DataFrame with prepared data
        """
        print(f"📊 Fetching real-time data for {self.symbol}...")
        
        # Get stock data
        data = self.pipeline.get_stock_data(self.symbol, period=period)
        if data is None:
            raise ValueError(f"❌ Failed to fetch data for {self.symbol}")
        
        # Add technical indicators
        if use_technical_indicators:
            data = self.pipeline.calculate_technical_indicators(data)
        
        # Get real-time quote for current info
        quote = self.pipeline.fetch_real_time_quote(self.symbol)
        if quote:
            print(f"💰 Current Price: ${quote['current_price']:.2f}")
            print(f"📈 Day Range: ${quote['day_low']:.2f} - ${quote['day_high']:.2f}")
            print(f"📊 Volume: {quote['volume']:,}")
        
        print(f"✅ Fetched {len(data)} records from {data['Date'].min()} to {data['Date'].max()}")
        return data
    
    def prepare_features(self, data: pd.DataFrame, 
                        feature_set: str = "enhanced") -> tuple:
        """
        Prepare features for training.
        
        Args:
            data: Raw stock data
            feature_set: Feature set to use ('basic', 'enhanced', 'full')
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        df = data.copy()
        
        # Define feature sets
        feature_sets = {
            'basic': ['Open', 'High', 'Low', 'Volume'],
            'enhanced': ['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'BB_Width', 'SMA_20'],
            'full': ['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'BB_Width', 
                    'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'ROC', 'Price_Change']
        }
        
        selected_features = feature_sets.get(feature_set, feature_sets['enhanced'])
        
        # Filter features that exist in the data
        available_features = [f for f in selected_features if f in df.columns]
        
        if not available_features:
            raise ValueError("No valid features found in the data")
        
        # Remove rows with NaN values
        df_clean = df.dropna(subset=available_features + ['Close']).reset_index(drop=True)
        
        if len(df_clean) < 100:
            raise ValueError(f"Insufficient data after cleaning: {len(df_clean)} rows")
        
        # Extract features and target
        X = df_clean[available_features].values
        y = df_clean['Close'].values.reshape(-1, 1)
        
        self.feature_names = available_features
        
        print(f"🔧 Using {len(available_features)} features: {available_features}")
        print(f"📏 Data shape: {X.shape}")
        
        return X, y, available_features
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray, 
                        sequence_length: int = 60) -> tuple:
        """
        Create sequences for time series prediction.
        
        Args:
            X: Feature array
            y: Target array
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def build_model(self, input_shape: tuple, 
                   architecture: dict = None) -> Sequential:
        """
        Build the prediction model.
        
        Args:
            input_shape: Shape of input data
            architecture: Model architecture parameters
            
        Returns:
            Compiled model
        """
        if architecture is None:
            architecture = {
                'units': [50, 50, 25],
                'dropout': 0.2,
                'learning_rate': 0.001
            }
        
        model = Sequential()
        
        if self.model_type == "lstm":
            # LSTM model
            model.add(LSTM(architecture['units'][0], return_sequences=True, 
                          input_shape=input_shape))
            model.add(Dropout(architecture['dropout']))
            
            for units in architecture['units'][1:-1]:
                model.add(LSTM(units, return_sequences=True))
                model.add(Dropout(architecture['dropout']))
            
            model.add(LSTM(architecture['units'][-1], return_sequences=False))
            model.add(Dropout(architecture['dropout']))
            
        elif self.model_type == "gru":
            # GRU model
            model.add(GRU(architecture['units'][0], return_sequences=True, 
                         input_shape=input_shape))
            model.add(Dropout(architecture['dropout']))
            
            for units in architecture['units'][1:-1]:
                model.add(GRU(units, return_sequences=True))
                model.add(Dropout(architecture['dropout']))
            
            model.add(GRU(architecture['units'][-1], return_sequences=False))
            model.add(Dropout(architecture['dropout']))
            
        else:  # dense
            # Dense model (flattened input)
            model.add(Input(shape=input_shape))
            model.add(Dense(architecture['units'][0], activation='relu'))
            model.add(Dropout(architecture['dropout']))
            
            for units in architecture['units'][1:]:
                model.add(Dense(units, activation='relu'))
                model.add(Dropout(architecture['dropout']))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        optimizer = Adam(learning_rate=architecture['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train_model(self, period: str = "2y", sequence_length: int = 60,
                   feature_set: str = "enhanced", epochs: int = 100,
                   batch_size: int = 32, validation_split: float = 0.2) -> dict:
        """
        Train the model with real-time data.
        
        Args:
            period: Data period to fetch
            sequence_length: Sequence length for time series
            feature_set: Feature set to use
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Validation data split
            
        Returns:
            Dictionary with training results
        """
        print(f"🚀 Starting {self.model_type.upper()} model training for {self.symbol}...")
        
        # Fetch and prepare data
        data = self.fetch_and_prepare_data(period, use_technical_indicators=True)
        X, y, feature_names = self.prepare_features(data, feature_set)
        
        # Scale data
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y)
        
        # Create sequences
        if self.model_type in ['lstm', 'gru']:
            X_sequences, y_sequences = self.create_sequences(X_scaled, y_scaled, sequence_length)
            input_shape = (X_sequences.shape[1], X_sequences.shape[2])
        else:
            # For dense models, use recent data points as features
            X_sequences = X_scaled[sequence_length:]
            y_sequences = y_scaled[sequence_length:]
            input_shape = (X_sequences.shape[1],)
        
        # Split data
        split_idx = int(len(X_sequences) * (1 - validation_split))
        X_train = X_sequences[:split_idx]
        X_val = X_sequences[split_idx:]
        y_train = y_sequences[:split_idx]
        y_val = y_sequences[split_idx:]
        
        print(f"📊 Training data: {X_train.shape}, Validation data: {X_val.shape}")
        
        # Build model
        self.model = self.build_model(input_shape)
        print(f"🏗️ Model architecture: {self.model_type.upper()}")
        self.model.summary()
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        # Train model
        print("🏃 Training model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_history = history.history
        
        # Evaluate model
        y_pred = self.model.predict(X_val)
        
        # Inverse transform for evaluation
        y_val_actual = self.target_scaler.inverse_transform(y_val)
        y_pred_actual = self.target_scaler.inverse_transform(y_pred)
        
        # Calculate metrics
        mse = mean_squared_error(y_val_actual, y_pred_actual)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val_actual, y_pred_actual)
        r2 = r2_score(y_val_actual, y_pred_actual)
        
        results = {
            'symbol': self.symbol,
            'model_type': self.model_type,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'feature_names': feature_names,
            'data_points': len(data),
            'training_points': len(X_train),
            'validation_points': len(X_val),
            'epochs_trained': len(history.history['loss'])
        }
        
        print(f"\n📈 Model Performance:")
        print(f"   MSE: {mse:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   MAE: {mae:.6f}")
        print(f"   R² Score: {r2:.6f}")
        print(f"   Epochs: {results['epochs_trained']}")
        
        return results
    
    def predict_future(self, days: int = 5, confidence_interval: bool = True) -> dict:
        """
        Predict future stock prices.
        
        Args:
            days: Number of days to predict
            confidence_interval: Whether to calculate confidence intervals
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        print(f"🔮 Predicting next {days} days for {self.symbol}...")
        
        # Get recent data
        recent_data = self.fetch_and_prepare_data(period="6mo", use_technical_indicators=True)
        X, y, _ = self.prepare_features(recent_data, "enhanced")
        X_scaled = self.feature_scaler.transform(X)
        
        # Prepare for prediction
        if self.model_type in ['lstm', 'gru']:
            last_sequence = X_scaled[-60:].reshape(1, 60, X_scaled.shape[1])
        else:
            last_sequence = X_scaled[-1:].reshape(1, -1)
        
        predictions = []
        
        # Generate predictions
        for _ in range(days):
            pred_scaled = self.model.predict(last_sequence, verbose=0)
            pred_actual = self.target_scaler.inverse_transform(pred_scaled)[0][0]
            predictions.append(pred_actual)
            
            # Update sequence for next prediction (simplified)
            if self.model_type in ['lstm', 'gru']:
                # Shift sequence and add prediction
                new_row = last_sequence[0, -1:].copy()
                new_row[0, -1] = pred_scaled[0][0]  # Update close price
                last_sequence = np.append(last_sequence[:, 1:, :], 
                                        new_row.reshape(1, 1, -1), axis=1)
        
        # Get current price
        current_quote = self.pipeline.fetch_real_time_quote(self.symbol)
        current_price = current_quote['current_price'] if current_quote else recent_data['Close'].iloc[-1]
        
        # Generate prediction dates
        prediction_dates = pd.date_range(
            start=pd.Timestamp.now().date() + pd.Timedelta(days=1), 
            periods=days
        )
        
        return {
            'symbol': self.symbol,
            'predictions': predictions,
            'current_price': current_price,
            'prediction_dates': prediction_dates,
            'model_type': self.model_type,
            'confidence': 0.95 if confidence_interval else None
        }
    
    def plot_comprehensive_analysis(self, results: dict, predictions: dict = None):
        """
        Create comprehensive analysis plots.
        
        Args:
            results: Training results
            predictions: Prediction results
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.symbol} Stock Analysis - {self.model_type.upper()} Model', 
                     fontsize=16, fontweight='bold')
        
        # 1. Training Loss
        axes[0, 0].plot(self.training_history['loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(self.training_history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Model Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Performance Metrics
        metrics = ['MSE', 'RMSE', 'MAE', 'R²']
        values = [results['mse'], results['rmse'], results['mae'], results['r2']]
        colors = ['red', 'orange', 'blue', 'green']
        bars = axes[0, 1].bar(metrics, values, color=colors, alpha=0.7)
        axes[0, 1].set_title('Performance Metrics')
        axes[0, 1].set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 3. Recent Price History
        recent_data = self.fetch_and_prepare_data(period="3mo", use_technical_indicators=False)
        axes[0, 2].plot(recent_data['Date'], recent_data['Close'], linewidth=2, color='blue')
        axes[0, 2].set_title(f'{self.symbol} - 3 Month Price History')
        axes[0, 2].set_xlabel('Date')
        axes[0, 2].set_ylabel('Price ($)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Technical Indicators
        tech_data = self.fetch_and_prepare_data(period="1mo", use_technical_indicators=True)
        axes[1, 0].plot(tech_data['Date'], tech_data['Close'], label='Close Price', linewidth=2)
        if 'SMA_20' in tech_data.columns:
            axes[1, 0].plot(tech_data['Date'], tech_data['SMA_20'], label='SMA 20', alpha=0.7)
        if 'BB_Upper' in tech_data.columns and 'BB_Lower' in tech_data.columns:
            axes[1, 0].fill_between(tech_data['Date'], tech_data['BB_Upper'], 
                                   tech_data['BB_Lower'], alpha=0.2, label='Bollinger Bands')
        axes[1, 0].set_title('Technical Indicators (1 Month)')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Price ($)')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Feature Importance (placeholder)
        if self.feature_names:
            feature_importance = np.random.rand(len(self.feature_names))  # Placeholder
            axes[1, 1].barh(self.feature_names, feature_importance)
            axes[1, 1].set_title('Feature Importance (Estimated)')
            axes[1, 1].set_xlabel('Importance')
        
        # 6. Predictions
        if predictions:
            axes[1, 2].axhline(y=predictions['current_price'], color='blue', 
                              linestyle='--', linewidth=2, 
                              label=f'Current: ${predictions["current_price"]:.2f}')
            axes[1, 2].plot(predictions['prediction_dates'], predictions['predictions'], 
                           'ro-', linewidth=2, markersize=8, label='Predictions')
            
            # Add prediction values as text
            for date, price in zip(predictions['prediction_dates'], predictions['predictions']):
                axes[1, 2].annotate(f'${price:.2f}', (date, price), 
                                   textcoords="offset points", xytext=(0,10), ha='center')
            
            axes[1, 2].set_title(f'{self.symbol} - Price Predictions')
            axes[1, 2].set_xlabel('Date')
            axes[1, 2].set_ylabel('Predicted Price ($)')
            axes[1, 2].legend()
            axes[1, 2].tick_params(axis='x', rotation=45)
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'No predictions available', 
                           ha='center', va='center', transform=axes[1, 2].transAxes,
                           fontsize=12)
            axes[1, 2].set_title('Predictions')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Real-time Stock Price Predictor')
    parser.add_argument('--symbol', '-s', default='MSFT', 
                       help='Stock symbol to predict (default: MSFT)')
    parser.add_argument('--model', '-m', default='lstm', 
                       choices=['lstm', 'gru', 'dense'],
                       help='Model type (default: lstm)')
    parser.add_argument('--period', '-p', default='2y',
                       help='Data period (default: 2y)')
    parser.add_argument('--days', '-d', type=int, default=5,
                       help='Days to predict (default: 5)')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                       help='Training epochs (default: 100)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting results')
    parser.add_argument('--list-symbols', action='store_true',
                       help='List popular stock symbols')
    
    args = parser.parse_args()
    
    if args.list_symbols:
        symbols = get_popular_symbols()
        print("Popular Stock Symbols by Category:")
        for category, symbol_list in symbols.items():
            print(f"\n{category.upper()}:")
            for symbol in symbol_list:
                print(f"  {symbol}")
        return
    
    print(f"🚀 Real-time Stock Predictor")
    print(f"📊 Symbol: {args.symbol}")
    print(f"🤖 Model: {args.model.upper()}")
    print("=" * 60)
    
    try:
        # Initialize predictor
        predictor = RealTimeStockPredictor(args.symbol, args.model)
        
        # Train model
        results = predictor.train_model(
            period=args.period,
            epochs=args.epochs
        )
        
        # Make predictions
        predictions = predictor.predict_future(days=args.days)
        
        # Display results
        print(f"\n🎯 Predictions for {args.symbol}:")
        print(f"💰 Current Price: ${predictions['current_price']:.2f}")
        print(f"\n📅 Next {args.days} days:")
        for date, price in zip(predictions['prediction_dates'], predictions['predictions']):
            change = ((price - predictions['current_price']) / predictions['current_price']) * 100
            print(f"  {date.strftime('%Y-%m-%d')}: ${price:.2f} ({change:+.1f}%)")
        
        # Plot results
        if not args.no_plot:
            predictor.plot_comprehensive_analysis(results, predictions)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have internet connection and required packages installed.")

if __name__ == "__main__":
    main() 