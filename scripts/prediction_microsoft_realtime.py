"""
Microsoft Stock Price Prediction using Real-time Data
This script predicts Microsoft stock prices using LSTM neural networks
with real-time data fetched from online sources.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
from data_pipeline import StockDataPipeline

warnings.filterwarnings('ignore')

class RealTimeMicrosoftPredictor:
    """Real-time Microsoft stock prediction using LSTM."""
    
    def __init__(self, symbol: str = "MSFT"):
        """
        Initialize the predictor.
        
        Args:
            symbol: Stock symbol to predict
        """
        self.symbol = symbol
        self.pipeline = StockDataPipeline()
        self.scaler = MinMaxScaler()
        self.model = None
        self.feature_columns = ['Volume', 'Open', 'High', 'Low']
        
    def fetch_real_time_data(self, period: str = "2y") -> pd.DataFrame:
        """
        Fetch real-time stock data.
        
        Args:
            period: Data period to fetch
            
        Returns:
            DataFrame with stock data and technical indicators
        """
        print(f"Fetching real-time data for {self.symbol}...")
        
        # Get base stock data
        data = self.pipeline.get_stock_data(self.symbol, period=period)
        if data is None:
            raise ValueError(f"Failed to fetch data for {self.symbol}")
        
        # Add technical indicators
        data_with_indicators = self.pipeline.calculate_technical_indicators(data)
        
        print(f"Fetched {len(data_with_indicators)} records")
        print(f"Date range: {data_with_indicators['Date'].min()} to {data_with_indicators['Date'].max()}")
        
        return data_with_indicators
    
    def prepare_features(self, data: pd.DataFrame, use_technical_indicators: bool = True) -> tuple:
        """
        Prepare features for training.
        
        Args:
            data: Raw stock data
            use_technical_indicators: Whether to include technical indicators
            
        Returns:
            Tuple of (features, target, feature_names)
        """
        df = data.copy()
        
        # Base features
        features = ['Volume', 'Open', 'High', 'Low']
        
        # Add technical indicators if requested
        if use_technical_indicators:
            technical_features = ['RSI', 'MACD', 'BB_Width', 'ROC', 'SMA_20', 'SMA_50']
            # Only add features that exist and have no NaN values
            for feature in technical_features:
                if feature in df.columns and not df[feature].isna().all():
                    features.append(feature)
        
        # Remove rows with NaN values
        df = df.dropna(subset=features + ['Close']).reset_index(drop=True)
        
        if len(df) == 0:
            raise ValueError("No valid data after removing NaN values")
        
        # Extract features and target
        X = df[features].values
        y = df['Close'].values.reshape(-1, 1)
        
        print(f"Using features: {features}")
        print(f"Data shape after preprocessing: {X.shape}")
        
        return X, y, features
    
    def scale_data(self, X: np.ndarray, y: np.ndarray, fit_scaler: bool = True) -> tuple:
        """
        Scale the features and target.
        
        Args:
            X: Feature array
            y: Target array
            fit_scaler: Whether to fit the scaler (True for training, False for prediction)
            
        Returns:
            Tuple of (scaled_X, scaled_y, target_scaler)
        """
        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Scale target separately
        target_scaler = MinMaxScaler()
        if fit_scaler:
            y_scaled = target_scaler.fit_transform(y)
        else:
            y_scaled = target_scaler.transform(y)
        
        return X_scaled, y_scaled, target_scaler
    
    def prepare_lstm_data(self, X_scaled: np.ndarray, y_scaled: np.ndarray, 
                         sequence_length: int = 60) -> tuple:
        """
        Prepare data for LSTM training.
        
        Args:
            X_scaled: Scaled features
            y_scaled: Scaled target
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i-sequence_length:i])
            y_sequences.append(y_scaled[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def build_model(self, input_shape: tuple) -> Sequential:
        """
        Build LSTM model.
        
        Args:
            input_shape: Shape of input data
            
        Returns:
            Compiled LSTM model
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model
    
    def train_model(self, period: str = "2y", sequence_length: int = 60, 
                   use_technical_indicators: bool = True) -> dict:
        """
        Train the LSTM model with real-time data.
        
        Args:
            period: Data period to fetch
            sequence_length: LSTM sequence length
            use_technical_indicators: Whether to use technical indicators
            
        Returns:
            Dictionary with training results
        """
        print("Starting model training with real-time data...")
        
        # Fetch data
        data = self.fetch_real_time_data(period)
        
        # Prepare features
        X, y, feature_names = self.prepare_features(data, use_technical_indicators)
        
        # Scale data
        X_scaled, y_scaled, self.target_scaler = self.scale_data(X, y, fit_scaler=True)
        
        # Prepare sequences
        X_sequences, y_sequences = self.prepare_lstm_data(X_scaled, y_scaled, sequence_length)
        
        # Split data
        split_index = int(0.8 * len(X_sequences))
        X_train = X_sequences[:split_index]
        X_test = X_sequences[split_index:]
        y_train = y_sequences[:split_index]
        y_test = y_sequences[split_index:]
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        
        # Build model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Train model
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        # Inverse transform for evaluation
        y_test_actual = self.target_scaler.inverse_transform(y_test)
        y_pred_actual = self.target_scaler.inverse_transform(y_pred)
        
        mse = mean_squared_error(y_test_actual, y_pred_actual)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_actual, y_pred_actual)
        
        results = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'feature_names': feature_names,
            'data_points': len(data),
            'training_points': len(X_train),
            'test_points': len(X_test),
            'history': history.history
        }
        
        print(f"\nModel Performance:")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R² Score: {r2:.6f}")
        
        return results
    
    def predict_next_days(self, days: int = 5) -> dict:
        """
        Predict stock prices for the next few days.
        
        Args:
            days: Number of days to predict
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        print(f"Predicting next {days} days for {self.symbol}...")
        
        # Get recent data
        recent_data = self.fetch_real_time_data(period="3mo")
        X, y, _ = self.prepare_features(recent_data, use_technical_indicators=True)
        X_scaled, _, _ = self.scale_data(X, y, fit_scaler=False)
        
        # Use last 60 days for prediction
        last_sequence = X_scaled[-60:].reshape(1, 60, X_scaled.shape[1])
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            pred = self.model.predict(current_sequence, verbose=0)
            pred_actual = self.target_scaler.inverse_transform(pred)[0][0]
            predictions.append(pred_actual)
            
            # Update sequence (simplified - in practice, you'd need to update all features)
            # For now, we'll just use the prediction as the new close price
            new_row = current_sequence[0, -1:].copy()
            new_row[0, -1] = pred[0][0]  # Update close price
            current_sequence = np.append(current_sequence[:, 1:, :], new_row.reshape(1, 1, -1), axis=1)
        
        # Get current price for comparison
        current_quote = self.pipeline.fetch_real_time_quote(self.symbol)
        current_price = current_quote['current_price'] if current_quote else recent_data['Close'].iloc[-1]
        
        return {
            'predictions': predictions,
            'current_price': current_price,
            'prediction_dates': pd.date_range(start=pd.Timestamp.now().date() + pd.Timedelta(days=1), periods=days),
            'symbol': self.symbol
        }
    
    def plot_results(self, results: dict, predictions: dict = None):
        """
        Plot training results and predictions.
        
        Args:
            results: Training results
            predictions: Prediction results (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        axes[0, 0].plot(results['history']['loss'], label='Training Loss')
        axes[0, 0].plot(results['history']['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Feature importance (simplified)
        feature_names = results['feature_names'][:10]  # Top 10 features
        feature_importance = np.random.rand(len(feature_names))  # Placeholder
        axes[0, 1].barh(feature_names, feature_importance)
        axes[0, 1].set_title('Feature Importance (Placeholder)')
        axes[0, 1].set_xlabel('Importance')
        
        # Recent price data
        recent_data = self.fetch_real_time_data(period="1mo")
        axes[1, 0].plot(recent_data['Date'], recent_data['Close'])
        axes[1, 0].set_title(f'{self.symbol} - Recent Price (1 Month)')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Price ($)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Predictions
        if predictions:
            axes[1, 1].axhline(y=predictions['current_price'], color='blue', linestyle='--', 
                              label=f'Current Price: ${predictions["current_price"]:.2f}')
            axes[1, 1].plot(predictions['prediction_dates'], predictions['predictions'], 
                           'ro-', label='Predictions')
            axes[1, 1].set_title(f'{self.symbol} - Price Predictions')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Predicted Price ($)')
            axes[1, 1].legend()
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No predictions available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Predictions')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main execution function."""
    print("Microsoft Stock Price Prediction - Real-time Data")
    print("=" * 60)
    
    # Initialize predictor
    predictor = RealTimeMicrosoftPredictor("MSFT")
    
    try:
        # Train model
        print("\n1. Training model with real-time data...")
        results = predictor.train_model(period="2y", use_technical_indicators=True)
        
        # Make predictions
        print("\n2. Making predictions...")
        predictions = predictor.predict_next_days(days=5)
        
        print(f"\nPredictions for {predictor.symbol}:")
        print(f"Current Price: ${predictions['current_price']:.2f}")
        print("\nNext 5 days:")
        for date, price in zip(predictions['prediction_dates'], predictions['predictions']):
            print(f"  {date.strftime('%Y-%m-%d')}: ${price:.2f}")
        
        # Plot results
        print("\n3. Plotting results...")
        predictor.plot_results(results, predictions)
        
        # Save model
        model_filename = f"microsoft_realtime_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.h5"
        predictor.model.save(model_filename)
        print(f"\nModel saved as: {model_filename}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have internet connection and required packages installed.")

if __name__ == "__main__":
    main() 