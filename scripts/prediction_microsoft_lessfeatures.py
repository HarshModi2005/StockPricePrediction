"""
Microsoft Stock Price Prediction using LSTM with Reduced Features
This script predicts Microsoft stock prices using LSTM neural networks
with reduced features (Open, High, Low - excluding Volume).
"""

import os
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

warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path):
    """Load and preprocess the stock data."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # Extract features (excluding Volume for this version)
        features = ['Open', 'High', 'Low']
        target = 'Close/Last'
        
        # Clean data by removing dollar signs
        for col in ['Open', 'High', 'Low', 'Close/Last']:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace('$', '').astype(float)
        
        # Extract feature arrays
        open_price = df['Open'].values.reshape(-1, 1)
        high = df['High'].values.reshape(-1, 1)
        low = df['Low'].values.reshape(-1, 1)
        target_var = df[target].values.reshape(-1, 1)
        
        return open_price, high, low, target_var, features
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None

def scale_features(open_price, high, low, target_var):
    """Scale all features using MinMaxScaler."""
    scaler = MinMaxScaler()
    
    # Scale each feature separately
    open_scaled = scaler.fit_transform(open_price)
    high_scaled = scaler.fit_transform(high)
    low_scaled = scaler.fit_transform(low)
    target_scaled = scaler.fit_transform(target_var)
    
    # Combine features (excluding volume)
    features_combined = np.column_stack([open_scaled, high_scaled, low_scaled])
    
    return features_combined, target_scaled, scaler

def prepare_lstm_data(features, target):
    """Prepare data for LSTM training."""
    # Use TimeSeriesSplit for proper time series validation
    tscv = TimeSeriesSplit(n_splits=10)
    
    # Get the last split for training/testing
    train_idx, test_idx = list(tscv.split(features))[-1]
    
    X_train = features[train_idx]
    X_test = features[test_idx]
    y_train = target[train_idx]
    y_test = target[test_idx]
    
    # Reshape for LSTM (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    return X_train, X_test, y_train, y_test

def build_lstm_model(input_shape):
    """Build and compile LSTM model for reduced features."""
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=input_shape, activation='tanh'),
        Dropout(0.2),
        LSTM(16, return_sequences=False, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """Train the LSTM model with early stopping."""
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R² Score: {r2:.6f}")
    
    return y_pred

def plot_results(y_test, y_pred):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Values', alpha=0.7)
    plt.plot(y_pred, label='LSTM Predictions', alpha=0.7)
    plt.title('Microsoft Stock Price Prediction - LSTM (Reduced Features: O,H,L)')
    plt.xlabel('Time Steps')
    plt.ylabel('Scaled Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function."""
    print("Microsoft Stock Price Prediction using LSTM (Reduced Features)")
    print("Features: Open, High, Low (Volume excluded)")
    print("=" * 60)
    
    # Load and preprocess data
    data_path = '../data/HistoricalData.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the data file exists in the data directory.")
        return
    
    open_price, high, low, target_var, features = load_and_preprocess_data(data_path)
    
    if open_price is None:
        return
    
    print(f"Using features: {features}")
    
    # Scale features
    features_scaled, target_scaled, scaler = scale_features(open_price, high, low, target_var)
    
    # Prepare LSTM data
    X_train, X_test, y_train, y_test = prepare_lstm_data(features_scaled, target_scaled)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Number of features: {X_train.shape[2]}")
    
    # Build model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    print(f"\nModel Summary:")
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Evaluate model
    y_pred = evaluate_model(model, X_test, y_test)
    
    # Plot results
    plot_results(y_test, y_pred)
    
    # Save model
    model.save('microsoft_lstm_lessfeatures.h5')
    print("\nModel saved as 'microsoft_lstm_lessfeatures.h5'")
    
    print("\nNote: This model uses reduced features (Open, High, Low)")
    print("Volume is excluded to test if it improves or degrades performance.")

if __name__ == "__main__":
    main()