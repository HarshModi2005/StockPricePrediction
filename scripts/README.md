# Multi-Strategy Stock Prediction Scripts

This module contains various Python scripts implementing different machine learning approaches for stock prediction. Each script focuses on specific strategies, datasets, or feature combinations for comprehensive experimentation.

## 📁 Script Overview

### Core Prediction Scripts
- **`prediction_microsoft_allfeatures.py`** - Microsoft stock prediction using all available features
- **`prediction_microsoft_lessfeatures.py`** - Microsoft stock with reduced feature set
- **`prediction_snp500.py`** - S&P 500 index prediction
- **`nifty_prediction_open.py`** - NIFTY index opening price prediction
- **`next_day_open_microsoft.py`** - Next-day opening price for Microsoft
- **`intraday.py`** - Intraday trading predictions

### Utility Scripts
- **`plot.py`** - Visualization utilities for results
- **`model.png`** - Model architecture visualization

## 📊 Available Datasets

### Stock Data Files
- **`stockdata.csv`** (29MB) - Comprehensive stock dataset
- **`nifty.csv`** (147KB) - NIFTY 50 index historical data  
- **`data.csv`** (72KB) - General stock data
- **`HistoricalData.csv`** (127KB) - Historical price data

*Note: Datasets are stored in the `../data/` directory for shared access.*

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Individual Scripts
```bash
# Microsoft stock prediction with all features
python prediction_microsoft_allfeatures.py

# NIFTY opening price prediction
python nifty_prediction_open.py

# Intraday predictions
python intraday.py
```

### 3. Visualize Results
```bash
python plot.py
```

## 🔧 Script Details

### Microsoft Stock Prediction Scripts

#### All Features (`prediction_microsoft_allfeatures.py`)
- **Features:** Volume, Open, High, Low, Close
- **Model:** LSTM with 32→16 neurons
- **Preprocessing:** MinMax scaling, TimeSeriesSplit validation
- **Output:** Price predictions with RMSE evaluation

#### Reduced Features (`prediction_microsoft_lessfeatures.py`)
- **Purpose:** Compare performance with fewer input features
- **Focus:** Essential price movements without volume complexity
- **Use Case:** When volume data is unreliable or unavailable

#### Next-Day Opening (`next_day_open_microsoft.py`)
- **Objective:** Predict tomorrow's opening price
- **Strategy:** Uses previous day's OHLC patterns
- **Application:** Pre-market trading decisions

### Market Index Predictions

#### S&P 500 (`prediction_snp500.py`)
- **Target:** Broad market index prediction
- **Challenge:** Higher complexity due to market aggregation
- **Features:** Standard OHLCV with technical indicators

#### NIFTY 50 (`nifty_prediction_open.py`)
- **Market:** Indian stock market index
- **Specialization:** Opening price prediction
- **Data:** Local market patterns and characteristics

### Intraday Trading (`intraday.py`)
- **Timeframe:** Short-term, within-day predictions
- **Strategy:** High-frequency pattern recognition
- **Risk:** Higher volatility, requires careful risk management

## 📈 Model Architectures

### LSTM Configuration
```python
# Standard configuration used across scripts
model = Sequential([
    LSTM(32, return_sequences=True, activation='linear'),
    LSTM(16, activation='relu'),
    Dense(1)
])

# Training parameters
epochs = 100
batch_size = 1
optimizer = 'adam'
loss = 'mean_squared_error'
```

### Feature Engineering
```python
# Standard preprocessing pipeline
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Time series split for validation
timesplit = TimeSeriesSplit(n_splits=10)
```

## 🎯 Performance Comparison

| Script | Target | Model | Features | Best Use Case |
|--------|--------|--------|----------|---------------|
| Microsoft (All) | Close Price | LSTM | OHLCV | Complete analysis |
| Microsoft (Less) | Close Price | LSTM | OHC | Quick predictions |
| S&P 500 | Index | LSTM | OHLCV | Market trends |
| NIFTY Open | Opening | LSTM | Custom | Pre-market |
| Next-Day Open | Opening | LSTM | OHLC | Day trading |
| Intraday | Various | LSTM | High-freq | Scalping |

## 🔧 Customization

### Adding New Stocks
1. **Prepare data in CSV format:**
```csv
Date,Open,High,Low,Close,Volume
2023-01-01,100.0,102.0,99.0,101.0,1000000
```

2. **Modify script data loading:**
```python
df = pd.read_csv('your_new_stock.csv')
```

3. **Adjust feature columns if needed:**
```python
features = ['Volume', 'Open', 'High', 'Low']  # Customize as needed
```

### Hyperparameter Tuning
```python
# LSTM architecture
lstm.add(LSTM(64, return_sequences=True))  # Increase neurons
lstm.add(LSTM(32, activation='tanh'))      # Change activation

# Training parameters  
epochs = 200          # More training
batch_size = 16       # Larger batches
learning_rate = 0.01  # Custom learning rate
```

### Feature Engineering Extensions
```python
# Add technical indicators
df['SMA_20'] = df['Close'].rolling(20).mean()
df['RSI'] = calculate_rsi(df['Close'])
df['MACD'] = calculate_macd(df['Close'])

# Include in features list
features = ['Volume', 'Open', 'High', 'Low', 'SMA_20', 'RSI', 'MACD']
```

## 📊 Visualization

The `plot.py` script provides:
- **Actual vs Predicted:** Time series comparison
- **Error Analysis:** Residual plots and distributions
- **Performance Metrics:** RMSE, MAE, R² visualization

```python
# Example usage
python plot.py --script prediction_microsoft_allfeatures --show-metrics
```

## 🐛 Troubleshooting

### Common Issues
1. **Data Format Errors**
   - Ensure CSV files have proper headers
   - Check for missing values or incorrect data types
   - Verify date formats are consistent

2. **Memory Issues**
   - Reduce batch size for large datasets
   - Use data generators for very large files
   - Monitor memory usage during training

3. **Poor Performance**
   - Check data quality and preprocessing
   - Experiment with different LSTM architectures
   - Validate feature scaling and normalization

### Performance Tips
- Use GPU acceleration when available
- Implement early stopping to prevent overfitting
- Cross-validate with multiple time periods
- Monitor training/validation loss curves

## 📚 References

- LSTM Networks for Time Series Prediction
- Technical Analysis in Algorithmic Trading
- Feature Engineering for Financial Data
- Time Series Cross-Validation Techniques

---

**Note:** These scripts are designed for experimentation and learning. For production trading, implement proper risk management, backtesting, and regulatory compliance. 