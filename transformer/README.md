# Transformer Model for Stock Prediction

This module implements a Transformer-based neural network for stock price prediction using multi-head attention mechanisms. The model excels at capturing complex temporal patterns in financial time series data.

## 🎯 Algorithm Overview

- **Architecture:** Multi-head attention transformer with layer normalization
- **Features:** Advanced technical indicators (Bollinger Bands, RSI, ROC)
- **Preprocessing:** Normalization and sequence generation
- **Performance:** R² scores of 0.89-0.97 depending on stock type

## 📋 Requirements

- Python 3.7 or higher
- TensorFlow 2.10+
- See `requirements.txt` for complete dependency list

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Jupyter Notebook
```bash
jupyter notebook nifty50.ipynb
```

### 3. Configure Stock Symbols
In the notebook, modify the `tickers` list to analyze different stocks:
```python
# Examples:
tickers = ['^DJI']        # Dow Jones Industrial Average
tickers = ['MSFT']        # Microsoft
tickers = ['^NSEI']       # NIFTY 50 (Indian market)
tickers = ['AAPL', 'GOOGL']  # Multiple stocks
```

### 4. Run All Cells
Execute the notebook cells sequentially to:
- Download stock data using yfinance
- Calculate technical indicators
- Train the transformer model
- Generate predictions and visualizations

## 📊 Model Architecture

```
Input Layer (OHLCV + Technical Indicators)
    ↓
Multi-Head Attention Layer
    ↓
Layer Normalization + Residual Connection
    ↓
Dense Layer with Dropout
    ↓
Global Average Pooling
    ↓
Output Layer (Price Prediction)
```

### Key Components
- **Multi-Head Attention:** Captures relationships across different time steps
- **Layer Normalization:** Stabilizes training and improves convergence
- **Residual Connections:** Prevents vanishing gradients
- **Dropout:** Reduces overfitting

## 🔧 Feature Engineering

The model uses comprehensive technical analysis features:

### Price-Based Features
- **Open, Close prices:** Raw and normalized
- **Bollinger Bands:** Upper/lower bands and width
- **Price Differences:** Absolute and percentage changes

### Technical Indicators
- **RSI (Relative Strength Index):** Momentum oscillator (14-period)
- **ROC (Rate of Change):** Price momentum (14-period)
- **Volume:** Trading volume normalization

### Feature Calculation Example
```python
def calculate_bollinger_bands(data, window=14, num_of_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return upper_band, lower_band
```

## 📈 Performance Insights

### Stock Type Performance
| Stock Type | R² Score Range | Best Use Case |
|------------|----------------|---------------|
| Individual Stocks | 0.96-0.97 | Microsoft, Apple, Google |
| Market Indices | 0.89-0.90 | S&P 500, Dow Jones, NIFTY |

### Why Individual Stocks Perform Better
- **Less Noise:** Individual stocks have clearer patterns than market aggregates
- **Company-Specific Signals:** Earnings, news, and events create predictable patterns
- **Lower Complexity:** Fewer conflicting signals compared to market indices

## 🔧 Customization

### Hyperparameter Tuning
Key parameters to experiment with:

```python
# Model architecture
SEQUENCE_LEN = 1          # Input sequence length
attention_heads = 8       # Number of attention heads
d_model = 64             # Model dimension

# Training parameters
epochs = 100
batch_size = 32
learning_rate = 0.001

# Feature engineering
bollinger_window = 14     # Bollinger Bands period
rsi_window = 14          # RSI calculation period
```

### Adding New Features
To add custom technical indicators:

1. **Calculate the indicator:**
```python
def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data.ewm(span=fast).mean()
    exp2 = data.ewm(span=slow).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line
```

2. **Add to feature DataFrame:**
```python
ticker_df[ticker+'_macd'] = calculate_macd(close)[0]
ticker_df[ticker+'_macd_signal'] = calculate_macd(close)[1]
```

3. **Update normalization:**
```python
MEAN = ticker_df.mean()
STD = ticker_df.std()
ticker_df = (ticker_df - MEAN) / STD
```

## 💾 Model Persistence

### Saving Trained Models
```python
# Save the model
model.save('my_transformer_model.keras')

# Save statistics for denormalization
stats.to_csv('feature_stats.csv', index=False)
```

### Loading for Inference
```python
from keras.models import load_model

# Load model and stats
model = load_model('my_transformer_model.keras')
stats = pd.read_csv('feature_stats.csv')
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce `SEQUENCE_LEN` or batch size
   - Use gradient checkpointing: `tf.config.experimental.enable_memory_growth()`

2. **Poor Convergence**
   - Check feature scaling (mean ≈ 0, std ≈ 1)
   - Reduce learning rate
   - Add more regularization (dropout, L2)

3. **Data Download Issues**
   - Verify ticker symbols are correct
   - Check internet connection
   - Use alternative data sources if yfinance fails

### Performance Optimization
```python
# Enable mixed precision training
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# Use GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

## 📚 References

- Vaswani et al. "Attention Is All You Need" (2017)
- Transformer architecture for time series forecasting
- Technical Analysis indicators in quantitative finance

---

**Note:** This implementation is optimized for research and experimentation. For production deployment, consider additional validation, error handling, and monitoring. 