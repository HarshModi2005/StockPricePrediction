# Stock Price Prediction - Multi-Strategy Repository

This repository contains multiple approaches to stock price prediction, each implementing different algorithms and methodologies. The project is organized into modular components for easy experimentation and comparison.

## 📁 Repository Structure

```
StockPricePrediction/
├── hmm/                    # Hidden Markov Model (C implementation)
├── transformer/            # Transformer Neural Network (Python/Keras)
├── scripts/               # Multi-strategy Python scripts (LSTM, etc.)
├── data/                  # Sample datasets and data utilities
└── README.md             # This file
```

## 🚀 Approaches Overview

### 1. Hidden Markov Model (HMM) - `hmm/`
- **Language:** C
- **Algorithm:** Baum-Welch training with forward-backward algorithm
- **Features:** 
  - Stock movement categorization
  - Min-max normalization
  - Multi-day prediction with loss analysis
- **Best for:** Understanding probabilistic state transitions in stock movements

### 2. Transformer Model - `transformer/`
- **Language:** Python (TensorFlow/Keras)
- **Algorithm:** Multi-head attention transformer
- **Features:**
  - Advanced feature engineering (Bollinger Bands, RSI, ROC)
  - Sequence-based prediction
  - High R² scores (0.89-0.97)
- **Best for:** Commercial stocks and time-series with complex patterns

### 3. Multi-Strategy Scripts - `scripts/`
- **Language:** Python (Scikit-learn, Keras)
- **Algorithms:** LSTM, Linear Regression, Feature Selection
- **Features:**
  - Multiple stock datasets (Microsoft, S&P 500, NIFTY)
  - Intraday and next-day predictions
  - Comparative feature analysis
- **Best for:** Quick experimentation and strategy comparison

## 📊 Performance Summary

| Approach | Best Use Case | R² Score Range | Prediction Horizon |
|----------|---------------|----------------|-------------------|
| HMM | Market state analysis | N/A (Loss-based) | 10-100 days |
| Transformer | Commercial stocks | 0.89-0.97 | Short-term |
| Multi-Strategy | Quick experiments | Varies | 1-day to multi-day |

## 🛠️ Getting Started

Each subdirectory contains its own README with specific setup instructions:

1. **HMM:** Requires C compiler (gcc), minimal dependencies
2. **Transformer:** Requires Python 3.7+, TensorFlow, pandas, yfinance
3. **Scripts:** Requires Python 3.7+, scikit-learn, keras, pandas

## 📈 Data Requirements

The models work with standard OHLCV (Open, High, Low, Close, Volume) stock data:
- CSV format with proper headers
- Historical data (daily/intraday)
- Sample datasets provided in `data/` folder

## 🔬 Research Insights

- **Market Indices vs Individual Stocks:** Transformer model performs better on individual stocks (R² 0.96-0.97) than market indices (R² 0.89-0.90)
- **Feature Importance:** Technical indicators (RSI, Bollinger Bands) significantly improve prediction accuracy
- **Time Horizons:** Short-term predictions (1-5 days) generally more accurate than long-term

## 🤝 Contributing

Each approach is self-contained and can be improved independently:
- Add new features or algorithms
- Optimize hyperparameters
- Extend to new datasets or markets
- Cross-validate between approaches

## 📝 License

This project is for educational and research purposes. Please ensure compliance with data provider terms when using financial data.

---

**Note:** This repository consolidates multiple experimental approaches. Each method has different strengths - choose based on your specific use case and requirements. 