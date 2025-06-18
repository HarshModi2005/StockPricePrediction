# Hidden Markov Model for Stock Prediction

This module implements a Hidden Markov Model (HMM) using the Baum-Welch algorithm to analyze and predict stock price movements. The model categorizes stock movements into discrete states and learns transition probabilities to forecast future market behavior.

## 🎯 Algorithm Overview

- **Training:** Baum-Welch algorithm with forward-backward computation
- **Prediction:** Viterbi-like state sequence prediction
- **Features:** Stock movement categorization based on price changes and volatility
- **Scaling:** Numerical stability through probability scaling

## 📋 Requirements

- **Compiler:** GCC or any C99-compatible compiler
- **Libraries:** Standard C libraries (math.h, stdio.h, stdlib.h, string.h, time.h)
- **Data:** CSV file with OHLCV stock data

## 🚀 Quick Start

### 1. Compile the Program
```bash
gcc hmm.c -o hmm -lm
```

### 2. Prepare Your Data
Ensure your CSV file follows this format:
```csv
"Index Name","Date","Open","High","Low","Close"
"NASDAQ","2023-01-01","100.50","102.30","99.80","101.20"
```

### 3. Update Data Path
Edit line ~80 in `hmm.c` to point to your data file:
```c
if (read_stock_data("../data/your_data.csv", stock_prices, &num_periods) != 0) {
```

### 4. Run the Program
```bash
./hmm
```

## 📊 Model Configuration

### States (N = 10)
The model uses 10 hidden states representing different market conditions:
- Bullish trends (various strengths)
- Bearish trends (various strengths)
- Sideways/neutral movements
- High volatility periods

### Observations (M = 20)
Stock movements are categorized into 10 observation symbols based on:
- **Percentage Change:** Strong increase (>3%), moderate (1-3%), small (0-1%), etc.
- **Volatility:** High/low range relative to price movement
- **Direction:** Bullish, bearish, or neutral

### Adjustable Parameters
```c
#define N 10          // Number of hidden states
#define M 20          // Number of observation symbols  
#define MAX_T 4147    // Maximum time periods (adjust for your dataset)
#define MAX_ITER 100  // Maximum Baum-Welch iterations
```

## 📈 Output Interpretation

The program outputs prediction accuracy for different time horizons:

```
Prediction for 10 days, Average Loss = 1.760682
Prediction for 20 days, Average Loss = 3.741657
...
Prediction for 100 days, Average Loss = 3.898718
```

**Lower loss values indicate better prediction accuracy.**

## 🔧 Customization

### Movement Categorization
Modify the `categorize_movement()` function to adjust how price movements are classified:
```c
int categorize_movement(double open, double close, double high, double low) {
    double change = ((close - open) / open) * 100;
    double range = high - low;
    
    // Customize these thresholds based on your market/timeframe
    if (change > 3.0 && range < 1.0) return 0;   // Strong small increase
    // ... add your custom logic
}
```

### Initial Parameters
The model starts with uniform distributions. For better convergence, you can initialize with domain knowledge:
```c
// Example: Initialize based on historical market behavior
double A[N][N] = { /* Custom transition probabilities */ };
double B[N][M] = { /* Custom emission probabilities */ };
```

## 🔬 Technical Details

### Numerical Stability
- **Scaling:** Forward-backward algorithms use scaling factors to prevent underflow
- **Convergence:** Baum-Welch stops when log-likelihood change < 1e-4
- **Initialization:** Random seed ensures reproducible results

### Data Processing
1. **Normalization:** Min-max scaling applied to OHLC values
2. **Categorization:** Continuous prices converted to discrete observations
3. **Sequence Processing:** Time series split for training/validation

## 🐛 Troubleshooting

### Common Issues
1. **File not found:** Check data file path in source code
2. **Compilation errors:** Ensure `-lm` flag for math library
3. **Poor convergence:** Increase MAX_ITER or adjust initial parameters
4. **Memory issues:** Reduce MAX_T for smaller datasets

### Performance Tips
- Use optimized compiler flags: `gcc -O3 hmm.c -o hmm -lm`
- For large datasets, consider increasing MAX_T
- Monitor convergence by uncommenting debug prints

## 📚 References

- Rabiner, L.R. "A Tutorial on Hidden Markov Models and Selected Applications"
- Baum-Welch Algorithm for HMM parameter estimation
- Forward-Backward algorithm for probability computation

---

**Note:** This implementation is optimized for educational purposes and stock market analysis. For production use, consider additional optimizations and error handling. 