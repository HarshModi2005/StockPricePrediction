# Data Directory

This directory contains the datasets used by the various stock prediction models. Due to file size constraints, the actual CSV files are not tracked in git.

## Required Datasets

To run the models, you'll need to place the following CSV files in this directory:

### For HMM Model (`hmm/`)
- **`sample_nasa_data.csv`** - Stock data in format: "Index Name","Date","Open","High","Low","Close"

### For Multi-Strategy Scripts (`scripts/`)
- **`HistoricalData.csv`** - Microsoft historical data with columns: Date, Open, High, Low, Close/Last, Volume
- **`stockdata.csv`** - Large comprehensive stock dataset 
- **`nifty.csv`** - NIFTY 50 index data
- **`data.csv`** - General stock data for intraday analysis

### For Transformer Model (`transformer/`)
- The transformer model downloads data automatically using `yfinance`
- No manual data files required

## Data Format

All CSV files should follow standard OHLCV format:
```csv
Date,Open,High,Low,Close,Volume
2023-01-01,100.0,102.0,99.0,101.0,1000000
```

## Data Sources

You can obtain stock data from:
- **Yahoo Finance** - Use `yfinance` Python library
- **Alpha Vantage API** - Free API with registration
- **Quandl** - Financial and economic data
- **Your broker's API** - Most brokers provide historical data

## Quick Data Download

For quick testing, you can use this Python script to download sample data:

```python
import yfinance as yf
import pandas as pd

# Download Microsoft data
msft = yf.download('MSFT', period='2y')
msft.to_csv('HistoricalData.csv')

# Download NIFTY data (if available)
nifty = yf.download('^NSEI', period='2y') 
nifty.to_csv('nifty.csv')

# Download S&P 500 data
sp500 = yf.download('^GSPC', period='5y')
sp500.to_csv('stockdata.csv')
```

## File Size Note

Some datasets (especially `stockdata.csv`) can be quite large (>25MB). This is why they're excluded from git tracking. Make sure you have sufficient disk space before downloading comprehensive datasets. 