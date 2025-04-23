"""Dataset Downloader for Stock Price Prediction

This script downloads historical stock price data from Yahoo Finance using the yfinance library.
It retrieves data for a specified ticker symbol (default: S&P 500 index) and saves it to a CSV file.

The downloaded data includes Open, High, Low, Close prices, Volume, and other stock metrics
that will be used for training and testing the stock prediction models.

Usage:
    Simply run this script to download the default S&P 500 data, or modify the configuration
    variables below to download data for different stocks or time periods.
"""

import yfinance as yf
import pandas as pd
import os

# Configuration parameters
file_name = "S&P500"  # Name of the output file (without extension)
ticker = "^GSPC"      # Yahoo Finance ticker symbol for S&P 500 index
start_date = None     # Start date for data download (None = earliest available)
end_date = None       # End date for data download (None = latest available)

# Create the csv directory if it doesn't exist
os.makedirs("csv", exist_ok=True)

# Download historical data from Yahoo Finance
print(f"Downloading {ticker} data...")
data = yf.download(ticker, start=start_date, end=end_date)

# Ensure the result is a DataFrame
if not isinstance(data, pd.DataFrame):
    data = pd.DataFrame(data)

# Save the data to a CSV file
output_path = os.path.join("csv", f"{file_name}.csv")
data.to_csv(output_path)
print(f"Data saved to {output_path}")
print(f"Downloaded {len(data)} records from {data.index.min()} to {data.index.max()}")