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
import datetime
import os

# Configuration parameters
start_date = None     # Start date for data download (None = earliest available)
end_date = None       # End date for data download (None = latest available)

# Parse command-line arguments
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Download historical stock price data from Yahoo Finance.')
    parser.add_argument('--ticker', type=str, help='Yahoo Finance ticker symbol (e.g., AAPL for Apple)')
    return parser.parse_args()

current_path = os.path.abspath(os.path.dirname(__file__))

# Get the ticker as argument or use the default ticker
args = parse_args()
if args.ticker:
    ticker = args.ticker
    print(f"Using ticker symbol from command line: {ticker}")
else:
    #Get the ticket from config.json
    import json
    with open(os.path.join(current_path,'config.json'), 'r') as f:
        config = json.load(f)
    ticker = config['general']['stock_symbol']
    if ticker == '' or ticker is None:
        print(f"No ticket symbol has been passed or configured")
        exit()
    print(f"Using ticker symbol from config.json: {ticker}")

file_name = ticker.replace('^', '')  # Remove ^ character for file naming

# Create the csv directory if it doesn't exist
# Get current path absolute path
folder = os.path.join(current_path, "csv")
os.makedirs(folder, exist_ok=True)

# Download historical data from Yahoo Finance
print(f"Downloading {ticker} data...")
if start_date and end_date:
    data = yf.download(ticker, start=start_date, end=end_date)
elif start_date and not end_date:
    data = yf.download(ticker, start=start_date, end=datetime.date.today().strftime("%Y-%m-%d"))
elif end_date and not start_date:
    data = yf.download(ticker, period="max")
    data = data.loc[:end_date]  
else:
    data = yf.download(ticker, period="max")

# Ensure the result is a DataFrame
if not isinstance(data, pd.DataFrame):
    data = pd.DataFrame(data)

# Save the data to a CSV file
output_path = os.path.join(folder, f"{file_name}.csv")

# Save to CSV first
data.to_csv(output_path)

# Read the CSV, remove the second and third rows, then save again
try:
    # Read the saved CSV file
    with open(output_path, 'r') as f:
        lines = f.readlines()
    
    # Remove the second and third rows (index 1 and 2)
    if len(lines) > 2:
        with open(output_path, 'w') as f:
            f.write(lines[0])  # Keep the header row
            f.writelines(lines[3:])  # Skip rows 1 and 2, keep the rest
        print(f"Removed 'Ticker' and 'Date' label rows from the CSV file")
    
    print(f"Data saved to {output_path}")
    print(f"Downloaded {len(data)} records from {data.index.min()} to {data.index.max()}")
except Exception as e:
    print(f"Error processing CSV file: {e}")
    # If there's an error, ensure the original data is saved
    data.to_csv(output_path)
    print(f"Original data saved to {output_path}")