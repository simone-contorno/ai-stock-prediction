import yfinance as yf
import pandas as pd

# Config
file_name = "Apple"
ticker = "AAPL"
start_date = None
end_date = None

# Get the data
data = yf.download(ticker, start=start_date, end=end_date)

if not isinstance(data, pd.DataFrame):
    data = pd.DataFrame(data)

# Save the data to a CSV file
data.to_csv("csv\\" + file_name + '.csv')