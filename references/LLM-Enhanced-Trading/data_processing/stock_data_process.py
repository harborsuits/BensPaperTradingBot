import os
import time
import requests
import pandas as pd
from collections import defaultdict
import numpy as np
import warnings
from datetime import datetime, timedelta
import logging
import pandas_market_calendars as mcal
from concurrent.futures import ThreadPoolExecutor
import gc  # For garbage collection
warnings.filterwarnings("ignore")
# Configure logging to output to a file
log_file = "trade_data.log"
logging.basicConfig(
    filename=log_file,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

headers = {
    "accept": "application/json",
    "APCA-API-KEY-ID": "YOUR-API-KEY",
    "APCA-API-SECRET-KEY": "YOUR-API-SECRET-KEY"
}

# Parent directory to store CSV files
parent_dir = "stock_data"
if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)

# List of stock symbols
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Define the market calendar for NYSE
nyse = mcal.get_calendar('NYSE')

# Function to get trading days for a given year
def get_trading_days(year):
    """
    Fetches all trading days for a given year based on the NYSE calendar. 
    """
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_days = schedule.index.strftime("%Y-%m-%d").tolist()
    return trading_days

# Function to clean trade data
def trade_clean(trade_dict, trades):
    """
    Cleans and organizes raw trade data by extracting key fields: last price, volume, 
    and clock time. 
    """
    for key in trades.keys():
        for trade in trades[key]:
            trade_dict['lastPrice'].append(trade['p'])
            trade_dict['Volume'].append(trade['s'])
            trade_dict['ClockTime'].append(trade['t'])

# Function to format datetime for URL
def format_datetime_for_url(dt):
    """
    Converts a datetime object to a string format suitable for API queries.
    """
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

# Main function to fetch and process data for a single date and stock symbol
def fetch_and_process_data(symbol, date):
    """
    Fetches, processes, and saves trade data for a stock symbol and date. 
    Handles API limitations with retries, computes Volume-Weighted Average Price on minute level (VWAP), 
    optimizes memory usage, removes duplicates, and stores the data in a structured directory.
    """
    # Check if the CSV file already exists for this symbol and date
    year = date[:4]
    symbol_dir = os.path.join(parent_dir, year, symbol)
    if not os.path.exists(symbol_dir):
        os.makedirs(symbol_dir)

    csv_file_path = f"{symbol_dir}/trades_{symbol}_{date}.csv"
    if os.path.exists(csv_file_path):
        logging.info(f"File already exists for {symbol} on {date}. Skipping.")
        return  # Skip this date if file already exists

    trade_dict = defaultdict(list)
    max_retries = 3
    retry_delay = 60  # 1 minute backoff for rate limit errors

    # Define time ranges
    time_ranges = [("09:30:00", "12:00:00"), ("13:00:00", "15:00:00"), ("15:00:00", "16:00:00")]

    # Loop over each time range
    for start_time, end_time in time_ranges:
        start_datetime = datetime.fromisoformat(f"{date}T{start_time}")
        end_datetime = datetime.fromisoformat(f"{date}T{end_time}")

        while start_datetime < end_datetime:
            current_end_datetime = start_datetime + timedelta(minutes=1)
            if current_end_datetime > end_datetime:
                current_end_datetime = end_datetime

            url_start = format_datetime_for_url(start_datetime)
            url_end = format_datetime_for_url(current_end_datetime)
            url = f"https://data.alpaca.markets/v2/stocks/trades?symbols={symbol}&start={url_start}&end={url_end}&limit=10000&feed=sip&sort=asc"

            retries = 0
            while retries < max_retries:
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    trades = data.get('trades', {})
                    trade_clean(trade_dict, trades)
                    logging.info(f"Successfully fetched data for {symbol} on {start_datetime} to {current_end_datetime}")
                    break  # Exit retry loop if successful
                elif response.status_code == 429:  # Rate limit error
                    logging.warning(f"Rate limit reached for {symbol} on {start_datetime}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)  # Wait before retrying
                    retries += 1
                else:
                    logging.error(f"Failed to fetch data for {symbol} on {start_datetime}: Status code {response.status_code}")
                    break  # Break on other errors

            start_datetime = current_end_datetime

    # Create a DataFrame from the collected trade data
    trade_df = pd.DataFrame.from_dict(trade_dict)
    
    # Optimize data types to reduce memory usage
    trade_df['Volume'] = trade_df['Volume'].astype('int32')  # Assuming Volume can fit in int32
    trade_df['lastPrice'] = trade_df['lastPrice'].astype('float32')

    # Convert ClockTime to datetime and format it
    trade_df['ClockTime'] = pd.to_datetime(trade_df['ClockTime'], errors='coerce')
    trade_df['ClockTime'] = trade_df['ClockTime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

    # Calculate the Volume-Weighted Average Price (VWAP)
    vwap_df = trade_df.groupby('ClockTime').apply(
        lambda x: (x['lastPrice'] * x['Volume']).sum() / x['Volume'].sum()
    ).reset_index(name='VWAP')
    
    # Merge the VWAP data back to the trade data
    trade_df = pd.merge(trade_df, vwap_df, on='ClockTime', how='left')
    
    # Remove duplicate entries based on 'ClockTime' and drop 'lastPrice' column
    trade_df = trade_df.drop_duplicates(subset=['ClockTime'])
    trade_df = trade_df.drop('lastPrice', axis=1)

    # Save the DataFrame to a CSV file
    trade_df.to_csv(csv_file_path, index=False)
    logging.info(f"Saved data to {csv_file_path}")

    # Clear trade_dict and DataFrame to free up memory, and run garbage collection
    del trade_dict, trade_df, vwap_df
    gc.collect()

# Function to process all data for a single stock across all trading days
def process_stock(symbol, trading_days):
    """
    The worker function for the threads  
    """
    for date in trading_days:
        fetch_and_process_data(symbol, date)

# Parallelize at the stock level using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    for symbol in symbols:
        all_trading_days = []
        for year in ["2021", "2022", "2023"]:
            all_trading_days.extend(get_trading_days(year))
        executor.submit(process_stock, symbol, all_trading_days)
