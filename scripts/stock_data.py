#!/usr/bin/env python3
"""
Simple script to fetch and display stock data using yfinance
"""
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def get_stock_data(ticker, period="1mo"):
    """
    Fetch stock data for the given ticker and period
    
    Parameters:
    ticker (str): Stock ticker symbol
    period (str): Time period to fetch data for (e.g., "1d", "1mo", "1y")
    
    Returns:
    pandas.DataFrame: Historical stock data
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    return hist

def display_stock_info(ticker):
    """
    Display basic information about a stock
    
    Parameters:
    ticker (str): Stock ticker symbol
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    
    print(f"\nStock Information for {ticker}:")
    print(f"Company Name: {info.get('shortName', 'N/A')}")
    print(f"Current Price: ${info.get('currentPrice', 'N/A')}")
    print(f"Previous Close: ${info.get('previousClose', 'N/A')}")
    print(f"Market Cap: ${info.get('marketCap', 'N/A'):,}")
    print(f"52-Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}")
    print(f"52-Week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}")
    
    # Show some financial metrics if available
    if 'trailingPE' in info:
        print(f"P/E Ratio: {info['trailingPE']:.2f}")
    if 'dividendYield' in info and info['dividendYield'] is not None:
        print(f"Dividend Yield: {info['dividendYield']*100:.2f}%")

def plot_stock_data(data, ticker):
    """
    Plot stock price data
    
    Parameters:
    data (pandas.DataFrame): Stock price data
    ticker (str): Stock ticker symbol
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label=f"{ticker} Close Price")
    plt.title(f"{ticker} Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the script"""
    # Get user input for ticker symbol
    ticker = input("Enter a stock ticker symbol (e.g., AAPL): ").upper()
    
    try:
        # Get stock data for the past month
        data = get_stock_data(ticker)
        
        # Display stock information
        display_stock_info(ticker)
        
        # Calculate basic statistics
        print("\nStock Statistics (Past Month):")
        print(f"Average Price: ${data['Close'].mean():.2f}")
        print(f"Highest Price: ${data['Close'].max():.2f}")
        print(f"Lowest Price: ${data['Close'].min():.2f}")
        
        # Calculate price change
        first_price = data['Close'].iloc[0]
        last_price = data['Close'].iloc[-1]
        price_change = ((last_price - first_price) / first_price) * 100
        print(f"Price Change: {price_change:.2f}%")
        
        # Ask if user wants to see a chart
        show_chart = input("\nWould you like to see a price chart? (yes/no): ").lower()
        if show_chart == 'yes' or show_chart == 'y':
            plot_stock_data(data, ticker)
    
    except Exception as e:
        print(f"Error: {e}")
        print("Please check the ticker symbol and try again.")

if __name__ == "__main__":
    main() 