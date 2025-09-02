#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha Vantage Integration Demo

This script demonstrates how to use the Alpha Vantage API integration
for technical analysis and backtesting.
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta

from trading_bot.signals.alpha_vantage_signals import AlphaVantageTechnicalSignals
from trading_bot.backtesting.alpha_vantage_backtester import AlphaVantageBacktester
from trading_bot.market.market_data import MarketData
from trading_bot.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to demonstrate Alpha Vantage integration"""
    
    # Get Alpha Vantage API key from environment variable or config
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY', None) or config.get('alpha_vantage', {}).get('api_key', None)
    
    if not api_key:
        logger.error("Alpha Vantage API key not found. Please set the ALPHA_VANTAGE_API_KEY environment variable.")
        return
    
    # Initialize market data
    market_data = MarketData()
    
    # Initialize Alpha Vantage technical signals
    av_signals = AlphaVantageTechnicalSignals(market_data=market_data, api_key=api_key)
    
    # Symbols to analyze
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Get technical summaries
    print("Getting technical summaries for symbols...")
    summaries = {}
    
    for symbol in symbols:
        print(f"Analyzing {symbol}...")
        summary = av_signals.get_technical_summary(symbol)
        summaries[symbol] = summary
        
        # Print summary
        print(f"\n--- Technical Summary for {symbol} ---")
        print(f"Current Price: ${summary.get('current_price', 'N/A')}")
        print(f"Overall Signal: {summary.get('overall_signal', 'N/A')}")
        
        print("\nIndicators:")
        for indicator, value in summary.get('indicators', {}).items():
            print(f"  {indicator}: {value}")
        
        print("\nSignals:")
        for signal_name, signal_value in summary.get('signals', {}).items():
            print(f"  {signal_name}: {signal_value}")
        
    # Save summaries to file
    with open('technical_summaries.json', 'w') as f:
        json.dump(summaries, f, indent=2)
    print("\nSaved technical summaries to technical_summaries.json")
    
    # Define backtesting parameters
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\nRunning backtest from {start_date} to {end_date}...")
    
    # Define indicator configuration
    indicators_config = {
        "auto_enrich": True,
        "indicators": [
            {"name": "SMA", "period": 20},
            {"name": "SMA", "period": 50},
            {"name": "SMA", "period": 200},
            {"name": "RSI", "period": 14},
            {"name": "MACD", "fast_period": 12, "slow_period": 26, "signal_period": 9},
            {"name": "BBANDS", "period": 20, "std_dev": 2.0},
            {"name": "ADX", "period": 14},
            {"name": "ATR", "period": 14}
        ],
        "filter_symbols": True,
        "filter_criteria": {
            "require_signal": "bullish",
            "min_adx": 20,
            "rsi_range": [30, 70]
        }
    }
    
    # Initialize backtester
    backtester = AlphaVantageBacktester(
        initial_capital=10000.0,
        start_date=start_date,
        end_date=end_date,
        data_source="alpha_vantage",
        api_key=api_key,
        indicators_config=indicators_config
    )
    
    # Define a simple moving average crossover strategy
    strategy = {
        "name": "sma_crossover",
        "type": "trend_following",
        "parameters": {
            "short_window": 20,
            "long_window": 50,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.15
        }
    }
    
    # Add strategy to backtester
    backtester.add_strategy(strategy)
    
    # Run backtest
    results = backtester.backtest_with_av_signals(
        strategy_name="sma_crossover",
        symbols=symbols
    )
    
    # Print backtest results
    print("\nBacktest Results:")
    print(f"Initial Capital: ${results.get('initial_capital', 0):.2f}")
    print(f"Final Portfolio Value: ${results.get('final_portfolio_value', 0):.2f}")
    print(f"Total Return: {results.get('total_return_pct', 0):.2f}%")
    print(f"Annualized Return: {results.get('annualized_return_pct', 0):.2f}%")
    print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {results.get('max_drawdown_pct', 0):.2f}%")
    print(f"Total Trades: {results.get('total_trades', 0)}")
    
    # Plot equity curve if available
    if 'equity_curve' in results:
        equity_curve = results['equity_curve']
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve)
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.savefig('equity_curve.png')
        print("Saved equity curve plot to equity_curve.png")
    
    # Save detailed results to file
    with open('backtest_results.json', 'w') as f:
        # Remove non-serializable objects
        serializable_results = {k: v for k, v in results.items() if k != 'equity_curve'}
        json.dump(serializable_results, f, indent=2)
    print("Saved detailed backtest results to backtest_results.json")

if __name__ == "__main__":
    main() 