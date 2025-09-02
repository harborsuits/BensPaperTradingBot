#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of using the backtester with real and mock market data.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import backtester and market data tools
from trading.backtester import Backtester, Portfolio, Position
from trading.data.fetch_market_data import fetch_market_data, fetch_technical_indicators
from trading.strategies.sma_crossover import SMACrossover
from trading.strategies.macd import MACDStrategy
from trading.strategies.mean_reversion import MeanReversionStrategy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_example_backtest(strategy_name='sma_crossover', use_mock=True, api_key=None):
    """
    Run a backtest with the specified strategy and data source.
    
    Parameters:
    -----------
    strategy_name : str
        Name of the strategy to use
    use_mock : bool
        Whether to use mock data or real API data
    api_key : str, optional
        Alpha Vantage API key
    """
    # Define backtest parameters
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    initial_capital = 100000
    position_size = 0.2  # 20% of portfolio per position
    commission = 0.001  # 0.1% commission
    stop_loss_pct = 0.05  # 5% stop loss
    take_profit_pct = 0.1  # 10% take profit
    
    # Fetch market data
    logger.info(f"Fetching market data for {symbols}")
    market_data = fetch_market_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        interval='daily',
        adjusted=True,
        use_mock=use_mock,
        api_key=api_key
    )
    
    # Initialize strategy based on the name
    if strategy_name == 'sma_crossover':
        # Define SMA parameters
        short_window = 20
        long_window = 50
        
        # Initialize strategy
        strategy = SMACrossover(
            short_window=short_window,
            long_window=long_window
        )
        
        logger.info(f"Initialized SMA Crossover strategy with short={short_window}, long={long_window}")
        
    elif strategy_name == 'macd':
        # Define MACD parameters
        fast_period = 12
        slow_period = 26
        signal_period = 9
        
        # Initialize strategy
        strategy = MACDStrategy(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period
        )
        
        logger.info(f"Initialized MACD strategy with fast={fast_period}, slow={slow_period}, signal={signal_period}")
        
    elif strategy_name == 'mean_reversion':
        # Define Mean Reversion parameters
        window = 20
        std_dev = 2.0
        
        # Initialize strategy
        strategy = MeanReversionStrategy(
            window=window,
            std_dev=std_dev
        )
        
        logger.info(f"Initialized Mean Reversion strategy with window={window}, std_dev={std_dev}")
        
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Initialize backtester
    backtester = Backtester(
        strategy=strategy,
        initial_capital=initial_capital,
        position_size=position_size,
        commission=commission,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct
    )
    
    # Run backtest for each symbol
    results = {}
    
    for symbol, data in market_data.items():
        logger.info(f"Running backtest for {symbol}")
        
        try:
            # Run the backtest
            result = backtester.run(data, symbol)
            
            # Save result
            results[symbol] = result
            
            # Plot equity curve
            backtester.plot_results(
                title=f"{strategy_name.upper()} - {symbol}",
                save_path=f"results/{strategy_name}_{symbol}_equity_curve.png"
            )
            
            # Log performance metrics
            logger.info(f"{symbol} Results: Return: {result['return_pct']:.2f}%, "
                        f"Win Rate: {result['win_rate']:.2f}%, "
                        f"Max Drawdown: {result['max_drawdown']:.2f}%")
            
        except Exception as e:
            logger.error(f"Error running backtest for {symbol}: {e}")
    
    # Save all results to a JSON file
    try:
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Save results
        filename = f"results/{strategy_name}_backtest_results.json"
        with open(filename, 'w') as f:
            # Convert non-serializable objects
            serializable_results = {
                symbol: {
                    k: v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
                    for k, v in result.items()
                }
                for symbol, result in results.items()
            }
            json.dump(serializable_results, f, indent=4)
        
        logger.info(f"Results saved to {filename}")
    
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    return results


def compare_strategies(use_mock=True, api_key=None):
    """
    Run and compare multiple trading strategies.
    
    Parameters:
    -----------
    use_mock : bool
        Whether to use mock data or real API data
    api_key : str, optional
        Alpha Vantage API key
    """
    strategies = ['sma_crossover', 'macd', 'mean_reversion']
    results = {}
    
    for strategy in strategies:
        logger.info(f"Testing {strategy} strategy")
        results[strategy] = run_example_backtest(strategy, use_mock, api_key)
    
    # Create a comparison table
    comparison = []
    
    for strategy, strategy_results in results.items():
        for symbol, result in strategy_results.items():
            comparison.append({
                'Strategy': strategy,
                'Symbol': symbol,
                'Return (%)': result.get('return_pct', 0),
                'Win Rate (%)': result.get('win_rate', 0),
                'Trades': result.get('num_trades', 0),
                'Max Drawdown (%)': result.get('max_drawdown', 0)
            })
    
    # Convert to DataFrame for easier analysis
    comparison_df = pd.DataFrame(comparison)
    
    # Display strategy comparison
    print("\nStrategy Comparison:")
    print(comparison_df)
    
    # Plot returns by strategy
    plt.figure(figsize=(12, 6))
    
    for strategy in strategies:
        strategy_data = comparison_df[comparison_df['Strategy'] == strategy]
        plt.bar(
            [f"{symbol} ({strategy})" for symbol in strategy_data['Symbol']],
            strategy_data['Return (%)'],
            label=strategy
        )
    
    plt.title('Strategy Returns Comparison')
    plt.xlabel('Symbol (Strategy)')
    plt.ylabel('Return (%)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save the comparison chart
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/strategy_comparison.png")
    plt.close()
    
    logger.info("Strategy comparison completed and saved to results/strategy_comparison.png")
    
    return comparison_df


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run trading strategy backtests')
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        choices=['sma_crossover', 'macd', 'mean_reversion', 'all'],
        default='all',
        help='Trading strategy to backtest'
    )
    parser.add_argument(
        '--use-real-data', '-r',
        action='store_true',
        help='Use real data instead of mock data'
    )
    parser.add_argument(
        '--api-key', '-k',
        type=str,
        help='Alpha Vantage API key'
    )
    
    args = parser.parse_args()
    
    # Run the example based on arguments
    if args.strategy == 'all':
        compare_strategies(
            use_mock=not args.use_real_data,
            api_key=args.api_key
        )
    else:
        run_example_backtest(
            strategy_name=args.strategy,
            use_mock=not args.use_real_data,
            api_key=args.api_key
        ) 