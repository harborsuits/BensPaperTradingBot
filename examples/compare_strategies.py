#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Comparison Script

Compares the performance of different trading strategies:
- MACD
- RSI 
- Combined MACD+RSI
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import strategies
from trading.strategies.macd import MACD
from trading.strategies.rsi import RSI
from trading.strategies.combined import CombinedStrategy
from trading.backtester import Backtester
from trading.data.fetch_market_data import fetch_market_data
from trading.data.data_validation import validate_dataset, prepare_data_for_backtest, analyze_strategy_data_requirements

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_backtest(strategy, data, symbol, initial_capital=100000, position_size=0.2, 
               commission=0.001, stop_loss=0.05, take_profit=0.1):
    """
    Run a backtest for a given strategy on a single symbol.
    
    Parameters:
    -----------
    strategy : object
        Strategy object with generate_signals method
    data : pandas.DataFrame
        Market data for backtesting
    symbol : str
        Symbol being tested
    initial_capital : float
        Initial capital for the portfolio
    position_size : float
        Position size as percentage of portfolio
    commission : float
        Commission rate per trade
    stop_loss : float
        Stop loss percentage
    take_profit : float
        Take profit percentage
        
    Returns:
    --------
    dict
        Dictionary of backtest results
    """
    logger.info(f"Running backtest for {strategy.name} on {symbol}")
    
    # Generate signals
    signals = strategy.generate_signals(data)
    
    # Initialize portfolio and backtester
    backtester = Backtester(
        initial_capital=initial_capital,
        position_size_pct=position_size,
        commission=commission,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit
    )
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Run backtest
    results = backtester.run_backtest(
        strategy_name=strategy.name,
        symbols=[symbol],
        start_date=data.index[0].strftime('%Y-%m-%d'),
        end_date=data.index[-1].strftime('%Y-%m-%d'),
        strategy_params=strategy.get_parameters()
    )
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"results/{symbol}_{strategy.name}_{timestamp}.json"
    
    try:
        backtester.save_results(results_file)
        
        # Plot results
        plot_file = f"results/{symbol}_{strategy.name}_{timestamp}.png"
        backtester.plot_results(save_path=plot_file)
    except Exception as e:
        logger.error(f"Error saving results or plotting: {e}")
    
    return results


def compare_strategies(symbols, start_date, end_date, use_mock=True):
    """
    Compare different trading strategies.
    
    Parameters:
    -----------
    symbols : list
        List of symbols to test
    start_date : str
        Start date for backtesting (YYYY-MM-DD)
    end_date : str
        End date for backtesting (YYYY-MM-DD)
    use_mock : bool
        Whether to use mock data
        
    Returns:
    --------
    pandas.DataFrame
        Comparison table of results
    """
    # Initialize strategies
    strategies = {
        'MACD': MACD(fast_period=12, slow_period=26, signal_period=9),
        'RSI': RSI(period=14, overbought=70, oversold=30),
        'Combined': CombinedStrategy(
            macd_fast=12, macd_slow=26, macd_signal=9,
            rsi_period=14, rsi_overbought=70, rsi_oversold=30,
            confirmation_window=3
        )
    }
    
    # Analyze strategy requirements
    strategy_requirements = {}
    min_bars_required = 100  # Default minimum
    
    for name, strategy in strategies.items():
        requirements = analyze_strategy_data_requirements(strategy)
        strategy_requirements[name] = requirements
        min_bars_required = max(min_bars_required, requirements['min_bars'])
    
    logger.info(f"Minimum bars required for all strategies: {min_bars_required}")
    
    # Fetch market data
    logger.info(f"Fetching market data for {len(symbols)} symbols from {start_date} to {end_date}")
    market_data = fetch_market_data(symbols, start_date, end_date, use_mock=use_mock)
    
    # Validate data
    logger.info("Validating dataset...")
    validation_results = validate_dataset(market_data, min_required_bars=min_bars_required)
    
    # Print validation summary
    logger.info(f"Data validation: {validation_results['summary']}")
    
    # Check if any symbols failed validation
    if not validation_results['is_valid_for_backtest']:
        logger.warning("Some symbols have data quality issues:")
        for symbol, result in validation_results['symbol_results'].items():
            if not result['is_valid']:
                logger.warning(f"  - {symbol}: {result['summary']}")
    
    # Prepare data for backtesting (clean and fix issues)
    logger.info("Preparing data for backtesting...")
    market_data = prepare_data_for_backtest(market_data)
    
    # Check if we still have symbols with data
    if not market_data:
        logger.error("No valid data available for backtesting!")
        return pd.DataFrame()
    
    # Store results
    all_results = []
    
    # Run backtests for each symbol and strategy
    for symbol, data in market_data.items():
        # Check if this symbol has enough data for our strategies
        if len(data) < min_bars_required:
            logger.warning(f"Skipping {symbol}: insufficient data ({len(data)} bars, {min_bars_required} required)")
            continue
            
        for strategy_name, strategy in strategies.items():
            try:
                # Get specific requirements for this strategy
                strategy_min_bars = strategy_requirements[strategy_name]['min_bars']
                
                # Check again with strategy-specific requirements
                if len(data) < strategy_min_bars:
                    logger.warning(f"Skipping {symbol} with {strategy_name}: insufficient data for this strategy")
                    continue
                
                # Run the backtest
                logger.info(f"Backtesting {symbol} with {strategy_name} strategy...")
                results = run_backtest(strategy, data, symbol)
                
                # Extract key metrics and add to results list
                all_results.append({
                    'Symbol': symbol,
                    'Strategy': strategy_name,
                    'Total Return (%)': results['total_return_pct'],
                    'Annualized Return (%)': results['annualized_return_pct'],
                    'Sharpe Ratio': results['sharpe_ratio'],
                    'Max Drawdown (%)': results['max_drawdown_pct'],
                    'Win Rate (%)': results['win_rate_pct'],
                    'Total Trades': results['total_trades']
                })
                
                logger.info(f"Completed backtest for {symbol} using {strategy_name} strategy")
                
            except Exception as e:
                logger.error(f"Error running backtest for {symbol} with {strategy_name}: {e}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Check if we have any results
    if results_df.empty:
        logger.error("No backtest results produced!")
        return results_df
    
    # Save comparison table
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = f"results/strategy_comparison_{timestamp}.csv"
        results_df.to_csv(csv_file, index=False)
        logger.info(f"Saved comparison results to {csv_file}")
    except Exception as e:
        logger.error(f"Error saving comparison results: {e}")
    
    # Create visual comparison
    try:
        create_comparison_charts(results_df, timestamp)
    except Exception as e:
        logger.error(f"Error creating comparison charts: {e}")
    
    return results_df


def create_comparison_charts(results_df, timestamp=None):
    """
    Create comparison charts for the backtest results.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with backtest results
    timestamp : str, optional
        Timestamp for filenames
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Check if we have enough data to create charts
    if results_df.empty:
        logger.warning("Cannot create charts: no data available")
        return
        
    # Check if there are enough unique symbols and strategies for comparison
    if len(results_df['Symbol'].unique()) < 1 or len(results_df['Strategy'].unique()) < 1:
        logger.warning("Not enough unique symbols or strategies for comparison charts")
        return
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Chart 1: Total Return by Strategy and Symbol
    plt.figure(figsize=(12, 8))
    
    try:
        # Group by symbol and strategy
        pivot_return = results_df.pivot(index='Symbol', columns='Strategy', values='Total Return (%)')
        
        # Plot grouped bar chart
        pivot_return.plot(kind='bar', ax=plt.gca())
        
        plt.title('Total Return by Strategy and Symbol')
        plt.xlabel('Symbol')
        plt.ylabel('Total Return (%)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Strategy')
        plt.tight_layout()
        plt.savefig(f"results/return_comparison_{timestamp}.png")
        logger.info(f"Saved return comparison chart to results/return_comparison_{timestamp}.png")
    except Exception as e:
        logger.error(f"Error creating return comparison chart: {e}")
    
    # Chart 2: Win Rate by Strategy and Symbol
    try:
        plt.figure(figsize=(12, 8))
        
        # Group by symbol and strategy
        pivot_winrate = results_df.pivot(index='Symbol', columns='Strategy', values='Win Rate (%)')
        
        # Plot grouped bar chart
        pivot_winrate.plot(kind='bar', ax=plt.gca())
        
        plt.title('Win Rate by Strategy and Symbol')
        plt.xlabel('Symbol')
        plt.ylabel('Win Rate (%)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Strategy')
        plt.tight_layout()
        plt.savefig(f"results/winrate_comparison_{timestamp}.png")
        logger.info(f"Saved win rate comparison chart to results/winrate_comparison_{timestamp}.png")
    except Exception as e:
        logger.error(f"Error creating win rate comparison chart: {e}")
    
    # Chart 3: Max Drawdown by Strategy and Symbol
    try:
        plt.figure(figsize=(12, 8))
        
        # Group by symbol and strategy
        pivot_drawdown = results_df.pivot(index='Symbol', columns='Strategy', values='Max Drawdown (%)')
        
        # Plot grouped bar chart
        pivot_drawdown.plot(kind='bar', ax=plt.gca())
        
        plt.title('Max Drawdown by Strategy and Symbol')
        plt.xlabel('Symbol')
        plt.ylabel('Max Drawdown (%)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Strategy')
        plt.tight_layout()
        plt.savefig(f"results/drawdown_comparison_{timestamp}.png")
        logger.info(f"Saved drawdown comparison chart to results/drawdown_comparison_{timestamp}.png")
    except Exception as e:
        logger.error(f"Error creating drawdown comparison chart: {e}")
    
    # Chart 4: Sharpe Ratio by Strategy and Symbol
    try:
        plt.figure(figsize=(12, 8))
        
        # Group by symbol and strategy
        pivot_sharpe = results_df.pivot(index='Symbol', columns='Strategy', values='Sharpe Ratio')
        
        # Plot grouped bar chart
        pivot_sharpe.plot(kind='bar', ax=plt.gca())
        
        plt.title('Sharpe Ratio by Strategy and Symbol')
        plt.xlabel('Symbol')
        plt.ylabel('Sharpe Ratio')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Strategy')
        plt.tight_layout()
        plt.savefig(f"results/sharpe_comparison_{timestamp}.png")
        logger.info(f"Saved Sharpe ratio comparison chart to results/sharpe_comparison_{timestamp}.png")
    except Exception as e:
        logger.error(f"Error creating Sharpe ratio comparison chart: {e}")
    
    # Chart 5: Number of Trades by Strategy and Symbol
    try:
        plt.figure(figsize=(12, 8))
        
        # Group by symbol and strategy
        pivot_trades = results_df.pivot(index='Symbol', columns='Strategy', values='Total Trades')
        
        # Plot grouped bar chart
        pivot_trades.plot(kind='bar', ax=plt.gca())
        
        plt.title('Number of Trades by Strategy and Symbol')
        plt.xlabel('Symbol')
        plt.ylabel('Number of Trades')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Strategy')
        plt.tight_layout()
        plt.savefig(f"results/trades_comparison_{timestamp}.png")
        logger.info(f"Saved trades comparison chart to results/trades_comparison_{timestamp}.png")
    except Exception as e:
        logger.error(f"Error creating trades comparison chart: {e}")
    
    logger.info("Created comparison charts in the results directory")
    plt.close('all')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare trading strategies')
    
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'],
                        help='Symbols to test')
    parser.add_argument('--start-date', default='2022-01-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2023-01-01',
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--use-mock', action='store_true',
                        help='Use mock data instead of real data')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip data validation checks')
    
    args = parser.parse_args()
    
    # Run comparison
    results = compare_strategies(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        use_mock=args.use_mock
    )
    
    # Print summary
    if not results.empty:
        print("\nStrategy Comparison Summary:")
        print(results.to_string(index=False))
        
        # Print conclusion
        try:
            best_strategies = results.loc[results.groupby('Symbol')['Total Return (%)'].idxmax()]
            print("\nBest strategy by total return for each symbol:")
            print(best_strategies[['Symbol', 'Strategy', 'Total Return (%)']].to_string(index=False))
            
            # Print risk-adjusted performance
            best_sharpe = results.loc[results.groupby('Symbol')['Sharpe Ratio'].idxmax()]
            print("\nBest strategy by Sharpe ratio (risk-adjusted) for each symbol:")
            print(best_sharpe[['Symbol', 'Strategy', 'Sharpe Ratio', 'Total Return (%)']].to_string(index=False))
        except Exception as e:
            print(f"\nCould not determine best strategies: {e}")
    else:
        print("\nNo results available. Please check the logs for errors.") 