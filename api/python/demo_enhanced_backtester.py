"""
Demo script for the Enhanced Backtester

This script demonstrates how to use the Enhanced Backtester to backtest
trading strategies and analyze results.
"""

import os
import json
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

from trading_bot.strategies import (
    TrendFollowingStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    BreakoutSwingStrategy,
    VolatilityBreakoutStrategy,
    OptionSpreadsStrategy
)
from trading_bot.utils.enhanced_backtester import EnhancedBacktester
from trading_bot.utils.mock_market_data import generate_mock_market_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("demo_backtester")

def load_config(config_path):
    """Load backtest configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def run_backtest(config_path, output_dir='results', start_date=None, end_date=None, symbols=None, use_mock=True):
    """
    Run backtest using the specified configuration.
    
    Args:
        config_path: Path to the backtest configuration file
        output_dir: Directory to save results
        start_date: Optional start date override
        end_date: Optional end date override
        symbols: Optional symbols override
        use_mock: Whether to use mock market data
    
    Returns:
        Dictionary of backtest results
    """
    # Load configuration
    config = load_config(config_path)
    
    # Override parameters if provided
    if start_date:
        config['start_date'] = start_date
    if end_date:
        config['end_date'] = end_date
    if symbols:
        config['symbols'] = symbols
    
    # Create strategies based on configuration
    strategies = {}
    for strategy_name in config['strategies']:
        if strategy_name == 'trend_following':
            strategies[strategy_name] = TrendFollowingStrategy(
                lookback_period=20,
                exit_lookback=10,
                risk_per_trade=0.02
            )
        elif strategy_name == 'momentum':
            strategies[strategy_name] = MomentumStrategy(
                short_window=10,
                long_window=30,
                rsi_period=14,
                rsi_overbought=70,
                rsi_oversold=30,
                risk_per_trade=0.02
            )
        elif strategy_name == 'mean_reversion':
            strategies[strategy_name] = MeanReversionStrategy(
                lookback_period=20,
                entry_std_dev=2.0,
                exit_std_dev=0.5,
                max_holding_period=10,
                risk_per_trade=0.02
            )
        elif strategy_name == 'breakout_swing':
            strategies[strategy_name] = BreakoutSwingStrategy(
                lookback_period=20,
                entry_breakout_pct=0.03,
                exit_reversal_pct=0.02,
                max_holding_period=15,
                risk_per_trade=0.02
            )
        elif strategy_name == 'volatility_breakout':
            strategies[strategy_name] = VolatilityBreakoutStrategy(
                atr_period=14,
                entry_atr_multiplier=2.0,
                exit_atr_multiplier=1.0,
                max_holding_period=10,
                risk_per_trade=0.02
            )
        elif strategy_name == 'option_spreads':
            strategies[strategy_name] = OptionSpreadsStrategy(
                min_iv_percentile=50,
                max_iv_percentile=85,
                min_days_to_expiration=20,
                max_days_to_expiration=45,
                max_delta=0.30,
                risk_per_trade=0.01
            )
    
    # Get market data
    if config.get('use_mock', use_mock):
        logger.info("Using mock market data for backtest")
        
        # Set default start and end dates if not provided
        if 'start_date' not in config:
            config['start_date'] = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        if 'end_date' not in config:
            config['end_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Set default symbols if not provided
        if 'symbols' not in config:
            config['symbols'] = ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']
        
        # Generate mock market data
        market_data = generate_mock_market_data(
            symbols=config['symbols'],
            start_date=config['start_date'],
            end_date=config['end_date'],
            include_options='option_spreads' in config['strategies']
        )
    else:
        # TODO: In a real implementation, we would fetch real market data
        logger.warning("Real market data fetching not implemented yet, falling back to mock data")
        market_data = generate_mock_market_data(
            symbols=config['symbols'],
            start_date=config['start_date'],
            end_date=config['end_date'],
            include_options='option_spreads' in config['strategies']
        )
    
    # Create backtester
    backtester = EnhancedBacktester(
        strategies=strategies,
        initial_capital=config.get('initial_capital', 100000.0),
        initial_allocations=config.get('initial_allocations', {}),
        commission_rate=config.get('commission_rate', 0.001),
        slippage=config.get('slippage', 0.001),
        rebalance_frequency=config.get('rebalance_frequency', 'weekly')
    )
    
    # Run backtest
    logger.info(f"Starting backtest for {', '.join(config['strategies'])}")
    results = backtester.run_backtest(market_data)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save results
    strategy_names = '-'.join(config['strategies'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f"{output_dir}/backtest_{strategy_names}_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump(results['summary'], f, indent=2)
    
    logger.info(f"Backtest results saved to {result_file}")
    
    # Save detailed trade log
    trade_log_file = f"{output_dir}/trade_log_{strategy_names}_{timestamp}.csv"
    pd.DataFrame(results['trade_log']).to_csv(trade_log_file, index=False)
    
    logger.info(f"Trade log saved to {trade_log_file}")
    
    # Save equity curve
    equity_curve_file = f"{output_dir}/equity_curve_{strategy_names}_{timestamp}.csv"
    pd.DataFrame(results['equity_curve']).to_csv(equity_curve_file, index=False)
    
    logger.info(f"Equity curve saved to {equity_curve_file}")
    
    return results

def visualize_results(results, output_dir='results'):
    """
    Visualize backtest results.
    
    Args:
        results: Dictionary of backtest results
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert equity curve to DataFrame
    equity_curve = pd.DataFrame(results['equity_curve'])
    equity_curve['date'] = pd.to_datetime(equity_curve['date'])
    equity_curve.set_index('date', inplace=True)
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve.index, equity_curve['portfolio_value'], label='Portfolio Value')
    
    # Calculate drawdowns
    rolling_max = equity_curve['portfolio_value'].cummax()
    drawdown = (equity_curve['portfolio_value'] - rolling_max) / rolling_max * 100
    
    # Plot drawdowns on secondary axis
    ax2 = plt.gca().twinx()
    ax2.fill_between(equity_curve.index, drawdown, 0, alpha=0.3, color='r', label='Drawdown %')
    ax2.set_ylabel('Drawdown %')
    
    plt.title('Portfolio Value and Drawdowns')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_file = f"{output_dir}/equity_curve_plot_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    
    plt.close()
    logger.info(f"Equity curve plot saved to {plot_file}")
    
    # Plot strategy contribution if multiple strategies
    if len(results['strategy_returns']) > 1:
        plt.figure(figsize=(12, 6))
        
        strategy_equity = pd.DataFrame()
        strategy_equity['date'] = pd.to_datetime([entry['date'] for entry in results['equity_curve']])
        strategy_equity.set_index('date', inplace=True)
        
        for strategy_name, returns in results['strategy_returns'].items():
            if len(returns) > 0:
                strategy_values = [entry['value'] for entry in returns]
                strategy_dates = pd.to_datetime([entry['date'] for entry in returns])
                temp_df = pd.DataFrame({'value': strategy_values}, index=strategy_dates)
                strategy_equity[strategy_name] = temp_df['value']
        
        strategy_equity.fillna(method='ffill', inplace=True)
        strategy_equity.plot(figsize=(12, 6))
        
        plt.title('Strategy Contribution')
        plt.xlabel('Date')
        plt.ylabel('Strategy Value ($)')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_file = f"{output_dir}/strategy_contribution_plot_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        plt.close()
        logger.info(f"Strategy contribution plot saved to {plot_file}")
    
    # Plot monthly returns heatmap
    if 'monthly_returns' in results:
        monthly_returns = pd.DataFrame(results['monthly_returns'])
        if not monthly_returns.empty:
            # Convert to proper datetime and extract month/year
            monthly_returns['date'] = pd.to_datetime(monthly_returns['date'])
            monthly_returns['year'] = monthly_returns['date'].dt.year
            monthly_returns['month'] = monthly_returns['date'].dt.month
            
            # Pivot the data for the heatmap
            pivot_returns = monthly_returns.pivot_table(
                values='return_pct', 
                index='year', 
                columns='month'
            )
            
            # Create heatmap
            plt.figure(figsize=(12, 6))
            
            # Set colormap with red for negative, green for positive
            cmap = plt.cm.RdYlGn
            
            # Plot heatmap
            im = plt.imshow(
                pivot_returns.values, 
                cmap=cmap,
                vmin=-max(5, abs(pivot_returns.min().min()), abs(pivot_returns.max().max())),
                vmax=max(5, abs(pivot_returns.min().min()), abs(pivot_returns.max().max()))
            )
            
            # Add colorbar
            plt.colorbar(im, label='Return %')
            
            # Add labels
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            plt.xticks(np.arange(12), month_labels)
            plt.yticks(np.arange(pivot_returns.shape[0]), pivot_returns.index)
            
            # Add title and labels
            plt.title('Monthly Returns Heatmap (%)')
            
            # Add values as text
            for i in range(pivot_returns.shape[0]):
                for j in range(pivot_returns.shape[1]):
                    if not np.isnan(pivot_returns.values[i, j]):
                        text = plt.text(j, i, f"{pivot_returns.values[i, j]:.1f}",
                                  ha="center", va="center", color="black" if abs(pivot_returns.values[i, j]) < 10 else "white")
            
            # Save plot
            plot_file = f"{output_dir}/monthly_returns_heatmap_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            
            plt.close()
            logger.info(f"Monthly returns heatmap saved to {plot_file}")

def main():
    """Main function to run demo backtest."""
    parser = argparse.ArgumentParser(description='Enhanced Backtester Demo')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to backtest configuration file')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--start_date', type=str, default=None,
                       help='Start date for the backtest (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                       help='End date for the backtest (YYYY-MM-DD)')
    parser.add_argument('--symbols', type=str, default=None,
                       help='Comma-separated list of symbols to include')
    
    args = parser.parse_args()
    
    # Parse symbols if provided
    symbols = None
    if args.symbols:
        symbols = args.symbols.split(',')
    
    # Run backtest
    results = run_backtest(
        config_path=args.config,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=symbols
    )
    
    # Visualize results
    visualize_results(results, args.output_dir)
    
    # Print summary
    print("\nBacktest Summary:")
    print(f"Total Return: {results['summary']['total_return_pct']:.2f}%")
    print(f"Annualized Return: {results['summary']['annualized_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['summary']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['summary']['max_drawdown_pct']:.2f}%")
    print(f"Win Rate: {results['summary']['win_rate']*100:.2f}%")
    print(f"Profit Factor: {results['summary']['profit_factor']:.2f}")
    
    return results

if __name__ == "__main__":
    main() 