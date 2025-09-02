#!/usr/bin/env python3
"""
Demo script for the PerformanceAnalyzer class.

This script demonstrates the functionality of the PerformanceAnalyzer class
by generating simulated trading data for multiple strategies and analyzing
their performance.

To run this script:
    python demo_performance_analyzer.py --strategies momentum trend_following mean_reversion --days 180 --num_trades 50 --plot --report
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt

# Add the parent directory to sys.path to import the PerformanceAnalyzer class
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trading_bot.performance.performance_analyzer import PerformanceAnalyzer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Demo script for PerformanceAnalyzer")
    
    parser.add_argument("--strategies", type=str, nargs="+", 
                        default=["momentum", "mean_reversion", "trend_following", "breakout"],
                        help="List of strategies to simulate")
    
    parser.add_argument("--days", type=int, default=180,
                        help="Number of days of data to generate")
    
    parser.add_argument("--num_trades", type=int, default=50,
                        help="Number of trades to generate for each strategy")
    
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save output files")
    
    parser.add_argument("--plot", action="store_true",
                        help="Generate and display plots")
    
    parser.add_argument("--report", action="store_true",
                        help="Generate and print performance report")
    
    return parser.parse_args()

def generate_random_returns(days: int, trend: float = 0.05, volatility: float = 0.01) -> Dict[str, float]:
    """
    Generate random daily returns for a strategy.
    
    Args:
        days: Number of days to generate returns for
        trend: Annualized trend (positive or negative)
        volatility: Annualized volatility
    
    Returns:
        Dictionary mapping dates to daily returns
    """
    # Daily trend and volatility
    daily_trend = trend / 252
    daily_vol = volatility / np.sqrt(252)
    
    # Generate random returns
    returns = np.random.normal(daily_trend, daily_vol, days)
    
    # Create dates (starting from today and going backward)
    end_date = datetime.now().date()
    dates = [(end_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
    
    # Create dictionary mapping dates to returns
    return {date: float(ret * 100) for date, ret in zip(dates, returns)}

def generate_random_trades(num_trades: int, 
                           start_date: datetime, 
                           end_date: datetime,
                           win_rate: float = 0.6,
                           avg_win: float = 2.0,
                           avg_loss: float = -1.0,
                           volatility: float = 0.3) -> List[Dict]:
    """
    Generate random trades for a strategy.
    
    Args:
        num_trades: Number of trades to generate
        start_date: Start date for trades
        end_date: End date for trades
        win_rate: Percentage of winning trades (0-1)
        avg_win: Average win percentage
        avg_loss: Average loss percentage (negative value)
        volatility: Volatility of trade returns
    
    Returns:
        List of trade dictionaries
    """
    trades = []
    date_range = (end_date - start_date).days
    
    for i in range(num_trades):
        # Randomize the trade duration (1-10 days)
        duration = random.randint(1, 10)
        
        # Randomize the exit date
        days_from_start = random.randint(0, date_range - duration)
        entry_date = start_date + timedelta(days=days_from_start)
        exit_date = entry_date + timedelta(days=duration)
        
        # Randomize the position size ($1,000 - $10,000)
        position_size = random.randint(1000, 10000)
        
        # Determine if it's a winning trade
        is_win = random.random() < win_rate
        
        # Generate the P&L percentage
        if is_win:
            pnl_pct = random.normalvariate(avg_win, volatility)
        else:
            pnl_pct = random.normalvariate(avg_loss, volatility * 0.5)
        
        # Calculate the P&L amount
        pnl = position_size * pnl_pct / 100
        
        # Generate random symbol
        symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "AMD", "INTC", "IBM"]
        symbol = random.choice(symbols)
        
        # Create the trade dictionary
        trade = {
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "exit_date": exit_date.strftime("%Y-%m-%d"),
            "symbol": symbol,
            "direction": "long" if random.random() < 0.7 else "short",
            "position_size": position_size,
            "entry_price": round(random.uniform(50, 500), 2),
            "exit_price": None,  # Will be calculated based on P&L
            "pnl": round(pnl, 2),
            "pnl_percent": round(pnl_pct, 2),
            "trade_duration": duration,
            "strategy_conviction": random.uniform(0.6, 1.0)
        }
        
        # Calculate exit price based on entry price and P&L percentage
        if trade["direction"] == "long":
            trade["exit_price"] = round(trade["entry_price"] * (1 + pnl_pct / 100), 2)
        else:
            trade["exit_price"] = round(trade["entry_price"] * (1 - pnl_pct / 100), 2)
        
        trades.append(trade)
    
    # Sort trades by exit date
    trades.sort(key=lambda x: x["exit_date"])
    
    return trades

def generate_strategy_data(strategy: str, 
                          days: int, 
                          num_trades: int,
                          strategy_params: Dict) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Generate simulated data for a strategy.
    
    Args:
        strategy: Name of the strategy
        days: Number of days of data
        num_trades: Number of trades to generate
        strategy_params: Parameters for data generation
    
    Returns:
        Tuple of (daily_returns, trades)
    """
    # Generate daily returns
    daily_returns = generate_random_returns(
        days=days,
        trend=strategy_params["trend"],
        volatility=strategy_params["volatility"]
    )
    
    # Generate trades
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    
    trades = generate_random_trades(
        num_trades=num_trades,
        start_date=datetime.combine(start_date, datetime.min.time()),
        end_date=datetime.combine(end_date, datetime.min.time()),
        win_rate=strategy_params["win_rate"],
        avg_win=strategy_params["avg_win"],
        avg_loss=strategy_params["avg_loss"],
        volatility=strategy_params["volatility"] * 20  # Scale volatility for trades
    )
    
    return daily_returns, trades

def print_separator(title: str = "", length: int = 80) -> None:
    """Print a separator line with an optional title."""
    if title:
        padding = (length - len(title) - 4) // 2
        print("=" * padding + f"[ {title} ]" + "=" * padding)
    else:
        print("=" * length)

def main():
    """Main function for the demo script."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Strategy parameters for data generation
    strategy_params = {
        "momentum": {
            "trend": 0.15,        # 15% annualized return
            "volatility": 0.18,   # 18% annualized volatility
            "win_rate": 0.55,     # 55% win rate
            "avg_win": 2.5,       # 2.5% average win
            "avg_loss": -1.5      # -1.5% average loss
        },
        "mean_reversion": {
            "trend": 0.10,        # 10% annualized return
            "volatility": 0.12,   # 12% annualized volatility
            "win_rate": 0.65,     # 65% win rate
            "avg_win": 1.8,       # 1.8% average win
            "avg_loss": -1.2      # -1.2% average loss
        },
        "trend_following": {
            "trend": 0.18,        # 18% annualized return
            "volatility": 0.22,   # 22% annualized volatility
            "win_rate": 0.45,     # 45% win rate
            "avg_win": 3.5,       # 3.5% average win
            "avg_loss": -1.8      # -1.8% average loss
        },
        "breakout": {
            "trend": 0.12,        # 12% annualized return
            "volatility": 0.20,   # 20% annualized volatility
            "win_rate": 0.50,     # 50% win rate
            "avg_win": 2.8,       # 2.8% average win
            "avg_loss": -1.6      # -1.6% average loss
        },
        "volatility_breakout": {
            "trend": 0.14,        # 14% annualized return
            "volatility": 0.25,   # 25% annualized volatility
            "win_rate": 0.48,     # 48% win rate
            "avg_win": 3.0,       # 3.0% average win
            "avg_loss": -1.7      # -1.7% average loss
        }
    }
    
    # Add default parameters for any strategy not in the predefined list
    for strategy in args.strategies:
        if strategy not in strategy_params:
            strategy_params[strategy] = {
                "trend": random.uniform(0.08, 0.18),
                "volatility": random.uniform(0.12, 0.25),
                "win_rate": random.uniform(0.45, 0.65),
                "avg_win": random.uniform(1.5, 3.5),
                "avg_loss": random.uniform(-1.0, -2.0)
            }
    
    # Initialize the performance analyzer
    data_dir = os.path.join(args.output_dir, "performance_data")
    os.makedirs(data_dir, exist_ok=True)
    
    analyzer = PerformanceAnalyzer(
        strategies=args.strategies,
        data_directory=data_dir,
        lookback_periods={
            "recent": 30,
            "medium": 90,
            "long": args.days
        }
    )
    
    print_separator("Initializing PerformanceAnalyzer Demo", 80)
    print(f"Strategies: {', '.join(args.strategies)}")
    print(f"Days of data: {args.days}")
    print(f"Number of trades per strategy: {args.num_trades}")
    print(f"Output directory: {args.output_dir}")
    print_separator()
    
    # Generate data for each strategy and add it to the analyzer
    for strategy in args.strategies:
        print(f"Generating data for {strategy}...")
        
        # Get strategy parameters or use defaults
        params = strategy_params.get(strategy, {
            "trend": 0.10,
            "volatility": 0.15,
            "win_rate": 0.55,
            "avg_win": 2.0,
            "avg_loss": -1.5
        })
        
        # Generate daily returns and trades
        daily_returns, trades = generate_strategy_data(
            strategy=strategy,
            days=args.days,
            num_trades=args.num_trades,
            strategy_params=params
        )
        
        # Add daily returns to the analyzer
        for date, return_value in daily_returns.items():
            analyzer.add_daily_return(strategy, date, return_value)
        
        # Add trades to the analyzer
        for trade in trades:
            analyzer.add_trade(strategy, trade)
        
        print(f"Added {len(daily_returns)} daily returns and {len(trades)} trades for {strategy}")
    
    print_separator()
    
    # Compare strategies
    print_separator("Strategy Comparison (All Time)", 80)
    comparison_df = analyzer.compare_strategies(period="all_time")
    print(comparison_df.to_string())
    print_separator()
    
    # Show recent performance
    print_separator("Recent Performance (Last 30 Days)", 80)
    recent_comparison = analyzer.compare_strategies(period="recent")
    print(recent_comparison.to_string())
    print_separator()
    
    # Display top performing strategies
    print_separator("Top Performing Strategies", 80)
    
    print("By Total Return (All Time):")
    top_by_return = analyzer.get_top_performing_strategies(
        metric="total_return", period="all_time"
    )
    for strategy, value in top_by_return:
        print(f"  {strategy}: {value:.2f}%")
    
    print("\nBy Sharpe Ratio (All Time):")
    top_by_sharpe = analyzer.get_top_performing_strategies(
        metric="sharpe_ratio", period="all_time"
    )
    for strategy, value in top_by_sharpe:
        print(f"  {strategy}: {value:.2f}")
    
    print("\nBy Win Rate (All Time):")
    top_by_win_rate = analyzer.get_top_performing_strategies(
        metric="win_rate", period="all_time"
    )
    for strategy, value in top_by_win_rate:
        print(f"  {strategy}: {value:.2f}%")
    
    print_separator()
    
    # Generate and display plots
    if args.plot:
        print("Generating plots...")
        
        # Plot equity curves
        equity_curve_path = os.path.join(args.output_dir, "equity_curves.png")
        analyzer.plot_equity_curves(
            strategies=args.strategies,
            figsize=(12, 8),
            save_path=equity_curve_path
        )
        print(f"Saved equity curves plot to {equity_curve_path}")
        
        # Plot performance metrics comparison
        plt.figure(figsize=(12, 8))
        all_metrics = comparison_df[["Total Return (%)", "Ann. Return (%)", "Sharpe Ratio"]]
        all_metrics.plot(kind="bar")
        plt.title("Performance Metrics Comparison")
        plt.ylabel("Value")
        plt.grid(axis="y")
        plt.tight_layout()
        
        metrics_path = os.path.join(args.output_dir, "performance_metrics.png")
        plt.savefig(metrics_path, dpi=300)
        print(f"Saved performance metrics plot to {metrics_path}")
        
        # Generate an example visualization for a specific strategy
        if args.strategies:
            example_strategy = args.strategies[0]
            trades = analyzer.get_strategy_trades(example_strategy)
            
            if trades:
                # Extract trade data
                exit_dates = [datetime.strptime(t["exit_date"], "%Y-%m-%d") for t in trades]
                pnl_values = [t["pnl"] for t in trades]
                
                # Create a plot
                plt.figure(figsize=(12, 6))
                colors = ["green" if pnl > 0 else "red" for pnl in pnl_values]
                plt.bar(exit_dates, pnl_values, color=colors)
                plt.title(f"{example_strategy} Trade P&L")
                plt.xlabel("Exit Date")
                plt.ylabel("P&L ($)")
                plt.grid(axis="y")
                plt.tight_layout()
                
                trade_pnl_path = os.path.join(args.output_dir, f"{example_strategy}_trade_pnl.png")
                plt.savefig(trade_pnl_path, dpi=300)
                print(f"Saved trade P&L plot to {trade_pnl_path}")
        
        print_separator()
    
    # Generate and print performance report
    if args.report:
        print("Generating performance report...")
        
        # Generate report for all strategies
        report = analyzer.generate_performance_report(period="all_time", include_trades=False)
        
        # Save report to file
        report_path = os.path.join(args.output_dir, "performance_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Saved performance report to {report_path}")
        
        print_separator("Performance Report Summary", 80)
        
        # Display report summary
        print(f"Report Date: {datetime.fromisoformat(report['report_date']).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Period: {report['period']}")
        print(f"Number of strategies: {len(report['strategies'])}")
        
        # Display top performers
        print("\nTop Performers by Total Return:")
        for strategy, value in report['top_performers']['total_return'].items():
            print(f"  {strategy}: {value:.2f}%")
        
        print("\nTop Performers by Sharpe Ratio:")
        for strategy, value in report['top_performers']['sharpe_ratio'].items():
            print(f"  {strategy}: {value:.2f}")
        
        print_separator()
    
    print("Demo completed successfully.")

if __name__ == "__main__":
    main() 