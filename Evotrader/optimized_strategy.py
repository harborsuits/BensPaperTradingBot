#!/usr/bin/env python3
"""
Optimized Trading Strategy Implementation

This script applies the optimized parameters discovered through evolutionary testing
and provides a simple framework for testing them on historical data.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
from typing import Dict, List, Any, Optional

# Add our tools to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class OptimizedMovingAverageCrossover:
    """
    Moving Average Crossover strategy with parameters optimized through evolution.
    """
    
    def __init__(self):
        # Optimized parameters from evolution
        self.parameters = {
            "fast_period": 16,       # Evolution optimized (was 18)
            "slow_period": 159,      # Evolution optimized (much longer than typical)
            "signal_threshold": 0.0375,  # Evolution optimized
            "position_size": 0.1     # Conservative position sizing
        }
        
        self.strategy_name = "OptimizedMovingAverageCrossover"
    
    def calculate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate trading signals based on optimized moving average crossover.
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Signal dictionary
        """
        if market_data is None or len(market_data) < self.parameters["slow_period"]:
            return {"signal": "none", "confidence": 0}
        
        # Calculate moving averages
        fast_ma = market_data['close'].rolling(self.parameters["fast_period"]).mean()
        slow_ma = market_data['close'].rolling(self.parameters["slow_period"]).mean()
        
        # Get current values
        current_fast = fast_ma.iloc[-1]
        current_slow = slow_ma.iloc[-1]
        
        # Calculate crossover conditions
        if len(fast_ma) > 1 and len(slow_ma) > 1:
            prev_fast = fast_ma.iloc[-2]
            prev_slow = slow_ma.iloc[-2]
            
            # Calculate percentage difference between MAs
            ma_diff_pct = abs(current_fast - current_slow) / current_slow
            
            # Check for significant crossovers
            bullish_crossover = (
                current_fast > current_slow and 
                prev_fast <= prev_slow and
                ma_diff_pct > self.parameters["signal_threshold"]
            )
            
            bearish_crossover = (
                current_fast < current_slow and 
                prev_fast >= prev_slow and
                ma_diff_pct > self.parameters["signal_threshold"]
            )
            
            # Generate signals
            if bullish_crossover:
                return {
                    "signal": "buy",
                    "confidence": min(1.0, ma_diff_pct / 0.05),  # Scale confidence
                    "reason": "bullish_crossover"
                }
            elif bearish_crossover:
                return {
                    "signal": "sell",
                    "confidence": min(1.0, ma_diff_pct / 0.05),  # Scale confidence
                    "reason": "bearish_crossover"
                }
        
        # No signal
        return {"signal": "none", "confidence": 0, "reason": "no_crossover"}


class OptimizedBollingerBands:
    """
    Bollinger Bands strategy with parameters optimized through evolution.
    """
    
    def __init__(self):
        # Optimized parameters from evolution
        self.parameters = {
            "period": 32,             # Evolution optimized (was 20)
            "std_dev": 2.35,          # Evolution optimized (was 2.0)
            "signal_threshold": 0.071, # Evolution optimized
            "position_size": 0.1      # Conservative position sizing
        }
        
        self.strategy_name = "OptimizedBollingerBands"
    
    def calculate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate trading signals based on optimized Bollinger Bands.
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Signal dictionary
        """
        if market_data is None or len(market_data) < self.parameters["period"]:
            return {"signal": "none", "confidence": 0}
        
        try:
            # Calculate Bollinger Bands
            rolling_mean = market_data['close'].rolling(window=self.parameters["period"]).mean()
            rolling_std = market_data['close'].rolling(window=self.parameters["period"]).std()
            
            upper_band = rolling_mean + (rolling_std * self.parameters["std_dev"])
            lower_band = rolling_mean - (rolling_std * self.parameters["std_dev"])
            
            # Make sure we have valid data at the end
            if pd.isna(rolling_mean.iloc[-1]) or pd.isna(rolling_std.iloc[-1]):
                return {"signal": "none", "confidence": 0, "reason": "insufficient_data"}
            
            # Current values - ensure we get scalar values, not Series
            current_price = float(market_data['close'].iloc[-1])
            current_upper = float(upper_band.iloc[-1])
            current_lower = float(lower_band.iloc[-1])
            current_middle = float(rolling_mean.iloc[-1])
            
            # Calculate band width as percentage of price
            band_width = float((current_upper - current_lower) / current_middle)
            
            # Calculate normalized position within bands
            if current_upper != current_lower:  # Avoid division by zero
                normalized_position = float((current_price - current_lower) / (current_upper - current_lower))
            else:
                normalized_position = 0.5
            
            # Generate signals based on optimized thresholds
            signal = "none"
            confidence = 0
            reason = "no_signal"
            
            # Calculate thresholds using scalar values
            lower_threshold = current_lower + (band_width * self.parameters["signal_threshold"])
            upper_threshold = current_upper - (band_width * self.parameters["signal_threshold"])
            
            # Price near or below lower band = buy signal
            if current_price <= lower_threshold:
                signal = "buy"
                # Higher confidence when price is further below the band
                confidence = min(1.0, (current_lower - current_price) / current_lower * 5 + 0.5)
                if confidence < 0:
                    confidence = 0.5  # Ensure positive confidence
                reason = "price_at_lower_band"
                
            # Price near or above upper band = sell signal
            elif current_price >= upper_threshold:
                signal = "sell"
                # Higher confidence when price is further above the band
                confidence = min(1.0, (current_price - current_upper) / current_upper * 5 + 0.5)
                if confidence < 0:
                    confidence = 0.5  # Ensure positive confidence
                reason = "price_at_upper_band"
        except Exception as e:
            print(f"Error calculating Bollinger Bands signal: {e}")
            return {"signal": "none", "confidence": 0, "reason": f"error: {str(e)}"}
        
        return {
            "signal": signal,
            "confidence": confidence,
            "reason": reason,
            "indicators": {
                "upper_band": current_upper,
                "middle_band": current_middle,
                "lower_band": current_lower,
                "band_width": band_width,
                "normalized_position": normalized_position
            }
        }


def fetch_historical_data(symbol: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """
    Fetch historical market data for the given symbol.
    
    Args:
        symbol: Ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (defaults to today)
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        # Rename columns to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        print(f"Fetched {len(df)} data points for {symbol}")
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


def backtest_strategy(strategy, market_data: pd.DataFrame, initial_capital: float = 10000.0) -> Dict[str, Any]:
    """
    Backtest a strategy on historical market data.
    
    Args:
        strategy: Strategy object with calculate_signal method
        market_data: DataFrame with OHLCV data
        initial_capital: Initial trading capital
        
    Returns:
        Dictionary with backtest results
    """
    print(f"Backtesting {strategy.strategy_name} on {len(market_data)} data points")
    
    # Initialize backtest variables
    equity = [initial_capital]
    position = 0
    entry_price = 0
    entry_idx = 0  # Track the index where position was entered
    trades = []
    
    # Process each day
    for i in range(100, len(market_data)-1):  # Start after warmup period
        # Get data up to current day
        current_data = market_data.iloc[:i+1]
        next_day = market_data.iloc[i+1]
        
        # Calculate signal
        signal = strategy.calculate_signal(current_data)
        
        # Process signal for next day
        if signal["signal"] == "buy" and position <= 0:
            # Close any existing short position
            if position < 0:
                exit_price = next_day["open"]
                profit = (entry_price - exit_price) * abs(position)
                trade = {
                    "type": "exit_short",
                    "entry_date": market_data.index[entry_idx],
                    "exit_date": next_day.name,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "profit": profit,
                    "profit_pct": profit / (entry_price * abs(position)) * 100
                }
                trades.append(trade)
            
            # Enter long position
            position = (equity[-1] * 0.95) / next_day["open"]  # Use 95% of capital
            entry_price = next_day["open"]
            entry_idx = i+1  # Store the entry index (next day)
            trades.append({
                "type": "entry_long",
                "date": next_day.name,
                "price": entry_price,
                "position": position
            })
            
        elif signal["signal"] == "sell" and position >= 0:
            # Close any existing long position
            if position > 0:
                exit_price = next_day["open"]
                profit = (exit_price - entry_price) * position
                trade = {
                    "type": "exit_long",
                    "entry_date": market_data.index[entry_idx],
                    "exit_date": next_day.name,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "profit": profit,
                    "profit_pct": profit / (entry_price * position) * 100
                }
                trades.append(trade)
            
            # Enter short position (if enabled)
            position = -(equity[-1] * 0.95) / next_day["open"]  # Use 95% of capital
            entry_price = next_day["open"]
            entry_idx = i+1  # Store the entry index (next day)
            trades.append({
                "type": "entry_short",
                "date": next_day.name,
                "price": entry_price,
                "position": position
            })
        
        # Calculate equity
        if position > 0:
            # Long position
            current_equity = equity[0] + (next_day["close"] - entry_price) * position
        elif position < 0:
            # Short position
            current_equity = equity[0] + (entry_price - next_day["close"]) * abs(position)
        else:
            # No position
            current_equity = equity[-1]
        
        equity.append(current_equity)
    
    # Close final position
    if position != 0:
        last_price = market_data["close"].iloc[-1]
        if position > 0:
            profit = (last_price - entry_price) * position
            trades.append({
                "type": "exit_long",
                "entry_date": market_data.index[entry_idx],
                "exit_date": market_data.index[-1],
                "entry_price": entry_price,
                "exit_price": last_price,
                "profit": profit,
                "profit_pct": profit / (entry_price * position) * 100
            })
        else:
            profit = (entry_price - last_price) * abs(position)
            trades.append({
                "type": "exit_short",
                "entry_date": market_data.index[entry_idx],
                "exit_date": market_data.index[-1],
                "entry_price": entry_price,
                "exit_price": last_price,
                "profit": profit,
                "profit_pct": profit / (entry_price * abs(position)) * 100
            })
    
    # Calculate performance metrics
    completed_trades = [t for t in trades if "profit" in t]
    win_trades = [t for t in completed_trades if t["profit"] > 0]
    lose_trades = [t for t in completed_trades if t["profit"] <= 0]
    
    total_profit = sum(t["profit"] for t in completed_trades)
    win_rate = len(win_trades) / len(completed_trades) if completed_trades else 0
    
    # Calculate drawdown
    max_equity = equity[0]
    max_drawdown = 0
    
    for eq in equity:
        if eq > max_equity:
            max_equity = eq
        drawdown = (max_equity - eq) / max_equity if max_equity > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
    
    # Calculate returns
    final_equity = equity[-1]
    total_return = (final_equity - initial_capital) / initial_capital
    
    results = {
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return": total_return,
        "total_return_pct": total_return * 100,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown * 100,
        "total_trades": len(completed_trades),
        "win_trades": len(win_trades),
        "lose_trades": len(lose_trades),
        "win_rate": win_rate,
        "win_rate_pct": win_rate * 100,
        "total_profit": total_profit,
        "equity_curve": equity,
        "trades": completed_trades
    }
    
    return results


def plot_equity_curve(results: Dict[str, Any], symbol: str, strategy_name: str, save_path: str = None):
    """
    Plot equity curve from backtest results.
    
    Args:
        results: Backtest results dictionary
        symbol: Ticker symbol
        strategy_name: Name of the strategy
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(results["equity_curve"])
    plt.title(f"Equity Curve: {strategy_name} on {symbol}")
    plt.xlabel("Trading Days")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved equity curve to {save_path}")
    
    plt.show()


def generate_report(results: Dict[str, Any], symbol: str, strategy_name: str, save_path: str = None) -> str:
    """
    Generate a markdown report of backtest results.
    
    Args:
        results: Backtest results dictionary
        symbol: Ticker symbol
        strategy_name: Name of the strategy
        save_path: Optional path to save the report
        
    Returns:
        Report text
    """
    report = f"# Backtest Report: {strategy_name} on {symbol}\n\n"
    report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report += "## Performance Summary\n\n"
    report += f"- **Initial Capital:** ${results['initial_capital']:.2f}\n"
    report += f"- **Final Equity:** ${results['final_equity']:.2f}\n"
    report += f"- **Total Return:** {results['total_return_pct']:.2f}%\n"
    report += f"- **Max Drawdown:** {results['max_drawdown_pct']:.2f}%\n"
    report += f"- **Win Rate:** {results['win_rate_pct']:.2f}%\n"
    report += f"- **Total Trades:** {results['total_trades']}\n"
    report += f"- **Winning Trades:** {results['win_trades']}\n"
    report += f"- **Losing Trades:** {results['lose_trades']}\n"
    report += f"- **Total Profit:** ${results['total_profit']:.2f}\n\n"
    
    report += "## Trade Analysis\n\n"
    
    # Sort trades by profit
    winning_trades = sorted([t for t in results["trades"] if t["profit"] > 0], 
                           key=lambda x: x["profit"], reverse=True)
    losing_trades = sorted([t for t in results["trades"] if t["profit"] <= 0], 
                          key=lambda x: x["profit"])
    
    # Show top winning trades
    report += "### Top Winning Trades\n\n"
    if winning_trades:
        report += "| Entry Date | Exit Date | Type | Entry | Exit | Profit | Return |\n"
        report += "| --- | --- | --- | --- | --- | --- | --- |\n"
        
        for trade in winning_trades[:5]:  # Show top 5
            trade_type = "Long" if trade["type"] == "exit_long" else "Short"
            report += f"| {trade['entry_date'].strftime('%Y-%m-%d')} | "
            report += f"{trade['exit_date'].strftime('%Y-%m-%d')} | "
            report += f"{trade_type} | "
            report += f"${trade['entry_price']:.2f} | "
            report += f"${trade['exit_price']:.2f} | "
            report += f"${trade['profit']:.2f} | "
            report += f"{trade['profit_pct']:.2f}% |\n"
    else:
        report += "No winning trades.\n"
    
    report += "\n"
    
    # Show worst losing trades
    report += "### Worst Losing Trades\n\n"
    if losing_trades:
        report += "| Entry Date | Exit Date | Type | Entry | Exit | Loss | Return |\n"
        report += "| --- | --- | --- | --- | --- | --- | --- |\n"
        
        for trade in losing_trades[:5]:  # Show worst 5
            trade_type = "Long" if trade["type"] == "exit_long" else "Short"
            report += f"| {trade['entry_date'].strftime('%Y-%m-%d')} | "
            report += f"{trade['exit_date'].strftime('%Y-%m-%d')} | "
            report += f"{trade_type} | "
            report += f"${trade['entry_price']:.2f} | "
            report += f"${trade['exit_price']:.2f} | "
            report += f"${trade['profit']:.2f} | "
            report += f"{trade['profit_pct']:.2f}% |\n"
    else:
        report += "No losing trades.\n"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Saved report to {save_path}")
    
    return report


def main():
    """Run backtest on optimized strategies."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test optimized trading strategies")
    parser.add_argument("--symbol", type=str, default="SPY", help="Ticker symbol to test")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--strategy", type=str, default="both", help="Strategy to test (ma, bb, or both)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = f"test_results/backtest_{int(datetime.now().timestamp())}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Fetch historical data
    market_data = fetch_historical_data(args.symbol, args.start, args.end)
    
    if market_data.empty:
        print(f"No data available for {args.symbol}")
        return
    
    # Run backtest for selected strategies
    if args.strategy.lower() in ["ma", "both"]:
        # Test Moving Average Crossover
        ma_strategy = OptimizedMovingAverageCrossover()
        ma_results = backtest_strategy(ma_strategy, market_data, args.capital)
        
        # Plot results
        plot_equity_curve(
            ma_results,
            args.symbol,
            ma_strategy.strategy_name,
            os.path.join(output_dir, f"{args.symbol}_{ma_strategy.strategy_name}_equity.png")
        )
        
        # Generate report
        generate_report(
            ma_results,
            args.symbol,
            ma_strategy.strategy_name,
            os.path.join(output_dir, f"{args.symbol}_{ma_strategy.strategy_name}_report.md")
        )
        
        print(f"\n{ma_strategy.strategy_name} Results:")
        print(f"Total Return: {ma_results['total_return_pct']:.2f}%")
        print(f"Win Rate: {ma_results['win_rate_pct']:.2f}%")
        print(f"Max Drawdown: {ma_results['max_drawdown_pct']:.2f}%")
        print(f"Total Trades: {ma_results['total_trades']}")
    
    if args.strategy.lower() in ["bb", "both"]:
        # Test Bollinger Bands
        bb_strategy = OptimizedBollingerBands()
        bb_results = backtest_strategy(bb_strategy, market_data, args.capital)
        
        # Plot results
        plot_equity_curve(
            bb_results,
            args.symbol,
            bb_strategy.strategy_name,
            os.path.join(output_dir, f"{args.symbol}_{bb_strategy.strategy_name}_equity.png")
        )
        
        # Generate report
        generate_report(
            bb_results,
            args.symbol,
            bb_strategy.strategy_name,
            os.path.join(output_dir, f"{args.symbol}_{bb_strategy.strategy_name}_report.md")
        )
        
        print(f"\n{bb_strategy.strategy_name} Results:")
        print(f"Total Return: {bb_results['total_return_pct']:.2f}%")
        print(f"Win Rate: {bb_results['win_rate_pct']:.2f}%")
        print(f"Max Drawdown: {bb_results['max_drawdown_pct']:.2f}%")
        print(f"Total Trades: {bb_results['total_trades']}")
    
    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
