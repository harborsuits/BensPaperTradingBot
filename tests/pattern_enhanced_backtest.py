#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pattern-Enhanced Backtest Runner

This script runs a backtest of the pattern-enhanced contextual strategy
using real market data from Yahoo Finance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional

from pattern_enhanced_strategy import (
    PatternEnhancedStrategy,
    add_indicators,
    detect_market_regime,
    detect_volatility
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PatternBacktest")

class BacktestRunner:
    """
    Runner for pattern-enhanced strategy backtests.
    """
    
    def __init__(self, data_dir: str = "data/market_data"):
        """
        Initialize the backtest runner.
        
        Args:
            data_dir: Directory for storing market data
        """
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize the strategy
        self.strategy = PatternEnhancedStrategy(initial_balance=10000.0)
        
        # Backtest results
        self.results = {
            "trades": [],
            "equity_curve": [],
            "metrics": {},
            "patterns": {}
        }
    
    def load_market_data(self, symbol: str, 
                        start_date: str, 
                        end_date: str,
                        interval: str = "1d",
                        use_cache: bool = True) -> pd.DataFrame:
        """
        Load market data from Yahoo Finance.
        
        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with market data
        """
        # Check if we should use cached data
        cache_file = os.path.join(self.data_dir, f"{symbol.replace('=', '_')}_{interval}.csv")
        
        # Handle forex pairs special format for Yahoo Finance
        if "USD" in symbol and len(symbol) == 6:
            yahoo_symbol = f"{symbol}=X"
        else:
            yahoo_symbol = symbol
            
        if use_cache and os.path.exists(cache_file):
            logger.info(f"Loading cached data for {symbol} from {cache_file}")
            
            # Skip the first 2 rows (which contain multi-level headers in yfinance's output)
            # And use the 3rd row as the header
            try:
                # Try first to load with header interpretation
                data = pd.read_csv(cache_file, skiprows=0)
                
                # Check if we have a multi-level header issue
                if data.shape[1] > 0 and any(col == 'EURUSD=X' for col in data.iloc[0] if isinstance(col, str)):
                    # Reload with proper handling of yfinance's multi-level header format
                    data = pd.read_csv(cache_file, skiprows=[1, 2])
                    
                # Standardize column names
                date_col = data.columns[0] if 'Date' not in data.columns else 'Date'
                data = data.rename(columns={date_col: 'Date'})
                
                # Ensure numeric columns
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in data.columns:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                
                # Handle date column and set index
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                data = data.dropna(subset=['Date'])  # Drop rows with invalid dates
                data.set_index('Date', inplace=True)
                
                # Make sure we have essential columns
                required_columns = ['Open', 'High', 'Low', 'Close']
                missing = [col for col in required_columns if col not in data.columns]
                
                if missing:
                    logger.warning(f"Missing columns {missing} in data, downloading again")
                    use_cache = False
            except Exception as e:
                logger.warning(f"Error loading cached data: {e}. Downloading again.")
                use_cache = False
            
            # Filter by date range if we have valid data
            if use_cache and not data.empty:
                # Make sure data is sorted by date
                data = data.sort_index()
                
                # Filter by date range
                try:
                    data = data.loc[start_date:end_date]
                except KeyError:
                    # If date filtering fails, it might be an index issue
                    logger.warning("Date filtering failed, downloading fresh data")
                    use_cache = False
        
        # Download fresh data if needed
        if not use_cache or not os.path.exists(cache_file):
            logger.info(f"Downloading data for {symbol} from {start_date} to {end_date}")
            
            # Download data
            data = yf.download(
                yahoo_symbol, 
                start=start_date, 
                end=end_date, 
                interval=interval
            )
            
            # Save to cache
            data.to_csv(cache_file)
        
        # Add technical indicators
        data = add_indicators(data)
        
        return data
    
    def run_backtest(self, symbol: str, 
                    start_date: str, 
                    end_date: str,
                    default_stop_loss_pips: int = 50) -> Dict[str, Any]:
        """
        Run backtest with pattern enhanced strategy.
        
        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            default_stop_loss_pips: Default stop loss in pips (will be adjusted with ATR)
            
        Returns:
            Dictionary with backtest results
        """
        # Load data
        data = self.load_market_data(symbol, start_date, end_date)
        
        # Initial account balance
        balance = self.strategy.balance
        initial_balance = balance
        
        # Reset results
        self.results = {
            "trades": [],
            "equity_curve": [],
            "metrics": {},
            "patterns": {},
            "daily_stats": {}
        }
        
        # Store initial equity point
        self.results["equity_curve"].append({
            "date": data.index[0].strftime("%Y-%m-%d"),
            "equity": balance
        })
        
        # Track active trades and performance metrics
        active_trade = None
        recent_trades = []
        max_balance = initial_balance
        drawdown = 0
        current_drawdown_pct = 0
        
        # Run backtest loop
        for i in range(50, len(data)):  # Start after enough bars for indicators
            # Get current window
            current_window = data.iloc[:i+1]
            current_bar = current_window.iloc[-1]
            prev_bar = current_window.iloc[-2]
            current_date = current_window.index[-1]
            
            # Track max balance and drawdown
            if balance > max_balance:
                max_balance = balance
            drawdown = max_balance - balance
            current_drawdown_pct = (drawdown / max_balance) if max_balance > 0 else 0
            
            # Track daily stats
            date_str = current_date.strftime("%Y-%m-%d")
            if date_str not in self.results["daily_stats"]:
                self.results["daily_stats"][date_str] = {
                    "equity": balance,
                    "drawdown": drawdown,
                    "drawdown_pct": current_drawdown_pct * 100,
                    "trades": 0
                }
            
            # Detect market regime and volatility
            market_regime = detect_market_regime(current_window)
            volatility_state = detect_volatility(current_window)
            
            # Update strategy context
            self.strategy.update_context(symbol, market_regime, volatility_state)
            
            # Handle active trade if it exists
            if active_trade:
                # Check for stop loss hit
                if active_trade["direction"] == "buy":
                    stop_hit = current_bar["Low"] <= active_trade["stop_loss_price"]
                    take_profit_hit = current_bar["High"] >= active_trade["take_profit_price"]
                else:  # sell
                    stop_hit = current_bar["High"] >= active_trade["stop_loss_price"]
                    take_profit_hit = current_bar["Low"] <= active_trade["take_profit_price"]
                
                # Close trade if stop loss or take profit was hit
                if stop_hit or take_profit_hit:
                    # Calculate profit/loss
                    if stop_hit:
                        exit_price = active_trade["stop_loss_price"]
                        exit_reason = "stop_loss"
                    else:
                        exit_price = active_trade["take_profit_price"]
                        exit_reason = "take_profit"
                    
                    if active_trade["direction"] == "buy":
                        profit_pips = (exit_price - active_trade["entry_price"]) * 10000
                    else:
                        profit_pips = (active_trade["entry_price"] - exit_price) * 10000
                    
                    # Calculate profit in account currency
                    profit_amount = profit_pips * active_trade["position_size"] * 10
                    
                    # Update balance
                    balance += profit_amount
                    self.strategy.update_balance(balance)
                    
                    # Record trade results
                    trade_result = {
                        "symbol": symbol,
                        "entry_date": active_trade["entry_date"].strftime("%Y-%m-%d"),
                        "exit_date": current_date.strftime("%Y-%m-%d"),
                        "direction": active_trade["direction"],
                        "entry_price": active_trade["entry_price"],
                        "exit_price": exit_price,
                        "position_size": active_trade["position_size"],
                        "stop_loss_pips": active_trade["stop_loss_pips"],
                        "profit_pips": profit_pips,
                        "profit_amount": profit_amount,
                        "exit_reason": exit_reason,
                        "pattern": active_trade["pattern"],
                        "market_regime": active_trade["regime"],
                        "balance_after": balance
                    }
                    
                    # Add trade to recent trades list (keeping last 10)
                    recent_trades.append(trade_result)
                    if len(recent_trades) > 10:
                        recent_trades.pop(0)  # Remove oldest trade
                    
                    self.results["trades"].append(trade_result)
                    
                    # Update daily stats
                    if date_str in self.results["daily_stats"]:
                        self.results["daily_stats"][date_str]["trades"] += 1
                        self.results["daily_stats"][date_str]["equity"] = balance
                    
                    # Update pattern performance
                    self.strategy.update_trade_outcome(
                        symbol=symbol,
                        entry_time=active_trade["entry_date"],
                        exit_time=current_date,
                        direction=active_trade["direction"],
                        pnl=profit_amount,
                        pips=profit_pips,
                        exit_reason=exit_reason,
                        pattern_name=active_trade["pattern"],
                        market_regime=active_trade["regime"]
                    )
                    
                    # Clear active trade
                    active_trade = None
                    
                    # Store equity point
                    self.results["equity_curve"].append({
                        "date": current_date.strftime("%Y-%m-%d"),
                        "equity": balance
                    })
            
            # Look for new signals if no active trade
            if not active_trade:
                # Check drawdown limit for capital preservation
                max_trades_per_day = 3  # Default max trades per day
                
                # ENHANCEMENT: Improved Capital Preservation
                # Reduce trading frequency in drawdown
                if current_drawdown_pct > 0.2:  # 20% drawdown
                    max_trades_per_day = 1  # Limit to 1 trade per day
                    logger.info(f"Reducing trading frequency due to {current_drawdown_pct:.1%} drawdown")
                
                # Check if we've exceeded max trades for today
                today_trades = self.results["daily_stats"].get(date_str, {}).get("trades", 0)
                if today_trades >= max_trades_per_day:
                    continue  # Skip trading for today
                
                # Select strategy with enhanced parameters
                strategy_info = self.strategy.select_strategy(symbol, current_window, recent_trades)
                
                # Check if we have a valid signal
                if (
                    not strategy_info["skip_trading"] and 
                    strategy_info["signal"] in ["buy", "sell"]
                ):
                    # ENHANCEMENT: Adaptive Stop Loss using ATR
                    # Get ATR value from strategy info or calculate it
                    atr_value = strategy_info.get("current_atr") or current_window['ATR'].iloc[-1]
                    
                    # Use ATR-based stop loss calculation
                    atr_multiplier = 1.5
                    atr_based_stop_pips = int(atr_value * 10000 * atr_multiplier)
                    
                    # Ensure minimum stop loss but don't go too tight
                    stop_loss_pips = max(30, min(atr_based_stop_pips, 100))
                    
                    # ENHANCEMENT: Adjust position sizing for drawdown
                    position_sizing_modifier = 1.0
                    
                    # Drastically reduce risk after significant drawdown
                    if current_drawdown_pct > 0.4:  # 40% drawdown
                        position_sizing_modifier *= 0.3  # 70% reduction
                    elif current_drawdown_pct > 0.2:  # 20% drawdown
                        position_sizing_modifier *= 0.7  # 30% reduction
                    
                    # Apply position sizing modifier from consecutive losses
                    if "position_size_modifier" in strategy_info:
                        position_sizing_modifier *= strategy_info["position_size_modifier"]
                    
                    # Calculate position size with all modifiers
                    position_info = self.strategy.calculate_position_size(
                        symbol=symbol,
                        entry_price=current_bar["Close"],
                        stop_loss_pips=stop_loss_pips,
                        account_balance=balance
                    )
                    
                    # Apply the final position size modifier
                    position_size = position_info["position_size"] * position_sizing_modifier
                    
                    # Calculate adaptive take profit based on TP/SL ratio
                    tp_sl_ratio = strategy_info["tp_sl_ratio"]
                    
                    # Calculate stop loss and take profit prices
                    if strategy_info["signal"] == "buy":
                        stop_loss_price = current_bar["Close"] - (stop_loss_pips / 10000)
                        take_profit_price = current_bar["Close"] + (stop_loss_pips * tp_sl_ratio / 10000)
                    else:  # sell
                        stop_loss_price = current_bar["Close"] + (stop_loss_pips / 10000)
                        take_profit_price = current_bar["Close"] - (stop_loss_pips * tp_sl_ratio / 10000)
                    
                    # Create new trade with enhanced parameters
                    active_trade = {
                        "entry_date": current_date,
                        "entry_price": current_bar["Close"],
                        "direction": strategy_info["signal"],
                        "position_size": position_size,
                        "stop_loss_price": stop_loss_price,
                        "take_profit_price": take_profit_price,
                        "stop_loss_pips": stop_loss_pips,
                        "tp_ratio": tp_sl_ratio,
                        "pattern": strategy_info["entry_conditions"].get("pattern", "unknown"),
                        "regime": market_regime,
                        "sizing_modifier": position_sizing_modifier
                    }
                    
                    # Update daily stats
                    if date_str in self.results["daily_stats"]:
                        self.results["daily_stats"][date_str]["trades"] += 1
                    
                    # Record trade start
                    self.strategy.start_trade(
                        symbol=symbol,
                        entry_time=current_date,
                        entry_price=current_bar["Close"],
                        direction=strategy_info["signal"],
                        pattern_name=active_trade["pattern"],
                        market_regime=market_regime
                    )
        
        # Calculate performance metrics
        self._calculate_metrics(initial_balance)
        
        return self.results
    
    def _calculate_metrics(self, initial_balance: float) -> None:
        """
        Calculate backtest performance metrics.
        
        Args:
            initial_balance: Initial account balance
        """
        trades = self.results["trades"]
        
        if not trades:
            logger.warning("No trades were executed in the backtest")
            self.results["metrics"] = {
                "total_trades": 0,
                "net_profit": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
                "max_drawdown_pct": 0,
                "return_pct": 0
            }
            return
        
        # Total trades
        total_trades = len(trades)
        
        # Winning trades
        winning_trades = [t for t in trades if t["profit_pips"] > 0]
        win_count = len(winning_trades)
        
        # Win rate
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Net profit
        net_profit = sum(t["profit_amount"] for t in trades)
        
        # Profit factor
        gross_profit = sum(t["profit_amount"] for t in trades if t["profit_amount"] > 0)
        gross_loss = sum(t["profit_amount"] for t in trades if t["profit_amount"] < 0)
        profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf')
        
        # Max drawdown
        balances = [initial_balance] + [t["balance_after"] for t in trades]
        peak = initial_balance
        drawdowns = []
        
        for balance in balances:
            if balance > peak:
                peak = balance
            drawdown = peak - balance
            drawdowns.append(drawdown)
        
        max_drawdown = max(drawdowns)
        max_drawdown_pct = max_drawdown / peak if peak > 0 else 0
        
        # Final return percentage
        final_balance = balances[-1]
        return_pct = (final_balance - initial_balance) / initial_balance * 100
        
        # Pattern performance
        pattern_stats = self.strategy.pattern_recognizer.pattern_stats
        
        # Store metrics
        self.results["metrics"] = {
            "total_trades": total_trades,
            "winning_trades": win_count,
            "losing_trades": total_trades - win_count,
            "win_rate": win_rate * 100,
            "net_profit": net_profit,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown_pct * 100,
            "return_pct": return_pct,
            "final_balance": final_balance
        }
        
        # Store pattern stats
        self.results["patterns"] = pattern_stats
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        Plot backtest results.
        
        Args:
            save_path: Path to save the plot to
        """
        if not self.results["equity_curve"]:
            logger.warning("No equity data to plot")
            return
        
        # Create a DataFrame from equity curve
        equity_df = pd.DataFrame(self.results["equity_curve"])
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        equity_df.set_index("date", inplace=True)
        
        # Create figure and axes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        equity_df["equity"].plot(ax=ax1)
        ax1.set_title("Equity Curve")
        ax1.set_ylabel("Balance ($)")
        ax1.grid(True)
        
        # Mark trades on the equity curve
        trades = self.results["trades"]
        for trade in trades:
            entry_date = datetime.strptime(trade["entry_date"], "%Y-%m-%d")
            exit_date = datetime.strptime(trade["exit_date"], "%Y-%m-%d")
            
            if trade["profit_amount"] > 0:
                color = 'green'
            else:
                color = 'red'
            
            # Plot the trade on the equity curve
            ax1.axvspan(entry_date, exit_date, alpha=0.2, color=color)
        
        # Plot trade P&L
        if trades:
            trade_df = pd.DataFrame(trades)
            trade_df["entry_date"] = pd.to_datetime(trade_df["entry_date"])
            trade_df.set_index("entry_date", inplace=True)
            
            trade_df["profit_amount"].plot(kind='bar', ax=ax2, color=trade_df["profit_amount"].apply(lambda x: 'green' if x > 0 else 'red'))
            ax2.set_title("Trade P&L")
            ax2.set_ylabel("Profit/Loss ($)")
            ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()
    
    def print_results(self) -> None:
        """Print backtest results to console"""
        metrics = self.results["metrics"]
        
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        
        print(f"Total Trades: {metrics.get('total_trades', 0)}")
        
        # Check if we have any trades before printing detailed metrics
        if metrics.get('total_trades', 0) == 0:
            print("\nNo trades were executed with the current strategy settings.")
            print("This may indicate that the pattern selection criteria are too strict")
            print("or that the current market conditions don't meet the strategy requirements.")
            return
            
        print(f"Winning Trades: {metrics['winning_trades']}")
        print(f"Losing Trades: {metrics['losing_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Net Profit: ${metrics['net_profit']:.2f}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Max Drawdown: ${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.2f}%)")
        print(f"Return: {metrics['return_pct']:.2f}%")
        print(f"Final Balance: ${metrics['final_balance']:.2f}")
        
        print("\n" + "-"*50)
        print("PATTERN PERFORMANCE")
        print("-"*50)
        
        pattern_stats = self.results["patterns"].get("by_pattern", {})
        
        for pattern_name, stats in pattern_stats.items():
            print(f"{pattern_name.upper()}:")
            print(f"  Win Rate: {stats['win_rate']*100:.2f}%")
            print(f"  Total Pips: {stats['total_pips']:.1f}")
            print(f"  Trades: {stats['successes'] + stats['failures']}")
            print("")
        
        print("\n" + "-"*50)
        print("REGIME-SPECIFIC PATTERN PERFORMANCE")
        print("-"*50)
        
        regime_stats = self.results["patterns"].get("by_regime", {})
        
        for regime, patterns in regime_stats.items():
            print(f"\n{regime.upper()} REGIME:")
            
            for pattern_name, stats in patterns.items():
                print(f"  {pattern_name}:")
                print(f"    Win Rate: {stats['win_rate']*100:.2f}%")
                print(f"    Total Pips: {stats['total_pips']:.1f}")
                print(f"    Trades: {stats['successes'] + stats['failures']}")
    
    def save_results(self, file_path: str) -> None:
        """
        Save backtest results to a JSON file.
        
        Args:
            file_path: Path to save results to
        """
        # Create a copy of results for serialization
        results_copy = self.results.copy()
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        logger.info(f"Saved backtest results to {file_path}")

if __name__ == "__main__":
    # Create backtest runner
    runner = BacktestRunner()
    
    # Run backtest on forex pair
    results = runner.run_backtest(
        symbol="EURUSD",
        start_date="2022-01-01",
        end_date="2022-12-31",
        default_stop_loss_pips=50
    )
    
    # Print results
    runner.print_results()
    
    # Plot results
    runner.plot_results(save_path="backtest_results.png")
    
    # Save results
    runner.save_results("backtest_results.json")
