#!/usr/bin/env python3
"""
Hybrid Strategy Combiner - Combines signals from multiple top evolved strategies
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from typing import Dict, List, Any, Optional, Union, Callable

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)


class HybridStrategySignals:
    """Signal combination methods for hybrid strategies"""
    
    @staticmethod
    def weighted_vote(signals: List[Dict[str, Any]], 
                     weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Combine signals using weighted voting.
        
        Args:
            signals: List of signal dictionaries from individual strategies
            weights: Optional weights for each strategy (defaults to equal weights)
            
        Returns:
            Combined signal dictionary
        """
        if not signals:
            return {"signal": "none", "confidence": 0, "method": "weighted_vote"}
            
        # Default to equal weights if not provided
        if weights is None:
            weights = [1.0] * len(signals)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0] * len(signals)
            total_weight = sum(weights)
        
        weights = [w / total_weight for w in weights]
        
        # Count weighted votes for each signal type
        vote_counts = {"buy": 0.0, "sell": 0.0, "none": 0.0}
        confidence_sum = {"buy": 0.0, "sell": 0.0, "none": 0.0}
        
        for i, signal in enumerate(signals):
            signal_type = signal.get("signal", "none")
            confidence = signal.get("confidence", 0.0)
            
            # Add weighted vote
            vote_counts[signal_type] += weights[i]
            confidence_sum[signal_type] += confidence * weights[i]
        
        # Determine winning signal
        if vote_counts["buy"] > vote_counts["sell"] and vote_counts["buy"] > vote_counts["none"]:
            final_signal = "buy"
            # Confidence is weighted average of buy confidences
            confidence = confidence_sum["buy"] / max(0.001, vote_counts["buy"])
        elif vote_counts["sell"] > vote_counts["buy"] and vote_counts["sell"] > vote_counts["none"]:
            final_signal = "sell"
            # Confidence is weighted average of sell confidences
            confidence = confidence_sum["sell"] / max(0.001, vote_counts["sell"])
        else:
            final_signal = "none"
            confidence = 0.0
        
        return {
            "signal": final_signal,
            "confidence": confidence,
            "method": "weighted_vote",
            "vote_counts": vote_counts,
            "component_signals": signals
        }
    
    @staticmethod
    def consensus(signals: List[Dict[str, Any]], 
                 threshold: float = 0.6) -> Dict[str, Any]:
        """
        Combine signals using consensus approach (majority must agree).
        
        Args:
            signals: List of signal dictionaries from individual strategies
            threshold: Minimum percentage of strategies that must agree
            
        Returns:
            Combined signal dictionary
        """
        if not signals:
            return {"signal": "none", "confidence": 0, "method": "consensus"}
            
        # Count votes for each signal type
        vote_counts = {"buy": 0, "sell": 0, "none": 0}
        confidence_sum = {"buy": 0.0, "sell": 0.0, "none": 0.0}
        
        for signal in signals:
            signal_type = signal.get("signal", "none")
            confidence = signal.get("confidence", 0.0)
            
            vote_counts[signal_type] += 1
            confidence_sum[signal_type] += confidence
        
        # Calculate percentages
        total_votes = len(signals)
        buy_pct = vote_counts["buy"] / total_votes
        sell_pct = vote_counts["sell"] / total_votes
        
        # Determine if we have consensus
        if buy_pct >= threshold:
            final_signal = "buy"
            # Average confidence of buy signals
            confidence = confidence_sum["buy"] / max(1, vote_counts["buy"])
        elif sell_pct >= threshold:
            final_signal = "sell"
            # Average confidence of sell signals
            confidence = confidence_sum["sell"] / max(1, vote_counts["sell"])
        else:
            final_signal = "none"
            confidence = 0.0
        
        return {
            "signal": final_signal,
            "confidence": confidence,
            "method": "consensus",
            "buy_pct": buy_pct,
            "sell_pct": sell_pct,
            "threshold": threshold,
            "component_signals": signals
        }
    
    @staticmethod
    def confidence_weighted(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine signals weighting by confidence level.
        
        Args:
            signals: List of signal dictionaries from individual strategies
            
        Returns:
            Combined signal dictionary
        """
        if not signals:
            return {"signal": "none", "confidence": 0, "method": "confidence_weighted"}
            
        # Weighted voting based on confidence
        buy_confidence_sum = 0.0
        sell_confidence_sum = 0.0
        
        for signal in signals:
            signal_type = signal.get("signal", "none")
            confidence = signal.get("confidence", 0.0)
            
            if signal_type == "buy":
                buy_confidence_sum += confidence
            elif signal_type == "sell":
                sell_confidence_sum += confidence
        
        # Determine final signal
        if buy_confidence_sum > sell_confidence_sum:
            final_signal = "buy"
            confidence = buy_confidence_sum / len(signals)  # Normalize by number of strategies
        elif sell_confidence_sum > buy_confidence_sum:
            final_signal = "sell"
            confidence = sell_confidence_sum / len(signals)  # Normalize by number of strategies
        else:
            final_signal = "none"
            confidence = 0.0
        
        return {
            "signal": final_signal,
            "confidence": min(1.0, confidence),  # Cap at 1.0
            "method": "confidence_weighted",
            "buy_confidence": buy_confidence_sum,
            "sell_confidence": sell_confidence_sum,
            "component_signals": signals
        }


class HybridStrategy:
    """
    Combines multiple evolved strategies into a single unified strategy.
    
    Features:
    - Multiple signal combination methods (weighted voting, consensus, etc.)
    - Strategy performance weighting
    - Risk management overlay
    - Comprehensive signal analysis
    """
    
    def __init__(self, 
                strategies: List[Any],
                weights: Optional[List[float]] = None,
                combination_method: str = "confidence_weighted",
                consensus_threshold: float = 0.6,
                min_confidence: float = 0.3):
        """
        Initialize the hybrid strategy.
        
        Args:
            strategies: List of strategy objects with calculate_signal method
            weights: Optional list of weights for each strategy
            combination_method: Method for combining signals ('weighted_vote', 'consensus', 'confidence_weighted')
            consensus_threshold: Threshold for consensus method
            min_confidence: Minimum confidence threshold for final signal
        """
        self.name = "HybridStrategy"
        self.strategies = strategies
        self.weights = weights if weights is not None else [1.0] * len(strategies)
        self.combination_method = combination_method
        self.consensus_threshold = consensus_threshold
        self.min_confidence = min_confidence
        
        # Map of combination methods
        self.combination_methods = {
            "weighted_vote": lambda signals: HybridStrategySignals.weighted_vote(signals, self.weights),
            "consensus": lambda signals: HybridStrategySignals.consensus(signals, self.consensus_threshold),
            "confidence_weighted": HybridStrategySignals.confidence_weighted
        }
    
    def calculate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate trading signal by combining component strategy signals.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            Combined signal dictionary
        """
        # Collect signals from all component strategies
        signals = []
        
        for strategy in self.strategies:
            try:
                signal = strategy.calculate_signal(market_data)
                signals.append(signal)
            except Exception as e:
                print(f"Error getting signal from strategy {strategy.__class__.__name__}: {e}")
                # Add a neutral signal as placeholder
                signals.append({"signal": "none", "confidence": 0, "error": str(e)})
        
        # Combine signals using the selected method
        combine_func = self.combination_methods.get(
            self.combination_method, 
            self.combination_methods["confidence_weighted"]
        )
        
        combined_signal = combine_func(signals)
        
        # Apply confidence threshold
        if combined_signal["confidence"] < self.min_confidence:
            combined_signal["signal"] = "none"
            combined_signal["filtered_reason"] = "below_confidence_threshold"
        
        # Add strategy metadata
        combined_signal["strategy_count"] = len(self.strategies)
        combined_signal["strategy_types"] = [s.__class__.__name__ for s in self.strategies]
        
        return combined_signal
    
    def backtest(self, 
                market_data: pd.DataFrame, 
                initial_capital: float = 10000.0,
                position_size_pct: float = 0.95,
                commission_pct: float = 0.001) -> Dict[str, Any]:
        """
        Backtest the hybrid strategy on historical data.
        
        Args:
            market_data: DataFrame with OHLCV data
            initial_capital: Starting capital
            position_size_pct: Percentage of capital to use per trade
            commission_pct: Commission percentage per trade
            
        Returns:
            Dictionary with backtest results
        """
        # Initialize backtest variables
        equity = [initial_capital]
        position = 0
        entry_price = 0
        entry_idx = 0
        trades = []
        
        # Process each day
        for i in range(100, len(market_data)-1):  # Start after warmup period
            # Get data up to current day
            current_data = market_data.iloc[:i+1]
            next_day = market_data.iloc[i+1]
            
            # Calculate signal
            signal = self.calculate_signal(current_data)
            
            # Process signal for next day
            if signal["signal"] == "buy" and position <= 0:
                # Close any existing short position
                if position < 0:
                    exit_price = next_day["open"]
                    profit = (entry_price - exit_price) * abs(position)
                    profit -= abs(position) * exit_price * commission_pct  # Subtract commission
                    
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
                    
                    # Update equity
                    equity.append(equity[-1] + profit)
                else:
                    # No position to close, just copy last equity value
                    equity.append(equity[-1])
                
                # Enter long position
                position = (equity[-1] * position_size_pct) / next_day["open"]
                position -= position * commission_pct  # Subtract commission
                entry_price = next_day["open"]
                entry_idx = i+1
                
                trades.append({
                    "type": "entry_long",
                    "date": next_day.name,
                    "price": entry_price,
                    "position": position,
                    "confidence": signal["confidence"],
                    "method": signal["method"]
                })
            
            elif signal["signal"] == "sell" and position >= 0:
                # Close any existing long position
                if position > 0:
                    exit_price = next_day["open"]
                    profit = (exit_price - entry_price) * position
                    profit -= position * exit_price * commission_pct  # Subtract commission
                    
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
                    
                    # Update equity
                    equity.append(equity[-1] + profit)
                else:
                    # No position to close, just copy last equity value
                    equity.append(equity[-1])
                
                # Enter short position (if enabled)
                position = -(equity[-1] * position_size_pct) / next_day["open"]
                position -= abs(position) * commission_pct  # Subtract commission
                entry_price = next_day["open"]
                entry_idx = i+1
                
                trades.append({
                    "type": "entry_short",
                    "date": next_day.name,
                    "price": entry_price,
                    "position": position,
                    "confidence": signal["confidence"],
                    "method": signal["method"]
                })
            
            else:
                # No action, just copy last equity value
                equity.append(equity[-1])
        
        # Close final position
        if position != 0:
            last_price = market_data["close"].iloc[-1]
            
            if position > 0:
                profit = (last_price - entry_price) * position
                profit -= position * last_price * commission_pct  # Subtract commission
                
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
                profit -= abs(position) * last_price * commission_pct  # Subtract commission
                
                trades.append({
                    "type": "exit_short",
                    "entry_date": market_data.index[entry_idx],
                    "exit_date": market_data.index[-1],
                    "entry_price": entry_price,
                    "exit_price": last_price,
                    "profit": profit,
                    "profit_pct": profit / (entry_price * abs(position)) * 100
                })
            
            # Update final equity
            equity.append(equity[-1] + profit)
        
        # Calculate performance metrics
        if len(trades) > 0:
            # Filter completed trades (entry and exit pairs)
            completed_trades = [t for t in trades if t["type"].startswith("exit")]
            
            # Calculate returns
            total_return_pct = (equity[-1] - initial_capital) / initial_capital * 100
            
            # Calculate winning trades
            winning_trades = [t for t in completed_trades if t["profit"] > 0]
            win_rate = len(winning_trades) / len(completed_trades) * 100 if completed_trades else 0
            
            # Calculate max drawdown
            peak = equity[0]
            max_drawdown = 0
            
            for eq in equity:
                if eq > peak:
                    peak = eq
                drawdown = (peak - eq) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
            
            # Calculate additional metrics
            avg_profit_pct = sum(t["profit_pct"] for t in completed_trades) / len(completed_trades) if completed_trades else 0
            
            profit_trades = [t["profit_pct"] for t in completed_trades if t["profit"] > 0]
            loss_trades = [t["profit_pct"] for t in completed_trades if t["profit"] <= 0]
            
            avg_win = sum(profit_trades) / len(profit_trades) if profit_trades else 0
            avg_loss = sum(loss_trades) / len(loss_trades) if loss_trades else 0
            
            profit_factor = abs(sum(profit_trades) / sum(loss_trades)) if sum(loss_trades) != 0 else float('inf')
        else:
            # No trades executed
            total_return_pct = 0
            win_rate = 0
            max_drawdown = 0
            avg_profit_pct = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            "equity_curve": equity,
            "trades": trades,
            "total_return_pct": total_return_pct,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "trade_count": len(completed_trades) if len(trades) > 0 else 0,
            "avg_profit_pct": avg_profit_pct,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "final_equity": equity[-1]
        }


def create_hybrid_from_deployment(deployment_dir: str, 
                                 combination_method: str = "confidence_weighted") -> HybridStrategy:
    """
    Create a hybrid strategy from deployed strategy files.
    
    Args:
        deployment_dir: Directory containing deployed strategy files
        combination_method: Method for combining signals
        
    Returns:
        HybridStrategy instance
    """
    if not os.path.exists(deployment_dir):
        raise FileNotFoundError(f"Deployment directory not found: {deployment_dir}")
    
    # Import all strategy files
    strategies = []
    strategy_files = []
    
    for entry in os.scandir(deployment_dir):
        if entry.is_file() and entry.name.endswith('.py') and entry.name != '__init__.py':
            strategy_files.append(entry.path)
    
    if not strategy_files:
        raise ValueError(f"No strategy files found in {deployment_dir}")
    
    print(f"Found {len(strategy_files)} strategy files")
    
    # Import and instantiate each strategy
    import importlib.util
    
    for file_path in strategy_files:
        try:
            # Extract file name without extension
            file_name = os.path.basename(file_path)
            module_name = os.path.splitext(file_name)[0]
            
            # Import the module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find strategy class (assumed to be the only class in the file)
            strategy_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and attr_name.startswith('Evolved'):
                    strategy_class = attr
                    break
            
            if strategy_class:
                strategy = strategy_class()
                strategies.append(strategy)
                print(f"Added strategy: {strategy_class.__name__}")
            else:
                print(f"No strategy class found in {file_name}")
        except Exception as e:
            print(f"Error importing strategy from {file_path}: {e}")
    
    if not strategies:
        raise ValueError("No strategies could be imported")
    
    # Create hybrid strategy
    return HybridStrategy(
        strategies=strategies,
        combination_method=combination_method
    )


def main():
    """Main function for running the hybrid strategy combiner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine multiple evolved strategies")
    
    parser.add_argument(
        "--deployment", 
        type=str, 
        default=None,
        help="Path to deployed strategies directory"
    )
    
    parser.add_argument(
        "--method", 
        type=str, 
        default="confidence_weighted",
        choices=["weighted_vote", "consensus", "confidence_weighted"],
        help="Method for combining signals"
    )
    
    parser.add_argument(
        "--symbol", 
        type=str, 
        default="SPY",
        help="Symbol to backtest on"
    )
    
    parser.add_argument(
        "--start", 
        type=str, 
        default="2022-01-01",
        help="Start date for backtest (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end", 
        type=str, 
        default=None,
        help="End date for backtest (YYYY-MM-DD), default is today"
    )
    
    args = parser.parse_args()
    
    # Find latest deployment if not specified
    if args.deployment is None:
        deploy_dirs = []
        deploy_base = "deployments"
        
        if os.path.exists(deploy_base):
            for entry in os.scandir(deploy_base):
                if entry.is_dir() and entry.name.startswith('evolved_strategies_'):
                    deploy_dirs.append(entry.path)
            
            if deploy_dirs:
                deploy_dirs.sort(reverse=True)
                args.deployment = deploy_dirs[0]
                print(f"Using latest deployment: {args.deployment}")
    
    if args.deployment is None:
        print("No deployment directory found or specified")
        return 1
    
    try:
        # Create hybrid strategy
        hybrid = create_hybrid_from_deployment(
            deployment_dir=args.deployment,
            combination_method=args.method
        )
        
        print(f"Created hybrid strategy with {len(hybrid.strategies)} component strategies")
        print(f"Using combination method: {args.method}")
        
        # Get market data
        try:
            import yfinance as yf
            
            print(f"Downloading data for {args.symbol} from {args.start} to {args.end or 'today'}")
            data = yf.download(args.symbol, start=args.start, end=args.end)
            
            if len(data) == 0:
                print(f"No data found for {args.symbol}")
                return 1
                
            print(f"Downloaded {len(data)} data points")
            
            # Run backtest
            print("Running backtest...")
            results = hybrid.backtest(data)
            
            # Display results
            print("\nBacktest Results:")
            print(f"Total Return: {results['total_return_pct']:.2f}%")
            print(f"Win Rate: {results['win_rate']:.1f}%")
            print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
            print(f"Total Trades: {results['trade_count']}")
            print(f"Profit Factor: {results['profit_factor']:.2f}")
            
            # Create output directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"test_results/hybrid_backtest_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save equity curve
            plt.figure(figsize=(12, 6))
            plt.plot(results['equity_curve'])
            plt.title(f"Hybrid Strategy Equity Curve - {args.symbol}")
            plt.xlabel("Trading Days")
            plt.ylabel("Equity ($)")
            plt.grid(True)
            
            equity_path = os.path.join(output_dir, f"{args.symbol}_hybrid_equity.png")
            plt.savefig(equity_path)
            print(f"Saved equity curve to {equity_path}")
            
            # Save detailed report
            trades_df = pd.DataFrame(results['trades'])
            if len(trades_df) > 0:
                trades_path = os.path.join(output_dir, f"{args.symbol}_hybrid_trades.csv")
                trades_df.to_csv(trades_path)
                print(f"Saved trade details to {trades_path}")
            
            report_path = os.path.join(output_dir, f"{args.symbol}_hybrid_report.md")
            with open(report_path, 'w') as f:
                f.write(f"# Hybrid Strategy Backtest Report\n\n")
                f.write(f"Symbol: {args.symbol}\n")
                f.write(f"Period: {args.start} to {args.end or 'Today'}\n")
                f.write(f"Combination Method: {args.method}\n")
                f.write(f"Component Strategies: {len(hybrid.strategies)}\n\n")
                
                f.write(f"## Results\n\n")
                f.write(f"Total Return: {results['total_return_pct']:.2f}%\n")
                f.write(f"Win Rate: {results['win_rate']:.1f}%\n")
                f.write(f"Max Drawdown: {results['max_drawdown']:.2f}%\n")
                f.write(f"Total Trades: {results['trade_count']}\n")
                f.write(f"Average Profit Per Trade: {results['avg_profit_pct']:.2f}%\n")
                f.write(f"Average Win: {results['avg_win']:.2f}%\n")
                f.write(f"Average Loss: {results['avg_loss']:.2f}%\n")
                f.write(f"Profit Factor: {results['profit_factor']:.2f}\n\n")
                
                f.write(f"## Component Strategies\n\n")
                for i, strategy in enumerate(hybrid.strategies):
                    f.write(f"{i+1}. {strategy.__class__.__name__}\n")
            
            print(f"Saved detailed report to {report_path}")
            
        except ImportError:
            print("yfinance not installed. Install with: pip install yfinance")
            return 1
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
