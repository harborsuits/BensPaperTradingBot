"""
Strategy Evaluator for BensBot-EvoTrader Integration

This module provides functionality to evaluate trading strategies through
backtesting, calculating performance metrics, and analyzing results.
"""

import os
import json
import logging
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger("benbot.research.evotrader.evaluator")

class StrategyEvaluator:
    """
    Evaluates trading strategies through backtesting and performance analysis.
    
    This class handles:
    - Running backtests across multiple symbols and timeframes
    - Calculating performance metrics (Sharpe, win rate, drawdown, etc.)
    - Analyzing trade patterns and behaviors
    - Comparing strategies against benchmarks
    """
    
    def __init__(self,
                symbols: List[str] = None,
                timeframes: List[str] = None,
                asset_class: str = "forex",
                output_dir: str = None):
        """
        Initialize the strategy evaluator.
        
        Args:
            symbols: List of symbols to test on
            timeframes: List of timeframes to use
            asset_class: "forex" or "crypto"
            output_dir: Directory for saving evaluation results
        """
        self.symbols = symbols or []
        self.timeframes = timeframes or []
        self.asset_class = asset_class
        
        # Create output directory
        self.output_dir = output_dir or "evaluation_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data connector (lazy-loaded when needed)
        self.data_connector = None
        
        logger.info(f"Strategy evaluator initialized for {asset_class}")
    
    def evaluate(self, strategy, detailed: bool = False) -> Dict[str, Any]:
        """
        Evaluate a strategy through backtesting.
        
        Args:
            strategy: Strategy to evaluate
            detailed: Whether to return detailed metrics
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Lazy-load data connector if not already loaded
        if self.data_connector is None:
            from trading_bot.research.evotrader.data_connector import EvoTraderDataConnector
            self.data_connector = EvoTraderDataConnector(
                cache_dir=os.path.join(self.output_dir, "data_cache")
            )
        
        # Get strategy ID
        strategy_id = getattr(strategy, "strategy_id", str(id(strategy)))
        
        # Run backtest for each symbol and timeframe
        all_results = []
        
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                logger.debug(f"Backtesting {strategy_id} on {symbol} {timeframe}")
                
                # Run backtest
                backtest_result = self._run_backtest(strategy, symbol, timeframe)
                
                if backtest_result:
                    all_results.append(backtest_result)
                    
                    # Save detailed backtest result if requested
                    if detailed:
                        backtest_file = os.path.join(
                            self.output_dir, 
                            f"{strategy_id}_{symbol}_{timeframe}.json"
                        )
                        with open(backtest_file, "w") as f:
                            json.dump(backtest_result, f, indent=2)
        
        # If no successful backtest results, return empty metrics
        if not all_results:
            logger.warning(f"No successful backtest results for {strategy_id}")
            return {"error": "No successful backtest results"}
        
        # Aggregate metrics across all backtests
        aggregated_metrics = self._aggregate_metrics(all_results)
        
        # Add overall stats
        aggregated_metrics["strategy_id"] = strategy_id
        aggregated_metrics["num_symbols_tested"] = len(self.symbols)
        aggregated_metrics["num_timeframes_tested"] = len(self.timeframes)
        aggregated_metrics["timestamp"] = datetime.datetime.now().isoformat()
        
        # If detailed is requested, include all backtest results
        if detailed:
            aggregated_metrics["detailed_results"] = all_results
        
        # Save aggregated metrics
        metrics_file = os.path.join(self.output_dir, f"{strategy_id}_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(aggregated_metrics, f, indent=2)
        
        logger.info(f"Evaluation completed for {strategy_id}")
        return aggregated_metrics
    
    def _run_backtest(self, strategy, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Run a backtest for a specific symbol and timeframe.
        
        Args:
            strategy: Strategy to test
            symbol: Symbol to test on
            timeframe: Timeframe to use
            
        Returns:
            Dictionary with backtest results
        """
        try:
            # Get market data
            if self.asset_class == "forex":
                data = self.data_connector.get_forex_data(symbol, timeframe)
            else:  # crypto
                data = self.data_connector.get_crypto_data(symbol, timeframe)
                
            if data.empty:
                logger.warning(f"No data available for {symbol} {timeframe}")
                return None
            
            # Prepare data for backtest
            price_data = data.copy()
            
            # Run simplified backtest (for demo purposes)
            # In a real implementation, this would use the strategy object to generate
            # signals and execute trades in a more sophisticated backtest engine
            
            # Extract strategy parameters
            params = strategy.parameters if hasattr(strategy, "parameters") else {}
            
            # Generate trading signals based on a simple moving average crossover
            # This is just a placeholder - real implementation would use the strategy's logic
            fast_period = params.get("fast_period", 10)
            slow_period = params.get("slow_period", 20)
            
            price_data['fast_ma'] = price_data['close'].rolling(window=fast_period).mean()
            price_data['slow_ma'] = price_data['close'].rolling(window=slow_period).mean()
            
            # Generate signals
            price_data['signal'] = 0
            price_data.loc[price_data['fast_ma'] > price_data['slow_ma'], 'signal'] = 1
            price_data.loc[price_data['fast_ma'] < price_data['slow_ma'], 'signal'] = -1
            
            # Generate position changes
            price_data['position'] = price_data['signal'].shift(1)
            price_data['position'].fillna(0, inplace=True)
            
            # Calculate daily returns
            price_data['returns'] = price_data['close'].pct_change()
            
            # Calculate strategy returns
            price_data['strategy_returns'] = price_data['position'] * price_data['returns']
            
            # Generate trades
            trades = self._generate_trades_from_signals(price_data, symbol)
            
            # Calculate metrics
            metrics = self._calculate_metrics(price_data, trades)
            
            # Add symbol and timeframe info
            metrics["symbol"] = symbol
            metrics["timeframe"] = timeframe
            
            # Include basic price data stats
            metrics["price_start"] = float(price_data['close'].iloc[0])
            metrics["price_end"] = float(price_data['close'].iloc[-1])
            metrics["price_change_pct"] = float((price_data['close'].iloc[-1] / price_data['close'].iloc[0] - 1) * 100)
            
            # If using the strategy's actual trading logic
            # In a real implementation, we would call the strategy's methods directly
            
            return {
                "metrics": metrics,
                "trades": [t for t in trades[:10]],  # Only include first 10 trades to save space
                "trade_count": len(trades)
            }
            
        except Exception as e:
            logger.error(f"Error running backtest for {symbol} {timeframe}: {e}")
            return None
    
    def _generate_trades_from_signals(self, price_data, symbol):
        """
        Generate trade list from signal data.
        
        Args:
            price_data: DataFrame with signals and positions
            symbol: Symbol being traded
            
        Returns:
            List of trade dictionaries
        """
        trades = []
        current_position = 0
        entry_price = None
        entry_time = None
        
        for idx, row in price_data.iterrows():
            new_position = row['position']
            
            # Position change - close existing and/or open new
            if new_position != current_position:
                # Close existing position if any
                if current_position != 0 and entry_price is not None:
                    exit_price = row['close']
                    pnl = (exit_price - entry_price) * current_position
                    pnl_pct = (exit_price / entry_price - 1) * 100 * current_position
                    
                    trades.append({
                        "symbol": symbol,
                        "entry_time": entry_time.isoformat(),
                        "exit_time": idx.isoformat(),
                        "direction": "LONG" if current_position > 0 else "SHORT",
                        "entry_price": float(entry_price),
                        "exit_price": float(exit_price),
                        "pnl": float(pnl),
                        "pnl_pct": float(pnl_pct)
                    })
                
                # Open new position if not flat
                if new_position != 0:
                    current_position = new_position
                    entry_price = row['close']
                    entry_time = idx
                else:
                    current_position = 0
                    entry_price = None
                    entry_time = None
        
        # Close any open positions at the end
        if current_position != 0 and entry_price is not None:
            exit_price = price_data['close'].iloc[-1]
            pnl = (exit_price - entry_price) * current_position
            pnl_pct = (exit_price / entry_price - 1) * 100 * current_position
            
            trades.append({
                "symbol": symbol,
                "entry_time": entry_time.isoformat(),
                "exit_time": price_data.index[-1].isoformat(),
                "direction": "LONG" if current_position > 0 else "SHORT",
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "pnl": float(pnl),
                "pnl_pct": float(pnl_pct)
            })
        
        return trades
    
    def _calculate_metrics(self, price_data, trades):
        """
        Calculate performance metrics from price data and trades.
        
        Args:
            price_data: DataFrame with price and return data
            trades: List of trade dictionaries
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {}
        
        # Skip if not enough data
        if len(price_data) < 2 or price_data['strategy_returns'].isnull().all():
            return {
                "error": "Insufficient data for metrics calculation",
                "sharpe_ratio": 0,
                "total_return_pct": 0,
                "win_rate_pct": 0,
                "max_drawdown_pct": 0,
                "profit_factor": 0
            }
        
        try:
            # Return metrics
            strategy_returns = price_data['strategy_returns'].dropna()
            if len(strategy_returns) > 0:
                # Cumulative return
                cumulative_return = (1 + strategy_returns).cumprod().iloc[-1] - 1
                metrics["total_return_pct"] = float(cumulative_return * 100)
                
                # Annualized return (assuming 252 trading days per year)
                if len(strategy_returns) > 1:
                    n_days = len(strategy_returns)
                    annualized_return = (1 + cumulative_return) ** (252 / n_days) - 1
                    metrics["annualized_return_pct"] = float(annualized_return * 100)
                
                # Volatility (annualized)
                if len(strategy_returns) > 1:
                    vol = strategy_returns.std() * np.sqrt(252)
                    metrics["annualized_volatility_pct"] = float(vol * 100)
                    
                    # Sharpe ratio (assuming 0% risk-free rate)
                    if vol > 0:
                        sharpe = annualized_return / vol
                        metrics["sharpe_ratio"] = float(sharpe)
                    else:
                        metrics["sharpe_ratio"] = 0
                
                # Drawdown
                cumulative = (1 + strategy_returns).cumprod()
                running_max = cumulative.cummax()
                drawdown = (cumulative / running_max - 1)
                max_drawdown = drawdown.min()
                metrics["max_drawdown_pct"] = float(max_drawdown * 100)
            else:
                metrics["total_return_pct"] = 0
                metrics["annualized_return_pct"] = 0
                metrics["annualized_volatility_pct"] = 0
                metrics["sharpe_ratio"] = 0
                metrics["max_drawdown_pct"] = 0
            
            # Trade metrics
            if trades:
                # Win rate
                winning_trades = [t for t in trades if t["pnl"] > 0]
                win_rate = len(winning_trades) / len(trades)
                metrics["win_rate_pct"] = float(win_rate * 100)
                
                # Average win/loss
                if winning_trades:
                    avg_win = sum(t["pnl"] for t in winning_trades) / len(winning_trades)
                    metrics["avg_win"] = float(avg_win)
                    
                    avg_win_pct = sum(t["pnl_pct"] for t in winning_trades) / len(winning_trades)
                    metrics["avg_win_pct"] = float(avg_win_pct)
                
                losing_trades = [t for t in trades if t["pnl"] <= 0]
                if losing_trades:
                    avg_loss = sum(t["pnl"] for t in losing_trades) / len(losing_trades)
                    metrics["avg_loss"] = float(avg_loss)
                    
                    avg_loss_pct = sum(t["pnl_pct"] for t in losing_trades) / len(losing_trades)
                    metrics["avg_loss_pct"] = float(avg_loss_pct)
                
                # Profit factor
                gross_profit = sum(t["pnl"] for t in winning_trades) if winning_trades else 0
                gross_loss = abs(sum(t["pnl"] for t in losing_trades)) if losing_trades else 0
                
                if gross_loss > 0:
                    profit_factor = gross_profit / gross_loss
                    metrics["profit_factor"] = float(profit_factor)
                else:
                    metrics["profit_factor"] = float('inf') if gross_profit > 0 else 0
                
                # Trade statistics
                metrics["total_trades"] = len(trades)
                metrics["winning_trades"] = len(winning_trades)
                metrics["losing_trades"] = len(losing_trades)
                
                # Average holding period
                if len(trades) > 0:
                    holding_periods = []
                    for trade in trades:
                        try:
                            entry = datetime.datetime.fromisoformat(trade["entry_time"])
                            exit = datetime.datetime.fromisoformat(trade["exit_time"])
                            holding_periods.append((exit - entry).total_seconds() / 3600)  # hours
                        except:
                            pass
                    
                    if holding_periods:
                        metrics["avg_holding_period_hours"] = float(sum(holding_periods) / len(holding_periods))
            else:
                metrics["win_rate_pct"] = 0
                metrics["profit_factor"] = 0
                metrics["total_trades"] = 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                "error": str(e),
                "sharpe_ratio": 0,
                "total_return_pct": 0,
                "win_rate_pct": 0,
                "max_drawdown_pct": 0,
                "profit_factor": 0
            }
    
    def _aggregate_metrics(self, backtest_results):
        """
        Aggregate metrics across multiple backtest results.
        
        Args:
            backtest_results: List of backtest result dictionaries
            
        Returns:
            Dictionary with aggregated metrics
        """
        # Extract all metrics
        all_metrics = [result["metrics"] for result in backtest_results if "metrics" in result]
        
        if not all_metrics:
            return {
                "error": "No metrics found in backtest results",
                "sharpe_ratio": 0,
                "total_return_pct": 0,
                "win_rate_pct": 0,
                "max_drawdown_pct": 0,
                "profit_factor": 0
            }
        
        # Initialize aggregated metrics
        aggregated = {}
        
        # Calculate average for key metrics
        for key in [
            "sharpe_ratio", "total_return_pct", "annualized_return_pct",
            "annualized_volatility_pct", "max_drawdown_pct", "win_rate_pct",
            "profit_factor", "avg_holding_period_hours"
        ]:
            values = [m.get(key, 0) for m in all_metrics]
            if values:
                aggregated[key] = float(sum(values) / len(values))
        
        # Calculate total trades
        total_trades = sum(m.get("total_trades", 0) for m in all_metrics)
        aggregated["total_trades"] = total_trades
        
        # Calculate overall win rate
        winning_trades = sum(m.get("winning_trades", 0) for m in all_metrics)
        aggregated["winning_trades"] = winning_trades
        
        if total_trades > 0:
            aggregated["win_rate_pct"] = float(winning_trades / total_trades * 100)
        
        # Get best and worst performing symbol/timeframe
        best_return_idx = max(range(len(all_metrics)), 
                             key=lambda i: all_metrics[i].get("total_return_pct", 0))
        worst_return_idx = min(range(len(all_metrics)), 
                             key=lambda i: all_metrics[i].get("total_return_pct", 0))
        
        aggregated["best_symbol"] = all_metrics[best_return_idx].get("symbol", "unknown")
        aggregated["best_timeframe"] = all_metrics[best_return_idx].get("timeframe", "unknown")
        aggregated["best_return_pct"] = all_metrics[best_return_idx].get("total_return_pct", 0)
        
        aggregated["worst_symbol"] = all_metrics[worst_return_idx].get("symbol", "unknown")
        aggregated["worst_timeframe"] = all_metrics[worst_return_idx].get("timeframe", "unknown")
        aggregated["worst_return_pct"] = all_metrics[worst_return_idx].get("total_return_pct", 0)
        
        return aggregated
    
    def compare_strategies(self, strategies, benchmark=None):
        """
        Compare multiple strategies against each other and optionally a benchmark.
        
        Args:
            strategies: List of strategies to compare
            benchmark: Optional benchmark strategy
            
        Returns:
            DataFrame with comparison metrics
        """
        results = []
        
        # Evaluate each strategy
        for strategy in strategies:
            metrics = self.evaluate(strategy)
            
            # Add strategy ID
            strategy_id = getattr(strategy, "strategy_id", str(id(strategy)))
            metrics["strategy_id"] = strategy_id
            
            results.append(metrics)
        
        # If benchmark provided, evaluate it too
        if benchmark:
            metrics = self.evaluate(benchmark)
            
            # Add benchmark ID
            benchmark_id = getattr(benchmark, "strategy_id", "benchmark")
            metrics["strategy_id"] = benchmark_id
            metrics["is_benchmark"] = True
            
            results.append(metrics)
        
        # Convert to DataFrame
        if results:
            df = pd.DataFrame(results)
            
            # Save comparison
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_file = os.path.join(self.output_dir, f"strategy_comparison_{timestamp}.csv")
            df.to_csv(comparison_file, index=False)
            
            logger.info(f"Strategy comparison saved to {comparison_file}")
            
            return df
        else:
            return pd.DataFrame()
