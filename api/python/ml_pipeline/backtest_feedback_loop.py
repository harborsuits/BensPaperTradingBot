"""
Backtest Feedback Loop - Self-Learning System
This module provides a feedback mechanism that automatically captures backtest results,
stores them in a structured format, and uses them to refine strategy-symbol pairs over time.
"""

import os
import sys
import json
import time
import logging
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import threading

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from our components
from trading_bot.market_context.market_context import get_market_context
from trading_bot.symbolranker.symbol_ranker import get_symbol_ranker
from trading_bot.ml_pipeline.adaptive_trainer import get_adaptive_trainer

class BacktestFeedbackSystem:
    """
    System that captures backtest results and uses them to automatically
    refine strategy-symbol pair rankings over time.
    """
    
    def __init__(self, config=None):
        """
        Initialize the backtest feedback system.
        
        Args:
            config: Configuration dictionary or None to use defaults
        """
        self._config = config or {}
        
        # Set up logging
        self.logger = logging.getLogger("BacktestFeedbackSystem")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Performance tracking
        self.results_db_path = self._config.get("results_db_path", "data/backtest_results")
        os.makedirs(self.results_db_path, exist_ok=True)
        
        # Current performance stats
        self.performance_stats = self._load_performance_stats()
        
        # Pair ranking adjustments
        self.pair_adjustments = {}
        
        # Thread lock for thread safety
        self._lock = threading.RLock()
        
        self.logger.info("BacktestFeedbackSystem initialized")
    
    def store_backtest_result(self, symbol, strategy, backtest_results):
        """
        Store a backtest result and update the feedback system.
        
        Args:
            symbol: Stock symbol
            strategy: Strategy ID
            backtest_results: Dictionary containing backtest results
        
        Returns:
            Boolean indicating success
        """
        self.logger.info(f"Storing backtest result for {symbol} with {strategy}")
        
        try:
            with self._lock:
                # Create a record with metadata
                record = {
                    "symbol": symbol,
                    "strategy": strategy,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "results": backtest_results
                }
                
                # Save to disk
                filename = f"backtest_{symbol}_{strategy}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                filepath = os.path.join(self.results_db_path, filename)
                
                with open(filepath, 'w') as f:
                    json.dump(record, f, indent=2)
                
                # Update performance stats
                self._update_performance_stats(symbol, strategy, backtest_results)
                
                # Update pair ranking adjustments
                self._update_pair_adjustments(symbol, strategy, backtest_results)
                
                # Apply the knowledge to the ML system
                self._apply_to_ml_system(symbol, strategy, backtest_results)
                
                self.logger.info(f"Backtest result stored successfully: {filepath}")
                
                return True
        
        except Exception as e:
            self.logger.error(f"Error storing backtest result: {str(e)}")
            return False
    
    def get_pair_performance(self, symbol, strategy):
        """
        Get performance statistics for a specific symbol-strategy pair.
        
        Args:
            symbol: Stock symbol
            strategy: Strategy ID
        
        Returns:
            Dictionary with performance statistics or None if not found
        """
        key = f"{symbol}_{strategy}"
        
        with self._lock:
            return self.performance_stats.get(key)
    
    def get_top_performing_pairs(self, limit=10):
        """
        Get the top performing symbol-strategy pairs.
        
        Args:
            limit: Maximum number of pairs to return
        
        Returns:
            List of top performing pairs
        """
        with self._lock:
            # Convert performance stats to list
            pairs = []
            
            for key, stats in self.performance_stats.items():
                symbol, strategy = key.split('_', 1)
                
                pairs.append({
                    "symbol": symbol,
                    "strategy": strategy,
                    "sharpe": stats.get("sharpe", 0),
                    "return": stats.get("return", 0),
                    "win_rate": stats.get("win_rate", 0),
                    "max_drawdown": stats.get("max_drawdown", 0),
                    "backtest_count": stats.get("backtest_count", 0)
                })
            
            # Sort by Sharpe ratio (descending)
            pairs.sort(key=lambda x: x["sharpe"], reverse=True)
            
            # Limit results
            return pairs[:limit]
    
    def get_adjustment_factor(self, symbol, strategy):
        """
        Get the adjustment factor for a specific symbol-strategy pair.
        This factor is used to adjust the score in the symbol ranker.
        
        Args:
            symbol: Stock symbol
            strategy: Strategy ID
        
        Returns:
            Adjustment factor (between 0.5 and 2.0)
        """
        key = f"{symbol}_{strategy}"
        
        with self._lock:
            return self.pair_adjustments.get(key, 1.0)
    
    def _load_performance_stats(self):
        """
        Load performance stats from disk.
        
        Returns:
            Dictionary with performance stats
        """
        try:
            filepath = os.path.join(self.results_db_path, "performance_stats.json")
            
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
            
            return {}
        
        except Exception as e:
            self.logger.error(f"Error loading performance stats: {str(e)}")
            return {}
    
    def _save_performance_stats(self):
        """
        Save performance stats to disk.
        """
        try:
            filepath = os.path.join(self.results_db_path, "performance_stats.json")
            
            with open(filepath, 'w') as f:
                json.dump(self.performance_stats, f, indent=2)
            
            self.logger.info(f"Performance stats saved to {filepath}")
        
        except Exception as e:
            self.logger.error(f"Error saving performance stats: {str(e)}")
    
    def _update_performance_stats(self, symbol, strategy, backtest_results):
        """
        Update performance stats with new backtest results.
        
        Args:
            symbol: Stock symbol
            strategy: Strategy ID
            backtest_results: Dictionary containing backtest results
        """
        key = f"{symbol}_{strategy}"
        
        # Extract metrics from backtest results
        sharpe = backtest_results.get("sharpe_ratio", 0)
        total_return = backtest_results.get("total_return", 0)
        win_rate = backtest_results.get("win_rate", 0)
        max_drawdown = backtest_results.get("max_drawdown", 0)
        
        # Get existing stats or initialize
        stats = self.performance_stats.get(key, {
            "sharpe": 0,
            "return": 0,
            "win_rate": 0,
            "max_drawdown": 0,
            "backtest_count": 0,
            "last_updated": None
        })
        
        # Update with exponential moving average (more weight to recent results)
        count = stats["backtest_count"]
        alpha = 0.3  # Weighting factor for new results
        
        if count == 0:
            # First result, just use the values
            stats["sharpe"] = sharpe
            stats["return"] = total_return
            stats["win_rate"] = win_rate
            stats["max_drawdown"] = max_drawdown
        else:
            # Update with exponential weighting
            stats["sharpe"] = stats["sharpe"] * (1 - alpha) + sharpe * alpha
            stats["return"] = stats["return"] * (1 - alpha) + total_return * alpha
            stats["win_rate"] = stats["win_rate"] * (1 - alpha) + win_rate * alpha
            stats["max_drawdown"] = stats["max_drawdown"] * (1 - alpha) + max_drawdown * alpha
        
        # Update count and timestamp
        stats["backtest_count"] = count + 1
        stats["last_updated"] = datetime.datetime.now().isoformat()
        
        # Store updated stats
        self.performance_stats[key] = stats
        
        # Save to disk
        self._save_performance_stats()
    
    def _update_pair_adjustments(self, symbol, strategy, backtest_results):
        """
        Update pair ranking adjustments based on backtest results.
        
        Args:
            symbol: Stock symbol
            strategy: Strategy ID
            backtest_results: Dictionary containing backtest results
        """
        key = f"{symbol}_{strategy}"
        
        # Extract key metrics
        sharpe = backtest_results.get("sharpe_ratio", 0)
        total_return = backtest_results.get("total_return", 0)
        
        # Calculate adjustment factor
        # Sharpe > 1.5 is very good, < 0.5 is poor
        sharpe_factor = min(max(sharpe / 1.0, 0.5), 2.0)
        
        # Return > 10% is good, < -5% is poor
        return_factor = min(max((total_return + 5) / 10, 0.5), 2.0)
        
        # Combine factors
        adjustment = (sharpe_factor * 0.7) + (return_factor * 0.3)
        
        # Store the adjustment
        self.pair_adjustments[key] = adjustment
        
        self.logger.info(f"Updated adjustment factor for {key}: {adjustment:.2f}")
    
    def _apply_to_ml_system(self, symbol, strategy, backtest_results):
        """
        Apply the backtest results to the ML system for continuous learning.
        
        Args:
            symbol: Stock symbol
            strategy: Strategy ID
            backtest_results: Dictionary containing backtest results
        """
        try:
            # Get adaptive trainer
            trainer = get_adaptive_trainer()
            
            # Create a structured learning example
            example = {
                "symbol": symbol,
                "strategy": strategy,
                "timestamp": datetime.datetime.now().isoformat(),
                "backtest_results": {
                    "sharpe_ratio": backtest_results.get("sharpe_ratio", 0),
                    "total_return": backtest_results.get("total_return", 0),
                    "win_rate": backtest_results.get("win_rate", 0),
                    "max_drawdown": backtest_results.get("max_drawdown", 0)
                },
                "market_regime": get_market_context().get_market_context().get("market", {}).get("regime", "unknown")
            }
            
            # In a real implementation, this would feed the example into the
            # ML system's training pipeline. For now, we'll just log it.
            self.logger.info(f"Applied backtest result to ML system: {example}")
            
            # Trigger training run if conditions are met
            # (e.g., significant result or enough new data)
            if abs(backtest_results.get("sharpe_ratio", 0)) > 1.5 or abs(backtest_results.get("total_return", 0)) > 15:
                self.logger.info("Significant backtest result detected, triggering ML training run")
                # In a real implementation, this would trigger an async training run
        
        except Exception as e:
            self.logger.error(f"Error applying to ML system: {str(e)}")


class BacktestExecutor:
    """
    Executes backtests for symbol-strategy pairs and feeds results
    into the feedback system.
    """
    
    def __init__(self, config=None):
        """
        Initialize the backtest executor.
        
        Args:
            config: Configuration dictionary or None to use defaults
        """
        self._config = config or {}
        
        # Set up logging
        self.logger = logging.getLogger("BacktestExecutor")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Get feedback system
        self.feedback_system = get_backtest_feedback_system()
        
        self.logger.info("BacktestExecutor initialized")
    
    def backtest_pair(self, symbol, strategy, start_date=None, end_date=None, params=None):
        """
        Backtest a symbol-strategy pair and store the results.
        
        Args:
            symbol: Stock symbol
            strategy: Strategy ID
            start_date: Start date for backtest or None for default
            end_date: End date for backtest or None for default
            params: Additional parameters for the backtest
        
        Returns:
            Dictionary with backtest results
        """
        self.logger.info(f"Backtesting {symbol} with {strategy}")
        
        try:
            # In a real implementation, this would call your existing backtest engine
            # For now, we'll simulate backtesting with random results
            
            # Default dates
            if start_date is None:
                start_date = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
            
            if end_date is None:
                end_date = datetime.datetime.now().strftime("%Y-%m-%d")
            
            # Log backtest configuration
            self.logger.info(f"Backtest config: {symbol}, {strategy}, {start_date} to {end_date}")
            
            # Simulate backtest duration
            time.sleep(0.5)
            
            # Generate plausible random results
            # In a real implementation, this would be the actual backtest results
            
            # Randomize with some strategy-specific biases
            base_sharpe = 0.8
            base_return = 5.0
            base_win_rate = 0.52
            base_max_dd = -8.0
            
            if strategy == "momentum_etf":
                # Higher returns but more drawdown
                base_sharpe += 0.3
                base_return += 3.0
                base_max_dd -= 2.0
            elif strategy == "value_dividend":
                # More consistent but lower returns
                base_sharpe += 0.1
                base_win_rate += 0.05
                base_max_dd += 3.0
            elif strategy == "mean_reversion":
                # Higher win rate but more volatility
                base_win_rate += 0.1
                base_sharpe -= 0.1
            
            # Add randomness
            results = {
                "sharpe_ratio": base_sharpe + np.random.normal(0, 0.3),
                "total_return": base_return + np.random.normal(0, 4.0),
                "win_rate": min(max(base_win_rate + np.random.normal(0, 0.05), 0.3), 0.8),
                "max_drawdown": base_max_dd + np.random.normal(0, 2.0),
                "trade_count": int(np.random.randint(20, 100)),
                "avg_trade_duration": float(np.random.randint(2, 10)),
                "start_date": start_date,
                "end_date": end_date,
                "params": params or {}
            }
            
            # Store in feedback system
            success = self.feedback_system.store_backtest_result(symbol, strategy, results)
            
            if success:
                self.logger.info(f"Backtest results stored for {symbol} with {strategy}")
            else:
                self.logger.warning(f"Failed to store backtest results for {symbol} with {strategy}")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error during backtesting: {str(e)}")
            return {"error": str(e)}
    
    def backtest_top_pairs(self, limit=5):
        """
        Backtest the top symbol-strategy pairs from the current market context.
        
        Args:
            limit: Maximum number of pairs to backtest
        
        Returns:
            List of backtest results
        """
        self.logger.info(f"Backtesting top {limit} pairs")
        
        try:
            # Get market intelligence controller
            from trading_bot.market_intelligence_controller import get_market_intelligence_controller
            controller = get_market_intelligence_controller()
            
            # Get top pairs
            top_pairs = controller.get_top_symbol_strategy_pairs(limit=limit)
            
            results = []
            
            for pair in top_pairs:
                symbol = pair.get("symbol")
                strategy = pair.get("strategy")
                
                # Backtest the pair
                backtest_result = self.backtest_pair(symbol, strategy)
                
                # Store result
                results.append({
                    "symbol": symbol,
                    "strategy": strategy,
                    "backtest_result": backtest_result
                })
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error backtesting top pairs: {str(e)}")
            return []


# Create singleton instances
_backtest_feedback_system = None
_backtest_executor = None

def get_backtest_feedback_system(config=None):
    """
    Get the singleton BacktestFeedbackSystem instance.
    
    Args:
        config: Optional configuration for the feedback system
    
    Returns:
        BacktestFeedbackSystem instance
    """
    global _backtest_feedback_system
    if _backtest_feedback_system is None:
        _backtest_feedback_system = BacktestFeedbackSystem(config)
    return _backtest_feedback_system

def get_backtest_executor(config=None):
    """
    Get the singleton BacktestExecutor instance.
    
    Args:
        config: Optional configuration for the executor
    
    Returns:
        BacktestExecutor instance
    """
    global _backtest_executor
    if _backtest_executor is None:
        _backtest_executor = BacktestExecutor(config)
    return _backtest_executor
