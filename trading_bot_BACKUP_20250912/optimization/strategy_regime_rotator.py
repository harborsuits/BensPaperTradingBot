#!/usr/bin/env python3
"""
Strategy Regime Rotator Module

This module provides a framework for rotating between different trading strategies
based on detected market regimes.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable, Optional, Union, Tuple
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import local modules
from trading_bot.optimization.advanced_market_regime_detector import AdvancedMarketRegimeDetector
from trading_bot.optimization.strategy_optimizer import StrategyOptimizer

# Configure logging
logger = logging.getLogger(__name__)

class StrategyRegimeRotator:
    """
    Rotates between different trading strategies based on detected market regimes.
    
    This class integrates the AdvancedMarketRegimeDetector with strategy optimization
    to select the best strategies for different market conditions.
    """
    
    def __init__(
        self,
        strategies: List[Any],
        initial_weights: Optional[Dict[str, float]] = None,
        regime_config: Optional[Dict[str, Any]] = None,
        lookback_window: int = 60,
        rebalance_frequency: str = "weekly",
        max_allocation_change: float = 0.2
    ):
        """
        Initialize the strategy regime rotator.
        
        Args:
            strategies: List of strategy instances or classes
            initial_weights: Optional initial strategy weights
            regime_config: Configuration for the market regime detector
            lookback_window: Number of days to look back for regime detection
            rebalance_frequency: How often to rebalance ('daily', 'weekly', 'monthly')
            max_allocation_change: Maximum allocation change per rebalance period
        """
        self.strategies = strategies
        self.strategy_names = self._get_strategy_names(strategies)
        
        # Initialize weights
        if initial_weights is None:
            # Equal weights by default
            self.weights = {name: 1.0 / len(strategies) for name in self.strategy_names}
        else:
            self.weights = initial_weights
        
        # Store configuration
        self.lookback_window = lookback_window
        self.rebalance_frequency = rebalance_frequency
        self.max_allocation_change = max_allocation_change
        
        # Initialize market regime detector
        self.regime_config = regime_config or {}
        self.market_regime_detector = AdvancedMarketRegimeDetector(self.regime_config)
        
        # Try to load pre-trained model
        model_path = self.regime_config.get("model_path", "models/market_regime_model.joblib")
        if os.path.exists(model_path):
            self.market_regime_detector.load_model(model_path)
        
        # Strategy performance by regime
        self.strategy_regime_performance = {}
        
        # Current market regime
        self.current_regime = "unknown"
        
        # Last rebalance date
        self.last_rebalance_date = None
        
        # History of weights and regimes
        self.weight_history = []
        self.regime_history = []
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    def _get_strategy_names(self, strategies):
        """Extract strategy names from strategy instances or classes."""
        names = []
        for strategy in strategies:
            if hasattr(strategy, 'name'):
                names.append(strategy.name)
            elif hasattr(strategy, '__name__'):
                names.append(strategy.__name__)
            else:
                names.append(str(strategy))
        return names
    
    def update_market_regime(self, market_data, date=None):
        """
        Update the current market regime based on market data.
        
        Args:
            market_data: DataFrame with OHLCV market data
            date: Optional specific date to detect regime for
            
        Returns:
            str: Detected market regime
        """
        # Detect regime
        regime = self.market_regime_detector.detect_regime(market_data, date)
        
        # Update current regime
        self.current_regime = regime
        
        # Add to history
        if date is None and not market_data.empty:
            date = market_data.index[-1]
            
        self.regime_history.append({
            'date': date,
            'regime': regime
        })
        
        self.logger.info(f"Market regime updated: {regime}")
        
        return regime
    
    def optimize_strategy_weights(self, market_data, strategy_returns):
        """
        Optimize strategy weights based on current market regime.
        
        Args:
            market_data: DataFrame with OHLCV market data
            strategy_returns: DataFrame with strategy returns
            
        Returns:
            Dict[str, float]: Optimized strategy weights
        """
        # Check if we have enough data
        if len(strategy_returns) < self.lookback_window:
            self.logger.warning(f"Not enough return data for optimization: {len(strategy_returns)} < {self.lookback_window}")
            return self.weights
        
        # Ensure we have detected the current regime
        if self.current_regime == "unknown":
            self.update_market_regime(market_data)
        
        # Get regime-specific weights
        regime_weights = self.get_regime_specific_weights()
        
        # Apply maximum change constraint
        constrained_weights = self.apply_weight_constraints(regime_weights)
        
        # Update weights
        self.weights = constrained_weights
        
        # Add to history
        current_date = market_data.index[-1] if not market_data.empty else datetime.now()
        self.weight_history.append({
            'date': current_date,
            'weights': self.weights.copy(),
            'regime': self.current_regime
        })
        
        # Update last rebalance date
        self.last_rebalance_date = current_date
        
        return self.weights
    
    def get_regime_specific_weights(self, regime=None):
        """
        Get optimal strategy weights for a specific market regime.
        
        Args:
            regime: Optional regime to get weights for (default: current regime)
            
        Returns:
            Dict[str, float]: Strategy weights for the regime
        """
        if regime is None:
            regime = self.current_regime
        
        # If we have pre-computed performance for this regime, use it
        if regime in self.strategy_regime_performance:
            return self._calculate_weights_from_performance(self.strategy_regime_performance[regime])
        
        # Otherwise, use default regime allocations
        return self._get_default_regime_weights(regime)
    
    def _calculate_weights_from_performance(self, performance_data):
        """Calculate weights based on strategy performance metrics."""
        # Extract key metrics for weighting
        performance = {}
        for strategy, metrics in performance_data.items():
            if strategy not in self.strategy_names:
                continue
                
            # Use Sharpe ratio as primary metric
            sharpe = metrics.get('sharpe', 0)
            
            # Assign negative score for strategies with negative Sharpe
            performance[strategy] = max(0.0001, sharpe)
        
        # Calculate weights proportional to performance
        total = sum(performance.values())
        if total > 0:
            weights = {strategy: perf / total for strategy, perf in performance.items()}
        else:
            # Equal weights if all perform poorly
            weights = {strategy: 1.0 / len(self.strategy_names) for strategy in self.strategy_names}
        
        return weights
    
    def _get_default_regime_weights(self, regime):
        """Get default weights for different regimes based on strategy characteristics."""
        # Define regime-specific weights based on strategy types
        regime_defaults = {
            "bullish": {
                "trend_following": 0.6,
                "momentum": 0.3,
                "mean_reversion": 0.0,
                "volatility": 0.1
            },
            "bearish": {
                "trend_following": 0.4,
                "momentum": 0.0,
                "mean_reversion": 0.3,
                "volatility": 0.3
            },
            "volatile_bullish": {
                "trend_following": 0.3,
                "momentum": 0.4,
                "mean_reversion": 0.0,
                "volatility": 0.3
            },
            "volatile_bearish": {
                "trend_following": 0.3,
                "momentum": 0.0, 
                "mean_reversion": 0.3,
                "volatility": 0.4
            },
            "sideways": {
                "trend_following": 0.0,
                "momentum": 0.2,
                "mean_reversion": 0.6,
                "volatility": 0.2
            },
            "low_volatility": {
                "trend_following": 0.0,
                "momentum": 0.3,
                "mean_reversion": 0.7,
                "volatility": 0.0
            },
            "high_volatility": {
                "trend_following": 0.0,
                "momentum": 0.0,
                "mean_reversion": 0.4,
                "volatility": 0.6
            }
        }
        
        # Default to equal weights if regime not recognized
        if regime not in regime_defaults:
            return {name: 1.0 / len(self.strategy_names) for name in self.strategy_names}
        
        # Get strategy types (implement this based on your strategy classes)
        strategy_types = self._get_strategy_types()
        
        # Calculate weights based on strategy types and regime defaults
        weights = {}
        type_counts = {}
        
        # Count strategies by type
        for name, type_name in strategy_types.items():
            if type_name not in type_counts:
                type_counts[type_name] = 0
            type_counts[type_name] += 1
        
        # Distribute weights by type
        for name, type_name in strategy_types.items():
            if type_name in regime_defaults[regime]:
                type_weight = regime_defaults[regime][type_name]
                # Distribute type weight equally among strategies of this type
                weights[name] = type_weight / type_counts[type_name] if type_counts[type_name] > 0 else 0
            else:
                weights[name] = 0
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {name: weight / total for name, weight in weights.items()}
        else:
            # Equal weights if no strategy types match
            weights = {name: 1.0 / len(self.strategy_names) for name in self.strategy_names}
        
        return weights
    
    def _get_strategy_types(self):
        """
        Determine strategy types based on strategy attributes or class names.
        Override this method to provide custom strategy type detection.
        """
        strategy_types = {}
        
        for i, strategy in enumerate(self.strategies):
            name = self.strategy_names[i]
            
            # Try to get type from strategy attribute
            if hasattr(strategy, 'strategy_type'):
                strategy_types[name] = strategy.strategy_type.lower()
            else:
                # Infer type from name
                name_lower = name.lower()
                
                if any(keyword in name_lower for keyword in ['trend', 'follow']):
                    strategy_types[name] = 'trend_following'
                elif any(keyword in name_lower for keyword in ['momentum', 'rsi']):
                    strategy_types[name] = 'momentum'
                elif any(keyword in name_lower for keyword in ['reversion', 'mean', 'contrarian']):
                    strategy_types[name] = 'mean_reversion'
                elif any(keyword in name_lower for keyword in ['vol', 'volatility']):
                    strategy_types[name] = 'volatility'
                else:
                    # Default type
                    strategy_types[name] = 'other'
        
        return strategy_types
    
    def apply_weight_constraints(self, target_weights):
        """
        Apply constraints to weight changes to avoid dramatic shifts.
        
        Args:
            target_weights: Target strategy weights
            
        Returns:
            Dict[str, float]: Constrained strategy weights
        """
        # Start with current weights
        new_weights = self.weights.copy()
        
        # Apply maximum change constraint
        for name, target_weight in target_weights.items():
            if name in new_weights:
                current_weight = new_weights[name]
                max_change = self.max_allocation_change
                
                # Limit change to maximum allowed
                weight_change = target_weight - current_weight
                if abs(weight_change) > max_change:
                    # Limit the change
                    weight_change = max_change if weight_change > 0 else -max_change
                
                new_weights[name] = current_weight + weight_change
        
        # Normalize weights to sum to 1
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {name: weight / total for name, weight in new_weights.items()}
        
        return new_weights
    
    def should_rebalance(self, current_date):
        """
        Check if we should rebalance based on rebalance frequency.
        
        Args:
            current_date: Current date to check against last rebalance
            
        Returns:
            bool: Whether to rebalance
        """
        if self.last_rebalance_date is None:
            return True
        
        # Convert to datetime if string
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        
        # Calculate time since last rebalance
        time_diff = current_date - self.last_rebalance_date
        
        # Check against rebalance frequency
        if self.rebalance_frequency == 'daily':
            return time_diff.days >= 1
        elif self.rebalance_frequency == 'weekly':
            return time_diff.days >= 7
        elif self.rebalance_frequency == 'monthly':
            return time_diff.days >= 30
        else:
            # Default to daily
            return time_diff.days >= 1
    
    def analyze_strategy_regime_performance(self, market_data, strategy_returns):
        """
        Analyze how each strategy performs in different market regimes.
        
        Args:
            market_data: DataFrame with OHLCV market data
            strategy_returns: DataFrame with strategy returns
            
        Returns:
            Dict: Strategy performance by regime
        """
        # Detect historical regimes if not already done
        if not hasattr(self.market_regime_detector, 'historical_regimes') or self.market_regime_detector.historical_regimes is None:
            regimes = self.market_regime_detector.detect_regime_history(market_data)
        else:
            regimes = self.market_regime_detector.historical_regimes
        
        # Combine regimes and strategy returns
        combined = pd.DataFrame(regimes).join(strategy_returns)
        combined = combined.dropna()
        
        # Calculate performance metrics by regime for each strategy
        performance_by_regime = {}
        
        # Get unique regimes
        unique_regimes = combined['regime'].unique()
        
        for regime in unique_regimes:
            # Filter data for this regime
            regime_data = combined[combined['regime'] == regime]
            
            # Skip if not enough data
            if len(regime_data) < 20:
                continue
            
            # Calculate metrics for each strategy
            regime_performance = {}
            
            for strategy in strategy_returns.columns:
                returns = regime_data[strategy]
                
                # Calculate performance metrics
                metrics = {
                    'count': len(returns),
                    'mean_return': returns.mean(),
                    'volatility': returns.std(),
                    'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0,
                    'win_rate': (returns > 0).mean(),
                    'max_drawdown': (returns.cumsum() - returns.cumsum().cummax()).min()
                }
                
                regime_performance[strategy] = metrics
            
            performance_by_regime[regime] = regime_performance
        
        # Store performance data
        self.strategy_regime_performance = performance_by_regime
        
        return performance_by_regime
    
    def get_allocation_for_date(self, date, market_data, strategy_returns):
        """
        Get strategy allocation for a specific date.
        
        Args:
            date: Date to get allocation for
            market_data: DataFrame with OHLCV market data
            strategy_returns: DataFrame with strategy returns
            
        Returns:
            Dict[str, float]: Strategy weights for the date
        """
        # Convert to datetime if string
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # Filter data up to the date
        market_data_filtered = market_data.loc[:date]
        strategy_returns_filtered = strategy_returns.loc[:date]
        
        # Update regime
        self.update_market_regime(market_data_filtered, date)
        
        # Check if we should rebalance
        if self.should_rebalance(date):
            # Optimize weights
            return self.optimize_strategy_weights(market_data_filtered, strategy_returns_filtered)
        else:
            # Return current weights
            return self.weights
    
    def backtest_regime_rotation(self, market_data, strategy_returns):
        """
        Backtest the regime rotation strategy.
        
        Args:
            market_data: DataFrame with OHLCV market data
            strategy_returns: DataFrame with strategy returns
            
        Returns:
            Dict: Backtest results
        """
        # Analyze strategy performance by regime
        self.analyze_strategy_regime_performance(market_data, strategy_returns)
        
        # Initialize results
        results = {
            'dates': [],
            'regimes': [],
            'weights': [],
            'portfolio_returns': []
        }
        
        # Reset weight history
        self.weight_history = []
        self.regime_history = []
        
        # Get common dates
        dates = strategy_returns.index.intersection(market_data.index)
        dates = sorted(dates)
        
        # Initialize weights
        current_weights = {col: 1.0 / len(strategy_returns.columns) for col in strategy_returns.columns}
        self.weights = current_weights
        self.last_rebalance_date = None
        
        # Run backtest
        for i, date in enumerate(dates):
            # Skip first day (need returns)
            if i == 0:
                continue
            
            # Get allocation for this date
            allocation = self.get_allocation_for_date(date, market_data.loc[:date], strategy_returns.loc[:date])
            
            # Calculate portfolio return
            daily_return = 0
            for strategy, weight in allocation.items():
                if strategy in strategy_returns.columns:
                    daily_return += weight * strategy_returns.loc[date, strategy]
            
            # Store results
            results['dates'].append(date)
            results['regimes'].append(self.current_regime)
            results['weights'].append(allocation.copy())
            results['portfolio_returns'].append(daily_return)
        
        # Calculate performance metrics
        returns_series = pd.Series(results['portfolio_returns'], index=results['dates'])
        
        metrics = {
            'total_return': (1 + returns_series).prod() - 1,
            'cagr': (1 + returns_series).prod() ** (252 / len(returns_series)) - 1,
            'volatility': returns_series.std() * np.sqrt(252),
            'sharpe_ratio': returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0,
            'max_drawdown': (returns_series.cumsum() - returns_series.cumsum().cummax()).min(),
            'win_rate': (returns_series > 0).mean(),
            'regime_counts': pd.Series(results['regimes']).value_counts().to_dict()
        }
        
        results['metrics'] = metrics
        results['returns_series'] = returns_series
        
        return results
    
    def visualize_regime_rotation(self, backtest_results, save_path=None):
        """
        Visualize the backtest results of regime rotation.
        
        Args:
            backtest_results: Results from backtest_regime_rotation
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib.Figure: Plot figure
        """
        # Extract data
        dates = backtest_results['dates']
        regimes = backtest_results['regimes']
        returns_series = backtest_results['returns_series']
        weights = backtest_results['weights']
        
        # Convert weights to DataFrame
        weights_df = pd.DataFrame(weights, index=dates)
        
        # Create cumulative returns
        cum_returns = (1 + returns_series).cumprod() - 1
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [2, 1, 2]})
        
        # 1. Plot cumulative returns
        axes[0].plot(cum_returns.index, cum_returns * 100, 'b-', linewidth=2)
        axes[0].set_title('Cumulative Returns (%)')
        axes[0].set_ylabel('Return (%)')
        axes[0].grid(True)
        
        # 2. Plot regime changes
        axes[1].plot(dates, [0] * len(dates), 'k-', alpha=0)  # Invisible line for scaling
        
        # Color background by regime
        regime_colors = {
            "bullish": "green",
            "bearish": "red",
            "volatile_bullish": "lightgreen",
            "volatile_bearish": "salmon",
            "sideways": "gray",
            "low_volatility": "lightblue",
            "high_volatility": "orange",
            "transition": "purple",
            "unknown": "white"
        }
        
        # Plot regime backgrounds
        unique_regimes = list(set(regimes))
        
        for regime in unique_regimes:
            # Create mask for this regime
            mask = [r == regime for r in regimes]
            regime_dates = [date for date, is_regime in zip(dates, mask) if is_regime]
            
            # Find contiguous blocks of this regime
            blocks = []
            block_start = None
            
            for date, is_regime in zip(dates, mask):
                if is_regime and block_start is None:
                    block_start = date
                elif not is_regime and block_start is not None:
                    blocks.append((block_start, date))
                    block_start = None
            
            # Add the last block if it hasn't been closed
            if block_start is not None:
                blocks.append((block_start, dates[-1]))
            
            # Plot each block
            for start, end in blocks:
                axes[1].axvspan(start, end, alpha=0.3, color=regime_colors.get(regime, "gray"))
        
        # Add legend for regimes
        import matplotlib.patches as mpatches
        
        patches = []
        for regime in unique_regimes:
            if regime in regime_colors:
                patch = mpatches.Patch(color=regime_colors[regime], alpha=0.3, label=regime.replace("_", " ").title())
                patches.append(patch)
        
        axes[1].legend(handles=patches, loc='upper right')
        axes[1].set_title('Market Regimes')
        axes[1].set_ylabel('Regime')
        axes[1].set_yticks([])
        
        # 3. Plot strategy weights
        weights_df.plot(ax=axes[2], linewidth=2)
        axes[2].set_title('Strategy Weights')
        axes[2].set_ylabel('Weight')
        axes[2].set_xlabel('Date')
        axes[2].grid(True)
        axes[2].legend(loc='upper right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_model(self, path=None):
        """
        Save the regime rotator model and weights.
        
        Args:
            path: Optional path to save the model
            
        Returns:
            str: Path to the saved model
        """
        if path is None:
            path = "models/strategy_regime_rotator.joblib"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save models
        model_data = {
            "weights": self.weights,
            "strategy_names": self.strategy_names,
            "strategy_regime_performance": self.strategy_regime_performance,
            "current_regime": self.current_regime,
            "weight_history": self.weight_history,
            "regime_history": self.regime_history,
            "last_rebalance_date": self.last_rebalance_date,
            "config": {
                "lookback_window": self.lookback_window,
                "rebalance_frequency": self.rebalance_frequency,
                "max_allocation_change": self.max_allocation_change
            }
        }
        
        # Save the market regime detector separately
        if hasattr(self, 'market_regime_detector'):
            self.market_regime_detector.save_model()
        
        import joblib
        joblib.dump(model_data, path)
        self.logger.info(f"Strategy regime rotator saved to {path}")
        
        return path
    
    def load_model(self, path=None):
        """
        Load the regime rotator model and weights.
        
        Args:
            path: Optional path to load the model from
            
        Returns:
            bool: Whether model was loaded successfully
        """
        if path is None:
            path = "models/strategy_regime_rotator.joblib"
        
        try:
            import joblib
            model_data = joblib.load(path)
            
            # Load weights and configuration
            self.weights = model_data["weights"]
            self.strategy_regime_performance = model_data["strategy_regime_performance"]
            self.current_regime = model_data["current_regime"]
            self.weight_history = model_data["weight_history"]
            self.regime_history = model_data["regime_history"]
            self.last_rebalance_date = model_data["last_rebalance_date"]
            
            # Update config
            if "config" in model_data:
                self.lookback_window = model_data["config"].get("lookback_window", self.lookback_window)
                self.rebalance_frequency = model_data["config"].get("rebalance_frequency", self.rebalance_frequency)
                self.max_allocation_change = model_data["config"].get("max_allocation_change", self.max_allocation_change)
            
            # Load the market regime detector separately
            if hasattr(self, 'market_regime_detector'):
                self.market_regime_detector.load_model()
            
            self.logger.info(f"Strategy regime rotator loaded from {path}")
            return True
        except (FileNotFoundError, KeyError) as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create some dummy strategies
    class DummyStrategy:
        def __init__(self, name, strategy_type):
            self.name = name
            self.strategy_type = strategy_type
    
    # Create strategy instances
    strategies = [
        DummyStrategy("TrendFollower", "trend_following"),
        DummyStrategy("MeanReversion", "mean_reversion"),
        DummyStrategy("MomentumStrategy", "momentum"),
        DummyStrategy("VolatilityStrategy", "volatility")
    ]
    
    # Create rotator
    rotator = StrategyRegimeRotator(
        strategies=strategies,
        lookback_window=30,
        rebalance_frequency='weekly'
    )
    
    # Create dummy market data
    dates = pd.date_range(start='2020-01-01', end='2022-01-01', freq='B')
    market_data = pd.DataFrame({
        'open': np.random.normal(100, 1, len(dates)),
        'high': np.random.normal(101, 1, len(dates)),
        'low': np.random.normal(99, 1, len(dates)),
        'close': np.random.normal(100, 1, len(dates)),
        'volume': np.random.normal(1000000, 100000, len(dates))
    }, index=dates)
    
    # Create dummy strategy returns
    strategy_returns = pd.DataFrame({
        'TrendFollower': np.random.normal(0.001, 0.01, len(dates)),
        'MeanReversion': np.random.normal(0.001, 0.012, len(dates)),
        'MomentumStrategy': np.random.normal(0.001, 0.015, len(dates)),
        'VolatilityStrategy': np.random.normal(0.001, 0.02, len(dates))
    }, index=dates)
    
    # Backtest regime rotation
    results = rotator.backtest_regime_rotation(market_data, strategy_returns)
    
    # Visualize results
    rotator.visualize_regime_rotation(results)
    
    # Print performance metrics
    print("Performance Metrics:")
    for metric, value in results['metrics'].items():
        if not isinstance(value, dict):
            print(f"{metric}: {value:.4f}")
    
    print("\nRegime Counts:")
    for regime, count in results['metrics']['regime_counts'].items():
        print(f"{regime}: {count}")
    
    # Save the model
    rotator.save_model() 