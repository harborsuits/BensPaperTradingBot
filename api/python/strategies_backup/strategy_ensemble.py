import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from collections import defaultdict
from enum import Enum

from .base_strategy import Strategy, SignalType, Position

logger = logging.getLogger(__name__)

class EnsembleMethod(Enum):
    """Methods for combining signals from multiple strategies"""
    MAJORITY_VOTE = "majority_vote"  # Simple majority vote (democratic)
    WEIGHTED_VOTE = "weighted_vote"  # Weighted voting based on strategy weights
    PERFORMANCE_WEIGHTED = "performance_weighted"  # Dynamic weights based on recent performance
    CONFIDENCE_WEIGHTED = "confidence_weighted"  # Weight by signal confidence
    SEQUENTIAL_FILTER = "sequential_filter"  # Sequential filtering of signals
    OPTIMAL_F = "optimal_f"  # Kelly criterion / optimal f allocation


class SignalCombination(Enum):
    """Methods for combining continuous signals (for position sizing)"""
    MEAN = "mean"  # Simple average of signals
    WEIGHTED_MEAN = "weighted_mean"  # Weighted average based on weights
    MEDIAN = "median"  # Median of signals (robust to outliers)
    MIN = "min"  # Minimum signal value (conservative)
    MAX = "max"  # Maximum signal value (aggressive)
    CONVEX_COMBINATION = "convex_combination"  # Dynamic convex combination


class WeightingMethod(Enum):
    """Method used to weight signals from different strategies"""
    EQUAL = "equal"                  # Equal weighting for all strategies
    CUSTOM = "custom"                # Custom fixed weights
    PERFORMANCE = "performance"      # Weight by historical performance
    REGIME_BASED = "regime_based"    # Weight by regime-specific performance
    VOLATILITY = "volatility"        # Inverse volatility weighting
    ADAPTIVE = "adaptive"            # Adaptive weights based on recent performance


class StrategyEnsemble(Strategy):
    """
    A composite strategy that combines signals from multiple strategies
    
    This class allows multiple trading strategies to be combined into a single
    strategy, with various methods for weighting and aggregating their signals.
    """
    
    def __init__(
        self,
        strategies: List[Strategy],
        weighting_method: WeightingMethod = WeightingMethod.EQUAL,
        strategy_weights: Optional[Dict[str, float]] = None,
        performance_window: int = 60,
        rebalance_frequency: int = 20,
        correlation_threshold: float = 0.7,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        strategy_names: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        name: str = "StrategyEnsemble",
    ):
        """
        Initialize the strategy ensemble
        
        Args:
            strategies: List of strategy instances to combine
            weighting_method: Method for weighting strategy signals
            strategy_weights: Optional dictionary of custom weights by strategy name
            performance_window: Window for calculating performance-based weights
            rebalance_frequency: How often to recalculate adaptive weights (bars)
            correlation_threshold: Correlation threshold for correlation adjustment
            min_weight: Minimum weight for any strategy
            max_weight: Maximum weight for any strategy
            strategy_names: Optional custom names for strategies
            symbols: Trading symbols (will use union of strategies if None)
            name: Name of this ensemble strategy
        """
        # Get unique symbols from all strategies if not specified
        if symbols is None:
            all_symbols = set()
            for strategy in strategies:
                if hasattr(strategy, 'symbols'):
                    all_symbols.update(strategy.symbols)
            symbols = list(all_symbols)
        
        super().__init__(symbols=symbols)
        
        self.strategies = strategies
        self.weighting_method = weighting_method
        self.performance_window = performance_window
        self.rebalance_frequency = rebalance_frequency
        self.correlation_threshold = correlation_threshold
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.name = name
        
        # Set strategy names
        if strategy_names and len(strategy_names) == len(strategies):
            self.strategy_names = strategy_names
        else:
            self.strategy_names = [
                getattr(s, 'name', f"Strategy_{i}") 
                for i, s in enumerate(strategies)
            ]
        
        # Initialize strategy weights
        self.strategy_weights = {}
        self._initialize_weights(strategy_weights)
        
        # Initialize performance tracking
        self.strategy_performances = {name: [] for name in self.strategy_names}
        self.strategy_signals = {name: pd.DataFrame() for name in self.strategy_names}
        self.last_rebalance = 0
        self.bar_count = 0
    
    def _initialize_weights(self, custom_weights: Optional[Dict[str, float]] = None) -> None:
        """
        Initialize strategy weights based on the weighting method
        
        Args:
            custom_weights: Optional dictionary of custom weights
        """
        if self.weighting_method == WeightingMethod.CUSTOM and custom_weights:
            # Use provided custom weights
            self.strategy_weights = custom_weights.copy()
            
            # Normalize weights
            total_weight = sum(self.strategy_weights.values())
            if total_weight > 0:
                for name in self.strategy_weights:
                    self.strategy_weights[name] /= total_weight
        else:
            # Default to equal weighting
            equal_weight = 1.0 / len(self.strategies)
            self.strategy_weights = {name: equal_weight for name in self.strategy_names}
        
        # Apply min/max constraints
        for name in self.strategy_weights:
            self.strategy_weights[name] = min(max(self.strategy_weights[name], self.min_weight), self.max_weight)
        
        # Normalize again after applying constraints
        self._normalize_weights()
        
        logger.info(f"Initialized strategy weights: {self.strategy_weights}")
    
    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0"""
        total_weight = sum(self.strategy_weights.values())
        if total_weight > 0:
            for name in self.strategy_weights:
                self.strategy_weights[name] /= total_weight
    
    def generate_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals by combining signals from all strategies
        
        Args:
            market_data: Market data DataFrame
            
        Returns:
            DataFrame with aggregated trading signals
        """
        self.bar_count += 1
        
        # Check if it's time to rebalance weights
        if (self.weighting_method in [WeightingMethod.PERFORMANCE, WeightingMethod.ADAPTIVE, 
                                     WeightingMethod.VOLATILITY] and 
            self.bar_count - self.last_rebalance >= self.rebalance_frequency):
            self._update_weights(market_data)
            self.last_rebalance = self.bar_count
        
        # Generate signals from each strategy
        all_signals = {}
        for i, strategy in enumerate(self.strategies):
            name = self.strategy_names[i]
            signals = strategy.generate_signals(market_data)
            
            # Store signals for performance tracking
            self.strategy_signals[name] = signals
            
            # Convert to dictionary for easier aggregation
            if not signals.empty:
                signals_dict = {}
                for symbol in self.symbols:
                    if symbol in signals.columns:
                        signals_dict[symbol] = signals.iloc[-1][symbol]
                    else:
                        signals_dict[symbol] = 0.0
                all_signals[name] = signals_dict
        
        # Combine signals using weighted average
        combined_signals = self._combine_signals(all_signals)
        
        # Convert combined signals to DataFrame format
        result = pd.DataFrame([combined_signals], columns=self.symbols)
        
        # Add timestamp if the input data has an index
        if not market_data.empty and isinstance(market_data.index, pd.DatetimeIndex):
            result.index = [market_data.index[-1]]
        
        return result
    
    def _combine_signals(self, all_signals: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Combine signals from all strategies using weighted average
        
        Args:
            all_signals: Dictionary of signals from each strategy
            
        Returns:
            Dictionary of combined signals for each symbol
        """
        combined = {symbol: 0.0 for symbol in self.symbols}
        
        for strategy_name, signals in all_signals.items():
            weight = self.strategy_weights.get(strategy_name, 0.0)
            for symbol, signal in signals.items():
                if symbol in combined:
                    combined[symbol] += signal * weight
        
        return combined
    
    def _update_weights(self, market_data: pd.DataFrame) -> None:
        """
        Update strategy weights based on the selected weighting method
        
        Args:
            market_data: Current market data
        """
        if self.weighting_method == WeightingMethod.PERFORMANCE:
            self._update_performance_weights(market_data)
        elif self.weighting_method == WeightingMethod.VOLATILITY:
            self._update_volatility_weights()
        elif self.weighting_method == WeightingMethod.ADAPTIVE:
            self._update_adaptive_weights(market_data)
        elif self.weighting_method == WeightingMethod.REGIME_BASED:
            self._update_regime_based_weights(market_data)
        
        # After updating, normalize weights
        self._normalize_weights()
        
        logger.info(f"Updated strategy weights: {self.strategy_weights}")
    
    def _update_performance_weights(self, market_data: pd.DataFrame) -> None:
        """
        Update weights based on historical performance of each strategy
        
        Args:
            market_data: Current market data
        """
        # We need enough history to calculate performance
        if self.bar_count < self.performance_window:
            return
        
        # Calculate performance for each strategy
        performances = {}
        
        for name in self.strategy_names:
            # Get signals history for this strategy
            signals = self.strategy_signals.get(name)
            if signals is None or signals.empty:
                performances[name] = 0.0
                continue
            
            # Calculate return series based on signals and market data
            returns = self._calculate_strategy_returns(signals, market_data, self.performance_window)
            
            # Calculate Sharpe ratio or other performance metric
            if len(returns) > 0:
                # Simple Sharpe ratio (annualized)
                sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
                sharpe = sharpe * np.sqrt(252)  # Annualize
                performances[name] = max(0.0, sharpe)  # Only positive performance impacts weights
            else:
                performances[name] = 0.0
        
        # Convert performances to weights
        total_performance = sum(performances.values())
        if total_performance > 0:
            for name in self.strategy_names:
                new_weight = performances[name] / total_performance
                
                # Smooth transition
                current_weight = self.strategy_weights.get(name, 0.0)
                self.strategy_weights[name] = current_weight * 0.7 + new_weight * 0.3
        
        # Apply min/max constraints
        for name in self.strategy_weights:
            self.strategy_weights[name] = min(max(self.strategy_weights[name], self.min_weight), self.max_weight)
    
    def _update_volatility_weights(self) -> None:
        """
        Update weights based on inverse volatility of strategy returns
        """
        volatilities = {}
        
        for name in self.strategy_names:
            # Get performance history
            returns = self.strategy_performances.get(name, [])
            
            if len(returns) >= 5:  # Need at least some history
                # Calculate volatility
                volatility = np.std(returns)
                if volatility > 0:
                    # Inverse volatility weight
                    volatilities[name] = 1.0 / volatility
                else:
                    volatilities[name] = 1.0
            else:
                volatilities[name] = 1.0
        
        # Convert inverse volatilities to weights
        total_inverse_vol = sum(volatilities.values())
        if total_inverse_vol > 0:
            for name in self.strategy_names:
                new_weight = volatilities[name] / total_inverse_vol
                
                # Smooth transition
                current_weight = self.strategy_weights.get(name, 0.0)
                self.strategy_weights[name] = current_weight * 0.7 + new_weight * 0.3
        
        # Apply min/max constraints
        for name in self.strategy_weights:
            self.strategy_weights[name] = min(max(self.strategy_weights[name], self.min_weight), self.max_weight)
    
    def _update_adaptive_weights(self, market_data: pd.DataFrame) -> None:
        """
        Update weights adaptively based on recent performance
        
        Args:
            market_data: Current market data
        """
        # Start with performance-based weights
        self._update_performance_weights(market_data)
        
        # Then adjust for correlation
        self._adjust_for_correlation()
    
    def _update_regime_based_weights(self, market_data: pd.DataFrame) -> None:
        """
        Update weights based on regime-specific performance
        
        Args:
            market_data: Current market data
        """
        # This requires a regime detector
        # For simplicity we'll use a basic volatility-based regime detection
        
        # Calculate recent volatility
        returns = market_data.pct_change().dropna()
        if len(returns) > 20:
            recent_vol = returns.std().mean() * np.sqrt(252)  # Annualized
            
            # Determine regime based on volatility
            high_vol_regime = recent_vol > 0.2  # 20% annualized vol threshold
            
            # Adjust weights based on regime
            if high_vol_regime:
                # In high volatility, favor strategies with better downside protection
                # This is a simplified example - a real implementation would be more complex
                self._favor_low_volatility_strategies()
            else:
                # In normal conditions, use standard performance weights
                self._update_performance_weights(market_data)
        else:
            # Not enough data, use default performance weights
            self._update_performance_weights(market_data)
    
    def _favor_low_volatility_strategies(self) -> None:
        """Adjust weights to favor strategies with lower volatility"""
        # Calculate downside deviation for each strategy
        downside_risks = {}
        
        for name in self.strategy_names:
            returns = self.strategy_performances.get(name, [])
            
            if len(returns) > 10:
                # Calculate downside deviation (only negative returns)
                negative_returns = [r for r in returns if r < 0]
                if negative_returns:
                    downside_risk = np.std(negative_returns)
                    downside_risks[name] = downside_risk
                else:
                    downside_risks[name] = 0.001  # Small positive value
            else:
                downside_risks[name] = 0.1  # Default
        
        # Convert to inverse downside risk weights
        if downside_risks:
            for name in self.strategy_names:
                risk = downside_risks.get(name, 0.1)
                if risk > 0:
                    # Inverse risk (lower risk = higher weight)
                    self.strategy_weights[name] = 1.0 / risk
                else:
                    self.strategy_weights[name] = 10.0  # High weight for no downside
            
            # Normalize
            total_weight = sum(self.strategy_weights.values())
            if total_weight > 0:
                for name in self.strategy_weights:
                    self.strategy_weights[name] /= total_weight
    
    def _adjust_for_correlation(self) -> None:
        """
        Adjust weights to account for correlation between strategies
        """
        # Need performance history for each strategy
        if not all(len(perf) >= 10 for perf in self.strategy_performances.values()):
            return
        
        # Create a returns DataFrame
        returns_data = {}
        for name, returns in self.strategy_performances.items():
            if len(returns) >= 10:
                returns_data[name] = returns[-10:]  # Last 10 returns
        
        if len(returns_data) < 2:
            return  # Need at least 2 strategies for correlation
        
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Adjust weights for highly correlated strategies
        for i, name1 in enumerate(self.strategy_names):
            for j, name2 in enumerate(self.strategy_names):
                if i < j:  # Only check each pair once
                    if name1 in corr_matrix.columns and name2 in corr_matrix.columns:
                        correlation = corr_matrix.loc[name1, name2]
                        
                        # If strategies are highly correlated
                        if correlation > self.correlation_threshold:
                            # Reduce weight of lower performing strategy
                            perf1 = np.mean(self.strategy_performances.get(name1, [0]))
                            perf2 = np.mean(self.strategy_performances.get(name2, [0]))
                            
                            if perf1 > perf2:
                                # Reduce weight of strategy 2
                                self.strategy_weights[name2] *= (1.0 - (correlation - self.correlation_threshold))
                            else:
                                # Reduce weight of strategy 1
                                self.strategy_weights[name1] *= (1.0 - (correlation - self.correlation_threshold))
    
    def _calculate_strategy_returns(
        self, 
        signals: pd.DataFrame, 
        market_data: pd.DataFrame,
        window: int
    ) -> pd.Series:
        """
        Calculate strategy returns based on signals and market data
        
        Args:
            signals: Strategy signals DataFrame
            market_data: Market data DataFrame
            window: Number of periods to calculate returns for
            
        Returns:
            Series of strategy returns
        """
        # Limit data to the window
        recent_data = market_data.tail(window + 1)  # +1 for calculating returns
        
        if len(recent_data) <= 1 or signals.empty:
            return pd.Series()
        
        # Calculate daily returns for all symbols
        market_returns = recent_data.pct_change().dropna()
        
        # Align signals with returns (signals are applied to next day's returns)
        aligned_signals = signals.reindex(market_returns.index, method='ffill')
        
        # Calculate strategy returns (simplified)
        strategy_returns = pd.Series(0.0, index=market_returns.index)
        
        for symbol in self.symbols:
            if symbol in aligned_signals.columns and symbol in market_returns.columns:
                # Multiply signal by next day's return
                symbol_contribution = aligned_signals[symbol] * market_returns[symbol]
                strategy_returns += symbol_contribution
        
        # Divide by number of symbols for equal weighting
        if len(self.symbols) > 0:
            strategy_returns /= len(self.symbols)
        
        return strategy_returns
    
    def calculate_position_size(self, signal: float, symbol: str, account_size: float) -> float:
        """
        Calculate position size based on the signal, symbol, and account size
        
        Args:
            signal: Trading signal (-1 to 1)
            symbol: The symbol to trade
            account_size: Current account size
            
        Returns:
            Position size in currency units
        """
        # For ensemble strategies, we can delegate to a "sizing strategy"
        # or implement a custom logic that considers the ensemble nature
        
        # Simple approach: take average of position sizes from all strategies
        total_size = 0.0
        count = 0
        
        for i, strategy in enumerate(self.strategies):
            # Get the weight for this strategy
            name = self.strategy_names[i]
            weight = self.strategy_weights.get(name, 0.0)
            
            # Only consider strategies with non-zero weight
            if weight > 0:
                # Get signal for this symbol from this strategy
                if hasattr(strategy, 'calculate_position_size'):
                    size = strategy.calculate_position_size(signal, symbol, account_size)
                    total_size += size * weight
                    count += weight
        
        # Return weighted average position size
        return total_size
    
    def add_strategy(self, strategy: Strategy, name: Optional[str] = None, weight: float = None) -> None:
        """
        Add a new strategy to the ensemble
        
        Args:
            strategy: Strategy instance to add
            name: Optional name for the strategy
            weight: Optional weight for the strategy
        """
        self.strategies.append(strategy)
        
        # Add strategy name
        if name is None:
            name = getattr(strategy, 'name', f"Strategy_{len(self.strategies)}")
        self.strategy_names.append(name)
        
        # Initialize performance tracking
        self.strategy_performances[name] = []
        self.strategy_signals[name] = pd.DataFrame()
        
        # Add strategy weight
        if weight is None:
            # Default weight depends on weighting method
            if self.weighting_method == WeightingMethod.EQUAL:
                equal_weight = 1.0 / len(self.strategies)
                self.strategy_weights = {name: equal_weight for name in self.strategy_names}
            else:
                # Start with minimum weight
                self.strategy_weights[name] = self.min_weight
        else:
            self.strategy_weights[name] = weight
        
        # Normalize weights
        self._normalize_weights()
        
        logger.info(f"Added strategy {name} to ensemble. Updated weights: {self.strategy_weights}")
    
    def remove_strategy(self, name: str) -> None:
        """
        Remove a strategy from the ensemble
        
        Args:
            name: Name of the strategy to remove
        """
        if name in self.strategy_names:
            index = self.strategy_names.index(name)
            
            # Remove from lists
            self.strategies.pop(index)
            self.strategy_names.remove(name)
            
            # Remove from dictionaries
            if name in self.strategy_weights:
                del self.strategy_weights[name]
            if name in self.strategy_performances:
                del self.strategy_performances[name]
            if name in self.strategy_signals:
                del self.strategy_signals[name]
            
            # Renormalize weights
            self._normalize_weights()
            
            logger.info(f"Removed strategy {name} from ensemble. Updated weights: {self.strategy_weights}")
        else:
            logger.warning(f"Strategy {name} not found in ensemble")
    
    def set_weighting_method(self, method: WeightingMethod) -> None:
        """
        Change the weighting method
        
        Args:
            method: New weighting method
        """
        self.weighting_method = method
        
        # Reset weights according to new method
        if method == WeightingMethod.EQUAL:
            equal_weight = 1.0 / len(self.strategies)
            self.strategy_weights = {name: equal_weight for name in self.strategy_names}
        elif method != WeightingMethod.CUSTOM:
            # For other methods, we'll recalculate on next signal generation
            self.last_rebalance = 0  # Force rebalance on next signal
        
        logger.info(f"Changed weighting method to {method.value}")
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """
        Get current strategy weights
        
        Returns:
            Dictionary of strategy weights
        """
        return self.strategy_weights.copy()
    
    def set_strategy_weight(self, name: str, weight: float) -> None:
        """
        Set weight for a specific strategy
        
        Args:
            name: Strategy name
            weight: New weight value
        """
        if name in self.strategy_names:
            # Apply min/max constraints
            weight = min(max(weight, self.min_weight), self.max_weight)
            self.strategy_weights[name] = weight
            
            # Normalize weights
            self._normalize_weights()
            
            logger.info(f"Set weight for strategy {name} to {weight}. Updated weights: {self.strategy_weights}")
        else:
            logger.warning(f"Strategy {name} not found in ensemble")


class DynamicEnsemble(StrategyEnsemble):
    """
    An extension of StrategyEnsemble that can dynamically adjust its component strategies
    based on market conditions and performance
    """
    
    def __init__(
        self,
        strategies: List[Strategy],
        strategy_selector: Optional[Callable[[pd.DataFrame], List[int]]] = None,
        min_active_strategies: int = 1,
        max_active_strategies: int = None,
        activation_threshold: float = 0.0,
        deactivation_threshold: float = -0.2,
        **kwargs
    ):
        """
        Initialize the dynamic ensemble
        
        Args:
            strategies: List of strategy instances
            strategy_selector: Optional function to select active strategies
            min_active_strategies: Minimum number of active strategies
            max_active_strategies: Maximum number of active strategies
            activation_threshold: Performance threshold for activation
            deactivation_threshold: Performance threshold for deactivation
            **kwargs: Additional arguments for StrategyEnsemble
        """
        super().__init__(strategies=strategies, **kwargs)
        
        self.strategy_selector = strategy_selector
        self.min_active_strategies = min_active_strategies
        self.max_active_strategies = max_active_strategies or len(strategies)
        self.activation_threshold = activation_threshold
        self.deactivation_threshold = deactivation_threshold
        
        # Initialize active strategies
        self.active_strategies = list(range(len(strategies)))
        self.strategy_active_status = {name: True for name in self.strategy_names}
    
    def generate_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals from the ensemble, dynamically adjusting active strategies
        
        Args:
            market_data: Market data DataFrame
            
        Returns:
            DataFrame with trading signals
        """
        self.bar_count += 1
        
        # Check if it's time to update active strategies
        if self.bar_count % self.rebalance_frequency == 0:
            self._update_active_strategies(market_data)
        
        # Generate signals only from active strategies
        all_signals = {}
        for i, strategy in enumerate(self.strategies):
            name = self.strategy_names[i]
            
            # Skip inactive strategies
            if not self.strategy_active_status.get(name, True):
                continue
            
            signals = strategy.generate_signals(market_data)
            
            # Store signals for performance tracking
            self.strategy_signals[name] = signals
            
            # Convert to dictionary for easier aggregation
            if not signals.empty:
                signals_dict = {}
                for symbol in self.symbols:
                    if symbol in signals.columns:
                        signals_dict[symbol] = signals.iloc[-1][symbol]
                    else:
                        signals_dict[symbol] = 0.0
                all_signals[name] = signals_dict
        
        # Combine signals using weighted average (only active strategies)
        combined_signals = self._combine_signals(all_signals)
        
        # Convert combined signals to DataFrame format
        result = pd.DataFrame([combined_signals], columns=self.symbols)
        
        # Add timestamp if the input data has an index
        if not market_data.empty and isinstance(market_data.index, pd.DatetimeIndex):
            result.index = [market_data.index[-1]]
        
        return result
    
    def _update_active_strategies(self, market_data: pd.DataFrame) -> None:
        """
        Update the list of active strategies based on performance
        
        Args:
            market_data: Current market data
        """
        # If we have a custom strategy selector, use it
        if self.strategy_selector is not None:
            selected_indices = self.strategy_selector(market_data)
            self.active_strategies = selected_indices
            
            # Update active status
            for i, name in enumerate(self.strategy_names):
                self.strategy_active_status[name] = i in self.active_strategies
            
            return
        
        # Otherwise, use performance-based selection
        if self.bar_count < self.performance_window:
            # Not enough history yet, keep all strategies active
            return
        
        # Calculate performance for each strategy
        performances = {}
        
        for name in self.strategy_names:
            # Get signals history for this strategy
            signals = self.strategy_signals.get(name)
            if signals is None or signals.empty:
                performances[name] = -1.0  # Penalize empty signals
                continue
            
            # Calculate return series based on signals and market data
            returns = self._calculate_strategy_returns(signals, market_data, self.performance_window)
            
            if len(returns) > 0:
                # Calculate Sharpe ratio
                sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
                sharpe = sharpe * np.sqrt(252)  # Annualize
                performances[name] = sharpe
            else:
                performances[name] = 0.0
        
        # Update active status based on performance
        active_count = sum(1 for status in self.strategy_active_status.values() if status)
        
        for name, performance in performances.items():
            current_status = self.strategy_active_status.get(name, True)
            
            if current_status:
                # Check if strategy should be deactivated
                if (performance < self.deactivation_threshold and 
                    active_count > self.min_active_strategies):
                    self.strategy_active_status[name] = False
                    active_count -= 1
                    logger.info(f"Deactivated strategy {name} (performance: {performance:.4f})")
            else:
                # Check if strategy should be activated
                if (performance > self.activation_threshold and 
                    active_count < self.max_active_strategies):
                    self.strategy_active_status[name] = True
                    active_count += 1
                    logger.info(f"Activated strategy {name} (performance: {performance:.4f})")
        
        # Update active strategies list
        self.active_strategies = [
            i for i, name in enumerate(self.strategy_names) 
            if self.strategy_active_status.get(name, True)
        ]
        
        # Make sure we have at least min_active_strategies
        if active_count < self.min_active_strategies:
            # Activate best performing inactive strategies
            inactive_performances = {
                name: perf for name, perf in performances.items() 
                if not self.strategy_active_status.get(name, True)
            }
            
            # Sort by performance
            sorted_inactive = sorted(
                inactive_performances.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Activate strategies until we reach minimum
            for name, _ in sorted_inactive:
                if active_count < self.min_active_strategies:
                    self.strategy_active_status[name] = True
                    active_count += 1
                    logger.info(f"Activated strategy {name} to meet minimum active requirement")
                else:
                    break
        
        # Update active strategies list again
        self.active_strategies = [
            i for i, name in enumerate(self.strategy_names) 
            if self.strategy_active_status.get(name, True)
        ]
        
        # Update weights to reflect active strategies
        self._update_weights_for_active_strategies()
    
    def _update_weights_for_active_strategies(self) -> None:
        """Update weights based on active strategies"""
        # Set inactive strategies to weight 0
        for name in self.strategy_names:
            if not self.strategy_active_status.get(name, True):
                self.strategy_weights[name] = 0.0
        
        # Normalize weights for active strategies
        self._normalize_weights()
    
    def get_active_strategies(self) -> List[str]:
        """
        Get list of active strategy names
        
        Returns:
            List of active strategy names
        """
        return [name for name, active in self.strategy_active_status.items() if active] 