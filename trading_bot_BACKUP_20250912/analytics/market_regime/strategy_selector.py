"""
Adaptive Strategy Selector

This module provides functionality to dynamically select and allocate capital to trading 
strategies based on their historical performance in different market regimes.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import threading
import json
import os

# Import local modules
from trading_bot.analytics.market_regime.detector import MarketRegimeType
from trading_bot.analytics.market_regime.performance import RegimePerformanceTracker

logger = logging.getLogger(__name__)

class StrategySelector:
    """
    Dynamically selects and weights trading strategies based on market regimes
    and historical performance data.
    
    Features:
    - Strategy scoring and selection based on regime-specific performance
    - Dynamic strategy weighting and allocation
    - Automatic strategy activation/deactivation
    - Performance monitoring and adaptation
    """
    
    def __init__(
        self,
        performance_tracker: RegimePerformanceTracker,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize strategy selector.
        
        Args:
            performance_tracker: RegimePerformanceTracker instance
            config: Configuration parameters
        """
        self.config = config or {}
        self.performance_tracker = performance_tracker
        
        # Directory to store strategy selector data
        self.selector_dir = self.config.get("selector_dir", "data/strategy_selector")
        
        # Strategy scores by regime
        self.strategy_scores: Dict[MarketRegimeType, Dict[str, float]] = {}
        
        # Current active strategies by symbol
        self.active_strategies: Dict[str, List[str]] = {}
        
        # Current strategy weights by symbol
        self.strategy_weights: Dict[str, Dict[str, float]] = {}
        
        # Strategy configuration data
        self.strategy_configs: Dict[str, Dict[str, Any]] = {}
        
        # Timeframe mappings
        self.timeframe_mappings: Dict[str, Dict[MarketRegimeType, str]] = {}
        
        # Scoring settings
        self.scoring_weights = self.config.get("scoring_weights", {
            "profit_factor": 0.3,
            "sharpe_ratio": 0.2,
            "win_rate": 0.1,
            "expectancy": 0.2,
            "correlation": 0.1,
            "sample_size": 0.1,
        })
        
        # Selection settings
        self.min_strategies = self.config.get("min_strategies", 1)
        self.max_strategies = self.config.get("max_strategies", 5)
        self.min_score_threshold = self.config.get("min_score_threshold", 0.4)
        
        # Diversification settings
        self.correlation_penalty = self.config.get("correlation_penalty", 0.5)
        self.min_correlation_sample = self.config.get("min_correlation_sample", 10)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize
        self._load_selector_data()
        
        logger.info("Strategy Selector initialized")
    
    def _load_selector_data(self) -> None:
        """Load strategy selector data from disk."""
        try:
            # Create selector directory if it doesn't exist
            os.makedirs(self.selector_dir, exist_ok=True)
            
            # Load strategy configuration data
            config_file = os.path.join(self.selector_dir, "strategy_configs.json")
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    self.strategy_configs = json.load(f)
            
            # Load timeframe mappings
            timeframe_file = os.path.join(self.selector_dir, "timeframe_mappings.json")
            if os.path.exists(timeframe_file):
                with open(timeframe_file, 'r') as f:
                    mappings_data = json.load(f)
                    
                    # Convert string keys to MarketRegimeType
                    for symbol, regimes in mappings_data.items():
                        converted_regimes = {}
                        for regime_str, timeframe in regimes.items():
                            try:
                                regime_type = MarketRegimeType(regime_str)
                                converted_regimes[regime_type] = timeframe
                            except ValueError:
                                logger.warning(f"Unknown regime type in timeframe mappings: {regime_str}")
                        
                        self.timeframe_mappings[symbol] = converted_regimes
            
            loaded_count = len(self.strategy_configs)
            if loaded_count > 0:
                logger.info(f"Loaded configuration data for {loaded_count} strategies")
            else:
                logger.info("No strategy configuration data found")
                
        except Exception as e:
            logger.error(f"Error loading strategy selector data: {str(e)}")
    
    def _save_selector_data(self) -> None:
        """Save strategy selector data to disk."""
        try:
            # Create selector directory if it doesn't exist
            os.makedirs(self.selector_dir, exist_ok=True)
            
            # Save strategy configuration data
            config_file = os.path.join(self.selector_dir, "strategy_configs.json")
            with open(config_file, 'w') as f:
                json.dump(self.strategy_configs, f, indent=2)
            
            # Save timeframe mappings
            timeframe_file = os.path.join(self.selector_dir, "timeframe_mappings.json")
            
            # Convert MarketRegimeType to strings for serialization
            serializable_mappings = {}
            for symbol, regimes in self.timeframe_mappings.items():
                serializable_regimes = {regime.value: timeframe for regime, timeframe in regimes.items()}
                serializable_mappings[symbol] = serializable_regimes
            
            with open(timeframe_file, 'w') as f:
                json.dump(serializable_mappings, f, indent=2)
            
            logger.info(f"Saved strategy selector data for {len(self.strategy_configs)} strategies")
            
        except Exception as e:
            logger.error(f"Error saving strategy selector data: {str(e)}")
    
    def register_strategy(
        self, strategy_id: str, strategy_config: Dict[str, Any]
    ) -> None:
        """
        Register a strategy with the selector.
        
        Args:
            strategy_id: Strategy identifier
            strategy_config: Strategy configuration data
        """
        with self._lock:
            try:
                self.strategy_configs[strategy_id] = strategy_config
                
                # Save to disk
                self._save_selector_data()
                
                logger.info(f"Registered strategy {strategy_id}")
                
            except Exception as e:
                logger.error(f"Error registering strategy: {str(e)}")
    
    def set_timeframe_mapping(
        self, symbol: str, regime_type: MarketRegimeType, timeframe: str
    ) -> None:
        """
        Set the preferred timeframe for a symbol in a specific market regime.
        
        Args:
            symbol: Symbol
            regime_type: Market regime type
            timeframe: Preferred timeframe
        """
        with self._lock:
            try:
                if symbol not in self.timeframe_mappings:
                    self.timeframe_mappings[symbol] = {}
                
                self.timeframe_mappings[symbol][regime_type] = timeframe
                
                # Save to disk
                self._save_selector_data()
                
                logger.info(f"Set preferred timeframe for {symbol} in {regime_type} to {timeframe}")
                
            except Exception as e:
                logger.error(f"Error setting timeframe mapping: {str(e)}")
    
    def get_preferred_timeframe(
        self, symbol: str, regime_type: MarketRegimeType, default_timeframe: str = "1h"
    ) -> str:
        """
        Get the preferred timeframe for a symbol in a specific market regime.
        
        Args:
            symbol: Symbol
            regime_type: Market regime type
            default_timeframe: Default timeframe to use if no mapping exists
            
        Returns:
            Preferred timeframe
        """
        try:
            if symbol in self.timeframe_mappings and regime_type in self.timeframe_mappings[symbol]:
                return self.timeframe_mappings[symbol][regime_type]
            
            # Fall back to NORMAL regime if it exists
            if symbol in self.timeframe_mappings and MarketRegimeType.NORMAL in self.timeframe_mappings[symbol]:
                return self.timeframe_mappings[symbol][MarketRegimeType.NORMAL]
            
            return default_timeframe
                
        except Exception as e:
            logger.error(f"Error getting preferred timeframe: {str(e)}")
            return default_timeframe
    
    def update_strategy_scores(
        self, regime_type: MarketRegimeType, available_strategies: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Update scores for all strategies in a specific market regime.
        
        Args:
            regime_type: Market regime type
            available_strategies: Optional list of strategies to score (defaults to all)
            
        Returns:
            Dict mapping strategy IDs to scores
        """
        with self._lock:
            try:
                # Initialize scores for this regime
                if regime_type not in self.strategy_scores:
                    self.strategy_scores[regime_type] = {}
                
                # Get list of strategies to score
                strategies_to_score = available_strategies or list(self.strategy_configs.keys())
                
                # Get relevant performance data
                for strategy_id in strategies_to_score:
                    # Skip if strategy doesn't exist
                    if strategy_id not in self.strategy_configs:
                        continue
                    
                    # Calculate score based on performance metrics
                    try:
                        regime_performance = self.performance_tracker.get_performance_by_regime(strategy_id)
                        
                        # Skip if no performance data for this regime
                        if regime_type not in regime_performance:
                            self.strategy_scores[regime_type][strategy_id] = 0.0
                            continue
                        
                        performance_metrics = regime_performance[regime_type]
                        
                        # Calculate score components
                        score_components = {}
                        
                        # Profit factor component
                        if "profit_factor_mean" in performance_metrics:
                            pf = min(performance_metrics["profit_factor_mean"], 3.0)  # Cap at 3.0
                            score_components["profit_factor"] = (pf - 1.0) / 2.0 if pf > 1.0 else 0.0
                        
                        # Sharpe ratio component
                        if "sharpe_ratio_mean" in performance_metrics:
                            sr = min(performance_metrics["sharpe_ratio_mean"], 3.0)  # Cap at 3.0
                            score_components["sharpe_ratio"] = sr / 3.0 if sr > 0 else 0.0
                        
                        # Win rate component
                        if "win_rate_mean" in performance_metrics:
                            wr = performance_metrics["win_rate_mean"]
                            score_components["win_rate"] = wr - 0.5 if wr > 0.5 else 0.0
                        
                        # Expectancy component
                        if "expectancy_mean" in performance_metrics:
                            exp = min(performance_metrics["expectancy_mean"], 2.0)  # Cap at 2.0
                            score_components["expectancy"] = exp / 2.0 if exp > 0 else 0.0
                        
                        # Sample size component (more samples = more confidence)
                        sample_size = performance_metrics.get("sample_size", 0)
                        max_samples = 50
                        score_components["sample_size"] = min(sample_size / max_samples, 1.0)
                        
                        # Correlation penalty (lower is better)
                        # We'll add this later when calculating final scores
                        
                        # Calculate weighted score
                        score = 0.0
                        total_weight = 0.0
                        
                        for component, value in score_components.items():
                            if component in self.scoring_weights:
                                score += value * self.scoring_weights[component]
                                total_weight += self.scoring_weights[component]
                        
                        # Normalize score
                        if total_weight > 0:
                            score = score / total_weight
                            
                            # Add trend bonus/penalty (recent improvement or degradation)
                            if "profit_factor_trend" in performance_metrics:
                                trend = performance_metrics["profit_factor_trend"]
                                trend_adjustment = max(min(trend * 5, 0.1), -0.1)  # ±10% max adjustment
                                score = max(0.0, min(1.0, score + trend_adjustment))
                        
                        # Store score
                        self.strategy_scores[regime_type][strategy_id] = score
                    
                    except Exception as e:
                        logger.warning(f"Error calculating score for strategy {strategy_id}: {str(e)}")
                        self.strategy_scores[regime_type][strategy_id] = 0.0
                
                # Apply correlation penalty to highly correlated strategies
                self._apply_correlation_penalty(regime_type, strategies_to_score)
                
                logger.debug(f"Updated strategy scores for {regime_type}")
                return self.strategy_scores[regime_type].copy()
                
            except Exception as e:
                logger.error(f"Error updating strategy scores: {str(e)}")
                return {}
    
    def _apply_correlation_penalty(
        self, regime_type: MarketRegimeType, strategy_ids: List[str]
    ) -> None:
        """
        Apply correlation penalty to highly correlated strategies.
        
        Args:
            regime_type: Market regime type
            strategy_ids: List of strategies to evaluate
        """
        try:
            # We need at least 2 strategies to calculate correlation
            if len(strategy_ids) < 2:
                return
            
            # Get correlation analysis
            correlation_analysis = self.performance_tracker.analyze_correlation(
                strategy_ids, regime_type, metric_name="returns"
            )
            
            # Skip if insufficient data
            if 'error' in correlation_analysis or correlation_analysis.get('sample_size', 0) < self.min_correlation_sample:
                return
            
            # Apply correlation penalty
            correlation_matrix = correlation_analysis.get('correlation_matrix', {})
            avg_correlations = correlation_analysis.get('avg_correlations', {})
            
            for strategy_id, avg_correlation in avg_correlations.items():
                # Skip strategies not in our list
                if strategy_id not in strategy_ids or strategy_id not in self.strategy_scores[regime_type]:
                    continue
                
                # Apply penalty (higher correlation = higher penalty)
                penalty = self.correlation_penalty * avg_correlation * self.scoring_weights.get("correlation", 0.1)
                self.strategy_scores[regime_type][strategy_id] = max(0.0, self.strategy_scores[regime_type][strategy_id] - penalty)
            
        except Exception as e:
            logger.warning(f"Error applying correlation penalty: {str(e)}")
    
    def select_strategies(
        self, symbol: str, regime_type: MarketRegimeType, timeframe: str,
        available_strategies: Optional[List[str]] = None,
        force_update: bool = False,
        min_strategies: Optional[int] = None,
        max_strategies: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Select best strategies for a symbol in a specific market regime.
        
        Args:
            symbol: Symbol
            regime_type: Market regime type
            timeframe: Timeframe
            available_strategies: Optional list of available strategies
            force_update: Force update of strategy scores
            min_strategies: Minimum number of strategies to select (overrides config)
            max_strategies: Maximum number of strategies to select (overrides config)
            
        Returns:
            List of selected strategy configurations with weights
        """
        with self._lock:
            try:
                # Use config values if not specified
                min_strat = min_strategies if min_strategies is not None else self.min_strategies
                max_strat = max_strategies if max_strategies is not None else self.max_strategies
                
                # Get list of available strategies
                strategies = available_strategies or list(self.strategy_configs.keys())
                
                # Filter to strategies compatible with this symbol and timeframe
                compatible_strategies = []
                for strategy_id in strategies:
                    if strategy_id not in self.strategy_configs:
                        continue
                    
                    config = self.strategy_configs[strategy_id]
                    
                    # Check symbol compatibility
                    symbols = config.get("compatible_symbols", [])
                    if symbols and symbol not in symbols:
                        continue
                    
                    # Check symbol exclusions
                    excluded_symbols = config.get("excluded_symbols", [])
                    if symbol in excluded_symbols:
                        continue
                    
                    # Check timeframe compatibility
                    timeframes = config.get("compatible_timeframes", [])
                    if timeframes and timeframe not in timeframes:
                        continue
                    
                    # Check regime compatibility
                    regimes = config.get("compatible_regimes", [])
                    if regimes and regime_type.value not in regimes:
                        continue
                    
                    # Strategy is compatible
                    compatible_strategies.append(strategy_id)
                
                # Update strategy scores if needed
                if force_update or regime_type not in self.strategy_scores:
                    self.update_strategy_scores(regime_type, compatible_strategies)
                
                # Get scores for compatible strategies
                strategy_scores = {}
                for strategy_id in compatible_strategies:
                    if regime_type in self.strategy_scores and strategy_id in self.strategy_scores[regime_type]:
                        strategy_scores[strategy_id] = self.strategy_scores[regime_type][strategy_id]
                    else:
                        strategy_scores[strategy_id] = 0.0
                
                # Sort strategies by score (descending)
                sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
                
                # Select top N strategies above threshold
                selected_strategies = []
                for strategy_id, score in sorted_strategies:
                    if score >= self.min_score_threshold:
                        selected_strategies.append(strategy_id)
                    
                    # Stop if we have enough strategies
                    if len(selected_strategies) >= max_strat:
                        break
                
                # Ensure minimum number of strategies
                if len(selected_strategies) < min_strat and len(sorted_strategies) > 0:
                    # Add strategies below threshold if needed
                    for strategy_id, score in sorted_strategies:
                        if strategy_id not in selected_strategies:
                            selected_strategies.append(strategy_id)
                        
                        # Stop if we have enough strategies
                        if len(selected_strategies) >= min_strat:
                            break
                
                # Calculate weights
                weights = self._calculate_strategy_weights(selected_strategies, strategy_scores)
                
                # Update active strategies
                self._update_active_strategies(symbol, selected_strategies)
                
                # Update strategy weights
                self._update_strategy_weights(symbol, weights)
                
                # Prepare result
                result = []
                for strategy_id in selected_strategies:
                    strategy_info = {
                        "strategy_id": strategy_id,
                        "score": strategy_scores.get(strategy_id, 0.0),
                        "weight": weights.get(strategy_id, 0.0),
                        "config": self.strategy_configs.get(strategy_id, {}).copy() if strategy_id in self.strategy_configs else {}
                    }
                    result.append(strategy_info)
                
                logger.info(f"Selected {len(result)} strategies for {symbol} in {regime_type}")
                return result
                
            except Exception as e:
                logger.error(f"Error selecting strategies: {str(e)}")
                return []
    
    def _calculate_strategy_weights(
        self, selected_strategies: List[str], scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate weights for selected strategies.
        
        Args:
            selected_strategies: List of selected strategy IDs
            scores: Dict mapping strategy IDs to scores
            
        Returns:
            Dict mapping strategy IDs to weights
        """
        try:
            # If no strategies, return empty dict
            if not selected_strategies:
                return {}
            
            # If only one strategy, give it 100% weight
            if len(selected_strategies) == 1:
                return {selected_strategies[0]: 1.0}
            
            # Calculate weighted scores
            total_score = sum(scores.get(strategy_id, 0.0) for strategy_id in selected_strategies)
            
            # Equal weights if total score is 0
            if total_score <= 0:
                weight = 1.0 / len(selected_strategies)
                return {strategy_id: weight for strategy_id in selected_strategies}
            
            # Calculate weights proportional to scores
            weights = {}
            for strategy_id in selected_strategies:
                score = scores.get(strategy_id, 0.0)
                weights[strategy_id] = score / total_score
            
            return weights
            
        except Exception as e:
            logger.error(f"Error calculating strategy weights: {str(e)}")
            
            # Fall back to equal weights
            weight = 1.0 / len(selected_strategies)
            return {strategy_id: weight for strategy_id in selected_strategies}
    
    def _update_active_strategies(self, symbol: str, selected_strategies: List[str]) -> None:
        """
        Update list of active strategies for a symbol.
        
        Args:
            symbol: Symbol
            selected_strategies: List of selected strategy IDs
        """
        self.active_strategies[symbol] = selected_strategies
    
    def _update_strategy_weights(self, symbol: str, weights: Dict[str, float]) -> None:
        """
        Update strategy weights for a symbol.
        
        Args:
            symbol: Symbol
            weights: Dict mapping strategy IDs to weights
        """
        self.strategy_weights[symbol] = weights
    
    def get_active_strategies(self, symbol: str) -> List[str]:
        """
        Get list of active strategies for a symbol.
        
        Args:
            symbol: Symbol
            
        Returns:
            List of active strategy IDs
        """
        return self.active_strategies.get(symbol, [])
    
    def get_strategy_weight(self, symbol: str, strategy_id: str) -> float:
        """
        Get weight for a strategy for a symbol.
        
        Args:
            symbol: Symbol
            strategy_id: Strategy identifier
            
        Returns:
            Strategy weight
        """
        if symbol in self.strategy_weights and strategy_id in self.strategy_weights[symbol]:
            return self.strategy_weights[symbol][strategy_id]
        
        return 0.0
    
    def get_strategy_scores(self, regime_type: MarketRegimeType) -> Dict[str, float]:
        """
        Get scores for all strategies in a specific market regime.
        
        Args:
            regime_type: Market regime type
            
        Returns:
            Dict mapping strategy IDs to scores
        """
        return self.strategy_scores.get(regime_type, {}).copy()
