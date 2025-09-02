"""
Strategy Prioritizer Module

This module uses LLM evaluations to analyze market conditions and
recommend which trading strategies should be prioritized based on
the current market regime, with dynamic adjustments based on recent
strategy performance metrics.
"""

import os
import json
import time
import logging
import openai
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import re
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StrategyPrioritizer")

# Import utilities
from trading_bot.utils.strategy_library import STRATEGY_METADATA, get_strategy_names
from trading_bot.utils.market_context_fetcher import get_current_market_context, get_mock_market_context
from trading_bot.ai_scoring.performance_adjuster import PerformanceAdjuster


class StrategyPrioritizer:
    """
    StrategyPrioritizer uses LLM evaluations to rank trading strategies
    based on current market conditions. It can also generate mock responses for testing.
    Includes performance-based adjustments to dynamically rebalance allocations.
    """
    
    def __init__(
        self,
        strategies: List[str] = None,
        api_key: Optional[str] = None,
        use_mock: bool = False,
        cache_duration: int = 60,
        cache_dir: Optional[str] = None,
        enable_performance_adjustment: bool = True,
        performance_weight: float = 0.3,
        performance_lookback_days: int = 30,
        performance_metrics: List[str] = None
    ):
        """
        Initialize the StrategyPrioritizer with configuration parameters.
        
        Args:
            strategies: List of strategy names to prioritize
            api_key: API key for GPT or other LLM service
            use_mock: Whether to use mock responses instead of API calls
            cache_duration: How long to cache results (in minutes)
            cache_dir: Directory to store disk-based cache files (if None, uses in-memory cache)
            enable_performance_adjustment: Whether to adjust allocations based on performance
            performance_weight: Weight to give performance metrics vs market regime (0-1)
            performance_lookback_days: Days of historical performance to consider
            performance_metrics: List of metrics to use for performance evaluation
        """
        self.strategies = strategies or [
            'trend_following', 'momentum', 'mean_reversion', 
            'breakout_swing', 'volatility_breakout', 'option_spreads'
        ]
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.use_mock = use_mock
        self.cache_duration = cache_duration
        self.cache_dir = cache_dir
        
        # Performance adjustment settings
        self.enable_performance_adjustment = enable_performance_adjustment
        self.performance_weight = performance_weight
        self.performance_lookback_days = performance_lookback_days
        self.performance_metrics = performance_metrics or ['sharpe_ratio', 'total_return', 'win_rate', 'max_drawdown']
        
        # Initialize performance adjuster and data storage
        self.performance_adjuster = PerformanceAdjuster(
            performance_weight=self.performance_weight,
            performance_metrics=self.performance_metrics
        )
        self.strategy_performance = {}
        
        # Initialize cache
        self.cache = {}
        self.cache_timestamp = {}
        
        # Create cache directory if it doesn't exist
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Created cache directory at {self.cache_dir}")
        
        logger.info(f"Initialized StrategyPrioritizer with {len(self.strategies)} strategies")
        logger.info(f"Using {'mock' if use_mock else 'API'} mode")
        if self.cache_dir:
            logger.info(f"Using disk-based cache at {self.cache_dir}")
        else:
            logger.info("Using in-memory cache")
    
    def get_strategy_allocation(
        self, 
        market_context: Dict[str, Any],
        force_refresh: bool = False,
        performance_data: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, float]:
        """
        Get recommended allocations for each strategy based on market context.
        
        Args:
            market_context: Dictionary with market data and indicators
            force_refresh: Whether to force a refresh even if cache is valid
            performance_data: Optional dictionary mapping strategy names to performance metrics
            
        Returns:
            Dictionary mapping strategy names to allocation percentages
        """
        # Generate a simple cache key based on market context
        cache_key = f"alloc_{market_context.get('regime', '')}"
        
        # Check if we have cached data and it's still valid
        if not force_refresh:
            # Try disk cache first if enabled
            if self.cache_dir:
                cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
                if os.path.exists(cache_file):
                    try:
                        # Check file modification time
                        mtime = os.path.getmtime(cache_file)
                        cache_time = datetime.fromtimestamp(mtime)
                        cache_age = (datetime.now() - cache_time).total_seconds() / 60
                        
                        if cache_age < self.cache_duration:
                            logger.debug(f"Using disk-cached allocations ({cache_age:.1f} min old)")
                            with open(cache_file, 'r') as f:
                                allocations = json.load(f)
                                # Apply performance adjustments if enabled
                                if self.enable_performance_adjustment and performance_data:
                                    logger.debug("Applying performance adjustments to cached allocations")
                                    return self._adjust_allocations_by_performance(allocations, performance_data)
                                return allocations
                    except Exception as e:
                        logger.warning(f"Error reading cache file: {e}")
            
            # Fall back to in-memory cache
            if cache_key in self.cache:
                cache_time = self.cache_timestamp.get(cache_key)
                if cache_time:
                    cache_age = (datetime.now() - cache_time).total_seconds() / 60
                    if cache_age < self.cache_duration:
                        logger.debug(f"Using in-memory cached allocations ({cache_age:.1f} min old)")
                        allocations = self.cache[cache_key]
                        # Apply performance adjustments if enabled
                        if self.enable_performance_adjustment and performance_data:
                            logger.debug("Applying performance adjustments to cached allocations")
                            return self._adjust_allocations_by_performance(allocations, performance_data)
                        return allocations
        
        try:
            # If using mock or API call fails, use mock response
            if self.use_mock:
                allocations = self._get_mock_prioritization(market_context)
            else:
                # Try to get allocations from LLM API
                allocations = self._get_api_prioritization(market_context)
                
                # If API call fails, fall back to mock
                if not allocations:
                    logger.warning("Failed to get LLM allocations, falling back to mock")
                    allocations = self._get_mock_prioritization(market_context)
        except Exception as e:
            logger.error(f"Error getting strategy allocations: {e}")
            allocations = self._get_mock_prioritization(market_context)
            
        # Apply performance-based adjustments if enabled
        if self.enable_performance_adjustment:
            perf_data = performance_data or self.strategy_performance
            if perf_data:
                logger.info("Applying performance-based adjustments to allocations")
                allocations = self.performance_adjuster.adjust_allocations(allocations, perf_data)
        
        # Ensure allocations are in float format
        allocations = {k: float(v) for k, v in allocations.items()}
        
        # Cache the result
        self.cache[cache_key] = allocations
        self.cache_timestamp[cache_key] = datetime.now()
        
        # Save to disk cache if enabled
        if self.cache_dir:
            try:
                cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
                with open(cache_file, 'w') as f:
                    json.dump(allocations, f)
                logger.debug(f"Saved allocations to disk cache: {cache_file}")
            except Exception as e:
                logger.warning(f"Error writing to disk cache: {e}")
        
        # Store performance data for future reference if provided
        if performance_data:
            self.strategy_performance = performance_data
            logger.debug(f"Stored performance data for {len(performance_data)} strategies")
        
        return allocations
    
    def _get_api_prioritization(self, market_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Get strategy allocations from LLM API based on market context.
        
        Args:
            market_context: Dictionary with market data and indicators
            
        Returns:
            Dictionary mapping strategy names to allocation percentages
        """
        # This would include API calls to LLM service
        # For now, just return empty dict to indicate it's not implemented
        logger.warning("LLM API prioritization not implemented, using mock")
        return {}
    
    def _get_mock_prioritization(self, market_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate mock strategy allocations based on market context.
        
        Args:
            market_context: Dictionary with market data and indicators
            
        Returns:
            Dictionary mapping strategy names to allocation percentages
        """
        regime = market_context.get('regime', 'sideways')
        
        # Define different allocation weights for different market regimes
        if regime == 'bullish':
            weights = {
                'trend_following': 0.3,
                'momentum': 0.35,
                'breakout_swing': 0.25,
                'mean_reversion': 0.03,
                'volatility_breakout': 0.02,
                'option_spreads': 0.05
            }
        elif regime == 'bearish':
            weights = {
                'trend_following': 0.25,
                'momentum': 0.05,
                'breakout_swing': 0.05,
                'mean_reversion': 0.20,
                'volatility_breakout': 0.20,
                'option_spreads': 0.25
            }
        elif regime == 'volatile':
            weights = {
                'trend_following': 0.03,
                'momentum': 0.01,
                'breakout_swing': 0.10,
                'mean_reversion': 0.20,
                'volatility_breakout': 0.35,
                'option_spreads': 0.31
            }
        else:  # sideways
            weights = {
                'trend_following': 0.10,
                'momentum': 0.05,
                'breakout_swing': 0.15,
                'mean_reversion': 0.40,
                'volatility_breakout': 0.10,
                'option_spreads': 0.20
            }
        
        # Add some randomness to weights (±5%)
        for strategy in weights:
            noise = random.uniform(-0.05, 0.05)
            weights[strategy] = max(0.01, weights[strategy] + noise)
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        normalized_weights = {s: w / total_weight for s, w in weights.items()}
        
        # Convert to percentages
        allocations = {s: w * 100 for s, w in normalized_weights.items()}
        
        # Log the types of allocations for debugging
        for strategy, allocation in allocations.items():
            logger.debug(f"Strategy {strategy} allocation: {allocation} (type: {type(allocation).__name__})")
        
        return allocations


# Command-line testing
if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Available strategies
    strategies = [
        "breakout_swing",
        "momentum",
        "mean_reversion",
        "trend_following",
        "volatility_breakout",
        "option_spreads"
    ]
    
    # Initialize prioritizer
    use_mock = os.getenv("OPENAI_API_KEY") is None
    prioritizer = StrategyPrioritizer(
        strategies=strategies,
        use_mock=use_mock
    )
    
    print("\nTesting Strategy Prioritizer")
    print(f"Using mock data: {use_mock}")
    
    try:
        # Test with current market conditions
        print("\nPrioritizing strategies for current market conditions...")
        result = prioritizer.get_strategy_allocation({})
        
        print("\nStrategy Allocations:")
        for strategy, allocation in result.items():
            print(f"- {strategy}: {allocation}%")
        
    except Exception as e:
        print(f"Error testing strategy prioritizer: {str(e)}") 