#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RegimeAwareStrategyPrioritizer - Integrates market regime classification with strategy prioritization
to deliver more context-aware strategy allocations based on detected market regimes.
"""

import os
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

from trading_bot.ai_scoring.strategy_prioritizer import StrategyPrioritizer
from trading_bot.utils.market_context_fetcher import MarketContextFetcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RegimeAwareStrategyPrioritizer")

class RegimeClassifier:
    """
    Classifies market regimes based on various indicators and market data.
    Identifies regimes such as: bullish, bearish, volatile, sideways, mean-reverting, etc.
    """
    
    def __init__(self, lookback_days: int = 60):
        """
        Initialize the RegimeClassifier.
        
        Args:
            lookback_days: Number of days to look back for regime classification
        """
        self.lookback_days = lookback_days
        self.market_fetcher = MarketContextFetcher()
        logger.info(f"RegimeClassifier initialized with lookback of {lookback_days} days")
        
    def classify_regime(self) -> Dict[str, Any]:
        """
        Classify the current market regime based on market indicators.
        
        Returns:
            Dict containing regime classification and confidence scores
        """
        # Fetch market data needed for classification
        market_data = self._get_market_data()
        
        # Detect primary regime (main classification)
        primary_regime = self._detect_primary_regime(market_data)
        
        # Detect secondary characteristics
        secondary_characteristics = self._detect_secondary_characteristics(market_data)
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(market_data, primary_regime)
        
        regime_data = {
            "primary_regime": primary_regime,
            "secondary_characteristics": secondary_characteristics,
            "confidence_scores": confidence_scores,
            "timestamp": datetime.now().isoformat(),
            "indicators_used": list(market_data.keys())
        }
        
        logger.info(f"Market regime classified as {primary_regime} with "
                   f"secondary characteristics: {secondary_characteristics}")
        
        return regime_data
    
    def _get_market_data(self) -> Dict[str, Any]:
        """
        Fetch and prepare market data for regime classification.
        
        Returns:
            Dict of market data metrics
        """
        # Get data from MarketContextFetcher
        market_context = self.market_fetcher.get_market_context()
        
        # Extract and compute key metrics for regime classification
        vix = market_context.get("vix", {}).get("current_value", 15.0)
        vix_percentile = market_context.get("vix", {}).get("percentile", 50.0)
        
        # Get trend data
        trend_strength = market_context.get("trend_strength", {}).get("value", 0.5)
        trend_direction = market_context.get("trend_direction", {}).get("value", 0)
        
        # Sector rotation data
        sector_dispersion = self._calculate_sector_dispersion(market_context.get("sector_performance", {}))
        
        # Compute additional metrics
        market_data = {
            "vix": vix,
            "vix_percentile": vix_percentile,
            "trend_strength": trend_strength,
            "trend_direction": trend_direction,
            "sector_dispersion": sector_dispersion,
            "breadth": market_context.get("market_breadth", {}).get("value", 0.5),
            "momentum": market_context.get("momentum", {}).get("value", 0.0),
            "volume": market_context.get("volume", {}).get("relative_change", 0.0),
        }
        
        return market_data
    
    def _calculate_sector_dispersion(self, sector_data: Dict[str, float]) -> float:
        """
        Calculate dispersion of sector performance as a regime indicator.
        High dispersion suggests rotational markets, while low dispersion suggests uniform trends.
        
        Args:
            sector_data: Dict of sector performance data
            
        Returns:
            Dispersion value as a float between 0 and 1
        """
        if not sector_data:
            return 0.5  # Default value
        
        try:
            sector_values = [float(v) for v in sector_data.values()]
            if len(sector_values) < 2:
                return 0.5
            
            # Calculate normalized standard deviation
            dispersion = np.std(sector_values) / (max(abs(np.max(sector_values)), abs(np.min(sector_values))) + 1e-6)
            return min(max(dispersion, 0.0), 1.0)  # Ensure value is between 0 and 1
        except (ValueError, TypeError):
            logger.warning("Could not calculate sector dispersion due to invalid values")
            return 0.5
    
    def _detect_primary_regime(self, market_data: Dict[str, Any]) -> str:
        """
        Detect the primary market regime based on market data.
        
        Args:
            market_data: Dict of market metrics
            
        Returns:
            String describing the primary regime
        """
        vix = market_data.get("vix", 15.0)
        trend_strength = market_data.get("trend_strength", 0.5)
        trend_direction = market_data.get("trend_direction", 0.0)
        
        # Simple rule-based classification
        if vix > 25:
            return "volatile"
        elif trend_strength > 0.7 and trend_direction > 0.5:
            return "bullish"
        elif trend_strength > 0.7 and trend_direction < -0.5:
            return "bearish"
        elif trend_strength < 0.3:
            return "sideways"
        elif trend_direction > 0.3:
            return "moderately_bullish"
        elif trend_direction < -0.3:
            return "moderately_bearish"
        else:
            return "neutral"
    
    def _detect_secondary_characteristics(self, market_data: Dict[str, Any]) -> List[str]:
        """
        Detect secondary market characteristics that overlay with the primary regime.
        
        Args:
            market_data: Dict of market metrics
            
        Returns:
            List of strings describing secondary characteristics
        """
        characteristics = []
        
        # Check for mean-reverting behavior
        if market_data.get("momentum", 0.0) < -0.5:
            characteristics.append("mean_reverting")
            
        # Check for rotational behavior
        if market_data.get("sector_dispersion", 0.0) > 0.7:
            characteristics.append("rotational")
            
        # Check for trend-following behavior
        if market_data.get("trend_strength", 0.0) > 0.8:
            characteristics.append("strongly_trending")
            
        # Check for range-bound behavior
        if (market_data.get("trend_strength", 0.5) < 0.3 and
                market_data.get("vix", 20) < 18):
            characteristics.append("range_bound")
        
        return characteristics
    
    def _calculate_confidence_scores(self, market_data: Dict[str, Any], 
                                    primary_regime: str) -> Dict[str, float]:
        """
        Calculate confidence scores for each possible regime classification.
        
        Args:
            market_data: Dict of market metrics
            primary_regime: The detected primary regime
            
        Returns:
            Dict of confidence scores for each regime
        """
        # Default confidence values
        confidence_scores = {
            "bullish": 0.0,
            "bearish": 0.0,
            "volatile": 0.0,
            "sideways": 0.0,
            "neutral": 0.0,
            "moderately_bullish": 0.0,
            "moderately_bearish": 0.0
        }
        
        # Calculate individual confidence values based on market data
        vix = market_data.get("vix", 15.0)
        vix_percentile = market_data.get("vix_percentile", 50.0)
        trend_strength = market_data.get("trend_strength", 0.5)
        trend_direction = market_data.get("trend_direction", 0.0)
        
        # Volatile regime confidence
        confidence_scores["volatile"] = min(vix / 40, 1.0) * 0.6 + vix_percentile / 100 * 0.4
        
        # Bullish regime confidence
        confidence_scores["bullish"] = (
            max(min(trend_direction, 1.0), 0.0) * 0.5 + 
            max(min(trend_strength, 1.0), 0.0) * 0.3 +
            (1.0 - min(vix / 30, 1.0)) * 0.2
        )
        
        # Bearish regime confidence
        confidence_scores["bearish"] = (
            max(min(-trend_direction, 1.0), 0.0) * 0.5 + 
            max(min(trend_strength, 1.0), 0.0) * 0.3 +
            min(vix / 25, 1.0) * 0.2
        )
        
        # Sideways regime confidence
        confidence_scores["sideways"] = (
            (1.0 - min(trend_strength * 2, 1.0)) * 0.7 +
            (1.0 - min(abs(trend_direction) * 2, 1.0)) * 0.3
        )
        
        # Moderately bullish confidence
        confidence_scores["moderately_bullish"] = (
            max(min(trend_direction * 1.5, 1.0), 0.0) * 0.6 +
            (1.0 - max(min(trend_strength * 1.5, 1.0), 0.0)) * 0.4
        )
        
        # Moderately bearish confidence
        confidence_scores["moderately_bearish"] = (
            max(min(-trend_direction * 1.5, 1.0), 0.0) * 0.6 +
            (1.0 - max(min(trend_strength * 1.5, 1.0), 0.0)) * 0.4
        )
        
        # Neutral confidence (when no clear regime)
        confidence_scores["neutral"] = (
            (1.0 - min(abs(trend_direction) * 2.5, 1.0)) * 0.5 +
            (1.0 - min(trend_strength * 2.5, 1.0)) * 0.3 +
            (1.0 - min(vix / 20, 1.0)) * 0.2
        )
        
        # Normalize confidence scores to ensure they sum to 1.0
        total = sum(confidence_scores.values()) + 1e-10  # Avoid division by zero
        confidence_scores = {k: v / total for k, v in confidence_scores.items()}
        
        return confidence_scores


class RegimeAwareStrategyPrioritizer:
    """
    Integrates market regime classification with strategy prioritization to deliver
    more context-aware strategy allocations based on detected market regimes.
    """
    
    def __init__(self, 
                strategies: List[str],
                use_mock: bool = False,
                regime_lookback_days: int = 60,
                cache_dir: str = None,
                api_key: str = None):
        """
        Initialize the RegimeAwareStrategyPrioritizer.
        
        Args:
            strategies: List of strategy names to prioritize
            use_mock: Whether to use mock responses instead of calling LLM API
            regime_lookback_days: Number of days to look back for regime classification
            cache_dir: Directory to cache responses
            api_key: API key for LLM service (defaults to environment variable)
        """
        self.strategies = strategies
        self.use_mock = use_mock
        
        # Initialize regime classifier
        self.regime_classifier = RegimeClassifier(lookback_days=regime_lookback_days)
        
        # Initialize base strategy prioritizer
        self.strategy_prioritizer = StrategyPrioritizer(
            strategies=strategies,
            use_mock=use_mock,
            cache_dir=cache_dir,
            api_key=api_key
        )
        
        # Initialize regime-specific weightings for strategies
        self._init_regime_weightings()
        
        logger.info(f"RegimeAwareStrategyPrioritizer initialized with {len(strategies)} strategies")
        if use_mock:
            logger.info("Using mock responses for strategy prioritization")
    
    def _init_regime_weightings(self):
        """
        Initialize regime-specific weightings for each strategy to be used
        as prior biases in the prioritization process.
        """
        # These weightings represent prior biases for each strategy in different regimes
        # They can be adjusted based on historical performance or domain knowledge
        self.regime_weightings = {
            "bullish": {
                "momentum": 0.9,
                "trend_following": 0.85,
                "breakout_swing": 0.75,
                "option_spreads": 0.5,
                "mean_reversion": 0.3,
                "volatility_breakout": 0.4
            },
            "bearish": {
                "momentum": 0.3,
                "trend_following": 0.7,
                "breakout_swing": 0.5,
                "option_spreads": 0.6,
                "mean_reversion": 0.4,
                "volatility_breakout": 0.7
            },
            "volatile": {
                "momentum": 0.2,
                "trend_following": 0.3,
                "breakout_swing": 0.5,
                "option_spreads": 0.8,
                "mean_reversion": 0.7,
                "volatility_breakout": 0.9
            },
            "sideways": {
                "momentum": 0.3,
                "trend_following": 0.2,
                "breakout_swing": 0.5,
                "option_spreads": 0.7,
                "mean_reversion": 0.9,
                "volatility_breakout": 0.5
            },
            "neutral": {
                "momentum": 0.5,
                "trend_following": 0.5,
                "breakout_swing": 0.6,
                "option_spreads": 0.6,
                "mean_reversion": 0.7,
                "volatility_breakout": 0.5
            },
            "moderately_bullish": {
                "momentum": 0.8,
                "trend_following": 0.7,
                "breakout_swing": 0.7,
                "option_spreads": 0.5,
                "mean_reversion": 0.5,
                "volatility_breakout": 0.4
            },
            "moderately_bearish": {
                "momentum": 0.4,
                "trend_following": 0.6,
                "breakout_swing": 0.5,
                "option_spreads": 0.6,
                "mean_reversion": 0.5,
                "volatility_breakout": 0.6
            }
        }
        
        # Initialize with default weightings for any strategy not explicitly listed
        default_weighting = 0.5
        for regime in self.regime_weightings:
            for strategy in self.strategies:
                if strategy not in self.regime_weightings[regime]:
                    self.regime_weightings[regime][strategy] = default_weighting
    
    def get_regime_classification(self) -> Dict[str, Any]:
        """
        Get the current market regime classification.
        
        Returns:
            Dict containing regime classification and confidence scores
        """
        return self.regime_classifier.classify_regime()
    
    def prioritize_strategies(self, 
                             market_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prioritize strategies based on market context and detected regime.
        
        Args:
            market_context: Optional market context data (fetched if not provided)
            
        Returns:
            Dict containing prioritized strategies with allocations and reasoning
        """
        # Get market context if not provided
        if market_context is None:
            market_fetcher = MarketContextFetcher()
            market_context = market_fetcher.get_market_context()
        
        # Get regime classification
        regime_data = self.get_regime_classification()
        primary_regime = regime_data["primary_regime"]
        confidence_scores = regime_data["confidence_scores"]
        
        # Enhanced market context with regime information
        enhanced_context = market_context.copy()
        enhanced_context.update({
            "detected_regime": primary_regime,
            "regime_confidence": confidence_scores[primary_regime],
            "regime_data": regime_data
        })
        
        # Get base prioritization from strategy prioritizer
        prioritization = self.strategy_prioritizer.get_strategy_allocation(
            market_context=enhanced_context
        )
        
        # Apply regime-based adjustments if not using mock data
        if not self.use_mock:
            prioritization = self._apply_regime_adjustments(
                prioritization, 
                primary_regime, 
                confidence_scores
            )
            
        return prioritization
    
    def _apply_regime_adjustments(self, 
                                 prioritization: Dict[str, Any],
                                 primary_regime: str,
                                 confidence_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Apply regime-based adjustments to the strategy prioritization.
        
        Args:
            prioritization: Base prioritization from strategy prioritizer
            primary_regime: Detected primary market regime
            confidence_scores: Confidence scores for each regime
            
        Returns:
            Updated prioritization with regime-based adjustments
        """
        # Get allocations from base prioritization
        allocations = prioritization.get("allocations", {})
        if not allocations:
            logger.warning("No allocations found in base prioritization")
            return prioritization
        
        # Apply regime weightings as a prior bias
        weighted_allocations = allocations.copy()
        
        # Calculate weighted average of regime biases based on confidence scores
        for strategy in self.strategies:
            regime_weight = 0.0
            for regime, confidence in confidence_scores.items():
                if regime in self.regime_weightings and strategy in self.regime_weightings[regime]:
                    regime_weight += self.regime_weightings[regime][strategy] * confidence
            
            # Apply regime weighting as a bias (adjust this factor as needed)
            bias_factor = 0.3  # How strongly the regime bias affects the final allocation
            if strategy in weighted_allocations:
                original_allocation = float(weighted_allocations[strategy])
                weighted_allocations[strategy] = original_allocation * (1 - bias_factor) + \
                                               regime_weight * bias_factor * original_allocation * 2
        
        # Normalize allocations to sum to 100%
        total_allocation = sum(float(v) for v in weighted_allocations.values())
        if total_allocation > 0:
            normalized_allocations = {k: float(v) / total_allocation * 100 
                                    for k, v in weighted_allocations.items()}
            
            # Update allocations in prioritization result
            prioritization["allocations"] = normalized_allocations
            
            # Add regime adjustment explanation
            prioritization["regime_adjustment_explanation"] = (
                f"Allocations adjusted based on detected {primary_regime} regime "
                f"with {confidence_scores[primary_regime]:.2f} confidence."
            )
        
        return prioritization
    
    def get_regime_weighted_allocation(self) -> Dict[str, Dict[str, float]]:
        """
        Get weightings for each strategy across different regimes.
        Useful for visualization and analysis.
        
        Returns:
            Dict mapping regimes to strategy weightings
        """
        return self.regime_weightings
    
    def update_regime_weightings(self, 
                                new_weightings: Dict[str, Dict[str, float]]) -> None:
        """
        Update the regime-specific weightings based on new data or insights.
        
        Args:
            new_weightings: Dict mapping regimes to strategy weightings
        """
        for regime, strategy_weights in new_weightings.items():
            if regime in self.regime_weightings:
                for strategy, weight in strategy_weights.items():
                    if strategy in self.strategies:
                        self.regime_weightings[regime][strategy] = weight
        
        logger.info("Regime weightings updated")


# Example usage
if __name__ == "__main__":
    # List of strategies to prioritize
    strategies = [
        "momentum", 
        "trend_following", 
        "breakout_swing", 
        "mean_reversion", 
        "volatility_breakout", 
        "option_spreads"
    ]
    
    # Initialize RegimeAwareStrategyPrioritizer
    prioritizer = RegimeAwareStrategyPrioritizer(
        strategies=strategies,
        use_mock=True,  # Use mock for testing
        regime_lookback_days=60
    )
    
    # Get regime classification
    regime_data = prioritizer.get_regime_classification()
    print(f"Detected regime: {regime_data['primary_regime']}")
    print(f"Confidence: {regime_data['confidence_scores'][regime_data['primary_regime']]:.2f}")
    print(f"Secondary characteristics: {regime_data['secondary_characteristics']}")
    
    # Prioritize strategies
    prioritization = prioritizer.prioritize_strategies()
    
    # Print results
    print("\nStrategy Allocations:")
    for strategy, allocation in prioritization["allocations"].items():
        print(f"{strategy}: {float(allocation):.2f}%")
    
    print("\nAI Reasoning:")
    print(prioritization.get("reasoning", "No reasoning provided")) 