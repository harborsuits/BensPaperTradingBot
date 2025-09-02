#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RegimeAwareStrategyPrioritizer integrates market regime classification 
with strategy prioritization for more accurate strategy allocation.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import joblib
from pathlib import Path
import os
import json
from datetime import datetime, timedelta

from trading_bot.ai_scoring.strategy_prioritizer import StrategyPrioritizer
from trading_bot.utils.market_context_fetcher import MarketContextFetcher

# Configure logging
logger = logging.getLogger(__name__)

class RegimeClassifier:
    """
    Market regime classifier using statistical and machine learning methods
    to determine the current market regime.
    """
    
    REGIMES = ["bullish", "bearish", "sideways", "volatile"]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_statistical: bool = True,
        lookback_period: int = 60,
        threshold_volatility: float = 20.0,
        threshold_trend: float = 5.0
    ):
        """
        Initialize the RegimeClassifier.
        
        Args:
            model_path: Path to a saved model file (optional)
            use_statistical: Whether to use statistical methods for classification
            lookback_period: Number of days to use for trend and volatility calculation
            threshold_volatility: VIX threshold for volatility regime
            threshold_trend: Trend strength threshold for bullish/bearish regime
        """
        self.model = self._load_model(model_path) if model_path else None
        self.use_statistical = use_statistical
        self.lookback_period = lookback_period
        self.threshold_volatility = threshold_volatility
        self.threshold_trend = threshold_trend
        
        # Feature importance for regimes if using statistical approach
        self.regime_features = {
            "bullish": ["trend_strength", "atr_ratio", "vix"],
            "bearish": ["trend_strength", "atr_ratio", "vix"],
            "volatile": ["vix", "atr_ratio", "trend_strength"],
            "sideways": ["vix", "trend_strength", "atr_ratio"]
        }
        
        # Feature weights by regime (importance)
        self.feature_weights = {
            "bullish": {"trend_strength": 0.6, "atr_ratio": 0.2, "vix": 0.2},
            "bearish": {"trend_strength": 0.6, "atr_ratio": 0.2, "vix": 0.2},
            "volatile": {"vix": 0.7, "atr_ratio": 0.2, "trend_strength": 0.1},
            "sideways": {"vix": 0.4, "trend_strength": 0.4, "atr_ratio": 0.2}
        }
        
        logger.info("RegimeClassifier initialized")
    
    def _load_model(self, model_path: str) -> Any:
        """
        Load a saved machine learning model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return None
    
    def _calculate_statistical_probabilities(
        self, 
        market_indicators: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate regime probabilities using simple statistical rules.
        
        Args:
            market_indicators: Dictionary of market indicators
            
        Returns:
            Dictionary of regime probabilities
        """
        vix = market_indicators.get("vix", 15.0)
        trend_strength = market_indicators.get("trend_strength", 0.0)
        atr_ratio = market_indicators.get("atr_ratio", 1.0)
        
        # Initialize probabilities
        probabilities = {regime: 0.0 for regime in self.REGIMES}
        
        # Volatility regime
        if vix >= self.threshold_volatility:
            volatility_score = min((vix - self.threshold_volatility) / 20.0 + 0.5, 1.0)
            probabilities["volatile"] += volatility_score * 0.7
        else:
            volatility_score = 1.0 - (vix / self.threshold_volatility * 0.8)
            
            # Trend-based regimes
            if trend_strength >= self.threshold_trend:
                trend_score = min((trend_strength - self.threshold_trend) / 10.0 + 0.5, 1.0)
                probabilities["bullish"] += trend_score * 0.8
            elif trend_strength <= -self.threshold_trend:
                trend_score = min((abs(trend_strength) - self.threshold_trend) / 10.0 + 0.5, 1.0)
                probabilities["bearish"] += trend_score * 0.8
            else:
                # Sideways regime
                sideways_score = 1.0 - (abs(trend_strength) / self.threshold_trend * 0.7)
                probabilities["sideways"] += sideways_score * volatility_score * 0.9
        
        # Use ATR ratio to further refine
        if atr_ratio >= 1.5:
            # Higher ATR suggests more volatility or stronger trends
            if probabilities["volatile"] > 0:
                probabilities["volatile"] += 0.2 * (atr_ratio - 1.0)
            elif probabilities["bullish"] > 0 or probabilities["bearish"] > 0:
                trend_regime = "bullish" if probabilities["bullish"] > probabilities["bearish"] else "bearish"
                probabilities[trend_regime] += 0.2 * (atr_ratio - 1.0)
        elif atr_ratio <= 0.7:
            # Lower ATR suggests more sideways action
            probabilities["sideways"] += 0.3 * (1.0 - atr_ratio)
        
        # Normalize probabilities
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {regime: prob / total for regime, prob in probabilities.items()}
        else:
            # Default to sideways if no clear signal
            probabilities["sideways"] = 1.0
        
        return probabilities
    
    def _predict_using_model(
        self, 
        market_indicators: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Predict regime probabilities using the trained model.
        
        Args:
            market_indicators: Dictionary of market indicators
            
        Returns:
            Dictionary of regime probabilities
        """
        if not self.model:
            logger.warning("No model loaded, falling back to statistical method")
            return self._calculate_statistical_probabilities(market_indicators)
        
        try:
            # Extract features used by the model
            features = pd.DataFrame([market_indicators])
            
            # Make prediction
            if hasattr(self.model, "predict_proba"):
                # For probabilistic models
                probas = self.model.predict_proba(features)
                return {regime: float(proba) for regime, proba in zip(self.REGIMES, probas[0])}
            else:
                # For non-probabilistic models, make a hard prediction
                prediction = self.model.predict(features)[0]
                return {regime: 1.0 if regime == prediction else 0.0 for regime in self.REGIMES}
        
        except Exception as e:
            logger.error(f"Error predicting with model: {e}")
            logger.info("Falling back to statistical method")
            return self._calculate_statistical_probabilities(market_indicators)
    
    def classify_regime(
        self, 
        market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify the current market regime based on market context.
        
        Args:
            market_context: Dictionary with market context
            
        Returns:
            Dictionary with regime classification and probabilities
        """
        market_indicators = market_context.get("market_indicators", {})
        
        # Get probabilities either statistically or via model
        if self.use_statistical or not self.model:
            probabilities = self._calculate_statistical_probabilities(market_indicators)
        else:
            probabilities = self._predict_using_model(market_indicators)
        
        # Get the most likely regime
        primary_regime = max(probabilities, key=probabilities.get)
        primary_probability = probabilities[primary_regime]
        
        # Get the second most likely regime
        probabilities_without_primary = {r: p for r, p in probabilities.items() if r != primary_regime}
        secondary_regime = max(probabilities_without_primary, key=probabilities_without_primary.get)
        secondary_probability = probabilities[secondary_regime]
        
        # Determine if we're in a mixed regime
        is_mixed = secondary_probability > 0.3 and primary_probability < 0.6
        
        # Calculate confidence score
        confidence = primary_probability
        if is_mixed:
            confidence = primary_probability - secondary_probability
        
        # Return classification with additional info
        result = {
            "primary_regime": primary_regime,
            "primary_probability": primary_probability,
            "secondary_regime": secondary_regime,
            "secondary_probability": secondary_probability,
            "is_mixed": is_mixed,
            "confidence": confidence,
            "all_probabilities": probabilities,
            "classification_method": "statistical" if self.use_statistical or not self.model else "model",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Regime classification: {primary_regime} (prob: {primary_probability:.2f})")
        if is_mixed:
            logger.info(f"Mixed regime detected: {primary_regime}/{secondary_regime}")
        
        return result


class RegimeAwareStrategyPrioritizer:
    """
    Integrates market regime classification with strategy prioritization
    for more accurate and tailored strategy allocations.
    """
    
    def __init__(
        self,
        strategies: List[str],
        api_key: Optional[str] = None,
        model_path: Optional[str] = None,
        use_statistical_classification: bool = True,
        cache_results: bool = True,
        cache_dir: Optional[str] = None,
        override_api_type: Optional[str] = None
    ):
        """
        Initialize the RegimeAwareStrategyPrioritizer.
        
        Args:
            strategies: List of strategy names
            api_key: API key for LLM service
            model_path: Path to a saved regime classification model
            use_statistical_classification: Whether to use statistical regime classification
            cache_results: Whether to cache results
            cache_dir: Directory to store cached results
            override_api_type: Override the API type for StrategyPrioritizer
        """
        self.strategies = strategies
        self.cache_dir = cache_dir or Path.home() / ".trading_bot" / "cache"
        
        # Initialize the regime classifier
        self.regime_classifier = RegimeClassifier(
            model_path=model_path,
            use_statistical=use_statistical_classification
        )
        
        # Initialize the strategy prioritizer
        self.strategy_prioritizer = StrategyPrioritizer(
            strategies=strategies,
            api_key=api_key,
            cache_results=cache_results,
            cache_dir=str(self.cache_dir),
            override_api_type=override_api_type
        )
        
        # Initialize market context fetcher
        self.market_fetcher = MarketContextFetcher()
        
        # Strategy performance adjustments based on regime
        self.regime_strategy_adjustments = {
            "bullish": {
                "momentum": 1.3,
                "trend_following": 1.2,
                "breakout_swing": 1.2,
                "mean_reversion": 0.8,
                "volatility_breakout": 0.7,
                "option_spreads": 0.9
            },
            "bearish": {
                "momentum": 0.7,
                "trend_following": 0.8,
                "breakout_swing": 0.7,
                "mean_reversion": 1.1,
                "volatility_breakout": 1.0,
                "option_spreads": 1.2
            },
            "volatile": {
                "momentum": 0.6,
                "trend_following": 0.7,
                "breakout_swing": 0.8,
                "mean_reversion": 1.0,
                "volatility_breakout": 1.4,
                "option_spreads": 1.3
            },
            "sideways": {
                "momentum": 0.7,
                "trend_following": 0.6,
                "breakout_swing": 0.9,
                "mean_reversion": 1.3,
                "volatility_breakout": 0.8,
                "option_spreads": 1.0
            }
        }
        
        logger.info(f"RegimeAwareStrategyPrioritizer initialized with {len(strategies)} strategies")
    
    def _apply_regime_adjustments(
        self, 
        allocations: Dict[str, float], 
        regime_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Apply regime-specific adjustments to strategy allocations.
        
        Args:
            allocations: Dictionary of strategy allocations
            regime_data: Dictionary with regime classification data
            
        Returns:
            Dictionary of adjusted strategy allocations
        """
        primary_regime = regime_data["primary_regime"]
        primary_prob = regime_data["primary_probability"]
        secondary_regime = regime_data["secondary_regime"]
        secondary_prob = regime_data["secondary_probability"]
        
        # Get adjustment multipliers
        primary_adjustments = self.regime_strategy_adjustments.get(primary_regime, {})
        secondary_adjustments = self.regime_strategy_adjustments.get(secondary_regime, {})
        
        # Apply weighted adjustments
        adjusted_allocations = {}
        for strategy, allocation in allocations.items():
            primary_adj = primary_adjustments.get(strategy, 1.0)
            secondary_adj = secondary_adjustments.get(strategy, 1.0)
            
            # Calculate weighted adjustment factor
            if regime_data["is_mixed"]:
                # Weight by probabilities for mixed regimes
                adj_factor = (primary_adj * primary_prob + secondary_adj * secondary_prob) / (primary_prob + secondary_prob)
            else:
                # Primarily use the main regime adjustment
                adj_factor = primary_adj
            
            # Apply adjustment
            adjusted_allocations[strategy] = allocation * adj_factor
        
        # Normalize to ensure allocations sum to 100%
        total_allocation = sum(adjusted_allocations.values())
        normalized_allocations = {
            strategy: (alloc / total_allocation) * 100 if total_allocation > 0 else 0.0
            for strategy, alloc in adjusted_allocations.items()
        }
        
        return normalized_allocations
    
    async def get_prioritized_strategies(
        self, 
        market_context: Optional[Dict[str, Any]] = None,
        use_cached: bool = True
    ) -> Dict[str, Any]:
        """
        Get prioritized strategies with regime awareness.
        
        Args:
            market_context: Dictionary with market context (optional)
            use_cached: Whether to use cached results if available
            
        Returns:
            Dictionary with prioritized strategies and regime information
        """
        # Fetch market context if not provided
        if not market_context:
            market_context = await self.market_fetcher.get_market_context()
        
        # Classify the market regime
        regime_data = self.regime_classifier.classify_regime(market_context)
        
        # Adjust market context with regime classification
        enhanced_context = market_context.copy()
        enhanced_context["regime_classification"] = regime_data
        
        # Use strategy prioritizer to get initial allocations
        prioritization_result = await self.strategy_prioritizer.get_prioritized_strategies(
            market_context=enhanced_context,
            use_cached=use_cached
        )
        
        # Extract strategy allocations
        strategy_allocations = prioritization_result.get("strategy_allocations", {})
        
        # Apply regime-specific adjustments
        adjusted_allocations = self._apply_regime_adjustments(
            strategy_allocations,
            regime_data
        )
        
        # Prepare result including regime information
        result = {
            "strategy_allocations": adjusted_allocations,
            "strategy_reasoning": prioritization_result.get("strategy_reasoning", {}),
            "market_summary": prioritization_result.get("market_summary", ""),
            "regime_data": regime_data,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def get_strategy_allocation(
        self, 
        strategy_name: str, 
        allocations: Dict[str, float]
    ) -> float:
        """
        Get the allocation for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            allocations: Dictionary of strategy allocations
            
        Returns:
            Allocation percentage for the strategy
        """
        return float(allocations.get(strategy_name, 0.0))
    
    def get_mock_prioritization(
        self, 
        market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get mock prioritization based on market context and regime.
        
        Args:
            market_context: Dictionary with market context
            
        Returns:
            Dictionary with mock prioritized strategies
        """
        # Classify regime
        regime_data = self.regime_classifier.classify_regime(market_context)
        primary_regime = regime_data["primary_regime"]
        
        # Get mock result from base prioritizer
        mock_result = self.strategy_prioritizer._get_mock_prioritization(market_context)
        
        # Apply regime-specific adjustments
        mock_allocations = mock_result.get("strategy_allocations", {})
        adjusted_allocations = self._apply_regime_adjustments(
            mock_allocations,
            regime_data
        )
        
        # Update result with adjusted allocations and regime data
        mock_result["strategy_allocations"] = adjusted_allocations
        mock_result["regime_data"] = regime_data
        
        return mock_result
    
    async def evaluate_regime_strategy_fit(
        self, 
        strategy_name: str,
        market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate how well a strategy fits with the current market regime.
        
        Args:
            strategy_name: Name of the strategy to evaluate
            market_context: Dictionary with market context
            
        Returns:
            Dictionary with evaluation results
        """
        # Classify regime
        regime_data = self.regime_classifier.classify_regime(market_context)
        primary_regime = regime_data["primary_regime"]
        
        # Get regime adjustment factor
        adjustment_factor = self.regime_strategy_adjustments.get(primary_regime, {}).get(strategy_name, 1.0)
        
        # Determine fit level
        if adjustment_factor >= 1.2:
            fit_level = "excellent"
        elif adjustment_factor >= 1.0:
            fit_level = "good"
        elif adjustment_factor >= 0.8:
            fit_level = "moderate"
        else:
            fit_level = "poor"
        
        return {
            "strategy": strategy_name,
            "regime": primary_regime,
            "fit_level": fit_level,
            "adjustment_factor": adjustment_factor,
            "is_recommended": adjustment_factor >= 1.0,
            "regime_data": regime_data
        }
    
    def save_regime_history(
        self, 
        regime_data: Dict[str, Any],
        path: Optional[str] = None
    ) -> str:
        """
        Save regime classification history to a file.
        
        Args:
            regime_data: Dictionary with regime classification data
            path: Path to save the history (optional)
            
        Returns:
            Path where the history was saved
        """
        history_path = path or str(self.cache_dir / "regime_history.json")
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        
        # Load existing history if available
        history = []
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
            except Exception as e:
                logger.error(f"Error loading regime history: {e}")
                history = []
        
        # Add new entry with timestamp
        entry = {
            "timestamp": datetime.now().isoformat(),
            **regime_data
        }
        history.append(entry)
        
        # Save updated history
        try:
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            logger.info(f"Regime history saved to {history_path}")
            return history_path
        except Exception as e:
            logger.error(f"Error saving regime history: {e}")
            return ""

if __name__ == "__main__":
    # Example usage
    import asyncio
    import os
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def demo():
        strategies = [
            "momentum", "trend_following", "mean_reversion",
            "breakout_swing", "volatility_breakout", "option_spreads"
        ]
        
        market_fetcher = MarketContextFetcher()
        market_context = await market_fetcher.get_market_context(use_mock=True)
        
        # Create a regime aware prioritizer
        prioritizer = RegimeAwareStrategyPrioritizer(
            strategies=strategies,
            use_statistical_classification=True
        )
        
        # Get prioritized strategies
        result = await prioritizer.get_prioritized_strategies(
            market_context=market_context
        )
        
        print("\n=== Market Regime ===")
        regime_data = result["regime_data"]
        print(f"Primary Regime: {regime_data['primary_regime']} (Probability: {regime_data['primary_probability']:.2f})")
        if regime_data["is_mixed"]:
            print(f"Secondary Regime: {regime_data['secondary_regime']} (Probability: {regime_data['secondary_probability']:.2f})")
        print(f"Confidence: {regime_data['confidence']:.2f}")
        
        print("\n=== Strategy Allocations ===")
        allocations = result["strategy_allocations"]
        for strategy, allocation in sorted(allocations.items(), key=lambda x: x[1], reverse=True):
            print(f"{strategy}: {allocation:.2f}%")
        
        print("\n=== Strategy Reasoning ===")
        reasoning = result["strategy_reasoning"]
        for strategy, reason in reasoning.items():
            print(f"{strategy}: {reason}")
        
        print("\n=== Market Summary ===")
        print(result["market_summary"])
        
        # Evaluate fit for each strategy
        print("\n=== Strategy Fit for Current Regime ===")
        for strategy in strategies:
            fit_result = await prioritizer.evaluate_regime_strategy_fit(
                strategy_name=strategy,
                market_context=market_context
            )
            print(f"{strategy}: {fit_result['fit_level']} fit for {fit_result['regime']} regime " +
                  f"(Factor: {fit_result['adjustment_factor']:.2f})")
    
    asyncio.run(demo()) 