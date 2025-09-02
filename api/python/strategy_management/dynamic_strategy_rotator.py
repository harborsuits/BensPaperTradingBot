import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
from datetime import datetime, timedelta
import json
import os
import time
import random

from trading_bot.core.event_system import EventListener, Event
from .interfaces import StrategyRotator, MarketContextProvider, MarketRegimeClassifier

logger = logging.getLogger(__name__)

class StrategyProfile:
    """
    Represents a trading strategy profile with performance metrics and
    activation status.
    """
    
    def __init__(self, 
                strategy_id: str,
                name: str,
                description: str = "",
                is_active: bool = False,
                regime_suitability: Dict[str, float] = None,
                max_allocation: float = 1.0):
        """
        Initialize a strategy profile
        
        Args:
            strategy_id: Unique identifier for the strategy
            name: Human-readable name
            description: Strategy description
            is_active: Whether the strategy is currently active
            regime_suitability: Dict mapping market regimes to suitability scores (0-1)
            max_allocation: Maximum capital allocation for this strategy (0-1)
        """
        self.strategy_id = strategy_id
        self.name = name
        self.description = description
        self.is_active = is_active
        self.regime_suitability = regime_suitability or {}
        self.max_allocation = max_allocation
        
        # Performance metrics
        self.performance_metrics = {
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "win_rate": None,
            "profit_factor": None,
            "max_drawdown": None,
            "avg_daily_return": None,
            "total_return": None,
            "volatility": None
        }
        
        # Historical performance
        self.historical_performance = {
            "daily_returns": [],
            "monthly_returns": [],
            "rolling_sharpe": []
        }
        
        # Rotation history
        self.activation_history = []
        
        # Last updated timestamp
        self.last_updated = datetime.now()
        
    def activate(self):
        """Activate the strategy"""
        if not self.is_active:
            self.is_active = True
            self.activation_history.append({
                "action": "activate",
                "timestamp": datetime.now().isoformat()
            })
            logger.info(f"Strategy {self.strategy_id} activated")
            
    def deactivate(self):
        """Deactivate the strategy"""
        if self.is_active:
            self.is_active = False
            self.activation_history.append({
                "action": "deactivate",
                "timestamp": datetime.now().isoformat()
            })
            logger.info(f"Strategy {self.strategy_id} deactivated")
            
    def update_performance_metrics(self, metrics: Dict[str, float]):
        """
        Update performance metrics
        
        Args:
            metrics: Dictionary of performance metrics
        """
        for key, value in metrics.items():
            if key in self.performance_metrics:
                self.performance_metrics[key] = value
                
        self.last_updated = datetime.now()
        
    def update_historical_performance(self, history_type: str, data: List[Dict[str, Any]]):
        """
        Update historical performance data
        
        Args:
            history_type: Type of history (daily_returns, monthly_returns, rolling_sharpe)
            data: Historical data
        """
        if history_type in self.historical_performance:
            self.historical_performance[history_type] = data
            
        self.last_updated = datetime.now()
        
    def calculate_regime_score(self, current_regime: str) -> float:
        """
        Calculate the suitability score for the current market regime
        
        Args:
            current_regime: Current market regime
            
        Returns:
            Suitability score between 0 and 1
        """
        # Default score if regime unknown
        default_score = 0.5
        
        # Get suitability score for this regime
        return self.regime_suitability.get(current_regime, default_score)
        
    def calculate_performance_score(self) -> float:
        """
        Calculate a performance score based on metrics
        
        Returns:
            Performance score between 0 and 1
        """
        # Start with base score
        score = 0.5
        
        # Sharpe ratio contribution
        sharpe = self.performance_metrics.get("sharpe_ratio")
        if sharpe is not None:
            # Normalize: 0 = 0, 1 = 0.5, 2+ = 1.0
            sharpe_score = min(max(sharpe / 2, 0), 1)
            score += sharpe_score * 0.3  # 30% weight
            
        # Win rate contribution
        win_rate = self.performance_metrics.get("win_rate")
        if win_rate is not None:
            # Normalize: 0.4 = 0, 0.5 = 0.5, 0.6+ = 1.0
            win_rate_score = min(max((win_rate - 0.4) / 0.2, 0), 1)
            score += win_rate_score * 0.2  # 20% weight
            
        # Profit factor contribution
        profit_factor = self.performance_metrics.get("profit_factor")
        if profit_factor is not None:
            # Normalize: 1.0 = 0, 1.5 = 0.5, 2.0+ = 1.0
            pf_score = min(max((profit_factor - 1.0) / 1.0, 0), 1)
            score += pf_score * 0.2  # 20% weight
            
        # Drawdown contribution (negative)
        drawdown = self.performance_metrics.get("max_drawdown")
        if drawdown is not None:
            # Normalize: 0% = 1, 10% = 0.5, 20%+ = 0
            dd_score = min(max(1.0 - drawdown * 5, 0), 1)
            score += dd_score * 0.1  # 10% weight
            
        # Return score capped between 0 and 1
        return min(max(score, 0), 1)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert profile to dictionary for serialization
        
        Returns:
            Dictionary representation
        """
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "description": self.description,
            "is_active": self.is_active,
            "regime_suitability": self.regime_suitability,
            "max_allocation": self.max_allocation,
            "performance_metrics": self.performance_metrics,
            "historical_performance": self.historical_performance,
            "activation_history": self.activation_history,
            "last_updated": self.last_updated.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyProfile':
        """
        Create profile from dictionary data
        
        Args:
            data: Dictionary data
            
        Returns:
            StrategyProfile instance
        """
        profile = cls(
            strategy_id=data["strategy_id"],
            name=data["name"],
            description=data["description"],
            is_active=data["is_active"],
            regime_suitability=data["regime_suitability"],
            max_allocation=data["max_allocation"]
        )
        
        profile.performance_metrics = data["performance_metrics"]
        profile.historical_performance = data["historical_performance"]
        profile.activation_history = data["activation_history"]
        
        if "last_updated" in data:
            profile.last_updated = datetime.fromisoformat(data["last_updated"])
            
        return profile


class DynamicStrategyRotator(StrategyRotator):
    """
    Dynamic Strategy Rotator that selects the most appropriate trading strategy 
    based on current market context and historical performance
    """
    
    def __init__(self, 
                config_path: str = None,
                market_context_provider: MarketContextProvider = None,
                regime_classifier: MarketRegimeClassifier = None):
        """
        Initialize the Dynamic Strategy Rotator
        
        Args:
            config_path: Path to configuration file
            market_context_provider: Provider for market context data
            regime_classifier: Market regime classifier
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Dynamic Strategy Rotator")
        
        # Initialize configurations
        self.config = self._load_config(config_path) if config_path else {}
        
        # Component references
        self.market_context_provider = market_context_provider
        self.regime_classifier = regime_classifier
        
        # Strategy registry
        self.strategies = {}
        self.performance_profiles = {}
        self.regime_performance = {}
        
        # Selection history
        self.rotation_history = []
        
        # Settings
        self.selection_weights = self.config.get("selection_weights", {
            "recent_performance": 0.4,
            "regime_match": 0.35,
            "volatility_match": 0.15,
            "diversification": 0.1
        })
        
        # Load registered strategies if available
        self._load_registered_strategies()
        
        self.logger.info("Dynamic Strategy Rotator initialized with %d strategies", 
                        len(self.strategies))

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            config: Configuration dictionary
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"Config path {config_path} does not exist, using defaults")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            return {}

    def _load_registered_strategies(self) -> None:
        """Load previously registered strategies from storage"""
        storage_path = self.config.get("strategy_storage_path", "strategies.json")
        try:
            if os.path.exists(storage_path):
                with open(storage_path, 'r') as f:
                    data = json.load(f)
                    self.strategies = data.get("strategies", {})
                    self.performance_profiles = data.get("performance_profiles", {})
                    self.regime_performance = data.get("regime_performance", {})
                self.logger.info(f"Loaded {len(self.strategies)} strategies from storage")
            else:
                self.logger.info("No stored strategies found")
        except Exception as e:
            self.logger.error(f"Error loading strategies: {str(e)}")

    def _save_registered_strategies(self) -> None:
        """Save registered strategies to storage"""
        storage_path = self.config.get("strategy_storage_path", "strategies.json")
        try:
            data = {
                "strategies": self.strategies,
                "performance_profiles": self.performance_profiles,
                "regime_performance": self.regime_performance
            }
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)
            with open(storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Saved {len(self.strategies)} strategies to storage")
        except Exception as e:
            self.logger.error(f"Error saving strategies: {str(e)}")

    def register_strategy(self, 
                         strategy_id: str, 
                         performance_profile: Dict[str, Any]) -> bool:
        """
        Register a new strategy with its performance profile
        
        Args:
            strategy_id: Unique identifier for the strategy
            performance_profile: Performance profile of the strategy
            
        Returns:
            success: Whether registration was successful
        """
        try:
            if strategy_id in self.strategies:
                self.logger.warning(f"Strategy {strategy_id} already registered, updating profile")
            
            required_fields = ["avg_return", "sharpe_ratio", "drawdown", "volatility", 
                              "win_rate", "preferred_regimes"]
            
            for field in required_fields:
                if field not in performance_profile:
                    self.logger.error(f"Missing required field '{field}' in performance profile")
                    return False
            
            # Register the strategy
            self.strategies[strategy_id] = {
                "id": strategy_id,
                "registered_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "times_selected": 0
            }
            
            self.performance_profiles[strategy_id] = performance_profile
            
            # Initialize regime performance if not exists
            if strategy_id not in self.regime_performance:
                self.regime_performance[strategy_id] = {}
            
            # Save to storage
            self._save_registered_strategies()
            
            self.logger.info(f"Successfully registered strategy: {strategy_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering strategy {strategy_id}: {str(e)}")
            return False

    def select_strategy(self, context: Dict[str, Any]) -> str:
        """
        Select the most appropriate strategy based on market context
        
        Args:
            context: Current market context containing market data, indicators, etc.
            
        Returns:
            strategy_id: Identifier of the selected strategy
        """
        if not self.strategies:
            self.logger.warning("No strategies registered, cannot select strategy")
            return ""
        
        try:
            # Get current market regime if classifier is available
            current_regime = {}
            if self.regime_classifier:
                current_regime = self.regime_classifier.classify_regime(context)
                self.logger.info(f"Current market regime: {current_regime}")
            elif "market_regime" in context:
                current_regime = context.get("market_regime", {})
                self.logger.info(f"Using provided market regime: {current_regime}")
            else:
                self.logger.warning("No regime classifier or regime data available")
            
            # Calculate scores for each strategy
            scores = {}
            for strategy_id in self.strategies:
                score = self._calculate_strategy_score(strategy_id, context, current_regime)
                scores[strategy_id] = score
            
            # Select the strategy with the highest score
            if not scores:
                self.logger.warning("No scores calculated, using random selection")
                import random
                selected_strategy = random.choice(list(self.strategies.keys()))
            else:
                selected_strategy = max(scores.items(), key=lambda x: x[1])[0]
            
            # Update selection history
            self._record_selection(selected_strategy, scores, current_regime, context)
            
            self.logger.info(f"Selected strategy: {selected_strategy} with scores: {scores}")
            return selected_strategy
            
        except Exception as e:
            self.logger.error(f"Error selecting strategy: {str(e)}")
            # Fall back to most recently used strategy if available
            if self.rotation_history:
                return self.rotation_history[-1]["selected_strategy"]
            # Otherwise return the first strategy in the registry
            return next(iter(self.strategies.keys())) if self.strategies else ""

    def _calculate_strategy_score(self, 
                                strategy_id: str, 
                                context: Dict[str, Any], 
                                current_regime: Dict[str, float]) -> float:
        """
        Calculate a score for how well a strategy matches the current market context
        
        Args:
            strategy_id: Identifier of the strategy
            context: Current market context
            current_regime: Current market regime probabilities
            
        Returns:
            score: Strategy suitability score (0-100)
        """
        if strategy_id not in self.performance_profiles:
            self.logger.warning(f"No performance profile for strategy {strategy_id}")
            return 0.0
        
        profile = self.performance_profiles[strategy_id]
        weights = self.selection_weights
        score_components = {}
        
        # 1. Recent performance score (0-100)
        recent_performance = self._get_recent_performance(strategy_id)
        score_components["recent_performance"] = recent_performance * weights["recent_performance"]
        
        # 2. Regime match score (0-100)
        regime_match = self._calculate_regime_match(strategy_id, current_regime)
        score_components["regime_match"] = regime_match * weights["regime_match"]
        
        # 3. Volatility match score (0-100)
        current_volatility = context.get("market_volatility", 0.0)
        if isinstance(current_volatility, dict):
            current_volatility = current_volatility.get("value", 0.0)
            
        volatility_match = self._calculate_volatility_match(profile, current_volatility)
        score_components["volatility_match"] = volatility_match * weights["volatility_match"]
        
        # 4. Diversification score (higher if different from recently used strategies) (0-100)
        diversification = self._calculate_diversification_score(strategy_id)
        score_components["diversification"] = diversification * weights["diversification"]
        
        # Calculate final weighted score
        final_score = sum(score_components.values())
        
        self.logger.debug(f"Strategy {strategy_id} score components: {score_components}, final: {final_score}")
        return final_score

    def _get_recent_performance(self, strategy_id: str) -> float:
        """
        Get recent performance score for a strategy
        
        Args:
            strategy_id: Identifier of the strategy
            
        Returns:
            score: Performance score (0-100)
        """
        profile = self.performance_profiles.get(strategy_id, {})
        
        # Base score on sharpe ratio, scaling between 0-100
        sharpe = profile.get("sharpe_ratio", 0)
        # Convert sharpe ratio to a 0-100 scale (assuming sharpe of 3 is excellent)
        sharpe_score = min(100, max(0, sharpe / 3 * 100))
        
        # Factor in win rate
        win_rate = profile.get("win_rate", 0.5)
        win_rate_score = win_rate * 100
        
        # Factor in drawdown (lower is better)
        drawdown = abs(profile.get("drawdown", 0))
        drawdown_score = 100 - min(100, drawdown * 100)
        
        # Combine the scores
        return (sharpe_score * 0.5) + (win_rate_score * 0.3) + (drawdown_score * 0.2)

    def _calculate_regime_match(self, 
                              strategy_id: str, 
                              current_regime: Dict[str, float]) -> float:
        """
        Calculate how well a strategy matches the current market regime
        
        Args:
            strategy_id: Identifier of the strategy
            current_regime: Current market regime probabilities
            
        Returns:
            match_score: Regime match score (0-100)
        """
        if not current_regime:
            return 50.0  # Neutral score if no regime data
            
        profile = self.performance_profiles.get(strategy_id, {})
        preferred_regimes = profile.get("preferred_regimes", {})
        
        if not preferred_regimes:
            return 50.0  # Neutral score if no preferred regimes
        
        # Calculate a weighted match score
        match_score = 0.0
        total_regime_probability = sum(current_regime.values())
        
        if total_regime_probability == 0:
            return 50.0
        
        # Normalize regime probabilities
        normalized_regime = {k: v / total_regime_probability for k, v in current_regime.items()}
        
        for regime, probability in normalized_regime.items():
            # How well does this strategy perform in this regime (0-100 scale)
            regime_performance = preferred_regimes.get(regime, 50.0)
            match_score += probability * regime_performance
            
        return match_score

    def _calculate_volatility_match(self, 
                                  profile: Dict[str, Any], 
                                  current_volatility: float) -> float:
        """
        Calculate how well a strategy matches the current market volatility
        
        Args:
            profile: Strategy performance profile
            current_volatility: Current market volatility
            
        Returns:
            match_score: Volatility match score (0-100)
        """
        strategy_preferred_volatility = profile.get("preferred_volatility", 0.0)
        if strategy_preferred_volatility == 0:
            return 50.0  # Neutral score if no preferred volatility
            
        # Calculate match score based on the difference between current and preferred
        # The closer they are, the higher the score
        volatility_diff = abs(current_volatility - strategy_preferred_volatility)
        max_diff = 0.5  # Maximum reasonable difference in volatility
        
        # Scale to 0-100, where 100 means perfect match
        if volatility_diff >= max_diff:
            return 0.0
        else:
            return 100 * (1 - volatility_diff / max_diff)

    def _calculate_diversification_score(self, strategy_id: str) -> float:
        """
        Calculate diversification score to encourage rotation
        
        Args:
            strategy_id: Identifier of the strategy
            
        Returns:
            diversification_score: Diversification score (0-100)
        """
        # If no history, max diversification
        if not self.rotation_history:
            return 100.0
            
        # Look at last 5 selections
        recent_selections = [h["selected_strategy"] for h in self.rotation_history[-5:]]
        
        # Count occurrences of this strategy
        occurrences = recent_selections.count(strategy_id)
        
        # More occurrences = lower diversification score
        if occurrences == 0:
            return 100.0  # Max diversification for unused strategies
        else:
            return max(0, 100 - (occurrences * 20))  # Each occurrence reduces score by 20

    def _record_selection(self, 
                        selected_strategy: str, 
                        scores: Dict[str, float], 
                        regime: Dict[str, float], 
                        context: Dict[str, Any]) -> None:
        """
        Record a strategy selection in the history
        
        Args:
            selected_strategy: Identifier of the selected strategy
            scores: Scores for all strategies
            regime: Current market regime
            context: Market context at selection time
        """
        # Update strategy stats
        if selected_strategy in self.strategies:
            self.strategies[selected_strategy]["times_selected"] += 1
            self.strategies[selected_strategy]["last_selected"] = datetime.now().isoformat()
        
        # Record selection event
        selection_record = {
            "timestamp": datetime.now().isoformat(),
            "selected_strategy": selected_strategy,
            "scores": scores,
            "market_regime": regime,
            "volatility": context.get("market_volatility", 0.0),
            "context_summary": {
                "trend": context.get("market_trend", "unknown"),
                "sentiment": context.get("market_sentiment", "unknown"),
                "volatility": context.get("market_volatility", 0.0)
            }
        }
        
        self.rotation_history.append(selection_record)
        
        # Trim history if too long
        max_history = self.config.get("max_rotation_history", 1000)
        if len(self.rotation_history) > max_history:
            self.rotation_history = self.rotation_history[-max_history:]

    def update_strategy_performance(self, 
                                  strategy_id: str, 
                                  trade_results: List[Dict[str, Any]],
                                  current_regime: Dict[str, float]) -> None:
        """
        Update performance metrics for a specific strategy
        
        Args:
            strategy_id: Identifier of the strategy
            trade_results: Recent trade results
            current_regime: Current market regime probabilities
        """
        if strategy_id not in self.performance_profiles:
            self.logger.warning(f"Strategy {strategy_id} not in performance profiles, cannot update")
            return
            
        try:
            profile = self.performance_profiles[strategy_id]
            
            # Extract performance data from trade results
            if not trade_results:
                self.logger.warning(f"No trade results provided for {strategy_id}")
                return
                
            # Calculate metrics from trade results
            total_return = sum(tr.get("return_pct", 0.0) for tr in trade_results)
            win_count = sum(1 for tr in trade_results if tr.get("return_pct", 0.0) > 0)
            total_trades = len(trade_results)
            
            # Update the performance profile with exponential weighting
            # More weight to recent performance
            alpha = 0.3  # Weight for new data
            
            profile["avg_return"] = (profile.get("avg_return", 0.0) * (1 - alpha) + 
                                    (total_return / len(trade_results)) * alpha)
            
            profile["win_rate"] = (profile.get("win_rate", 0.5) * (1 - alpha) + 
                                 (win_count / total_trades if total_trades > 0 else 0.5) * alpha)
            
            # Update last modified
            profile["last_updated"] = datetime.now().isoformat()
            
            # Update regime-specific performance
            self._update_regime_performance(strategy_id, trade_results, current_regime)
            
            # Update strategy metadata
            self.strategies[strategy_id]["last_updated"] = datetime.now().isoformat()
            
            # Save changes
            self._save_registered_strategies()
            
            self.logger.info(f"Updated performance for strategy {strategy_id}")
            
        except Exception as e:
            self.logger.error(f"Error updating strategy performance for {strategy_id}: {str(e)}")

    def _update_regime_performance(self, 
                                 strategy_id: str, 
                                 trade_results: List[Dict[str, Any]],
                                 current_regime: Dict[str, float]) -> None:
        """
        Update regime-specific performance for a strategy
        
        Args:
            strategy_id: Identifier of the strategy
            trade_results: Recent trade results
            current_regime: Current market regime probabilities
        """
        if not current_regime:
            self.logger.debug(f"No regime data provided for {strategy_id}")
            return
            
        # Get the highest probability regime
        top_regime = max(current_regime.items(), key=lambda x: x[1])[0]
        
        # Initialize regime performance if not exists
        if strategy_id not in self.regime_performance:
            self.regime_performance[strategy_id] = {}
            
        if top_regime not in self.regime_performance[strategy_id]:
            self.regime_performance[strategy_id][top_regime] = {
                "avg_return": 0.0,
                "win_rate": 0.0,
                "trade_count": 0
            }
        
        # Calculate metrics from trade results
        if not trade_results:
            return
            
        total_return = sum(tr.get("return_pct", 0.0) for tr in trade_results)
        win_count = sum(1 for tr in trade_results if tr.get("return_pct", 0.0) > 0)
        total_trades = len(trade_results)
        
        # Get current regime stats
        regime_stats = self.regime_performance[strategy_id][top_regime]
        
        # Update with exponential weighting
        alpha = 0.3  # Weight for new data
        
        current_trade_count = regime_stats.get("trade_count", 0)
        new_trade_count = current_trade_count + total_trades
        
        # Update regime stats
        regime_stats["avg_return"] = (regime_stats.get("avg_return", 0.0) * (1 - alpha) + 
                                     (total_return / len(trade_results)) * alpha)
        
        regime_stats["win_rate"] = (regime_stats.get("win_rate", 0.0) * (1 - alpha) + 
                                  (win_count / total_trades if total_trades > 0 else 0.0) * alpha)
        
        regime_stats["trade_count"] = new_trade_count
        
        # Update performance profile's preferred regimes based on regime performance
        profile = self.performance_profiles[strategy_id]
        if "preferred_regimes" not in profile:
            profile["preferred_regimes"] = {}
            
        # Adjust preferred regime score (0-100 scale)
        # Higher avg_return and win_rate = higher score
        avg_return = regime_stats["avg_return"]
        win_rate = regime_stats["win_rate"]
        
        # Scale avg_return to 0-50 (assuming max reasonable return is 10%)
        return_score = min(50, max(0, avg_return / 0.1 * 50))
        
        # Scale win_rate to 0-50
        win_rate_score = min(50, max(0, win_rate * 50))
        
        # Combine for 0-100 score
        regime_score = return_score + win_rate_score
        
        # Update preferred regimes with exponential weighting
        current_score = profile["preferred_regimes"].get(top_regime, 50.0)
        profile["preferred_regimes"][top_regime] = current_score * (1 - alpha) + regime_score * alpha

    def get_rotation_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of strategy rotations
        
        Returns:
            history: List of historical strategy rotations
        """
        return self.rotation_history

    def get_strategy_performance(self, strategy_id: str = None) -> Dict[str, Any]:
        """
        Get performance data for strategies
        
        Args:
            strategy_id: Optional specific strategy ID to get data for
            
        Returns:
            performance_data: Dictionary of performance data
        """
        if strategy_id:
            if strategy_id not in self.performance_profiles:
                self.logger.warning(f"Strategy {strategy_id} not found")
                return {}
                
            return {
                "profile": self.performance_profiles.get(strategy_id, {}),
                "metadata": self.strategies.get(strategy_id, {}),
                "regime_performance": self.regime_performance.get(strategy_id, {})
            }
        else:
            # Return summary of all strategies
            return {
                "strategies": self.strategies,
                "performance_summary": {
                    sid: {
                        "avg_return": profile.get("avg_return", 0.0),
                        "sharpe_ratio": profile.get("sharpe_ratio", 0.0),
                        "win_rate": profile.get("win_rate", 0.0)
                    }
                    for sid, profile in self.performance_profiles.items()
                }
            }

    def remove_strategy(self, strategy_id: str) -> bool:
        """
        Remove a strategy from the rotator
        
        Args:
            strategy_id: Identifier of the strategy to remove
            
        Returns:
            success: Whether removal was successful
        """
        try:
            if strategy_id not in self.strategies:
                self.logger.warning(f"Strategy {strategy_id} not found")
                return False
                
            # Remove from all registries
            self.strategies.pop(strategy_id, None)
            self.performance_profiles.pop(strategy_id, None)
            self.regime_performance.pop(strategy_id, None)
            
            # Save changes
            self._save_registered_strategies()
            
            self.logger.info(f"Removed strategy {strategy_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing strategy {strategy_id}: {str(e)}")
            return False 