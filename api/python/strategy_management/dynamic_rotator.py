import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, deque
import json
import os

from trading_bot.strategy_management.interfaces import CoreContext, MarketContext, Strategy, EventListener, StrategySelector, MarketRegimeClassifier, MarketContextProvider
from trading_bot.common.market_types import MarketRegime

# Setup logging
logger = logging.getLogger(__name__)

class DynamicStrategyRotator(StrategySelector):
    """
    Dynamically selects trading strategies based on market regime, 
    historical performance, and current market conditions.
    """
    
    def __init__(self, 
                regime_classifier: MarketRegimeClassifier,
                context_provider: MarketContextProvider,
                config_path: Optional[str] = None):
        """
        Initialize the dynamic strategy rotator
        
        Args:
            regime_classifier: Component that classifies market regimes
            context_provider: Component that provides market context
            config_path: Path to configuration file
        """
        self.regime_classifier = regime_classifier
        self.context_provider = context_provider
        self.strategies = {}  # Strategy database
        self.strategy_performance = {}  # Performance history by strategy
        self.regime_performance = {}  # Performance by regime
        
        self.current_strategy = None
        self.strategy_history = []
        self.confidence_level = 0.0
        self.last_rotation_time = datetime.now()
        self.min_strategy_duration = timedelta(hours=6)  # Minimum time before rotation
        
        # Default weights for selection factors
        self.selection_weights = {
            "regime_match": 0.40,
            "historical_performance": 0.30,
            "recent_performance": 0.20,
            "volatility_match": 0.10
        }
        
        # Load configuration if provided
        if config_path:
            self._load_config(config_path)
            
    def select_strategy(self, market_context: Dict[str, Any]) -> str:
        """
        Select the most appropriate strategy based on market context
        
        Args:
            market_context: Dictionary containing current market data
            
        Returns:
            strategy_id: Identifier of the selected strategy
        """
        # Only allow rotation if minimum duration has passed
        if self.current_strategy and (datetime.now() - self.last_rotation_time < self.min_strategy_duration):
            logger.info(f"Strategy rotation prevented - minimum duration not met. Current: {self.current_strategy}")
            return self.current_strategy
        
        # Get current market regime
        current_regime = self.regime_classifier.classify_regime(market_context)
        regime_confidence = self.regime_classifier.get_regime_confidence()
        
        logger.info(f"Current market regime: {current_regime} (confidence: {regime_confidence:.2f})")
        
        # Calculate scores for each strategy
        strategy_scores = {}
        
        for strategy_id, strategy_info in self.strategies.items():
            # Skip disabled strategies
            if not strategy_info.get("enabled", True):
                continue
                
            score = self._calculate_strategy_score(
                strategy_id, 
                strategy_info, 
                current_regime, 
                market_context
            )
            
            strategy_scores[strategy_id] = score
            
        # No strategies available
        if not strategy_scores:
            logger.warning("No strategies available for selection")
            return self.current_strategy or "default_strategy"
            
        # Select the strategy with the highest score
        selected_strategy = max(strategy_scores.items(), key=lambda x: x[1])
        strategy_id = selected_strategy[0]
        score = selected_strategy[1]
        
        # Update confidence level
        self.confidence_level = score / 100.0  # Normalize to 0-1 range
        
        # Record the rotation
        if self.current_strategy != strategy_id:
            logger.info(f"Strategy rotation: {self.current_strategy} -> {strategy_id} (score: {score:.2f})")
            self.strategy_history.append({
                "timestamp": datetime.now().isoformat(),
                "previous": self.current_strategy,
                "new": strategy_id,
                "regime": current_regime,
                "confidence": self.confidence_level,
                "market_context": {k: v for k, v in market_context.items() if k in [
                    "volatility", "trend", "volume", "sentiment"
                ]}
            })
            
            # Update rotation time
            self.last_rotation_time = datetime.now()
            
        self.current_strategy = strategy_id
        return strategy_id
        
    def register_strategy(self, strategy_id: str, 
                         performance_profile: Dict[str, Any]) -> bool:
        """
        Register a new strategy with its performance profile
        
        Args:
            strategy_id: Unique identifier for the strategy
            performance_profile: Performance characteristics of the strategy
            
        Returns:
            success: Whether registration was successful
        """
        if strategy_id in self.strategies:
            logger.warning(f"Strategy {strategy_id} already registered. Updating profile.")
            
        # Ensure the profile contains required fields
        required_fields = ["regime_affinity", "volatility_range", "description"]
        for field in required_fields:
            if field not in performance_profile:
                logger.error(f"Strategy profile missing required field: {field}")
                return False
                
        # Initialize performance history if new
        if strategy_id not in self.strategy_performance:
            self.strategy_performance[strategy_id] = {
                "overall": {
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "total_trades": 0
                },
                "by_regime": {},
                "recent_trades": []
            }
            
        # Store the strategy
        self.strategies[strategy_id] = performance_profile
        logger.info(f"Strategy {strategy_id} registered successfully")
        
        return True
        
    def get_strategy_confidence(self) -> float:
        """
        Get confidence level for the current strategy selection
        
        Returns:
            confidence: Confidence level (0.0-1.0)
        """
        return self.confidence_level
        
    def update_strategy_performance(self, strategy_id: str,
                                  performance_metrics: Dict[str, float]) -> None:
        """
        Update performance metrics for a specific strategy
        
        Args:
            strategy_id: Identifier of the strategy
            performance_metrics: Updated performance metrics
        """
        if strategy_id not in self.strategy_performance:
            logger.warning(f"Strategy {strategy_id} not found in performance database. Initializing.")
            self.strategy_performance[strategy_id] = {
                "overall": {
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "total_trades": 0
                },
                "by_regime": {},
                "recent_trades": []
            }
            
        # Get current market regime
        market_context = self.context_provider.get_market_context()
        current_regime = self.regime_classifier.classify_regime(market_context)
        
        # Update overall performance with weighted average
        current_stats = self.strategy_performance[strategy_id]["overall"]
        total_trades = current_stats["total_trades"]
        new_total = total_trades + 1
        
        # Update overall performance metrics
        for metric in ["win_rate", "profit_factor", "sharpe_ratio", "max_drawdown"]:
            if metric in performance_metrics:
                # Weighted average based on number of trades
                current_value = current_stats[metric]
                new_value = performance_metrics[metric]
                
                updated_value = (current_value * total_trades + new_value) / new_total
                self.strategy_performance[strategy_id]["overall"][metric] = updated_value
                
        # Update trade count
        self.strategy_performance[strategy_id]["overall"]["total_trades"] = new_total
        
        # Update regime-specific performance
        if current_regime not in self.strategy_performance[strategy_id]["by_regime"]:
            self.strategy_performance[strategy_id]["by_regime"][current_regime] = {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_trades": 0
            }
            
        regime_stats = self.strategy_performance[strategy_id]["by_regime"][current_regime]
        regime_trades = regime_stats["total_trades"]
        new_regime_total = regime_trades + 1
        
        # Update regime-specific metrics
        for metric in ["win_rate", "profit_factor", "sharpe_ratio", "max_drawdown"]:
            if metric in performance_metrics:
                current_value = regime_stats[metric]
                new_value = performance_metrics[metric]
                
                updated_value = (current_value * regime_trades + new_value) / new_regime_total
                self.strategy_performance[strategy_id]["by_regime"][current_regime][metric] = updated_value
                
        # Update regime trade count
        self.strategy_performance[strategy_id]["by_regime"][current_regime]["total_trades"] = new_regime_total
        
        # Add to recent trades (keep last 20)
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "regime": current_regime,
            "metrics": performance_metrics
        }
        
        recent_trades = self.strategy_performance[strategy_id]["recent_trades"]
        recent_trades.append(trade_record)
        
        # Keep only the most recent 20 trades
        if len(recent_trades) > 20:
            self.strategy_performance[strategy_id]["recent_trades"] = recent_trades[-20:]
            
        logger.debug(f"Updated performance for strategy {strategy_id} in regime {current_regime}")
        
    def get_rotation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the history of strategy rotations
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            history: List of strategy rotation events
        """
        return self.strategy_history[-limit:] if self.strategy_history else []
        
    def get_strategy_details(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed information about a strategy
        
        Args:
            strategy_id: Strategy identifier, or None for current strategy
            
        Returns:
            details: Dictionary containing strategy details
        """
        target_id = strategy_id or self.current_strategy
        
        if not target_id or target_id not in self.strategies:
            return {"error": "Strategy not found"}
            
        # Combine profile and performance data
        profile = self.strategies[target_id]
        performance = self.strategy_performance.get(target_id, {})
        
        details = {
            "id": target_id,
            "description": profile.get("description", "No description"),
            "regime_affinity": profile.get("regime_affinity", {}),
            "volatility_range": profile.get("volatility_range", {}),
            "performance": performance.get("overall", {}),
            "regime_performance": performance.get("by_regime", {}),
            "recent_trades": performance.get("recent_trades", [])
        }
        
        return details
        
    def save_state(self, file_path: str) -> bool:
        """
        Save the current state to a file
        
        Args:
            file_path: Path to save the state
            
        Returns:
            success: Whether the save was successful
        """
        try:
            state = {
                "strategies": self.strategies,
                "strategy_performance": self.strategy_performance,
                "strategy_history": self.strategy_history,
                "current_strategy": self.current_strategy,
                "selection_weights": self.selection_weights,
                "last_rotation_time": self.last_rotation_time.isoformat(),
                "min_strategy_duration": self.min_strategy_duration.total_seconds()
            }
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Strategy rotator state saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save rotator state: {str(e)}")
            return False
            
    def load_state(self, file_path: str) -> bool:
        """
        Load state from a file
        
        Args:
            file_path: Path to load the state from
            
        Returns:
            success: Whether the load was successful
        """
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
                
            self.strategies = state.get("strategies", {})
            self.strategy_performance = state.get("strategy_performance", {})
            self.strategy_history = state.get("strategy_history", [])
            self.current_strategy = state.get("current_strategy")
            self.selection_weights = state.get("selection_weights", self.selection_weights)
            
            # Convert ISO strings back to datetime
            self.last_rotation_time = datetime.fromisoformat(state.get("last_rotation_time", 
                                                                     datetime.now().isoformat()))
            
            # Convert seconds to timedelta
            duration_seconds = state.get("min_strategy_duration", 
                                       self.min_strategy_duration.total_seconds())
            self.min_strategy_duration = timedelta(seconds=duration_seconds)
            
            logger.info(f"Strategy rotator state loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load rotator state: {str(e)}")
            return False
            
    def _calculate_strategy_score(self, strategy_id: str, 
                               strategy_info: Dict[str, Any],
                               current_regime: str,
                               market_context: Dict[str, Any]) -> float:
        """
        Calculate a score for a strategy based on the current market conditions
        
        Args:
            strategy_id: Identifier of the strategy
            strategy_info: Information about the strategy
            current_regime: Current market regime
            market_context: Current market data
            
        Returns:
            score: Strategy suitability score (0-100)
        """
        # Start with base score
        score = 50.0
        
        # Get strategy performance data
        performance = self.strategy_performance.get(strategy_id, {})
        
        # 1. Regime match score (0-40 points)
        regime_affinities = strategy_info.get("regime_affinity", {})
        regime_match_score = regime_affinities.get(current_regime, 0.0) * 40.0
        
        # 2. Historical performance score (0-30 points)
        historical_score = 0.0
        if strategy_id in self.strategy_performance:
            # Overall metrics
            overall = performance.get("overall", {})
            win_rate = overall.get("win_rate", 0.0)
            profit_factor = overall.get("profit_factor", 1.0)
            sharpe = overall.get("sharpe_ratio", 0.0)
            
            # Calculate historical score
            win_rate_score = min(win_rate * 10.0, 10.0)  # 0-10 points
            profit_factor_score = min((profit_factor - 1.0) * 5.0, 10.0)  # 0-10 points
            sharpe_score = min(sharpe * 2.5, 10.0)  # 0-10 points
            
            historical_score = win_rate_score + profit_factor_score + sharpe_score
            
        # 3. Recent performance in current regime (0-20 points)
        recent_score = 0.0
        if strategy_id in self.strategy_performance:
            # Regime-specific performance
            regime_performance = performance.get("by_regime", {}).get(current_regime, {})
            if regime_performance:
                regime_win_rate = regime_performance.get("win_rate", 0.0)
                regime_profit_factor = regime_performance.get("profit_factor", 1.0)
                
                # Recent regime score
                recent_score = min(regime_win_rate * 10.0, 10.0) + min((regime_profit_factor - 1.0) * 5.0, 10.0)
                
        # 4. Volatility match score (0-10 points)
        volatility_score = 0.0
        if "volatility" in market_context and "volatility_range" in strategy_info:
            current_vol = market_context["volatility"]
            vol_range = strategy_info["volatility_range"]
            
            # Check if current volatility is within strategy's preferred range
            if vol_range.get("min", 0) <= current_vol <= vol_range.get("max", float('inf')):
                # Full score if in the middle of the range
                mid_range = (vol_range.get("min", 0) + vol_range.get("max", 100)) / 2
                distance = abs(current_vol - mid_range)
                range_width = max(vol_range.get("max", 100) - vol_range.get("min", 0), 1)
                
                # Score based on proximity to the middle of the preferred range
                volatility_score = 10.0 * (1.0 - min(distance / (range_width / 2), 1.0))
                
        # Calculate weighted total score
        total_score = (
            self.selection_weights["regime_match"] * regime_match_score +
            self.selection_weights["historical_performance"] * historical_score +
            self.selection_weights["recent_performance"] * recent_score +
            self.selection_weights["volatility_match"] * volatility_score
        )
        
        logger.debug(f"Strategy {strategy_id} score: {total_score:.2f} " +
                  f"(regime: {regime_match_score:.2f}, " +
                  f"hist: {historical_score:.2f}, " +
                  f"recent: {recent_score:.2f}, " +
                  f"vol: {volatility_score:.2f})")
        
        return total_score
        
    def _load_config(self, config_path: str) -> None:
        """
        Load configuration from a file
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Load selection weights
            if "selection_weights" in config:
                self.selection_weights = config["selection_weights"]
                
            # Load minimum strategy duration
            if "min_strategy_duration_hours" in config:
                hours = config["min_strategy_duration_hours"]
                self.min_strategy_duration = timedelta(hours=hours)
                
            # Load any pre-defined strategies
            if "strategies" in config:
                for strategy_id, profile in config["strategies"].items():
                    self.register_strategy(strategy_id, profile)
                    
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}") 