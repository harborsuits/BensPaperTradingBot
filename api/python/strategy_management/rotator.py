#!/usr/bin/env python3
"""
Dynamic Strategy Rotator

This module implements the dynamic strategy rotation system that adapts
to market conditions and performance metrics.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

from trading_bot.strategy_management.interfaces import (
    CoreContext, Strategy, MarketContext, StrategyPrioritizer, EventListener, EventPayload
)

logger = logging.getLogger(__name__)

class RotationEvent:
    """Represents a strategy rotation event"""
    
    def __init__(self, 
                 trigger_type: str,
                 old_allocations: Dict[str, float],
                 new_allocations: Dict[str, float],
                 reasoning: Optional[Dict[str, Any]] = None,
                 timestamp: Optional[datetime] = None):
        self.trigger_type = trigger_type
        self.old_allocations = old_allocations
        self.new_allocations = new_allocations
        self.reasoning = reasoning or {}
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "trigger_type": self.trigger_type,
            "old_allocations": self.old_allocations,
            "new_allocations": self.new_allocations,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RotationEvent':
        """Create instance from dictionary"""
        return cls(
            trigger_type=data["trigger_type"],
            old_allocations=data["old_allocations"],
            new_allocations=data["new_allocations"],
            reasoning=data.get("reasoning", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) 
                if isinstance(data["timestamp"], str) else data["timestamp"]
        )
    
    def __str__(self) -> str:
        """String representation"""
        return (f"RotationEvent(trigger={self.trigger_type}, "
                f"timestamp={self.timestamp.strftime('%Y-%m-%d %H:%M:%S')})")


class DynamicStrategyRotator(EventListener):
    """
    Dynamic Strategy Rotator
    
    Responsible for dynamically rotating strategy allocations based on:
    - Market regime changes
    - Strategy performance shifts
    - Volatility changes
    - News impacts
    
    Acts as an event listener to respond to various system events.
    """
    
    def __init__(self, 
                 core_context: CoreContext,
                 strategy_prioritizer: StrategyPrioritizer,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the rotator
        
        Args:
            core_context: Core context instance
            strategy_prioritizer: Strategy prioritizer
            config: Configuration dictionary
        """
        super().__init__(event_types=[
            "market_context_updated",
            "regime_changed",
            "strategy_performance_updated",
            "news_impact_analyzed",
            "rotation_requested"
        ])
        
        self.core_context = core_context
        self.strategy_prioritizer = strategy_prioritizer
        self.config = config or {}
        
        # Set default configuration values
        self._set_default_config()
        
        # Register as listener for relevant events
        for event_type in self.event_types:
            core_context.add_listener(event_type, self)
        
        # Initialize rotation history
        self.rotation_history: List[RotationEvent] = []
        
        # Initialize last rotation timestamp
        self.last_rotation_timestamp = datetime.now()
        
        # Track last known allocations (for computing changes)
        self.current_allocations = self._get_current_allocations()
        
        logger.info("Dynamic Strategy Rotator initialized")
    
    def _set_default_config(self) -> None:
        """Set default configuration values"""
        defaults = {
            "min_rotation_interval_minutes": 60,
            "performance_threshold_pct": 5.0,
            "max_daily_rotations": 5,
            "rotation_cooldown_minutes": 30,
            "regime_change_force_rotation": True,
            "enforce_gradual_transitions": True,
            "max_allocation_change_pct": 25.0,
            "volatility_sensitivity": "medium",
            "news_impact_threshold": 0.5,
            "enable_auto_rotation": True
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def _get_current_allocations(self) -> Dict[str, float]:
        """Get current allocations from strategy instances"""
        allocations = {}
        for strategy_id, strategy in self.core_context.get_all_strategies().items():
            allocations[strategy_id] = strategy.get_allocation()
        return allocations
    
    def _enforce_allocation_constraints(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """
        Enforce allocation constraints
        
        Args:
            allocations: Raw allocations from prioritizer
            
        Returns:
            Adjusted allocations that respect constraints
        """
        constraints = self.core_context.get_global_constraints()
        max_per_strategy = constraints.get("max_allocation_per_strategy", 80.0)
        min_per_strategy = constraints.get("min_allocation_per_strategy", 0.0)
        total_limit = constraints.get("total_allocation_limit", 100.0)
        
        # Enforce min/max per strategy
        for strategy_id in allocations:
            allocations[strategy_id] = max(min_per_strategy, min(max_per_strategy, allocations[strategy_id]))
        
        # Enforce total allocation limit
        total_allocation = sum(allocations.values())
        if total_allocation > total_limit:
            # Scale down proportionally
            scale_factor = total_limit / total_allocation
            for strategy_id in allocations:
                allocations[strategy_id] *= scale_factor
        
        return allocations
    
    def _enforce_gradual_transition(self, new_allocations: Dict[str, float]) -> Dict[str, float]:
        """
        Enforce gradual transition between old and new allocations
        
        Args:
            new_allocations: Target allocations
            
        Returns:
            Allocations with limited changes from current allocations
        """
        if not self.config.get("enforce_gradual_transitions", True):
            return new_allocations
        
        max_change_pct = self.config.get("max_allocation_change_pct", 25.0)
        gradual_allocations = {}
        
        for strategy_id, new_alloc in new_allocations.items():
            current_alloc = self.current_allocations.get(strategy_id, 0.0)
            max_change = current_alloc * (max_change_pct / 100.0)
            
            # Limit the change to max_change
            if new_alloc > current_alloc:
                gradual_allocations[strategy_id] = min(new_alloc, current_alloc + max_change)
            else:
                gradual_allocations[strategy_id] = max(new_alloc, current_alloc - max_change)
        
        # Add any missing strategies (could happen if new strategies were added)
        for strategy_id in self.current_allocations:
            if strategy_id not in gradual_allocations:
                gradual_allocations[strategy_id] = self.current_allocations[strategy_id]
        
        return gradual_allocations
    
    def _apply_allocations(self, new_allocations: Dict[str, float]) -> None:
        """
        Apply new allocations to strategy instances
        
        Args:
            new_allocations: Target allocations
        """
        available_strategies = self.core_context.get_all_strategies()
        
        for strategy_id, allocation in new_allocations.items():
            if strategy_id in available_strategies:
                strategy = available_strategies[strategy_id]
                current_allocation = strategy.get_allocation()
                
                if abs(current_allocation - allocation) > 0.1:  # Only update if change is significant
                    strategy.set_allocation(allocation)
                    logger.info(f"Updated allocation for {strategy_id}: {current_allocation:.2f}% -> {allocation:.2f}%")
            else:
                logger.warning(f"Cannot apply allocation to unknown strategy: {strategy_id}")
    
    def _record_rotation_event(self, trigger_type: str, new_allocations: Dict[str, float], 
                              reasoning: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a rotation event to history
        
        Args:
            trigger_type: What triggered this rotation
            new_allocations: New allocations being applied
            reasoning: Optional reasoning data
        """
        event = RotationEvent(
            trigger_type=trigger_type,
            old_allocations=self.current_allocations.copy(),
            new_allocations=new_allocations.copy(),
            reasoning=reasoning,
            timestamp=datetime.now()
        )
        
        # Add to history and update last rotation timestamp
        self.rotation_history.append(event)
        self.last_rotation_timestamp = event.timestamp
        
        # Limit history size
        max_history = 100
        if len(self.rotation_history) > max_history:
            self.rotation_history = self.rotation_history[-max_history:]
        
        # Update current allocations reference
        self.current_allocations = new_allocations.copy()
        
        # Notify about rotation
        self.core_context._notify_listeners("rotation_executed", event.to_dict())
        
        logger.info(f"Executed rotation: {trigger_type} at {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _is_rotation_allowed(self) -> Tuple[bool, str]:
        """
        Check if rotation is currently allowed based on constraints
        
        Returns:
            Tuple of (is_allowed, reason)
        """
        if not self.config.get("enable_auto_rotation", True):
            return False, "Auto-rotation is disabled in configuration"
        
        # Check cooldown period
        cooldown_minutes = self.config.get("rotation_cooldown_minutes", 30)
        time_since_last = datetime.now() - self.last_rotation_timestamp
        if time_since_last < timedelta(minutes=cooldown_minutes):
            return False, f"Cooldown period active ({cooldown_minutes}min)"
        
        # Check max daily rotations
        max_daily = self.config.get("max_daily_rotations", 5)
        today_rotations = sum(1 for event in self.rotation_history 
                            if event.timestamp.date() == datetime.now().date())
        if today_rotations >= max_daily:
            return False, f"Daily rotation limit reached ({max_daily})"
        
        return True, "Rotation allowed"
    
    def trigger_rotation(self, trigger_type: str) -> bool:
        """
        Trigger a strategy rotation
        
        Args:
            trigger_type: What triggered this rotation
            
        Returns:
            True if rotation was executed, False otherwise
        """
        # Check if rotation is allowed
        is_allowed, reason = self._is_rotation_allowed()
        if not is_allowed and trigger_type != "forced":
            logger.info(f"Rotation not allowed: {reason}")
            return False
        
        # Get current market context
        market_context = self.core_context.get_market_context()
        
        # Get prioritized allocations with reasoning
        prioritization_result = self.strategy_prioritizer.prioritize_strategies(
            market_context, include_reasoning=True)
        
        # Extract allocations and reasoning
        new_allocations = prioritization_result.get("allocations", {})
        reasoning = prioritization_result.get("reasoning", {})
        
        # Apply constraints to allocations
        constrained_allocations = self._enforce_allocation_constraints(new_allocations)
        
        # Apply gradual transition (if enabled)
        if trigger_type != "forced" and self.config.get("enforce_gradual_transitions", True):
            final_allocations = self._enforce_gradual_transition(constrained_allocations)
        else:
            final_allocations = constrained_allocations
        
        # Check if the change is significant
        is_significant = False
        for strategy_id, new_alloc in final_allocations.items():
            current_alloc = self.current_allocations.get(strategy_id, 0.0)
            if abs(new_alloc - current_alloc) > 1.0:  # More than 1% change
                is_significant = True
                break
        
        if not is_significant and trigger_type != "forced":
            logger.info("No significant allocation changes, skipping rotation")
            return False
        
        # Apply the new allocations to strategies
        self._apply_allocations(final_allocations)
        
        # Record the rotation event
        self._record_rotation_event(trigger_type, final_allocations, reasoning)
        
        return True
    
    def get_rotation_history(self) -> List[Dict[str, Any]]:
        """Get history of rotation events as dictionaries"""
        return [event.to_dict() for event in self.rotation_history]
    
    def force_rotation(self) -> bool:
        """Force a rotation regardless of constraints"""
        return self.trigger_rotation("forced")
    
    # Event handler methods
    
    def on_market_context_updated(self, market_context: MarketContext) -> None:
        """Handle market context updated event"""
        # Check for significant volatility change
        if hasattr(market_context, "volatility") and hasattr(market_context, "previous_volatility"):
            vol_change = abs(market_context.volatility - market_context.previous_volatility)
            vol_sensitivity = self.config.get("volatility_sensitivity", "medium")
            
            # Convert sensitivity to threshold
            sensitivity_map = {
                "low": 0.5,
                "medium": 0.3,
                "high": 0.15
            }
            threshold = sensitivity_map.get(vol_sensitivity, 0.3)
            
            if vol_change > threshold:
                logger.info(f"Significant volatility change detected: {vol_change:.2f}")
                self.trigger_rotation("volatility_change")
    
    def on_regime_changed(self, event_data: Dict[str, Any]) -> None:
        """Handle regime change event"""
        if self.config.get("regime_change_force_rotation", True):
            old_regime = event_data.get("old_regime", "unknown")
            new_regime = event_data.get("new_regime", "unknown")
            
            logger.info(f"Market regime changed from {old_regime} to {new_regime}")
            self.trigger_rotation("regime_change")
    
    def on_strategy_performance_updated(self, event_data: Dict[str, Any]) -> None:
        """Handle strategy performance updated event"""
        # Check if any strategy has significantly changed performance
        strategy_id = event_data.get("strategy_id")
        
        # Get performance threshold
        perf_threshold = self.config.get("performance_threshold_pct", 5.0)
        
        # Get strategy history
        history = self.core_context.get_performance_history(strategy_id)
        
        if not history or strategy_id not in history:
            return
        
        # Get the 2 most recent performance entries
        entries = history[strategy_id]
        if len(entries) < 2:
            return
        
        recent = entries[-1]
        previous = entries[-2]
        
        # Check for significant Sharpe ratio change
        if "sharpe_ratio" in recent and "sharpe_ratio" in previous:
            sharpe_change = abs(recent["sharpe_ratio"] - previous["sharpe_ratio"])
            if sharpe_change > (perf_threshold / 100):
                logger.info(f"Significant performance change for {strategy_id}: Sharpe change {sharpe_change:.2f}")
                self.trigger_rotation("performance_change")
        
        # Check for significant return change
        if "return_pct" in recent and "return_pct" in previous:
            return_change = abs(recent["return_pct"] - previous["return_pct"])
            if return_change > perf_threshold:
                logger.info(f"Significant performance change for {strategy_id}: Return change {return_change:.2f}%")
                self.trigger_rotation("performance_change")
    
    def on_news_impact_analyzed(self, event_data: Dict[str, Any]) -> None:
        """Handle news impact analysis event"""
        impact_score = event_data.get("impact_score", 0)
        threshold = self.config.get("news_impact_threshold", 0.5)
        
        if abs(impact_score) >= threshold:
            logger.info(f"Significant news impact detected: {impact_score:.2f}")
            
            # Add impact details to the context before rotation
            self.core_context.update_market_context({
                "recent_news_impact": impact_score,
                "news_impact_details": event_data
            })
            
            self.trigger_rotation("news_impact")
    
    def on_rotation_requested(self, _: Any) -> None:
        """Handle explicit rotation request"""
        self.trigger_rotation("manual_request")


class WeightedPerformanceStrategyPrioritizer(StrategyPrioritizer):
    """
    Weighted performance-based strategy prioritization
    
    Prioritizes strategies based on:
    - Recent performance
    - Historical regime-specific performance
    - Volatility adaptation
    - Parameter sensitivity
    """
    
    def __init__(self, core_context: CoreContext, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the prioritizer
        
        Args:
            core_context: Core context instance
            config: Configuration dictionary
        """
        self.core_context = core_context
        self.config = config or {}
        
        # Set default weights
        self._set_default_weights()
        
        # Initialize regime-specific strategy performance cache
        self.regime_performance = defaultdict(lambda: defaultdict(list))
        
        logger.info("Weighted Performance Strategy Prioritizer initialized")
    
    def _set_default_weights(self) -> None:
        """Set default weight configuration"""
        defaults = {
            "weight_recent_performance": 0.35,
            "weight_regime_performance": 0.3,
            "weight_volatility_adaptation": 0.2,
            "weight_risk_adjusted": 0.15,
            "default_allocation": 5.0,  # Default allocation for unknown strategies
            "min_data_points": 3,  # Minimum data points needed for reliable scoring
            "lookback_periods": 10  # How many periods to look back for performance
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def _calculate_recent_performance_score(self, strategy_id: str) -> float:
        """
        Calculate recent performance score for a strategy
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Score from 0-1 based on recent performance
        """
        history = self.core_context.get_performance_history(strategy_id)
        if not history or strategy_id not in history:
            return 0.0
        
        entries = history[strategy_id]
        
        if not entries:
            return 0.0
        
        # Get recent entries
        lookback = min(self.config["lookback_periods"], len(entries))
        recent_entries = entries[-lookback:]
        
        if not recent_entries:
            return 0.0
        
        # Calculate weighted average of returns (more recent entries have higher weight)
        weights = np.linspace(0.5, 1.0, len(recent_entries))
        returns = [entry.get("return_pct", 0) for entry in recent_entries]
        
        if not returns or len(returns) < self.config["min_data_points"]:
            return 0.0
        
        weighted_return = np.average(returns, weights=weights)
        
        # Also consider Sharpe ratio if available
        sharpe_values = [entry.get("sharpe_ratio", None) for entry in recent_entries]
        sharpe_values = [s for s in sharpe_values if s is not None]
        
        if len(sharpe_values) >= self.config["min_data_points"]:
            avg_sharpe = np.mean(sharpe_values)
            # Combine return and Sharpe (normalize Sharpe to 0-1 range)
            normalized_sharpe = min(max(avg_sharpe, 0), 3) / 3
            score = 0.7 * weighted_return / 100 + 0.3 * normalized_sharpe
        else:
            # Just use returns if not enough Sharpe data
            score = weighted_return / 100
        
        # Normalize to 0-1 range (assuming max return of 20%)
        return min(max(score, 0), 1)
    
    def _calculate_regime_performance_score(self, strategy_id: str, current_regime: str) -> float:
        """
        Calculate regime-specific performance score
        
        Args:
            strategy_id: Strategy ID
            current_regime: Current market regime
            
        Returns:
            Score from 0-1 based on historical performance in this regime
        """
        if current_regime not in self.regime_performance or strategy_id not in self.regime_performance[current_regime]:
            return 0.5  # Neutral score for unknown regime performance
        
        regime_data = self.regime_performance[current_regime][strategy_id]
        
        if len(regime_data) < self.config["min_data_points"]:
            return 0.5  # Not enough data points
        
        # Calculate average return in this regime
        avg_return = np.mean([d.get("return_pct", 0) for d in regime_data])
        
        # Normalize to 0-1 range (assuming max return of 15% in a regime)
        normalized_return = (avg_return + 15) / 30
        return min(max(normalized_return, 0), 1)
    
    def _calculate_volatility_adaptation_score(self, strategy_id: str, volatility: float) -> float:
        """
        Calculate volatility adaptation score
        
        Args:
            strategy_id: Strategy ID
            volatility: Current market volatility
            
        Returns:
            Score from 0-1 based on how well strategy handles current volatility
        """
        history = self.core_context.get_performance_history(strategy_id)
        if not history or strategy_id not in history:
            return 0.5  # Neutral score
        
        entries = history[strategy_id]
        
        if len(entries) < self.config["min_data_points"]:
            return 0.5  # Not enough data
        
        # Collect volatility and return pairs
        vol_return_pairs = []
        for entry in entries:
            if "market_volatility" in entry and "return_pct" in entry:
                vol_return_pairs.append((entry["market_volatility"], entry["return_pct"]))
        
        if len(vol_return_pairs) < self.config["min_data_points"]:
            return 0.5  # Not enough paired data
        
        # Find performance in similar volatility conditions
        similar_vol_returns = [ret for vol, ret in vol_return_pairs 
                              if abs(vol - volatility) < 0.1]
        
        if not similar_vol_returns:
            return 0.5  # No similar volatility data
        
        avg_return = np.mean(similar_vol_returns)
        
        # Normalize to 0-1 range
        normalized_return = (avg_return + 10) / 20
        return min(max(normalized_return, 0), 1)
    
    def _calculate_risk_adjusted_score(self, strategy_id: str) -> float:
        """
        Calculate risk-adjusted performance score
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Score from 0-1 based on risk-adjusted metrics
        """
        history = self.core_context.get_performance_history(strategy_id)
        if not history or strategy_id not in history:
            return 0.5  # Neutral score
        
        entries = history[strategy_id]
        
        if len(entries) < self.config["min_data_points"]:
            return 0.5  # Not enough data
        
        # Collect relevant metrics
        sharpe_values = [entry.get("sharpe_ratio", None) for entry in entries]
        sharpe_values = [s for s in sharpe_values if s is not None]
        
        sortino_values = [entry.get("sortino_ratio", None) for entry in entries]
        sortino_values = [s for s in sortino_values if s is not None]
        
        max_drawdown_values = [entry.get("max_drawdown", None) for entry in entries]
        max_drawdown_values = [d for d in max_drawdown_values if d is not None]
        
        if not sharpe_values and not sortino_values and not max_drawdown_values:
            return 0.5  # No risk metrics available
        
        # Calculate score components
        sharpe_score = np.mean(sharpe_values) / 3 if sharpe_values else 0.5
        sortino_score = np.mean(sortino_values) / 4 if sortino_values else 0.5
        
        # Lower drawdown is better
        if max_drawdown_values:
            avg_drawdown = np.mean(max_drawdown_values)
            drawdown_score = 1 - (avg_drawdown / 30)  # Assuming 30% is worst case
        else:
            drawdown_score = 0.5
        
        # Combine scores
        available_metrics = sum(1 for x in [sharpe_values, sortino_values, max_drawdown_values] if x)
        
        if available_metrics == 0:
            return 0.5  # No metrics available
        
        weights = []
        scores = []
        
        if sharpe_values:
            weights.append(0.4)
            scores.append(sharpe_score)
        
        if sortino_values:
            weights.append(0.3)
            scores.append(sortino_score)
        
        if max_drawdown_values:
            weights.append(0.3)
            scores.append(drawdown_score)
        
        # Normalize weights
        weights = [w/sum(weights) for w in weights]
        
        # Weighted average
        combined_score = sum(w * s for w, s in zip(weights, scores))
        
        return min(max(combined_score, 0), 1)
    
    def _update_regime_performance(self, strategy_id: str, regime: str, 
                                  performance: Dict[str, Any]) -> None:
        """
        Update regime-specific performance data
        
        Args:
            strategy_id: Strategy ID
            regime: Market regime
            performance: Performance metrics
        """
        self.regime_performance[regime][strategy_id].append(performance)
        
        # Limit history size
        max_regime_history = 20
        if len(self.regime_performance[regime][strategy_id]) > max_regime_history:
            self.regime_performance[regime][strategy_id] = self.regime_performance[regime][strategy_id][-max_regime_history:]
    
    def _calculate_composite_score(self, strategy_id: str, market_context: MarketContext) -> Tuple[float, Dict[str, float]]:
        """
        Calculate composite strategy score
        
        Args:
            strategy_id: Strategy ID
            market_context: Current market context
            
        Returns:
            Tuple of (composite_score, component_scores)
        """
        # Get current regime and volatility
        regime = getattr(market_context, "regime", "unknown")
        volatility = getattr(market_context, "volatility", 0.2)
        
        # Calculate component scores
        recent_score = self._calculate_recent_performance_score(strategy_id)
        regime_score = self._calculate_regime_performance_score(strategy_id, regime)
        volatility_score = self._calculate_volatility_adaptation_score(strategy_id, volatility)
        risk_adjusted_score = self._calculate_risk_adjusted_score(strategy_id)
        
        # Apply weights
        weights = {
            "recent": self.config["weight_recent_performance"],
            "regime": self.config["weight_regime_performance"],
            "volatility": self.config["weight_volatility_adaptation"],
            "risk_adjusted": self.config["weight_risk_adjusted"]
        }
        
        composite_score = (
            weights["recent"] * recent_score +
            weights["regime"] * regime_score +
            weights["volatility"] * volatility_score +
            weights["risk_adjusted"] * risk_adjusted_score
        )
        
        component_scores = {
            "recent_performance": recent_score,
            "regime_performance": regime_score,
            "volatility_adaptation": volatility_score,
            "risk_adjusted": risk_adjusted_score
        }
        
        return composite_score, component_scores
    
    def prioritize_strategies(self, market_context: MarketContext, include_reasoning: bool = False) -> Dict[str, Any]:
        """
        Prioritize strategies based on market context
        
        Args:
            market_context: Current market context
            include_reasoning: Whether to include reasoning in the result
            
        Returns:
            Dictionary with allocations and optional reasoning
        """
        strategy_ids = self.core_context.get_strategy_ids()
        
        if not strategy_ids:
            logger.warning("No strategies found for prioritization")
            return {"allocations": {}, "reasoning": {}}
        
        # Calculate scores for each strategy
        scores = {}
        components = {}
        
        for strategy_id in strategy_ids:
            score, component_scores = self._calculate_composite_score(strategy_id, market_context)
            scores[strategy_id] = score
            components[strategy_id] = component_scores
        
        # Handle case with no scores
        if not scores:
            return {"allocations": {strategy_id: self.config["default_allocation"] for strategy_id in strategy_ids},
                    "reasoning": {}}
        
        # Normalize scores to allocations summing to 100%
        total_score = sum(scores.values())
        
        if total_score > 0:
            allocations = {strategy_id: (score / total_score) * 100 for strategy_id, score in scores.items()}
        else:
            # Equal allocation if all scores are 0
            equal_allocation = 100 / len(strategy_ids)
            allocations = {strategy_id: equal_allocation for strategy_id in strategy_ids}
        
        result = {"allocations": allocations}
        
        # Include reasoning if requested
        if include_reasoning:
            reasoning = {
                "scores": scores,
                "component_scores": components,
                "market_context": {
                    "regime": getattr(market_context, "regime", "unknown"),
                    "volatility": getattr(market_context, "volatility", "unknown"),
                    "timestamp": datetime.now().isoformat()
                }
            }
            result["reasoning"] = reasoning
        
        return result 