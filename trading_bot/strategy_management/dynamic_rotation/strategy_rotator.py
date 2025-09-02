import os
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Import interfaces from parent modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from strategy_management.interfaces import CoreContext, MarketContext, StrategyPrioritizer

logger = logging.getLogger("strategy_rotator")

class DynamicStrategyRotator:
    """
    Rotates strategies based on market conditions and constraints.
    Implements dynamic allocation adjustments to optimize performance
    while maintaining risk management rules.
    """
    
    def __init__(self, core_context: CoreContext, strategy_prioritizer: StrategyPrioritizer, 
                 config: Dict[str, Any]):
        self.core_context = core_context
        self.strategy_prioritizer = strategy_prioritizer
        self.config = config
        
        # Strategy allocation history
        self.allocation_history = []
        
        # Current allocations
        self.current_allocations = {}
        
        # Last rotation timestamp
        self.last_rotation = datetime.now() - timedelta(days=1)
        
        # Rotation frequency (in days)
        self.rotation_frequency_days = config.get("rotation_frequency_days", 7)
        
        # Minimum change threshold to trigger a rotation (percent)
        self.min_change_threshold = config.get("min_change_threshold", 5.0)
        
        # Force rotation on regime change
        self.force_on_regime_change = config.get("force_on_regime_change", True)
        
        # Risk-based allocation constraints
        self.drawdown_allocation_reduction = config.get("drawdown_allocation_reduction", 0.5)
        self.max_drawdown_threshold = config.get("max_drawdown_threshold", 10.0)
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Initialize with equal allocations if needed
        if not self.current_allocations:
            strategies = config.get("strategies", [])
            if strategies:
                equal_allocation = 100.0 / len(strategies)
                self.current_allocations = {s: equal_allocation for s in strategies}
        
        # Register for market context updates
        self.core_context.add_event_listener("market_context_updated", self._on_market_context_update)

    def _on_market_context_update(self, market_context):
        """Handle market context updates, potentially triggering a rotation"""
        try:
            with self.lock:
                # Check if we should force rotation due to regime change
                if self.force_on_regime_change:
                    current_regime = getattr(market_context, "regime", None)
                    prev_regime = getattr(self, "_prev_regime", None)
                    
                    if current_regime and prev_regime and current_regime != prev_regime:
                        logger.info(f"Market regime change detected: {prev_regime} -> {current_regime}")
                        # Force a rotation
                        self.rotate_strategies(force=True)
                    
                    # Update previous regime
                    self._prev_regime = current_regime
        except Exception as e:
            logger.error(f"Error handling market context update: {str(e)}")

    def should_rotate(self):
        """Determine if it's time to rotate strategies"""
        with self.lock:
            # Check if it's been long enough since last rotation
            days_since_rotation = (datetime.now() - self.last_rotation).total_seconds() / 86400
            
            return days_since_rotation >= self.rotation_frequency_days

    def get_current_allocations(self):
        """Get current strategy allocations"""
        with self.lock:
            return self.current_allocations.copy()

    def rotate_strategies(self, market_context=None, force=False):
        """
        Rotate strategies based on current market conditions and constraints.
        
        Args:
            market_context: Optional market context (if None, gets from core_context)
            force: Whether to force rotation even if not scheduled
            
        Returns:
            Dict containing rotation results
        """
        try:
            with self.lock:
                # Check if we should rotate
                if not force and not self.should_rotate():
                    return {
                        "rotated": False,
                        "message": "Rotation not scheduled yet",
                        "current_allocations": self.current_allocations
                    }
                
                # Get market context if not provided
                if market_context is None:
                    market_context = self.core_context.market_context
                
                # Prioritize strategies
                prioritization_result = self.strategy_prioritizer.prioritize_strategies(
                    market_context=market_context,
                    include_reasoning=True
                )
                
                # Get the recommended allocations
                target_allocations = prioritization_result.get("allocations", {})
                
                # Apply risk adjustments based on drawdowns
                risk_adjusted_allocations = self._apply_risk_adjustments(target_allocations)
                
                # Apply change constraints to limit allocation shifts
                constrained_allocations = self._apply_change_constraints(risk_adjusted_allocations)
                
                # Check if changes exceed minimum threshold
                significant_change = False
                for strategy, allocation in constrained_allocations.items():
                    current = self.current_allocations.get(strategy, 0.0)
                    change = abs(allocation - current)
                    if change >= self.min_change_threshold:
                        significant_change = True
                        break
                
                if not force and not significant_change:
                    return {
                        "rotated": False,
                        "message": "Changes below threshold",
                        "current_allocations": self.current_allocations,
                        "target_allocations": constrained_allocations
                    }
                
                # Apply the new allocations
                previous_allocations = self.current_allocations.copy()
                self.current_allocations = constrained_allocations
                
                # Record in history
                self.allocation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "regime": getattr(market_context, "regime", "unknown"),
                    "previous_allocations": previous_allocations,
                    "new_allocations": constrained_allocations,
                    "reasoning": prioritization_result.get("reasoning", {})
                })
                
                # Trim history if it gets too large
                if len(self.allocation_history) > 100:
                    self.allocation_history = self.allocation_history[-100:]
                
                # Update last rotation timestamp
                self.last_rotation = datetime.now()
                
                # Update strategies in core context
                self._update_strategy_allocations()
                
                # Prepare result
                result = {
                    "rotated": True,
                    "previous_allocations": previous_allocations,
                    "new_allocations": constrained_allocations,
                    "regime": getattr(market_context, "regime", "unknown"),
                    "market_summary": prioritization_result.get("market_summary", ""),
                    "reasoning": prioritization_result.get("reasoning", {})
                }
                
                logger.info(f"Completed strategy rotation. New allocations: {constrained_allocations}")
                
                return result
                
        except Exception as e:
            logger.error(f"Error rotating strategies: {str(e)}")
            return {
                "rotated": False,
                "message": f"Error: {str(e)}",
                "current_allocations": self.current_allocations
            }

    def _apply_risk_adjustments(self, allocations):
        """Apply risk-based adjustments to allocations"""
        adjusted = allocations.copy()
        
        # Check portfolio drawdown
        portfolio = getattr(self.core_context, "portfolio", None)
        portfolio_drawdown = getattr(portfolio, "drawdown_pct", 0.0) if portfolio else 0.0
        
        # Check individual strategy drawdowns
        strategy_drawdowns = {}
        for strat_name, strategy in self.core_context.strategies.items():
            strategy_drawdowns[strat_name] = getattr(strategy, "max_drawdown", 0.0)
        
        # If portfolio drawdown exceeds threshold, reduce allocations to riskier strategies
        if portfolio_drawdown > self.max_drawdown_threshold:
            logger.info(f"Portfolio drawdown ({portfolio_drawdown:.2f}%) exceeds threshold, adjusting allocations")
            
            # Calculate reduction factor based on severity of drawdown
            reduction_factor = min(1.0, portfolio_drawdown / self.max_drawdown_threshold * self.drawdown_allocation_reduction)
            
            # Identify the riskiest strategies (those with highest drawdowns)
            sorted_by_drawdown = sorted(
                strategy_drawdowns.items(),
                key=lambda x: x[1],
                reverse=True  # Highest drawdown first
            )
            
            # Calculate reduction for each strategy based on its risk
            reductions = {}
            total_reduction = 0.0
            
            # Apply reductions starting with riskiest strategies
            for strat_name, drawdown in sorted_by_drawdown:
                if strat_name not in adjusted:
                    continue
                
                # Skip strategies with minimal allocations
                if adjusted[strat_name] <= 5.0:
                    continue
                
                # Calculate reduction based on strategy risk and portfolio drawdown
                strategy_reduction = adjusted[strat_name] * reduction_factor * (drawdown / max(1.0, portfolio_drawdown))
                reductions[strat_name] = min(adjusted[strat_name] - 5.0, strategy_reduction)  # Don't go below 5%
                total_reduction += reductions[strat_name]
            
            # Apply reductions
            for strat_name, reduction in reductions.items():
                adjusted[strat_name] -= reduction
            
            # Redistribute the reduced allocation to less risky strategies
            if total_reduction > 0:
                # Find the least risky strategies (lowest drawdowns)
                low_risk_strategies = sorted_by_drawdown[-3:]  # Bottom 3
                
                # Calculate total current allocation for these strategies
                current_low_risk_allocation = sum(adjusted.get(s[0], 0.0) for s in low_risk_strategies)
                
                # Redistribute proportionally to their current allocations
                if current_low_risk_allocation > 0:
                    for strat_name, _ in low_risk_strategies:
                        if strat_name in adjusted:
                            added = (adjusted[strat_name] / current_low_risk_allocation) * total_reduction
                            adjusted[strat_name] += added
        
        # Normalize to ensure allocations sum to 100%
        total = sum(adjusted.values())
        if total > 0:
            for strategy in adjusted:
                adjusted[strategy] = (adjusted[strategy] / total) * 100.0
        
        return adjusted

    def _apply_change_constraints(self, target_allocations):
        """Limit allocation changes to max_allocation_change"""
        constrained = {}
        max_change = self.config.get("max_allocation_change", 15.0)
        
        for strategy in target_allocations:
            current = self.current_allocations.get(strategy, 0.0)
            target = target_allocations[strategy]
            
            # Limit change to max_allocation_change
            change = target - current
            if abs(change) > max_change:
                # Limit the change
                if change > 0:
                    constrained[strategy] = current + max_change
                else:
                    constrained[strategy] = current - max_change
            else:
                constrained[strategy] = target
        
        # Normalize to ensure allocations sum to 100%
        total = sum(constrained.values())
        if total > 0:
            for strategy in constrained:
                constrained[strategy] = (constrained[strategy] / total) * 100.0
        
        return constrained

    def _update_strategy_allocations(self):
        """Update strategy allocations in the core context"""
        for strategy, allocation in self.current_allocations.items():
            self.core_context.upsert_strategy(strategy, {
                "allocation_pct": allocation
            })

    def get_allocation_history(self, days=30):
        """Get allocation history for the specified number of days"""
        with self.lock:
            cutoff = datetime.now() - timedelta(days=days)
            history = []
            
            for entry in self.allocation_history:
                timestamp = datetime.fromisoformat(entry["timestamp"])
                if timestamp >= cutoff:
                    history.append(entry)
            
            return history

    def get_allocation_chart_data(self):
        """Get allocation data in a format suitable for charting"""
        with self.lock:
            # Prepare data structure
            data = {
                "timestamps": [],
                "strategies": {},
                "regimes": []
            }
            
            # Initialize strategy data series
            for strategy in self.current_allocations:
                data["strategies"][strategy] = []
            
            # Process history entries
            for entry in sorted(self.allocation_history, key=lambda x: x["timestamp"]):
                timestamp = entry["timestamp"]
                data["timestamps"].append(timestamp)
                data["regimes"].append(entry["regime"])
                
                # Add allocation for each strategy
                for strategy in data["strategies"]:
                    allocation = entry["new_allocations"].get(strategy, 0.0)
                    data["strategies"][strategy].append(allocation)
            
            # Add current point if not in history
            if not self.allocation_history or self.allocation_history[-1]["timestamp"] != datetime.now().isoformat():
                data["timestamps"].append(datetime.now().isoformat())
                data["regimes"].append(getattr(self.core_context.market_context, "regime", "unknown"))
                
                for strategy in data["strategies"]:
                    allocation = self.current_allocations.get(strategy, 0.0)
                    data["strategies"][strategy].append(allocation)
            
            return data
            
    def save_state(self, filepath):
        """Save rotator state to a file"""
        with self.lock:
            state = {
                "current_allocations": self.current_allocations,
                "last_rotation": self.last_rotation.isoformat(),
                "prev_regime": getattr(self, "_prev_regime", None),
                "allocation_history": self.allocation_history
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
    
    def load_state(self, filepath):
        """Load rotator state from a file"""
        with self.lock:
            if not os.path.exists(filepath):
                logger.warning(f"State file not found: {filepath}")
                return False
            
            try:
                with open(filepath, 'r') as f:
                    state = json.load(f)
                
                self.current_allocations = state.get("current_allocations", {})
                self.last_rotation = datetime.fromisoformat(state.get("last_rotation", datetime.now().isoformat()))
                self._prev_regime = state.get("prev_regime")
                self.allocation_history = state.get("allocation_history", [])
                
                # Update strategies with loaded allocations
                self._update_strategy_allocations()
                
                logger.info(f"Loaded strategy rotator state from {filepath}")
                return True
            except Exception as e:
                logger.error(f"Error loading rotator state: {str(e)}")
                return False 