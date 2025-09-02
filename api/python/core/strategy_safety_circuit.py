"""
Strategy Safety Circuit - Implements critical safety measures for live trading strategies.

This module provides automated safety protocols for strategies transitioning from
paper to live trading, including:
- Position size ramping
- Automated circuit breakers
- Correlation monitoring
- Shadow paper trading
"""
import logging
from typing import Dict, List, Any, Optional, Set, Callable
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from threading import Lock

from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.core.event_bus import EventBus
from trading_bot.core.constants import EventType, StrategyStatus, StrategyPhase

logger = logging.getLogger(__name__)

class StrategySafetyCircuit:
    """
    Implements safety measures for live trading strategies.
    
    Key features:
    1. Position Size Ramping: Gradually increases position size for new live strategies
    2. Automated Circuit Breakers: Suspends strategies that breach risk thresholds
    3. Correlation Monitoring: Warns about highly correlated strategies
    4. Shadow Trading: Continues paper trading alongside live for comparison
    """
    
    def __init__(self):
        """Initialize the strategy safety circuit."""
        self.service_registry = ServiceRegistry.get_instance()
        self.service_registry.register_service("strategy_safety_circuit", self)
        
        # Get event bus
        self.event_bus = self.service_registry.get_service("event_bus")
        if not self.event_bus:
            logger.error("Event bus service not available")
            self.event_bus = EventBus()
            
        # Strategy safety settings
        self.safety_settings: Dict[str, Dict[str, Any]] = {}
        
        # Strategy suspension tracking
        self.suspended_strategies: Set[str] = set()
        self.suspension_lock = Lock()
        
        # Correlation tracking
        self.strategy_returns: Dict[str, List[float]] = {}
        
        # Shadow trading tracking
        self.shadow_trading_enabled: Dict[str, bool] = {}
        
        # Subscribe to events
        self._subscribe_to_events()
        
        logger.info("Strategy safety circuit initialized")
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events."""
        self.event_bus.subscribe(EventType.STRATEGY_UPDATE, self._on_strategy_update)
        self.event_bus.subscribe(EventType.POSITION_UPDATE, self._on_position_update)
        self.event_bus.subscribe(EventType.ORDER_PLACED, self._on_order_placed)
        self.event_bus.subscribe(EventType.PAPER_TO_LIVE_TRANSITION, self._on_strategy_promotion)
        
        logger.info("Subscribed to trading events for safety monitoring")
    
    def setup_position_ramping(
        self, 
        strategy_id: str, 
        initial_size_pct: float = 0.25, 
        target_size_pct: float = 1.0,
        ramp_days: int = 30,
        steps: int = 3
    ) -> None:
        """
        Setup position size ramping for a newly promoted strategy.
        
        Args:
            strategy_id: Strategy identifier
            initial_size_pct: Initial position size as percentage of target
            target_size_pct: Target final position size percentage
            ramp_days: Days over which to ramp up position size
            steps: Number of steps to increase position size
        """
        if strategy_id not in self.safety_settings:
            self.safety_settings[strategy_id] = {}
            
        # Calculate step sizes and dates
        step_pct = (target_size_pct - initial_size_pct) / steps
        days_per_step = ramp_days / steps
        
        ramp_schedule = []
        current_pct = initial_size_pct
        current_day = 0
        
        for i in range(steps):
            next_pct = current_pct + step_pct
            next_day = current_day + days_per_step
            
            ramp_schedule.append({
                "day": int(next_day),
                "position_size_pct": next_pct,
                "date": (datetime.now() + timedelta(days=next_day)).date().isoformat()
            })
            
            current_pct = next_pct
            current_day = next_day
        
        self.safety_settings[strategy_id]["position_ramping"] = {
            "enabled": True,
            "current_size_pct": initial_size_pct,
            "target_size_pct": target_size_pct,
            "start_date": datetime.now().date().isoformat(),
            "schedule": ramp_schedule
        }
        
        logger.info(f"Position ramping configured for {strategy_id} starting at {initial_size_pct*100}%")
    
    def setup_circuit_breakers(
        self, 
        strategy_id: str,
        daily_loss_pct: float = -5.0,
        max_drawdown_pct: float = -10.0,
        max_consecutive_losses: int = 5,
        auto_suspend: bool = True
    ) -> None:
        """
        Setup circuit breakers for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            daily_loss_pct: Daily loss percentage that triggers circuit breaker
            max_drawdown_pct: Maximum drawdown percentage that triggers circuit breaker
            max_consecutive_losses: Number of consecutive losses that triggers circuit breaker
            auto_suspend: Whether to automatically suspend the strategy when breached
        """
        if strategy_id not in self.safety_settings:
            self.safety_settings[strategy_id] = {}
        
        self.safety_settings[strategy_id]["circuit_breakers"] = {
            "enabled": True,
            "daily_loss_pct": daily_loss_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "max_consecutive_losses": max_consecutive_losses,
            "auto_suspend": auto_suspend,
            "consecutive_losses": 0,  # Current count
            "last_reset_date": datetime.now().date().isoformat()
        }
        
        logger.info(f"Circuit breakers configured for {strategy_id}")
    
    def enable_shadow_trading(self, strategy_id: str, enabled: bool = True) -> None:
        """
        Enable or disable shadow paper trading for a live strategy.
        
        Args:
            strategy_id: Strategy identifier
            enabled: Whether to enable shadow trading
        """
        self.shadow_trading_enabled[strategy_id] = enabled
        
        logger.info(f"Shadow trading {'enabled' if enabled else 'disabled'} for {strategy_id}")
    
    def get_current_position_size_pct(self, strategy_id: str) -> float:
        """
        Get current position size percentage for a strategy based on ramping schedule.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            float: Current position size percentage (0.0-1.0)
        """
        if (strategy_id not in self.safety_settings or 
            "position_ramping" not in self.safety_settings[strategy_id] or
            not self.safety_settings[strategy_id]["position_ramping"]["enabled"]):
            return 1.0
        
        ramping = self.safety_settings[strategy_id]["position_ramping"]
        start_date = datetime.fromisoformat(ramping["start_date"])
        days_active = (datetime.now().date() - start_date.date()).days
        
        # Check if we're at the end of the schedule
        if not ramping["schedule"] or days_active >= ramping["schedule"][-1]["day"]:
            return ramping["target_size_pct"]
        
        # Find the appropriate step in the schedule
        for step in ramping["schedule"]:
            if days_active < step["day"]:
                return ramping["current_size_pct"]
            ramping["current_size_pct"] = step["position_size_pct"]
        
        return ramping["current_size_pct"]
    
    def is_strategy_suspended(self, strategy_id: str) -> bool:
        """
        Check if a strategy is currently suspended by safety circuit.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            bool: True if suspended
        """
        return strategy_id in self.suspended_strategies
    
    def suspend_strategy(self, strategy_id: str, reason: str) -> bool:
        """
        Suspend a strategy due to safety concerns.
        
        Args:
            strategy_id: Strategy identifier
            reason: Reason for suspension
            
        Returns:
            bool: Success status
        """
        with self.suspension_lock:
            # Don't suspend if already suspended
            if strategy_id in self.suspended_strategies:
                return True
            
            # Get strategy workflow
            workflow = self.service_registry.get_service("strategy_trial_workflow")
            if not workflow:
                logger.error("Strategy workflow service not available")
                return False
            
            # Suspend strategy
            try:
                success = workflow.pause_strategy(strategy_id)
                
                if success:
                    self.suspended_strategies.add(strategy_id)
                    
                    # Publish event
                    self.event_bus.publish(
                        EventType.STRATEGY_UPDATE,
                        {
                            "strategy_id": strategy_id,
                            "status": "PAUSED",
                            "reason": f"Safety circuit: {reason}",
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    
                    logger.warning(f"Strategy {strategy_id} suspended by safety circuit: {reason}")
                    return True
                else:
                    logger.error(f"Failed to suspend strategy {strategy_id}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error suspending strategy {strategy_id}: {str(e)}")
                return False
    
    def resume_strategy(self, strategy_id: str) -> bool:
        """
        Resume a previously suspended strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            bool: Success status
        """
        with self.suspension_lock:
            # Not suspended
            if strategy_id not in self.suspended_strategies:
                return True
            
            # Get strategy workflow
            workflow = self.service_registry.get_service("strategy_trial_workflow")
            if not workflow:
                logger.error("Strategy workflow service not available")
                return False
            
            # Resume strategy
            try:
                success = workflow.resume_strategy(strategy_id)
                
                if success:
                    self.suspended_strategies.remove(strategy_id)
                    
                    # Reset circuit breaker counters
                    if (strategy_id in self.safety_settings and 
                        "circuit_breakers" in self.safety_settings[strategy_id]):
                        self.safety_settings[strategy_id]["circuit_breakers"]["consecutive_losses"] = 0
                    
                    # Publish event
                    self.event_bus.publish(
                        EventType.STRATEGY_UPDATE,
                        {
                            "strategy_id": strategy_id,
                            "status": "ACTIVE",
                            "reason": "Safety circuit: Manually resumed",
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    
                    logger.info(f"Strategy {strategy_id} resumed")
                    return True
                else:
                    logger.error(f"Failed to resume strategy {strategy_id}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error resuming strategy {strategy_id}: {str(e)}")
                return False
    
    def _on_strategy_update(self, event_data: Dict[str, Any]) -> None:
        """
        Handle strategy update events.
        
        Args:
            event_data: Strategy event data
        """
        strategy_id = event_data.get("strategy_id")
        if not strategy_id:
            return
            
        # Update daily return tracking for correlation monitoring
        performance_tracker = self.service_registry.get_service("performance_tracker")
        if performance_tracker:
            try:
                metrics = performance_tracker.get_strategy_metrics(strategy_id)
                daily_return = metrics.get("daily_return_pct", 0)
                
                if strategy_id not in self.strategy_returns:
                    self.strategy_returns[strategy_id] = []
                
                self.strategy_returns[strategy_id].append(daily_return)
                
                # Limit history length
                if len(self.strategy_returns[strategy_id]) > 30:
                    self.strategy_returns[strategy_id] = self.strategy_returns[strategy_id][-30:]
                
                # Check correlations if we have enough data
                self._check_strategy_correlations()
                
            except Exception as e:
                logger.error(f"Error updating strategy returns: {str(e)}")
    
    def _on_position_update(self, event_data: Dict[str, Any]) -> None:
        """
        Handle position update events to check for circuit breakers.
        
        Args:
            event_data: Position event data
        """
        strategy_id = event_data.get("strategy_id")
        if not strategy_id:
            return
            
        # Skip if strategy is not in live mode
        workflow = self.service_registry.get_service("strategy_trial_workflow")
        if workflow:
            strategy_info = workflow.get_strategy_info(strategy_id)
            if strategy_info and strategy_info.get("phase") != "LIVE":
                return
                
        # Check for circuit breaker conditions
        self._check_circuit_breakers(strategy_id)
    
    def _on_order_placed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle order placed events for position size adjustment.
        
        Args:
            event_data: Order event data
        """
        strategy_id = event_data.get("strategy_id")
        if not strategy_id:
            return
            
        # Skip if strategy is not in live mode
        workflow = self.service_registry.get_service("strategy_trial_workflow")
        if workflow:
            strategy_info = workflow.get_strategy_info(strategy_id)
            if strategy_info and strategy_info.get("phase") != "LIVE":
                return
        
        # Handle position sizing if ramping is enabled
        if (strategy_id in self.safety_settings and 
            "position_ramping" in self.safety_settings[strategy_id] and
            self.safety_settings[strategy_id]["position_ramping"]["enabled"]):
            
            # Get current position size percentage
            current_pct = self.get_current_position_size_pct(strategy_id)
            
            # Adjust quantity if needed
            if current_pct < 1.0:
                original_quantity = event_data.get("quantity", 0)
                adjusted_quantity = int(original_quantity * current_pct)
                
                if adjusted_quantity != original_quantity:
                    logger.info(f"Adjusted order quantity for {strategy_id} from {original_quantity} to {adjusted_quantity} ({current_pct*100:.0f}% position size)")
                    
                    # We can't modify the event directly, but we can publish a new event
                    # with the adjusted quantity that downstream handlers can use
                    self.event_bus.publish(
                        EventType.ORDER_ADJUSTED,
                        {
                            **event_data,
                            "original_quantity": original_quantity,
                            "adjusted_quantity": adjusted_quantity,
                            "adjustment_reason": "Position size ramping",
                            "position_size_pct": current_pct
                        }
                    )
    
    def _on_strategy_promotion(self, event_data: Dict[str, Any]) -> None:
        """
        Handle strategy promotion events to setup safety measures.
        
        Args:
            event_data: Promotion event data
        """
        strategy_id = event_data.get("strategy_id")
        if not strategy_id:
            return
            
        logger.info(f"Setting up safety measures for newly promoted strategy {strategy_id}")
        
        # Setup default safety measures
        self.setup_position_ramping(
            strategy_id=strategy_id,
            initial_size_pct=0.25,
            target_size_pct=1.0,
            ramp_days=30,
            steps=3
        )
        
        self.setup_circuit_breakers(
            strategy_id=strategy_id,
            daily_loss_pct=-5.0,
            max_drawdown_pct=-10.0,
            max_consecutive_losses=5,
            auto_suspend=True
        )
        
        # Enable shadow trading
        self.enable_shadow_trading(strategy_id)
    
    def _check_circuit_breakers(self, strategy_id: str) -> None:
        """
        Check if a strategy has triggered any circuit breakers.
        
        Args:
            strategy_id: Strategy identifier
        """
        if (strategy_id not in self.safety_settings or 
            "circuit_breakers" not in self.safety_settings[strategy_id] or
            not self.safety_settings[strategy_id]["circuit_breakers"]["enabled"]):
            return
            
        # Get performance metrics
        performance_tracker = self.service_registry.get_service("performance_tracker")
        if not performance_tracker:
            return
            
        try:
            metrics = performance_tracker.get_strategy_metrics(strategy_id)
            if not metrics:
                return
                
            circuit_breakers = self.safety_settings[strategy_id]["circuit_breakers"]
            
            # Check daily loss
            daily_return = metrics.get("daily_return_pct", 0)
            if daily_return <= circuit_breakers["daily_loss_pct"]:
                if circuit_breakers["auto_suspend"]:
                    self.suspend_strategy(
                        strategy_id=strategy_id,
                        reason=f"Daily loss threshold exceeded: {daily_return:.2f}% (threshold: {circuit_breakers['daily_loss_pct']:.2f}%)"
                    )
                return  # Exit after suspension
            
            # Check drawdown
            max_drawdown = metrics.get("max_drawdown_pct", 0)
            if max_drawdown <= circuit_breakers["max_drawdown_pct"]:
                if circuit_breakers["auto_suspend"]:
                    self.suspend_strategy(
                        strategy_id=strategy_id,
                        reason=f"Max drawdown threshold exceeded: {max_drawdown:.2f}% (threshold: {circuit_breakers['max_drawdown_pct']:.2f}%)"
                    )
                return  # Exit after suspension
            
            # Check consecutive losses on new trade
            recent_trades = performance_tracker.get_recent_trades(strategy_id, limit=10)
            if recent_trades:
                latest_trade = recent_trades[0]
                
                # If this is a new losing trade
                if latest_trade.get("pnl", 0) < 0:
                    circuit_breakers["consecutive_losses"] += 1
                    
                    if circuit_breakers["consecutive_losses"] >= circuit_breakers["max_consecutive_losses"]:
                        if circuit_breakers["auto_suspend"]:
                            self.suspend_strategy(
                                strategy_id=strategy_id,
                                reason=f"Consecutive losses threshold exceeded: {circuit_breakers['consecutive_losses']} (threshold: {circuit_breakers['max_consecutive_losses']})"
                            )
                else:
                    # Reset consecutive losses on winning trade
                    circuit_breakers["consecutive_losses"] = 0
                    
        except Exception as e:
            logger.error(f"Error checking circuit breakers for {strategy_id}: {str(e)}")
    
    def _check_strategy_correlations(self) -> None:
        """Check for high correlations between strategies."""
        # Need at least 2 strategies with data
        if len(self.strategy_returns) < 2:
            return
            
        # Need at least 5 data points for meaningful correlation
        strategies_with_data = {s: returns for s, returns in self.strategy_returns.items() 
                               if len(returns) >= 5}
        
        if len(strategies_with_data) < 2:
            return
            
        try:
            # Convert to DataFrame for correlation calculation
            df = pd.DataFrame(strategies_with_data)
            
            # Calculate correlation matrix
            corr_matrix = df.corr()
            
            # Check for high correlations
            high_correlations = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    s1 = corr_matrix.columns[i]
                    s2 = corr_matrix.columns[j]
                    corr = corr_matrix.iloc[i, j]
                    
                    if abs(corr) >= 0.7:  # High correlation threshold
                        high_correlations.append({
                            "strategy1": s1,
                            "strategy2": s2,
                            "correlation": corr
                        })
            
            # Report high correlations
            if high_correlations:
                logger.warning(f"High strategy correlations detected: {high_correlations}")
                
                # Publish correlation event
                self.event_bus.publish(
                    EventType.RISK_ALERT,
                    {
                        "alert_type": "strategy_correlation",
                        "timestamp": datetime.now().isoformat(),
                        "correlations": high_correlations
                    }
                )
                
        except Exception as e:
            logger.error(f"Error checking strategy correlations: {str(e)}")
    
    def get_safety_status(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get safety status for one or all strategies.
        
        Args:
            strategy_id: Optional strategy identifier
            
        Returns:
            Dict: Safety status
        """
        if strategy_id:
            if strategy_id not in self.safety_settings:
                return {"strategy_id": strategy_id, "safety_measures": []}
                
            return {
                "strategy_id": strategy_id,
                "suspended": strategy_id in self.suspended_strategies,
                "safety_measures": self.safety_settings[strategy_id],
                "current_position_size_pct": self.get_current_position_size_pct(strategy_id),
                "shadow_trading_enabled": self.shadow_trading_enabled.get(strategy_id, False)
            }
        else:
            # Return status for all strategies
            all_status = {}
            for s_id in self.safety_settings:
                all_status[s_id] = self.get_safety_status(s_id)
            return all_status
