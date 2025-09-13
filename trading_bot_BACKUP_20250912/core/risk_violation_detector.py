"""
Risk Violation Detector for identifying risk management issues during staging.
"""
import logging
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timedelta
import pandas as pd

from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.core.event_bus import EventBus
from trading_bot.core.constants import EventType

logger = logging.getLogger(__name__)

class RiskViolationDetector:
    """
    Monitors strategies during staging to detect any risk management violations.
    
    This helps identify strategies that might exceed risk parameters
    before they go live with real capital.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the risk violation detector.
        
        Args:
            config: Configuration dictionary with risk thresholds
        """
        self.service_registry = ServiceRegistry.get_instance()
        self.service_registry.register_service("risk_violation_detector", self)
        
        # Get event bus
        self.event_bus = self.service_registry.get_service("event_bus")
        if not self.event_bus:
            logger.error("Event bus service not available")
            self.event_bus = EventBus()
        
        # Configuration
        self.config = config or {}
        self.risk_tolerance_multiplier = self.config.get("risk_tolerance_multiplier", 0.8)
        
        # Default risk thresholds (will be overridden by any strategy-specific thresholds)
        self.default_thresholds = {
            "max_position_size_pct": self.config.get("max_position_size_pct", 5.0),
            "max_daily_loss_pct": self.config.get("max_daily_loss_pct", -3.0),
            "max_drawdown_pct": self.config.get("max_drawdown_pct", -10.0),
            "max_strategy_correlation": self.config.get("max_strategy_correlation", 0.7),
            "max_asset_concentration_pct": self.config.get("max_asset_concentration_pct", 20.0),
            "max_sector_concentration_pct": self.config.get("max_sector_concentration_pct", 30.0),
            "max_trade_frequency_per_day": self.config.get("max_trade_frequency_per_day", 100),
            "min_risk_reward_ratio": self.config.get("min_risk_reward_ratio", 1.5),
        }
        
        # Strategy-specific thresholds
        self.strategy_thresholds: Dict[str, Dict[str, float]] = {}
        
        # Violation tracking
        self.violations: Dict[str, List[Dict[str, Any]]] = {}
        self.active_violations: Set[str] = set()
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # Subscribe to events
        self._subscribe_to_events()
        
        logger.info("Risk violation detector initialized")
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events for monitoring."""
        if not self.event_bus:
            return
            
        self.event_bus.subscribe(EventType.ORDER_PLACED, self._on_order_placed)
        self.event_bus.subscribe(EventType.POSITION_UPDATE, self._on_position_update)
        self.event_bus.subscribe(EventType.STRATEGY_UPDATE, self._on_strategy_update)
        
        logger.info("Subscribed to trading events for risk monitoring")
    
    def set_strategy_threshold(self, strategy_id: str, threshold_name: str, value: float) -> None:
        """
        Set a strategy-specific risk threshold.
        
        Args:
            strategy_id: Strategy identifier
            threshold_name: Name of the threshold
            value: Threshold value
        """
        if strategy_id not in self.strategy_thresholds:
            self.strategy_thresholds[strategy_id] = {}
            
        self.strategy_thresholds[strategy_id][threshold_name] = value
        logger.info(f"Set {threshold_name} = {value} for strategy {strategy_id}")
    
    def get_threshold(self, strategy_id: str, threshold_name: str) -> float:
        """
        Get the threshold value for a strategy, with fallback to default.
        
        Args:
            strategy_id: Strategy identifier
            threshold_name: Name of the threshold
            
        Returns:
            float: Threshold value
        """
        # Check for strategy-specific threshold
        if strategy_id in self.strategy_thresholds and threshold_name in self.strategy_thresholds[strategy_id]:
            return self.strategy_thresholds[strategy_id][threshold_name]
            
        # Fall back to default
        if threshold_name in self.default_thresholds:
            return self.default_thresholds[threshold_name]
            
        # If not found, return a safe default
        logger.warning(f"No threshold found for {threshold_name}, using default")
        return 0.0
    
    def register_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Register a callback for risk violation alerts.
        
        Args:
            callback: Function to call with violation info
        """
        self.alert_callbacks.append(callback)
    
    def _on_order_placed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle order placed events to check for risk violations.
        
        Args:
            event_data: Order event data
        """
        strategy_id = event_data.get("strategy_id")
        if not strategy_id:
            return
            
        # Check position size
        self._check_position_size_violation(strategy_id, event_data)
        
        # Check trade frequency
        self._check_trade_frequency_violation(strategy_id)
    
    def _on_position_update(self, event_data: Dict[str, Any]) -> None:
        """
        Handle position update events to check for risk violations.
        
        Args:
            event_data: Position event data
        """
        strategy_id = event_data.get("strategy_id")
        if not strategy_id:
            return
            
        # Check for drawdown violation
        self._check_drawdown_violation(strategy_id, event_data)
        
        # Check for daily loss violation
        self._check_daily_loss_violation(strategy_id, event_data)
        
        # Check for concentration violations
        self._check_concentration_violations(strategy_id)
    
    def _on_strategy_update(self, event_data: Dict[str, Any]) -> None:
        """
        Handle strategy update events.
        
        Args:
            event_data: Strategy event data
        """
        # Check for correlation violations across strategies
        self._check_strategy_correlations()
    
    def _check_position_size_violation(self, strategy_id: str, order_data: Dict[str, Any]) -> None:
        """
        Check if an order would violate position size limits.
        
        Args:
            strategy_id: Strategy identifier
            order_data: Order data
        """
        # Get threshold value adjusted by risk tolerance
        threshold = self.get_threshold(strategy_id, "max_position_size_pct") * self.risk_tolerance_multiplier
        
        # Get performance tracker to check account size
        performance_tracker = self.service_registry.get_service("performance_tracker")
        if not performance_tracker:
            return
            
        # Calculate position size as percentage of account
        account_value = performance_tracker.get_strategy_account_value(strategy_id)
        if not account_value or account_value <= 0:
            return
            
        order_value = order_data.get("quantity", 0) * order_data.get("price", 0)
        position_pct = (order_value / account_value) * 100
        
        if position_pct > threshold:
            violation = {
                "type": "position_size",
                "strategy_id": strategy_id,
                "timestamp": datetime.now().isoformat(),
                "details": {
                    "position_pct": position_pct,
                    "threshold_pct": threshold,
                    "order_value": order_value,
                    "account_value": account_value,
                    "symbol": order_data.get("symbol")
                }
            }
            
            self._record_violation(strategy_id, violation)
    
    def _check_drawdown_violation(self, strategy_id: str, position_data: Dict[str, Any]) -> None:
        """
        Check if a strategy has exceeded maximum drawdown.
        
        Args:
            strategy_id: Strategy identifier
            position_data: Position update data
        """
        # Get threshold value adjusted by risk tolerance
        threshold = self.get_threshold(strategy_id, "max_drawdown_pct") * self.risk_tolerance_multiplier
        
        # Get performance tracker to check drawdown
        performance_tracker = self.service_registry.get_service("performance_tracker")
        if not performance_tracker:
            return
            
        metrics = performance_tracker.get_strategy_metrics(strategy_id)
        if not metrics:
            return
            
        max_drawdown = metrics.get("max_drawdown_pct", 0)
        
        # Note: max_drawdown is typically negative, and threshold is negative
        # So we're checking if drawdown is worse (more negative) than threshold
        if max_drawdown < threshold:
            violation = {
                "type": "max_drawdown",
                "strategy_id": strategy_id,
                "timestamp": datetime.now().isoformat(),
                "details": {
                    "max_drawdown_pct": max_drawdown,
                    "threshold_pct": threshold
                }
            }
            
            self._record_violation(strategy_id, violation)
    
    def _check_daily_loss_violation(self, strategy_id: str, position_data: Dict[str, Any]) -> None:
        """
        Check if a strategy has exceeded daily loss limit.
        
        Args:
            strategy_id: Strategy identifier
            position_data: Position update data
        """
        # Get threshold value adjusted by risk tolerance
        threshold = self.get_threshold(strategy_id, "max_daily_loss_pct") * self.risk_tolerance_multiplier
        
        # Get performance tracker
        performance_tracker = self.service_registry.get_service("performance_tracker")
        if not performance_tracker:
            return
            
        metrics = performance_tracker.get_strategy_metrics(strategy_id)
        if not metrics:
            return
            
        daily_return = metrics.get("daily_return_pct", 0)
        
        # Check if daily return is below threshold (more negative)
        if daily_return < threshold:
            violation = {
                "type": "daily_loss",
                "strategy_id": strategy_id,
                "timestamp": datetime.now().isoformat(),
                "details": {
                    "daily_loss_pct": daily_return,
                    "threshold_pct": threshold
                }
            }
            
            self._record_violation(strategy_id, violation)
    
    def _check_trade_frequency_violation(self, strategy_id: str) -> None:
        """
        Check if a strategy is trading too frequently.
        
        Args:
            strategy_id: Strategy identifier
        """
        # Get threshold value
        threshold = self.get_threshold(strategy_id, "max_trade_frequency_per_day")
        
        # Get performance tracker
        performance_tracker = self.service_registry.get_service("performance_tracker")
        if not performance_tracker:
            return
            
        # Get today's trades
        today = datetime.now().date()
        trades = performance_tracker.get_trades_by_date(strategy_id, today)
        
        if len(trades) > threshold:
            violation = {
                "type": "trade_frequency",
                "strategy_id": strategy_id,
                "timestamp": datetime.now().isoformat(),
                "details": {
                    "trade_count": len(trades),
                    "threshold": threshold,
                    "date": today.isoformat()
                }
            }
            
            self._record_violation(strategy_id, violation)
    
    def _check_concentration_violations(self, strategy_id: str) -> None:
        """
        Check for asset and sector concentration violations.
        
        Args:
            strategy_id: Strategy identifier
        """
        # Get broker router
        broker_router = self.service_registry.get_service("strategy_broker_router")
        if not broker_router:
            return
            
        # Get broker for strategy
        broker = broker_router.get_broker_for_strategy(strategy_id)
        if not broker:
            return
            
        # Get positions
        try:
            positions = broker.get_positions(strategy_id=strategy_id)
        except Exception:
            return
            
        if not positions:
            return
            
        # Get account value
        account_value = sum(p.get("market_value", 0) for p in positions)
        if account_value <= 0:
            return
            
        # Check asset concentration
        asset_threshold = self.get_threshold(strategy_id, "max_asset_concentration_pct") * self.risk_tolerance_multiplier
        
        # Group by symbol
        symbol_values = {}
        for position in positions:
            symbol = position.get("symbol")
            market_value = position.get("market_value", 0)
            if symbol:
                symbol_values[symbol] = symbol_values.get(symbol, 0) + market_value
        
        # Check each symbol's concentration
        for symbol, value in symbol_values.items():
            concentration = (value / account_value) * 100
            if concentration > asset_threshold:
                violation = {
                    "type": "asset_concentration",
                    "strategy_id": strategy_id,
                    "timestamp": datetime.now().isoformat(),
                    "details": {
                        "symbol": symbol,
                        "concentration_pct": concentration,
                        "threshold_pct": asset_threshold,
                        "market_value": value,
                        "account_value": account_value
                    }
                }
                
                self._record_violation(strategy_id, violation)
        
        # Sector concentration would be similar but requires additional data about sectors
    
    def _check_strategy_correlations(self) -> None:
        """Check for excessive correlation between strategies."""
        # This would require:
        # 1. Getting performance data for all strategies
        # 2. Calculating correlation matrix
        # 3. Checking for pairs with correlation above threshold
        
        # For simplicity, this implementation is placeholder
        # A full implementation would need daily/hourly returns for each strategy
        # to calculate meaningful correlations
        pass
    
    def _record_violation(self, strategy_id: str, violation: Dict[str, Any]) -> None:
        """
        Record a risk violation and send alerts.
        
        Args:
            strategy_id: Strategy identifier
            violation: Violation details
        """
        # Initialize violations list for this strategy if needed
        if strategy_id not in self.violations:
            self.violations[strategy_id] = []
            
        # Add to violations list
        self.violations[strategy_id].append(violation)
        
        # Track as active violation
        violation_key = f"{strategy_id}:{violation['type']}"
        self.active_violations.add(violation_key)
        
        # Log the violation
        logger.warning(f"Risk violation detected: {violation['type']} for strategy {strategy_id}")
        
        # Send alerts to callbacks
        for callback in self.alert_callbacks:
            try:
                callback(violation['type'], violation)
            except Exception as e:
                logger.error(f"Error in violation alert callback: {str(e)}")
    
    def clear_violation(self, strategy_id: str, violation_type: str) -> None:
        """
        Clear a specific violation type for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            violation_type: Type of violation to clear
        """
        violation_key = f"{strategy_id}:{violation_type}"
        if violation_key in self.active_violations:
            self.active_violations.remove(violation_key)
            logger.info(f"Cleared {violation_type} violation for strategy {strategy_id}")
    
    def has_active_violations(self, strategy_id: str) -> bool:
        """
        Check if a strategy has any active violations.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            bool: True if violations exist
        """
        return any(f"{strategy_id}:{vtype}" in self.active_violations 
                  for vtype in ["position_size", "max_drawdown", "daily_loss", 
                                "trade_frequency", "asset_concentration"])
    
    def get_violations(self, strategy_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all violations, optionally filtered by strategy.
        
        Args:
            strategy_id: Optional strategy filter
            
        Returns:
            Dict: Violations by strategy
        """
        if strategy_id:
            return {strategy_id: self.violations.get(strategy_id, [])}
        return self.violations
    
    def get_active_violations_count(self) -> int:
        """Get count of currently active violations."""
        return len(self.active_violations)
    
    def generate_risk_report(self) -> Dict[str, Any]:
        """Generate a comprehensive risk report."""
        # Count violations by type
        violation_counts = {}
        for strategy_violations in self.violations.values():
            for violation in strategy_violations:
                vtype = violation["type"]
                violation_counts[vtype] = violation_counts.get(vtype, 0) + 1
        
        # Count violations by strategy
        strategy_counts = {strategy_id: len(violations) 
                          for strategy_id, violations in self.violations.items()}
        
        # Strategies with current active violations
        active_violation_strategies = {key.split(":")[0] for key in self.active_violations}
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_violations": sum(len(v) for v in self.violations.values()),
            "active_violations": len(self.active_violations),
            "violation_types": violation_counts,
            "violations_by_strategy": strategy_counts,
            "strategies_with_active_violations": list(active_violation_strategies)
        }
