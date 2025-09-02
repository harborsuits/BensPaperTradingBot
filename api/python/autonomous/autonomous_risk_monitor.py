#!/usr/bin/env python3
"""
Autonomous Risk Monitor

This module continuously monitors risk conditions and autonomously makes 
adjustments to strategy allocations, risk parameters, and trading activity
based on predefined rules and market conditions.
"""

import logging
import os
import json
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

# Import risk components
from trading_bot.autonomous.risk_integration import get_autonomous_risk_manager
from trading_bot.autonomous.strategy_deployment_pipeline import get_deployment_pipeline, DeploymentStatus
from trading_bot.risk.risk_manager import RiskLevel, StopLossType
from trading_bot.event_system import EventBus, Event, EventType

logger = logging.getLogger(__name__)

class RiskCondition:
    """Represents a risk condition with threshold and action"""
    
    def __init__(self, 
                 name: str, 
                 metric_path: List[str], 
                 threshold: float,
                 comparison: str,
                 severity: str,
                 action: str,
                 action_params: Optional[Dict[str, Any]] = None):
        """
        Initialize a risk condition.
        
        Args:
            name: Condition name
            metric_path: Path to metric in risk metrics dictionary
            threshold: Threshold value
            comparison: Comparison operator (>, <, >=, <=, ==)
            severity: Severity level (info, warning, critical)
            action: Action to take when triggered
            action_params: Additional parameters for the action
        """
        self.name = name
        self.metric_path = metric_path
        self.threshold = threshold
        self.comparison = comparison
        self.severity = severity
        self.action = action
        self.action_params = action_params or {}
        
        # Statistics
        self.trigger_count = 0
        self.last_triggered = None
        self.currently_triggered = False
    
    def evaluate(self, metrics: Dict[str, Any]) -> bool:
        """
        Evaluate the condition against metrics.
        
        Args:
            metrics: Risk metrics dictionary
            
        Returns:
            bool: True if condition is triggered
        """
        # Extract metric value from nested dictionary
        value = metrics
        for key in self.metric_path:
            if key not in value:
                return False
            value = value[key]
        
        # Handle non-numeric values
        if not isinstance(value, (int, float)):
            return False
        
        # Evaluate comparison
        triggered = False
        if self.comparison == ">":
            triggered = value > self.threshold
        elif self.comparison == "<":
            triggered = value < self.threshold
        elif self.comparison == ">=":
            triggered = value >= self.threshold
        elif self.comparison == "<=":
            triggered = value <= self.threshold
        elif self.comparison == "==":
            triggered = value == self.threshold
        
        # Update statistics
        if triggered:
            self.trigger_count += 1
            self.last_triggered = datetime.now()
            
            if not self.currently_triggered:
                # Log first trigger
                logger.warning(f"Risk condition triggered: {self.name} - {value} {self.comparison} {self.threshold}")
                self.currently_triggered = True
        else:
            if self.currently_triggered:
                # Log condition recovery
                logger.info(f"Risk condition recovered: {self.name}")
                self.currently_triggered = False
        
        return triggered


class AutonomousRiskMonitor:
    """
    Autonomous monitor that continuously evaluates risk conditions and takes 
    actions based on predefined rules.
    """
    
    def __init__(self, 
                 event_bus: Optional[EventBus] = None,
                 config_path: Optional[str] = None):
        """
        Initialize the autonomous risk monitor.
        
        Args:
            event_bus: Event bus for communication
            config_path: Path to configuration file
        """
        self.event_bus = event_bus or EventBus()
        
        # Get components
        self.risk_manager = get_autonomous_risk_manager()
        self.deployment_pipeline = get_deployment_pipeline()
        
        # Risk conditions
        self.conditions = []
        
        # Configuration
        self.config = {
            "monitoring_interval_seconds": 60,
            "max_allocation_percentage": 20.0,
            "correlation_threshold": 0.7,
            "auto_pause_enabled": True,
            "auto_resume_enabled": True,
            "dynamic_allocation_enabled": True,
            "minimum_free_allocation": 20.0  # Keep at least 20% allocation free
        }
        
        # Load config if provided
        if config_path:
            self._load_config(config_path)
        
        # Load default conditions
        self._load_default_conditions()
        
        # State
        self.is_running = False
        self.monitor_thread = None
        self.last_check_time = None
        self.triggered_conditions = {}
        
        # Register for events
        self._register_for_events()
        
        logger.info("Autonomous Risk Monitor initialized")
    
    def _load_config(self, config_path: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                
                # Update config with loaded values
                self.config.update(loaded_config)
                
                logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def _load_default_conditions(self) -> None:
        """Load default risk conditions."""
        self.conditions = [
            # Portfolio-wide conditions
            RiskCondition(
                name="High Portfolio Drawdown",
                metric_path=["portfolio_metrics", "current_drawdown_pct"],
                threshold=12.0,
                comparison=">",
                severity="critical",
                action="reduce_all_allocations",
                action_params={"factor": 0.7}
            ),
            RiskCondition(
                name="Extreme Portfolio Drawdown",
                metric_path=["portfolio_metrics", "current_drawdown_pct"],
                threshold=18.0,
                comparison=">",
                severity="critical",
                action="pause_all_strategies",
                action_params={"reason": "Extreme portfolio drawdown"}
            ),
            RiskCondition(
                name="High Daily Loss",
                metric_path=["portfolio_metrics", "daily_profit_loss_pct"],
                threshold=-3.0,
                comparison="<",
                severity="warning",
                action="reduce_all_allocations",
                action_params={"factor": 0.8}
            ),
            RiskCondition(
                name="Excessive Portfolio Risk",
                metric_path=["portfolio_metrics", "total_portfolio_risk"],
                threshold=85.0,
                comparison=">",
                severity="warning",
                action="reduce_all_allocations",
                action_params={"factor": 0.9}
            ),
            # Strategy-specific conditions handled dynamically at runtime
        ]
        
        logger.info(f"Loaded {len(self.conditions)} default risk conditions")
    
    def _register_for_events(self) -> None:
        """Register for relevant events."""
        self.event_bus.register(EventType.STRATEGY_DEPLOYED_WITH_RISK, self._handle_strategy_deployed)
        self.event_bus.register(EventType.POSITION_CLOSED, self._handle_position_closed)
        self.event_bus.register(EventType.CIRCUIT_BREAKER_TRIGGERED, self._handle_circuit_breaker)
        
        logger.info("Registered for events")
    
    def start_monitoring(self) -> None:
        """Start autonomous risk monitoring."""
        if self.is_running:
            logger.warning("Monitor is already running")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Started autonomous risk monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop autonomous risk monitoring."""
        if not self.is_running:
            logger.warning("Monitor is not running")
            return
        
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("Stopped autonomous risk monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Get current risk metrics
                risk_report = self.risk_manager.get_risk_report()
                
                # Get deployments
                active_deployments = self.deployment_pipeline.get_deployments(
                    status=DeploymentStatus.ACTIVE
                )
                
                # Set timestamp
                self.last_check_time = datetime.now()
                
                # Evaluate conditions
                triggered = self._evaluate_conditions(risk_report, active_deployments)
                
                # Take actions for triggered conditions
                if triggered:
                    self._take_actions(triggered, risk_report, active_deployments)
                
                # Dynamic allocation adjustment (if enabled)
                if self.config["dynamic_allocation_enabled"]:
                    self._adjust_allocations(risk_report, active_deployments)
                
                # Add strategy-specific conditions for any new strategies
                self._update_strategy_conditions(active_deployments)
                
                # Emit monitoring event
                self._emit_monitoring_event(risk_report, triggered)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Sleep until next check
            time.sleep(self.config["monitoring_interval_seconds"])
    
    def _evaluate_conditions(self, 
                            risk_report: Dict[str, Any],
                            deployments: List[Dict[str, Any]]) -> List[RiskCondition]:
        """
        Evaluate all risk conditions.
        
        Args:
            risk_report: Risk report data
            deployments: Active deployments
            
        Returns:
            List of triggered conditions
        """
        triggered = []
        
        # Evaluate each condition
        for condition in self.conditions:
            if condition.evaluate(risk_report):
                triggered.append(condition)
        
        # Strategy-specific evaluations
        for deployment in deployments:
            strategy_id = deployment.get("strategy_id")
            if not strategy_id:
                continue
                
            # Check strategy drawdown
            strategy_metrics = risk_report.get("strategy_metrics", {}).get(strategy_id, {})
            current_drawdown = strategy_metrics.get("risk_metrics", {}).get("current_drawdown", 0)
            
            if current_drawdown > self.config.get("strategy_drawdown_threshold", 25.0):
                # Create dynamic condition
                condition = RiskCondition(
                    name=f"High Strategy Drawdown: {strategy_id}",
                    metric_path=["strategy_metrics", strategy_id, "risk_metrics", "current_drawdown"],
                    threshold=self.config.get("strategy_drawdown_threshold", 25.0),
                    comparison=">",
                    severity="warning",
                    action="pause_strategy",
                    action_params={"strategy_id": strategy_id, "reason": "Excessive drawdown"}
                )
                
                # Mark as triggered
                condition.trigger_count += 1
                condition.last_triggered = datetime.now()
                condition.currently_triggered = True
                
                triggered.append(condition)
        
        return triggered
    
    def _take_actions(self, 
                     triggered_conditions: List[RiskCondition],
                     risk_report: Dict[str, Any],
                     deployments: List[Dict[str, Any]]) -> None:
        """
        Take actions for triggered conditions.
        
        Args:
            triggered_conditions: List of triggered conditions
            risk_report: Risk report data
            deployments: Active deployments
        """
        for condition in triggered_conditions:
            action = condition.action
            params = condition.action_params
            
            logger.info(f"Taking action '{action}' for condition '{condition.name}'")
            
            if action == "reduce_all_allocations":
                factor = params.get("factor", 0.8)
                self._reduce_all_allocations(factor)
                
            elif action == "pause_all_strategies":
                reason = params.get("reason", "Autonomous risk control")
                self._pause_all_strategies(reason)
                
            elif action == "pause_strategy":
                strategy_id = params.get("strategy_id")
                reason = params.get("reason", "Autonomous risk control")
                
                if strategy_id:
                    self._pause_strategy(strategy_id, reason)
                
            elif action == "adjust_risk_level":
                new_level = params.get("risk_level", RiskLevel.HIGH)
                self._adjust_risk_level(new_level)
                
            elif action == "emit_alert":
                alert_type = params.get("alert_type", "RISK_WARNING")
                message = params.get("message", condition.name)
                self._emit_alert(alert_type, message)
    
    def _reduce_all_allocations(self, factor: float) -> None:
        """
        Reduce allocation for all active strategies.
        
        Args:
            factor: Reduction factor (0.0-1.0)
        """
        # Get active deployments
        active_deployments = self.deployment_pipeline.get_deployments(
            status=DeploymentStatus.ACTIVE
        )
        
        adjusted_count = 0
        
        for deployment in active_deployments:
            deployment_id = deployment.get("deployment_id")
            if not deployment_id:
                continue
                
            strategy_id = deployment.get("strategy_id")
            current_allocation = deployment.get("risk_params", {}).get("allocation_percentage", 5.0)
            new_allocation = current_allocation * factor
            
            # Adjust through risk manager
            if self.risk_manager.adjust_allocation(strategy_id, new_allocation):
                adjusted_count += 1
                logger.info(f"Reduced allocation for {strategy_id} from {current_allocation:.1f}% to {new_allocation:.1f}%")
        
        # Emit event
        self._emit_event(
            event_type="ALLOCATION_REDUCED",
            data={
                "factor": factor,
                "affected_strategies": adjusted_count,
                "reason": "Autonomous risk control",
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def _pause_all_strategies(self, reason: str) -> None:
        """
        Pause all active strategies.
        
        Args:
            reason: Reason for pausing
        """
        # Only do this if auto-pause is enabled
        if not self.config["auto_pause_enabled"]:
            logger.info("Auto-pause is disabled, skipping pause all operation")
            return
            
        # Get active deployments
        active_deployments = self.deployment_pipeline.get_deployments(
            status=DeploymentStatus.ACTIVE
        )
        
        paused_count = 0
        
        for deployment in active_deployments:
            deployment_id = deployment.get("deployment_id")
            if not deployment_id:
                continue
                
            # Pause through pipeline
            if self.deployment_pipeline.pause_deployment(deployment_id, reason):
                paused_count += 1
        
        logger.warning(f"Paused {paused_count} strategies: {reason}")
        
        # Emit event
        self._emit_event(
            event_type="ALL_STRATEGIES_PAUSED",
            data={
                "count": paused_count,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def _pause_strategy(self, strategy_id: str, reason: str) -> None:
        """
        Pause a specific strategy.
        
        Args:
            strategy_id: Strategy ID
            reason: Reason for pausing
        """
        # Only do this if auto-pause is enabled
        if not self.config["auto_pause_enabled"]:
            logger.info(f"Auto-pause is disabled, skipping pause for {strategy_id}")
            return
            
        # Get deployment ID
        deployment_id = self.deployment_pipeline.strategy_to_deployment.get(strategy_id)
        
        if not deployment_id:
            logger.warning(f"Cannot find deployment for strategy {strategy_id}")
            return
        
        # Pause through pipeline
        if self.deployment_pipeline.pause_deployment(deployment_id, reason):
            logger.info(f"Paused strategy {strategy_id}: {reason}")
    
    def _adjust_risk_level(self, new_level: RiskLevel) -> None:
        """
        Adjust system risk level.
        
        Args:
            new_level: New risk level
        """
        # Emit risk level change event
        self._emit_event(
            event_type=EventType.RISK_LEVEL_CHANGED,
            data={
                "old_level": self.risk_manager.risk_level,
                "new_level": new_level,
                "timestamp": datetime.now().isoformat(),
                "source": "AutonomousRiskMonitor"
            }
        )
        
        logger.info(f"Adjusted risk level to {new_level}")
    
    def _emit_alert(self, alert_type: str, message: str) -> None:
        """
        Emit risk alert.
        
        Args:
            alert_type: Alert type
            message: Alert message
        """
        self._emit_event(
            event_type=EventType.RISK_ALERT,
            data={
                "alert_type": alert_type,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "source": "AutonomousRiskMonitor"
            }
        )
        
        logger.info(f"Emitted risk alert: {alert_type} - {message}")
    
    def _adjust_allocations(self, 
                           risk_report: Dict[str, Any],
                           deployments: List[Dict[str, Any]]) -> None:
        """
        Dynamically adjust strategy allocations based on performance.
        
        Args:
            risk_report: Risk report data
            deployments: Active deployments
        """
        # Skip if no deployments
        if not deployments:
            return
            
        # Calculate performance metrics
        performance_data = {}
        total_allocation = 0.0
        
        for deployment in deployments:
            strategy_id = deployment.get("strategy_id")
            if not strategy_id:
                continue
                
            # Get current allocation
            current_allocation = deployment.get("risk_params", {}).get("allocation_percentage", 5.0)
            total_allocation += current_allocation
            
            # Get performance data
            profit_loss = deployment.get("performance", {}).get("profit_loss", 0.0)
            win_rate = deployment.get("performance", {}).get("win_rate", 0.0)
            trades = deployment.get("performance", {}).get("trades", 0)
            
            # Skip if not enough trades
            if trades < 5:
                continue
                
            # Calculate performance score
            # Simple score: combination of P&L and win rate
            score = (profit_loss / 1000.0) + (win_rate / 100.0)
            
            performance_data[strategy_id] = {
                "score": score,
                "current_allocation": current_allocation
            }
        
        # Check if we have enough data to adjust
        if len(performance_data) < 2:
            return
            
        # Check if we're at the allocation limit
        free_allocation = 100.0 - total_allocation
        if free_allocation < self.config["minimum_free_allocation"]:
            logger.info(f"Not enough free allocation ({free_allocation:.1f}%) for adjustment")
            return
        
        # Sort strategies by performance score
        sorted_strategies = sorted(
            performance_data.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )
        
        # Adjust allocations
        # Increase allocation for top performer, decrease for bottom performer
        if len(sorted_strategies) >= 2:
            top_strategy = sorted_strategies[0][0]
            bottom_strategy = sorted_strategies[-1][0]
            
            # Don't adjust if the scores are too close
            top_score = sorted_strategies[0][1]["score"]
            bottom_score = sorted_strategies[-1][1]["score"]
            
            if top_score - bottom_score < 0.5:
                return
                
            # Calculate adjustment
            adjustment = min(1.0, free_allocation / 4)  # Small incremental adjustment
            
            # Adjust top strategy up
            top_current = performance_data[top_strategy]["current_allocation"]
            if top_current < self.config["max_allocation_percentage"]:
                self.risk_manager.adjust_allocation(top_strategy, top_current + adjustment)
                logger.info(f"Increased allocation for top performer {top_strategy} by {adjustment:.1f}%")
            
            # Adjust bottom strategy down (but not below minimum)
            bottom_current = performance_data[bottom_strategy]["current_allocation"]
            if bottom_current > 3.0:  # Don't go below 2% allocation
                self.risk_manager.adjust_allocation(bottom_strategy, bottom_current - adjustment)
                logger.info(f"Decreased allocation for bottom performer {bottom_strategy} by {adjustment:.1f}%")
    
    def _update_strategy_conditions(self, deployments: List[Dict[str, Any]]) -> None:
        """
        Update strategy-specific conditions based on active deployments.
        
        Args:
            deployments: Active deployments
        """
        # Skip if no deployments
        if not deployments:
            return
        
        # Get list of strategy IDs that already have conditions
        existing_strategies = set()
        for condition in self.conditions:
            if "strategy_id" in condition.action_params:
                existing_strategies.add(condition.action_params["strategy_id"])
        
        # Add conditions for new strategies
        for deployment in deployments:
            strategy_id = deployment.get("strategy_id")
            if not strategy_id or strategy_id in existing_strategies:
                continue
                
            # Add drawdown condition
            drawdown_condition = RiskCondition(
                name=f"High Drawdown: {strategy_id}",
                metric_path=["strategy_metrics", strategy_id, "risk_metrics", "current_drawdown"],
                threshold=self.config.get("strategy_drawdown_threshold", 25.0),
                comparison=">",
                severity="warning",
                action="pause_strategy",
                action_params={"strategy_id": strategy_id, "reason": "Excessive drawdown"}
            )
            
            # Add daily loss condition
            loss_condition = RiskCondition(
                name=f"High Daily Loss: {strategy_id}",
                metric_path=["strategy_metrics", strategy_id, "risk_metrics", "daily_profit_loss"],
                threshold=-1000.0,  # $1000 daily loss
                comparison="<",
                severity="warning",
                action="pause_strategy",
                action_params={"strategy_id": strategy_id, "reason": "Excessive daily loss"}
            )
            
            # Add conditions
            self.conditions.append(drawdown_condition)
            self.conditions.append(loss_condition)
            
            logger.info(f"Added risk conditions for strategy {strategy_id}")
    
    def _handle_strategy_deployed(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle strategy deployed event."""
        strategy_id = data.get("strategy_id")
        if not strategy_id:
            return
            
        logger.info(f"Strategy deployed: {strategy_id}")
        
        # Make sure we have conditions for this strategy
        self._update_strategy_conditions([{"strategy_id": strategy_id}])
    
    def _handle_position_closed(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle position closed event."""
        strategy_id = data.get("strategy_id")
        if not strategy_id:
            return
            
        profit_loss = data.get("profit_loss", 0.0)
        
        # Log for significant P&L
        if abs(profit_loss) > 500:
            logger.info(f"Significant P&L for {strategy_id}: ${profit_loss:.2f}")
    
    def _handle_circuit_breaker(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle circuit breaker event."""
        reasons = data.get("reasons", [])
        
        logger.warning(f"Circuit breaker triggered: {reasons}")
        
        # If configured for auto-pause, pause all strategies
        if self.config["auto_pause_enabled"]:
            self._pause_all_strategies("Circuit breaker triggered")
    
    def _emit_monitoring_event(self, 
                              risk_report: Dict[str, Any],
                              triggered_conditions: List[RiskCondition]) -> None:
        """
        Emit monitoring event.
        
        Args:
            risk_report: Risk report data
            triggered_conditions: List of triggered conditions
        """
        self._emit_event(
            event_type="RISK_MONITORING_UPDATE",
            data={
                "timestamp": datetime.now().isoformat(),
                "triggered_conditions": [c.name for c in triggered_conditions],
                "risk_level": risk_report.get("portfolio_metrics", {}).get("risk_level", "MEDIUM"),
                "drawdown": risk_report.get("portfolio_metrics", {}).get("current_drawdown_pct", 0.0),
                "active_strategies": risk_report.get("active_strategies", 0)
            }
        )
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Emit event to event bus.
        
        Args:
            event_type: Event type
            data: Event data
        """
        if not self.event_bus:
            return
            
        try:
            event = Event(
                event_type=event_type,
                source="AutonomousRiskMonitor",
                data=data,
                timestamp=datetime.now()
            )
            self.event_bus.publish(event)
        except Exception as e:
            logger.error(f"Error emitting event: {e}")

# Singleton instance for global access
_autonomous_risk_monitor = None

def get_autonomous_risk_monitor(event_bus: Optional[EventBus] = None,
                               config_path: Optional[str] = None) -> AutonomousRiskMonitor:
    """
    Get singleton instance of autonomous risk monitor.
    
    Args:
        event_bus: Event bus for communication
        config_path: Path to configuration file
        
    Returns:
        AutonomousRiskMonitor instance
    """
    global _autonomous_risk_monitor
    
    if _autonomous_risk_monitor is None:
        _autonomous_risk_monitor = AutonomousRiskMonitor(event_bus, config_path)
        
    return _autonomous_risk_monitor
