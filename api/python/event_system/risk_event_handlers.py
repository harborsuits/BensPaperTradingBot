#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk Event Handlers

This module contains handlers for risk-related events, connecting
the risk management system with the UI and other system components.
Following our established event-driven architecture pattern.
"""

import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta

from trading_bot.event_system import Event, EventHandler, EventBus, EventType

logger = logging.getLogger(__name__)

class RiskEventTracker:
    """
    Tracks risk-related events and risk metrics.
    Acts as a central hub for risk data that the UI and other components can query.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the risk event tracker.
        
        Args:
            event_bus: Event bus to register handlers with
        """
        self.risk_alerts = []  # Stores all risk alerts
        self.latest_events = {}  # Stores the latest events by type
        self.strategy_risk = {}  # Stores risk data by strategy_id
        self.circuit_breaker_history = []  # History of circuit breaker triggers
        self.risk_level_changes = []  # History of risk level changes
        
        # Real-time metrics
        self.current_risk_level = "MEDIUM"  # Default risk level
        self.portfolio_metrics = {}  # Current portfolio risk metrics
        
        # Risk thresholds
        self.risk_thresholds = {
            "portfolio_drawdown": 15.0,  # %
            "strategy_drawdown": 25.0,   # %
            "daily_loss": 5.0,           # %
            "trade_frequency": 20,       # per day
            "correlation_threshold": 0.7  # correlation coefficient
        }
        
        self.event_bus = event_bus
        
        # Register with event bus if provided
        if event_bus:
            self._register_handlers()
    
    def _register_handlers(self) -> None:
        """Register event handlers with the event bus."""
        # Risk alert events
        self.event_bus.register_handler(
            EventHandler(
                callback=self._handle_risk_event,
                event_type=EventType.CIRCUIT_BREAKER_TRIGGERED,
                name="circuit_breaker_handler"
            )
        )
        
        self.event_bus.register_handler(
            EventHandler(
                callback=self._handle_risk_event,
                event_type=EventType.RISK_LEVEL_CHANGED,
                name="risk_level_handler"
            )
        )
        
        self.event_bus.register_handler(
            EventHandler(
                callback=self._handle_risk_event,
                event_type=EventType.RISK_ALERT,
                name="risk_alert_handler"
            )
        )
        
        # Strategy risk events
        self.event_bus.register_handler(
            EventHandler(
                callback=self._handle_strategy_risk_event,
                event_type=EventType.STRATEGY_DEPLOYED_WITH_RISK,
                name="strategy_risk_deploy_handler"
            )
        )
        
        self.event_bus.register_handler(
            EventHandler(
                callback=self._handle_strategy_risk_event,
                event_type=EventType.STRATEGY_PAUSED,
                name="strategy_paused_handler"
            )
        )
        
        self.event_bus.register_handler(
            EventHandler(
                callback=self._handle_strategy_risk_event,
                event_type=EventType.STRATEGY_RESUMED,
                name="strategy_resumed_handler"
            )
        )
        
        # Risk metric update events
        self.event_bus.register_handler(
            EventHandler(
                callback=self._handle_risk_metrics_update,
                event_type=EventType.RISK_METRICS_UPDATED,
                name="risk_metrics_handler"
            )
        )
        
        logger.info("Registered risk event handlers")
    
    def _handle_risk_event(self, event: Event) -> None:
        """
        Handle risk alert event.
        
        Args:
            event: Event to handle
        """
        try:
            # Store event data
            self.latest_events[event.event_type] = event.data
            
            # Record the event
            event_record = {
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "data": event.data
            }
            
            # Add to risk alerts
            self.risk_alerts.append(event_record)
            
            # Process specific event types
            if event.event_type == EventType.CIRCUIT_BREAKER_TRIGGERED:
                self.circuit_breaker_history.append(event_record)
                logger.warning(f"Circuit breaker triggered: {event.data.get('reasons', [])}")
                
            elif event.event_type == EventType.RISK_LEVEL_CHANGED:
                old_level = event.data.get("old_level")
                new_level = event.data.get("new_level")
                self.current_risk_level = new_level
                self.risk_level_changes.append(event_record)
                logger.info(f"Risk level changed from {old_level} to {new_level}")
                
            elif event.event_type == EventType.RISK_ALERT:
                alert_type = event.data.get("alert_type")
                message = event.data.get("message", "")
                logger.warning(f"Risk alert ({alert_type}): {message}")
            
            logger.debug(f"Processed risk event: {event.event_type}")
            
        except Exception as e:
            logger.error(f"Error handling risk event: {e}")
    
    def _handle_strategy_risk_event(self, event: Event) -> None:
        """
        Handle strategy risk event.
        
        Args:
            event: Event to handle
        """
        try:
            strategy_id = event.data.get("strategy_id")
            if not strategy_id:
                logger.warning(f"Strategy risk event missing strategy_id: {event.event_type}")
                return
                
            # Ensure strategy exists in our tracking
            if strategy_id not in self.strategy_risk:
                self.strategy_risk[strategy_id] = {
                    "status": "unknown",
                    "history": [],
                    "last_updated": datetime.now().isoformat(),
                    "risk_metrics": {},
                    "allocation": 0.0
                }
                
            strategy_record = self.strategy_risk[strategy_id]
            
            # Update strategy status based on event type
            if event.event_type == EventType.STRATEGY_DEPLOYED_WITH_RISK:
                strategy_record["status"] = "active"
                strategy_record["allocation"] = event.data.get("allocation_percentage", 0.0)
                strategy_record["risk_level"] = event.data.get("risk_level", "MEDIUM")
                strategy_record["risk_params"] = event.data.get("risk_params", {})
                
            elif event.event_type == EventType.STRATEGY_PAUSED:
                strategy_record["status"] = "paused"
                strategy_record["pause_reason"] = event.data.get("reason", "")
                strategy_record["pause_time"] = event.timestamp.isoformat()
                
            elif event.event_type == EventType.STRATEGY_RESUMED:
                strategy_record["status"] = "active"
                if "pause_reason" in strategy_record:
                    del strategy_record["pause_reason"]
                if "pause_time" in strategy_record:
                    del strategy_record["pause_time"]
            
            # Add event to history
            event_record = {
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "data": event.data
            }
            strategy_record["history"].append(event_record)
            strategy_record["last_updated"] = datetime.now().isoformat()
            
            logger.info(f"Updated risk data for strategy {strategy_id}: {event.event_type}")
            
        except Exception as e:
            logger.error(f"Error handling strategy risk event: {e}")
    
    def _handle_risk_metrics_update(self, event: Event) -> None:
        """
        Handle risk metrics update event.
        
        Args:
            event: Event to handle
        """
        try:
            # Update portfolio metrics
            portfolio_metrics = event.data.get("portfolio_metrics", {})
            if portfolio_metrics:
                self.portfolio_metrics = portfolio_metrics
            
            # Update strategy-specific metrics
            strategy_metrics = event.data.get("strategy_metrics", {})
            for strategy_id, metrics in strategy_metrics.items():
                if strategy_id in self.strategy_risk:
                    self.strategy_risk[strategy_id]["risk_metrics"] = metrics
                    self.strategy_risk[strategy_id]["last_updated"] = datetime.now().isoformat()
            
            # Update risk thresholds
            thresholds = event.data.get("thresholds", {})
            if thresholds:
                self.risk_thresholds.update(thresholds)
            
            logger.debug(f"Updated risk metrics at {datetime.now().isoformat()}")
            
        except Exception as e:
            logger.error(f"Error handling risk metrics update: {e}")
    
    def get_risk_alerts(self, 
                        limit: int = 50, 
                        alert_type: Optional[str] = None,
                        since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get recent risk alerts, optionally filtered.
        
        Args:
            limit: Maximum number of alerts to return
            alert_type: Specific alert type to filter for
            since: Only return alerts since this time
            
        Returns:
            List of alert dictionaries
        """
        filtered_alerts = self.risk_alerts
        
        # Apply filters
        if alert_type:
            filtered_alerts = [a for a in filtered_alerts if a["event_type"] == alert_type]
            
        if since:
            filtered_alerts = [
                a for a in filtered_alerts 
                if datetime.fromisoformat(a["timestamp"]) >= since
            ]
        
        # Sort by timestamp, newest first
        sorted_alerts = sorted(
            filtered_alerts,
            key=lambda a: a["timestamp"],
            reverse=True
        )
        
        # Apply limit
        return sorted_alerts[:limit]
    
    def get_strategy_risk_data(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get risk data for strategies.
        
        Args:
            strategy_id: Specific strategy ID to get data for, or None for all
            
        Returns:
            Dictionary with risk data
        """
        if strategy_id:
            return self.strategy_risk.get(strategy_id, {})
        return self.strategy_risk
    
    def get_current_risk_status(self) -> Dict[str, Any]:
        """
        Get current risk status.
        
        Returns:
            Dictionary with current risk status
        """
        active_strategies = sum(1 for s in self.strategy_risk.values() if s.get("status") == "active")
        paused_strategies = sum(1 for s in self.strategy_risk.values() if s.get("status") == "paused")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "risk_level": self.current_risk_level,
            "portfolio_metrics": self.portfolio_metrics,
            "active_strategies": active_strategies,
            "paused_strategies": paused_strategies,
            "recent_alerts": self.get_risk_alerts(limit=5),
            "circuit_breakers": {
                "status": "active",
                "thresholds": self.risk_thresholds,
                "recent_triggers": len(self.circuit_breaker_history[-5:]) if self.circuit_breaker_history else 0
            }
        }
    
    def get_risk_statistics(self, time_period: str = "1d") -> Dict[str, Any]:
        """
        Get risk statistics for a specific time period.
        
        Args:
            time_period: Time period to get statistics for ("1d", "1w", "1m")
            
        Returns:
            Dictionary with risk statistics
        """
        now = datetime.now()
        
        # Determine cutoff time
        if time_period == "1d":
            cutoff = now - timedelta(days=1)
        elif time_period == "1w":
            cutoff = now - timedelta(weeks=1)
        elif time_period == "1m":
            cutoff = now - timedelta(days=30)
        else:
            cutoff = now - timedelta(days=1)  # Default to 1 day
        
        # Filter alerts by cutoff
        period_alerts = [
            a for a in self.risk_alerts 
            if datetime.fromisoformat(a["timestamp"]) >= cutoff
        ]
        
        # Count by type
        alert_counts = {}
        for alert in period_alerts:
            event_type = alert["event_type"]
            if event_type not in alert_counts:
                alert_counts[event_type] = 0
            alert_counts[event_type] += 1
        
        # Count circuit breaker triggers
        circuit_breaker_count = sum(
            1 for a in self.circuit_breaker_history
            if datetime.fromisoformat(a["timestamp"]) >= cutoff
        )
        
        # Count risk level changes
        risk_level_changes = sum(
            1 for a in self.risk_level_changes
            if datetime.fromisoformat(a["timestamp"]) >= cutoff
        )
        
        return {
            "time_period": time_period,
            "total_alerts": len(period_alerts),
            "alert_counts": alert_counts,
            "circuit_breaker_triggers": circuit_breaker_count,
            "risk_level_changes": risk_level_changes,
            "timestamp": now.isoformat()
        }

# Singleton instance for global access
risk_event_tracker = None

def get_risk_event_tracker(event_bus: Optional[EventBus] = None) -> RiskEventTracker:
    """
    Get the risk event tracker instance.
    
    Args:
        event_bus: Event bus to register handlers with
        
    Returns:
        RiskEventTracker instance
    """
    global risk_event_tracker
    if risk_event_tracker is None:
        risk_event_tracker = RiskEventTracker(event_bus)
    return risk_event_tracker
