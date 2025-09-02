"""
Emergency Controls Module

This module provides mechanisms to protect trading accounts from excessive losses
or unusual behavior, including kill switches, position limits, and circuit breakers.

Features:
- Kill switch functionality (manual or automated)
- Per-strategy position limits
- Circuit breakers for unusual activity or market conditions
- Account risk monitoring
"""
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Callable, Union
import pandas as pd
import numpy as np

from trading_bot.core.event_bus import get_global_event_bus, Event
from trading_bot.core.constants import EventType, TradingMode
from trading_bot.security.secure_logger import SecureLogger

logger = SecureLogger(name=__name__)

class EmergencyControls:
    """
    Provides emergency trading controls for risk management.
    Can halt trading or specific strategies when risk thresholds are exceeded.
    """
    
    def __init__(self, 
                 max_daily_loss_pct: float = 0.02,
                 max_position_pct: float = 0.05,
                 max_strategy_loss_pct: float = 0.01,
                 max_drawdown_pct: float = 0.05,
                 circuit_breaker_window: int = 60,
                 auto_enable: bool = True):
        """
        Initialize emergency controls.
        
        Args:
            max_daily_loss_pct: Maximum allowed daily loss as percentage of account value
            max_position_pct: Maximum allowed position size as percentage of account value
            max_strategy_loss_pct: Maximum allowed loss per strategy
            max_drawdown_pct: Maximum allowed drawdown before kill switch triggers
            circuit_breaker_window: Window in seconds for circuit breaker monitoring
            auto_enable: Whether to automatically enable controls on initialization
        """
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_position_pct = max_position_pct
        self.max_strategy_loss_pct = max_strategy_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.circuit_breaker_window = circuit_breaker_window
        
        self.event_bus = get_global_event_bus()
        self.enabled = auto_enable
        
        # Global kill switch status
        self.kill_switch_activated = False
        
        # Strategy status tracking
        self.strategy_status: Dict[str, Dict[str, Any]] = {}
        
        # Position tracking
        self.positions: Dict[str, Dict[str, Any]] = {}
        
        # Account value tracking
        self.account_value = 0.0
        self.starting_value = 0.0
        self.high_watermark = 0.0
        
        # Risk metrics
        self.daily_pnl = 0.0
        self.current_drawdown_pct = 0.0
        
        # Circuit breaker data
        self.recent_orders: List[Dict[str, Any]] = []
        self.recent_rejections: List[Dict[str, Any]] = []
        self.trade_frequency: Dict[str, int] = {}
        
        # Initialize event subscriptions
        self._initialize_event_listeners()
        
        logger.info("Emergency controls initialized")
        
        if auto_enable:
            logger.info("Emergency controls automatically enabled")
    
    def _initialize_event_listeners(self):
        """Set up event bus subscriptions."""
        # Order and trade events
        self.event_bus.subscribe(EventType.ORDER_CREATED, self.on_order_created)
        self.event_bus.subscribe(EventType.ORDER_FILLED, self.on_order_filled)
        self.event_bus.subscribe(EventType.ORDER_REJECTED, self.on_order_rejected)
        self.event_bus.subscribe(EventType.TRADE_EXECUTED, self.on_trade_executed)
        
        # Account and position events
        self.event_bus.subscribe(EventType.CAPITAL_ADJUSTED, self.on_capital_adjusted)
        self.event_bus.subscribe(EventType.POSITION_UPDATE, self.on_position_update)
        
        # Risk events
        self.event_bus.subscribe(EventType.RISK_LIMIT_REACHED, self.on_risk_limit_reached)
        self.event_bus.subscribe(EventType.DRAWDOWN_ALERT, self.on_drawdown_alert)
        self.event_bus.subscribe(EventType.DRAWDOWN_THRESHOLD_EXCEEDED, self.on_drawdown_threshold_exceeded)
    
    def enable(self):
        """Enable emergency controls."""
        self.enabled = True
        logger.info("Emergency controls enabled")
        
        # Publish event
        self.event_bus.create_and_publish(
            event_type=EventType.RISK_CONTROL_STATUS_CHANGED,
            data={"status": "enabled", "timestamp": datetime.now().isoformat()},
            source="emergency_controls"
        )
    
    def disable(self):
        """Disable emergency controls (USE WITH CAUTION)."""
        self.enabled = False
        logger.warning("Emergency controls disabled - USE CAUTION")
        
        # Publish event
        self.event_bus.create_and_publish(
            event_type=EventType.RISK_CONTROL_STATUS_CHANGED,
            data={"status": "disabled", "timestamp": datetime.now().isoformat()},
            source="emergency_controls"
        )
    
    def activate_kill_switch(self, reason: str = "Manual activation"):
        """
        Activate the emergency kill switch to halt all trading.
        
        Args:
            reason: Reason for kill switch activation
        """
        if self.kill_switch_activated:
            logger.warning(f"Kill switch already activated. Current reason: {reason}")
            return
        
        self.kill_switch_activated = True
        
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        
        # Publish event
        self.event_bus.create_and_publish(
            event_type=EventType.KILL_SWITCH_ACTIVATED,
            data={
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "account_value": self.account_value,
                "daily_pnl": self.daily_pnl,
                "drawdown": self.current_drawdown_pct
            },
            source="emergency_controls"
        )
        
        # Change trading mode to STOPPED
        self.event_bus.create_and_publish(
            event_type=EventType.MODE_CHANGED,
            data={"new_mode": TradingMode.STOPPED, "reason": f"Kill switch activated: {reason}"},
            source="emergency_controls"
        )
    
    def deactivate_kill_switch(self, override_reason: str = "Manual deactivation"):
        """
        Deactivate the kill switch (requires manual intervention).
        
        Args:
            override_reason: Reason for manual override
        """
        if not self.kill_switch_activated:
            logger.info("Kill switch is not currently activated")
            return
        
        self.kill_switch_activated = False
        
        logger.warning(f"KILL SWITCH DEACTIVATED: {override_reason}")
        
        # Publish event
        self.event_bus.create_and_publish(
            event_type=EventType.KILL_SWITCH_DEACTIVATED,
            data={
                "override_reason": override_reason, 
                "timestamp": datetime.now().isoformat()
            },
            source="emergency_controls"
        )
    
    def disable_strategy(self, strategy_id: str, reason: str = "Risk limit exceeded"):
        """
        Disable a specific strategy.
        
        Args:
            strategy_id: ID of the strategy to disable
            reason: Reason for disabling
        """
        if strategy_id not in self.strategy_status:
            self.strategy_status[strategy_id] = {"enabled": False, "reason": reason}
        else:
            self.strategy_status[strategy_id]["enabled"] = False
            self.strategy_status[strategy_id]["reason"] = reason
        
        logger.warning(f"Strategy disabled: {strategy_id}. Reason: {reason}")
        
        # Publish event
        self.event_bus.create_and_publish(
            event_type=EventType.STRATEGY_STOPPED,
            data={"strategy_id": strategy_id, "reason": reason},
            source="emergency_controls"
        )
    
    def enable_strategy(self, strategy_id: str, reason: str = "Manual override"):
        """
        Enable a previously disabled strategy.
        
        Args:
            strategy_id: ID of the strategy to enable
            reason: Reason for enabling
        """
        if strategy_id not in self.strategy_status:
            self.strategy_status[strategy_id] = {"enabled": True, "reason": reason}
        else:
            self.strategy_status[strategy_id]["enabled"] = True
            self.strategy_status[strategy_id]["reason"] = reason
        
        logger.info(f"Strategy enabled: {strategy_id}. Reason: {reason}")
        
        # Publish event
        self.event_bus.create_and_publish(
            event_type=EventType.STRATEGY_STARTED,
            data={"strategy_id": strategy_id, "reason": reason},
            source="emergency_controls"
        )
    
    def is_strategy_enabled(self, strategy_id: str) -> bool:
        """
        Check if a strategy is enabled.
        
        Args:
            strategy_id: Strategy ID to check
            
        Returns:
            True if strategy is enabled or unknown, False if explicitly disabled
        """
        if not self.enabled or self.kill_switch_activated:
            return False
            
        if strategy_id not in self.strategy_status:
            return True
            
        return self.strategy_status[strategy_id].get("enabled", True)
    
    def set_position_limit(self, symbol: str, max_size: float, max_notional: Optional[float] = None):
        """
        Set position limit for a symbol.
        
        Args:
            symbol: Symbol to set limit for
            max_size: Maximum position size in units
            max_notional: Maximum position value (price * size)
        """
        if symbol not in self.positions:
            self.positions[symbol] = {
                "current_size": 0.0,
                "max_size": max_size,
                "max_notional": max_notional,
                "current_notional": 0.0
            }
        else:
            self.positions[symbol]["max_size"] = max_size
            if max_notional is not None:
                self.positions[symbol]["max_notional"] = max_notional
                
        logger.info(f"Position limit set for {symbol}: {max_size} units, {max_notional if max_notional else 'no'} notional limit")
    
    def check_position_limit(self, symbol: str, new_size: float, price: float) -> bool:
        """
        Check if a new position size would exceed limits.
        
        Args:
            symbol: Symbol to check
            new_size: New total position size
            price: Current price
            
        Returns:
            True if position is within limits, False otherwise
        """
        if not self.enabled:
            return True
            
        # If kill switch is active, no new positions
        if self.kill_switch_activated:
            logger.warning(f"Position check failed: Kill switch is active")
            return False
            
        if symbol not in self.positions:
            # No specific limits set, check against global limits
            notional_value = abs(new_size) * price
            account_pct = notional_value / self.account_value if self.account_value > 0 else 1.0
            
            if account_pct > self.max_position_pct:
                logger.warning(
                    f"Position limit exceeded for {symbol}: {account_pct:.2%} of account " + 
                    f"(limit: {self.max_position_pct:.2%})"
                )
                return False
                
            return True
            
        position = self.positions[symbol]
        
        # Check size limit
        if position.get("max_size") is not None and abs(new_size) > position["max_size"]:
            logger.warning(
                f"Position size limit exceeded for {symbol}: {abs(new_size)} units " + 
                f"(limit: {position['max_size']})"
            )
            return False
            
        # Check notional limit
        notional_value = abs(new_size) * price
        if position.get("max_notional") is not None and notional_value > position["max_notional"]:
            logger.warning(
                f"Position notional limit exceeded for {symbol}: ${notional_value:.2f} " + 
                f"(limit: ${position['max_notional']:.2f})"
            )
            return False
            
        # Check account percentage limit
        account_pct = notional_value / self.account_value if self.account_value > 0 else 1.0
        if account_pct > self.max_position_pct:
            logger.warning(
                f"Position account percentage limit exceeded for {symbol}: {account_pct:.2%} " + 
                f"(limit: {self.max_position_pct:.2%})"
            )
            return False
            
        return True
    
    def update_position(self, symbol: str, size: float, price: float):
        """
        Update tracked position for a symbol.
        
        Args:
            symbol: Symbol to update
            size: New position size
            price: Current price
        """
        if symbol not in self.positions:
            self.positions[symbol] = {
                "current_size": size,
                "current_notional": abs(size) * price,
                "max_size": None,
                "max_notional": None
            }
        else:
            self.positions[symbol]["current_size"] = size
            self.positions[symbol]["current_notional"] = abs(size) * price
            
        # Check if position exceeds limits and log warning
        if self.enabled and not self.check_position_limit(symbol, size, price):
            logger.warning(
                f"Current position for {symbol} exceeds limits: {size} units at ${price:.2f}"
            )
    
    def circuit_breaker_check(self) -> Optional[str]:
        """
        Check if circuit breaker conditions are met.
        
        Returns:
            Reason string if circuit breaker should trigger, None otherwise
        """
        if not self.enabled:
            return None
            
        # Check excessive order rejection rate
        now = datetime.now()
        cutoff_time = now - timedelta(seconds=self.circuit_breaker_window)
        
        # Filter recent orders and rejections
        recent_orders = [o for o in self.recent_orders if o["timestamp"] >= cutoff_time]
        recent_rejections = [r for r in self.recent_rejections if r["timestamp"] >= cutoff_time]
        
        # Calculate rejection rate
        total_orders = len(recent_orders)
        total_rejections = len(recent_rejections)
        
        if total_orders > 10 and total_rejections / total_orders > 0.5:
            reason = (
                f"High order rejection rate: {total_rejections}/{total_orders} " + 
                f"({total_rejections/total_orders:.1%}) in the last {self.circuit_breaker_window} seconds"
            )
            return reason
            
        # Check for excessive trading frequency (potential algo runaway)
        if self.check_excessive_frequency():
            reason = "Excessive trading frequency detected"
            return reason
            
        # Check account drawdown
        if self.current_drawdown_pct >= self.max_drawdown_pct:
            reason = f"Maximum drawdown exceeded: {self.current_drawdown_pct:.2%} (limit: {self.max_drawdown_pct:.2%})"
            return reason
            
        # Check daily loss
        daily_loss_pct = self.daily_pnl / self.starting_value if self.starting_value > 0 else 0
        if self.daily_pnl < 0 and abs(daily_loss_pct) >= self.max_daily_loss_pct:
            reason = f"Maximum daily loss exceeded: {daily_loss_pct:.2%} (limit: -{self.max_daily_loss_pct:.2%})"
            return reason
            
        return None
    
    def check_excessive_frequency(self) -> bool:
        """
        Check if trading frequency is excessive.
        
        Returns:
            True if frequency is excessive, False otherwise
        """
        now = datetime.now()
        cutoff_time = now - timedelta(seconds=self.circuit_breaker_window)
        
        # Count orders by strategy in the window
        strategy_counts = {}
        for order in self.recent_orders:
            if order["timestamp"] < cutoff_time:
                continue
                
            strategy_id = order.get("strategy_id", "unknown")
            if strategy_id not in strategy_counts:
                strategy_counts[strategy_id] = 0
            strategy_counts[strategy_id] += 1
            
        # Check if any strategy exceeds threshold
        # (threshold is dynamic based on timeframe, e.g., 20 orders per minute)
        threshold = 20
        
        for strategy_id, count in strategy_counts.items():
            if count > threshold:
                logger.warning(
                    f"Excessive order frequency for strategy {strategy_id}: " + 
                    f"{count} orders in {self.circuit_breaker_window} seconds"
                )
                return True
                
        return False
    
    def on_order_created(self, event: Event):
        """Handle order created events."""
        if not self.enabled:
            return
            
        order_data = event.data
        timestamp = datetime.fromisoformat(order_data.get("timestamp", datetime.now().isoformat()))
        
        # Add to recent orders
        self.recent_orders.append({
            "order_id": order_data.get("order_id", "unknown"),
            "symbol": order_data.get("symbol", "unknown"),
            "size": order_data.get("quantity", 0),
            "price": order_data.get("price", 0),
            "strategy_id": order_data.get("strategy_id", "unknown"),
            "timestamp": timestamp
        })
        
        # Trim recent orders list (keep last 100)
        if len(self.recent_orders) > 100:
            self.recent_orders = self.recent_orders[-100:]
    
    def on_order_filled(self, event: Event):
        """Handle order filled events."""
        if not self.enabled:
            return
            
        fill_data = event.data
        symbol = fill_data.get("symbol", "unknown")
        quantity = fill_data.get("quantity", 0)
        price = fill_data.get("price", 0)
        
        # Update position tracking
        if symbol in self.positions:
            current_size = self.positions[symbol].get("current_size", 0)
            new_size = current_size + quantity
            self.update_position(symbol, new_size, price)
    
    def on_order_rejected(self, event: Event):
        """Handle order rejected events."""
        if not self.enabled:
            return
            
        rejection_data = event.data
        timestamp = datetime.fromisoformat(rejection_data.get("timestamp", datetime.now().isoformat()))
        
        # Add to recent rejections
        self.recent_rejections.append({
            "order_id": rejection_data.get("order_id", "unknown"),
            "symbol": rejection_data.get("symbol", "unknown"),
            "reason": rejection_data.get("reason", "unknown"),
            "strategy_id": rejection_data.get("strategy_id", "unknown"),
            "timestamp": timestamp
        })
        
        # Trim list (keep last 100)
        if len(self.recent_rejections) > 100:
            self.recent_rejections = self.recent_rejections[-100:]
            
        # Run circuit breaker check on rejection
        circuit_breaker_reason = self.circuit_breaker_check()
        if circuit_breaker_reason:
            self.activate_kill_switch(reason=f"Circuit breaker: {circuit_breaker_reason}")
    
    def on_trade_executed(self, event: Event):
        """Handle trade executed events."""
        if not self.enabled:
            return
            
        trade_data = event.data
        pnl = trade_data.get("pnl", 0.0)
        strategy_id = trade_data.get("strategy_id", "unknown")
        
        # Update daily P&L
        self.daily_pnl += pnl
        
        # Check P&L impact
        self._check_pnl_impact(pnl, strategy_id)
        
        # Check circuit breaker after trade
        circuit_breaker_reason = self.circuit_breaker_check()
        if circuit_breaker_reason:
            self.activate_kill_switch(reason=f"Circuit breaker: {circuit_breaker_reason}")
    
    def on_capital_adjusted(self, event: Event):
        """Handle capital adjusted events."""
        capital_data = event.data
        new_capital = capital_data.get("new_capital", self.account_value)
        
        # If this is the first update of the day, set as starting value
        if self.starting_value == 0:
            self.starting_value = new_capital
            
        # Update account value and high watermark
        self.account_value = new_capital
        if new_capital > self.high_watermark:
            self.high_watermark = new_capital
        
        # Update drawdown
        if self.high_watermark > 0:
            self.current_drawdown_pct = (self.high_watermark - self.account_value) / self.high_watermark
    
    def on_position_update(self, event: Event):
        """Handle position update events."""
        if not self.enabled:
            return
            
        position_data = event.data
        symbol = position_data.get("symbol", "unknown")
        size = position_data.get("size", 0)
        price = position_data.get("price", 0)
        
        # Update position tracking
        self.update_position(symbol, size, price)
    
    def on_risk_limit_reached(self, event: Event):
        """Handle risk limit reached events."""
        if not self.enabled:
            return
            
        risk_data = event.data
        risk_type = risk_data.get("risk_type", "unknown")
        strategy_id = risk_data.get("strategy_id", None)
        severity = risk_data.get("severity", "warning")
        
        logger.warning(f"Risk limit reached: {risk_type} for {strategy_id or 'global'} - {severity}")
        
        # Disable specific strategy or activate kill switch based on severity
        if severity == "critical":
            reason = f"Risk limit reached: {risk_type}"
            self.activate_kill_switch(reason=reason)
        elif strategy_id and severity == "high":
            self.disable_strategy(strategy_id, reason=f"Risk limit reached: {risk_type}")
    
    def on_drawdown_alert(self, event: Event):
        """Handle drawdown alert events."""
        if not self.enabled:
            return
            
        drawdown_data = event.data
        strategy_id = drawdown_data.get("strategy_id", None)
        drawdown_pct = drawdown_data.get("drawdown_pct", 0.0)
        
        # Log alert
        if strategy_id:
            logger.warning(f"Drawdown alert for strategy {strategy_id}: {drawdown_pct:.2%}")
        else:
            logger.warning(f"Account drawdown alert: {drawdown_pct:.2%}")
    
    def on_drawdown_threshold_exceeded(self, event: Event):
        """Handle drawdown threshold exceeded events."""
        if not self.enabled:
            return
            
        drawdown_data = event.data
        strategy_id = drawdown_data.get("strategy_id", None)
        drawdown_pct = drawdown_data.get("drawdown_pct", 0.0)
        threshold = drawdown_data.get("threshold", 0.0)
        
        if strategy_id:
            # Disable specific strategy
            reason = f"Strategy drawdown exceeded: {drawdown_pct:.2%} > {threshold:.2%}"
            self.disable_strategy(strategy_id, reason=reason)
        else:
            # Account-level drawdown
            reason = f"Account drawdown exceeded: {drawdown_pct:.2%} > {threshold:.2%}"
            self.activate_kill_switch(reason=reason)
    
    def _check_pnl_impact(self, pnl: float, strategy_id: str):
        """
        Check P&L impact against limits.
        
        Args:
            pnl: P&L amount
            strategy_id: Strategy ID
        """
        # Skip if P&L is positive
        if pnl >= 0:
            return
            
        # Check strategy loss limit
        if strategy_id in self.strategy_status:
            strategy_daily_pnl = self.strategy_status[strategy_id].get("daily_pnl", 0.0) + pnl
            self.strategy_status[strategy_id]["daily_pnl"] = strategy_daily_pnl
            
            # Calculate percentage loss
            strategy_loss_pct = abs(strategy_daily_pnl) / self.starting_value if self.starting_value > 0 else 0
            
            if strategy_loss_pct >= self.max_strategy_loss_pct and strategy_daily_pnl < 0:
                reason = (
                    f"Strategy loss limit exceeded: {strategy_loss_pct:.2%} " + 
                    f"(limit: {self.max_strategy_loss_pct:.2%})"
                )
                self.disable_strategy(strategy_id, reason=reason)
                
        # Check account loss limit
        daily_loss_pct = abs(self.daily_pnl) / self.starting_value if self.starting_value > 0 else 0
        
        if self.daily_pnl < 0 and daily_loss_pct >= self.max_daily_loss_pct:
            reason = f"Daily loss limit exceeded: {daily_loss_pct:.2%} (limit: {self.max_daily_loss_pct:.2%})"
            self.activate_kill_switch(reason=reason)
    
    def reset_daily_metrics(self):
        """Reset daily metrics (should be called at start of trading day)."""
        old_value = self.account_value or 0
        
        self.starting_value = old_value
        self.daily_pnl = 0.0
        
        # Reset strategy-specific daily P&L
        for strategy_id in self.strategy_status:
            if "daily_pnl" in self.strategy_status[strategy_id]:
                self.strategy_status[strategy_id]["daily_pnl"] = 0.0
        
        # Clear recent orders and rejections
        self.recent_orders = []
        self.recent_rejections = []
        
        logger.info(f"Daily metrics reset. Starting account value: ${old_value:.2f}")
    
    def get_status_report(self) -> Dict[str, Any]:
        """
        Get a status report of emergency controls.
        
        Returns:
            Dictionary with status information
        """
        return {
            "enabled": self.enabled,
            "kill_switch_activated": self.kill_switch_activated,
            "account_value": self.account_value,
            "starting_value": self.starting_value,
            "high_watermark": self.high_watermark,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.daily_pnl / self.starting_value if self.starting_value > 0 else 0,
            "current_drawdown_pct": self.current_drawdown_pct,
            "position_count": len(self.positions),
            "positions": {
                symbol: {
                    "size": pos.get("current_size", 0),
                    "notional": pos.get("current_notional", 0),
                    "max_size": pos.get("max_size", "unlimited"),
                    "max_notional": pos.get("max_notional", "unlimited")
                }
                for symbol, pos in self.positions.items()
            },
            "disabled_strategies": {
                strategy_id: status.get("reason", "unknown")
                for strategy_id, status in self.strategy_status.items()
                if not status.get("enabled", True)
            },
            "limits": {
                "max_daily_loss_pct": self.max_daily_loss_pct,
                "max_position_pct": self.max_position_pct,
                "max_strategy_loss_pct": self.max_strategy_loss_pct,
                "max_drawdown_pct": self.max_drawdown_pct
            },
            "timestamp": datetime.now().isoformat()
        }
