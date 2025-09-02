"""
Circuit breaker system for preventing excessive losses in the trading system.

This module provides functionality to automatically halt trading when certain
thresholds are exceeded, such as maximum daily loss, maximum drawdown,
or maximum number of consecutive losses.
"""

import logging
import time
import datetime
import threading
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

import pandas as pd

from trading_bot.core.event_bus import EventBus
from trading_bot.core.events import Event

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "enabled": True,
    "maxDailyLoss": 500.0,  # dollars
    "maxDrawdownPercent": 5.0,  # percent
    "maxTradesPerDay": 20,
    "maxConsecutiveLosses": 3
}

class CircuitBreakerManager:
    """
    Manages circuit breakers to halt trading automatically when predefined
    risk thresholds are exceeded.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None, config_path: Optional[str] = None):
        """
        Initialize the circuit breaker manager.
        
        Args:
            event_bus: Event bus for publishing and subscribing to events
            config_path: Path to configuration file for circuit breaker settings
        """
        self.event_bus = event_bus or EventBus()
        self.config_path = config_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Circuit breaker state
        self._active = False
        self._reason = None
        self._triggered_at = None
        self._triggered_info = {}
        
        # Daily tracking
        self._daily_loss = 0.0
        self._daily_trades = 0
        self._consecutive_losses = 0
        self._today = datetime.datetime.now().date()
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Subscribe to trade events
        if self.event_bus:
            self.event_bus.subscribe("trade_executed", self._on_trade_executed)
            self.event_bus.subscribe("position_closed", self._on_position_closed)
            self.event_bus.subscribe("daily_summary", self._on_daily_summary)
            
        # Start monitoring thread
        self._monitoring_thread = threading.Thread(target=self._daily_reset_monitor, daemon=True)
        self._monitoring_thread.start()
            
    def _load_config(self) -> Dict[str, Any]:
        """Load circuit breaker configuration from file or use defaults."""
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded circuit breaker configuration from {self.config_path}")
                return config
            except Exception as e:
                logger.error(f"Error loading circuit breaker config: {e}")
        
        logger.info("Using default circuit breaker configuration")
        return DEFAULT_CONFIG
        
    def save_config(self) -> None:
        """Save current configuration to file."""
        if not self.config_path:
            logger.warning("No config path specified, can't save circuit breaker config")
            return
            
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved circuit breaker configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving circuit breaker config: {e}")
    
    def _daily_reset_monitor(self) -> None:
        """Monitor thread that resets daily counters at market open."""
        while True:
            current_date = datetime.datetime.now().date()
            if current_date != self._today:
                with self._lock:
                    self._reset_daily_counters()
                    self._today = current_date
                    logger.info("Reset daily circuit breaker counters")
            time.sleep(60)  # Check every minute
    
    def _reset_daily_counters(self) -> None:
        """Reset daily tracking counters."""
        self._daily_loss = 0.0
        self._daily_trades = 0
        # Don't reset consecutive losses as they can span multiple days
        
        # Auto-reset circuit breaker if it was triggered due to daily limits
        if self._active and self._reason in ["daily_loss_exceeded", "max_trades_exceeded"]:
            self.reset()
    
    def is_active(self) -> bool:
        """Return whether the circuit breaker is currently active."""
        return self._active
        
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the circuit breaker."""
        with self._lock:
            status = {
                "active": self._active,
                "reason": self._reason,
                "triggeredAt": self._triggered_at.isoformat() if self._triggered_at else None,
                "maxDailyLoss": self.config["maxDailyLoss"],
                "currentDailyLoss": self._daily_loss,
                "maxTradesPerDay": self.config["maxTradesPerDay"],
                "currentTradeCount": self._daily_trades
            }
            
            if self._triggered_info:
                status.update(self._triggered_info)
                
            return status
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update the circuit breaker configuration."""
        with self._lock:
            # Only update valid keys
            for key in DEFAULT_CONFIG:
                if key in new_config:
                    self.config[key] = new_config[key]
            
            # Save updated config
            self.save_config()
            
            logger.info(f"Updated circuit breaker configuration: {self.config}")
    
    def register_loss(self, amount: float, trade_id: str = None) -> None:
        """
        Register a loss with the circuit breaker.
        
        Args:
            amount: Loss amount (positive value)
            trade_id: Optional trade ID for reference
        """
        if amount <= 0:
            logger.warning(f"Ignoring non-positive loss amount: {amount}")
            return
            
        with self._lock:
            self._daily_loss += amount
            self._consecutive_losses += 1
            self._daily_trades += 1
            
            logger.info(f"Registered loss: ${amount:.2f}, daily total: ${self._daily_loss:.2f}, consecutive: {self._consecutive_losses}")
            
            # Check if circuit breaker should be triggered
            self._check_thresholds()
    
    def register_profit(self, amount: float, trade_id: str = None) -> None:
        """
        Register a profit with the circuit breaker.
        
        Args:
            amount: Profit amount (positive value)
            trade_id: Optional trade ID for reference
        """
        if amount <= 0:
            logger.warning(f"Ignoring non-positive profit amount: {amount}")
            return
            
        with self._lock:
            # Offset daily loss if present
            if self._daily_loss > 0:
                self._daily_loss = max(0, self._daily_loss - amount)
            
            # Reset consecutive losses counter
            self._consecutive_losses = 0
            
            # Increment trade counter
            self._daily_trades += 1
            
            logger.info(f"Registered profit: ${amount:.2f}, daily loss adjusted to: ${self._daily_loss:.2f}")
            
            # Check trade count threshold
            if self.config["enabled"] and self._daily_trades >= self.config["maxTradesPerDay"]:
                self._trigger("max_trades_exceeded", {
                    "maxTradesPerDay": self.config["maxTradesPerDay"],
                    "currentTradeCount": self._daily_trades
                })
    
    def register_trade(self, profit_loss: float, trade_id: str = None) -> None:
        """
        Register a trade result with the circuit breaker.
        
        Args:
            profit_loss: Profit (positive) or loss (negative) amount
            trade_id: Optional trade ID for reference
        """
        if profit_loss >= 0:
            self.register_profit(profit_loss, trade_id)
        else:
            self.register_loss(abs(profit_loss), trade_id)
    
    def _check_thresholds(self) -> None:
        """Check all circuit breaker thresholds and trigger if exceeded."""
        if not self.config["enabled"]:
            return
            
        # Check maximum daily loss
        if self._daily_loss >= self.config["maxDailyLoss"]:
            self._trigger("daily_loss_exceeded", {
                "maxDailyLoss": self.config["maxDailyLoss"],
                "currentDailyLoss": self._daily_loss
            })
            return
            
        # Check consecutive losses
        if self._consecutive_losses >= self.config["maxConsecutiveLosses"]:
            self._trigger("consecutive_losses_exceeded", {
                "maxConsecutiveLosses": self.config["maxConsecutiveLosses"],
                "currentConsecutiveLosses": self._consecutive_losses
            })
            return
    
    def _trigger(self, reason: str, details: Dict[str, Any]) -> None:
        """
        Trigger the circuit breaker.
        
        Args:
            reason: Reason for triggering
            details: Additional details about the trigger
        """
        if self._active:
            logger.debug(f"Circuit breaker already active, ignoring trigger: {reason}")
            return
            
        with self._lock:
            self._active = True
            self._reason = reason
            self._triggered_at = datetime.datetime.now()
            self._triggered_info = details
            
            logger.warning(f"Circuit breaker TRIGGERED: {reason}, details: {details}")
            
            # Publish event
            if self.event_bus:
                self.event_bus.publish(Event(
                    type="circuit_breaker_triggered",
                    data={
                        "reason": reason,
                        "triggered_at": self._triggered_at.isoformat(),
                        "details": details
                    }
                ))
    
    def reset(self) -> None:
        """Reset (deactivate) the circuit breaker."""
        if not self._active:
            logger.debug("Circuit breaker not active, ignoring reset")
            return
            
        with self._lock:
            previous_reason = self._reason
            previous_details = self._triggered_info
            
            self._active = False
            self._reason = None
            self._triggered_at = None
            self._triggered_info = {}
            
            logger.info(f"Circuit breaker RESET. Was active due to: {previous_reason}")
            
            # Publish event
            if self.event_bus:
                self.event_bus.publish(Event(
                    type="circuit_breaker_reset",
                    data={
                        "previous_reason": previous_reason,
                        "previous_details": previous_details,
                        "reset_at": datetime.datetime.now().isoformat()
                    }
                ))
    
    def _on_trade_executed(self, event: Event) -> None:
        """Handle trade executed events."""
        if not event.data:
            return
            
        trade_data = event.data
        if "profit_loss" in trade_data:
            self.register_trade(trade_data["profit_loss"], trade_data.get("trade_id"))
    
    def _on_position_closed(self, event: Event) -> None:
        """Handle position closed events."""
        if not event.data:
            return
            
        position_data = event.data
        if "realized_pl" in position_data:
            self.register_trade(position_data["realized_pl"], position_data.get("position_id"))
    
    def _on_daily_summary(self, event: Event) -> None:
        """Handle daily summary events to check drawdown."""
        if not event.data or not self.config["enabled"]:
            return
            
        summary_data = event.data
        if "drawdown_percent" in summary_data:
            drawdown = summary_data["drawdown_percent"]
            if drawdown >= self.config["maxDrawdownPercent"]:
                self._trigger("drawdown_exceeded", {
                    "maxDrawdownPercent": self.config["maxDrawdownPercent"],
                    "currentDrawdownPercent": drawdown
                })
