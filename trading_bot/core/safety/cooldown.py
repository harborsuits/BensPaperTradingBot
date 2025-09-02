"""
Cooldown system for enforcing trading pauses after losses or rapid trading.

This module implements a cooldown system that temporarily pauses trading
after certain conditions are met, such as consecutive losses or rapid trading.
"""

import logging
import time
import datetime
import threading
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

from trading_bot.core.event_bus import EventBus
from trading_bot.core.events import Event

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "enabled": True,
    "durationSeconds": 900,  # 15 minutes
    "afterConsecutiveLosses": 3,
    "afterMaxDrawdown": True
}

class CooldownManager:
    """
    Manages trading cooldowns to enforce pauses after losses or other triggers.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None, config_path: Optional[str] = None):
        """
        Initialize the cooldown manager.
        
        Args:
            event_bus: Event bus for publishing and subscribing to events
            config_path: Path to configuration file for cooldown settings
        """
        self.event_bus = event_bus or EventBus()
        self.config_path = config_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Cooldown state
        self._active = False
        self._reason = None
        self._start_time = None
        self._end_time = None
        
        # Tracking
        self._consecutive_losses = 0
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Cooldown monitor thread
        self._monitoring_thread = threading.Thread(target=self._cooldown_monitor, daemon=True)
        self._monitoring_thread.start()
        
        # Subscribe to events
        if self.event_bus:
            self.event_bus.subscribe("trade_executed", self._on_trade_executed)
            self.event_bus.subscribe("circuit_breaker_triggered", self._on_circuit_breaker_triggered)
            
    def _load_config(self) -> Dict[str, Any]:
        """Load cooldown configuration from file or use defaults."""
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded cooldown configuration from {self.config_path}")
                return config
            except Exception as e:
                logger.error(f"Error loading cooldown config: {e}")
        
        logger.info("Using default cooldown configuration")
        return DEFAULT_CONFIG
        
    def save_config(self) -> None:
        """Save current configuration to file."""
        if not self.config_path:
            logger.warning("No config path specified, can't save cooldown config")
            return
            
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved cooldown configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving cooldown config: {e}")
    
    def _cooldown_monitor(self) -> None:
        """Monitor thread that checks cooldown expiration."""
        while True:
            if self._active and datetime.datetime.now() >= self._end_time:
                with self._lock:
                    self._end_cooldown("expired")
                    logger.info("Cooldown period expired")
            time.sleep(1)  # Check every second
    
    def is_active(self) -> bool:
        """Return whether a cooldown is currently active."""
        return self._active
        
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the cooldown."""
        with self._lock:
            status = {
                "active": self._active,
                "reason": self._reason,
                "endsAt": self._end_time.isoformat() if self._end_time else None,
                "remainingSeconds": self.get_remaining_time() if self._active else 0
            }
            return status
    
    def get_remaining_time(self) -> int:
        """Get the remaining time in seconds for the current cooldown."""
        if not self._active or not self._end_time:
            return 0
            
        remaining = (self._end_time - datetime.datetime.now()).total_seconds()
        return max(0, int(remaining))
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update the cooldown configuration."""
        with self._lock:
            # Only update valid keys
            for key in DEFAULT_CONFIG:
                if key in new_config:
                    self.config[key] = new_config[key]
            
            # Save updated config
            self.save_config()
            
            logger.info(f"Updated cooldown configuration: {self.config}")
    
    def start_cooldown(self, reason: str, duration_override: Optional[int] = None) -> None:
        """
        Start a cooldown period.
        
        Args:
            reason: Reason for the cooldown
            duration_override: Optional override for cooldown duration in seconds
        """
        if self._active:
            logger.info(f"Cooldown already active, ignoring new request: {reason}")
            return
            
        with self._lock:
            self._active = True
            self._reason = reason
            self._start_time = datetime.datetime.now()
            
            duration = duration_override if duration_override is not None else self.config["durationSeconds"]
            self._end_time = self._start_time + datetime.timedelta(seconds=duration)
            
            logger.warning(f"Cooldown STARTED: {reason}, duration: {duration}s, ends at: {self._end_time}")
            
            # Publish event
            if self.event_bus:
                self.event_bus.publish(Event(
                    type="cooldown_started",
                    data={
                        "reason": reason,
                        "start_time": self._start_time.isoformat(),
                        "end_time": self._end_time.isoformat(),
                        "duration_seconds": duration
                    }
                ))
    
    def _end_cooldown(self, end_reason: str) -> None:
        """
        End the current cooldown.
        
        Args:
            end_reason: Reason for ending the cooldown (expired/reset)
        """
        if not self._active:
            return
            
        with self._lock:
            previous_reason = self._reason
            
            self._active = False
            self._reason = None
            
            logger.info(f"Cooldown ENDED ({end_reason}). Was active due to: {previous_reason}")
            
            # Publish event
            if self.event_bus:
                self.event_bus.publish(Event(
                    type="cooldown_ended",
                    data={
                        "previous_reason": previous_reason,
                        "end_reason": end_reason,
                        "end_time": datetime.datetime.now().isoformat()
                    }
                ))
    
    def reset_cooldown(self) -> None:
        """Manually reset (end) the current cooldown."""
        if not self._active:
            logger.debug("No active cooldown to reset")
            return
            
        self._end_cooldown("manual_reset")
    
    def register_loss(self) -> None:
        """Register a trading loss and potentially trigger a cooldown."""
        with self._lock:
            self._consecutive_losses += 1
            
            # Check if we should trigger a cooldown
            if self.config["enabled"] and self._consecutive_losses >= self.config["afterConsecutiveLosses"]:
                self.start_cooldown(f"consecutive_losses_{self._consecutive_losses}")
                # Reset the counter
                self._consecutive_losses = 0
    
    def register_profit(self) -> None:
        """Register a trading profit and reset the consecutive loss counter."""
        with self._lock:
            self._consecutive_losses = 0
    
    def _on_trade_executed(self, event: Event) -> None:
        """Handle trade executed events."""
        if not event.data:
            return
            
        # Skip processing during active cooldown
        if self._active:
            return
            
        trade_data = event.data
        if "profit_loss" in trade_data:
            profit_loss = trade_data["profit_loss"]
            if profit_loss < 0:
                self.register_loss()
            else:
                self.register_profit()
    
    def _on_circuit_breaker_triggered(self, event: Event) -> None:
        """Handle circuit breaker triggered events."""
        if not event.data:
            return
            
        trigger_data = event.data
        reason = trigger_data.get("reason", "unknown")
        
        # Start cooldown after circuit breaker trigger if configured
        if self.config["enabled"] and self.config["afterMaxDrawdown"] and reason == "drawdown_exceeded":
            # Use a longer cooldown for drawdown events
            extended_duration = self.config["durationSeconds"] * 2
            self.start_cooldown(f"circuit_breaker_{reason}", extended_duration)
