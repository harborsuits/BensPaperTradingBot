"""
Central safety manager for coordinating all trading safety guardrails.

This module provides a unified interface to manage all safety systems:
- Circuit breakers to limit losses
- Cooldown periods after losses
- Emergency stop functionality
- Trading mode control (live vs paper)
"""

import logging
import datetime
import threading
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import uuid

from trading_bot.core.event_bus import EventBus
from trading_bot.core.events import Event
from trading_bot.core.safety.circuit_breakers import CircuitBreakerManager
from trading_bot.core.safety.cooldown import CooldownManager
from trading_bot.core.safety.emergency_stop import EmergencyStopManager
from trading_bot.core.safety.trading_mode import TradingModeManager

logger = logging.getLogger(__name__)

class SafetyEvent:
    """Represents a safety-related event for logging and auditing."""
    
    def __init__(self, 
                 event_type: str, 
                 action: str, 
                 reason: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None,
                 actor: Optional[str] = None):
        """
        Initialize a new safety event.
        
        Args:
            event_type: Type of safety event (emergency_stop, circuit_breaker, cooldown, mode_change)
            action: Action taken (activated, deactivated, triggered, reset)
            reason: Reason for the event
            details: Additional details about the event
            actor: Entity that triggered the event
        """
        self.id = str(uuid.uuid4())
        self.type = event_type
        self.action = action
        self.reason = reason
        self.details = details or {}
        self.actor = actor
        self.timestamp = datetime.datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type,
            "action": self.action,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "details": self.details,
            "actor": self.actor
        }

class SafetyManager:
    """
    Central manager for all trading safety guardrails.
    
    This class coordinates circuit breakers, cooldowns, emergency stop,
    and trading mode functionality to provide a unified safety system.
    """
    
    def __init__(self, 
                 event_bus: Optional[EventBus] = None, 
                 config_dir: Optional[str] = None,
                 safety_db_path: Optional[str] = None):
        """
        Initialize the safety manager.
        
        Args:
            event_bus: Event bus for publishing and subscribing to events
            config_dir: Directory for configuration files
            safety_db_path: Path to the safety events database file
        """
        self.event_bus = event_bus or EventBus()
        
        # Set up configuration paths
        if config_dir:
            os.makedirs(config_dir, exist_ok=True)
            self.circuit_breaker_config = os.path.join(config_dir, "circuit_breakers.json")
            self.cooldown_config = os.path.join(config_dir, "cooldown.json")
            self.emergency_stop_config = os.path.join(config_dir, "emergency_stop.json")
            self.trading_mode_config = os.path.join(config_dir, "trading_mode.json")
        else:
            self.circuit_breaker_config = None
            self.cooldown_config = None
            self.emergency_stop_config = None
            self.trading_mode_config = None
        
        # Safety event database
        self.safety_db_path = safety_db_path
        self._events: List[SafetyEvent] = []
        self._load_events()
        
        # Initialize components
        self.circuit_breaker = CircuitBreakerManager(
            event_bus=self.event_bus,
            config_path=self.circuit_breaker_config
        )
        
        self.cooldown = CooldownManager(
            event_bus=self.event_bus,
            config_path=self.cooldown_config
        )
        
        self.emergency_stop = EmergencyStopManager(
            event_bus=self.event_bus,
            config_path=self.emergency_stop_config
        )
        
        self.trading_mode = TradingModeManager(
            event_bus=self.event_bus,
            config_path=self.trading_mode_config
        )
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Subscribe to safety events
        self._setup_event_listeners()
        
        logger.info("Safety manager initialized")
    
    def _setup_event_listeners(self) -> None:
        """Set up event listeners for safety-related events."""
        if not self.event_bus:
            return
            
        # Circuit breaker events
        self.event_bus.subscribe("circuit_breaker_triggered", self._on_circuit_breaker_triggered)
        self.event_bus.subscribe("circuit_breaker_reset", self._on_circuit_breaker_reset)
        
        # Cooldown events
        self.event_bus.subscribe("cooldown_started", self._on_cooldown_started)
        self.event_bus.subscribe("cooldown_ended", self._on_cooldown_ended)
        
        # Emergency stop events
        self.event_bus.subscribe("emergency_stop_activated", self._on_emergency_stop_activated)
        self.event_bus.subscribe("emergency_stop_deactivated", self._on_emergency_stop_deactivated)
        
        # Trading mode events
        self.event_bus.subscribe("trading_mode_changed", self._on_trading_mode_changed)
    
    def _load_events(self) -> None:
        """Load safety events from the database file."""
        if not self.safety_db_path or not os.path.exists(self.safety_db_path):
            return
            
        try:
            with open(self.safety_db_path, 'r') as f:
                events_data = json.load(f)
                
            for event_data in events_data:
                event = SafetyEvent(
                    event_type=event_data.get('type', 'unknown'),
                    action=event_data.get('action', 'unknown'),
                    reason=event_data.get('reason'),
                    details=event_data.get('details'),
                    actor=event_data.get('actor')
                )
                event.id = event_data.get('id', str(uuid.uuid4()))
                event.timestamp = datetime.datetime.fromisoformat(event_data.get('timestamp'))
                self._events.append(event)
                
            logger.info(f"Loaded {len(self._events)} safety events from {self.safety_db_path}")
        except Exception as e:
            logger.error(f"Error loading safety events: {e}")
            self._events = []
    
    def _save_events(self) -> None:
        """Save safety events to the database file."""
        if not self.safety_db_path:
            return
            
        try:
            # Create parent directory if needed
            os.makedirs(os.path.dirname(self.safety_db_path), exist_ok=True)
            
            with self._lock:
                events_data = [event.to_dict() for event in self._events]
                
            with open(self.safety_db_path, 'w') as f:
                json.dump(events_data, f, indent=2)
                
            logger.debug(f"Saved {len(self._events)} safety events to {self.safety_db_path}")
        except Exception as e:
            logger.error(f"Error saving safety events: {e}")
    
    def add_event(self, event: SafetyEvent) -> None:
        """
        Add a safety event to the database.
        
        Args:
            event: The safety event to add
        """
        with self._lock:
            self._events.append(event)
            
        # Save to disk
        self._save_events()
        
        # Publish event
        if self.event_bus:
            self.event_bus.publish(Event(
                type="safety_event_recorded",
                data=event.to_dict()
            ))
    
    def get_events(self, limit: int = 50, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get safety events from the database.
        
        Args:
            limit: Maximum number of events to return
            event_type: Optional filter by event type
            
        Returns:
            List of safety events as dictionaries
        """
        with self._lock:
            filtered_events = [
                event for event in self._events
                if event_type is None or event.type == event_type
            ]
            
            # Sort by timestamp (newest first)
            sorted_events = sorted(
                filtered_events,
                key=lambda e: e.timestamp,
                reverse=True
            )
            
            # Apply limit
            limited_events = sorted_events[:limit]
            
            # Convert to dictionaries
            return [event.to_dict() for event in limited_events]
    
    def check_trading_allowed(self) -> Tuple[bool, str]:
        """
        Check if trading is currently allowed by all safety systems.
        
        Returns:
            Tuple of (allowed, reason)
            - allowed: True if trading is allowed, False otherwise
            - reason: Reason trading is not allowed, or None if allowed
        """
        # Check emergency stop (highest priority)
        if self.emergency_stop.is_active():
            return False, "emergency_stop_active"
            
        # Check circuit breaker
        if self.circuit_breaker.is_active():
            return False, "circuit_breaker_active"
            
        # Check cooldown
        if self.cooldown.is_active():
            return False, "cooldown_active"
            
        # All checks passed
        return True, ""
    
    def get_safety_status(self) -> Dict[str, Any]:
        """
        Get the comprehensive status of all safety systems.
        
        Returns:
            Dictionary with the status of all safety systems
        """
        return {
            "tradingMode": self.trading_mode.get_mode(),
            "emergencyStopActive": self.emergency_stop.is_active(),
            "circuitBreakers": self.circuit_breaker.get_status(),
            "cooldowns": self.cooldown.get_status(),
            "tradingAllowed": self.check_trading_allowed()[0]
        }
    
    def set_emergency_stop(self, active: bool, actor: Optional[str] = None, reason: Optional[str] = None) -> bool:
        """
        Set the emergency stop state.
        
        Args:
            active: Whether to activate (True) or deactivate (False) the emergency stop
            actor: Entity setting the emergency stop
            reason: Reason for the change (if activating)
            
        Returns:
            Boolean indicating whether the operation was successful
        """
        if active:
            return self.emergency_stop.activate(actor, reason)
        else:
            return self.emergency_stop.deactivate(actor)
    
    def set_trading_mode(self, mode: str, actor: Optional[str] = None) -> bool:
        """
        Set the trading mode.
        
        Args:
            mode: The trading mode ('live' or 'paper')
            actor: Entity changing the mode
            
        Returns:
            Boolean indicating whether the operation was successful
        """
        if mode not in ['live', 'paper']:
            logger.error(f"Invalid trading mode: {mode}")
            return False
            
        return self.trading_mode.set_mode(mode, actor)
    
    def reset_circuit_breaker(self, actor: Optional[str] = None) -> None:
        """
        Reset the circuit breaker.
        
        Args:
            actor: Entity resetting the circuit breaker
        """
        if not self.circuit_breaker.is_active():
            logger.info("Circuit breaker not active, no need to reset")
            return
            
        self.circuit_breaker.reset()
        
        # Record event
        self.add_event(SafetyEvent(
            event_type="circuit_breaker",
            action="reset",
            actor=actor
        ))
    
    def reset_cooldown(self, actor: Optional[str] = None) -> None:
        """
        Reset the cooldown.
        
        Args:
            actor: Entity resetting the cooldown
        """
        if not self.cooldown.is_active():
            logger.info("Cooldown not active, no need to reset")
            return
            
        self.cooldown.reset_cooldown()
        
        # Record event
        self.add_event(SafetyEvent(
            event_type="cooldown",
            action="reset",
            actor=actor
        ))
    
    def update_circuit_breaker_config(self, config: Dict[str, Any]) -> None:
        """
        Update circuit breaker configuration.
        
        Args:
            config: New configuration settings
        """
        self.circuit_breaker.update_config(config)
    
    def update_cooldown_config(self, config: Dict[str, Any]) -> None:
        """
        Update cooldown configuration.
        
        Args:
            config: New configuration settings
        """
        self.cooldown.update_config(config)
    
    # Event handlers
    
    def _on_circuit_breaker_triggered(self, event: Event) -> None:
        """Handle circuit breaker triggered events."""
        if not event.data:
            return
            
        self.add_event(SafetyEvent(
            event_type="circuit_breaker",
            action="triggered",
            reason=event.data.get("reason"),
            details=event.data.get("details")
        ))
    
    def _on_circuit_breaker_reset(self, event: Event) -> None:
        """Handle circuit breaker reset events."""
        if not event.data:
            return
            
        self.add_event(SafetyEvent(
            event_type="circuit_breaker",
            action="reset",
            details=event.data
        ))
    
    def _on_cooldown_started(self, event: Event) -> None:
        """Handle cooldown started events."""
        if not event.data:
            return
            
        self.add_event(SafetyEvent(
            event_type="cooldown",
            action="activated",
            reason=event.data.get("reason"),
            details={
                "start_time": event.data.get("start_time"),
                "end_time": event.data.get("end_time"),
                "duration_seconds": event.data.get("duration_seconds")
            }
        ))
    
    def _on_cooldown_ended(self, event: Event) -> None:
        """Handle cooldown ended events."""
        if not event.data:
            return
            
        self.add_event(SafetyEvent(
            event_type="cooldown",
            action="deactivated",
            reason=event.data.get("end_reason"),
            details=event.data
        ))
    
    def _on_emergency_stop_activated(self, event: Event) -> None:
        """Handle emergency stop activated events."""
        if not event.data:
            return
            
        self.add_event(SafetyEvent(
            event_type="emergency_stop",
            action="activated",
            reason=event.data.get("reason"),
            actor=event.data.get("activated_by")
        ))
    
    def _on_emergency_stop_deactivated(self, event: Event) -> None:
        """Handle emergency stop deactivated events."""
        if not event.data:
            return
            
        self.add_event(SafetyEvent(
            event_type="emergency_stop",
            action="deactivated",
            actor=event.data.get("deactivated_by")
        ))
    
    def _on_trading_mode_changed(self, event: Event) -> None:
        """Handle trading mode changed events."""
        if not event.data:
            return
            
        self.add_event(SafetyEvent(
            event_type="mode_change",
            action="changed",
            details={
                "previous_mode": event.data.get("previous_mode"),
                "new_mode": event.data.get("new_mode")
            },
            actor=event.data.get("changed_by")
        ))
