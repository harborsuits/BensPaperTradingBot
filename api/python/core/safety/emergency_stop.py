"""
Emergency stop system for instantly halting all trading activity.

This module provides a global emergency stop that can be triggered manually
to immediately halt all trading activity across the entire system.
"""

import logging
import datetime
import threading
import json
from typing import Dict, Optional, Any
from pathlib import Path

from trading_bot.core.event_bus import EventBus
from trading_bot.core.events import Event

logger = logging.getLogger(__name__)

class EmergencyStopManager:
    """
    Manages the global emergency stop functionality that can halt all trading.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None, config_path: Optional[str] = None):
        """
        Initialize the emergency stop manager.
        
        Args:
            event_bus: Event bus for publishing and subscribing to events
            config_path: Path to configuration file (optional)
        """
        self.event_bus = event_bus or EventBus()
        self.config_path = config_path
        
        # Emergency stop state
        self._active = False
        self._activated_at = None
        self._activated_by = None
        self._reason = None
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Load state if available
        self._load_state()
            
    def _load_state(self) -> None:
        """Load emergency stop state from file if available."""
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    state = json.load(f)
                
                self._active = state.get('active', False)
                self._activated_at = datetime.datetime.fromisoformat(state['activated_at']) if state.get('activated_at') else None
                self._activated_by = state.get('activated_by')
                self._reason = state.get('reason')
                
                logger.info(f"Loaded emergency stop state from {self.config_path}: active={self._active}")
                
                # Publish state if active on startup
                if self._active and self.event_bus:
                    self.event_bus.publish(Event(
                        type="emergency_stop_active",
                        data={
                            "activated_at": self._activated_at.isoformat() if self._activated_at else None,
                            "activated_by": self._activated_by,
                            "reason": self._reason
                        }
                    ))
            except Exception as e:
                logger.error(f"Error loading emergency stop state: {e}")
        
    def _save_state(self) -> None:
        """Save current emergency stop state to file."""
        if not self.config_path:
            return
            
        try:
            state = {
                'active': self._active,
                'activated_at': self._activated_at.isoformat() if self._activated_at else None,
                'activated_by': self._activated_by,
                'reason': self._reason
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Saved emergency stop state to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving emergency stop state: {e}")
    
    def is_active(self) -> bool:
        """Return whether the emergency stop is currently active."""
        return self._active
        
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the emergency stop."""
        with self._lock:
            return {
                "active": self._active,
                "activatedAt": self._activated_at.isoformat() if self._activated_at else None,
                "activatedBy": self._activated_by,
                "reason": self._reason
            }
    
    def activate(self, activated_by: Optional[str] = None, reason: Optional[str] = None) -> bool:
        """
        Activate the emergency stop.
        
        Args:
            activated_by: Identifier of the entity activating the stop (user, system, etc.)
            reason: Reason for the emergency stop
            
        Returns:
            bool: True if emergency stop was activated, False if already active
        """
        if self._active:
            logger.info("Emergency stop already active, ignoring activation request")
            return False
            
        with self._lock:
            self._active = True
            self._activated_at = datetime.datetime.now()
            self._activated_by = activated_by or "system"
            self._reason = reason or "manual_activation"
            
            logger.warning(f"EMERGENCY STOP ACTIVATED by {self._activated_by}: {self._reason}")
            
            # Save state
            self._save_state()
            
            # Publish event
            if self.event_bus:
                self.event_bus.publish(Event(
                    type="emergency_stop_activated",
                    data={
                        "activated_at": self._activated_at.isoformat(),
                        "activated_by": self._activated_by,
                        "reason": self._reason
                    }
                ))
                
            return True
    
    def deactivate(self, deactivated_by: Optional[str] = None) -> bool:
        """
        Deactivate the emergency stop.
        
        Args:
            deactivated_by: Identifier of the entity deactivating the stop
            
        Returns:
            bool: True if emergency stop was deactivated, False if not active
        """
        if not self._active:
            logger.info("Emergency stop not active, ignoring deactivation request")
            return False
            
        with self._lock:
            previous_activated_by = self._activated_by
            previous_reason = self._reason
            
            self._active = False
            self._activated_at = None
            self._activated_by = None
            self._reason = None
            
            logger.warning(f"EMERGENCY STOP DEACTIVATED by {deactivated_by or 'system'}")
            
            # Save state
            self._save_state()
            
            # Publish event
            if self.event_bus:
                self.event_bus.publish(Event(
                    type="emergency_stop_deactivated",
                    data={
                        "deactivated_at": datetime.datetime.now().isoformat(),
                        "deactivated_by": deactivated_by or "system",
                        "previous_activated_by": previous_activated_by,
                        "previous_reason": previous_reason
                    }
                ))
                
            return True
    
    def toggle(self, actor: Optional[str] = None, reason: Optional[str] = None) -> bool:
        """
        Toggle the emergency stop state.
        
        Args:
            actor: Identifier of the entity toggling the stop
            reason: Reason for the emergency stop (if activating)
            
        Returns:
            bool: The new state of the emergency stop (True = active)
        """
        if self._active:
            self.deactivate(actor)
            return False
        else:
            self.activate(actor, reason)
            return True
