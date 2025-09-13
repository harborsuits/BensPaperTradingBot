"""
Trading mode manager for controlling live vs paper trading modes.

This module provides functionality to control and enforce trading modes
(live or paper) across the entire trading system.
"""

import logging
import datetime
import threading
import json
from typing import Dict, Optional, Any, Literal
from pathlib import Path

from trading_bot.core.event_bus import EventBus
from trading_bot.core.events import Event

logger = logging.getLogger(__name__)

# Trading mode type
TradingMode = Literal['live', 'paper']

class TradingModeManager:
    """
    Manages and enforces the trading mode (live or paper) across the trading system.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None, config_path: Optional[str] = None):
        """
        Initialize the trading mode manager.
        
        Args:
            event_bus: Event bus for publishing and subscribing to events
            config_path: Path to configuration file for mode settings
        """
        self.event_bus = event_bus or EventBus()
        self.config_path = config_path
        
        # Default to paper trading for safety
        self._mode = 'paper'
        self._last_changed = None
        self._changed_by = None
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Load state if available
        self._load_state()
            
    def _load_state(self) -> None:
        """Load trading mode state from file if available."""
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    state = json.load(f)
                
                mode = state.get('mode', 'paper')
                # Validate the mode is valid
                self._mode = mode if mode in ['live', 'paper'] else 'paper'
                
                self._last_changed = datetime.datetime.fromisoformat(state['last_changed']) if state.get('last_changed') else None
                self._changed_by = state.get('changed_by')
                
                logger.info(f"Loaded trading mode state from {self.config_path}: mode={self._mode}")
                
                # Publish current mode on startup
                if self.event_bus:
                    self.event_bus.publish(Event(
                        type="trading_mode_current",
                        data={
                            "mode": self._mode,
                            "last_changed": self._last_changed.isoformat() if self._last_changed else None,
                            "changed_by": self._changed_by
                        }
                    ))
            except Exception as e:
                logger.error(f"Error loading trading mode state: {e}")
                # Default to paper trading on any error
                self._mode = 'paper'
        
    def _save_state(self) -> None:
        """Save current trading mode state to file."""
        if not self.config_path:
            return
            
        try:
            state = {
                'mode': self._mode,
                'last_changed': self._last_changed.isoformat() if self._last_changed else None,
                'changed_by': self._changed_by
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Saved trading mode state to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving trading mode state: {e}")
    
    def get_mode(self) -> TradingMode:
        """Get the current trading mode."""
        return self._mode
        
    def is_live_trading(self) -> bool:
        """Check if currently in live trading mode."""
        return self._mode == 'live'
        
    def is_paper_trading(self) -> bool:
        """Check if currently in paper trading mode."""
        return self._mode == 'paper'
        
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the trading mode."""
        with self._lock:
            return {
                "mode": self._mode,
                "lastChanged": self._last_changed.isoformat() if self._last_changed else None,
                "changedBy": self._changed_by
            }
    
    def set_mode(self, mode: TradingMode, changed_by: Optional[str] = None) -> bool:
        """
        Set the trading mode.
        
        Args:
            mode: The trading mode to set ('live' or 'paper')
            changed_by: Identifier of the entity changing the mode
            
        Returns:
            bool: True if mode was changed, False if already in that mode
        """
        if mode not in ['live', 'paper']:
            logger.error(f"Invalid trading mode: {mode}, must be 'live' or 'paper'")
            return False
            
        if self._mode == mode:
            logger.info(f"Already in {mode} trading mode, no change needed")
            return False
            
        with self._lock:
            previous_mode = self._mode
            self._mode = mode
            self._last_changed = datetime.datetime.now()
            self._changed_by = changed_by or "system"
            
            logger.warning(f"Trading mode changed from {previous_mode} to {mode} by {self._changed_by}")
            
            # Save state
            self._save_state()
            
            # Publish event
            if self.event_bus:
                self.event_bus.publish(Event(
                    type="trading_mode_changed",
                    data={
                        "previous_mode": previous_mode,
                        "new_mode": mode,
                        "changed_at": self._last_changed.isoformat(),
                        "changed_by": self._changed_by
                    }
                ))
                
            return True
    
    def set_live_trading(self, changed_by: Optional[str] = None) -> bool:
        """
        Set to live trading mode.
        
        Args:
            changed_by: Identifier of the entity changing the mode
            
        Returns:
            bool: True if mode was changed, False if already in live mode
        """
        return self.set_mode('live', changed_by)
    
    def set_paper_trading(self, changed_by: Optional[str] = None) -> bool:
        """
        Set to paper trading mode.
        
        Args:
            changed_by: Identifier of the entity changing the mode
            
        Returns:
            bool: True if mode was changed, False if already in paper mode
        """
        return self.set_mode('paper', changed_by)
    
    def toggle_mode(self, changed_by: Optional[str] = None) -> TradingMode:
        """
        Toggle between live and paper trading modes.
        
        Args:
            changed_by: Identifier of the entity toggling the mode
            
        Returns:
            The new trading mode after toggling
        """
        new_mode = 'paper' if self._mode == 'live' else 'live'
        self.set_mode(new_mode, changed_by)
        return new_mode
