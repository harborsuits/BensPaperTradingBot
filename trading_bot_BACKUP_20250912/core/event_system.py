#!/usr/bin/env python3
"""
Event system for handling trading bot events.
"""

import logging
from enum import Enum
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Enum representing different types of events."""
    MARKET_DATA = "market_data"
    SIGNAL = "signal"
    ORDER = "order"
    TRADE = "trade"
    POSITION = "position"
    REGIME_CHANGE = "regime_change"
    ERROR = "error"
    SYSTEM = "system"

@dataclass
class Event:
    """Class representing an event in the system."""
    type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    source: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary format."""
        return {
            'type': self.type.value,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'source': self.source
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary format."""
        return cls(
            type=EventType(data['type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            data=data['data'],
            source=data['source']
        )

class EventBus:
    """Event bus for managing event subscriptions and publishing."""
    
    def __init__(self):
        """Initialize the event bus."""
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {}
        
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to {event_type.value} events")
        
    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            callback: Function to remove from subscribers
        """
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(callback)
            logger.debug(f"Unsubscribed from {event_type.value} events")
            
    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event: Event to publish
        """
        if event.type in self._subscribers:
            for callback in self._subscribers[event.type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")
                    
    def clear_subscribers(self, event_type: Optional[EventType] = None) -> None:
        """
        Clear all subscribers for an event type or all event types.
        
        Args:
            event_type: Optional event type to clear subscribers for
        """
        if event_type:
            self._subscribers[event_type] = []
            logger.debug(f"Cleared subscribers for {event_type.value} events")
        else:
            self._subscribers.clear()
            logger.debug("Cleared all subscribers")
