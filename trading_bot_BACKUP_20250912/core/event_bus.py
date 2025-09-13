"""
Event Bus System for BensBot

This module implements an event-driven architecture for the trading system,
allowing components to publish and subscribe to events without tight coupling.

Features:
- Event publishing and subscription
- Priority-based event processing
- Event history for replay and analysis
- Event correlation detection
"""
import logging
import uuid
import heapq
from typing import Callable, Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, Counter, deque

from trading_bot.core.constants import EventType

logger = logging.getLogger(__name__)

@dataclass
class Event:
    """Event class for publishing to the event bus."""
    
    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class EventBus:
    """
    Enhanced event bus implementation for publishing and subscribing to events.
    
    This is a central component that facilitates communication between
    different parts of the trading system without tight coupling.
    
    Features:
    - Priority-based event processing
    - Event history for replay and analysis
    - Event correlation detection
    - Transaction grouping for related events
    """
    
    # Define event priority levels (lower number = higher priority)
    PRIORITY_CRITICAL = 0   # Risk alerts, system errors
    PRIORITY_HIGH = 1       # Trading signals, order events
    PRIORITY_MEDIUM = 2     # Market data updates
    PRIORITY_LOW = 3        # Logging, metrics
    
    # Map event types to priority levels
    DEFAULT_PRIORITY_MAP = {
        # Critical events
        EventType.RISK_LIMIT_REACHED: PRIORITY_CRITICAL,
        EventType.DRAWDOWN_ALERT: PRIORITY_CRITICAL,
        EventType.DRAWDOWN_THRESHOLD_EXCEEDED: PRIORITY_CRITICAL,
        EventType.ERROR_OCCURRED: PRIORITY_CRITICAL,
        EventType.CORRELATION_RISK_ALERT: PRIORITY_CRITICAL,
        
        # High priority events
        EventType.SIGNAL_GENERATED: PRIORITY_HIGH,
        EventType.ORDER_CREATED: PRIORITY_HIGH,
        EventType.ORDER_FILLED: PRIORITY_HIGH,
        EventType.ORDER_REJECTED: PRIORITY_HIGH,
        EventType.TRADE_EXECUTED: PRIORITY_HIGH,
        EventType.MARKET_REGIME_CHANGED: PRIORITY_HIGH,
        
        # Medium priority events
        EventType.MARKET_DATA_RECEIVED: PRIORITY_MEDIUM,
        EventType.TICK_RECEIVED: PRIORITY_MEDIUM,
        EventType.BAR_CLOSED: PRIORITY_MEDIUM,
        EventType.PATTERN_DISCOVERED: PRIORITY_MEDIUM,
        
        # Low priority events
        EventType.LOG_MESSAGE: PRIORITY_LOW,
        EventType.HEALTH_CHECK: PRIORITY_LOW,
    }
    
    def __init__(self, verbose_logging: bool = False, max_history: int = 1000):
        """
        Initialize the enhanced event bus.
        
        Args:
            verbose_logging: Whether to log all events (can be very noisy)
            max_history: Maximum number of events to store in history
        """
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {}
        self._global_subscribers: List[Callable[[Event], None]] = []
        self._verbose_logging = verbose_logging
        
        # Enhanced features
        self._event_history = deque(maxlen=max_history)  # Circular buffer for event history
        self._event_queue = []  # Priority queue for events
        self._processing_events = False  # Flag to prevent recursive event processing
        self._priority_map = self.DEFAULT_PRIORITY_MAP.copy()  # Event type to priority mapping
        
        # Event correlation tracking
        self._recent_events = defaultdict(list)  # Event type to recent events
        self._correlation_patterns = defaultdict(Counter)  # Track event type sequences
        self._correlation_threshold = 5  # Minimum count to establish a correlation
        self._correlation_window = timedelta(minutes=5)  # Time window for correlation
        
        # Transaction grouping
        self._current_transaction_id = None
        self._transaction_events = defaultdict(list)  # Transaction ID to events
        
        logger.info("Enhanced event bus initialized with history capacity: %d", max_history)
    
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """
        Subscribe to a specific event type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to event type: {event_type}")
    
    def subscribe_all(self, callback: Callable[[Event], None]) -> None:
        """
        Subscribe to all events.
        
        Args:
            callback: Function to call for any event
        """
        self._global_subscribers.append(callback)
        logger.debug("Subscribed to all events")
    
    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """
        Unsubscribe from a specific event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            callback: Function to remove from subscribers
        """
        if event_type in self._subscribers and callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
            logger.debug(f"Unsubscribed from event type: {event_type}")
    
    def unsubscribe_all(self, callback: Callable[[Event], None]) -> None:
        """
        Unsubscribe from all events.
        
        Args:
            callback: Function to remove from global subscribers
        """
        if callback in self._global_subscribers:
            self._global_subscribers.remove(callback)
            logger.debug("Unsubscribed from all events")
    
    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers with priority handling.
        
        Args:
            event: Event to publish
        """
        if self._verbose_logging:
            logger.debug(f"Publishing event: {event.event_type}, data: {event.data}, source: {event.source}")
        
        # Add event to history
        self._event_history.append(event)
        
        # Add to correlation tracking
        self._track_for_correlation(event)
        
        # Get priority for this event type (default to MEDIUM if not specified)
        priority = self._priority_map.get(event.event_type, self.PRIORITY_MEDIUM)
        
        # Add to priority queue with transaction grouping
        transaction_id = event.data.get('transaction_id', event.event_id)
        heapq.heappush(self._event_queue, (priority, datetime.now(), event, transaction_id))
        
        # Process event queue if not already processing
        if not self._processing_events:
            self._process_event_queue()
    
    def _process_event_queue(self):
        """
        Process events in the queue according to priority.
        """
        self._processing_events = True
        try:
            # Process all events in the queue
            while self._event_queue:
                # Get highest priority event
                _, _, event, transaction_id = heapq.heappop(self._event_queue)
                
                # Add to transaction tracking
                self._transaction_events[transaction_id].append(event)
                
                # Notify specific subscribers
                if event.event_type in self._subscribers:
                    for callback in self._subscribers[event.event_type]:
                        try:
                            callback(event)
                        except Exception as e:
                            logger.error(f"Error in event callback for {event.event_type}: {str(e)}")
                
                # Notify global subscribers
                for callback in self._global_subscribers:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Error in global event callback for {event.event_type}: {str(e)}")
                        
                # Check if this was the last event in a transaction
                self._check_transaction_completion(transaction_id)
                
        finally:
            self._processing_events = False
    
    def _track_for_correlation(self, event: Event):
        """
        Track events for correlation detection.
        
        Args:
            event: Event to track
        """
        now = datetime.now()
        
        # Add event to recent events
        self._recent_events[event.event_type].append((now, event))
        
        # Clean up old events
        for event_type in list(self._recent_events.keys()):
            self._recent_events[event_type] = [
                (ts, evt) for ts, evt in self._recent_events[event_type]
                if now - ts <= self._correlation_window
            ]
            
        # Look for patterns: events that frequently follow this event type
        for other_type in self._recent_events.keys():
            if other_type != event.event_type:
                # Find events of other_type that occurred within 5 seconds after events of this type
                for prev_time, prev_event in self._recent_events[event.event_type][:-1]:  # Exclude current event
                    for curr_time, curr_event in self._recent_events[other_type]:
                        time_diff = (curr_time - prev_time).total_seconds()
                        if 0 < time_diff < 5:  # Within 5 seconds
                            pattern = (prev_event.event_type, curr_event.event_type)
                            self._correlation_patterns[prev_event.event_type][curr_event.event_type] += 1
        
    def _check_transaction_completion(self, transaction_id: str):
        """
        Check if a transaction is complete and publish a summary event.
        
        Args:
            transaction_id: ID of the transaction to check
        """
        # Simple heuristic: if no events in queue with this transaction ID, consider it complete
        if transaction_id in self._transaction_events and not any(t_id == transaction_id for _, _, _, t_id in self._event_queue):
            events = self._transaction_events.pop(transaction_id)
            
            # Only summarize transactions with multiple events
            if len(events) > 1:
                summary = {
                    'transaction_id': transaction_id,
                    'event_count': len(events),
                    'event_types': [e.event_type for e in events],
                    'start_time': min(e.timestamp for e in events),
                    'end_time': max(e.timestamp for e in events),
                    'duration_ms': (max(e.timestamp for e in events) - min(e.timestamp for e in events)).total_seconds() * 1000
                }
                
                # Create a summary event - but don't process it as part of the transaction to avoid recursion
                summary_event = Event(
                    event_type='transaction_completed',
                    data=summary,
                    source='event_bus'
                )
    
    def create_and_publish(self, 
                         event_type: EventType, 
                         data: Optional[Dict[str, Any]] = None, 
                         source: str = "") -> Event:
        """
        Create and publish an event in one step.
        
        Args:
            event_type: Type of event to create
            data: Event data
            source: Source of the event
        
        Returns:
            Created event
        """
        event = Event(
            event_type=event_type,
            data=data or {},
            source=source,
            timestamp=datetime.now()
        )
        self.publish(event)
        return event
    
    def clear_subscribers(self) -> None:
        """Clear all subscribers from the event bus."""
        self._subscribers = {}
        self._global_subscribers = []
        logger.info("All subscribers cleared from event bus")
    
    def set_event_priority(self, event_type: EventType, priority: int) -> None:
        """
        Set the priority level for a specific event type.
        
        Args:
            event_type: The event type to set priority for
            priority: Priority level (0-3, lower is higher priority)
        """
        if priority not in range(4):  # 0-3
            raise ValueError("Priority must be between 0 and 3")
        self._priority_map[event_type] = priority
        logger.debug(f"Set priority {priority} for event type {event_type}")
    
    def get_event_history(self, event_type: Optional[EventType] = None, 
                         limit: int = 100) -> List[Event]:
        """
        Get historical events, optionally filtered by type.
        
        Args:
            event_type: Optional filter for specific event type
            limit: Maximum number of events to return
            
        Returns:
            List of historical events
        """
        if event_type:
            return [e for e in self._event_history if e.event_type == event_type][-limit:]
        return list(self._event_history)[-limit:]
    
    def replay_events(self, events: List[Event]) -> None:
        """
        Replay a sequence of historical events.
        
        Args:
            events: List of events to replay
        """
        logger.info(f"Replaying {len(events)} historical events")
        for event in events:
            # Mark as replay to avoid recursion and confusion with real-time events
            event.data['is_replay'] = True
            self.publish(event)
            
    def get_correlations(self, min_confidence: float = 0.7) -> Dict[Tuple[EventType, EventType], float]:
        """
        Get discovered event correlations that exceed confidence threshold.
        
        Args:
            min_confidence: Minimum confidence level for correlations
            
        Returns:
            Dictionary of correlated event pairs with confidence scores
        """
        correlations = {}
        
        for event_type, counters in self._correlation_patterns.items():
            total = sum(counters.values())
            if total >= self._correlation_threshold:
                for other_type, count in counters.items():
                    confidence = count / total
                    if confidence >= min_confidence:
                        correlations[(event_type, other_type)] = confidence
                        
        return correlations


# Global event bus instance
_global_event_bus: Optional[EventBus] = None


def get_global_event_bus() -> EventBus:
    """
    Get the global event bus instance.
    
    Returns:
        The global event bus
    """
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus
