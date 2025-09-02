"""
Event Handler System

Provides a clean architecture for handling different market events with:
- Separate handlers for different event types
- Clean initialization and shutdown procedures
- Clear separation of concerns

Based on concepts from EA31337-Libre.
"""

import logging
import threading
import time
import abc
from typing import Dict, List, Any, Optional, Union, Callable, Type, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import queue

from trading_bot.event_system import EventBus, MessageQueue, Event

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Types of events that can be handled by the system"""
    MARKET_DATA = auto()
    TICK = auto()
    PRICE_CHANGE = auto()
    NEW_CANDLE = auto()
    INDICATOR = auto()
    SIGNAL = auto()
    PENDING_ORDER = auto()
    ORDER_FILLED = auto()
    ORDER_CANCELLED = auto()
    POSITION_OPENED = auto()
    POSITION_MODIFIED = auto()
    POSITION_CLOSED = auto()
    TRADE = auto()
    NEWS = auto()
    SYSTEM = auto()
    CUSTOM = auto()

@dataclass
class HandlerMetadata:
    """Metadata for an event handler"""
    name: str
    event_types: Set[EventType]
    priority: int = 0
    is_active: bool = True
    stats: Dict[str, Any] = field(default_factory=dict)
    last_execution: Optional[datetime] = None
    execution_count: int = 0
    avg_execution_time: float = 0.0

class BaseEventHandler(abc.ABC):
    """Base class for all event handlers"""
    
    def __init__(self, name: str, event_types: List[EventType], priority: int = 0):
        """
        Initialize the event handler
        
        Args:
            name: Name of the handler
            event_types: List of event types this handler can process
            priority: Priority of the handler (higher = processed first)
        """
        self.metadata = HandlerMetadata(
            name=name,
            event_types=set(event_types),
            priority=priority
        )
        
        self._is_initialized = False
        self._is_running = False
        self._lock = threading.RLock()
    
    def initialize(self):
        """Initialize the handler"""
        with self._lock:
            if not self._is_initialized:
                try:
                    self._initialize()
                    self._is_initialized = True
                    logger.info(f"Handler {self.metadata.name} initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize handler {self.metadata.name}: {e}")
                    raise
    
    def shutdown(self):
        """Shutdown the handler"""
        with self._lock:
            if self._is_initialized:
                try:
                    self._shutdown()
                    self._is_initialized = False
                    self._is_running = False
                    logger.info(f"Handler {self.metadata.name} shutdown")
                except Exception as e:
                    logger.error(f"Failed to shutdown handler {self.metadata.name}: {e}")
                    raise
    
    def start(self):
        """Start the handler"""
        with self._lock:
            if not self._is_running and self._is_initialized:
                try:
                    self._start()
                    self._is_running = True
                    logger.info(f"Handler {self.metadata.name} started")
                except Exception as e:
                    logger.error(f"Failed to start handler {self.metadata.name}: {e}")
                    raise
    
    def stop(self):
        """Stop the handler"""
        with self._lock:
            if self._is_running:
                try:
                    self._stop()
                    self._is_running = False
                    logger.info(f"Handler {self.metadata.name} stopped")
                except Exception as e:
                    logger.error(f"Failed to stop handler {self.metadata.name}: {e}")
                    raise
    
    def handle_event(self, event: Event) -> bool:
        """
        Handle an event
        
        Args:
            event: Event to handle
            
        Returns:
            True if the event was handled, False otherwise
        """
        if not self._is_running or not self.metadata.is_active:
            return False
            
        # Check if this handler can process this event type
        event_type = event.metadata.get('type')
        if event_type is None or EventType(event_type) not in self.metadata.event_types:
            return False
        
        # Process the event
        start_time = time.time()
        try:
            result = self._handle_event(event)
            
            # Update stats
            self.metadata.last_execution = datetime.now()
            self.metadata.execution_count += 1
            
            execution_time = time.time() - start_time
            # Update average execution time with exponential moving average
            if self.metadata.avg_execution_time == 0:
                self.metadata.avg_execution_time = execution_time
            else:
                alpha = 0.1  # EMA factor
                self.metadata.avg_execution_time = (1 - alpha) * self.metadata.avg_execution_time + alpha * execution_time
            
            self.metadata.stats['last_execution_time'] = execution_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling event in {self.metadata.name}: {e}")
            self.metadata.stats['last_error'] = str(e)
            self.metadata.stats['last_error_time'] = datetime.now()
            return False
    
    @abc.abstractmethod
    def _initialize(self):
        """Initialize handler resources"""
        pass
    
    @abc.abstractmethod
    def _shutdown(self):
        """Clean up handler resources"""
        pass
    
    def _start(self):
        """Start handler processing"""
        pass
    
    def _stop(self):
        """Stop handler processing"""
        pass
    
    @abc.abstractmethod
    def _handle_event(self, event: Event) -> bool:
        """
        Process an event
        
        Args:
            event: Event to handle
            
        Returns:
            True if the event was handled, False otherwise
        """
        pass
    
    def __str__(self):
        return f"{self.metadata.name} (priority: {self.metadata.priority}, active: {self.metadata.is_active})"

class EventHandlerManager:
    """
    Manages the registration and execution of event handlers.
    
    This class is responsible for:
    1. Registering and unregistering handlers
    2. Dispatching events to appropriate handlers
    3. Managing handler lifecycle
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the event handler manager
        
        Args:
            event_bus: Optional event bus to subscribe to
        """
        self.handlers: Dict[str, BaseEventHandler] = {}
        self.event_bus = event_bus
        self.event_types_map: Dict[EventType, List[BaseEventHandler]] = {
            event_type: [] for event_type in EventType
        }
        
        self._lock = threading.RLock()
        self._is_running = False
        self._event_thread = None
        self._stop_event = threading.Event()
        
        # Register with event bus if provided
        if event_bus:
            event_bus.subscribe("event_handler_manager", self._on_event)
    
    def register_handler(self, handler: BaseEventHandler) -> bool:
        """
        Register a new event handler
        
        Args:
            handler: Handler to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        with self._lock:
            name = handler.metadata.name
            if name in self.handlers:
                logger.warning(f"Handler {name} already registered")
                return False
            
            # Add to handlers dictionary
            self.handlers[name] = handler
            
            # Add to event types map
            for event_type in handler.metadata.event_types:
                self.event_types_map[event_type].append(handler)
                # Sort handlers by priority
                self.event_types_map[event_type].sort(key=lambda h: h.metadata.priority, reverse=True)
            
            # Initialize the handler
            handler.initialize()
            
            # Start if manager is running
            if self._is_running:
                handler.start()
            
            logger.info(f"Registered handler: {name}")
            return True
    
    def unregister_handler(self, name: str) -> bool:
        """
        Unregister an event handler
        
        Args:
            name: Name of the handler to unregister
            
        Returns:
            True if unregistration was successful, False otherwise
        """
        with self._lock:
            if name not in self.handlers:
                logger.warning(f"Handler {name} not registered")
                return False
            
            handler = self.handlers[name]
            
            # Stop and shutdown the handler
            handler.stop()
            handler.shutdown()
            
            # Remove from event types map
            for event_type in handler.metadata.event_types:
                if handler in self.event_types_map[event_type]:
                    self.event_types_map[event_type].remove(handler)
            
            # Remove from handlers dictionary
            del self.handlers[name]
            
            logger.info(f"Unregistered handler: {name}")
            return True
    
    def start(self):
        """Start all registered handlers"""
        with self._lock:
            if self._is_running:
                logger.warning("Event handler manager already running")
                return
            
            # Start all handlers
            for handler in self.handlers.values():
                try:
                    handler.start()
                except Exception as e:
                    logger.error(f"Failed to start handler {handler.metadata.name}: {e}")
            
            self._is_running = True
            self._stop_event.clear()
            
            # Start processing thread if event bus is not provided
            if not self.event_bus:
                self._event_thread = threading.Thread(target=self._process_events, daemon=True)
                self._event_thread.start()
            
            logger.info("Event handler manager started")
    
    def stop(self):
        """Stop all registered handlers"""
        with self._lock:
            if not self._is_running:
                logger.warning("Event handler manager not running")
                return
            
            # Stop all handlers
            for handler in self.handlers.values():
                try:
                    handler.stop()
                except Exception as e:
                    logger.error(f"Failed to stop handler {handler.metadata.name}: {e}")
            
            self._is_running = False
            self._stop_event.set()
            
            # Wait for processing thread to finish
            if self._event_thread and self._event_thread.is_alive():
                self._event_thread.join(timeout=5.0)
            
            logger.info("Event handler manager stopped")
    
    def dispatch_event(self, event: Event) -> int:
        """
        Dispatch an event to all registered handlers
        
        Args:
            event: Event to dispatch
            
        Returns:
            Number of handlers that processed the event
        """
        if not self._is_running:
            return 0
            
        event_type_str = event.metadata.get('type')
        if event_type_str is None:
            logger.warning(f"Event has no type: {event}")
            return 0
            
        try:
            event_type = EventType(event_type_str)
        except (ValueError, TypeError):
            logger.warning(f"Unknown event type: {event_type_str}")
            return 0
        
        handlers = self.event_types_map.get(event_type, [])
        if not handlers:
            return 0
        
        handled_count = 0
        for handler in handlers:
            if handler.handle_event(event):
                handled_count += 1
        
        return handled_count
    
    def _on_event(self, event: Event):
        """Callback for events from event bus"""
        self.dispatch_event(event)
    
    def _process_events(self):
        """Process events in the queue (used when no event bus is provided)"""
        logger.info("Event processing thread started")
        while not self._stop_event.is_set():
            # Process any manually dispatched events
            time.sleep(0.01)  # Prevent CPU spinning
        logger.info("Event processing thread stopped")
    
    def get_handler(self, name: str) -> Optional[BaseEventHandler]:
        """
        Get a handler by name
        
        Args:
            name: Name of the handler
            
        Returns:
            Handler object or None if not found
        """
        return self.handlers.get(name)
    
    def get_handlers_by_type(self, event_type: EventType) -> List[BaseEventHandler]:
        """
        Get all handlers for a specific event type
        
        Args:
            event_type: Type of event
            
        Returns:
            List of handlers for this event type
        """
        return self.event_types_map.get(event_type, []).copy()
    
    def get_handler_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all handlers
        
        Returns:
            Dictionary with handler statistics
        """
        stats = {}
        for name, handler in self.handlers.items():
            stats[name] = {
                'priority': handler.metadata.priority,
                'is_active': handler.metadata.is_active,
                'event_types': [et.name for et in handler.metadata.event_types],
                'execution_count': handler.metadata.execution_count,
                'avg_execution_time': handler.metadata.avg_execution_time,
                'last_execution': handler.metadata.last_execution.isoformat() if handler.metadata.last_execution else None,
                **handler.metadata.stats
            }
        return stats
    
    def __str__(self):
        return f"EventHandlerManager (running: {self._is_running}, handlers: {len(self.handlers)})"


# Concrete handler implementations

class MarketDataHandler(BaseEventHandler):
    """Handler for market data events"""
    
    def __init__(self, name: str = "MarketDataHandler", priority: int = 100):
        super().__init__(name, [EventType.MARKET_DATA, EventType.TICK, EventType.PRICE_CHANGE], priority)
        self.processors = []
    
    def _initialize(self):
        """Initialize resources"""
        pass
    
    def _shutdown(self):
        """Clean up resources"""
        pass
    
    def _handle_event(self, event: Event) -> bool:
        """Process market data events"""
        # Implementation will depend on specific market data needs
        return True
    
    def add_processor(self, processor: Callable):
        """Add a data processor function"""
        self.processors.append(processor)


class IndicatorHandler(BaseEventHandler):
    """Handler for indicator events"""
    
    def __init__(self, name: str = "IndicatorHandler", priority: int = 90):
        super().__init__(name, [EventType.INDICATOR, EventType.NEW_CANDLE], priority)
        self.indicators = {}
    
    def _initialize(self):
        """Initialize resources"""
        pass
    
    def _shutdown(self):
        """Clean up resources"""
        pass
    
    def _handle_event(self, event: Event) -> bool:
        """Process indicator events"""
        # Implementation will depend on specific indicator needs
        return True
    
    def register_indicator(self, name: str, indicator_func: Callable):
        """Register a new indicator"""
        self.indicators[name] = indicator_func


class SignalHandler(BaseEventHandler):
    """Handler for trading signal events"""
    
    def __init__(self, name: str = "SignalHandler", priority: int = 80):
        super().__init__(name, [EventType.SIGNAL], priority)
        self.strategies = {}
    
    def _initialize(self):
        """Initialize resources"""
        pass
    
    def _shutdown(self):
        """Clean up resources"""
        pass
    
    def _handle_event(self, event: Event) -> bool:
        """Process signal events"""
        # Implementation will depend on specific signal handling needs
        return True
    
    def register_strategy(self, name: str, strategy):
        """Register a strategy for signal handling"""
        self.strategies[name] = strategy


class OrderHandler(BaseEventHandler):
    """Handler for order events"""
    
    def __init__(self, name: str = "OrderHandler", priority: int = 70):
        super().__init__(name, [EventType.PENDING_ORDER, EventType.ORDER_FILLED, EventType.ORDER_CANCELLED], priority)
    
    def _initialize(self):
        """Initialize resources"""
        pass
    
    def _shutdown(self):
        """Clean up resources"""
        pass
    
    def _handle_event(self, event: Event) -> bool:
        """Process order events"""
        # Implementation will depend on specific order handling needs
        return True


class PositionHandler(BaseEventHandler):
    """Handler for position events"""
    
    def __init__(self, name: str = "PositionHandler", priority: int = 60):
        super().__init__(name, [EventType.POSITION_OPENED, EventType.POSITION_MODIFIED, EventType.POSITION_CLOSED], priority)
    
    def _initialize(self):
        """Initialize resources"""
        pass
    
    def _shutdown(self):
        """Clean up resources"""
        pass
    
    def _handle_event(self, event: Event) -> bool:
        """Process position events"""
        # Implementation will depend on specific position handling needs
        return True


class NewsHandler(BaseEventHandler):
    """Handler for news events"""
    
    def __init__(self, name: str = "NewsHandler", priority: int = 50):
        super().__init__(name, [EventType.NEWS], priority)
    
    def _initialize(self):
        """Initialize resources"""
        pass
    
    def _shutdown(self):
        """Clean up resources"""
        pass
    
    def _handle_event(self, event: Event) -> bool:
        """Process news events"""
        # Implementation will depend on specific news handling needs
        return True


class SystemHandler(BaseEventHandler):
    """Handler for system events"""
    
    def __init__(self, name: str = "SystemHandler", priority: int = 100):
        super().__init__(name, [EventType.SYSTEM], priority)
    
    def _initialize(self):
        """Initialize resources"""
        pass
    
    def _shutdown(self):
        """Clean up resources"""
        pass
    
    def _handle_event(self, event: Event) -> bool:
        """Process system events"""
        # Implementation will depend on specific system needs
        return True
