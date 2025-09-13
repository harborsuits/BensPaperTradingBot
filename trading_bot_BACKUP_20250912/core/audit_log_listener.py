"""
Audit Log Listener

Connects the event bus to the audit log system, automatically logging
all relevant trading events for compliance, analysis, and debugging.
"""

import logging
from typing import Dict, Any, Optional

from trading_bot.core.event_bus import Event, EventBus, get_global_event_bus
from trading_bot.core.constants import EventType
from trading_bot.brokers.trade_audit_log import TradeAuditLog, AuditEventType

# Configure logging
logger = logging.getLogger(__name__)


class AuditLogListener:
    """
    Listens to the event bus and logs relevant events to the audit log.
    
    This class creates a bridge between the event-driven architecture
    and the persistent audit logging system.
    """
    
    def __init__(self, audit_log: TradeAuditLog, event_bus: Optional[EventBus] = None):
        """
        Initialize the audit log listener.
        
        Args:
            audit_log: Audit log instance to record events
            event_bus: Event bus to listen to (uses global if None)
        """
        self.audit_log = audit_log
        self.event_bus = event_bus or get_global_event_bus()
        self._event_mapping = self._create_event_mapping()
        self._registered = False
        logger.info("Audit log listener initialized")
    
    def _create_event_mapping(self) -> Dict[EventType, AuditEventType]:
        """Create mapping between event bus events and audit log events."""
        return {
            # Order events
            EventType.ORDER_CREATED: AuditEventType.ORDER_SUBMITTED,
            EventType.ORDER_SUBMITTED: AuditEventType.ORDER_SUBMITTED,
            EventType.ORDER_FILLED: AuditEventType.ORDER_FILLED,
            EventType.ORDER_CANCELLED: AuditEventType.ORDER_CANCELLED,
            EventType.ORDER_REJECTED: AuditEventType.ORDER_REJECTED,
            
            # Trade events
            EventType.TRADE_EXECUTED: AuditEventType.ORDER_FILLED,
            EventType.TRADE_CLOSED: AuditEventType.POSITION_CLOSED,
            
            # Strategy events
            EventType.STRATEGY_STARTED: AuditEventType.SYSTEM_ERROR,  # Using SYSTEM_ERROR as generic for system events
            EventType.STRATEGY_STOPPED: AuditEventType.SYSTEM_ERROR,
            EventType.SIGNAL_GENERATED: AuditEventType.STRATEGY_SIGNAL,
            
            # System events
            EventType.SYSTEM_STARTED: AuditEventType.SYSTEM_ERROR,
            EventType.SYSTEM_STOPPED: AuditEventType.SYSTEM_ERROR,
            EventType.ERROR_OCCURRED: AuditEventType.SYSTEM_ERROR,
            
            # Risk events
            EventType.RISK_LIMIT_REACHED: AuditEventType.RISK_LIMIT_BREACH,
            EventType.DRAWDOWN_ALERT: AuditEventType.RISK_LIMIT_BREACH,
            EventType.POSITION_SIZE_CALCULATED: AuditEventType.POSITION_UPDATED,
            
            # Broker events
            EventType.HEALTH_STATUS_CHANGED: AuditEventType.BROKER_OPERATION,
            
            # Mode events
            EventType.MODE_CHANGED: AuditEventType.CONFIG_CHANGE
        }
    
    def register(self) -> None:
        """Register event handlers with the event bus."""
        if self._registered:
            logger.warning("Audit log listener already registered")
            return
            
        # Subscribe to specific event types
        for event_type in self._event_mapping.keys():
            self.event_bus.subscribe(event_type, self._handle_event)
        
        # Also subscribe to all events for comprehensive audit if needed
        # self.event_bus.subscribe_all(self._handle_all_events)
        
        self._registered = True
        logger.info("Audit log listener registered with event bus")
    
    def unregister(self) -> None:
        """Unregister event handlers from the event bus."""
        if not self._registered:
            return
            
        # Unsubscribe from specific event types
        for event_type in self._event_mapping.keys():
            self.event_bus.unsubscribe(event_type, self._handle_event)
        
        # Also unsubscribe from all events
        # self.event_bus.unsubscribe_all(self._handle_all_events)
        
        self._registered = False
        logger.info("Audit log listener unregistered from event bus")
    
    def _handle_event(self, event: Event) -> None:
        """
        Handle a specific event by logging it to the audit log.
        
        Args:
            event: Event from the event bus
        """
        if event.event_type not in self._event_mapping:
            return
            
        audit_event_type = self._event_mapping[event.event_type]
        
        # Extract event details
        details = event.data.copy()
        
        # Add source and timestamp info
        details["event_source"] = event.source
        details["event_bus_timestamp"] = event.timestamp.isoformat()
        
        # Extract broker_id and order_id if present
        broker_id = details.pop("broker_id", None) or details.get("broker", None)
        order_id = details.pop("order_id", None) or details.get("id", None)
        strategy_id = details.pop("strategy_id", None) or details.get("strategy", None)
        
        try:
            # Log to audit trail
            self.audit_log.log_event(
                audit_event_type,
                details,
                broker_id=broker_id,
                order_id=order_id,
                strategy_id=strategy_id
            )
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Logged {event.event_type} to audit log as {audit_event_type}")
        except Exception as e:
            logger.error(f"Error logging event to audit log: {str(e)}")
    
    def _handle_all_events(self, event: Event) -> None:
        """
        Handle all events (optional comprehensive logging).
        
        Args:
            event: Event from the event bus
        """
        # Only log events that aren't already covered by specific handlers
        if event.event_type in self._event_mapping:
            return
            
        # For now, just log the event type and basic info
        details = {
            "event_type": event.event_type,
            "source": event.source,
            "timestamp": event.timestamp.isoformat(),
            "data_keys": list(event.data.keys())
        }
        
        try:
            # Log to audit trail as a system event
            self.audit_log.log_event(
                AuditEventType.SYSTEM_ERROR,  # Using as a generic for other events
                details
            )
        except Exception as e:
            logger.error(f"Error logging event to audit log: {str(e)}")


def create_audit_log_listener(audit_log: TradeAuditLog) -> AuditLogListener:
    """
    Create and register an audit log listener.
    
    Args:
        audit_log: Audit log instance to record events
        
    Returns:
        Registered audit log listener
    """
    listener = AuditLogListener(audit_log)
    listener.register()
    return listener
