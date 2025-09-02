#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event Types - Defines all event types and structures used in the event-driven
architecture of the trading system.

Based on best practices from OctoBot and EA31337.
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import uuid
import json

logger = logging.getLogger("EventSystem")

class EventType(Enum):
    """Types of events that can flow through the system"""
    
    # System Events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    SYSTEM_ERROR = "system_error"
    CONFIG_UPDATE = "config_update"
    
    # Market Events
    MARKET_DATA = "market_data"
    CANDLE_CLOSE = "candle_close"
    PRICE_UPDATE = "price_update"
    VOLUME_SPIKE = "volume_spike"
    TICK_DATA = "tick_data"
    
    # Analysis Events
    TECHNICAL_INDICATOR = "technical_indicator"
    PATTERN_DETECTED = "pattern_detected"
    REGIME_CHANGE = "regime_change"
    ML_PREDICTION = "ml_prediction"
    LLM_ANALYSIS = "llm_analysis"
    SENTIMENT_UPDATE = "sentiment_update"
    
    # Trading Events
    SIGNAL_GENERATED = "signal_generated"
    ORDER_CREATED = "order_created"
    ORDER_UPDATED = "order_updated"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELED = "order_canceled"
    TRADE_CLOSED = "trade_closed"
    POSITION_UPDATED = "position_updated"
    
    # Risk Events
    RISK_LIMIT_BREACHED = "risk_limit_breached"
    RISK_LEVEL_CHANGED = "risk_level_changed"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    TAKE_PROFIT_TRIGGERED = "take_profit_triggered"
    
    # Account Events
    BALANCE_UPDATE = "balance_update"
    MARGIN_CALL = "margin_call"
    
    # Notification Events
    NOTIFICATION = "notification"
    ALERT_TRIGGERED = "alert_triggered"
    
    # Scheduled Events
    SCHEDULED_TASK = "scheduled_task"
    DAILY_SUMMARY = "daily_summary"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    
    # Memory Events
    MEMORY_CREATED = "memory_created"
    MEMORY_UPDATED = "memory_updated"
    MEMORY_CONSOLIDATED = "memory_consolidated"
    
    # Approval Workflow Events
    APPROVAL_REQUEST_CREATED = "approval_request_created"
    APPROVAL_REQUEST_APPROVED = "approval_request_approved"
    APPROVAL_REQUEST_REJECTED = "approval_request_rejected"
    
    # Broker Intelligence Events
    BROKER_INTELLIGENCE_UPDATE = "broker_intelligence_update"
    BROKER_CIRCUIT_BREAKER = "broker_circuit_breaker"
    BROKER_HEALTH_STATUS_CHANGE = "broker_health_status_change"
    BROKER_ORDER_PLACED = "broker_order_placed"
    BROKER_ORDER_FILLED = "broker_order_filled"
    BROKER_ORDER_FAILED = "broker_order_failed"
    BROKER_CONNECTION_ERROR = "broker_connection_error"
    BROKER_QUOTE_RECEIVED = "broker_quote_received"
    BROKER_DATA_RECEIVED = "broker_data_received"
    
    # Orchestrator Advisory Events
    ORCHESTRATOR_ADVISORY_UPDATE = "orchestrator_advisory_update"
    ORCHESTRATOR_ADVISORY_ALERT = "orchestrator_advisory_alert"


class Event:
    """
    Base event class that contains common properties for all events
    flowing through the system.
    """
    
    def __init__(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        source: str,
        timestamp: Optional[datetime] = None,
        event_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new event
        
        Args:
            event_type: Type of event
            data: Event data payload
            source: Component that generated the event
            timestamp: When the event occurred (defaults to now)
            event_id: Unique identifier (auto-generated if not provided)
            metadata: Additional metadata about the event
        """
        self.event_type = event_type
        self.data = data
        self.source = source
        self.timestamp = timestamp or datetime.now()
        self.event_id = event_id or str(uuid.uuid4())
        self.metadata = metadata or {}
        
        # Additional fields for event routing and processing
        self.processed_by = []
        self.priority = self.metadata.get("priority", 1)  # Higher is more important
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "data": self.data,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "processed_by": self.processed_by,
            "priority": self.priority
        }
    
    @classmethod
    def from_dict(cls, event_dict: Dict[str, Any]) -> 'Event':
        """Create event from dictionary representation"""
        event = cls(
            event_type=EventType(event_dict["event_type"]),
            data=event_dict["data"],
            source=event_dict["source"],
            timestamp=datetime.fromisoformat(event_dict["timestamp"]),
            event_id=event_dict["event_id"],
            metadata=event_dict.get("metadata", {})
        )
        
        # Set additional fields
        if "processed_by" in event_dict:
            event.processed_by = event_dict["processed_by"]
        if "priority" in event_dict:
            event.priority = event_dict["priority"]
        if "created_at" in event_dict:
            event.created_at = datetime.fromisoformat(event_dict["created_at"])
        
        return event
    
    def __str__(self) -> str:
        """String representation of the event"""
        return (f"Event(type={self.event_type.value}, id={self.event_id}, "
                f"source={self.source}, priority={self.priority})")


class MarketDataEvent(Event):
    """Event containing market data updates (candles, ticks, etc.)"""
    
    def __init__(
        self,
        symbol: str,
        data: Dict[str, Any],
        source: str,
        timestamp: Optional[datetime] = None,
        event_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize market data event
        
        Args:
            symbol: Market symbol
            data: Market data
            source: Data source
            timestamp: When the data was generated
            event_id: Unique identifier
            metadata: Additional metadata
        """
        # Add symbol to data
        event_data = {"symbol": symbol, **data}
        
        super().__init__(
            event_type=EventType.MARKET_DATA,
            data=event_data,
            source=source,
            timestamp=timestamp,
            event_id=event_id,
            metadata=metadata
        )
        
        # Set symbol as explicit property for easier access
        self.symbol = symbol


class SignalEvent(Event):
    """Event containing trading signals from strategies"""
    
    def __init__(
        self,
        symbol: str,
        signal_type: str,
        direction: int,
        strength: float,
        strategy: str,
        source: str,
        timestamp: Optional[datetime] = None,
        event_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize signal event
        
        Args:
            symbol: Market symbol
            signal_type: Type of signal (entry, exit, etc.)
            direction: Signal direction (1=long, -1=short, 0=neutral)
            strength: Signal strength (0 to 1)
            strategy: Strategy that generated the signal
            source: Component that generated the event
            timestamp: When the signal was generated
            event_id: Unique identifier
            metadata: Additional metadata
        """
        event_data = {
            "symbol": symbol,
            "signal_type": signal_type,
            "direction": direction,
            "strength": strength,
            "strategy": strategy
        }
        
        super().__init__(
            event_type=EventType.SIGNAL_GENERATED,
            data=event_data,
            source=source,
            timestamp=timestamp,
            event_id=event_id,
            metadata=metadata
        )


class OrderEvent(Event):
    """Event containing order information (created, updated, filled, etc.)"""
    
    def __init__(
        self,
        order_id: str,
        symbol: str,
        order_type: str,
        side: int,
        quantity: float,
        price: Optional[float],
        status: str,
        source: str,
        timestamp: Optional[datetime] = None,
        event_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize order event
        
        Args:
            order_id: Order identifier
            symbol: Market symbol
            order_type: Type of order (market, limit, etc.)
            side: Order side (1=buy, -1=sell)
            quantity: Order quantity
            price: Order price
            status: Order status
            source: Component that generated the event
            timestamp: When the order event occurred
            event_id: Unique identifier
            metadata: Additional metadata
        """
        event_data = {
            "order_id": order_id,
            "symbol": symbol,
            "order_type": order_type,
            "side": side,
            "quantity": quantity,
            "price": price,
            "status": status
        }
        
        # Determine event type based on status
        if status == "created":
            event_type = EventType.ORDER_CREATED
        elif status == "filled":
            event_type = EventType.ORDER_FILLED
        elif status == "canceled":
            event_type = EventType.ORDER_CANCELED
        else:
            event_type = EventType.ORDER_UPDATED
        
        super().__init__(
            event_type=event_type,
            data=event_data,
            source=source,
            timestamp=timestamp,
            event_id=event_id,
            metadata=metadata
        )


class RiskEvent(Event):
    """Event containing risk management information"""
    
    def __init__(
        self,
        risk_type: str,
        level: Union[int, str],
        details: Dict[str, Any],
        source: str,
        timestamp: Optional[datetime] = None,
        event_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize risk event
        
        Args:
            risk_type: Type of risk event
            level: Risk level (1-5 or descriptive string)
            details: Risk details
            source: Component that generated the event
            timestamp: When the risk event occurred
            event_id: Unique identifier
            metadata: Additional metadata
        """
        event_data = {
            "risk_type": risk_type,
            "level": level,
            "details": details
        }
        
        # Determine event type based on risk type
        if risk_type == "limit_breach":
            event_type = EventType.RISK_LIMIT_BREACHED
        elif risk_type == "level_change":
            event_type = EventType.RISK_LEVEL_CHANGED
        elif risk_type == "stop_loss":
            event_type = EventType.STOP_LOSS_TRIGGERED
        elif risk_type == "take_profit":
            event_type = EventType.TAKE_PROFIT_TRIGGERED
        else:
            event_type = EventType.RISK_LIMIT_BREACHED
        
        super().__init__(
            event_type=event_type,
            data=event_data,
            source=source,
            timestamp=timestamp,
            event_id=event_id,
            metadata=metadata
        )


class AnalysisEvent(Event):
    """Event containing analysis results (technical indicators, ML predictions, etc.)"""
    
    def __init__(
        self,
        analysis_type: str,
        symbol: str,
        results: Dict[str, Any],
        confidence: float,
        source: str,
        timestamp: Optional[datetime] = None,
        event_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize analysis event
        
        Args:
            analysis_type: Type of analysis
            symbol: Market symbol
            results: Analysis results
            confidence: Confidence score (0 to 1)
            source: Component that generated the event
            timestamp: When the analysis was performed
            event_id: Unique identifier
            metadata: Additional metadata
        """
        event_data = {
            "analysis_type": analysis_type,
            "symbol": symbol,
            "results": results,
            "confidence": confidence
        }
        
        # Determine event type based on analysis type
        if analysis_type == "technical_indicator":
            event_type = EventType.TECHNICAL_INDICATOR
        elif analysis_type == "pattern":
            event_type = EventType.PATTERN_DETECTED
        elif analysis_type == "regime":
            event_type = EventType.REGIME_CHANGE
        elif analysis_type == "ml_prediction":
            event_type = EventType.ML_PREDICTION
        elif analysis_type == "llm_analysis":
            event_type = EventType.LLM_ANALYSIS
        elif analysis_type == "sentiment":
            event_type = EventType.SENTIMENT_UPDATE
        else:
            event_type = EventType.TECHNICAL_INDICATOR
        
        super().__init__(
            event_type=event_type,
            data=event_data,
            source=source,
            timestamp=timestamp,
            event_id=event_id,
            metadata=metadata
        )
