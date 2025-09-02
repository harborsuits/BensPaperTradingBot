#!/usr/bin/env python3
"""
Event definitions for the trading bot system.

This module defines all event types used in the trading system's event bus.
Events provide a decoupled way for components to communicate.
"""

from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum

# Re-export EventBus from event_bus.py to maintain backward compatibility
from trading_bot.core.event_bus import EventBus, Event as EventObj, get_global_event_bus


class EventType(Enum):
    """Enumeration of event types in the system"""
    STRATEGY_SIGNAL = "strategy_signal"
    MARKET_DATA = "market_data"
    ORDER_UPDATE = "order_update"
    POSITION_UPDATE = "position_update"
    ACCOUNT_UPDATE = "account_update"
    BROKER_STATUS = "broker_status"
    BROKER_METRIC = "broker_metric"
    BROKER_INTELLIGENCE = "broker_intelligence"
    REGIME_UPDATE = "regime_update"
    RISK_UPDATE = "risk_update"
    SYSTEM_STATUS = "system_status"
    
    # Risk management events
    MARGIN_CALL = "margin_call"
    MARGIN_WARNING = "margin_warning"
    TRADING_PAUSED = "trading_paused"
    TRADING_RESUMED = "trading_resumed"
    FORCED_EXIT = "forced_exit"
    PORTFOLIO_EQUITY_UPDATE = "portfolio_equity_update"
    VOLATILITY_UPDATE = "volatility_update"
    
    # Order lifecycle events
    ORDER_ACKNOWLEDGED = "order_acknowledged"
    ORDER_PARTIAL_FILL = "order_partial_fill"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    
    # Execution intelligence events
    SLIPPAGE_METRIC = "slippage_metric"


class Event:
    """Base class for all events (not a dataclass)"""
    
    def __init__(self):
        self.timestamp = datetime.now()


@dataclass
class OrderAcknowledged:
    """Event emitted when an order is acknowledged by the broker"""
    order_id: str
    broker: str
    symbol: str
    side: str
    quantity: float
    order_type: str
    limit_price: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class OrderPartialFill:
    """Event emitted when an order is partially filled"""
    order_id: str
    broker: str
    symbol: str
    side: str
    filled_qty: float
    remaining_qty: float
    fill_price: float
    timestamp: datetime = None
    type: EventType = EventType.ORDER_PARTIAL_FILL
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class OrderFilled:
    """Event emitted when an order is completely filled"""
    order_id: str
    broker: str
    symbol: str
    side: str
    total_qty: float
    avg_fill_price: float
    trade_id: Optional[str] = None
    timestamp: datetime = None
    type: EventType = EventType.ORDER_FILLED
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class OrderCancelled:
    """Event emitted when an order is cancelled"""
    order_id: str
    broker: str
    reason: Optional[str] = None
    timestamp: datetime = None
    type: EventType = EventType.ORDER_CANCELLED
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class OrderRejected:
    """Event emitted when an order is rejected by the broker"""
    broker: str
    symbol: str
    side: str
    quantity: float
    order_type: str
    reason: str
    order_id: Optional[str] = None  # May not have an ID if rejected immediately
    limit_price: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        self.type = EventType.ORDER_REJECTED
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class SlippageMetric:
    """Event emitted to record slippage metrics"""
    broker: str
    symbol: str
    asset_class: str
    side: str
    expected_price: float
    fill_price: float
    slippage_amount: float  # Positive means worse execution than expected
    slippage_bps: float     # Slippage in basis points
    order_id: str
    trade_id: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        self.type = EventType.SLIPPAGE_METRIC
        if self.timestamp is None:
            self.timestamp = datetime.now()


# Other existing event classes would remain below
@dataclass
class StrategySignal(Event):
    """Event emitted when a strategy generates a trading signal"""
    strategy_id: str
    signal_type: str
    symbol: str
    direction: str
    price: float
    confidence: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.type = EventType.STRATEGY_SIGNAL


@dataclass
class MarketDataEvent(Event):
    """Event containing market data updates"""
    symbol: str
    data_type: str
    price: float
    volume: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.type = EventType.MARKET_DATA


@dataclass
class OrderUpdate(Event):
    """Event for order status updates"""
    order_id: str
    broker_id: str
    status: str
    filled_quantity: float
    remaining_quantity: float
    avg_fill_price: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.type = EventType.ORDER_UPDATE


@dataclass
class PositionUpdate(Event):
    """Event for position updates"""
    symbol: str
    quantity: float
    avg_price: float
    broker_id: str
    unrealized_pnl: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.type = EventType.POSITION_UPDATE


@dataclass
class AccountUpdate(Event):
    """Event for account updates"""
    broker_id: str
    balance: float
    equity: float
    margin_used: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.type = EventType.ACCOUNT_UPDATE


@dataclass
class BrokerStatus(Event):
    """Event for broker status updates"""
    broker_id: str
    status: str  # "connected", "disconnected", "error"
    message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.type = EventType.BROKER_STATUS


@dataclass
class BrokerMetric(Event):
    """Event for broker performance metrics"""
    broker_id: str
    metric_type: str  # "latency", "error_rate", "fill_rate", etc.
    value: float
    unit: str
    operation: Optional[str] = None  # "order", "quote", "data", etc.
    asset_class: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.type = EventType.BROKER_METRIC


@dataclass
class BrokerIntelligence(Event):
    """Event for broker intelligence updates"""
    broker_id: str
    score: float
    alert_type: Optional[str] = None
    message: Optional[str] = None
    recommended_action: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.type = EventType.BROKER_INTELLIGENCE


@dataclass
class RegimeUpdate(Event):
    """Event for market regime updates"""
    regime_type: str
    confidence: float
    affected_assets: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.type = EventType.REGIME_UPDATE


@dataclass
class RiskUpdate(Event):
    """Event for risk metric updates"""
    risk_type: str
    level: float
    affected_components: List[str]
    message: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.type = EventType.RISK_UPDATE


@dataclass
class SystemStatus(Event):
    """Event for system status updates"""
    component: str
    status: str
    message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.type = EventType.SYSTEM_STATUS


# Risk Management Events

@dataclass
class MarginCallWarning(Event):
    """Event emitted when a margin warning threshold is reached"""
    broker: str
    ratio: float  # Current margin usage ratio
    threshold: float  # Warning threshold that was breached
    account_id: str
    remaining_buffer: float  # Percentage buffer remaining before actual margin call
    
    def __post_init__(self):
        super().__post_init__()
        self.type = EventType.MARGIN_WARNING


@dataclass
class MarginCall(Event):
    """Event emitted when a margin call is triggered"""
    broker: str
    ratio: float  # Current margin usage ratio
    threshold: float  # Call threshold that was breached
    account_id: str
    severity: str = "high"  # low, medium, high, critical
    
    def __post_init__(self):
        super().__post_init__()
        self.type = EventType.MARGIN_CALL


@dataclass
class TradingPaused(Event):
    """Event emitted when trading is paused due to risk issues"""
    reason: str
    component: Optional[str] = None  # Component that triggered the pause
    threshold_breached: Optional[float] = None
    current_value: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.type = EventType.TRADING_PAUSED


@dataclass
class TradingResumed(Event):
    """Event emitted when trading is resumed after being paused"""
    component: Optional[str] = None  # Component that resumed trading
    resume_policy: Optional[str] = None  # Auto or manual resume
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.type = EventType.TRADING_RESUMED


@dataclass
class ForcedExitOrder(Event):
    """Event emitted when a position needs to be forcibly exited due to risk"""
    symbol: str
    qty: float
    reason: str
    broker: Optional[str] = None
    order_type: str = "market"  # Default to market order for forced exits
    max_slippage_bps: Optional[int] = None  # Maximum acceptable slippage in basis points
    
    def __post_init__(self):
        super().__post_init__()
        self.type = EventType.FORCED_EXIT


@dataclass
class PortfolioEquityUpdate(Event):
    """Event for tracking portfolio equity for circuit breakers"""
    equity: float
    peak_equity: float
    drawdown: float  # Current drawdown percentage
    broker_id: Optional[str] = None  # None means aggregated across all brokers
    currency: str = "USD"
    
    def __post_init__(self):
        super().__post_init__()
        self.type = EventType.PORTFOLIO_EQUITY_UPDATE


@dataclass
class VolatilityUpdate:
    """Event for tracking realized volatility for circuit breakers"""
    timeframe: str  # e.g., "1H", "1D"
    realized_vol: float  # Annualized realized volatility
    symbol: Optional[str] = None  # None means portfolio-level volatility
    vol_z_score: Optional[float] = None  # Standard deviations from baseline
    lookback_periods: int = 21  # Number of periods used in calculation
    timestamp: datetime = None
    
    def __post_init__(self):
        self.type = EventType.VOLATILITY_UPDATE
        if self.timestamp is None:
            self.timestamp = datetime.now()
