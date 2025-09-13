#!/usr/bin/env python3
"""
Advanced Order Types for BensBot

This module defines complex order types including:
- Combo/multi-leg orders
- Iceberg orders for splitting large orders
- VWAP/TWAP orders for time-weighted execution

These advanced orders integrate with BensBot's execution and position tracking system.
"""

from typing import List, Dict, Optional, Literal, Any, Union
from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel, Field


class ComboLeg(BaseModel):
    """
    Single leg of a combo/multi-leg order
    """
    symbol: str
    quantity: float
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit", "stop", "stop_limit"]
    price: Optional[float] = None
    stop_price: Optional[float] = None
    broker_id: Optional[str] = None  # Specific broker override for this leg
    leg_id: str = Field(default_factory=lambda: str(uuid4()))
    asset_class: Optional[str] = None  # Equity, option, futures, etc.


class ComboOrder(BaseModel):
    """
    Combo/multi-leg order for executing multiple related legs
    """
    combo_id: str = Field(default_factory=lambda: str(uuid4()))
    legs: List[ComboLeg]
    order_strategy: Literal["spread", "straddle", "iron_condor", "butterfly", "custom"]
    routing_instructions: Dict[str, Any] = Field(default_factory=dict)  # e.g. {"equities": "tradier", "options": "alpaca"}
    time_in_force: Literal["day", "gtc", "ioc", "fok"] = "day"
    user_metadata: Dict[str, Any] = Field(default_factory=dict)  # e.g. strategy name, tags
    risk_limits: Dict[str, Any] = Field(default_factory=dict)  # e.g. max slippage, max delay between legs
    expected_prices: Dict[str, float] = Field(default_factory=dict)  # Expected prices for slippage calculation


class IcebergOrder(BaseModel):
    """
    Iceberg order for splitting large orders into smaller chunks
    """
    order_id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    total_quantity: float
    display_quantity: float  # Visible portion per child order
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit", "stop", "stop_limit"]
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: Literal["day", "gtc", "ioc", "fok"] = "day"
    child_delay_ms: int  # Wait between child orders
    max_retries: int = 3  # Maximum retry attempts per child order
    broker_id: Optional[str] = None  # Specific broker override
    user_metadata: Dict[str, Any] = Field(default_factory=dict)
    expected_price: Optional[float] = None  # Expected execution price


class TimeWeightedOrder(BaseModel):
    """
    Base class for time-weighted execution orders (VWAP/TWAP)
    """
    order_id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    total_quantity: float
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit"]
    price: Optional[float] = None
    time_in_force: Literal["day", "ioc"] = "day"
    start_time: datetime
    end_time: datetime
    broker_id: Optional[str] = None
    user_metadata: Dict[str, Any] = Field(default_factory=dict)
    max_participation_rate: float = 0.3  # Maximum percentage of market volume


class TWAPOrder(TimeWeightedOrder):
    """
    Time-Weighted Average Price order
    Splits execution evenly across time intervals
    """
    interval_minutes: int = 5  # Time interval between child orders


class VWAPOrder(TimeWeightedOrder):
    """
    Volume-Weighted Average Price order
    Weights execution based on historical volume profile
    """
    volume_profile: Literal["historical", "custom", "auto"] = "historical"
    custom_profile: Optional[Dict[str, float]] = None  # Custom distribution if specified
    lookback_days: int = 10  # Days of historical data to use for volume profile
    deviation_tolerance: float = 0.1  # Allowed deviation from target schedule


# Events related to advanced orders
class ComboOrderEvent(BaseModel):
    """Base class for combo order events"""
    combo_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IcebergOrderEvent(BaseModel):
    """Base class for iceberg order events"""
    order_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TimeWeightedOrderEvent(BaseModel):
    """Base class for time-weighted order events"""
    order_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Individual event types
class ComboOrderPlaced(ComboOrderEvent):
    """Event when a combo order is placed"""
    legs: List[ComboLeg]
    order_strategy: str


class ComboLegFilled(ComboOrderEvent):
    """Event when a combo leg is filled"""
    leg_id: str
    symbol: str
    quantity: float
    price: float
    side: str


class ComboOrderCompleted(ComboOrderEvent):
    """Event when all legs of a combo order are filled"""
    fill_details: Dict[str, Any]
    execution_time_ms: int


class ComboOrderFailed(ComboOrderEvent):
    """Event when a combo order fails"""
    reason: str
    failed_legs: List[str]
    partial_fills: Dict[str, Any] = Field(default_factory=dict)


class IcebergOrderStarted(IcebergOrderEvent):
    """Event when an iceberg order starts execution"""
    symbol: str
    total_quantity: float
    display_quantity: float
    side: str


class IcebergChunkPlaced(IcebergOrderEvent):
    """Event when an iceberg chunk is placed"""
    chunk_index: int
    quantity: float
    remaining_quantity: float
    chunk_order_id: str


class IcebergChunkFilled(IcebergOrderEvent):
    """Event when an iceberg chunk is filled"""
    chunk_index: int
    quantity: float
    price: float
    remaining_quantity: float


class IcebergOrderCompleted(IcebergOrderEvent):
    """Event when an iceberg order is completely filled"""
    total_filled: float
    avg_price: float
    execution_time_ms: int
    chunks_count: int


class IcebergOrderFailed(IcebergOrderEvent):
    """Event when an iceberg order fails"""
    reason: str
    filled_quantity: float
    remaining_quantity: float


class TimeWeightedOrderStarted(TimeWeightedOrderEvent):
    """Event when a time-weighted order starts execution"""
    symbol: str
    total_quantity: float
    side: str
    schedule_type: str  # "TWAP" or "VWAP"


class TimeWeightedChunkPlaced(TimeWeightedOrderEvent):
    """Event when a time-weighted chunk is placed"""
    chunk_index: int
    timestamp: datetime
    target_quantity: float
    actual_quantity: float
    remaining_quantity: float


class TimeWeightedOrderProgress(TimeWeightedOrderEvent):
    """Event for time-weighted order progress updates"""
    percent_complete: float
    quantity_executed: float
    remaining_quantity: float
    current_deviation: float  # Deviation from ideal schedule


class TimeWeightedOrderCompleted(TimeWeightedOrderEvent):
    """Event when a time-weighted order is completely filled"""
    total_filled: float
    avg_price: float
    execution_time_ms: int
    market_impact: float
    implementation_shortfall: float


class TimeWeightedOrderFailed(TimeWeightedOrderEvent):
    """Event when a time-weighted order fails"""
    reason: str
    filled_quantity: float
    remaining_quantity: float
