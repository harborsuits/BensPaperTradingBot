#!/usr/bin/env python3
"""
Order lifecycle tracking utilities for broker adapters.

This module provides helper methods for broker adapters to emit standardized
order lifecycle events (acknowledgment, partial fills, full fills) and 
track execution metrics like slippage.
"""

from datetime import datetime
from typing import Dict, Any, Optional, Union

from trading_bot.core.events import (
    OrderAcknowledged, OrderPartialFill, OrderFilled, 
    OrderCancelled, OrderRejected, SlippageMetric
)
from trading_bot.event_system.event_bus import EventBus


class OrderLifecycleTracker:
    """Helper class for tracking order lifecycle and emitting events"""
    
    def __init__(self, broker_id: str, event_bus: EventBus):
        """
        Initialize the order lifecycle tracker
        
        Args:
            broker_id: ID of the broker
            event_bus: EventBus for emitting events
        """
        self.broker_id = broker_id
        self.event_bus = event_bus
        
        # Track filled quantities for partial fill detection
        self.last_filled_qty = {}  # order_id -> last_filled_qty
        
        # Track expected prices for slippage calculation
        self.expected_prices = {}  # order_id -> expected_price
        
        # Track if we've acknowledged orders
        self.acknowledged = set()  # set of acknowledged order_ids
    
    def emit_order_acknowledged(
        self, 
        order_id: str, 
        symbol: str, 
        side: str,
        quantity: float,
        order_type: str,
        limit_price: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Emit an OrderAcknowledged event
        
        Args:
            order_id: Broker's order ID
            symbol: Asset symbol
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            order_type: Order type ('market', 'limit', etc.)
            limit_price: Limit price if applicable
            timestamp: Event timestamp (defaults to now)
        """
        if not self.event_bus:
            return
        
        if order_id in self.acknowledged:
            return  # Don't acknowledge twice
        
        self.acknowledged.add(order_id)
        
        event = OrderAcknowledged(
            order_id=order_id,
            broker=self.broker_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            timestamp=timestamp or datetime.now()
        )
        
        self.event_bus.emit(event)
    
    def record_expected_price(self, order_id: str, expected_price: float):
        """
        Record the expected execution price for an order
        
        Args:
            order_id: Order ID
            expected_price: Expected execution price
        """
        self.expected_prices[order_id] = expected_price
    
    def emit_order_partial_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        filled_qty: float,
        remaining_qty: float,
        fill_price: float,
        timestamp: Optional[datetime] = None
    ):
        """
        Emit an OrderPartialFill event
        
        Args:
            order_id: Broker's order ID
            symbol: Asset symbol
            side: Order side ('buy' or 'sell')
            filled_qty: Quantity filled in this execution
            remaining_qty: Quantity remaining to be filled
            fill_price: Execution price
            timestamp: Event timestamp (defaults to now)
        """
        if not self.event_bus:
            return
            
        # Track filled quantity
        last_filled = self.last_filled_qty.get(order_id, 0)
        self.last_filled_qty[order_id] = filled_qty
        
        # Only emit if there's been a change
        if filled_qty <= last_filled:
            return
        
        event = OrderPartialFill(
            order_id=order_id,
            broker=self.broker_id,
            symbol=symbol,
            side=side,
            filled_qty=filled_qty,
            remaining_qty=remaining_qty,
            fill_price=fill_price,
            timestamp=timestamp or datetime.now()
        )
        
        self.event_bus.emit(event)
        
        # Calculate and emit slippage if we have an expected price
        if order_id in self.expected_prices:
            self._emit_slippage_metric(
                order_id=order_id,
                symbol=symbol,
                side=side,
                expected_price=self.expected_prices[order_id],
                fill_price=fill_price,
                fill_qty=filled_qty - last_filled,
                asset_class=self._get_asset_class(symbol)
            )
    
    def emit_order_filled(
        self,
        order_id: str,
        symbol: str,
        side: str,
        total_qty: float,
        avg_fill_price: float,
        trade_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Emit an OrderFilled event
        
        Args:
            order_id: Broker's order ID
            symbol: Asset symbol
            side: Order side ('buy' or 'sell')
            total_qty: Total quantity filled
            avg_fill_price: Average fill price
            trade_id: Optional trade ID
            timestamp: Event timestamp (defaults to now)
        """
        if not self.event_bus:
            return
        
        # Track filled quantity and clean up
        self.last_filled_qty[order_id] = total_qty
        
        event = OrderFilled(
            order_id=order_id,
            broker=self.broker_id,
            symbol=symbol,
            side=side,
            total_qty=total_qty,
            avg_fill_price=avg_fill_price,
            trade_id=trade_id,
            timestamp=timestamp or datetime.now()
        )
        
        self.event_bus.emit(event)
        
        # Calculate and emit final slippage if we have an expected price
        if order_id in self.expected_prices:
            self._emit_slippage_metric(
                order_id=order_id,
                symbol=symbol,
                side=side,
                expected_price=self.expected_prices[order_id],
                fill_price=avg_fill_price,
                fill_qty=total_qty,
                asset_class=self._get_asset_class(symbol),
                is_final=True,
                trade_id=trade_id
            )
            
            # Clean up expected price
            del self.expected_prices[order_id]
    
    def emit_order_cancelled(
        self,
        order_id: str,
        reason: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Emit an OrderCancelled event
        
        Args:
            order_id: Broker's order ID
            reason: Optional cancellation reason
            timestamp: Event timestamp (defaults to now)
        """
        if not self.event_bus:
            return
        
        # Clean up tracking
        if order_id in self.last_filled_qty:
            del self.last_filled_qty[order_id]
        if order_id in self.expected_prices:
            del self.expected_prices[order_id]
        if order_id in self.acknowledged:
            self.acknowledged.remove(order_id)
        
        event = OrderCancelled(
            order_id=order_id,
            broker=self.broker_id,
            reason=reason,
            timestamp=timestamp or datetime.now()
        )
        
        self.event_bus.emit(event)
    
    def emit_order_rejected(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        reason: str,
        order_id: Optional[str] = None,
        limit_price: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Emit an OrderRejected event
        
        Args:
            symbol: Asset symbol
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            order_type: Order type ('market', 'limit', etc.)
            reason: Rejection reason
            order_id: Optional broker order ID (if assigned)
            limit_price: Optional limit price
            timestamp: Event timestamp (defaults to now)
        """
        if not self.event_bus:
            return
        
        # Clean up tracking if order_id exists
        if order_id:
            if order_id in self.last_filled_qty:
                del self.last_filled_qty[order_id]
            if order_id in self.expected_prices:
                del self.expected_prices[order_id]
            if order_id in self.acknowledged:
                self.acknowledged.remove(order_id)
        
        event = OrderRejected(
            order_id=order_id,
            broker=self.broker_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            reason=reason,
            limit_price=limit_price,
            timestamp=timestamp or datetime.now()
        )
        
        self.event_bus.emit(event)
    
    def _emit_slippage_metric(
        self,
        order_id: str,
        symbol: str,
        side: str,
        expected_price: float,
        fill_price: float,
        fill_qty: float,
        asset_class: str,
        is_final: bool = False,
        trade_id: Optional[str] = None
    ):
        """
        Calculate and emit slippage metrics
        
        Args:
            order_id: Order ID
            symbol: Asset symbol
            side: Order side ('buy' or 'sell')
            expected_price: Expected execution price
            fill_price: Actual fill price
            fill_qty: Fill quantity
            asset_class: Asset class ('equity', 'option', 'forex', etc.)
            is_final: Whether this is the final fill
            trade_id: Optional trade ID
        """
        if not self.event_bus:
            return
        
        # Calculate slippage (positive means worse execution price)
        if side.lower() == 'buy':
            slippage_amount = fill_price - expected_price
        else:  # sell
            slippage_amount = expected_price - fill_price
        
        # Calculate slippage in basis points (1bp = 0.01%)
        if expected_price > 0:
            slippage_bps = (slippage_amount / expected_price) * 10000
        else:
            slippage_bps = 0
        
        event = SlippageMetric(
            broker=self.broker_id,
            symbol=symbol,
            asset_class=asset_class,
            side=side,
            expected_price=expected_price,
            fill_price=fill_price,
            slippage_amount=slippage_amount,
            slippage_bps=slippage_bps,
            order_id=order_id,
            trade_id=trade_id,
            timestamp=datetime.now()
        )
        
        self.event_bus.emit(event)
    
    def _get_asset_class(self, symbol: str) -> str:
        """
        Determine asset class from symbol
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Asset class string
        """
        # Simple determination based on symbol format
        # This could be enhanced with a more sophisticated lookup
        if '.' in symbol:  # e.g. EUR.USD
            return 'forex'
        elif '/' in symbol:  # e.g. BTC/USD
            return 'crypto'
        elif symbol.endswith('C') or symbol.endswith('P'):  # Simple option check
            return 'option'
        else:
            return 'equity'
    
    def cleanup_order(self, order_id: str):
        """
        Clean up tracking for an order
        
        Args:
            order_id: Order ID to clean up
        """
        if order_id in self.last_filled_qty:
            del self.last_filled_qty[order_id]
        if order_id in self.expected_prices:
            del self.expected_prices[order_id]
        if order_id in self.acknowledged:
            self.acknowledged.remove(order_id)
