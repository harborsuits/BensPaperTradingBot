#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper Broker Implementation

This module implements a paper trading broker that simulates order execution
without sending real orders to any exchange. It uses real market data to
drive simulated fills while maintaining a separate portfolio and order book.
"""

import logging
import uuid
import json
from datetime import datetime, time, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from decimal import Decimal, ROUND_DOWN
import threading
import time as time_mod

from trading_bot.brokers.broker_interface import BrokerInterface
from trading_bot.core.constants import OrderStatus, OrderType, OrderSide, TimeInForce
from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType

logger = logging.getLogger(__name__)


class PaperOrder:
    """Represents a paper trading order."""
    
    def __init__(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
        submitted_at: Optional[datetime] = None,
        strategy_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """Initialize a paper order."""
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.filled_quantity = 0.0
        self.order_type = order_type
        self.price = price
        self.stop_price = stop_price
        self.time_in_force = time_in_force
        self.status = "open"
        self.submitted_at = submitted_at or datetime.now()
        self.filled_at = None
        self.canceled_at = None
        self.strategy_id = strategy_id
        self.tags = tags or ["PAPER"]
        self.fill_price = None
        self.average_price = None
        self.fees = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "order_type": self.order_type,
            "price": self.price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force,
            "status": self.status,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None, 
            "canceled_at": self.canceled_at.isoformat() if self.canceled_at else None,
            "strategy_id": self.strategy_id,
            "tags": self.tags,
            "fill_price": self.fill_price,
            "average_price": self.average_price,
            "fees": self.fees
        }


class PaperPosition:
    """Represents a paper trading position."""
    
    def __init__(
        self,
        symbol: str,
        quantity: float = 0.0,
        average_price: float = 0.0,
        cost_basis: float = 0.0
    ):
        """Initialize a paper position."""
        self.symbol = symbol
        self.quantity = quantity
        self.average_price = average_price
        self.cost_basis = cost_basis
        self.unrealized_pl = 0.0
        self.realized_pl = 0.0
        self.current_price = 0.0
        self.updated_at = datetime.now()
    
    def update_price(self, price: float) -> None:
        """Update position with current market price."""
        self.current_price = price
        self.unrealized_pl = self.quantity * (price - self.average_price)
        self.updated_at = datetime.now()
    
    def add(self, quantity: float, price: float) -> None:
        """Add to position (buy)."""
        if self.quantity == 0:
            self.average_price = price
            self.cost_basis = quantity * price
            self.quantity = quantity
        else:
            total_cost = self.cost_basis + (quantity * price)
            self.quantity += quantity
            self.average_price = total_cost / self.quantity
            self.cost_basis = total_cost
        
        self.updated_at = datetime.now()
    
    def reduce(self, quantity: float, price: float) -> float:
        """Reduce position (sell) and return realized P&L."""
        if quantity > self.quantity:
            quantity = self.quantity  # Cannot sell more than we have
        
        if self.quantity == 0:
            return 0.0
        
        # Calculate realized P&L
        realized = quantity * (price - self.average_price)
        self.realized_pl += realized
        
        # Update position
        self.quantity -= quantity
        
        # If position is closed, reset price
        if self.quantity == 0:
            self.average_price = 0.0
            self.cost_basis = 0.0
        else:
            self.cost_basis = self.quantity * self.average_price
            
        self.updated_at = datetime.now()
        return realized
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "average_price": self.average_price,
            "cost_basis": self.cost_basis,
            "unrealized_pl": self.unrealized_pl,
            "realized_pl": self.realized_pl,
            "current_price": self.current_price,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class PaperBroker(BrokerInterface):
    """
    Paper Trading Broker Implementation
    
    Simulates a broker for paper trading without sending real orders.
    Uses real market data to drive simulated fills.
    """
    
    def __init__(
        self,
        name: str = "PaperBroker",
        initial_balance: float = 100000.0,
        base_currency: str = "USD",
        slippage_model: Optional[Dict[str, Any]] = None,
        commission_model: Optional[Dict[str, Any]] = None,
        data_source: Optional[Any] = None
    ):
        """
        Initialize the paper broker.
        
        Args:
            name: Name for the paper broker
            initial_balance: Starting balance in base currency
            base_currency: Base currency for the account
            slippage_model: Configuration for simulated slippage
            commission_model: Configuration for simulated commissions
            data_source: Optional real data source for quotes
        """
        self._name = name
        self._status = "connected"
        self._balance = initial_balance
        self._initial_balance = initial_balance
        self._base_currency = base_currency
        
        # Slippage model (default: none)
        self._slippage_model = slippage_model or {"type": "none"}
        
        # Commission model (default: none)
        self._commission_model = commission_model or {"type": "none"}
        
        # Tracking collections
        self._positions: Dict[str, PaperPosition] = {}
        self._open_orders: Dict[str, PaperOrder] = {}
        self._filled_orders: Dict[str, PaperOrder] = {}
        self._canceled_orders: Dict[str, PaperOrder] = {}
        
        # Transaction history
        self._transactions: List[Dict[str, Any]] = []
        
        # Last order ID (for sequential IDs)
        self._last_order_id = 0
        
        # Market data source
        self._data_source = data_source
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Event bus for notifications
        self._event_bus = EventBus()
        
        # Order processing thread
        self._stop_processing = False
        self._processing_thread = threading.Thread(
            target=self._process_orders_thread,
            daemon=True
        )
        self._processing_thread.start()
        
        logger.info(f"Paper broker initialized with {initial_balance} {base_currency}")
    
    def _process_orders_thread(self) -> None:
        """Background thread to process pending orders."""
        while not self._stop_processing:
            try:
                self._process_pending_orders()
                time_mod.sleep(1.0)  # Check every second
            except Exception as e:
                logger.error(f"Error processing paper orders: {str(e)}")
    
    def _process_pending_orders(self) -> None:
        """Process any pending orders that can be filled."""
        # Make a copy to avoid modification during iteration
        with self._lock:
            orders_to_process = list(self._open_orders.values())
        
        for order in orders_to_process:
            try:
                # Get current market price
                current_price = self._get_market_price(order.symbol)
                if current_price is None:
                    continue
                
                # Check if order can be filled
                fill_price = self._check_order_fill(order, current_price)
                
                if fill_price is not None:
                    self._fill_order(order, fill_price)
            except Exception as e:
                logger.error(f"Error processing order {order.order_id}: {str(e)}")
    
    def _check_order_fill(self, order: PaperOrder, current_price: float) -> Optional[float]:
        """
        Check if an order should be filled based on current price.
        
        Args:
            order: The order to check
            current_price: Current market price
            
        Returns:
            Fill price if order should be filled, None otherwise
        """
        # Market orders fill immediately
        if order.order_type.lower() == "market":
            return self._apply_slippage(current_price, order.side)
        
        # Limit buy orders fill when price <= limit price
        if order.order_type.lower() == "limit" and order.side.lower() == "buy":
            if current_price <= order.price:
                return self._apply_slippage(order.price, order.side)
        
        # Limit sell orders fill when price >= limit price
        if order.order_type.lower() == "limit" and order.side.lower() == "sell":
            if current_price >= order.price:
                return self._apply_slippage(order.price, order.side)
        
        # Stop buy orders trigger when price >= stop price
        if order.order_type.lower() == "stop" and order.side.lower() == "buy":
            if current_price >= order.stop_price:
                return self._apply_slippage(current_price, order.side)
        
        # Stop sell orders trigger when price <= stop price
        if order.order_type.lower() == "stop" and order.side.lower() == "sell":
            if current_price <= order.stop_price:
                return self._apply_slippage(current_price, order.side)
        
        # Stop limit orders are more complex
        if order.order_type.lower() == "stop_limit":
            # Stop price has been triggered, now it's a limit order
            if (order.side.lower() == "buy" and current_price >= order.stop_price) or \
               (order.side.lower() == "sell" and current_price <= order.stop_price):
                # Convert to a limit order logic
                if (order.side.lower() == "buy" and current_price <= order.price) or \
                   (order.side.lower() == "sell" and current_price >= order.price):
                    return self._apply_slippage(order.price, order.side)
        
        return None
    
    def _apply_slippage(self, price: float, side: str) -> float:
        """
        Apply slippage model to the fill price.
        
        Args:
            price: Base price
            side: Buy or sell
            
        Returns:
            Price with slippage applied
        """
        slippage_type = self._slippage_model.get("type", "none")
        
        if slippage_type == "none":
            return price
        
        if slippage_type == "fixed":
            # Fixed basis points (e.g., 5 = 0.05%)
            basis_points = self._slippage_model.get("basis_points", 0)
            slippage_factor = basis_points / 10000.0  # Convert to decimal
            
            if side.lower() == "buy":
                return price * (1 + slippage_factor)
            else:
                return price * (1 - slippage_factor)
        
        if slippage_type == "random":
            # Random slippage within range
            import random
            min_bp = self._slippage_model.get("min_basis_points", 0)
            max_bp = self._slippage_model.get("max_basis_points", 5)
            basis_points = random.uniform(min_bp, max_bp)
            slippage_factor = basis_points / 10000.0
            
            if side.lower() == "buy":
                return price * (1 + slippage_factor)
            else:
                return price * (1 - slippage_factor)
        
        return price
    
    def _calculate_commission(self, order: PaperOrder, fill_price: float) -> float:
        """
        Calculate commission for an order.
        
        Args:
            order: The order
            fill_price: Fill price
            
        Returns:
            Commission amount
        """
        commission_type = self._commission_model.get("type", "none")
        
        if commission_type == "none":
            return 0.0
        
        if commission_type == "fixed":
            # Fixed per order
            return self._commission_model.get("per_order", 0.0)
        
        if commission_type == "per_share":
            # Per share with minimum
            per_share = self._commission_model.get("per_share", 0.0)
            minimum = self._commission_model.get("minimum", 0.0)
            commission = per_share * order.quantity
            return max(commission, minimum)
        
        if commission_type == "percentage":
            # Percentage of trade value
            percentage = self._commission_model.get("percentage", 0.0)
            return (percentage / 100.0) * (fill_price * order.quantity)
        
        return 0.0
    
    def _fill_order(self, order: PaperOrder, fill_price: float) -> None:
        """
        Fill an order at the specified price.
        
        Args:
            order: Order to fill
            fill_price: Price to fill at
        """
        with self._lock:
            # Calculate commission
            commission = self._calculate_commission(order, fill_price)
            
            # Update order
            order.status = "filled"
            order.filled_quantity = order.quantity
            order.fill_price = fill_price
            order.average_price = fill_price
            order.filled_at = datetime.now()
            order.fees = commission
            
            # Move from open to filled orders
            if order.order_id in self._open_orders:
                del self._open_orders[order.order_id]
            self._filled_orders[order.order_id] = order
            
            # Update position
            symbol = order.symbol
            if symbol not in self._positions:
                self._positions[symbol] = PaperPosition(symbol)
            
            position = self._positions[symbol]
            
            # Update cash balance and position
            total_value = order.quantity * fill_price
            total_cost = total_value + commission
            
            if order.side.lower() == "buy":
                position.add(order.quantity, fill_price)
                self._balance -= total_cost
            else:  # sell
                realized_pl = position.reduce(order.quantity, fill_price)
                self._balance += total_value - commission
            
            # Record transaction
            transaction = {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "price": fill_price,
                "fees": commission,
                "timestamp": datetime.now().isoformat(),
                "strategy_id": order.strategy_id,
                "tags": order.tags
            }
            self._transactions.append(transaction)
            
            # Publish event
            self._publish_fill_event(order)
            
            logger.info(f"[PAPER] Order filled: {order.order_id} {order.side} {order.quantity} {order.symbol} @ {fill_price}")
    
    def _publish_fill_event(self, order: PaperOrder) -> None:
        """
        Publish order fill event.
        
        Args:
            order: Filled order
        """
        event_data = {
            "broker": "paper",
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "price": order.fill_price,
            "type": order.order_type,
            "status": order.status,
            "strategy_id": order.strategy_id,
            "tags": order.tags,
            "timestamp": datetime.now().isoformat()
        }
        
        self._event_bus.publish(Event(
            event_type=EventType.ORDER_FILLED,
            data=event_data
        ))
    
    def _get_market_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price for a symbol.
        
        Args:
            symbol: Symbol to get price for
            
        Returns:
            Current price or None if unavailable
        """
        # If we have a data source, use it
        if self._data_source:
            try:
                quote = self._data_source.get_quote(symbol)
                if quote:
                    return (quote.get("bid") + quote.get("ask")) / 2.0
            except Exception as e:
                logger.error(f"Error getting quote from data source: {str(e)}")
        
        # Otherwise, check if we already have a price in positions
        if symbol in self._positions and self._positions[symbol].current_price > 0:
            return self._positions[symbol].current_price
        
        # As a last resort, try to use the get_quote method
        try:
            quote = self.get_quote(symbol)
            if quote:
                return (quote.get("bid") + quote.get("ask")) / 2.0
        except Exception:
            pass
        
        return None
    
    def _generate_order_id(self) -> str:
        """Generate a unique paper order ID."""
        with self._lock:
            self._last_order_id += 1
            return f"paper-{self._last_order_id}"
    
    # BrokerInterface implementation
    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        # Paper broker is always "open"
        return True
    
    def get_next_market_open(self) -> datetime:
        """Get the next market open datetime."""
        # Paper broker is always open, but return reasonable value
        return datetime.now()
    
    def get_trading_hours(self) -> Dict[str, Any]:
        """Get trading hours information."""
        now = datetime.now()
        return {
            "is_open": True,
            "opens_at": now.replace(hour=9, minute=30, second=0, microsecond=0).isoformat(),
            "closes_at": now.replace(hour=16, minute=0, second=0, microsecond=0).isoformat(),
            "extended_hours": True
        }
    
    def get_account_balances(self) -> Dict[str, Any]:
        """Get account balance information."""
        with self._lock:
            # Calculate total equity (cash + positions)
            positions_value = 0.0
            for position in self._positions.values():
                if position.quantity > 0:
                    current_price = self._get_market_price(position.symbol) or position.average_price
                    positions_value += position.quantity * current_price
            
            return {
                "cash": self._balance,
                "equity": self._balance + positions_value,
                "buying_power": self._balance * 2.0,  # Assume 2x margin
                "initial_balance": self._initial_balance,
                "currency": self._base_currency,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        with self._lock:
            positions = []
            for symbol, position in self._positions.items():
                if position.quantity > 0:
                    # Update current price
                    current_price = self._get_market_price(symbol)
                    if current_price:
                        position.update_price(current_price)
                    
                    positions.append(position.to_dict())
            return positions
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """Get current orders."""
        with self._lock:
            return [order.to_dict() for order in self._open_orders.values()]
    
    def place_equity_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        duration: str = "day",
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        strategy_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Place an equity order.
        
        Args:
            symbol: Stock symbol
            side: Order side ('buy' or 'sell')
            quantity: Number of shares
            order_type: Order type ('market', 'limit', 'stop', 'stop_limit')
            duration: Order duration ('day', 'gtc')
            price: Limit price (required for limit and stop_limit orders)
            stop_price: Stop price (required for stop and stop_limit orders)
            strategy_id: Optional strategy identifier
            tags: Optional tags for the order
            
        Returns:
            Dict: Order result with ID and status
        """
        # Validate order
        if order_type.lower() in ["limit", "stop_limit"] and price is None:
            raise ValueError("Limit price required for limit orders")
        
        if order_type.lower() in ["stop", "stop_limit"] and stop_price is None:
            raise ValueError("Stop price required for stop orders")
        
        with self._lock:
            # Generate order ID
            order_id = self._generate_order_id()
            
            # Create tags if none provided
            if tags is None:
                tags = []
            
            # Always add PAPER tag
            if "PAPER" not in tags:
                tags.append("PAPER")
            
            # Create order
            order = PaperOrder(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                stop_price=stop_price,
                time_in_force=duration,
                strategy_id=strategy_id,
                tags=tags
            )
            
            # Add to open orders
            self._open_orders[order_id] = order
            
            # If market order, try to fill immediately
            if order_type.lower() == "market":
                current_price = self._get_market_price(symbol)
                if current_price:
                    fill_price = self._apply_slippage(current_price, side)
                    self._fill_order(order, fill_price)
            
            logger.info(f"[PAPER] Order placed: {order_id} {side} {quantity} {symbol} ({order_type})")
            
            return {
                "order_id": order_id,
                "status": order.status,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "type": order_type
            }
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get order status.
        
        Args:
            order_id: Order ID
            
        Returns:
            Dict: Order status details
        """
        with self._lock:
            if order_id in self._open_orders:
                return self._open_orders[order_id].to_dict()
            elif order_id in self._filled_orders:
                return self._filled_orders[order_id].to_dict()
            elif order_id in self._canceled_orders:
                return self._canceled_orders[order_id].to_dict()
            else:
                raise ValueError(f"Order {order_id} not found")
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Dict: Cancellation result
        """
        with self._lock:
            if order_id not in self._open_orders:
                return {"success": False, "reason": "Order not found or already executed"}
            
            order = self._open_orders[order_id]
            order.status = "canceled"
            order.canceled_at = datetime.now()
            
            # Move from open to canceled orders
            del self._open_orders[order_id]
            self._canceled_orders[order_id] = order
            
            logger.info(f"[PAPER] Order canceled: {order_id}")
            
            return {
                "success": True,
                "order_id": order_id,
                "status": "canceled"
            }
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time quote for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict: Quote information
        """
        # If we have a data source, use it
        if self._data_source:
            try:
                return self._data_source.get_quote(symbol)
            except Exception as e:
                logger.error(f"Error getting quote from data source: {str(e)}")
        
        # Return a simulated quote if needed
        if symbol in self._positions and self._positions[symbol].current_price > 0:
            price = self._positions[symbol].current_price
            return {
                "symbol": symbol,
                "bid": price * 0.999,
                "ask": price * 1.001,
                "last": price,
                "volume": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        # No price information available
        raise ValueError(f"No quote data available for {symbol}")
    
    def get_historical_data(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical market data.
        
        Args:
            symbol: Stock symbol
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d')
            start_date: Start date
            end_date: End date (defaults to current time)
            
        Returns:
            List[Dict]: Historical data points
        """
        # If we have a data source, delegate to it
        if self._data_source:
            try:
                return self._data_source.get_historical_data(
                    symbol, interval, start_date, end_date
                )
            except Exception as e:
                logger.error(f"Error getting historical data from data source: {str(e)}")
        
        # Return empty list if no data source
        return []
    
    def name(self) -> str:
        """Get broker name."""
        return f"{self._name} [PAPER]"
    
    def status(self) -> str:
        """Get broker connection status."""
        return self._status
    
    def supports_extended_hours(self) -> bool:
        """Check if broker supports extended hours trading."""
        return True
    
    def supports_fractional_shares(self) -> bool:
        """Check if broker supports fractional shares."""
        return True
    
    def api_calls_remaining(self) -> Optional[int]:
        """Get number of API calls remaining (rate limiting)."""
        return None  # No rate limiting for paper broker
    
    def get_broker_time(self) -> datetime:
        """Get current time from broker's servers."""
        return datetime.now()
    
    def needs_refresh(self) -> bool:
        """Check if broker connection needs refresh."""
        return False  # Paper broker never needs refresh
    
    def refresh_connection(self) -> bool:
        """Refresh broker connection (re-authenticate)."""
        return True  # Always successful for paper broker
    
    # Additional methods specific to paper broker
    def get_transactions(self) -> List[Dict[str, Any]]:
        """Get transaction history."""
        with self._lock:
            return self._transactions.copy()
    
    def set_balance(self, balance: float) -> None:
        """
        Set account balance.
        
        Args:
            balance: New balance
        """
        with self._lock:
            self._balance = balance
    
    def set_data_source(self, data_source: Any) -> None:
        """
        Set data source for quotes.
        
        Args:
            data_source: Data source that implements get_quote
        """
        self._data_source = data_source
    
    def reset(self) -> None:
        """Reset paper broker to initial state."""
        with self._lock:
            self._balance = self._initial_balance
            self._positions = {}
            self._open_orders = {}
            self._filled_orders = {}
            self._canceled_orders = {}
            self._transactions = []
            self._last_order_id = 0
    
    def __del__(self):
        """Clean up resources."""
        self._stop_processing = True
        if self._processing_thread.is_alive():
            self._processing_thread.join(timeout=1.0)
