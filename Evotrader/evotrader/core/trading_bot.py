"""TradingBot abstract interface."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import time
import uuid


class OrderType(Enum):
    """Types of orders that can be placed."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Side of an order."""
    BUY = "buy"
    SELL = "sell"


class Order:
    """Represents a trading order."""
    
    def __init__(self, 
                 symbol: str, 
                 side: OrderSide, 
                 quantity: float, 
                 order_type: OrderType = OrderType.MARKET,
                 price: Optional[float] = None,
                 stop_price: Optional[float] = None):
        self.id = str(uuid.uuid4())
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.price = price
        self.stop_price = stop_price
        self.created_at = time.time()
        self.executed_at: Optional[float] = None
        self.executed_price: Optional[float] = None
        self.status = "created"  # created, executed, canceled, rejected
        self.fees: float = 0.0


class Position:
    """Represents an open trading position."""
    
    def __init__(self, symbol: str, quantity: float, entry_price: float):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = time.time()
        # For tracking partial exits
        self.realized_pnl = 0.0
        
    def update(self, quantity_change: float, price: float) -> float:
        """Update position after a trade, return realized P/L."""
        if abs(quantity_change) > abs(self.quantity):
            raise ValueError(f"Cannot change position by {quantity_change} as only {self.quantity} remains")
        
        # Calculate P/L for the portion being closed
        if quantity_change < 0 and self.quantity > 0:  # Selling long position
            realized_pnl = -quantity_change * (price - self.entry_price)
        elif quantity_change > 0 and self.quantity < 0:  # Buying to cover short
            realized_pnl = -quantity_change * (self.entry_price - price)
        else:  # Adding to position
            realized_pnl = 0.0
            
        # Update position
        old_quantity = self.quantity
        self.quantity += quantity_change
        
        # Update entry price for adds to position
        if old_quantity * quantity_change > 0:  # Adding to position in same direction
            self.entry_price = ((old_quantity * self.entry_price) + 
                              (quantity_change * price)) / self.quantity
            
        # Track realized P/L
        self.realized_pnl += realized_pnl
        return realized_pnl


class Trade:
    """Record of a completed trade."""
    
    def __init__(self, 
                 symbol: str, 
                 side: OrderSide, 
                 quantity: float, 
                 price: float,
                 timestamp: float,
                 fees: float = 0.0,
                 order_id: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.price = price
        self.timestamp = timestamp
        self.fees = fees
        self.order_id = order_id


class TradingBot(ABC):
    """Abstract base class for all trading bots."""
    
    def __init__(self, bot_id: str):
        """Initialize bot with unique ID."""
        self.bot_id = bot_id
        self.creation_time = time.time()
        self.last_update_time: Optional[float] = None
        
        # Financial state
        self.initial_balance: float = 0.0
        self.balance: float = 0.0
        self.equity: float = 0.0  # Balance + unrealized P/L
        
        # Trading state
        self.positions: Dict[str, Position] = {}  # Symbol -> Position
        self.open_orders: Dict[str, Order] = {}  # Order ID -> Order
        self.trades: List[Trade] = []
        
        # Performance metrics
        self.metrics: Dict[str, Any] = {
            "win_count": 0,
            "loss_count": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "max_equity": 0.0,
            "daily_returns": {},
            "sharpe_ratio": None,
        }
        
        # State tracking
        self.is_active = True
    
    @abstractmethod
    def initialize(self, initial_balance: float = 1.0):
        """Initialize bot state before simulation starts."""
        pass

    @abstractmethod
    def on_data(self, market_data: Dict[str, Any]) -> List[Order]:
        """Process incoming market data and execute strategy logic."""
        pass
    
    def update_equity(self, market_data: Dict[str, Any]) -> float:
        """Update equity calculation with current market prices."""
        unrealized_pnl = 0.0
        for symbol, position in self.positions.items():
            # Assume market_data contains the latest price for this symbol
            if symbol in market_data and "price" in market_data[symbol]:
                current_price = market_data[symbol]["price"]
                if position.quantity > 0:  # Long position
                    unrealized_pnl += position.quantity * (current_price - position.entry_price)
                elif position.quantity < 0:  # Short position
                    unrealized_pnl += position.quantity * (position.entry_price - current_price)
        
        self.equity = self.balance + unrealized_pnl
        
        # Update max equity and drawdown metrics
        if self.equity > self.metrics["max_equity"]:
            self.metrics["max_equity"] = self.equity
        
        if self.metrics["max_equity"] > 0:
            current_drawdown = (self.metrics["max_equity"] - self.equity) / self.metrics["max_equity"]
            if current_drawdown > self.metrics["max_drawdown"]:
                self.metrics["max_drawdown"] = current_drawdown
        
        return self.equity
    
    def place_order(self, order: Order) -> Order:
        """Place a new order."""
        # Add to open orders
        self.open_orders[order.id] = order
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by ID."""
        if order_id in self.open_orders:
            self.open_orders[order_id].status = "canceled"
            del self.open_orders[order_id]
            return True
        return False
    
    def handle_order_fill(self, order: Order, fill_price: float, timestamp: float, fees: float = 0.0) -> Tuple[Trade, float]:
        """Process an order fill and update position."""
        # Create trade record
        trade = Trade(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=timestamp,
            fees=fees,
            order_id=order.id
        )
        self.trades.append(trade)
        
        # Update order status
        order.status = "executed"
        order.executed_at = timestamp
        order.executed_price = fill_price
        order.fees = fees
        
        # Remove from open orders
        if order.id in self.open_orders:
            del self.open_orders[order.id]
        
        # Update positions
        position_change = order.quantity if order.side == OrderSide.BUY else -order.quantity
        realized_pnl = 0.0
        
        # Get or create position
        if order.symbol not in self.positions:
            self.positions[order.symbol] = Position(order.symbol, 0, fill_price)
        
        # Apply position change
        position = self.positions[order.symbol]
        realized_pnl = position.update(position_change, fill_price)
        
        # Clean up if position is closed
        if position.quantity == 0:
            del self.positions[order.symbol]
            # Update win/loss count and profit/loss metrics
            if position.realized_pnl > 0:
                self.metrics["win_count"] += 1
                self.metrics["gross_profit"] += position.realized_pnl  # Track gross profit
            elif position.realized_pnl < 0:
                self.metrics["loss_count"] += 1
                self.metrics["gross_loss"] += abs(position.realized_pnl)  # Track gross loss (as positive value)
                
        # Update balance (reduced by fees)
        self.balance += realized_pnl - fees
        self.metrics["total_pnl"] += realized_pnl
        
        return trade, realized_pnl
    
    def get_position_size(self, symbol: str) -> float:
        """Get current position size for a symbol."""
        if symbol in self.positions:
            return self.positions[symbol].quantity
        return 0.0
    
    def get_unrealized_pnl(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate unrealized profit/loss for all positions."""
        result = {}
        
        for symbol, position in self.positions.items():
            if symbol in market_data and "price" in market_data[symbol]:
                current_price = market_data[symbol]["price"]
                
                if position.quantity > 0:  # Long
                    pnl = position.quantity * (current_price - position.entry_price)
                else:  # Short
                    pnl = position.quantity * (position.entry_price - current_price)
                    
                result[symbol] = pnl
                
        return result
        
    @abstractmethod
    def finalize(self) -> Dict[str, Any]:
        """Finalize any state and cleanup after simulation ends.
        
        Returns:
            Dict: Final metrics and performance summary
        """
        pass
