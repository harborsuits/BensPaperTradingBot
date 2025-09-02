#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Trading Mode - Core component for separating signal generation,
trading logic, and risk management.

Inspired by OctoBot and FreqTrade architecture.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import uuid

# Import local components
from trading_bot.strategies.base_strategy import Strategy, SignalType
from trading_bot.risk.risk_manager import RiskManager, RiskLevel

# Setup logging
logger = logging.getLogger("TradingMode")

class OrderType(Enum):
    """Order types supported by the trading system"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class Order:
    """
    Represents a trading order with comprehensive details including order type, 
    price, quantity, and associated risk parameters.
    """
    
    def __init__(
        self,
        symbol: str,
        order_type: OrderType,
        side: SignalType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "gtc",
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        reduce_only: bool = False,
        exchange: Optional[str] = None,
        group_id: Optional[str] = None,
        order_id: Optional[str] = None,
        status: str = "new"
    ):
        """
        Initialize an order with comprehensive parameters
        
        Args:
            symbol: Trading symbol
            order_type: Type of order (market, limit, etc.)
            side: Order direction (long/short)
            quantity: Order size
            price: Limit price (required for limit orders)
            stop_price: Stop price (required for stop orders)
            time_in_force: Time validity (gtc, ioc, fok)
            take_profit: Take profit price
            stop_loss: Stop loss price
            trailing_stop: Trailing stop percentage
            reduce_only: Whether order should only reduce position
            exchange: Exchange identifier
            group_id: Order group ID for linked orders
            order_id: Order ID (auto-generated if not provided)
            status: Order status
        """
        self.symbol = symbol
        self.order_type = order_type
        self.side = side
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.time_in_force = time_in_force
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.trailing_stop = trailing_stop
        self.reduce_only = reduce_only
        self.exchange = exchange
        self.group_id = group_id
        self.order_id = order_id or str(uuid.uuid4())
        self.status = status
        
        # Timestamps
        self.created_time = datetime.now()
        self.updated_time = datetime.now()
        self.filled_time = None
        self.canceled_time = None
        
        # Execution details
        self.filled_quantity = 0.0
        self.average_fill_price = None
        self.commission = 0.0
        self.realized_pnl = 0.0
        
        # Additional metadata
        self.metadata = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary representation"""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "order_type": self.order_type.value,
            "side": self.side.name,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
            "trailing_stop": self.trailing_stop,
            "reduce_only": self.reduce_only,
            "exchange": self.exchange,
            "group_id": self.group_id,
            "status": self.status,
            "created_time": self.created_time.isoformat() if self.created_time else None,
            "updated_time": self.updated_time.isoformat() if self.updated_time else None,
            "filled_time": self.filled_time.isoformat() if self.filled_time else None,
            "canceled_time": self.canceled_time.isoformat() if self.canceled_time else None,
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "commission": self.commission,
            "realized_pnl": self.realized_pnl,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Create order from dictionary representation"""
        order = cls(
            symbol=data["symbol"],
            order_type=OrderType(data["order_type"]),
            side=SignalType[data["side"]],
            quantity=data["quantity"],
            price=data.get("price"),
            stop_price=data.get("stop_price"),
            time_in_force=data.get("time_in_force", "gtc"),
            take_profit=data.get("take_profit"),
            stop_loss=data.get("stop_loss"),
            trailing_stop=data.get("trailing_stop"),
            reduce_only=data.get("reduce_only", False),
            exchange=data.get("exchange"),
            group_id=data.get("group_id"),
            order_id=data.get("order_id"),
            status=data.get("status", "new")
        )
        
        # Set timestamps
        if data.get("created_time"):
            order.created_time = datetime.fromisoformat(data["created_time"])
        if data.get("updated_time"):
            order.updated_time = datetime.fromisoformat(data["updated_time"])
        if data.get("filled_time"):
            order.filled_time = datetime.fromisoformat(data["filled_time"])
        if data.get("canceled_time"):
            order.canceled_time = datetime.fromisoformat(data["canceled_time"])
            
        # Set execution details
        order.filled_quantity = data.get("filled_quantity", 0.0)
        order.average_fill_price = data.get("average_fill_price")
        order.commission = data.get("commission", 0.0)
        order.realized_pnl = data.get("realized_pnl", 0.0)
        
        # Set metadata
        order.metadata = data.get("metadata", {})
        
        return order
    
    def __repr__(self) -> str:
        """String representation of order"""
        return (f"Order(id={self.order_id}, symbol={self.symbol}, "
                f"type={self.order_type.value}, side={self.side.name}, "
                f"qty={self.quantity}, price={self.price}, status={self.status})")


class BaseTradingMode(ABC):
    """
    Abstract base class for all trading modes.
    
    Trading modes define the execution logic layer that sits between
    signal generation (strategies) and the actual order execution,
    providing a clear separation of concerns.
    
    Each trading mode implements a specific approach to order management,
    portfolio allocation, and entry/exit logic while leveraging signals
    from one or more strategies.
    """
    
    def __init__(
        self,
        name: str,
        strategies: Dict[str, Strategy],
        risk_manager: RiskManager,
        symbols: List[str],
        parameters: Dict[str, Any] = None
    ):
        """
        Initialize a trading mode
        
        Args:
            name: Trading mode name
            strategies: Dictionary of strategy name -> strategy instance
            risk_manager: Risk management system
            symbols: Symbols to trade
            parameters: Trading mode parameters
        """
        self.name = name
        self.strategies = strategies
        self.risk_manager = risk_manager
        self.symbols = symbols
        self.parameters = parameters or {}
        
        # Run-time state
        self.active = True
        self.positions = {}
        self.pending_orders = {}
        self.order_history = []
        
        # Performance tracking
        self.equity_curve = []
        self.performance_metrics = {}
        
        logger.info(f"Initialized trading mode: {self.name}")
    
    @abstractmethod
    def process_signals(
        self, 
        strategy_signals: Dict[str, Dict[str, SignalType]], 
        market_data: Dict[str, Dict[str, Any]],
        current_time: pd.Timestamp,
        account_balance: float
    ) -> List[Order]:
        """
        Process signals from strategies and generate orders
        
        Args:
            strategy_signals: Dictionary of strategy name -> {symbol -> signal}
            market_data: Market data for each symbol
            current_time: Current timestamp
            account_balance: Current account balance
            
        Returns:
            List of orders to execute
        """
        pass
    
    @abstractmethod
    def manage_positions(
        self,
        market_data: Dict[str, Dict[str, Any]],
        current_time: pd.Timestamp
    ) -> List[Order]:
        """
        Manage existing positions and generate exit orders if needed
        
        Args:
            market_data: Market data for each symbol
            current_time: Current timestamp
            
        Returns:
            List of orders to execute
        """
        pass
    
    def update(
        self,
        market_data: Dict[str, Dict[str, Any]],
        current_time: pd.Timestamp,
        account_balance: float
    ) -> Dict[str, Any]:
        """
        Update trading mode with new market data and account information
        
        Args:
            market_data: Market data for each symbol
            current_time: Current timestamp
            account_balance: Current account balance
            
        Returns:
            Dictionary of update results including any new orders
        """
        if not self.active:
            return {"orders": [], "positions": list(self.positions.values())}
        
        # Collect signals from all strategies
        strategy_signals = {}
        for name, strategy in self.strategies.items():
            signals = strategy.generate_signals(market_data, current_time)
            strategy_signals[name] = signals
        
        # Process signals to generate orders
        entry_orders = self.process_signals(strategy_signals, market_data, current_time, account_balance)
        
        # Manage existing positions
        exit_orders = self.manage_positions(market_data, current_time)
        
        # Combine all orders
        all_orders = entry_orders + exit_orders
        
        # Add to order history
        self.order_history.extend(all_orders)
        
        # Track equity
        self.equity_curve.append({
            "timestamp": current_time,
            "account_balance": account_balance,
            "positions": len(self.positions)
        })
        
        # Update performance metrics
        self._update_performance_metrics()
        
        return {
            "orders": all_orders,
            "positions": list(self.positions.values()),
            "metrics": self.performance_metrics
        }
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics based on equity curve and trade history"""
        if len(self.equity_curve) < 2:
            return
        
        # Basic metrics
        start_equity = self.equity_curve[0]["account_balance"]
        current_equity = self.equity_curve[-1]["account_balance"]
        total_return = (current_equity / start_equity) - 1.0
        
        # Create equity array and calculate returns
        equity_values = [point["account_balance"] for point in self.equity_curve]
        returns = np.diff(equity_values) / equity_values[:-1]
        
        # Calculate drawdown
        peak = np.maximum.accumulate(equity_values)
        drawdown = 1 - np.array(equity_values) / peak
        max_drawdown = np.max(drawdown)
        
        # Win rate
        if len(self.order_history) > 0:
            winning_trades = sum(1 for order in self.order_history if order.realized_pnl > 0)
            win_rate = winning_trades / len(self.order_history)
        else:
            win_rate = 0.0
        
        # Store metrics
        self.performance_metrics = {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "trades": len(self.order_history),
            "sharpe_ratio": self._calculate_sharpe_ratio(returns),
            "current_drawdown": drawdown[-1]
        }
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio from returns"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
            
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
    
    def set_active(self, active: bool) -> None:
        """Set the active state of the trading mode"""
        self.active = active
        logger.info(f"Trading mode {self.name} active state set to: {active}")

    def handle_order_update(self, order_update: Dict[str, Any]) -> None:
        """
        Handle order status updates from the execution system
        
        Args:
            order_update: Order update information
        """
        order_id = order_update.get("order_id")
        if not order_id:
            logger.warning("Received order update without order_id")
            return
        
        if order_id in self.pending_orders:
            # Update pending order
            order = self.pending_orders[order_id]
            order.status = order_update.get("status", order.status)
            order.filled_quantity = order_update.get("filled_quantity", order.filled_quantity)
            order.average_fill_price = order_update.get("average_fill_price", order.average_fill_price)
            order.updated_time = datetime.now()
            
            # Handle fills and cancellations
            if order.status == "filled":
                order.filled_time = datetime.now()
                # Update positions if order is filled
                self._update_positions(order)
                # Remove from pending orders
                self.pending_orders.pop(order_id)
                # Add to order history
                self.order_history.append(order)
            elif order.status == "canceled":
                order.canceled_time = datetime.now()
                # Remove from pending orders
                self.pending_orders.pop(order_id)
                # Add to order history
                self.order_history.append(order)
                
            logger.info(f"Updated order {order_id}: {order.status}")
        else:
            logger.warning(f"Received update for unknown order: {order_id}")

    def _update_positions(self, order: Order) -> None:
        """
        Update positions based on filled order
        
        Args:
            order: Filled order
        """
        symbol = order.symbol
        
        # Position calculation logic
        if symbol not in self.positions:
            # New position
            if order.side in [SignalType.LONG, SignalType.SHORT]:
                self.positions[symbol] = {
                    "symbol": symbol,
                    "side": order.side,
                    "quantity": order.filled_quantity,
                    "entry_price": order.average_fill_price,
                    "entry_time": order.filled_time,
                    "stop_loss": order.stop_loss,
                    "take_profit": order.take_profit,
                    "trailing_stop": order.trailing_stop
                }
        else:
            # Existing position
            position = self.positions[symbol]
            
            # Check if reducing or closing position
            if ((position["side"] == SignalType.LONG and order.side == SignalType.SHORT) or
                (position["side"] == SignalType.SHORT and order.side == SignalType.LONG)):
                
                # Reduce position
                position["quantity"] -= order.filled_quantity
                
                # Close position if quantity at or below zero
                if position["quantity"] <= 0:
                    self.positions.pop(symbol)
            else:
                # Increase position
                old_qty = position["quantity"]
                new_qty = old_qty + order.filled_quantity
                
                # Calculate new average entry price
                position["entry_price"] = (
                    (old_qty * position["entry_price"]) + 
                    (order.filled_quantity * order.average_fill_price)
                ) / new_qty
                
                position["quantity"] = new_qty
                
                # Update risk parameters if provided
                if order.stop_loss is not None:
                    position["stop_loss"] = order.stop_loss
                if order.take_profit is not None:
                    position["take_profit"] = order.take_profit
                if order.trailing_stop is not None:
                    position["trailing_stop"] = order.trailing_stop
