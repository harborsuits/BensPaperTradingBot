"""
Paper Trading Adapter

Implements the BrokerInterface for paper trading and backtesting.
"""

import logging
import time
import uuid
import json
import os
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Any, Optional, Union, Tuple
import threading
import random

import pandas as pd
import numpy as np

from trading_bot.brokers.broker_interface import BrokerInterface, MarketSession
from trading_bot.core.events import (
    OrderAcknowledged, OrderPartialFill, OrderFilled, 
    OrderCancelled, OrderRejected, SlippageMetric
)
from trading_bot.event_system.event_bus import EventBus


class PaperTradeConfig:
    """Configuration for paper trading."""
    
    def __init__(self, 
                initial_cash: float = 100000.0,
                data_source: str = 'yfinance',
                commission_rate: float = 0.0,
                slippage_model: str = 'random',
                slippage_range: Tuple[float, float] = (0.0001, 0.0005),
                fill_latency_range: Tuple[float, float] = (0.1, 0.5),
                partial_fills_probability: float = 0.2,
                enable_shorting: bool = True,
                margin_requirement: float = 0.5,
                leverage_limit: float = 2.0,
                simulation_mode: str = 'realtime',
                state_file: Optional[str] = None):
        """
        Initialize paper trading configuration.
        
        Args:
            initial_cash: Initial cash balance
            data_source: Price data source ('yfinance', 'alphavantage', 'custom')
            commission_rate: Commission rate as percentage (0.003 = 0.3%)
            slippage_model: Slippage simulation model ('random', 'fixed', 'none')
            slippage_range: Range of random slippage as percentage
            fill_latency_range: Range of simulated order fill latency in seconds
            partial_fills_probability: Probability of partial fills
            enable_shorting: Whether to allow short selling
            margin_requirement: Margin requirement as percentage of position value
            leverage_limit: Maximum allowed leverage
            simulation_mode: 'realtime' or 'backtest'
            state_file: File path to save/load persistent state
        """
        self.initial_cash = initial_cash
        self.data_source = data_source
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model
        self.slippage_range = slippage_range
        self.fill_latency_range = fill_latency_range
        self.partial_fills_probability = partial_fills_probability
        self.enable_shorting = enable_shorting
        self.margin_requirement = margin_requirement
        self.leverage_limit = leverage_limit
        self.simulation_mode = simulation_mode
        self.state_file = state_file
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'initial_cash': self.initial_cash,
            'data_source': self.data_source,
            'commission_rate': self.commission_rate,
            'slippage_model': self.slippage_model,
            'slippage_range': self.slippage_range,
            'fill_latency_range': self.fill_latency_range,
            'partial_fills_probability': self.partial_fills_probability,
            'enable_shorting': self.enable_shorting,
            'margin_requirement': self.margin_requirement,
            'leverage_limit': self.leverage_limit,
            'simulation_mode': self.simulation_mode,
            'state_file': self.state_file
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaperTradeConfig':
        """Create configuration from dictionary."""
        return cls(
            initial_cash=data.get('initial_cash', 100000.0),
            data_source=data.get('data_source', 'yfinance'),
            commission_rate=data.get('commission_rate', 0.0),
            slippage_model=data.get('slippage_model', 'random'),
            slippage_range=data.get('slippage_range', (0.0001, 0.0005)),
            fill_latency_range=data.get('fill_latency_range', (0.1, 0.5)),
            partial_fills_probability=data.get('partial_fills_probability', 0.2),
            enable_shorting=data.get('enable_shorting', True),
            margin_requirement=data.get('margin_requirement', 0.5),
            leverage_limit=data.get('leverage_limit', 2.0),
            simulation_mode=data.get('simulation_mode', 'realtime'),
            state_file=data.get('state_file')
        )


class Position:
    """Represents a paper trading position."""
    
    def __init__(self, symbol: str, quantity: float, avg_price: float, side: str):
        self.symbol = symbol
        self.quantity = quantity  # Positive for long, negative for short
        self.avg_price = avg_price
        self.side = side
        self.market_value = 0.0
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_pct = 0.0
        self.cost_basis = abs(quantity) * avg_price
        
    def update_market_price(self, current_price: float):
        """Update position with current market price."""
        self.market_value = self.quantity * current_price
        self.unrealized_pnl = self.market_value - self.cost_basis if self.side == 'long' else self.cost_basis - self.market_value
        self.unrealized_pnl_pct = self.unrealized_pnl / self.cost_basis if self.cost_basis != 0 else 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_entry_price': self.avg_price,
            'cost_basis': self.cost_basis,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'side': self.side
        }


class Order:
    """Represents a paper trading order."""
    
    def __init__(self, 
                 order_id: str,
                 client_order_id: str,
                 symbol: str, 
                 quantity: float,
                 side: str,
                 order_type: str,
                 time_in_force: str = 'day',
                 limit_price: Optional[float] = None,
                 stop_price: Optional[float] = None,
                 status: str = 'new',
                 filled_quantity: float = 0.0,
                 filled_avg_price: Optional[float] = None,
                 created_at: Optional[datetime] = None,
                 filled_at: Optional[datetime] = None,
                 expected_price: Optional[float] = None):
        """
        Initialize order.
        
        Args:
            order_id: Unique order ID
            client_order_id: Client-side order ID
            symbol: Stock symbol
            quantity: Order quantity
            side: Order side ('buy' or 'sell')
            order_type: Order type ('market', 'limit', 'stop', 'stop_limit')
            time_in_force: Time in force ('day', 'gtc', 'fok', 'ioc')
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            status: Order status
            filled_quantity: Filled quantity
            filled_avg_price: Filled average price
            created_at: Creation timestamp
            filled_at: Fill timestamp
            expected_price: Expected execution price
        """
        self.order_id = order_id
        self.client_order_id = client_order_id
        self.symbol = symbol
        self.quantity = quantity
        self.side = side
        self.order_type = order_type
        self.time_in_force = time_in_force
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.status = status
        self.filled_quantity = filled_quantity
        self.filled_avg_price = filled_avg_price
        self.created_at = created_at or datetime.now()
        self.filled_at = filled_at
        self.expected_price = expected_price
        self.last_update = datetime.now()
        
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == 'filled'
    
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in ['new', 'partially_filled', 'accepted', 'pending']
    
    def update_fill(self, fill_quantity: float, fill_price: float):
        """Update order with a partial or full fill."""
        if fill_quantity <= 0:
            return
            
        remaining = self.quantity - self.filled_quantity
        fill_quantity = min(fill_quantity, remaining)
        
        if self.filled_quantity == 0:
            self.filled_avg_price = fill_price
        else:
            # Calculate new average fill price
            total_value = self.filled_quantity * self.filled_avg_price
            total_value += fill_quantity * fill_price
            self.filled_avg_price = total_value / (self.filled_quantity + fill_quantity)
            
        self.filled_quantity += fill_quantity
        
        if abs(self.filled_quantity - self.quantity) < 0.0001:
            self.status = 'filled'
            self.filled_at = datetime.now()
        else:
            self.status = 'partially_filled'
            
        self.last_update = datetime.now()
        
    def cancel(self):
        """Cancel the order."""
        if self.is_active():
            self.status = 'cancelled'
            self.last_update = datetime.now()
            
    def reject(self, reason: str = 'unknown'):
        """Reject the order."""
        self.status = 'rejected'
        self.last_update = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            'id': self.order_id,
            'client_order_id': self.client_order_id,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'filled_quantity': self.filled_quantity,
            'side': self.side,
            'type': self.order_type,
            'time_in_force': self.time_in_force,
            'limit_price': self.limit_price,
            'stop_price': self.stop_price,
            'status': self.status,
            'filled_avg_price': self.filled_avg_price,
            'created_at': self.created_at,
            'filled_at': self.filled_at,
            'last_update': self.last_update
        }

class PaperTradeAdapter(BrokerInterface):
    """Paper trading adapter implementing the BrokerInterface."""
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """Initialize paper trading adapter."""
        super().__init__(event_bus)
        self.logger = logging.getLogger(__name__)
        self.broker_id = 'paper'
        
        # Paper trading state
        self.config = None
        self.cash = 0.0
        self.initial_cash = 0.0
        self.positions = {}  # symbol -> Position
        self.orders = {}  # order_id -> Order
        self.quotes = {}  # symbol -> Quote
        self.market_data_source = None
        self.simulation_mode = 'realtime'
        self.connected = False
        self.order_processor_thread = None
        self.running = False
        self.lock = threading.RLock()
        
        # Backtesting variables
        self.historical_data = {}  # symbol -> DataFrame
        self.current_time = None
        
    def connect(self, config) -> bool:
        """Connect to paper trading system."""
        if isinstance(config, dict):
            config = PaperTradeConfig.from_dict(config)
        
        self.config = config
        self.cash = config.initial_cash
        self.initial_cash = config.initial_cash
        self.simulation_mode = config.simulation_mode
        
        # Initialize market data source
        if config.data_source == 'yfinance':
            # We'll use yfinance for real-time data in paper mode
            import yfinance as yf
            self.market_data_source = 'yfinance'
        
        # Load saved state if available
        if config.state_file and os.path.exists(config.state_file):
            self._load_state(config.state_file)
        
        # Start order processor thread for real-time simulation
        if self.simulation_mode == 'realtime':
            self.running = True
            self.order_processor_thread = threading.Thread(target=self._order_processor, daemon=True)
            self.order_processor_thread.start()
        
        self.connected = True
        self.logger.info(f"Connected to paper trading ({self.simulation_mode} mode)")
        return True
    
    def disconnect(self) -> bool:
        """Disconnect from paper trading system."""
        self.running = False
        
        # Save state if configured
        if self.config and self.config.state_file:
            self._save_state(self.config.state_file)
        
        # Clear state
        self.positions = {}
        self.orders = {}
        self.quotes = {}
        self.connected = False
        
        return True
    
    def is_connected(self) -> bool:
        """Check if connected to paper trading system."""
        return self.connected
    
    def _save_state(self, file_path: str):
        """Save paper trading state to file."""
        with self.lock:
            state = {
                'cash': self.cash,
                'positions': [p.to_dict() for p in self.positions.values()],
                'orders': [o.to_dict() for o in self.orders.values() if o.status in ['new', 'partially_filled']]
            }
            
            try:
                with open(file_path, 'w') as f:
                    json.dump(state, f, default=str)
                self.logger.info(f"Saved paper trading state to {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to save state: {str(e)}")
    
    def _load_state(self, file_path: str):
        """Load paper trading state from file."""
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            with self.lock:
                self.cash = state.get('cash', self.config.initial_cash)
                
                # Load positions
                for pos_data in state.get('positions', []):
                    position = Position(
                        symbol=pos_data['symbol'],
                        quantity=pos_data['quantity'],
                        avg_price=pos_data['avg_entry_price'],
                        side=pos_data['side']
                    )
                    self.positions[pos_data['symbol']] = position
                
                # Load open orders
                for order_data in state.get('orders', []):
                    if order_data['status'] in ['new', 'partially_filled']:
                        order = Order(
                            order_id=order_data['id'],
                            client_order_id=order_data.get('client_order_id', f"loaded-{uuid.uuid4().hex[:8]}"),
                            symbol=order_data['symbol'],
                            quantity=order_data['quantity'],
                            side=order_data['side'],
                            order_type=order_data['type'],
                            time_in_force=order_data.get('time_in_force', 'day'),
                            limit_price=order_data.get('limit_price'),
                            stop_price=order_data.get('stop_price'),
                            status=order_data['status'],
                            filled_quantity=order_data.get('filled_quantity', 0),
                            filled_avg_price=order_data.get('filled_avg_price')
                        )
                        self.orders[order_data['id']] = order
                        
                self.logger.info(f"Loaded paper trading state from {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to load state: {str(e)}")

    def _order_processor(self):
        """Background thread to process orders."""
        while self.running:
            try:
                self._process_orders()
                time.sleep(0.1)  # Check orders 10 times per second
            except Exception as e:
                self.logger.error(f"Error in order processor: {str(e)}")
    
    def _process_orders(self):
        """Process all active orders."""
        with self.lock:
            active_orders = [o for o in self.orders.values() if o.is_active()]
            
            for order in active_orders:
                # Skip processing if not enough time has passed since creation
                # (simulates order processing latency)
                elapsed = (datetime.now() - order.created_at).total_seconds()
                min_latency, max_latency = self.config.fill_latency_range
                if elapsed < random.uniform(min_latency, max_latency):
                    continue
                
                # Get current price for the symbol
                current_price = self._get_current_price(order.symbol)
                if current_price is None:
                    continue
                
                # Check if order should be filled based on type and price
                should_fill, fill_price = self._should_fill_order(order, current_price)
                
                if should_fill:
                    # Determine fill quantity (partial or full)
                    remaining = order.quantity - order.filled_quantity
                    
                    # Simulate partial fills
                    if (remaining > 1 and 
                        order.order_type == 'market' and 
                        random.random() < self.config.partial_fills_probability):
                        # Partial fill
                        fill_pct = random.uniform(0.2, 0.8)
                        fill_qty = round(remaining * fill_pct, 2)
                    else:
                        # Full fill
                        fill_qty = remaining
                    
                    # Apply slippage
                    fill_price = self._apply_slippage(fill_price, order.side)
                    
                    # Update order
                    order.update_fill(fill_qty, fill_price)
                    
                    # Update portfolio
                    self._execute_fill(order.symbol, fill_qty, fill_price, order.side, order.order_id)
                    
                    # Publish events
                    if order.is_filled():
                        self._publish_order_filled(order)
                    else:
                        self._publish_order_partial_fill(order, fill_qty, fill_price)
    
    def _execute_fill(self, symbol: str, quantity: float, price: float, side: str, order_id: str):
        """Execute a fill by updating positions and cash."""
        # Calculate trade value and commission
        trade_value = quantity * price
        commission = trade_value * self.config.commission_rate
        
        if side == 'buy':
            # Deduct cash
            self.cash -= (trade_value + commission)
            
            # Update position
            if symbol in self.positions:
                position = self.positions[symbol]
                
                if position.side == 'long':
                    # Adding to long position
                    new_quantity = position.quantity + quantity
                    new_cost = position.cost_basis + trade_value
                    new_avg_price = new_cost / new_quantity
                    
                    position.quantity = new_quantity
                    position.avg_price = new_avg_price
                    position.cost_basis = new_cost
                else:
                    # Reducing short position
                    new_quantity = position.quantity - quantity
                    
                    if new_quantity < 0:
                        # Flipping to long
                        new_quantity = abs(new_quantity)
                        position.side = 'long'
                        position.quantity = new_quantity
                        position.avg_price = price
                        position.cost_basis = new_quantity * price
                    else:
                        # Remaining short
                        position.quantity = new_quantity
                        if new_quantity == 0:
                            del self.positions[symbol]
            else:
                # New long position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=price,
                    side='long'
                )
        
        elif side == 'sell':
            # For sell orders, add cash (minus commission)
            self.cash += (trade_value - commission)
            
            # Update position
            if symbol in self.positions:
                position = self.positions[symbol]
                
                if position.side == 'long':
                    # Reducing long position
                    new_quantity = position.quantity - quantity
                    
                    if new_quantity < 0:
                        # Flipping to short
                        new_quantity = abs(new_quantity)
                        position.side = 'short'
                        position.quantity = new_quantity
                        position.avg_price = price
                        position.cost_basis = new_quantity * price
                    else:
                        # Remaining long
                        position.quantity = new_quantity
                        if new_quantity == 0:
                            del self.positions[symbol]
                else:
                    # Adding to short position
                    new_quantity = position.quantity + quantity
                    new_cost = position.cost_basis + trade_value
                    new_avg_price = new_cost / new_quantity
                    
                    position.quantity = new_quantity
                    position.avg_price = new_avg_price
                    position.cost_basis = new_cost
            else:
                # New short position (if shorting enabled)
                if self.config.enable_shorting:
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=-quantity,  # Negative for short
                        avg_price=price,
                        side='short'
                    )
                else:
                    self.logger.warning(f"Attempted to short {symbol} but shorting is disabled")
        
        # Update position market values
        self._update_position_values()
        
        self.logger.info(f"Executed {side} {quantity} {symbol} @ {price} (commission: {commission:.2f})")
    
    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply simulated slippage to price."""
        if self.config.slippage_model == 'none':
            return price
            
        min_slip, max_slip = self.config.slippage_range
        
        if self.config.slippage_model == 'fixed':
            slippage = max_slip
        else:  # random
            slippage = random.uniform(min_slip, max_slip)
        
        # For buys, price goes up; for sells, price goes down
        if side == 'buy':
            return price * (1 + slippage)
        else:
            return price * (1 - slippage)
    
    def _should_fill_order(self, order: Order, current_price: float) -> Tuple[bool, float]:
        """Determine if an order should be filled based on current price."""
        if order.order_type == 'market':
            return True, current_price
            
        elif order.order_type == 'limit':
            if order.side == 'buy' and current_price <= order.limit_price:
                return True, order.limit_price
            elif order.side == 'sell' and current_price >= order.limit_price:
                return True, order.limit_price
                
        elif order.order_type == 'stop':
            if order.side == 'buy' and current_price >= order.stop_price:
                return True, current_price
            elif order.side == 'sell' and current_price <= order.stop_price:
                return True, current_price
                
        elif order.order_type == 'stop_limit':
            if order.side == 'buy' and current_price >= order.stop_price:
                # Triggered, use limit price
                if current_price <= order.limit_price:
                    return True, order.limit_price
                else:
                    return False, 0
            elif order.side == 'sell' and current_price <= order.stop_price:
                # Triggered, use limit price
                if current_price >= order.limit_price:
                    return True, order.limit_price
                else:
                    return False, 0
                    
        return False, 0
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        # In real-time mode, fetch from data source
        if self.simulation_mode == 'realtime':
            quote = self.get_quote(symbol)
            if quote:
                return (quote.get('bid', 0) + quote.get('ask', 0)) / 2
            return None
            
        # In backtest mode, use historical data at current time
        elif self.simulation_mode == 'backtest':
            if symbol not in self.historical_data or self.current_time is None:
                return None
                
            data = self.historical_data[symbol]
            # Find the closest data point <= current time
            mask = data.index <= self.current_time
            if not mask.any():
                return None
                
            row = data[mask].iloc[-1]
            return row['close']
    
    def _update_position_values(self):
        """Update market values for all positions."""
        for symbol, position in list(self.positions.items()):
            current_price = self._get_current_price(symbol)
            if current_price:
                position.update_market_price(current_price)
            else:
                self.logger.warning(f"Could not update position for {symbol}: price not available")
    
    def load_historical_data(self, data: Dict[str, pd.DataFrame]):
        """Load historical data for backtesting."""
        self.historical_data = data
        self.simulation_mode = 'backtest'
        self.logger.info(f"Loaded historical data for {len(data)} symbols")
        
    def set_backtest_time(self, dt: datetime):
        """Set current time for backtesting."""
        self.current_time = dt
        
        # Process any pending orders at this time
        self._process_orders()
        
        # Update position values
        self._update_position_values()
    
    def _publish_order_filled(self, order: Order):
        """Publish order filled event."""
        if not self.event_bus:
            return
            
        self.event_bus.publish(OrderFilled(
            broker_id=self.broker_id,
            order_id=order.order_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            quantity=order.quantity,
            price=order.filled_avg_price,
            side=order.side,
            timestamp=datetime.now() if self.simulation_mode == 'realtime' else self.current_time
        ))
        
        # Publish slippage metric if expected price was provided
        if order.expected_price and order.filled_avg_price:
            slippage = abs(order.filled_avg_price - order.expected_price) / order.expected_price
            self.event_bus.publish(SlippageMetric(
                broker_id=self.broker_id,
                order_id=order.order_id,
                symbol=order.symbol,
                expected_price=order.expected_price,
                execution_price=order.filled_avg_price,
                slippage=slippage,
                timestamp=datetime.now() if self.simulation_mode == 'realtime' else self.current_time
            ))
    
    def _publish_order_partial_fill(self, order: Order, fill_qty: float, fill_price: float):
        """Publish order partial fill event."""
        if not self.event_bus:
            return
            
        self.event_bus.publish(OrderPartialFill(
            broker_id=self.broker_id,
            order_id=order.order_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            fill_quantity=fill_qty,
            fill_price=fill_price,
            remaining_quantity=order.quantity - order.filled_quantity,
            side=order.side,
            timestamp=datetime.now() if self.simulation_mode == 'realtime' else self.current_time
        ))
    
    def is_market_open(self) -> bool:
        """Check if market is open."""
        if self.simulation_mode == 'backtest':
            # In backtest mode, always consider market open
            return True
            
        # Get current time (system time or simulation time)
        now = datetime.now() if self.simulation_mode == 'realtime' else self.current_time
        
        # Check if it's a trading day and during market hours
        return MarketSession.is_market_open(now)
    
    def get_next_market_open(self) -> datetime:
        """Get next market open time."""
        now = datetime.now() if self.simulation_mode == 'realtime' else self.current_time
        return MarketSession.get_next_market_open(now)
    
    def get_trading_hours(self) -> Dict[str, Any]:
        """Get trading hours information."""
        now = datetime.now() if self.simulation_mode == 'realtime' else self.current_time
        
        if not MarketSession.is_trading_day(now):
            next_open = MarketSession.get_next_market_open(now)
            return {
                'is_open': False,
                'next_open': next_open
            }
        
        current_date = now.date()
        market_open = datetime.combine(current_date, MarketSession.REGULAR_OPEN)
        market_close = datetime.combine(current_date, MarketSession.REGULAR_CLOSE)
        
        return {
            'is_open': MarketSession.is_market_open(now),
            'market_open': market_open,
            'market_close': market_close,
            'pre_market_open': datetime.combine(current_date, MarketSession.PREMARKET_OPEN),
            'after_market_close': datetime.combine(current_date, MarketSession.AFTERHOURS_CLOSE)
        }
    
    def get_account_balances(self) -> Dict[str, Any]:
        """Get account balance information."""
        # Calculate total equity (cash + position values)
        total_position_value = sum(p.market_value for p in self.positions.values())
        equity = self.cash + total_position_value
        
        # Calculate buying power based on margin
        buying_power = equity * self.config.leverage_limit
        
        return {
            'cash': self.cash,
            'equity': equity,
            'initial_cash': self.initial_cash,
            'buying_power': buying_power,
            'long_market_value': sum(p.market_value for p in self.positions.values() if p.side == 'long'),
            'short_market_value': sum(abs(p.market_value) for p in self.positions.values() if p.side == 'short'),
            'total_positions_value': total_position_value,
            'day_pnl': equity - self.initial_cash,
            'day_pnl_pct': (equity - self.initial_cash) / self.initial_cash if self.initial_cash > 0 else 0
        }
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        self._update_position_values()
        return [position.to_dict() for position in self.positions.values()]
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """Get current orders."""
        return [order.to_dict() for order in self.orders.values()]
    
    def place_equity_order(self, symbol: str, quantity: int, side: str, order_type: str, 
                         time_in_force: str = 'day', limit_price: float = None, 
                         stop_price: float = None, expected_price: float = None) -> Dict[str, Any]:
        """Place an equity order."""
        if not self.is_connected():
            return {'success': False, 'error': 'Not connected to paper trading system'}
        
        # Validate inputs
        if side not in ['buy', 'sell']:
            return {'success': False, 'error': f"Invalid side: {side}"}
            
        if order_type not in ['market', 'limit', 'stop', 'stop_limit']:
            return {'success': False, 'error': f"Invalid order type: {order_type}"}
            
        if order_type in ['limit', 'stop_limit'] and limit_price is None:
            return {'success': False, 'error': f"Limit price required for {order_type} orders"}
            
        if order_type in ['stop', 'stop_limit'] and stop_price is None:
            return {'success': False, 'error': f"Stop price required for {order_type} orders"}
        
        # Check if we have enough cash for buy orders or shares for sell orders
        if side == 'buy':
            # Estimate maximum cost including slippage and commission
            price_estimate = self._get_current_price(symbol)
            if price_estimate is None:
                return {'success': False, 'error': f"Could not get current price for {symbol}"}
                
            max_slippage = max(self.config.slippage_range)
            estimated_price = price_estimate * (1 + max_slippage)
            estimated_cost = quantity * estimated_price
            estimated_commission = estimated_cost * self.config.commission_rate
            total_cost = estimated_cost + estimated_commission
            
            if total_cost > self.cash and order_type in ['market']:
                return {'success': False, 'error': f"Insufficient cash: {self.cash:.2f} < {total_cost:.2f}"}
                
        elif side == 'sell':
            # Check if we have enough shares to sell
            if symbol in self.positions:
                position = self.positions[symbol]
                if position.side == 'long' and position.quantity < quantity:
                    return {'success': False, 'error': f"Insufficient shares: {position.quantity} < {quantity}"}
            else:
                # Trying to sell something we don't have
                if not self.config.enable_shorting:
                    return {'success': False, 'error': f"No position in {symbol} and shorting is disabled"}
        
        # Create the order
        order_id = f"paper-{uuid.uuid4().hex}"
        client_order_id = f"bensbot-{uuid.uuid4().hex[:12]}"
        
        order = Order(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            quantity=quantity,
            side=side,
            order_type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price,
            stop_price=stop_price,
            status='new',
            expected_price=expected_price
        )
        
        # Store the order
        with self.lock:
            self.orders[order_id] = order
        
        # Publish order acknowledged event
        if self.event_bus:
            self.event_bus.publish(OrderAcknowledged(
                broker_id=self.broker_id,
                order_id=order_id,
                client_order_id=client_order_id,
                symbol=symbol,
                quantity=quantity,
                side=side,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                timestamp=datetime.now() if self.simulation_mode == 'realtime' else self.current_time
            ))
        
        # If it's a market order, process it immediately for backtest mode
        if self.simulation_mode == 'backtest' and order_type == 'market':
            self._process_orders()
        
        return {
            'success': True,
            'order_id': order_id,
            'client_order_id': client_order_id,
            'status': 'accepted'
        }
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status."""
        if order_id not in self.orders:
            return {'success': False, 'error': f"Order not found: {order_id}"}
            
        order = self.orders[order_id]
        return {
            'success': True,
            'order_id': order_id,
            'client_order_id': order.client_order_id,
            'symbol': order.symbol,
            'quantity': order.quantity,
            'filled_quantity': order.filled_quantity,
            'side': order.side,
            'type': order.order_type,
            'status': order.status,
            'limit_price': order.limit_price,
            'stop_price': order.stop_price,
            'filled_avg_price': order.filled_avg_price,
            'created_at': order.created_at,
            'filled_at': order.filled_at
        }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order."""
        if order_id not in self.orders:
            return {'success': False, 'error': f"Order not found: {order_id}"}
            
        order = self.orders[order_id]
        
        if not order.is_active():
            return {'success': False, 'error': f"Cannot cancel order with status: {order.status}"}
            
        order.cancel()
        
        # Publish cancellation event
        if self.event_bus:
            self.event_bus.publish(OrderCancelled(
                broker_id=self.broker_id,
                order_id=order_id,
                timestamp=datetime.now() if self.simulation_mode == 'realtime' else self.current_time
            ))
            
        return {
            'success': True,
            'order_id': order_id,
            'status': 'cancelled'
        }
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get quote for a symbol."""
        if self.simulation_mode == 'realtime':
            # Use actual market data source
            if self.market_data_source == 'yfinance':
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period='1d')
                    
                    if data.empty:
                        return {}
                        
                    last_row = data.iloc[-1]
                    
                    # Simulate bid/ask spread
                    last_price = last_row['Close']
                    spread = last_price * 0.0005  # 0.05% spread
                    
                    quote = {
                        'symbol': symbol,
                        'bid': last_price - spread/2,
                        'ask': last_price + spread/2,
                        'last': last_price,
                        'volume': int(last_row['Volume']),
                        'timestamp': datetime.now()
                    }
                    
                    # Cache the quote
                    self.quotes[symbol] = quote
                    return quote
                except Exception as e:
                    self.logger.error(f"Error getting quote for {symbol}: {str(e)}")
                    return {}
            
            # Fallback to cached quote
            return self.quotes.get(symbol, {})
            
        elif self.simulation_mode == 'backtest':
            if symbol not in self.historical_data or self.current_time is None:
                return {}
                
            data = self.historical_data[symbol]
            # Find the closest data point <= current time
            mask = data.index <= self.current_time
            if not mask.any():
                return {}
                
            row = data[mask].iloc[-1]
            
            # Simulate bid/ask spread
            last_price = row['close']
            spread = last_price * 0.0005  # 0.05% spread
            
            return {
                'symbol': symbol,
                'bid': last_price - spread/2,
                'ask': last_price + spread/2,
                'last': last_price,
                'volume': int(row['volume']) if 'volume' in row else 0,
                'timestamp': row.name if isinstance(row.name, datetime) else self.current_time
            }
    
    def get_historical_data(self, symbol: str, interval: str, start_date: datetime, 
                          end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get historical market data."""
        if end_date is None:
            end_date = datetime.now() if self.simulation_mode == 'realtime' else self.current_time
            
        if self.simulation_mode == 'backtest' and symbol in self.historical_data:
            # Use loaded historical data
            data = self.historical_data[symbol]
            mask = (data.index >= start_date) & (data.index <= end_date)
            filtered = data[mask]
            
            result = []
            for timestamp, row in filtered.iterrows():
                result.append({
                    'timestamp': timestamp,
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'] if 'volume' in row else 0
                })
                
            return result
            
        elif self.market_data_source == 'yfinance':
            try:
                import yfinance as yf
                
                # Map interval to yfinance format
                yf_interval = {
                    '1m': '1m',
                    '5m': '5m',
                    '15m': '15m',
                    '30m': '30m',
                    '1h': '1h',
                    '1d': '1d'
                }
                
                interval_str = yf_interval.get(interval, '1d')
                
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    interval=interval_str,
                    start=start_date,
                    end=end_date
                )
                
                result = []
                for timestamp, row in df.iterrows():
                    result.append({
                        'timestamp': timestamp.to_pydatetime(),
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'volume': int(row['Volume'])
                    })
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error getting historical data for {symbol}: {str(e)}")
                return []
        
        return []
    
    def name(self) -> str:
        """Get broker name."""
        mode = "Backtest" if self.simulation_mode == 'backtest' else "Paper"
        return f"{mode} Trading"
    
    def status(self) -> str:
        """Get broker status."""
        if self.is_connected():
            return 'connected'
        else:
            return 'disconnected'
    
    def supports_extended_hours(self) -> bool:
        """Check if broker supports extended hours trading."""
        return True
    
    def supports_fractional_shares(self) -> bool:
        """Check if broker supports fractional shares."""
        return True
    
    def api_calls_remaining(self) -> Optional[int]:
        """Get number of API calls remaining."""
        return None  # Not applicable for paper trading
    
    def get_broker_time(self) -> datetime:
        """Get broker time."""
        return self.current_time if self.simulation_mode == 'backtest' else datetime.now()
    
    def get_margin_status(self) -> Dict[str, Any]:
        """Get margin account status."""
        # Calculate total position value
        total_position_value = sum(abs(p.market_value) for p in self.positions.values())
        
        # Calculate margin used
        margin_used = total_position_value * self.config.margin_requirement
        
        # Calculate equity
        equity = self.cash + sum(p.market_value for p in self.positions.values())
        
        # Calculate margin percentage
        margin_percentage = 1.0
        if margin_used > 0:
            margin_percentage = equity / margin_used
        
        return {
            'cash': self.cash,
            'equity': equity,
            'margin_used': margin_used,
            'margin_available': equity - margin_used if equity > margin_used else 0.0,
            'margin_percentage': margin_percentage,
            'leverage_limit': self.config.leverage_limit,
            'last_updated': self.current_time if self.simulation_mode == 'backtest' else datetime.now()
        }
    
    def refresh_connection(self) -> bool:
        """Refresh broker connection."""
        # Not needed for paper trading, always return True
        return True
    
    def needs_refresh(self) -> bool:
        """Check if broker connection needs refresh."""
        # Not needed for paper trading, always return False
        return False
