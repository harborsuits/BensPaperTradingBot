#!/usr/bin/env python3
"""
Strategy Base Class

This module defines the base class for all trading strategies in the system.
All strategy types (ML models, indicator rules, pattern detectors, etc.)
must inherit from this base class.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable
from enum import Enum, auto
from functools import partial

from trading_bot.core.constants import EventType, AssetType
from trading_bot.core.event_bus import Event, get_global_event_bus
from trading_bot.models.order import Order, OrderSide, OrderType, TimeInForce
from trading_bot.models.position import Position
from trading_bot.models.account import Account
from trading_bot.models.market_data import MarketData, Bar, Quote, Trade

logger = logging.getLogger(__name__)


class StrategyPriority(Enum):
    """Priority levels for strategy execution and conflict resolution."""
    LOWEST = 1
    LOW = 2
    NORMAL = 3
    HIGH = 4
    HIGHEST = 5


class ConflictResolutionMode(Enum):
    """How to handle conflicts between multiple strategies on the same asset."""
    PRIORITY_BASED = "priority_based"    # Higher priority strategy wins
    PERFORMANCE_BASED = "performance_based"  # Better performing strategy wins
    NEWER_SIGNAL = "newer_signal"      # Most recent signal wins
    CONSERVATIVE = "conservative"      # Don't trade if conflict exists
    SPLIT_ALLOCATION = "split_allocation"  # Allow both strategies to trade with reduced size
    MANUAL = "manual"                # Require manual intervention


class SignalTag(Enum):
    """Tags that can be applied to strategy signals for filtering and routing."""
    ENTRY = "entry"                # Position entry
    EXIT = "exit"                  # Position exit
    MARKET_REGIME = "market_regime"  # Market regime change indicator
    ADJUSTMENT = "adjustment"      # Position adjustment (stop, target, etc.)
    HIGH_CONFIDENCE = "high_confidence"  # High confidence signal
    LOW_CONFIDENCE = "low_confidence"    # Low confidence signal
    SCALPING = "scalping"          # Short-term signal
    SWING = "swing"                # Medium-term signal
    POSITION = "position"          # Long-term signal
    # Allow custom tags through string values


class StrategyState(Enum):
    """Enum representing the possible states of a strategy."""
    INITIALIZED = "initialized"  # Strategy is initialized but not running
    RUNNING = "running"          # Strategy is active and processing data
    PAUSED = "paused"            # Strategy is temporarily paused (e.g., due to performance issues)
    STOPPED = "stopped"          # Strategy is manually stopped
    ERROR = "error"              # Strategy encountered an error
    PROMOTED = "promoted"        # Strategy has been promoted due to good performance
    PROBATION = "probation"      # Strategy is underperforming but still active
    RETIRED = "retired"          # Strategy has been retired due to poor performance


class StrategyType(Enum):
    """Enum representing the type of strategy."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    TREND_FOLLOWING = "trend_following"
    MACHINE_LEARNING = "machine_learning"
    ARBITRAGE = "arbitrage"
    STATISTICAL = "statistical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    PATTERN_RECOGNITION = "pattern_recognition"
    EXTERNAL_SIGNAL = "external_signal"
    INDICATOR_BASED = "indicator_based"
    CUSTOM = "custom"
    ENSEMBLE = "ensemble"
    # Added for options strategies
    VOLATILITY = "volatility"
    INCOME = "income"


class StrategyMetrics:
    """Class to track performance metrics for a strategy."""
    
    def __init__(self):
        # Profitability metrics
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.daily_pnl = 0.0
        self.cumulative_returns = 0.0
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.volatility = 0.0
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.calmar_ratio = 0.0
        
        # Trading metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_rate = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.profit_factor = 0.0
        self.expectancy = 0.0
        
        # Time metrics
        self.start_time = datetime.now()
        self.runtime = 0.0
        self.last_update = datetime.now()

    def update(self, new_metrics: Dict[str, Any]) -> None:
        """Update metrics with new values."""
        for key, value in new_metrics.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.last_update = datetime.now()
        self.runtime = (datetime.now() - self.start_time).total_seconds()
        
        # Calculate derived metrics
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
        
        if self.losing_trades > 0 and self.winning_trades > 0:
            self.profit_factor = (self.avg_win * self.winning_trades) / (abs(self.avg_loss) * self.losing_trades)
            self.expectancy = (self.win_rate * self.avg_win) - ((1 - self.win_rate) * abs(self.avg_loss))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            # Profitability metrics
            "total_pnl": self.total_pnl,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "daily_pnl": self.daily_pnl,
            "cumulative_returns": self.cumulative_returns,
            
            # Risk metrics
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            
            # Trading metrics
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            
            # Time metrics
            "start_time": self.start_time.isoformat(),
            "runtime": self.runtime,
            "last_update": self.last_update.isoformat()
        }


class Strategy(ABC):
    """
    Base class for all trading strategies.
    
    All strategies must inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(
        self,
        strategy_id: str = None,
        name: str = None,
        description: str = None,
        symbols: List[str] = None,
        asset_type: AssetType = None,
        timeframe: str = "1m",
        parameters: Dict[str, Any] = None,
        risk_limits: Dict[str, Any] = None,
        broker_id: str = None,
        enabled: bool = True,
        # Unified strategy handling extensions
        strategy_type: StrategyType = StrategyType.CUSTOM,
        priority: StrategyPriority = StrategyPriority.NORMAL,
        signal_tags: Set[Union[str, SignalTag]] = None,
        conflict_resolution: ConflictResolutionMode = ConflictResolutionMode.PRIORITY_BASED,
        subaccount_id: Optional[str] = None,
        order_tag_prefix: Optional[str] = None
    ):
        """
        Initialize a new strategy.
        
        Args:
            strategy_id: Unique identifier for the strategy
            name: Human-readable name of the strategy
            description: Detailed description of the strategy
            symbols: List of symbols this strategy trades
            asset_type: Type of asset this strategy trades
            timeframe: Timeframe for data analysis (e.g., "1m", "5m", "1h", "1d")
            parameters: Strategy-specific parameters
            risk_limits: Risk management parameters
            broker_id: ID of the broker to use (None for any/default)
            enabled: Whether the strategy is enabled by default
        """
        # Basic strategy information
        self.strategy_id = strategy_id or str(uuid.uuid4())
        self.name = name or f"Strategy_{self.strategy_id[:8]}"
        self.description = description or "No description"
        self.symbols = symbols or []
        self.asset_type = asset_type
        self.timeframe = timeframe
        self.parameters = parameters or {}
        self.risk_limits = risk_limits or {}
        self.broker_id = broker_id
        self.enabled = enabled
        
        # State tracking
        self.state = StrategyState.INITIALIZED
        self.metrics = StrategyMetrics()
        self.positions = {}
        self.pending_orders = {}
        self.historical_orders = []
        
        # Unified strategy handling extensions
        self.strategy_type = strategy_type
        self.priority = priority
        self.signal_tags = set(signal_tags) if signal_tags else set()
        self.conflict_resolution = conflict_resolution
        self.subaccount_id = subaccount_id
        self.order_tag_prefix = order_tag_prefix or self.strategy_id[:8]
        
        # Signal tracking
        self.generated_signals = []
        self.active_signals = {}
        self.signal_performance = {}
        
        # Conflict tracking
        self.conflicts = []
        self.conflict_resolutions = []
        
        # Get a reference to the global event bus
        self._event_bus = get_global_event_bus()
        
        # Subscribe to relevant events
        self._subscribe_to_events()
        
        logger.info(f"Initialized strategy: {self.name} ({self.strategy_id})")
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events on the event bus."""
        # Core market data events
        self._event_bus.subscribe(EventType.MARKET_DATA_UPDATE, self._on_market_data_event)
        self._event_bus.subscribe(EventType.QUOTE_UPDATE, self._on_quote_event)
        self._event_bus.subscribe(EventType.TRADE_UPDATE, self._on_trade_event)
        self._event_bus.subscribe(EventType.BAR_UPDATE, self._on_bar_event)
        
        # Order and position events
        self._event_bus.subscribe(EventType.ORDER_FILLED, self._on_order_filled)
        self._event_bus.subscribe(EventType.ORDER_CANCELED, self._on_order_canceled)
        self._event_bus.subscribe(EventType.ORDER_REJECTED, self._on_order_rejected)
        self._event_bus.subscribe(EventType.POSITION_UPDATE, self._on_position_update)
    
    # Event handlers
    def _on_market_data_event(self, event: Event) -> None:
        """Handle general market data events."""
        if not self.is_running():
            return
        
        data = event.data
        if data.get("symbol") in self.symbols:
            self.last_data[data.get("symbol")] = data
            self.on_data(data)
    
    def _on_quote_event(self, event: Event) -> None:
        """Handle quote update events."""
        if not self.is_running():
            return
        
        quote = event.data
        if quote.get("symbol") in self.symbols:
            self.on_quote(quote)
    
    def _on_trade_event(self, event: Event) -> None:
        """Handle trade update events."""
        if not self.is_running():
            return
        
        trade = event.data
        if trade.get("symbol") in self.symbols:
            self.on_trade(trade)
    
    def _on_bar_event(self, event: Event) -> None:
        """Handle bar/candle update events."""
        if not self.is_running():
            return
        
        bar = event.data
        if bar.get("symbol") in self.symbols:
            self.on_bar(bar)
    
    def _on_order_filled(self, event: Event) -> None:
        """Handle order filled events."""
        order = event.data
        if order.get("strategy_id") == self.strategy_id:
            # Update metrics
            price = order.get("fill_price", 0.0)
            quantity = order.get("quantity", 0)
            order_side = order.get("side", "")
            
            # Add to historical orders
            self.historical_orders.append(order)
            
            # Remove from pending orders
            order_id = order.get("order_id")
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
            
            # Call strategy-specific handler
            self.on_order_filled(order)
    
    def _on_order_canceled(self, event: Event) -> None:
        """Handle order canceled events."""
        order = event.data
        if order.get("strategy_id") == self.strategy_id:
            # Remove from pending orders
            order_id = order.get("order_id")
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
            
            # Call strategy-specific handler
            self.on_order_canceled(order)
    
    def _on_order_rejected(self, event: Event) -> None:
        """Handle order rejected events."""
        order = event.data
        if order.get("strategy_id") == self.strategy_id:
            # Remove from pending orders
            order_id = order.get("order_id")
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
            
            # Call strategy-specific handler
            self.on_order_rejected(order)
    
    def _on_position_update(self, event: Event) -> None:
        """Handle position update events."""
        position = event.data
        if position.get("strategy_id") == self.strategy_id:
            symbol = position.get("symbol")
            if symbol:
                self.positions[symbol] = position
            
            # Call strategy-specific handler
            self.on_position_update(position)
    
    # State management
    def start(self) -> None:
        """Start the strategy."""
        if self.state == StrategyState.INITIALIZED or self.state == StrategyState.PAUSED or self.state == StrategyState.STOPPED:
            self.state = StrategyState.RUNNING
            self.metrics.start_time = datetime.now()
            logger.info(f"Starting strategy: {self.name} ({self.strategy_id})")
            self.on_start()
    
    def pause(self) -> None:
        """Pause the strategy."""
        if self.state == StrategyState.RUNNING:
            self.state = StrategyState.PAUSED
            logger.info(f"Pausing strategy: {self.name} ({self.strategy_id})")
            self.on_pause()
    
    def resume(self) -> None:
        """Resume the strategy after pausing."""
        if self.state == StrategyState.PAUSED:
            self.state = StrategyState.RUNNING
            logger.info(f"Resuming strategy: {self.name} ({self.strategy_id})")
            self.on_resume()
    
    def stop(self) -> None:
        """Stop the strategy."""
        if self.state != StrategyState.STOPPED:
            self.state = StrategyState.STOPPED
            logger.info(f"Stopping strategy: {self.name} ({self.strategy_id})")
            self.on_stop()
    
    def is_running(self) -> bool:
        """Check if the strategy is currently running."""
        return self.state == StrategyState.RUNNING and self.enabled
    
    # Order management
    def create_order(
        self,
        symbol: str,
        quantity: float,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        price: float = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        stop_price: float = None,
        client_order_id: str = None,
        signal_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Create a new order.
        
        Returns:
            The order ID if successful, None otherwise.
        """
        # Create a unique client order ID if not provided
        if not client_order_id:
            client_order_id = f"{self.strategy_id}_{uuid.uuid4().hex[:8]}"
        
        # Add strategy tag prefix to ensure all orders are identifiable
        strategy_tag = self.get_tag_for_order(side.value.lower())
        
        # Combine with any additional tags
        order_tags = [strategy_tag]
        if tags:
            order_tags.extend(tags)
        
        # Create the order data
        order_data = {
            "symbol": symbol,
            "quantity": quantity,
            "side": side.value,
            "type": order_type.value,
            "time_in_force": time_in_force.value,
            "client_order_id": client_order_id,
            "strategy_id": self.strategy_id,
            "strategy_type": self.strategy_type.value,
            "strategy_priority": self.priority.value,
            "signal_id": signal_id,
            "tags": order_tags,
            "subaccount_id": self.subaccount_id
        }
        
        # Record the order in pending orders
        self.pending_orders[order_data["client_order_id"]] = order_data
        
        # If this order is tied to a signal, mark the signal as executed
        if signal_id and signal_id in self.active_signals:
            self.mark_signal_executed(signal_id, {
                'order_id': order_data["client_order_id"],
                'status': 'submitted',
                'timestamp': datetime.now()
            })
        
        logger.info(f"Strategy {self.name} created order: {order_data['client_order_id']} for {symbol} {side.value} {quantity}")
        
        return order_data["client_order_id"]
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Returns:
            True if the cancel request was sent, False otherwise.
        """
        if order_id not in self.pending_orders:
            logger.warning(f"Strategy {self.name} attempted to cancel non-existent order: {order_id}")
            return False
        
        # Publish order cancellation event
        self.event_bus.publish(Event(
            event_type=EventType.ORDER_CANCEL_REQUESTED,
            data={"order_id": order_id, "strategy_id": self.strategy_id},
            source=self.strategy_id
        ))
        
        logger.info(f"Strategy {self.name} requested cancellation of order: {order_id}")
        
        return True
    
    def cancel_all_orders(self) -> int:
        """
        Cancel all pending orders for this strategy.
        
        Returns:
            Number of orders for which cancellation was requested.
        """
        count = 0
        for order_id in list(self.pending_orders.keys()):
            if self.cancel_order(order_id):
                count += 1
        
        return count
    
    # Position management
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get the current position for a symbol."""
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if there is an open position for a symbol."""
        position = self.get_position(symbol)
        return position is not None and position.quantity != 0
    
    def position_value(self, symbol: str) -> float:
        """Get the current market value of a position."""
        position = self.get_position(symbol)
        if not position:
            return 0.0
        
        # Try to get the latest price from our cached data
        latest_data = self.last_data.get(symbol, {})
        price = latest_data.get("price", position.current_price or position.avg_price or 0.0)
        
        return position.quantity * price
    
    def close_position(self, symbol: str) -> Optional[str]:
        """
        Close an existing position.
        
        Returns:
            The order ID of the closing order if successful, None otherwise.
        """
        position = self.get_position(symbol)
        if not position or position.quantity == 0:
            logger.warning(f"Strategy {self.name} attempted to close non-existent position for {symbol}")
            return None
        
        # Determine the side for the closing order
        side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
        
        # Create an order to close the position
        return self.create_order(
            symbol=symbol,
            quantity=abs(position.quantity),
            side=side,
            order_type=OrderType.MARKET
        )
    
    def close_all_positions(self) -> Dict[str, Optional[str]]:
        """
        Close all existing positions for this strategy.
        
        Returns:
            A dictionary mapping symbols to order IDs of closing orders.
        """
        results = {}
        for symbol in list(self.positions.keys()):
            order_id = self.close_position(symbol)
            results[symbol] = order_id
        
        return results
    
    # Signal generation and abstract methods
    def generate_signal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate a trading signal based on the given data.
        This is the main entry point for strategy-specific logic.
        
        Args:
            data: Market data or event that triggered the signal generation
            
        Returns:
            A dictionary containing signal details if a signal is generated,
            None otherwise.
        """
        # First check if we should process this data
        if not self._should_process_data(data):
            return None
            
        # Default implementation delegates to abstract method
        signal = self.on_signal_generation(data)
        
        # If a signal was generated, record it and attach metadata
        if signal is not None:
            signal_id = self.record_signal(signal)
            signal['id'] = signal_id
        
        return signal
        
    def _should_process_data(self, data: Dict[str, Any]) -> bool:
        """
        Determine if this strategy should process the given data.
        
        Args:
            data: Market data or event
            
        Returns:
            True if the strategy should process this data, False otherwise
        """
        # Check if strategy is enabled and running
        if not self.enabled or self.state != StrategyState.RUNNING:
            return False
            
        # Check if the data is for a symbol we're monitoring
        symbol = data.get('symbol')
        if symbol and self.symbols and symbol not in self.symbols:
            return False
            
        # Check timeframe if applicable
        timeframe = data.get('timeframe')
        if timeframe and timeframe != self.timeframe:
            return False
            
        return True
    
    # Abstract methods to be implemented by subclasses
    @abstractmethod
    def on_signal_generation(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate a trading signal based on the given data.
        Must be implemented by subclasses.
        
        Args:
            data: Market data or event that triggered the signal generation
            
        Returns:
            A dictionary containing signal details if a signal is generated,
            None otherwise.
        """
        pass
    
    @abstractmethod
    def on_data(self, data: Dict[str, Any]) -> None:
        """
        Process new market data.
        Must be implemented by subclasses.
        
        Args:
            data: New market data
        """
        pass
    
    # Optional event handlers with default no-op implementations
    def on_start(self) -> None:
        """Called when the strategy is started."""
        pass
    
    def on_pause(self) -> None:
        """Called when the strategy is paused."""
        pass
    
    def on_resume(self) -> None:
        """Called when the strategy is resumed after pausing."""
        pass
    
    def on_stop(self) -> None:
        """Called when the strategy is stopped."""
        pass
    
    def on_quote(self, quote: Dict[str, Any]) -> None:
        """Process a new quote update."""
        pass
    
    def on_trade(self, trade: Dict[str, Any]) -> None:
        """Process a new trade update."""
        pass
    
    def on_bar(self, bar: Dict[str, Any]) -> None:
        """Process a new bar/candle update."""
        pass
    
    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """Called when an order from this strategy is filled."""
        pass
    
    def on_order_canceled(self, order: Dict[str, Any]) -> None:
        """Called when an order from this strategy is canceled."""
        pass
    
    def on_order_rejected(self, order: Dict[str, Any]) -> None:
        """Called when an order from this strategy is rejected."""
        pass
    
    def on_position_update(self, position: Dict[str, Any]) -> None:
        """Called when a position for this strategy is updated."""
        pass
    
    # Unified strategy handling methods
    def add_signal_tag(self, tag: Union[str, SignalTag]) -> None:
        """Add a tag to this strategy's signal tags."""
        if isinstance(tag, str):
            self.signal_tags.add(tag)
        else:
            self.signal_tags.add(tag.value)
    
    def remove_signal_tag(self, tag: Union[str, SignalTag]) -> None:
        """Remove a tag from this strategy's signal tags."""
        if isinstance(tag, str):
            self.signal_tags.discard(tag)
        else:
            self.signal_tags.discard(tag.value)
    
    def has_signal_tag(self, tag: Union[str, SignalTag]) -> bool:
        """Check if this strategy has a specific signal tag."""
        if isinstance(tag, str):
            return tag in self.signal_tags
        return tag.value in self.signal_tags
    
    def get_tag_for_order(self, order_type: str = None) -> str:
        """Generate a tag for an order from this strategy."""
        tag = f"{self.order_tag_prefix}_{self.strategy_id[:8]}"
        if order_type:
            tag += f"_{order_type}"
        return tag
    
    def record_signal(self, signal: Dict[str, Any]) -> str:
        """Record a signal generated by this strategy."""
        # Generate a unique ID for the signal
        signal_id = signal.get('id', str(uuid.uuid4()))
        
        # Add metadata
        signal['strategy_id'] = self.strategy_id
        signal['strategy_name'] = self.name
        signal['strategy_type'] = self.strategy_type.value
        signal['priority'] = self.priority.value
        signal['timestamp'] = signal.get('timestamp', datetime.now())
        signal['tags'] = list(self.signal_tags) + signal.get('tags', [])
        
        # Store the signal
        self.generated_signals.append(signal)
        self.active_signals[signal_id] = signal
        
        return signal_id
    
    def mark_signal_executed(self, signal_id: str, result: Dict[str, Any]) -> None:
        """Mark a signal as executed and record its performance."""
        if signal_id in self.active_signals:
            signal = self.active_signals[signal_id]
            signal['executed'] = True
            signal['execution_timestamp'] = datetime.now()
            signal['execution_result'] = result
            
            # Move from active to performance tracking
            self.signal_performance[signal_id] = signal
            del self.active_signals[signal_id]
    
    def record_conflict(self, conflict_data: Dict[str, Any]) -> None:
        """Record a conflict with another strategy."""
        conflict_data['timestamp'] = datetime.now()
        conflict_data['strategy_id'] = self.strategy_id
        self.conflicts.append(conflict_data)
    
    def record_conflict_resolution(self, resolution_data: Dict[str, Any]) -> None:
        """Record how a conflict was resolved."""
        resolution_data['timestamp'] = datetime.now()
        resolution_data['strategy_id'] = self.strategy_id
        self.conflict_resolutions.append(resolution_data)
    
    def get_conflict_resolution_preference(self) -> Dict[str, Any]:
        """Get this strategy's preferences for conflict resolution."""
        return {
            'mode': self.conflict_resolution.value,
            'priority': self.priority.value,
            'strategy_id': self.strategy_id,
            'strategy_type': self.strategy_type.value
        }
    
    # Utility methods
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary representation."""
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "description": self.description,
            "symbols": self.symbols,
            "asset_type": self.asset_type.value if self.asset_type else None,
            "timeframe": self.timeframe,
            "parameters": self.parameters,
            "risk_limits": self.risk_limits,
            "broker_id": self.broker_id,
            "enabled": self.enabled,
            "state": self.state.value,
            "strategy_type": self.strategy_type.value,
            "priority": self.priority.value,
            "signal_tags": list(self.signal_tags),
            "conflict_resolution": self.conflict_resolution.value,
            "subaccount_id": self.subaccount_id,
            "metrics": self.metrics.to_dict(),
            "positions": {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
            "pending_orders": len(self.pending_orders),
            "historical_orders": len(self.historical_orders),
            "generated_signals": len(self.generated_signals),
            "active_signals": len(self.active_signals),
            "conflicts": len(self.conflicts)
        }
