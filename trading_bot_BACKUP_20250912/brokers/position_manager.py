#!/usr/bin/env python3
"""
Position Manager

Manages positions across brokers by reconciling order fill events
with internal position state. Maintains accurate position tracking
even with partial fills and updates position P&L.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from decimal import Decimal

from trading_bot.core.events import OrderPartialFill, OrderFilled, OrderCancelled, SlippageMetric
from trading_bot.event_system.event_bus import EventBus


class Position:
    """Represents a trading position"""
    
    def __init__(
        self,
        symbol: str,
        quantity: float,
        avg_price: float,
        broker_id: str,
        entry_time: Optional[datetime] = None,
        trade_id: Optional[str] = None,
        asset_class: Optional[str] = None,
        position_id: Optional[str] = None
    ):
        """
        Initialize a position
        
        Args:
            symbol: Asset symbol
            quantity: Position quantity (positive for long, negative for short)
            avg_price: Average entry price
            broker_id: ID of the broker
            entry_time: Entry timestamp
            trade_id: ID of the associated trade
            asset_class: Asset class
            position_id: Optional position ID
        """
        self.symbol = symbol
        self.quantity = quantity
        self.avg_price = avg_price
        self.broker_id = broker_id
        self.entry_time = entry_time or datetime.now()
        self.trade_id = trade_id
        self.asset_class = asset_class
        self.position_id = position_id or f"{broker_id}_{symbol}_{self.entry_time.timestamp()}"
        
        # Track fills that contributed to this position
        self.fills = []
        
        # P&L tracking
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.current_price = avg_price
        self.last_update_time = self.entry_time
    
    def add_fill(self, fill_data: Dict[str, Any]):
        """
        Add a fill to the position
        
        Args:
            fill_data: Fill data
        """
        self.fills.append(fill_data)
    
    def update_unrealized_pnl(self, current_price: float):
        """
        Update unrealized P&L based on current price
        
        Args:
            current_price: Current price of the asset
        """
        self.current_price = current_price
        self.last_update_time = datetime.now()
        
        if self.quantity == 0:
            self.unrealized_pnl = 0.0
            return
        
        # Calculate P&L
        price_diff = current_price - self.avg_price
        if self.quantity < 0:  # Short position
            price_diff = -price_diff
        
        self.unrealized_pnl = price_diff * abs(self.quantity)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary"""
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'broker_id': self.broker_id,
            'entry_time': self.entry_time.isoformat(),
            'trade_id': self.trade_id,
            'asset_class': self.asset_class,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'current_price': self.current_price,
            'last_update_time': self.last_update_time.isoformat(),
            'fill_count': len(self.fills)
        }


class PositionManager:
    """
    Manages positions across brokers
    
    Reconciles order fill events with internal position state.
    Tracks position P&L and provides position information for trading decisions.
    """
    
    def __init__(self, event_bus: EventBus):
        """
        Initialize the position manager
        
        Args:
            event_bus: Event bus for event subscription and emission
        """
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        
        # Position tracking by broker and symbol
        self.positions: Dict[str, Dict[str, Position]] = {}  # broker_id -> symbol -> Position
        
        # Mapping of order IDs to position IDs for tracking order fill reconciliation
        self.order_position_map: Dict[str, str] = {}  # order_id -> position_id
        
        # Mapping of trade IDs to position IDs
        self.trade_position_map: Dict[str, str] = {}  # trade_id -> position_id
        
        # Set of closed position IDs
        self.closed_positions: Set[str] = set()
        
        # Subscribe to order events
        self._subscribe_to_events()
    
    def _subscribe_to_events(self):
        """Subscribe to order events"""
        self.event_bus.on(OrderPartialFill, self._on_partial_fill)
        self.event_bus.on(OrderFilled, self._on_full_fill)
        self.event_bus.on(OrderCancelled, self._on_order_cancelled)
        self.event_bus.on(SlippageMetric, self._on_slippage_metric)
    
    def _on_partial_fill(self, event: OrderPartialFill):
        """
        Handle partial fill event
        
        Args:
            event: OrderPartialFill event
        """
        try:
            self.logger.info(f"Processing partial fill: {event.order_id}, {event.filled_qty} @ {event.fill_price}")
            
            broker_id = event.broker
            symbol = event.symbol
            side = event.side
            filled_qty = event.filled_qty
            remaining_qty = event.remaining_qty
            fill_price = event.fill_price
            order_id = event.order_id
            
            # Determine quantity direction
            if side.lower() == 'buy':
                quantity = filled_qty
            else:  # sell
                quantity = -filled_qty
            
            # Create fill data
            fill_data = {
                'order_id': order_id,
                'fill_price': fill_price,
                'fill_qty': filled_qty,
                'remaining_qty': remaining_qty,
                'side': side,
                'timestamp': event.timestamp
            }
            
            # Check if order is mapped to a position
            position_id = self.order_position_map.get(order_id)
            position = None
            
            if position_id:
                # Find the position
                for broker_positions in self.positions.values():
                    for pos in broker_positions.values():
                        if pos.position_id == position_id:
                            position = pos
                            break
                    if position:
                        break
            
            if position:
                # Update existing position
                self._update_position_with_fill(position, quantity, fill_price, fill_data)
            else:
                # Create a new position and map it to the order
                position = self._create_position_from_fill(
                    broker_id=broker_id,
                    symbol=symbol,
                    quantity=quantity,
                    fill_price=fill_price,
                    order_id=order_id,
                    fill_data=fill_data
                )
                
                # Map the order to the position
                self.order_position_map[order_id] = position.position_id
            
            self.logger.debug(f"Updated position after partial fill: {position.position_id}, {position.quantity} @ {position.avg_price}")
            
        except Exception as e:
            self.logger.error(f"Error processing partial fill: {str(e)}")
    
    def _on_full_fill(self, event: OrderFilled):
        """
        Handle full fill event
        
        Args:
            event: OrderFilled event
        """
        try:
            self.logger.info(f"Processing full fill: {event.order_id}, {event.total_qty} @ {event.avg_fill_price}")
            
            broker_id = event.broker
            symbol = event.symbol
            side = event.side
            total_qty = event.total_qty
            avg_fill_price = event.avg_fill_price
            order_id = event.order_id
            trade_id = event.trade_id
            
            # Determine quantity direction
            if side.lower() == 'buy':
                quantity = total_qty
            else:  # sell
                quantity = -total_qty
            
            # Create fill data
            fill_data = {
                'order_id': order_id,
                'fill_price': avg_fill_price,
                'fill_qty': total_qty,
                'remaining_qty': 0,
                'side': side,
                'timestamp': event.timestamp,
                'is_final': True,
                'trade_id': trade_id
            }
            
            # Check if order is mapped to a position
            position_id = self.order_position_map.get(order_id)
            position = None
            
            if position_id:
                # Find the position
                for broker_positions in self.positions.values():
                    for pos in broker_positions.values():
                        if pos.position_id == position_id:
                            position = pos
                            break
                    if position:
                        break
            
            if position:
                # Update existing position
                self._update_position_with_fill(position, quantity, avg_fill_price, fill_data)
                
                # If trade ID is provided, map it to the position
                if trade_id:
                    self.trade_position_map[trade_id] = position.position_id
                    position.trade_id = trade_id
            else:
                # Create a new position and map it to the order
                position = self._create_position_from_fill(
                    broker_id=broker_id,
                    symbol=symbol,
                    quantity=quantity,
                    fill_price=avg_fill_price,
                    order_id=order_id,
                    fill_data=fill_data,
                    trade_id=trade_id
                )
                
                # Map the order and trade to the position
                self.order_position_map[order_id] = position.position_id
                if trade_id:
                    self.trade_position_map[trade_id] = position.position_id
            
            # Check if position is closed (quantity = 0)
            if position.quantity == 0:
                self._close_position(position)
            
            self.logger.debug(f"Updated position after full fill: {position.position_id}, {position.quantity} @ {position.avg_price}")
            
        except Exception as e:
            self.logger.error(f"Error processing full fill: {str(e)}")
    
    def _on_order_cancelled(self, event: OrderCancelled):
        """
        Handle order cancelled event
        
        Args:
            event: OrderCancelled event
        """
        try:
            self.logger.info(f"Processing order cancelled: {event.order_id}")
            
            order_id = event.order_id
            
            # Remove from order-position map
            if order_id in self.order_position_map:
                del self.order_position_map[order_id]
            
        except Exception as e:
            self.logger.error(f"Error processing order cancelled: {str(e)}")
    
    def _on_slippage_metric(self, event: SlippageMetric):
        """
        Handle slippage metric event
        
        Args:
            event: SlippageMetric event
        """
        try:
            self.logger.info(f"Processing slippage metric: {event.order_id}, {event.slippage_bps} bps")
            
            # Could track slippage statistics here
            # For now, just log it
            
        except Exception as e:
            self.logger.error(f"Error processing slippage metric: {str(e)}")
    
    def _update_position_with_fill(
        self,
        position: Position,
        fill_quantity: float,
        fill_price: float,
        fill_data: Dict[str, Any]
    ):
        """
        Update a position with a fill
        
        Args:
            position: Position to update
            fill_quantity: Fill quantity (signed)
            fill_price: Fill price
            fill_data: Additional fill data
        """
        # Save original position state
        original_quantity = position.quantity
        original_avg_price = position.avg_price
        
        # Calculate new position
        new_quantity = original_quantity + fill_quantity
        
        # Add fill data
        position.add_fill(fill_data)
        
        # If closing or reducing a position
        if (original_quantity > 0 and fill_quantity < 0) or (original_quantity < 0 and fill_quantity > 0):
            # Calculate realized P&L for the closed portion
            closed_quantity = min(abs(original_quantity), abs(fill_quantity))
            if original_quantity > 0:  # Long position
                realized_pnl = (fill_price - original_avg_price) * closed_quantity
            else:  # Short position
                realized_pnl = (original_avg_price - fill_price) * closed_quantity
            
            position.realized_pnl += realized_pnl
            
            # If completely closing
            if abs(fill_quantity) >= abs(original_quantity):
                position.quantity = 0
                position.avg_price = 0
                return
            
            # If reducing, keep original avg_price
            position.quantity = new_quantity
            return
        
        # If increasing position, recalculate average price
        if new_quantity != 0:
            position.avg_price = ((original_quantity * original_avg_price) + (fill_quantity * fill_price)) / new_quantity
        
        position.quantity = new_quantity
    
    def _create_position_from_fill(
        self,
        broker_id: str,
        symbol: str,
        quantity: float,
        fill_price: float,
        order_id: str,
        fill_data: Dict[str, Any],
        trade_id: Optional[str] = None,
        asset_class: Optional[str] = None
    ) -> Position:
        """
        Create a new position from a fill
        
        Args:
            broker_id: Broker ID
            symbol: Asset symbol
            quantity: Position quantity
            fill_price: Fill price
            order_id: Order ID
            fill_data: Fill data
            trade_id: Optional trade ID
            asset_class: Optional asset class
            
        Returns:
            Created position
        """
        # Ensure broker dictionary exists
        if broker_id not in self.positions:
            self.positions[broker_id] = {}
        
        # Check if we already have a position for this symbol
        if symbol in self.positions[broker_id]:
            # Get existing position
            position = self.positions[broker_id][symbol]
            
            # Update with fill
            self._update_position_with_fill(position, quantity, fill_price, fill_data)
            
            return position
        
        # Create new position
        position = Position(
            symbol=symbol,
            quantity=quantity,
            avg_price=fill_price,
            broker_id=broker_id,
            entry_time=fill_data.get('timestamp', datetime.now()),
            trade_id=trade_id,
            asset_class=asset_class
        )
        
        # Add fill data
        position.add_fill(fill_data)
        
        # Store position
        self.positions[broker_id][symbol] = position
        
        return position
    
    def _close_position(self, position: Position):
        """
        Mark a position as closed
        
        Args:
            position: Position to close
        """
        self.logger.info(f"Closing position: {position.position_id}")
        
        # Add to closed positions
        self.closed_positions.add(position.position_id)
        
        # Remove from active positions
        if position.broker_id in self.positions and position.symbol in self.positions[position.broker_id]:
            del self.positions[position.broker_id][position.symbol]
        
        # Emit event or call callback here for trade logging
        # This is where you would persist the completed trade record
    
    def get_position(self, broker_id: str, symbol: str) -> Optional[Position]:
        """
        Get a position by broker ID and symbol
        
        Args:
            broker_id: Broker ID
            symbol: Asset symbol
            
        Returns:
            Position or None if not found
        """
        if broker_id in self.positions and symbol in self.positions[broker_id]:
            return self.positions[broker_id][symbol]
        return None
    
    def get_positions_by_broker(self, broker_id: str) -> List[Position]:
        """
        Get all positions for a broker
        
        Args:
            broker_id: Broker ID
            
        Returns:
            List of positions
        """
        if broker_id in self.positions:
            return list(self.positions[broker_id].values())
        return []
    
    def get_all_positions(self) -> List[Position]:
        """
        Get all positions across all brokers
        
        Returns:
            List of positions
        """
        result = []
        for broker_positions in self.positions.values():
            result.extend(broker_positions.values())
        return result
    
    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """
        Get all positions for a symbol across all brokers
        
        Args:
            symbol: Asset symbol
            
        Returns:
            List of positions
        """
        result = []
        for broker_positions in self.positions.values():
            if symbol in broker_positions:
                result.append(broker_positions[symbol])
        return result
    
    def get_position_by_trade_id(self, trade_id: str) -> Optional[Position]:
        """
        Get position by trade ID
        
        Args:
            trade_id: Trade ID
            
        Returns:
            Position or None if not found
        """
        position_id = self.trade_position_map.get(trade_id)
        if not position_id:
            return None
        
        # Find position by ID
        for broker_positions in self.positions.values():
            for position in broker_positions.values():
                if position.position_id == position_id:
                    return position
        
        return None
    
    def update_position_price(self, broker_id: str, symbol: str, current_price: float):
        """
        Update a position's current price and unrealized P&L
        
        Args:
            broker_id: Broker ID
            symbol: Asset symbol
            current_price: Current asset price
        """
        position = self.get_position(broker_id, symbol)
        if position:
            position.update_unrealized_pnl(current_price)
    
    def update_all_prices(self, prices: Dict[str, float]):
        """
        Update prices for all positions
        
        Args:
            prices: Dictionary mapping symbols to prices
        """
        for broker_positions in self.positions.values():
            for symbol, position in broker_positions.items():
                if symbol in prices:
                    position.update_unrealized_pnl(prices[symbol])
    
    def reconcile_with_broker_positions(self, broker_id: str, broker_positions: List[Dict[str, Any]]):
        """
        Reconcile internal positions with broker-reported positions
        
        Args:
            broker_id: Broker ID
            broker_positions: List of positions from broker API
        """
        self.logger.info(f"Reconciling positions for broker {broker_id}")
        
        # Convert broker positions to a dictionary by symbol
        broker_pos_dict = {p['symbol']: p for p in broker_positions}
        
        # Get internal positions for this broker
        internal_positions = self.get_positions_by_broker(broker_id)
        internal_pos_dict = {p.symbol: p for p in internal_positions}
        
        # Find positions in broker but not in internal state
        for symbol, broker_pos in broker_pos_dict.items():
            if symbol not in internal_pos_dict:
                # Create a new position from broker data
                self.logger.warning(f"Found position in broker not in internal state: {symbol}")
                
                # Create position
                quantity = float(broker_pos['quantity'])
                if broker_pos.get('side') == 'short':
                    quantity = -quantity
                
                position = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=float(broker_pos['avg_price']),
                    broker_id=broker_id,
                    entry_time=datetime.now(),  # We don't know the real entry time
                    asset_class=broker_pos.get('asset_class')
                )
                
                # Store position
                if broker_id not in self.positions:
                    self.positions[broker_id] = {}
                self.positions[broker_id][symbol] = position
            else:
                # Check if quantities match
                internal_pos = internal_pos_dict[symbol]
                broker_qty = float(broker_pos['quantity'])
                if broker_pos.get('side') == 'short':
                    broker_qty = -broker_qty
                
                if abs(internal_pos.quantity - broker_qty) > 0.0001:  # Allow for small floating point differences
                    self.logger.warning(f"Position quantity mismatch for {symbol}: internal={internal_pos.quantity}, broker={broker_qty}")
                    
                    # Update internal position to match broker
                    internal_pos.quantity = broker_qty
        
        # Find positions in internal state but not in broker
        for symbol, internal_pos in internal_pos_dict.items():
            if symbol not in broker_pos_dict:
                self.logger.warning(f"Found position in internal state not in broker: {symbol}")
                
                # Close the position
                self._close_position(internal_pos)
