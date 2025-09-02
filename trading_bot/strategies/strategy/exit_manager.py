"""
Exit Strategy Manager

This module provides sophisticated exit strategy management for trading positions,
supporting fixed exits, dynamic exits like trailing stops, and advanced exit methods
that adapt to market conditions.

Features:
- Trailing stops with volatility adaptation
- Multi-stage exits with partial position closing
- Time-based and session-based exits
- Market regime-aware exit adaptation
- Integration with broker for order execution
"""

import logging
import threading
import time
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import json

# Import broker and position related components
from trading_bot.position.position_manager import PositionManager
from trading_bot.brokers.multi_broker_manager import MultiBrokerManager

# Import event system components
from trading_bot.event_system import EventBus, Event

logger = logging.getLogger(__name__)

class ExitType:
    """Exit strategy types."""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    TIME_EXIT = "time_exit"
    BREAKEVEN = "breakeven"
    SCALE_OUT = "scale_out"
    VOLATILITY_STOP = "volatility_stop"
    SESSION_EXIT = "session_exit"
    CHANDELIER_EXIT = "chandelier_exit"
    REGIME_BASED = "regime_based"
    PATTERN_BASED = "pattern_based"
    MANUAL = "manual"

class ExitStatus:
    """Exit strategy status."""
    ACTIVE = "active"
    TRIGGERED = "triggered"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    PENDING = "pending"
    FAILED = "failed"

class ExitStrategyManager:
    """
    Sophisticated exit strategy manager that handles all types of exits
    for trading positions including trailing stops, scaled exits, and
    volatility-adjusted strategies.
    """
    
    def __init__(self, 
                 position_manager: PositionManager,
                 broker_manager: MultiBrokerManager,
                 market_data_service=None,
                 event_bus: Optional[EventBus] = None):
        """
        Initialize the exit strategy manager.
        
        Args:
            position_manager: Manager for position tracking
            broker_manager: Manager for broker connections
            market_data_service: Optional service for market data access
            event_bus: Optional event bus for system events
        """
        # Core dependencies
        self.position_manager = position_manager
        self.broker_manager = broker_manager
        self.market_data_service = market_data_service
        self.event_bus = event_bus
        
        # Exit strategy storage
        self.trailing_stops: Dict[str, Dict[str, Any]] = {}  # position_id -> trailing settings
        self.take_profits: Dict[str, Dict[str, Any]] = {}    # position_id -> profit targets
        self.stop_losses: Dict[str, Dict[str, Any]] = {}     # position_id -> stop settings
        self.time_exits: Dict[str, Dict[str, Any]] = {}      # position_id -> time exit settings
        self.scale_outs: Dict[str, Dict[str, Any]] = {}      # position_id -> scale out levels
        
        # Exit strategy history
        self.exit_history: List[Dict[str, Any]] = []
        
        # Monitoring control
        self.monitoring_thread = None
        self.monitoring_active = False
        self.check_interval = 1.0  # seconds
        
        # Market conditions
        self.volatility_metrics: Dict[str, Dict[str, Any]] = {}  # symbol -> volatility data
        self.regime_states: Dict[str, str] = {}  # symbol -> current regime
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Register for events if event bus is provided
        if self.event_bus:
            self._register_event_handlers()
            
        logger.info("Exit Strategy Manager initialized")
    
    def _register_event_handlers(self):
        """Register for relevant system events."""
        try:
            # Register for market regime events
            self.event_bus.subscribe("MARKET_REGIME_CHANGED", self.handle_regime_change)
            
            # Register for volatility update events
            self.event_bus.subscribe("VOLATILITY_UPDATED", self.handle_volatility_update)
            
            # Register for position events
            self.event_bus.subscribe("POSITION_OPENED", self.handle_position_opened)
            self.event_bus.subscribe("POSITION_CLOSED", self.handle_position_closed)
            self.event_bus.subscribe("POSITION_UPDATED", self.handle_position_updated)
            
            logger.info("Registered event handlers for Exit Strategy Manager")
        except Exception as e:
            logger.error(f"Error registering event handlers: {str(e)}")
    
    def handle_regime_change(self, event: Event):
        """Handle market regime change events."""
        try:
            data = event.data
            symbol = data.get('symbol')
            regime = data.get('regime')
            
            if symbol and regime:
                self.regime_states[symbol] = regime
                logger.info(f"Market regime for {symbol} changed to {regime}")
                
                # Adapt exit strategies for all positions with this symbol
                for position_id, position in self.position_manager.internal_positions.items():
                    if position.get('symbol') == symbol:
                        self.adapt_to_market_regime(position_id, regime)
        except Exception as e:
            logger.error(f"Error handling regime change: {str(e)}")
    
    def handle_volatility_update(self, event: Event):
        """Handle volatility update events."""
        try:
            data = event.data
            symbol = data.get('symbol')
            volatility = data.get('volatility')
            
            if symbol and volatility is not None:
                self.volatility_metrics[symbol] = volatility
                logger.info(f"Volatility for {symbol} updated to {volatility}")
                
                # Adjust exit strategies for all positions with this symbol
                for position_id, position in self.position_manager.internal_positions.items():
                    if position.get('symbol') == symbol:
                        self.adjust_exits_for_volatility(position_id)
        except Exception as e:
            logger.error(f"Error handling volatility update: {str(e)}")
    
    def handle_position_opened(self, event: Event):
        """Handle position opened events."""
        try:
            data = event.data
            position_id = data.get('position_id')
            
            if position_id:
                logger.info(f"New position opened: {position_id}")
                
                # Add default exit strategies if configured
                self.apply_default_exits(position_id)
        except Exception as e:
            logger.error(f"Error handling position opened: {str(e)}")
    
    def handle_position_closed(self, event: Event):
        """Handle position closed events."""
        try:
            data = event.data
            position_id = data.get('position_id')
            
            if position_id:
                logger.info(f"Position closed: {position_id}")
                
                # Clean up exit strategies for this position
                self.remove_all_exits(position_id)
        except Exception as e:
            logger.error(f"Error handling position closed: {str(e)}")
    
    def handle_position_updated(self, event: Event):
        """Handle position updated events."""
        try:
            data = event.data
            position_id = data.get('position_id')
            
            if position_id:
                logger.info(f"Position updated: {position_id}")
                
                # May need to adjust exit strategies based on position changes
                position = self.position_manager.get_position(position_id)
                if position:
                    # Re-evaluate any dynamic exits
                    self._check_exit_conditions(position_id)
        except Exception as e:
            logger.error(f"Error handling position updated: {str(e)}")
    
    def apply_default_exits(self, position_id: str) -> bool:
        """
        Apply default exit strategies to a position.
        
        Args:
            position_id: Position ID
            
        Returns:
            bool: Success status
        """
        position = self.position_manager.get_position(position_id)
        if not position:
            logger.warning(f"Cannot apply default exits - position {position_id} not found")
            return False
            
        try:
            # Default stop loss (if entry_price is available)
            if 'entry_price' in position:
                entry_price = float(position['entry_price'])
                if position.get('direction') == 'long':
                    stop_price = entry_price * 0.98  # 2% stop loss
                else:
                    stop_price = entry_price * 1.02  # 2% stop loss
                
                self.add_stop_loss(position_id, price=stop_price)
            
            # Default take profit (1.5:1 reward-to-risk)
            if 'entry_price' in position and position_id in self.stop_losses:
                entry_price = float(position['entry_price'])
                stop_price = self.stop_losses[position_id]['price']
                risk = abs(entry_price - stop_price)
                
                if position.get('direction') == 'long':
                    take_profit = entry_price + (risk * 1.5)
                else:
                    take_profit = entry_price - (risk * 1.5)
                
                self.add_take_profit(position_id, price=take_profit)
            
            # Default trailing stop (after 1:1 reward-to-risk)
            if 'entry_price' in position and position_id in self.stop_losses:
                entry_price = float(position['entry_price'])
                stop_price = self.stop_losses[position_id]['price']
                risk = abs(entry_price - stop_price)
                
                if position.get('direction') == 'long':
                    activation = entry_price + risk  # 1:1 reward-to-risk
                else:
                    activation = entry_price - risk  # 1:1 reward-to-risk
                
                self.add_trailing_stop(
                    position_id, 
                    trail_percent=0.5,  # 0.5% trailing stop
                    activation_threshold=activation
                )
            
            logger.info(f"Applied default exit strategies to position {position_id}")
            return True
        except Exception as e:
            logger.error(f"Error applying default exits: {str(e)}")
            return False
    
    def remove_all_exits(self, position_id: str) -> bool:
        """
        Remove all exit strategies for a position.
        
        Args:
            position_id: Position ID
            
        Returns:
            bool: Success status
        """
        with self._lock:
            try:
                # Remove from all exit strategy dictionaries
                if position_id in self.trailing_stops:
                    del self.trailing_stops[position_id]
                
                if position_id in self.take_profits:
                    del self.take_profits[position_id]
                
                if position_id in self.stop_losses:
                    del self.stop_losses[position_id]
                
                if position_id in self.time_exits:
                    del self.time_exits[position_id]
                
                if position_id in self.scale_outs:
                    del self.scale_outs[position_id]
                
                logger.info(f"Removed all exit strategies for position {position_id}")
                return True
            except Exception as e:
                logger.error(f"Error removing exits: {str(e)}")
                return False
    
    def add_stop_loss(self, 
                     position_id: str, 
                     price: Optional[float] = None, 
                     pips: Optional[float] = None, 
                     percent: Optional[float] = None,
                     volatility_multiple: Optional[float] = None) -> bool:
        """
        Add a stop loss to a position.
        
        Args:
            position_id: Position ID
            price: Explicit stop price
            pips: Stop distance in pips
            percent: Stop distance as percentage of entry price
            volatility_multiple: Stop distance as multiple of volatility (e.g., ATR)
            
        Returns:
            bool: Success status
        """
        position = self.position_manager.get_position(position_id)
        if not position:
            logger.warning(f"Cannot add stop loss - position {position_id} not found")
            return False
            
        with self._lock:
            try:
                symbol = position.get('symbol')
                direction = position.get('direction', 'long')
                entry_price = float(position.get('entry_price', 0))
                
                # Calculate stop price based on provided parameters
                stop_price = None
                
                if price is not None:
                    # Explicit price
                    stop_price = price
                elif pips is not None:
                    # Distance in pips
                    pip_value = 0.0001 if 'USD' in symbol or 'EUR' in symbol else 0.01
                    if direction == 'long':
                        stop_price = entry_price - (pips * pip_value)
                    else:
                        stop_price = entry_price + (pips * pip_value)
                elif percent is not None:
                    # Distance as percentage
                    if direction == 'long':
                        stop_price = entry_price * (1 - percent/100)
                    else:
                        stop_price = entry_price * (1 + percent/100)
                elif volatility_multiple is not None and self.market_data_service:
                    # Distance as volatility multiple
                    try:
                        atr = self.market_data_service.get_indicator(symbol, 'ATR', period=14)
                        if direction == 'long':
                            stop_price = entry_price - (atr * volatility_multiple)
                        else:
                            stop_price = entry_price + (atr * volatility_multiple)
                    except Exception as e:
                        logger.error(f"Error calculating ATR-based stop: {str(e)}")
                        return False
                
                if stop_price is None:
                    logger.warning(f"Cannot add stop loss - no valid stop price parameters")
                    return False
                
                # Store stop loss
                self.stop_losses[position_id] = {
                    'price': stop_price,
                    'status': ExitStatus.ACTIVE,
                    'type': ExitType.STOP_LOSS,
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'order_id': None  # Will be set when order is placed
                }
                
                # Place order with broker if connected
                try:
                    # Determine if we should place an order with the broker
                    # This depends on broker capabilities and trading mode
                    broker_id = position.get('broker_id', 'unknown')
                    broker = self.broker_manager.brokers.get(broker_id)
                    
                    if broker and hasattr(broker, 'place_stop_order'):
                        order_result = broker.place_stop_order(
                            symbol=symbol,
                            quantity=position.get('quantity'),
                            stop_price=stop_price,
                            direction='sell' if direction == 'long' else 'buy'
                        )
                        
                        # Store order ID
                        if order_result and 'order_id' in order_result:
                            self.stop_losses[position_id]['order_id'] = order_result['order_id']
                            logger.info(f"Placed stop loss order for {position_id} at {stop_price}")
                except Exception as e:
                    logger.error(f"Error placing stop loss order: {str(e)}")
                
                # Log the action
                logger.info(f"Added stop loss for {position_id} at {stop_price}")
                return True
                
            except Exception as e:
                logger.error(f"Error adding stop loss: {str(e)}")
                return False
    
    def add_take_profit(self,
                      position_id: str,
                      price: Optional[float] = None,
                      percent: Optional[float] = None,
                      scale_levels: Optional[List[float]] = None) -> bool:
        """
        Add a take profit to a position.
        
        Args:
            position_id: Position ID
            price: Explicit take profit price
            percent: Profit target as percentage of entry price
            scale_levels: Multiple profit targets as percentages
            
        Returns:
            bool: Success status
        """
        position = self.position_manager.get_position(position_id)
        if not position:
            logger.warning(f"Cannot add take profit - position {position_id} not found")
            return False
            
        with self._lock:
            try:
                symbol = position.get('symbol')
                direction = position.get('direction', 'long')
                entry_price = float(position.get('entry_price', 0))
                
                # Calculate take profit price based on provided parameters
                take_profit_price = None
                
                if price is not None:
                    # Explicit price
                    take_profit_price = price
                elif percent is not None:
                    # Target as percentage
                    if direction == 'long':
                        take_profit_price = entry_price * (1 + percent/100)
                    else:
                        take_profit_price = entry_price * (1 - percent/100)
                elif scale_levels is not None:
                    # Handle scale levels separately
                    return self.add_scale_out_strategy(position_id, levels=scale_levels)
                
                if take_profit_price is None:
                    logger.warning(f"Cannot add take profit - no valid price parameters")
                    return False
                
                # Store take profit
                self.take_profits[position_id] = {
                    'price': take_profit_price,
                    'status': ExitStatus.ACTIVE,
                    'type': ExitType.TAKE_PROFIT,
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'order_id': None  # Will be set when order is placed
                }
                
                # Place order with broker if connected
                try:
                    # Determine if we should place an order with the broker
                    broker_id = position.get('broker_id', 'unknown')
                    broker = self.broker_manager.brokers.get(broker_id)
                    
                    if broker and hasattr(broker, 'place_limit_order'):
                        order_result = broker.place_limit_order(
                            symbol=symbol,
                            quantity=position.get('quantity'),
                            limit_price=take_profit_price,
                            direction='sell' if direction == 'long' else 'buy'
                        )
                        
                        # Store order ID
                        if order_result and 'order_id' in order_result:
                            self.take_profits[position_id]['order_id'] = order_result['order_id']
                            logger.info(f"Placed take profit order for {position_id} at {take_profit_price}")
                except Exception as e:
                    logger.error(f"Error placing take profit order: {str(e)}")
                
                # Log the action
                logger.info(f"Added take profit for {position_id} at {take_profit_price}")
                return True
                
            except Exception as e:
                logger.error(f"Error adding take profit: {str(e)}")
                return False
    
    def add_trailing_stop(self,
                         position_id: str,
                         trail_pips: Optional[float] = None,
                         trail_percent: Optional[float] = None,
                         activation_threshold: Optional[float] = None) -> bool:
        """
        Add a trailing stop to a position.
        
        Args:
            position_id: Position ID
            trail_pips: Distance to trail by in pips
            trail_percent: Distance to trail by as percentage
            activation_threshold: Price at which trailing begins
            
        Returns:
            bool: Success status
        """
        position = self.position_manager.get_position(position_id)
        if not position:
            logger.warning(f"Cannot add trailing stop - position {position_id} not found")
            return False
            
        with self._lock:
            try:
                symbol = position.get('symbol')
                direction = position.get('direction', 'long')
                entry_price = float(position.get('entry_price', 0))
                
                # Calculate trailing parameters
                trail_amount = None
                
                if trail_pips is not None:
                    # Convert pips to price movement
                    pip_value = 0.0001 if 'USD' in symbol or 'EUR' in symbol else 0.01
                    trail_amount = trail_pips * pip_value
                elif trail_percent is not None:
                    # Convert percentage to decimal
                    trail_amount = entry_price * (trail_percent / 100)
                else:
                    # Default to 1% trailing stop
                    trail_amount = entry_price * 0.01
                
                # Set activation threshold if not provided
                if activation_threshold is None:
                    # Default to immediate activation
                    activation_threshold = 0
                
                # Initialize reference price (highest/lowest seen so far)
                reference_price = None
                current_price = None
                
                # Try to get current price
                try:
                    if self.market_data_service:
                        current_price = self.market_data_service.get_price(symbol)
                    else:
                        # Try to get from broker
                        broker_id = position.get('broker_id', 'unknown')
                        broker = self.broker_manager.brokers.get(broker_id)
                        if broker:
                            quote = broker.get_quote(symbol)
                            current_price = quote.get('last', entry_price)
                except Exception as e:
                    logger.warning(f"Could not get current price: {str(e)}")
                    current_price = entry_price
                
                # Initialize reference price
                if direction == 'long':
                    reference_price = max(entry_price, current_price) if current_price else entry_price
                else:
                    reference_price = min(entry_price, current_price) if current_price else entry_price
                
                # Calculate initial stop price
                if direction == 'long':
                    stop_price = reference_price - trail_amount
                else:
                    stop_price = reference_price + trail_amount
                
                # Store trailing stop
                self.trailing_stops[position_id] = {
                    'trail_amount': trail_amount,
                    'reference_price': reference_price,
                    'stop_price': stop_price,
                    'activation_threshold': activation_threshold,
                    'status': ExitStatus.ACTIVE,
                    'type': ExitType.TRAILING_STOP,
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'order_id': None,  # Will be set when order is placed
                    'is_activated': activation_threshold == 0  # Activated immediately if no threshold
                }
                
                # Log the action
                logger.info(f"Added trailing stop for {position_id} with trail amount {trail_amount}")
                
                # Start monitoring if not already running
                if not self.monitoring_active:
                    self.start_monitoring()
                
                return True
                
            except Exception as e:
                logger.error(f"Error adding trailing stop: {str(e)}")
                return False
    
    def add_time_exit(self,
                     position_id: str,
                     duration: Optional[int] = None,
                     specific_time: Optional[str] = None,
                     market_session: Optional[str] = None) -> bool:
        """
        Add a time-based exit to a position.
        
        Args:
            position_id: Position ID
            duration: Number of minutes to hold position
            specific_time: Specific time to exit (ISO format)
            market_session: Market session to exit at (e.g., 'close')
            
        Returns:
            bool: Success status
        """
        position = self.position_manager.get_position(position_id)
        if not position:
            logger.warning(f"Cannot add time exit - position {position_id} not found")
            return False
            
        with self._lock:
            try:
                # Calculate exit time based on provided parameters
                exit_time = None
                reason = ""
                
                if duration is not None:
                    # Exit after duration
                    entry_time = datetime.fromisoformat(position.get('entry_date'))
                    exit_time = entry_time + timedelta(minutes=duration)
                    reason = f"Duration: {duration} minutes"
                elif specific_time is not None:
                    # Exit at specific time
                    exit_time = datetime.fromisoformat(specific_time)
                    reason = f"Specific time: {specific_time}"
                elif market_session is not None:
                    # Exit at market session
                    reason = f"Market session: {market_session}"
                    if market_session == 'close':
                        # Get market close time from broker
                        try:
                            broker_id = position.get('broker_id', 'unknown')
                            broker = self.broker_manager.brokers.get(broker_id)
                            if broker and hasattr(broker, 'get_market_close_time'):
                                exit_time = broker.get_market_close_time()
                            else:
                                # Default to 16:00 New York time
                                now = datetime.now()
                                exit_time = datetime.combine(now.date(), datetime.strptime('16:00', '%H:%M').time())
                        except Exception as e:
                            logger.error(f"Error getting market close time: {str(e)}")
                            return False
                    else:
                        logger.warning(f"Unsupported market session: {market_session}")
                        return False
                else:
                    # Default to end of trading day
                    now = datetime.now()
                    exit_time = datetime.combine(now.date(), datetime.strptime('16:00', '%H:%M').time())
                    reason = "Default: End of trading day"
                
                # Store time exit
                self.time_exits[position_id] = {
                    'exit_time': exit_time.isoformat() if exit_time else None,
                    'market_session': market_session,
                    'reason': reason,
                    'status': ExitStatus.ACTIVE,
                    'type': ExitType.TIME_EXIT,
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                }
                
                # Log the action
                exit_info = exit_time.isoformat() if exit_time else market_session
                logger.info(f"Added time exit for {position_id} at {exit_info}")
                
                # Start monitoring if not already running
                if not self.monitoring_active:
                    self.start_monitoring()
                
                return True
                
            except Exception as e:
                logger.error(f"Error adding time exit: {str(e)}")
                return False
    
    def add_scale_out_strategy(self,
                              position_id: str,
                              levels: List[float] = [25, 50, 75, 100],
                              prices: Optional[List[float]] = None,
                              trailing: bool = False) -> bool:
        """
        Add a scale-out strategy with multiple exit levels.
        
        Args:
            position_id: Position ID
            levels: Percentage levels for scaling out
            prices: Explicit prices for scaling out
            trailing: Whether to use trailing stops for each level
            
        Returns:
            bool: Success status
        """
        position = self.position_manager.get_position(position_id)
        if not position:
            logger.warning(f"Cannot add scale-out strategy - position {position_id} not found")
            return False
            
        with self._lock:
            try:
                symbol = position.get('symbol')
                direction = position.get('direction', 'long')
                entry_price = float(position.get('entry_price', 0))
                quantity = float(position.get('quantity', 0))
                
                # Calculate exit levels
                exit_levels = []
                
                if prices is not None:
                    # Use explicit prices
                    for i, price in enumerate(prices):
                        # Calculate level percentage
                        if i == len(prices) - 1:
                            level_pct = 100
                        else:
                            level_pct = levels[i] if i < len(levels) else (i + 1) * 25
                        
                        # Calculate quantity for this level
                        level_qty = (quantity * level_pct / 100) if i == 0 else \
                                   (quantity * (level_pct - levels[i-1]) / 100)
                        
                        exit_levels.append({
                            'price': price,
                            'quantity': level_qty,
                            'percentage': level_pct,
                            'executed': False
                        })
                else:
                    # Calculate prices based on percentage levels
                    # First try to use stop loss for relative R-multiples
                    stop_loss = None
                    if position_id in self.stop_losses:
                        stop_loss = self.stop_losses[position_id]['price']
                    
                    for i, level in enumerate(levels):
                        # Calculate quantity for this level
                        level_qty = (quantity * level / 100) if i == 0 else \
                                   (quantity * (level - levels[i-1]) / 100)
                        
                        # Calculate price for this level
                        level_price = None
                        
                        if stop_loss is not None:
                            # Calculate based on R-multiple
                            risk = abs(entry_price - stop_loss)
                            if direction == 'long':
                                r_multiple = i + 1  # 1R, 2R, 3R, etc.
                                level_price = entry_price + (risk * r_multiple)
                            else:
                                r_multiple = i + 1
                                level_price = entry_price - (risk * r_multiple)
                        else:
                            # Calculate based on percentage of entry price
                            if direction == 'long':
                                level_price = entry_price * (1 + (level / 100))
                            else:
                                level_price = entry_price * (1 - (level / 100))
                        
                        exit_levels.append({
                            'price': level_price,
                            'quantity': level_qty,
                            'percentage': level,
                            'executed': False
                        })
                
                # Store scale-out strategy
                self.scale_outs[position_id] = {
                    'levels': exit_levels,
                    'trailing': trailing,
                    'status': ExitStatus.ACTIVE,
                    'type': ExitType.SCALE_OUT,
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'executed_levels': []
                }
                
                # If trailing, create trailing stops for each level
                if trailing:
                    for i, level in enumerate(exit_levels):
                        # Only create trailing stops for levels after first (first is take profit)
                        if i > 0:
                            trail_percent = 0.5  # Default 0.5% trail
                            activation = level['price']
                            
                            # Create unique "sub-position" ID for this level
                            sub_position_id = f"{position_id}-level-{i+1}"
                            
                            # Add trailing stop for this level
                            self.trailing_stops[sub_position_id] = {
                                'trail_amount': level['price'] * (trail_percent / 100),
                                'reference_price': level['price'],
                                'stop_price': level['price'] * (1 - trail_percent / 100) if direction == 'long' else \
                                              level['price'] * (1 + trail_percent / 100),
                                'activation_threshold': activation,
                                'status': ExitStatus.ACTIVE,
                                'type': ExitType.TRAILING_STOP,
                                'parent_position_id': position_id,
                                'level_index': i,
                                'created_at': datetime.now().isoformat(),
                                'updated_at': datetime.now().isoformat(),
                                'is_activated': False
                            }
                
                # Log the action
                level_str = ", ".join([f"{l['percentage']}% at {l['price']}" for l in exit_levels])
                logger.info(f"Added scale-out strategy for {position_id} with levels: {level_str}")
                
                # Start monitoring if not already running
                if not self.monitoring_active:
                    self.start_monitoring()
                
                return True
                
            except Exception as e:
                logger.error(f"Error adding scale-out strategy: {str(e)}")
                return False
    
    def update_stop_loss(self, position_id: str, new_price: float) -> bool:
        """
        Update an existing stop loss.
        
        Args:
            position_id: Position ID
            new_price: New stop price
            
        Returns:
            bool: Success status
        """
        if position_id not in self.stop_losses:
            logger.warning(f"Cannot update stop loss - no stop loss found for {position_id}")
            return False
            
        with self._lock:
            try:
                # Update stop loss price
                old_price = self.stop_losses[position_id]['price']
                self.stop_losses[position_id]['price'] = new_price
                self.stop_losses[position_id]['updated_at'] = datetime.now().isoformat()
                
                # Update broker order if applicable
                order_id = self.stop_losses[position_id].get('order_id')
                if order_id:
                    try:
                        position = self.position_manager.get_position(position_id)
                        if position:
                            broker_id = position.get('broker_id', 'unknown')
                            broker = self.broker_manager.brokers.get(broker_id)
                            
                            if broker and hasattr(broker, 'update_order'):
                                broker.update_order(order_id, new_price=new_price)
                                logger.info(f"Updated broker stop order for {position_id} to {new_price}")
                    except Exception as e:
                        logger.error(f"Error updating broker stop order: {str(e)}")
                
                logger.info(f"Updated stop loss for {position_id} from {old_price} to {new_price}")
                return True
                
            except Exception as e:
                logger.error(f"Error updating stop loss: {str(e)}")
                return False
    
    def move_stop_to_breakeven(self, position_id: str, buffer_percent: float = 0.1) -> bool:
        """
        Move stop loss to breakeven (entry price) with an optional buffer.
        
        Args:
            position_id: Position ID
            buffer_percent: Buffer percentage below/above entry (default 0.1%)
            
        Returns:
            bool: Success status
        """
        if position_id not in self.stop_losses:
            logger.warning(f"Cannot move stop to breakeven - no stop loss found for {position_id}")
            return False
            
        position = self.position_manager.get_position(position_id)
        if not position:
            logger.warning(f"Cannot move stop to breakeven - position {position_id} not found")
            return False
            
        with self._lock:
            try:
                # Get entry price
                entry_price = float(position.get('entry_price', 0))
                direction = position.get('direction', 'long')
                
                # Calculate breakeven price with buffer
                if direction == 'long':
                    breakeven = entry_price * (1 - buffer_percent/100)  # Slightly below entry
                else:
                    breakeven = entry_price * (1 + buffer_percent/100)  # Slightly above entry
                
                # Update stop loss
                return self.update_stop_loss(position_id, breakeven)
                
            except Exception as e:
                logger.error(f"Error moving stop to breakeven: {str(e)}")
                return False
    
    def start_monitoring(self) -> bool:
        """
        Start the monitoring thread for exit strategies.
        
        Returns:
            bool: Success status
        """
        if self.monitoring_active:
            logger.info("Monitoring already active")
            return True
            
        try:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="ExitStrategyMonitor",
                daemon=True
            )
            self.monitoring_thread.start()
            logger.info("Started exit strategy monitoring thread")
            return True
        except Exception as e:
            self.monitoring_active = False
            logger.error(f"Error starting monitoring thread: {str(e)}")
            return False
    
    def stop_monitoring(self) -> bool:
        """
        Stop the monitoring thread.
        
        Returns:
            bool: Success status
        """
        if not self.monitoring_active:
            logger.info("Monitoring not active")
            return True
            
        try:
            self.monitoring_active = False
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                # Thread will terminate on next loop iteration
                # Wait for it to finish (timeout after 5 seconds)
                self.monitoring_thread.join(timeout=5.0)
                
            logger.info("Stopped exit strategy monitoring thread")
            return True
        except Exception as e:
            logger.error(f"Error stopping monitoring thread: {str(e)}")
            return False
    
    def _monitoring_loop(self) -> None:
        """
        Main monitoring loop that checks exit conditions.
        Runs in a separate thread.
        """
        logger.info("Exit strategy monitoring loop started")
        
        while self.monitoring_active:
            try:
                # Get all active positions
                positions = self.position_manager.get_all_positions()
                position_ids = list(positions.keys()) if isinstance(positions, dict) else \
                               [p.get('position_id') for p in positions]
                
                # Check exit conditions for each position
                for position_id in position_ids:
                    try:
                        self._check_exit_conditions(position_id)
                    except Exception as e:
                        logger.error(f"Error checking exit conditions for {position_id}: {str(e)}")
                
                # Sleep for check interval
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5.0)  # Longer sleep on error
        
        logger.info("Exit strategy monitoring loop terminated")
    
    def _check_exit_conditions(self, position_id: str) -> None:
        """
        Check all exit conditions for a position.
        
        Args:
            position_id: Position ID to check
        """
        # Skip if position doesn't exist
        position = self.position_manager.get_position(position_id)
        if not position:
            return
            
        with self._lock:
            try:
                symbol = position.get('symbol')
                direction = position.get('direction', 'long')
                
                # Get current price
                current_price = None
                try:
                    if self.market_data_service:
                        current_price = self.market_data_service.get_price(symbol)
                    else:
                        # Try to get from broker
                        broker_id = position.get('broker_id', 'unknown')
                        broker = self.broker_manager.brokers.get(broker_id)
                        if broker:
                            quote = broker.get_quote(symbol)
                            current_price = quote.get('last')
                except Exception as e:
                    logger.warning(f"Could not get current price for {symbol}: {str(e)}")
                    return  # Can't check without price
                
                if current_price is None:
                    return  # No price available
                
                # Check stop loss
                if position_id in self.stop_losses and self.stop_losses[position_id]['status'] == ExitStatus.ACTIVE:
                    stop_price = self.stop_losses[position_id]['price']
                    
                    # Check if stop is hit
                    if (direction == 'long' and current_price <= stop_price) or \
                       (direction == 'short' and current_price >= stop_price):
                        # Execute stop loss
                        logger.info(f"Stop loss triggered for {position_id} at {current_price}")
                        self._execute_exit(position_id, ExitType.STOP_LOSS, current_price)
                        return  # Exit processed, no need to check others
                
                # Check take profit
                if position_id in self.take_profits and self.take_profits[position_id]['status'] == ExitStatus.ACTIVE:
                    take_profit_price = self.take_profits[position_id]['price']
                    
                    # Check if take profit is hit
                    if (direction == 'long' and current_price >= take_profit_price) or \
                       (direction == 'short' and current_price <= take_profit_price):
                        # Execute take profit
                        logger.info(f"Take profit triggered for {position_id} at {current_price}")
                        self._execute_exit(position_id, ExitType.TAKE_PROFIT, current_price)
                        return  # Exit processed, no need to check others
                
                # Check trailing stop
                if position_id in self.trailing_stops and self.trailing_stops[position_id]['status'] == ExitStatus.ACTIVE:
                    trailing = self.trailing_stops[position_id]
                    
                    # Check if not yet activated
                    if not trailing.get('is_activated', False):
                        # Check if price has reached activation threshold
                        activation_threshold = trailing.get('activation_threshold', 0)
                        if activation_threshold == 0 or \
                           (direction == 'long' and current_price >= activation_threshold) or \
                           (direction == 'short' and current_price <= activation_threshold):
                            # Activate trailing stop
                            trailing['is_activated'] = True
                            logger.info(f"Activated trailing stop for {position_id} at {current_price}")
                    
                    # If activated, check if stop is hit or update reference price
                    if trailing.get('is_activated', False):
                        # Check if stop is hit
                        stop_price = trailing['stop_price']
                        if (direction == 'long' and current_price <= stop_price) or \
                           (direction == 'short' and current_price >= stop_price):
                            # Execute trailing stop
                            logger.info(f"Trailing stop triggered for {position_id} at {current_price}")
                            self._execute_exit(position_id, ExitType.TRAILING_STOP, current_price)
                            return  # Exit processed, no need to check others
                        
                        # Update trail reference and stop price if price has moved favorably
                        reference_price = trailing['reference_price']
                        trail_amount = trailing['trail_amount']
                        
                        if (direction == 'long' and current_price > reference_price) or \
                           (direction == 'short' and current_price < reference_price):
                            # Update reference price
                            trailing['reference_price'] = current_price
                            
                            # Calculate new stop price
                            if direction == 'long':
                                trailing['stop_price'] = current_price - trail_amount
                            else:
                                trailing['stop_price'] = current_price + trail_amount
                            
                            trailing['updated_at'] = datetime.now().isoformat()
                            logger.debug(f"Updated trailing stop for {position_id} to {trailing['stop_price']}")
                
                # Check time exit
                if position_id in self.time_exits and self.time_exits[position_id]['status'] == ExitStatus.ACTIVE:
                    time_exit = self.time_exits[position_id]
                    exit_time_str = time_exit.get('exit_time')
                    
                    if exit_time_str:
                        exit_time = datetime.fromisoformat(exit_time_str)
                        now = datetime.now()
                        
                        if now >= exit_time:
                            # Execute time exit
                            logger.info(f"Time exit triggered for {position_id} at {now.isoformat()}")
                            self._execute_exit(position_id, ExitType.TIME_EXIT, current_price)
                            return  # Exit processed, no need to check others
                
                # Check scale out strategy
                if position_id in self.scale_outs and self.scale_outs[position_id]['status'] == ExitStatus.ACTIVE:
                    scale_out = self.scale_outs[position_id]
                    levels = scale_out.get('levels', [])
                    
                    for i, level in enumerate(levels):
                        if not level.get('executed', False):
                            level_price = level.get('price')
                            
                            # Check if level is hit
                            if (direction == 'long' and current_price >= level_price) or \
                               (direction == 'short' and current_price <= level_price):
                                # Execute scale out for this level
                                logger.info(f"Scale-out level {i+1} triggered for {position_id} at {current_price}")
                                
                                # Execute partial exit
                                quantity = level.get('quantity')
                                self._execute_exit(position_id, ExitType.SCALE_OUT, current_price, quantity=quantity)
                                
                                # Mark level as executed
                                level['executed'] = True
                                level['executed_at'] = datetime.now().isoformat()
                                level['executed_price'] = current_price
                                
                                # Add to executed levels
                                executed_level = level.copy()
                                executed_level['level_index'] = i
                                scale_out['executed_levels'].append(executed_level)
                                
                                # Update scale out status
                                scale_out['updated_at'] = datetime.now().isoformat()
                                
                                # Check if last level
                                if i == len(levels) - 1:
                                    scale_out['status'] = ExitStatus.EXECUTED
                                
                                # Don't check further levels in this iteration
                                break
                
            except Exception as e:
                logger.error(f"Error in _check_exit_conditions for {position_id}: {str(e)}")
    
    def _execute_exit(self, position_id: str, exit_type: str, price: float, quantity: Optional[float] = None) -> bool:
        """
        Execute an exit strategy.
        
        Args:
            position_id: Position ID
            exit_type: Type of exit (from ExitType)
            price: Current price
            quantity: Optional quantity for partial exits
            
        Returns:
            bool: Success status
        """
        try:
            position = self.position_manager.get_position(position_id)
            if not position:
                logger.warning(f"Cannot execute exit - position {position_id} not found")
                return False
            
            # Determine exit quantity
            position_quantity = float(position.get('quantity', 0))
            exit_quantity = quantity if quantity is not None else position_quantity
            
            # Check if valid quantity
            if exit_quantity <= 0 or exit_quantity > position_quantity:
                logger.warning(f"Invalid exit quantity {exit_quantity} for position {position_id}")
                return False
            
            # Determine if this is a full or partial exit
            is_partial = exit_quantity < position_quantity
            
            # Execute exit with broker
            symbol = position.get('symbol')
            direction = position.get('direction', 'long')
            broker_id = position.get('broker_id', 'unknown')
            broker = self.broker_manager.brokers.get(broker_id)
            
            exit_result = None
            if broker:
                try:
                    # Determine exit order parameters
                    exit_side = 'sell' if direction == 'long' else 'buy'
                    
                    # Place market order to exit
                    exit_result = broker.place_equity_order(
                        symbol=symbol,
                        quantity=exit_quantity,
                        order_type='market',
                        side=exit_side,
                        time_in_force='day',
                        position_id=position_id
                    )
                    
                    if exit_result:
                        logger.info(f"Placed exit order for {position_id}: {exit_type} at {price}")
                except Exception as e:
                    logger.error(f"Error placing exit order: {str(e)}")
            
            # Update exit strategy status
            self._update_exit_status(position_id, exit_type, ExitStatus.EXECUTED, price, exit_quantity)
            
            # Add to exit history
            exit_record = {
                'position_id': position_id,
                'exit_type': exit_type,
                'price': price,
                'quantity': exit_quantity,
                'is_partial': is_partial,
                'timestamp': datetime.now().isoformat(),
                'broker_id': broker_id,
                'order_id': exit_result.get('order_id') if exit_result else None
            }
            self.exit_history.append(exit_record)
            
            # Fire exit event if event bus is available
            if self.event_bus:
                event_data = {
                    'position_id': position_id,
                    'exit_type': exit_type,
                    'price': price,
                    'quantity': exit_quantity,
                    'is_partial': is_partial
                }
                self.event_bus.emit("POSITION_EXIT", event_data)
            
            # If this is a full exit, clean up all exit strategies
            if not is_partial:
                self.remove_all_exits(position_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing exit: {str(e)}")
            return False
    
    def _update_exit_status(self, position_id: str, exit_type: str, new_status: str, price: float, quantity: float) -> None:
        """
        Update the status of an exit strategy.
        
        Args:
            position_id: Position ID
            exit_type: Type of exit
            new_status: New status
            price: Price at which status was updated
            quantity: Quantity being exited
        """
        with self._lock:
            try:
                # Update status based on exit type
                if exit_type == ExitType.STOP_LOSS and position_id in self.stop_losses:
                    self.stop_losses[position_id]['status'] = new_status
                    self.stop_losses[position_id]['updated_at'] = datetime.now().isoformat()
                    self.stop_losses[position_id]['exit_price'] = price
                    self.stop_losses[position_id]['exit_quantity'] = quantity
                
                elif exit_type == ExitType.TAKE_PROFIT and position_id in self.take_profits:
                    self.take_profits[position_id]['status'] = new_status
                    self.take_profits[position_id]['updated_at'] = datetime.now().isoformat()
                    self.take_profits[position_id]['exit_price'] = price
                    self.take_profits[position_id]['exit_quantity'] = quantity
                
                elif exit_type == ExitType.TRAILING_STOP and position_id in self.trailing_stops:
                    self.trailing_stops[position_id]['status'] = new_status
                    self.trailing_stops[position_id]['updated_at'] = datetime.now().isoformat()
                    self.trailing_stops[position_id]['exit_price'] = price
                    self.trailing_stops[position_id]['exit_quantity'] = quantity
                
                elif exit_type == ExitType.TIME_EXIT and position_id in self.time_exits:
                    self.time_exits[position_id]['status'] = new_status
                    self.time_exits[position_id]['updated_at'] = datetime.now().isoformat()
                    self.time_exits[position_id]['exit_price'] = price
                    self.time_exits[position_id]['exit_quantity'] = quantity
            except Exception as e:
                logger.error(f"Error updating exit status: {str(e)}")
    
    def adapt_to_market_regime(self, position_id: str, regime: str) -> bool:
        """
        Adapt exit strategies based on detected market regime.
        
        Args:
            position_id: Position ID
            regime: Market regime ('trending', 'volatile', 'ranging', etc.)
            
        Returns:
            bool: Success status
        """
        position = self.position_manager.get_position(position_id)
        if not position:
            logger.warning(f"Cannot adapt to market regime - position {position_id} not found")
            return False
        
        with self._lock:
            try:
                symbol = position.get('symbol')
                direction = position.get('direction', 'long')
                logger.info(f"Adapting exit strategies for {position_id} to {regime} regime")
                
                # Store the regime in our state
                self.regime_states[symbol] = regime
                
                # Apply regime-specific adaptations
                if regime == 'trending':
                    return self._adjust_to_trending(position_id)
                elif regime == 'volatile':
                    return self._adjust_to_volatile(position_id)
                elif regime == 'ranging':
                    return self._adjust_to_ranging(position_id)
                elif regime == 'choppy':
                    return self._adjust_to_choppy(position_id)
                else:
                    logger.warning(f"Unknown market regime: {regime}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error adapting to market regime: {str(e)}")
                return False
    
    def _adjust_to_trending(self, position_id: str) -> bool:
        """
        Adjust exit strategies for trending markets.
        In trending markets, we want to:
        - Use wider trailing stops to capture the trend
        - Possibly remove fixed take profits
        - Use chandelier exits if available
        
        Args:
            position_id: Position ID
            
        Returns:
            bool: Success status
        """
        try:
            position = self.position_manager.get_position(position_id)
            if not position:
                return False
                
            symbol = position.get('symbol')
            direction = position.get('direction', 'long')
            current_price = None
            
            # Try to get current price
            try:
                if self.market_data_service:
                    current_price = self.market_data_service.get_price(symbol)
                else:
                    broker_id = position.get('broker_id', 'unknown')
                    broker = self.broker_manager.brokers.get(broker_id)
                    if broker:
                        quote = broker.get_quote(symbol)
                        current_price = quote.get('last')
            except Exception:
                pass
            
            # Adjust trailing stop if exists
            if position_id in self.trailing_stops:
                trail = self.trailing_stops[position_id]
                entry_price = float(position.get('entry_price', 0))
                
                # Widen trailing stop for trending market
                # Use 3x normal trailing amount
                trail_amount = trail.get('trail_amount', entry_price * 0.01)  # Default 1%
                new_trail_amount = trail_amount * 3.0  # Triple for trending
                
                # Update trail amount
                trail['trail_amount'] = new_trail_amount
                
                # Recalculate stop price
                if current_price is not None and trail.get('is_activated', False):
                    if direction == 'long':
                        trail['stop_price'] = current_price - new_trail_amount
                    else:
                        trail['stop_price'] = current_price + new_trail_amount
                
                logger.info(f"Adjusted trailing stop for trending market: {position_id}, new trail amount: {new_trail_amount}")
            
            # Consider removing fixed take profits in strong trend
            # (This is optional and could be conditionally applied)
            if position_id in self.take_profits and self.take_profits[position_id]['status'] == ExitStatus.ACTIVE:
                # Instead of removing, we could move it further to capture more of the trend
                take_profit = self.take_profits[position_id]
                current_tp = take_profit.get('price')
                entry_price = float(position.get('entry_price', 0))
                
                # Calculate a more distant take profit (e.g., 1.5x current distance from entry)
                if current_tp and entry_price:
                    current_distance = abs(current_tp - entry_price)
                    new_distance = current_distance * 1.5
                    
                    if direction == 'long':
                        new_tp = entry_price + new_distance
                    else:
                        new_tp = entry_price - new_distance
                    
                    # Update take profit
                    take_profit['price'] = new_tp
                    take_profit['updated_at'] = datetime.now().isoformat()
                    
                    logger.info(f"Adjusted take profit for trending market: {position_id}, new price: {new_tp}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adjusting to trending market: {str(e)}")
            return False
    
    def _adjust_to_volatile(self, position_id: str) -> bool:
        """
        Adjust exit strategies for volatile markets.
        In volatile markets, we want to:
        - Use tighter stops to reduce risk
        - Implement faster profit-taking and breakeven strategies
        - Possibly use scaled exits
        
        Args:
            position_id: Position ID
            
        Returns:
            bool: Success status
        """
        try:
            position = self.position_manager.get_position(position_id)
            if not position:
                return False
                
            symbol = position.get('symbol')
            direction = position.get('direction', 'long')
            entry_price = float(position.get('entry_price', 0))
            
            # Tighten trailing stop if exists
            if position_id in self.trailing_stops:
                trail = self.trailing_stops[position_id]
                
                # Tighten trailing stop for volatile market
                # Use 0.75x normal trailing amount
                trail_amount = trail.get('trail_amount', entry_price * 0.01)  # Default 1%
                new_trail_amount = trail_amount * 0.75  # Tighter for volatile
                
                # Update trail amount
                trail['trail_amount'] = new_trail_amount
                
                # Recalculate stop price if activated
                reference_price = trail.get('reference_price')
                if reference_price is not None and trail.get('is_activated', False):
                    if direction == 'long':
                        trail['stop_price'] = reference_price - new_trail_amount
                    else:
                        trail['stop_price'] = reference_price + new_trail_amount
                
                logger.info(f"Adjusted trailing stop for volatile market: {position_id}, new trail amount: {new_trail_amount}")
            
            # Consider moving take profit closer in volatile markets
            if position_id in self.take_profits and self.take_profits[position_id]['status'] == ExitStatus.ACTIVE:
                take_profit = self.take_profits[position_id]
                current_tp = take_profit.get('price')
                
                # Calculate a closer take profit (e.g., 0.75x current distance from entry)
                if current_tp:
                    current_distance = abs(current_tp - entry_price)
                    new_distance = current_distance * 0.75
                    
                    if direction == 'long':
                        new_tp = entry_price + new_distance
                    else:
                        new_tp = entry_price - new_distance
                    
                    # Update take profit
                    take_profit['price'] = new_tp
                    take_profit['updated_at'] = datetime.now().isoformat()
                    
                    logger.info(f"Adjusted take profit for volatile market: {position_id}, new price: {new_tp}")
            
            # Consider implementing or modifying scale out strategy
            if position_id not in self.scale_outs:
                # Create a new scale-out strategy for volatile markets
                # Take profits at 25%, 50%, 75%, and 100% of position
                # at progressively further price levels (0.5R, 1R, 1.5R, 2R)
                if position_id in self.stop_losses:
                    stop_price = self.stop_losses[position_id]['price']
                    risk = abs(entry_price - stop_price)
                    
                    # Early partial exits for volatile markets
                    self.add_scale_out_strategy(
                        position_id,
                        levels=[25, 50, 75, 100],
                        prices=[
                            entry_price + (risk * 0.5) if direction == 'long' else entry_price - (risk * 0.5),
                            entry_price + (risk * 1.0) if direction == 'long' else entry_price - (risk * 1.0),
                            entry_price + (risk * 1.5) if direction == 'long' else entry_price - (risk * 1.5),
                            entry_price + (risk * 2.0) if direction == 'long' else entry_price - (risk * 2.0)
                        ],
                        trailing=True  # Use trailing stops for later exits
                    )
                    logger.info(f"Added scale-out strategy for volatile market: {position_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adjusting to volatile market: {str(e)}")
            return False
    
    def _adjust_to_ranging(self, position_id: str) -> bool:
        """
        Adjust exit strategies for ranging markets.
        In ranging markets, we want to:
        - Use fixed take profits rather than trailing stops
        - Place tighter stops near range boundaries
        - Consider time-based exits
        
        Args:
            position_id: Position ID
            
        Returns:
            bool: Success status
        """
        try:
            position = self.position_manager.get_position(position_id)
            if not position:
                return False
                
            symbol = position.get('symbol')
            direction = position.get('direction', 'long')
            entry_price = float(position.get('entry_price', 0))
            
            # Favor fixed take profits in ranging markets
            # If we don't have a take profit, add one
            if position_id not in self.take_profits and position_id in self.stop_losses:
                stop_price = self.stop_losses[position_id]['price']
                risk = abs(entry_price - stop_price)
                
                # Use a 1:1 reward-to-risk ratio for ranging markets
                if direction == 'long':
                    take_profit_price = entry_price + risk
                else:
                    take_profit_price = entry_price - risk
                
                # Add take profit
                self.add_take_profit(position_id, price=take_profit_price)
                logger.info(f"Added take profit for ranging market: {position_id}, price: {take_profit_price}")
            
            # Consider removing or adjusting trailing stops in ranging markets
            if position_id in self.trailing_stops and self.trailing_stops[position_id]['status'] == ExitStatus.ACTIVE:
                # Option 1: Remove trailing stop entirely
                # del self.trailing_stops[position_id]
                
                # Option 2: Make trailing stop tighter
                trail = self.trailing_stops[position_id]
                trail_amount = trail.get('trail_amount', entry_price * 0.01)  # Default 1%
                new_trail_amount = trail_amount * 0.5  # Half for ranging markets
                
                # Update trail amount
                trail['trail_amount'] = new_trail_amount
                
                # Recalculate stop price if activated
                reference_price = trail.get('reference_price')
                if reference_price is not None and trail.get('is_activated', False):
                    if direction == 'long':
                        trail['stop_price'] = reference_price - new_trail_amount
                    else:
                        trail['stop_price'] = reference_price + new_trail_amount
                
                logger.info(f"Adjusted trailing stop for ranging market: {position_id}, new trail amount: {new_trail_amount}")
            
            # Consider adding a time-based exit for ranging markets
            if position_id not in self.time_exits:
                # Exit after 2 hours by default in ranging markets
                self.add_time_exit(position_id, duration=120)  # 120 minutes
                logger.info(f"Added time-based exit for ranging market: {position_id}, duration: 120 minutes")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adjusting to ranging market: {str(e)}")
            return False
    
    def _adjust_to_choppy(self, position_id: str) -> bool:
        """
        Adjust exit strategies for choppy/consolidating markets.
        In choppy markets, we want to:
        - Use very tight stops or consider closing position entirely
        - Take quick profits at smaller targets
        - Consider time-based exits
        
        Args:
            position_id: Position ID
            
        Returns:
            bool: Success status
        """
        try:
            position = self.position_manager.get_position(position_id)
            if not position:
                return False
                
            symbol = position.get('symbol')
            direction = position.get('direction', 'long')
            entry_price = float(position.get('entry_price', 0))
            current_price = None
            
            # Try to get current price
            try:
                if self.market_data_service:
                    current_price = self.market_data_service.get_price(symbol)
                else:
                    broker_id = position.get('broker_id', 'unknown')
                    broker = self.broker_manager.brokers.get(broker_id)
                    if broker:
                        quote = broker.get_quote(symbol)
                        current_price = quote.get('last')
            except Exception:
                pass
            
            # Calculate current profit
            current_profit_pct = None
            if current_price is not None:
                if direction == 'long':
                    current_profit_pct = ((current_price / entry_price) - 1) * 100
                else:
                    current_profit_pct = ((entry_price / current_price) - 1) * 100
            
            # If we're in small profit, consider exiting entirely
            if current_profit_pct is not None and current_profit_pct > 0.5:  # If > 0.5% in profit
                logger.info(f"Choppy market exit for {position_id} at {current_profit_pct:.2f}% profit")
                return self._execute_exit(position_id, ExitType.REGIME_BASED, current_price)
            
            # Otherwise, adjust for choppy markets
            # Make take profit very close (0.5-1% from entry)
            if position_id in self.take_profits:
                take_profit = self.take_profits[position_id]
                
                # Set a very close take profit (0.75% from entry)
                if direction == 'long':
                    new_tp = entry_price * 1.0075
                else:
                    new_tp = entry_price * 0.9925
                
                # Update take profit
                take_profit['price'] = new_tp
                take_profit['updated_at'] = datetime.now().isoformat()
                
                logger.info(f"Adjusted take profit for choppy market: {position_id}, new price: {new_tp}")
            elif current_price is not None:
                # Add a new close take profit if none exists
                if direction == 'long':
                    new_tp = entry_price * 1.0075
                else:
                    new_tp = entry_price * 0.9925
                
                self.add_take_profit(position_id, price=new_tp)
                logger.info(f"Added take profit for choppy market: {position_id}, price: {new_tp}")
            
            # Make stop loss very tight
            if position_id in self.stop_losses:
                # Move stop to breakeven or slightly worse
                self.move_stop_to_breakeven(position_id, buffer_percent=0.2)
                logger.info(f"Moved stop to breakeven for choppy market: {position_id}")
            
            # Add very short time exit if not present
            if position_id not in self.time_exits:
                # Exit after 30 minutes in choppy markets
                self.add_time_exit(position_id, duration=30)
                logger.info(f"Added short time-based exit for choppy market: {position_id}, duration: 30 minutes")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adjusting to choppy market: {str(e)}")
            return False
    
    def adjust_exits_for_volatility(self, position_id: str) -> bool:
        """
        Adjust exit strategies based on current market volatility.
        
        Args:
            position_id: Position ID
            
        Returns:
            bool: Success status
        """
        position = self.position_manager.get_position(position_id)
        if not position:
            logger.warning(f"Cannot adjust volatility exits - position {position_id} not found")
            return False
        
        with self._lock:
            try:
                symbol = position.get('symbol')
                direction = position.get('direction', 'long')
                
                # Get volatility metrics
                atr = None
                try:
                    if self.market_data_service and hasattr(self.market_data_service, 'get_indicator'):
                        atr = self.market_data_service.get_indicator(symbol, 'ATR', period=14)
                    else:
                        # Check if we have cached volatility
                        if symbol in self.volatility_metrics:
                            vol_data = self.volatility_metrics.get(symbol, {})
                            atr = vol_data.get('atr')
                except Exception as e:
                    logger.warning(f"Error getting ATR for {symbol}: {str(e)}")
                
                if atr is None:
                    logger.warning(f"Cannot adjust for volatility - no ATR available for {symbol}")
                    return False
                
                # Get entry price
                entry_price = float(position.get('entry_price', 0))
                
                # Calculate relative volatility (ATR as % of price)
                rel_volatility = (atr / entry_price) * 100  # ATR as percentage of entry price
                
                # Store volatility for future reference
                self.volatility_metrics[symbol] = {
                    'atr': atr,
                    'relative_volatility': rel_volatility,
                    'updated_at': datetime.now().isoformat()
                }
                
                # Adjust stop loss based on volatility
                if position_id in self.stop_losses:
                    stop_loss = self.stop_losses[position_id]
                    
                    # Calculate new stop price based on 2x ATR
                    if direction == 'long':
                        new_stop = entry_price - (atr * 2)
                    else:
                        new_stop = entry_price + (atr * 2)
                    
                    # Don't move stop in wrong direction
                    current_stop = stop_loss.get('price')
                    if current_stop is not None:
                        if (direction == 'long' and new_stop < current_stop) or \
                           (direction == 'short' and new_stop > current_stop):
                            # Update stop loss
                            self.update_stop_loss(position_id, new_stop)
                            logger.info(f"Adjusted stop loss for volatility: {position_id}, new stop: {new_stop}")
                
                # Adjust trailing stop based on volatility
                if position_id in self.trailing_stops:
                    trailing = self.trailing_stops[position_id]
                    
                    # Set trail amount to 1x ATR
                    new_trail = atr * 1.0
                    trailing['trail_amount'] = new_trail
                    
                    # Update stop price if activated
                    if trailing.get('is_activated', False):
                        reference_price = trailing.get('reference_price')
                        if reference_price is not None:
                            if direction == 'long':
                                trailing['stop_price'] = reference_price - new_trail
                            else:
                                trailing['stop_price'] = reference_price + new_trail
                    
                    trailing['updated_at'] = datetime.now().isoformat()
                    logger.info(f"Adjusted trailing stop for volatility: {position_id}, new trail amount: {new_trail}")
                
                return True
                
            except Exception as e:
                logger.error(f"Error adjusting for volatility: {str(e)}")
                return False
    
    def adjust_exits_for_time_of_day(self, position_id: str, market_session: str) -> bool:
        """
        Adjust exit strategies based on time of day or market session.
        
        Args:
            position_id: Position ID
            market_session: Market session ('open', 'mid', 'close', 'overnight')
            
        Returns:
            bool: Success status
        """
        position = self.position_manager.get_position(position_id)
        if not position:
            return False
        
        with self._lock:
            try:
                # Adjust based on market session
                if market_session == 'open':  # Market open - high volatility
                    # Wider stops to accommodate volatility around open
                    if position_id in self.stop_losses:
                        stop_loss = self.stop_losses[position_id]
                        stop_price = stop_loss.get('price')
                        entry_price = float(position.get('entry_price', 0))
                        direction = position.get('direction', 'long')
                        
                        # Widen stop by 50%
                        current_distance = abs(entry_price - stop_price)
                        new_distance = current_distance * 1.5
                        
                        if direction == 'long':
                            new_stop = entry_price - new_distance
                        else:
                            new_stop = entry_price + new_distance
                        
                        self.update_stop_loss(position_id, new_stop)
                        logger.info(f"Widened stop for market open: {position_id}, new stop: {new_stop}")
                        
                elif market_session == 'close':  # Market close - reduce exposure
                    # Add time exit if not present to ensure we exit before market close
                    if position_id not in self.time_exits:
                        now = datetime.now()
                        # Default to 16:00 (market close)
                        exit_time = datetime.combine(now.date(), datetime.strptime('15:50', '%H:%M').time())
                        
                        self.add_time_exit(position_id, specific_time=exit_time.isoformat())
                        logger.info(f"Added market close exit for {position_id} at {exit_time.isoformat()}")
                    
                elif market_session == 'overnight':  # Reduce overnight exposure
                    # Consider closing position entirely or using very tight parameters
                    try:
                        # Option 1: Close position entirely
                        if self.market_data_service:
                            current_price = self.market_data_service.get_price(position.get('symbol'))
                            if current_price is not None:
                                logger.info(f"Closing position {position_id} to avoid overnight exposure")
                                return self._execute_exit(position_id, ExitType.SESSION_EXIT, current_price)
                        
                        # Option 2: Move stop to breakeven
                        self.move_stop_to_breakeven(position_id, buffer_percent=0)
                        logger.info(f"Moved stop to exact breakeven for overnight: {position_id}")
                    except Exception as e:
                        logger.error(f"Error handling overnight session: {str(e)}")
                
                return True
                
            except Exception as e:
                logger.error(f"Error adjusting for time of day: {str(e)}")
                return False
    
    def add_chandelier_exit(self, position_id: str, periods: int = 22, multiplier: float = 3.0) -> bool:
        """
        Add a chandelier exit - trailing stop based on highest high minus ATR.
        Useful for trending markets.
        
        Args:
            position_id: Position ID
            periods: Number of periods to look back for highest high/lowest low
            multiplier: ATR multiplier for stop distance
            
        Returns:
            bool: Success status
        """
        position = self.position_manager.get_position(position_id)
        if not position:
            return False
            
        with self._lock:
            try:
                symbol = position.get('symbol')
                direction = position.get('direction', 'long')
                
                # Need market data service with historical data for this exit type
                if not self.market_data_service or not hasattr(self.market_data_service, 'get_historic_ohlc'):
                    logger.warning(f"Cannot add chandelier exit - market data service not available")
                    return False
                
                # Get historical data
                try:
                    # Get ATR
                    atr = self.market_data_service.get_indicator(symbol, 'ATR', period=14)
                    
                    # Get historical highs/lows
                    hist_data = self.market_data_service.get_historic_ohlc(symbol, periods=periods)
                    
                    if not hist_data or not atr:
                        logger.warning(f"Cannot add chandelier exit - historical data not available")
                        return False
                    
                    # Calculate highest high or lowest low
                    if direction == 'long':
                        extreme_price = max([bar['high'] for bar in hist_data])
                        stop_price = extreme_price - (atr * multiplier)
                    else:
                        extreme_price = min([bar['low'] for bar in hist_data])
                        stop_price = extreme_price + (atr * multiplier)
                    
                    # Add trailing stop with this initial value
                    chandelier_exit = {
                        'stop_price': stop_price,
                        'reference_price': extreme_price,
                        'trail_amount': atr * multiplier,
                        'activation_threshold': 0,  # Activated immediately
                        'is_activated': True,
                        'periods': periods,
                        'multiplier': multiplier,
                        'status': ExitStatus.ACTIVE,
                        'type': ExitType.CHANDELIER_EXIT,
                        'created_at': datetime.now().isoformat(),
                        'updated_at': datetime.now().isoformat(),
                        'order_id': None
                    }
                    
                    # Store in trailing stops dictionary with special key
                    chandelier_key = f"{position_id}-chandelier"
                    self.trailing_stops[chandelier_key] = chandelier_exit
                    
                    logger.info(f"Added chandelier exit for {position_id} at {stop_price}")
                    
                    # Start monitoring if not already running
                    if not self.monitoring_active:
                        self.start_monitoring()
                    
                    return True
                    
                except Exception as e:
                    logger.error(f"Error calculating chandelier exit: {str(e)}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error adding chandelier exit: {str(e)}")
                return False
    
