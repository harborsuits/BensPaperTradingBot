#!/usr/bin/env python
"""
Volatility-Adjusted Stop Loss System

This module implements an advanced stop loss system that:
1. Adjusts stop distances based on market volatility
2. Dynamically updates stops based on price action
3. Implements multiple stop types (fixed, ATR-based, Chandelier)
4. Provides trailing stop mechanisms with smart adjustments
5. Adapts to changing market conditions
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta

from trading_bot.core.event_bus import EventBus, Event, get_global_event_bus
from trading_bot.core.constants import EventType

logger = logging.getLogger(__name__)


class VolatilityAdjustedStops:
    """
    Advanced stop loss system that dynamically adjusts stop distances
    based on market volatility and price action.
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the volatility-adjusted stop system.
        
        Args:
            event_bus: Event bus for events
            config: Configuration dictionary
        """
        self.event_bus = event_bus or get_global_event_bus()
        self.config = config or {}
        
        # Stop loss settings
        self.default_atr_multiple = self.config.get("default_atr_multiple", 2.0)
        self.min_atr_multiple = self.config.get("min_atr_multiple", 1.0)
        self.max_atr_multiple = self.config.get("max_atr_multiple", 4.0)
        
        # Volatility calculation settings
        self.vol_lookback = self.config.get("vol_lookback", 20)
        self.use_averaged_vol = self.config.get("use_averaged_vol", True)
        self.vol_type = self.config.get("vol_type", "atr")  # "atr", "stdev", "parkinson"
        
        # Trailing stop settings
        self.trailing_stop_activation_pct = self.config.get("trailing_stop_activation_pct", 0.01)  # 1% profit
        self.trailing_stop_buffer_pct = self.config.get("trailing_stop_buffer_pct", 0.2)  # 20% of ATR
        self.chandelier_exit_multiplier = self.config.get("chandelier_exit_multiplier", 3.0)
        
        # Regime-based adjustments
        self.regime_stop_adjustments = self.config.get("regime_stop_adjustments", {
            "trending": 0.8,      # Wider stops in trending markets (inverted for stops)
            "ranging": 1.2,       # Tighter stops in ranging markets
            "volatile": 1.5,      # Much tighter stops in volatile markets
            "low_volatility": 0.9 # Wider stops in low volatility
        })
        
        # Time-based adjustment
        self.time_decay_factor = self.config.get("time_decay_factor", 0.95)  # Tighten by 5% per day
        self.apply_time_decay = self.config.get("apply_time_decay", True)
        
        # Strategy-specific stop adjustments
        self.strategy_stop_adjustments = self.config.get("strategy_stop_adjustments", {})
        
        # Active stops tracking
        self.active_stops = {}
        
        # Performance tracking for adaptivity
        self.stop_performance = {
            "premature_exits": 0,
            "delayed_exits": 0,
            "optimal_exits": 0,
            "total_stops": 0
        }
        
        # Cache recent calculations
        self.calculation_cache = {}
        
        # Subscribe to relevant events
        self._subscribe_to_events()
        
        logger.info("Volatility-Adjusted Stops system initialized")
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events for stop adjustment"""
        self.event_bus.subscribe(EventType.MARKET_REGIME_CHANGED, self._handle_regime_change)
        self.event_bus.subscribe(EventType.TRADE_EXECUTED, self._handle_trade_executed)
        self.event_bus.subscribe(EventType.TRADE_CLOSED, self._handle_trade_closed)
        self.event_bus.subscribe(EventType.PRICE_UPDATED, self._handle_price_update)
    
    def _handle_regime_change(self, event: Event):
        """
        Handle market regime change events
        
        Args:
            event: Market regime change event
        """
        symbol = event.data.get('symbol')
        regime = event.data.get('current_regime')
        
        if symbol and regime:
            # Update stops for this symbol based on new regime
            for stop_id, stop_info in list(self.active_stops.items()):
                if stop_info.get('symbol') == symbol:
                    self._recalculate_stop(stop_id, regime=regime)
                    
            logger.info(f"Stops adjusted for {symbol} based on regime change to {regime}")
    
    def _handle_trade_executed(self, event: Event):
        """
        Handle trade executed events
        
        Args:
            event: Trade executed event
        """
        strategy = event.data.get('strategy')
        symbol = event.data.get('symbol')
        entry_price = event.data.get('price')
        position_size = event.data.get('position_size')
        direction = event.data.get('direction', 'long')
        trade_id = event.data.get('trade_id')
        market_data = event.data.get('market_data')
        
        if strategy and symbol and entry_price and position_size and trade_id:
            # Register default stop for this trade
            self.register_stop(
                trade_id=trade_id,
                symbol=symbol,
                strategy=strategy,
                entry_price=entry_price,
                direction=direction,
                market_data=market_data,
                stop_type="atr",
                custom_params=None
            )
    
    def _handle_trade_closed(self, event: Event):
        """
        Handle trade closed events
        
        Args:
            event: Trade closed event
        """
        trade_id = event.data.get('trade_id')
        exit_price = event.data.get('exit_price')
        stop_triggered = event.data.get('stop_triggered', False)
        
        if trade_id and trade_id in self.active_stops:
            stop_info = self.active_stops[trade_id]
            stop_price = stop_info.get('current_stop_price')
            direction = stop_info.get('direction', 'long')
            
            # Evaluate stop performance for adaptivity
            if stop_price and exit_price:
                self.stop_performance['total_stops'] += 1
                
                if stop_triggered:
                    # Stop was hit - was it optimal?
                    if direction == 'long':
                        if exit_price < stop_price * 0.995:  # Price moved significantly below stop
                            self.stop_performance['optimal_exits'] += 1
                        else:
                            self.stop_performance['premature_exits'] += 1
                    else:  # short
                        if exit_price > stop_price * 1.005:  # Price moved significantly above stop
                            self.stop_performance['optimal_exits'] += 1
                        else:
                            self.stop_performance['premature_exits'] += 1
                else:
                    # Stop wasn't hit - should it have been?
                    if direction == 'long':
                        if exit_price < stop_price:
                            self.stop_performance['delayed_exits'] += 1
                    else:  # short
                        if exit_price > stop_price:
                            self.stop_performance['delayed_exits'] += 1
            
            # Clean up active stop
            del self.active_stops[trade_id]
            logger.debug(f"Removed stop for closed trade {trade_id}")
    
    def _handle_price_update(self, event: Event):
        """
        Handle price update events
        
        Args:
            event: Price update event
        """
        symbol = event.data.get('symbol')
        price = event.data.get('price')
        high = event.data.get('high', price)
        low = event.data.get('low', price)
        timestamp = event.data.get('timestamp', datetime.now())
        
        if symbol and price:
            # Update all stops for this symbol
            for stop_id, stop_info in list(self.active_stops.items()):
                if stop_info.get('symbol') == symbol:
                    self._update_stop_for_price(stop_id, price, high, low, timestamp)
    
    def register_stop(
        self,
        trade_id: str,
        symbol: str,
        strategy: str,
        entry_price: float,
        direction: str,
        market_data: Optional[Dict[str, Any]] = None,
        stop_type: str = "atr",
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Register a new stop for a trade.
        
        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            strategy: Strategy name
            entry_price: Entry price
            direction: Trade direction ('long' or 'short')
            market_data: Optional market data dictionary
            stop_type: Stop type ('fixed', 'atr', 'percent', 'chandelier')
            custom_params: Optional custom parameters for the stop
            
        Returns:
            Dictionary with stop details
        """
        params = custom_params or {}
        
        # Extract volatility measure
        volatility = None
        regime = None
        if market_data:
            volatility = market_data.get('atr')
            if not volatility and 'atr' in market_data:
                volatility = market_data['atr']
            regime = market_data.get('regime')
        
        # Calculate initial stop price
        stop_price = self._calculate_initial_stop(
            entry_price=entry_price,
            direction=direction,
            volatility=volatility,
            stop_type=stop_type,
            regime=regime,
            symbol=symbol,
            strategy=strategy,
            params=params
        )
        
        # Store stop information
        stop_info = {
            'trade_id': trade_id,
            'symbol': symbol,
            'strategy': strategy,
            'entry_price': entry_price,
            'direction': direction,
            'initial_stop_price': stop_price,
            'current_stop_price': stop_price,
            'high_watermark': entry_price if direction == 'long' else entry_price,
            'low_watermark': entry_price if direction == 'short' else entry_price,
            'stop_type': stop_type,
            'custom_params': params,
            'created_at': datetime.now(),
            'last_updated': datetime.now(),
            'trailing_active': False
        }
        
        self.active_stops[trade_id] = stop_info
        
        logger.info(f"Registered {stop_type} stop for {symbol} {direction} trade at {stop_price:.4f}")
        return stop_info
    
    def update_stop(
        self,
        trade_id: str,
        new_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Update an existing stop with new parameters.
        
        Args:
            trade_id: Trade identifier
            new_params: New parameters to update
            
        Returns:
            Updated stop info or None if not found
        """
        if trade_id not in self.active_stops:
            logger.warning(f"Cannot update stop: trade {trade_id} not found")
            return None
        
        stop_info = self.active_stops[trade_id]
        
        # Update allowed fields
        updatable_fields = [
            'stop_type', 'custom_params', 'current_stop_price',
            'trailing_active'
        ]
        
        for field in updatable_fields:
            if field in new_params:
                stop_info[field] = new_params[field]
        
        stop_info['last_updated'] = datetime.now()
        
        # Recalculate stop if needed
        if 'stop_type' in new_params or 'custom_params' in new_params:
            self._recalculate_stop(trade_id)
        
        return stop_info
    
    def get_stop_price(self, trade_id: str) -> Optional[float]:
        """
        Get the current stop price for a trade.
        
        Args:
            trade_id: Trade identifier
            
        Returns:
            Current stop price or None if not found
        """
        if trade_id in self.active_stops:
            return self.active_stops[trade_id]['current_stop_price']
        return None
    
    def check_stop_triggered(
        self,
        trade_id: str,
        current_price: float
    ) -> bool:
        """
        Check if a stop has been triggered.
        
        Args:
            trade_id: Trade identifier
            current_price: Current price
            
        Returns:
            True if stop triggered, False otherwise
        """
        if trade_id not in self.active_stops:
            return False
        
        stop_info = self.active_stops[trade_id]
        stop_price = stop_info['current_stop_price']
        direction = stop_info['direction']
        
        if direction == 'long':
            return current_price <= stop_price
        else:  # short
            return current_price >= stop_price
    
    def _calculate_initial_stop(
        self,
        entry_price: float,
        direction: str,
        volatility: Optional[float] = None,
        stop_type: str = "atr",
        regime: Optional[str] = None,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate initial stop price based on various factors.
        
        Args:
            entry_price: Entry price
            direction: Trade direction ('long' or 'short')
            volatility: Volatility measure (ATR or standard deviation)
            stop_type: Stop type ('fixed', 'atr', 'percent', 'chandelier')
            regime: Market regime
            symbol: Trading symbol
            strategy: Strategy name
            params: Additional parameters
            
        Returns:
            Initial stop price
        """
        params = params or {}
        
        # Apply regime-based adjustment if available
        regime_factor = 1.0
        if regime and regime in self.regime_stop_adjustments:
            regime_factor = self.regime_stop_adjustments[regime]
        
        # Apply strategy-specific adjustment if available
        strategy_factor = 1.0
        if strategy and strategy in self.strategy_stop_adjustments:
            strategy_factor = self.strategy_stop_adjustments[strategy]
        
        combined_factor = regime_factor * strategy_factor
        
        # Calculate stop based on type
        if stop_type == "fixed":
            stop_price = params.get('stop_price', 0)
            
        elif stop_type == "percent":
            percent = params.get('percent', 0.02) * combined_factor
            if direction == 'long':
                stop_price = entry_price * (1 - percent)
            else:
                stop_price = entry_price * (1 + percent)
                
        elif stop_type == "atr":
            if volatility is None:
                # Default to 2% if volatility not provided
                volatility = entry_price * 0.02
                
            atr_multiple = params.get('atr_multiple', self.default_atr_multiple) * combined_factor
            
            # Apply min/max constraints to ATR multiple
            atr_multiple = max(self.min_atr_multiple, min(atr_multiple, self.max_atr_multiple))
            
            stop_distance = volatility * atr_multiple
            
            if direction == 'long':
                stop_price = entry_price - stop_distance
            else:
                stop_price = entry_price + stop_distance
                
        elif stop_type == "chandelier":
            if volatility is None:
                # Default to 2% if volatility not provided
                volatility = entry_price * 0.02
                
            chandelier_multiple = params.get(
                'chandelier_multiple', 
                self.chandelier_exit_multiplier
            ) * combined_factor
            
            stop_distance = volatility * chandelier_multiple
            
            if direction == 'long':
                stop_price = entry_price - stop_distance
            else:
                stop_price = entry_price + stop_distance
                
        else:
            # Default to 2% fixed stop
            if direction == 'long':
                stop_price = entry_price * 0.98
            else:
                stop_price = entry_price * 1.02
        
        return stop_price
    
    def _recalculate_stop(
        self,
        trade_id: str,
        regime: Optional[str] = None,
        market_data: Optional[Dict[str, Any]] = None
    ):
        """
        Recalculate stop based on updated parameters.
        
        Args:
            trade_id: Trade identifier
            regime: Optional market regime
            market_data: Optional market data
        """
        if trade_id not in self.active_stops:
            return
        
        stop_info = self.active_stops[trade_id]
        
        # Only recalculate if trailing stop is not active
        if stop_info['trailing_active']:
            return
        
        # Get parameters
        entry_price = stop_info['entry_price']
        direction = stop_info['direction']
        stop_type = stop_info['stop_type']
        custom_params = stop_info['custom_params'] or {}
        strategy = stop_info['strategy']
        symbol = stop_info['symbol']
        
        # Apply time decay if enabled
        time_factor = 1.0
        if self.apply_time_decay:
            days_active = (datetime.now() - stop_info['created_at']).days
            if days_active > 0:
                time_factor = self.time_decay_factor ** days_active
        
        # Extract volatility from market data
        volatility = None
        if market_data:
            volatility = market_data.get('atr')
            if not volatility and 'atr' in market_data:
                volatility = market_data['atr']
            if not regime:
                regime = market_data.get('regime')
        
        # Calculate new stop
        new_stop = self._calculate_initial_stop(
            entry_price=entry_price,
            direction=direction,
            volatility=volatility,
            stop_type=stop_type,
            regime=regime,
            symbol=symbol,
            strategy=strategy,
            params={**custom_params, 'time_factor': time_factor}
        )
        
        # Apply watermark constraints for trailing
        if direction == 'long':
            high_watermark = stop_info['high_watermark']
            if new_stop < stop_info['current_stop_price']:
                # Don't allow stop to move lower
                new_stop = stop_info['current_stop_price']
        else:  # short
            low_watermark = stop_info['low_watermark']
            if new_stop > stop_info['current_stop_price']:
                # Don't allow stop to move higher
                new_stop = stop_info['current_stop_price']
        
        # Update stop
        if new_stop != stop_info['current_stop_price']:
            stop_info['current_stop_price'] = new_stop
            stop_info['last_updated'] = datetime.now()
    
    def _update_stop_for_price(
        self,
        trade_id: str,
        price: float,
        high: float,
        low: float,
        timestamp: datetime
    ):
        """
        Update stop based on new price information.
        
        Args:
            trade_id: Trade identifier
            price: Current price
            high: Current bar high price
            low: Current bar low price
            timestamp: Price timestamp
        """
        if trade_id not in self.active_stops:
            return
        
        stop_info = self.active_stops[trade_id]
        direction = stop_info['direction']
        
        # Update watermarks
        if direction == 'long':
            if high > stop_info['high_watermark']:
                stop_info['high_watermark'] = high
                
                # Check if trailing stop should activate
                entry_price = stop_info['entry_price']
                activation_threshold = entry_price * (1 + self.trailing_stop_activation_pct)
                
                if high >= activation_threshold and not stop_info['trailing_active']:
                    # Activate trailing stop
                    stop_info['trailing_active'] = True
                    logger.debug(f"Trailing stop activated for {trade_id}")
                
                # Update trailing stop if active
                if stop_info['trailing_active']:
                    # Calculate new stop based on high watermark
                    stop_type = stop_info['stop_type']
                    stop_distance = 0
                    
                    if stop_type == 'atr' and 'last_atr' in stop_info:
                        stop_distance = stop_info['last_atr'] * stop_info.get('custom_params', {}).get(
                            'atr_multiple', self.default_atr_multiple)
                    elif stop_type == 'chandelier' and 'last_atr' in stop_info:
                        stop_distance = stop_info['last_atr'] * stop_info.get('custom_params', {}).get(
                            'chandelier_multiple', self.chandelier_exit_multiplier)
                    elif stop_type == 'percent':
                        percent = stop_info.get('custom_params', {}).get('percent', 0.02)
                        stop_distance = high * percent
                    else:
                        # Default to 2% for other types
                        stop_distance = high * 0.02
                    
                    # Calculate new stop price
                    new_stop = high - stop_distance
                    
                    # Only move stop up, never down
                    if new_stop > stop_info['current_stop_price']:
                        stop_info['current_stop_price'] = new_stop
                        stop_info['last_updated'] = timestamp
                        logger.debug(f"Updated trailing stop for {trade_id} to {new_stop:.4f}")
        
        else:  # short direction
            if low < stop_info['low_watermark']:
                stop_info['low_watermark'] = low
                
                # Check if trailing stop should activate
                entry_price = stop_info['entry_price']
                activation_threshold = entry_price * (1 - self.trailing_stop_activation_pct)
                
                if low <= activation_threshold and not stop_info['trailing_active']:
                    # Activate trailing stop
                    stop_info['trailing_active'] = True
                    logger.debug(f"Trailing stop activated for {trade_id}")
                
                # Update trailing stop if active
                if stop_info['trailing_active']:
                    # Calculate new stop based on low watermark
                    stop_type = stop_info['stop_type']
                    stop_distance = 0
                    
                    if stop_type == 'atr' and 'last_atr' in stop_info:
                        stop_distance = stop_info['last_atr'] * stop_info.get('custom_params', {}).get(
                            'atr_multiple', self.default_atr_multiple)
                    elif stop_type == 'chandelier' and 'last_atr' in stop_info:
                        stop_distance = stop_info['last_atr'] * stop_info.get('custom_params', {}).get(
                            'chandelier_multiple', self.chandelier_exit_multiplier)
                    elif stop_type == 'percent':
                        percent = stop_info.get('custom_params', {}).get('percent', 0.02)
                        stop_distance = low * percent
                    else:
                        # Default to 2% for other types
                        stop_distance = low * 0.02
                    
                    # Calculate new stop price
                    new_stop = low + stop_distance
                    
                    # Only move stop down, never up
                    if new_stop < stop_info['current_stop_price']:
                        stop_info['current_stop_price'] = new_stop
                        stop_info['last_updated'] = timestamp
                        logger.debug(f"Updated trailing stop for {trade_id} to {new_stop:.4f}")
    
    def get_all_active_stops(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active stops.
        
        Returns:
            Dictionary of all active stops
        """
        return self.active_stops
    
    def get_stop_performance(self) -> Dict[str, Any]:
        """
        Get stop performance metrics.
        
        Returns:
            Dictionary of stop performance metrics
        """
        performance = self.stop_performance.copy()
        
        # Calculate percentages
        total = performance.get('total_stops', 0)
        if total > 0:
            performance['premature_exits_pct'] = performance.get('premature_exits', 0) / total
            performance['delayed_exits_pct'] = performance.get('delayed_exits', 0) / total
            performance['optimal_exits_pct'] = performance.get('optimal_exits', 0) / total
        
        return performance
    
    def adapt_to_performance(self):
        """
        Adapt stop parameters based on performance metrics
        """
        performance = self.get_stop_performance()
        total = performance.get('total_stops', 0)
        
        if total < 20:
            # Not enough data to adapt
            return
        
        premature_pct = performance.get('premature_exits_pct', 0)
        delayed_pct = performance.get('delayed_exits_pct', 0)
        
        if premature_pct > 0.4:
            # Too many premature exits, widen stops
            self.default_atr_multiple *= 1.05
            self.min_atr_multiple *= 1.05
            self.max_atr_multiple *= 1.05
            logger.info(f"Adapted stops wider due to high premature exits: {premature_pct:.1%}")
        
        elif delayed_pct > 0.3:
            # Too many delayed exits, tighten stops
            self.default_atr_multiple *= 0.95
            self.min_atr_multiple *= 0.95
            self.max_atr_multiple *= 0.95
            logger.info(f"Adapted stops tighter due to high delayed exits: {delayed_pct:.1%}")
            
        # Reset counters
        self.stop_performance = {
            "premature_exits": 0,
            "delayed_exits": 0,
            "optimal_exits": 0,
            "total_stops": 0
        }
