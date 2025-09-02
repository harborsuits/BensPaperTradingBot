#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
London Session Breakout Strategy

This module implements a forex breakout strategy focused on the London trading session.
It identifies price ranges during pre-London hours and trades breakouts when London opens.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, time, timedelta

from trading_bot.strategies.forex.base.forex_base_strategy import ForexBaseStrategy, ForexSession
from trading_bot.strategies.factory.strategy_registry import register_strategy, StrategyType, AssetClass, MarketRegime, TimeFrame
from trading_bot.core.event_system import EventBus, Event, EventType
from trading_bot.strategies.factory.strategy_template import Signal, SignalType

logger = logging.getLogger(__name__)

@register_strategy({
    'asset_class': 'forex',
    'strategy_type': 'breakout',
    'compatible_market_regimes': ['volatile', 'trending'],
    'timeframe': 'intraday',
    'regime_compatibility_scores': {
        'trending': 0.80,       # Strong compatibility with trending markets
        'ranging': 0.40,        # Poor compatibility with ranging markets
        'volatile': 0.85,       # Excellent compatibility with volatile markets
        'low_volatility': 0.30, # Very poor compatibility with low volatility
        'all_weather': 0.65     # Good overall compatibility
    },
    'optimal_parameters': {
        'volatile': {
            'range_hours': 2,
            'breakout_threshold': 0.2,
            'adr_filter_percentage': 25,
            'max_spread_pips': 3.5
        },
        'trending': {
            'range_hours': 3,
            'breakout_threshold': 0.15,
            'adr_filter_percentage': 20,
            'max_spread_pips': 2.5
        }
    }
})
class LondonSessionBreakoutStrategy(ForexBaseStrategy):
    """
    London Session Breakout Strategy for forex
    
    This strategy capitalizes on the volatility of the London trading session open:
    - Measures the pre-London range (typically Asian session range)
    - Enters on breakouts of this range when London opens
    - Uses session time filters and volatility filters
    - Employs pip-based position sizing for precise risk management
    
    The strategy works best with EUR/USD, GBP/USD, and GBP/JPY pairs
    during periods of high market volatility or trending conditions.
    """
    
    # Default parameters - can be overridden via constructor
    DEFAULT_PARAMS = {
        # Session timing parameters
        'pre_london_start_hour': 2,   # 2:00 UTC
        'pre_london_end_hour': 7,     # 7:00 UTC (30 min before London)
        'london_open_hour': 7,        # 7:00 UTC (8:00 London time, winter)
        'london_close_hour': 16,      # 16:00 UTC
        'range_hours': 3,             # Hours before London to establish range
        
        # Breakout parameters
        'breakout_threshold': 0.15,   # Minimum pips beyond range as % of range size
        'min_range_pips': 20,         # Minimum range size in pips
        'max_range_pips': 120,        # Maximum range size in pips
        'entry_window_minutes': 120,  # Time to allow entry after London open
        
        # Volatility filters
        'adr_period': 14,              # Period for Average Daily Range
        'adr_filter_percentage': 20,   # Minimum ADR percentage for the session range
        
        # Currency pairs to trade (majors and crosses with high London volatility)
        'pairs': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'GBP/JPY', 'EUR/JPY'],
        
        # Trade management
        'max_spread_pips': 3.0,        # Maximum spread to allow trade
        'stop_loss_multiplier': 1.0,   # Stop loss as multiple of range size
        'take_profit_multiplier': 1.5, # Take profit as multiple of risk
        'partial_exit_level': 0.8,     # Exit 50% at this profit level (0.8 = 80% to take profit)
        
        # Risk management
        'max_risk_per_trade_percent': 0.01,  # 1% risk per trade
        'max_daily_trades': 2,              # Maximum trades per day
        'pip_value': 0.0001               # Standard pip value for 4-digit pairs
    }
    
    def __init__(self, name: str = "LondonSessionBreakoutStrategy", 
                parameters: Optional[Dict[str, Any]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize London Session Breakout Strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters (will be merged with DEFAULT_PARAMS)
            metadata: Strategy metadata
        """
        # Initialize the base class
        super().__init__(name, parameters, metadata)
        
        # Override defaults with provided parameters
        self.parameters = self.DEFAULT_PARAMS.copy()
        if parameters:
            self.parameters.update(parameters)
        
        # Strategy-specific state
        self.session_ranges = {}  # Ranges detected for each pair
        self.active_signals = {}  # Active signals by pair
        self.daily_trades = {}    # Count of trades taken per day
        self.current_date = None  # Current trading date
        
        logger.info(f"{name} initialized with parameters: {self.parameters}")
    
    def register_events(self, event_bus: EventBus) -> None:
        """
        Register strategy events with the event bus.
        
        Args:
            event_bus: Event bus to register with
        """
        self.event_bus = event_bus
        
        # Register for market data events
        event_bus.register(EventType.MARKET_DATA_UPDATED, self._on_market_data_updated)
        event_bus.register(EventType.TIMEFRAME_COMPLETED, self._on_timeframe_completed)
        
        # Register for session events
        event_bus.register(EventType.SESSION_STARTED, self._on_session_started)
        event_bus.register(EventType.SESSION_ENDED, self._on_session_ended)
        
        logger.info(f"{self.name} registered for events")
    
    def _on_session_started(self, event: Event) -> None:
        """
        Handle session started events.
        
        This is called when a new forex session begins.
        """
        session_name = event.data.get('session')
        if not session_name:
            return
            
        # Reset for a new trading day if needed
        current_date = datetime.now().date()
        if self.current_date != current_date:
            self.daily_trades = {}
            self.current_date = current_date
            
        # Handle London session start
        if session_name == ForexSession.LONDON:
            logger.info("London session started, checking for breakout opportunities")
            
            # Process each pair for potential breakouts
            for pair in self.parameters['pairs']:
                if pair in self.session_ranges:
                    # Only set up new trades if we haven't hit daily limit
                    if pair not in self.daily_trades or self.daily_trades[pair] < self.parameters['max_daily_trades']:
                        self._monitor_for_breakout(pair)
    
    def _on_session_ended(self, event: Event) -> None:
        """
        Handle session ended events.
        
        This is called when a forex session ends.
        """
        session_name = event.data.get('session')
        if not session_name:
            return
            
        # When Asian session ends (before London), finalize ranges
        if session_name == ForexSession.ASIAN:
            logger.info("Asian session ended, finalizing pre-London ranges")
            # Nothing specific to do here as ranges are updated continuously
        
        # When London session ends, clear any pending signals
        elif session_name == ForexSession.LONDON:
            logger.info("London session ended, clearing signals")
            for pair in list(self.active_signals.keys()):
                # Only clear signals that were for entry, not active trades
                if self.active_signals[pair].get('status') == 'pending':
                    del self.active_signals[pair]
    
    def _on_market_data_updated(self, event: Event) -> None:
        """
        Handle market data updated events.
        
        This is where we'll continuously update our session ranges and monitor for breakouts.
        """
        # Extract data from the event
        data = event.data.get('data', {})
        symbol = event.data.get('symbol')
        
        if not symbol or not data or len(data) < 2:
            return
            
        # Only process pairs we're interested in
        if symbol not in self.parameters['pairs']:
            return
            
        # Check current hour
        current_hour = datetime.now().hour
        
        # During pre-London hours, update the range
        if self._is_pre_london_hours(current_hour):
            self._update_session_range(symbol, data)
        
        # During London hours, monitor for breakouts
        elif self._is_london_hours(current_hour):
            self._check_breakout(symbol, data)
    
    def _on_timeframe_completed(self, event: Event) -> None:
        """
        Handle timeframe completed events.
        
        For London breakout, this is where we'll update our daily statistics
        and perform analysis of completed breakout trades.
        """
        # Extract data from the event
        data = event.data.get('data', {})
        symbol = event.data.get('symbol')
        timeframe = event.data.get('timeframe')
        
        if not symbol or not data or not timeframe:
            return
        
        # Only process for specific timeframes we care about (15m and higher)
        if timeframe not in ['15m', '30m', '1h', '4h']:
            return
            
        # Update signals if we have any for this pair
        if symbol in self.active_signals:
            self._update_signal(symbol, data)
    
    def _is_pre_london_hours(self, hour: int) -> bool:
        """
        Check if current time is in pre-London session hours.
        
        Args:
            hour: Current hour in UTC
            
        Returns:
            bool: True if in pre-London hours
        """
        start_hour = self.parameters['pre_london_start_hour']
        end_hour = self.parameters['pre_london_end_hour']
        
        return start_hour <= hour < end_hour
    
    def _is_london_hours(self, hour: int) -> bool:
        """
        Check if current time is in London session hours.
        
        Args:
            hour: Current hour in UTC
            
        Returns:
            bool: True if in London hours
        """
        start_hour = self.parameters['london_open_hour']
        end_hour = self.parameters['london_close_hour']
        
        return start_hour <= hour < end_hour
    
    def _is_in_entry_window(self) -> bool:
        """
        Check if current time is within the London breakout entry window.
        
        Returns:
            bool: True if in entry window
        """
        # Get current time
        now = datetime.now()
        
        # Calculate London open time (today)
        london_open = datetime.combine(now.date(), time(hour=self.parameters['london_open_hour']))
        
        # Calculate end of entry window
        entry_window_end = london_open + timedelta(minutes=self.parameters['entry_window_minutes'])
        
        # Check if current time is within entry window
        return london_open <= now <= entry_window_end
    
    def _update_session_range(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Update the pre-London session range for a symbol.
        
        Args:
            symbol: Symbol to update range for
            data: DataFrame with OHLCV data
        """
        # Initialize if needed
        if symbol not in self.session_ranges:
            self.session_ranges[symbol] = {
                'high': float('-inf'),
                'low': float('inf'),
                'start_time': datetime.now(),
                'last_updated': datetime.now()
            }
        
        # Calculate how many hours of data to look back
        range_hours = self.parameters['range_hours']
        
        # Limit data to the range we're interested in
        lookback_minutes = range_hours * 60
        if len(data) >= lookback_minutes:
            range_data = data.tail(lookback_minutes)
        else:
            range_data = data  # Use all available data if not enough history
        
        # Update range
        session_high = range_data['high'].max()
        session_low = range_data['low'].min()
        
        # Update the range if needed
        if session_high > self.session_ranges[symbol]['high']:
            self.session_ranges[symbol]['high'] = session_high
            
        if session_low < self.session_ranges[symbol]['low']:
            self.session_ranges[symbol]['low'] = session_low
            
        # Update last updated time
        self.session_ranges[symbol]['last_updated'] = datetime.now()
        
        # Calculate range in pips
        range_pips = self._calculate_pips(symbol, self.session_ranges[symbol]['high'] - self.session_ranges[symbol]['low'])
        
        # Store the range size
        self.session_ranges[symbol]['range_pips'] = range_pips
        
        # Log significant range updates
        if range_pips >= self.parameters['min_range_pips']:
            logger.debug(f"Updated {symbol} pre-London range: {range_pips:.1f} pips ({self.session_ranges[symbol]['low']:.5f} - {self.session_ranges[symbol]['high']:.5f})")
    
    def _monitor_for_breakout(self, symbol: str) -> None:
        """
        Set up monitoring for breakouts on a symbol.
        
        Args:
            symbol: Symbol to monitor for breakouts
        """
        # Check if we have a valid range
        if symbol not in self.session_ranges:
            return
            
        range_data = self.session_ranges[symbol]
        range_pips = range_data.get('range_pips', 0)
        
        # Validate the range
        if not self._is_valid_range(symbol, range_data):
            logger.info(f"Range for {symbol} not valid for breakout monitoring: {range_pips:.1f} pips")
            return
            
        # Create pending signals for both directions
        self.active_signals[symbol] = {
            'status': 'pending',
            'range_high': range_data['high'],
            'range_low': range_data['low'],
            'range_pips': range_pips,
            'monitoring_start': datetime.now(),
            'signals': {
                'buy': None,  # Will be populated when breakout occurs
                'sell': None
            }
        }
        
        logger.info(f"Monitoring {symbol} for London session breakout. Range: {range_pips:.1f} pips ({range_data['low']:.5f} - {range_data['high']:.5f})")
    
    def _is_valid_range(self, symbol: str, range_data: Dict[str, Any]) -> bool:
        """
        Validate if a range is suitable for breakout trading.
        
        Args:
            symbol: Symbol to validate range for
            range_data: Range data
            
        Returns:
            bool: True if range is valid
        """
        range_pips = range_data.get('range_pips', 0)
        
        # Check if range is within min/max parameters
        if range_pips < self.parameters['min_range_pips']:
            return False
            
        if range_pips > self.parameters['max_range_pips']:
            return False
            
        # Validate against ADR if available
        if 'adr' in range_data and range_data['adr'] > 0:
            adr_percentage = (range_pips / range_data['adr']) * 100
            if adr_percentage < self.parameters['adr_filter_percentage']:
                return False
        
        return True
    
    def _check_breakout(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Check if a breakout has occurred for a monitored symbol.
        
        Args:
            symbol: Symbol to check
            data: DataFrame with OHLCV data
        """
        # Make sure we're in the entry window and have a pending signal
        if not self._is_in_entry_window():
            return
            
        if symbol not in self.active_signals or self.active_signals[symbol]['status'] != 'pending':
            return
            
        # Get current price
        current_price = data['close'].iloc[-1]
        
        # Get range data
        range_data = self.active_signals[symbol]
        range_high = range_data['range_high']
        range_low = range_data['range_low']
        range_pips = range_data['range_pips']
        
        # Calculate breakout threshold in pips
        threshold_pips = range_pips * self.parameters['breakout_threshold']
        
        # Check for upward breakout
        if current_price > range_high + (threshold_pips * self.parameters['pip_value']):
            logger.info(f"Upward breakout detected for {symbol} at {current_price:.5f}")
            
            # Check if we already have a buy signal
            if range_data['signals']['buy'] is None:
                # Generate buy signal
                signal = self._create_breakout_signal(symbol, SignalType.BUY, current_price, range_data)
                
                # Store the signal
                range_data['signals']['buy'] = signal
                
                # Publish the signal
                self._publish_signal(symbol, signal)
                
                # Increment trade count
                if symbol not in self.daily_trades:
                    self.daily_trades[symbol] = 0
                self.daily_trades[symbol] += 1
        
        # Check for downward breakout
        elif current_price < range_low - (threshold_pips * self.parameters['pip_value']):
            logger.info(f"Downward breakout detected for {symbol} at {current_price:.5f}")
            
            # Check if we already have a sell signal
            if range_data['signals']['sell'] is None:
                # Generate sell signal
                signal = self._create_breakout_signal(symbol, SignalType.SELL, current_price, range_data)
                
                # Store the signal
                range_data['signals']['sell'] = signal
                
                # Publish the signal
                self._publish_signal(symbol, signal)
                
                # Increment trade count
                if symbol not in self.daily_trades:
                    self.daily_trades[symbol] = 0
                self.daily_trades[symbol] += 1
    
    def _create_breakout_signal(self, symbol: str, signal_type: SignalType, 
                              price: float, range_data: Dict[str, Any]) -> Signal:
        """
        Create a breakout signal for the given symbol and type.
        
        Args:
            symbol: Symbol to create signal for
            signal_type: Type of signal
            price: Current price
            range_data: Range data
            
        Returns:
            Signal: Trading signal
        """
        # Range info
        range_high = range_data['range_high']
        range_low = range_data['range_low']
        range_pips = range_data['range_pips']
        
        # Calculate stop loss and take profit
        if signal_type == SignalType.BUY:
            # For buy signals, stop loss below the range low
            stop_loss = range_low - (range_pips * self.parameters['stop_loss_multiplier'] * self.parameters['pip_value'])
            # Take profit above entry
            take_profit = price + ((price - stop_loss) * self.parameters['take_profit_multiplier'])
        else:
            # For sell signals, stop loss above the range high
            stop_loss = range_high + (range_pips * self.parameters['stop_loss_multiplier'] * self.parameters['pip_value'])
            # Take profit below entry
            take_profit = price - ((stop_loss - price) * self.parameters['take_profit_multiplier'])
        
        # Calculate confidence based on multiple factors
        # 1. Range size relative to min/max parameters
        range_factor = min(1.0, range_pips / (self.parameters['max_range_pips'] * 0.5))
        
        # 2. Entry timing (earlier is better)
        entry_time = datetime.now()
        london_open = datetime.combine(entry_time.date(), time(hour=self.parameters['london_open_hour']))
        minutes_since_open = (entry_time - london_open).total_seconds() / 60
        timing_factor = max(0.5, 1.0 - (minutes_since_open / self.parameters['entry_window_minutes']))
        
        # 3. Breakout strength
        if signal_type == SignalType.BUY:
            breakout_pips = self._calculate_pips(symbol, price - range_high)
        else:
            breakout_pips = self._calculate_pips(symbol, range_low - price)
            
        strength_factor = min(1.0, breakout_pips / (range_pips * 0.3))
        
        # Final confidence score
        confidence = min(0.95, (0.4 * range_factor + 0.3 * timing_factor + 0.3 * strength_factor))
        
        # Create signal
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            price=price,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'strategy': self.name,
                'session': 'london',
                'range_pips': range_pips,
                'range_high': range_high,
                'range_low': range_low,
                'breakout_pips': breakout_pips,
                'entry_time': entry_time.isoformat()
            }
        )
    
    def _update_signal(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Update an existing signal based on new data.
        
        Args:
            symbol: Symbol to update signal for
            data: DataFrame with OHLCV data
        """
        # Check if we have a signal for this pair
        if symbol not in self.active_signals:
            return
            
        # If we're not in London hours anymore, expire the signal
        current_hour = datetime.now().hour
        if not self._is_london_hours(current_hour) and self.active_signals[symbol]['status'] == 'pending':
            logger.info(f"Expiring pending signal for {symbol} as London session has ended")
            del self.active_signals[symbol]
            return
            
        # If we have an active buy signal, check for stop/target
        if self.active_signals[symbol]['signals']['buy'] is not None:
            signal = self.active_signals[symbol]['signals']['buy']
            current_price = data['close'].iloc[-1]
            
            if current_price <= signal.stop_loss:
                logger.info(f"Stop loss hit for {symbol} buy signal")
                # Remove the signal
                self.active_signals[symbol]['signals']['buy'] = None
                
            elif current_price >= signal.take_profit:
                logger.info(f"Take profit hit for {symbol} buy signal")
                # Remove the signal
                self.active_signals[symbol]['signals']['buy'] = None
        
        # If we have an active sell signal, check for stop/target
        if self.active_signals[symbol]['signals']['sell'] is not None:
            signal = self.active_signals[symbol]['signals']['sell']
            current_price = data['close'].iloc[-1]
            
            if current_price >= signal.stop_loss:
                logger.info(f"Stop loss hit for {symbol} sell signal")
                # Remove the signal
                self.active_signals[symbol]['signals']['sell'] = None
                
            elif current_price <= signal.take_profit:
                logger.info(f"Take profit hit for {symbol} sell signal")
                # Remove the signal
                self.active_signals[symbol]['signals']['sell'] = None
        
        # If both signals are gone, remove the pair from active signals
        if (self.active_signals[symbol]['signals']['buy'] is None and 
                self.active_signals[symbol]['signals']['sell'] is None):
            del self.active_signals[symbol]
    
    def _publish_signal(self, symbol: str, signal: Signal) -> None:
        """
        Publish a signal to the event bus.
        
        Args:
            symbol: Symbol for the signal
            signal: Signal to publish
        """
        if self.event_bus:
            self.event_bus.publish(
                EventType.SIGNAL_GENERATED,
                {
                    'symbol': symbol,
                    'signal': signal,
                    'strategy': self.name
                }
            )
    
    def _calculate_pips(self, symbol: str, price_difference: float) -> float:
        """
        Convert a price difference to pips.
        
        Args:
            symbol: Symbol to calculate pips for
            price_difference: Price difference
            
        Returns:
            float: Difference in pips
        """
        # Most forex pairs have 4 decimal places (pip is 0.0001)
        pip_value = self.parameters['pip_value']
        
        # JPY pairs have 2 decimal places (pip is 0.01)
        if symbol.endswith('JPY'):
            pip_value = 0.01
            
        return abs(price_difference) / pip_value
    
    def calculate_indicators(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Calculate technical indicators for the strategy.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol for the data
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        # Calculate Average Daily Range (ADR)
        if len(data) >= self.parameters['adr_period'] * 24:  # Assuming hourly data
            # Calculate daily ranges
            daily_ranges = []
            for i in range(self.parameters['adr_period']):
                # Get data for each day
                start_idx = len(data) - (i + 1) * 24
                end_idx = len(data) - i * 24
                
                if start_idx >= 0:
                    day_data = data.iloc[start_idx:end_idx]
                    daily_high = day_data['high'].max()
                    daily_low = day_data['low'].min()
                    daily_range = self._calculate_pips(symbol, daily_high - daily_low)
                    daily_ranges.append(daily_range)
            
            # Calculate ADR
            if daily_ranges:
                indicators['adr'] = sum(daily_ranges) / len(daily_ranges)
                
                # Update session range with ADR
                if symbol in self.session_ranges:
                    self.session_ranges[symbol]['adr'] = indicators['adr']
        
        return indicators
    
    def generate_signals(self, universe: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate trading signals for the universe of symbols.
        
        Args:
            universe: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        signals = {}
        
        # Only generate signals during London session
        current_hour = datetime.now().hour
        if not self._is_london_hours(current_hour):
            return signals
            
        # Process each active pair
        for symbol in self.active_signals:
            # Skip if we don't have data for this pair
            if symbol not in universe:
                continue
                
            data = universe[symbol]
            
            # Check for pending signals
            if self.active_signals[symbol]['status'] == 'pending':
                self._check_breakout(symbol, data)
            
            # Add any active signals to the result
            if self.active_signals[symbol]['signals']['buy'] is not None:
                signals[symbol] = self.active_signals[symbol]['signals']['buy']
            elif self.active_signals[symbol]['signals']['sell'] is not None:
                signals[symbol] = self.active_signals[symbol]['signals']['sell']
        
        return signals
    
    def calculate_position_size(self, signal: Signal, account_balance: float) -> float:
        """
        Calculate position size for the signal based on risk management rules.
        
        Args:
            signal: Trading signal
            account_balance: Current account balance
            
        Returns:
            Position size in units (lot size)
        """
        # Extract parameters
        max_risk_percent = self.parameters['max_risk_per_trade_percent']
        risk_amount = account_balance * max_risk_percent
        
        # Calculate stop loss distance in pips
        entry_price = signal.price
        stop_loss = signal.stop_loss
        pip_value = self.parameters['pip_value']
        
        # Convert price difference to pips
        if signal.signal_type == SignalType.BUY:
            stop_loss_pips = self._calculate_pips(signal.symbol, entry_price - stop_loss)
        else:
            stop_loss_pips = self._calculate_pips(signal.symbol, stop_loss - entry_price)
        
        # Calculate position size based on pips
        return self.calculate_position_size_pips(
            symbol=signal.symbol,
            entry_price=entry_price,
            stop_loss_pips=stop_loss_pips,
            risk_amount=risk_amount
        )
