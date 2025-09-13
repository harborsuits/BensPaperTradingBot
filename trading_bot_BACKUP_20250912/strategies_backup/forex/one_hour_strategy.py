#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex One-Hour Trading Strategy

A specialized forex strategy designed specifically for the 1-hour timeframe
with a focus on intraday trading opportunities, session transitions, and
short-term market inefficiencies.

Key Features:
1. Exclusive focus on 1-hour charts
2. Session transition trading (Asian-European-US)
3. Intraday support/resistance identification
4. Time-based position management (close by EOD)
5. Specialized for high-probability intraday setups
6. Adaptive intraday volatility-based position sizing

Author: Ben Dickinson
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import pytz
from enum import Enum

from trading_bot.strategies.base.forex_base import (
    ForexBaseStrategy, 
    MarketRegime, 
    MarketSession, 
    TradeDirection
)
from trading_bot.utils.event_bus import EventBus

logger = logging.getLogger(__name__)

class HourlyPattern(Enum):
    """Patterns specific to 1-hour timeframe analysis"""
    MOMENTUM_CONTINUATION = "momentum_continuation"
    COUNTER_TREND_REVERSAL = "counter_trend_reversal"
    SESSION_BREAKOUT = "session_breakout"
    MIDDAY_CONSOLIDATION = "midday_consolidation"
    NEW_SESSION_REVERSAL = "new_session_reversal"
    HOURLY_SUPPORT_BOUNCE = "hourly_support_bounce"
    HOURLY_RESISTANCE_REJECT = "hourly_resistance_reject"
    VOLATILITY_EXPANSION = "volatility_expansion"

class OneHourForexStrategy(ForexBaseStrategy):
    """
    One-Hour Forex Trading Strategy
    
    A specialized strategy designed exclusively for the 1-hour timeframe with
    a focus on intraday trading, session transitions, and time-based position management.
    
    This strategy excels at:
    - Capturing session transition movements
    - Identifying intraday support/resistance levels
    - Trading 1-hour momentum and reversal patterns
    - Managing positions with time-based decay (higher risk earlier in session)
    - Adapting to intraday volatility profiles
    """
    
    def __init__(self, 
                 parameters: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the One-Hour Forex Strategy
        
        Args:
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Call the parent constructor first
        super().__init__("one_hour_forex", parameters, metadata)
        
        # Default parameters specific to one-hour strategy
        default_params = {
            # Core timeframe settings - strictly enforced
            'primary_timeframe': '1h',
            'secondary_timeframes': ['15m', '4h'],
            
            # Signal generation
            'lookback_periods': 24,        # Look back 24 hours (1 day)
            'momentum_threshold': 0.3,     # Minimum momentum for signal
            'pattern_recognition': True,   # Use pattern recognition
            'session_transition_focus': True, # Focus on session transitions
            
            # Session settings
            'trade_asian_session': True,   
            'trade_european_session': True,
            'trade_us_session': True,
            'avoid_session_overlap': False,
            
            # Entry rules
            'min_hourly_atr': 0.0003,      # Minimum ATR for entry (EUR/USD scale)
            'max_hourly_atr': 0.0015,      # Maximum ATR for entry (EUR/USD scale)
            'min_pattern_quality': 0.6,    # Minimum pattern quality score
            'require_multi_timeframe_confirmation': True,
            
            # Time-based position management
            'max_holding_periods': 8,      # Maximum holding time in hours
            'reduce_exposure_after_hours': 4, # Start reducing exposure after 4 hours
            'close_all_eod': True,         # Close all positions by end of day
            'eod_cutoff_utc': 20,          # UTC hour to consider end of day
            
            # Risk management
            'base_position_size': 0.02,    # Base position size (% of account)
            'max_daily_risk': 0.05,        # Maximum daily risk (% of account)
            'time_decay_factor': 0.8,      # Position size decay per hour held
            'hourly_volatility_adjustment': True, # Adjust for hourly volatility
            
            # Exit rules
            'take_profit_atr_multiple': 1.5, # Take profit as multiple of ATR
            'stop_loss_atr_multiple': 1.0,   # Stop loss as multiple of ATR
            'trailing_stop_activation': 0.5,  # Activate trailing stop after this % of take profit
            'trailing_stop_step': 0.2,        # Trailing stop step size in ATR
            
            # Indicators
            'volatility_indicator': 'atr',   # ATR for volatility measurement
            'momentum_indicators': ['rsi', 'macd', 'stochastic'], # Momentum indicators
            'use_volume_profile': True,     # Use volume profile analysis
            'use_support_resistance': True,  # Use S/R level detection
            
            # Filters
            'news_filter_enabled': True,     # Filter out high-impact news
            'news_filter_window_hours': 1,   # Hours around news to avoid trading
            'filter_low_liquidity_hours': True, # Filter low liquidity hours
            'min_hourly_range_pips': 5,      # Minimum hourly range in pips
        }
        
        # Update default parameters with provided ones
        if self.parameters is None:
            self.parameters = {}
            
        # Apply defaults for any missing parameters
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Initialize strategy state
        self._initialize_state()
        
        logger.info(f"Initialized {self.__class__.__name__} with parameters: {self.parameters}")
        
    def _initialize_state(self):
        """Initialize strategy state variables"""
        # Historical data and patterns
        self.hourly_patterns = {}          # Store detected hourly patterns
        self.support_resistance_levels = {}  # Store S/R levels by symbol
        self.intraday_volatility = {}      # Store intraday volatility profiles
        self.session_performance = {}      # Track performance by session
        
        # Position management
        self.active_positions = {}         # Track active positions
        self.daily_trades = {}             # Track trades by day
        self.exposure_reduction_schedule = {}  # Schedule for reducing exposure
        
        # Performance tracking
        self.pattern_performance = {p.value: {'count': 0, 'win': 0, 'loss': 0, 'pnl': 0.0} 
                                    for p in HourlyPattern}
        self.hour_performance = {hour: {'count': 0, 'win': 0, 'loss': 0, 'pnl': 0.0} 
                                 for hour in range(24)}
        
        # Register event listeners
        self._register_events()
        
    def _register_events(self):
        """Register strategy event listeners"""
        event_bus = EventBus.get_instance()
        event_bus.subscribe("market_data_update", self._on_market_data_update)
        event_bus.subscribe("session_change", self._on_session_change)
        event_bus.subscribe("eod_approaching", self._on_eod_approaching)
        event_bus.subscribe("news_announcement", self._on_news_announcement)
        
    def _on_market_data_update(self, data):
        """Handle market data updates"""
        # We only care about 1-hour data updates
        if 'timeframe' in data and data['timeframe'] == self.parameters['primary_timeframe']:
            symbol = data.get('symbol')
            self._update_intraday_analysis(symbol, data.get('ohlcv'))
    
    def _on_session_change(self, data):
        """Handle trading session changes"""
        new_session = data.get('new_session')
        old_session = data.get('old_session')
        timestamp = data.get('timestamp')
        
        # Log session transition
        logger.info(f"Trading session changed from {old_session} to {new_session} at {timestamp}")
        
        # Check if we should focus on this session transition
        if self.parameters['session_transition_focus']:
            # Session transitions often present trading opportunities
            symbols = data.get('symbols', [])
            for symbol in symbols:
                if self._is_tradable_session(new_session):
                    # Flag this symbol for session transition analysis
                    self._analyze_session_transition(symbol, old_session, new_session, timestamp)
    
    def _on_eod_approaching(self, data):
        """Handle end-of-day approaching notification"""
        # Check if we should close all positions by EOD
        if self.parameters['close_all_eod']:
            cutoff_hour = self.parameters['eod_cutoff_utc']
            current_hour = data.get('current_hour', 0)
            
            # If we're approaching the cutoff hour, start closing positions
            if current_hour >= cutoff_hour - 1:
                logger.info(f"EOD approaching, preparing to close positions at {data.get('timestamp')}")
                self._close_all_positions()
    
    def _on_news_announcement(self, data):
        """Handle economic news announcement"""
        if self.parameters['news_filter_enabled']:
            impact = data.get('impact', 'low')
            # Only filter for medium and high impact news
            if impact.lower() in ['medium', 'high']:
                symbol = data.get('symbol')
                window_hours = self.parameters['news_filter_window_hours']
                
                # Add this symbol to the news filter list with expiry time
                news_time = pd.to_datetime(data.get('time'))
                expiry_time = news_time + pd.Timedelta(hours=window_hours)
                
                logger.info(f"Filtering {symbol} until {expiry_time} due to {impact} impact news")
                
                # Store in news filter with expiry time
                if not hasattr(self, 'news_filter'):
                    self.news_filter = {}
                
                self.news_filter[symbol] = {
                    'expiry': expiry_time,
                    'impact': impact
                }
                
    def _is_tradable_session(self, session):
        """Check if we should trade in the given session"""
        if session == MarketSession.ASIAN and self.parameters['trade_asian_session']:
            return True
        elif session == MarketSession.EUROPEAN and self.parameters['trade_european_session']:
            return True
        elif session == MarketSession.US and self.parameters['trade_us_session']:
            return True
        return False
                
    def _update_intraday_analysis(self, symbol, ohlcv_data):
        """Update intraday analysis for the given symbol"""
        if symbol is None or ohlcv_data is None or ohlcv_data.empty:
            return
            
        # Ensure we have enough data
        min_periods = self.parameters['lookback_periods']
        if len(ohlcv_data) < min_periods:
            logger.warning(f"Insufficient data for {symbol}, need {min_periods} hourly bars")
            return
            
        # Update support/resistance levels
        if self.parameters['use_support_resistance']:
            self._update_support_resistance_levels(symbol, ohlcv_data)
            
        # Update intraday volatility profile
        self._update_intraday_volatility(symbol, ohlcv_data)
        
        # Detect hourly patterns
        if self.parameters['pattern_recognition']:
            self._detect_hourly_patterns(symbol, ohlcv_data)
            
        # Publish intraday analysis update event
        EventBus.get_instance().publish('intraday_analysis_update', {
            'symbol': symbol,
            'timestamp': pd.Timestamp.now(),
            'timeframe': self.parameters['primary_timeframe'],
            'support_levels': self.support_resistance_levels.get(symbol, {}).get('support', []),
            'resistance_levels': self.support_resistance_levels.get(symbol, {}).get('resistance', []),
            'hourly_patterns': self.hourly_patterns.get(symbol, []),
            'volatility_profile': self.intraday_volatility.get(symbol, {})
        })
            
    def _update_support_resistance_levels(self, symbol, ohlcv_data):
        """Identify intraday support and resistance levels"""
        # Initialize S/R storage for this symbol if needed
        if symbol not in self.support_resistance_levels:
            self.support_resistance_levels[symbol] = {'support': [], 'resistance': []}
            
        # Get recent data for intraday levels
        lookback = min(self.parameters['lookback_periods'], len(ohlcv_data))
        recent_data = ohlcv_data.iloc[-lookback:]
        
        # Find swing highs and lows
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        closes = recent_data['close'].values
        
        # Simple swing high/low detection (can be enhanced with more sophisticated methods)
        resistance_levels = []
        support_levels = []
        
        window_size = 3  # Window for swing point detection
        
        # Find swing highs (resistance)
        for i in range(window_size, len(highs) - window_size):
            # Check if this is a local maximum
            if all(highs[i] > highs[i-j] for j in range(1, window_size+1)) and \
               all(highs[i] > highs[i+j] for j in range(1, window_size+1)):
                resistance_levels.append(highs[i])
                
        # Find swing lows (support)
        for i in range(window_size, len(lows) - window_size):
            # Check if this is a local minimum
            if all(lows[i] < lows[i-j] for j in range(1, window_size+1)) and \
               all(lows[i] < lows[i+j] for j in range(1, window_size+1)):
                support_levels.append(lows[i])
                
        # Add current day's high and low
        today_data = ohlcv_data[ohlcv_data.index.date == ohlcv_data.index[-1].date()]
        if not today_data.empty:
            daily_high = today_data['high'].max()
            daily_low = today_data['low'].min()
            
            if daily_high not in resistance_levels:
                resistance_levels.append(daily_high)
            if daily_low not in support_levels:
                support_levels.append(daily_low)
                
        # Cluster levels that are close together (within 0.1% of price)
        current_price = ohlcv_data['close'].iloc[-1]
        price_threshold = current_price * 0.001  # 0.1% of price
        
        clustered_resistance = self._cluster_price_levels(resistance_levels, price_threshold)
        clustered_support = self._cluster_price_levels(support_levels, price_threshold)
        
        # Update the stored levels
        self.support_resistance_levels[symbol]['resistance'] = clustered_resistance
        self.support_resistance_levels[symbol]['support'] = clustered_support
        
        logger.debug(f"Updated S/R levels for {symbol}: {len(clustered_support)} support, {len(clustered_resistance)} resistance")
        
    def _cluster_price_levels(self, levels, threshold):
        """Cluster price levels that are within the threshold"""
        if not levels:
            return []
            
        # Sort levels
        sorted_levels = sorted(levels)
        
        # Initialize clusters
        clusters = [[sorted_levels[0]]]
        
        # Cluster similar levels
        for level in sorted_levels[1:]:
            if level - clusters[-1][-1] < threshold:
                # Add to current cluster
                clusters[-1].append(level)
            else:
                # Start a new cluster
                clusters.append([level])
                
        # Calculate average of each cluster
        clustered_levels = [sum(cluster) / len(cluster) for cluster in clusters]
        
        return clustered_levels
        
    def _update_intraday_volatility(self, symbol, ohlcv_data):
        """Update intraday volatility profile"""
        # Initialize volatility storage for this symbol if needed
        if symbol not in self.intraday_volatility:
            self.intraday_volatility[symbol] = {hour: [] for hour in range(24)}
            
        # Calculate hourly volatility (ATR)
        high = ohlcv_data['high']
        low = ohlcv_data['low']
        close = ohlcv_data['close'].shift(1)
        
        # Simple ATR calculation
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        # True range is the max of these three values
        tr = pd.DataFrame({
            'tr1': tr1,
            'tr2': tr2,
            'tr3': tr3
        }).max(axis=1)
        
        # Calculate hourly ATR
        ohlcv_data.loc[:, 'atr'] = tr.rolling(window=14).mean()
        
        # Group by hour and calculate average volatility
        for hour, group in ohlcv_data.groupby(ohlcv_data.index.hour):
            if not group.empty and not group['atr'].isna().all():
                avg_atr = group['atr'].mean()
                self.intraday_volatility[symbol][hour].append(avg_atr)
                
                # Keep only the most recent 30 days of data
                if len(self.intraday_volatility[symbol][hour]) > 30:
                    self.intraday_volatility[symbol][hour] = self.intraday_volatility[symbol][hour][-30:]
                    
        # Calculate current volatility percentile
        current_hour = ohlcv_data.index[-1].hour
        current_atr = ohlcv_data['atr'].iloc[-1]
        
        if symbol in self.intraday_volatility and current_hour in self.intraday_volatility[symbol]:
            hour_volatility = self.intraday_volatility[symbol][current_hour]
            if hour_volatility:
                percentile = sum(1 for x in hour_volatility if x < current_atr) / len(hour_volatility)
                logger.debug(f"{symbol} current volatility at {current_hour}:00 UTC is in the {percentile:.1%} percentile")
                
    def _detect_hourly_patterns(self, symbol, ohlcv_data):
        """Detect patterns in 1-hour timeframe"""
        # Initialize pattern storage for this symbol if needed
        if symbol not in self.hourly_patterns:
            self.hourly_patterns[symbol] = []
            
        # Get the last few bars for pattern detection
        lookback = min(10, len(ohlcv_data))
        recent_data = ohlcv_data.iloc[-lookback:].copy()
        
        # Add basic indicators for pattern detection
        recent_data['rsi'] = self._calculate_rsi(recent_data['close'], window=14)
        recent_data['sma20'] = recent_data['close'].rolling(window=20).mean()
        recent_data['sma50'] = recent_data['close'].rolling(window=50).mean()
        
        # Calculate price change and body size
        recent_data['price_change'] = recent_data['close'].pct_change()
        recent_data['body_size'] = abs(recent_data['close'] - recent_data['open'])
        recent_data['body_size_pct'] = recent_data['body_size'] / recent_data['open']
        recent_data['upper_wick'] = recent_data['high'] - recent_data[['open', 'close']].max(axis=1)
        recent_data['lower_wick'] = recent_data[['open', 'close']].min(axis=1) - recent_data['low']
        
        # Variables to store detected patterns
        detected_patterns = []
        
        # Focus on the last completed bar
        current_bar = recent_data.iloc[-1]
        prev_bar = recent_data.iloc[-2] if len(recent_data) > 1 else None
        
        # Skip if we don't have enough data
        if prev_bar is None or pd.isna(current_bar['sma20']) or pd.isna(current_bar['rsi']):
            return
            
        # Get current session
        current_hour = recent_data.index[-1].hour
        current_session = self.determine_trading_session(recent_data.index[-1])
        
        # 1. Momentum Continuation Pattern
        if ((current_bar['close'] > current_bar['open']) and 
            (prev_bar['close'] > prev_bar['open']) and 
            (current_bar['close'] > current_bar['sma20']) and 
            (current_bar['rsi'] > 60)):
            
            detected_patterns.append({
                'type': HourlyPattern.MOMENTUM_CONTINUATION.value,
                'direction': TradeDirection.LONG,
                'strength': min(1.0, current_bar['rsi'] / 100),
                'timestamp': recent_data.index[-1]
            })
            
        elif ((current_bar['close'] < current_bar['open']) and 
              (prev_bar['close'] < prev_bar['open']) and 
              (current_bar['close'] < current_bar['sma20']) and 
              (current_bar['rsi'] < 40)):
              
            detected_patterns.append({
                'type': HourlyPattern.MOMENTUM_CONTINUATION.value,
                'direction': TradeDirection.SHORT,
                'strength': min(1.0, (100 - current_bar['rsi']) / 100),
                'timestamp': recent_data.index[-1]
            })
        
        # 2. Counter-Trend Reversal Pattern
        if ((current_bar['close'] > current_bar['open']) and 
            (prev_bar['close'] < prev_bar['open']) and 
            (current_bar['body_size_pct'] > 0.001) and  # Significant body
            (current_bar['close'] < current_bar['sma20']) and  # Still below MA
            (current_bar['rsi'] < 30)):
            
            detected_patterns.append({
                'type': HourlyPattern.COUNTER_TREND_REVERSAL.value,
                'direction': TradeDirection.LONG,
                'strength': min(1.0, (30 - current_bar['rsi']) / 30),
                'timestamp': recent_data.index[-1]
            })
            
        elif ((current_bar['close'] < current_bar['open']) and 
              (prev_bar['close'] > prev_bar['open']) and 
              (current_bar['body_size_pct'] > 0.001) and  # Significant body
              (current_bar['close'] > current_bar['sma20']) and  # Still above MA
              (current_bar['rsi'] > 70)):
              
            detected_patterns.append({
                'type': HourlyPattern.COUNTER_TREND_REVERSAL.value,
                'direction': TradeDirection.SHORT,
                'strength': min(1.0, (current_bar['rsi'] - 70) / 30),
                'timestamp': recent_data.index[-1]
            })
        
        # 3. Session Breakout Pattern
        # Detect if we're at the start of a new session
        is_session_start = False
        if current_session == MarketSession.ASIAN and current_hour in [0, 1, 2]:
            is_session_start = True
        elif current_session == MarketSession.EUROPEAN and current_hour in [7, 8, 9]:
            is_session_start = True
        elif current_session == MarketSession.US and current_hour in [13, 14, 15]:
            is_session_start = True
            
        if is_session_start:
            # Look for breakout of previous session's range
            prev_session_data = recent_data.iloc[:-1]  # Exclude current bar
            if not prev_session_data.empty:
                prev_high = prev_session_data['high'].max()
                prev_low = prev_session_data['low'].min()
                
                if current_bar['close'] > prev_high and current_bar['body_size_pct'] > 0.001:
                    detected_patterns.append({
                        'type': HourlyPattern.SESSION_BREAKOUT.value,
                        'direction': TradeDirection.LONG,
                        'strength': min(1.0, (current_bar['close'] - prev_high) / prev_high * 100),
                        'timestamp': recent_data.index[-1]
                    })
                    
                elif current_bar['close'] < prev_low and current_bar['body_size_pct'] > 0.001:
                    detected_patterns.append({
                        'type': HourlyPattern.SESSION_BREAKOUT.value,
                        'direction': TradeDirection.SHORT,
                        'strength': min(1.0, (prev_low - current_bar['close']) / prev_low * 100),
                        'timestamp': recent_data.index[-1]
                    })
        
        # 4. Midday Consolidation Breakout (primarily during European/US session)
        if current_session in [MarketSession.EUROPEAN, MarketSession.US]:
            # Check if we're in the middle of a session
            is_midday = (current_session == MarketSession.EUROPEAN and current_hour in [10, 11, 12]) or \
                       (current_session == MarketSession.US and current_hour in [16, 17, 18])
                       
            if is_midday:
                # Get the last few hours (mid-session consolidation)
                mid_session_data = recent_data.iloc[-4:-1]  # Last 3 bars excluding current
                
                if len(mid_session_data) >= 3:
                    range_high = mid_session_data['high'].max()
                    range_low = mid_session_data['low'].min()
                    range_size = range_high - range_low
                    
                    # Check for low volatility consolidation followed by breakout
                    if range_size < current_bar['atr'] * 0.8:  # Tight consolidation
                        if current_bar['close'] > range_high and current_bar['body_size'] > range_size:
                            detected_patterns.append({
                                'type': HourlyPattern.MIDDAY_CONSOLIDATION.value,
                                'direction': TradeDirection.LONG,
                                'strength': min(1.0, current_bar['body_size'] / (range_size * 2)),
                                'timestamp': recent_data.index[-1]
                            })
                            
                        elif current_bar['close'] < range_low and current_bar['body_size'] > range_size:
                            detected_patterns.append({
                                'type': HourlyPattern.MIDDAY_CONSOLIDATION.value,
                                'direction': TradeDirection.SHORT,
                                'strength': min(1.0, current_bar['body_size'] / (range_size * 2)),
                                'timestamp': recent_data.index[-1]
                            })
        
        # 5. Support/Resistance tests
        if symbol in self.support_resistance_levels:
            # Get the levels
            support_levels = self.support_resistance_levels[symbol]['support']
            resistance_levels = self.support_resistance_levels[symbol]['resistance']
            
            # Calculate distances to nearest levels
            current_price = current_bar['close']
            price_threshold = current_price * 0.001  # 0.1% of price
            
            # Check for support bounce
            for support in support_levels:
                # If price approached support and bounced
                if abs(current_bar['low'] - support) < price_threshold and current_bar['close'] > current_bar['open']:
                    detected_patterns.append({
                        'type': HourlyPattern.HOURLY_SUPPORT_BOUNCE.value,
                        'direction': TradeDirection.LONG,
                        'strength': min(1.0, (current_bar['close'] - current_bar['low']) / current_bar['low'] * 50),
                        'timestamp': recent_data.index[-1],
                        'level': support
                    })
                    break  # Only detect one support bounce
                    
            # Check for resistance rejection
            for resistance in resistance_levels:
                # If price approached resistance and rejected
                if abs(current_bar['high'] - resistance) < price_threshold and current_bar['close'] < current_bar['open']:
                    detected_patterns.append({
                        'type': HourlyPattern.HOURLY_RESISTANCE_REJECT.value,
                        'direction': TradeDirection.SHORT,
                        'strength': min(1.0, (current_bar['high'] - current_bar['close']) / current_bar['high'] * 50),
                        'timestamp': recent_data.index[-1],
                        'level': resistance
                    })
                    break  # Only detect one resistance rejection
        
        # Store detected patterns
        if detected_patterns:
            # Append to patterns list, keep only last 24 hours of patterns
            self.hourly_patterns[symbol] = detected_patterns + self.hourly_patterns.get(symbol, [])
            if len(self.hourly_patterns[symbol]) > 24:
                self.hourly_patterns[symbol] = self.hourly_patterns[symbol][:24]
                
            # Log detected patterns
            logger.info(f"Detected hourly patterns for {symbol}: {[p['type'] for p in detected_patterns]}")
            
            # Publish pattern detection event
            EventBus.get_instance().publish('hourly_pattern_detected', {
                'symbol': symbol,
                'patterns': detected_patterns,
                'timestamp': recent_data.index[-1].isoformat()
            })
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI for pattern detection"""
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gain = gains.rolling(window=window, min_periods=1).mean()
        avg_loss = losses.rolling(window=window, min_periods=1).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def _analyze_session_transition(self, symbol, old_session, new_session, timestamp):
        """Analyze potential trading opportunities at session transitions"""
        # This method would analyze potential setups during session transitions
        # and flag potential entry points for the trading algorithm
        
        logger.debug(f"Analyzing session transition for {symbol}: {old_session} -> {new_session}")
        
        # We'll publish an event for the session transition analysis
        EventBus.get_instance().publish('session_transition_analysis', {
            'symbol': symbol,
            'old_session': old_session.name if isinstance(old_session, MarketSession) else old_session,
            'new_session': new_session.name if isinstance(new_session, MarketSession) else new_session,
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else timestamp,
            'analyzed_by': self.__class__.__name__
        })
        
    def _close_all_positions(self):
        """Close all positions due to EOD cutoff"""
        # This would be implemented to close all positions when EOD is approaching
        # In a real system, this would interact with the position manager
        
        logger.info("Closing all positions due to EOD cutoff")
        
        # Publish EOD close event
        EventBus.get_instance().publish('eod_close_positions', {
            'strategy': self.__class__.__name__,
            'timestamp': pd.Timestamp.now().isoformat(),
            'reason': 'EOD cutoff reached'
        })
        
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_time: pd.Timestamp) -> Dict[str, Any]:
        """
        Generate trading signals based on 1-hour chart patterns and setups
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            current_time: Current timestamp
            
        Returns:
            Dictionary of signal information
        """
        signals = {}
        
        # Check if we're in an active trading session
        if not self.is_active_trading_session(current_time):
            return signals
            
        # Process each symbol
        for symbol, ohlcv_data in data.items():
            # Ensure we have the 1-hour timeframe data
            if self.parameters['primary_timeframe'] not in ohlcv_data:
                logger.warning(f"Missing 1-hour data for {symbol}")
                continue
                
            # Get 1-hour data
            hourly_data = ohlcv_data[self.parameters['primary_timeframe']]
            
            # Skip if we have insufficient data
            if len(hourly_data) < self.parameters['lookback_periods']:
                logger.debug(f"Insufficient 1-hour data for {symbol}, have {len(hourly_data)} bars, need {self.parameters['lookback_periods']}")
                continue
                
            # Check if we should skip due to news filter
            if self._should_skip_for_news(symbol, current_time):
                logger.info(f"Skipping {symbol} due to news filter")
                continue
                
            # Check if the symbol is active for trading in this session
            current_session = self.determine_trading_session(current_time)
            if not self._is_tradable_session(current_session):
                logger.debug(f"Skipping {symbol} - session {current_session} not tradable")
                continue
            
            # Update intraday analysis (support/resistance, patterns, etc.)
            self._update_intraday_analysis(symbol, hourly_data)
            
            # Check for available patterns
            if symbol not in self.hourly_patterns or not self.hourly_patterns[symbol]:
                continue
                
            # Get recent patterns (last hour)
            recent_patterns = []
            for pattern in self.hourly_patterns[symbol]:
                pattern_time = pattern['timestamp']
                if isinstance(pattern_time, str):
                    pattern_time = pd.Timestamp(pattern_time)
                    
                time_diff = (current_time - pattern_time).total_seconds() / 3600
                if time_diff <= 1.0:  # Within the last hour
                    recent_patterns.append(pattern)
            
            if not recent_patterns:
                continue
                
            # Find the strongest pattern
            strongest_pattern = max(recent_patterns, key=lambda x: x['strength'])
            
            # Check if it meets minimum pattern quality
            if strongest_pattern['strength'] < self.parameters['min_pattern_quality']:
                continue
                
            # Check for multi-timeframe confirmation if required
            if self.parameters['require_multi_timeframe_confirmation']:
                confirmed = self._check_multi_timeframe_confirmation(
                    symbol, 
                    strongest_pattern['direction'], 
                    ohlcv_data, 
                    current_time
                )
                
                if not confirmed:
                    logger.debug(f"Pattern for {symbol} lacks multi-timeframe confirmation")
                    continue
            
            # Get current price info
            current_price = hourly_data['close'].iloc[-1]
            hourly_atr = self._calculate_hourly_atr(hourly_data)
            
            # Check hourly ATR is within acceptable range
            if hourly_atr < self.parameters['min_hourly_atr'] or hourly_atr > self.parameters['max_hourly_atr']:
                logger.debug(f"Hourly ATR for {symbol} outside acceptable range: {hourly_atr}")
                continue
                
            # Calculate position size with time-based decay
            hours_in_session = self._calculate_hours_in_session(current_time, current_session)
            position_size = self._calculate_position_size(symbol, hours_in_session, hourly_atr)
            
            # Calculate stop loss and take profit levels
            if strongest_pattern['direction'] == TradeDirection.LONG:
                entry_price = current_price
                stop_loss = entry_price - (hourly_atr * self.parameters['stop_loss_atr_multiple'])
                take_profit = entry_price + (hourly_atr * self.parameters['take_profit_atr_multiple'])
            else:  # SHORT
                entry_price = current_price
                stop_loss = entry_price + (hourly_atr * self.parameters['stop_loss_atr_multiple'])
                take_profit = entry_price - (hourly_atr * self.parameters['take_profit_atr_multiple'])
            
            # Generate signal ID
            signal_id = f"1h_{symbol}_{strongest_pattern['type']}_{current_time.strftime('%Y%m%d%H%M')}"
            
            # Create signal object
            signal = {
                'id': signal_id,
                'symbol': symbol,
                'direction': strongest_pattern['direction'],
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strength': strongest_pattern['strength'],
                'position_size': position_size,
                'pattern': strongest_pattern['type'],
                'timeframe': self.parameters['primary_timeframe'],
                'session': current_session.name if isinstance(current_session, MarketSession) else current_session,
                'hour_of_day': current_time.hour,
                'atr': hourly_atr,
                'time': current_time
            }
            
            # Add extra information if available
            if 'level' in strongest_pattern:
                signal['key_level'] = strongest_pattern['level']
                
            # Store the signal
            signals[symbol] = signal
            
            # Log signal generation
            logger.info(f"Generated 1-hour signal for {symbol}: {strongest_pattern['direction'].name} based on {strongest_pattern['type']} pattern")
            
            # Publish signal event
            EventBus.get_instance().publish('one_hour_signal', {
                'id': signal_id,
                'symbol': symbol,
                'direction': strongest_pattern['direction'].name,
                'pattern': strongest_pattern['type'],
                'strength': strongest_pattern['strength'],
                'timeframe': self.parameters['primary_timeframe'],
                'hour': current_time.hour,
                'timestamp': current_time.isoformat()
            })
            
        return signals
    
    def _should_skip_for_news(self, symbol: str, current_time: pd.Timestamp) -> bool:
        """Check if we should skip trading due to news filter"""
        if not self.parameters['news_filter_enabled'] or not hasattr(self, 'news_filter'):
            return False
            
        # Check if symbol is in news filter
        if symbol in self.news_filter:
            news_info = self.news_filter[symbol]
            expiry_time = news_info['expiry']
            
            # If current time is before expiry, skip trading
            if current_time < expiry_time:
                return True
                
            # If expired, remove from filter
            else:
                del self.news_filter[symbol]
                
        return False
    
    def _check_multi_timeframe_confirmation(self, symbol: str, direction: TradeDirection, 
                                          data: Dict[str, pd.DataFrame], current_time: pd.Timestamp) -> bool:
        """Check if signal is confirmed on multiple timeframes"""
        # Get secondary timeframes
        secondary_timeframes = self.parameters['secondary_timeframes']
        
        # Ensure we have data for these timeframes
        available_timeframes = [tf for tf in secondary_timeframes if tf in data]
        
        if not available_timeframes:
            logger.warning(f"No secondary timeframes available for {symbol}")
            return False
            
        confirmations = 0
        needed_confirmations = len(available_timeframes) // 2 + 1  # More than half must confirm
        
        for timeframe in available_timeframes:
            tf_data = data[timeframe]
            
            # Skip if insufficient data
            if len(tf_data) < 10:
                continue
                
            # Get basic indicators for confirmation
            close = tf_data['close']
            sma20 = close.rolling(window=20).mean()
            sma50 = close.rolling(window=50).mean()
            
            # Get recent data
            current_close = close.iloc[-1]
            current_sma20 = sma20.iloc[-1]
            current_sma50 = sma50.iloc[-1]
            
            # Check for confirmation based on direction
            if direction == TradeDirection.LONG:
                # For longs, price should be above key MAs or trending up
                if current_close > current_sma20 or (current_sma20 > current_sma50):
                    confirmations += 1
            else:  # SHORT
                # For shorts, price should be below key MAs or trending down
                if current_close < current_sma20 or (current_sma20 < current_sma50):
                    confirmations += 1
                    
        # Check if we have enough confirmations
        return confirmations >= needed_confirmations
    
    def _calculate_hourly_atr(self, data: pd.DataFrame, window: int = 14) -> float:
        """Calculate hourly Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        # Calculate true range components
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        # True range is the max of these
        tr = pd.DataFrame({
            'tr1': tr1,
            'tr2': tr2,
            'tr3': tr3
        }).max(axis=1)
        
        # ATR is the moving average of true range
        atr = tr.rolling(window=window).mean().iloc[-1]
        
        return atr
    
    def _calculate_hours_in_session(self, current_time: pd.Timestamp, session: MarketSession) -> int:
        """Calculate hours elapsed in the current session"""
        session_start_hour = {
            MarketSession.ASIAN: 0,      # 00:00 UTC
            MarketSession.EUROPEAN: 7,  # 07:00 UTC
            MarketSession.US: 13        # 13:00 UTC
        }.get(session, 0)
        
        current_hour = current_time.hour
        
        # Handle session that crosses midnight
        if session == MarketSession.ASIAN and current_hour > 12:
            # This would be next day's Asian session starting
            return current_hour - 24
            
        return current_hour - session_start_hour
    
    def _calculate_position_size(self, symbol: str, hours_in_session: int, hourly_atr: float) -> float:
        """Calculate position size with time-based decay"""
        # Base position size
        base_size = self.parameters['base_position_size']
        
        # Apply time decay if configured
        reduce_after = self.parameters['reduce_exposure_after_hours']
        if hours_in_session > reduce_after:
            # Exponential decay
            hours_over = hours_in_session - reduce_after
            decay_factor = self.parameters['time_decay_factor'] ** hours_over
            position_size = base_size * decay_factor
        else:
            position_size = base_size
            
        # Adjust for volatility if configured
        if self.parameters['hourly_volatility_adjustment']:
            # Get typical volatility range for this symbol
            symbol_base_atr = 0.0003  # Default for major pairs
            if hasattr(self, 'symbol_base_atr') and symbol in self.symbol_base_atr:
                symbol_base_atr = self.symbol_base_atr[symbol]
                
            # Adjust position size inversely with volatility
            volatility_ratio = symbol_base_atr / hourly_atr if hourly_atr > 0 else 1.0
            # Cap the adjustment to prevent extreme sizing
            volatility_ratio = max(0.5, min(2.0, volatility_ratio))
            
            position_size *= volatility_ratio
            
        # Ensure position size doesn't exceed max daily risk
        max_size = self.parameters['max_daily_risk']
        position_size = min(position_size, max_size)
        
        return position_size
    
    def update(self, data: Dict[str, pd.DataFrame], current_time: pd.Timestamp, account_size: float) -> None:
        """Update strategy on new data"""
        # Generate new signals
        new_signals = self.generate_signals(data, current_time)
        
        # Manage existing positions
        self._manage_existing_positions(data, current_time, account_size)
        
        # Check for end of day cutoff
        if self.parameters['close_all_eod'] and current_time.hour >= self.parameters['eod_cutoff_utc']:
            self._close_all_positions()
            
        # Update hourly performance metrics
        self._update_performance_metrics(current_time)
    
    def _manage_existing_positions(self, data: Dict[str, pd.DataFrame], current_time: pd.Timestamp, account_size: float) -> None:
        """Manage existing positions - adjust stops, take profits, etc."""
        # In a real implementation, this would check active positions and adjust based on:
        # 1. Time in trade (reduce exposure over time)
        # 2. Trailing stop adjustments
        # 3. Partial take profits at session boundaries
        # 4. EOD position closure
        pass
    
    def _update_performance_metrics(self, current_time: pd.Timestamp) -> None:
        """Update performance metrics for strategy optimization"""
        # This would track performance by pattern type, hour of day, etc.
        # to inform future parameter optimization
        pass
    
    def get_regime_compatibility(self, regime: MarketRegime) -> float:
        """
        Get the compatibility score of this strategy with the specified market regime
        
        The One-Hour Forex Strategy performs best in moderately trending markets
        with defined intraday movements and session transitions.
        
        Args:
            regime: The market regime to check compatibility with
            
        Returns:
            Compatibility score from 0.0 to 1.0
        """
        # Define compatibility with different regimes
        compatibility_map = {
            MarketRegime.TRENDING: 0.85,      # Strong in trending markets
            MarketRegime.RANGING: 0.40,       # Weaker in ranging markets
            MarketRegime.VOLATILE: 0.60,      # Moderate in volatile markets
            MarketRegime.BREAKOUT: 0.75,      # Good in breakout regimes (catches session breakouts)
            MarketRegime.REVERSAL: 0.65,      # Decent in reversals (catches counter-trend moves)
            MarketRegime.LOW_VOLATILITY: 0.30, # Poor in low volatility
            MarketRegime.HIGH_LIQUIDITY: 0.80, # Strong in liquid markets
            MarketRegime.LOW_LIQUIDITY: 0.35,  # Poor in illiquid conditions
            MarketRegime.NORMAL: 0.70,        # Good in normal conditions
            MarketRegime.UNKNOWN: 0.50        # Average in unknown conditions
        }
        
        # Return the compatibility score
        return compatibility_map.get(regime, 0.5)
    
    def get_optimal_parameters(self, regime: MarketRegime) -> Dict[str, Any]:
        """
        Get regime-optimized parameters for the strategy
        
        Args:
            regime: The market regime to optimize for
            
        Returns:
            Dictionary of optimized parameters
        """
        # Base parameters (default)
        optimized_params = self.parameters.copy()
        
        # Adjust based on regime
        if regime == MarketRegime.TRENDING:
            # In trending markets, focus on momentum continuation
            optimized_params.update({
                'momentum_threshold': 0.25,  # Lower threshold to catch more trends
                'take_profit_atr_multiple': 2.0,  # Larger take profit to capture trend moves
                'stop_loss_atr_multiple': 1.0,   # Standard stop loss
                'base_position_size': 0.025      # Slightly larger position
            })
            
        elif regime == MarketRegime.RANGING:
            # In ranging markets, focus on support/resistance bounces
            optimized_params.update({
                'min_pattern_quality': 0.7,    # Higher quality threshold
                'take_profit_atr_multiple': 1.2, # Smaller take profit
                'stop_loss_atr_multiple': 0.8,  # Tighter stop loss
                'base_position_size': 0.015     # Smaller position size
            })
            
        elif regime == MarketRegime.VOLATILE:
            # In volatile markets, be more selective and protect capital
            optimized_params.update({
                'min_pattern_quality': 0.75,    # Higher quality threshold
                'min_hourly_atr': 0.0005,      # Look for higher volatility
                'take_profit_atr_multiple': 1.8, # Wider take profit
                'stop_loss_atr_multiple': 1.5,  # Wider stop loss
                'base_position_size': 0.015     # Smaller position size
            })
            
        elif regime == MarketRegime.BREAKOUT:
            # In breakout regimes, focus on session breakouts
            optimized_params.update({
                'session_transition_focus': True, # Focus on session transitions
                'take_profit_atr_multiple': 2.2, # Larger take profit
                'stop_loss_atr_multiple': 1.2,  # Slightly wider stop loss
                'base_position_size': 0.022     # Moderate position size
            })
            
        # Return the optimized parameters
        return optimized_params
    
    def optimize(self, data: Dict[str, pd.DataFrame], param_grid: Optional[Dict[str, List[Any]]] = None) -> Dict[str, Any]:
        """
        Optimize strategy parameters based on historical performance
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            param_grid: Optional parameter grid to search
            
        Returns:
            Dictionary of optimized parameters
        """
        # Default parameter grid if none provided
        if param_grid is None:
            param_grid = {
                'momentum_threshold': [0.2, 0.3, 0.4],
                'min_pattern_quality': [0.5, 0.6, 0.7],
                'take_profit_atr_multiple': [1.0, 1.5, 2.0],
                'stop_loss_atr_multiple': [0.8, 1.0, 1.2],
                'base_position_size': [0.01, 0.02, 0.03],
                'time_decay_factor': [0.7, 0.8, 0.9]
            }
            
        # Simple grid search implementation (would be more sophisticated in production)
        best_params = self.parameters.copy()
        best_score = float('-inf')
        
        # For each parameter combination, backtest and score
        # This is a simplified version for illustration
        # In production, this would use a proper backtesting engine
        
        # Log optimization completion
        logger.info(f"Optimized parameters for {self.__class__.__name__}")
        
        # Return the best parameters
        return best_params
