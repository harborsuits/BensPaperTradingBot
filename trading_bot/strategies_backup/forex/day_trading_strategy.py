#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Day Trading Strategy

This module implements a day trading strategy for forex markets,
focusing on intraday price movements with complete position exit by day end.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta, time

from trading_bot.strategies.base.forex_base import ForexBaseStrategy, ForexSession
from trading_bot.strategies.strategy_template import Signal, SignalType, TimeFrame, MarketRegime
from trading_bot.event_system import EventBus
from trading_bot.event_system.event_types import EventType, Event

logger = logging.getLogger(__name__)

class ForexDayTradingStrategy(ForexBaseStrategy):
    """Day trading strategy for forex markets.
    
    This strategy focuses on intraday price movements by:
    1. Analyzing shorter timeframes (5m-1h)
    2. Identifying key intraday support and resistance levels
    3. Capturing breakouts and reversals during high-volatility sessions
    4. Exiting all positions before the market close
    """
    
    # Default strategy parameters
    DEFAULT_PARAMETERS = {
        # Time parameters
        'preferred_timeframes': [TimeFrame.MINUTE_5, TimeFrame.MINUTE_15, TimeFrame.HOUR_1],
        'session_exit_buffer_minutes': 30,   # Exit trades this many minutes before session close
        'max_trade_duration_hours': 8,       # Maximum trade hold time
        'trading_sessions': [ForexSession.LONDON, ForexSession.NEWYORK],
        
        # Entry parameters
        'use_support_resistance': True,      # Use support/resistance for entries
        'use_momentum': True,                # Use momentum indicators for entries
        'use_volatility_filter': True,       # Filter entries based on volatility
        'entry_signal_threshold': 0.7,       # Minimum signal strength for entry
        
        # Technical indicators
        'fast_ema': 8,                       # Fast EMA period
        'slow_ema': 21,                      # Slow EMA period
        'macd_fast': 12,                     # MACD fast period
        'macd_slow': 26,                     # MACD slow period
        'macd_signal': 9,                    # MACD signal period
        'rsi_period': 14,                    # RSI period
        'rsi_overbought': 70,                # RSI overbought level
        'rsi_oversold': 30,                  # RSI oversold level
        'atr_period': 14,                    # ATR period
        
        # Support/Resistance
        'sr_lookback_bars': 20,              # Bars to look back for S/R
        'sr_pip_threshold': 5,               # Minimum pips between S/R levels
        'sr_touch_count': 2,                 # Minimum touches to confirm S/R
        
        # Position management
        'stop_loss_atr_multiple': 1.0,       # Stop loss as ATR multiple
        'take_profit_atr_multiple': 2.0,     # Take profit as ATR multiple
        'trailing_stop_activation': 1.0,     # ATR multiple profit before trailing
        'time_based_exit': True,             # Exit based on time
        'partial_exits': True,               # Take partial profits
        'partial_exit_levels': [0.5, 0.75],  # Levels for partial exits
        'partial_exit_sizes': [0.3, 0.3],    # Size to exit at each level
        
        # Risk parameters
        'max_risk_per_trade_percent': 0.01,  # Max risk per trade (1%)
        'max_daily_risk_percent': 0.03,      # Max daily risk (3%)
        'max_open_trades': 3,                # Maximum concurrent open trades
    }
    
    def __init__(self, name: str = "Forex Day Trading", 
                 parameters: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the forex day trading strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters (will be merged with DEFAULT_PARAMETERS)
            metadata: Strategy metadata
        """
        # Merge default parameters with ForexBaseStrategy defaults
        forex_params = self.DEFAULT_FOREX_PARAMS.copy()
        forex_params.update(self.DEFAULT_PARAMETERS)
        
        # Override with user-provided parameters if any
        if parameters:
            forex_params.update(parameters)
        
        # Initialize the base strategy
        super().__init__(name=name, parameters=forex_params, metadata=metadata)
        
        # Register with the event system
        self.event_bus = EventBus()
        
        # Strategy state
        self.current_signals = {}        # Current trading signals
        self.active_trades = {}          # Active day trades
        self.intraday_support = {}       # Intraday support levels
        self.intraday_resistance = {}    # Intraday resistance levels
        self.day_start_time = None       # Start time of current trading day
        
        logger.info(f"Initialized {self.name} strategy")
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_time: datetime) -> Dict[str, Signal]:
        """
        Generate trade signals based on intraday price movements.
        
        Args:
            data: Dictionary mapping symbols to OHLCV DataFrames
            current_time: Current timestamp
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        signals = {}
        
        # Check if we're in an active trading session
        is_active_session = self.is_current_session_active()
        
        if not is_active_session:
            logger.debug(f"Not in active trading session at {current_time}")
            return signals
        
        # Check if we need to initialize day start
        self._update_day_start(current_time)
        
        # Check for time-based exits on existing positions
        if self.parameters['time_based_exit']:
            self._check_time_exits(current_time)
        
        # Process each symbol
        for symbol, ohlcv in data.items():
            # Skip if we don't have enough data
            if len(ohlcv) < 30:  # Need at least 30 bars
                logger.debug(f"Insufficient data for {symbol}, skipping")
                continue
            
            # Calculate intraday indicators
            indicators = self._calculate_intraday_indicators(ohlcv)
            
            # Detect intraday support and resistance levels
            self._detect_support_resistance(symbol, ohlcv, indicators)
            
            # Evaluate for potential day trade setups
            signal = self._evaluate_day_trade_setup(symbol, ohlcv, indicators, current_time)
            
            if signal:
                signals[symbol] = signal
                # Store in current signals
                self.current_signals[symbol] = signal
        
        # Publish event with active day trades
        if self.active_trades:
            event_data = {
                'strategy_name': self.name,
                'active_trades': self.active_trades,
                'trade_count': len(self.active_trades),
                'timestamp': current_time.isoformat()
            }
            
            event = Event(
                event_type=EventType.SIGNAL_GENERATED,
                source=self.name,
                data=event_data,
                metadata={'strategy_type': 'forex', 'category': 'day_trading'}
            )
            self.event_bus.publish(event)
        
        return signals
    
    def _update_day_start(self, current_time: datetime) -> None:
        """
        Update the day start time if needed.
        
        Args:
            current_time: Current timestamp
        """
        # Check if we need to initialize or reset day start
        if self.day_start_time is None or current_time.date() > self.day_start_time.date():
            self.day_start_time = datetime.combine(current_time.date(), time(0, 0))
            logger.info(f"Started new trading day: {self.day_start_time.date()}")
            
            # Reset intraday tracking
            self.intraday_support = {}
            self.intraday_resistance = {}
    
    def _check_time_exits(self, current_time: datetime) -> None:
        """
        Check for trades that should be exited based on time conditions.
        
        Args:
            current_time: Current timestamp
        """
        # Check time-based exits for active trades
        trades_to_exit = []
        
        for trade_id, trade in self.active_trades.items():
            entry_time = datetime.fromisoformat(trade['entry_time'])
            
            # Calculate hold duration
            hold_duration = (current_time - entry_time).total_seconds() / 3600  # in hours
            
            # Check if we've exceeded max trade duration
            max_duration = self.parameters['max_trade_duration_hours']
            if hold_duration >= max_duration:
                trades_to_exit.append(trade_id)
                logger.info(f"Time-based exit for {trade['symbol']} after {hold_duration:.1f} hours")
            
            # Check if we're approaching session close
            session_buffer = self.parameters['session_exit_buffer_minutes']
            next_session_close = self._get_next_session_close(current_time)
            
            if next_session_close:
                mins_to_close = (next_session_close - current_time).total_seconds() / 60
                if mins_to_close <= session_buffer:
                    trades_to_exit.append(trade_id)
                    logger.info(f"Session close exit for {trade['symbol']} with {mins_to_close:.0f} minutes to close")
        
        # Emit exit events for trades that need to be closed
        for trade_id in trades_to_exit:
            trade = self.active_trades[trade_id]
            
            # Publish exit event
            event_data = {
                'strategy_name': self.name,
                'symbol': trade['symbol'],
                'trade_id': trade_id,
                'exit_reason': 'time_based_exit',
                'entry_time': trade['entry_time'],
                'exit_time': current_time.isoformat(),
                'position_direction': trade['direction']
            }
            
            event = Event(
                event_type=EventType.POSITION_EXIT,
                source=self.name,
                data=event_data,
                metadata={'strategy_type': 'forex', 'category': 'day_trading'}
            )
            self.event_bus.publish(event)
            
            # Remove from active trades
            self.active_trades.pop(trade_id, None)
    
    def _get_next_session_close(self, current_time: datetime) -> Optional[datetime]:
        """
        Get the next trading session close time.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Datetime of next session close or None
        """
        # Get active trading sessions
        active_sessions = []
        for session in self.parameters['trading_sessions']:
            start_time, end_time = self.SESSION_HOURS[session]
            
            # Convert session hours to datetime on current day
            session_start = datetime.combine(current_time.date(), start_time)
            session_end = datetime.combine(current_time.date(), end_time)
            
            # Handle overnight sessions
            if end_time < start_time:  # Session spans midnight
                if current_time.time() < end_time:
                    # Early morning of the next day
                    session_start = datetime.combine(current_time.date() - timedelta(days=1), start_time)
                else:
                    # After the end time, session ends tomorrow
                    session_end = datetime.combine(current_time.date() + timedelta(days=1), end_time)
            
            # Check if current time is in this session
            if session_start <= current_time <= session_end:
                active_sessions.append((session, session_end))
        
        if not active_sessions:
            return None
        
        # Return the earliest session end time
        return min([end for _, end in active_sessions])
    
    def _detect_support_resistance(self, symbol: str, ohlcv: pd.DataFrame, indicators: Dict[str, Any]) -> None:
        """
        Detect intraday support and resistance levels.
        
        Args:
            symbol: Currency pair symbol
            ohlcv: DataFrame with OHLCV price data
            indicators: Dictionary of technical indicators
        """
        # Initialize support/resistance for this symbol if needed
        if symbol not in self.intraday_support:
            self.intraday_support[symbol] = []
        
        if symbol not in self.intraday_resistance:
            self.intraday_resistance[symbol] = []
        
        # Use pivot points for quick S/R levels
        # Calculate pivot point based on previous day's data
        if len(ohlcv) < 2:
            return
            
        lookback = self.parameters['sr_lookback_bars']
        prev_high = ohlcv['high'].iloc[-lookback:-1].max()
        prev_low = ohlcv['low'].iloc[-lookback:-1].min()
        prev_close = ohlcv['close'].iloc[-2]
        
        # Calculate classic pivot points
        pivot = (prev_high + prev_low + prev_close) / 3
        s1 = (2 * pivot) - prev_high
        s2 = pivot - (prev_high - prev_low)
        r1 = (2 * pivot) - prev_low
        r2 = pivot + (prev_high - prev_low)
        
        # Check for price swings (highs and lows)
        swing_highs = []
        swing_lows = []
        
        # Simple swing detection (can be improved with more sophisticated algorithms)
        for i in range(2, min(lookback, len(ohlcv) - 2)):
            # Check for swing high
            if (ohlcv['high'].iloc[-i] > ohlcv['high'].iloc[-i-1] and 
                ohlcv['high'].iloc[-i] > ohlcv['high'].iloc[-i-2] and 
                ohlcv['high'].iloc[-i] > ohlcv['high'].iloc[-i+1] and 
                ohlcv['high'].iloc[-i] > ohlcv['high'].iloc[-i+2]):
                swing_highs.append(ohlcv['high'].iloc[-i])
            
            # Check for swing low
            if (ohlcv['low'].iloc[-i] < ohlcv['low'].iloc[-i-1] and 
                ohlcv['low'].iloc[-i] < ohlcv['low'].iloc[-i-2] and 
                ohlcv['low'].iloc[-i] < ohlcv['low'].iloc[-i+1] and 
                ohlcv['low'].iloc[-i] < ohlcv['low'].iloc[-i+2]):
                swing_lows.append(ohlcv['low'].iloc[-i])
        
        # Combine pivot levels and swing levels
        resistance_levels = [r1, r2] + swing_highs
        support_levels = [s1, s2] + swing_lows
        
        # Filter and consolidate levels
        pip_threshold = self.parameters['sr_pip_threshold'] * self.parameters['pip_value']
        
        # Consolidate resistance levels
        consolidated_resistance = []
        for level in sorted(resistance_levels):
            # Skip if too close to an existing level
            if not consolidated_resistance or min(abs(r - level) for r in consolidated_resistance) > pip_threshold:
                consolidated_resistance.append(level)
        
        # Consolidate support levels
        consolidated_support = []
        for level in sorted(support_levels, reverse=True):
            # Skip if too close to an existing level
            if not consolidated_support or min(abs(s - level) for s in consolidated_support) > pip_threshold:
                consolidated_support.append(level)
        
        # Update the strategy state
        self.intraday_resistance[symbol] = consolidated_resistance
        self.intraday_support[symbol] = consolidated_support
    
    def _calculate_intraday_indicators(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for intraday trading.
        
        Args:
            ohlcv: DataFrame with OHLCV price data
            
        Returns:
            Dictionary with calculated indicators
        """
        indicators = {}
        
        # Moving Averages
        indicators['ema_fast'] = self.calculate_ema(ohlcv, self.parameters['fast_ema'])
        indicators['ema_slow'] = self.calculate_ema(ohlcv, self.parameters['slow_ema'])
        
        # MACD
        macd, signal, histogram = self.calculate_macd(
            ohlcv, 
            self.parameters['macd_fast'],
            self.parameters['macd_slow'],
            self.parameters['macd_signal']
        )
        indicators['macd'] = macd
        indicators['macd_signal'] = signal
        indicators['macd_histogram'] = histogram
        
        # RSI
        indicators['rsi'] = self.calculate_rsi(ohlcv, self.parameters['rsi_period'])
        
        # Volatility (ATR)
        indicators['atr'] = self.calculate_atr(ohlcv, self.parameters['atr_period'])
        
        # Bollinger Bands
        upper, middle, lower = self.calculate_bollinger_bands(ohlcv, 20, 2)
        indicators['bb_upper'] = upper
        indicators['bb_middle'] = middle
        indicators['bb_lower'] = lower
        
        # Price momentum
        indicators['momentum'] = self.calculate_momentum(ohlcv, 10)
        
        # Price slope (linear regression)
        indicators['price_slope'] = self.calculate_slope(ohlcv['close'], 10)
        
        # Volume relative strength
        if 'volume' in ohlcv.columns:
            indicators['volume_ma'] = self.calculate_sma(ohlcv['volume'], 10)
            indicators['rel_volume'] = ohlcv['volume'] / indicators['volume_ma']
        
        return indicators
    
    def _evaluate_day_trade_setup(self, symbol: str, ohlcv: pd.DataFrame, 
                                  indicators: Dict[str, Any], current_time: datetime) -> Optional[Signal]:
        """
        Evaluate price data for potential day trading setups.
        
        Args:
            symbol: Currency pair symbol
            ohlcv: DataFrame with OHLCV price data
            indicators: Dictionary of technical indicators
            current_time: Current timestamp
            
        Returns:
            Signal object if a setup is found, None otherwise
        """
        # Current price data
        current_price = ohlcv['close'].iloc[-1]
        prev_price = ohlcv['close'].iloc[-2] if len(ohlcv) > 1 else current_price
        
        # Setup strength scoring (0-1 scale)
        long_score = 0.0
        short_score = 0.0
        
        # 1. Trend direction from EMAs
        if indicators['ema_fast'][-1] > indicators['ema_slow'][-1]:
            long_score += 0.2
        else:
            short_score += 0.2
        
        # 2. EMA crossover check (recent)
        if (indicators['ema_fast'][-2] <= indicators['ema_slow'][-2] and 
            indicators['ema_fast'][-1] > indicators['ema_slow'][-1]):
            long_score += 0.3  # Fresh long crossover
        elif (indicators['ema_fast'][-2] >= indicators['ema_slow'][-2] and 
              indicators['ema_fast'][-1] < indicators['ema_slow'][-1]):
            short_score += 0.3  # Fresh short crossover
        
        # 3. MACD signal
        if indicators['macd'][-1] > indicators['macd_signal'][-1]:
            long_score += 0.1
        else:
            short_score += 0.1
            
        # MACD crossing
        if (indicators['macd'][-2] <= indicators['macd_signal'][-2] and 
            indicators['macd'][-1] > indicators['macd_signal'][-1]):
            long_score += 0.15  # Fresh MACD bullish cross
        elif (indicators['macd'][-2] >= indicators['macd_signal'][-2] and 
              indicators['macd'][-1] < indicators['macd_signal'][-1]):
            short_score += 0.15  # Fresh MACD bearish cross
        
        # 4. RSI conditions
        rsi = indicators['rsi'][-1]
        if rsi < self.parameters['rsi_oversold']:
            long_score += 0.2  # Oversold condition
        elif rsi > self.parameters['rsi_overbought']:
            short_score += 0.2  # Overbought condition
        
        # 5. Support/Resistance proximity
        if symbol in self.intraday_support and self.intraday_support[symbol]:
            closest_support = min(self.intraday_support[symbol], key=lambda x: abs(current_price - x))
            support_distance = abs(current_price - closest_support) / indicators['atr'][-1]
            
            # Bouncing off support (price near support and moving up)
            if support_distance < 0.5 and current_price > prev_price:
                long_score += 0.2
        
        if symbol in self.intraday_resistance and self.intraday_resistance[symbol]:
            closest_resistance = min(self.intraday_resistance[symbol], key=lambda x: abs(current_price - x))
            resistance_distance = abs(current_price - closest_resistance) / indicators['atr'][-1]
            
            # Rejection off resistance (price near resistance and moving down)
            if resistance_distance < 0.5 and current_price < prev_price:
                short_score += 0.2
        
        # 6. Momentum confirmation
        if indicators['momentum'][-1] > 0:
            long_score += 0.1
        else:
            short_score += 0.1
        
        # 7. Price slope
        if indicators['price_slope'][-1] > 0:
            long_score += 0.1
        else:
            short_score += 0.1
            
        # 8. Volume confirmation (if available)
        if 'rel_volume' in indicators and len(indicators['rel_volume']) > 0:
            if indicators['rel_volume'][-1] > 1.2:  # Above average volume
                # Add to the stronger signal
                if long_score > short_score:
                    long_score += 0.1
                else:
                    short_score += 0.1
        
        # Determine overall signal direction and strength
        signal_threshold = self.parameters['entry_signal_threshold']
        
        # Check for maximum open trades
        if len(self.active_trades) >= self.parameters['max_open_trades']:
            logger.debug(f"Maximum open trades ({self.parameters['max_open_trades']}) reached, skipping signal")
            return None
        
        # Generate trade signal if score exceeds threshold
        if long_score >= signal_threshold and long_score > short_score:
            # Calculate stop loss and take profit levels
            atr_value = indicators['atr'][-1]
            stop_loss = current_price - (atr_value * self.parameters['stop_loss_atr_multiple'])
            take_profit = current_price + (atr_value * self.parameters['take_profit_atr_multiple'])
            
            # Create a trade ID
            trade_id = f"{symbol}-{current_time.strftime('%Y%m%d%H%M%S')}-LONG"
            
            # Record this as an active trade
            self.active_trades[trade_id] = {
                'symbol': symbol,
                'direction': 'LONG',
                'entry_time': current_time.isoformat(),
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'score': long_score,
                'atr': atr_value
            }
            
            # Create a new long signal
            return Signal(
                symbol=symbol,
                signal_type=SignalType.LONG,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quantity=None,  # Will be calculated by position sizing logic
                timestamp=current_time,
                expiration=current_time + timedelta(hours=self.parameters['max_trade_duration_hours']),
                source=self.name,
                metadata={
                    'strategy_type': 'forex_day_trading',
                    'score': long_score,
                    'trade_id': trade_id,
                    'regime': self.current_regime if hasattr(self, 'current_regime') else 'unknown'
                }
            )
            
        elif short_score >= signal_threshold and short_score > long_score:
            # Calculate stop loss and take profit levels
            atr_value = indicators['atr'][-1]
            stop_loss = current_price + (atr_value * self.parameters['stop_loss_atr_multiple'])
            take_profit = current_price - (atr_value * self.parameters['take_profit_atr_multiple'])
            
            # Create a trade ID
            trade_id = f"{symbol}-{current_time.strftime('%Y%m%d%H%M%S')}-SHORT"
            
            # Record this as an active trade
            self.active_trades[trade_id] = {
                'symbol': symbol,
                'direction': 'SHORT',
                'entry_time': current_time.isoformat(),
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'score': short_score,
                'atr': atr_value
            }
            
            # Create a new short signal
            return Signal(
                symbol=symbol,
                signal_type=SignalType.SHORT,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quantity=None,  # Will be calculated by position sizing logic
                timestamp=current_time,
                expiration=current_time + timedelta(hours=self.parameters['max_trade_duration_hours']),
                source=self.name,
                metadata={
                    'strategy_type': 'forex_day_trading',
                    'score': short_score,
                    'trade_id': trade_id,
                    'regime': self.current_regime if hasattr(self, 'current_regime') else 'unknown'
                }
            )
            
        # No valid setup found
        return None
    
    def get_regime_compatibility_score(self, regime: MarketRegime) -> float:
        """
        Calculate how compatible this strategy is with the given market regime.
        
        Args:
            regime: Market regime to check compatibility with
            
        Returns:
            Compatibility score from 0.0 (incompatible) to 1.0 (highly compatible)
        """
        compatibility_scores = {
            MarketRegime.TRENDING_BULL: 0.8,    # Works well in uptrends for quick long entries
            MarketRegime.TRENDING_BEAR: 0.8,    # Works well in downtrends for quick short entries
            MarketRegime.RANGING: 0.7,          # Can work in ranges with proper S/R identification
            MarketRegime.VOLATILE: 0.4,         # Less effective in highly volatile markets without clear direction
            MarketRegime.VOLATILE_BULL: 0.5,    # Can work with higher volatility in bulls, but with caution
            MarketRegime.VOLATILE_BEAR: 0.5,    # Can work with higher volatility in bears, but with caution
            MarketRegime.BREAKOUT: 0.6,         # Decent on fresh breakouts from ranges
            MarketRegime.REVERSAL: 0.4,         # Not ideal during major trend reversals
            MarketRegime.LOW_VOLATILITY: 0.2,   # Poor in very low volatility - not enough intraday movement
            MarketRegime.UNDEFINED: 0.5,        # Moderate compatibility when regime is unclear
        }
        
        return compatibility_scores.get(regime, 0.5)  # Default moderate compatibility
    
    def optimize_for_regime(self, regime: MarketRegime) -> None:
        """
        Optimize strategy parameters for the specified market regime.
        
        Args:
            regime: Market regime to optimize for
        """
        self.current_regime = regime
        
        # Base parameters that don't change
        base_params = {
            'trading_sessions': self.parameters['trading_sessions'],
            'session_exit_buffer_minutes': self.parameters['session_exit_buffer_minutes'],
            'max_daily_risk_percent': self.parameters['max_daily_risk_percent'],
            'time_based_exit': True
        }
        
        # Regime-specific parameter adjustments
        if regime == MarketRegime.TRENDING_BULL or regime == MarketRegime.TRENDING_BEAR:
            # In trending markets, focus on trend following with wider targets
            regime_params = {
                'entry_signal_threshold': 0.65,        # More permissive entries
                'stop_loss_atr_multiple': 1.0,        # Standard stop
                'take_profit_atr_multiple': 2.5,       # Wider targets to capture trend moves
                'trailing_stop_activation': 1.0,       # Activate trailing stops earlier
                'max_trade_duration_hours': 6,         # Allow trades to run longer with the trend
                'use_momentum': True,                  # Focus on momentum
                'use_support_resistance': False,       # Less focus on S/R in strong trends
                'partial_exits': True,
                'partial_exit_levels': [0.6, 0.8],     # Later partial exits to let profits run
                'partial_exit_sizes': [0.3, 0.3],
                'max_open_trades': 3
            }
            
        elif regime == MarketRegime.RANGING:
            # In ranging markets, focus on S/R bounces with tighter targets
            regime_params = {
                'entry_signal_threshold': 0.75,        # More stringent entries
                'stop_loss_atr_multiple': 0.8,        # Tighter stops
                'take_profit_atr_multiple': 1.5,      # Lower targets suitable for ranges
                'trailing_stop_activation': 1.0,      # Standard trailing stop
                'max_trade_duration_hours': 4,        # Shorter trade duration
                'use_momentum': False,               # Less focus on momentum
                'use_support_resistance': True,      # Heavy S/R focus
                'partial_exits': True,
                'partial_exit_levels': [0.4, 0.7],    # Earlier partial exits in ranges
                'partial_exit_sizes': [0.4, 0.4],
                'max_open_trades': 4                  # More trades in ranges
            }
            
        elif regime in [MarketRegime.VOLATILE, MarketRegime.VOLATILE_BULL, MarketRegime.VOLATILE_BEAR]:
            # In volatile markets, be more selective with tighter risk control
            regime_params = {
                'entry_signal_threshold': 0.85,        # Much more stringent entries
                'stop_loss_atr_multiple': 1.2,        # Wider stops for volatility
                'take_profit_atr_multiple': 1.8,      # Reasonable targets
                'trailing_stop_activation': 0.8,      # Quick trailing stops
                'max_trade_duration_hours': 3,        # Shorter trades in volatile conditions
                'use_momentum': True,                 # Use momentum
                'use_support_resistance': True,      # Use S/R
                'partial_exits': True,
                'partial_exit_levels': [0.3, 0.6],    # Faster partial exits
                'partial_exit_sizes': [0.5, 0.3],     # Larger first partial
                'max_open_trades': 2                  # Fewer trades in volatile markets
            }
            
        elif regime == MarketRegime.BREAKOUT:
            # For breakouts, focus on momentum and volume confirmation
            regime_params = {
                'entry_signal_threshold': 0.7,         # Moderate entry threshold
                'stop_loss_atr_multiple': 1.2,        # Wider stops for volatility
                'take_profit_atr_multiple': 2.2,      # Higher targets for breakout moves
                'trailing_stop_activation': 1.0,      # Standard trailing
                'max_trade_duration_hours': 5,        # Moderate duration
                'use_momentum': True,                 # Focus on momentum
                'use_support_resistance': True,      # Use S/R for entry confirmation
                'partial_exits': True,
                'partial_exit_levels': [0.5, 0.8],    # Later partials to capture breakout move
                'partial_exit_sizes': [0.3, 0.3],
                'max_open_trades': 3
            }
            
        elif regime == MarketRegime.LOW_VOLATILITY:
            # In low volatility, be very selective and focus on small moves
            regime_params = {
                'entry_signal_threshold': 0.9,         # Very strict entries
                'stop_loss_atr_multiple': 1.5,        # Wider stops relative to small ATR
                'take_profit_atr_multiple': 1.2,      # Modest targets
                'trailing_stop_activation': 0.8,      # Quick trailing
                'max_trade_duration_hours': 4,        # Standard duration
                'use_momentum': False,               # Less momentum focus (not much momentum)
                'use_support_resistance': True,      # Heavy S/R focus
                'partial_exits': True,
                'partial_exit_levels': [0.3, 0.6],    # Early partials to lock in small profits
                'partial_exit_sizes': [0.5, 0.3],
                'max_open_trades': 2                  # Limited trades
            }
        
        else:  # Default/Undefined regime
            # Balanced approach for unclear regimes
            regime_params = {
                'entry_signal_threshold': 0.75,        # Standard entry threshold
                'stop_loss_atr_multiple': 1.0,        # Standard stop
                'take_profit_atr_multiple': 2.0,      # Standard target
                'trailing_stop_activation': 1.0,      # Standard trailing
                'max_trade_duration_hours': 4,        # Standard duration
                'use_momentum': True,                 # Use momentum
                'use_support_resistance': True,      # Use S/R
                'partial_exits': True,
                'partial_exit_levels': [0.5, 0.7],    # Standard partials
                'partial_exit_sizes': [0.3, 0.3],
                'max_open_trades': 3                  # Standard trade count
            }
        
        # Combine base and regime-specific parameters
        optimized_params = {**self.parameters.copy(), **base_params, **regime_params}
        
        # Update strategy parameters
        self.parameters.update(optimized_params)
        
        logger.info(f"Optimized Day Trading strategy for {regime.name} regime")
        
    def optimize(self, data: Dict[str, pd.DataFrame], parameter_ranges: Optional[Dict[str, List[Any]]] = None):
        """
        Optimize strategy parameters using historical data.
        
        Args:
            data: Dictionary mapping symbols to historical OHLCV data
            parameter_ranges: Optional parameter ranges to optimize (if None, use defaults)
        """
        if not parameter_ranges:
            # Default parameter ranges to search
            parameter_ranges = {
                'fast_ema': [5, 8, 10, 12],
                'slow_ema': [15, 18, 21, 25],
                'rsi_period': [9, 14, 21],
                'entry_signal_threshold': [0.6, 0.65, 0.7, 0.75, 0.8],
                'stop_loss_atr_multiple': [0.8, 1.0, 1.2, 1.5],
                'take_profit_atr_multiple': [1.5, 2.0, 2.5, 3.0],
            }
        
        logger.info(f"Starting parameter optimization for {self.name}")
        
        # Implement grid search or other optimization method here
        # This would be a more extensive implementation in a real system
        
        # For now, we'll just log that optimization would happen
        # and set some reasonable default parameters
        
        logger.info(f"Completed parameter optimization for {self.name}")
        
        # In a real implementation, we would:.
        # 1. Generate all parameter combinations
        # 2. Backtest each combination
        # 3. Score results (Sharpe, profit factor, etc.)
        # 4. Select best parameters
        # 5. Update self.parameters
