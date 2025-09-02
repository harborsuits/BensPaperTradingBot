#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Swing Trading Strategy

This module implements a swing trading strategy for forex markets,
focusing on multi-day position holding and larger price moves.
Uses multiple timeframe analysis for better entries and exits.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

from trading_bot.strategies.base.forex_base import ForexBaseStrategy, ForexSession
from trading_bot.strategies.strategy_template import Signal, SignalType, TimeFrame, MarketRegime
from trading_bot.event_system import EventBus
from trading_bot.event_system.event_types import EventType, Event

logger = logging.getLogger(__name__)

class ForexSwingTradingStrategy(ForexBaseStrategy):
    """
    Swing trading strategy for forex markets.
    
    This strategy focuses on intermediate-term trades with:
    1. Multi-day position holding (typically 1-7 days)
    2. Multiple timeframe analysis for entry/exit
    3. Larger profit targets (50-200 pips)
    4. Fundamental analysis incorporation
    5. Focus on key support/resistance levels
    """
    
    # Default strategy parameters
    DEFAULT_PARAMETERS = {
        # Basic parameters
        'primary_timeframe': TimeFrame.HOUR_4,    # 4-hour primary timeframe
        'trend_timeframe': TimeFrame.DAY_1,      # Daily timeframe for trend
        'entry_timeframe': TimeFrame.HOUR_1,      # 1-hour timeframe for entries
        'profit_target_pips': 100,            # Take profit at 100 pips
        'stop_loss_pips': 50,                 # Stop loss at 50 pips
        'risk_per_trade_pct': 1.0,            # Risk 1% per trade
        'max_holding_days': 10,               # Maximum holding period
        
        # Trade management
        'trailing_stop_activation': 50,       # Activate trailing stop after 50 pips
        'trailing_stop_distance': 30,         # Trailing stop distance in pips
        'partial_take_profit': True,          # Take partial profits
        'partial_tp_threshold': 70,           # Take partial at 70 pips
        'partial_tp_percentage': 0.5,         # Close 50% of position
        
        # Multiple timeframe indicators
        'daily_ma_period': 50,                # 50-day moving average
        'daily_atr_period': 14,               # Daily ATR period
        'h4_ma_period': 20,                   # 4-hour moving average
        'h1_ma_period': 10,                   # 1-hour moving average
        
        # Oscillator parameters
        'rsi_period': 14,                     # RSI period
        'rsi_overbought': 70,                 # RSI overbought level
        'rsi_oversold': 30,                   # RSI oversold level
        'macd_fast': 12,                      # MACD fast period
        'macd_slow': 26,                      # MACD slow period
        'macd_signal': 9,                     # MACD signal period
        
        # Support/resistance parameters
        'sr_lookback_periods': 100,           # Lookback for S/R levels
        'sr_pip_threshold': 20,               # Minimum pips between S/R levels
        'sr_touch_count': 2,                  # Minimum touches to confirm S/R
        
        # Fundamental analysis
        'use_economic_calendar': True,        # Consider economic events
        'avoid_high_impact_news': True,       # Avoid trading during high-impact news
        'news_avoidance_hours': 12,           # Hours to avoid trading before high-impact news
        
        # Filter parameters
        'min_daily_atr_pips': 50,             # Minimum daily ATR in pips
        'max_daily_atr_pips': 250,            # Maximum daily ATR in pips
        'min_risk_reward': 1.5,               # Minimum risk-reward ratio
    }
    
    def __init__(self, name: str = "Forex Swing Trading Strategy", 
                 parameters: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the forex swing trading strategy.
        
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
        self.last_updates = {}           # Last update timestamps
        self.support_resistance = {}     # Support/resistance levels by symbol
        self.fundamental_events = {}     # Economic calendar events by currency
        self.active_swing_trades = {}    # Active trades being monitored for management
        
        logger.info(f"Initialized {self.name} with primary timeframe {self.parameters['primary_timeframe']}")
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_time: datetime):
        """
        Generate trade signals for swing trading opportunities.
        
        Args:
            data: Dictionary mapping symbols to OHLCV DataFrames (primary timeframe)
            current_time: Current timestamp
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        signals = {}
        
        # Process each symbol
        for symbol, ohlcv in data.items():
            # Skip if we don't have enough data
            required_bars = max(
                self.parameters['h4_ma_period'],
                self.parameters['rsi_period'],
                self.parameters['macd_slow'] + self.parameters['macd_signal']
            ) + 20  # Add buffer
            
            if len(ohlcv) < required_bars:
                logger.warning(f"Not enough data for {symbol} to generate swing trading signals")
                continue
            
            # Get trend data from daily timeframe
            # In a real implementation, this would come from separate timeframe data
            # For this demo, we'll assume the primary timeframe data is sufficient
            trend_data = self._get_resampled_data(ohlcv, self.parameters['trend_timeframe'])
            entry_data = self._get_resampled_data(ohlcv, self.parameters['entry_timeframe'])
            
            # Skip if daily ATR is outside acceptable range
            daily_atr_pips = self._calculate_atr(trend_data, self.parameters['daily_atr_period'])
            if daily_atr_pips < self.parameters['min_daily_atr_pips'] or daily_atr_pips > self.parameters['max_daily_atr_pips']:
                logger.debug(f"Daily ATR outside range for {symbol}: {daily_atr_pips:.2f} pips")
                continue
            
            # Update support/resistance levels
            self._update_support_resistance(symbol, trend_data)
            
            # Check for fundamental events that might impact trading
            if self.parameters['use_economic_calendar'] and self._check_economic_calendar(symbol, current_time):
                logger.debug(f"Skipping {symbol} due to upcoming economic events")
                continue
            
            # Calculate indicators on multiple timeframes
            trend_indicators = self._calculate_trend_indicators(trend_data)
            primary_indicators = self._calculate_swing_indicators(ohlcv)
            entry_indicators = self._calculate_entry_indicators(entry_data)
            
            # Evaluate swing trading opportunity
            signal = self._evaluate_swing_opportunity(symbol, ohlcv, trend_indicators, 
                                                primary_indicators, entry_indicators, current_time)
            
            if signal:
                # Store the signal
                signals[symbol] = signal
                self.current_signals[symbol] = signal
                
                # Add to active trades for management
                self.active_swing_trades[symbol] = {
                    'entry_time': current_time,
                    'entry_price': signal.metadata['entry_price'],
                    'stop_loss': signal.metadata['stop_loss'],
                    'take_profit': signal.metadata['take_profit'],
                    'partial_taken': False,
                    'direction': signal.signal_type
                }
                
                # Emit the signal event
                self.emit_strategy_event({
                    'symbol': symbol,
                    'signal_type': signal.signal_type.name,
                    'confidence': signal.confidence,
                    'entry_price': signal.metadata.get('entry_price'),
                    'stop_loss': signal.metadata.get('stop_loss'),
                    'take_profit': signal.metadata.get('take_profit'),
                    'holding_period': f"{self.parameters['max_holding_days']} days max",
                    'pip_target': self.parameters['profit_target_pips'],
                    'risk_reward': signal.metadata.get('risk_reward_ratio')
                })
                
                logger.info(
                    f"Generated {signal.signal_type.name} swing trading signal for {symbol} "
                    f"with confidence {signal.confidence:.2f}, target {self.parameters['profit_target_pips']} pips"
                )
        
        return signals
    
    def _get_resampled_data(self, data: pd.DataFrame, target_timeframe: TimeFrame) -> pd.DataFrame:
        """
        Resample OHLCV data to a different timeframe.
        
        Args:
            data: Original OHLCV DataFrame
            target_timeframe: Target timeframe to resample to
            
        Returns:
            Resampled OHLCV DataFrame
        """
        # In a production system, we would properly resample the data
        # For this demo, we'll just return the original data and pretend it's resampled
        # This would be replaced with actual timeframe conversion in production
        
        # Example of proper resampling (commented out):
        # timeframe_map = {
        #     TimeFrame.M5: '5T',
        #     TimeFrame.M15: '15T',
        #     TimeFrame.H1: '1H',
        #     TimeFrame.H4: '4H',
        #     TimeFrame.D1: '1D'
        # }
        # rule = timeframe_map.get(target_timeframe)
        # resampled = data.resample(rule, on='datetime').agg({
        #     'open': 'first',
        #     'high': 'max',
        #     'low': 'min',
        #     'close': 'last',
        #     'volume': 'sum'
        # })
        
        return data.copy()
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> float:
        """
        Calculate Average True Range (ATR) in pips.
        
        Args:
            data: OHLCV DataFrame
            period: ATR period
            
        Returns:
            ATR value in pips
        """
        high = data['high']
        low = data['low']
        close = data['close']
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        # Convert to pips
        pip_value = self.parameters['pip_value']
        atr_pips = atr / pip_value
        
        return atr_pips
    
    def _update_support_resistance(self, symbol: str, data: pd.DataFrame):
        """
        Update support and resistance levels for the symbol.
        
        Args:
            symbol: Currency pair symbol
            data: OHLCV DataFrame (trend timeframe)
        """
        # Initialize support/resistance if not exists
        if symbol not in self.support_resistance:
            self.support_resistance[symbol] = {'support': [], 'resistance': []}
        
        # Get parameters
        lookback = self.parameters['sr_lookback_periods']
        min_touches = self.parameters['sr_touch_count']
        pip_threshold = self.parameters['sr_pip_threshold'] * self.parameters['pip_value']
        
        # Use recent data only
        recent_data = data.iloc[-lookback:].copy() if len(data) > lookback else data.copy()
        
        # Find swing highs and lows (simplified approach)
        highs = recent_data['high']
        lows = recent_data['low']
        
        # A point is a swing high if it's higher than the n points before and after it
        window_size = 5
        swing_highs = []
        swing_lows = []
        
        # Find swing highs (resistance levels)
        for i in range(window_size, len(highs) - window_size):
            if all(highs[i] > highs[i-j] for j in range(1, window_size+1)) and \
               all(highs[i] > highs[i+j] for j in range(1, window_size+1)):
                swing_highs.append((recent_data.index[i], highs[i]))
        
        # Find swing lows (support levels)
        for i in range(window_size, len(lows) - window_size):
            if all(lows[i] < lows[i-j] for j in range(1, window_size+1)) and \
               all(lows[i] < lows[i+j] for j in range(1, window_size+1)):
                swing_lows.append((recent_data.index[i], lows[i]))
        
        # Group nearby levels (within pip_threshold)
        grouped_highs = self._group_levels([h[1] for h in swing_highs], pip_threshold)
        grouped_lows = self._group_levels([l[1] for l in swing_lows], pip_threshold)
        
        # Keep levels with enough touches
        resistance_levels = [level for level, count in grouped_highs.items() if count >= min_touches]
        support_levels = [level for level, count in grouped_lows.items() if count >= min_touches]
        
        # Update the support/resistance dictionary
        self.support_resistance[symbol]['resistance'] = resistance_levels
        self.support_resistance[symbol]['support'] = support_levels
    
    def _group_levels(self, levels: List[float], threshold: float) -> Dict[float, int]:
        """
        Group nearby price levels and count occurrences.
        
        Args:
            levels: List of price levels
            threshold: Maximum distance to consider levels as the same
            
        Returns:
            Dictionary mapping representative levels to touch counts
        """
        if not levels:
            return {}
            
        # Sort levels
        sorted_levels = sorted(levels)
        groups = {}
        
        for level in sorted_levels:
            # Find closest existing group
            closest_group = None
            min_distance = float('inf')
            
            for group_level in groups.keys():
                distance = abs(level - group_level)
                if distance < min_distance and distance < threshold:
                    min_distance = distance
                    closest_group = group_level
            
            if closest_group is not None:
                # Add to existing group
                groups[closest_group] += 1
            else:
                # Create new group
                groups[level] = 1
        
        return groups
    
    def _check_economic_calendar(self, symbol: str, current_time: datetime) -> bool:
        """
        Check if there are upcoming economic events that might impact trading.
        
        Args:
            symbol: Currency pair symbol
            current_time: Current timestamp
            
        Returns:
            True if trading should be avoided, False otherwise
        """
        # In a production system, we would check a real economic calendar
        # For this demo, we'll just return False (allow trading)
        
        if not self.parameters['avoid_high_impact_news']:
            return False
            
        # Extract currencies from the symbol (e.g., 'EURUSD' -> ['EUR', 'USD'])
        base_currency = symbol[:3]
        quote_currency = symbol[3:6]
        
        # Check if we have any high-impact events for these currencies
        currencies_to_check = [base_currency, quote_currency]
        avoidance_hours = self.parameters['news_avoidance_hours']
        
        for currency in currencies_to_check:
            if currency in self.fundamental_events:
                for event_time, impact in self.fundamental_events[currency]:
                    # Check if event is upcoming within avoidance window
                    if current_time <= event_time <= current_time + timedelta(hours=avoidance_hours) and impact == 'high':
                        logger.info(f"Avoiding {symbol} due to upcoming high-impact {currency} event at {event_time}")
                        return True
        
        return False
    
    def _calculate_trend_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicators for the trend timeframe.
        
        Args:
            data: OHLCV DataFrame (trend timeframe)
            
        Returns:
            Dictionary of trend indicators
        """
        indicators = {}
        
        # Get close prices
        close = data['close']
        high = data['high']
        low = data['low']
        
        # Calculate moving average on the daily timeframe
        ma_period = self.parameters['daily_ma_period']
        indicators['daily_ma'] = close.rolling(window=ma_period).mean()
        
        # Calculate daily ATR
        atr_period = self.parameters['daily_atr_period']
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        indicators['daily_atr'] = tr.rolling(window=atr_period).mean()
        
        # Determine trend direction based on price vs MA
        indicators['price_vs_ma'] = close.iloc[-1] / indicators['daily_ma'].iloc[-1] - 1
        indicators['ma_slope'] = (indicators['daily_ma'].iloc[-1] / indicators['daily_ma'].iloc[-5] - 1) * 100
        
        # Trend strength
        indicators['trend_strength'] = abs(indicators['ma_slope'])
        
        # Classify trend
        if indicators['price_vs_ma'] > 0.001 and indicators['ma_slope'] > 0.05:  # 0.1% above MA, 0.05% upward slope
            indicators['trend_direction'] = 'up'
        elif indicators['price_vs_ma'] < -0.001 and indicators['ma_slope'] < -0.05:  # 0.1% below MA, 0.05% downward slope
            indicators['trend_direction'] = 'down'
        else:
            indicators['trend_direction'] = 'sideways'
        
        return indicators
    
    def _calculate_swing_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicators for the primary timeframe.
        
        Args:
            data: OHLCV DataFrame (primary timeframe)
            
        Returns:
            Dictionary of swing indicators
        """
        indicators = {}
        
        # Get close prices
        close = data['close']
        
        # Calculate RSI
        rsi_period = self.parameters['rsi_period']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        
        # Avoid division by zero
        loss = loss.replace(0, 0.00001)
        
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        fast_period = self.parameters['macd_fast']
        slow_period = self.parameters['macd_slow']
        signal_period = self.parameters['macd_signal']
        
        fast_ema = close.ewm(span=fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=slow_period, adjust=False).mean()
        indicators['macd'] = fast_ema - slow_ema
        indicators['macd_signal'] = indicators['macd'].ewm(span=signal_period, adjust=False).mean()
        indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # Calculate moving average for H4 timeframe
        h4_ma_period = self.parameters['h4_ma_period']
        indicators['h4_ma'] = close.rolling(window=h4_ma_period).mean()
        
        # Calculate distance to support and resistance
        # This would normally use the support/resistance levels we calculated earlier
        # Here we'll have placeholder values
        indicators['distance_to_nearest_support'] = 100  # pips
        indicators['distance_to_nearest_resistance'] = 100  # pips
        
        return indicators
    
    def _calculate_entry_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicators for entry timing (entry timeframe).
        
        Args:
            data: OHLCV DataFrame (entry timeframe)
            
        Returns:
            Dictionary of entry indicators
        """
        indicators = {}
        
        # Get close prices
        close = data['close']
        
        # Calculate H1 MA
        h1_ma_period = self.parameters['h1_ma_period']
        indicators['h1_ma'] = close.rolling(window=h1_ma_period).mean()
        
        # Calculate momentum
        indicators['momentum'] = close.diff(3)
        indicators['momentum_pct'] = indicators['momentum'] / close.shift(3) * 100
        
        # Recent price action (last few candles)
        indicators['recent_bullish'] = (close.iloc[-1] > close.iloc[-2] > close.iloc[-3])
        indicators['recent_bearish'] = (close.iloc[-1] < close.iloc[-2] < close.iloc[-3])
        
        # Entry readiness
        indicators['entry_ready'] = False
        
        # Price position relative to H1 MA
        indicators['price_vs_h1_ma'] = close.iloc[-1] - indicators['h1_ma'].iloc[-1]
        indicators['h1_ma_slope'] = indicators['h1_ma'].diff()
        
        # Entry qualification
        if indicators['momentum_pct'].iloc[-1] > 0 and indicators['h1_ma_slope'].iloc[-1] > 0:
            indicators['long_entry_quality'] = 'good'
        elif indicators['momentum_pct'].iloc[-1] > 0:
            indicators['long_entry_quality'] = 'moderate'
        else:
            indicators['long_entry_quality'] = 'poor'
            
        if indicators['momentum_pct'].iloc[-1] < 0 and indicators['h1_ma_slope'].iloc[-1] < 0:
            indicators['short_entry_quality'] = 'good'
        elif indicators['momentum_pct'].iloc[-1] < 0:
            indicators['short_entry_quality'] = 'moderate'
        else:
            indicators['short_entry_quality'] = 'poor'
        
        return indicators
    
    def _evaluate_swing_opportunity(self, symbol: str, ohlcv: pd.DataFrame,
                                  trend_indicators: Dict[str, Any],
                                  primary_indicators: Dict[str, Any],
                                  entry_indicators: Dict[str, Any],
                                  current_time: datetime) -> Optional[Signal]:
        """
        Evaluate market conditions for swing trading opportunities.
        
        Args:
            symbol: Currency pair symbol
            ohlcv: OHLCV DataFrame (primary timeframe)
            trend_indicators: Indicators from trend timeframe
            primary_indicators: Indicators from primary timeframe
            entry_indicators: Indicators from entry timeframe
            current_time: Current timestamp
            
        Returns:
            Signal object if opportunity found, None otherwise
        """
        # Get current price
        current_price = ohlcv['close'].iloc[-1]
        
        # Support and resistance levels (if available)
        support_levels = self.support_resistance.get(symbol, {}).get('support', [])
        resistance_levels = self.support_resistance.get(symbol, {}).get('resistance', [])
        
        # Get key indicator values
        trend_direction = trend_indicators['trend_direction']
        trend_strength = trend_indicators['trend_strength']
        rsi = primary_indicators['rsi'].iloc[-1]
        macd = primary_indicators['macd'].iloc[-1]
        macd_signal = primary_indicators['macd_signal'].iloc[-1]
        macd_histogram = primary_indicators['macd_histogram'].iloc[-1]
        
        # Entry quality
        long_entry_quality = entry_indicators.get('long_entry_quality', 'poor')
        short_entry_quality = entry_indicators.get('short_entry_quality', 'poor')
        
        # Initialize signal variables
        signal_type = SignalType.FLAT
        confidence = 0.0
        
        # Strategy for long swing trades
        long_conditions = [
            # Trend alignment (daily timeframe showing uptrend)
            trend_direction == 'up',
            trend_strength > 0.1,  # Minimum trend strength
            
            # Momentum indicators
            rsi > 40 and rsi < 70,  # Not overbought or oversold
            macd > macd_signal,     # MACD crossover or above signal
            macd_histogram > 0,     # Positive MACD histogram
            
            # Entry timing
            long_entry_quality in ['good', 'moderate'],
            
            # Support proximity (if available)
            len(support_levels) > 0
        ]
        
        # Strategy for short swing trades
        short_conditions = [
            # Trend alignment (daily timeframe showing downtrend)
            trend_direction == 'down',
            trend_strength > 0.1,  # Minimum trend strength
            
            # Momentum indicators
            rsi < 60 and rsi > 30,  # Not overbought or oversold
            macd < macd_signal,     # MACD crossover or below signal
            macd_histogram < 0,     # Negative MACD histogram
            
            # Entry timing
            short_entry_quality in ['good', 'moderate'],
            
            # Resistance proximity (if available)
            len(resistance_levels) > 0
        ]
        
        # Count true conditions
        long_count = sum(long_conditions)
        short_count = sum(short_conditions)
        
        # Set minimum threshold for signal generation
        min_conditions = 5
        
        # Check if we have enough conditions for a signal
        if long_count >= min_conditions and long_count > short_count:
            signal_type = SignalType.LONG
            # Scale confidence based on conditions met and entry quality
            base_confidence = 0.6 + (long_count - min_conditions) * 0.05
            quality_boost = 0.1 if long_entry_quality == 'good' else 0.0
            confidence = base_confidence + quality_boost
            
        elif short_count >= min_conditions and short_count > long_count:
            signal_type = SignalType.SHORT
            # Scale confidence based on conditions met and entry quality
            base_confidence = 0.6 + (short_count - min_conditions) * 0.05
            quality_boost = 0.1 if short_entry_quality == 'good' else 0.0
            confidence = base_confidence + quality_boost
        
        # If confidence is high enough, create a signal
        if confidence >= 0.6:
            # Calculate stop loss and take profit in pips
            stop_loss_pips = self.parameters['stop_loss_pips']
            take_profit_pips = self.parameters['profit_target_pips']
            
            # Adjust based on ATR if available
            if 'daily_atr' in trend_indicators:
                daily_atr_pips = trend_indicators['daily_atr'].iloc[-1] / self.parameters['pip_value']
                # Use ATR-based stops if larger than minimum
                atr_stop = daily_atr_pips * 0.8  # 80% of daily ATR
                if atr_stop > stop_loss_pips:
                    stop_loss_pips = atr_stop
                    # Maintain minimum risk-reward ratio
                    take_profit_pips = stop_loss_pips * self.parameters['min_risk_reward']
            
            # Round to whole pips
            stop_loss_pips = round(stop_loss_pips)
            take_profit_pips = round(take_profit_pips)
            
            # Convert pips to price
            pip_value = self.parameters['pip_value']
            
            # Set entry, stop loss and take profit prices
            entry_price = current_price
            
            if signal_type == SignalType.LONG:
                stop_loss = entry_price - (stop_loss_pips * pip_value)
                take_profit = entry_price + (take_profit_pips * pip_value)
            else:  # SHORT
                stop_loss = entry_price + (stop_loss_pips * pip_value)
                take_profit = entry_price - (take_profit_pips * pip_value)
            
            # Risk-reward ratio
            risk_reward_ratio = take_profit_pips / stop_loss_pips
            
            # Check minimum risk-reward
            if risk_reward_ratio < self.parameters['min_risk_reward']:
                logger.debug(f"Rejecting {signal_type.name} signal for {symbol} due to poor risk-reward ratio: {risk_reward_ratio:.2f}")
                return None
            
            # Create the signal
            signal = Signal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                timestamp=ohlcv.index[-1],
                metadata={
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'trend_direction': trend_direction,
                    'trend_strength': trend_strength,
                    'rsi': rsi,
                    'macd_histogram': macd_histogram,
                    'risk_reward_ratio': risk_reward_ratio,
                    'entry_quality': long_entry_quality if signal_type == SignalType.LONG else short_entry_quality,
                    'stop_loss_pips': stop_loss_pips,
                    'take_profit_pips': take_profit_pips,
                    'trailing_stop': self.parameters['trailing_stop_activation'],
                    'max_holding_days': self.parameters['max_holding_days']
                }
            )
            
            return signal
        
        return None
    
    def update(self, data: Dict[str, pd.DataFrame], current_time: datetime, account_size: float = 10000.0):
        """
        Update the strategy with new data and generate signals.
        
        Args:
            data: Dictionary mapping symbols to OHLCV DataFrames
            current_time: Current timestamp
            account_size: Current account size in base currency
            
        Returns:
            Dictionary of signals
        """
        # Generate signals
        signals = self.generate_signals(data, current_time)
        
        # Manage active swing trades
        self._manage_active_trades(data, current_time, account_size)
        
        return signals
    
    def _manage_active_trades(self, data: Dict[str, pd.DataFrame], current_time: datetime, account_size: float):
        """
        Manage active swing trades (partial take profits, trailing stops, etc.).
        
        Args:
            data: Dictionary mapping symbols to OHLCV DataFrames
            current_time: Current timestamp
            account_size: Current account size in base currency
        """
        symbols_to_remove = []
        
        # Process each active trade
        for symbol, trade_data in self.active_swing_trades.items():
            # Skip if the symbol is not in the data
            if symbol not in data:
                continue
            
            # Get current price
            current_price = data[symbol]['close'].iloc[-1]
            
            # Calculate time held
            days_held = (current_time - trade_data['entry_time']).days
            
            # Check if we've exceeded max holding period
            if days_held > self.parameters['max_holding_days']:
                # Emit exit event
                self.emit_strategy_event({
                    'symbol': symbol,
                    'signal_type': 'EXIT',
                    'reason': 'Max holding period reached',
                    'days_held': days_held,
                    'entry_price': trade_data['entry_price'],
                    'exit_price': current_price
                })
                
                symbols_to_remove.append(symbol)
                continue
            
            # Handle long positions
            if trade_data['direction'] == SignalType.LONG:
                # Check for partial take profit
                if self.parameters['partial_take_profit'] and not trade_data['partial_taken']:
                    partial_tp_price = trade_data['entry_price'] + (self.parameters['partial_tp_threshold'] * self.parameters['pip_value'])
                    if current_price >= partial_tp_price:
                        # Emit partial take profit event
                        self.emit_strategy_event({
                            'symbol': symbol,
                            'signal_type': 'PARTIAL_TP',
                            'percentage': self.parameters['partial_tp_percentage'],
                            'entry_price': trade_data['entry_price'],
                            'exit_price': current_price,
                            'pips_gained': (current_price - trade_data['entry_price']) / self.parameters['pip_value']
                        })
                        
                        trade_data['partial_taken'] = True
                
                # Check for trailing stop activation
                if current_price >= trade_data['entry_price'] + (self.parameters['trailing_stop_activation'] * self.parameters['pip_value']):
                    # Calculate new stop loss based on trailing distance
                    new_stop = current_price - (self.parameters['trailing_stop_distance'] * self.parameters['pip_value'])
                    
                    # Only update if it would raise the stop loss
                    if new_stop > trade_data['stop_loss']:
                        # Emit trailing stop update event
                        self.emit_strategy_event({
                            'symbol': symbol,
                            'signal_type': 'TRAILING_STOP_UPDATE',
                            'old_stop': trade_data['stop_loss'],
                            'new_stop': new_stop,
                            'price': current_price
                        })
                        
                        trade_data['stop_loss'] = new_stop
            
            # Handle short positions
            elif trade_data['direction'] == SignalType.SHORT:
                # Check for partial take profit
                if self.parameters['partial_take_profit'] and not trade_data['partial_taken']:
                    partial_tp_price = trade_data['entry_price'] - (self.parameters['partial_tp_threshold'] * self.parameters['pip_value'])
                    if current_price <= partial_tp_price:
                        # Emit partial take profit event
                        self.emit_strategy_event({
                            'symbol': symbol,
                            'signal_type': 'PARTIAL_TP',
                            'percentage': self.parameters['partial_tp_percentage'],
                            'entry_price': trade_data['entry_price'],
                            'exit_price': current_price,
                            'pips_gained': (trade_data['entry_price'] - current_price) / self.parameters['pip_value']
                        })
                        
                        trade_data['partial_taken'] = True
                
                # Check for trailing stop activation
                if current_price <= trade_data['entry_price'] - (self.parameters['trailing_stop_activation'] * self.parameters['pip_value']):
                    # Calculate new stop loss based on trailing distance
                    new_stop = current_price + (self.parameters['trailing_stop_distance'] * self.parameters['pip_value'])
                    
                    # Only update if it would lower the stop loss
                    if new_stop < trade_data['stop_loss']:
                        # Emit trailing stop update event
                        self.emit_strategy_event({
                            'symbol': symbol,
                            'signal_type': 'TRAILING_STOP_UPDATE',
                            'old_stop': trade_data['stop_loss'],
                            'new_stop': new_stop,
                            'price': current_price
                        })
                        
                        trade_data['stop_loss'] = new_stop
        
        # Remove trades that have been closed
        for symbol in symbols_to_remove:
            if symbol in self.active_swing_trades:
                del self.active_swing_trades[symbol]
                del self.current_signals[symbol]
    
    def get_market_regime_compatibility(self, regime: MarketRegime) -> float:
        """
        Get the compatibility score of this strategy with the given market regime.
        
        Args:
            regime: Market regime to evaluate
            
        Returns:
            Compatibility score (0.0 to 1.0)
        """
        # Swing trading compatibility with different market regimes
        compatibility_scores = {
            MarketRegime.BULL_TREND: 0.85,       # Excellent in established uptrends
            MarketRegime.BEAR_TREND: 0.80,       # Very good in established downtrends
            MarketRegime.CONSOLIDATION: 0.35,    # Poor in tight ranges (needs room to swing)
            MarketRegime.HIGH_VOLATILITY: 0.70,  # Good with volatility but needs direction
            MarketRegime.LOW_VOLATILITY: 0.50,   # Moderate - can work but needs patience
            MarketRegime.UNKNOWN: 0.50           # Moderate default
        }
        
        return compatibility_scores.get(regime, 0.5)
    
    def optimize_for_market_regime(self, regime: MarketRegime):
        """
        Optimize strategy parameters for the given market regime.
        
        Args:
            regime: Market regime to optimize for
        """
        # Start with base parameters
        optimized_params = {}
        
        # Optimize for Bull Trend
        if regime == MarketRegime.BULL_TREND:
            optimized_params = {
                'profit_target_pips': 120,        # Larger profit targets in trending markets
                'stop_loss_pips': 60,             # Wider stops for trend continuation
                'trailing_stop_activation': 70,   # Activate trail after decent move
                'trailing_stop_distance': 40,     # More room in trending markets
                'partial_tp_threshold': 85,       # Take partials later
                'daily_ma_period': 50,            # Longer MA for trend confirmation
                'h4_ma_period': 20,              # Standard for trends
                'h1_ma_period': 10               # Standard for entries
            }
        
        # Optimize for Bear Trend
        elif regime == MarketRegime.BEAR_TREND:
            optimized_params = {
                'profit_target_pips': 130,        # Slightly larger targets (bear markets move faster)
                'stop_loss_pips': 65,             # Slightly wider stops for volatility
                'trailing_stop_activation': 75,   # Activate trail after good move
                'trailing_stop_distance': 45,     # More room for volatile moves
                'partial_tp_threshold': 90,       # Take partials later
                'daily_ma_period': 50,            # Longer MA for trend confirmation
                'h4_ma_period': 18,              # Slightly shorter for faster trends
                'h1_ma_period': 8                # Shorter for faster entries
            }
        
        # Optimize for Consolidation
        elif regime == MarketRegime.CONSOLIDATION:
            optimized_params = {
                'profit_target_pips': 60,         # Smaller targets in ranges
                'stop_loss_pips': 40,             # Tight stops to minimize drawdowns
                'trailing_stop_activation': 30,   # Early trail activation
                'trailing_stop_distance': 20,     # Tight trail in ranges
                'partial_tp_threshold': 40,       # Take partials earlier
                'daily_ma_period': 20,            # Shorter MA in ranges
                'h4_ma_period': 10,              # Shorter for ranges
                'h1_ma_period': 5                # Shorter for quicker entries
            }
        
        # Optimize for High Volatility
        elif regime == MarketRegime.HIGH_VOLATILITY:
            optimized_params = {
                'profit_target_pips': 150,        # Larger targets for volatile moves
                'stop_loss_pips': 75,             # Wider stops for volatility
                'trailing_stop_activation': 80,   # Later trail activation
                'trailing_stop_distance': 50,     # Wider trail for volatility
                'partial_tp_threshold': 100,      # Take partials later
                'daily_ma_period': 30,            # Moderate MA for volatility
                'h4_ma_period': 15,              # Moderate for volatility
                'h1_ma_period': 7                # Moderate for entries
            }
        
        # Optimize for Low Volatility
        elif regime == MarketRegime.LOW_VOLATILITY:
            optimized_params = {
                'profit_target_pips': 80,         # Moderate targets for slow markets
                'stop_loss_pips': 45,             # Moderate stops
                'trailing_stop_activation': 50,   # Standard trail activation
                'trailing_stop_distance': 30,     # Standard trail distance
                'partial_tp_threshold': 60,       # Standard partial take profit
                'daily_ma_period': 40,            # Longer MA for slow markets
                'h4_ma_period': 18,              # Standard for slow markets
                'h1_ma_period': 9                # Standard for entries
            }
        
        # Update parameters if optimized_params is not empty
        if optimized_params:
            for key, value in optimized_params.items():
                self.parameters[key] = value
            
            logger.info(f"Optimized {self.name} for {regime.name} market regime")
    
    def emit_strategy_event(self, event_data: Dict[str, Any]):
        """
        Emit a strategy event to the event bus.
        
        Args:
            event_data: Event data to emit
        """
        # Add strategy name
        event_data['strategy_name'] = self.name
        
        # Add timestamp if not present
        if 'timestamp' not in event_data:
            event_data['timestamp'] = datetime.now()
        
        # Emit the event
        event = Event(EventType.STRATEGY_SIGNAL, event_data)
        self.event_bus.publish(event)
