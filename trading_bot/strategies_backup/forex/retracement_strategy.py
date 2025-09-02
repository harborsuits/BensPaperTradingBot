#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Retracement (Pullback) Trading Strategy

This strategy focuses on identifying and trading retracements/pullbacks in established trends.
It identifies key Fibonacci retracement levels, trend strength, and optimal entry points
during temporary price corrections against the primary trend direction.

Key Features:
1. Fibonacci retracement level identification
2. Trend strength confirmation
3. Multiple timeframe analysis
4. Pullback strength evaluation
5. Entry/exit timing optimization
6. Regime-aware parameter adjustment

Author: Ben Dickinson
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from enum import Enum
from datetime import datetime, timedelta
import json

from trading_bot.strategies.base.forex_base import ForexBaseStrategy
from trading_bot.utils.event_bus import EventBus
from trading_bot.utils.technical_indicators import (
    calculate_atr, calculate_rsi, calculate_macd, 
    calculate_bollinger_bands, calculate_stochastic
)
from trading_bot.strategies.strategy_template import (
    Strategy, SignalType, PositionSizing, OrderStatus, 
    TradeDirection, MarketRegime, TimeFrame
)

logger = logging.getLogger(__name__)

class RetracementLevel(Enum):
    """Fibonacci retracement levels."""
    LEVEL_236 = 0.236
    LEVEL_382 = 0.382
    LEVEL_500 = 0.500
    LEVEL_618 = 0.618
    LEVEL_786 = 0.786

class TrendStrength(Enum):
    """Classification of trend strength."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"

class ForexRetracementStrategy(ForexBaseStrategy):
    """
    Forex Retracement (Pullback) Trading Strategy
    
    This strategy identifies retracements in established trends and enters trades
    when price pulls back to key Fibonacci levels with confirmation signals.
    """
    
    def __init__(self,
                 parameters: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the Retracement Strategy.
        
        Args:
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters
        default_params = {
            # Trend identification
            'trend_lookback_period': 50,     # Bars to identify trend
            'short_ma_period': 20,           # Short MA for trend direction
            'long_ma_period': 50,            # Long MA for trend direction
            'adx_period': 14,                # ADX period for trend strength
            'adx_threshold': 25,             # Minimum ADX for strong trend
            
            # Retracement levels to consider
            'fib_levels': [0.382, 0.5, 0.618, 0.786],
            'primary_fib_level': 0.618,      # Primary level to focus on
            'level_tolerance': 0.05,         # Tolerance around levels (5%)
            
            # Entry/exit parameters
            'min_candles_in_pullback': 2,    # Minimum candles in pullback 
            'max_candles_in_pullback': 10,   # Maximum candles in pullback
            'stop_loss_atr_multiple': 1.5,   # ATR multiple for stop loss
            'take_profit_atr_multiple': 2.5, # ATR multiple for take profit
            'use_fibonacci_extensions': True, # Use fib extensions for TP
            'extension_levels': [1.27, 1.618, 2.0],  # Fibonacci extension levels
            
            # Confirmation indicators
            'use_rsi': True,                 # Use RSI for confirmation
            'rsi_period': 14,                # RSI period
            'rsi_oversold': 30,              # RSI oversold threshold
            'rsi_overbought': 70,            # RSI overbought threshold
            
            'use_macd': True,                # Use MACD for confirmation
            'macd_fast': 12,                 # MACD fast period
            'macd_slow': 26,                 # MACD slow period
            'macd_signal': 9,                # MACD signal period
            
            'use_stochastic': True,          # Use stochastic for confirmation
            'stoch_k_period': 14,            # Stochastic K period
            'stoch_d_period': 3,             # Stochastic D period
            'stoch_oversold': 20,            # Stochastic oversold threshold
            'stoch_overbought': 80,          # Stochastic overbought threshold
            
            # Risk management
            'risk_per_trade': 0.01,          # 1% risk per trade
            'max_trades_per_trend': 2,       # Maximum trades in one trend
            
            # Advanced
            'use_swing_points': True,        # Use swing points for level calculation
            'pullback_min_strength': 0.4,    # Minimum pullback strength (0-1)
            
            # Multi-timeframe analysis
            'use_mtf_confirmation': True,    # Use multiple timeframes
            'confirmation_timeframes': ['15M', '1H', '4H'],  # Timeframes to check
            'mtf_agreement_threshold': 0.7,  # Required agreement between timeframes
            
            # Optimization
            'optimize_per_pair': True,       # Optimize separately for each pair
            'regime_specific_params': True,  # Use regime-specific parameters
        }
        
        # Initialize base class
        default_metadata = {
            'name': 'Forex Retracement Strategy',
            'description': 'Trading strategy that identifies and enters on retracements in established trends',
            'version': '1.0.0',
            'author': 'Ben Dickinson',
            'type': 'forex_retracement',
            'tags': ['forex', 'retracement', 'pullback', 'fibonacci', 'trend_following'],
            'preferences': {
                'timeframes': ['15M', '1H', '4H', 'D'],
                'default_timeframe': '1H'
            }
        }
        
        # Update metadata if provided
        if metadata:
            default_metadata.update(metadata)
            
        super().__init__('forex_retracement', parameters, default_metadata)
        
        # Merge provided parameters with defaults
        if parameters:
            for key, value in parameters.items():
                if key in default_params:
                    self.parameters[key] = value
        else:
            self.parameters = default_params
        
        # State tracking
        self.identified_trends = {}  # symbol -> trend info
        self.active_retracements = {}  # symbol -> retracement info
        self.fibonacci_levels = {}  # symbol -> calculated fib levels
        self.mtf_data = {}  # symbol -> timeframe -> data
        
        # Register with event bus
        EventBus.get_instance().register(self)
    
    def _identify_trend(self, data: pd.DataFrame) -> Tuple[TradeDirection, TrendStrength, Dict[str, Any]]:
        """
        Identify the current market trend and its strength.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Tuple of (trend_direction, trend_strength, trend_details)
        """
        if len(data) < self.parameters['trend_lookback_period']:
            return TradeDirection.FLAT, TrendStrength.NONE, {}
        
        # Use moving averages for trend direction
        short_period = self.parameters['short_ma_period']
        long_period = self.parameters['long_ma_period']
        
        # Calculate moving averages
        data['short_ma'] = data['close'].rolling(window=short_period).mean()
        data['long_ma'] = data['close'].rolling(window=long_period).mean()
        
        # Calculate ADX for trend strength
        adx_period = self.parameters['adx_period']
        adx_threshold = self.parameters['adx_threshold']
        
        # Add ADX calculation (typically in technical_indicators.py)
        # For now, we'll simulate it with a placeholder calculation
        data['adx'] = data['close'].rolling(window=adx_period).std() * 10
        adx_value = data['adx'].iloc[-1]
        
        # Get recent MA values
        recent_short_ma = data['short_ma'].dropna().iloc[-5:]
        recent_long_ma = data['long_ma'].dropna().iloc[-5:]
        
        # Determine trend direction
        if recent_short_ma.iloc[-1] > recent_long_ma.iloc[-1]:
            # Check if short MA is consistently above long MA
            ma_diff = recent_short_ma - recent_long_ma
            if all(ma_diff > 0):
                trend_direction = TradeDirection.LONG
            else:
                trend_direction = TradeDirection.FLAT  # Mixed signals
        elif recent_short_ma.iloc[-1] < recent_long_ma.iloc[-1]:
            # Check if short MA is consistently below long MA
            ma_diff = recent_long_ma - recent_short_ma
            if all(ma_diff > 0):
                trend_direction = TradeDirection.SHORT
            else:
                trend_direction = TradeDirection.FLAT  # Mixed signals
        else:
            trend_direction = TradeDirection.FLAT
        
        # Determine trend strength based on ADX
        if adx_value >= adx_threshold * 1.5:
            trend_strength = TrendStrength.STRONG
        elif adx_value >= adx_threshold:
            trend_strength = TrendStrength.MODERATE
        elif adx_value >= adx_threshold * 0.7:
            trend_strength = TrendStrength.WEAK
        else:
            trend_strength = TrendStrength.NONE
            trend_direction = TradeDirection.FLAT  # No real trend
        
        # Calculate additional trend metrics
        recent_closes = data['close'].iloc[-self.parameters['trend_lookback_period']:]
        trend_slope = (recent_closes.iloc[-1] - recent_closes.iloc[0]) / len(recent_closes)
        trend_duration = len(recent_closes)
        
        # Package all trend details
        trend_details = {
            'adx': adx_value,
            'short_ma': recent_short_ma.iloc[-1],
            'long_ma': recent_long_ma.iloc[-1],
            'slope': trend_slope,
            'duration': trend_duration,
            'last_price': data['close'].iloc[-1],
            'time': data.index[-1]
        }
        
        return trend_direction, trend_strength, trend_details
    
    def _find_swing_points(self, data: pd.DataFrame, lookback: int = 5) -> Tuple[List[int], List[int]]:
        """
        Find swing high and swing low points in the price data.
        
        Args:
            data: OHLCV DataFrame
            lookback: Number of bars to look back/forward for swing detection
            
        Returns:
            Tuple of (swing_high_indices, swing_low_indices)
        """
        if len(data) < 2 * lookback + 1:
            return [], []
        
        swing_highs = []
        swing_lows = []
        
        # Find swing highs and lows
        for i in range(lookback, len(data) - lookback):
            # Check for swing high
            is_swing_high = True
            for j in range(1, lookback + 1):
                if data['high'].iloc[i] <= data['high'].iloc[i - j] or \
                   data['high'].iloc[i] <= data['high'].iloc[i + j]:
                    is_swing_high = False
                    break
                    
            if is_swing_high:
                swing_highs.append(i)
                
            # Check for swing low
            is_swing_low = True
            for j in range(1, lookback + 1):
                if data['low'].iloc[i] >= data['low'].iloc[i - j] or \
                   data['low'].iloc[i] >= data['low'].iloc[i + j]:
                    is_swing_low = False
                    break
                    
            if is_swing_low:
                swing_lows.append(i)
                
        return swing_highs, swing_lows
    
    def _calculate_fibonacci_levels(self, data: pd.DataFrame, trend_direction: TradeDirection) -> Dict[float, float]:
        """
        Calculate Fibonacci retracement levels based on the trend.
        
        Args:
            data: OHLCV DataFrame
            trend_direction: Detected trend direction
            
        Returns:
            Dictionary of Fibonacci level -> price
        """
        # Get Fibonacci levels from parameters
        fib_levels = self.parameters['fib_levels']
        
        if self.parameters['use_swing_points']:
            # Find swing points
            swing_highs, swing_lows = self._find_swing_points(data)
            
            if not swing_highs or not swing_lows:
                # Fallback if no swing points found
                if trend_direction == TradeDirection.LONG:
                    swing_low_idx = data['low'].idxmin()
                    swing_high_idx = data.index[-1]
                    swing_low = data.loc[swing_low_idx, 'low']
                    swing_high = data.iloc[-1]['high']
                else:
                    swing_high_idx = data['high'].idxmax()
                    swing_low_idx = data.index[-1]
                    swing_high = data.loc[swing_high_idx, 'high']
                    swing_low = data.iloc[-1]['low']
            else:
                if trend_direction == TradeDirection.LONG:
                    # Find most recent significant swing low before the current price
                    recent_swing_lows = [idx for idx in swing_lows if idx < len(data) - 5]
                    if recent_swing_lows:
                        swing_low_idx = recent_swing_lows[-1]
                        swing_low = data['low'].iloc[swing_low_idx]
                        # Find the highest high after this swing low
                        if swing_low_idx < len(data) - 1:
                            high_section = data['high'].iloc[swing_low_idx:].max()
                            swing_high = high_section
                        else:
                            swing_high = data['high'].iloc[-1]
                    else:
                        # Fallback
                        swing_low = data['low'].min()
                        swing_high = data['high'].iloc[-10:].max()
                else:  # SHORT trend
                    # Find most recent significant swing high before the current price
                    recent_swing_highs = [idx for idx in swing_highs if idx < len(data) - 5]
                    if recent_swing_highs:
                        swing_high_idx = recent_swing_highs[-1]
                        swing_high = data['high'].iloc[swing_high_idx]
                        # Find the lowest low after this swing high
                        if swing_high_idx < len(data) - 1:
                            low_section = data['low'].iloc[swing_high_idx:].min()
                            swing_low = low_section
                        else:
                            swing_low = data['low'].iloc[-1]
                    else:
                        # Fallback
                        swing_high = data['high'].max()
                        swing_low = data['low'].iloc[-10:].min()
        else:
            # Simple method - use highest high and lowest low in the lookback period
            lookback = self.parameters['trend_lookback_period']
            recent_data = data.iloc[-lookback:]
            
            if trend_direction == TradeDirection.LONG:
                swing_low = recent_data['low'].min()
                swing_high = recent_data['high'].max()
            else:  # SHORT trend
                swing_high = recent_data['high'].max()
                swing_low = recent_data['low'].min()
        
        # Calculate price range
        price_range = abs(swing_high - swing_low)
        
        # Calculate Fibonacci levels
        levels = {}
        
        if trend_direction == TradeDirection.LONG:
            # For uptrends, retracements move down from high to low
            for level in fib_levels:
                levels[level] = swing_high - (price_range * level)
                
            # Add swing points as key levels
            levels[0.0] = swing_high
            levels[1.0] = swing_low
                
            # Add extension levels if configured
            if self.parameters['use_fibonacci_extensions']:
                for ext in self.parameters['extension_levels']:
                    levels[-ext] = swing_high + (price_range * ext)
        else:  # SHORT trend
            # For downtrends, retracements move up from low to high
            for level in fib_levels:
                levels[level] = swing_low + (price_range * level)
                
            # Add swing points as key levels
            levels[0.0] = swing_low
            levels[1.0] = swing_high
                
            # Add extension levels if configured
            if self.parameters['use_fibonacci_extensions']:
                for ext in self.parameters['extension_levels']:
                    levels[-ext] = swing_low - (price_range * ext)
        
        return levels
    
    def _detect_retracement(self, data: pd.DataFrame, trend_direction: TradeDirection, 
                        fibonacci_levels: Dict[float, float]) -> Tuple[bool, float, str]:
        """
        Detect if price is currently in a retracement.
        
        Args:
            data: OHLCV DataFrame
            trend_direction: Current trend direction
            fibonacci_levels: Calculated Fibonacci levels
            
        Returns:
            Tuple of (is_retracement, retracement_strength, nearest_level)
        """
        if len(data) < 5:  # Need enough data
            return False, 0.0, ""
        
        # Get current price
        current_price = data['close'].iloc[-1]
        
        # Get primary fib level and tolerance
        primary_level = self.parameters['primary_fib_level']
        tolerance = self.parameters['level_tolerance']
        
        # Check if we're in a pullback
        if trend_direction == TradeDirection.LONG:
            # In uptrend, pullback is price moving down
            # Check if we've moved down from recent highs
            high_point = data['high'].iloc[-5:].max()
            pullback_pct = (high_point - current_price) / high_point
            
            # Must have pulled back enough but not too much
            min_pullback = self.parameters['pullback_min_strength']
            if pullback_pct < min_pullback:
                return False, pullback_pct, ""
                
            # Check if price is near any Fibonacci level
            nearest_level = None
            nearest_distance = float('inf')
            
            for level, price in fibonacci_levels.items():
                if level > 0 and level <= 1.0:  # Only consider retracement levels (not extensions)
                    distance = abs(current_price - price) / price
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_level = level
            
            # Check if we're close enough to a Fibonacci level
            if nearest_level is not None and nearest_distance <= tolerance:
                return True, pullback_pct, f"{nearest_level:.3f}"
                
        elif trend_direction == TradeDirection.SHORT:
            # In downtrend, pullback is price moving up
            # Check if we've moved up from recent lows
            low_point = data['low'].iloc[-5:].min()
            if low_point == 0:  # Avoid division by zero
                return False, 0.0, ""
                
            pullback_pct = (current_price - low_point) / low_point
            
            # Must have pulled back enough but not too much
            min_pullback = self.parameters['pullback_min_strength']
            if pullback_pct < min_pullback:
                return False, pullback_pct, ""
                
            # Check if price is near any Fibonacci level
            nearest_level = None
            nearest_distance = float('inf')
            
            for level, price in fibonacci_levels.items():
                if level > 0 and level <= 1.0:  # Only consider retracement levels (not extensions)
                    distance = abs(current_price - price) / price
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_level = level
            
            # Check if we're close enough to a Fibonacci level
            if nearest_level is not None and nearest_distance <= tolerance:
                return True, pullback_pct, f"{nearest_level:.3f}"
        
        return False, 0.0, ""
    
    def _check_confirmation_indicators(self, data: pd.DataFrame, trend_direction: TradeDirection) -> Tuple[bool, float, str]:
        """
        Check if confirmation indicators support a retracement entry.
        
        Args:
            data: OHLCV DataFrame
            trend_direction: Current trend direction
            
        Returns:
            Tuple of (confirmed, confidence, reason)
        """
        if len(data) < 50:  # Need enough data for indicators
            return False, 0.0, "Insufficient data for confirmation"
        
        confirmations = []
        confidence_scores = []
        reasons = []
        
        # Check RSI
        if self.parameters['use_rsi']:
            rsi_period = self.parameters['rsi_period']
            rsi = calculate_rsi(data, rsi_period).iloc[-1]
            
            if trend_direction == TradeDirection.LONG:
                # In uptrends, look for oversold RSI showing pullback ending
                if rsi <= self.parameters['rsi_oversold']:
                    confirmations.append(True)
                    confidence_scores.append(0.8)
                    reasons.append(f"RSI oversold ({rsi:.2f})")
                elif rsi <= 45:  # Moderately oversold
                    confirmations.append(True)
                    confidence_scores.append(0.6)
                    reasons.append(f"RSI moderately low ({rsi:.2f})")
                else:
                    confirmations.append(False)
                    confidence_scores.append(0.3)
            else:  # SHORT trend
                # In downtrends, look for overbought RSI showing pullback ending
                if rsi >= self.parameters['rsi_overbought']:
                    confirmations.append(True)
                    confidence_scores.append(0.8)
                    reasons.append(f"RSI overbought ({rsi:.2f})")
                elif rsi >= 55:  # Moderately overbought
                    confirmations.append(True)
                    confidence_scores.append(0.6)
                    reasons.append(f"RSI moderately high ({rsi:.2f})")
                else:
                    confirmations.append(False)
                    confidence_scores.append(0.3)
        
        # Check MACD
        if self.parameters['use_macd']:
            fast = self.parameters['macd_fast']
            slow = self.parameters['macd_slow']
            signal = self.parameters['macd_signal']
            
            macd_result = calculate_macd(data, fast, slow, signal)
            macd = macd_result['macd'].iloc[-1]
            signal = macd_result['signal'].iloc[-1]
            histogram = macd_result['histogram'].iloc[-1]
            
            # Look for MACD momentum in trend direction after pullback
            if trend_direction == TradeDirection.LONG:
                if histogram > 0 and histogram > macd_result['histogram'].iloc[-2]:
                    # Positive and increasing histogram in uptrend
                    confirmations.append(True)
                    confidence_scores.append(0.7)
                    reasons.append("MACD histogram turning positive")
                else:
                    confirmations.append(False)
                    confidence_scores.append(0.4)
            else:  # SHORT trend
                if histogram < 0 and histogram < macd_result['histogram'].iloc[-2]:
                    # Negative and decreasing histogram in downtrend
                    confirmations.append(True)
                    confidence_scores.append(0.7)
                    reasons.append("MACD histogram turning negative")
                else:
                    confirmations.append(False)
                    confidence_scores.append(0.4)
        
        # Check Stochastic
        if self.parameters['use_stochastic']:
            k_period = self.parameters['stoch_k_period']
            d_period = self.parameters['stoch_d_period']
            
            stoch = calculate_stochastic(data, k_period, d_period)
            k = stoch['K'].iloc[-1]
            d = stoch['D'].iloc[-1]
            
            if trend_direction == TradeDirection.LONG:
                # Look for stochastic exiting oversold in uptrend
                if k > d and k < 50 and k > stoch['K'].iloc[-2]:
                    confirmations.append(True)
                    confidence_scores.append(0.7)
                    reasons.append(f"Stochastic turning up ({k:.2f})")
                elif k <= self.parameters['stoch_oversold'] and k > d:
                    confirmations.append(True)
                    confidence_scores.append(0.8)
                    reasons.append(f"Stochastic exiting oversold ({k:.2f})")
                else:
                    confirmations.append(False)
                    confidence_scores.append(0.3)
            else:  # SHORT trend
                # Look for stochastic exiting overbought in downtrend
                if k < d and k > 50 and k < stoch['K'].iloc[-2]:
                    confirmations.append(True)
                    confidence_scores.append(0.7)
                    reasons.append(f"Stochastic turning down ({k:.2f})")
                elif k >= self.parameters['stoch_overbought'] and k < d:
                    confirmations.append(True)
                    confidence_scores.append(0.8)
                    reasons.append(f"Stochastic exiting overbought ({k:.2f})")
                else:
                    confirmations.append(False)
                    confidence_scores.append(0.3)
        
        # Calculate overall confirmation
        if not confirmations:
            return False, 0.0, "No confirmation indicators enabled"
            
        # Need majority of indicators to confirm
        confirmed_count = sum(1 for c in confirmations if c)
        
        if confirmed_count >= len(confirmations) / 2:
            # At least half of the indicators confirm
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            reason = "; ".join([r for r, c in zip(reasons, confirmations) if c])
            return True, avg_confidence, reason
        else:
            return False, 0.3, "Insufficient indicator confirmation"
    
    def _check_mtf_agreement(self, symbol: str, trend_direction: TradeDirection) -> float:
        """
        Check if multiple timeframes agree on the trend direction.
        
        Args:
            symbol: Currency pair symbol
            trend_direction: Current trend direction
            
        Returns:
            Agreement score (0.0 to 1.0)
        """
        if not self.parameters['use_mtf_confirmation'] or symbol not in self.mtf_data:
            return 0.5  # Neutral if not using MTF or no data
        
        # Count agreeing timeframes
        agreements = 0
        total = 0
        
        for tf, data in self.mtf_data[symbol].items():
            if len(data) < self.parameters['trend_lookback_period']:
                continue
                
            total += 1
            
            # Get trend on this timeframe
            tf_direction, _, _ = self._identify_trend(data)
            
            # Check if trend matches the primary timeframe
            if tf_direction == trend_direction:
                agreements += 1
        
        # Calculate agreement score
        if total == 0:
            return 0.5  # Neutral if no data
            
        return agreements / total
    
    def _prepare_mtf_data(self, symbol: str, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Prepare multi-timeframe data for analysis.
        
        Args:
            symbol: Currency pair symbol
            data: Primary timeframe data
            
        Returns:
            Dictionary of timeframe -> resampled data
        """
        if not self.parameters['use_mtf_confirmation']:
            return {}
            
        # Skip if not enough data
        if len(data) < 100:
            return {}
        
        # Ensure we have timestamp index
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning(f"Data for {symbol} doesn't have timestamp index")
            return {}
        
        results = {}
        
        # Resample to other timeframes
        for tf in self.parameters['confirmation_timeframes']:
            try:
                # Create resampled data
                freq = tf.replace('M', 'T') if 'M' in tf else tf  # Convert 15M to 15T for pandas
                resampled = data.resample(freq).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                if len(resampled) > 20:  # Ensure we have enough data
                    results[tf] = resampled
            except Exception as e:
                logger.error(f"Error resampling {symbol} to {tf}: {str(e)}")
        
        # Cache the results
        self.mtf_data[symbol] = results
        
        return results
        
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_time: pd.Timestamp) -> Dict[str, Any]:
        """
        Generate trading signals for retracement opportunities.
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            current_time: Current time
            
        Returns:
            Dictionary of signal information
        """
        signals = {}
        
        # Skip if not in an active trading session
        if not self.is_active_trading_session(current_time):
            return signals
            
        # Process each symbol
        for symbol, ohlcv in data.items():
            if len(ohlcv) < self.parameters['trend_lookback_period']:
                continue
                
            # Prepare multi-timeframe data if configured
            if self.parameters['use_mtf_confirmation']:
                self._prepare_mtf_data(symbol, ohlcv)
                
            # Step 1: Identify current trend
            trend_direction, trend_strength, trend_details = self._identify_trend(ohlcv)
            
            # Skip if no clear trend or very weak trend
            if trend_direction == TradeDirection.FLAT or trend_strength == TrendStrength.NONE:
                continue
                
            # Update trend information
            self.identified_trends[symbol] = {
                'direction': trend_direction,
                'strength': trend_strength,
                'details': trend_details,
                'time': current_time
            }
            
            # Step 2: Calculate Fibonacci levels
            fib_levels = self._calculate_fibonacci_levels(ohlcv, trend_direction)
            self.fibonacci_levels[symbol] = fib_levels
            
            # Step 3: Detect if price is in retracement
            is_retracement, retracement_strength, nearest_level = self._detect_retracement(
                ohlcv, trend_direction, fib_levels)
                
            if is_retracement:
                # Update active retracement info
                self.active_retracements[symbol] = {
                    'direction': trend_direction,
                    'strength': retracement_strength,
                    'level': nearest_level,
                    'start_time': current_time,
                    'start_price': ohlcv['close'].iloc[-1]
                }
                
                # Step 4: Check confirmation indicators
                confirmed, confidence, reason = self._check_confirmation_indicators(ohlcv, trend_direction)
                
                # Step 5: Check multi-timeframe agreement if configured
                mtf_agreement = self._check_mtf_agreement(symbol, trend_direction)
                
                # Only generate signal if indicators confirm and MTF agrees
                mtf_threshold = self.parameters['mtf_agreement_threshold']
                
                if confirmed and (not self.parameters['use_mtf_confirmation'] or mtf_agreement >= mtf_threshold):
                    # Create signal
                    current_price = ohlcv['close'].iloc[-1]
                    atr = calculate_atr(ohlcv, 14).iloc[-1]
                    
                    # Calculate stop loss and take profit
                    if trend_direction == TradeDirection.LONG:
                        # For long trades, stop loss below the retracement
                        stop_loss = current_price - (atr * self.parameters['stop_loss_atr_multiple'])
                        
                        # Take profit at next target
                        if self.parameters['use_fibonacci_extensions']:
                            # Use the first extension level
                            ext_level = self.parameters['extension_levels'][0]
                            if -ext_level in fib_levels:
                                take_profit = fib_levels[-ext_level]
                            else:
                                take_profit = current_price + (atr * self.parameters['take_profit_atr_multiple'])
                        else:
                            take_profit = current_price + (atr * self.parameters['take_profit_atr_multiple'])
                    else:
                        # For short trades, stop loss above the retracement
                        stop_loss = current_price + (atr * self.parameters['stop_loss_atr_multiple'])
                        
                        # Take profit at next target
                        if self.parameters['use_fibonacci_extensions']:
                            # Use the first extension level
                            ext_level = self.parameters['extension_levels'][0]
                            if -ext_level in fib_levels:
                                take_profit = fib_levels[-ext_level]
                            else:
                                take_profit = current_price - (atr * self.parameters['take_profit_atr_multiple'])
                        else:
                            take_profit = current_price - (atr * self.parameters['take_profit_atr_multiple'])
                    
                    # Create signal with details
                    signal_strength = (confidence * 0.6) + (mtf_agreement * 0.4)  # Weighted score
                    
                    # Generate trade id
                    trade_id = f"retracement_{symbol}_{current_time.strftime('%Y%m%d%H%M%S')}"
                    
                    signals[symbol] = {
                        'id': trade_id,
                        'symbol': symbol,
                        'direction': trend_direction,
                        'strength': signal_strength,
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'retracement_level': nearest_level,
                        'trend_strength': trend_strength.name,
                        'confirmation_reason': reason,
                        'mtf_agreement': mtf_agreement,
                        'time': current_time
                    }
                    
                    # Publish event for the signal
                    EventBus.get_instance().publish('retracement_signal', {
                        'id': trade_id,
                        'symbol': symbol,
                        'direction': trend_direction.name,
                        'retracement_level': nearest_level,
                        'strength': signal_strength,
                        'time': current_time.isoformat()
                    })
        
        return signals
    
    def get_position_size(self, symbol: str, entry_price: float, stop_loss: float, 
                          account_balance: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol: Currency pair symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            account_balance: Current account balance
            
        Returns:
            Position size in units
        """
        # Get risk per trade
        risk_percent = self.parameters['risk_per_trade']
        risk_amount = account_balance * risk_percent
        
        # Calculate stop loss distance in pips
        pip_value = self.get_pip_value(symbol)
        stop_distance = abs(entry_price - stop_loss) / pip_value
        
        # Calculate position size
        position_size = self.calculate_position_size_pips(symbol, entry_price, stop_distance, risk_amount)
        
        return position_size
    
    def get_regime_compatibility_score(self, regime: MarketRegime) -> float:
        """
        Calculate compatibility score with the given market regime.
        
        Args:
            regime: Market regime to check compatibility with
            
        Returns:
            Compatibility score (0.0 to 1.0)
        """
        # Retracement strategy works well in certain regimes
        regime_scores = {
            MarketRegime.TRENDING: 0.85,        # Excellent in trending markets
            MarketRegime.RANGING: 0.60,         # Can work in ranges but less optimal
            MarketRegime.VOLATILE: 0.50,        # Challenging in volatile markets
            MarketRegime.BREAKOUT: 0.70,        # Can work after breakouts for continuation
            MarketRegime.LOW_VOLATILITY: 0.75,  # Good in low volatility with clear pullbacks
            MarketRegime.HIGH_VOLATILITY: 0.40, # Difficult in high volatility
            MarketRegime.BULLISH: 0.80,         # Great in bullish markets
            MarketRegime.BEARISH: 0.80,         # Great in bearish markets
            MarketRegime.CONSOLIDATION: 0.50,   # Suboptimal in consolidation
            MarketRegime.UNKNOWN: 0.60          # Average score as fallback
        }
        
        return regime_scores.get(regime, 0.5)
    
    def get_optimal_timeframe(self, regime: MarketRegime) -> str:
        """
        Get the optimal timeframe for this strategy under the given regime.
        
        Args:
            regime: Current market regime
            
        Returns:
            Optimal timeframe string
        """
        # Retracement strategy works better on certain timeframes depending on regime
        regime_timeframes = {
            MarketRegime.TRENDING: '1H',       # Hourly for trending markets
            MarketRegime.RANGING: '4H',        # Higher timeframe for ranges
            MarketRegime.VOLATILE: '4H',       # Higher timeframe for volatility
            MarketRegime.BREAKOUT: '15M',      # Shorter for breakouts
            MarketRegime.LOW_VOLATILITY: '1H', # Hourly for low volatility
            MarketRegime.HIGH_VOLATILITY: '4H', # Higher for high volatility
            MarketRegime.BULLISH: '1H',        # Standard for bullish
            MarketRegime.BEARISH: '1H',        # Standard for bearish
            MarketRegime.CONSOLIDATION: '4H',  # Higher for consolidation
            MarketRegime.UNKNOWN: '1H'         # Default
        }
        
        return regime_timeframes.get(regime, '1H')
    
    def optimize(self, data: Dict[str, pd.DataFrame], 
                market_regime: MarketRegime = MarketRegime.UNKNOWN) -> Dict[str, Any]:
        """
        Optimize strategy parameters for the given market regime.
        
        Args:
            data: Dictionary of symbol -> DataFrame with market data
            market_regime: Market regime to optimize for
            
        Returns:
            Optimized parameters dictionary
        """
        logger.info(f"Optimizing Retracement strategy for {market_regime} regime")
        
        # Define parameter ranges based on regime
        param_ranges = {}
        
        # Base ranges depending on market regime
        if market_regime == MarketRegime.TRENDING:
            param_ranges = {
                'trend_lookback_period': [40, 50, 60],
                'adx_threshold': [20, 25, 30],
                'primary_fib_level': [0.5, 0.618],
                'level_tolerance': [0.03, 0.05, 0.07],
                'stop_loss_atr_multiple': [1.2, 1.5, 1.8],
                'take_profit_atr_multiple': [2.0, 2.5, 3.0],
                'pullback_min_strength': [0.3, 0.4, 0.5],
                'mtf_agreement_threshold': [0.6, 0.7, 0.8]
            }
        elif market_regime == MarketRegime.RANGING:
            param_ranges = {
                'trend_lookback_period': [30, 40, 50],
                'adx_threshold': [15, 20, 25],
                'primary_fib_level': [0.382, 0.5],
                'level_tolerance': [0.05, 0.07, 0.1],
                'stop_loss_atr_multiple': [1.0, 1.2, 1.5],
                'take_profit_atr_multiple': [1.5, 2.0, 2.5],
                'pullback_min_strength': [0.2, 0.3, 0.4],
                'mtf_agreement_threshold': [0.6, 0.7, 0.8]
            }
        elif market_regime in [MarketRegime.VOLATILE, MarketRegime.HIGH_VOLATILITY]:
            param_ranges = {
                'trend_lookback_period': [50, 60, 70],
                'adx_threshold': [25, 30, 35],
                'primary_fib_level': [0.5, 0.618, 0.786],
                'level_tolerance': [0.07, 0.1, 0.12],
                'stop_loss_atr_multiple': [1.5, 2.0, 2.5],
                'take_profit_atr_multiple': [2.0, 2.5, 3.0],
                'pullback_min_strength': [0.4, 0.5, 0.6],
                'mtf_agreement_threshold': [0.7, 0.8, 0.9]
            }
        elif market_regime == MarketRegime.BREAKOUT:
            param_ranges = {
                'trend_lookback_period': [30, 40, 50],
                'adx_threshold': [25, 30, 35],
                'primary_fib_level': [0.382, 0.5],
                'level_tolerance': [0.05, 0.07, 0.1],
                'stop_loss_atr_multiple': [1.2, 1.5, 1.8],
                'take_profit_atr_multiple': [2.0, 2.5, 3.0],
                'pullback_min_strength': [0.3, 0.4, 0.5],
                'mtf_agreement_threshold': [0.7, 0.8, 0.9]
            }
        else:  # Default/unknown regime
            param_ranges = {
                'trend_lookback_period': [40, 50, 60],
                'adx_threshold': [20, 25, 30],
                'primary_fib_level': [0.5, 0.618],
                'level_tolerance': [0.05, 0.07],
                'stop_loss_atr_multiple': [1.2, 1.5, 1.8],
                'take_profit_atr_multiple': [2.0, 2.5, 3.0],
                'pullback_min_strength': [0.3, 0.4, 0.5],
                'mtf_agreement_threshold': [0.6, 0.7, 0.8]
            }
        
        # TODO: Implement grid search optimization
        # For now, return the middle value of each parameter range as a reasonable default
        optimized = {}
        for param, values in param_ranges.items():
            middle_idx = len(values) // 2
            optimized[param] = values[middle_idx]
            
        # Set appropriate fib levels based on regime
        if market_regime == MarketRegime.TRENDING:
            optimized['fib_levels'] = [0.382, 0.5, 0.618]
        elif market_regime == MarketRegime.RANGING:
            optimized['fib_levels'] = [0.382, 0.5, 0.618, 0.786]
        elif market_regime in [MarketRegime.VOLATILE, MarketRegime.HIGH_VOLATILITY]:
            optimized['fib_levels'] = [0.5, 0.618, 0.786]
        else:
            optimized['fib_levels'] = [0.382, 0.5, 0.618, 0.786]
            
        # Extension levels based on regime
        if market_regime == MarketRegime.TRENDING:
            optimized['extension_levels'] = [1.27, 1.618, 2.0]
        elif market_regime == MarketRegime.RANGING:
            optimized['extension_levels'] = [1.27, 1.618]
        elif market_regime in [MarketRegime.VOLATILE, MarketRegime.HIGH_VOLATILITY]:
            optimized['extension_levels'] = [1.27, 1.618, 2.0, 2.618]
        else:
            optimized['extension_levels'] = [1.27, 1.618, 2.0]
        
        logger.info(f"Optimization complete for Retracement strategy")
        
        return optimized
    
    def save_state(self, file_path: str) -> bool:
        """
        Save the current state of trends and retracements.
        
        Args:
            file_path: Path to save the state
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create serializable state
            state = {
                'identified_trends': {
                    symbol: {
                        'direction': trend['direction'].name,
                        'strength': trend['strength'].name,
                        'details': trend['details'],
                        'time': trend['time'].isoformat()
                    } for symbol, trend in self.identified_trends.items()
                },
                'active_retracements': {
                    symbol: {
                        'direction': retrace['direction'].name,
                        'strength': retrace['strength'],
                        'level': retrace['level'],
                        'start_time': retrace['start_time'].isoformat(),
                        'start_price': retrace['start_price']
                    } for symbol, retrace in self.active_retracements.items()
                },
                'fibonacci_levels': self.fibonacci_levels,
                'parameters': self.parameters,
                'saved_at': datetime.now().isoformat()
            }
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Saved Retracement strategy state to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving Retracement strategy state: {str(e)}")
            return False
    
    def load_state(self, file_path: str) -> bool:
        """
        Load a previously saved state.
        
        Args:
            file_path: Path to the state file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"State file does not exist: {file_path}")
                return False
                
            # Load from file
            with open(file_path, 'r') as f:
                state = json.load(f)
                
            # Restore parameters
            if 'parameters' in state:
                self.parameters.update(state['parameters'])
                
            # Restore fibonacci levels
            if 'fibonacci_levels' in state:
                self.fibonacci_levels = state['fibonacci_levels']
                
            # Restore identified trends
            if 'identified_trends' in state:
                self.identified_trends = {}
                for symbol, trend in state['identified_trends'].items():
                    self.identified_trends[symbol] = {
                        'direction': TradeDirection[trend['direction']],
                        'strength': TrendStrength[trend['strength']],
                        'details': trend['details'],
                        'time': pd.Timestamp(trend['time'])
                    }
                    
            # Restore active retracements
            if 'active_retracements' in state:
                self.active_retracements = {}
                for symbol, retrace in state['active_retracements'].items():
                    self.active_retracements[symbol] = {
                        'direction': TradeDirection[retrace['direction']],
                        'strength': retrace['strength'],
                        'level': retrace['level'],
                        'start_time': pd.Timestamp(retrace['start_time']),
                        'start_price': retrace['start_price']
                    }
                    
            logger.info(f"Loaded Retracement strategy state from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading Retracement strategy state: {str(e)}")
            return False
