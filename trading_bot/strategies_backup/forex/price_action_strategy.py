#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Price Action Trading Strategy

This strategy focuses on analyzing raw price movements and candlestick patterns
without relying heavily on technical indicators. It identifies key price action
signals such as pin bars, engulfing patterns, inside bars, and price rejection
at significant levels.

Key Features:
1. Pure price action analysis with minimal indicators
2. Support/resistance level identification and tracking
3. Multiple timeframe confirmation
4. Advanced candlestick pattern recognition
5. Session-aware trade execution
6. Regime-aware parameter optimization

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
from dataclasses import dataclass

from trading_bot.strategies.base.forex_base import ForexBaseStrategy
from trading_bot.utils.event_bus import EventBus
from trading_bot.utils.technical_indicators import calculate_atr
from trading_bot.utils.visualization import plot_price_levels, plot_candlestick_patterns
from trading_bot.strategy_selection.trading_time_optimizer import TradingSession
from trading_bot.strategies.strategy_template import (
    Strategy, SignalType, PositionSizing, OrderStatus, 
    TradeDirection, MarketRegime, TimeFrame
)

logger = logging.getLogger(__name__)

class CandlestickPattern(Enum):
    """Enum defining supported candlestick patterns."""
    PIN_BAR = "pin_bar"
    ENGULFING = "engulfing"
    INSIDE_BAR = "inside_bar"
    EVENING_STAR = "evening_star"
    MORNING_STAR = "morning_star"
    DOJI = "doji"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    DARK_CLOUD_COVER = "dark_cloud_cover"
    PIERCING_LINE = "piercing_line"
    BULLISH_HARAMI = "bullish_harami"
    BEARISH_HARAMI = "bearish_harami"

@dataclass
class PriceLevel:
    """Represents a significant price level with metadata."""
    price: float
    type: str  # 'support', 'resistance', 'swing_high', 'swing_low', 'pivot'
    strength: float  # 0.0 to 1.0
    touches: int
    created_at: datetime
    timeframe: str
    last_test: Optional[datetime] = None
    broken: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'price': self.price,
            'type': self.type,
            'strength': self.strength,
            'touches': self.touches,
            'created_at': self.created_at.isoformat(),
            'timeframe': self.timeframe,
            'last_test': self.last_test.isoformat() if self.last_test else None,
            'broken': self.broken
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PriceLevel':
        """Create from dictionary after deserialization."""
        return cls(
            price=data['price'],
            type=data['type'],
            strength=data['strength'],
            touches=data['touches'],
            created_at=datetime.fromisoformat(data['created_at']),
            timeframe=data['timeframe'],
            last_test=datetime.fromisoformat(data['last_test']) if data.get('last_test') else None,
            broken=data.get('broken', False)
        )

class PriceActionStrategy(ForexBaseStrategy):
    """
    Forex Price Action Trading Strategy implementation.
    
    This strategy focuses on pure price action signals with minimal indicators,
    emphasizing candlestick patterns, support/resistance levels, and price momentum.
    """
    
    def __init__(self, 
                 parameters: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the Price Action Trading strategy.
        
        Args:
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters
        default_params = {
            # Pattern recognition parameters
            'pin_bar_nose_ratio': 0.33,  # Ratio of nose to total candle for pin bar
            'engulfing_size_factor': 1.1,  # How much larger engulfing body must be
            'inside_bar_lookback': 3,  # How many bars to look back for inside bar patterns
            'pattern_confirmation_bars': 2,  # Bars to confirm pattern completion
            
            # Level identification parameters
            'swing_high_lookback': 5,  # Bars on each side for swing high/low detection
            'price_level_proximity': 0.0010,  # 10 pips proximity for merging levels
            'level_strength_decay': 0.95,  # Strength decay factor for unused levels
            'min_level_strength': 0.3,  # Minimum strength to keep a level
            'max_tracked_levels': 20,  # Maximum number of levels to track
            
            # Timeframes for analysis (strings like '1H', '4H')
            'primary_timeframe': '1H',
            'confirmation_timeframes': ['15M', '4H'],  # For multi-timeframe confirmation
            
            # Trade parameters
            'stop_loss_atr_factor': 1.5,  # ATR multiplier for stop loss
            'take_profit_atr_factor': 2.5,  # ATR multiplier for take profit
            'risk_per_trade': 0.01,  # 1% risk per trade
            'max_active_trades': 3,
            'pattern_expiry_bars': 3,  # Patterns expire after this many bars
            
            # Filtering
            'min_daily_range_pips': 50,  # Minimum daily range in pips
            'min_pattern_strength': 0.7,  # Minimum strength for a valid pattern
            
            # MTF confirmation
            'mtf_agreement_threshold': 0.7,  # Threshold for timeframe agreement (0-1)
            
            # Advanced
            'pattern_weightings': {  # Relative importance of patterns (0-1)
                'pin_bar': 0.85,
                'engulfing': 0.9,
                'inside_bar': 0.75,
                'evening_star': 0.95,
                'morning_star': 0.95,
                'doji': 0.6,
                'hammer': 0.8,
                'shooting_star': 0.8,
                'dark_cloud_cover': 0.8,
                'piercing_line': 0.8,
                'bullish_harami': 0.7,
                'bearish_harami': 0.7
            }
        }
        
        # Initialize base class
        default_metadata = {
            'name': 'Forex Price Action Strategy',
            'description': 'Trading strategy based on pure price action signals with minimal indicators',
            'version': '1.0.0',
            'author': 'Ben Dickinson',
            'type': 'forex_price_action',
            'tags': ['forex', 'price_action', 'swing_trading', 'candlestick_patterns'],
            'preferences': {
                'timeframes': ['5M', '15M', '1H', '4H', 'D'],
                'default_timeframe': '1H'
            }
        }
        
        if metadata:
            default_metadata.update(metadata)
            
        super().__init__('forex_price_action', parameters, default_metadata)
        
        # Merge provided parameters with defaults
        if parameters:
            for key, value in parameters.items():
                if key in default_params:
                    self.parameters[key] = value
        else:
            self.parameters = default_params
        
        # Initialize price level tracking
        self.price_levels: Dict[str, List[PriceLevel]] = {}  # symbol -> list of price levels
        
        # Pattern detection history
        self.detected_patterns: Dict[str, List[Dict[str, Any]]] = {}  # symbol -> list of patterns
        
        # MTF data storage
        self.mtf_data: Dict[str, Dict[str, pd.DataFrame]] = {}  # symbol -> {timeframe -> data}
        
        # Cache for recent calculations
        self._calculation_cache = {}
        
        # Register with event bus
        EventBus.get_instance().register(self)
        
    def detect_pin_bar(self, data: pd.DataFrame, index: int) -> Tuple[bool, TradeDirection, float]:
        """
        Detect pin bar candlestick pattern.
        
        Args:
            data: OHLCV DataFrame
            index: Current candle index
            
        Returns:
            Tuple of (is_pin_bar, direction, strength)
        """
        if index < 1 or index >= len(data) - 1:
            return False, TradeDirection.FLAT, 0.0
            
        candle = data.iloc[index]
        prev_candle = data.iloc[index-1]
        
        # Calculate candle parts
        body_size = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        total_range = candle['high'] - candle['low']
        
        # Skip if candle range is too small (avoid division by zero)
        if total_range < 0.0001:
            return False, TradeDirection.FLAT, 0.0
            
        # Pin bar conditions
        nose_ratio = self.parameters['pin_bar_nose_ratio']
        
        # Bullish pin (long lower wick)
        if (lower_wick / total_range > nose_ratio and 
                body_size / total_range < (1 - nose_ratio) and 
                upper_wick / total_range < 0.2):
            
            # Strength based on wick size and previous trend
            strength = min(1.0, lower_wick / total_range * 1.5)
            
            # Confirm it's at a swing low or after a downtrend
            if index > 5 and data['close'].iloc[index-5:index].mean() > candle['close']:
                strength *= 1.2  # Increase strength if potential reversal
                
            return True, TradeDirection.LONG, min(1.0, strength)
            
        # Bearish pin (long upper wick)
        elif (upper_wick / total_range > nose_ratio and 
              body_size / total_range < (1 - nose_ratio) and 
              lower_wick / total_range < 0.2):
            
            # Strength based on wick size and previous trend
            strength = min(1.0, upper_wick / total_range * 1.5)
            
            # Confirm it's at a swing high or after an uptrend
            if index > 5 and data['close'].iloc[index-5:index].mean() < candle['close']:
                strength *= 1.2  # Increase strength if potential reversal
                
            return True, TradeDirection.SHORT, min(1.0, strength)
            
        return False, TradeDirection.FLAT, 0.0
        
    def detect_engulfing(self, data: pd.DataFrame, index: int) -> Tuple[bool, TradeDirection, float]:
        """
        Detect engulfing candlestick pattern.
        
        Args:
            data: OHLCV DataFrame
            index: Current candle index
            
        Returns:
            Tuple of (is_engulfing, direction, strength)
        """
        if index < 1 or index >= len(data) - 1:
            return False, TradeDirection.FLAT, 0.0
            
        current = data.iloc[index]
        previous = data.iloc[index-1]
        
        # Calculate body sizes
        current_body = abs(current['close'] - current['open'])
        previous_body = abs(previous['close'] - previous['open'])
        
        # Skip if body sizes are too small
        if previous_body < 0.0001 or current_body < 0.0001:
            return False, TradeDirection.FLAT, 0.0
            
        # Get current candle direction
        current_bullish = current['close'] > current['open']
        previous_bullish = previous['close'] > previous['open']
        
        # Size factor from parameters
        size_factor = self.parameters['engulfing_size_factor']
        
        # Check for engulfing pattern - opposite direction and larger body
        if current_bullish and not previous_bullish:  # Bullish engulfing
            # Check if current body engulfs previous body
            if (current_body > previous_body * size_factor and
                current['open'] <= previous['close'] and
                current['close'] >= previous['open']):
                
                # Calculate strength based on size difference and location
                strength = min(1.0, current_body / previous_body * 0.7)
                
                # Higher strength if at support level or after downtrend
                if index > 5 and data['close'].iloc[index-5:index].mean() > current['open']:
                    strength *= 1.2
                    
                return True, TradeDirection.LONG, min(1.0, strength)
                
        elif not current_bullish and previous_bullish:  # Bearish engulfing
            # Check if current body engulfs previous body
            if (current_body > previous_body * size_factor and
                current['open'] >= previous['close'] and
                current['close'] <= previous['open']):
                
                # Calculate strength based on size difference and location
                strength = min(1.0, current_body / previous_body * 0.7)
                
                # Higher strength if at resistance level or after uptrend
                if index > 5 and data['close'].iloc[index-5:index].mean() < current['open']:
                    strength *= 1.2
                    
                return True, TradeDirection.SHORT, min(1.0, strength)
                
        return False, TradeDirection.FLAT, 0.0
        
    def detect_inside_bar(self, data: pd.DataFrame, index: int) -> Tuple[bool, TradeDirection, float]:
        """
        Detect inside bar pattern (current bar range inside previous bar range).
        
        Args:
            data: OHLCV DataFrame
            index: Current candle index
            
        Returns:
            Tuple of (is_inside_bar, direction, strength)
        """
        if index < 1 or index >= len(data) - 1:
            return False, TradeDirection.FLAT, 0.0
            
        current = data.iloc[index]
        previous = data.iloc[index-1]
        
        # Inside bar conditions: high lower than previous high, low higher than previous low
        is_inside = (current['high'] <= previous['high'] and 
                     current['low'] >= previous['low'])
                     
        if not is_inside:
            return False, TradeDirection.FLAT, 0.0
            
        # Calculate strength based on size difference
        current_range = current['high'] - current['low']
        previous_range = previous['high'] - previous['low']
        
        if previous_range < 0.0001:  # Avoid division by zero
            return False, TradeDirection.FLAT, 0.0
            
        # Inside bars are more significant when they're much smaller than the mother bar
        # and when the mother bar is a strong trending candle
        size_ratio = current_range / previous_range
        strength = 1.0 - min(0.9, size_ratio)  # Smaller ratio = higher strength
        
        # Determine direction based on the mother bar and the trend
        mother_bar_bullish = previous['close'] > previous['open']
        
        # Look at short-term trend direction
        if index >= 5:
            short_trend = data['close'].iloc[index-5:index].mean()
            trend_up = current['close'] > short_trend
            
            if mother_bar_bullish and trend_up:
                return True, TradeDirection.LONG, strength
            elif not mother_bar_bullish and not trend_up:
                return True, TradeDirection.SHORT, strength
            else:
                # Inside bar with conflicting signals - lower strength
                if mother_bar_bullish:
                    return True, TradeDirection.LONG, strength * 0.6
                else:
                    return True, TradeDirection.SHORT, strength * 0.6
        else:
            # Not enough data for trend, rely on mother bar direction
            if mother_bar_bullish:
                return True, TradeDirection.LONG, strength * 0.8
            else:
                return True, TradeDirection.SHORT, strength * 0.8
                
    def detect_doji(self, data: pd.DataFrame, index: int) -> Tuple[bool, TradeDirection, float]:
        """
        Detect doji candlestick pattern (small body with wicks).
        
        Args:
            data: OHLCV DataFrame
            index: Current candle index
            
        Returns:
            Tuple of (is_doji, direction, strength)
        """
        if index < 1 or index >= len(data) - 1:
            return False, TradeDirection.FLAT, 0.0
            
        candle = data.iloc[index]
        
        # Calculate body and range
        body = abs(candle['close'] - candle['open'])
        range_size = candle['high'] - candle['low']
        
        # Avoid division by zero
        if range_size < 0.0001:
            return False, TradeDirection.FLAT, 0.0
            
        # Doji has very small body compared to range
        body_ratio = body / range_size
        
        if body_ratio < 0.1:  # Very small body
            # Calculate direction based on wicks
            upper_wick = candle['high'] - max(candle['open'], candle['close'])
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            
            # Indecision doji
            if abs(upper_wick - lower_wick) / range_size < 0.2:  # Balanced wicks
                # Doji in a trend can signal reversal
                if index > 5:
                    trend = data['close'].iloc[index-5:index].mean()
                    if candle['close'] > trend:
                        return True, TradeDirection.SHORT, 0.6  # Potential reversal down
                    else:
                        return True, TradeDirection.LONG, 0.6  # Potential reversal up
                else:
                    return True, TradeDirection.FLAT, 0.5  # Indecision
                    
            # Dragonfly doji (long lower wick, almost no upper wick)
            elif lower_wick > 0.7 * range_size and upper_wick < 0.1 * range_size:
                return True, TradeDirection.LONG, 0.8
                
            # Gravestone doji (long upper wick, almost no lower wick)
            elif upper_wick > 0.7 * range_size and lower_wick < 0.1 * range_size:
                return True, TradeDirection.SHORT, 0.8
                
            # Regular doji - direction depends on trend
            else:
                if index > 5:
                    trend = data['close'].iloc[index-5:index].mean()
                    if candle['close'] > trend:
                        return True, TradeDirection.SHORT, 0.5  # Potential reversal down
                    else:
                        return True, TradeDirection.LONG, 0.5  # Potential reversal up
                else:
                    return True, TradeDirection.FLAT, 0.4  # Weaker signal
                    
        return False, TradeDirection.FLAT, 0.0
    
    def detect_candle_patterns(self, data: pd.DataFrame, index: int) -> List[Dict[str, Any]]:
        """
        Detect all candlestick patterns at the current index.
        
        Args:
            data: OHLCV DataFrame
            index: Current index
            
        Returns:
            List of dictionaries containing pattern details
        """
        patterns = []
        
        # Check for pin bar
        is_pin, pin_direction, pin_strength = self.detect_pin_bar(data, index)
        if is_pin and pin_strength >= self.parameters['min_pattern_strength']:
            patterns.append({
                'type': CandlestickPattern.PIN_BAR.value,
                'direction': pin_direction,
                'strength': pin_strength * self.parameters['pattern_weightings']['pin_bar'],
                'index': index,
                'time': data.index[index],
                'expiry': index + self.parameters['pattern_expiry_bars']
            })
            
        # Check for engulfing pattern
        is_engulfing, engulfing_direction, engulfing_strength = self.detect_engulfing(data, index)
        if is_engulfing and engulfing_strength >= self.parameters['min_pattern_strength']:
            patterns.append({
                'type': CandlestickPattern.ENGULFING.value,
                'direction': engulfing_direction,
                'strength': engulfing_strength * self.parameters['pattern_weightings']['engulfing'],
                'index': index,
                'time': data.index[index],
                'expiry': index + self.parameters['pattern_expiry_bars']
            })
            
        # Check for inside bar
        is_inside, inside_direction, inside_strength = self.detect_inside_bar(data, index)
        if is_inside and inside_strength >= self.parameters['min_pattern_strength']:
            patterns.append({
                'type': CandlestickPattern.INSIDE_BAR.value,
                'direction': inside_direction,
                'strength': inside_strength * self.parameters['pattern_weightings']['inside_bar'],
                'index': index,
                'time': data.index[index],
                'expiry': index + self.parameters['pattern_expiry_bars']
            })
            
        # Check for doji
        is_doji, doji_direction, doji_strength = self.detect_doji(data, index)
        if is_doji and doji_strength >= self.parameters['min_pattern_strength']:
            patterns.append({
                'type': CandlestickPattern.DOJI.value,
                'direction': doji_direction,
                'strength': doji_strength * self.parameters['pattern_weightings']['doji'],
                'index': index,
                'time': data.index[index],
                'expiry': index + self.parameters['pattern_expiry_bars']
            })
            
        return patterns
    
    def identify_support_resistance(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[PriceLevel]:
        """
        Identify support and resistance levels from price data.
        
        Args:
            data: OHLCV DataFrame
            symbol: Currency pair symbol
            timeframe: Timeframe of the data
            
        Returns:
            List of PriceLevel objects
        """
        # Skip if not enough data
        if len(data) < self.parameters['swing_high_lookback'] * 2 + 1:
            return []
            
        # Find swing highs and lows
        levels = []
        lookback = self.parameters['swing_high_lookback']
        
        # Process for swing highs (potential resistance)
        for i in range(lookback, len(data) - lookback):
            # Check for swing high
            is_swing_high = True
            for j in range(1, lookback + 1):
                if data['high'].iloc[i] <= data['high'].iloc[i - j] or \
                   data['high'].iloc[i] <= data['high'].iloc[i + j]:
                    is_swing_high = False
                    break
                    
            if is_swing_high:
                # Create resistance level
                level = PriceLevel(
                    price=data['high'].iloc[i],
                    type='resistance',
                    strength=0.8,  # Initial strength
                    touches=1,
                    created_at=data.index[i].to_pydatetime(),
                    timeframe=timeframe
                )
                levels.append(level)
                
            # Check for swing low
            is_swing_low = True
            for j in range(1, lookback + 1):
                if data['low'].iloc[i] >= data['low'].iloc[i - j] or \
                   data['low'].iloc[i] >= data['low'].iloc[i + j]:
                    is_swing_low = False
                    break
                    
            if is_swing_low:
                # Create support level
                level = PriceLevel(
                    price=data['low'].iloc[i],
                    type='support',
                    strength=0.8,  # Initial strength
                    touches=1,
                    created_at=data.index[i].to_pydatetime(),
                    timeframe=timeframe
                )
                levels.append(level)
        
        # Merge nearby levels
        return self._consolidate_price_levels(levels, symbol)
        
    def _consolidate_price_levels(self, levels: List[PriceLevel], symbol: str) -> List[PriceLevel]:
        """
        Merge nearby price levels and maintain only the strongest ones.
        
        Args:
            levels: List of price levels
            symbol: Currency pair symbol
            
        Returns:
            Consolidated list of price levels
        """
        if not levels:
            return []
            
        # Get pip value for this symbol
        pip_value = self.get_pip_value(symbol)
        proximity_pips = self.parameters['price_level_proximity'] / pip_value
        
        # Sort levels by price
        sorted_levels = sorted(levels, key=lambda x: x.price)
        consolidated = []
        
        # Group nearby levels
        current_group = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # If this level is close to the previous one, group them
            if abs(level.price - current_group[0].price) < proximity_pips:
                current_group.append(level)
            else:
                # Process the current group
                if current_group:
                    # Select the strongest level from the group
                    strongest = max(current_group, key=lambda x: x.strength * x.touches)
                    consolidated.append(strongest)
                    
                # Start a new group
                current_group = [level]
                
        # Process the last group
        if current_group:
            strongest = max(current_group, key=lambda x: x.strength * x.touches)
            consolidated.append(strongest)
            
        # Limit the number of levels
        if len(consolidated) > self.parameters['max_tracked_levels']:
            consolidated = sorted(consolidated, key=lambda x: x.strength * x.touches, reverse=True)
            consolidated = consolidated[:self.parameters['max_tracked_levels']]
            
        return consolidated
    
    def _update_price_levels(self, symbol: str, current_price: float, current_time: datetime) -> None:
        """
        Update price levels with new price information.
        
        Args:
            symbol: Currency pair symbol
            current_price: Current price
            current_time: Current time
        """
        if symbol not in self.price_levels or not self.price_levels[symbol]:
            return
            
        pip_value = self.get_pip_value(symbol)
        proximity_pips = self.parameters['price_level_proximity'] / pip_value
        min_strength = self.parameters['min_level_strength']
        decay_factor = self.parameters['level_strength_decay']
        
        updated_levels = []
        
        for level in self.price_levels[symbol]:
            # Apply time decay to strength
            if (current_time - level.created_at).days > 7:  # Older than a week
                level.strength *= decay_factor
                
            # Check if level was tested
            if abs(current_price - level.price) < proximity_pips * 0.5:  # Half of proximity for test
                level.touches += 1
                level.last_test = current_time
                level.strength = min(1.0, level.strength * 1.1)  # Strengthen tested levels
                
            # Check if level was broken
            if ((level.type == 'resistance' and current_price > level.price * 1.0015) or 
                (level.type == 'support' and current_price < level.price * 0.9985)):
                # Price significantly beyond the level
                if not level.broken:
                    level.broken = True
                    level.strength *= 0.5  # Broken levels lose strength but don't disappear
                
            # Keep if strong enough
            if level.strength >= min_strength:
                updated_levels.append(level)
                
        # Update the levels
        self.price_levels[symbol] = updated_levels
        
    def _prepare_mtf_data(self, symbol: str, data: pd.DataFrame, 
                          current_time: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """
        Prepare multi-timeframe data for analysis.
        
        Args:
            symbol: Currency pair symbol
            data: Primary timeframe data
            current_time: Current time
            
        Returns:
            Dictionary of timeframe -> resampled data
        """
        # Skip if not enough data
        if len(data) < 100:
            return {self.parameters['primary_timeframe']: data}
        
        # Ensure we have timestamp index
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning(f"Data for {symbol} doesn't have timestamp index")
            return {self.parameters['primary_timeframe']: data}
        
        results = {self.parameters['primary_timeframe']: data}
        
        # Resample to other timeframes
        for tf in self.parameters['confirmation_timeframes']:
            if tf in results or tf == self.parameters['primary_timeframe']:
                continue
                
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
    
    def _check_mtf_agreement(self, symbol: str, direction: TradeDirection) -> float:
        """
        Check if multiple timeframes agree on the trading direction.
        
        Args:
            symbol: Currency pair symbol
            direction: Proposed trade direction
            
        Returns:
            Agreement score (0.0 to 1.0)
        """
        if symbol not in self.mtf_data:
            return 0.5  # Neutral if no data
            
        # Count agreeing timeframes
        agreements = 0
        total = 0
        
        for tf, data in self.mtf_data[symbol].items():
            if len(data) < 10:
                continue
                
            total += 1
            current_index = len(data) - 1
            
            # Get the last detected patterns
            patterns = self.detect_candle_patterns(data, current_index)
            if not patterns:
                continue
                
            # Check if any pattern agrees with our direction
            for pattern in patterns:
                if pattern['direction'] == direction:
                    agreements += 1 * pattern['strength']
                    break
        
        # Calculate agreement score
        if total == 0:
            return 0.5  # Neutral
            
        return min(1.0, agreements / total)
        
    def _evaluate_trade_setup(self, symbol: str, direction: TradeDirection, 
                             pattern_strength: float, price: float) -> Tuple[bool, float, str]:
        """
        Evaluate a potential trade setup considering price levels and patterns.
        
        Args:
            symbol: Currency pair symbol
            direction: Trade direction
            pattern_strength: Strength of the pattern
            price: Current price
            
        Returns:
            Tuple of (is_valid, score, reason)
        """
        # Check if symbol has price levels
        if symbol not in self.price_levels or not self.price_levels[symbol]:
            return False, 0.0, "No price levels identified"
            
        # Get pip value
        pip_value = self.get_pip_value(symbol)
        proximity_pips = self.parameters['price_level_proximity'] / pip_value
        
        # Find nearest levels
        support_levels = [l for l in self.price_levels[symbol] if l.type == 'support' and not l.broken]
        resistance_levels = [l for l in self.price_levels[symbol] if l.type == 'resistance' and not l.broken]
        
        # Calculate distances
        nearest_support = None
        support_distance = float('inf')
        for level in support_levels:
            dist = (price - level.price) / pip_value  # Distance in pips
            if dist > 0 and dist < support_distance:  # Price above support
                support_distance = dist
                nearest_support = level
                
        nearest_resistance = None
        resistance_distance = float('inf')
        for level in resistance_levels:
            dist = (level.price - price) / pip_value  # Distance in pips
            if dist > 0 and dist < resistance_distance:  # Price below resistance
                resistance_distance = dist
                nearest_resistance = level
        
        # Check for support/resistance confirmation
        if direction == TradeDirection.LONG:
            # Long trade should be near support
            if nearest_support and support_distance < 20:  # Within 20 pips of support
                support_score = nearest_support.strength * (1.0 - min(1.0, support_distance / 20))
                reason = f"Price near support at {nearest_support.price:.5f}"
                
                # Check if we have room to run before hitting resistance
                if nearest_resistance:
                    room_to_run = resistance_distance / 100  # Normalize to 0-1 for reasonable distances
                    setup_score = (pattern_strength * 0.4 + support_score * 0.4 + room_to_run * 0.2)
                    
                    if setup_score >= 0.7 and room_to_run > 0.3:
                        return True, setup_score, reason
                    else:
                        return False, setup_score, f"Insufficient room before resistance at {nearest_resistance.price:.5f}"
                else:
                    # No nearby resistance - good for long
                    setup_score = pattern_strength * 0.6 + support_score * 0.4
                    return setup_score >= 0.7, setup_score, reason
            else:
                return False, 0.3, "Not near support level"
                
        elif direction == TradeDirection.SHORT:
            # Short trade should be near resistance
            if nearest_resistance and resistance_distance < 20:  # Within 20 pips of resistance
                resistance_score = nearest_resistance.strength * (1.0 - min(1.0, resistance_distance / 20))
                reason = f"Price near resistance at {nearest_resistance.price:.5f}"
                
                # Check if we have room to run before hitting support
                if nearest_support:
                    room_to_run = support_distance / 100  # Normalize to 0-1 for reasonable distances
                    setup_score = (pattern_strength * 0.4 + resistance_score * 0.4 + room_to_run * 0.2)
                    
                    if setup_score >= 0.7 and room_to_run > 0.3:
                        return True, setup_score, reason
                    else:
                        return False, setup_score, f"Insufficient room before support at {nearest_support.price:.5f}"
                else:
                    # No nearby support - good for short
                    setup_score = pattern_strength * 0.6 + resistance_score * 0.4
                    return setup_score >= 0.7, setup_score, reason
            else:
                return False, 0.3, "Not near resistance level"
        
        return False, 0.0, "Invalid direction"
        
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_time: pd.Timestamp) -> Dict[str, Any]:
        """
        Generate trading signals based on price action patterns.
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            current_time: Current time
            
        Returns:
            Dictionary of signal information
        """
        signals = {}
        
        # Check for active trading session
        if not self.is_active_trading_session(current_time):
            return signals
            
        # Process each symbol
        for symbol, ohlcv in data.items():
            if len(ohlcv) < 30:  # Need enough historical data
                continue
                
            # Prepare multi-timeframe data
            mtf_data = self._prepare_mtf_data(symbol, ohlcv, current_time)
            
            # Initialize price levels if needed
            if symbol not in self.price_levels or not self.price_levels[symbol]:
                # Get primary timeframe
                primary_tf = self.parameters['primary_timeframe']
                if primary_tf in mtf_data:
                    self.price_levels[symbol] = self.identify_support_resistance(
                        mtf_data[primary_tf], symbol, primary_tf)
                    
            # Get current price
            current_price = ohlcv['close'].iloc[-1]
            
            # Update existing price levels
            self._update_price_levels(symbol, current_price, current_time.to_pydatetime())
            
            # Initialize detected patterns for this symbol if needed
            if symbol not in self.detected_patterns:
                self.detected_patterns[symbol] = []
                
            # Remove expired patterns
            current_index = len(ohlcv) - 1
            self.detected_patterns[symbol] = [p for p in self.detected_patterns[symbol] 
                                             if p['expiry'] > current_index]
            
            # Detect new patterns
            primary_tf = self.parameters['primary_timeframe']
            if primary_tf in mtf_data:
                new_patterns = self.detect_candle_patterns(mtf_data[primary_tf], current_index)
                self.detected_patterns[symbol].extend(new_patterns)
                
            # Get currently active patterns
            active_patterns = self.detected_patterns[symbol]
            
            if not active_patterns:
                continue
                
            # Find the strongest pattern for each direction
            strongest_long = {'strength': 0}
            strongest_short = {'strength': 0}
            
            for pattern in active_patterns:
                if pattern['direction'] == TradeDirection.LONG and pattern['strength'] > strongest_long.get('strength', 0):
                    strongest_long = pattern
                elif pattern['direction'] == TradeDirection.SHORT and pattern['strength'] > strongest_short.get('strength', 0):
                    strongest_short = pattern
            
            # Evaluate potential trade setups
            long_signal = None
            short_signal = None
            
            # Check long setup
            if strongest_long.get('strength', 0) >= self.parameters['min_pattern_strength']:
                # Check MTF agreement
                mtf_agreement = self._check_mtf_agreement(symbol, TradeDirection.LONG)
                
                # Evaluate setup quality
                is_valid, score, reason = self._evaluate_trade_setup(
                    symbol, TradeDirection.LONG, strongest_long['strength'], current_price)
                
                # Adjust score based on MTF agreement
                adjusted_score = score * (0.7 + 0.3 * mtf_agreement)  # MTF influences 30% of score
                
                if is_valid and adjusted_score >= 0.7:
                    # Calculate stop loss and take profit
                    atr = calculate_atr(ohlcv, 14).iloc[-1]
                    stop_loss = current_price - atr * self.parameters['stop_loss_atr_factor']
                    take_profit = current_price + atr * self.parameters['take_profit_atr_factor']
                    
                    # Create signal
                    long_signal = {
                        'symbol': symbol,
                        'direction': TradeDirection.LONG,
                        'strength': adjusted_score,
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'pattern_type': strongest_long['type'],
                        'reason': reason,
                        'mtf_agreement': mtf_agreement
                    }
            
            # Check short setup
            if strongest_short.get('strength', 0) >= self.parameters['min_pattern_strength']:
                # Check MTF agreement
                mtf_agreement = self._check_mtf_agreement(symbol, TradeDirection.SHORT)
                
                # Evaluate setup quality
                is_valid, score, reason = self._evaluate_trade_setup(
                    symbol, TradeDirection.SHORT, strongest_short['strength'], current_price)
                
                # Adjust score based on MTF agreement
                adjusted_score = score * (0.7 + 0.3 * mtf_agreement)  # MTF influences 30% of score
                
                if is_valid and adjusted_score >= 0.7:
                    # Calculate stop loss and take profit
                    atr = calculate_atr(ohlcv, 14).iloc[-1]
                    stop_loss = current_price + atr * self.parameters['stop_loss_atr_factor']
                    take_profit = current_price - atr * self.parameters['take_profit_atr_factor']
                    
                    # Create signal
                    short_signal = {
                        'symbol': symbol,
                        'direction': TradeDirection.SHORT,
                        'strength': adjusted_score,
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'pattern_type': strongest_short['type'],
                        'reason': reason,
                        'mtf_agreement': mtf_agreement
                    }
            
            # Only use the stronger signal if both directions are valid
            if long_signal and short_signal:
                if long_signal['strength'] >= short_signal['strength']:
                    signals[symbol] = long_signal
                else:
                    signals[symbol] = short_signal
            elif long_signal:
                signals[symbol] = long_signal
            elif short_signal:
                signals[symbol] = short_signal
        
        # Publish signals event
        if signals:
            EventBus.get_instance().publish('price_action_signals', {
                'time': current_time.isoformat(),
                'signals': signals,
                'source': self.metadata['name']
            })
            
        return signals
    
    def get_regime_compatibility_score(self, regime: MarketRegime) -> float:
        """
        Calculate compatibility score with the given market regime.
        
        Args:
            regime: Market regime to check compatibility with
            
        Returns:
            Compatibility score (0.0 to 1.0)
        """
        # Price action works well in certain regimes but not others
        regime_scores = {
            MarketRegime.TRENDING: 0.75,      # Good in trends for continuation patterns
            MarketRegime.RANGING: 0.85,       # Excellent in ranges for reversal patterns
            MarketRegime.VOLATILE: 0.65,      # Can work if levels are respected
            MarketRegime.BREAKOUT: 0.90,      # Great for breakout patterns
            MarketRegime.LOW_VOLATILITY: 0.60,  # Mixed results in low volatility
            MarketRegime.HIGH_VOLATILITY: 0.55,  # Dangerous in high volatility, need filters
            MarketRegime.BULLISH: 0.75,       # Good for identifying bullish continuation
            MarketRegime.BEARISH: 0.75,       # Good for identifying bearish continuation
            MarketRegime.CONSOLIDATION: 0.85,  # Great for identifying range boundaries
            MarketRegime.UNKNOWN: 0.65        # Average performance as a baseline
        }
        
        return regime_scores.get(regime, 0.5)
    
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
        logger.info(f"Optimizing Price Action strategy for {market_regime} regime")
        
        # Define parameter ranges based on regime
        param_ranges = {}
        
        # Base parameter ranges
        if market_regime == MarketRegime.TRENDING:
            param_ranges = {
                'pin_bar_nose_ratio': [0.30, 0.33, 0.36],
                'engulfing_size_factor': [1.05, 1.1, 1.15],
                'min_pattern_strength': [0.65, 0.7, 0.75],
                'mtf_agreement_threshold': [0.6, 0.7, 0.8],
                'stop_loss_atr_factor': [1.2, 1.5, 1.8],
                'take_profit_atr_factor': [2.0, 2.5, 3.0]
            }
        elif market_regime == MarketRegime.RANGING:
            param_ranges = {
                'pin_bar_nose_ratio': [0.33, 0.36, 0.40],
                'engulfing_size_factor': [1.1, 1.15, 1.2],
                'min_pattern_strength': [0.7, 0.75, 0.8],
                'mtf_agreement_threshold': [0.7, 0.75, 0.8],
                'stop_loss_atr_factor': [1.0, 1.2, 1.5],
                'take_profit_atr_factor': [1.5, 2.0, 2.5]
            }
        elif market_regime == MarketRegime.VOLATILE or market_regime == MarketRegime.HIGH_VOLATILITY:
            param_ranges = {
                'pin_bar_nose_ratio': [0.36, 0.40, 0.45],
                'engulfing_size_factor': [1.15, 1.2, 1.25],
                'min_pattern_strength': [0.75, 0.8, 0.85],
                'mtf_agreement_threshold': [0.75, 0.8, 0.85],
                'stop_loss_atr_factor': [1.5, 1.8, 2.0],
                'take_profit_atr_factor': [2.0, 2.5, 3.0]
            }
        elif market_regime == MarketRegime.BREAKOUT:
            param_ranges = {
                'pin_bar_nose_ratio': [0.30, 0.33, 0.36],
                'engulfing_size_factor': [1.1, 1.15, 1.2],
                'min_pattern_strength': [0.7, 0.75, 0.8],
                'mtf_agreement_threshold': [0.7, 0.75, 0.8],
                'stop_loss_atr_factor': [1.5, 1.8, 2.0],
                'take_profit_atr_factor': [2.5, 3.0, 3.5]
            }
        else:  # Default/unknown regime
            param_ranges = {
                'pin_bar_nose_ratio': [0.33, 0.36],
                'engulfing_size_factor': [1.1, 1.15],
                'min_pattern_strength': [0.7, 0.75],
                'mtf_agreement_threshold': [0.7, 0.75],
                'stop_loss_atr_factor': [1.3, 1.5, 1.7],
                'take_profit_atr_factor': [2.0, 2.5, 3.0]
            }
        
        # TODO: Implement grid search optimization
        # For now, return the middle value of each parameter range as a reasonable default
        optimized = {}
        for param, values in param_ranges.items():
            middle_idx = len(values) // 2
            optimized[param] = values[middle_idx]
            
        # Update pattern weightings based on regime
        pattern_weightings = dict(self.parameters['pattern_weightings'])
        
        if market_regime == MarketRegime.TRENDING:
            # In trending markets, engulfing and pin bars are strong
            pattern_weightings['engulfing'] = 0.95
            pattern_weightings['pin_bar'] = 0.90
            pattern_weightings['inside_bar'] = 0.75  # Less relevant in trends
        elif market_regime == MarketRegime.RANGING:
            # In ranging markets, inside bars and dojis work well
            pattern_weightings['inside_bar'] = 0.85
            pattern_weightings['doji'] = 0.75
        elif market_regime == MarketRegime.BREAKOUT:
            # For breakouts, engulfing patterns are powerful
            pattern_weightings['engulfing'] = 0.95
            pattern_weightings['pin_bar'] = 0.85
            
        optimized['pattern_weightings'] = pattern_weightings
        
        logger.info(f"Optimization complete for Price Action strategy")
        
        return optimized
    
    def save_state(self, file_path: str) -> bool:
        """
        Save the current state of price levels and pattern data.
        
        Args:
            file_path: Path to save the state
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare price levels for serialization
            serialized_levels = {}
            for symbol, levels in self.price_levels.items():
                serialized_levels[symbol] = [level.to_dict() for level in levels]
            
            # Create state dictionary
            state = {
                'price_levels': serialized_levels,
                'detected_patterns': self.detected_patterns,
                'parameters': self.parameters,
                'saved_at': datetime.now().isoformat()
            }
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Saved Price Action state to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving Price Action state: {str(e)}")
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
                
            # Restore patterns
            if 'detected_patterns' in state:
                self.detected_patterns = state['detected_patterns']
                
            # Restore price levels
            if 'price_levels' in state:
                self.price_levels = {}
                for symbol, levels in state['price_levels'].items():
                    self.price_levels[symbol] = [PriceLevel.from_dict(level) for level in levels]
                    
            logger.info(f"Loaded Price Action state from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading Price Action state: {str(e)}")
            return False
