"""
Signal Filter Components

Implementation of various filter components for the modular strategy system.
These components filter signals based on market conditions, time, and other criteria.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import logging
from datetime import datetime, timedelta, time

from trading_bot.strategies.base_strategy import SignalType
from trading_bot.strategies.modular_strategy_system import (
    StrategyComponent, ComponentType, FilterComponent, MarketCondition
)

logger = logging.getLogger(__name__)

class VolumeFilter(FilterComponent):
    """Filters signals based on volume thresholds."""
    
    def __init__(self, 
                component_id: Optional[str] = None,
                min_volume_percentile: float = 50.0,
                lookback_period: int = 20):
        """
        Initialize volume filter
        
        Args:
            component_id: Unique component ID
            min_volume_percentile: Minimum volume percentile (0-100)
            lookback_period: Period for volume comparison
        """
        super().__init__(component_id)
        self.parameters = {
            'min_volume_percentile': min_volume_percentile,
            'lookback_period': lookback_period
        }
        self.description = f"Volume Filter (>{min_volume_percentile}th percentile)"
    
    def filter_signals(self, signals: Dict[str, SignalType], data: Dict[str, pd.DataFrame], 
                      context: Dict[str, Any]) -> Dict[str, SignalType]:
        """
        Filter signals based on volume requirements
        
        Args:
            signals: Input signals to filter
            data: Market data
            context: Filtering context
            
        Returns:
            Filtered signals
        """
        filtered_signals = signals.copy()
        
        min_percentile = self.parameters['min_volume_percentile']
        lookback = self.parameters['lookback_period']
        
        for symbol, signal in signals.items():
            # Skip FLAT signals (nothing to filter)
            if signal == SignalType.FLAT:
                continue
            
            # Get market data for this symbol
            if symbol not in data:
                # No data, remove signal
                filtered_signals[symbol] = SignalType.FLAT
                continue
            
            df = data[symbol]
            if len(df) < lookback:
                # Not enough history, remove signal
                filtered_signals[symbol] = SignalType.FLAT
                continue
            
            # Get latest volume
            latest_volume = df['volume'].iloc[-1]
            
            # Calculate volume percentile
            volume_history = df['volume'].iloc[-lookback:-1]  # Exclude current volume
            if len(volume_history) == 0:
                continue
                
            volume_percentile = 100 * (sum(latest_volume > vol for vol in volume_history) / len(volume_history))
            
            # Filter based on percentile
            if volume_percentile < min_percentile:
                filtered_signals[symbol] = SignalType.FLAT
        
        return filtered_signals

class VolatilityFilter(FilterComponent):
    """Filters signals based on volatility measures."""
    
    def __init__(self, 
                component_id: Optional[str] = None,
                min_atr_percentile: float = 30.0,
                max_atr_percentile: float = 90.0,
                atr_period: int = 14,
                lookback_period: int = 30):
        """
        Initialize volatility filter
        
        Args:
            component_id: Unique component ID
            min_atr_percentile: Minimum ATR percentile (0-100)
            max_atr_percentile: Maximum ATR percentile (0-100)
            atr_period: ATR calculation period
            lookback_period: Period for ATR comparison
        """
        super().__init__(component_id)
        self.parameters = {
            'min_atr_percentile': min_atr_percentile,
            'max_atr_percentile': max_atr_percentile,
            'atr_period': atr_period,
            'lookback_period': lookback_period
        }
        self.description = f"Volatility Filter ({min_atr_percentile}%-{max_atr_percentile}%)"
    
    def filter_signals(self, signals: Dict[str, SignalType], data: Dict[str, pd.DataFrame], 
                      context: Dict[str, Any]) -> Dict[str, SignalType]:
        """
        Filter signals based on volatility requirements
        
        Args:
            signals: Input signals to filter
            data: Market data
            context: Filtering context
            
        Returns:
            Filtered signals
        """
        filtered_signals = signals.copy()
        
        min_percentile = self.parameters['min_atr_percentile']
        max_percentile = self.parameters['max_atr_percentile']
        atr_period = self.parameters['atr_period']
        lookback = self.parameters['lookback_period']
        
        for symbol, signal in signals.items():
            # Skip FLAT signals (nothing to filter)
            if signal == SignalType.FLAT:
                continue
            
            # Get market data for this symbol
            if symbol not in data:
                # No data, remove signal
                filtered_signals[symbol] = SignalType.FLAT
                continue
            
            df = data[symbol]
            if len(df) < max(atr_period, lookback):
                # Not enough history, remove signal
                filtered_signals[symbol] = SignalType.FLAT
                continue
            
            # Calculate ATR if needed
            if 'atr' not in df.columns:
                # Calculate True Range
                high_low = df['high'] - df['low']
                high_close_prev = abs(df['high'] - df['close'].shift(1))
                low_close_prev = abs(df['low'] - df['close'].shift(1))
                tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
                
                # Calculate ATR
                df['atr'] = tr.rolling(window=atr_period).mean()
            
            # Get latest ATR
            latest_atr = df['atr'].iloc[-1]
            if pd.isna(latest_atr):
                # Invalid ATR, remove signal
                filtered_signals[symbol] = SignalType.FLAT
                continue
            
            # Calculate ATR percentile
            atr_history = df['atr'].iloc[-lookback:-1]  # Exclude current ATR
            if len(atr_history) == 0:
                continue
                
            atr_percentile = 100 * (sum(latest_atr > atr for atr in atr_history) / len(atr_history))
            
            # Filter based on percentile
            if atr_percentile < min_percentile or atr_percentile > max_percentile:
                filtered_signals[symbol] = SignalType.FLAT
        
        return filtered_signals

class TimeOfDayFilter(FilterComponent):
    """Filters signals based on time of day."""
    
    def __init__(self, 
                component_id: Optional[str] = None,
                start_time: str = "09:30",
                end_time: str = "16:00",
                time_zone: str = "America/New_York",
                days_of_week: Optional[List[int]] = None):
        """
        Initialize time of day filter
        
        Args:
            component_id: Unique component ID
            start_time: Start time (HH:MM)
            end_time: End time (HH:MM)
            time_zone: Time zone
            days_of_week: Days of week (0=Monday, 6=Sunday), None for all days
        """
        super().__init__(component_id)
        self.parameters = {
            'start_time': start_time,
            'end_time': end_time,
            'time_zone': time_zone,
            'days_of_week': days_of_week or list(range(7))
        }
        self.description = f"Time Filter ({start_time}-{end_time})"
    
    def filter_signals(self, signals: Dict[str, SignalType], data: Dict[str, pd.DataFrame], 
                      context: Dict[str, Any]) -> Dict[str, SignalType]:
        """
        Filter signals based on time requirements
        
        Args:
            signals: Input signals to filter
            data: Market data
            context: Filtering context
            
        Returns:
            Filtered signals
        """
        # Get current time from context or use current time
        current_time = context.get('current_time', datetime.now())
        
        # Parse time parameters
        start_time_str = self.parameters['start_time']
        end_time_str = self.parameters['end_time']
        
        start_hour, start_min = map(int, start_time_str.split(':'))
        end_hour, end_min = map(int, end_time_str.split(':'))
        
        start_time_obj = time(start_hour, start_min)
        end_time_obj = time(end_hour, end_min)
        
        # Get day of week (0=Monday, 6=Sunday)
        day_of_week = current_time.weekday()
        allowed_days = self.parameters['days_of_week']
        
        # Check if current time is within allowed range
        current_time_obj = current_time.time()
        is_allowed_time = (
            day_of_week in allowed_days and
            start_time_obj <= current_time_obj <= end_time_obj
        )
        
        if not is_allowed_time:
            # Outside allowed time range, set all signals to FLAT
            return {symbol: SignalType.FLAT for symbol in signals}
        
        # Within allowed time range, return signals unchanged
        return signals

class TrendFilter(FilterComponent):
    """Filters signals based on trend direction."""
    
    def __init__(self, 
                component_id: Optional[str] = None,
                trend_period: int = 50,
                uptrend_threshold: float = 0.0,
                downtrend_threshold: float = 0.0,
                allow_counter_trend: bool = False,
                price_column: str = 'close'):
        """
        Initialize trend filter
        
        Args:
            component_id: Unique component ID
            trend_period: Period for trend determination
            uptrend_threshold: Minimum slope for uptrend
            downtrend_threshold: Maximum slope for downtrend
            allow_counter_trend: Allow counter-trend signals
            price_column: Price column to use
        """
        super().__init__(component_id)
        self.parameters = {
            'trend_period': trend_period,
            'uptrend_threshold': uptrend_threshold,
            'downtrend_threshold': downtrend_threshold,
            'allow_counter_trend': allow_counter_trend,
            'price_column': price_column
        }
        self.description = f"Trend Filter ({trend_period} period)"
    
    def filter_signals(self, signals: Dict[str, SignalType], data: Dict[str, pd.DataFrame], 
                      context: Dict[str, Any]) -> Dict[str, SignalType]:
        """
        Filter signals based on trend requirements
        
        Args:
            signals: Input signals to filter
            data: Market data
            context: Filtering context
            
        Returns:
            Filtered signals
        """
        filtered_signals = signals.copy()
        
        trend_period = self.parameters['trend_period']
        uptrend_threshold = self.parameters['uptrend_threshold']
        downtrend_threshold = self.parameters['downtrend_threshold']
        allow_counter_trend = self.parameters['allow_counter_trend']
        price_col = self.parameters['price_column']
        
        for symbol, signal in signals.items():
            # Skip FLAT signals (nothing to filter)
            if signal == SignalType.FLAT:
                continue
            
            # Get market data for this symbol
            if symbol not in data:
                # No data, remove signal
                filtered_signals[symbol] = SignalType.FLAT
                continue
            
            df = data[symbol]
            if len(df) < trend_period:
                # Not enough history, remove signal
                filtered_signals[symbol] = SignalType.FLAT
                continue
            
            # Calculate trend
            prices = df[price_col].iloc[-trend_period:].values
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            
            # Normalize slope as percentage
            slope_pct = slope / prices[0] * 100
            
            # Determine trend direction
            if slope_pct > uptrend_threshold:
                trend = 'up'
            elif slope_pct < downtrend_threshold:
                trend = 'down'
            else:
                trend = 'sideways'
            
            # Filter based on trend direction
            if not allow_counter_trend:
                if (signal in [SignalType.LONG, SignalType.SCALE_UP] and trend != 'up'):
                    filtered_signals[symbol] = SignalType.FLAT
                elif (signal in [SignalType.SHORT, SignalType.SCALE_DOWN] and trend != 'down'):
                    filtered_signals[symbol] = SignalType.FLAT
        
        return filtered_signals

class MarketRegimeFilter(FilterComponent):
    """Filters signals based on market regime."""
    
    def __init__(self, 
                component_id: Optional[str] = None,
                allowed_conditions: List[MarketCondition] = None):
        """
        Initialize market regime filter
        
        Args:
            component_id: Unique component ID
            allowed_conditions: List of allowed market conditions
        """
        super().__init__(component_id)
        self.parameters = {
            'allowed_conditions': [c.name for c in (allowed_conditions or [])]
        }
        
        condition_names = [c.name for c in (allowed_conditions or [])]
        self.description = f"Market Regime Filter ({', '.join(condition_names)})"
    
    def filter_signals(self, signals: Dict[str, SignalType], data: Dict[str, pd.DataFrame], 
                      context: Dict[str, Any]) -> Dict[str, SignalType]:
        """
        Filter signals based on market regime
        
        Args:
            signals: Input signals to filter
            data: Market data
            context: Filtering context
            
        Returns:
            Filtered signals
        """
        # Get detected market conditions from context
        market_conditions = context.get('market_conditions', {})
        if not market_conditions:
            # No market conditions detected, can't filter
            return signals
        
        # Convert parameter names back to enums
        allowed_conditions = [
            MarketCondition[name] for name in self.parameters['allowed_conditions']
        ]
        
        if not allowed_conditions:
            # No allowed conditions specified, return signals unchanged
            return signals
        
        filtered_signals = signals.copy()
        
        for symbol, signal in signals.items():
            # Skip FLAT signals (nothing to filter)
            if signal == SignalType.FLAT:
                continue
            
            # Get conditions for this symbol
            symbol_conditions = market_conditions.get(symbol, [])
            
            # Check if any allowed condition is present
            if not any(condition in allowed_conditions for condition in symbol_conditions):
                # No allowed conditions present, filter out signal
                filtered_signals[symbol] = SignalType.FLAT
        
        return filtered_signals

class ConsolidationFilter(FilterComponent):
    """Filters signals based on price consolidation."""
    
    def __init__(self, 
                component_id: Optional[str] = None,
                lookback_period: int = 20,
                max_range_percentile: float = 30.0):
        """
        Initialize consolidation filter
        
        Args:
            component_id: Unique component ID
            lookback_period: Period for range calculation
            max_range_percentile: Maximum range percentile (0-100)
        """
        super().__init__(component_id)
        self.parameters = {
            'lookback_period': lookback_period,
            'max_range_percentile': max_range_percentile
        }
        self.description = f"Consolidation Filter ({lookback_period} bars)"
    
    def filter_signals(self, signals: Dict[str, SignalType], data: Dict[str, pd.DataFrame], 
                      context: Dict[str, Any]) -> Dict[str, SignalType]:
        """
        Filter signals based on price consolidation
        
        Args:
            signals: Input signals to filter
            data: Market data
            context: Filtering context
            
        Returns:
            Filtered signals
        """
        filtered_signals = signals.copy()
        
        lookback = self.parameters['lookback_period']
        max_percentile = self.parameters['max_range_percentile']
        
        for symbol, signal in signals.items():
            # Skip FLAT signals (nothing to filter)
            if signal == SignalType.FLAT:
                continue
            
            # Get market data for this symbol
            if symbol not in data:
                # No data, remove signal
                filtered_signals[symbol] = SignalType.FLAT
                continue
            
            df = data[symbol]
            if len(df) < lookback:
                # Not enough history, remove signal
                filtered_signals[symbol] = SignalType.FLAT
                continue
            
            # Calculate price ranges for each bar
            ranges = df['high'] - df['low']
            
            # Calculate current range
            current_range = ranges.iloc[-1]
            
            # Calculate range percentile
            range_history = ranges.iloc[-lookback:-1]  # Exclude current range
            if len(range_history) == 0:
                continue
                
            range_percentile = 100 * (sum(current_range < r for r in range_history) / len(range_history))
            
            # Filter based on percentile (only allow trading in tight ranges)
            if range_percentile > max_percentile:
                filtered_signals[symbol] = SignalType.FLAT
        
        return filtered_signals

class SignalConsistencyFilter(FilterComponent):
    """Filters signals based on consistency over multiple timeframes."""
    
    def __init__(self, 
                component_id: Optional[str] = None,
                required_consistency: float = 0.7,
                lookback_bars: int = 3):
        """
        Initialize signal consistency filter
        
        Args:
            component_id: Unique component ID
            required_consistency: Required signal consistency (0-1)
            lookback_bars: Number of bars to check for consistency
        """
        super().__init__(component_id)
        self.parameters = {
            'required_consistency': required_consistency,
            'lookback_bars': lookback_bars
        }
        self.description = f"Consistency Filter ({lookback_bars} bars, {required_consistency:.0%})"
        
        # Keep track of previous signals
        self.previous_signals = {}
    
    def filter_signals(self, signals: Dict[str, SignalType], data: Dict[str, pd.DataFrame], 
                      context: Dict[str, Any]) -> Dict[str, SignalType]:
        """
        Filter signals based on consistency over time
        
        Args:
            signals: Input signals to filter
            data: Market data
            context: Filtering context
            
        Returns:
            Filtered signals
        """
        filtered_signals = signals.copy()
        required_consistency = self.parameters['required_consistency']
        lookback = self.parameters['lookback_bars']
        
        # Get current time from context
        current_time = context.get('current_time', datetime.now())
        
        # Update signal history for each symbol
        for symbol, signal in signals.items():
            # Initialize history if needed
            if symbol not in self.previous_signals:
                self.previous_signals[symbol] = []
            
            # Add current signal to history
            self.previous_signals[symbol].append({
                'signal': signal,
                'time': current_time
            })
            
            # Keep only recent signals
            self.previous_signals[symbol] = self.previous_signals[symbol][-lookback:]
            
            # Skip if we don't have enough history
            if len(self.previous_signals[symbol]) < lookback:
                continue
            
            # Check consistency
            history = self.previous_signals[symbol]
            current_signal_type = signal
            
            # Count occurrences of current signal type
            signal_count = sum(1 for entry in history if entry['signal'] == current_signal_type)
            consistency = signal_count / len(history)
            
            # Filter based on consistency
            if consistency < required_consistency:
                filtered_signals[symbol] = SignalType.FLAT
        
        return filtered_signals
