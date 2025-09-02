"""
Advanced technical indicators for EvoTrader.

This module extends the indicator_system with more sophisticated technical indicators
including pivot points, Ichimoku Cloud, and other advanced analysis tools.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from abc import ABC

from .indicator_system import Indicator, IndicatorFactory

logger = logging.getLogger(__name__)


class PivotPoints(Indicator):
    """
    Pivot Point indicator with support and resistance levels.
    
    Calculate standard pivot points and support/resistance levels
    based on previous periods' high, low, and close prices.
    """
    
    def __init__(
        self, 
        symbol: str, 
        pivot_type: str = 'standard', 
        timeframe: str = 'daily'
    ):
        """
        Initialize Pivot Points indicator.
        
        Args:
            symbol: Trading symbol
            pivot_type: Type of pivot calculation ('standard', 'fibonacci', 'woodie', 'camarilla', 'demark')
            timeframe: Timeframe for calculation ('daily', 'weekly', 'monthly')
        """
        super().__init__(symbol, {
            'pivot_type': pivot_type,
            'timeframe': timeframe
        })
        
        # Storage for pivot levels
        self.pivot: Optional[float] = None
        self.supports: List[Optional[float]] = [None, None, None]  # S1, S2, S3
        self.resistances: List[Optional[float]] = [None, None, None]  # R1, R2, R3
        self.mid_points: List[Optional[float]] = [None, None, None, None]  # M0, M1, M2, M3
        
        # Previous period's price data
        self.prev_high: Optional[float] = None
        self.prev_low: Optional[float] = None
        self.prev_close: Optional[float] = None
        self.prev_open: Optional[float] = None
        
        # Data tracking
        self.last_update_day: int = -1
        self.min_data_points = 2  # Need previous period data
    
    def update(self, candle: Dict[str, Any]) -> Optional[float]:
        """
        Update pivot points with new price data.
        
        Args:
            candle: Price candle with OHLCV data
            
        Returns:
            Pivot point value
        """
        # Check for required fields
        required_fields = ['high', 'low', 'close', 'timestamp']
        if not all(field in candle for field in required_fields):
            logger.warning(f"Missing required fields for pivot points: {[f for f in required_fields if f not in candle]}")
            return None
        
        # Extract current timestamp and determine if we need to recalculate
        timestamp = candle['timestamp']
        
        # Determine if we're in a new period for recalculation
        # In production, use actual date/time parsing
        # For now, use a simple tracking approach
        current_day = timestamp // (24 * 60 * 60)  # Convert to day
        
        # Update previous period's data
        if 'open' in candle:
            self.prev_open = candle['open']
        
        # Only recalculate once per period
        if current_day != self.last_update_day:
            # Save current period's values for next period's calculation
            self.prev_high = candle['high']
            self.prev_low = candle['low']
            self.prev_close = candle['close']
            
            # Calculate pivot points
            self._calculate_pivot_points()
            
            self.last_update_day = current_day
        
        # Store pivot as the primary indicator value
        if self.pivot is not None:
            self.values.append(self.pivot)
            self.is_ready = True
        else:
            self.values.append(None)
        
        return self.pivot
    
    def _calculate_pivot_points(self) -> None:
        """
        Calculate pivot points and support/resistance levels based on specified method.
        """
        if self.prev_high is None or self.prev_low is None or self.prev_close is None:
            return
        
        pivot_type = self.params['pivot_type'].lower()
        
        # Standard pivot points calculation
        if pivot_type == 'standard':
            self.pivot = (self.prev_high + self.prev_low + self.prev_close) / 3
            self.supports[0] = (2 * self.pivot) - self.prev_high  # S1
            self.resistances[0] = (2 * self.pivot) - self.prev_low  # R1
            self.supports[1] = self.pivot - (self.prev_high - self.prev_low)  # S2
            self.resistances[1] = self.pivot + (self.prev_high - self.prev_low)  # R2
            self.supports[2] = self.supports[0] - (self.prev_high - self.prev_low)  # S3
            self.resistances[2] = self.resistances[0] + (self.prev_high - self.prev_low)  # R3
            
            # Calculate mid-points (between pivot and S1/R1)
            self.mid_points[0] = (self.pivot + self.supports[0]) / 2  # M0
            self.mid_points[1] = (self.pivot + self.resistances[0]) / 2  # M1
            self.mid_points[2] = (self.supports[0] + self.supports[1]) / 2  # M2
            self.mid_points[3] = (self.resistances[0] + self.resistances[1]) / 2  # M3
            
        # Fibonacci pivot points
        elif pivot_type == 'fibonacci':
            self.pivot = (self.prev_high + self.prev_low + self.prev_close) / 3
            range_value = self.prev_high - self.prev_low
            
            self.supports[0] = self.pivot - 0.382 * range_value  # S1
            self.resistances[0] = self.pivot + 0.382 * range_value  # R1
            self.supports[1] = self.pivot - 0.618 * range_value  # S2
            self.resistances[1] = self.pivot + 0.618 * range_value  # R2
            self.supports[2] = self.pivot - 1.0 * range_value  # S3
            self.resistances[2] = self.pivot + 1.0 * range_value  # R3
            
        # Woodie's pivot points
        elif pivot_type == 'woodie':
            self.pivot = (self.prev_high + self.prev_low + 2 * self.prev_close) / 4
            
            self.supports[0] = (2 * self.pivot) - self.prev_high  # S1
            self.resistances[0] = (2 * self.pivot) - self.prev_low  # R1
            self.supports[1] = self.pivot - (self.prev_high - self.prev_low)  # S2
            self.resistances[1] = self.pivot + (self.prev_high - self.prev_low)  # R2
            
        # Camarilla pivot points
        elif pivot_type == 'camarilla':
            range_value = self.prev_high - self.prev_low
            
            self.pivot = (self.prev_high + self.prev_low + self.prev_close) / 3
            self.supports[0] = self.prev_close - (range_value * 1.1 / 12)  # S1
            self.resistances[0] = self.prev_close + (range_value * 1.1 / 12)  # R1
            self.supports[1] = self.prev_close - (range_value * 1.1 / 6)  # S2
            self.resistances[1] = self.prev_close + (range_value * 1.1 / 6)  # R2
            self.supports[2] = self.prev_close - (range_value * 1.1 / 4)  # S3
            self.resistances[2] = self.prev_close + (range_value * 1.1 / 4)  # R3
            
        # DeMark's pivot points
        elif pivot_type == 'demark':
            if self.prev_close > self.prev_open:
                self.pivot = self.prev_high + (2 * self.prev_low) + self.prev_close
            elif self.prev_close < self.prev_open:
                self.pivot = (2 * self.prev_high) + self.prev_low + self.prev_close
            else:
                self.pivot = self.prev_high + self.prev_low + (2 * self.prev_close)
                
            self.pivot = self.pivot / 4
            
            self.supports[0] = self.pivot - (self.prev_high - self.prev_low)  # S1
            self.resistances[0] = self.pivot + (self.prev_high - self.prev_low)  # R1
    
    def get_pivot(self) -> Optional[float]:
        """Get current pivot point value."""
        return self.pivot
    
    def get_supports(self) -> List[Optional[float]]:
        """Get current support levels."""
        return self.supports
    
    def get_resistances(self) -> List[Optional[float]]:
        """Get current resistance levels."""
        return self.resistances
    
    def get_mid_points(self) -> List[Optional[float]]:
        """Get current mid-point levels."""
        return self.mid_points
    
    def get_closest_level(self, price: float) -> Tuple[str, float, float]:
        """
        Get the closest pivot, support or resistance level to the current price.
        
        Args:
            price: Current price to compare against levels
            
        Returns:
            Tuple of (level_type, level_value, distance)
        """
        levels = []
        
        # Add pivot
        if self.pivot is not None:
            levels.append(('P', self.pivot))
            
        # Add supports
        for i, level in enumerate(self.supports):
            if level is not None:
                levels.append((f'S{i+1}', level))
                
        # Add resistances
        for i, level in enumerate(self.resistances):
            if level is not None:
                levels.append((f'R{i+1}', level))
                
        # Add mid-points
        for i, level in enumerate(self.mid_points):
            if level is not None:
                levels.append((f'M{i}', level))
        
        # Find closest level
        if not levels:
            return ('NONE', 0, float('inf'))
            
        closest = min(levels, key=lambda x: abs(x[1] - price))
        return (closest[0], closest[1], abs(closest[1] - price))


class IchimokuCloud(Indicator):
    """
    Ichimoku Cloud indicator.
    
    A comprehensive indicator showing support/resistance, trend direction,
    and momentum using five distinct lines and a "cloud" area.
    """
    
    def __init__(
        self, 
        symbol: str, 
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        displacement: int = 26
    ):
        """
        Initialize Ichimoku Cloud indicator.
        
        Args:
            symbol: Trading symbol
            tenkan_period: Tenkan-sen (Conversion Line) period
            kijun_period: Kijun-sen (Base Line) period
            senkou_b_period: Senkou Span B period
            displacement: Displacement period for Senkou Span projections
        """
        super().__init__(symbol, {
            'tenkan_period': tenkan_period,
            'kijun_period': kijun_period,
            'senkou_b_period': senkou_b_period,
            'displacement': displacement
        })
        
        # History of highs and lows
        self.highs: List[Optional[float]] = []
        self.lows: List[Optional[float]] = []
        
        # Ichimoku components
        self.tenkan_sen: List[Optional[float]] = []  # Conversion Line
        self.kijun_sen: List[Optional[float]] = []  # Base Line
        self.senkou_span_a: List[Optional[float]] = []  # Leading Span A
        self.senkou_span_b: List[Optional[float]] = []  # Leading Span B
        self.chikou_span: List[Optional[float]] = []  # Lagging Span
        
        self.min_data_points = max(tenkan_period, kijun_period, senkou_b_period)
    
    def update(self, candle: Dict[str, Any]) -> Optional[float]:
        """
        Update Ichimoku Cloud with new price data.
        
        Args:
            candle: Price candle with OHLCV data
            
        Returns:
            Tenkan-sen value (as the primary value)
        """
        # Check for required fields
        if not all(field in candle for field in ['high', 'low', 'close']):
            logger.warning(f"Missing required fields for Ichimoku Cloud: high, low, or close")
            return None
        
        # Add to price history
        self.highs.append(candle['high'])
        self.lows.append(candle['low'])
        
        # Calculate components if we have enough data
        tp = self.params['tenkan_period']
        kp = self.params['kijun_period']
        sp = self.params['senkou_b_period']
        disp = self.params['displacement']
        
        # Calculate Tenkan-sen (Conversion Line)
        if len(self.highs) >= tp:
            tenkan = self._calculate_average_line(self.highs[-tp:], self.lows[-tp:])
            self.tenkan_sen.append(tenkan)
        else:
            self.tenkan_sen.append(None)
        
        # Calculate Kijun-sen (Base Line)
        if len(self.highs) >= kp:
            kijun = self._calculate_average_line(self.highs[-kp:], self.lows[-kp:])
            self.kijun_sen.append(kijun)
        else:
            self.kijun_sen.append(None)
        
        # Calculate Senkou Span A (Leading Span A)
        if len(self.tenkan_sen) > 0 and len(self.kijun_sen) > 0:
            tenkan_current = self.tenkan_sen[-1]
            kijun_current = self.kijun_sen[-1]
            
            if tenkan_current is not None and kijun_current is not None:
                senkou_a = (tenkan_current + kijun_current) / 2
                self.senkou_span_a.append(senkou_a)
            else:
                self.senkou_span_a.append(None)
        else:
            self.senkou_span_a.append(None)
        
        # Calculate Senkou Span B (Leading Span B)
        if len(self.highs) >= sp:
            senkou_b = self._calculate_average_line(self.highs[-sp:], self.lows[-sp:])
            self.senkou_span_b.append(senkou_b)
        else:
            self.senkou_span_b.append(None)
        
        # Calculate Chikou Span (Lagging Span)
        # This is just the current close price, but it's plotted disp periods back
        self.chikou_span.append(candle['close'])
        
        # Use Tenkan-sen as the primary indicator value
        if self.tenkan_sen and self.tenkan_sen[-1] is not None:
            self.values.append(self.tenkan_sen[-1])
            
            # Mark as ready when we have enough data for all components
            if (len(self.tenkan_sen) > 0 and len(self.kijun_sen) > 0 and 
                len(self.senkou_span_a) > 0 and len(self.senkou_span_b) > 0 and
                all(v is not None for v in [
                    self.tenkan_sen[-1], self.kijun_sen[-1], 
                    self.senkou_span_a[-1], self.senkou_span_b[-1]])):
                self.is_ready = True
        else:
            self.values.append(None)
        
        return self.tenkan_sen[-1] if self.tenkan_sen else None
    
    def _calculate_average_line(
        self, 
        highs: List[Optional[float]], 
        lows: List[Optional[float]]
    ) -> Optional[float]:
        """
        Calculate the average of highest high and lowest low.
        
        Args:
            highs: List of high prices
            lows: List of low prices
            
        Returns:
            Average of highest high and lowest low
        """
        # Filter out None values
        highs_filtered = [h for h in highs if h is not None]
        lows_filtered = [l for l in lows if l is not None]
        
        if not highs_filtered or not lows_filtered:
            return None
            
        highest_high = max(highs_filtered)
        lowest_low = min(lows_filtered)
        
        return (highest_high + lowest_low) / 2
    
    def get_tenkan_sen(self) -> List[Optional[float]]:
        """Get Tenkan-sen (Conversion Line) values."""
        return self.tenkan_sen
    
    def get_kijun_sen(self) -> List[Optional[float]]:
        """Get Kijun-sen (Base Line) values."""
        return self.kijun_sen
    
    def get_senkou_span_a(self) -> List[Optional[float]]:
        """Get Senkou Span A (Leading Span A) values."""
        return self.senkou_span_a
    
    def get_senkou_span_b(self) -> List[Optional[float]]:
        """Get Senkou Span B (Leading Span B) values."""
        return self.senkou_span_b
    
    def get_chikou_span(self) -> List[Optional[float]]:
        """Get Chikou Span (Lagging Span) values."""
        return self.chikou_span
    
    def get_cloud_top(self) -> List[Optional[float]]:
        """Get the top of the cloud for each period."""
        result = []
        for i in range(len(self.senkou_span_a)):
            if i < len(self.senkou_span_a) and i < len(self.senkou_span_b):
                a = self.senkou_span_a[i]
                b = self.senkou_span_b[i]
                if a is not None and b is not None:
                    result.append(max(a, b))
                else:
                    result.append(None)
            else:
                result.append(None)
        return result
    
    def get_cloud_bottom(self) -> List[Optional[float]]:
        """Get the bottom of the cloud for each period."""
        result = []
        for i in range(len(self.senkou_span_a)):
            if i < len(self.senkou_span_a) and i < len(self.senkou_span_b):
                a = self.senkou_span_a[i]
                b = self.senkou_span_b[i]
                if a is not None and b is not None:
                    result.append(min(a, b))
                else:
                    result.append(None)
            else:
                result.append(None)
        return result
    
    def get_cloud_color(self, index: int = -1) -> str:
        """
        Get the color of the cloud at a specific index.
        
        Args:
            index: Index to check, default is the latest
            
        Returns:
            'green' if bullish (A > B), 'red' if bearish (A < B), 'none' if undetermined
        """
        if 0 <= index < len(self.senkou_span_a) and 0 <= index < len(self.senkou_span_b):
            a = self.senkou_span_a[index]
            b = self.senkou_span_b[index]
            if a is not None and b is not None:
                return 'green' if a > b else 'red'
        
        return 'none'
    
    def is_price_above_cloud(self, price: float, index: int = -1) -> bool:
        """
        Check if price is above the cloud.
        
        Args:
            price: Price to check
            index: Index to check, default is the latest
            
        Returns:
            True if price is above the cloud, False otherwise
        """
        if 0 <= index < len(self.senkou_span_a) and 0 <= index < len(self.senkou_span_b):
            a = self.senkou_span_a[index]
            b = self.senkou_span_b[index]
            if a is not None and b is not None:
                return price > max(a, b)
        
        return False
    
    def is_price_below_cloud(self, price: float, index: int = -1) -> bool:
        """
        Check if price is below the cloud.
        
        Args:
            price: Price to check
            index: Index to check, default is the latest
            
        Returns:
            True if price is below the cloud, False otherwise
        """
        if 0 <= index < len(self.senkou_span_a) and 0 <= index < len(self.senkou_span_b):
            a = self.senkou_span_a[index]
            b = self.senkou_span_b[index]
            if a is not None and b is not None:
                return price < min(a, b)
        
        return False
    
    def is_tenkan_kijun_cross(self, index: int = -1) -> Optional[str]:
        """
        Check if there's a Tenkan/Kijun cross at a specific index.
        
        Args:
            index: Index to check, default is the latest
            
        Returns:
            'bullish' for bullish cross, 'bearish' for bearish cross, None if no cross or cannot determine
        """
        if (len(self.tenkan_sen) < 2 or len(self.kijun_sen) < 2 or 
            index < 0 or index >= len(self.tenkan_sen) - 1 or index >= len(self.kijun_sen) - 1):
            return None
            
        curr_tenkan = self.tenkan_sen[index]
        prev_tenkan = self.tenkan_sen[index - 1]
        curr_kijun = self.kijun_sen[index]
        prev_kijun = self.kijun_sen[index - 1]
        
        if all(v is not None for v in [curr_tenkan, prev_tenkan, curr_kijun, prev_kijun]):
            # Bullish cross: Tenkan crosses above Kijun
            if curr_tenkan > curr_kijun and prev_tenkan <= prev_kijun:
                return 'bullish'
                
            # Bearish cross: Tenkan crosses below Kijun
            elif curr_tenkan < curr_kijun and prev_tenkan >= prev_kijun:
                return 'bearish'
        
        return None


# Register advanced indicators with the factory
def register_advanced_indicators() -> None:
    """Register advanced indicators with the IndicatorFactory."""
    IndicatorFactory.register_indicator('pivot_points', PivotPoints)
    IndicatorFactory.register_indicator('ichimoku', IchimokuCloud)
