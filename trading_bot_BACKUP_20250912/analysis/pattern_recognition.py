#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pattern Recognition Module

This module provides classes and functions for detecting trading patterns
in market data and calculating pattern match scores.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime
import logging
import os

# Import our pattern structure classes
from trading_bot.analysis.pattern_structure import (
    Pattern, PatternType, MarketContext, PatternRegistry,
    PriceActionPattern, CandlestickPattern, ChartFormationPattern,
    IndicatorSignalPattern, VolatilityPattern, MultiTimeframePattern
)

logger = logging.getLogger(__name__)

class PatternDetector:
    """Base class for pattern detection algorithms"""
    
    def __init__(self, lookback_periods: int = 20):
        """
        Initialize pattern detector.
        
        Args:
            lookback_periods: Number of bars to look back when analyzing patterns
        """
        self.lookback_periods = lookback_periods
    
    def detect(self, data: pd.DataFrame, pattern: Pattern) -> Dict[str, Any]:
        """
        Detect if the specified pattern exists in the data.
        
        Args:
            data: Market data in DataFrame format
            pattern: Pattern to detect
            
        Returns:
            Dictionary containing detection results with match information
        """
        # Base implementation returns no match
        return {
            "match": False,
            "confidence": 0.0,
            "location": None,
            "direction": None,
            "metadata": {}
        }

class PriceActionDetector(PatternDetector):
    """Detector for price action patterns"""
    
    def detect(self, data: pd.DataFrame, pattern: Pattern) -> Dict[str, Any]:
        """Detect price action patterns in the data"""
        if len(data) < pattern.parameters.get("bar_count", 3) + 1:
            return {"match": False, "confidence": 0.0}
        
        # Get the most recent bars for analysis
        recent_data = data.iloc[-pattern.parameters.get("bar_count", 3) - 1:]
        
        result = {
            "match": False, 
            "confidence": 0.0,
            "location": len(data) - 1,
            "direction": None,
            "metadata": {}
        }
        
        # Detect specific price action patterns
        if pattern.name == "pin_bar":
            result = self._detect_pin_bar(recent_data, pattern)
        elif pattern.name == "inside_bar":
            result = self._detect_inside_bar(recent_data, pattern)
        elif pattern.name == "outside_bar":
            result = self._detect_outside_bar(recent_data, pattern)
        
        return result
    
    def _detect_pin_bar(self, data: pd.DataFrame, pattern: Pattern) -> Dict[str, Any]:
        """Detect pin bar pattern"""
        if len(data) < 2:
            return {"match": False, "confidence": 0.0}
        
        current_bar = data.iloc[-1]
        
        # Calculate body and wick sizes
        body_size = abs(current_bar['Close'] - current_bar['Open'])
        total_range = current_bar['High'] - current_bar['Low']
        
        if total_range == 0:  # Avoid division by zero
            return {"match": False, "confidence": 0.0}
        
        body_ratio = body_size / total_range
        
        # Calculate wick sizes
        if current_bar['Close'] >= current_bar['Open']:  # Bullish candle
            upper_wick = current_bar['High'] - current_bar['Close']
            lower_wick = current_bar['Open'] - current_bar['Low']
            is_bullish = True
        else:  # Bearish candle
            upper_wick = current_bar['High'] - current_bar['Open']
            lower_wick = current_bar['Close'] - current_bar['Low']
            is_bullish = False
        
        upper_wick_ratio = upper_wick / total_range if total_range > 0 else 0
        lower_wick_ratio = lower_wick / total_range if total_range > 0 else 0
        
        # Pin bar criteria:
        # 1. Small body (less than half the total range)
        # 2. One wick at least 2x the size of the body
        # 3. The other wick is small
        min_body_size = pattern.parameters.get("min_body_size", 0.5)
        max_wick_ratio = pattern.parameters.get("max_wick_ratio", 0.3)
        
        is_pin_bar = (
            body_ratio <= min_body_size and
            (
                (upper_wick_ratio >= 2 * body_ratio and lower_wick_ratio <= max_wick_ratio) or
                (lower_wick_ratio >= 2 * body_ratio and upper_wick_ratio <= max_wick_ratio)
            )
        )
        
        # Determine direction
        direction = None
        if is_pin_bar:
            if upper_wick_ratio > lower_wick_ratio:
                direction = "sell"  # Pin bar with long upper wick suggests bearish move
            else:
                direction = "buy"   # Pin bar with long lower wick suggests bullish move
        
        # Calculate confidence based on the wick-to-body ratio
        dominant_wick = max(upper_wick_ratio, lower_wick_ratio)
        confidence = min(0.95, dominant_wick / body_ratio * 0.5) if body_ratio > 0 else 0.5
        
        return {
            "match": is_pin_bar,
            "confidence": confidence if is_pin_bar else 0.0,
            "location": len(data) - 1,
            "direction": direction,
            "metadata": {
                "body_ratio": body_ratio,
                "upper_wick_ratio": upper_wick_ratio,
                "lower_wick_ratio": lower_wick_ratio,
                "is_bullish": is_bullish
            }
        }
    
    def _detect_inside_bar(self, data: pd.DataFrame, pattern: Pattern) -> Dict[str, Any]:
        """Detect inside bar pattern"""
        if len(data) < 2:
            return {"match": False, "confidence": 0.0}
        
        current_bar = data.iloc[-1]
        previous_bar = data.iloc[-2]
        
        # Inside bar criteria
        is_inside = (
            current_bar['High'] <= previous_bar['High'] and
            current_bar['Low'] >= previous_bar['Low']
        )
        
        # Calculate how "inside" the bar is
        if is_inside:
            current_range = current_bar['High'] - current_bar['Low']
            previous_range = previous_bar['High'] - previous_bar['Low']
            
            # Avoid division by zero
            if previous_range == 0:
                inside_ratio = 0
            else:
                inside_ratio = 1 - (current_range / previous_range)
                
            # Higher confidence when the inside bar is much smaller
            confidence = min(0.9, inside_ratio + 0.3)
            
            # Direction is based on previous bar trend
            direction = "buy" if previous_bar['Close'] > previous_bar['Open'] else "sell"
        else:
            confidence = 0.0
            direction = None
        
        return {
            "match": is_inside,
            "confidence": confidence if is_inside else 0.0,
            "location": len(data) - 1,
            "direction": direction,
            "metadata": {
                "inside_ratio": inside_ratio if is_inside else 0
            }
        }
    
    def _detect_outside_bar(self, data: pd.DataFrame, pattern: Pattern) -> Dict[str, Any]:
        """Detect outside bar pattern"""
        if len(data) < 2:
            return {"match": False, "confidence": 0.0}
        
        current_bar = data.iloc[-1]
        previous_bar = data.iloc[-2]
        
        # Outside bar criteria
        is_outside = (
            current_bar['High'] > previous_bar['High'] and
            current_bar['Low'] < previous_bar['Low']
        )
        
        # Calculate how "outside" the bar is
        if is_outside:
            current_range = current_bar['High'] - current_bar['Low']
            previous_range = previous_bar['High'] - previous_bar['Low']
            
            # Avoid division by zero
            if previous_range == 0:
                outside_ratio = 1
            else:
                outside_ratio = (current_range / previous_range) - 1
                
            # Higher confidence when the outside bar is much larger
            confidence = min(0.9, outside_ratio + 0.3)
            
            # Direction is based on current bar
            direction = "buy" if current_bar['Close'] > current_bar['Open'] else "sell"
        else:
            confidence = 0.0
            direction = None
        
        return {
            "match": is_outside,
            "confidence": confidence if is_outside else 0.0,
            "location": len(data) - 1,
            "direction": direction,
            "metadata": {
                "outside_ratio": outside_ratio if is_outside else 0
            }
        }

class CandlestickDetector(PatternDetector):
    """Detector for candlestick patterns"""
    
    def detect(self, data: pd.DataFrame, pattern: Pattern) -> Dict[str, Any]:
        """Detect candlestick patterns in the data"""
        result = {
            "match": False, 
            "confidence": 0.0,
            "location": len(data) - 1,
            "direction": None,
            "metadata": {}
        }
        
        # Detect specific candlestick patterns
        if pattern.name == "doji":
            result = self._detect_doji(data, pattern)
        elif pattern.name == "engulfing":
            result = self._detect_engulfing(data, pattern)
        elif pattern.name == "evening_star":
            result = self._detect_evening_star(data, pattern)
        elif pattern.name == "morning_star":
            result = self._detect_morning_star(data, pattern)
        
        return result
    
    def _detect_doji(self, data: pd.DataFrame, pattern: Pattern) -> Dict[str, Any]:
        """Detect doji candlestick pattern"""
        if len(data) < 1:
            return {"match": False, "confidence": 0.0}
        
        current_bar = data.iloc[-1]
        
        # Calculate body and range
        body_size = abs(current_bar['Close'] - current_bar['Open'])
        bar_range = current_bar['High'] - current_bar['Low']
        
        # Avoid division by zero
        if bar_range == 0:
            return {"match": False, "confidence": 0.0}
        
        body_ratio = body_size / bar_range
        
        # Doji criteria: very small body relative to range
        is_doji = body_ratio <= 0.1
        
        # Calculate confidence
        confidence = max(0, 0.9 - (body_ratio * 5)) if is_doji else 0.0
        
        return {
            "match": is_doji,
            "confidence": confidence,
            "location": len(data) - 1,
            "direction": None,  # Doji represents indecision
            "metadata": {
                "body_ratio": body_ratio
            }
        }
    
    def _detect_engulfing(self, data: pd.DataFrame, pattern: Pattern) -> Dict[str, Any]:
        """Detect engulfing candlestick pattern"""
        if len(data) < 2:
            return {"match": False, "confidence": 0.0}
        
        current_bar = data.iloc[-1]
        previous_bar = data.iloc[-2]
        
        # Calculate bodies (uses real body, not including wicks)
        current_high = max(current_bar['Open'], current_bar['Close'])
        current_low = min(current_bar['Open'], current_bar['Close'])
        
        previous_high = max(previous_bar['Open'], previous_bar['Close'])
        previous_low = min(previous_bar['Open'], previous_bar['Close'])
        
        # Check if current bar's body completely engulfs previous bar's body
        is_engulfing = (current_high > previous_high and current_low < previous_low)
        
        # Determine pattern direction
        is_bullish = current_bar['Close'] > current_bar['Open']
        is_bearish = current_bar['Close'] < current_bar['Open']
        
        direction = None
        if is_engulfing:
            # Bullish engulfing: current bar bullish, previous bar bearish
            if is_bullish and previous_bar['Close'] < previous_bar['Open']:
                direction = "buy"
            # Bearish engulfing: current bar bearish, previous bar bullish
            elif is_bearish and previous_bar['Close'] > previous_bar['Open']:
                direction = "sell"
            else:
                # Not a true engulfing if bars are same direction
                is_engulfing = False
        
        # Calculate confidence based on size difference
        confidence = 0.0
        if is_engulfing:
            current_body_size = abs(current_bar['Close'] - current_bar['Open'])
            previous_body_size = abs(previous_bar['Close'] - previous_bar['Open'])
            
            # More confidence when current body is significantly larger
            if previous_body_size > 0:
                size_ratio = current_body_size / previous_body_size
                confidence = min(0.95, 0.5 + (size_ratio * 0.2))
        
        return {
            "match": is_engulfing,
            "confidence": confidence,
            "location": len(data) - 1,
            "direction": direction,
            "metadata": {
                "is_bullish": is_bullish,
                "is_bearish": is_bearish
            }
        }
    
    def _detect_evening_star(self, data: pd.DataFrame, pattern: Pattern) -> Dict[str, Any]:
        """Detect evening star candlestick pattern"""
        if len(data) < 3:
            return {"match": False, "confidence": 0.0}
        
        # Get the last three bars
        first_bar = data.iloc[-3]
        second_bar = data.iloc[-2]
        third_bar = data.iloc[-1]
        
        # Calculate bodies
        first_body = abs(first_bar['Close'] - first_bar['Open'])
        second_body = abs(second_bar['Close'] - second_bar['Open'])
        third_body = abs(third_bar['Close'] - third_bar['Open'])
        
        # Evening star criteria:
        # 1. First bar is bullish with large body
        # 2. Second bar gaps up and has small body (doji-like)
        # 3. Third bar is bearish and closes deep into first bar's body
        is_first_bullish = first_bar['Close'] > first_bar['Open']
        
        # Check if second bar has small body
        second_bar_range = second_bar['High'] - second_bar['Low']
        is_second_small_body = (second_body / second_bar_range <= 0.3) if second_bar_range > 0 else False
        
        # Check if second bar gaps up
        gaps_up = min(second_bar['Open'], second_bar['Close']) > first_bar['Close']
        
        # Check if third bar is bearish
        is_third_bearish = third_bar['Close'] < third_bar['Open']
        
        # Check if third bar closes deep into first bar's body
        closes_into_first = third_bar['Close'] < (first_bar['Open'] + (first_body * 0.5))
        
        is_evening_star = (
            is_first_bullish and 
            is_second_small_body and 
            (gaps_up or is_second_small_body) and  # Allow for no gap if second bar is very small
            is_third_bearish and 
            closes_into_first
        )
        
        confidence = 0.0
        if is_evening_star:
            # Calculate confidence based on third bar's penetration into first bar
            penetration = (first_bar['Close'] - third_bar['Close']) / first_body if first_body > 0 else 0
            confidence = min(0.95, 0.5 + (penetration * 0.5))
        
        return {
            "match": is_evening_star,
            "confidence": confidence,
            "location": len(data) - 1,
            "direction": "sell" if is_evening_star else None,
            "metadata": {
                "first_body": first_body,
                "second_body": second_body,
                "third_body": third_body
            }
        }
    
    def _detect_morning_star(self, data: pd.DataFrame, pattern: Pattern) -> Dict[str, Any]:
        """Detect morning star candlestick pattern"""
        if len(data) < 3:
            return {"match": False, "confidence": 0.0}
        
        # Get the last three bars
        first_bar = data.iloc[-3]
        second_bar = data.iloc[-2]
        third_bar = data.iloc[-1]
        
        # Calculate bodies
        first_body = abs(first_bar['Close'] - first_bar['Open'])
        second_body = abs(second_bar['Close'] - second_bar['Open'])
        third_body = abs(third_bar['Close'] - third_bar['Open'])
        
        # Morning star criteria:
        # 1. First bar is bearish with large body
        # 2. Second bar gaps down and has small body (doji-like)
        # 3. Third bar is bullish and closes deep into first bar's body
        is_first_bearish = first_bar['Close'] < first_bar['Open']
        
        # Check if second bar has small body
        second_bar_range = second_bar['High'] - second_bar['Low']
        is_second_small_body = (second_body / second_bar_range <= 0.3) if second_bar_range > 0 else False
        
        # Check if second bar gaps down
        gaps_down = max(second_bar['Open'], second_bar['Close']) < first_bar['Close']
        
        # Check if third bar is bullish
        is_third_bullish = third_bar['Close'] > third_bar['Open']
        
        # Check if third bar closes deep into first bar's body
        closes_into_first = third_bar['Close'] > (first_bar['Open'] - (first_body * 0.5))
        
        is_morning_star = (
            is_first_bearish and 
            is_second_small_body and 
            (gaps_down or is_second_small_body) and  # Allow for no gap if second bar is very small
            is_third_bullish and 
            closes_into_first
        )
        
        confidence = 0.0
        if is_morning_star:
            # Calculate confidence based on third bar's penetration into first bar
            penetration = (third_bar['Close'] - first_bar['Close']) / first_body if first_body > 0 else 0
            confidence = min(0.95, 0.5 + (penetration * 0.5))
        
        return {
            "match": is_morning_star,
            "confidence": confidence,
            "location": len(data) - 1,
            "direction": "buy" if is_morning_star else None,
            "metadata": {
                "first_body": first_body,
                "second_body": second_body,
                "third_body": third_body
            }
        }

# Initialize standard pattern detectors
PATTERN_DETECTORS = {
    PatternType.PRICE_ACTION.value: PriceActionDetector(),
    PatternType.CANDLESTICK.value: CandlestickDetector()
}

# Continue with other detector implementations in pattern_recognition_ext.py
