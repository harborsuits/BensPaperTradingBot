#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Pattern Recognition Module

This module provides additional pattern detectors for chart formations,
indicator signals, and volatility patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
import logging

# Import our pattern structure classes
from trading_bot.analysis.pattern_structure import (
    Pattern, PatternType, MarketContext, PatternRegistry,
    ChartFormationPattern, IndicatorSignalPattern, VolatilityPattern
)

# Import base detector class
from trading_bot.analysis.pattern_recognition import PatternDetector

logger = logging.getLogger(__name__)

class ChartFormationDetector(PatternDetector):
    """Detector for chart formation patterns"""
    
    def detect(self, data: pd.DataFrame, pattern: Pattern) -> Dict[str, Any]:
        """Detect chart formations in the data"""
        result = {
            "match": False, 
            "confidence": 0.0,
            "location": len(data) - 1,
            "direction": None,
            "metadata": {}
        }
        
        # Detect specific chart formations
        if pattern.name == "double_top":
            result = self._detect_double_top(data, pattern)
        elif pattern.name == "double_bottom":
            result = self._detect_double_bottom(data, pattern)
        elif pattern.name == "head_and_shoulders":
            result = self._detect_head_and_shoulders(data, pattern)
        
        return result
    
    def _detect_double_top(self, data: pd.DataFrame, pattern: Pattern) -> Dict[str, Any]:
        """Detect double top chart pattern"""
        # Need at least 10 bars to detect a valid double top
        lookback = pattern.parameters.get("min_duration_bars", 10)
        if len(data) < lookback + 5:  # Need extra bars before pattern
            return {"match": False, "confidence": 0.0}
        
        # Get data window for analysis
        analysis_data = data.iloc[-lookback-5:]
        
        # Find peaks (local maxima)
        peaks = []
        for i in range(1, len(analysis_data) - 1):
            if (analysis_data.iloc[i]['High'] > analysis_data.iloc[i-1]['High'] and 
                analysis_data.iloc[i]['High'] > analysis_data.iloc[i+1]['High']):
                peaks.append((i, analysis_data.iloc[i]['High']))
        
        # Need at least 2 peaks to form a double top
        if len(peaks) < 2:
            return {"match": False, "confidence": 0.0}
        
        # Find the two highest peaks
        peaks.sort(key=lambda x: x[1], reverse=True)
        top_peaks = peaks[:2]
        top_peaks.sort(key=lambda x: x[0])  # Sort by index for time order
        
        peak1_idx, peak1_val = top_peaks[0]
        peak2_idx, peak2_val = top_peaks[1]
        
        # Double top criteria:
        # 1. Two peaks at approximately the same level
        # 2. Peaks should be separated by at least a few bars
        # 3. There should be a trough between them
        # 4. Price should break below the trough after the second peak
        
        # Check peak separation
        min_separation = 3  # Minimum bars between peaks
        if peak2_idx - peak1_idx < min_separation:
            return {"match": False, "confidence": 0.0}
        
        # Check peak equality
        max_deviation = pattern.parameters.get("max_deviation", 0.1)
        avg_peak = (peak1_val + peak2_val) / 2
        peaks_equal = abs(peak1_val - peak2_val) / avg_peak <= max_deviation
        
        if not peaks_equal:
            return {"match": False, "confidence": 0.0}
        
        # Find the trough between peaks
        trough_val = float('inf')
        trough_idx = -1
        
        for i in range(peak1_idx + 1, peak2_idx):
            if analysis_data.iloc[i]['Low'] < trough_val:
                trough_val = analysis_data.iloc[i]['Low']
                trough_idx = i
        
        if trough_idx == -1:
            return {"match": False, "confidence": 0.0}
        
        # Check if price has broken below the trough after the second peak
        has_broken = False
        for i in range(peak2_idx + 1, len(analysis_data)):
            if analysis_data.iloc[i]['Close'] < trough_val:
                has_broken = True
                break
        
        # Calculate confidence based on:
        # - How equal the peaks are
        # - The depth of the trough relative to the peaks
        # - Whether a breakdown has occurred
        peak_diff_pct = abs(peak1_val - peak2_val) / avg_peak
        trough_depth = avg_peak - trough_val
        pattern_height = avg_peak - trough_val
        
        confidence = 0.0
        if peaks_equal:
            confidence = 0.5  # Base confidence for equal peaks
            
            # Add confidence for deeper trough
            if pattern_height > 0:
                depth_ratio = trough_depth / avg_peak
                confidence += min(0.3, depth_ratio * 2)
            
            # Add confidence for breakdown
            if has_broken:
                confidence += 0.2
        
        is_double_top = peaks_equal and (has_broken or (peak2_idx >= len(analysis_data) - 3))
        
        return {
            "match": is_double_top,
            "confidence": confidence if is_double_top else 0.0,
            "location": len(data) - 1,
            "direction": "sell" if is_double_top else None,
            "metadata": {
                "peak1": (peak1_idx, peak1_val),
                "peak2": (peak2_idx, peak2_val),
                "trough": (trough_idx, trough_val),
                "has_broken": has_broken
            }
        }
    
    def _detect_double_bottom(self, data: pd.DataFrame, pattern: Pattern) -> Dict[str, Any]:
        """Detect double bottom chart pattern"""
        # Need at least 10 bars to detect a valid double bottom
        lookback = pattern.parameters.get("min_duration_bars", 10)
        if len(data) < lookback + 5:  # Need extra bars before pattern
            return {"match": False, "confidence": 0.0}
        
        # Get data window for analysis
        analysis_data = data.iloc[-lookback-5:]
        
        # Find troughs (local minima)
        troughs = []
        for i in range(1, len(analysis_data) - 1):
            if (analysis_data.iloc[i]['Low'] < analysis_data.iloc[i-1]['Low'] and 
                analysis_data.iloc[i]['Low'] < analysis_data.iloc[i+1]['Low']):
                troughs.append((i, analysis_data.iloc[i]['Low']))
        
        # Need at least 2 troughs to form a double bottom
        if len(troughs) < 2:
            return {"match": False, "confidence": 0.0}
        
        # Find the two lowest troughs
        troughs.sort(key=lambda x: x[1])
        bottom_troughs = troughs[:2]
        bottom_troughs.sort(key=lambda x: x[0])  # Sort by index for time order
        
        trough1_idx, trough1_val = bottom_troughs[0]
        trough2_idx, trough2_val = bottom_troughs[1]
        
        # Double bottom criteria:
        # 1. Two troughs at approximately the same level
        # 2. Troughs should be separated by at least a few bars
        # 3. There should be a peak between them
        # 4. Price should break above the peak after the second trough
        
        # Check trough separation
        min_separation = 3  # Minimum bars between troughs
        if trough2_idx - trough1_idx < min_separation:
            return {"match": False, "confidence": 0.0}
        
        # Check trough equality
        max_deviation = pattern.parameters.get("max_deviation", 0.1)
        avg_trough = (trough1_val + trough2_val) / 2
        troughs_equal = abs(trough1_val - trough2_val) / avg_trough <= max_deviation
        
        if not troughs_equal:
            return {"match": False, "confidence": 0.0}
        
        # Find the peak between troughs
        peak_val = float('-inf')
        peak_idx = -1
        
        for i in range(trough1_idx + 1, trough2_idx):
            if analysis_data.iloc[i]['High'] > peak_val:
                peak_val = analysis_data.iloc[i]['High']
                peak_idx = i
        
        if peak_idx == -1:
            return {"match": False, "confidence": 0.0}
        
        # Check if price has broken above the peak after the second trough
        has_broken = False
        for i in range(trough2_idx + 1, len(analysis_data)):
            if analysis_data.iloc[i]['Close'] > peak_val:
                has_broken = True
                break
        
        # Calculate confidence based on:
        # - How equal the troughs are
        # - The height of the peak relative to the troughs
        # - Whether a breakout has occurred
        trough_diff_pct = abs(trough1_val - trough2_val) / avg_trough
        peak_height = peak_val - avg_trough
        pattern_height = peak_val - avg_trough
        
        confidence = 0.0
        if troughs_equal:
            confidence = 0.5  # Base confidence for equal troughs
            
            # Add confidence for higher peak
            if pattern_height > 0:
                height_ratio = peak_height / avg_trough
                confidence += min(0.3, height_ratio * 2)
            
            # Add confidence for breakout
            if has_broken:
                confidence += 0.2
        
        is_double_bottom = troughs_equal and (has_broken or (trough2_idx >= len(analysis_data) - 3))
        
        return {
            "match": is_double_bottom,
            "confidence": confidence if is_double_bottom else 0.0,
            "location": len(data) - 1,
            "direction": "buy" if is_double_bottom else None,
            "metadata": {
                "trough1": (trough1_idx, trough1_val),
                "trough2": (trough2_idx, trough2_val),
                "peak": (peak_idx, peak_val),
                "has_broken": has_broken
            }
        }
    
    def _detect_head_and_shoulders(self, data: pd.DataFrame, pattern: Pattern) -> Dict[str, Any]:
        """Detect head and shoulders pattern - a reliable reversal pattern"""
        # Need enough bars to detect this complex pattern
        lookback = pattern.parameters.get("min_duration_bars", 20)
        if len(data) < lookback:
            return {"match": False, "confidence": 0.0}
        
        # Use only the specified lookback period
        analysis_data = data.iloc[-lookback:]
        
        # Find peaks (local maxima)
        peaks = []
        for i in range(2, len(analysis_data) - 2):
            if (analysis_data.iloc[i]['High'] > analysis_data.iloc[i-1]['High'] and 
                analysis_data.iloc[i]['High'] > analysis_data.iloc[i-2]['High'] and
                analysis_data.iloc[i]['High'] > analysis_data.iloc[i+1]['High'] and
                analysis_data.iloc[i]['High'] > analysis_data.iloc[i+2]['High']):
                peaks.append((i, analysis_data.iloc[i]['High']))
        
        # Need at least 3 peaks to form a head and shoulders
        if len(peaks) < 3:
            return {"match": False, "confidence": 0.0}
        
        # Sort peaks by height
        peaks_by_height = sorted(peaks, key=lambda x: x[1], reverse=True)
        
        # Get the highest peak (head)
        head_idx, head_val = peaks_by_height[0]
        
        # Find candidates for left and right shoulders
        left_candidates = [(i, val) for i, val in peaks if i < head_idx]
        right_candidates = [(i, val) for i, val in peaks if i > head_idx]
        
        # Need at least one peak on each side
        if not left_candidates or not right_candidates:
            return {"match": False, "confidence": 0.0}
        
        # Get the highest peak on each side
        left_shoulder = max(left_candidates, key=lambda x: x[1])
        right_shoulder = max(right_candidates, key=lambda x: x[1])
        
        left_idx, left_val = left_shoulder
        right_idx, right_val = right_shoulder
        
        # H&S criteria:
        # 1. Three peaks with the middle one (head) being the highest
        # 2. The two shoulders should be at approximately the same level
        # 3. There should be a neckline connecting the troughs between peaks
        # 4. Price should break below the neckline after the right shoulder
        
        # Check if head is higher than both shoulders
        if head_val <= left_val or head_val <= right_val:
            return {"match": False, "confidence": 0.0}
        
        # Check if shoulders are at similar levels
        max_deviation = pattern.parameters.get("max_deviation", 0.1)
        avg_shoulder = (left_val + right_val) / 2
        shoulders_equal = abs(left_val - right_val) / avg_shoulder <= max_deviation
        
        if not shoulders_equal:
            return {"match": False, "confidence": 0.0}
        
        # Find the troughs between peaks (neckline points)
        left_trough_idx = -1
        left_trough_val = float('inf')
        for i in range(left_idx + 1, head_idx):
            if analysis_data.iloc[i]['Low'] < left_trough_val:
                left_trough_val = analysis_data.iloc[i]['Low']
                left_trough_idx = i
        
        right_trough_idx = -1
        right_trough_val = float('inf')
        for i in range(head_idx + 1, right_idx):
            if analysis_data.iloc[i]['Low'] < right_trough_val:
                right_trough_val = analysis_data.iloc[i]['Low']
                right_trough_idx = i
        
        if left_trough_idx == -1 or right_trough_idx == -1:
            return {"match": False, "confidence": 0.0}
        
        # Calculate neckline (can be sloped)
        # Simplified: take average of the two troughs
        neckline_val = (left_trough_val + right_trough_val) / 2
        
        # Check for breakout below neckline
        has_broken = False
        for i in range(right_idx + 1, len(analysis_data)):
            if analysis_data.iloc[i]['Close'] < neckline_val:
                has_broken = True
                break
        
        # Calculate pattern metrics for confidence
        pattern_height = head_val - neckline_val
        shoulder_ratio = avg_shoulder / head_val
        
        # Calculate confidence
        confidence = 0.0
        if head_val > left_val and head_val > right_val:
            # Base confidence
            confidence = 0.4
            
            # Add for shoulder equality
            shoulder_equality = 1 - (abs(left_val - right_val) / avg_shoulder)
            confidence += min(0.3, shoulder_equality * 0.3)
            
            # Add for proper shoulder ratio (ideally around 0.7-0.85 of head height)
            if 0.65 <= shoulder_ratio <= 0.9:
                confidence += 0.2
            
            # Add for breakout
            if has_broken:
                confidence += 0.1
        
        is_head_shoulders = (head_val > left_val and head_val > right_val and 
                            shoulders_equal and 
                            (has_broken or right_idx >= len(analysis_data) - 5))
        
        return {
            "match": is_head_shoulders,
            "confidence": confidence if is_head_shoulders else 0.0,
            "location": len(data) - 1,
            "direction": "sell" if is_head_shoulders else None,
            "metadata": {
                "head": (head_idx, head_val),
                "left_shoulder": (left_idx, left_val),
                "right_shoulder": (right_idx, right_val),
                "neckline": neckline_val,
                "has_broken": has_broken
            }
        }

class IndicatorSignalDetector(PatternDetector):
    """Detector for indicator-based signals"""
    
    def detect(self, data: pd.DataFrame, pattern: Pattern) -> Dict[str, Any]:
        """Detect indicator signals in the data"""
        result = {
            "match": False, 
            "confidence": 0.0,
            "location": len(data) - 1,
            "direction": None,
            "metadata": {}
        }
        
        # Calculate indicators if not already in data
        data = self._ensure_indicators(data, pattern)
        
        # Detect specific indicator signals
        if pattern.name == "macd_crossover":
            result = self._detect_macd_crossover(data, pattern)
        elif pattern.name == "rsi_oversold_bounce":
            result = self._detect_rsi_oversold_bounce(data, pattern)
        elif pattern.name == "rsi_overbought_drop":
            result = self._detect_rsi_overbought_drop(data, pattern)
        
        return result
    
    def _ensure_indicators(self, data: pd.DataFrame, pattern: Pattern) -> pd.DataFrame:
        """Calculate necessary indicators if they don't exist in data"""
        df = data.copy()
        
        # RSI
        if pattern.name.startswith("rsi_") and "RSI" not in df.columns:
            df = self._calculate_rsi(df)
        
        # MACD
        if pattern.name.startswith("macd_") and "MACD" not in df.columns:
            df = self._calculate_macd(df)
        
        return df
    
    def _calculate_rsi(self, data: pd.DataFrame, period=14) -> pd.DataFrame:
        """Calculate RSI indicator"""
        df = data.copy()
        
        # Calculate price changes
        delta = df['Close'].diff()
        
        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = -loss  # Make losses positive
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _calculate_macd(self, data: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
        """Calculate MACD indicator"""
        df = data.copy()
        
        # Calculate EMAs
        df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
        df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
        
        # Calculate MACD line and signal line
        df['MACD'] = df['EMA_fast'] - df['EMA_slow']
        df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        return df
    
    def _detect_macd_crossover(self, data: pd.DataFrame, pattern: Pattern) -> Dict[str, Any]:
        """Detect MACD line crossing signal line"""
        if len(data) < 3 or 'MACD' not in data.columns or 'MACD_signal' not in data.columns:
            return {"match": False, "confidence": 0.0}
        
        # Get current and previous values
        curr = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Check for crossover
        bullish_cross = prev['MACD'] <= prev['MACD_signal'] and curr['MACD'] > curr['MACD_signal']
        bearish_cross = prev['MACD'] >= prev['MACD_signal'] and curr['MACD'] < curr['MACD_signal']
        
        is_crossover = bullish_cross or bearish_cross
        direction = "buy" if bullish_cross else "sell" if bearish_cross else None
        
        # Calculate confidence based on:
        # - Strength of the crossover (difference between MACD and signal)
        # - Histogram growth
        confidence = 0.0
        if is_crossover:
            # Base confidence for crossover
            confidence = 0.5
            
            # Add confidence for stronger crossover
            cross_strength = abs(curr['MACD'] - curr['MACD_signal']) / abs(curr['MACD'])
            confidence += min(0.3, cross_strength * 3)
            
            # Add confidence based on histogram growth
            if 'MACD_hist' in data.columns:
                hist_growth = abs(curr['MACD_hist']) / abs(prev['MACD_hist']) if abs(prev['MACD_hist']) > 0 else 1
                if hist_growth > 1:
                    confidence += min(0.2, (hist_growth - 1) * 0.5)
        
        return {
            "match": is_crossover,
            "confidence": min(0.95, confidence) if is_crossover else 0.0,
            "location": len(data) - 1,
            "direction": direction,
            "metadata": {
                "bullish": bullish_cross,
                "bearish": bearish_cross,
                "macd_value": curr['MACD'],
                "signal_value": curr['MACD_signal'],
                "histogram": curr.get('MACD_hist', 0)
            }
        }
    
    def _detect_rsi_oversold_bounce(self, data: pd.DataFrame, pattern: Pattern) -> Dict[str, Any]:
        """Detect RSI moving up from oversold territory"""
        if len(data) < 3 or 'RSI' not in data.columns:
            return {"match": False, "confidence": 0.0}
        
        # Get recent values
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Get oversold threshold (default 30)
        oversold = pattern.parameters.get("oversold_threshold", 30)
        
        # Check for oversold bounce:
        # - Previous RSI was below the oversold threshold
        # - Current RSI is moving up
        was_oversold = previous['RSI'] < oversold
        is_bouncing = current['RSI'] > previous['RSI']
        has_crossed = previous['RSI'] < oversold and current['RSI'] >= oversold
        
        is_oversold_bounce = (was_oversold and is_bouncing)
        
        # Calculate confidence based on:
        # - How oversold it was (lower = stronger signal)
        # - Strength of bounce
        # - Whether it crossed back above threshold
        confidence = 0.0
        if is_oversold_bounce:
            # Base confidence
            confidence = 0.4
            
            # Add confidence based on how oversold it was
            oversold_strength = (oversold - previous['RSI']) / oversold
            confidence += min(0.3, oversold_strength * 0.6)
            
            # Add confidence based on bounce strength
            bounce_strength = (current['RSI'] - previous['RSI']) / previous['RSI']
            confidence += min(0.2, bounce_strength * 2)
            
            # Add if crossed back above threshold
            if has_crossed:
                confidence += 0.1
        
        return {
            "match": is_oversold_bounce,
            "confidence": confidence if is_oversold_bounce else 0.0,
            "location": len(data) - 1,
            "direction": "buy" if is_oversold_bounce else None,
            "metadata": {
                "current_rsi": current['RSI'],
                "previous_rsi": previous['RSI'],
                "threshold": oversold,
                "crossed_threshold": has_crossed
            }
        }
    
    def _detect_rsi_overbought_drop(self, data: pd.DataFrame, pattern: Pattern) -> Dict[str, Any]:
        """Detect RSI moving down from overbought territory"""
        if len(data) < 3 or 'RSI' not in data.columns:
            return {"match": False, "confidence": 0.0}
        
        # Get recent values
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Get overbought threshold (default 70)
        overbought = pattern.parameters.get("overbought_threshold", 70)
        
        # Check for overbought drop:
        # - Previous RSI was above the overbought threshold
        # - Current RSI is moving down
        was_overbought = previous['RSI'] > overbought
        is_dropping = current['RSI'] < previous['RSI']
        has_crossed = previous['RSI'] > overbought and current['RSI'] <= overbought
        
        is_overbought_drop = (was_overbought and is_dropping)
        
        # Calculate confidence based on:
        # - How overbought it was (higher = stronger signal)
        # - Strength of drop
        # - Whether it crossed back below threshold
        confidence = 0.0
        if is_overbought_drop:
            # Base confidence
            confidence = 0.4
            
            # Add confidence based on how overbought it was
            overbought_strength = (previous['RSI'] - overbought) / (100 - overbought)
            confidence += min(0.3, overbought_strength * 0.6)
            
            # Add confidence based on drop strength
            drop_strength = (previous['RSI'] - current['RSI']) / previous['RSI']
            confidence += min(0.2, drop_strength * 2)
            
            # Add if crossed back below threshold
            if has_crossed:
                confidence += 0.1
        
        return {
            "match": is_overbought_drop,
            "confidence": confidence if is_overbought_drop else None,
            "location": len(data) - 1,
            "direction": "sell" if is_overbought_drop else None,
            "metadata": {
                "current_rsi": current['RSI'],
                "previous_rsi": previous['RSI'],
                "threshold": overbought,
                "crossed_threshold": has_crossed
            }
        }

# Add the detectors to the global dictionary
def register_extended_detectors():
    from trading_bot.analysis.pattern_recognition import PATTERN_DETECTORS
    
    PATTERN_DETECTORS[PatternType.CHART_FORMATION.value] = ChartFormationDetector()
    PATTERN_DETECTORS[PatternType.INDICATOR_SIGNAL.value] = IndicatorSignalDetector()
    
    return PATTERN_DETECTORS
