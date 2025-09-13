#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Breakout Strategy Helper Methods

This module contains helper methods for the Breakout Trading Strategy
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

# Configure logging
logger = logging.getLogger(__name__)


def identify_key_levels(self, data: pd.DataFrame) -> None:
    """
    Identify potential support and resistance levels based on historical price action.
    
    Args:
        data: Market data DataFrame with OHLCV columns
    """
    try:
        # Use only the lookback period for analysis
        lookback = self.parameters['lookback_period']
        if len(data) < lookback:
            return
            
        analysis_data = data.tail(lookback)
        
        # Find local minima (support) and maxima (resistance)
        local_min = []
        local_max = []
        
        for i in range(1, len(analysis_data)-1):
            # Local minimum (support)
            if (analysis_data['low'].iloc[i] < analysis_data['low'].iloc[i-1] and 
                analysis_data['low'].iloc[i] < analysis_data['low'].iloc[i+1]):
                local_min.append((i, analysis_data['low'].iloc[i]))
            
            # Local maximum (resistance)
            if (analysis_data['high'].iloc[i] > analysis_data['high'].iloc[i-1] and 
                analysis_data['high'].iloc[i] > analysis_data['high'].iloc[i+1]):
                local_max.append((i, analysis_data['high'].iloc[i]))
        
        # Group similar price levels (within tolerance)
        tolerance = self.parameters['level_tolerance']
        support_clusters = cluster_price_levels(self, [price for _, price in local_min], tolerance)
        resistance_clusters = cluster_price_levels(self, [price for _, price in local_max], tolerance)
        
        # Get levels with at least the minimum number of touches
        min_touches = self.parameters['min_touches']
        self.support_levels = [level for level, touches in support_clusters if touches >= min_touches]
        self.resistance_levels = [level for level, touches in resistance_clusters if touches >= min_touches]
        
        logger.debug(f"Identified {len(self.support_levels)} support and {len(self.resistance_levels)} resistance levels")
        
    except Exception as e:
        logger.error(f"Error identifying key levels: {str(e)}")


def cluster_price_levels(self, price_levels: List[float], tolerance: float = 0.005) -> List[Tuple[float, int]]:
    """
    Cluster similar price levels together and count touches.
    
    Args:
        price_levels: List of price levels to cluster
        tolerance: Percentage tolerance for clustering
        
    Returns:
        List of tuples with (price_level, touch_count) sorted by touch count
    """
    if not price_levels:
        return []
        
    # Sort price levels
    sorted_levels = sorted(price_levels)
    
    # Cluster similar levels
    clusters = []
    current_cluster = [sorted_levels[0]]
    
    for i in range(1, len(sorted_levels)):
        current_price = sorted_levels[i]
        prev_price = current_cluster[-1]
        
        # If within tolerance, add to current cluster
        if abs(current_price - prev_price) / prev_price <= tolerance:
            current_cluster.append(current_price)
        else:
            # New cluster
            avg_price = sum(current_cluster) / len(current_cluster)
            clusters.append((avg_price, len(current_cluster)))
            current_cluster = [current_price]
    
    # Add the last cluster
    if current_cluster:
        avg_price = sum(current_cluster) / len(current_cluster)
        clusters.append((avg_price, len(current_cluster)))
    
    # Sort clusters by touch count (descending)
    return sorted(clusters, key=lambda x: x[1], reverse=True)


def identify_consolidation_patterns(self, data: pd.DataFrame) -> None:
    """
    Identify potential consolidation patterns (price channels, rectangles).
    
    Args:
        data: Market data DataFrame with OHLCV columns
    """
    try:
        # Reset consolidation patterns
        self.consolidation_patterns = []
        
        # Check for minimum data required
        min_bars = self.parameters['min_consolidation_bars']
        if len(data) < min_bars * 2:  # Need enough data to find patterns
            return
        
        # Threshold for consolidation (max range percent)
        threshold = self.parameters['consolidation_threshold']
        
        # Analyze potential consolidation patterns
        for start in range(len(data) - min_bars * 2, len(data) - min_bars + 1):
            # Look at windows of specified minimum size
            window = data.iloc[start:start + min_bars]
            
            # Calculate high and low of the window
            window_high = window['high'].max()
            window_low = window['low'].min()
            window_avg = (window_high + window_low) / 2
            
            # Calculate range as percentage of average price
            window_range_pct = (window_high - window_low) / window_avg
            
            # If range is tight enough, consider it consolidation
            if window_range_pct <= threshold:
                touches_high = sum(1 for i in range(len(window)) if 
                                 window['high'].iloc[i] >= window_high * 0.995)
                touches_low = sum(1 for i in range(len(window)) if 
                                window['low'].iloc[i] <= window_low * 1.005)
                
                # Must have at least 2 touches on each boundary
                if touches_high >= 2 and touches_low >= 2:
                    self.consolidation_patterns.append({
                        'start_idx': start,
                        'end_idx': start + min_bars - 1,
                        'upper': window_high,
                        'lower': window_low,
                        'touches_high': touches_high,
                        'touches_low': touches_low,
                        'range_pct': window_range_pct
                    })
        
        # Sort by recency (most recent first)
        self.consolidation_patterns = sorted(
            self.consolidation_patterns, 
            key=lambda x: x['end_idx'],
            reverse=True
        )
        
        logger.debug(f"Identified {len(self.consolidation_patterns)} consolidation patterns")
        
    except Exception as e:
        logger.error(f"Error identifying consolidation patterns: {str(e)}")


def detect_breakouts(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> None:
    """
    Detect potential price breakouts from key levels or consolidation patterns.
    
    Args:
        data: Market data DataFrame with OHLCV columns
        indicators: Pre-calculated indicators
    """
    try:
        if data.empty or len(data) < 3:
            return
            
        # Get current and recent prices
        current_price = data['close'].iloc[-1]
        current_high = data['high'].iloc[-1]
        current_low = data['low'].iloc[-1]
        prev_close = data['close'].iloc[-2]
        
        # Check volume confirmation if available
        volume_confirmed = False
        if 'volume_ratio' in indicators:
            volume_ratio = indicators['volume_ratio'].iloc[-1]
            volume_confirmed = volume_ratio >= self.parameters['volume_multiplier']
        
        # Check MACD confirmation if enabled
        macd_confirmed = False
        if self.parameters['macd_confirmation'] and 'macd_histogram' in indicators:
            macd_hist = indicators['macd_histogram'].iloc[-1]
            macd_hist_prev = indicators['macd_histogram'].iloc[-2]
            macd_confirmed = macd_hist > 0 and macd_hist > macd_hist_prev  # Increasing momentum
        
        # Reset current breakout if price moved back significantly
        if self.current_breakout['level'] is not None:
            breakout_level = self.current_breakout['level']
            reversal_threshold = 0.01  # 1% threshold for reversal
            
            if ((self.current_breakout['direction'] == 'up' and 
                current_price < breakout_level * (1 - reversal_threshold)) or
               (self.current_breakout['direction'] == 'down' and 
                current_price > breakout_level * (1 + reversal_threshold))):
                # Reset breakout - appears to be a false breakout
                self.current_breakout = {
                    'type': None,
                    'level': None,
                    'direction': None,
                    'confirmed': False,
                    'detected_at': None,
                    'strength': 0.0
                }
        
        # If we don't have a current active breakout, look for new ones
        if self.current_breakout['level'] is None:
            # Check for resistance breakouts (price breaking above resistance)
            for level in self.resistance_levels:
                # Breakout occurs when price closes above resistance
                if prev_close <= level and current_price > level * 1.01:  # 1% above resistance
                    strength = min(1.0, (current_price - level) / level * 10)  # Scale based on breakout size
                    
                    # Apply confirmation factors
                    if volume_confirmed:
                        strength *= 1.2  # Increase strength with volume confirmation
                    if macd_confirmed:
                        strength *= 1.2  # Increase strength with MACD confirmation
                        
                    self.current_breakout = {
                        'type': 'resistance',
                        'level': level,
                        'direction': 'up',
                        'confirmed': False,  # Needs confirmation bars
                        'detected_at': len(data) - 1,
                        'strength': min(1.0, strength)  # Cap at 1.0
                    }
                    logger.info(f"Detected potential resistance breakout at {current_price:.2f}, level: {level:.2f}")
                    break
            
            # If no resistance breakout found, check for support breakouts (price breaking below support)
            if self.current_breakout['level'] is None:
                for level in self.support_levels:
                    # Breakout occurs when price closes below support
                    if prev_close >= level and current_price < level * 0.99:  # 1% below support
                        strength = min(1.0, (level - current_price) / level * 10)  # Scale based on breakout size
                        
                        # Apply confirmation factors
                        if volume_confirmed:
                            strength *= 1.2  # Increase strength with volume confirmation
                        if macd_confirmed:
                            strength *= 1.2  # Increase strength with MACD confirmation
                            
                        self.current_breakout = {
                            'type': 'support',
                            'level': level,
                            'direction': 'down',
                            'confirmed': False,  # Needs confirmation bars
                            'detected_at': len(data) - 1,
                            'strength': min(1.0, strength)  # Cap at 1.0
                        }
                        logger.info(f"Detected potential support breakout at {current_price:.2f}, level: {level:.2f}")
                        break
            
            # If no level breakouts found, check for consolidation pattern breakouts
            if self.current_breakout['level'] is None and self.consolidation_patterns:
                # Check most recent consolidation pattern
                pattern = self.consolidation_patterns[0]
                upper_level = pattern['upper']
                lower_level = pattern['lower']
                
                # Breakout above consolidation
                if prev_close <= upper_level and current_price > upper_level * 1.01:  # 1% above upper bound
                    strength = min(1.0, (current_price - upper_level) / upper_level * 10)
                    
                    if volume_confirmed:
                        strength *= 1.2
                    if macd_confirmed:
                        strength *= 1.2
                        
                    self.current_breakout = {
                        'type': 'consolidation',
                        'level': upper_level,
                        'direction': 'up',
                        'confirmed': False,
                        'detected_at': len(data) - 1,
                        'strength': min(1.0, strength)
                    }
                    logger.info(f"Detected potential consolidation breakout (upward) at {current_price:.2f}, level: {upper_level:.2f}")
                
                # Breakout below consolidation
                elif prev_close >= lower_level and current_price < lower_level * 0.99:  # 1% below lower bound
                    strength = min(1.0, (lower_level - current_price) / lower_level * 10)
                    
                    if volume_confirmed:
                        strength *= 1.2
                    if macd_confirmed:
                        strength *= 1.2
                        
                    self.current_breakout = {
                        'type': 'consolidation',
                        'level': lower_level,
                        'direction': 'down',
                        'confirmed': False,
                        'detected_at': len(data) - 1,
                        'strength': min(1.0, strength)
                    }
                    logger.info(f"Detected potential consolidation breakout (downward) at {current_price:.2f}, level: {lower_level:.2f}")
        
        # Check if breakout is confirmed (price continues in breakout direction for specified bars)
        elif not self.current_breakout['confirmed']:
            confirmation_bars = self.parameters['breakout_confirmation_bars']
            detected_idx = self.current_breakout['detected_at']
            current_idx = len(data) - 1
            
            # Need enough bars since detection for confirmation
            if current_idx - detected_idx >= confirmation_bars:
                # Check if price maintained breakout direction
                breakout_level = self.current_breakout['level']
                direction = self.current_breakout['direction']
                
                if direction == 'up' and current_price > breakout_level:
                    # Confirmed upward breakout
                    self.current_breakout['confirmed'] = True
                    logger.info(f"Confirmed {self.current_breakout['type']} breakout at {breakout_level:.2f}")
                elif direction == 'down' and current_price < breakout_level:
                    # Confirmed downward breakout
                    self.current_breakout['confirmed'] = True
                    logger.info(f"Confirmed {self.current_breakout['type']} breakout at {breakout_level:.2f}")
                else:
                    # Failed breakout, reset
                    self.current_breakout = {
                        'type': None,
                        'level': None,
                        'direction': None,
                        'confirmed': False,
                        'detected_at': None,
                        'strength': 0.0
                    }
                    logger.info("Breakout failed confirmation, resetting")
            
    except Exception as e:
        logger.error(f"Error detecting breakouts: {str(e)}")
