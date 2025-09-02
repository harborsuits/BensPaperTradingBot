"""
Robust technical indicator calculations with proper None-value handling.
This module provides safe wrappers around the technical indicators to ensure
they properly handle None values and don't cause errors during calculation.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from ..utils.indicators import (
    sma as orig_sma,
    ema as orig_ema,
    rsi as orig_rsi,
    macd as orig_macd,
    bollinger_bands as orig_bb,
    atr as orig_atr,
    stochastic as orig_stochastic
)

logger = logging.getLogger(__name__)

def safe_sma(data: List[float], period: int) -> List[float]:
    """Safely calculate Simple Moving Average with None-value protection."""
    try:
        # Filter out None values
        clean_data = [p for p in data if p is not None]
        if len(clean_data) < period:
            return [None] * len(data)
        
        # Calculate SMA on clean data
        result = orig_sma(clean_data, period)
        
        # Pad with None values to match original length
        padding = [None] * (len(data) - len(result))
        return padding + result
    except Exception as e:
        logger.warning(f"Error calculating SMA: {str(e)}")
        return [None] * len(data)

def safe_ema(data: List[float], period: int, smoothing: float = 2.0) -> List[float]:
    """Safely calculate Exponential Moving Average with None-value protection."""
    try:
        # Filter out None values
        clean_data = [p for p in data if p is not None]
        if len(clean_data) < period:
            return [None] * len(data)
        
        # Calculate EMA on clean data
        result = orig_ema(clean_data, period, smoothing)
        
        # Pad with None values to match original length
        padding = [None] * (len(data) - len(result))
        return padding + result
    except Exception as e:
        logger.warning(f"Error calculating EMA: {str(e)}")
        return [None] * len(data)

def safe_rsi(data: List[float], period: int = 14) -> List[float]:
    """Safely calculate RSI with None-value protection."""
    try:
        # Filter out None values
        clean_data = [p for p in data if p is not None]
        if len(clean_data) < period + 1:  # Need at least period+1 for price changes
            return [None] * len(data)
        
        # Calculate RSI on clean data
        result = orig_rsi(clean_data, period)
        
        # Pad with None values to match original length
        padding = [None] * (len(data) - len(result))
        return padding + result
    except Exception as e:
        logger.warning(f"Error calculating RSI: {str(e)}")
        return [None] * len(data)

def safe_macd(
    data: List[float], 
    fast_period: int = 12, 
    slow_period: int = 26, 
    signal_period: int = 9
) -> Tuple[List[float], List[float], List[float]]:
    """Safely calculate MACD with None-value protection."""
    try:
        # Filter out None values
        clean_data = [p for p in data if p is not None]
        if len(clean_data) < slow_period + signal_period:
            empty = [None] * len(data)
            return empty, empty, empty
        
        # Calculate MACD on clean data
        macd_line, signal, hist = orig_macd(
            clean_data, fast_period, slow_period, signal_period
        )
        
        # Pad with None values to match original length
        padding_len = len(data) - len(macd_line)
        padding = [None] * padding_len
        
        return (
            padding + macd_line,
            padding + signal,
            padding + hist
        )
    except Exception as e:
        logger.warning(f"Error calculating MACD: {str(e)}")
        empty = [None] * len(data)
        return empty, empty, empty

def safe_bollinger_bands(
    data: List[float], 
    period: int = 20, 
    std_dev: float = 2.0
) -> Tuple[List[float], List[float], List[float]]:
    """Safely calculate Bollinger Bands with None-value protection."""
    try:
        # Filter out None values
        clean_data = [p for p in data if p is not None]
        if len(clean_data) < period:
            empty = [None] * len(data)
            return empty, empty, empty
        
        # Calculate BB on clean data
        middle, upper, lower = orig_bb(clean_data, period, std_dev)
        
        # Pad with None values to match original length
        padding_len = len(data) - len(middle)
        padding = [None] * padding_len
        
        return (
            padding + middle,
            padding + upper,
            padding + lower
        )
    except Exception as e:
        logger.warning(f"Error calculating Bollinger Bands: {str(e)}")
        empty = [None] * len(data)
        return empty, empty, empty

def safe_atr(
    highs: List[float], 
    lows: List[float], 
    closes: List[float], 
    period: int = 14
) -> List[float]:
    """Safely calculate ATR with None-value protection."""
    try:
        # Ensure all lists have same length
        if not (len(highs) == len(lows) == len(closes)):
            logger.warning("ATR input lists have different lengths")
            return [None] * len(closes)
        
        # Filter out None values (keeping aligned indices)
        clean_highs, clean_lows, clean_closes = [], [], []
        for h, l, c in zip(highs, lows, closes):
            if h is not None and l is not None and c is not None:
                clean_highs.append(h)
                clean_lows.append(l)
                clean_closes.append(c)
        
        if len(clean_closes) < period + 1:  # Need prev close for TR
            return [None] * len(closes)
        
        # Calculate ATR on clean data
        result = orig_atr(clean_highs, clean_lows, clean_closes, period)
        
        # Pad with None values to match original length
        padding = [None] * (len(closes) - len(result))
        return padding + result
    except Exception as e:
        logger.warning(f"Error calculating ATR: {str(e)}")
        return [None] * len(closes)

def safe_stochastic(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[List[float], List[float]]:
    """Safely calculate Stochastic Oscillator with None-value protection."""
    try:
        # Ensure all lists have same length
        if not (len(highs) == len(lows) == len(closes)):
            logger.warning("Stochastic input lists have different lengths")
            return [None] * len(closes), [None] * len(closes)
        
        # Filter out None values (keeping aligned indices)
        clean_highs, clean_lows, clean_closes = [], [], []
        for h, l, c in zip(highs, lows, closes):
            if h is not None and l is not None and c is not None:
                clean_highs.append(h)
                clean_lows.append(l)
                clean_closes.append(c)
        
        if len(clean_closes) < k_period:
            return [None] * len(closes), [None] * len(closes)
        
        # Calculate Stochastic on clean data
        k_values, d_values = orig_stochastic(
            clean_highs, clean_lows, clean_closes, k_period, d_period
        )
        
        # Pad with None values to match original length
        padding_len = len(closes) - len(k_values)
        padding = [None] * padding_len
        
        return padding + k_values, padding + d_values
    except Exception as e:
        logger.warning(f"Error calculating Stochastic: {str(e)}")
        return [None] * len(closes), [None] * len(closes)


# Helper functions for indicator calculations

def is_bullish_crossover(current_fast: float, current_slow: float, 
                         prev_fast: float, prev_slow: float) -> bool:
    """Detect a bullish crossover (fast line crosses above slow line)"""
    if None in (current_fast, current_slow, prev_fast, prev_slow):
        return False
    return current_fast > current_slow and prev_fast <= prev_slow

def is_bearish_crossover(current_fast: float, current_slow: float, 
                          prev_fast: float, prev_slow: float) -> bool:
    """Detect a bearish crossover (fast line crosses below slow line)"""
    if None in (current_fast, current_slow, prev_fast, prev_slow):
        return False
    return current_fast < current_slow and prev_fast >= prev_slow
