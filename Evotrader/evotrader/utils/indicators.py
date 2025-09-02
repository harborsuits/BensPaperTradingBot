"""Technical indicators for trading strategies."""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import math


def sma(data: List[float], period: int) -> List[float]:
    """
    Calculate Simple Moving Average.
    
    Args:
        data: List of price values
        period: Period for the moving average
        
    Returns:
        List of SMA values (with None for the first period-1 entries)
    """
    if len(data) < period:
        return [None] * len(data)
        
    result = [None] * (period - 1)
    
    for i in range(period - 1, len(data)):
        result.append(sum(data[i - period + 1:i + 1]) / period)
        
    return result


def ema(data: List[float], period: int, smoothing: float = 2.0) -> List[float]:
    """
    Calculate Exponential Moving Average.
    
    Args:
        data: List of price values
        period: Period for the moving average
        smoothing: Smoothing factor (default: 2.0)
        
    Returns:
        List of EMA values (with None for the first period-1 entries)
    """
    if len(data) < period:
        return [None] * len(data)
        
    result = [None] * (period - 1)
    
    # First EMA value is SMA
    result.append(sum(data[:period]) / period)
    
    # Calculate alpha (smoothing factor)
    alpha = smoothing / (period + 1.0)
    
    # Calculate EMA values
    for i in range(period, len(data)):
        ema_value = data[i] * alpha + result[-1] * (1 - alpha)
        result.append(ema_value)
        
    return result


def rsi(data: List[float], period: int = 14) -> List[float]:
    """
    Calculate Relative Strength Index.
    
    Args:
        data: List of price values
        period: Period for RSI calculation
        
    Returns:
        List of RSI values (with None for the first period entries)
    """
    if len(data) <= period:
        return [None] * len(data)
        
    # Calculate price changes
    changes = [data[i+1] - data[i] for i in range(len(data)-1)]
    
    # Create result array with leading None values
    result = [None] * (period + 1)
    
    # Calculate first average gain and loss
    gains = [max(0, change) for change in changes[:period]]
    losses = [max(0, -change) for change in changes[:period]]
    
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    
    # Add first RSI value
    if avg_loss == 0:
        result.append(100.0)
    else:
        rs = avg_gain / avg_loss
        result.append(100.0 - (100.0 / (1.0 + rs)))
    
    # Calculate remaining RSI values
    for i in range(period, len(changes)):
        # Update average gain and loss using smoothing
        current_gain = max(0, changes[i])
        current_loss = max(0, -changes[i])
        
        avg_gain = (avg_gain * (period - 1) + current_gain) / period
        avg_loss = (avg_loss * (period - 1) + current_loss) / period
        
        # Calculate RSI
        if avg_loss == 0:
            result.append(100.0)
        else:
            rs = avg_gain / avg_loss
            result.append(100.0 - (100.0 / (1.0 + rs)))
            
    return result


def macd(data: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Args:
        data: List of price values
        fast_period: Period for fast EMA
        slow_period: Period for slow EMA
        signal_period: Period for signal line
        
    Returns:
        Tuple of (MACD line, signal line, histogram)
    """
    # Calculate EMAs
    fast_ema = ema(data, fast_period)
    slow_ema = ema(data, slow_period)
    
    # Calculate MACD line
    macd_line = [None] * max(len(fast_ema), len(slow_ema))
    for i in range(len(macd_line)):
        if i < len(fast_ema) and i < len(slow_ema) and fast_ema[i] is not None and slow_ema[i] is not None:
            macd_line[i] = fast_ema[i] - slow_ema[i]
            
    # Calculate signal line (EMA of MACD line)
    # Filter out None values for signal line calculation
    valid_macd = [x for x in macd_line if x is not None]
    if len(valid_macd) < signal_period:
        signal_line = [None] * len(macd_line)
    else:
        # Pad signal line with None values to match MACD line length
        pad_length = len(macd_line) - len(valid_macd)
        signal_line = [None] * pad_length + ema(valid_macd, signal_period)
        
    # Calculate histogram (MACD line - signal line)
    histogram = [None] * len(macd_line)
    for i in range(len(histogram)):
        if macd_line[i] is not None and signal_line[i] is not None:
            histogram[i] = macd_line[i] - signal_line[i]
            
    return macd_line, signal_line, histogram


def bollinger_bands(data: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate Bollinger Bands.
    
    Args:
        data: List of price values
        period: Period for the moving average
        std_dev: Number of standard deviations for the bands
        
    Returns:
        Tuple of (middle band, upper band, lower band)
    """
    # Calculate middle band (SMA)
    middle_band = sma(data, period)
    
    # Calculate standard deviation
    upper_band = [None] * len(middle_band)
    lower_band = [None] * len(middle_band)
    
    for i in range(period - 1, len(data)):
        if middle_band[i] is not None:
            # Calculate standard deviation for the period
            period_values = data[i - period + 1:i + 1]
            std = math.sqrt(sum((x - middle_band[i]) ** 2 for x in period_values) / period)
            
            # Calculate upper and lower bands
            upper_band[i] = middle_band[i] + (std_dev * std)
            lower_band[i] = middle_band[i] - (std_dev * std)
            
    return middle_band, upper_band, lower_band


def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
    """
    Calculate Average True Range.
    
    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        period: Period for ATR calculation
        
    Returns:
        List of ATR values
    """
    if len(highs) != len(lows) or len(highs) != len(closes):
        raise ValueError("Input arrays must have the same length")
        
    if len(highs) < period + 1:
        return [None] * len(highs)
        
    # Calculate True Range
    tr_values = [None]  # First TR requires previous close, so it's None
    
    for i in range(1, len(highs)):
        prev_close = closes[i-1]
        true_range = max(
            highs[i] - lows[i],  # Current high - current low
            abs(highs[i] - prev_close),  # Current high - previous close
            abs(lows[i] - prev_close)  # Current low - previous close
        )
        tr_values.append(true_range)
        
    # Calculate ATR
    result = [None] * period
    
    # First ATR is simple average of TR values
    result.append(sum(tr_values[1:period+1]) / period)
    
    # Calculate remaining ATR values using smoothing
    for i in range(period + 1, len(tr_values)):
        atr_value = (result[-1] * (period - 1) + tr_values[i]) / period
        result.append(atr_value)
        
    return result


def stochastic(highs: List[float], lows: List[float], closes: List[float], 
                k_period: int = 14, d_period: int = 3) -> Tuple[List[float], List[float]]:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        k_period: Period for %K
        d_period: Period for %D (SMA of %K)
        
    Returns:
        Tuple of (%K, %D)
    """
    if len(highs) != len(lows) or len(highs) != len(closes):
        raise ValueError("Input arrays must have the same length")
        
    if len(highs) < k_period:
        return [None] * len(highs), [None] * len(highs)
        
    # Calculate %K
    k_values = [None] * (k_period - 1)
    
    for i in range(k_period - 1, len(closes)):
        period_highs = highs[i - k_period + 1:i + 1]
        period_lows = lows[i - k_period + 1:i + 1]
        highest_high = max(period_highs)
        lowest_low = min(period_lows)
        
        if highest_high == lowest_low:
            # Avoid division by zero
            k_values.append(50.0)
        else:
            # Calculate %K
            k_value = 100.0 * (closes[i] - lowest_low) / (highest_high - lowest_low)
            k_values.append(k_value)
            
    # Calculate %D (SMA of %K)
    d_values = sma(k_values, d_period)
    
    return k_values, d_values
