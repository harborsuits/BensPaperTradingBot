#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Technical Trading Strategies

This module implements various technical analysis based trading strategies
that can be used with the Alpha Vantage backtester.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime

def sma_crossover_strategy(date: datetime, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Simple Moving Average crossover strategy.
    
    Generates buy signals when short SMA crosses above long SMA,
    and sell signals when short SMA crosses below long SMA.
    
    Args:
        date: Current backtest date
        data: Dictionary with price and indicator data for each symbol
        params: Strategy parameters
        
    Returns:
        Dictionary of signals for each symbol
    """
    # Default parameters
    short_period = params.get("short_period", 20)
    long_period = params.get("long_period", 50)
    lookback = params.get("lookback", 1)  # Days to look back for crossover
    
    signals = {}
    
    for symbol, symbol_data in data.items():
        try:
            # Get indicators
            indicators = symbol_data["indicators"]
            
            # Check if required indicators exist
            short_sma_key = f"SMA_{short_period}"
            long_sma_key = f"SMA_{long_period}"
            
            if short_sma_key not in indicators or long_sma_key not in indicators:
                continue
            
            short_sma = indicators[short_sma_key]
            long_sma = indicators[long_sma_key]
            
            # Need at least lookback+1 days of data
            if len(short_sma) <= lookback or len(long_sma) <= lookback:
                continue
            
            # Check for crossover
            current_diff = short_sma.iloc[-1] - long_sma.iloc[-1]
            previous_diff = short_sma.iloc[-lookback-1] - long_sma.iloc[-lookback-1]
            
            # Initialize with neutral signal
            signal = "neutral"
            
            # Bullish crossover (short SMA crosses above long SMA)
            if previous_diff <= 0 and current_diff > 0:
                signal = "buy"
            
            # Bearish crossover (short SMA crosses below long SMA)
            elif previous_diff >= 0 and current_diff < 0:
                signal = "sell"
            
            signals[symbol] = {
                "signal": signal,
                "strength": abs(current_diff),
                "short_sma": short_sma.iloc[-1],
                "long_sma": long_sma.iloc[-1]
            }
            
        except Exception as e:
            print(f"Error in SMA crossover strategy for {symbol}: {str(e)}")
    
    return signals

def macd_strategy(date: datetime, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    MACD-based trading strategy.
    
    Generates buy signals when MACD crosses above signal line,
    and sell signals when MACD crosses below signal line.
    Strong signals are generated when both the MACD cross and histogram direction align.
    
    Args:
        date: Current backtest date
        data: Dictionary with price and indicator data for each symbol
        params: Strategy parameters
        
    Returns:
        Dictionary of signals for each symbol
    """
    # Default parameters
    lookback = params.get("lookback", 1)  # Days to look back for crossover
    
    signals = {}
    
    for symbol, symbol_data in data.items():
        try:
            # Get indicators
            indicators = symbol_data["indicators"]
            
            # Check if required indicators exist
            if "MACD" not in indicators or "MACD_Signal" not in indicators or "MACD_Hist" not in indicators:
                continue
            
            macd = indicators["MACD"]
            signal_line = indicators["MACD_Signal"]
            histogram = indicators["MACD_Hist"]
            
            # Need at least lookback+1 days of data
            if len(macd) <= lookback or len(signal_line) <= lookback or len(histogram) <= lookback:
                continue
            
            # Check for crossover
            current_diff = macd.iloc[-1] - signal_line.iloc[-1]
            previous_diff = macd.iloc[-lookback-1] - signal_line.iloc[-lookback-1]
            
            current_hist = histogram.iloc[-1]
            previous_hist = histogram.iloc[-lookback-1]
            
            # Initialize with neutral signal
            signal = "neutral"
            
            # Bullish crossover (MACD crosses above signal line)
            if previous_diff <= 0 and current_diff > 0:
                # Strong buy if histogram is also increasing
                if current_hist > previous_hist:
                    signal = "strong_buy"
                else:
                    signal = "buy"
            
            # Bearish crossover (MACD crosses below signal line)
            elif previous_diff >= 0 and current_diff < 0:
                # Strong sell if histogram is also decreasing
                if current_hist < previous_hist:
                    signal = "strong_sell"
                else:
                    signal = "sell"
            
            signals[symbol] = {
                "signal": signal,
                "strength": abs(current_diff),
                "macd": macd.iloc[-1],
                "signal_line": signal_line.iloc[-1],
                "histogram": current_hist
            }
            
        except Exception as e:
            print(f"Error in MACD strategy for {symbol}: {str(e)}")
    
    return signals

def rsi_strategy(date: datetime, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Relative Strength Index (RSI) trading strategy.
    
    Generates buy signals when RSI crosses above oversold threshold,
    and sell signals when RSI crosses below overbought threshold.
    
    Args:
        date: Current backtest date
        data: Dictionary with price and indicator data for each symbol
        params: Strategy parameters
        
    Returns:
        Dictionary of signals for each symbol
    """
    # Default parameters
    oversold = params.get("oversold", 30)
    overbought = params.get("overbought", 70)
    lookback = params.get("lookback", 1)  # Days to look back for crossover
    
    signals = {}
    
    for symbol, symbol_data in data.items():
        try:
            # Get indicators
            indicators = symbol_data["indicators"]
            
            # Check if required indicators exist
            if "RSI" not in indicators:
                continue
            
            rsi = indicators["RSI"]
            
            # Need at least lookback+1 days of data
            if len(rsi) <= lookback:
                continue
            
            current_rsi = rsi.iloc[-1]
            previous_rsi = rsi.iloc[-lookback-1]
            
            # Initialize with neutral signal
            signal = "neutral"
            
            # Bullish signal (RSI crosses above oversold threshold)
            if previous_rsi <= oversold and current_rsi > oversold:
                # Strong buy if deeply oversold
                if previous_rsi < oversold - 10:
                    signal = "strong_buy"
                else:
                    signal = "buy"
            
            # Bearish signal (RSI crosses below overbought threshold)
            elif previous_rsi >= overbought and current_rsi < overbought:
                # Strong sell if deeply overbought
                if previous_rsi > overbought + 10:
                    signal = "strong_sell"
                else:
                    signal = "sell"
            
            signals[symbol] = {
                "signal": signal,
                "strength": abs(current_rsi - 50),  # Distance from neutral (50)
                "rsi": current_rsi
            }
            
        except Exception as e:
            print(f"Error in RSI strategy for {symbol}: {str(e)}")
    
    return signals

def bollinger_band_strategy(date: datetime, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Bollinger Bands trading strategy.
    
    Generates buy signals when price touches or crosses below lower band,
    and sell signals when price touches or crosses above upper band.
    
    Args:
        date: Current backtest date
        data: Dictionary with price and indicator data for each symbol
        params: Strategy parameters
        
    Returns:
        Dictionary of signals for each symbol
    """
    # Default parameters
    lookback = params.get("lookback", 1)  # Days to look back for crossover
    
    signals = {}
    
    for symbol, symbol_data in data.items():
        try:
            # Get indicators and price data
            indicators = symbol_data["indicators"]
            prices = symbol_data["prices"]
            
            # Check if required indicators exist
            if "BB_Upper" not in indicators or "BB_Lower" not in indicators or "BB_Middle" not in indicators:
                continue
            
            # Get price and bands
            close_prices = prices["Close"]
            upper_band = indicators["BB_Upper"]
            lower_band = indicators["BB_Lower"]
            middle_band = indicators["BB_Middle"]
            
            # Need at least lookback+1 days of data
            if len(close_prices) <= lookback or len(upper_band) <= lookback or len(lower_band) <= lookback:
                continue
            
            current_price = close_prices.iloc[-1]
            previous_price = close_prices.iloc[-lookback-1]
            
            current_upper = upper_band.iloc[-1]
            previous_upper = upper_band.iloc[-lookback-1]
            
            current_lower = lower_band.iloc[-1]
            previous_lower = lower_band.iloc[-lookback-1]
            
            current_middle = middle_band.iloc[-1]
            
            # Initialize with neutral signal
            signal = "neutral"
            
            # Bullish signal (price crosses above lower band from below)
            if previous_price <= previous_lower and current_price > current_lower:
                # Strong buy if price was deeply below lower band
                if previous_price < previous_lower * 0.98:
                    signal = "strong_buy"
                else:
                    signal = "buy"
            
            # Bearish signal (price crosses below upper band from above)
            elif previous_price >= previous_upper and current_price < current_upper:
                # Strong sell if price was deeply above upper band
                if previous_price > previous_upper * 1.02:
                    signal = "strong_sell"
                else:
                    signal = "sell"
            
            # Calculate percent bandwidth (price position within bands)
            bandwidth = (current_upper - current_lower) / current_middle
            percent_b = (current_price - current_lower) / (current_upper - current_lower)
            
            signals[symbol] = {
                "signal": signal,
                "strength": abs(percent_b - 0.5) * 2,  # Normalized distance from middle
                "percent_b": percent_b,
                "bandwidth": bandwidth,
                "price": current_price,
                "upper_band": current_upper,
                "lower_band": current_lower
            }
            
        except Exception as e:
            print(f"Error in Bollinger Band strategy for {symbol}: {str(e)}")
    
    return signals

def multi_indicator_strategy(date: datetime, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Multi-indicator strategy combining several technical indicators.
    
    Combines signals from RSI, MACD, and Bollinger Bands to generate
    composite trading signals.
    
    Args:
        date: Current backtest date
        data: Dictionary with price and indicator data for each symbol
        params: Strategy parameters
        
    Returns:
        Dictionary of signals for each symbol
    """
    # Default parameters
    rsi_oversold = params.get("rsi_oversold", 30)
    rsi_overbought = params.get("rsi_overbought", 70)
    
    signals = {}
    
    for symbol, symbol_data in data.items():
        try:
            # Get indicators and price data
            indicators = symbol_data["indicators"]
            prices = symbol_data["prices"]
            
            # Skip if missing indicators
            required_indicators = ["RSI", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower"]
            if not all(ind in indicators for ind in required_indicators):
                continue
            
            close_prices = prices["Close"]
            
            # Get indicator values
            current_rsi = indicators["RSI"].iloc[-1]
            current_price = close_prices.iloc[-1]
            current_macd = indicators["MACD"].iloc[-1]
            current_signal = indicators["MACD_Signal"].iloc[-1]
            current_upper = indicators["BB_Upper"].iloc[-1]
            current_lower = indicators["BB_Lower"].iloc[-1]
            
            # Score-based approach (from -3 to +3)
            score = 0
            
            # RSI component
            if current_rsi < rsi_oversold:
                # More oversold = stronger buy signal
                score += 1 + max(0, (rsi_oversold - current_rsi) / 10)
            elif current_rsi > rsi_overbought:
                # More overbought = stronger sell signal
                score -= 1 + max(0, (current_rsi - rsi_overbought) / 10)
            
            # MACD component
            if current_macd > current_signal:
                # Larger difference = stronger buy signal
                score += min(1, (current_macd - current_signal) * 10)
            else:
                # Larger difference = stronger sell signal
                score -= min(1, (current_signal - current_macd) * 10)
            
            # Bollinger Bands component
            bb_width = current_upper - current_lower
            if bb_width > 0:
                bb_position = (current_price - current_lower) / bb_width
                
                if bb_position < 0.2:  # Near or below lower band
                    score += 1
                elif bb_position > 0.8:  # Near or above upper band
                    score -= 1
            
            # Determine signal based on score
            if score >= 2:
                signal = "strong_buy"
            elif score >= 0.5:
                signal = "buy"
            elif score <= -2:
                signal = "strong_sell"
            elif score <= -0.5:
                signal = "sell"
            else:
                signal = "neutral"
            
            signals[symbol] = {
                "signal": signal,
                "score": score,
                "rsi": current_rsi,
                "macd_diff": current_macd - current_signal,
                "bb_position": (current_price - current_lower) / (current_upper - current_lower) if current_upper > current_lower else 0.5
            }
            
        except Exception as e:
            print(f"Error in multi-indicator strategy for {symbol}: {str(e)}")
    
    return signals

def adx_trend_strategy(date: datetime, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Average Directional Index (ADX) trend-following strategy.
    
    Uses ADX to determine trend strength and +DI/-DI for direction.
    Only trades when ADX indicates a strong trend.
    
    Args:
        date: Current backtest date
        data: Dictionary with price and indicator data for each symbol
        params: Strategy parameters
        
    Returns:
        Dictionary of signals for each symbol
    """
    # Default parameters
    adx_threshold = params.get("adx_threshold", 25)  # ADX above this indicates strong trend
    
    signals = {}
    
    for symbol, symbol_data in data.items():
        try:
            # Get indicators
            indicators = symbol_data["indicators"]
            
            # Skip if missing indicators
            if "ADX" not in indicators or "+DI" not in indicators or "-DI" not in indicators:
                continue
            
            adx = indicators["ADX"].iloc[-1]
            plus_di = indicators["+DI"].iloc[-1]
            minus_di = indicators["-DI"].iloc[-1]
            
            # Initialize with neutral signal
            signal = "neutral"
            
            # Only generate signals in strong trends
            if adx > adx_threshold:
                # Bullish trend when +DI > -DI
                if plus_di > minus_di:
                    # Strong buy if ADX is very high and +DI substantially larger than -DI
                    if adx > adx_threshold * 1.5 and plus_di > minus_di * 1.5:
                        signal = "strong_buy"
                    else:
                        signal = "buy"
                
                # Bearish trend when -DI > +DI
                elif minus_di > plus_di:
                    # Strong sell if ADX is very high and -DI substantially larger than +DI
                    if adx > adx_threshold * 1.5 and minus_di > plus_di * 1.5:
                        signal = "strong_sell"
                    else:
                        signal = "sell"
            
            signals[symbol] = {
                "signal": signal,
                "adx": adx,
                "plus_di": plus_di,
                "minus_di": minus_di,
                "strength": adx / 100  # Normalize ADX
            }
            
        except Exception as e:
            print(f"Error in ADX trend strategy for {symbol}: {str(e)}")
    
    return signals 