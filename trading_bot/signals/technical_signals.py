#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Technical Signals Module

This module provides the TechnicalSignals class for analyzing
technical trading signals and indicators.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, date, timedelta

from trading_bot.market.market_data import MarketData

logger = logging.getLogger(__name__)

class TechnicalSignals:
    """
    Technical Signals class for generating and analyzing technical indicators and signals.
    
    This class provides methods to calculate various technical indicators and 
    generate trading signals based on technical analysis.
    """
    
    def __init__(self, market_data: MarketData):
        """
        Initialize the TechnicalSignals object.
        
        Args:
            market_data: MarketData instance for retrieving historical data
        """
        self.market_data = market_data
        
        logger.info("Initialized TechnicalSignals")
    
    def calculate_sma(self, symbol: str, period: int = 20) -> Optional[pd.Series]:
        """
        Calculate Simple Moving Average for a symbol.
        
        Args:
            symbol: Stock symbol
            period: SMA period in days
            
        Returns:
            Series with SMA values or None if data not available
        """
        try:
            # Get historical close data
            hist_data = self.market_data.get_historical_data(
                symbol, 
                days=period*2,  # Get more days to ensure we have enough data
                fields=["close"]
            )
            
            if hist_data is None or len(hist_data) < period:
                logger.warning(f"Insufficient data for {symbol} to calculate {period}-day SMA")
                return None
            
            # Calculate SMA
            sma = hist_data["close"].rolling(window=period).mean()
            
            return sma
            
        except Exception as e:
            logger.error(f"Error calculating SMA for {symbol}: {str(e)}")
            return None
    
    def calculate_ema(self, symbol: str, period: int = 20) -> Optional[pd.Series]:
        """
        Calculate Exponential Moving Average for a symbol.
        
        Args:
            symbol: Stock symbol
            period: EMA period in days
            
        Returns:
            Series with EMA values or None if data not available
        """
        try:
            # Get historical close data
            hist_data = self.market_data.get_historical_data(
                symbol, 
                days=period*3,  # Get more days to ensure we have enough data
                fields=["close"]
            )
            
            if hist_data is None or len(hist_data) < period:
                logger.warning(f"Insufficient data for {symbol} to calculate {period}-day EMA")
                return None
            
            # Calculate EMA
            ema = hist_data["close"].ewm(span=period, adjust=False).mean()
            
            return ema
            
        except Exception as e:
            logger.error(f"Error calculating EMA for {symbol}: {str(e)}")
            return None
    
    def calculate_rsi(self, symbol: str, period: int = 14) -> Optional[pd.Series]:
        """
        Calculate Relative Strength Index for a symbol.
        
        Args:
            symbol: Stock symbol
            period: RSI period in days
            
        Returns:
            Series with RSI values or None if data not available
        """
        try:
            # Get historical close data
            hist_data = self.market_data.get_historical_data(
                symbol, 
                days=period*3,  # Get more days to ensure we have enough data
                fields=["close"]
            )
            
            if hist_data is None or len(hist_data) < period+1:
                logger.warning(f"Insufficient data for {symbol} to calculate {period}-day RSI")
                return None
            
            # Calculate price changes
            delta = hist_data["close"].diff()
            
            # Split gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gain = gains.rolling(window=period).mean()
            avg_loss = losses.rolling(window=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI for {symbol}: {str(e)}")
            return None
    
    def calculate_macd(self, symbol: str, fast_period: int = 12, 
                      slow_period: int = 26, signal_period: int = 9) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
        """
        Calculate Moving Average Convergence Divergence for a symbol.
        
        Args:
            symbol: Stock symbol
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram) Series or None if data not available
        """
        try:
            # Get historical close data
            hist_data = self.market_data.get_historical_data(
                symbol, 
                days=slow_period*3,  # Get more days to ensure we have enough data
                fields=["close"]
            )
            
            if hist_data is None or len(hist_data) < slow_period:
                logger.warning(f"Insufficient data for {symbol} to calculate MACD")
                return None, None, None
            
            # Calculate fast and slow EMAs
            fast_ema = hist_data["close"].ewm(span=fast_period, adjust=False).mean()
            slow_ema = hist_data["close"].ewm(span=slow_period, adjust=False).mean()
            
            # Calculate MACD line
            macd_line = fast_ema - slow_ema
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram
            
        except Exception as e:
            logger.error(f"Error calculating MACD for {symbol}: {str(e)}")
            return None, None, None
    
    def calculate_bollinger_bands(self, symbol: str, period: int = 20, 
                                std_dev: float = 2.0) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
        """
        Calculate Bollinger Bands for a symbol.
        
        Args:
            symbol: Stock symbol
            period: SMA period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (Upper Band, Middle Band, Lower Band) Series or None if data not available
        """
        try:
            # Get historical close data
            hist_data = self.market_data.get_historical_data(
                symbol, 
                days=period*3,  # Get more days to ensure we have enough data
                fields=["close"]
            )
            
            if hist_data is None or len(hist_data) < period:
                logger.warning(f"Insufficient data for {symbol} to calculate Bollinger Bands")
                return None, None, None
            
            # Calculate middle band (SMA)
            middle_band = hist_data["close"].rolling(window=period).mean()
            
            # Calculate standard deviation
            std = hist_data["close"].rolling(window=period).std()
            
            # Calculate upper and lower bands
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)
            
            return upper_band, middle_band, lower_band
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands for {symbol}: {str(e)}")
            return None, None, None
    
    def is_sma_bullish(self, symbol: str, short_period: int = 50, 
                     long_period: int = 200) -> bool:
        """
        Check if SMA configuration is bullish (short above long).
        
        Args:
            symbol: Stock symbol
            short_period: Short SMA period
            long_period: Long SMA period
            
        Returns:
            True if bullish, False otherwise
        """
        # Calculate SMAs
        short_sma = self.calculate_sma(symbol, period=short_period)
        long_sma = self.calculate_sma(symbol, period=long_period)
        
        if short_sma is None or long_sma is None:
            return False
        
        # Check if short SMA is above long SMA
        return short_sma.iloc[-1] > long_sma.iloc[-1]
    
    def is_ema_bullish(self, symbol: str, short_period: int = 20, 
                     long_period: int = 50) -> bool:
        """
        Check if EMA configuration is bullish (short above long).
        
        Args:
            symbol: Stock symbol
            short_period: Short EMA period
            long_period: Long EMA period
            
        Returns:
            True if bullish, False otherwise
        """
        # Calculate EMAs
        short_ema = self.calculate_ema(symbol, period=short_period)
        long_ema = self.calculate_ema(symbol, period=long_period)
        
        if short_ema is None or long_ema is None:
            return False
        
        # Check if short EMA is above long EMA
        return short_ema.iloc[-1] > long_ema.iloc[-1]
    
    def is_uptrend(self, symbol: str, days: int = 20) -> bool:
        """
        Check if a symbol is in an uptrend over the specified period.
        
        Args:
            symbol: Stock symbol
            days: Number of days to check
            
        Returns:
            True if in uptrend, False otherwise
        """
        try:
            # Get historical close data
            hist_data = self.market_data.get_historical_data(
                symbol, 
                days=days*2,  # Get more days to ensure we have enough data
                fields=["close"]
            )
            
            if hist_data is None or len(hist_data) < days:
                logger.warning(f"Insufficient data for {symbol} to check uptrend")
                return False
            
            # Calculate linear regression
            y = hist_data["close"].values[-days:]
            x = np.arange(len(y))
            
            slope, _, _, _, _ = np.polyfit(x, y, 1, full=True)
            
            # Check if slope is positive
            return slope[0] > 0
            
        except Exception as e:
            logger.error(f"Error checking uptrend for {symbol}: {str(e)}")
            return False
    
    def is_overbought(self, symbol: str, rsi_threshold: float = 70.0) -> bool:
        """
        Check if a symbol is overbought based on RSI.
        
        Args:
            symbol: Stock symbol
            rsi_threshold: RSI threshold (default: 70)
            
        Returns:
            True if overbought, False otherwise
        """
        # Calculate RSI
        rsi = self.calculate_rsi(symbol)
        
        if rsi is None:
            return False
        
        # Check if RSI is above threshold
        return rsi.iloc[-1] > rsi_threshold
    
    def is_oversold(self, symbol: str, rsi_threshold: float = 30.0) -> bool:
        """
        Check if a symbol is oversold based on RSI.
        
        Args:
            symbol: Stock symbol
            rsi_threshold: RSI threshold (default: 30)
            
        Returns:
            True if oversold, False otherwise
        """
        # Calculate RSI
        rsi = self.calculate_rsi(symbol)
        
        if rsi is None:
            return False
        
        # Check if RSI is below threshold
        return rsi.iloc[-1] < rsi_threshold
    
    def is_macd_bullish(self, symbol: str) -> bool:
        """
        Check if MACD is bullish (histogram positive and increasing).
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if bullish, False otherwise
        """
        # Calculate MACD
        _, _, histogram = self.calculate_macd(symbol)
        
        if histogram is None:
            return False
        
        # Check if histogram is positive and increasing
        latest = histogram.iloc[-1]
        previous = histogram.iloc[-2] if len(histogram) > 1 else 0
        
        return latest > 0 and latest > previous
    
    def is_macd_bearish(self, symbol: str) -> bool:
        """
        Check if MACD is bearish (histogram negative and decreasing).
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if bearish, False otherwise
        """
        # Calculate MACD
        _, _, histogram = self.calculate_macd(symbol)
        
        if histogram is None:
            return False
        
        # Check if histogram is negative and decreasing
        latest = histogram.iloc[-1]
        previous = histogram.iloc[-2] if len(histogram) > 1 else 0
        
        return latest < 0 and latest < previous
    
    def is_support_level(self, symbol: str, price: float, 
                       lookback: int = 20, tolerance: float = 0.02) -> bool:
        """
        Check if a price level is a support level.
        
        Args:
            symbol: Stock symbol
            price: Price level to check
            lookback: Number of days to look back
            tolerance: Tolerance percentage
            
        Returns:
            True if support level, False otherwise
        """
        try:
            # Get historical low data
            hist_data = self.market_data.get_historical_data(
                symbol, 
                days=lookback*2,  # Get more days to ensure we have enough data
                fields=["low"]
            )
            
            if hist_data is None or hist_data.empty:
                return False
            
            # Calculate minimum price with tolerance
            min_price = price * (1 - tolerance)
            max_price = price * (1 + tolerance)
            
            # Count how many times price was near this level
            near_level = hist_data[(hist_data["low"] >= min_price) & (hist_data["low"] <= max_price)]
            
            # Require at least 2 instances to consider it a support level
            return len(near_level) >= 2
            
        except Exception as e:
            logger.error(f"Error checking support level for {symbol}: {str(e)}")
            return False
    
    def is_resistance_level(self, symbol: str, price: float, 
                         lookback: int = 20, tolerance: float = 0.02) -> bool:
        """
        Check if a price level is a resistance level.
        
        Args:
            symbol: Stock symbol
            price: Price level to check
            lookback: Number of days to look back
            tolerance: Tolerance percentage
            
        Returns:
            True if resistance level, False otherwise
        """
        try:
            # Get historical high data
            hist_data = self.market_data.get_historical_data(
                symbol, 
                days=lookback*2,  # Get more days to ensure we have enough data
                fields=["high"]
            )
            
            if hist_data is None or hist_data.empty:
                return False
            
            # Calculate minimum price with tolerance
            min_price = price * (1 - tolerance)
            max_price = price * (1 + tolerance)
            
            # Count how many times price was near this level
            near_level = hist_data[(hist_data["high"] >= min_price) & (hist_data["high"] <= max_price)]
            
            # Require at least 2 instances to consider it a resistance level
            return len(near_level) >= 2
            
        except Exception as e:
            logger.error(f"Error checking resistance level for {symbol}: {str(e)}")
            return False
    
    def get_trend_strength(self, symbol: str, days: int = 20) -> Optional[float]:
        """
        Calculate the strength of a trend using linear regression.
        
        Args:
            symbol: Stock symbol
            days: Number of days to check
            
        Returns:
            Trend strength (r-squared) or None if data not available
        """
        try:
            # Get historical close data
            hist_data = self.market_data.get_historical_data(
                symbol, 
                days=days*2,  # Get more days to ensure we have enough data
                fields=["close"]
            )
            
            if hist_data is None or len(hist_data) < days:
                logger.warning(f"Insufficient data for {symbol} to calculate trend strength")
                return None
            
            # Calculate linear regression
            y = hist_data["close"].values[-days:]
            x = np.arange(len(y))
            
            _, _, r_value, _, _ = np.polyfit(x, y, 1, full=True)
            
            # Return r-squared value
            return r_value * r_value
            
        except Exception as e:
            logger.error(f"Error calculating trend strength for {symbol}: {str(e)}")
            return None 