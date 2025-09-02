#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volatility Signals Module

This module provides the VolatilitySignals class for analyzing
volatility-based trading signals and metrics.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, date, timedelta

from trading_bot.market.market_data import MarketData

logger = logging.getLogger(__name__)

class VolatilitySignals:
    """
    Volatility Signals class for generating and analyzing volatility-based trading signals.
    
    This class provides methods to calculate various volatility metrics and 
    generate trading signals based on volatility patterns.
    """
    
    def __init__(self, market_data: MarketData):
        """
        Initialize the VolatilitySignals object.
        
        Args:
            market_data: MarketData instance for retrieving historical data
        """
        self.market_data = market_data
        self._iv_cache = {}  # Cache for implied volatility data
        
        logger.info("Initialized VolatilitySignals")
    
    def get_historical_volatility(self, symbol: str, window: int = 20, 
                                trading_days: int = 252) -> Optional[float]:
        """
        Calculate the historical volatility for a symbol.
        
        Args:
            symbol: Stock symbol
            window: Number of days for rolling window
            trading_days: Number of trading days in a year
            
        Returns:
            Historical volatility as a decimal or None if data not available
        """
        try:
            # Get historical close data
            hist_data = self.market_data.get_historical_data(
                symbol, 
                days=window+10,  # Get a bit more days to ensure we have enough data
                fields=["close"]
            )
            
            if hist_data is None or len(hist_data) < window:
                logger.warning(f"Insufficient data for {symbol} to calculate historical volatility")
                return None
            
            # Calculate log returns
            returns = np.log(hist_data["close"] / hist_data["close"].shift(1)).dropna()
            
            # Calculate standard deviation of returns
            std_dev = returns.std()
            
            # Annualize the standard deviation
            hist_vol = std_dev * np.sqrt(trading_days)
            
            return hist_vol
            
        except Exception as e:
            logger.error(f"Error calculating historical volatility for {symbol}: {str(e)}")
            return None
    
    def get_iv_rank(self, symbol: str, lookback_days: int = 252) -> Optional[float]:
        """
        Calculate the implied volatility rank for a symbol.
        
        IV Rank is the percentile ranking of the current IV relative to the
        historical IV range over a specified lookback period.
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days to look back for historical IV
            
        Returns:
            IV Rank as a percentage (0-100) or None if data not available
        """
        try:
            # In a real implementation, we would retrieve historical IV data
            # Here, we'll use a mock implementation
            
            # Check cache
            cache_key = f"{symbol}_{lookback_days}"
            if cache_key in self._iv_cache:
                return self._iv_cache[cache_key]
            
            # Get current IV (approximated by historical volatility)
            current_iv = self.get_historical_volatility(symbol, window=20)
            
            if current_iv is None:
                return None
            
            # Generate mock historical IV data
            np.random.seed(sum(ord(c) for c in symbol))  # Seed for reproducibility
            base_iv = current_iv * 0.8  # Base IV level
            range_iv = current_iv * 0.4  # Range of IV values
            
            # Generate IV values with some randomness and mean-reversion
            hist_ivs = []
            prev_iv = base_iv
            for _ in range(lookback_days):
                # Mean-reverting random walk
                new_iv = prev_iv + np.random.normal(0, 0.01) + 0.1 * (base_iv - prev_iv)
                # Ensure IV is positive and within reasonable range
                new_iv = max(0.05, min(1.0, new_iv))
                hist_ivs.append(new_iv)
                prev_iv = new_iv
            
            # Calculate IV rank
            min_iv = min(hist_ivs)
            max_iv = max(hist_ivs)
            
            if max_iv == min_iv:
                iv_rank = 50.0  # Default to middle if min = max
            else:
                iv_rank = 100 * (current_iv - min_iv) / (max_iv - min_iv)
            
            # Cache result
            self._iv_cache[cache_key] = iv_rank
            
            return iv_rank
            
        except Exception as e:
            logger.error(f"Error calculating IV rank for {symbol}: {str(e)}")
            return None
    
    def get_iv_percentile(self, symbol: str, lookback_days: int = 252) -> Optional[float]:
        """
        Calculate the implied volatility percentile for a symbol.
        
        IV Percentile is the percentage of days in the lookback period
        where IV was lower than the current IV.
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days to look back for historical IV
            
        Returns:
            IV Percentile as a percentage (0-100) or None if data not available
        """
        try:
            # In a real implementation, we would retrieve historical IV data
            # Here, we'll use a mock implementation
            
            # Check cache
            cache_key = f"{symbol}_pct_{lookback_days}"
            if cache_key in self._iv_cache:
                return self._iv_cache[cache_key]
            
            # Get current IV (approximated by historical volatility)
            current_iv = self.get_historical_volatility(symbol, window=20)
            
            if current_iv is None:
                return None
            
            # Generate mock historical IV data
            np.random.seed(sum(ord(c) for c in symbol))  # Seed for reproducibility
            base_iv = current_iv * 0.8  # Base IV level
            range_iv = current_iv * 0.4  # Range of IV values
            
            # Generate IV values with some randomness and mean-reversion
            hist_ivs = []
            prev_iv = base_iv
            for _ in range(lookback_days):
                # Mean-reverting random walk
                new_iv = prev_iv + np.random.normal(0, 0.01) + 0.1 * (base_iv - prev_iv)
                # Ensure IV is positive and within reasonable range
                new_iv = max(0.05, min(1.0, new_iv))
                hist_ivs.append(new_iv)
                prev_iv = new_iv
            
            # Calculate IV percentile
            count_lower = sum(1 for iv in hist_ivs if iv < current_iv)
            iv_percentile = 100 * count_lower / len(hist_ivs)
            
            # Cache result
            self._iv_cache[cache_key] = iv_percentile
            
            return iv_percentile
            
        except Exception as e:
            logger.error(f"Error calculating IV percentile for {symbol}: {str(e)}")
            return None
    
    def is_iv_relatively_high(self, symbol: str, threshold: float = 70.0) -> bool:
        """
        Check if implied volatility is relatively high for a symbol.
        
        Args:
            symbol: Stock symbol
            threshold: IV rank/percentile threshold (default: 70%)
            
        Returns:
            True if IV is relatively high, False otherwise
        """
        iv_rank = self.get_iv_rank(symbol)
        
        if iv_rank is None:
            return False
            
        return iv_rank >= threshold
    
    def is_iv_relatively_low(self, symbol: str, threshold: float = 30.0) -> bool:
        """
        Check if implied volatility is relatively low for a symbol.
        
        Args:
            symbol: Stock symbol
            threshold: IV rank/percentile threshold (default: 30%)
            
        Returns:
            True if IV is relatively low, False otherwise
        """
        iv_rank = self.get_iv_rank(symbol)
        
        if iv_rank is None:
            return False
            
        return iv_rank <= threshold
    
    def volatility_regime(self, symbol: str) -> str:
        """
        Determine the current volatility regime for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            String indicating the volatility regime ('high', 'normal', 'low')
        """
        iv_rank = self.get_iv_rank(symbol)
        
        if iv_rank is None:
            return "unknown"
            
        if iv_rank >= 70:
            return "high"
        elif iv_rank <= 30:
            return "low"
        else:
            return "normal"
    
    def is_iv_expanding(self, symbol: str, days: int = 5) -> bool:
        """
        Check if implied volatility is expanding recently.
        
        Args:
            symbol: Stock symbol
            days: Number of days to check for expansion
            
        Returns:
            True if IV is expanding, False otherwise
        """
        try:
            # In a real implementation, we would retrieve historical IV data for the past days
            # and check if there's an upward trend
            
            # As a simple mock, we'll use a random value but seed based on symbol
            # for reproducibility
            np.random.seed(sum(ord(c) for c in symbol) + int(datetime.now().timestamp() / 86400))
            
            # 40% chance of expanding IV
            return np.random.random() < 0.4
            
        except Exception as e:
            logger.error(f"Error checking if IV is expanding for {symbol}: {str(e)}")
            return False
    
    def is_iv_contracting(self, symbol: str, days: int = 5) -> bool:
        """
        Check if implied volatility is contracting recently.
        
        Args:
            symbol: Stock symbol
            days: Number of days to check for contraction
            
        Returns:
            True if IV is contracting, False otherwise
        """
        try:
            # In a real implementation, we would retrieve historical IV data for the past days
            # and check if there's a downward trend
            
            # As a simple mock, we'll use a random value but seed based on symbol
            # for reproducibility
            np.random.seed(sum(ord(c) for c in symbol) + int(datetime.now().timestamp() / 86400))
            
            # 40% chance of contracting IV
            return np.random.random() < 0.4
            
        except Exception as e:
            logger.error(f"Error checking if IV is contracting for {symbol}: {str(e)}")
            return False
    
    def iv_skew(self, symbol: str) -> Optional[float]:
        """
        Calculate the implied volatility skew for a symbol.
        
        IV skew measures the difference in IV between OTM puts and OTM calls.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            IV skew value or None if data not available
        """
        try:
            # In a real implementation, we would retrieve option chain data
            # and calculate the IV skew from OTM puts and calls
            
            # As a simple mock, we'll generate a value between -0.2 and 0.2
            # with a bias towards positive (puts having higher IV than calls)
            np.random.seed(sum(ord(c) for c in symbol) + int(datetime.now().timestamp() / 86400))
            
            # Generate IV skew with a bias towards positive values
            return np.random.normal(0.05, 0.1)
            
        except Exception as e:
            logger.error(f"Error calculating IV skew for {symbol}: {str(e)}")
            return None 