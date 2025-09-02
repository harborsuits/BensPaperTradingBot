#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standard Strategy Implementations - Common trading strategies
"""

import logging
import numpy as np
from typing import Dict, Any
from datetime import datetime

from trading_bot.strategy.base.strategy import Strategy

# Setup logging
logger = logging.getLogger("StandardStrategies")

class MomentumStrategy(Strategy):
    """Simple momentum strategy based on recent price action"""
    
    def generate_signal(self, market_data: Dict[str, Any]) -> float:
        """Generate momentum-based signal."""
        # Extract data
        prices = market_data.get("prices", [])
        
        if len(prices) < 2:
            logger.warning(f"{self.name}: Insufficient price data for signal generation")
            return 0.0
        
        # Get config parameters
        fast_period = self.config.get("fast_period", 5)
        slow_period = self.config.get("slow_period", 20)
        
        # Ensure enough data
        if len(prices) < slow_period:
            logger.debug(f"{self.name}: Not enough price history for slow period ({len(prices)}/{slow_period})")
            return 0.0
        
        # Test array shape to determine if we're dealing with an uptrend or downtrend
        # In uptrends, earlier prices are lower than recent prices
        # In downtrends, earlier prices are higher than recent prices
        
        # Calculate recent price changes for fast and slow periods
        fast_change = prices[-1] - prices[-fast_period]
        slow_change = prices[-1] - prices[-slow_period]
        
        # Normalize by price level to get percentage changes
        fast_pct = fast_change / prices[-fast_period]
        slow_pct = slow_change / prices[-slow_period]
        
        # Compute signal: in uptrends fast_pct is positive
        # In stronger uptrends, fast_pct > slow_pct, giving a positive signal
        # In downtrends, fast_pct is negative and usually more negative than slow_pct
        signal = fast_pct * 10
        
        # Scale signal to be between -1 and 1
        scaled_signal = np.clip(signal, -1.0, 1.0)
        
        # Update last signal and time
        self.last_signal = scaled_signal
        self.last_update_time = datetime.now()
        
        logger.debug(f"{self.name} generated signal: {scaled_signal:.4f}")
        return scaled_signal


class TrendFollowingStrategy(Strategy):
    """Trend following strategy based on moving averages"""
    
    def generate_signal(self, market_data: Dict[str, Any]) -> float:
        """Generate trend-following signal."""
        # Extract data
        prices = market_data.get("prices", [])
        
        if len(prices) < 2:
            logger.warning(f"{self.name}: Insufficient price data for signal generation")
            return 0.0
        
        # Get config parameters
        short_ma_period = self.config.get("short_ma_period", 10)
        long_ma_period = self.config.get("long_ma_period", 30)
        
        # Ensure enough data
        if len(prices) < long_ma_period:
            logger.debug(f"{self.name}: Not enough price history for long MA period ({len(prices)}/{long_ma_period})")
            return 0.0
        
        # Calculate moving averages
        short_ma = np.mean(prices[-short_ma_period:])
        long_ma = np.mean(prices[-long_ma_period:])
        
        # Calculate trend strength
        trend_strength = (short_ma / long_ma - 1) * 10
        
        # Generate signal between -1 and 1
        signal = np.clip(trend_strength * 5, -1.0, 1.0)
        
        # Update last signal and time
        self.last_signal = signal
        self.last_update_time = datetime.now()
        
        logger.debug(f"{self.name} generated signal: {signal:.4f}")
        return signal


class MeanReversionStrategy(Strategy):
    """Mean reversion strategy looking for oversold/overbought conditions"""
    
    def generate_signal(self, market_data: Dict[str, Any]) -> float:
        """Generate mean-reversion signal."""
        # Extract data
        prices = market_data.get("prices", [])
        
        if len(prices) < 2:
            logger.warning(f"{self.name}: Insufficient price data for signal generation")
            return 0.0
        
        # Get config parameters
        period = self.config.get("period", 20)
        std_dev_factor = self.config.get("std_dev_factor", 2.0)
        
        # Ensure enough data
        if len(prices) < period:
            logger.debug(f"{self.name}: Not enough price history for period ({len(prices)}/{period})")
            return 0.0
        
        # Calculate mean and standard deviation
        mean_price = np.mean(prices[-period:])
        std_dev = np.std(prices[-period:])
        
        # Calculate z-score (how many std devs from mean)
        z_score = (prices[-1] - mean_price) / std_dev if std_dev > 0 else 0
        
        # Mean reversion signal (negative of z-score, scaled)
        # When price is high (positive z-score), signal is negative (sell)
        # When price is low (negative z-score), signal is positive (buy)
        signal = -z_score / std_dev_factor
        
        # Clip signal to be between -1 and 1
        signal = np.clip(signal, -1.0, 1.0)
        
        # Update last signal and time
        self.last_signal = signal
        self.last_update_time = datetime.now()
        
        logger.debug(f"{self.name} generated signal: {signal:.4f}, z-score: {z_score:.4f}")
        return signal 