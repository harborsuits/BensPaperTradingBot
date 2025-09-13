#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Breakout Strategy

This strategy specializes in identifying and trading breakouts from consolidation
patterns in cryptocurrency markets. It uses volatility contraction, volume surge,
and price action to detect and trade high-probability breakouts.

Key characteristics:
- Detects periods of price consolidation (volatility contraction)
- Identifies breakouts with volume confirmation
- Dynamically adjusts position size based on volatility expansion
- Offers multiple breakout confirmation methods to reduce false signals
- Implements robust risk management specific to breakout trading
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from trading_bot.strategies_new.crypto.base import CryptoBaseStrategy, CryptoSession
from trading_bot.core.events import Event, EventType
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.strategies_new.factory.registry import register_strategy

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="CryptoBreakoutStrategy",
    market_type="crypto",
    description="Breakout strategy for crypto markets specializing in volatility expansion after consolidation",
    timeframes=["M5", "M15", "H1", "H4", "D1"],  # Multiple timeframes as breakouts can occur at various scales
    parameters={
        # Consolidation detection
        "volatility_lookback": {"type": "int", "default": 20, "min": 10, "max": 50},
        "consolidation_threshold": {"type": "float", "default": 0.5, "min": 0.2, "max": 0.8},
        "min_consolidation_periods": {"type": "int", "default": 5, "min": 3, "max": 20},
        
        # Breakout detection
        "range_lookback": {"type": "int", "default": 14, "min": 5, "max": 30},
        "range_extension": {"type": "float", "default": 0.2, "min": 0.05, "max": 0.5},
        "atr_period": {"type": "int", "default": 14, "min": 7, "max": 21},
        "atr_multiplier": {"type": "float", "default": 1.5, "min": 0.5, "max": 3.0},
        
        # Volume confirmation
        "require_volume": {"type": "bool", "default": True},
        "volume_ratio_threshold": {"type": "float", "default": 1.5, "min": 1.0, "max": 5.0},
        "volume_ma_period": {"type": "int", "default": 20, "min": 10, "max": 50},
        
        # Confirmation filters
        "use_bollinger_bands": {"type": "bool", "default": True},
        "bb_period": {"type": "int", "default": 20, "min": 10, "max": 50},
        "bb_std": {"type": "float", "default": 2.0, "min": 1.0, "max": 3.0},
        "use_keltner_channels": {"type": "bool", "default": False},
        "kc_period": {"type": "int", "default": 20, "min": 10, "max": 50},
        "kc_multiplier": {"type": "float", "default": 2.0, "min": 1.0, "max": 3.0},
        
        # Pattern recognition
        "detect_rectangles": {"type": "bool", "default": True},
        "detect_triangles": {"type": "bool", "default": True},
        "detect_flags": {"type": "bool", "default": True},
        "pattern_min_points": {"type": "int", "default": 4, "min": 3, "max": 7},
        
        # Trade execution
        "breakout_confirmation_periods": {"type": "int", "default": 1, "min": 1, "max": 3},
        "use_limit_orders": {"type": "bool", "default": False},
        "limit_order_distance": {"type": "float", "default": 0.01, "min": 0.001, "max": 0.05},
        
        # Stop loss and take profit
        "stop_type": {"type": "str", "default": "volatility", "enum": ["fixed", "volatility", "support_resistance"]},
        "fixed_stop_percent": {"type": "float", "default": 0.02, "min": 0.005, "max": 0.05},
        "stop_atr_multiple": {"type": "float", "default": 1.5, "min": 0.5, "max": 3.0},
        "take_profit_ratio": {"type": "float", "default": 2.0, "min": 1.0, "max": 5.0},
        
        # Risk management
        "risk_per_trade": {"type": "float", "default": 0.01, "min": 0.005, "max": 0.02},
        "max_open_positions": {"type": "int", "default": 2, "min": 1, "max": 5},
        "scale_out_targets": {"type": "int", "default": 3, "min": 1, "max": 5},
        "scale_out_percentages": {"type": "str", "default": "50,30,20", "description": "Comma-separated percentages for scaling out"},
        
        # Advanced features
        "use_trailing_stop": {"type": "bool", "default": True},
        "trail_activation_percent": {"type": "float", "default": 0.01, "min": 0.005, "max": 0.03},
        "false_breakout_filter": {"type": "bool", "default": True},
        "multi_timeframe_filter": {"type": "bool", "default": False},
    }
)
class CryptoBreakoutStrategy(CryptoBaseStrategy):
    """
    A breakout strategy for cryptocurrency markets.
    
    This strategy:
    1. Identifies periods of consolidation (volatility contraction)
    2. Detects breakouts from support/resistance levels
    3. Uses volume confirmation to filter false breakouts
    4. Implements robust risk management for volatile breakout trades
    5. Offers trailing stops and scale-out capabilities to maximize profits
    """
    
    def __init__(self, session: CryptoSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """Initialize the crypto breakout strategy."""
        super().__init__(session, data_pipeline, parameters)
        
        # Strategy-specific state
        self.consolidation_detected = False
        self.consolidation_start = None
        self.range_high = None
        self.range_low = None
        self.breakout_direction = None
        self.breakout_candle = None
        self.breakout_price = None
        self.is_monitoring_breakout = False
        self.breakout_confirmation_count = 0
        self.trailing_stops = {}
        self.scale_out_levels = []
        self.position_targets = {}
        self.false_breakout_count = 0
        self.successful_breakout_count = 0
        
        # Parse scale-out percentages
        scale_out_str = self.parameters["scale_out_percentages"]
        try:
            self.scale_out_percentages = [int(p.strip()) for p in scale_out_str.split(",")]
            # Normalize to ensure they sum to 100
            total = sum(self.scale_out_percentages)
            if total != 100:
                self.scale_out_percentages = [int(p * 100 / total) for p in self.scale_out_percentages]
                logger.info(f"Normalized scale-out percentages to: {self.scale_out_percentages}")
        except:
            self.scale_out_percentages = [50, 30, 20]
            logger.warning(f"Failed to parse scale-out percentages, using defaults: {self.scale_out_percentages}")
        
        # Configuration check
        if self.parameters["use_bollinger_bands"] and self.parameters["use_keltner_channels"]:
            logger.warning("Both Bollinger Bands and Keltner Channels are enabled. This may generate conflicting signals.")
        
        logger.info(f"Initialized crypto breakout strategy with volatility lookback: {self.parameters['volatility_lookback']}, "
                   f"range lookback: {self.parameters['range_lookback']}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicators for breakout detection including volatility measures,
        support/resistance levels, and consolidation patterns.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        if data.empty or len(data) < self.parameters["volatility_lookback"]:
            return indicators
        
        # Calculate ATR for volatility measurement
        atr_period = self.parameters["atr_period"]
        high_low = data["high"] - data["low"]
        high_close = abs(data["high"] - data["close"].shift())
        low_close = abs(data["low"] - data["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators["atr"] = true_range.rolling(window=atr_period).mean()
        
        # Calculate historical volatility
        vol_lookback = self.parameters["volatility_lookback"]
        log_returns = np.log(data["close"] / data["close"].shift(1))
        indicators["volatility"] = log_returns.rolling(window=vol_lookback).std() * np.sqrt(252)  # Annualized
        
        # Calculate current vs historical volatility ratio
        current_vol = indicators["volatility"].iloc[-1]
        vol_ratio = current_vol / indicators["volatility"].rolling(window=vol_lookback).mean().iloc[-1]
        indicators["volatility_ratio"] = vol_ratio
        
        # Detect volatility contraction (consolidation)
        volatility_window = indicators["volatility"].iloc[-vol_lookback:]
        is_contracting = volatility_window.iloc[-1] < volatility_window.mean() * self.parameters["consolidation_threshold"]
        indicators["is_consolidating"] = is_contracting
        
        # Calculate range high and low
        range_lookback = self.parameters["range_lookback"]
        indicators["range_high"] = data["high"].rolling(window=range_lookback).max()
        indicators["range_low"] = data["low"].rolling(window=range_lookback).min()
        
        # Calculate trading range
        indicators["range_width"] = indicators["range_high"] - indicators["range_low"]
        indicators["range_percentage"] = indicators["range_width"] / data["close"] * 100
        
        # Detect if price is currently near range boundaries
        current_close = data["close"].iloc[-1]
        current_high = data["high"].iloc[-1]
        current_low = data["low"].iloc[-1]
        range_high = indicators["range_high"].iloc[-1]
        range_low = indicators["range_low"].iloc[-1]
        range_extension = self.parameters["range_extension"]
        
        # Check if we're testing upper boundary
        upper_threshold = range_high * (1 - range_extension * 0.1)  # Near upper bound
        indicators["near_upper_bound"] = current_high >= upper_threshold
        indicators["breakout_up"] = current_close > range_high
        
        # Check if we're testing lower boundary
        lower_threshold = range_low * (1 + range_extension * 0.1)  # Near lower bound
        indicators["near_lower_bound"] = current_low <= lower_threshold
        indicators["breakout_down"] = current_close < range_low
        
        # Calculate support and resistance levels
        self._calculate_support_resistance(data, indicators)
        
        # Volume analysis
        if self.parameters["require_volume"]:
            vol_ma_period = self.parameters["volume_ma_period"]
            indicators["volume_ma"] = data["volume"].rolling(window=vol_ma_period).mean()
            indicators["volume_ratio"] = data["volume"] / indicators["volume_ma"]
            indicators["high_volume"] = indicators["volume_ratio"] > self.parameters["volume_ratio_threshold"]
            indicators["rising_volume"] = data["volume"] > data["volume"].shift(1)
        
        # Bollinger Bands (for volatility-based breakout confirmation)
        if self.parameters["use_bollinger_bands"]:
            bb_period = self.parameters["bb_period"]
            bb_std = self.parameters["bb_std"]
            indicators["bb_middle"] = data["close"].rolling(window=bb_period).mean()
            price_std = data["close"].rolling(window=bb_period).std()
            indicators["bb_upper"] = indicators["bb_middle"] + (price_std * bb_std)
            indicators["bb_lower"] = indicators["bb_middle"] - (price_std * bb_std)
            indicators["bb_width"] = (indicators["bb_upper"] - indicators["bb_lower"]) / indicators["bb_middle"]
            
            # Detect Bollinger Band squeeze (volatility contraction)
            indicators["bb_squeeze"] = indicators["bb_width"] < indicators["bb_width"].rolling(window=bb_period).mean() * 0.8
            
            # Check for price outside bands (volatility expansion / breakout)
            indicators["above_upper_band"] = data["close"] > indicators["bb_upper"]
            indicators["below_lower_band"] = data["close"] < indicators["bb_lower"]
        
        # Keltner Channels (alternative volatility measure)
        if self.parameters["use_keltner_channels"]:
            kc_period = self.parameters["kc_period"]
            kc_mult = self.parameters["kc_multiplier"]
            indicators["kc_middle"] = data["close"].rolling(window=kc_period).mean()
            kc_range = indicators["atr"].rolling(window=kc_period).mean()
            indicators["kc_upper"] = indicators["kc_middle"] + (kc_range * kc_mult)
            indicators["kc_lower"] = indicators["kc_middle"] - (kc_range * kc_mult)
            
            # Check for price outside channels
            indicators["above_upper_kc"] = data["close"] > indicators["kc_upper"]
            indicators["below_lower_kc"] = data["close"] < indicators["kc_lower"]
        
        # Detect specific pattern types if enabled
        if self.parameters["detect_rectangles"]:
            self._detect_rectangle_pattern(data, indicators)
        
        if self.parameters["detect_triangles"]:
            self._detect_triangle_pattern(data, indicators)
        
        if self.parameters["detect_flags"]:
            self._detect_flag_pattern(data, indicators)
        
        # Update the state based on new indicators
        self._update_consolidation_state(data, indicators)
        
        return indicators
    
    def _calculate_support_resistance(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> None:
        """
        Calculate dynamic support and resistance levels based on recent price action.
        
        Uses pivot points, historical highs/lows, and volume profiles.
        
        Args:
            data: Market data DataFrame
            indicators: Dictionary to update with calculated levels
        """
        if len(data) < 30:
            return
        
        # Keep track of potential support/resistance levels
        levels = []
        
        # Method 1: Find local highs and lows
        window_size = min(20, len(data) // 4)
        price_high = data["high"]
        price_low = data["low"]
        price_close = data["close"]
        
        # Find local highs (price higher than window before and after)
        for i in range(window_size, len(data) - window_size):
            if all(price_high.iloc[i] > price_high.iloc[i-window_size:i]) and \
               all(price_high.iloc[i] > price_high.iloc[i+1:i+window_size+1]):
                levels.append((data.index[i], price_high.iloc[i], "resistance"))
        
        # Find local lows (price lower than window before and after)
        for i in range(window_size, len(data) - window_size):
            if all(price_low.iloc[i] < price_low.iloc[i-window_size:i]) and \
               all(price_low.iloc[i] < price_low.iloc[i+1:i+window_size+1]):
                levels.append((data.index[i], price_low.iloc[i], "support"))
        
        # Method 2: Round number levels often act as support/resistance in crypto
        # Find significant round numbers near current price
        current_price = data["close"].iloc[-1]
        price_magnitude = 10 ** int(np.log10(current_price))
        
        # Major round numbers (e.g., 10,000, 20,000 for BTC)
        for i in range(1, 10):
            round_level = i * price_magnitude
            if 0.7 * current_price <= round_level <= 1.3 * current_price:
                levels.append((data.index[-1], round_level, "round_number"))
        
        # Half and quarter levels (e.g., 5,000, 2,500)
        for fraction in [0.25, 0.5, 0.75]:
            for i in range(1, 10):
                round_level = i * price_magnitude * fraction
                if 0.7 * current_price <= round_level <= 1.3 * current_price:
                    levels.append((data.index[-1], round_level, "round_number"))
        
        # Convert to DataFrames for easier filtering
        if levels:
            levels_df = pd.DataFrame(levels, columns=["time", "level", "type"])
            
            # Store key levels in indicators
            resistance_levels = levels_df[levels_df["type"].isin(["resistance", "round_number"])]["level"].tolist()
            support_levels = levels_df[levels_df["type"].isin(["support", "round_number"])]["level"].tolist()
            
            # Sort levels by proximity to current price
            resistance_levels = sorted([r for r in resistance_levels if r > current_price])
            support_levels = sorted([s for s in support_levels if s < current_price], reverse=True)
            
            indicators["resistance_levels"] = resistance_levels[:3] if resistance_levels else []  # Top 3 closest
            indicators["support_levels"] = support_levels[:3] if support_levels else []  # Top 3 closest
            
            # Identify closest support and resistance
            if resistance_levels:
                indicators["closest_resistance"] = min(resistance_levels)
            
            if support_levels:
                indicators["closest_support"] = max(support_levels)
    
    def _detect_rectangle_pattern(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> None:
        """
        Detect rectangle consolidation patterns (horizontal support/resistance).
        
        Args:
            data: Market data DataFrame
            indicators: Dictionary to update with pattern detections
        """
        # Default to no pattern
        indicators["rectangle_pattern"] = False
        
        if len(data) < self.parameters["range_lookback"] * 2:
            return
            
        lookback = self.parameters["range_lookback"]
        min_points = self.parameters["pattern_min_points"]
        
        # Check for horizontal resistance (multiple tests of same level)
        highs = data["high"].iloc[-lookback:]
        high_std = highs.std() / highs.mean()  # Normalized standard deviation
        
        # Check for horizontal support (multiple tests of same level)
        lows = data["low"].iloc[-lookback:]
        low_std = lows.std() / lows.mean()  # Normalized standard deviation
        
        # Rectangle pattern: both support and resistance are horizontal (low std dev)
        if high_std < 0.02 and low_std < 0.02 and \
           (highs.max() - lows.min()) / lows.min() < 0.15:  # Range is less than 15%
            indicators["rectangle_pattern"] = True
            indicators["rectangle_high"] = highs.max()
            indicators["rectangle_low"] = lows.min()
            indicators["pattern_type"] = "rectangle"
            logger.info(f"Detected rectangle pattern: high: {highs.max():.2f}, low: {lows.min():.2f}")
    
    def _detect_triangle_pattern(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> None:
        """
        Detect triangle patterns (converging support/resistance lines).
        
        Args:
            data: Market data DataFrame
            indicators: Dictionary to update with pattern detections
        """
        # Default to no pattern
        indicators["triangle_pattern"] = False
        
        if len(data) < self.parameters["range_lookback"] * 2:
            return
            
        lookback = self.parameters["range_lookback"]
        
        # Get recent price data
        recent_data = data.iloc[-lookback:]
        highs = recent_data["high"]
        lows = recent_data["low"]
        
        # Try to fit lines to highs and lows
        high_x = np.arange(len(highs))
        low_x = np.arange(len(lows))
        
        # Only proceed if we have sufficient data
        if len(highs) < 5 or len(lows) < 5:
            return
            
        try:
            # Linear regression for resistance line
            high_coef = np.polyfit(high_x, highs, 1)
            high_slope = high_coef[0]
            high_intercept = high_coef[1]
            
            # Linear regression for support line
            low_coef = np.polyfit(low_x, lows, 1)
            low_slope = low_coef[0]
            low_intercept = low_coef[1]
            
            # Identify triangle types
            # Symmetrical: resistance sloping down, support sloping up
            if high_slope < -0.0001 and low_slope > 0.0001:
                indicators["triangle_pattern"] = True
                indicators["triangle_type"] = "symmetrical"
                indicators["pattern_type"] = "symmetrical_triangle"
                logger.info("Detected symmetrical triangle pattern")
                
            # Ascending: resistance roughly horizontal, support sloping up
            elif abs(high_slope) < 0.0005 and low_slope > 0.0001:
                indicators["triangle_pattern"] = True
                indicators["triangle_type"] = "ascending"
                indicators["pattern_type"] = "ascending_triangle"
                logger.info("Detected ascending triangle pattern")
                
            # Descending: resistance sloping down, support roughly horizontal
            elif high_slope < -0.0001 and abs(low_slope) < 0.0005:
                indicators["triangle_pattern"] = True
                indicators["triangle_type"] = "descending"
                indicators["pattern_type"] = "descending_triangle"
                logger.info("Detected descending triangle pattern")
                
            # Store regression coefficients
            if indicators["triangle_pattern"]:
                indicators["resistance_slope"] = high_slope
                indicators["resistance_intercept"] = high_intercept
                indicators["support_slope"] = low_slope
                indicators["support_intercept"] = low_intercept
                
                # Calculate apex (where the lines converge)
                if abs(high_slope - low_slope) > 0.0001:  # Avoid division by near-zero
                    apex_x = (low_intercept - high_intercept) / (high_slope - low_slope)
                    indicators["apex_distance"] = apex_x - len(high_x)
        except:
            # Linear regression can fail with certain data patterns
            logger.warning("Failed to detect triangle pattern due to regression error")
    
    def _detect_flag_pattern(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> None:
        """
        Detect flag/pennant patterns (small consolidation after strong trend).
        
        Args:
            data: Market data DataFrame
            indicators: Dictionary to update with pattern detections
        """
        # Default to no pattern
        indicators["flag_pattern"] = False
        
        lookback = self.parameters["range_lookback"]
        if len(data) < lookback * 2:
            return
            
        # Check for preceding strong trend ("flag pole")
        pole_data = data.iloc[-lookback*2:-lookback]
        pole_returns = (pole_data["close"].iloc[-1] / pole_data["close"].iloc[0]) - 1
        
        # Consolidation data (the "flag" part)
        flag_data = data.iloc[-lookback:]
        flag_returns = abs(flag_data["close"].iloc[-1] / flag_data["close"].iloc[0] - 1)
        
        # Flag conditions: strong trend followed by tight consolidation
        is_bullish_flag = pole_returns > 0.15 and flag_returns < 0.07 and flag_data["close"].std() / flag_data["close"].mean() < 0.03
        is_bearish_flag = pole_returns < -0.15 and flag_returns < 0.07 and flag_data["close"].std() / flag_data["close"].mean() < 0.03
        
        if is_bullish_flag or is_bearish_flag:
            indicators["flag_pattern"] = True
            indicators["flag_type"] = "bullish" if is_bullish_flag else "bearish"
            indicators["pattern_type"] = f"{indicators['flag_type']}_flag"
            indicators["flag_high"] = flag_data["high"].max()
            indicators["flag_low"] = flag_data["low"].min()
            logger.info(f"Detected {indicators['flag_type']} flag pattern")
    
    def _update_consolidation_state(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> None:
        """
        Update the internal state tracking consolidation and breakout progress.
        
        Args:
            data: Market data DataFrame
            indicators: Dictionary of current indicators
        """
        is_consolidating = indicators.get("is_consolidating", False) or \
                           indicators.get("bb_squeeze", False) or \
                           indicators.get("rectangle_pattern", False) or \
                           indicators.get("triangle_pattern", False) or \
                           indicators.get("flag_pattern", False)
        
        # Start tracking a new consolidation
        if is_consolidating and not self.consolidation_detected:
            self.consolidation_detected = True
            self.consolidation_start = data.index[-1]
            self.range_high = indicators.get("range_high", data["high"].iloc[-self.parameters["range_lookback"]:].max()).iloc[-1]
            self.range_low = indicators.get("range_low", data["low"].iloc[-self.parameters["range_lookback"]:].min()).iloc[-1]
            logger.info(f"Started tracking consolidation: high={self.range_high:.2f}, low={self.range_low:.2f}")
            
        # Update existing consolidation
        elif is_consolidating and self.consolidation_detected:
            # Update range bounds if we have pattern-specific values
            if "rectangle_high" in indicators:
                self.range_high = indicators["rectangle_high"]
                self.range_low = indicators["rectangle_low"]
            elif "flag_high" in indicators:
                self.range_high = indicators["flag_high"]
                self.range_low = indicators["flag_low"]
            elif "triangle_type" in indicators:
                # Use dynamic support/resistance based on slope
                x_pos = len(data) - 1  # Current position
                if "resistance_slope" in indicators and "resistance_intercept" in indicators:
                    self.range_high = indicators["resistance_slope"] * x_pos + indicators["resistance_intercept"]
                if "support_slope" in indicators and "support_intercept" in indicators:
                    self.range_low = indicators["support_slope"] * x_pos + indicators["support_intercept"]
            
        # Reset if consolidation ends without breakout
        elif not is_consolidating and self.consolidation_detected and not self.is_monitoring_breakout:
            consolidation_duration = (data.index[-1] - self.consolidation_start).total_seconds() / 86400  # Days
            logger.info(f"Consolidation ended without breakout after {consolidation_duration:.1f} days")
            self.consolidation_detected = False
            self.consolidation_start = None
            
        # Check for potential breakout
        if self.consolidation_detected and not self.is_monitoring_breakout:
            breakout_up = indicators.get("breakout_up", False)
            breakout_down = indicators.get("breakout_down", False)
            volume_confirmed = indicators.get("high_volume", True)  # Default to True if volume check disabled
            
            # Check for upward breakout
            if breakout_up and volume_confirmed:
                self.breakout_direction = "up"
                self.breakout_candle = data.iloc[-1]
                self.breakout_price = data["close"].iloc[-1]
                self.is_monitoring_breakout = True
                self.breakout_confirmation_count = 0
                logger.info(f"Potential upward breakout detected at {self.breakout_price:.2f}")
                
            # Check for downward breakout
            elif breakout_down and volume_confirmed:
                self.breakout_direction = "down"
                self.breakout_candle = data.iloc[-1]
                self.breakout_price = data["close"].iloc[-1]
                self.is_monitoring_breakout = True
                self.breakout_confirmation_count = 0
                logger.info(f"Potential downward breakout detected at {self.breakout_price:.2f}")
                
        # Update breakout monitoring
        if self.is_monitoring_breakout:
            confirmation_periods = self.parameters["breakout_confirmation_periods"]
            current_close = data["close"].iloc[-1]
            
            # Check if breakout is still valid
            if self.breakout_direction == "up" and current_close > self.range_high:
                self.breakout_confirmation_count += 1
                logger.info(f"Breakout confirmation: {self.breakout_confirmation_count}/{confirmation_periods}")
                
            elif self.breakout_direction == "down" and current_close < self.range_low:
                self.breakout_confirmation_count += 1
                logger.info(f"Breakout confirmation: {self.breakout_confirmation_count}/{confirmation_periods}")
                
            else:
                # Failed breakout
                logger.info(f"Breakout failed - price returned to range")
                self.is_monitoring_breakout = False
                self.breakout_direction = None
                self.false_breakout_count += 1
                
            # Check if breakout is fully confirmed
            if self.breakout_confirmation_count >= confirmation_periods:
                logger.info(f"Breakout confirmed after {confirmation_periods} periods")
                # Keep monitoring for trade management, but mark as confirmed
                self.successful_breakout_count += 1
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on breakout patterns and confirmations.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            "long_entry": False,
            "short_entry": False,
            "long_exit": False,
            "short_exit": False,
            "signal_strength": 0.0,
            "stop_loss": None,
            "take_profit": None,
        }
        
        if data.empty or not indicators:
            return signals
        
        # Check for confirmed breakouts
        is_breakout_confirmed = self.is_monitoring_breakout and self.breakout_confirmation_count >= self.parameters["breakout_confirmation_periods"]
        
        # Entry signals based on confirmed breakouts
        if is_breakout_confirmed:
            # Calculate signal strength based on multiple factors
            base_strength = 0.7  # Confirmed breakouts start with strong base signal
            
            # Adjust strength based on volume
            volume_bonus = 0.0
            if self.parameters["require_volume"] and indicators.get("high_volume", False):
                volume_ratio = indicators.get("volume_ratio", 1.0)
                volume_bonus = min(0.2, (volume_ratio - 1) * 0.1)  # Up to +0.2 bonus for high volume
            
            # Adjust strength based on pattern type
            pattern_bonus = 0.0
            pattern_type = indicators.get("pattern_type", None)
            if pattern_type in ["rectangle", "symmetrical_triangle"]:
                pattern_bonus = 0.05
            elif pattern_type in ["ascending_triangle", "descending_triangle"]:
                pattern_bonus = 0.1
            elif pattern_type in ["bullish_flag", "bearish_flag"]:
                pattern_bonus = 0.15  # Flags have highest reliability post-breakout
            
            # Adjust strength based on volatility expansion
            volatility_bonus = 0.0
            if "volatility_ratio" in indicators:
                volatility_ratio = indicators["volatility_ratio"]
                if volatility_ratio > 1.2:  # Volatility expanding
                    volatility_bonus = min(0.1, (volatility_ratio - 1) * 0.1)
            
            # Calculate total signal strength (cap at 1.0)
            signal_strength = min(1.0, base_strength + volume_bonus + pattern_bonus + volatility_bonus)
            signals["signal_strength"] = signal_strength
            
            # Generate directional signals
            if self.breakout_direction == "up":
                signals["long_entry"] = True
                logger.info(f"Confirmed upward breakout signal generated with strength {signal_strength:.2f}")
                
                # Calculate take profit based on pattern height projection
                range_height = self.range_high - self.range_low
                signals["take_profit"] = self.range_high + (range_height * self.parameters["take_profit_ratio"])
                
                # Calculate stop loss based on configuration
                if self.parameters["stop_type"] == "fixed":
                    signals["stop_loss"] = data["close"].iloc[-1] * (1 - self.parameters["fixed_stop_percent"])
                elif self.parameters["stop_type"] == "volatility" and "atr" in indicators:
                    atr = indicators["atr"].iloc[-1]
                    signals["stop_loss"] = data["close"].iloc[-1] - (atr * self.parameters["stop_atr_multiple"])
                elif self.parameters["stop_type"] == "support_resistance":
                    # Use the breakout level as the stop (failed breakout)
                    signals["stop_loss"] = self.range_high * 0.995  # Slightly below breakout level
                
            elif self.breakout_direction == "down":
                signals["short_entry"] = True
                logger.info(f"Confirmed downward breakout signal generated with strength {signal_strength:.2f}")
                
                # Calculate take profit based on pattern height projection
                range_height = self.range_high - self.range_low
                signals["take_profit"] = self.range_low - (range_height * self.parameters["take_profit_ratio"])
                
                # Calculate stop loss based on configuration
                if self.parameters["stop_type"] == "fixed":
                    signals["stop_loss"] = data["close"].iloc[-1] * (1 + self.parameters["fixed_stop_percent"])
                elif self.parameters["stop_type"] == "volatility" and "atr" in indicators:
                    atr = indicators["atr"].iloc[-1]
                    signals["stop_loss"] = data["close"].iloc[-1] + (atr * self.parameters["stop_atr_multiple"])
                elif self.parameters["stop_type"] == "support_resistance":
                    # Use the breakout level as the stop (failed breakout)
                    signals["stop_loss"] = self.range_low * 1.005  # Slightly above breakout level
                    
            # Calculate scale-out targets
            if self.parameters["scale_out_targets"] > 1:
                self._set_scale_out_targets(data, signals)
                
        # Exit signals for existing positions
        for position in self.positions:
            # Long position exit conditions
            if position.direction == "long":
                # Exit on opposite breakout
                if self.is_monitoring_breakout and self.breakout_direction == "down" and \
                   self.breakout_confirmation_count > 0:
                    signals["long_exit"] = True
                    logger.info(f"Exit long signal: Confirmed downward breakout")
                
                # Failed retest of breakout level (potential false breakout)
                elif indicators.get("below_lower_band", False) or indicators.get("below_lower_kc", False):
                    signals["long_exit"] = True
                    logger.info(f"Exit long signal: Price broke below volatility bands")
                    
            # Short position exit conditions
            elif position.direction == "short":
                # Exit on opposite breakout
                if self.is_monitoring_breakout and self.breakout_direction == "up" and \
                   self.breakout_confirmation_count > 0:
                    signals["short_exit"] = True
                    logger.info(f"Exit short signal: Confirmed upward breakout")
                
                # Failed retest of breakout level (potential false breakout)
                elif indicators.get("above_upper_band", False) or indicators.get("above_upper_kc", False):
                    signals["short_exit"] = True
                    logger.info(f"Exit short signal: Price broke above volatility bands")
        
        return signals
    
    def _set_scale_out_targets(self, data: pd.DataFrame, signals: Dict[str, Any]) -> None:
        """
        Calculate scale-out targets for position management.
        
        Args:
            data: Market data DataFrame
            signals: Trading signals to update with targets
        """
        if not signals.get("take_profit"):
            return
        
        current_price = data["close"].iloc[-1]
        take_profit = signals["take_profit"]
        scale_out_targets = self.parameters["scale_out_targets"]
        
        # Calculate intermediary targets
        target_prices = []
        if signals.get("long_entry"):
            # For long positions, targets are between current price and take profit
            price_range = take_profit - current_price
            for i in range(1, scale_out_targets):
                target_pct = i / scale_out_targets
                target_prices.append(current_price + (price_range * target_pct))
            target_prices.append(take_profit)  # Final target
            
        elif signals.get("short_entry"):
            # For short positions, targets are between current price and take profit
            price_range = current_price - take_profit
            for i in range(1, scale_out_targets):
                target_pct = i / scale_out_targets
                target_prices.append(current_price - (price_range * target_pct))
            target_prices.append(take_profit)  # Final target
        
        # Store targets with corresponding percentages
        if target_prices:
            self.scale_out_levels = target_prices
            signals["scale_out_targets"] = dict(zip(target_prices, self.scale_out_percentages))
            logger.info(f"Set {len(target_prices)} scale-out targets: {[f'{p:.2f}' for p in target_prices]}")
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate position size based on risk parameters and breakout volatility.
        
        Breakout trades often have higher volatility, so position sizing is adjusted
        to maintain consistent risk levels.
        
        Args:
            direction: Direction of the trade ('long' or 'short')
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Position size in crypto units
        """
        if data.empty or not indicators:
            return 0.0
        
        # Account balance (in base currency)
        account_balance = 10000.0  # Mock value, would come from exchange API
        risk_percentage = self.parameters["risk_per_trade"]
        risk_amount = account_balance * risk_percentage
        
        # Current price
        current_price = data["close"].iloc[-1]
        
        # Stop distance based on stop loss
        stop_loss = self.signals.get("stop_loss", None)
        if stop_loss is None:
            # Fallback to volatility-based stop if not explicitly set
            if "atr" in indicators:
                atr = indicators["atr"].iloc[-1]
                stop_distance = atr * self.parameters["stop_atr_multiple"]
            else:
                # Use fixed percentage as last resort
                stop_distance = current_price * self.parameters["fixed_stop_percent"]
        else:
            stop_distance = abs(current_price - stop_loss)
        
        # Calculate position size based on risk and stop distance
        position_size_usd = risk_amount
        position_size_crypto = position_size_usd / current_price
        
        # Adjust based on stop distance
        if stop_distance > 0:
            position_size_crypto = risk_amount / stop_distance
        
        # Adjust position size based on signal strength
        signal_strength = self.signals.get("signal_strength", 0.5)
        strength_factor = max(0.3, signal_strength)  # Minimum 30% of calculated size
        position_size_crypto *= strength_factor
        
        # Adjust position size based on volatility ratio
        if "volatility_ratio" in indicators and indicators["volatility_ratio"] > 1.3:
            # Reduce size for very high volatility
            vol_ratio = indicators["volatility_ratio"]
            vol_adjustment = max(0.5, 1.0 / vol_ratio)
            position_size_crypto *= vol_adjustment
            logger.info(f"Reduced position size to {vol_adjustment:.2f}x due to high volatility")
        
        # Apply precision appropriate for the asset
        decimals = 8 if self.session.symbol.startswith("BTC") else 6
        position_size_crypto = round(position_size_crypto, decimals)
        
        # Ensure minimum trade size
        min_trade_size = self.session.min_trade_size
        position_size_crypto = max(position_size_crypto, min_trade_size)
        
        logger.info(f"Breakout position size: {position_size_crypto} {self.session.symbol.split('-')[0]} "
                  f"(signal strength: {signal_strength:.2f})")
        
        return position_size_crypto
    
    def _on_timeframe_completed(self, event: Event) -> None:
        """
        Handle timeframe completed events.
        
        For breakout strategies, we process signals at the close of each candle.
        We also update trailing stops for open positions.
        """
        super()._on_timeframe_completed(event)
        
        # Only process if this is our timeframe
        if event.data.get('timeframe') != self.session.timeframe:
            return
        
        # Calculate new indicators with the latest data
        self.indicators = self.calculate_indicators(self.market_data)
        
        # Generate trading signals
        self.signals = self.generate_signals(self.market_data, self.indicators)
        
        # Check for trade opportunities
        self._check_for_trade_opportunities()
        
        # Update trailing stops if enabled
        if self.parameters["use_trailing_stop"]:
            self._update_trailing_stops()
    
    def _update_trailing_stops(self) -> None:
        """
        Update trailing stops for open positions based on current price and ATR.
        """
        if not self.positions or self.market_data.empty:
            return
        
        current_price = self.market_data["close"].iloc[-1]
        activation_pct = self.parameters["trail_activation_percent"]
        
        # Use ATR for dynamic trailing if available
        atr_multiple = self.parameters["stop_atr_multiple"] * 0.75  # Tighter trailing than initial stop
        atr = self.indicators.get("atr", None)
        atr_value = atr.iloc[-1] if atr is not None else None
        
        # Process each open position
        for position in self.positions:
            if not hasattr(position, "id") or position.id is None:
                continue
                
            # Initialize trailing stop if not already set
            if position.id not in self.trailing_stops:
                if position.direction == "long":
                    # For long positions, set initial trailing stop below entry price
                    if atr_value:
                        self.trailing_stops[position.id] = position.entry_price - (atr_value * atr_multiple)
                    else:
                        self.trailing_stops[position.id] = position.entry_price * (1 - activation_pct * 2)
                else:  # short
                    # For short positions, set initial trailing stop above entry price
                    if atr_value:
                        self.trailing_stops[position.id] = position.entry_price + (atr_value * atr_multiple)
                    else:
                        self.trailing_stops[position.id] = position.entry_price * (1 + activation_pct * 2)
                
                logger.info(f"Set initial trailing stop for {position.direction} at {self.trailing_stops[position.id]:.2f}")
            
            # Update trailing stop if price moved favorably
            if position.direction == "long":
                # Calculate activation threshold (position needs to be in profit)
                activation_threshold = position.entry_price * (1 + activation_pct)
                
                # Only trail if price has moved beyond activation threshold
                if current_price > activation_threshold:
                    # Calculate new stop price
                    new_stop = current_price * (1 - activation_pct)
                    if atr_value:
                        new_stop = current_price - (atr_value * atr_multiple)
                    
                    # Update only if new stop is higher than current stop
                    if new_stop > self.trailing_stops[position.id]:
                        self.trailing_stops[position.id] = new_stop
                        logger.info(f"Updated trailing stop for long position {position.id}: {new_stop:.2f}")
            
            else:  # short position
                # Calculate activation threshold
                activation_threshold = position.entry_price * (1 - activation_pct)
                
                # Only trail if price has moved beyond activation threshold
                if current_price < activation_threshold:
                    # Calculate new stop price
                    new_stop = current_price * (1 + activation_pct)
                    if atr_value:
                        new_stop = current_price + (atr_value * atr_multiple)
                    
                    # Update only if new stop is lower than current stop
                    if new_stop < self.trailing_stops[position.id]:
                        self.trailing_stops[position.id] = new_stop
                        logger.info(f"Updated trailing stop for short position {position.id}: {new_stop:.2f}")
    
    def _on_market_data_updated(self, event: Event) -> None:
        """
        Handle market data updated events.
        
        For breakout strategies, we check trailing stops and scale-out targets on each tick.
        """
        super()._on_market_data_updated(event)
        
        # Skip if no open positions or no trailing stops
        if not self.positions:
            return
        
        # Check if we need to exit any positions based on trailing stops
        current_price = event.data.get('close', None)
        if current_price is None:
            return
        
        # Check trailing stops
        for position in self.positions:
            if not hasattr(position, "id") or position.id not in self.trailing_stops:
                continue
                
            trailing_stop = self.trailing_stops[position.id]
            
            # Check if trailing stop is hit
            if (position.direction == "long" and current_price < trailing_stop) or \
               (position.direction == "short" and current_price > trailing_stop):
                logger.info(f"Trailing stop triggered for {position.direction} position {position.id} at {trailing_stop:.2f}")
                self._close_position(position.id)
        
        # Check scale-out targets
        self._check_scale_out_targets(current_price)
    
    def _check_scale_out_targets(self, current_price: float) -> None:
        """
        Check if any scale-out targets have been hit and execute partial exits.
        
        Args:
            current_price: Current market price
        """
        if not self.scale_out_levels or not self.positions:
            return
        
        # Process each position
        for position in self.positions:
            if not hasattr(position, "id") or position.id not in self.position_targets:
                # Initialize targets for this position
                if position.direction == "long":
                    # For long positions, targets are in ascending order
                    targets = sorted([t for t in self.scale_out_levels if t > position.entry_price])
                    if targets:
                        self.position_targets[position.id] = targets
                else:  # short
                    # For short positions, targets are in descending order
                    targets = sorted([t for t in self.scale_out_levels if t < position.entry_price], reverse=True)
                    if targets:
                        self.position_targets[position.id] = targets
                        
            # Check if any targets have been hit
            if position.id in self.position_targets and self.position_targets[position.id]:
                if position.direction == "long" and current_price >= self.position_targets[position.id][0]:
                    target_price = self.position_targets[position.id].pop(0)
                    self._execute_scale_out(position, target_price)
                    
                elif position.direction == "short" and current_price <= self.position_targets[position.id][0]:
                    target_price = self.position_targets[position.id].pop(0)
                    self._execute_scale_out(position, target_price)
    
    def _execute_scale_out(self, position, target_price: float) -> None:
        """
        Execute a partial position exit at the specified target price.
        
        Args:
            position: Position to scale out of
            target_price: Target price that was hit
        """
        if not hasattr(position, "id") or position.id is None:
            return
            
        # Determine which target number this is
        target_idx = len(self.scale_out_percentages) - len(self.position_targets.get(position.id, []))
        target_pct = self.scale_out_percentages[target_idx - 1] / 100  # Convert to decimal
        
        # Calculate portion to close
        close_qty = position.quantity * target_pct
        
        logger.info(f"Scale-out target {target_idx} hit at {target_price:.2f} for {position.direction} position {position.id} "
                   f"(closing {target_pct:.0%})")
        
        # In a real system, you would execute a partial close here
        # For simulation, we'll just log the scale-out
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible this strategy is with the current market regime.
        
        Breakout strategies work best during volatility expansion phases and poorly
        during choppy, range-bound markets.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            "trending": 0.75,         # Good in established trends (continuation breakouts)
            "ranging": 0.30,          # Poor in ranging markets (false breakouts)
            "volatile": 0.60,         # Moderate in volatile markets (needs confirmation)
            "calm": 0.20,             # Poor in very calm markets (no momentum)
            "breakout": 0.95,         # Excellent during actual breakouts
            "high_volume": 0.85,      # Very good during high volume periods
            "low_volume": 0.25,       # Poor during low volume (false breakouts likely)
            "high_liquidity": 0.80,   # Very good in high liquidity markets
            "low_liquidity": 0.40,    # Moderate in low liquidity (slippage risk)
            "building_momentum": 0.90, # Excellent when momentum is building
        }
        
        return compatibility_map.get(market_regime, 0.60)  # Default compatibility
