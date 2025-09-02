#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Moving Average Crossover Strategy

This strategy uses moving average crossovers to identify trend changes and
generate trading signals. It can be configured with various types of moving
averages (SMA, EMA, WMA) and different timeframes for the faster and slower MAs.

Key characteristics:
- Trend-following strategy
- Multiple MA types and periods
- Simple but effective entry and exit signals
- Can be enhanced with volume confirmation
- Adaptable to different market conditions
"""

import logging
from typing import Dict, Any, List, Optional
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
    name="CryptoMAStrategy",
    market_type="crypto",
    description="Moving average crossover strategy for crypto markets using configurable MA types and periods",
    timeframes=["M15", "H1", "H4", "D1"],  # MA crossovers typically work best on higher timeframes
    parameters={
        # Moving average parameters
        "fast_ma_type": {"type": "str", "default": "EMA", "enum": ["SMA", "EMA", "WMA"]},
        "slow_ma_type": {"type": "str", "default": "EMA", "enum": ["SMA", "EMA", "WMA"]},
        "fast_ma_period": {"type": "int", "default": 20, "min": 5, "max": 50},
        "slow_ma_period": {"type": "int", "default": 50, "min": 20, "max": 200},
        "signal_ma_period": {"type": "int", "default": 9, "min": 3, "max": 20},
        
        # Additional confirmation indicators
        "use_volume_confirmation": {"type": "bool", "default": True},
        "volume_ma_period": {"type": "int", "default": 20, "min": 5, "max": 50},
        "use_atr_for_stops": {"type": "bool", "default": True},
        "atr_period": {"type": "int", "default": 14, "min": 7, "max": 21},
        
        # Trade execution parameters
        "stop_loss_atr_multiple": {"type": "float", "default": 1.5, "min": 0.5, "max": 3.0},
        "take_profit_atr_multiple": {"type": "float", "default": 3.0, "min": 1.0, "max": 6.0},
        "trail_stop_enabled": {"type": "bool", "default": True},
        "trail_stop_activation_pct": {"type": "float", "default": 0.01, "min": 0.005, "max": 0.03},
        
        # Risk management
        "risk_per_trade": {"type": "float", "default": 0.01, "min": 0.005, "max": 0.02},
        "max_open_positions": {"type": "int", "default": 3, "min": 1, "max": 5},
        "require_trend_confirmation": {"type": "bool", "default": True},
        "min_crossover_strength": {"type": "float", "default": 0.001, "min": 0.0001, "max": 0.01},
    }
)
class CryptoMAStrategy(CryptoBaseStrategy):
    """
    A moving average crossover strategy for cryptocurrency markets.
    
    This strategy:
    1. Uses configurable moving averages (SMA, EMA, WMA) for trend identification
    2. Generates signals based on MA crossovers (fast MA crossing above/below slow MA)
    3. Can include volume confirmation to filter signals
    4. Uses ATR for position sizing and stop placement
    5. Offers trailing stop capabilities for profit maximization
    """
    
    def __init__(self, session: CryptoSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """Initialize the crypto moving average crossover strategy."""
        super().__init__(session, data_pipeline, parameters)
        
        # Strategy-specific state
        self.ma_type_map = {
            "SMA": self._calculate_sma,
            "EMA": self._calculate_ema,
            "WMA": self._calculate_wma
        }
        
        self.current_trend = "neutral"
        self.previous_fast_ma = None
        self.previous_slow_ma = None
        self.crossover_detected = False
        self.crossover_time = None
        self.crossover_price = None
        self.trailing_stops = {}  # Position ID -> trailing stop price
        
        # Get MA configuration
        self.fast_ma_type = self.parameters["fast_ma_type"]
        self.slow_ma_type = self.parameters["slow_ma_type"]
        self.fast_ma_period = self.parameters["fast_ma_period"]
        self.slow_ma_period = self.parameters["slow_ma_period"]
        self.signal_ma_period = self.parameters["signal_ma_period"]
        
        logger.info(f"Initialized crypto MA crossover strategy: {self.fast_ma_type}({self.fast_ma_period}) x {self.slow_ma_type}({self.slow_ma_period})")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate moving averages and related indicators.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        if data.empty or len(data) < self.slow_ma_period:
            return indicators
        
        # Calculate fast and slow moving averages
        fast_ma_func = self.ma_type_map.get(self.fast_ma_type, self._calculate_sma)
        slow_ma_func = self.ma_type_map.get(self.slow_ma_type, self._calculate_sma)
        
        indicators["fast_ma"] = fast_ma_func(data["close"], self.fast_ma_period)
        indicators["slow_ma"] = slow_ma_func(data["close"], self.slow_ma_period)
        
        # Calculate crossover and trend indicators
        if len(indicators["fast_ma"]) > 1 and len(indicators["slow_ma"]) > 1:
            # Current values
            current_fast = indicators["fast_ma"].iloc[-1]
            current_slow = indicators["slow_ma"].iloc[-1]
            
            # Previous values
            prev_fast = indicators["fast_ma"].iloc[-2]
            prev_slow = indicators["slow_ma"].iloc[-2]
            
            # Detect crossover
            crossover_up = prev_fast <= prev_slow and current_fast > current_slow
            crossover_down = prev_fast >= prev_slow and current_fast < current_slow
            
            indicators["crossover_up"] = crossover_up
            indicators["crossover_down"] = crossover_down
            indicators["crossover"] = crossover_up or crossover_down
            
            # Calculate crossover strength (how far the MAs crossed)
            indicators["crossover_strength"] = abs(current_fast - current_slow) / current_slow
            
            # Determine current trend
            if current_fast > current_slow:
                indicators["trend"] = "bullish"
            elif current_fast < current_slow:
                indicators["trend"] = "bearish"
            else:
                indicators["trend"] = "neutral"
                
            # Store previous values for next iteration
            self.previous_fast_ma = current_fast
            self.previous_slow_ma = current_slow
            
            # Update trend state
            self.current_trend = indicators["trend"]
        
        # Calculate ATR for stop placement if enabled
        if self.parameters["use_atr_for_stops"]:
            atr_period = self.parameters["atr_period"]
            high_low = data["high"] - data["low"]
            high_close = abs(data["high"] - data["close"].shift())
            low_close = abs(data["low"] - data["close"].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            indicators["atr"] = true_range.rolling(window=atr_period).mean()
        
        # Volume confirmation if enabled
        if self.parameters["use_volume_confirmation"]:
            volume_ma_period = self.parameters["volume_ma_period"]
            indicators["volume_ma"] = data["volume"].rolling(window=volume_ma_period).mean()
            indicators["volume_trend"] = data["volume"] > indicators["volume_ma"]
        
        return indicators
    
    def _calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return data.rolling(window=period).mean()
    
    def _calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data.ewm(span=period, adjust=False).mean()
    
    def _calculate_wma(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Weighted Moving Average."""
        weights = np.arange(1, period + 1)
        return data.rolling(period).apply(lambda x: np.sum(weights * x) / weights.sum(), raw=True)
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on moving average crossovers.
        
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
        
        if not indicators or data.empty:
            return signals
        
        # Check for crossovers
        crossover_up = indicators.get("crossover_up", False)
        crossover_down = indicators.get("crossover_down", False)
        trend = indicators.get("trend", "neutral")
        
        # Volume confirmation if enabled
        volume_confirmed = True
        if self.parameters["use_volume_confirmation"] and "volume_trend" in indicators:
            volume_confirmed = indicators["volume_trend"].iloc[-1]
        
        # Check crossover strength
        crossover_strength = indicators.get("crossover_strength", 0)
        min_strength = self.parameters["min_crossover_strength"]
        strength_confirmed = crossover_strength >= min_strength
        
        # Long entry signal
        if crossover_up and volume_confirmed and strength_confirmed:
            signals["long_entry"] = True
            signals["signal_strength"] = min(1.0, crossover_strength / (min_strength * 2))
            logger.info(f"MA crossover LONG signal: {self.fast_ma_type}({self.fast_ma_period}) crossed above {self.slow_ma_type}({self.slow_ma_period})")
            
            # Store crossover information
            self.crossover_detected = True
            self.crossover_time = data.index[-1]
            self.crossover_price = data["close"].iloc[-1]
        
        # Short entry signal
        elif crossover_down and volume_confirmed and strength_confirmed:
            signals["short_entry"] = True
            signals["signal_strength"] = min(1.0, crossover_strength / (min_strength * 2))
            logger.info(f"MA crossover SHORT signal: {self.fast_ma_type}({self.fast_ma_period}) crossed below {self.slow_ma_type}({self.slow_ma_period})")
            
            # Store crossover information
            self.crossover_detected = True
            self.crossover_time = data.index[-1]
            self.crossover_price = data["close"].iloc[-1]
        
        # Exit signals - exit when trend changes (opposite crossover)
        for position in self.positions:
            if position.direction == "long" and crossover_down:
                signals["long_exit"] = True
                logger.info(f"MA crossover exit LONG signal: {self.fast_ma_type}({self.fast_ma_period}) crossed below {self.slow_ma_type}({self.slow_ma_period})")
            
            elif position.direction == "short" and crossover_up:
                signals["short_exit"] = True
                logger.info(f"MA crossover exit SHORT signal: {self.fast_ma_type}({self.fast_ma_period}) crossed above {self.slow_ma_type}({self.slow_ma_period})")
        
        # Calculate stop loss and take profit levels if ATR is available
        current_price = data["close"].iloc[-1]
        
        if self.parameters["use_atr_for_stops"] and "atr" in indicators:
            atr = indicators["atr"].iloc[-1]
            stop_multiple = self.parameters["stop_loss_atr_multiple"]
            tp_multiple = self.parameters["take_profit_atr_multiple"]
            
            if signals["long_entry"]:
                signals["stop_loss"] = current_price - (atr * stop_multiple)
                signals["take_profit"] = current_price + (atr * tp_multiple)
            
            elif signals["short_entry"]:
                signals["stop_loss"] = current_price + (atr * stop_multiple)
                signals["take_profit"] = current_price - (atr * tp_multiple)
        
        return signals
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate position size based on risk parameters and ATR.
        
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
        
        # Risk per trade
        risk_percentage = self.parameters["risk_per_trade"]
        risk_amount = account_balance * risk_percentage
        
        # Current price
        current_price = data["close"].iloc[-1]
        
        # Set default position size (fixed percentage of account)
        default_position_size = (account_balance * 0.1) / current_price
        
        # If ATR is available, use it for position sizing based on stop distance
        if self.parameters["use_atr_for_stops"] and "atr" in indicators:
            atr = indicators["atr"].iloc[-1]
            stop_distance = atr * self.parameters["stop_loss_atr_multiple"]
            
            # Calculate position size based on risked amount and stop distance
            if stop_distance > 0:
                position_size_base = risk_amount / (stop_distance / current_price)
                position_size_crypto = position_size_base / current_price
                
                # Apply signal strength adjustment
                signal_strength = self.signals.get("signal_strength", 0.5)
                position_size_crypto *= max(0.5, signal_strength)
                
                # Ensure minimum trade size
                min_trade_size = self.session.min_trade_size
                position_size_crypto = max(position_size_crypto, min_trade_size)
                
                # Apply precision appropriate for the asset
                decimals = 8 if self.session.symbol.startswith("BTC") else 6
                position_size_crypto = round(position_size_crypto, decimals)
                
                return position_size_crypto
        
        # Fallback to default sizing if ATR approach fails
        return default_position_size
    
    def _on_timeframe_completed(self, event: Event) -> None:
        """
        Handle timeframe completed events.
        
        For MA strategies, we process signals at the close of each candle.
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
        if self.parameters["trail_stop_enabled"]:
            self._update_trailing_stops()
    
    def _update_trailing_stops(self) -> None:
        """Update trailing stops for open positions based on current price."""
        if not self.positions or not self.market_data.empty:
            return
        
        current_price = self.market_data["close"].iloc[-1]
        activation_pct = self.parameters["trail_stop_activation_pct"]
        
        # Process each open position
        for position in self.positions:
            # Skip if position has no ID
            if not hasattr(position, "id") or position.id is None:
                continue
                
            # Initialize trailing stop if not already set
            if position.id not in self.trailing_stops:
                if position.direction == "long":
                    self.trailing_stops[position.id] = position.entry_price * (1 - activation_pct * 2)
                else:  # short
                    self.trailing_stops[position.id] = position.entry_price * (1 + activation_pct * 2)
            
            # Update trailing stop if price moved favorably
            if position.direction == "long":
                # Calculate activation threshold
                activation_threshold = position.entry_price * (1 + activation_pct)
                
                # Only trail if price has moved beyond activation threshold
                if current_price > activation_threshold:
                    # Calculate new stop price
                    new_stop = current_price * (1 - activation_pct)
                    
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
                    
                    # Update only if new stop is lower than current stop
                    if new_stop < self.trailing_stops[position.id]:
                        self.trailing_stops[position.id] = new_stop
                        logger.info(f"Updated trailing stop for short position {position.id}: {new_stop:.2f}")
    
    def _on_market_data_updated(self, event: Event) -> None:
        """
        Handle market data updated events.
        
        For MA strategies, we don't need to act on every tick, but we use this to
        check trailing stops.
        """
        super()._on_market_data_updated(event)
        
        # Skip if no open positions with trailing stops
        if not self.positions or not self.trailing_stops:
            return
        
        # Check if we need to exit any positions based on trailing stops
        current_price = event.data.get('close', None)
        if current_price is None:
            return
        
        # Check each position against its trailing stop
        for position in self.positions:
            if not hasattr(position, "id") or position.id not in self.trailing_stops:
                continue
                
            trailing_stop = self.trailing_stops[position.id]
            
            # Check if trailing stop is hit
            if (position.direction == "long" and current_price < trailing_stop) or \
               (position.direction == "short" and current_price > trailing_stop):
                logger.info(f"Trailing stop triggered for {position.direction} position {position.id} at {trailing_stop:.2f}")
                self._close_position(position.id)
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible this strategy is with the current market regime.
        
        Moving average strategies work best in trending markets and struggle in 
        choppy or ranging markets.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            "trending": 0.90,        # Excellent in trending markets
            "ranging": 0.40,         # Poor in ranging markets (whipsaws)
            "volatile": 0.60,        # Moderate in volatile markets
            "calm": 0.70,            # Good in calm markets if trending
            "breakout": 0.75,        # Good during breakouts (may catch the start of trends)
            "high_volume": 0.80,     # Very good during high volume periods
            "low_volume": 0.60,      # Moderate during low volume
            "high_liquidity": 0.80,  # Very good in high liquidity markets
            "low_liquidity": 0.50,   # Moderate in low liquidity markets
        }
        
        return compatibility_map.get(market_regime, 0.65)  # Default compatibility
