#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto MACD Strategy

This strategy uses the Moving Average Convergence Divergence (MACD) indicator to
identify momentum shifts and generate trading signals. It can be configured for 
different timeframes and sensitivity levels.

Key characteristics:
- Trend-momentum indicator combining moving averages
- Signal line crossovers for entry/exit
- Histogram divergence for early signals
- Can be used in both trending and ranging markets
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
    name="CryptoMACDStrategy",
    market_type="crypto",
    description="MACD-based strategy for crypto markets to identify momentum shifts and trend direction",
    timeframes=["M15", "H1", "H4", "D1"],
    parameters={
        # MACD parameters
        "fast_ema_period": {"type": "int", "default": 12, "min": 5, "max": 25},
        "slow_ema_period": {"type": "int", "default": 26, "min": 10, "max": 50},
        "signal_period": {"type": "int", "default": 9, "min": 3, "max": 20},
        
        # Signal generation
        "use_histogram": {"type": "bool", "default": True},
        "histogram_threshold": {"type": "float", "default": 0.0, "min": 0.0, "max": 0.5},
        "require_zero_line_cross": {"type": "bool", "default": False},
        "use_divergence": {"type": "bool", "default": True},
        "divergence_lookback": {"type": "int", "default": 10, "min": 5, "max": 20},
        
        # Confirmation filters
        "use_price_filter": {"type": "bool", "default": True},
        "ma_period": {"type": "int", "default": 50, "min": 20, "max": 200},
        "use_volume_confirmation": {"type": "bool", "default": True},
        "volume_ma_period": {"type": "int", "default": 20, "min": 5, "max": 50},
        
        # Execution parameters
        "signal_quality_threshold": {"type": "float", "default": 0.2, "min": 0.1, "max": 0.9},
        "exit_on_opposite_signal": {"type": "bool", "default": True},
        "trailing_exit": {"type": "bool", "default": True},
        
        # Risk management
        "risk_per_trade": {"type": "float", "default": 0.01, "min": 0.005, "max": 0.02},
        "max_open_positions": {"type": "int", "default": 2, "min": 1, "max": 5},
        "use_atr_for_stops": {"type": "bool", "default": True},
        "atr_period": {"type": "int", "default": 14, "min": 7, "max": 21},
        "stop_loss_atr_multiple": {"type": "float", "default": 1.5, "min": 0.5, "max": 3.0},
        "take_profit_atr_multiple": {"type": "float", "default": 3.0, "min": 1.0, "max": 6.0},
    }
)
class CryptoMACDStrategy(CryptoBaseStrategy):
    """
    A strategy based on the Moving Average Convergence Divergence (MACD) indicator.
    
    This strategy:
    1. Uses MACD for trend and momentum identification
    2. Generates signals on MACD line crossovers and histogram reversals
    3. Can detect and trade MACD divergences with price
    4. Includes volume confirmation for higher quality signals
    5. Uses ATR for position sizing and stop placement
    """
    
    def __init__(self, session: CryptoSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """Initialize the crypto MACD strategy."""
        super().__init__(session, data_pipeline, parameters)
        
        # Strategy-specific state
        self.macd_history = []
        self.histogram_history = []
        self.price_history = []
        self.divergence_signals = {
            "bullish": False,
            "bearish": False,
            "last_bullish": None,
            "last_bearish": None,
        }
        self.last_cross_above = None
        self.last_cross_below = None
        self.trailing_stops = {}
        
        # Get MACD configuration
        self.fast_period = self.parameters["fast_ema_period"]
        self.slow_period = self.parameters["slow_ema_period"]
        self.signal_period = self.parameters["signal_period"]
        
        logger.info(f"Initialized crypto MACD strategy with periods {self.fast_period}/{self.slow_period}/{self.signal_period}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate MACD and additional indicators.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        if data.empty or len(data) < self.parameters["slow_ema_period"] + self.parameters["signal_period"]:
            return indicators
        
        # Calculate MACD components
        fast_ema = data["close"].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = data["close"].ewm(span=self.slow_period, adjust=False).mean()
        
        # MACD Line = Fast EMA - Slow EMA
        macd_line = fast_ema - slow_ema
        
        # Signal Line = EMA of MACD Line
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        
        # MACD Histogram = MACD Line - Signal Line
        histogram = macd_line - signal_line
        
        # Store in indicators
        indicators["macd_line"] = macd_line
        indicators["signal_line"] = signal_line
        indicators["histogram"] = histogram
        
        # Calculate zero-line crosses
        indicators["above_zero"] = macd_line > 0
        indicators["crossed_above_zero"] = (macd_line > 0) & (macd_line.shift(1) <= 0)
        indicators["crossed_below_zero"] = (macd_line < 0) & (macd_line.shift(1) >= 0)
        
        # Calculate signal line crosses
        indicators["above_signal"] = macd_line > signal_line
        indicators["crossed_above_signal"] = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        indicators["crossed_below_signal"] = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        
        # Calculate histogram reversals
        indicators["histogram_positive"] = histogram > 0
        indicators["histogram_increasing"] = histogram > histogram.shift(1)
        indicators["histogram_decreasing"] = histogram < histogram.shift(1)
        indicators["histogram_reversal_up"] = (histogram > histogram.shift(1)) & (histogram.shift(1) <= histogram.shift(2))
        indicators["histogram_reversal_down"] = (histogram < histogram.shift(1)) & (histogram.shift(1) >= histogram.shift(2))
        
        # Price filter if enabled
        if self.parameters["use_price_filter"]:
            ma_period = self.parameters["ma_period"]
            indicators["price_ma"] = data["close"].rolling(window=ma_period).mean()
            indicators["above_ma"] = data["close"] > indicators["price_ma"]
        
        # Volume confirmation if enabled
        if self.parameters["use_volume_confirmation"]:
            vol_ma_period = self.parameters["volume_ma_period"]
            indicators["volume_ma"] = data["volume"].rolling(window=vol_ma_period).mean()
            indicators["high_volume"] = data["volume"] > indicators["volume_ma"]
        
        # Calculate ATR for position sizing if needed
        atr_period = self.parameters["atr_period"]
        high_low = data["high"] - data["low"]
        high_close = abs(data["high"] - data["close"].shift())
        low_close = abs(data["low"] - data["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators["atr"] = true_range.rolling(window=atr_period).mean()
        
        # Store history for divergence detection
        if self.parameters["use_divergence"]:
            lookback = self.parameters["divergence_lookback"]
            self.macd_history = macd_line.iloc[-lookback:].tolist()
            self.histogram_history = histogram.iloc[-lookback:].tolist()
            self.price_history = data["close"].iloc[-lookback:].tolist()
            
            # Detect MACD divergences
            if len(self.macd_history) >= lookback and len(self.price_history) >= lookback:
                self._detect_divergences(data, indicators)
        
        return indicators
    
    def _detect_divergences(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> None:
        """
        Detect MACD divergences with price.
        
        A bullish divergence occurs when price makes a lower low but MACD makes a higher low.
        A bearish divergence occurs when price makes a higher high but MACD makes a lower high.
        
        Args:
            data: Market data DataFrame
            indicators: Dictionary of pre-calculated indicators
        """
        # Reset divergence signals
        indicators["bullish_divergence"] = False
        indicators["bearish_divergence"] = False
        
        lookback = self.parameters["divergence_lookback"]
        if len(self.price_history) < lookback or len(self.macd_history) < lookback:
            return
            
        # Create series for easier analysis
        price_series = pd.Series(self.price_history[-lookback:])
        macd_series = pd.Series(self.macd_history[-lookback:])
        
        # Simple approach: find lowest points and compare
        # For bullish divergence: price making lower lows but MACD making higher lows
        price_min_idx = price_series.idxmin()
        macd_min_idx = macd_series.idxmin()
        
        # For bearish divergence: price making higher highs but MACD making lower highs
        price_max_idx = price_series.idxmax()
        macd_max_idx = macd_series.idxmax()
        
        # Check for bullish divergence (focus on recent data)
        recent_window = min(5, lookback // 2)
        if (price_min_idx > (lookback - recent_window) and  # Price low is recent
            macd_series.iloc[price_min_idx] > macd_series.iloc[macd_min_idx]):  # MACD higher at price low
            indicators["bullish_divergence"] = True
            self.divergence_signals["bullish"] = True
            self.divergence_signals["last_bullish"] = data.index[-1]
            logger.info(f"Detected bullish MACD divergence")
        
        # Check for bearish divergence (focus on recent data)
        if (price_max_idx > (lookback - recent_window) and  # Price high is recent
            macd_series.iloc[price_max_idx] < macd_series.iloc[macd_max_idx]):  # MACD lower at price high
            indicators["bearish_divergence"] = True
            self.divergence_signals["bearish"] = True
            self.divergence_signals["last_bearish"] = data.index[-1]
            logger.info(f"Detected bearish MACD divergence")
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on MACD indicator values.
        
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
        
        if not indicators or data.empty or "macd_line" not in indicators:
            return signals
        
        # Get current values
        current_macd = indicators["macd_line"].iloc[-1]
        current_signal = indicators["signal_line"].iloc[-1]
        current_histogram = indicators["histogram"].iloc[-1]
        
        # Get confirmation filters
        price_filter = True
        if self.parameters["use_price_filter"] and "above_ma" in indicators:
            price_filter = indicators["above_ma"].iloc[-1]
            
        volume_filter = True
        if self.parameters["use_volume_confirmation"] and "high_volume" in indicators:
            volume_filter = indicators["high_volume"].iloc[-1]
        
        # Check for signal line crossovers
        crossover_up = indicators["crossed_above_signal"].iloc[-1]
        crossover_down = indicators["crossed_below_signal"].iloc[-1]
        
        # Check for histogram reversals
        histogram_reversal_up = indicators["histogram_reversal_up"].iloc[-1]
        histogram_reversal_down = indicators["histogram_reversal_down"].iloc[-1]
        
        # Check for zero line requirement
        zero_line_check = True
        if self.parameters["require_zero_line_cross"]:
            if crossover_up or histogram_reversal_up:
                zero_line_check = indicators["above_zero"].iloc[-1]
            elif crossover_down or histogram_reversal_down:
                zero_line_check = not indicators["above_zero"].iloc[-1]
        
        # Divergence signals
        bullish_divergence = indicators.get("bullish_divergence", False)
        bearish_divergence = indicators.get("bearish_divergence", False)
        
        # Calculate signal strength based on MACD values
        signal_strength = min(1.0, abs(current_macd - current_signal) / 0.002)
        if signal_strength < self.parameters["signal_quality_threshold"]:
            signal_strength = 0.0
            
        signals["signal_strength"] = signal_strength
        
        # Long entry signals
        if ((crossover_up or (self.parameters["use_histogram"] and histogram_reversal_up)) and 
            zero_line_check and volume_filter and 
            (price_filter or bullish_divergence)):
            
            signals["long_entry"] = True
            
            # Record the crossover for future reference
            self.last_cross_above = data.index[-1]
            
            # Log the signal with details
            if crossover_up:
                logger.info(f"MACD LONG signal: Crossed above signal line (MACD: {current_macd:.4f}, Signal: {current_signal:.4f})")
            elif histogram_reversal_up:
                logger.info(f"MACD LONG signal: Histogram reversal upward (Histogram: {current_histogram:.4f})")
            if bullish_divergence:
                signals["signal_strength"] = max(signals["signal_strength"], 0.7)  # Divergence strengthens signal
                logger.info(f"MACD LONG signal boosted by bullish divergence")
        
        # Short entry signals
        if ((crossover_down or (self.parameters["use_histogram"] and histogram_reversal_down)) and 
            zero_line_check and volume_filter and 
            (not price_filter or bearish_divergence)):
            
            signals["short_entry"] = True
            
            # Record the crossover for future reference
            self.last_cross_below = data.index[-1]
            
            # Log the signal with details
            if crossover_down:
                logger.info(f"MACD SHORT signal: Crossed below signal line (MACD: {current_macd:.4f}, Signal: {current_signal:.4f})")
            elif histogram_reversal_down:
                logger.info(f"MACD SHORT signal: Histogram reversal downward (Histogram: {current_histogram:.4f})")
            if bearish_divergence:
                signals["signal_strength"] = max(signals["signal_strength"], 0.7)  # Divergence strengthens signal
                logger.info(f"MACD SHORT signal boosted by bearish divergence")
        
        # Exit signals
        for position in self.positions:
            # Exit long positions
            if position.direction == "long":
                if self.parameters["exit_on_opposite_signal"] and (crossover_down or histogram_reversal_down):
                    signals["long_exit"] = True
                    logger.info(f"MACD exit LONG signal: Opposite signal detected")
                    
            # Exit short positions
            elif position.direction == "short":
                if self.parameters["exit_on_opposite_signal"] and (crossover_up or histogram_reversal_up):
                    signals["short_exit"] = True
                    logger.info(f"MACD exit SHORT signal: Opposite signal detected")
        
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
        risk_percentage = self.parameters["risk_per_trade"]
        risk_amount = account_balance * risk_percentage
        
        # Current price
        current_price = data["close"].iloc[-1]
        
        # Default position size
        default_position_size = (account_balance * 0.1) / current_price
        
        # ATR-based position sizing
        if self.parameters["use_atr_for_stops"] and "atr" in indicators:
            atr = indicators["atr"].iloc[-1]
            stop_multiple = self.parameters["stop_loss_atr_multiple"]
            stop_distance = atr * stop_multiple
            
            # Calculate position size based on risk amount and stop distance
            if stop_distance > 0:
                position_size_base = risk_amount / stop_distance
                position_size_crypto = position_size_base / current_price
                
                # Adjust position size based on signal strength
                signal_strength = self.signals.get("signal_strength", 0.5)
                position_size_crypto *= max(0.5, signal_strength)
                
                # Apply precision appropriate for the asset
                decimals = 8 if self.session.symbol.startswith("BTC") else 6
                position_size_crypto = round(position_size_crypto, decimals)
                
                # Ensure minimum trade size
                min_trade_size = self.session.min_trade_size
                position_size_crypto = max(position_size_crypto, min_trade_size)
                
                logger.info(f"MACD position size: {position_size_crypto} {self.session.symbol.split('-')[0]} "
                          f"(signal strength: {signal_strength:.2f})")
                          
                return position_size_crypto
        
        # Fallback to default if ATR calculation failed
        return default_position_size
    
    def _on_timeframe_completed(self, event: Event) -> None:
        """
        Handle timeframe completed events.
        
        For MACD strategies, we process signals at the close of each candle.
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
        if self.parameters["trailing_exit"]:
            self._update_trailing_stops()
    
    def _update_trailing_stops(self) -> None:
        """Update trailing stops for open positions based on current indicators."""
        if not self.positions or self.market_data.empty:
            return
        
        current_price = self.market_data["close"].iloc[-1]
        if "atr" not in self.indicators:
            return
            
        atr = self.indicators["atr"].iloc[-1]
        stop_multiple = self.parameters["stop_loss_atr_multiple"] * 0.75  # Tighter trailing stop
        
        # Process each open position
        for position in self.positions:
            if not hasattr(position, "id") or position.id is None:
                continue
                
            # Initialize trailing stop if not already set
            if position.id not in self.trailing_stops:
                if position.direction == "long":
                    self.trailing_stops[position.id] = position.entry_price - (atr * stop_multiple)
                else:  # short
                    self.trailing_stops[position.id] = position.entry_price + (atr * stop_multiple)
            
            # Update trailing stop if price moved favorably
            if position.direction == "long":
                new_stop = current_price - (atr * stop_multiple)
                if new_stop > self.trailing_stops[position.id]:
                    self.trailing_stops[position.id] = new_stop
                    logger.info(f"Updated trailing stop for long position {position.id}: {new_stop:.2f}")
            
            else:  # short position
                new_stop = current_price + (atr * stop_multiple)
                if new_stop < self.trailing_stops[position.id]:
                    self.trailing_stops[position.id] = new_stop
                    logger.info(f"Updated trailing stop for short position {position.id}: {new_stop:.2f}")
    
    def _on_market_data_updated(self, event: Event) -> None:
        """
        Handle market data updated events.
        
        For MACD strategies, we check trailing stops on each price update.
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
        
        MACD strategies work well in trending markets but can also adapt to
        ranging markets by focusing on the histogram.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            "trending": 0.85,        # Very good in trending markets
            "ranging": 0.60,         # Moderate in ranging markets (histogram helps)
            "volatile": 0.70,        # Good in volatile markets
            "calm": 0.65,            # Moderate in calm markets
            "breakout": 0.80,        # Very good during breakouts
            "high_volume": 0.80,     # Very good during high volume periods
            "low_volume": 0.60,      # Moderate during low volume
            "high_liquidity": 0.75,  # Good in high liquidity markets
            "low_liquidity": 0.55,   # Moderate in low liquidity markets
        }
        
        return compatibility_map.get(market_regime, 0.65)  # Default compatibility
