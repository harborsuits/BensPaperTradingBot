#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto RSI Strategy

This strategy uses the Relative Strength Index (RSI) to identify overbought 
and oversold conditions in cryptocurrency markets. It can generate both mean 
reversion and trend continuation signals based on RSI levels and divergences.

Key characteristics:
- Mean reversion when RSI reaches extreme levels
- Trend confirmation when RSI shows momentum
- Divergence detection for potential reversals
- Additional filters to reduce false signals
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
    name="CryptoRSIStrategy",
    market_type="crypto",
    description="RSI-based strategy for crypto markets to identify overbought/oversold conditions and divergences",
    timeframes=["M5", "M15", "H1", "H4", "D1"],
    parameters={
        # RSI parameters
        "rsi_period": {"type": "int", "default": 14, "min": 7, "max": 30},
        "overbought_threshold": {"type": "int", "default": 70, "min": 60, "max": 90},
        "oversold_threshold": {"type": "int", "default": 30, "min": 10, "max": 40},
        "entry_threshold": {"type": "int", "default": 5, "min": 3, "max": 10},
        "exit_threshold": {"type": "int", "default": 5, "min": 3, "max": 10},
        
        # Divergence detection
        "detect_divergence": {"type": "bool", "default": True},
        "divergence_lookback": {"type": "int", "default": 10, "min": 5, "max": 20},
        "divergence_threshold": {"type": "float", "default": 0.05, "min": 0.01, "max": 0.1},
        
        # Signal filtering
        "use_ma_filter": {"type": "bool", "default": True},
        "ma_period": {"type": "int", "default": 50, "min": 20, "max": 200},
        "use_volume_filter": {"type": "bool", "default": True},
        "volume_ma_period": {"type": "int", "default": 20, "min": 10, "max": 50},
        
        # Strategic approach
        "strategy_mode": {"type": "str", "default": "mean_reversion", "enum": ["mean_reversion", "trend_following", "both"]},
        "use_rsi_trend": {"type": "bool", "default": True},
        "rsi_trend_period": {"type": "int", "default": 5, "min": 3, "max": 10},
        
        # Trading parameters
        "risk_per_trade": {"type": "float", "default": 0.01, "min": 0.005, "max": 0.02},
        "max_open_positions": {"type": "int", "default": 2, "min": 1, "max": 5},
        "stop_loss_atr_multiple": {"type": "float", "default": 1.5, "min": 0.5, "max": 3.0},
        "take_profit_atr_multiple": {"type": "float", "default": 2.0, "min": 1.0, "max": 5.0},
        "atr_period": {"type": "int", "default": 14, "min": 7, "max": 21},
    }
)
class CryptoRSIStrategy(CryptoBaseStrategy):
    """
    A Relative Strength Index (RSI) based strategy for cryptocurrency markets.
    
    This strategy:
    1. Uses RSI to identify overbought and oversold conditions
    2. Can detect RSI divergences for potential reversals
    3. Optional moving average filter for trend direction
    4. Works as either mean-reversion or trend-following based on configuration
    5. Uses ATR for position sizing and stop placement
    """
    
    def __init__(self, session: CryptoSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """Initialize the crypto RSI strategy."""
        super().__init__(session, data_pipeline, parameters)
        
        # Strategy-specific state
        self.rsi_history = []
        self.price_history = []
        self.divergence_signals = {
            "bullish": False,
            "bearish": False,
            "last_bullish": None,
            "last_bearish": None,
        }
        
        # Configure trading mode
        self.strategy_mode = self.parameters["strategy_mode"]
        
        logger.info(f"Initialized crypto RSI strategy with period {self.parameters['rsi_period']}, "
                   f"mode: {self.strategy_mode}, "
                   f"thresholds: {self.parameters['oversold_threshold']}/{self.parameters['overbought_threshold']}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate RSI and additional indicators for trading decisions.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        if data.empty or len(data) < self.parameters["rsi_period"]:
            return indicators
        
        # Calculate RSI
        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.parameters["rsi_period"]).mean()
        avg_loss = loss.rolling(window=self.parameters["rsi_period"]).mean()
        
        # Handle division by zero
        rs = pd.Series(np.where(avg_loss == 0, 100, avg_gain / avg_loss), index=avg_gain.index)
        indicators["rsi"] = 100 - (100 / (1 + rs))
        
        # Calculate RSI trend if enabled
        if self.parameters["use_rsi_trend"]:
            rsi_trend_period = self.parameters["rsi_trend_period"]
            indicators["rsi_ma"] = indicators["rsi"].rolling(window=rsi_trend_period).mean()
            indicators["rsi_trend"] = indicators["rsi"] > indicators["rsi_ma"]
        
        # Moving average filter if enabled
        if self.parameters["use_ma_filter"]:
            ma_period = self.parameters["ma_period"]
            indicators["price_ma"] = data["close"].rolling(window=ma_period).mean()
            indicators["above_ma"] = data["close"] > indicators["price_ma"]
        
        # Volume filter if enabled
        if self.parameters["use_volume_filter"]:
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
        if self.parameters["detect_divergence"]:
            self.rsi_history = indicators["rsi"].iloc[-self.parameters["divergence_lookback"]:].tolist()
            self.price_history = data["close"].iloc[-self.parameters["divergence_lookback"]:].tolist()
            
            # Check for divergences
            if len(self.rsi_history) >= self.parameters["divergence_lookback"]:
                self._detect_divergences(data, indicators)
        
        return indicators
    
    def _detect_divergences(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> None:
        """
        Detect RSI divergences with price.
        
        A bullish divergence occurs when price makes lower lows but RSI makes higher lows.
        A bearish divergence occurs when price makes higher highs but RSI makes lower highs.
        
        Args:
            data: Market data DataFrame
            indicators: Dictionary of pre-calculated indicators
        """
        # Reset divergence signals
        indicators["bullish_divergence"] = False
        indicators["bearish_divergence"] = False
        
        if len(self.price_history) < self.parameters["divergence_lookback"]:
            return
            
        # Get local highs and lows
        lookback = self.parameters["divergence_lookback"]
        price_series = pd.Series(self.price_history[-lookback:])
        rsi_series = pd.Series(self.rsi_history[-lookback:])
        
        # Simple method: compare start, middle, and end points
        mid_point = lookback // 2
        
        # Bullish divergence (lower price lows but higher RSI lows)
        if (price_series.iloc[-1] < price_series.iloc[0]) and (rsi_series.iloc[-1] > rsi_series.iloc[0]):
            diff_pct = abs((price_series.iloc[-1] / price_series.iloc[0]) - 1)
            if diff_pct > self.parameters["divergence_threshold"]:
                indicators["bullish_divergence"] = True
                self.divergence_signals["bullish"] = True
                self.divergence_signals["last_bullish"] = data.index[-1]
                logger.info(f"Detected bullish RSI divergence (price: {price_series.iloc[-1]:.2f}, RSI: {rsi_series.iloc[-1]:.1f})")
        
        # Bearish divergence (higher price highs but lower RSI highs)
        if (price_series.iloc[-1] > price_series.iloc[0]) and (rsi_series.iloc[-1] < rsi_series.iloc[0]):
            diff_pct = abs((price_series.iloc[-1] / price_series.iloc[0]) - 1)
            if diff_pct > self.parameters["divergence_threshold"]:
                indicators["bearish_divergence"] = True
                self.divergence_signals["bearish"] = True
                self.divergence_signals["last_bearish"] = data.index[-1]
                logger.info(f"Detected bearish RSI divergence (price: {price_series.iloc[-1]:.2f}, RSI: {rsi_series.iloc[-1]:.1f})")
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on RSI levels and conditions.
        
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
        
        if not indicators or data.empty or "rsi" not in indicators:
            return signals
        
        # Get current RSI and configuration values
        current_rsi = indicators["rsi"].iloc[-1]
        overbought = self.parameters["overbought_threshold"]
        oversold = self.parameters["oversold_threshold"]
        entry_threshold = self.parameters["entry_threshold"]
        exit_threshold = self.parameters["exit_threshold"]
        
        # Get additional filter conditions if available
        trend_filter = True
        if self.parameters["use_ma_filter"] and "above_ma" in indicators:
            trend_filter = indicators["above_ma"].iloc[-1]  # Above MA is bullish
            
        volume_filter = True
        if self.parameters["use_volume_filter"] and "high_volume" in indicators:
            volume_filter = indicators["high_volume"].iloc[-1]  # High volume is good
        
        # Get divergence signals if available
        bullish_divergence = indicators.get("bullish_divergence", False)
        bearish_divergence = indicators.get("bearish_divergence", False)
        
        # Signal strength based on RSI extremity
        signal_strength = 0.0
        if current_rsi <= oversold:
            # Stronger signal the lower the RSI gets below oversold
            signal_strength = min(1.0, (oversold - current_rsi) / 30.0)
        elif current_rsi >= overbought:
            # Stronger signal the higher the RSI gets above overbought
            signal_strength = min(1.0, (current_rsi - overbought) / 30.0)
            
        signals["signal_strength"] = signal_strength
        
        # Mean Reversion Mode
        if self.strategy_mode in ["mean_reversion", "both"]:
            # Long signal when RSI exits oversold zone (from below)
            if current_rsi <= oversold + entry_threshold and indicators["rsi"].iloc[-2] < oversold:
                if volume_filter:
                    signals["long_entry"] = True
                    logger.info(f"RSI mean reversion LONG signal: RSI {current_rsi:.1f} exiting oversold zone")
                
            # Short signal when RSI exits overbought zone (from above)
            if current_rsi >= overbought - entry_threshold and indicators["rsi"].iloc[-2] > overbought:
                if volume_filter:
                    signals["short_entry"] = True
                    logger.info(f"RSI mean reversion SHORT signal: RSI {current_rsi:.1f} exiting overbought zone")
                    
            # Exit long when RSI reaches overbought
            for position in self.positions:
                if position.direction == "long" and current_rsi >= overbought - exit_threshold:
                    signals["long_exit"] = True
                    logger.info(f"RSI mean reversion exit LONG signal: RSI {current_rsi:.1f} approaching overbought")
                    
                # Exit short when RSI reaches oversold
                elif position.direction == "short" and current_rsi <= oversold + exit_threshold:
                    signals["short_exit"] = True
                    logger.info(f"RSI mean reversion exit SHORT signal: RSI {current_rsi:.1f} approaching oversold")
        
        # Trend Following Mode
        if self.strategy_mode in ["trend_following", "both"]:
            rsi_trend = indicators.get("rsi_trend", True) if self.parameters["use_rsi_trend"] else True
            
            # Long signal on bullish divergence or RSI trend
            if bullish_divergence and trend_filter:
                signals["long_entry"] = True
                signals["signal_strength"] = max(signals["signal_strength"], 0.7)  # Divergence is a strong signal
                logger.info(f"RSI trend following LONG signal: Bullish divergence detected")
                
            # Short signal on bearish divergence or RSI trend
            if bearish_divergence and not trend_filter:
                signals["short_entry"] = True
                signals["signal_strength"] = max(signals["signal_strength"], 0.7)  # Divergence is a strong signal
                logger.info(f"RSI trend following SHORT signal: Bearish divergence detected")
                
            # Exit signals based on RSI momentum change
            for position in self.positions:
                if position.direction == "long" and not rsi_trend and current_rsi < indicators["rsi"].iloc[-2]:
                    signals["long_exit"] = True
                    logger.info(f"RSI trend following exit LONG signal: RSI momentum weakened")
                    
                elif position.direction == "short" and rsi_trend and current_rsi > indicators["rsi"].iloc[-2]:
                    signals["short_exit"] = True
                    logger.info(f"RSI trend following exit SHORT signal: RSI momentum strengthened")
        
        # Calculate stop loss and take profit levels if ATR is available
        current_price = data["close"].iloc[-1]
        
        if "atr" in indicators:
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
        if "atr" in indicators:
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
                
                logger.info(f"RSI position size: {position_size_crypto} {self.session.symbol.split('-')[0]} "
                          f"(signal strength: {signal_strength:.2f})")
                          
                return position_size_crypto
        
        # Fallback to default if ATR calculation failed
        return default_position_size
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible this strategy is with the current market regime.
        
        RSI strategies work best in ranging markets but can adapt to other regimes
        based on strategy_mode configuration.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        # Base compatibility map
        base_compatibility = {
            "ranging": 0.90,       # Excellent in ranging markets (mean reversion)
            "trending": 0.60,      # Moderate in trending markets with trend mode
            "volatile": 0.70,      # Good in volatile markets (catching extremes)
            "calm": 0.70,          # Good in calm markets
            "breakout": 0.40,      # Poor during breakouts (false signals)
            "high_volume": 0.75,   # Good in high volume conditions
            "low_volume": 0.60,    # Moderate in low volume
            "high_liquidity": 0.80, # Very good in high liquidity
            "low_liquidity": 0.50,  # Moderate in low liquidity
        }
        
        # Adjust compatibility based on strategy mode
        if self.strategy_mode == "trend_following":
            # Adjust for trend-following RSI configuration
            trend_adjustments = {
                "ranging": -0.3,    # Worse in ranging markets
                "trending": +0.2,   # Better in trending markets
                "breakout": +0.2,   # Better for breakouts
            }
            
            for regime, adjustment in trend_adjustments.items():
                if regime == market_regime and regime in base_compatibility:
                    return max(0.1, min(1.0, base_compatibility[regime] + adjustment))
        
        return base_compatibility.get(market_regime, 0.65)  # Default compatibility
