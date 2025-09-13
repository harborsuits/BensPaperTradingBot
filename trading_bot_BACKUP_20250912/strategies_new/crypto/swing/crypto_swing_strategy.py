#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Swing Trading Strategy

This strategy aims to capture medium-term price swings in cryptocurrency markets.
It identifies overbought and oversold conditions, combined with trend analysis
to enter positions at swing points. The strategy holds positions for days to weeks
to capture larger market movements.

Key characteristics:
- Medium-term holding periods (days to weeks)
- Uses multiple technical indicators for confirmation
- Aims for larger profit targets than day trading
- Focuses on high-quality setups with multiple confirmations
- Adapts to changing market conditions
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
    name="CryptoSwingStrategy",
    market_type="crypto",
    description="Medium-term swing trading strategy for crypto markets focusing on overbought/oversold conditions and trend confirmation",
    timeframes=["H1", "H4", "D1"],  # Swing trading works best on higher timeframes
    parameters={
        # Technical indicators
        "rsi_period": {"type": "int", "default": 14, "min": 7, "max": 21},
        "rsi_overbought": {"type": "float", "default": 70.0, "min": 65.0, "max": 80.0},
        "rsi_oversold": {"type": "float", "default": 30.0, "min": 20.0, "max": 35.0},
        "ema_short": {"type": "int", "default": 20, "min": 10, "max": 30},
        "ema_medium": {"type": "int", "default": 50, "min": 30, "max": 80},
        "ema_long": {"type": "int", "default": 200, "min": 100, "max": 300},
        "macd_fast": {"type": "int", "default": 12, "min": 8, "max": 16},
        "macd_slow": {"type": "int", "default": 26, "min": 20, "max": 30},
        "macd_signal": {"type": "int", "default": 9, "min": 7, "max": 12},
        "atr_period": {"type": "int", "default": 14, "min": 7, "max": 21},
        "stoch_k_period": {"type": "int", "default": 14, "min": 7, "max": 21},
        "stoch_d_period": {"type": "int", "default": 3, "min": 2, "max": 5},
        "stoch_overbought": {"type": "float", "default": 80.0, "min": 75.0, "max": 85.0},
        "stoch_oversold": {"type": "float", "default": 20.0, "min": 15.0, "max": 25.0},
        
        # Trade execution
        "profit_target_atr_multi": {"type": "float", "default": 3.0, "min": 1.5, "max": 5.0},
        "stop_loss_atr_multi": {"type": "float", "default": 1.5, "min": 1.0, "max": 3.0},
        "trailing_stop_enabled": {"type": "bool", "default": True},
        "trailing_stop_activation_pct": {"type": "float", "default": 0.02, "min": 0.01, "max": 0.05},
        "trailing_stop_distance_atr": {"type": "float", "default": 2.0, "min": 1.0, "max": 4.0},
        
        # Risk management
        "risk_per_trade": {"type": "float", "default": 0.01, "min": 0.005, "max": 0.02},
        "max_open_positions": {"type": "int", "default": 3, "min": 1, "max": 5},
        "wait_bars_after_signal": {"type": "int", "default": 1, "min": 0, "max": 3},
        "correlation_threshold": {"type": "float", "default": 0.7, "min": 0.5, "max": 0.9},
    }
)
class CryptoSwingStrategy(CryptoBaseStrategy):
    """
    A medium-term swing trading strategy for cryptocurrency markets.
    
    This strategy:
    1. Uses multiple technical indicators (RSI, EMA, MACD, Stochastics) to identify swing points
    2. Focuses on medium-term price movements over days or weeks
    3. Employs strict trend-confirmation rules to avoid false signals
    4. Uses trailing stops to maximize profits on successful trades
    5. Adapts position sizing based on volatility and signal strength
    """
    
    def __init__(self, session: CryptoSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """
        Initialize the crypto swing trading strategy.
        
        Args:
            session: Crypto trading session
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        super().__init__(session, data_pipeline, parameters)
        
        # Trade monitoring
        self.open_swing_trades = []
        self.pending_signals = []
        self.last_signal_time = None
        self.bars_since_signal = 0
        self.last_evaluated_candle = None
        
        # Performance tracking
        self.swing_trades_total = 0
        self.swing_trades_successful = 0
        self.swing_trades_failed = 0
        self.average_holding_period = 0  # in days
        
        # Signal quality assessment
        self.signal_quality_history = []
        self.recent_market_structure = {
            "trend_strength": 0,
            "current_trend": "neutral",
            "volatility_regime": "normal",
            "key_levels": [],
        }
        
        logger.info(f"Initialized crypto swing trading strategy for {self.session.symbol} on {self.session.timeframe}")
        logger.info(f"Using RSI({self.parameters['rsi_period']}), EMAs ({self.parameters['ema_short']}/{self.parameters['ema_medium']}/{self.parameters['ema_long']}), MACD, and Stochastics")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for swing trading strategy.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary of calculated indicators
        """
        if data.empty or len(data) < self.parameters["ema_long"]:
            return {}
        
        indicators = {}
        
        # RSI - Momentum indicator for overbought/oversold conditions
        rsi_period = self.parameters["rsi_period"]
        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        rs = avg_gain / avg_loss
        indicators["rsi"] = 100 - (100 / (1 + rs))
        
        # EMAs - Trend indicators
        indicators["ema_short"] = data["close"].ewm(span=self.parameters["ema_short"], adjust=False).mean()
        indicators["ema_medium"] = data["close"].ewm(span=self.parameters["ema_medium"], adjust=False).mean()
        indicators["ema_long"] = data["close"].ewm(span=self.parameters["ema_long"], adjust=False).mean()
        
        # MACD - Trend and momentum indicator
        ema_fast = data["close"].ewm(span=self.parameters["macd_fast"], adjust=False).mean()
        ema_slow = data["close"].ewm(span=self.parameters["macd_slow"], adjust=False).mean()
        indicators["macd_line"] = ema_fast - ema_slow
        indicators["macd_signal"] = indicators["macd_line"].ewm(span=self.parameters["macd_signal"], adjust=False).mean()
        indicators["macd_histogram"] = indicators["macd_line"] - indicators["macd_signal"]
        
        # Stochastic Oscillator - Momentum indicator
        k_period = self.parameters["stoch_k_period"]
        d_period = self.parameters["stoch_d_period"]
        lowest_low = data["low"].rolling(window=k_period).min()
        highest_high = data["high"].rolling(window=k_period).max()
        indicators["stoch_k"] = 100 * ((data["close"] - lowest_low) / (highest_high - lowest_low))
        indicators["stoch_d"] = indicators["stoch_k"].rolling(window=d_period).mean()
        
        # ATR - Volatility indicator for position sizing and stop placement
        high_low = data["high"] - data["low"]
        high_close = abs(data["high"] - data["close"].shift())
        low_close = abs(data["low"] - data["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators["atr"] = true_range.rolling(window=self.parameters["atr_period"]).mean()
        
        # Trend direction and strength
        indicators["trend_direction"] = np.where(
            indicators["ema_short"] > indicators["ema_medium"], 1,
            np.where(indicators["ema_short"] < indicators["ema_medium"], -1, 0)
        )
        
        # Swing high/low detection
        indicators["swing_high"] = self._detect_swing_highs(data)
        indicators["swing_low"] = self._detect_swing_lows(data)
        
        # Market volatility regime
        mean_atr = indicators["atr"].mean()
        indicators["volatility_regime"] = np.where(
            indicators["atr"].iloc[-1] > mean_atr * 1.5, "high",
            np.where(indicators["atr"].iloc[-1] < mean_atr * 0.75, "low", "normal")
        )
        
        return indicators
    
    def _detect_swing_highs(self, data: pd.DataFrame, lookback: int = 3) -> pd.Series:
        """Detect swing highs in price data."""
        highs = data["high"]
        swing_highs = pd.Series(index=data.index, dtype=bool)
        
        for i in range(lookback, len(data) - lookback):
            left_higher = all(highs.iloc[i] > highs.iloc[i-j] for j in range(1, lookback+1))
            right_higher = all(highs.iloc[i] > highs.iloc[i+j] for j in range(1, lookback+1))
            swing_highs.iloc[i] = left_higher and right_higher
            
        return swing_highs
    
    def _detect_swing_lows(self, data: pd.DataFrame, lookback: int = 3) -> pd.Series:
        """Detect swing lows in price data."""
        lows = data["low"]
        swing_lows = pd.Series(index=data.index, dtype=bool)
        
        for i in range(lookback, len(data) - lookback):
            left_lower = all(lows.iloc[i] < lows.iloc[i-j] for j in range(1, lookback+1))
            right_lower = all(lows.iloc[i] < lows.iloc[i+j] for j in range(1, lookback+1))
            swing_lows.iloc[i] = left_lower and right_lower
            
        return swing_lows
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on calculated indicators.
        
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
        
        if not indicators or data.empty or len(data) < 20:
            return signals
        
        # Current indicator values
        current_close = data["close"].iloc[-1]
        current_rsi = indicators["rsi"].iloc[-1]
        prev_rsi = indicators["rsi"].iloc[-2] if len(indicators["rsi"]) > 1 else 50
        current_macd = indicators["macd_line"].iloc[-1]
        current_macd_signal = indicators["macd_signal"].iloc[-1]
        prev_macd = indicators["macd_line"].iloc[-2] if len(indicators["macd_line"]) > 1 else 0
        prev_macd_signal = indicators["macd_signal"].iloc[-2] if len(indicators["macd_signal"]) > 1 else 0
        current_stoch_k = indicators["stoch_k"].iloc[-1]
        current_stoch_d = indicators["stoch_d"].iloc[-1]
        
        # Trend assessment
        ema_short_current = indicators["ema_short"].iloc[-1]
        ema_medium_current = indicators["ema_medium"].iloc[-1]
        ema_long_current = indicators["ema_long"].iloc[-1]
        
        # Detect trend direction
        if ema_short_current > ema_medium_current and ema_medium_current > ema_long_current:
            signals["trend"] = "bullish"
            trend_strength = 2
        elif ema_short_current > ema_medium_current:
            signals["trend"] = "bullish"
            trend_strength = 1
        elif ema_short_current < ema_medium_current and ema_medium_current < ema_long_current:
            signals["trend"] = "bearish"
            trend_strength = 2
        elif ema_short_current < ema_medium_current:
            signals["trend"] = "bearish"
            trend_strength = 1
        else:
            signals["trend"] = "neutral"
            trend_strength = 0
        
        # Update recent market structure
        self.recent_market_structure["current_trend"] = signals["trend"]
        self.recent_market_structure["trend_strength"] = trend_strength
        self.recent_market_structure["volatility_regime"] = indicators["volatility_regime"]
        
        # Long entry conditions
        long_conditions = [
            # Oversold condition recovering
            current_rsi < self.parameters["rsi_oversold"] and prev_rsi < current_rsi,
            
            # MACD crossover (bullish)
            prev_macd < prev_macd_signal and current_macd > current_macd_signal,
            
            # Stochastic oversold and rising
            current_stoch_k < self.parameters["stoch_oversold"] and current_stoch_k > current_stoch_d,
            
            # Confirmed swing low
            indicators["swing_low"].iloc[-1]
        ]
        
        # Short entry conditions
        short_conditions = [
            # Overbought condition declining
            current_rsi > self.parameters["rsi_overbought"] and prev_rsi > current_rsi,
            
            # MACD crossover (bearish)
            prev_macd > prev_macd_signal and current_macd < current_macd_signal,
            
            # Stochastic overbought and falling
            current_stoch_k > self.parameters["stoch_overbought"] and current_stoch_k < current_stoch_d,
            
            # Confirmed swing high
            indicators["swing_high"].iloc[-1]
        ]
        
        # Long signal - needs at least 2 conditions plus favorable trend
        long_condition_count = sum(long_conditions)
        if long_condition_count >= 2 and (signals["trend"] in ["bullish", "neutral"]):
            # Signal strength based on condition count and trend strength
            signal_strength = (long_condition_count / len(long_conditions)) * (trend_strength + 1) / 3
            signals["long_entry"] = True
            signals["signal_strength"] = signal_strength
            
            # Calculate stop loss and take profit (ATR-based)
            atr = indicators["atr"].iloc[-1]
            signals["stop_loss"] = current_close - (atr * self.parameters["stop_loss_atr_multi"])
            signals["take_profit"] = current_close + (atr * self.parameters["profit_target_atr_multi"])
            
            logger.info(f"SWING LONG signal generated for {self.session.symbol} with strength {signal_strength:.2f}")
        
        # Short signal - needs at least 2 conditions plus favorable trend
        short_condition_count = sum(short_conditions)
        if short_condition_count >= 2 and (signals["trend"] in ["bearish", "neutral"]):
            # Signal strength based on condition count and trend strength
            signal_strength = (short_condition_count / len(short_conditions)) * (trend_strength + 1) / 3
            signals["short_entry"] = True
            signals["signal_strength"] = signal_strength
            
            # Calculate stop loss and take profit (ATR-based)
            atr = indicators["atr"].iloc[-1]
            signals["stop_loss"] = current_close + (atr * self.parameters["stop_loss_atr_multi"])
            signals["take_profit"] = current_close - (atr * self.parameters["profit_target_atr_multi"])
            
            logger.info(f"SWING SHORT signal generated for {self.session.symbol} with strength {signal_strength:.2f}")
        
        # Exit signals based on trend change or indicator reversal
        for position in self.positions:
            if position.direction == "long":
                # Exit long if trend becomes strongly bearish or technical indicators reverse
                if (signals["trend"] == "bearish" and trend_strength == 2) or \
                   (current_rsi > self.parameters["rsi_overbought"] and prev_rsi > current_rsi) or \
                   (indicators["swing_high"].iloc[-1]):
                    signals["long_exit"] = True
                    logger.info(f"Exit LONG signal for {self.session.symbol}")
            elif position.direction == "short":
                # Exit short if trend becomes strongly bullish or technical indicators reverse
                if (signals["trend"] == "bullish" and trend_strength == 2) or \
                   (current_rsi < self.parameters["rsi_oversold"] and prev_rsi < current_rsi) or \
                   (indicators["swing_low"].iloc[-1]):
                    signals["short_exit"] = True
                    logger.info(f"Exit SHORT signal for {self.session.symbol}")
        
        # Track the last signal time
        if signals["long_entry"] or signals["short_entry"]:
            self.last_signal_time = data.index[-1]
            self.bars_since_signal = 0
        else:
            # Increment bars since last signal
            if self.last_signal_time is not None:
                self.bars_since_signal += 1
        
        return signals
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate position size based on risk parameters, ATR, and signal strength.
        
        For swing trading, we use ATR to determine stop distance, then calculate
        position size to risk a fixed percentage of account on each trade.
        
        Args:
            direction: Direction of the trade ('long' or 'short')
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Position size in crypto units
        """
        if data.empty or not indicators or "atr" not in indicators:
            return 0.0
        
        # Account balance (in base currency)
        account_balance = 10000.0  # Mock value, would come from exchange API
        
        # Risk per trade (as percentage of account)
        risk_per_trade = self.parameters["risk_per_trade"]
        risk_amount = account_balance * risk_per_trade
        
        # Current price and ATR
        current_price = data["close"].iloc[-1]
        atr = indicators["atr"].iloc[-1]
        
        # Stop loss distance based on ATR
        stop_loss_multiple = self.parameters["stop_loss_atr_multi"]
        if direction == "long":
            stop_distance = atr * stop_loss_multiple
        else:  # short
            stop_distance = atr * stop_loss_multiple
        
        # Position size calculation (in base currency)
        if stop_distance > 0 and current_price > 0:
            position_size_base = risk_amount / (stop_distance / current_price)
            
            # Convert to crypto units
            position_size_crypto = position_size_base / current_price
            
            # Adjust for signal strength if available
            signal_strength = indicators.get("signal_strength", 0.5)
            if signal_strength > 0:
                position_size_crypto = position_size_crypto * min(signal_strength * 1.5, 1.0)
            
            # Ensure minimum trade size
            min_trade_size = self.session.min_trade_size
            position_size_crypto = max(position_size_crypto, min_trade_size)
            
            # Round to appropriate precision
            decimals = 8 if self.session.symbol.startswith("BTC") else 6
            position_size_crypto = round(position_size_crypto, decimals)
            
            return position_size_crypto
        
        return 0.0
    
    def _on_timeframe_completed(self, event: Event) -> None:
        """
        Handle timeframe completed events.
        
        For swing trading, we primarily make decisions at the close of each candle,
        especially on higher timeframes.
        """
        super()._on_timeframe_completed(event)
        
        # Only process if this is our timeframe
        if event.data.get('timeframe') != self.session.timeframe:
            return
        
        # Wait for enough data
        if self.market_data.empty or len(self.market_data) < self.parameters["ema_long"]:
            return
        
        # Avoid processing the same candle multiple times
        current_candle_time = self.market_data.index[-1]
        if self.last_evaluated_candle == current_candle_time:
            return
            
        self.last_evaluated_candle = current_candle_time
        
        # Calculate new indicators with the latest data
        self.indicators = self.calculate_indicators(self.market_data)
        
        # Generate trading signals
        self.signals = self.generate_signals(self.market_data, self.indicators)
        
        # Check for trade opportunities
        self._check_for_trade_opportunities()
        
        # Update trailing stops for open positions
        if self.parameters["trailing_stop_enabled"]:
            self._update_trailing_stops()
    
    def _update_trailing_stops(self) -> None:
        """Update trailing stops for open positions to lock in profits."""
        if not self.positions or not self.indicators or "atr" not in self.indicators:
            return
        
        current_price = self.market_data["close"].iloc[-1]
        atr = self.indicators["atr"].iloc[-1]
        trailing_distance = atr * self.parameters["trailing_stop_distance_atr"]
        activation_threshold = self.parameters["trailing_stop_activation_pct"]
        
        for position in self.positions:
            # Calculate profit for this position
            if position.direction == "long":
                current_profit_pct = (current_price - position.entry_price) / position.entry_price
                
                # Only activate trailing stop if we've reached the activation threshold
                if current_profit_pct > activation_threshold:
                    # Calculate new stop level
                    new_stop = current_price - trailing_distance
                    
                    # Only move stop up, never down
                    if position.stop_loss is None or new_stop > position.stop_loss:
                        position.stop_loss = new_stop
                        logger.info(f"Updated trailing stop for LONG position to {new_stop:.2f}")
                
            elif position.direction == "short":
                current_profit_pct = (position.entry_price - current_price) / position.entry_price
                
                # Only activate trailing stop if we've reached the activation threshold
                if current_profit_pct > activation_threshold:
                    # Calculate new stop level
                    new_stop = current_price + trailing_distance
                    
                    # Only move stop down, never up
                    if position.stop_loss is None or new_stop < position.stop_loss:
                        position.stop_loss = new_stop
                        logger.info(f"Updated trailing stop for SHORT position to {new_stop:.2f}")
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible this strategy is with the current market regime.
        
        Swing trading works well in both trending and ranging markets, but prefers
        less volatile conditions where swing points are well-defined.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            "trending": 0.85,        # Very good in trending markets
            "ranging": 0.80,         # Good in ranging markets too
            "volatile": 0.60,        # Moderate in volatile markets
            "calm": 0.75,            # Good in calm markets
            "breakout": 0.65,        # Moderate during breakouts
            "high_volume": 0.80,     # Good during high volume periods
            "low_volume": 0.65,      # Moderate during low volume
            "high_liquidity": 0.85,  # Very good in high liquidity markets
            "low_liquidity": 0.60,   # Moderate in low liquidity markets
        }
        
        return compatibility_map.get(market_regime, 0.70)  # Default is decent
