#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Swing Trading Strategy Module

This module implements swing trading strategies for capturing medium-term
price movements over periods of days to weeks.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from trading_bot.strategies.strategy_template import (
    StrategyTemplate, 
    StrategyOptimizable,
    Signal, 
    SignalType,
    TimeFrame,
    MarketRegime
)
from trading_bot.strategies.position_trading import PositionTradingStrategy

# Setup logging
logger = logging.getLogger(__name__)

class SwingTradingStrategy(StrategyOptimizable):
    """
    Swing Trading Strategy designed to capture medium-term price swings.
    
    This strategy combines technical indicators (moving averages, RSI, MACD) 
    with support/resistance levels to identify potential swing trade opportunities
    that typically last 2-10 days.
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Swing Trading strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters
        default_params = {
            "fast_ma_period": 20,
            "slow_ma_period": 50,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "min_swing_days": 2,
            "max_swing_days": 10,
            "pullback_threshold": 0.03,  # 3% pullback considered for entry
            "trend_strength_threshold": 0.05,  # 5% for trend confirmation
            "volume_filter": True,
            "min_volume_percentile": 60,
            "atr_period": 14,
            "stop_loss_atr_multiple": 2.0,
            "take_profit_atr_multiple": 3.0,
            "trailing_stop": True,
            "trailing_stop_activation_percent": 1.5
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        logger.info(f"Initialized Swing Trading strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "fast_ma_period": [10, 15, 20, 25],
            "slow_ma_period": [40, 50, 60, 70],
            "rsi_period": [10, 14, 21],
            "rsi_overbought": [65, 70, 75, 80],
            "rsi_oversold": [20, 25, 30, 35],
            "macd_fast": [8, 12, 16],
            "macd_slow": [22, 26, 30],
            "macd_signal": [7, 9, 11],
            "min_swing_days": [1, 2, 3],
            "max_swing_days": [7, 10, 14],
            "pullback_threshold": [0.02, 0.03, 0.04, 0.05],
            "trend_strength_threshold": [0.03, 0.05, 0.07],
            "volume_filter": [True, False],
            "min_volume_percentile": [50, 60, 70],
            "stop_loss_atr_multiple": [1.5, 2.0, 2.5, 3.0],
            "take_profit_atr_multiple": [2.0, 3.0, 4.0, 5.0],
            "trailing_stop": [True, False],
            "trailing_stop_activation_percent": [1.0, 1.5, 2.0, 2.5]
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Series of price data
            period: RSI period
            
        Returns:
            Series with RSI values
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses over the period
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        # Calculate relative strength
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast_period: int = 12, 
                       slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD indicator.
        
        Args:
            prices: Series of price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal EMA period
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        # Calculate EMAs
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _find_support_resistance(self, highs: pd.Series, lows: pd.Series, closes: pd.Series,
                                volume: pd.Series, lookback: int = 50, 
                                min_distance: float = 0.02) -> Tuple[List[float], List[float]]:
        """
        Find significant support and resistance levels.
        
        Args:
            highs: Series of high prices
            lows: Series of low prices
            closes: Series of close prices
            volume: Series of volume data
            lookback: Period to look back
            min_distance: Minimum price distance (as %) for unique levels
            
        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        if len(closes) < lookback:
            return [], []
        
        # Use only lookback period
        recent_highs = highs[-lookback:].values
        recent_lows = lows[-lookback:].values
        recent_closes = closes[-lookback:].values
        recent_volume = volume[-lookback:].values if volume is not None else np.ones_like(recent_closes)
        
        # Initialize levels
        support_levels = []
        resistance_levels = []
        
        # Find local minima/maxima with volume confirmation
        for i in range(2, len(recent_closes)-2):
            # Check for potential resistance (local high)
            if (recent_closes[i] > recent_closes[i-1] and 
                recent_closes[i] > recent_closes[i-2] and
                recent_closes[i] > recent_closes[i+1] and
                recent_closes[i] > recent_closes[i+2] and
                recent_volume[i] > np.mean(recent_volume[i-5:i+5])):
                
                # New potential resistance level
                new_level = recent_closes[i]
                
                # Check if it's far enough from existing resistance levels
                is_unique = True
                for level in resistance_levels:
                    if abs(new_level/level - 1) < min_distance:
                        is_unique = False
                        break
                
                if is_unique:
                    resistance_levels.append(new_level)
            
            # Check for potential support (local low)
            if (recent_closes[i] < recent_closes[i-1] and 
                recent_closes[i] < recent_closes[i-2] and
                recent_closes[i] < recent_closes[i+1] and
                recent_closes[i] < recent_closes[i+2] and
                recent_volume[i] > np.mean(recent_volume[i-5:i+5])):
                
                # New potential support level
                new_level = recent_closes[i]
                
                # Check if it's far enough from existing support levels
                is_unique = True
                for level in support_levels:
                    if abs(new_level/level - 1) < min_distance:
                        is_unique = False
                        break
                
                if is_unique:
                    support_levels.append(new_level)
        
        # Sort levels
        support_levels.sort()
        resistance_levels.sort()
        
        return support_levels, resistance_levels
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate swing trading indicators for all symbols.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary of calculated indicators for each symbol
        """
        indicators = {}
        
        # Get parameters
        fast_ma = self.parameters.get("fast_ma_period", 20)
        slow_ma = self.parameters.get("slow_ma_period", 50)
        rsi_period = self.parameters.get("rsi_period", 14)
        macd_fast = self.parameters.get("macd_fast", 12)
        macd_slow = self.parameters.get("macd_slow", 26)
        macd_signal = self.parameters.get("macd_signal", 9)
        atr_period = self.parameters.get("atr_period", 14)
        
        for symbol, df in data.items():
            # Ensure required columns exist
            if not all(col in df.columns for col in ['high', 'low', 'close']):
                logger.warning(f"Required price columns not found for {symbol}")
                continue
            
            try:
                # Calculate moving averages
                fast_ma_values = df['close'].rolling(window=fast_ma).mean()
                slow_ma_values = df['close'].rolling(window=slow_ma).mean()
                
                # Calculate RSI
                rsi_values = self._calculate_rsi(df['close'], period=rsi_period)
                
                # Calculate MACD
                macd_line, signal_line, histogram = self._calculate_macd(
                    df['close'], fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal
                )
                
                # Calculate ATR for stop loss/take profit
                high_low = df['high'] - df['low']
                high_close_prev = np.abs(df['high'] - df['close'].shift(1))
                low_close_prev = np.abs(df['low'] - df['close'].shift(1))
                tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
                atr = tr.rolling(window=atr_period).mean()
                
                # Calculate support and resistance levels
                volume = df['volume'] if 'volume' in df.columns else None
                support_levels, resistance_levels = self._find_support_resistance(
                    df['high'], df['low'], df['close'], volume
                )
                
                # Calculate pullbacks from recent swing highs/lows
                recent_high = df['high'].rolling(window=20).max()
                recent_low = df['low'].rolling(window=20).min()
                
                # Pullback from high (for bullish setups)
                pullback_from_high = (recent_high - df['close']) / recent_high
                
                # Pullback from low (for bearish setups)
                pullback_from_low = (df['close'] - recent_low) / df['close']
                
                # Calculate volume indicators if volume data is available
                volume_percentile = None
                if 'volume' in df.columns:
                    volume_percentile = df['volume'].rolling(window=20).apply(
                        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
                    )
                
                # Store indicators
                indicators[symbol] = {
                    "fast_ma": pd.DataFrame({"fast_ma": fast_ma_values}),
                    "slow_ma": pd.DataFrame({"slow_ma": slow_ma_values}),
                    "rsi": pd.DataFrame({"rsi": rsi_values}),
                    "macd_line": pd.DataFrame({"macd_line": macd_line}),
                    "macd_signal": pd.DataFrame({"macd_signal": signal_line}),
                    "macd_histogram": pd.DataFrame({"macd_histogram": histogram}),
                    "atr": pd.DataFrame({"atr": atr}),
                    "pullback_from_high": pd.DataFrame({"pullback_from_high": pullback_from_high}),
                    "pullback_from_low": pd.DataFrame({"pullback_from_low": pullback_from_low}),
                    "support_levels": support_levels,
                    "resistance_levels": resistance_levels
                }
                
                if volume_percentile is not None:
                    indicators[symbol]["volume_percentile"] = pd.DataFrame({"volume_percentile": volume_percentile})
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
        
        return indicators
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate swing trading signals based on indicator combinations.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Get parameters
        rsi_overbought = self.parameters.get("rsi_overbought", 70)
        rsi_oversold = self.parameters.get("rsi_oversold", 30)
        pullback_threshold = self.parameters.get("pullback_threshold", 0.03)
        trend_strength = self.parameters.get("trend_strength_threshold", 0.05)
        volume_filter = self.parameters.get("volume_filter", True)
        min_volume_percentile = self.parameters.get("min_volume_percentile", 60)
        stop_loss_atr_multiple = self.parameters.get("stop_loss_atr_multiple", 2.0)
        take_profit_atr_multiple = self.parameters.get("take_profit_atr_multiple", 3.0)
        trailing_stop = self.parameters.get("trailing_stop", True)
        trailing_activation = self.parameters.get("trailing_stop_activation_percent", 1.5)
        
        # Calculate indicators
        indicators = self.calculate_indicators(data)
        
        # Generate signals
        signals = {}
        
        for symbol, symbol_indicators in indicators.items():
            try:
                # Get latest data
                latest_data = data[symbol].iloc[-1]
                prev_data = data[symbol].iloc[-2] if len(data[symbol]) > 1 else None
                latest_price = latest_data['close']
                latest_timestamp = latest_data.name if isinstance(latest_data.name, datetime) else datetime.now()
                
                # Get latest indicator values
                latest_fast_ma = symbol_indicators["fast_ma"].iloc[-1]["fast_ma"]
                latest_slow_ma = symbol_indicators["slow_ma"].iloc[-1]["slow_ma"]
                latest_rsi = symbol_indicators["rsi"].iloc[-1]["rsi"]
                latest_macd = symbol_indicators["macd_line"].iloc[-1]["macd_line"]
                latest_macd_signal = symbol_indicators["macd_signal"].iloc[-1]["macd_signal"]
                latest_macd_hist = symbol_indicators["macd_histogram"].iloc[-1]["macd_histogram"]
                latest_atr = symbol_indicators["atr"].iloc[-1]["atr"]
                latest_pullback_high = symbol_indicators["pullback_from_high"].iloc[-1]["pullback_from_high"]
                latest_pullback_low = symbol_indicators["pullback_from_low"].iloc[-1]["pullback_from_low"]
                
                # Get previous indicator values if available
                prev_macd_hist = symbol_indicators["macd_histogram"].iloc[-2]["macd_histogram"] if len(symbol_indicators["macd_histogram"]) > 1 else 0
                
                # Check volume filter if enabled
                volume_ok = True
                if volume_filter and "volume_percentile" in symbol_indicators:
                    vol_pct = symbol_indicators["volume_percentile"].iloc[-1]["volume_percentile"]
                    volume_ok = vol_pct >= min_volume_percentile
                
                # Get support/resistance levels
                support_levels = symbol_indicators.get("support_levels", [])
                resistance_levels = symbol_indicators.get("resistance_levels", [])
                
                # Find nearest support/resistance
                nearest_support = max([level for level in support_levels if level < latest_price], default=None)
                nearest_resistance = min([level for level in resistance_levels if level > latest_price], default=None)
                
                # Generate signal based on swing trading conditions
                signal_type = None
                confidence = 0.0
                
                # BULLISH SWING SETUP
                if (latest_fast_ma > latest_slow_ma and  # Uptrend confirmed by MA crossover
                    latest_price > latest_slow_ma * (1 + trend_strength) and  # Price well above slow MA
                    latest_pullback_high >= pullback_threshold and  # Recent pullback from high
                    latest_rsi > 40 and latest_rsi < 60 and  # RSI in neutral zone after reset
                    latest_macd_hist > 0 and latest_macd_hist > prev_macd_hist and  # MACD histogram rising
                    volume_ok):
                    
                    signal_type = SignalType.BUY
                    
                    # Calculate confidence based on multiple factors
                    # 1. Trend strength
                    trend_confidence = min(0.3, (latest_price / latest_slow_ma - 1) * 5)
                    
                    # 2. Pullback quality
                    pullback_confidence = min(0.2, latest_pullback_high * 5)
                    
                    # 3. MACD momentum
                    macd_momentum = (latest_macd_hist - prev_macd_hist) / abs(prev_macd_hist) if prev_macd_hist != 0 else 0
                    macd_confidence = min(0.2, max(0.1, macd_momentum))
                    
                    # 4. Support strength
                    support_confidence = 0.0
                    if nearest_support is not None:
                        support_proximity = (latest_price - nearest_support) / latest_price
                        if support_proximity < 0.05:  # Price close to support
                            support_confidence = min(0.2, 0.2 - support_proximity * 2)
                    
                    # 5. Volume
                    volume_confidence = 0.0
                    if "volume_percentile" in symbol_indicators:
                        vol_pct = symbol_indicators["volume_percentile"].iloc[-1]["volume_percentile"]
                        volume_confidence = min(0.1, vol_pct / 1000)
                    
                    confidence = min(0.9, trend_confidence + pullback_confidence + macd_confidence + support_confidence + volume_confidence)
                    
                    # Calculate stop loss and take profit
                    stop_loss = latest_price - (latest_atr * stop_loss_atr_multiple)
                    if nearest_support is not None and nearest_support > stop_loss:
                        # Use support level if it's tighter than ATR-based stop
                        stop_loss = nearest_support * 0.99
                    
                    take_profit = latest_price + (latest_atr * take_profit_atr_multiple)
                    if nearest_resistance is not None:
                        # Target nearest resistance for take profit
                        take_profit = min(take_profit, nearest_resistance)
                
                # BEARISH SWING SETUP
                elif (latest_fast_ma < latest_slow_ma and  # Downtrend confirmed by MA crossover
                      latest_price < latest_slow_ma * (1 - trend_strength) and  # Price well below slow MA
                      latest_pullback_low >= pullback_threshold and  # Recent pullback from low
                      latest_rsi < 60 and latest_rsi > 40 and  # RSI in neutral zone after reset
                      latest_macd_hist < 0 and latest_macd_hist < prev_macd_hist and  # MACD histogram falling
                      volume_ok):
                    
                    signal_type = SignalType.SELL
                    
                    # Calculate confidence based on multiple factors
                    # 1. Trend strength
                    trend_confidence = min(0.3, (1 - latest_price / latest_slow_ma) * 5)
                    
                    # 2. Pullback quality
                    pullback_confidence = min(0.2, latest_pullback_low * 5)
                    
                    # 3. MACD momentum
                    macd_momentum = (prev_macd_hist - latest_macd_hist) / abs(prev_macd_hist) if prev_macd_hist != 0 else 0
                    macd_confidence = min(0.2, max(0.1, macd_momentum))
                    
                    # 4. Resistance strength
                    resistance_confidence = 0.0
                    if nearest_resistance is not None:
                        resistance_proximity = (nearest_resistance - latest_price) / latest_price
                        if resistance_proximity < 0.05:  # Price close to resistance
                            resistance_confidence = min(0.2, 0.2 - resistance_proximity * 2)
                    
                    # 5. Volume
                    volume_confidence = 0.0
                    if "volume_percentile" in symbol_indicators:
                        vol_pct = symbol_indicators["volume_percentile"].iloc[-1]["volume_percentile"]
                        volume_confidence = min(0.1, vol_pct / 1000)
                    
                    confidence = min(0.9, trend_confidence + pullback_confidence + macd_confidence + resistance_confidence + volume_confidence)
                    
                    # Calculate stop loss and take profit
                    stop_loss = latest_price + (latest_atr * stop_loss_atr_multiple)
                    if nearest_resistance is not None and nearest_resistance < stop_loss:
                        # Use resistance level if it's tighter than ATR-based stop
                        stop_loss = nearest_resistance * 1.01
                    
                    take_profit = latest_price - (latest_atr * take_profit_atr_multiple)
                    if nearest_support is not None:
                        # Target nearest support for take profit
                        take_profit = max(take_profit, nearest_support)
                
                # MOMENTUM REVERSAL SETUP (Alternative swing entry)
                elif volume_ok:
                    # Bullish reversal
                    if (latest_rsi < rsi_oversold and  # Oversold condition
                        latest_rsi > symbol_indicators["rsi"].iloc[-2]["rsi"] and  # RSI turning up
                        latest_macd_hist > prev_macd_hist and  # MACD histogram improving
                        nearest_support is not None and 
                        (latest_price - nearest_support) / latest_price < 0.03):  # Price near support
                        
                        signal_type = SignalType.BUY
                        
                        # Calculate confidence (lower than trend setup)
                        oversold_conf = min(0.25, (rsi_oversold - latest_rsi + 10) / 100)
                        macd_conf = min(0.15, (latest_macd_hist - prev_macd_hist) * 20) if prev_macd_hist < 0 else 0.05
                        support_conf = min(0.2, 0.2 - (latest_price - nearest_support) / latest_price * 5)
                        volume_conf = min(0.1, symbol_indicators["volume_percentile"].iloc[-1]["volume_percentile"] / 1000) if "volume_percentile" in symbol_indicators else 0
                        
                        confidence = min(0.8, oversold_conf + macd_conf + support_conf + volume_conf)
                        
                        # Calculate stop loss and take profit
                        stop_loss = latest_price - (latest_atr * stop_loss_atr_multiple)
                        take_profit = latest_price + (latest_atr * take_profit_atr_multiple)
                    
                    # Bearish reversal
                    elif (latest_rsi > rsi_overbought and  # Overbought condition
                          latest_rsi < symbol_indicators["rsi"].iloc[-2]["rsi"] and  # RSI turning down
                          latest_macd_hist < prev_macd_hist and  # MACD histogram worsening
                          nearest_resistance is not None and
                          (nearest_resistance - latest_price) / latest_price < 0.03):  # Price near resistance
                        
                        signal_type = SignalType.SELL
                        
                        # Calculate confidence (lower than trend setup)
                        overbought_conf = min(0.25, (latest_rsi - rsi_overbought + 10) / 100)
                        macd_conf = min(0.15, (prev_macd_hist - latest_macd_hist) * 20) if prev_macd_hist > 0 else 0.05
                        resistance_conf = min(0.2, 0.2 - (nearest_resistance - latest_price) / latest_price * 5)
                        volume_conf = min(0.1, symbol_indicators["volume_percentile"].iloc[-1]["volume_percentile"] / 1000) if "volume_percentile" in symbol_indicators else 0
                        
                        confidence = min(0.8, overbought_conf + macd_conf + resistance_conf + volume_conf)
                        
                        # Calculate stop loss and take profit
                        stop_loss = latest_price + (latest_atr * stop_loss_atr_multiple)
                        take_profit = latest_price - (latest_atr * take_profit_atr_multiple)
                
                # Create signal if we have a valid signal type
                if signal_type:
                    signals[symbol] = Signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        price=latest_price,
                        timestamp=latest_timestamp,
                        confidence=confidence,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            "fast_ma": latest_fast_ma,
                            "slow_ma": latest_slow_ma,
                            "rsi": latest_rsi,
                            "macd_histogram": latest_macd_hist,
                            "atr": latest_atr,
                            "trailing_stop": trailing_stop,
                            "trailing_activation": trailing_activation,
                            "strategy_type": "swing_trading"
                        }
                    )
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals 