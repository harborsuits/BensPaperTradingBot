#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trend Following Strategy Module

This module implements various trend following trading strategies.
Trend following strategies aim to identify and capitalize on established
market trends, entering positions in the direction of the trend.
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

# Setup logging
logger = logging.getLogger(__name__)

class MovingAverageCrossStrategy(StrategyOptimizable):
    """
    Trend following strategy based on moving average crossovers.
    
    This strategy generates buy signals when a faster moving average
    crosses above a slower moving average, and sell signals when the
    faster moving average crosses below the slower moving average.
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Moving Average Crossover strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters
        default_params = {
            "fast_ma_period": 20,
            "slow_ma_period": 50,
            "fast_ma_type": "sma",  # simple, exponential, weighted
            "slow_ma_type": "sma",
            "trend_filter_period": 200,  # longer-term trend filter
            "volume_filter": True,
            "min_volume_percentile": 50,
            "stop_loss_atr_multiple": 2.0,
            "atr_period": 14,
            "take_profit_atr_multiple": 4.0
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        logger.info(f"Initialized Moving Average Cross strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "fast_ma_period": [10, 15, 20, 25, 30],
            "slow_ma_period": [40, 50, 60, 80, 100],
            "fast_ma_type": ["sma", "ema", "wma"],
            "slow_ma_type": ["sma", "ema", "wma"],
            "trend_filter_period": [0, 100, 150, 200, 250],  # 0 means no filter
            "volume_filter": [True, False],
            "min_volume_percentile": [30, 40, 50, 60, 70],
            "stop_loss_atr_multiple": [1.5, 2.0, 2.5, 3.0],
            "atr_period": [10, 14, 20],
            "take_profit_atr_multiple": [3.0, 4.0, 5.0, 6.0]
        }
    
    def _calculate_moving_average(self, series: pd.Series, period: int, ma_type: str) -> pd.Series:
        """
        Calculate moving average based on specified type.
        
        Args:
            series: Price series
            period: MA period
            ma_type: Type of moving average (sma, ema, wma)
            
        Returns:
            Series with calculated moving average
        """
        if ma_type.lower() == "sma":
            return series.rolling(window=period).mean()
        elif ma_type.lower() == "ema":
            return series.ewm(span=period, adjust=False).mean()
        elif ma_type.lower() == "wma":
            weights = np.arange(1, period + 1)
            return series.rolling(window=period).apply(
                lambda x: np.sum(weights * x) / weights.sum(), raw=True
            )
        else:
            logger.warning(f"Unknown MA type: {ma_type}, defaulting to SMA")
            return series.rolling(window=period).mean()
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate moving averages and ATR for all symbols.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary of calculated indicators for each symbol
        """
        indicators = {}
        
        # Get parameters
        fast_period = self.parameters.get("fast_ma_period", 20)
        slow_period = self.parameters.get("slow_ma_period", 50)
        fast_type = self.parameters.get("fast_ma_type", "sma")
        slow_type = self.parameters.get("slow_ma_type", "sma")
        trend_period = self.parameters.get("trend_filter_period", 200)
        atr_period = self.parameters.get("atr_period", 14)
        
        for symbol, df in data.items():
            # Ensure required columns exist
            if 'close' not in df.columns:
                logger.warning(f"Close price column not found for {symbol}")
                continue
            
            try:
                # Calculate moving averages
                fast_ma = self._calculate_moving_average(df['close'], fast_period, fast_type)
                slow_ma = self._calculate_moving_average(df['close'], slow_period, slow_type)
                
                # Calculate longer-term trend MA if specified
                trend_ma = None
                if trend_period > 0:
                    trend_ma = self._calculate_moving_average(df['close'], trend_period, "sma")
                
                # Calculate ATR for volatility assessment
                high_low = df['high'] - df['low']
                high_close_prev = np.abs(df['high'] - df['close'].shift(1))
                low_close_prev = np.abs(df['low'] - df['close'].shift(1))
                tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
                atr = tr.rolling(window=atr_period).mean()
                
                # Calculate volume percentile if volume data is available
                volume_percentile = None
                if 'volume' in df.columns:
                    volume_percentile = df['volume'].rolling(window=20).apply(
                        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
                    )
                
                # Calculate additional trend strength indicators
                price_vs_ma = (df['close'] / slow_ma - 1) * 100  # Percent above/below slow MA
                
                # Store indicators
                indicators[symbol] = {
                    "fast_ma": pd.DataFrame({"fast_ma": fast_ma}),
                    "slow_ma": pd.DataFrame({"slow_ma": slow_ma}),
                    "atr": pd.DataFrame({"atr": atr}),
                    "price_vs_ma": pd.DataFrame({"price_vs_ma": price_vs_ma})
                }
                
                if trend_ma is not None:
                    indicators[symbol]["trend_ma"] = pd.DataFrame({"trend_ma": trend_ma})
                
                if volume_percentile is not None:
                    indicators[symbol]["volume_percentile"] = pd.DataFrame({"volume_percentile": volume_percentile})
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
        
        return indicators
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate buy/sell signals based on moving average crossovers.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Get parameters
        trend_period = self.parameters.get("trend_filter_period", 200)
        volume_filter = self.parameters.get("volume_filter", True)
        min_volume_percentile = self.parameters.get("min_volume_percentile", 50)
        stop_loss_atr_multiple = self.parameters.get("stop_loss_atr_multiple", 2.0)
        take_profit_atr_multiple = self.parameters.get("take_profit_atr_multiple", 4.0)
        
        # Calculate indicators
        indicators = self.calculate_indicators(data)
        
        # Generate signals
        signals = {}
        
        for symbol, symbol_indicators in indicators.items():
            try:
                # Get latest data points
                latest_data = data[symbol].iloc[-1]
                latest_price = latest_data['close']
                latest_timestamp = latest_data.name if isinstance(latest_data.name, datetime) else datetime.now()
                
                # Get previous data point for crossover detection
                previous_data = data[symbol].iloc[-2]
                
                # Get latest indicator values
                fast_ma_now = symbol_indicators["fast_ma"].iloc[-1]["fast_ma"]
                slow_ma_now = symbol_indicators["slow_ma"].iloc[-1]["slow_ma"]
                fast_ma_prev = symbol_indicators["fast_ma"].iloc[-2]["fast_ma"]
                slow_ma_prev = symbol_indicators["slow_ma"].iloc[-2]["slow_ma"]
                latest_atr = symbol_indicators["atr"].iloc[-1]["atr"]
                
                # Check volume filter if enabled
                volume_ok = True
                if volume_filter and "volume_percentile" in symbol_indicators:
                    vol_pct = symbol_indicators["volume_percentile"].iloc[-1]["volume_percentile"]
                    volume_ok = vol_pct >= min_volume_percentile
                
                # Check trend filter if enabled
                trend_ok = True
                if trend_period > 0 and "trend_ma" in symbol_indicators:
                    trend_ma = symbol_indicators["trend_ma"].iloc[-1]["trend_ma"]
                    # Only allow long signals when price is above trend MA
                    # and short signals when price is below trend MA
                    trend_ok = latest_price > trend_ma 
                
                # Generate signal based on moving average crossover
                signal_type = None
                confidence = 0.0
                
                # Detect crossover
                bullish_cross = (fast_ma_prev < slow_ma_prev) and (fast_ma_now >= slow_ma_now)
                bearish_cross = (fast_ma_prev > slow_ma_prev) and (fast_ma_now <= slow_ma_now)
                
                if bullish_cross and volume_ok and trend_ok:
                    # Bullish crossover - buy signal
                    signal_type = SignalType.BUY
                    
                    # Calculate confidence based on distance from slow MA and trend
                    price_vs_ma = symbol_indicators["price_vs_ma"].iloc[-1]["price_vs_ma"]
                    # Higher confidence when price has just started moving above MA
                    ma_confidence = 0.6 + min(0.2, abs(price_vs_ma) * 0.01)
                    
                    # Higher confidence when price is accelerating away from MAs
                    acceleration = (latest_price / fast_ma_now - 1) - (fast_ma_now / slow_ma_now - 1)
                    accel_confidence = 0.0
                    if acceleration > 0:
                        accel_confidence = min(0.1, acceleration * 10)
                    
                    confidence = min(0.9, ma_confidence + accel_confidence)
                    
                    # Calculate stop loss and take profit based on ATR
                    stop_loss = latest_price - (latest_atr * stop_loss_atr_multiple)
                    take_profit = latest_price + (latest_atr * take_profit_atr_multiple)
                    
                elif bearish_cross and volume_ok and not trend_ok:  # Reverse trend check for short
                    # Bearish crossover - sell signal
                    signal_type = SignalType.SELL
                    
                    # Calculate confidence based on distance from slow MA and trend
                    price_vs_ma = symbol_indicators["price_vs_ma"].iloc[-1]["price_vs_ma"]
                    # Higher confidence when price has just started moving below MA
                    ma_confidence = 0.6 + min(0.2, abs(price_vs_ma) * 0.01)
                    
                    # Higher confidence when price is accelerating away from MAs (downwards)
                    acceleration = (latest_price / fast_ma_now - 1) - (fast_ma_now / slow_ma_now - 1)
                    accel_confidence = 0.0
                    if acceleration < 0:
                        accel_confidence = min(0.1, abs(acceleration) * 10)
                    
                    confidence = min(0.9, ma_confidence + accel_confidence)
                    
                    # Calculate stop loss and take profit based on ATR
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
                            "fast_ma": fast_ma_now,
                            "slow_ma": slow_ma_now,
                            "atr": latest_atr,
                            "strategy_type": "trend_following"
                        }
                    )
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals


class ADXTrendStrategy(StrategyOptimizable):
    """
    Trend following strategy based on ADX (Average Directional Index).
    
    This strategy uses ADX to identify the strength of a trend and
    generates signals based on directional movement indicators (DMI).
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ADX Trend strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters
        default_params = {
            "adx_period": 14,
            "adx_threshold": 25,
            "dmi_period": 14,
            "ma_filter_period": 50,
            "ma_filter_type": "ema",
            "stop_loss_atr_multiple": 2.0,
            "atr_period": 14,
            "take_profit_atr_multiple": 4.0
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        logger.info(f"Initialized ADX Trend strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "adx_period": [10, 14, 20, 25],
            "adx_threshold": [20, 25, 30, 35],
            "dmi_period": [10, 14, 20, 25],
            "ma_filter_period": [0, 20, 50, 100, 200],  # 0 means no filter
            "ma_filter_type": ["sma", "ema"],
            "stop_loss_atr_multiple": [1.5, 2.0, 2.5, 3.0],
            "atr_period": [10, 14, 20],
            "take_profit_atr_multiple": [3.0, 4.0, 5.0, 6.0]
        }
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate ADX, DMI, and other indicators for all symbols.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary of calculated indicators for each symbol
        """
        indicators = {}
        
        # Get parameters
        adx_period = self.parameters.get("adx_period", 14)
        dmi_period = self.parameters.get("dmi_period", 14)
        ma_filter_period = self.parameters.get("ma_filter_period", 50)
        ma_filter_type = self.parameters.get("ma_filter_type", "ema")
        atr_period = self.parameters.get("atr_period", 14)
        
        for symbol, df in data.items():
            # Ensure required columns exist
            if not all(col in df.columns for col in ['high', 'low', 'close']):
                logger.warning(f"Required price columns not found for {symbol}")
                continue
            
            try:
                # Calculate True Range
                high_low = df['high'] - df['low']
                high_close_prev = np.abs(df['high'] - df['close'].shift(1))
                low_close_prev = np.abs(df['low'] - df['close'].shift(1))
                tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
                
                # Calculate ATR
                atr = tr.rolling(window=atr_period).mean()
                
                # Calculate Directional Movement
                up_move = df['high'] - df['high'].shift(1)
                down_move = df['low'].shift(1) - df['low']
                
                plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
                minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
                
                # Calculate smoothed DM
                plus_di = pd.Series(plus_dm).rolling(window=dmi_period).mean() / atr * 100
                minus_di = pd.Series(minus_dm).rolling(window=dmi_period).mean() / atr * 100
                
                # Calculate DX and ADX
                dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
                adx = pd.Series(dx).rolling(window=adx_period).mean()
                
                # Calculate MA filter if specified
                ma_filter = None
                if ma_filter_period > 0:
                    if ma_filter_type.lower() == "sma":
                        ma_filter = df['close'].rolling(window=ma_filter_period).mean()
                    elif ma_filter_type.lower() == "ema":
                        ma_filter = df['close'].ewm(span=ma_filter_period, adjust=False).mean()
                
                # Store indicators
                indicators[symbol] = {
                    "adx": pd.DataFrame({"adx": adx}),
                    "plus_di": pd.DataFrame({"plus_di": plus_di}),
                    "minus_di": pd.DataFrame({"minus_di": minus_di}),
                    "atr": pd.DataFrame({"atr": atr})
                }
                
                if ma_filter is not None:
                    indicators[symbol]["ma_filter"] = pd.DataFrame({"ma_filter": ma_filter})
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
        
        return indicators
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate buy/sell signals based on ADX and DMI.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Get parameters
        adx_threshold = self.parameters.get("adx_threshold", 25)
        ma_filter_period = self.parameters.get("ma_filter_period", 50)
        stop_loss_atr_multiple = self.parameters.get("stop_loss_atr_multiple", 2.0)
        take_profit_atr_multiple = self.parameters.get("take_profit_atr_multiple", 4.0)
        
        # Calculate indicators
        indicators = self.calculate_indicators(data)
        
        # Generate signals
        signals = {}
        
        for symbol, symbol_indicators in indicators.items():
            try:
                # Get latest data point
                latest_data = data[symbol].iloc[-1]
                latest_price = latest_data['close']
                latest_timestamp = latest_data.name if isinstance(latest_data.name, datetime) else datetime.now()
                
                # Get latest indicator values
                adx_value = symbol_indicators["adx"].iloc[-1]["adx"]
                plus_di = symbol_indicators["plus_di"].iloc[-1]["plus_di"]
                minus_di = symbol_indicators["minus_di"].iloc[-1]["minus_di"]
                latest_atr = symbol_indicators["atr"].iloc[-1]["atr"]
                
                # Check MA filter if specified
                ma_filter_ok = True
                if ma_filter_period > 0 and "ma_filter" in symbol_indicators:
                    ma_value = symbol_indicators["ma_filter"].iloc[-1]["ma_filter"]
                    ma_filter_ok = latest_price > ma_value
                
                # Generate signal based on ADX and DMI
                signal_type = None
                confidence = 0.0
                
                # Strong trend (ADX above threshold)
                if adx_value > adx_threshold:
                    # Bullish trend (+DI above -DI)
                    if plus_di > minus_di and ma_filter_ok:
                        signal_type = SignalType.BUY
                        
                        # Calculate confidence based on ADX strength and DI difference
                        adx_confidence = min(0.5, 0.3 + (adx_value - adx_threshold) / 100)
                        di_diff_confidence = min(0.4, (plus_di - minus_di) / 100)
                        confidence = min(0.9, adx_confidence + di_diff_confidence)
                        
                        # Calculate stop loss and take profit based on ATR
                        stop_loss = latest_price - (latest_atr * stop_loss_atr_multiple)
                        take_profit = latest_price + (latest_atr * take_profit_atr_multiple)
                        
                    # Bearish trend (-DI above +DI)
                    elif minus_di > plus_di and not ma_filter_ok:  # Reverse check for shorts
                        signal_type = SignalType.SELL
                        
                        # Calculate confidence based on ADX strength and DI difference
                        adx_confidence = min(0.5, 0.3 + (adx_value - adx_threshold) / 100)
                        di_diff_confidence = min(0.4, (minus_di - plus_di) / 100)
                        confidence = min(0.9, adx_confidence + di_diff_confidence)
                        
                        # Calculate stop loss and take profit based on ATR
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
                            "adx": adx_value,
                            "plus_di": plus_di,
                            "minus_di": minus_di,
                            "atr": latest_atr,
                            "strategy_type": "trend_following"
                        }
                    )
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals


class ParabolicSARTrendStrategy(StrategyOptimizable):
    """
    Trend following strategy based on Parabolic SAR.
    
    This strategy uses the Parabolic Stop and Reverse (SAR) indicator to
    identify potential trend changes and generate entry and exit signals.
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Parabolic SAR Trend strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters
        default_params = {
            "sar_acceleration": 0.02,
            "sar_maximum": 0.2,
            "ema_filter_period": 50,
            "adx_filter": True,
            "adx_period": 14,
            "adx_threshold": 25,
            "stop_loss_atr_multiple": 2.0,
            "atr_period": 14,
            "take_profit_atr_multiple": 4.0
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        logger.info(f"Initialized Parabolic SAR Trend strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "sar_acceleration": [0.01, 0.02, 0.03],
            "sar_maximum": [0.1, 0.2, 0.3],
            "ema_filter_period": [0, 20, 50, 100],  # 0 means no filter
            "adx_filter": [True, False],
            "adx_period": [10, 14, 20],
            "adx_threshold": [20, 25, 30],
            "stop_loss_atr_multiple": [1.5, 2.0, 2.5, 3.0],
            "atr_period": [10, 14, 20],
            "take_profit_atr_multiple": [3.0, 4.0, 5.0, 6.0]
        }
    
    def calculate_parabolic_sar(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                                acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
        """
        Calculate Parabolic SAR.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            acceleration: Acceleration factor
            maximum: Maximum acceleration factor
            
        Returns:
            Series with calculated Parabolic SAR values
        """
        length = len(close)
        psar = np.zeros(length)
        psarbull = [None] * length  # Bullish SAR
        psarbear = [None] * length  # Bearish SAR
        bull = True
        af = acceleration
        ep = low[0]
        hp = high[0]
        lp = low[0]
        
        for i in range(2, length):
            if bull:
                psar[i] = psar[i-1] + af * (hp - psar[i-1])
            else:
                psar[i] = psar[i-1] + af * (lp - psar[i-1])
            
            reverse = False
            
            if bull:
                if low[i] < psar[i]:
                    bull = False
                    reverse = True
                    psar[i] = hp
                    lp = low[i]
                    af = acceleration
            else:
                if high[i] > psar[i]:
                    bull = True
                    reverse = True
                    psar[i] = lp
                    hp = high[i]
                    af = acceleration
            
            if not reverse:
                if bull:
                    if high[i] > hp:
                        hp = high[i]
                        af = min(af + acceleration, maximum)
                    if low[i-1] < psar[i]:
                        psar[i] = low[i-1]
                    if low[i-2] < psar[i]:
                        psar[i] = low[i-2]
                else:
                    if low[i] < lp:
                        lp = low[i]
                        af = min(af + acceleration, maximum)
                    if high[i-1] > psar[i]:
                        psar[i] = high[i-1]
                    if high[i-2] > psar[i]:
                        psar[i] = high[i-2]
            
            if bull:
                psarbull[i] = psar[i]
            else:
                psarbear[i] = psar[i]
        
        return pd.Series(psar, index=close.index)
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate Parabolic SAR, EMA, ADX, and other indicators for all symbols.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary of calculated indicators for each symbol
        """
        indicators = {}
        
        # Get parameters
        sar_acceleration = self.parameters.get("sar_acceleration", 0.02)
        sar_maximum = self.parameters.get("sar_maximum", 0.2)
        ema_period = self.parameters.get("ema_filter_period", 50)
        adx_filter = self.parameters.get("adx_filter", True)
        adx_period = self.parameters.get("adx_period", 14)
        atr_period = self.parameters.get("atr_period", 14)
        
        for symbol, df in data.items():
            # Ensure required columns exist
            if not all(col in df.columns for col in ['high', 'low', 'close']):
                logger.warning(f"Required price columns not found for {symbol}")
                continue
            
            try:
                # Calculate Parabolic SAR
                psar = self.calculate_parabolic_sar(
                    df['high'], df['low'], df['close'], 
                    acceleration=sar_acceleration, maximum=sar_maximum
                )
                
                # Calculate EMA filter if specified
                ema_filter = None
                if ema_period > 0:
                    ema_filter = df['close'].ewm(span=ema_period, adjust=False).mean()
                
                # Calculate ATR
                high_low = df['high'] - df['low']
                high_close_prev = np.abs(df['high'] - df['close'].shift(1))
                low_close_prev = np.abs(df['low'] - df['close'].shift(1))
                tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
                atr = tr.rolling(window=atr_period).mean()
                
                # Calculate ADX if filter is enabled
                adx = None
                plus_di = None
                minus_di = None
                
                if adx_filter:
                    # Calculate Directional Movement
                    up_move = df['high'] - df['high'].shift(1)
                    down_move = df['low'].shift(1) - df['low']
                    
                    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
                    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
                    
                    # Calculate smoothed DM
                    plus_di = pd.Series(plus_dm).rolling(window=adx_period).mean() / atr * 100
                    minus_di = pd.Series(minus_dm).rolling(window=adx_period).mean() / atr * 100
                    
                    # Calculate DX and ADX
                    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
                    adx = pd.Series(dx).rolling(window=adx_period).mean()
                
                # Store indicators
                indicators[symbol] = {
                    "psar": pd.DataFrame({"psar": psar}),
                    "atr": pd.DataFrame({"atr": atr})
                }
                
                if ema_filter is not None:
                    indicators[symbol]["ema_filter"] = pd.DataFrame({"ema_filter": ema_filter})
                
                if adx_filter:
                    indicators[symbol]["adx"] = pd.DataFrame({"adx": adx})
                    indicators[symbol]["plus_di"] = pd.DataFrame({"plus_di": plus_di})
                    indicators[symbol]["minus_di"] = pd.DataFrame({"minus_di": minus_di})
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
        
        return indicators
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate buy/sell signals based on Parabolic SAR.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Get parameters
        ema_period = self.parameters.get("ema_filter_period", 50)
        adx_filter = self.parameters.get("adx_filter", True)
        adx_threshold = self.parameters.get("adx_threshold", 25)
        stop_loss_atr_multiple = self.parameters.get("stop_loss_atr_multiple", 2.0)
        take_profit_atr_multiple = self.parameters.get("take_profit_atr_multiple", 4.0)
        
        # Calculate indicators
        indicators = self.calculate_indicators(data)
        
        # Generate signals
        signals = {}
        
        for symbol, symbol_indicators in indicators.items():
            try:
                # Get latest data points
                latest_data = data[symbol].iloc[-1]
                prev_data = data[symbol].iloc[-2]
                latest_price = latest_data['close']
                prev_price = prev_data['close']
                latest_timestamp = latest_data.name if isinstance(latest_data.name, datetime) else datetime.now()
                
                # Get latest indicator values
                latest_psar = symbol_indicators["psar"].iloc[-1]["psar"]
                prev_psar = symbol_indicators["psar"].iloc[-2]["psar"]
                latest_atr = symbol_indicators["atr"].iloc[-1]["atr"]
                
                # Check EMA filter if specified
                ema_filter_ok = True
                if ema_period > 0 and "ema_filter" in symbol_indicators:
                    ema_value = symbol_indicators["ema_filter"].iloc[-1]["ema_filter"]
                    ema_filter_ok = latest_price > ema_value
                
                # Check ADX filter if enabled
                adx_filter_ok = True
                if adx_filter and "adx" in symbol_indicators:
                    adx_value = symbol_indicators["adx"].iloc[-1]["adx"]
                    plus_di = symbol_indicators["plus_di"].iloc[-1]["plus_di"]
                    minus_di = symbol_indicators["minus_di"].iloc[-1]["minus_di"]
                    
                    adx_filter_ok = adx_value > adx_threshold
                
                # Generate signal based on Parabolic SAR
                signal_type = None
                confidence = 0.0
                
                # Detect SAR crossover - bullish when price crosses above SAR
                bullish_cross = prev_price < prev_psar and latest_price > latest_psar
                
                # Detect SAR crossover - bearish when price crosses below SAR
                bearish_cross = prev_price > prev_psar and latest_price < latest_psar
                
                if bullish_cross and ema_filter_ok and adx_filter_ok:
                    # Bullish SAR cross - buy signal
                    signal_type = SignalType.BUY
                    
                    # Base confidence
                    confidence = 0.7
                    
                    # Calculate stop loss and take profit based on ATR
                    stop_loss = latest_price - (latest_atr * stop_loss_atr_multiple)
                    take_profit = latest_price + (latest_atr * take_profit_atr_multiple)
                    
                elif bearish_cross and not ema_filter_ok and adx_filter_ok:
                    # Bearish SAR cross - sell signal
                    signal_type = SignalType.SELL
                    
                    # Base confidence
                    confidence = 0.7
                    
                    # Calculate stop loss and take profit based on ATR
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
                            "psar": latest_psar,
                            "atr": latest_atr,
                            "strategy_type": "trend_following"
                        }
                    )
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals 