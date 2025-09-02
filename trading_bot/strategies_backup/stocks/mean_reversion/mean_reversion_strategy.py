#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mean Reversion Strategy Module

This module implements various mean reversion trading strategies.
Mean reversion strategies aim to capitalize on asset prices returning
to their historical mean or average value after deviating significantly.
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

class RSIMeanReversionStrategy(StrategyOptimizable):
    """
    RSI (Relative Strength Index) based mean reversion strategy.
    
    This strategy generates buy signals when RSI is oversold and
    sell signals when RSI is overbought, based on the premise that
    extreme RSI values tend to revert back to normal levels.
    
    Key features:
    - Uses RSI to identify overbought and oversold conditions in price movements
    - Implements optional moving average filter to trade only in the direction of the primary trend
    - Incorporates volume filters to ensure sufficient market participation
    - Features ATR-based position sizing and risk management
    - Includes automated exit logic based on RSI returning to neutral levels
    - Offers trailing stop loss capability to capture extended price moves
    
    Trading logic:
    - Buy signals: Generated when RSI moves below the oversold threshold and then begins to rise
      indicating a potential reversal from oversold conditions
    - Sell signals: Generated when RSI moves above the overbought threshold and then begins to fall
      indicating a potential reversal from overbought conditions
    - Trend filter: Optional filter to only take trades in the direction of the longer-term trend
      as defined by price position relative to a moving average
    - Exit logic: Positions are closed when RSI crosses a neutral level (typically 50) or
      when stop loss or take profit levels are reached
    
    Ideal market conditions:
    - Range-bound markets with defined upper and lower boundaries
    - Assets with cyclical price patterns and mean-reverting behavior
    - Periods of price oscillation rather than strong directional trends
    - Markets with sufficient volatility to produce tradable overbought/oversold conditions
    
    Limitations:
    - Underperforms in strongly trending markets where overbought/oversold conditions persist
    - Can generate false signals during trend changes or high volatility environments
    - May struggle during periods of structural market changes or regime shifts
    - Requires careful parameter optimization based on each asset's volatility characteristics
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize RSI Mean Reversion strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters
        default_params = {
            "rsi_period": 14,
            "oversold_threshold": 30,
            "overbought_threshold": 70,
            "ma_filter_period": 200,  # Moving average filter period
            "use_ma_filter": True,    # Use moving average filter for trend direction
            "ma_filter_type": "sma",  # SMA or EMA
            "volume_filter": True,
            "min_volume_percentile": 40,
            "atr_period": 14,
            "stop_loss_atr_multiple": 2.0,
            "take_profit_atr_multiple": 2.0,
            "trailing_stop": True,
            "trailing_stop_activation_percent": 1.0,
            "exit_rsi_level": 50      # Exit when RSI crosses this level
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        logger.info(f"Initialized RSI Mean Reversion strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "rsi_period": [7, 14, 21],
            "oversold_threshold": [20, 25, 30, 35],
            "overbought_threshold": [65, 70, 75, 80],
            "ma_filter_period": [50, 100, 200],
            "use_ma_filter": [True, False],
            "ma_filter_type": ["sma", "ema"],
            "volume_filter": [True, False],
            "min_volume_percentile": [30, 40, 50],
            "atr_period": [10, 14, 21],
            "stop_loss_atr_multiple": [1.5, 2.0, 2.5, 3.0],
            "take_profit_atr_multiple": [1.0, 1.5, 2.0, 2.5],
            "trailing_stop": [True, False],
            "trailing_stop_activation_percent": [0.5, 1.0, 1.5],
            "exit_rsi_level": [45, 50, 55]
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate the Relative Strength Index (RSI).
        
        Args:
            prices: Series of price data
            period: RSI period
            
        Returns:
            Series of RSI values
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
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate RSI and other indicators for all symbols.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary of calculated indicators for each symbol
        """
        indicators = {}
        
        # Get parameters
        rsi_period = self.parameters.get("rsi_period", 14)
        ma_filter_period = self.parameters.get("ma_filter_period", 200)
        ma_filter_type = self.parameters.get("ma_filter_type", "sma")
        atr_period = self.parameters.get("atr_period", 14)
        
        for symbol, df in data.items():
            # Ensure required columns exist
            if not all(col in df.columns for col in ['close']):
                logger.warning(f"Required price columns not found for {symbol}")
                continue
            
            try:
                # Calculate RSI
                rsi = self._calculate_rsi(df['close'], period=rsi_period)
                
                # Calculate moving average for trend filter
                if ma_filter_type.lower() == "sma":
                    ma = df['close'].rolling(window=ma_filter_period).mean()
                else:  # EMA
                    ma = df['close'].ewm(span=ma_filter_period, adjust=False).mean()
                
                # Calculate trend direction (1 = uptrend, -1 = downtrend)
                trend = np.where(df['close'] > ma, 1, -1)
                
                # Calculate ATR for stop loss and take profit
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
                
                # Store indicators
                indicators[symbol] = {
                    "rsi": pd.DataFrame({"rsi": rsi}),
                    "ma": pd.DataFrame({"ma": ma}),
                    "trend": pd.DataFrame({"trend": trend}),
                    "atr": pd.DataFrame({"atr": atr})
                }
                
                if volume_percentile is not None:
                    indicators[symbol]["volume_percentile"] = pd.DataFrame({"volume_percentile": volume_percentile})
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
        
        return indicators
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate buy/sell signals based on RSI mean reversion.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Get parameters
        oversold = self.parameters.get("oversold_threshold", 30)
        overbought = self.parameters.get("overbought_threshold", 70)
        use_ma_filter = self.parameters.get("use_ma_filter", True)
        volume_filter = self.parameters.get("volume_filter", True)
        min_volume_percentile = self.parameters.get("min_volume_percentile", 40)
        stop_loss_atr_multiple = self.parameters.get("stop_loss_atr_multiple", 2.0)
        take_profit_atr_multiple = self.parameters.get("take_profit_atr_multiple", 2.0)
        trailing_stop = self.parameters.get("trailing_stop", True)
        trailing_activation = self.parameters.get("trailing_stop_activation_percent", 1.0)
        
        # Calculate indicators
        indicators = self.calculate_indicators(data)
        
        # Generate signals
        signals = {}
        
        for symbol, symbol_indicators in indicators.items():
            try:
                # Get latest data
                latest_data = data[symbol].iloc[-1]
                prev_data = data[symbol].iloc[-2]
                latest_price = latest_data['close']
                latest_timestamp = latest_data.name if isinstance(latest_data.name, datetime) else datetime.now()
                
                # Get indicator values
                latest_rsi = symbol_indicators["rsi"].iloc[-1]["rsi"]
                prev_rsi = symbol_indicators["rsi"].iloc[-2]["rsi"]
                latest_trend = symbol_indicators["trend"].iloc[-1]["trend"] if use_ma_filter else 1
                latest_atr = symbol_indicators["atr"].iloc[-1]["atr"]
                
                # Check volume filter if enabled
                volume_ok = True
                if volume_filter and "volume_percentile" in symbol_indicators:
                    vol_pct = symbol_indicators["volume_percentile"].iloc[-1]["volume_percentile"]
                    volume_ok = vol_pct >= min_volume_percentile
                
                # Generate signal based on RSI
                signal_type = None
                confidence = 0.0
                
                # Buy signal: RSI crosses below oversold threshold (upward)
                if prev_rsi <= oversold < latest_rsi and volume_ok:
                    # Only buy in uptrend if MA filter is enabled
                    if not use_ma_filter or latest_trend > 0:
                        signal_type = SignalType.BUY
                        
                        # Calculate confidence based on:
                        # 1. Distance from oversold level
                        oversold_distance = oversold - min(oversold, prev_rsi)
                        distance_confidence = min(0.3, oversold_distance / 10)
                        
                        # 2. RSI momentum (strength of bounce)
                        rsi_momentum = latest_rsi - prev_rsi
                        momentum_confidence = min(0.3, rsi_momentum / 10)
                        
                        # 3. Trend alignment
                        trend_confidence = 0.2 if latest_trend > 0 else 0.0
                        
                        # 4. Volume strength
                        volume_confidence = 0.0
                        if "volume_percentile" in symbol_indicators:
                            vol_pct = symbol_indicators["volume_percentile"].iloc[-1]["volume_percentile"]
                            volume_confidence = min(0.2, vol_pct / 100)
                        
                        confidence = min(0.9, distance_confidence + momentum_confidence + trend_confidence + volume_confidence)
                        
                        # Calculate stop loss and take profit based on ATR
                        stop_loss = latest_price - (latest_atr * stop_loss_atr_multiple)
                        take_profit = latest_price + (latest_atr * take_profit_atr_multiple)
                
                # Sell signal: RSI crosses above overbought threshold (downward)
                elif prev_rsi >= overbought > latest_rsi and volume_ok:
                    # Only sell in downtrend if MA filter is enabled
                    if not use_ma_filter or latest_trend < 0:
                        signal_type = SignalType.SELL
                        
                        # Calculate confidence based on:
                        # 1. Distance from overbought level
                        overbought_distance = max(overbought, prev_rsi) - overbought
                        distance_confidence = min(0.3, overbought_distance / 10)
                        
                        # 2. RSI momentum (strength of reversal)
                        rsi_momentum = prev_rsi - latest_rsi
                        momentum_confidence = min(0.3, rsi_momentum / 10)
                        
                        # 3. Trend alignment
                        trend_confidence = 0.2 if latest_trend < 0 else 0.0
                        
                        # 4. Volume strength
                        volume_confidence = 0.0
                        if "volume_percentile" in symbol_indicators:
                            vol_pct = symbol_indicators["volume_percentile"].iloc[-1]["volume_percentile"]
                            volume_confidence = min(0.2, vol_pct / 100)
                        
                        confidence = min(0.9, distance_confidence + momentum_confidence + trend_confidence + volume_confidence)
                        
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
                            "rsi": latest_rsi,
                            "trend": latest_trend,
                            "atr": latest_atr,
                            "trailing_stop": trailing_stop,
                            "trailing_activation": trailing_activation,
                            "strategy_type": "mean_reversion"
                        }
                    )
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals


class BollingerBandMeanReversionStrategy(StrategyOptimizable):
    """
    Bollinger Band based mean reversion strategy.
    
    This strategy generates buy signals when price reaches the lower band
    and sell signals when price reaches the upper band, based on the premise
    that prices tend to return to their mean when they deviate too far.
    
    Key features:
    - Uses Bollinger Bands to identify statistical extremes in price movements
    - Incorporates band width analysis to detect volatility contractions and expansions
    - Features band squeeze detection to identify potential breakout opportunities
    - Implements volume confirmation to validate reversal signals
    - Includes dynamic risk management with ATR-based stop loss and take profit levels
    - Offers automatic exit when price crosses the middle band (mean reversion target)
    
    Trading logic:
    - Buy signals: Generated when price touches or penetrates the lower band, indicating
      a statistically significant deviation below the mean with reversion potential
    - Sell signals: Generated when price touches or penetrates the upper band, indicating
      a statistically significant deviation above the mean with reversion potential
    - Band squeeze filter: Optional filter to identify periods of volatility contraction
      that often precede strong price movements
    - Risk management: Uses ATR to size positions and set stop loss levels proportional
      to current market volatility
    - Exit strategy: Takes profit when price reverts to the mean (middle band) or
      when trailing stops are triggered
    
    Ideal market conditions:
    - Range-bound markets with clear price boundaries
    - Assets that demonstrate statistical mean-reverting properties
    - Markets with cyclical volatility patterns (periods of contraction followed by expansion)
    - Liquid markets where price movements are smoother and more statistically predictable
    
    Limitations:
    - Can generate false signals during strong trending markets
    - Band penetrations may not lead to immediate reversals
    - Less effective during regime changes or market structure shifts
    - Performance varies based on band parameter selection (period and standard deviation)
    - May require different parameters for different volatility environments
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Bollinger Band Mean Reversion strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters
        default_params = {
            "bb_period": 20,
            "bb_std_dev": 2.0,
            "entry_threshold_pct": 0.0,  # Enter when price is exactly at the band
            "volume_filter": True,
            "min_volume_percentile": 40,
            "band_squeeze_filter": True,
            "squeeze_threshold": 0.5,  # 50% of the average width
            "atr_period": 14,
            "stop_loss_atr_multiple": 2.0,
            "take_profit_atr_multiple": 2.0,
            "trailing_stop": True,
            "trailing_stop_activation_percent": 1.0,
            "exit_on_middle_band": True  # Exit when price crosses middle band
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        logger.info(f"Initialized Bollinger Band Mean Reversion strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "bb_period": [10, 20, 30],
            "bb_std_dev": [1.5, 2.0, 2.5, 3.0],
            "entry_threshold_pct": [0.0, 0.1, 0.2],
            "volume_filter": [True, False],
            "min_volume_percentile": [30, 40, 50],
            "band_squeeze_filter": [True, False],
            "squeeze_threshold": [0.3, 0.5, 0.7],
            "atr_period": [10, 14, 21],
            "stop_loss_atr_multiple": [1.5, 2.0, 2.5, 3.0],
            "take_profit_atr_multiple": [1.0, 1.5, 2.0, 2.5],
            "trailing_stop": [True, False],
            "trailing_stop_activation_percent": [0.5, 1.0, 1.5],
            "exit_on_middle_band": [True, False]
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Series of price data
            period: Bollinger Band period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        # Calculate middle band (SMA)
        middle_band = prices.rolling(window=period).mean()
        
        # Calculate standard deviation
        std = prices.rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    def _is_band_squeezing(self, upper_band: pd.Series, lower_band: pd.Series, 
                          lookback: int = 20, threshold: float = 0.5) -> bool:
        """
        Detect if Bollinger Bands are squeezing (narrowing).
        
        Args:
            upper_band: Series of upper band values
            lower_band: Series of lower band values
            lookback: Lookback period for detecting squeeze
            threshold: Threshold for determining squeeze (as a proportion of average width)
            
        Returns:
            Boolean indicating if bands are squeezing
        """
        if len(upper_band) < lookback * 2:
            return False
        
        # Calculate band width
        band_width = upper_band - lower_band
        
        # Calculate average band width over last N periods
        avg_width = band_width[-lookback*2:-lookback].mean()
        current_width = band_width[-1]
        
        # Check if current width is below threshold of average width
        return current_width < (avg_width * threshold)
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate Bollinger Bands and other indicators for all symbols.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary of calculated indicators for each symbol
        """
        indicators = {}
        
        # Get parameters
        bb_period = self.parameters.get("bb_period", 20)
        bb_std_dev = self.parameters.get("bb_std_dev", 2.0)
        atr_period = self.parameters.get("atr_period", 14)
        squeeze_threshold = self.parameters.get("squeeze_threshold", 0.5)
        
        for symbol, df in data.items():
            # Ensure required columns exist
            if not all(col in df.columns for col in ['close']):
                logger.warning(f"Required price columns not found for {symbol}")
                continue
            
            try:
                # Calculate Bollinger Bands
                upper_band, middle_band, lower_band = self._calculate_bollinger_bands(
                    df['close'], period=bb_period, std_dev=bb_std_dev
                )
                
                # Calculate band width
                band_width = upper_band - lower_band
                
                # Calculate band width percentile (normalized width)
                band_width_percentile = band_width.rolling(window=100).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
                )
                
                # Calculate price position within bands
                price_position = (df['close'] - lower_band) / (upper_band - lower_band)
                
                # Calculate ATR for stop loss and take profit
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
                
                # Detect band squeeze
                is_squeezing = self._is_band_squeezing(upper_band, lower_band, 
                                                     lookback=bb_period, 
                                                     threshold=squeeze_threshold)
                
                # Store indicators
                indicators[symbol] = {
                    "upper_band": pd.DataFrame({"upper_band": upper_band}),
                    "middle_band": pd.DataFrame({"middle_band": middle_band}),
                    "lower_band": pd.DataFrame({"lower_band": lower_band}),
                    "band_width": pd.DataFrame({"band_width": band_width}),
                    "band_width_percentile": pd.DataFrame({"band_width_percentile": band_width_percentile}),
                    "price_position": pd.DataFrame({"price_position": price_position}),
                    "atr": pd.DataFrame({"atr": atr}),
                    "is_squeezing": is_squeezing
                }
                
                if volume_percentile is not None:
                    indicators[symbol]["volume_percentile"] = pd.DataFrame({"volume_percentile": volume_percentile})
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
        
        return indicators
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate buy/sell signals based on Bollinger Band mean reversion.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Get parameters
        entry_threshold = self.parameters.get("entry_threshold_pct", 0.0)
        volume_filter = self.parameters.get("volume_filter", True)
        min_volume_percentile = self.parameters.get("min_volume_percentile", 40)
        band_squeeze_filter = self.parameters.get("band_squeeze_filter", True)
        stop_loss_atr_multiple = self.parameters.get("stop_loss_atr_multiple", 2.0)
        take_profit_atr_multiple = self.parameters.get("take_profit_atr_multiple", 2.0)
        trailing_stop = self.parameters.get("trailing_stop", True)
        trailing_activation = self.parameters.get("trailing_stop_activation_percent", 1.0)
        
        # Calculate indicators
        indicators = self.calculate_indicators(data)
        
        # Generate signals
        signals = {}
        
        for symbol, symbol_indicators in indicators.items():
            try:
                # Get latest data
                latest_data = data[symbol].iloc[-1]
                prev_data = data[symbol].iloc[-2]
                latest_price = latest_data['close']
                latest_timestamp = latest_data.name if isinstance(latest_data.name, datetime) else datetime.now()
                
                # Get indicator values
                latest_upper = symbol_indicators["upper_band"].iloc[-1]["upper_band"]
                latest_middle = symbol_indicators["middle_band"].iloc[-1]["middle_band"]
                latest_lower = symbol_indicators["lower_band"].iloc[-1]["lower_band"]
                latest_atr = symbol_indicators["atr"].iloc[-1]["atr"]
                is_squeezing = symbol_indicators["is_squeezing"]
                
                # Calculate band penetration thresholds with entry threshold offset
                lower_entry = latest_lower * (1 + entry_threshold/100)
                upper_entry = latest_upper * (1 - entry_threshold/100)
                
                # Check volume filter if enabled
                volume_ok = True
                if volume_filter and "volume_percentile" in symbol_indicators:
                    vol_pct = symbol_indicators["volume_percentile"].iloc[-1]["volume_percentile"]
                    volume_ok = vol_pct >= min_volume_percentile
                
                # Check band squeeze filter if enabled
                squeeze_ok = True
                if band_squeeze_filter:
                    squeeze_ok = is_squeezing
                
                # Generate signal based on Bollinger Bands
                signal_type = None
                confidence = 0.0
                
                # Buy signal: Price crosses below lower band
                if latest_price <= lower_entry and prev_data['close'] > lower_entry and volume_ok and squeeze_ok:
                    signal_type = SignalType.BUY
                    
                    # Calculate confidence based on:
                    # 1. Distance from lower band
                    band_distance_pct = (lower_entry - latest_price) / latest_price * 100
                    distance_confidence = min(0.3, band_distance_pct * 0.1)
                    
                    # 2. Band width (narrower bands = more reliable signals)
                    band_width_pct = symbol_indicators["band_width_percentile"].iloc[-1]["band_width_percentile"]
                    band_confidence = min(0.3, 0.3 - (band_width_pct / 200))  # Lower percentile = higher confidence
                    
                    # 3. Volume strength
                    volume_confidence = 0.0
                    if "volume_percentile" in symbol_indicators:
                        vol_pct = symbol_indicators["volume_percentile"].iloc[-1]["volume_percentile"]
                        volume_confidence = min(0.2, vol_pct / 100)
                    
                    # 4. Squeeze confidence
                    squeeze_confidence = 0.2 if is_squeezing else 0.0
                    
                    confidence = min(0.9, distance_confidence + band_confidence + volume_confidence + squeeze_confidence)
                    
                    # Calculate stop loss and take profit based on ATR
                    stop_loss = latest_price - (latest_atr * stop_loss_atr_multiple)
                    take_profit = latest_middle  # Target the middle band
                
                # Sell signal: Price crosses above upper band
                elif latest_price >= upper_entry and prev_data['close'] < upper_entry and volume_ok and squeeze_ok:
                    signal_type = SignalType.SELL
                    
                    # Calculate confidence based on:
                    # 1. Distance from upper band
                    band_distance_pct = (latest_price - upper_entry) / latest_price * 100
                    distance_confidence = min(0.3, band_distance_pct * 0.1)
                    
                    # 2. Band width
                    band_width_pct = symbol_indicators["band_width_percentile"].iloc[-1]["band_width_percentile"]
                    band_confidence = min(0.3, 0.3 - (band_width_pct / 200))
                    
                    # 3. Volume strength
                    volume_confidence = 0.0
                    if "volume_percentile" in symbol_indicators:
                        vol_pct = symbol_indicators["volume_percentile"].iloc[-1]["volume_percentile"]
                        volume_confidence = min(0.2, vol_pct / 100)
                    
                    # 4. Squeeze confidence
                    squeeze_confidence = 0.2 if is_squeezing else 0.0
                    
                    confidence = min(0.9, distance_confidence + band_confidence + volume_confidence + squeeze_confidence)
                    
                    # Calculate stop loss and take profit based on ATR
                    stop_loss = latest_price + (latest_atr * stop_loss_atr_multiple)
                    take_profit = latest_middle  # Target the middle band
                
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
                            "upper_band": latest_upper,
                            "middle_band": latest_middle,
                            "lower_band": latest_lower,
                            "atr": latest_atr,
                            "is_squeezing": is_squeezing,
                            "trailing_stop": trailing_stop,
                            "trailing_activation": trailing_activation,
                            "strategy_type": "mean_reversion"
                        }
                    )
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals


class StatisticalMeanReversionStrategy(StrategyOptimizable):
    """
    Statistical mean reversion strategy based on z-scores.
    
    This strategy calculates z-scores of prices relative to a moving average
    and generates signals when prices deviate significantly from the mean,
    assuming they will revert back to normal levels.
    
    Key features:
    - Uses statistical z-scores to quantify deviations from the mean
    - Provides options for different central tendency measures (mean or median)
    - Supports both simple and exponential moving averages for baseline calculation
    - Implements configurable entry and exit thresholds based on standard deviations
    - Incorporates volume filters to validate statistical signals
    - Features adaptive risk management based on recent volatility measures
    
    Trading logic:
    - Buy signals: Generated when z-score crosses back above a negative threshold
      (e.g., -2.0), indicating a potential upward reversion after significant downside deviation
    - Sell signals: Generated when z-score crosses back below a positive threshold
      (e.g., +2.0), indicating a potential downward reversion after significant upside deviation
    - Exit logic: Positions are closed when z-score approaches the mean (configurable threshold)
      or when stop loss/take profit levels are reached
    - Take profit: Typically set at the mean/median price level, representing complete reversion
    
    Ideal market conditions:
    - Markets with identifiable statistical properties and stable variance
    - Assets that display predictable mean-reverting behavior
    - Instruments with sufficient history to calculate meaningful statistical measures
    - Environments with stable correlation structures and market regimes
    
    Limitations:
    - Relies on the assumption of price normality which may not always hold
    - Sensitive to outliers and regime changes that affect statistical properties
    - May generate signals too early during sustained trending movements
    - Requires careful selection of lookback periods to balance sensitivity and reliability
    - Statistical parameters need periodic recalibration as market conditions evolve
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Statistical Mean Reversion strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters
        default_params = {
            "lookback_period": 100,  # Period for calculating mean and std dev
            "ma_type": "sma",        # SMA or EMA
            "entry_z_score": 2.0,    # Z-score threshold for entry
            "exit_z_score": 0.5,     # Z-score threshold for exit
            "volume_filter": True,
            "min_volume_percentile": 40,
            "use_median": False,     # Use median instead of mean
            "atr_period": 14,
            "stop_loss_atr_multiple": 2.0,
            "take_profit_atr_multiple": 1.5,
            "trailing_stop": True,
            "trailing_stop_activation_percent": 1.0
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        logger.info(f"Initialized Statistical Mean Reversion strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "lookback_period": [50, 100, 200],
            "ma_type": ["sma", "ema"],
            "entry_z_score": [1.5, 2.0, 2.5, 3.0],
            "exit_z_score": [0.0, 0.5, 1.0],
            "volume_filter": [True, False],
            "min_volume_percentile": [30, 40, 50],
            "use_median": [True, False],
            "atr_period": [10, 14, 21],
            "stop_loss_atr_multiple": [1.5, 2.0, 2.5],
            "take_profit_atr_multiple": [1.0, 1.5, 2.0],
            "trailing_stop": [True, False],
            "trailing_stop_activation_percent": [0.5, 1.0, 1.5]
        }
    
    def _calculate_z_score(self, prices: pd.Series, lookback: int = 100, ma_type: str = "sma", use_median: bool = False) -> pd.Series:
        """
        Calculate z-score of price relative to its moving average.
        
        Args:
            prices: Series of price data
            lookback: Lookback period for calculating mean and std dev
            ma_type: Type of moving average (sma or ema)
            use_median: Use median instead of mean
            
        Returns:
            Series of z-scores
        """
        # Calculate mean
        if ma_type.lower() == "sma":
            if use_median:
                # Use rolling median
                ma = prices.rolling(window=lookback).median()
            else:
                # Use rolling mean (SMA)
                ma = prices.rolling(window=lookback).mean()
        else:
            # Use exponential moving average (EMA)
            ma = prices.ewm(span=lookback, adjust=False).mean()
        
        # Calculate standard deviation
        std = prices.rolling(window=lookback).std()
        
        # Calculate z-score
        z_score = (prices - ma) / std
        
        return z_score
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate z-scores and other indicators for all symbols.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary of calculated indicators for each symbol
        """
        indicators = {}
        
        # Get parameters
        lookback = self.parameters.get("lookback_period", 100)
        ma_type = self.parameters.get("ma_type", "sma")
        use_median = self.parameters.get("use_median", False)
        atr_period = self.parameters.get("atr_period", 14)
        
        for symbol, df in data.items():
            # Ensure required columns exist
            if not all(col in df.columns for col in ['close']):
                logger.warning(f"Required price columns not found for {symbol}")
                continue
            
            try:
                # Calculate z-score
                z_score = self._calculate_z_score(
                    df['close'], lookback=lookback, ma_type=ma_type, use_median=use_median
                )
                
                # Calculate moving average
                if ma_type.lower() == "sma":
                    if use_median:
                        ma = df['close'].rolling(window=lookback).median()
                    else:
                        ma = df['close'].rolling(window=lookback).mean()
                else:
                    ma = df['close'].ewm(span=lookback, adjust=False).mean()
                
                # Calculate ATR for stop loss and take profit
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
                
                # Store indicators
                indicators[symbol] = {
                    "z_score": pd.DataFrame({"z_score": z_score}),
                    "ma": pd.DataFrame({"ma": ma}),
                    "atr": pd.DataFrame({"atr": atr})
                }
                
                if volume_percentile is not None:
                    indicators[symbol]["volume_percentile"] = pd.DataFrame({"volume_percentile": volume_percentile})
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
        
        return indicators
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate buy/sell signals based on statistical mean reversion.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Get parameters
        entry_z_score = self.parameters.get("entry_z_score", 2.0)
        exit_z_score = self.parameters.get("exit_z_score", 0.5)
        volume_filter = self.parameters.get("volume_filter", True)
        min_volume_percentile = self.parameters.get("min_volume_percentile", 40)
        stop_loss_atr_multiple = self.parameters.get("stop_loss_atr_multiple", 2.0)
        take_profit_atr_multiple = self.parameters.get("take_profit_atr_multiple", 1.5)
        trailing_stop = self.parameters.get("trailing_stop", True)
        trailing_activation = self.parameters.get("trailing_stop_activation_percent", 1.0)
        
        # Calculate indicators
        indicators = self.calculate_indicators(data)
        
        # Generate signals
        signals = {}
        
        for symbol, symbol_indicators in indicators.items():
            try:
                # Get latest data
                latest_data = data[symbol].iloc[-1]
                prev_data = data[symbol].iloc[-2]
                latest_price = latest_data['close']
                latest_timestamp = latest_data.name if isinstance(latest_data.name, datetime) else datetime.now()
                
                # Get indicator values
                latest_z_score = symbol_indicators["z_score"].iloc[-1]["z_score"]
                prev_z_score = symbol_indicators["z_score"].iloc[-2]["z_score"]
                latest_ma = symbol_indicators["ma"].iloc[-1]["ma"]
                latest_atr = symbol_indicators["atr"].iloc[-1]["atr"]
                
                # Check volume filter if enabled
                volume_ok = True
                if volume_filter and "volume_percentile" in symbol_indicators:
                    vol_pct = symbol_indicators["volume_percentile"].iloc[-1]["volume_percentile"]
                    volume_ok = vol_pct >= min_volume_percentile
                
                # Generate signal based on z-score
                signal_type = None
                confidence = 0.0
                
                # Buy signal: z-score crosses below negative entry threshold (upward)
                if prev_z_score <= -entry_z_score < latest_z_score and latest_z_score < -exit_z_score and volume_ok:
                    signal_type = SignalType.BUY
                    
                    # Calculate confidence based on:
                    # 1. Z-score extreme level
                    z_extreme = min(-entry_z_score, prev_z_score)
                    z_confidence = min(0.3, abs(z_extreme) / 5)
                    
                    # 2. Z-score momentum (strength of reversal)
                    z_momentum = latest_z_score - prev_z_score
                    momentum_confidence = min(0.3, z_momentum / 1.0)
                    
                    # 3. Volume strength
                    volume_confidence = 0.0
                    if "volume_percentile" in symbol_indicators:
                        vol_pct = symbol_indicators["volume_percentile"].iloc[-1]["volume_percentile"]
                        volume_confidence = min(0.2, vol_pct / 100)
                    
                    # 4. Price vs MA gap
                    price_gap = (latest_price / latest_ma - 1) * 100
                    gap_confidence = min(0.2, abs(price_gap) / 10)
                    
                    confidence = min(0.9, z_confidence + momentum_confidence + volume_confidence + gap_confidence)
                    
                    # Calculate stop loss and take profit based on ATR
                    stop_loss = latest_price - (latest_atr * stop_loss_atr_multiple)
                    take_profit = latest_ma  # Target the mean
                
                # Sell signal: z-score crosses above positive entry threshold (downward)
                elif prev_z_score >= entry_z_score > latest_z_score and latest_z_score > exit_z_score and volume_ok:
                    signal_type = SignalType.SELL
                    
                    # Calculate confidence based on:
                    # 1. Z-score extreme level
                    z_extreme = max(entry_z_score, prev_z_score)
                    z_confidence = min(0.3, abs(z_extreme) / 5)
                    
                    # 2. Z-score momentum (strength of reversal)
                    z_momentum = prev_z_score - latest_z_score
                    momentum_confidence = min(0.3, z_momentum / 1.0)
                    
                    # 3. Volume strength
                    volume_confidence = 0.0
                    if "volume_percentile" in symbol_indicators:
                        vol_pct = symbol_indicators["volume_percentile"].iloc[-1]["volume_percentile"]
                        volume_confidence = min(0.2, vol_pct / 100)
                    
                    # 4. Price vs MA gap
                    price_gap = (latest_price / latest_ma - 1) * 100
                    gap_confidence = min(0.2, abs(price_gap) / 10)
                    
                    confidence = min(0.9, z_confidence + momentum_confidence + volume_confidence + gap_confidence)
                    
                    # Calculate stop loss and take profit based on ATR
                    stop_loss = latest_price + (latest_atr * stop_loss_atr_multiple)
                    take_profit = latest_ma  # Target the mean
                
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
                            "z_score": latest_z_score,
                            "ma": latest_ma,
                            "atr": latest_atr,
                            "trailing_stop": trailing_stop,
                            "trailing_activation": trailing_activation,
                            "strategy_type": "mean_reversion"
                        }
                    )
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals


class MeanReversionStrategy(StrategyOptimizable):
    """
    Mean Reversion Strategy
    
    This class provides a unified interface for mean reversion strategies.
    By default, it uses the RSI-based mean reversion strategy implementation
    but can be configured to use other mean reversion approaches like
    Bollinger Bands or statistical z-score approaches.
    
    The strategy aims to capitalize on the tendency of asset prices to
    revert to their mean after significant deviations.
    """
    
    def __init__(
        self,
        name: str = "MeanReversionStrategy",
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Mean Reversion strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters
        default_params = {
            "strategy_type": "rsi",  # Options: "rsi", "bollinger", "statistical"
            # RSI parameters
            "rsi_period": 14,
            "oversold_threshold": 30,
            "overbought_threshold": 70,
            # Bollinger Band parameters
            "bb_period": 20,
            "bb_std_dev": 2.0,
            # Statistical parameters
            "lookback_period": 100,
            "z_score_entry": 2.0,
            "z_score_exit": 0.0,
            # Common parameters
            "atr_period": 14,
            "stop_loss_atr_multiple": 2.0,
            "take_profit_atr_multiple": 2.0,
            "trailing_stop": True,
            "volume_filter": True
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        # Initialize the appropriate sub-strategy based on strategy_type
        strategy_type = default_params.get("strategy_type", "rsi").lower()
        
        if strategy_type == "rsi":
            self.strategy = RSIMeanReversionStrategy(
                name=f"{name}_RSI",
                parameters=default_params,
                metadata=metadata
            )
        elif strategy_type == "bollinger":
            self.strategy = BollingerBandMeanReversionStrategy(
                name=f"{name}_BollingerBand",
                parameters=default_params,
                metadata=metadata
            )
        elif strategy_type == "statistical":
            self.strategy = StatisticalMeanReversionStrategy(
                name=f"{name}_Statistical",
                parameters=default_params,
                metadata=metadata
            )
        else:
            logger.warning(f"Unknown strategy type: {strategy_type}. Using RSI as default.")
            self.strategy = RSIMeanReversionStrategy(
                name=f"{name}_RSI",
                parameters=default_params,
                metadata=metadata
            )
        
        logger.info(f"Initialized Mean Reversion strategy ({strategy_type}): {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        # Get parameter space based on strategy type
        strategy_type = self.parameters.get("strategy_type", "rsi").lower()
        
        # Common parameters
        common_params = {
            "strategy_type": ["rsi", "bollinger", "statistical"],
            "atr_period": [10, 14, 21],
            "stop_loss_atr_multiple": [1.5, 2.0, 2.5, 3.0],
            "take_profit_atr_multiple": [1.0, 1.5, 2.0, 2.5],
            "trailing_stop": [True, False],
            "volume_filter": [True, False]
        }
        
        # Get strategy-specific parameters
        if strategy_type == "rsi":
            strategy_params = {
                "rsi_period": [7, 14, 21],
                "oversold_threshold": [20, 25, 30, 35],
                "overbought_threshold": [65, 70, 75, 80]
            }
        elif strategy_type == "bollinger":
            strategy_params = {
                "bb_period": [10, 20, 30],
                "bb_std_dev": [1.5, 2.0, 2.5, 3.0]
            }
        elif strategy_type == "statistical":
            strategy_params = {
                "lookback_period": [50, 100, 200],
                "z_score_entry": [1.5, 2.0, 2.5, 3.0],
                "z_score_exit": [0.0, 0.5, 1.0]
            }
        else:
            strategy_params = {}
        
        # Combine parameters
        param_space = {**common_params, **strategy_params}
        return param_space
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate indicators by delegating to the selected strategy.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary of calculated indicators for each symbol
        """
        return self.strategy.calculate_indicators(data)
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate signals by delegating to the selected strategy.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        return self.strategy.generate_signals(data)
    
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Update strategy parameters.
        
        Args:
            parameters: New parameters
        """
        super().update_parameters(parameters)
        
        # Check if strategy_type has changed
        new_strategy_type = parameters.get("strategy_type")
        current_strategy_type = self.parameters.get("strategy_type")
        
        # If strategy type has changed, reinitialize the strategy
        if new_strategy_type and new_strategy_type != current_strategy_type:
            self.__init__(name=self.name, parameters=self.parameters, metadata=self.metadata)
        else:
            # Update the sub-strategy parameters
            self.strategy.update_parameters(parameters) 