#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Breakout Strategy Module

This module implements various breakout trading strategies.
Breakout strategies aim to identify and capitalize on price movements
beyond identified support and resistance levels.
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

class PriceChannelBreakoutStrategy(StrategyOptimizable):
    """
    Breakout strategy based on price channels.
    
    This strategy generates buy signals when price breaks above the highest high
    of a defined period, and sell signals when price breaks below the lowest low
    of a defined period.
    
    Key features:
    - Uses rolling highest highs and lowest lows to establish dynamic support/resistance channels
    - Applies confirmation rules to filter out false breakouts and whipsaws
    - Implements volume filters to verify breakout authenticity and strength
    - Incorporates volatility analysis to adapt to changing market conditions
    - Adjusts position sizing and risk parameters based on ATR volatility measures
    - Features both fixed and trailing stop mechanisms for risk management
    
    Trading logic:
    - Buy signals: Triggered when price closes above the highest high of the channel period
      with confirmation over specified number of bars and volume/volatility validation
    - Sell signals: Generated when price closes below the lowest low of the channel period
      with similar confirmation requirements and filters
    - Stop placement: Calculated dynamically using ATR to adapt to each asset's volatility profile
    - Take profit: Set as a multiple of the risk, creating a favorable risk-reward ratio
    
    Ideal market conditions:
    - Markets transitioning from consolidation to trending phases
    - Assets with well-defined trading ranges before breakouts
    - Sufficient volatility to generate meaningful price movements
    - Adequate liquidity and volume to support sustained moves after breakouts
    
    Limitations:
    - Susceptible to false breakouts in choppy or highly volatile markets
    - May have delayed entries due to confirmation requirements
    - Performance dependent on proper channel period selection for each asset
    - Requires careful parameter optimization across different market regimes
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Price Channel Breakout strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters
        default_params = {
            "channel_period": 20,
            "breakout_confirmation_bars": 1,
            "volume_filter": True,
            "min_volume_percentile": 70,
            "volatility_filter": True,
            "atr_period": 14,
            "min_atr_percentile": 50,
            "stop_loss_atr_multiple": 1.5,
            "take_profit_atr_multiple": 3.0,
            "trailing_stop": True,
            "trailing_stop_activation_percent": 1.0
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        logger.info(f"Initialized Price Channel Breakout strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "channel_period": [10, 20, 30, 40, 50],
            "breakout_confirmation_bars": [0, 1, 2],
            "volume_filter": [True, False],
            "min_volume_percentile": [50, 60, 70, 80],
            "volatility_filter": [True, False],
            "atr_period": [10, 14, 20],
            "min_atr_percentile": [30, 50, 70],
            "stop_loss_atr_multiple": [1.0, 1.5, 2.0, 2.5],
            "take_profit_atr_multiple": [2.0, 3.0, 4.0, 5.0],
            "trailing_stop": [True, False],
            "trailing_stop_activation_percent": [0.5, 1.0, 1.5, 2.0]
        }
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate price channels and related indicators for all symbols.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary of calculated indicators for each symbol
        """
        indicators = {}
        
        # Get parameters
        channel_period = self.parameters.get("channel_period", 20)
        atr_period = self.parameters.get("atr_period", 14)
        
        for symbol, df in data.items():
            # Ensure required columns exist
            if not all(col in df.columns for col in ['high', 'low', 'close']):
                logger.warning(f"Required price columns not found for {symbol}")
                continue
            
            try:
                # Calculate highest high and lowest low over the channel period
                upper_channel = df['high'].rolling(window=channel_period).max()
                lower_channel = df['low'].rolling(window=channel_period).min()
                
                # Calculate channel width
                channel_width = upper_channel - lower_channel
                
                # Calculate ATR for volatility assessment
                high_low = df['high'] - df['low']
                high_close_prev = np.abs(df['high'] - df['close'].shift(1))
                low_close_prev = np.abs(df['low'] - df['close'].shift(1))
                tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
                atr = tr.rolling(window=atr_period).mean()
                
                # Calculate ATR percentile over last 100 periods
                atr_percentile = atr.rolling(window=100).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
                )
                
                # Calculate volume percentile if volume data is available
                volume_percentile = None
                if 'volume' in df.columns:
                    volume_percentile = df['volume'].rolling(window=20).apply(
                        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
                    )
                
                # Calculate proximity to channel boundaries (for confidence calculation)
                price_position = (df['close'] - lower_channel) / (upper_channel - lower_channel)
                
                # Store indicators
                indicators[symbol] = {
                    "upper_channel": pd.DataFrame({"upper_channel": upper_channel}),
                    "lower_channel": pd.DataFrame({"lower_channel": lower_channel}),
                    "channel_width": pd.DataFrame({"channel_width": channel_width}),
                    "atr": pd.DataFrame({"atr": atr}),
                    "atr_percentile": pd.DataFrame({"atr_percentile": atr_percentile}),
                    "price_position": pd.DataFrame({"price_position": price_position})
                }
                
                if volume_percentile is not None:
                    indicators[symbol]["volume_percentile"] = pd.DataFrame({"volume_percentile": volume_percentile})
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
        
        return indicators
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate buy/sell signals based on price channel breakouts.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Get parameters
        channel_period = self.parameters.get("channel_period", 20)
        confirmation_bars = self.parameters.get("breakout_confirmation_bars", 1)
        volume_filter = self.parameters.get("volume_filter", True)
        min_volume_percentile = self.parameters.get("min_volume_percentile", 70)
        volatility_filter = self.parameters.get("volatility_filter", True)
        min_atr_percentile = self.parameters.get("min_atr_percentile", 50)
        stop_loss_atr_multiple = self.parameters.get("stop_loss_atr_multiple", 1.5)
        take_profit_atr_multiple = self.parameters.get("take_profit_atr_multiple", 3.0)
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
                latest_price = latest_data['close']
                latest_timestamp = latest_data.name if isinstance(latest_data.name, datetime) else datetime.now()
                
                # Get indicator values
                upper_channel = symbol_indicators["upper_channel"].iloc[-(confirmation_bars+1)]["upper_channel"]
                lower_channel = symbol_indicators["lower_channel"].iloc[-(confirmation_bars+1)]["lower_channel"]
                
                latest_atr = symbol_indicators["atr"].iloc[-1]["atr"]
                channel_width = symbol_indicators["channel_width"].iloc[-1]["channel_width"]
                price_position = symbol_indicators["price_position"].iloc[-1]["price_position"]
                
                # Check filters
                volume_ok = True
                if volume_filter and "volume_percentile" in symbol_indicators:
                    vol_pct = symbol_indicators["volume_percentile"].iloc[-1]["volume_percentile"]
                    volume_ok = vol_pct >= min_volume_percentile
                
                volatility_ok = True
                if volatility_filter:
                    atr_pct = symbol_indicators["atr_percentile"].iloc[-1]["atr_percentile"]
                    volatility_ok = atr_pct >= min_atr_percentile
                
                # Get price history for confirmation
                price_history = data[symbol].iloc[-(channel_period+confirmation_bars+1):]
                
                # Generate signal based on breakout
                signal_type = None
                confidence = 0.0
                
                # Check for upside breakout (confirmation_bars ago price was below the upper channel
                # and now it's above)
                if (confirmation_bars == 0 and latest_price > upper_channel) or \
                   (confirmation_bars > 0 and 
                    price_history['close'].iloc[-(confirmation_bars+1)] <= upper_channel and
                    all(price_history['close'].iloc[-confirmation_bars:] > upper_channel)):
                    
                    # Upside breakout - buy signal
                    if volume_ok and volatility_ok:
                        signal_type = SignalType.BUY
                        
                        # Calculate confidence based on:
                        # 1. Channel width relative to ATR (wider channels = more significant breakouts)
                        channel_atr_ratio = channel_width / latest_atr
                        channel_confidence = min(0.3, 0.15 + channel_atr_ratio * 0.05)
                        
                        # 2. Volume strength
                        volume_confidence = 0.0
                        if "volume_percentile" in symbol_indicators:
                            vol_pct = symbol_indicators["volume_percentile"].iloc[-1]["volume_percentile"]
                            volume_confidence = min(0.3, vol_pct / 200)  # Max 0.3 at 60th percentile
                        
                        # 3. Volatility
                        vol_confidence = min(0.2, symbol_indicators["atr_percentile"].iloc[-1]["atr_percentile"] / 500)
                        
                        # 4. Distance from breakout level
                        breakout_distance = (latest_price / upper_channel - 1) * 100
                        # Not too close, not too far
                        distance_confidence = min(0.2, 0.1 + min(breakout_distance, 2.0) * 0.05)
                        
                        confidence = min(0.9, channel_confidence + volume_confidence + vol_confidence + distance_confidence)
                        
                        # Calculate stop loss and take profit based on ATR
                        stop_loss = latest_price - (latest_atr * stop_loss_atr_multiple)
                        take_profit = latest_price + (latest_atr * take_profit_atr_multiple)
                
                # Check for downside breakout
                elif (confirmation_bars == 0 and latest_price < lower_channel) or \
                     (confirmation_bars > 0 and 
                      price_history['close'].iloc[-(confirmation_bars+1)] >= lower_channel and
                      all(price_history['close'].iloc[-confirmation_bars:] < lower_channel)):
                    
                    # Downside breakout - sell signal
                    if volume_ok and volatility_ok:
                        signal_type = SignalType.SELL
                        
                        # Calculate confidence using same factors as upside breakout
                        channel_atr_ratio = channel_width / latest_atr
                        channel_confidence = min(0.3, 0.15 + channel_atr_ratio * 0.05)
                        
                        volume_confidence = 0.0
                        if "volume_percentile" in symbol_indicators:
                            vol_pct = symbol_indicators["volume_percentile"].iloc[-1]["volume_percentile"]
                            volume_confidence = min(0.3, vol_pct / 200)
                        
                        vol_confidence = min(0.2, symbol_indicators["atr_percentile"].iloc[-1]["atr_percentile"] / 500)
                        
                        breakout_distance = (1 - latest_price / lower_channel) * 100
                        distance_confidence = min(0.2, 0.1 + min(breakout_distance, 2.0) * 0.05)
                        
                        confidence = min(0.9, channel_confidence + volume_confidence + vol_confidence + distance_confidence)
                        
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
                            "channel_period": channel_period,
                            "channel_width": channel_width,
                            "atr": latest_atr,
                            "trailing_stop": trailing_stop,
                            "trailing_activation": trailing_activation,
                            "strategy_type": "breakout"
                        }
                    )
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals


class VolumeBreakoutStrategy(StrategyOptimizable):
    """
    Breakout strategy based on volume-confirmed price movements.
    
    This strategy identifies key support/resistance levels and generates
    signals when price breaks through these levels with above-average volume.
    
    Key features:
    - Identifies significant support and resistance levels using price action analysis
    - Requires volume confirmation to validate the authenticity of breakouts
    - Focuses on consolidation patterns before breakouts for higher probability trades
    - Implements adaptive stop loss based on market volatility and support/resistance
    - Features volatility-adjusted position sizing through ATR calculations
    - Includes price pattern filters to identify higher probability breakout setups
    
    Trading logic:
    - Buy signals: Generated when price breaks above resistance with volume exceeding threshold
      after a period of price consolidation within a defined range
    - Sell signals: Triggered when price breaks below support with strong volume confirmation
      following a period of consolidation or distribution pattern
    - Stop placement: Placed at the opposite side of the consolidation range or key support/resistance
    - Take profit: Calculated using measured move projections or ATR multiples
    
    Ideal market conditions:
    - Markets with clear volume trends and price-volume correlations
    - Assets showing periods of consolidation before significant moves
    - Trading environments with institutional participation (visible in volume patterns)
    - Markets with sufficient liquidity to provide meaningful volume signals
    
    Limitations:
    - Volume data may be less reliable in certain markets (e.g., thinly traded stocks)
    - Difficulty in distinguishing between breakout volume and exhaustion volume
    - May generate too few signals in low-volatility environments
    - Requires continuous monitoring of support/resistance level validity
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Volume Breakout strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters
        default_params = {
            "lookback_period": 20,
            "resistance_lookback": 50,
            "support_tolerance": 0.01,  # 1% tolerance for support/resistance
            "volume_threshold": 1.5,    # Volume must be 1.5x the average
            "price_filter": True,
            "min_consolidation_days": 5,
            "max_consolidation_range": 0.05,  # 5% max range for consolidation
            "atr_period": 14,
            "stop_loss_atr_multiple": 1.5,
            "take_profit_atr_multiple": 3.0,
            "trailing_stop": True,
            "trailing_stop_activation_percent": 1.0
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        logger.info(f"Initialized Volume Breakout strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "lookback_period": [10, 20, 30],
            "resistance_lookback": [30, 50, 100],
            "support_tolerance": [0.005, 0.01, 0.02],
            "volume_threshold": [1.2, 1.5, 2.0],
            "price_filter": [True, False],
            "min_consolidation_days": [3, 5, 7, 10],
            "max_consolidation_range": [0.03, 0.05, 0.07],
            "atr_period": [10, 14, 20],
            "stop_loss_atr_multiple": [1.0, 1.5, 2.0],
            "take_profit_atr_multiple": [2.0, 3.0, 4.0],
            "trailing_stop": [True, False],
            "trailing_stop_activation_percent": [0.5, 1.0, 1.5]
        }
    
    def _find_support_resistance(self, highs: pd.Series, lows: pd.Series, 
                                 lookback: int, tolerance: float) -> Tuple[List[float], List[float]]:
        """
        Find support and resistance levels in price data.
        
        Args:
            highs: Series of high prices
            lows: Series of low prices
            lookback: Number of periods to look back
            tolerance: Tolerance for identifying price levels (as a percentage)
            
        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        support_levels = []
        resistance_levels = []
        
        # Use only the lookback period
        if len(highs) > lookback:
            highs = highs[-lookback:]
            lows = lows[-lookback:]
        
        # Find local minima and maxima
        for i in range(1, len(lows)-1):
            # Local minimum (support)
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                # Check if we already have a similar level
                is_new_level = True
                for level in support_levels:
                    if abs(lows[i] / level - 1) < tolerance:
                        is_new_level = False
                        break
                
                if is_new_level:
                    support_levels.append(lows[i])
            
            # Local maximum (resistance)
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                # Check if we already have a similar level
                is_new_level = True
                for level in resistance_levels:
                    if abs(highs[i] / level - 1) < tolerance:
                        is_new_level = False
                        break
                
                if is_new_level:
                    resistance_levels.append(highs[i])
        
        return support_levels, resistance_levels
    
    def _is_consolidating(self, prices: pd.Series, days: int, max_range: float) -> bool:
        """
        Check if the price is consolidating within a range.
        
        Args:
            prices: Series of price data
            days: Number of days to check
            max_range: Maximum allowed range as a percentage
            
        Returns:
            Boolean indicating if the price is consolidating
        """
        if len(prices) < days:
            return False
        
        recent_prices = prices[-days:]
        price_range = (recent_prices.max() - recent_prices.min()) / recent_prices.min()
        
        return price_range <= max_range
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate support/resistance levels and other indicators for all symbols.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary of calculated indicators for each symbol
        """
        indicators = {}
        
        # Get parameters
        lookback = self.parameters.get("lookback_period", 20)
        resistance_lookback = self.parameters.get("resistance_lookback", 50)
        support_tolerance = self.parameters.get("support_tolerance", 0.01)
        atr_period = self.parameters.get("atr_period", 14)
        
        for symbol, df in data.items():
            # Ensure required columns exist
            if not all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
                logger.warning(f"Required price/volume columns not found for {symbol}")
                continue
            
            try:
                # Calculate average volume
                avg_volume = df['volume'].rolling(window=lookback).mean()
                
                # Calculate volume ratio (current volume / average volume)
                volume_ratio = df['volume'] / avg_volume
                
                # Find support and resistance levels
                support_levels, resistance_levels = self._find_support_resistance(
                    df['high'], df['low'], resistance_lookback, support_tolerance
                )
                
                # Calculate ATR for volatility assessment
                high_low = df['high'] - df['low']
                high_close_prev = np.abs(df['high'] - df['close'].shift(1))
                low_close_prev = np.abs(df['low'] - df['close'].shift(1))
                tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
                atr = tr.rolling(window=atr_period).mean()
                
                # Store indicators
                indicators[symbol] = {
                    "avg_volume": pd.DataFrame({"avg_volume": avg_volume}),
                    "volume_ratio": pd.DataFrame({"volume_ratio": volume_ratio}),
                    "atr": pd.DataFrame({"atr": atr})
                }
                
                # Store support and resistance levels
                indicators[symbol]["support_levels"] = support_levels
                indicators[symbol]["resistance_levels"] = resistance_levels
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
        
        return indicators
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate buy/sell signals based on volume breakouts.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Get parameters
        volume_threshold = self.parameters.get("volume_threshold", 1.5)
        price_filter = self.parameters.get("price_filter", True)
        min_consolidation_days = self.parameters.get("min_consolidation_days", 5)
        max_consolidation_range = self.parameters.get("max_consolidation_range", 0.05)
        stop_loss_atr_multiple = self.parameters.get("stop_loss_atr_multiple", 1.5)
        take_profit_atr_multiple = self.parameters.get("take_profit_atr_multiple", 3.0)
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
                latest_price = latest_data['close']
                latest_high = latest_data['high']
                latest_low = latest_data['low']
                latest_timestamp = latest_data.name if isinstance(latest_data.name, datetime) else datetime.now()
                
                # Get volume indicators
                latest_volume_ratio = symbol_indicators["volume_ratio"].iloc[-1]["volume_ratio"]
                latest_atr = symbol_indicators["atr"].iloc[-1]["atr"]
                
                # Check consolidation if price filter is enabled
                consolidation_ok = True
                if price_filter:
                    consolidation_ok = self._is_consolidating(
                        data[symbol]['close'], min_consolidation_days, max_consolidation_range
                    )
                
                # Check volume threshold
                volume_ok = latest_volume_ratio >= volume_threshold
                
                # Generate signal if conditions are met
                signal_type = None
                confidence = 0.0
                
                if volume_ok and consolidation_ok:
                    # Check for breakout of resistance level (buy signal)
                    resistance_levels = symbol_indicators.get("resistance_levels", [])
                    for level in resistance_levels:
                        # If the price is breaking above a resistance level
                        if latest_data['close'] > level > data[symbol].iloc[-2]['close']:
                            # Bullish breakout - buy signal
                            signal_type = SignalType.BUY
                            
                            # Calculate confidence based on volume and breakout strength
                            volume_confidence = min(0.4, 0.2 + (latest_volume_ratio - volume_threshold) * 0.1)
                            breakout_strength = (latest_price / level - 1) * 100
                            breakout_confidence = min(0.3, breakout_strength * 0.1)
                            consolidation_confidence = 0.2 if consolidation_ok else 0.0
                            
                            confidence = min(0.9, volume_confidence + breakout_confidence + consolidation_confidence)
                            
                            # Calculate stop loss and take profit based on ATR
                            stop_loss = latest_price - (latest_atr * stop_loss_atr_multiple)
                            take_profit = latest_price + (latest_atr * take_profit_atr_multiple)
                            
                            # Only use the closest resistance breakout
                            break
                    
                    # If no buy signal was found, check for breakdown of support (sell signal)
                    if signal_type is None:
                        support_levels = symbol_indicators.get("support_levels", [])
                        for level in support_levels:
                            # If the price is breaking below a support level
                            if latest_data['close'] < level < data[symbol].iloc[-2]['close']:
                                # Bearish breakdown - sell signal
                                signal_type = SignalType.SELL
                                
                                # Calculate confidence based on volume and breakdown strength
                                volume_confidence = min(0.4, 0.2 + (latest_volume_ratio - volume_threshold) * 0.1)
                                breakdown_strength = (1 - latest_price / level) * 100
                                breakdown_confidence = min(0.3, breakdown_strength * 0.1)
                                consolidation_confidence = 0.2 if consolidation_ok else 0.0
                                
                                confidence = min(0.9, volume_confidence + breakdown_confidence + consolidation_confidence)
                                
                                # Calculate stop loss and take profit based on ATR
                                stop_loss = latest_price + (latest_atr * stop_loss_atr_multiple)
                                take_profit = latest_price - (latest_atr * take_profit_atr_multiple)
                                
                                # Only use the closest support breakdown
                                break
                
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
                            "volume_ratio": latest_volume_ratio,
                            "atr": latest_atr,
                            "trailing_stop": trailing_stop,
                            "trailing_activation": trailing_activation,
                            "strategy_type": "breakout"
                        }
                    )
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals


class VolatilityBreakoutStrategy(StrategyOptimizable):
    """
    Breakout strategy based on volatility expansion.
    
    This strategy identifies periods of low volatility (contraction) followed
    by volatility expansion, which often signals the beginning of a new trend.
    
    Key features:
    - Identifies volatility contraction periods using ATR (Average True Range) analysis
    - Detects significant volatility expansion events that often precede trending moves
    - Uses price direction during expansion to determine trade direction
    - Incorporates volume confirmation to validate the authenticity of the breakout
    - Adapts position sizing and stop loss parameters to current volatility conditions
    - Features dynamic take-profit targets based on volatility multiples
    
    Trading logic:
    - Identifies periods of decreasing volatility (contraction phases)
    - Monitors for sudden expansion in volatility (20%+ increase in ATR)
    - Generates buy signals when volatility expands with upward price movement
    - Generates sell signals when volatility expands with downward price movement
    - Uses close price movement relative to prior close to determine direction
    - Sets stop loss using volatility-adjusted distance from entry price
    
    Ideal market conditions:
    - Assets transitioning from consolidation to trending behavior
    - Markets with cyclical volatility patterns (compression followed by expansion)
    - Assets with sufficient liquidity to handle volatility events
    - Environments where volatility signals precede directional price movements
    
    Limitations:
    - May generate false signals during choppy market conditions
    - Performance varies across different types of volatility environments
    - Can be triggered by short-term volatility spikes without follow-through
    - May miss opportunities in steadily trending markets without volatility events
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Volatility Breakout strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters
        default_params = {
            "atr_period": 14,
            "atr_ma_period": 50,
            "min_volatility_expansion": 1.2,  # ATR must expand by at least 20%
            "lookback_period": 20,
            "min_vol_contraction_days": 5,
            "volume_filter": True,
            "min_volume_expansion": 1.5,  # Volume must expand by at least 50%
            "stop_loss_atr_multiple": 1.0,
            "take_profit_atr_multiple": 3.0,
            "trailing_stop": True,
            "trailing_stop_activation_percent": 1.0
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        logger.info(f"Initialized Volatility Breakout strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "atr_period": [10, 14, 20],
            "atr_ma_period": [30, 50, 70],
            "min_volatility_expansion": [1.1, 1.2, 1.3, 1.4],
            "lookback_period": [10, 20, 30],
            "min_vol_contraction_days": [3, 5, 7, 10],
            "volume_filter": [True, False],
            "min_volume_expansion": [1.3, 1.5, 1.7, 2.0],
            "stop_loss_atr_multiple": [0.75, 1.0, 1.5],
            "take_profit_atr_multiple": [2.0, 3.0, 4.0],
            "trailing_stop": [True, False],
            "trailing_stop_activation_percent": [0.5, 1.0, 1.5]
        }
    
    def _detect_volatility_contraction(self, atr: pd.Series, days: int) -> bool:
        """
        Detect if the market has been in a volatility contraction phase.
        
        Args:
            atr: Series of ATR values
            days: Minimum number of days for contraction
            
        Returns:
            Boolean indicating if volatility contraction is detected
        """
        if len(atr) < days + 1:
            return False
        
        # Get recent ATR values
        recent_atr = atr[-days-1:-1]
        
        # Calculate the slope of the ATR
        atr_slope = (recent_atr.iloc[-1] - recent_atr.iloc[0]) / recent_atr.iloc[0]
        
        # Negative slope indicates contracting volatility
        return atr_slope < 0
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate volatility indicators for all symbols.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary of calculated indicators for each symbol
        """
        indicators = {}
        
        # Get parameters
        atr_period = self.parameters.get("atr_period", 14)
        atr_ma_period = self.parameters.get("atr_ma_period", 50)
        
        for symbol, df in data.items():
            # Ensure required columns exist
            if not all(col in df.columns for col in ['high', 'low', 'close']):
                logger.warning(f"Required price columns not found for {symbol}")
                continue
            
            try:
                # Calculate ATR for volatility assessment
                high_low = df['high'] - df['low']
                high_close_prev = np.abs(df['high'] - df['close'].shift(1))
                low_close_prev = np.abs(df['low'] - df['close'].shift(1))
                tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
                atr = tr.rolling(window=atr_period).mean()
                
                # Calculate ATR moving average for longer-term volatility reference
                atr_ma = atr.rolling(window=atr_ma_period).mean()
                
                # Calculate ATR expansion ratio
                atr_expansion = atr / atr.shift(1)
                
                # Calculate ATR percentile over last 100 periods
                atr_percentile = atr.rolling(window=100).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
                )
                
                # Calculate volume expansion if volume data is available
                volume_expansion = None
                if 'volume' in df.columns:
                    volume_ma = df['volume'].rolling(window=20).mean()
                    volume_expansion = df['volume'] / volume_ma
                
                # Store indicators
                indicators[symbol] = {
                    "atr": pd.DataFrame({"atr": atr}),
                    "atr_ma": pd.DataFrame({"atr_ma": atr_ma}),
                    "atr_expansion": pd.DataFrame({"atr_expansion": atr_expansion}),
                    "atr_percentile": pd.DataFrame({"atr_percentile": atr_percentile})
                }
                
                if volume_expansion is not None:
                    indicators[symbol]["volume_expansion"] = pd.DataFrame({"volume_expansion": volume_expansion})
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
        
        return indicators
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate buy/sell signals based on volatility breakouts.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Get parameters
        min_vol_expansion = self.parameters.get("min_volatility_expansion", 1.2)
        lookback_period = self.parameters.get("lookback_period", 20)
        min_vol_contraction_days = self.parameters.get("min_vol_contraction_days", 5)
        volume_filter = self.parameters.get("volume_filter", True)
        min_volume_expansion = self.parameters.get("min_volume_expansion", 1.5)
        stop_loss_atr_multiple = self.parameters.get("stop_loss_atr_multiple", 1.0)
        take_profit_atr_multiple = self.parameters.get("take_profit_atr_multiple", 3.0)
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
                previous_data = data[symbol].iloc[-2]
                latest_price = latest_data['close']
                latest_timestamp = latest_data.name if isinstance(latest_data.name, datetime) else datetime.now()
                
                # Get volatility indicators
                latest_atr = symbol_indicators["atr"].iloc[-1]["atr"]
                latest_atr_expansion = symbol_indicators["atr_expansion"].iloc[-1]["atr_expansion"]
                
                # Check if we have a volatility expansion
                vol_expansion_ok = latest_atr_expansion >= min_vol_expansion
                
                # Check if we had a volatility contraction before the expansion
                vol_contraction_ok = self._detect_volatility_contraction(
                    symbol_indicators["atr"].iloc[:, 0], min_vol_contraction_days
                )
                
                # Check volume expansion if volume filter is enabled
                volume_ok = True
                if volume_filter and "volume_expansion" in symbol_indicators:
                    latest_vol_expansion = symbol_indicators["volume_expansion"].iloc[-1]["volume_expansion"]
                    volume_ok = latest_vol_expansion >= min_volume_expansion
                
                # Generate signal based on volatility breakout
                signal_type = None
                confidence = 0.0
                
                if vol_expansion_ok and vol_contraction_ok and volume_ok:
                    # Determine direction based on price movement
                    price_change = latest_price / previous_data['close'] - 1
                    
                    if price_change > 0:
                        # Bullish volatility breakout - buy signal
                        signal_type = SignalType.BUY
                        
                        # Calculate confidence based on volatility expansion and volume
                        vol_confidence = min(0.4, 0.2 + (latest_atr_expansion - min_vol_expansion) * 0.5)
                        
                        volume_confidence = 0.0
                        if "volume_expansion" in symbol_indicators:
                            latest_vol_expansion = symbol_indicators["volume_expansion"].iloc[-1]["volume_expansion"]
                            volume_confidence = min(0.3, 0.1 + (latest_vol_expansion - min_volume_expansion) * 0.2)
                        
                        price_confidence = min(0.3, price_change * 10)  # Higher confidence with stronger price move
                        
                        confidence = min(0.9, vol_confidence + volume_confidence + price_confidence)
                        
                        # Calculate stop loss and take profit based on ATR
                        stop_loss = latest_price - (latest_atr * stop_loss_atr_multiple)
                        take_profit = latest_price + (latest_atr * take_profit_atr_multiple)
                        
                    elif price_change < 0:
                        # Bearish volatility breakout - sell signal
                        signal_type = SignalType.SELL
                        
                        # Calculate confidence based on volatility expansion and volume
                        vol_confidence = min(0.4, 0.2 + (latest_atr_expansion - min_vol_expansion) * 0.5)
                        
                        volume_confidence = 0.0
                        if "volume_expansion" in symbol_indicators:
                            latest_vol_expansion = symbol_indicators["volume_expansion"].iloc[-1]["volume_expansion"]
                            volume_confidence = min(0.3, 0.1 + (latest_vol_expansion - min_volume_expansion) * 0.2)
                        
                        price_confidence = min(0.3, abs(price_change) * 10)  # Higher confidence with stronger price move
                        
                        confidence = min(0.9, vol_confidence + volume_confidence + price_confidence)
                        
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
                            "atr": latest_atr,
                            "atr_expansion": latest_atr_expansion,
                            "trailing_stop": trailing_stop,
                            "trailing_activation": trailing_activation,
                            "strategy_type": "breakout"
                        }
                    )
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals 