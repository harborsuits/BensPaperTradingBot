#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Breakout Strategy Template

This module implements a comprehensive breakout strategy template
that can be optimized and extended for specific trading approaches.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import talib

from trading_bot.strategies.strategy_template import (
    StrategyOptimizable,
    StrategyMetadata,
    Signal,
    SignalType,
    TimeFrame,
    MarketRegime
)

logger = logging.getLogger(__name__)

class BreakoutTemplate(StrategyOptimizable):
    """
    Breakout Strategy Template that identifies and trades price breakouts
    from significant support/resistance levels and chart patterns.
    
    Supported breakout types include:
    - Range breakouts (boxes, channels)
    - Support/resistance level breakouts
    - Chart pattern breakouts (triangles, flags, etc.)
    - Volatility breakouts
    - Volume-confirmed breakouts
    
    This template can be customized and extended for specific breakout approaches.
    """
    
    def __init__(
        self,
        name: str = "breakout",
        parameters: Dict[str, Any] = None,
        metadata: Optional[StrategyMetadata] = None
    ):
        """
        Initialize breakout strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy-specific parameters
            metadata: Strategy metadata
        """
        # Default parameters for breakout strategy
        default_params = {
            # General parameters
            "lookback_period": 30,
            "min_consolidation_bars": 5,
            "breakout_threshold": 0.5,  # % above resistance / below support
            "stop_loss_atr_multiple": 1.5,
            "take_profit_atr_multiple": 3.0,
            "risk_per_trade": 0.015,  # Slightly higher risk for breakout trades
            
            # Range breakout parameters
            "use_range_breakout": True,
            "range_period": 20,
            "range_threshold": 0.03,  # Max range as % of price to qualify as consolidation
            
            # Support/Resistance parameters
            "use_support_resistance": True,
            "sr_lookback": 50,
            "sr_confirmation_touches": 2,
            "sr_proximity_threshold": 0.02,  # % distance to consider "at" a level
            
            # Volatility breakout parameters
            "use_volatility_breakout": True,
            "volatility_period": 20,
            "volatility_multiple": 2.0,  # ATR multiple for breakout
            "volatility_squeeze_threshold": 0.7,  # Bollinger band width threshold for squeeze
            
            # Volume parameters
            "require_volume_confirmation": True,
            "volume_confirmation_threshold": 1.5,  # Volume increase for confirmation
            "volume_lookback": 10,
            
            # Confirmation parameters
            "require_close_confirmation": True,  # Require close beyond level, not just wick
            "false_breakout_filter": True,  # Additional filtering for false breakouts
            "min_indicators_for_signal": 2,
            "signal_threshold": 0.6  # Minimum confidence threshold
        }
        
        # Override defaults with provided parameters
        if parameters:
            default_params.update(parameters)
        
        # Create default metadata if not provided
        if metadata is None:
            metadata = StrategyMetadata(
                name=name,
                version="1.0.0",
                description="Breakout strategy that identifies and trades price breakouts from key levels",
                author="Trading Bot",
                timeframes=[
                    TimeFrame.MINUTE_15,
                    TimeFrame.MINUTE_30,
                    TimeFrame.HOUR_1,
                    TimeFrame.HOUR_4,
                    TimeFrame.DAY_1
                ],
                asset_classes=["stocks", "forex", "crypto", "futures", "etf"],
                tags=["breakout", "volatility", "support_resistance", "volume", "momentum"]
            )
        
        # Initialize parent class
        super().__init__(name, default_params, metadata)
        
        logger.info(f"Initialized breakout strategy: {name}")
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate breakout indicators for the given data.
        
        Args:
            data: Dictionary of symbol -> DataFrame with price/volume data
            
        Returns:
            Dictionary of symbol -> indicators dictionary
        """
        indicators = {}
        
        for symbol, df in data.items():
            # Skip if insufficient data
            lookback = max(
                self.parameters["lookback_period"],
                self.parameters["sr_lookback"] + 10
            )
            if len(df) < lookback:
                continue
            
            symbol_indicators = {}
            
            # Extract price and volume data
            close_prices = df['close'].values
            high_prices = df['high'].values if 'high' in df else close_prices
            low_prices = df['low'].values if 'low' in df else close_prices
            volumes = df['volume'].values if 'volume' in df else np.ones_like(close_prices)
            
            # Calculate range breakout indicators
            if self.parameters["use_range_breakout"]:
                try:
                    range_period = self.parameters["range_period"]
                    
                    # Calculate rolling high and low
                    rolling_high = pd.Series(high_prices).rolling(window=range_period).max()
                    rolling_low = pd.Series(low_prices).rolling(window=range_period).min()
                    
                    # Calculate percentage range
                    price_range = (rolling_high - rolling_low) / rolling_low
                    
                    # Identify consolidation periods (low range)
                    range_threshold = self.parameters["range_threshold"]
                    is_consolidating = price_range < range_threshold
                    
                    # Check for minimum consecutive consolidation bars
                    min_bars = self.parameters["min_consolidation_bars"]
                    consolidation_count = is_consolidating.rolling(window=min_bars).sum()
                    consolidated = consolidation_count >= min_bars
                    
                    # Store indicators
                    symbol_indicators['rolling_high'] = rolling_high
                    symbol_indicators['rolling_low'] = rolling_low
                    symbol_indicators['price_range'] = price_range
                    symbol_indicators['is_consolidating'] = is_consolidating
                    symbol_indicators['consolidated'] = consolidated
                except Exception as e:
                    logger.error(f"Error calculating range breakout indicators for {symbol}: {e}")
            
            # Calculate support/resistance levels
            if self.parameters["use_support_resistance"]:
                try:
                    sr_lookback = self.parameters["sr_lookback"]
                    min_touches = self.parameters["sr_confirmation_touches"]
                    
                    # Identify swing highs and lows
                    swing_high = pd.Series(high_prices)
                    swing_low = pd.Series(low_prices)
                    
                    # Simple swing detection (local maxima/minima)
                    window = 5  # Window for local extrema
                    
                    # Local maxima (high is higher than n periods before and after)
                    max_idx = []
                    for i in range(window, len(high_prices) - window):
                        if (high_prices[i] > high_prices[i-window:i]).all() and (high_prices[i] > high_prices[i+1:i+window+1]).all():
                            max_idx.append(i)
                    
                    # Local minima (low is lower than n periods before and after)
                    min_idx = []
                    for i in range(window, len(low_prices) - window):
                        if (low_prices[i] < low_prices[i-window:i]).all() and (low_prices[i] < low_prices[i+1:i+window+1]).all():
                            min_idx.append(i)
                    
                    # Get resistance levels from swing highs
                    resistance_levels = {}
                    for idx in max_idx:
                        level = high_prices[idx]
                        proximity = self.parameters["sr_proximity_threshold"] * level
                        
                        # Count levels in proximity
                        close_levels = [l for l in resistance_levels.keys() if abs(l - level) < proximity]
                        
                        if close_levels:
                            # Average with existing close level
                            avg_level = sum(close_levels) / len(close_levels) * resistance_levels[close_levels[0]] + level
                            avg_level /= (resistance_levels[close_levels[0]] + 1)
                            resistance_levels[close_levels[0]] += 1
                        else:
                            # New level
                            resistance_levels[level] = 1
                    
                    # Get support levels from swing lows
                    support_levels = {}
                    for idx in min_idx:
                        level = low_prices[idx]
                        proximity = self.parameters["sr_proximity_threshold"] * level
                        
                        # Count levels in proximity
                        close_levels = [l for l in support_levels.keys() if abs(l - level) < proximity]
                        
                        if close_levels:
                            # Average with existing close level
                            avg_level = sum(close_levels) / len(close_levels) * support_levels[close_levels[0]] + level
                            avg_level /= (support_levels[close_levels[0]] + 1)
                            support_levels[close_levels[0]] += 1
                        else:
                            # New level
                            support_levels[level] = 1
                    
                    # Filter to significant levels (with enough touches)
                    significant_resistance = [level for level, touches in resistance_levels.items() if touches >= min_touches]
                    significant_support = [level for level, touches in support_levels.items() if touches >= min_touches]
                    
                    # Store levels in ascending order
                    symbol_indicators['resistance_levels'] = sorted(significant_resistance)
                    symbol_indicators['support_levels'] = sorted(significant_support)
                    
                    # Calculate nearest levels to current price
                    current_price = close_prices[-1]
                    
                    # Find nearest resistance above current price
                    above_resistance = [r for r in significant_resistance if r > current_price]
                    nearest_resistance = min(above_resistance) if above_resistance else None
                    
                    # Find nearest support below current price
                    below_support = [s for s in significant_support if s < current_price]
                    nearest_support = max(below_support) if below_support else None
                    
                    symbol_indicators['nearest_resistance'] = nearest_resistance
                    symbol_indicators['nearest_support'] = nearest_support
                    
                    # Calculate distance to nearest levels (as percentage of price)
                    if nearest_resistance:
                        symbol_indicators['resistance_distance'] = (nearest_resistance - current_price) / current_price
                    
                    if nearest_support:
                        symbol_indicators['support_distance'] = (current_price - nearest_support) / current_price
                except Exception as e:
                    logger.error(f"Error calculating support/resistance for {symbol}: {e}")
            
            # Calculate volatility breakout indicators
            if self.parameters["use_volatility_breakout"]:
                try:
                    # Calculate ATR
                    vol_period = self.parameters["volatility_period"]
                    atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=vol_period)
                    symbol_indicators['atr'] = pd.Series(atr, index=df.index)
                    
                    # Calculate Bollinger Bands
                    upper, middle, lower = talib.BBANDS(
                        close_prices,
                        timeperiod=vol_period,
                        nbdevup=2,
                        nbdevdn=2
                    )
                    
                    symbol_indicators['bb_upper'] = pd.Series(upper, index=df.index)
                    symbol_indicators['bb_middle'] = pd.Series(middle, index=df.index)
                    symbol_indicators['bb_lower'] = pd.Series(lower, index=df.index)
                    
                    # Calculate Bollinger Band width (for detecting volatility squeezes)
                    bb_width = (upper - lower) / middle
                    symbol_indicators['bb_width'] = pd.Series(bb_width, index=df.index)
                    
                    # Detect volatility squeeze (narrowing BB width)
                    squeeze_threshold = self.parameters["volatility_squeeze_threshold"]
                    
                    # Current BB width vs average of last 20 periods
                    avg_bb_width = pd.Series(bb_width).rolling(window=20).mean()
                    squeeze_ratio = bb_width / avg_bb_width
                    
                    # Is price in a volatility squeeze?
                    is_squeezed = squeeze_ratio < squeeze_threshold
                    symbol_indicators['is_squeezed'] = pd.Series(is_squeezed, index=df.index)
                    
                    # Detect potential breakout levels
                    vol_multiple = self.parameters["volatility_multiple"]
                    breakout_level_up = close_prices + (atr * vol_multiple)
                    breakout_level_down = close_prices - (atr * vol_multiple)
                    
                    symbol_indicators['breakout_level_up'] = pd.Series(breakout_level_up, index=df.index)
                    symbol_indicators['breakout_level_down'] = pd.Series(breakout_level_down, index=df.index)
                except Exception as e:
                    logger.error(f"Error calculating volatility indicators for {symbol}: {e}")
            
            # Calculate volume indicators
            if self.parameters["require_volume_confirmation"] and 'volume' in df:
                try:
                    volume_lookback = self.parameters["volume_lookback"]
                    
                    # Calculate average volume
                    avg_volume = pd.Series(volumes).rolling(window=volume_lookback).mean()
                    symbol_indicators['avg_volume'] = pd.Series(avg_volume, index=df.index)
                    
                    # Calculate volume ratio
                    volume_ratio = volumes / avg_volume
                    symbol_indicators['volume_ratio'] = pd.Series(volume_ratio, index=df.index)
                    
                    # Is volume confirming?
                    vol_threshold = self.parameters["volume_confirmation_threshold"]
                    volume_confirmed = volume_ratio > vol_threshold
                    symbol_indicators['volume_confirmed'] = pd.Series(volume_confirmed, index=df.index)
                except Exception as e:
                    logger.error(f"Error calculating volume indicators for {symbol}: {e}")
            
            # Store indicators for this symbol
            indicators[symbol] = symbol_indicators
        
        return indicators
    
    def generate_signals(
        self, 
        data: Dict[str, pd.DataFrame], 
        context: Dict[str, Any] = None
    ) -> Dict[str, Signal]:
        """
        Generate breakout trading signals for the given data.
        
        Args:
            data: Dictionary of symbol -> DataFrame with price/indicator data
            context: Additional context information (market regime, etc.)
            
        Returns:
            Dictionary of symbol -> Signal objects
        """
        signals = {}
        
        # Calculate indicators
        indicators = self.calculate_indicators(data)
        
        # Get current market regime if available
        market_regime = context.get('market_regime', MarketRegime.UNKNOWN) if context else MarketRegime.UNKNOWN
        
        for symbol, symbol_indicators in indicators.items():
            # Skip if not enough indicators available
            if len(symbol_indicators) < 3:
                continue
            
            df = data[symbol]
            current_price = df['close'].iloc[-1]
            current_high = df['high'].iloc[-1] if 'high' in df else current_price
            current_low = df['low'].iloc[-1] if 'low' in df else current_price
            current_time = df.index[-1]
            
            # Initialize signal tracking
            breakout_signals = 0
            breakdown_signals = 0
            total_potential_signals = 0
            
            # Check for range breakouts
            if 'consolidated' in symbol_indicators and self.parameters["use_range_breakout"]:
                total_potential_signals += 1
                
                # Get consolidation status
                consolidated = symbol_indicators['consolidated'].iloc[-2]  # Check if we were consolidating before current bar
                
                if consolidated:
                    # Get range high and low
                    range_high = symbol_indicators['rolling_high'].iloc[-2]
                    range_low = symbol_indicators['rolling_low'].iloc[-2]
                    
                    # Calculate breakout thresholds
                    threshold = self.parameters["breakout_threshold"] / 100
                    upside_threshold = range_high * (1 + threshold)
                    downside_threshold = range_low * (1 - threshold)
                    
                    # Check for upside breakout
                    if current_high > upside_threshold:
                        # If requiring close confirmation, check close vs range high
                        if not self.parameters["require_close_confirmation"] or current_price > range_high:
                            breakout_signals += 1
                            
                            # Add metadata for this breakout
                            symbol_indicators['range_breakout_level'] = range_high
                            symbol_indicators['range_breakout_distance'] = (current_price - range_high) / range_high
                    
                    # Check for downside breakdown
                    elif current_low < downside_threshold:
                        # If requiring close confirmation, check close vs range low
                        if not self.parameters["require_close_confirmation"] or current_price < range_low:
                            breakdown_signals += 1
                            
                            # Add metadata for this breakdown
                            symbol_indicators['range_breakdown_level'] = range_low
                            symbol_indicators['range_breakdown_distance'] = (range_low - current_price) / range_low
            
            # Check for support/resistance breakouts
            if 'nearest_resistance' in symbol_indicators and 'nearest_support' in symbol_indicators and self.parameters["use_support_resistance"]:
                total_potential_signals += 1
                
                resistance = symbol_indicators['nearest_resistance']
                support = symbol_indicators['nearest_support']
                
                # Check for resistance breakout
                if resistance:
                    threshold = self.parameters["breakout_threshold"] / 100
                    breakout_level = resistance * (1 + threshold)
                    
                    if current_high > breakout_level:
                        # If requiring close confirmation, check close vs resistance
                        if not self.parameters["require_close_confirmation"] or current_price > resistance:
                            breakout_signals += 1
                            
                            # Add metadata for this breakout
                            symbol_indicators['sr_breakout_level'] = resistance
                            symbol_indicators['sr_breakout_distance'] = (current_price - resistance) / resistance
                
                # Check for support breakdown
                if support:
                    threshold = self.parameters["breakout_threshold"] / 100
                    breakdown_level = support * (1 - threshold)
                    
                    if current_low < breakdown_level:
                        # If requiring close confirmation, check close vs support
                        if not self.parameters["require_close_confirmation"] or current_price < support:
                            breakdown_signals += 1
                            
                            # Add metadata for this breakdown
                            symbol_indicators['sr_breakdown_level'] = support
                            symbol_indicators['sr_breakdown_distance'] = (support - current_price) / support
            
            # Check for volatility breakouts
            if 'is_squeezed' in symbol_indicators and self.parameters["use_volatility_breakout"]:
                total_potential_signals += 1
                
                # Check if price was in a squeeze before current bar
                was_squeezed = symbol_indicators['is_squeezed'].iloc[-2] if len(symbol_indicators['is_squeezed']) > 1 else False
                
                if was_squeezed:
                    # Get Bollinger Bands
                    upper = symbol_indicators['bb_upper'].iloc[-1]
                    lower = symbol_indicators['bb_lower'].iloc[-1]
                    
                    # Check for Bollinger Band breakout
                    if current_high > upper:
                        # If requiring close confirmation, check close vs upper band
                        if not self.parameters["require_close_confirmation"] or current_price > upper:
                            breakout_signals += 1
                            
                            # Add metadata for this breakout
                            symbol_indicators['bb_breakout_level'] = upper
                            symbol_indicators['bb_breakout_distance'] = (current_price - upper) / upper
                    
                    # Check for Bollinger Band breakdown
                    elif current_low < lower:
                        # If requiring close confirmation, check close vs lower band
                        if not self.parameters["require_close_confirmation"] or current_price < lower:
                            breakdown_signals += 1
                            
                            # Add metadata for this breakdown
                            symbol_indicators['bb_breakdown_level'] = lower
                            symbol_indicators['bb_breakdown_distance'] = (lower - current_price) / lower
            
            # Check volume confirmation if required
            volume_confirmed = True
            if self.parameters["require_volume_confirmation"] and 'volume_confirmed' in symbol_indicators:
                volume_confirmed = symbol_indicators['volume_confirmed'].iloc[-1]
                
                if not volume_confirmed:
                    # Reduce signal strength if volume confirmation is required but not present
                    breakout_signals *= 0.5
                    breakdown_signals *= 0.5
            
            # Filter false breakouts if enabled
            if self.parameters["false_breakout_filter"]:
                # Get price behavior after potential breakout
                if len(df) >= 3:
                    # Check for quick reversals (potential false breakouts)
                    prev_close = df['close'].iloc[-2]
                    prev_high = df['high'].iloc[-2] if 'high' in df else prev_close
                    prev_low = df['low'].iloc[-2] if 'low' in df else prev_close
                    
                    # If breaking up but closing below previous close = potential false breakout
                    if breakout_signals > 0 and current_price < prev_close:
                        breakout_signals *= 0.3
                    
                    # If breaking down but closing above previous close = potential false breakdown
                    if breakdown_signals > 0 and current_price > prev_close:
                        breakdown_signals *= 0.3
            
            # Determine signal type
            signal_type = None
            signal_strength = 0
            min_signals_required = min(
                self.parameters["min_indicators_for_signal"],
                total_potential_signals
            )
            
            if breakout_signals >= min_signals_required:
                signal_type = SignalType.BUY
                signal_strength = breakout_signals / total_potential_signals
            elif breakdown_signals >= min_signals_required:
                signal_type = SignalType.SELL
                signal_strength = breakdown_signals / total_potential_signals
            
            # Apply threshold filter
            if signal_strength < self.parameters["signal_threshold"]:
                continue
            
            # Generate signal if conditions met
            if signal_type:
                # Adjust confidence based on market regime
                regime_adjustment = self._get_regime_adjustment(market_regime, signal_type)
                confidence = min(0.95, signal_strength * regime_adjustment)
                
                # Apply volume confirmation boost if confirmed
                if volume_confirmed and self.parameters["require_volume_confirmation"]:
                    confidence = min(0.95, confidence * 1.2)
                
                # Calculate stop loss and take profit
                stop_loss = None
                take_profit = None
                
                if 'atr' in symbol_indicators:
                    atr = symbol_indicators['atr'].iloc[-1]
                    
                    if signal_type == SignalType.BUY:
                        stop_loss = current_price - atr * self.parameters["stop_loss_atr_multiple"]
                        take_profit = current_price + atr * self.parameters["take_profit_atr_multiple"]
                    else:  # SELL
                        stop_loss = current_price + atr * self.parameters["stop_loss_atr_multiple"]
                        take_profit = current_price - atr * self.parameters["take_profit_atr_multiple"]
                else:
                    # Fallback stop loss calculation
                    if signal_type == SignalType.BUY and 'range_breakout_level' in symbol_indicators:
                        # Place stop below the breakout level
                        stop_loss = symbol_indicators['range_breakout_level'] * 0.99
                    elif signal_type == SignalType.BUY and 'sr_breakout_level' in symbol_indicators:
                        # Place stop below the resistance (now support) level
                        stop_loss = symbol_indicators['sr_breakout_level'] * 0.99
                    elif signal_type == SignalType.SELL and 'range_breakdown_level' in symbol_indicators:
                        # Place stop above the breakdown level
                        stop_loss = symbol_indicators['range_breakdown_level'] * 1.01
                    elif signal_type == SignalType.SELL and 'sr_breakdown_level' in symbol_indicators:
                        # Place stop above the support (now resistance) level
                        stop_loss = symbol_indicators['sr_breakdown_level'] * 1.01
                
                # Create signal metadata
                signal_metadata = {
                    "strategy": self.name,
                    "volume_confirmed": volume_confirmed,
                    "breakout_type": []
                }
                
                # Add breakout type to metadata
                if 'range_breakout_level' in symbol_indicators:
                    signal_metadata["breakout_type"].append("range")
                    signal_metadata["range_breakout_level"] = symbol_indicators['range_breakout_level']
                
                if 'sr_breakout_level' in symbol_indicators:
                    signal_metadata["breakout_type"].append("support_resistance")
                    signal_metadata["sr_breakout_level"] = symbol_indicators['sr_breakout_level']
                
                if 'bb_breakout_level' in symbol_indicators:
                    signal_metadata["breakout_type"].append("volatility")
                    signal_metadata["bb_breakout_level"] = symbol_indicators['bb_breakout_level']
                
                # Create signal
                signals[symbol] = Signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    timestamp=current_time,
                    price=current_price,
                    confidence=confidence,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    trailing_stop_pct=0.03,  # Default 3% trailing stop
                    metadata=signal_metadata
                )
        
        return signals
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get the parameter space for optimization.
        
        Returns:
            Dictionary of parameter names and possible values
        """
        return {
            # General parameters
            "lookback_period": [20, 30, 40, 50],
            "min_consolidation_bars": [3, 5, 7, 10],
            "breakout_threshold": [0.2, 0.5, 1.0, 1.5],
            "stop_loss_atr_multiple": [1.0, 1.5, 2.0, 2.5],
            "take_profit_atr_multiple": [2.0, 3.0, 4.0, 5.0],
            "risk_per_trade": [0.01, 0.015, 0.02],
            
            # Range breakout parameters
            "range_period": [10, 15, 20, 30],
            "range_threshold": [0.02, 0.03, 0.05, 0.08],
            
            # Support/Resistance parameters
            "sr_lookback": [30, 50, 75, 100],
            "sr_confirmation_touches": [2, 3, 4],
            "sr_proximity_threshold": [0.01, 0.02, 0.03],
            
            # Volatility breakout parameters
            "volatility_period": [10, 15, 20, 30],
            "volatility_multiple": [1.5, 2.0, 2.5, 3.0],
            "volatility_squeeze_threshold": [0.5, 0.7, 0.9],
            
            # Volume parameters
            "volume_confirmation_threshold": [1.3, 1.5, 2.0, 2.5],
            "volume_lookback": [5, 10, 15, 20],
            
            # Confirmation parameters
            "require_close_confirmation": [True, False],
            "min_indicators_for_signal": [1, 2, 3],
            "signal_threshold": [0.4, 0.5, 0.6, 0.7]
        }
    
    def get_regime_compatibility(self, regime: MarketRegime) -> float:
        """
        Get compatibility score for this strategy in the given market regime.
        
        Args:
            regime: Market regime
            
        Returns:
            Compatibility score (0-1, higher is better)
        """
        # Breakout strategies typically work well in transitions between regimes or at start of trends
        compatibility_map = {
            MarketRegime.VOLATILE: 0.8,    # Good in volatile transitions
            MarketRegime.RANGE_BOUND: 0.7, # Good for breaking out of ranges
            MarketRegime.TRENDING: 0.6,    # Can work at start of trends
            MarketRegime.BULLISH: 0.5,     # Moderate in established bullish markets
            MarketRegime.BEARISH: 0.5,     # Moderate in established bearish markets
            MarketRegime.NEUTRAL: 0.4,     # Less effective in neutral markets
            MarketRegime.UNKNOWN: 0.5      # Moderate in unknown regimes
        }
        
        return compatibility_map.get(regime, 0.5)
    
    def _get_regime_adjustment(self, regime: MarketRegime, signal_type: SignalType) -> float:
        """
        Get adjustment factor for signal confidence based on market regime.
        
        Args:
            regime: Current market regime
            signal_type: Signal type (BUY/SELL)
            
        Returns:
            Adjustment factor (0-1.5)
        """
        # Default adjustment - no change
        adjustment = 1.0
        
        # Adjust based on regime
        if regime == MarketRegime.VOLATILE:
            # Ideal for breakouts
            adjustment = 1.2
        elif regime == MarketRegime.RANGE_BOUND:
            # Good for breaking out of ranges
            adjustment = 1.1
        elif regime == MarketRegime.TRENDING:
            # Depends on direction of breakout vs trend
            pass  # Use default
        elif regime == MarketRegime.BULLISH and signal_type == SignalType.SELL:
            # Reduce confidence for breakdowns in bullish market
            adjustment = 0.7
        elif regime == MarketRegime.BEARISH and signal_type == SignalType.BUY:
            # Reduce confidence for breakouts in bearish market
            adjustment = 0.7
        
        return adjustment
    
    def detect_stock_characteristics(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Detect stock-specific characteristics relevant for breakout strategy.
        
        Args:
            data: Dictionary of symbol -> DataFrame with price/volume data
            
        Returns:
            Dictionary of symbol -> characteristics dictionary
        """
        characteristics = {}
        
        for symbol, df in data.items():
            # Skip if insufficient data
            if len(df) < 100:  # Need enough data for meaningful statistics
                continue
            
            # Calculate returns
            returns = df['close'].pct_change().dropna()
            
            # Calculate volatility
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate average true range (ATR) as percentage of price
            atr_pct = None
            if 'high' in df and 'low' in df:
                high = df['high']
                low = df['low']
                close = df['close']
                tr1 = high - low
                tr2 = np.abs(high - close.shift(1))
                tr3 = np.abs(low - close.shift(1))
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(window=14).mean()
                atr_pct = (atr / close).mean()
            
            # Calculate average range
            avg_daily_range = None
            if 'high' in df and 'low' in df:
                daily_range = (df['high'] - df['low']) / df['close']
                avg_daily_range = daily_range.mean()
            
            # Calculate gap frequency and size
            gap_stats = None
            if len(df) > 20:
                gaps = (df['open'] - df['close'].shift(1)).abs() / df['close'].shift(1)
                significant_gaps = gaps[gaps > 0.01]  # Gaps larger than 1%
                
                gap_stats = {
                    "frequency": len(significant_gaps) / len(df),
                    "avg_size": significant_gaps.mean() if len(significant_gaps) > 0 else 0
                }
            
            # Analyze volume characteristics
            volume_stats = None
            if 'volume' in df:
                volume = df['volume']
                avg_volume = volume.mean()
                volume_std = volume.std()
                
                # Volume spikes
                volume_ratio = volume / volume.rolling(window=20).mean()
                spikes = volume_ratio[volume_ratio > 2]
                
                volume_stats = {
                    "avg_volume": avg_volume,
                    "volume_std": volume_std,
                    "coefficient_of_variation": volume_std / avg_volume if avg_volume > 0 else 0,
                    "spike_frequency": len(spikes) / len(df)
                }
            
            # Calculate range-bound tendency
            range_bound_score = 0
            if len(df) > 50:
                # Calculate 20-day highs and lows
                rolling_high_20d = df['high'].rolling(window=20).max()
                rolling_low_20d = df['low'].rolling(window=20).min()
                
                # Calculate range as percentage of price
                price_range_20d = (rolling_high_20d - rolling_low_20d) / rolling_low_20d
                
                # Tight ranges (less than 10%)
                tight_ranges = price_range_20d[price_range_20d < 0.1]
                range_bound_score = len(tight_ranges) / len(price_range_20d)
            
            # Categorize breakout suitability
            breakout_suitability = "unknown"
            
            if range_bound_score > 0.7:
                breakout_suitability = "excellent"  # Stock frequently forms ranges
            elif range_bound_score > 0.5:
                breakout_suitability = "good"
            elif range_bound_score > 0.3:
                breakout_suitability = "moderate"
            else:
                breakout_suitability = "poor"  # Stock rarely forms clear ranges
            
            # Adjust based on gap frequency
            if gap_stats and gap_stats["frequency"] > 0.2:
                # Frequent gaps can create more breakout opportunities
                if breakout_suitability == "moderate":
                    breakout_suitability = "good"
                elif breakout_suitability == "good":
                    breakout_suitability = "excellent"
            
            # Store characteristics
            characteristics[symbol] = {
                "volatility": volatility,
                "atr_percentage": atr_pct,
                "avg_daily_range": avg_daily_range,
                "range_bound_score": range_bound_score,
                "gap_statistics": gap_stats,
                "volume_statistics": volume_stats,
                "breakout_suitability": breakout_suitability,
                "suitable_for_strategy": breakout_suitability in ["excellent", "good"]
            }
        
        return characteristics 