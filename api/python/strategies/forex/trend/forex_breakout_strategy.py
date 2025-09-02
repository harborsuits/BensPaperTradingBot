#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Breakout Strategy

This module implements a breakout trading strategy for forex markets,
focusing on identifying and trading price movements that break through
significant support or resistance levels after periods of consolidation.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from trading_bot.strategies.factory.strategy_registry import register_strategy

from trading_bot.strategies.base.forex_base import ForexBaseStrategy, ForexSession
from trading_bot.strategies.strategy_template import Signal, SignalType, TimeFrame, MarketRegime
from trading_bot.event_system import EventBus
from trading_bot.event_system.event_types import EventType, Event

logger = logging.getLogger(__name__)


@register_strategy({
    'asset_class': 'forex',
    'strategy_type': 'trend',
    'compatible_market_regimes': ['all_weather'],
    'timeframe': 'daily',
    'regime_compatibility_scores': {'trending': 0.7, 'ranging': 0.5, 'volatile': 0.9, 'low_volatility': 0.3, 'all_weather': 0.6}
})
class ForexBreakoutStrategy(ForexBaseStrategy):
    """
    Breakout strategy for forex markets.
    
    This strategy identifies and trades breakouts by:
    1. Detecting significant support and resistance levels
    2. Monitoring price action for breakout confirmations
    3. Using volume and volatility to confirm valid breakouts
    4. Applying filters to reduce false breakouts
    """
    
    # Default strategy parameters
    DEFAULT_PARAMETERS = {
        # Breakout identification parameters
        'lookback_period': 30,         # Period to identify support/resistance
        'consolidation_days': 10,      # Min days of consolidation before breakout
        'breakout_threshold': 0.01,    # Min % move to qualify as breakout
        
        # Volatility parameters
        'atr_period': 14,              # ATR calculation period
        'atr_multiplier': 1.5,         # Multiplier for volatility filtering
        
        # Volume confirmation
        'volume_threshold': 1.5,       # Volume increase required (multiple of avg)
        
        # False breakout protection
        'confirmation_candles': 2,     # Candles needed to confirm breakout
        'rejection_threshold': 0.5,    # % of breakout range to consider rejected
        
        # Trade management parameters
        'stop_loss_pips': 50,          # Stop loss in pips
        'take_profit_factor': 2.0,     # Take profit as multiple of stop loss
        
        # Donchian channel parameters
        'donchian_period': 20,         # Period for Donchian channels
        
        # Session preferences (inherited from ForexBaseStrategy)
        'trading_sessions': [ForexSession.LONDON, ForexSession.NEWYORK],
        'avoid_news_releases': True,   # Whether to avoid trading during news
        
        # Filter parameters
        'min_breakout_pips': 20,       # Minimum breakout size in pips
        'max_spread_pips': 3.0,        # Maximum allowed spread in pips
    }
    
    def __init__(self, name: str = "Forex Breakout Strategy", 
                 parameters: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the forex breakout strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters (will be merged with DEFAULT_PARAMETERS)
            metadata: Strategy metadata
        """
        # Merge default parameters with ForexBaseStrategy defaults
        forex_params = self.DEFAULT_FOREX_PARAMS.copy()
        forex_params.update(self.DEFAULT_PARAMETERS)
        
        # Override with user-provided parameters if any
        if parameters:
            forex_params.update(parameters)
        
        # Initialize the base strategy
        super().__init__(name=name, parameters=forex_params, metadata=metadata)
        
        # Register with the event system
        self.event_bus = EventBus()
        
        # Strategy state
        self.current_signals = {}       # Current trading signals
        self.identified_breakouts = {}  # Tracked breakout levels
        self.last_updates = {}          # Last update timestamps
        
        logger.info(f"Initialized {self.name} strategy")
        
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_time: datetime):
        """
        Generate trade signals for breakout opportunities.
        
        Args:
            data: Dictionary mapping symbols to OHLCV DataFrames
            current_time: Current timestamp
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        signals = {}
        
        # Process each symbol
        for symbol, ohlcv in data.items():
            # Skip if we don't have enough data
            if len(ohlcv) < self.parameters['lookback_period']:
                logger.warning(f"Not enough data for {symbol} to generate breakout signals")
                continue
            
            # Check if symbol passes filters (spread, session, etc.)
            if not self._passes_filters(symbol, ohlcv, current_time):
                continue
            
            # Calculate indicators for breakout detection
            indicators = self._calculate_breakout_indicators(ohlcv)
            
            # Identify significant levels and consolidation patterns
            breakout_levels = self._identify_breakout_levels(symbol, ohlcv, indicators)
            
            # Evaluate for breakout signals
            signal = self._evaluate_breakout(symbol, ohlcv, indicators, breakout_levels)
            
            if signal:
                # Store the signal
                signals[symbol] = signal
                self.current_signals[symbol] = signal
                
                # Emit the signal event
                self.emit_strategy_event({
                    'symbol': symbol,
                    'signal_type': signal.signal_type.name,
                    'confidence': signal.confidence,
                    'entry_price': signal.metadata.get('entry_price'),
                    'stop_loss': signal.metadata.get('stop_loss'),
                    'take_profit': signal.metadata.get('take_profit'),
                    'breakout_type': signal.metadata.get('breakout_type')
                })
                
                logger.info(
                    f"Generated {signal.signal_type.name} breakout signal for {symbol} "
                    f"with confidence {signal.confidence:.2f}"
                )
        
        return signals
    
    def _calculate_breakout_indicators(self, ohlcv: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate indicators for breakout analysis.
        
        Args:
            ohlcv: DataFrame with OHLCV price data
            
        Returns:
            Dictionary of calculated indicators
        """
        # Copy parameters to local variables for readability
        atr_period = self.parameters['atr_period']
        donchian_period = self.parameters['donchian_period']
        
        # Calculate indicators
        indicators = {}
        
        # Calculate Average True Range (ATR) for volatility measurement
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        
        # True Range calculations
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR calculation
        indicators['atr'] = tr.rolling(window=atr_period).mean()
        
        # Donchian Channels for range identification
        indicators['upper_band'] = high.rolling(window=donchian_period).max()
        indicators['lower_band'] = low.rolling(window=donchian_period).min()
        indicators['mid_band'] = (indicators['upper_band'] + indicators['lower_band']) / 2
        
        # Calculate volume metrics
        if 'volume' in ohlcv.columns:
            # Volume moving average
            indicators['volume_ma'] = ohlcv['volume'].rolling(window=20).mean()
            # Volume ratio (current vs average)
            indicators['volume_ratio'] = ohlcv['volume'] / indicators['volume_ma']
        else:
            # If volume data is missing, use placeholder values
            indicators['volume_ma'] = pd.Series(1.0, index=ohlcv.index)
            indicators['volume_ratio'] = pd.Series(1.0, index=ohlcv.index)
        
        # Calculate volatility ratio for breakout confirmation
        indicators['volatility_ratio'] = tr / indicators['atr']
        
        # Price rate of change for momentum confirmation
        indicators['price_roc'] = close.pct_change(periods=5) * 100
        
        return indicators
    
    def _identify_breakout_levels(self, symbol: str, ohlcv: pd.DataFrame, 
                                 indicators: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Identify significant support/resistance levels and consolidation patterns.
        
        Args:
            symbol: Currency pair symbol
            ohlcv: DataFrame with OHLCV price data
            indicators: Dictionary of technical indicators
            
        Returns:
            Dictionary with breakout levels and context
        """
        # Copy parameters to local variables for readability
        lookback_period = self.parameters['lookback_period']
        consolidation_days = self.parameters['consolidation_days']
        
        # Get recent price data
        recent_high = ohlcv['high'].iloc[-lookback_period:]
        recent_low = ohlcv['low'].iloc[-lookback_period:]
        recent_close = ohlcv['close'].iloc[-lookback_period:]
        
        # Calculate key price levels
        donchian_high = indicators['upper_band'].iloc[-1]
        donchian_low = indicators['lower_band'].iloc[-1]
        
        # Check for consolidation pattern (narrowing price range)
        is_consolidating = False
        consolidation_range = 0.0
        
        # Calculate the range over the consolidation_days window
        if len(ohlcv) >= consolidation_days:
            recent_range = recent_high.max() - recent_low.min()
            consolidation_high = ohlcv['high'].iloc[-consolidation_days:].max()
            consolidation_low = ohlcv['low'].iloc[-consolidation_days:].min()
            consolidation_range = consolidation_high - consolidation_low
            
            # Calculate average daily range during consolidation
            daily_ranges = ohlcv['high'].iloc[-consolidation_days:] - ohlcv['low'].iloc[-consolidation_days:]
            avg_daily_range = daily_ranges.mean()
            
            # Consolidation is identified by a narrowing range compared to the lookback_period
            is_consolidating = (consolidation_range < recent_range * 0.6) and (avg_daily_range < indicators['atr'].iloc[-1])
        
        # Identify potential breakout levels
        resistance_levels = self._find_resistance_levels(ohlcv, lookback_period)
        support_levels = self._find_support_levels(ohlcv, lookback_period)
        
        # Get current and previous close prices
        current_close = ohlcv['close'].iloc[-1]
        previous_close = ohlcv['close'].iloc[-2]
        
        # Check if price is near a significant level
        nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_close)) if resistance_levels else None
        nearest_support = min(support_levels, key=lambda x: abs(x - current_close)) if support_levels else None
        
        # Calculate distances to nearest levels in pips
        pip_value = self.parameters['pip_value']
        distance_to_resistance = (nearest_resistance - current_close) / pip_value if nearest_resistance else float('inf')
        distance_to_support = (current_close - nearest_support) / pip_value if nearest_support else float('inf')
        
        # Determine if we have potential breakout conditions
        potential_upside_breakout = (current_close > previous_close) and (distance_to_resistance < 20) 
        potential_downside_breakout = (current_close < previous_close) and (distance_to_support < 20)
        
        # Return the identified levels and context
        return {
            'resistance_levels': resistance_levels,
            'support_levels': support_levels,
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_support,
            'distance_to_resistance': distance_to_resistance,
            'distance_to_support': distance_to_support,
            'is_consolidating': is_consolidating,
            'consolidation_range': consolidation_range,
            'donchian_high': donchian_high,
            'donchian_low': donchian_low,
            'potential_upside_breakout': potential_upside_breakout,
            'potential_downside_breakout': potential_downside_breakout
        }
    
    def _find_resistance_levels(self, ohlcv: pd.DataFrame, lookback: int) -> List[float]:
        """
        Identify significant resistance levels based on price history.
        
        Args:
            ohlcv: DataFrame with OHLCV price data
            lookback: Number of periods to look back
            
        Returns:
            List of resistance price levels
        """
        # Extract relevant price data
        high = ohlcv['high'].iloc[-lookback:]
        close = ohlcv['close'].iloc[-lookback:]
        
        # Find local maxima (swing highs)
        potential_levels = []
        
        # Use swing high detection (current high higher than n bars on either side)
        window_size = 5
        for i in range(window_size, len(high) - window_size):
            if all(high.iloc[i] > high.iloc[i-window_size:i]) and all(high.iloc[i] > high.iloc[i+1:i+window_size+1]):
                potential_levels.append(high.iloc[i])
        
        # If we don't have enough levels, add recent highs
        if len(potential_levels) < 3:
            potential_levels.append(high.max())
            
        # Group nearby levels (within 0.3% of each other) to avoid duplicates
        grouped_levels = self._group_price_levels(potential_levels, 0.003)
        
        return grouped_levels
    
    def _find_support_levels(self, ohlcv: pd.DataFrame, lookback: int) -> List[float]:
        """
        Identify significant support levels based on price history.
        
        Args:
            ohlcv: DataFrame with OHLCV price data
            lookback: Number of periods to look back
            
        Returns:
            List of support price levels
        """
        # Extract relevant price data
        low = ohlcv['low'].iloc[-lookback:]
        close = ohlcv['close'].iloc[-lookback:]
        
        # Find local minima (swing lows)
        potential_levels = []
        
        # Use swing low detection (current low lower than n bars on either side)
        window_size = 5
        for i in range(window_size, len(low) - window_size):
            if all(low.iloc[i] < low.iloc[i-window_size:i]) and all(low.iloc[i] < low.iloc[i+1:i+window_size+1]):
                potential_levels.append(low.iloc[i])
        
        # If we don't have enough levels, add recent lows
        if len(potential_levels) < 3:
            potential_levels.append(low.min())
            
        # Group nearby levels (within 0.3% of each other) to avoid duplicates
        grouped_levels = self._group_price_levels(potential_levels, 0.003)
        
        return grouped_levels
    
    def _group_price_levels(self, levels: List[float], tolerance: float) -> List[float]:
        """
        Group price levels that are within the specified tolerance of each other.
        
        Args:
            levels: List of price levels
            tolerance: Maximum percentage difference to group levels
            
        Returns:
            List of grouped price levels
        """
        if not levels:
            return []
            
        # Sort levels
        sorted_levels = sorted(levels)
        
        # Group nearby levels
        grouped = []
        current_group = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # If this level is close to the current group's average, add it
            group_avg = sum(current_group) / len(current_group)
            if abs(level - group_avg) / group_avg <= tolerance:
                current_group.append(level)
            else:
                # Otherwise, finish the current group and start a new one
                grouped.append(sum(current_group) / len(current_group))
                current_group = [level]
        
        # Add the last group
        if current_group:
            grouped.append(sum(current_group) / len(current_group))
        
        return grouped
    
    def _evaluate_breakout(self, symbol: str, ohlcv: pd.DataFrame, indicators: Dict[str, pd.Series],
                          breakout_levels: Dict[str, Any]) -> Optional[Signal]:
        """
        Evaluate potential breakout trading opportunities and generate signals.
        
        Args:
            symbol: Currency pair symbol
            ohlcv: DataFrame with OHLCV price data
            indicators: Dictionary of technical indicators
            breakout_levels: Dictionary with breakout level information
            
        Returns:
            Signal object with trade recommendation or None
        """
        # Extract parameters
        breakout_threshold = self.parameters['breakout_threshold']
        volume_threshold = self.parameters['volume_threshold']
        pip_value = self.parameters['pip_value']
        stop_loss_pips = self.parameters['stop_loss_pips']
        take_profit_factor = self.parameters['take_profit_factor']
        min_breakout_pips = self.parameters['min_breakout_pips']
        
        # Extract current market data
        current_price = ohlcv['close'].iloc[-1]
        current_high = ohlcv['high'].iloc[-1]
        current_low = ohlcv['low'].iloc[-1]
        atr = indicators['atr'].iloc[-1]
        
        # Get volume confirmation if available
        volume_confirmed = indicators['volume_ratio'].iloc[-1] >= volume_threshold
        
        # Initialize signal variables
        signal_type = SignalType.FLAT
        entry_price = current_price
        stop_loss = 0.0
        take_profit = 0.0
        confidence = 0.0
        breakout_type = "none"
        
        # Check for Donchian channel breakouts
        donchian_high_breakout = current_high > breakout_levels['donchian_high']
        donchian_low_breakout = current_low < breakout_levels['donchian_low']
        
        # Check for support/resistance breakouts
        resistance_breakout = False
        support_breakout = False
        
        # If we have resistance levels, check for breakouts
        if breakout_levels['resistance_levels']:
            for level in breakout_levels['resistance_levels']:
                # Check if price has exceeded the resistance level by the threshold
                if current_price > level * (1 + breakout_threshold):
                    resistance_breakout = True
                    breakout_distance = abs(current_price - level) / pip_value
                    
                    # Verify the breakout is significant enough
                    if breakout_distance >= min_breakout_pips:
                        signal_type = SignalType.LONG
                        stop_loss = level - (stop_loss_pips * pip_value)
                        take_profit = current_price + (stop_loss_pips * take_profit_factor * pip_value)
                        confidence = 0.6  # Base confidence
                        breakout_type = "resistance"
                        
                        # Add confidence based on volume confirmation
                        if volume_confirmed:
                            confidence += 0.2
                        
                        # Add confidence if this follows a consolidation pattern
                        if breakout_levels['is_consolidating']:
                            confidence += 0.2
                        
                        break
        
        # If no resistance breakout and we have support levels, check for support breakouts
        if signal_type == SignalType.FLAT and breakout_levels['support_levels']:
            for level in breakout_levels['support_levels']:
                # Check if price has broken below the support level by the threshold
                if current_price < level * (1 - breakout_threshold):
                    support_breakout = True
                    breakout_distance = abs(current_price - level) / pip_value
                    
                    # Verify the breakout is significant enough
                    if breakout_distance >= min_breakout_pips:
                        signal_type = SignalType.SHORT
                        stop_loss = level + (stop_loss_pips * pip_value)
                        take_profit = current_price - (stop_loss_pips * take_profit_factor * pip_value)
                        confidence = 0.6  # Base confidence
                        breakout_type = "support"
                        
                        # Add confidence based on volume confirmation
                        if volume_confirmed:
                            confidence += 0.2
                        
                        # Add confidence if this follows a consolidation pattern
                        if breakout_levels['is_consolidating']:
                            confidence += 0.2
                        
                        break
        
        # If no support/resistance breakouts, check for Donchian breakouts
        if signal_type == SignalType.FLAT:
            if donchian_high_breakout:
                signal_type = SignalType.LONG
                stop_loss = current_price - (stop_loss_pips * pip_value)
                take_profit = current_price + (stop_loss_pips * take_profit_factor * pip_value)
                confidence = 0.5  # Lower base confidence for Donchian breakouts
                breakout_type = "donchian_high"
                
                # Add confidence based on volume confirmation
                if volume_confirmed:
                    confidence += 0.2
                
            elif donchian_low_breakout:
                signal_type = SignalType.SHORT
                stop_loss = current_price + (stop_loss_pips * pip_value)
                take_profit = current_price - (stop_loss_pips * take_profit_factor * pip_value)
                confidence = 0.5  # Lower base confidence for Donchian breakouts
                breakout_type = "donchian_low"
                
                # Add confidence based on volume confirmation
                if volume_confirmed:
                    confidence += 0.2
        
        # Create signal if we have a trade
        if signal_type != SignalType.FLAT:
            signal = Signal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                timestamp=ohlcv.index[-1],
                metadata={
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'atr': atr,
                    'volume_ratio': indicators['volume_ratio'].iloc[-1],
                    'breakout_type': breakout_type,
                    'is_consolidation_breakout': breakout_levels['is_consolidating']
                }
            )
            
            # Log signal details
            signal_desc = 'LONG' if signal_type == SignalType.LONG else 'SHORT'
            logger.info(
                f"Generated {signal_desc} breakout signal for {symbol} with confidence {confidence:.2f}, "
                f"breakout type: {breakout_type}"
            )
            
            return signal
        
        return None
    
    def get_compatibility_score(self, market_regime: MarketRegime) -> float:
        """
        Calculate compatibility score with the given market regime.
        
        Args:
            market_regime: The current market regime
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        # Breakout strategies perform well in transitional markets between
        # consolidation and trending conditions, and in high volatility environments
        compatibility_map = {
            # Trending regimes - moderate to good compatibility 
            MarketRegime.BULL_TREND: 0.65,      # Good compatibility with established bull trends
            MarketRegime.BEAR_TREND: 0.65,      # Good compatibility with established bear trends
            
            # Volatile regimes - best for breakout strategies
            MarketRegime.HIGH_VOLATILITY: 0.90, # Excellent compatibility with volatile markets
            
            # Sideways/ranging regimes - moderate compatibility when consolidating before breakout
            MarketRegime.CONSOLIDATION: 0.60,   # Moderate compatibility with consolidation
            MarketRegime.LOW_VOLATILITY: 0.40,  # Low compatibility with low vol markets
            
            # Default for unknown regimes
            MarketRegime.UNKNOWN: 0.60          # Above average compatibility with unknown conditions
        }
        
        # Return the compatibility score or default to 0.6 if regime unknown
        return compatibility_map.get(market_regime, 0.6)
        
    def optimize_for_regime(self, market_regime: MarketRegime) -> Dict[str, Any]:
        """
        Optimize strategy parameters for the given market regime.
        
        Args:
            market_regime: The current market regime
            
        Returns:
            Dictionary of optimized parameters
        """
        # Start with current parameters
        optimized_params = self.parameters.copy()
        
        # Adjust parameters based on regime
        if market_regime == MarketRegime.HIGH_VOLATILITY:
            # For high volatility, use larger confirmations but aggressive profit targets
            optimized_params['breakout_threshold'] = 0.015      # Higher threshold for stronger moves
            optimized_params['confirmation_candles'] = 3        # More candles to confirm
            optimized_params['take_profit_factor'] = 3.0        # Larger profit targets
            optimized_params['stop_loss_pips'] = 60            # Wider stops
            optimized_params['min_breakout_pips'] = 30         # Larger breakouts expected
            
        elif market_regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
            # For trending markets, look for trend continuation breakouts
            optimized_params['breakout_threshold'] = 0.01
            optimized_params['confirmation_candles'] = 2
            optimized_params['take_profit_factor'] = 2.5
            optimized_params['atr_multiplier'] = 1.2           # Less stringent volatility filter
            
        elif market_regime == MarketRegime.CONSOLIDATION:
            # For consolidation, focus on the strongest breakouts
            optimized_params['breakout_threshold'] = 0.02      # Higher threshold to avoid false breakouts
            optimized_params['confirmation_candles'] = 3       # More confirmation needed
            optimized_params['volume_threshold'] = 2.0         # Stronger volume confirmation required
            optimized_params['atr_multiplier'] = 1.8          # More stringent volatility filter
            
        elif market_regime == MarketRegime.LOW_VOLATILITY:
            # For low volatility, focus on tight ranges with minimal noise
            optimized_params['breakout_threshold'] = 0.008     # Lower threshold for smaller moves
            optimized_params['stop_loss_pips'] = 30            # Tighter stops
            optimized_params['min_breakout_pips'] = 15         # Accept smaller breakouts
            optimized_params['atr_multiplier'] = 2.0           # More stringent volatility requirements
        
        # Log the optimization
        logger.info(f"Optimized {self.name} for {market_regime} regime")
        
        return optimized_params
    
    def emit_strategy_event(self, data: Dict[str, Any]):
        """
        Emit a strategy update event to the event bus.
        
        Args:
            data: Event data to emit
        """
        # Add standard fields
        event_data = {
            'strategy_name': self.name,
            'timestamp': datetime.now(),
            **data
        }
        
        # Create and publish event
        event = Event(
            event_type=EventType.SIGNAL_GENERATED,
            source=self.name,
            data=event_data,
            metadata={'strategy_type': 'forex', 'category': 'breakout'}
        )
        self.event_bus.publish(event)
