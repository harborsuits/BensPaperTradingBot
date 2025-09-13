#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Range Trading Strategy

This module implements a range trading strategy for forex markets,
focusing on identifying and trading within established price ranges
during consolidation periods.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
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
    'regime_compatibility_scores': {'trending': 0.4, 'ranging': 0.9, 'volatile': 0.3, 'low_volatility': 0.8, 'all_weather': 0.6}
})
class ForexRangeTradingStrategy(ForexBaseStrategy):
    """
    Range trading strategy for forex markets.
    
    This strategy identifies and trades within price ranges by:
    1. Detecting consolidation patterns using various indicators
    2. Identifying support and resistance levels using price action
    3. Trading bounces off range boundaries with confirmation
    4. Using oscillators to confirm overbought/oversold conditions
    """
    
    # Default strategy parameters
    DEFAULT_PARAMETERS = {
        # Range identification parameters
        'lookback_period': 50,      # Period to determine range formation
        'range_threshold': 0.03,    # Max range as % of price for range qualification
        'min_touches': 2,           # Min touches of support/resistance to confirm
        
        # Bollinger Band parameters
        'bb_period': 20,            # Period for Bollinger Bands
        'bb_std_dev': 2.0,          # Standard deviations for bands
        
        # RSI parameters for overbought/oversold
        'rsi_period': 14,           # Period for RSI calculation
        'rsi_overbought': 70,       # Threshold for overbought
        'rsi_oversold': 30,         # Threshold for oversold
        
        # Stochastic oscillator parameters
        'stoch_k_period': 14,       # %K period
        'stoch_d_period': 3,        # %D period
        
        # Trade management parameters
        'stop_loss_pips': 30,       # Stop loss in pips
        'take_profit_factor': 1.5,  # Take profit as multiple of stop loss
        
        # Session preferences (inherited from ForexBaseStrategy)
        'trading_sessions': [ForexSession.LONDON, ForexSession.NEWYORK],
        'avoid_news_releases': True, # Whether to avoid trading during news releases
        
        # Filter parameters
        'min_range_pips': 20,       # Minimum range size in pips
        'max_range_pips': 200,      # Maximum range size in pips
        'max_spread_pips': 3.0,     # Maximum allowed spread in pips
    }
    
    def __init__(self, name: str = "Forex Range Trading", 
                 parameters: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the forex range trading strategy.
        
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
        self.current_signals = {}    # Current trading signals
        self.identified_ranges = {}  # Tracked price ranges
        self.last_updates = {}       # Last update timestamps
        
        logger.info(f"Initialized {self.name} strategy")
        
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_time: datetime) -> Dict[str, Signal]:
        """
        Generate trade signals for range trading opportunities.
        
        Args:
            data: Dictionary mapping symbols to OHLCV DataFrames
            current_time: Current timestamp
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        signals = {}
        
        for symbol, ohlcv in data.items():
            # Skip pairs with insufficient data
            if len(ohlcv) < self.parameters['lookback_period']:
                logger.debug(f"Insufficient data for {symbol}, skipping")
                continue
                
            # Skip if currently avoiding news releases and news is expected
            if self.parameters['avoid_news_releases'] and self.is_news_release_time():
                logger.debug(f"Skipping {symbol} due to news release time")
                continue
                
            # Check if spread is too high
            current_spread = ohlcv['high'].iloc[-1] - ohlcv['low'].iloc[-1]
            pip_value = self.parameters['pip_value']
            spread_pips = current_spread / pip_value
            
            if spread_pips > self.parameters['max_spread_pips']:
                logger.debug(f"Spread too high for {symbol}: {spread_pips} pips")
                continue
                
            # Calculate range trading indicators
            indicators = self._calculate_range_indicators(ohlcv)
            
            # Detect price ranges and bounces
            range_details = self._detect_price_range(symbol, ohlcv, indicators)
            
            # If we have a valid range
            if range_details['is_in_range']:
                # Is symbol already in our tracked ranges?
                if symbol not in self.identified_ranges:
                    self.identified_ranges[symbol] = range_details
                    logger.info(f"Identified new range for {symbol}: support={range_details['support']:.5f}, resistance={range_details['resistance']:.5f}")
                
                # Evaluate for potential trade signals
                signal = self._evaluate_range_trade(symbol, ohlcv, indicators, range_details)
                
                # Only include signals with sufficient confidence
                if signal and signal.signal_type != SignalType.FLAT and signal.confidence >= 0.5:
                    signals[symbol] = signal
            else:
                # Remove from tracked ranges if it breaks out
                if symbol in self.identified_ranges:
                    logger.info(f"Range broken for {symbol}")
                    self.identified_ranges.pop(symbol, None)
        
        return signals
        
    def _calculate_range_indicators(self, ohlcv: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate indicators for range trading analysis.
        
        Args:
            ohlcv: DataFrame with OHLCV price data
            
        Returns:
            Dictionary of calculated indicators
        """
        # Make a copy of dataframe to avoid modifying original
        df = ohlcv.copy()
        
        # Get parameters
        bb_period = self.parameters['bb_period']
        bb_std_dev = self.parameters['bb_std_dev']
        rsi_period = self.parameters['rsi_period']
        stoch_k_period = self.parameters['stoch_k_period']
        stoch_d_period = self.parameters['stoch_d_period']
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        df['bb_std'] = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * bb_std_dev)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * bb_std_dev)
        
        # Calculate %B (position within Bollinger Bands)
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Calculate Bandwidth (range size indicator)
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate Stochastic Oscillator
        df['stoch_lowest_low'] = df['low'].rolling(window=stoch_k_period).min()
        df['stoch_highest_high'] = df['high'].rolling(window=stoch_k_period).max()
        df['stoch_k'] = 100 * ((df['close'] - df['stoch_lowest_low']) / 
                             (df['stoch_highest_high'] - df['stoch_lowest_low']))
        df['stoch_d'] = df['stoch_k'].rolling(window=stoch_d_period).mean()
        
        # Calculate Average True Range (ATR) for volatility assessment
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Calculate historical volatility
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=21).std() * np.sqrt(252)
        
        # Range identification metrics
        df['range_width'] = (df['high'].rolling(window=self.parameters['lookback_period']).max() - 
                            df['low'].rolling(window=self.parameters['lookback_period']).min())
        df['range_width_pct'] = df['range_width'] / df['close']
        
        # Return indicators as dictionary
        indicators = {
            'bb_upper': df['bb_upper'],
            'bb_middle': df['bb_middle'],
            'bb_lower': df['bb_lower'],
            'bb_pct': df['bb_pct'],
            'bb_bandwidth': df['bb_bandwidth'],
            'rsi': df['rsi'],
            'stoch_k': df['stoch_k'],
            'stoch_d': df['stoch_d'],
            'atr': df['atr'],
            'volatility': df['volatility'],
            'range_width': df['range_width'],
            'range_width_pct': df['range_width_pct']
        }
        
        return indicators
        
    def _detect_price_range(self, symbol: str, ohlcv: pd.DataFrame, indicators: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Detect if price is trading within a range and identify support/resistance levels.
        
        Args:
            symbol: Currency pair symbol
            ohlcv: DataFrame with OHLCV price data
            indicators: Dictionary of technical indicators
            
        Returns:
            Dictionary with range details including support, resistance, and range status
        """
        # Get the most recent values
        current_price = ohlcv['close'].iloc[-1]
        current_bb_pct = indicators['bb_pct'].iloc[-1]
        current_bandwidth = indicators['bb_bandwidth'].iloc[-1]
        current_range_width_pct = indicators['range_width_pct'].iloc[-1]
        
        # Get parameters
        lookback = self.parameters['lookback_period']
        range_threshold = self.parameters['range_threshold']
        min_touches = self.parameters['min_touches']
        min_range_pips = self.parameters['min_range_pips']
        max_range_pips = self.parameters['max_range_pips']
        pip_value = self.parameters['pip_value']
        
        # Initialize result dictionary
        result = {
            'is_in_range': False,
            'support': None,
            'resistance': None,
            'range_width': None,
            'range_width_pips': None,
            'touches_support': 0,
            'touches_resistance': 0,
            'confidence': 0.0,
        }
        
        # Check if bandwidth is low enough to indicate a range
        recent_bandwidths = indicators['bb_bandwidth'].iloc[-lookback:]
        mean_bandwidth = recent_bandwidths.mean()
        
        # Not enough data
        if np.isnan(mean_bandwidth):
            return result
        
        # Check for NaN values
        if (np.isnan(current_bandwidth) or np.isnan(current_range_width_pct)):
            return result
        
        # First condition: Price range is within threshold
        # Tight bollinger bands indicate range-bound market
        range_condition = current_range_width_pct < range_threshold
        
        if not range_condition:
            return result  # Not in a range
        
        # Find potential support and resistance levels using recent highs/lows
        lookback_highs = ohlcv['high'].iloc[-lookback:]
        lookback_lows = ohlcv['low'].iloc[-lookback:]
        
        # Group nearby price levels to find clusters (support/resistance zones)
        resistance_levels = self._find_price_clusters(lookback_highs, pip_value * 10)
        support_levels = self._find_price_clusters(lookback_lows, pip_value * 10)
        
        # Find most prominent levels
        if resistance_levels and support_levels:
            # Get the most touched levels
            resistance = max(resistance_levels, key=lambda x: x[1])[0]
            support = max(support_levels, key=lambda x: x[1])[0]
            
            # Count touches of support/resistance
            touches_resistance = max(resistance_levels, key=lambda x: x[1])[1]
            touches_support = max(support_levels, key=lambda x: x[1])[1]
            
            # Check if range size is within limits
            range_width = resistance - support
            range_width_pips = range_width / pip_value
            
            valid_range_size = min_range_pips <= range_width_pips <= max_range_pips
            
            # Check if we have enough touches to confirm the range
            enough_touches = touches_support >= min_touches and touches_resistance >= min_touches
            
            # Check if current price is within the range
            price_in_range = support <= current_price <= resistance
            
            # Calculate distance from boundaries as percentage of range
            if range_width > 0 and price_in_range:
                distance_from_support = (current_price - support) / range_width
                distance_from_resistance = (resistance - current_price) / range_width
                # Close to boundary = better signal
                closest_boundary = min(distance_from_support, distance_from_resistance)
                # Convert to a 0-1 score where 1 means very close to boundary
                boundary_score = 1 - min(closest_boundary * 4, 0.9)  # Cap at 0.9
            else:
                boundary_score = 0
            
            # Combine conditions and calculate confidence
            result['is_in_range'] = valid_range_size and enough_touches and price_in_range
            
            # Store range details
            result['support'] = support
            result['resistance'] = resistance
            result['range_width'] = range_width
            result['range_width_pips'] = range_width_pips
            result['touches_support'] = touches_support
            result['touches_resistance'] = touches_resistance
            
            # Calculate confidence based on multiple factors
            if result['is_in_range']:
                # Stronger ranges have more touches
                touch_score = min((touches_support + touches_resistance) / (min_touches * 4), 1.0)
                # Tighter ranges are more reliable
                size_score = 1 - (range_width_pips / max_range_pips)
                # Pure range confidence is a combination of factors
                result['confidence'] = 0.5 + (touch_score * 0.2 + size_score * 0.1 + boundary_score * 0.2)
        
        return result
    
    def _find_price_clusters(self, price_series: pd.Series, tolerance: float) -> List[Tuple[float, int]]:
        """
        Group nearby price points into clusters to identify support/resistance zones.
        
        Args:
            price_series: Series of prices to analyze
            tolerance: Maximum difference to consider prices as part of the same cluster
            
        Returns:
            List of tuples containing (cluster_price, number_of_points)
        """
        # Sort prices
        sorted_prices = sorted(price_series.dropna().tolist())
        
        if not sorted_prices:
            return []
        
        # Group nearby prices
        clusters = []
        current_cluster = [sorted_prices[0]]
        
        for price in sorted_prices[1:]:
            if price - current_cluster[0] <= tolerance:
                # Add to current cluster
                current_cluster.append(price)
            else:
                # Start a new cluster
                avg_price = sum(current_cluster) / len(current_cluster)
                clusters.append((avg_price, len(current_cluster)))
                current_cluster = [price]
        
        # Add the last cluster
        if current_cluster:
            avg_price = sum(current_cluster) / len(current_cluster)
            clusters.append((avg_price, len(current_cluster)))
            
        return clusters
        
    def _evaluate_range_trade(self, symbol: str, ohlcv: pd.DataFrame, indicators: Dict[str, pd.Series], 
                           range_details: Dict[str, Any]) -> Optional[Signal]:
        """
        Evaluate potential range trading opportunities and generate signals.
        
        Args:
            symbol: Currency pair symbol
            ohlcv: DataFrame with OHLCV price data
            indicators: Dictionary of technical indicators
            range_details: Dictionary with range information
            
        Returns:
            Signal object with trade recommendation or None
        """
        # If not in a range, no signal
        if not range_details['is_in_range']:
            return None
            
        # Get current values
        current_price = ohlcv['close'].iloc[-1]
        current_rsi = indicators['rsi'].iloc[-1]
        current_stoch_k = indicators['stoch_k'].iloc[-1]
        current_stoch_d = indicators['stoch_d'].iloc[-1]
        atr = indicators['atr'].iloc[-1]
        
        # Get range boundaries
        support = range_details['support']
        resistance = range_details['resistance']
        range_width = range_details['range_width']
        
        # Get parameters
        rsi_oversold = self.parameters['rsi_oversold']
        rsi_overbought = self.parameters['rsi_overbought']
        stop_loss_pips = self.parameters['stop_loss_pips']
        take_profit_factor = self.parameters['take_profit_factor']
        pip_value = self.parameters['pip_value']
        
        # Check for NaN values
        if np.isnan(current_rsi) or np.isnan(current_stoch_k) or np.isnan(current_stoch_d):
            return None
            
        # Default is flat signal
        signal_type = SignalType.FLAT
        confidence = 0.0
        entry_price = current_price
        stop_loss = None
        take_profit = None
        
        # Calculate distance from boundaries as percentage of range
        # Lower value means closer to boundary
        distance_from_support = (current_price - support) / range_width
        distance_from_resistance = (resistance - current_price) / range_width
        
        # Near support - potential LONG opportunity
        near_support = distance_from_support < 0.2  # Within 20% of range from support
        
        # Near resistance - potential SHORT opportunity
        near_resistance = distance_from_resistance < 0.2  # Within 20% of range from resistance
        
        # Proximity multiplier - closer to boundary means stronger signal
        proximity = 1 - min(distance_from_support, distance_from_resistance) * 3
        proximity = max(0.5, min(proximity, 0.9))  # Limit to 0.5-0.9 range
        
        # LONG signal conditions (near support)
        if near_support:
            # Oversold condition
            oversold = current_rsi < rsi_oversold
            
            # Stochastic confirms oversold
            stoch_oversold = current_stoch_k < 25 and current_stoch_d < 25
            
            # Stochastic K crossing above D
            stoch_crossover = (current_stoch_k > current_stoch_d and 
                              indicators['stoch_k'].iloc[-2] <= indicators['stoch_d'].iloc[-2])
            
            # Combine conditions with weights
            long_score = 0.4  # Base score for near support
            if oversold: long_score += 0.2
            if stoch_oversold: long_score += 0.2
            if stoch_crossover: long_score += 0.2
            
            # Scale by proximity to boundary
            confidence = long_score * proximity * range_details['confidence']
            
            # Generate LONG signal if confidence is high enough
            if confidence >= 0.5:
                signal_type = SignalType.LONG
                stop_loss = support - (stop_loss_pips * pip_value)
                take_profit = current_price + (stop_loss_pips * take_profit_factor * pip_value)
        
        # SHORT signal conditions (near resistance)
        elif near_resistance:
            # Overbought condition
            overbought = current_rsi > rsi_overbought
            
            # Stochastic confirms overbought
            stoch_overbought = current_stoch_k > 75 and current_stoch_d > 75
            
            # Stochastic K crossing below D
            stoch_crossover = (current_stoch_k < current_stoch_d and 
                              indicators['stoch_k'].iloc[-2] >= indicators['stoch_d'].iloc[-2])
            
            # Combine conditions with weights
            short_score = 0.4  # Base score for near resistance
            if overbought: short_score += 0.2
            if stoch_overbought: short_score += 0.2
            if stoch_crossover: short_score += 0.2
            
            # Scale by proximity to boundary
            confidence = short_score * proximity * range_details['confidence']
            
            # Generate SHORT signal if confidence is high enough
            if confidence >= 0.5:
                signal_type = SignalType.SHORT
                stop_loss = resistance + (stop_loss_pips * pip_value)
                take_profit = current_price - (stop_loss_pips * take_profit_factor * pip_value)
        
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
                    'range_support': support,
                    'range_resistance': resistance,
                    'rsi': current_rsi,
                    'stochastic_k': current_stoch_k,
                    'stochastic_d': current_stoch_d
                }
            )
            
            # Log signal details
            signal_desc = 'LONG' if signal_type == SignalType.LONG else 'SHORT'
            logger.info(
                f"Generated {signal_desc} range signal for {symbol} with confidence {confidence:.2f}, "
                f"price={current_price:.5f}, support={support:.5f}, resistance={resistance:.5f}"
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
        # Range trading strategies perform well in consolidation/sideways markets
        # and poorly in trending markets (opposite of trend-following strategies)
        compatibility_map = {
            # Trending regimes - worst for range trading
            MarketRegime.BULL_TREND: 0.30,      # Poor compatibility with bull trends
            MarketRegime.BEAR_TREND: 0.35,      # Poor compatibility with bear trends
            
            # Volatile regimes - moderately compatible with proper stops
            MarketRegime.HIGH_VOLATILITY: 0.50, # Moderate compatibility with volatile markets
            
            # Sideways/ranging regimes - best for range trading
            MarketRegime.CONSOLIDATION: 0.90,   # Excellent compatibility with consolidation
            MarketRegime.LOW_VOLATILITY: 0.85,  # Strong compatibility with low vol markets
            
            # Default for unknown regimes
            MarketRegime.UNKNOWN: 0.60          # Above average compatibility with unknown conditions
        }
        
        # Return the compatibility score or default to 0.5 if regime unknown
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
        if market_regime == MarketRegime.CONSOLIDATION:
            # For strong consolidation, use tighter ranges
            optimized_params['range_threshold'] = 0.02
            optimized_params['min_touches'] = 3      # Require more touches
            optimized_params['bb_std_dev'] = 1.8     # Tighter Bollinger Bands
            optimized_params['stop_loss_pips'] = 20  # Tighter stops
            
        elif market_regime == MarketRegime.LOW_VOLATILITY:
            # For low volatility, use smaller ranges and tighter stops
            optimized_params['range_threshold'] = 0.015
            optimized_params['min_range_pips'] = 15
            optimized_params['max_range_pips'] = 150
            optimized_params['stop_loss_pips'] = 15
            
        elif market_regime == MarketRegime.HIGH_VOLATILITY:
            # For high volatility, use wider ranges and stops
            optimized_params['range_threshold'] = 0.04
            optimized_params['min_range_pips'] = 30
            optimized_params['max_range_pips'] = 300
            optimized_params['bb_std_dev'] = 2.5     # Wider Bollinger Bands
            optimized_params['stop_loss_pips'] = 40  # Wider stops
            
        elif market_regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
            # For trending markets, be more selective with ranges
            optimized_params['range_threshold'] = 0.025
            optimized_params['min_touches'] = 4       # Require more touches to confirm range
            optimized_params['stop_loss_pips'] = 25   # Normal stops
        
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
            metadata={'strategy_type': 'forex', 'category': 'range_trading'}
        )
        self.event_bus.publish(event)
