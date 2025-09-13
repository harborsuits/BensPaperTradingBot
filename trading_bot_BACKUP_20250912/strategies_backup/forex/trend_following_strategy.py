#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Trend-Following Strategy

This module implements a trend-following strategy for forex markets,
using moving averages and ADX to identify and trade strong trends.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

from trading_bot.strategies.base.forex_base import ForexBaseStrategy, ForexSession
from trading_bot.strategies.strategy_template import Signal, SignalType, TimeFrame, MarketRegime
from trading_bot.event_system import EventBus
from trading_bot.event_system.event_types import EventType, Event

logger = logging.getLogger(__name__)

class ForexTrendFollowingStrategy(ForexBaseStrategy):
    """Trend-following strategy for forex markets.
    
    This strategy identifies and follows strong trends using a combination of:
    1. Moving average crossovers (fast/slow)
    2. ADX (Average Directional Index) for trend strength confirmation
    3. Support/resistance levels for entry/exit points
    4. Session-specific adjustments
    """
    
    # Default strategy parameters
    DEFAULT_PARAMETERS = {
        # MA parameters
        'fast_ma_period': 8,
        'slow_ma_period': 21,
        'ma_type': 'ema',  # 'sma', 'ema', or 'wma'
        
        # Trend strength parameters
        'adx_period': 14,
        'adx_threshold': 25,  # Minimum ADX value for trend confirmation
        
        # Trade management parameters  
        'stop_loss_atr_multiple': 1.5,
        'take_profit_atr_multiple': 2.5,
        'trailing_stop_activation': 1.0,  # ATR multiple profit before activating
        
        # Session preferences (inherited from ForexBaseStrategy)
        'trading_sessions': [ForexSession.LONDON, ForexSession.NEWYORK],
        'session_boost_factor': 1.2,  # Confidence boost during preferred sessions
        
        # Filter parameters
        'min_volatility_percentile': 40,  # Minimum volatility (ATR percentile)
        'max_spread_pips': 3.0,  # Maximum allowed spread in pips
        'trade_only_major_pairs': False,  # Whether to trade only major pairs
    }
    
    def __init__(self, name: str = "Forex Trend-Following", 
                 parameters: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the forex trend-following strategy.
        
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
        self.current_signals = {}  # Current trading signals
        self.active_trends = {}   # Track identified trends
        self.last_updates = {}    # Last update timestamps
        
        logger.info(f"Initialized {self.name} strategy")
        
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_time: datetime) -> Dict[str, Signal]:
        """
        Generate trade signals based on trend-following indicators.
        
        Args:
            data: Dictionary mapping symbols to OHLCV DataFrames
            current_time: Current timestamp
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        signals = {}
        
        for symbol, ohlcv in data.items():
            # Skip pairs with insufficient data
            if len(ohlcv) < self.parameters['slow_ma_period'] + 10:
                logger.debug(f"Insufficient data for {symbol}, skipping")
                continue
                
            # Skip if not during preferred trading sessions (if enabled)
            if not self.is_current_session_active() and self.parameters.get('strict_session_filter', False):
                continue
                
            # Calculate indicators
            indicators = self._calculate_trend_indicators(ohlcv)
            
            # Skip if spread is too high
            current_spread = ohlcv['high'].iloc[-1] - ohlcv['low'].iloc[-1]
            pip_value = self.parameters['pip_value']
            spread_pips = current_spread / pip_value
            
            if spread_pips > self.parameters['max_spread_pips']:
                logger.debug(f"Spread too high for {symbol}: {spread_pips} pips")
                continue
                
            # Generate signal based on indicators
            signal = self._evaluate_trend(symbol, ohlcv, indicators)
            
            # Adjust signal based on trading session
            if signal and signal.signal_type != SignalType.FLAT:
                adjusted_signals = self.adjust_for_trading_session({symbol: signal})
                signal = adjusted_signals[symbol]
                
                # Only include signals with sufficient confidence
                if signal.confidence >= 0.5:
                    signals[symbol] = signal
                    
        # Track active trends
        self._update_active_trends(signals)
        
        return signals
        
    def _calculate_trend_indicators(self, ohlcv: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate trend-following technical indicators.
        
        Args:
            ohlcv: DataFrame with OHLCV price data
            
        Returns:
            Dictionary of calculated indicators
        """
        # Make a copy of dataframe to avoid modifying original
        df = ohlcv.copy()
        
        # Get parameters
        fast_period = self.parameters['fast_ma_period']
        slow_period = self.parameters['slow_ma_period']
        ma_type = self.parameters['ma_type']
        adx_period = self.parameters['adx_period']
        
        # Calculate moving averages
        if ma_type.lower() == 'sma':
            df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
            df['slow_ma'] = df['close'].rolling(window=slow_period).mean()
        elif ma_type.lower() == 'ema':
            df['fast_ma'] = df['close'].ewm(span=fast_period, adjust=False).mean()
            df['slow_ma'] = df['close'].ewm(span=slow_period, adjust=False).mean()
        elif ma_type.lower() == 'wma':
            # Weighted moving average
            weights_fast = np.arange(1, fast_period + 1)
            weights_slow = np.arange(1, slow_period + 1)
            df['fast_ma'] = df['close'].rolling(window=fast_period).apply(
                lambda x: np.sum(weights_fast * x) / weights_fast.sum(), raw=True)
            df['slow_ma'] = df['close'].rolling(window=slow_period).apply(
                lambda x: np.sum(weights_slow * x) / weights_slow.sum(), raw=True)
        
        # Calculate moving average crossover signals
        df['ma_diff'] = df['fast_ma'] - df['slow_ma']
        df['ma_cross'] = np.where(
            (df['ma_diff'].shift(1) <= 0) & (df['ma_diff'] > 0), 1,  # Bullish crossover
            np.where((df['ma_diff'].shift(1) >= 0) & (df['ma_diff'] < 0), -1, 0)  # Bearish crossover
        )
        
        # Calculate ATR for volatility assessment
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Calculate ADX (Average Directional Index) for trend strength
        # Step 1: Calculate +DM, -DM, +DI, -DI
        df['up_move'] = df['high'].diff()
        df['down_move'] = df['low'].shift().diff(-1).abs()  # negative diff to get correct direction
        
        # +DM: high movement with positive change and greater than low movement
        df['+dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0),
                            df['up_move'], 0)
        # -DM: low movement with positive change and greater than high movement
        df['-dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0),
                            df['down_move'], 0)
        
        # Step 2: Smooth DM values with the ATR period
        df['+di'] = 100 * (df['+dm'].rolling(window=adx_period).mean() / df['atr'])
        df['-di'] = 100 * (df['-dm'].rolling(window=adx_period).mean() / df['atr'])
        
        # Step 3: Calculate directional movement index (DX)
        df['di_diff'] = np.abs(df['+di'] - df['-di'])
        df['di_sum'] = df['+di'] + df['-di']
        df['dx'] = 100 * (df['di_diff'] / df['di_sum'])
        
        # Step 4: Calculate ADX as smoothed DX
        df['adx'] = df['dx'].rolling(window=adx_period).mean()
        
        # Calculate MACD for additional trend confirmation
        df['macd_line'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        df['signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd_line'] - df['signal_line']
        
        # Return indicators as dictionary
        indicators = {
            'fast_ma': df['fast_ma'],
            'slow_ma': df['slow_ma'],
            'ma_diff': df['ma_diff'],
            'ma_cross': df['ma_cross'],
            'atr': df['atr'],
            '+di': df['+di'],
            '-di': df['-di'],
            'adx': df['adx'],
            'macd_line': df['macd_line'],
            'signal_line': df['signal_line'],
            'macd_hist': df['macd_hist']
        }
        
        return indicators
        
    def _evaluate_trend(self, symbol: str, ohlcv: pd.DataFrame, indicators: Dict[str, pd.Series]) -> Signal:
        """
        Evaluate trend strength and direction to generate a trading signal.
        
        Args:
            symbol: Currency pair symbol
            ohlcv: DataFrame with OHLCV price data
            indicators: Dictionary of technical indicators
            
        Returns:
            Signal object with trade recommendation
        """
        # Get current values (most recent bar)
        current_price = ohlcv['close'].iloc[-1]
        adx = indicators['adx'].iloc[-1]
        plus_di = indicators['+di'].iloc[-1] 
        minus_di = indicators['-di'].iloc[-1]
        ma_cross = indicators['ma_cross'].iloc[-1]
        atr = indicators['atr'].iloc[-1]
        macd_hist = indicators['macd_hist'].iloc[-1]
        macd_line = indicators['macd_line'].iloc[-1]
        signal_line = indicators['signal_line'].iloc[-1]
        
        # Check for NaN values
        if np.isnan(adx) or np.isnan(plus_di) or np.isnan(minus_di):
            return Signal(symbol=symbol, signal_type=SignalType.FLAT, 
                         confidence=0.0, timestamp=ohlcv.index[-1])
        
        # Default to FLAT signal
        signal_type = SignalType.FLAT
        confidence = 0.0
        entry_price = None
        stop_loss = None
        take_profit = None
        
        # Check for strong trend
        adx_threshold = self.parameters['adx_threshold']
        strong_trend = adx > adx_threshold
        
        # Determine trend direction
        bullish_trend = plus_di > minus_di
        bearish_trend = minus_di > plus_di
        
        # Bullish conditions
        if strong_trend and bullish_trend:
            # Look for confirmation from other indicators
            ma_bullish = indicators['ma_diff'].iloc[-1] > 0
            macd_bullish = macd_line > signal_line
            
            # Calculate confidence based on indicator alignment
            base_confidence = 0.5  # Start with base confidence
            
            # Add confidence for each aligned indicator
            if ma_bullish:
                base_confidence += 0.1
            if ma_cross == 1:  # Recent bullish crossover
                base_confidence += 0.2
            if macd_bullish:
                base_confidence += 0.1
            if macd_hist > 0 and macd_hist > macd_hist.shift(1).iloc[-1]:
                base_confidence += 0.1
            
            # Scale confidence by trend strength
            adx_factor = min(adx / 50.0, 1.0)  # Normalize ADX to [0, 1]
            confidence = min(base_confidence * (1.0 + adx_factor), 0.95)
            
            # Generate LONG signal
            signal_type = SignalType.LONG
            entry_price = current_price
            
            # Set stop loss and take profit based on ATR
            stop_loss = entry_price - (self.parameters['stop_loss_atr_multiple'] * atr)
            take_profit = entry_price + (self.parameters['take_profit_atr_multiple'] * atr)
        
        # Bearish conditions
        elif strong_trend and bearish_trend:
            # Look for confirmation from other indicators
            ma_bearish = indicators['ma_diff'].iloc[-1] < 0
            macd_bearish = macd_line < signal_line
            
            # Calculate confidence based on indicator alignment
            base_confidence = 0.5  # Start with base confidence
            
            # Add confidence for each aligned indicator
            if ma_bearish:
                base_confidence += 0.1
            if ma_cross == -1:  # Recent bearish crossover
                base_confidence += 0.2
            if macd_bearish:
                base_confidence += 0.1
            if macd_hist < 0 and macd_hist < macd_hist.shift(1).iloc[-1]:
                base_confidence += 0.1
            
            # Scale confidence by trend strength
            adx_factor = min(adx / 50.0, 1.0)  # Normalize ADX to [0, 1]
            confidence = min(base_confidence * (1.0 + adx_factor), 0.95)
            
            # Generate SHORT signal
            signal_type = SignalType.SHORT
            entry_price = current_price
            
            # Set stop loss and take profit based on ATR
            stop_loss = entry_price + (self.parameters['stop_loss_atr_multiple'] * atr)
            take_profit = entry_price - (self.parameters['take_profit_atr_multiple'] * atr)
        
        # Apply trading session analysis if available
        if self.is_current_session_active():
            session_boost = self.parameters.get('session_boost_factor', 1.0)
            confidence = min(confidence * session_boost, 0.95)
        
        # Create the signal object
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
                'adx': adx
            }
        )
        
        # Log the signal
        signal_desc = 'FLAT' if signal_type == SignalType.FLAT else \
                     'LONG' if signal_type == SignalType.LONG else 'SHORT'
        logger.info(
            f"Generated {signal_desc} signal for {symbol} with confidence {confidence:.2f}, "
            f"ADX={adx:.2f}, ATR={atr:.6f}"
        )
        
        return signal
        
    def _update_active_trends(self, signals: Dict[str, Signal]):
        """
        Update and maintain the list of active trends.
        
        Args:
            signals: Dictionary of newly generated signals
        """
        current_time = datetime.now()
        
        # Update active trends based on new signals
        for symbol, signal in signals.items():
            if signal.signal_type != SignalType.FLAT and signal.confidence >= 0.5:
                # Create or update trend record
                self.active_trends[symbol] = {
                    'direction': signal.signal_type,
                    'start_time': self.active_trends.get(symbol, {}).get('start_time', current_time),
                    'last_update': current_time,
                    'strength': signal.confidence,
                    'entry_price': signal.metadata.get('entry_price'),
                    'stop_loss': signal.metadata.get('stop_loss'),
                    'take_profit': signal.metadata.get('take_profit'),
                    'atr': signal.metadata.get('atr')
                }
            elif symbol in self.active_trends:
                # If we get a FLAT signal for a previously tracked trend, keep it but flag for review
                self.active_trends[symbol]['needs_review'] = True
                self.active_trends[symbol]['last_update'] = current_time
        
        # Remove expired trends (no update for more than 24 hours)
        expired_symbols = []
        for symbol, trend in self.active_trends.items():
            hours_since_update = (current_time - trend.get('last_update', current_time)).total_seconds() / 3600
            if hours_since_update > 24:
                expired_symbols.append(symbol)
                
        for symbol in expired_symbols:
            logger.info(f"Removing expired trend for {symbol}")
            self.active_trends.pop(symbol, None)
        
        # Emit event for active trends update
        if self.active_trends:
            event_data = {
                'strategy_name': self.name,
                'active_trends': self.active_trends,
                'trend_count': len(self.active_trends),
                'timestamp': current_time
            }
            
            # Create and publish event
            event = Event(
                event_type=EventType.SIGNAL_GENERATED,  # Using SIGNAL_GENERATED from the system EventType enum
                source=self.name,
                data=event_data,
                metadata={'strategy_type': 'forex', 'category': 'trend_following'}
            )
            self.event_bus.publish(event)
            
    def get_compatibility_score(self, market_regime: MarketRegime) -> float:
        """
        Calculate compatibility score with the given market regime.
        
        Args:
            market_regime: The current market regime
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        # Trend-following strategies perform well in trending markets and poorly in sideways markets
        compatibility_map = {
            # Trending regimes - best for trend following
            MarketRegime.BULL_TREND: 0.90,  # Strong compatibility with bull trends
            MarketRegime.BEAR_TREND: 0.85,  # Strong compatibility with bear trends
            
            # Volatile regimes - trend following with stops can work
            MarketRegime.HIGH_VOLATILITY: 0.60,  # Moderate compatibility with volatile markets
            
            # Sideways/ranging regimes - worst for trend following
            MarketRegime.CONSOLIDATION: 0.30,  # Low compatibility with ranging markets
            MarketRegime.LOW_VOLATILITY: 0.25,  # Low compatibility with low vol markets
            
            # Default for unknown regimes
            MarketRegime.UNKNOWN: 0.40  # Below average compatibility with unknown conditions
        }
        
        # Return the compatibility score or default to 0.5 if regime unknown
        return compatibility_map.get(market_regime, 0.5)
        
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
        if market_regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
            # For strong trends, use faster moving averages and lower ADX threshold
            optimized_params['fast_ma_period'] = 5
            optimized_params['slow_ma_period'] = 15
            optimized_params['adx_threshold'] = 20
            optimized_params['take_profit_atr_multiple'] = 3.0
            
        elif market_regime == MarketRegime.HIGH_VOLATILITY:
            # For volatile regimes, use slower moving averages and higher ADX threshold
            optimized_params['fast_ma_period'] = 10
            optimized_params['slow_ma_period'] = 30
            optimized_params['adx_threshold'] = 30
            optimized_params['stop_loss_atr_multiple'] = 2.0
            
        elif market_regime in [MarketRegime.CONSOLIDATION, MarketRegime.LOW_VOLATILITY]:
            # For sideways markets, be more conservative
            optimized_params['fast_ma_period'] = 12
            optimized_params['slow_ma_period'] = 30
            optimized_params['adx_threshold'] = 35  # Higher threshold to only catch strongest trends
            optimized_params['stop_loss_atr_multiple'] = 1.0  # Tighter stops
            
        # Log the optimization
        logger.info(f"Optimized {self.name} for {market_regime} regime")
        
        return optimized_params
