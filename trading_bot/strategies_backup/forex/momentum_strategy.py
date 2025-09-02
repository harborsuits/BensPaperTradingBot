#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Momentum Strategy

This module implements a momentum trading strategy for forex markets,
focusing on identifying currencies with strong directional momentum
using various momentum indicators and oscillators.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

from trading_bot.strategies.base.forex_base import ForexBaseStrategy, ForexSession
from trading_bot.strategies.strategy_template import Signal, SignalType, TimeFrame, MarketRegime
from trading_bot.event_system import EventBus
from trading_bot.event_system.event_types import EventType, Event

logger = logging.getLogger(__name__)

class ForexMomentumStrategy(ForexBaseStrategy):
    """
    Momentum strategy for forex markets.
    
    This strategy identifies and trades currency pairs with strong momentum by:
    1. Using multiple momentum indicators (ROC, RSI, Momentum oscillator)
    2. Confirming momentum strength and direction
    3. Timing entries for accelerating price movement
    4. Implementing precise risk management based on volatility
    """
    
    # Default strategy parameters
    DEFAULT_PARAMETERS = {
        # Momentum indicator parameters
        'roc_period': 14,              # Rate of Change period
        'roc_threshold': 1.0,          # ROC threshold for signal generation
        'rsi_period': 14,              # RSI period
        'rsi_overbought': 70,          # RSI overbought threshold
        'rsi_oversold': 30,            # RSI oversold threshold
        'momentum_period': 10,         # Momentum oscillator period
        
        # Additional filters
        'adx_period': 14,              # ADX period for trend strength
        'adx_threshold': 25,           # ADX threshold for trend strength
        
        # Money management parameters
        'stop_loss_atr_mult': 1.5,     # Stop loss as multiple of ATR
        'take_profit_atr_mult': 3.0,   # Take profit as multiple of ATR
        'atr_period': 14,              # ATR period
        
        # Volume considerations
        'volume_ma_period': 20,        # Volume moving average period
        'volume_threshold': 1.2,       # Volume threshold as multiple of average
        
        # Session preferences (inherited from ForexBaseStrategy)
        'trading_sessions': [ForexSession.LONDON, ForexSession.NEWYORK],
        'avoid_news_releases': True,   # Whether to avoid trading during news releases
        
        # Filter parameters
        'min_atr_pips': 15,            # Minimum ATR in pips for sufficient volatility
        'max_spread_pips': 3.0,        # Maximum allowed spread in pips
    }
    
    def __init__(self, name: str = "Forex Momentum Strategy", 
                 parameters: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the forex momentum strategy.
        
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
        self.current_signals = {}      # Current trading signals
        self.last_updates = {}         # Last update timestamps
        self.momentum_states = {}      # Track momentum states for each symbol
        
        logger.info(f"Initialized {self.name} strategy")
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_time: datetime):
        """
        Generate trade signals for momentum opportunities.
        
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
            min_required = max(
                self.parameters['roc_period'],
                self.parameters['rsi_period'],
                self.parameters['momentum_period'],
                self.parameters['adx_period'],
                self.parameters['atr_period']
            ) + 10  # Add buffer for calculations
            
            if len(ohlcv) < min_required:
                logger.warning(f"Not enough data for {symbol} to generate momentum signals")
                continue
            
            # Check if symbol passes filters (spread, session, etc.)
            if not self._passes_filters(symbol, ohlcv, current_time):
                continue
            
            # Calculate momentum indicators
            indicators = self._calculate_momentum_indicators(ohlcv)
            
            # Evaluate momentum and generate signal
            signal = self._evaluate_momentum(symbol, ohlcv, indicators)
            
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
                    'roc': signal.metadata.get('roc'),
                    'rsi': signal.metadata.get('rsi'),
                    'momentum': signal.metadata.get('momentum')
                })
                
                logger.info(
                    f"Generated {signal.signal_type.name} momentum signal for {symbol} "
                    f"with confidence {signal.confidence:.2f}"
                )
        
        return signals
    
    def _calculate_momentum_indicators(self, ohlcv: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate momentum indicators for analysis.
        
        Args:
            ohlcv: DataFrame with OHLCV price data
            
        Returns:
            Dictionary of calculated indicators
        """
        # Copy parameters to local variables for readability
        roc_period = self.parameters['roc_period']
        rsi_period = self.parameters['rsi_period']
        momentum_period = self.parameters['momentum_period']
        adx_period = self.parameters['adx_period']
        atr_period = self.parameters['atr_period']
        
        # Calculate indicators
        indicators = {}
        
        # Rate of Change (ROC)
        indicators['roc'] = ((ohlcv['close'] / ohlcv['close'].shift(roc_period)) - 1) * 100
        
        # RSI calculation
        delta = ohlcv['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        
        # Avoid division by zero
        loss = loss.replace(0, 0.00001)
        
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Momentum oscillator (close price - n periods ago close price)
        indicators['momentum'] = ohlcv['close'] - ohlcv['close'].shift(momentum_period)
        
        # Normalized momentum (as percentage of price)
        indicators['momentum_pct'] = (indicators['momentum'] / ohlcv['close'].shift(momentum_period)) * 100
        
        # Calculate ATR for volatility
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
        
        # ATR as percentage of price
        indicators['atr_pct'] = (indicators['atr'] / ohlcv['close']) * 100
        
        # Calculate ADX for trend strength
        # First, calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm.abs()), 0)
        minus_dm = minus_dm.abs().where((minus_dm < 0) & (minus_dm.abs() > plus_dm), 0)
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(window=adx_period).sum() / tr.rolling(window=adx_period).sum())
        minus_di = 100 * (minus_dm.rolling(window=adx_period).sum() / tr.rolling(window=adx_period).sum())
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        indicators['adx'] = dx.rolling(window=adx_period).mean()
        
        # Volume analysis if available
        if 'volume' in ohlcv.columns:
            # Volume moving average
            volume_ma_period = self.parameters['volume_ma_period']
            indicators['volume_ma'] = ohlcv['volume'].rolling(window=volume_ma_period).mean()
            # Volume ratio (current vs average)
            indicators['volume_ratio'] = ohlcv['volume'] / indicators['volume_ma']
        else:
            # If volume data is missing, use placeholder values
            indicators['volume_ma'] = pd.Series(1.0, index=ohlcv.index)
            indicators['volume_ratio'] = pd.Series(1.0, index=ohlcv.index)
        
        # Acceleration of momentum (second derivative of price)
        indicators['momentum_acceleration'] = indicators['momentum_pct'].diff()
        
        return indicators
    
    def _evaluate_momentum(self, symbol: str, ohlcv: pd.DataFrame, 
                          indicators: Dict[str, pd.Series]) -> Optional[Signal]:
        """
        Evaluate momentum indicators and generate a trading signal if appropriate.
        
        Args:
            symbol: Currency pair symbol
            ohlcv: DataFrame with OHLCV price data
            indicators: Dictionary of technical indicators
            
        Returns:
            Signal object with trade recommendation or None
        """
        # Extract parameters
        roc_threshold = self.parameters['roc_threshold']
        rsi_overbought = self.parameters['rsi_overbought']
        rsi_oversold = self.parameters['rsi_oversold']
        adx_threshold = self.parameters['adx_threshold']
        volume_threshold = self.parameters['volume_threshold']
        stop_loss_atr_mult = self.parameters['stop_loss_atr_mult']
        take_profit_atr_mult = self.parameters['take_profit_atr_mult']
        
        # Get current values
        current_price = ohlcv['close'].iloc[-1]
        current_roc = indicators['roc'].iloc[-1]
        current_rsi = indicators['rsi'].iloc[-1]
        current_momentum = indicators['momentum_pct'].iloc[-1]
        current_adx = indicators['adx'].iloc[-1]
        current_volume_ratio = indicators['volume_ratio'].iloc[-1]
        current_atr = indicators['atr'].iloc[-1]
        current_momentum_accel = indicators['momentum_acceleration'].iloc[-1]
        
        # Initialize signal variables
        signal_type = SignalType.FLAT
        confidence = 0.0
        
        # Long signal conditions
        long_conditions = (
            current_roc > roc_threshold and                    # Positive ROC
            current_rsi < 70 and                               # Not overbought
            current_momentum > 0 and                           # Positive momentum
            current_adx > adx_threshold and                    # Strong trend
            current_momentum_accel > 0                         # Accelerating momentum
        )
        
        # Short signal conditions
        short_conditions = (
            current_roc < -roc_threshold and                   # Negative ROC
            current_rsi > 30 and                               # Not oversold
            current_momentum < 0 and                           # Negative momentum
            current_adx > adx_threshold and                    # Strong trend
            current_momentum_accel < 0                         # Accelerating momentum downward
        )
        
        # Enhanced long conditions with RSI and volume
        enhanced_long = (
            long_conditions and
            current_rsi < 50 and                               # RSI in lower half (more upside potential)
            current_volume_ratio > volume_threshold            # Above average volume
        )
        
        # Enhanced short conditions with RSI and volume
        enhanced_short = (
            short_conditions and
            current_rsi > 50 and                               # RSI in upper half (more downside potential)
            current_volume_ratio > volume_threshold            # Above average volume
        )
        
        # Optimal entry for long (RSI turning up from oversold)
        optimal_long_entry = (
            long_conditions and
            current_rsi > indicators['rsi'].iloc[-2] and       # RSI turning up
            current_rsi > rsi_oversold and                     # RSI emerged from oversold
            current_rsi < 50                                   # Still in lower half of range
        )
        
        # Optimal entry for short (RSI turning down from overbought)
        optimal_short_entry = (
            short_conditions and
            current_rsi < indicators['rsi'].iloc[-2] and       # RSI turning down
            current_rsi < rsi_overbought and                   # RSI emerged from overbought
            current_rsi > 50                                   # Still in upper half of range
        )
        
        # Determine signal type and confidence
        if optimal_long_entry:
            signal_type = SignalType.LONG
            confidence = 0.85  # Highest confidence for optimal entry
        elif optimal_short_entry:
            signal_type = SignalType.SHORT
            confidence = 0.85  # Highest confidence for optimal entry
        elif enhanced_long:
            signal_type = SignalType.LONG
            confidence = 0.75  # High confidence for enhanced conditions
        elif enhanced_short:
            signal_type = SignalType.SHORT
            confidence = 0.75  # High confidence for enhanced conditions
        elif long_conditions:
            signal_type = SignalType.LONG
            confidence = 0.65  # Standard confidence for basic conditions
        elif short_conditions:
            signal_type = SignalType.SHORT
            confidence = 0.65  # Standard confidence for basic conditions
        
        # Calculate risk management levels
        entry_price = current_price
        pip_value = self.parameters['pip_value']
        atr_pips = current_atr / pip_value
        
        # Set stop loss and take profit based on ATR
        if signal_type == SignalType.LONG:
            stop_loss = entry_price - (stop_loss_atr_mult * current_atr)
            take_profit = entry_price + (take_profit_atr_mult * current_atr)
        elif signal_type == SignalType.SHORT:
            stop_loss = entry_price + (stop_loss_atr_mult * current_atr)
            take_profit = entry_price - (take_profit_atr_mult * current_atr)
        else:
            stop_loss = 0.0
            take_profit = 0.0
            
        # Only generate signal if confidence is high enough
        if confidence >= 0.6 and signal_type != SignalType.FLAT:
            # Create the signal
            signal = Signal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                timestamp=ohlcv.index[-1],
                metadata={
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'roc': current_roc,
                    'rsi': current_rsi,
                    'momentum': current_momentum,
                    'adx': current_adx,
                    'atr': current_atr,
                    'atr_pips': atr_pips,
                    'volume_ratio': current_volume_ratio,
                    'momentum_acceleration': current_momentum_accel
                }
            )
            
            # Log the signal
            signal_desc = 'LONG' if signal_type == SignalType.LONG else 'SHORT'
            logger.info(
                f"Generated {signal_desc} momentum signal for {symbol} with confidence {confidence:.2f}, "
                f"ROC={current_roc:.2f}, RSI={current_rsi:.2f}, Momentum={current_momentum:.2f}, "
                f"ADX={current_adx:.2f}"
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
        # Momentum strategies perform exceptionally well in markets with emerging trends
        # where momentum is building and volatility is moderate to high
        compatibility_map = {
            # Trending regimes - good for momentum
            MarketRegime.BULL_TREND: 0.80,      # Strong compatibility with bull trends
            MarketRegime.BEAR_TREND: 0.95,      # Exceptional compatibility with bear trends
            
            # Volatile regimes - excellent for momentum
            MarketRegime.HIGH_VOLATILITY: 0.85, # Very strong compatibility with volatile markets
            
            # Sideways/ranging regimes - poor for momentum
            MarketRegime.CONSOLIDATION: 0.40,   # Poor compatibility with consolidation
            MarketRegime.LOW_VOLATILITY: 0.30,  # Poor compatibility with low vol markets
            
            # Default for unknown regimes
            MarketRegime.UNKNOWN: 0.60          # Moderate compatibility with unknown conditions
        }
        
        # Return the compatibility score or default to 0.60 if regime unknown
        return compatibility_map.get(market_regime, 0.60)
    
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
        if market_regime == MarketRegime.BULL_TREND:
            # For bull trends, focus on strong upward momentum
            optimized_params['roc_threshold'] = 0.8           # Lower threshold to catch more opportunities
            optimized_params['adx_threshold'] = 20            # Lower ADX requirement
            optimized_params['take_profit_atr_mult'] = 3.5    # Larger profit target in established trend
            optimized_params['stop_loss_atr_mult'] = 1.3      # Tighter stop loss in established trend
            
        elif market_regime == MarketRegime.BEAR_TREND:
            # For bear trends, focus on strong downward momentum
            optimized_params['roc_threshold'] = 0.8           # Lower threshold to catch more opportunities
            optimized_params['adx_threshold'] = 20            # Lower ADX requirement
            optimized_params['take_profit_atr_mult'] = 3.5    # Larger profit target in established trend
            optimized_params['stop_loss_atr_mult'] = 1.3      # Tighter stop loss in established trend
            
        elif market_regime == MarketRegime.HIGH_VOLATILITY:
            # For high volatility, focus on strong momentum bursts
            optimized_params['roc_threshold'] = 1.5           # Higher threshold for stronger moves
            optimized_params['adx_threshold'] = 25            # Moderate ADX requirement
            optimized_params['take_profit_atr_mult'] = 2.5    # Moderate profit target due to volatility
            optimized_params['stop_loss_atr_mult'] = 2.0      # Wider stop loss due to volatility
            optimized_params['volume_threshold'] = 1.5        # Require stronger volume confirmation
            
        elif market_regime == MarketRegime.CONSOLIDATION:
            # For consolidation, be very selective
            optimized_params['roc_threshold'] = 2.0           # Higher threshold to avoid false signals
            optimized_params['adx_threshold'] = 30            # Require stronger trend confirmation
            optimized_params['take_profit_atr_mult'] = 2.0    # Smaller profit target in consolidation
            optimized_params['stop_loss_atr_mult'] = 1.5      # Moderate stop loss
            optimized_params['volume_threshold'] = 2.0        # Require much stronger volume confirmation
            
        elif market_regime == MarketRegime.LOW_VOLATILITY:
            # For low volatility, focus on breakout momentum from tight ranges
            optimized_params['roc_threshold'] = 1.2           # Moderate threshold
            optimized_params['adx_threshold'] = 25            # Moderate ADX requirement
            optimized_params['take_profit_atr_mult'] = 4.0    # Larger profit target for breakouts
            optimized_params['stop_loss_atr_mult'] = 1.2      # Tighter stop loss given low volatility
            
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
            metadata={'strategy_type': 'forex', 'category': 'momentum'}
        )
        self.event_bus.publish(event)
