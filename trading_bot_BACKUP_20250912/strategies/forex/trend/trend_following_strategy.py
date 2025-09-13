#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Trend Following Strategy

This module implements a trend following strategy for forex markets using:
- Multiple moving averages (SMA, EMA, WMA)
- ADX for trend strength
- MACD for trend confirmation
- ATR for volatility measurement and position sizing
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from trading_bot.strategies_new.forex.base.forex_base_strategy import ForexBaseStrategy, ForexSession
from trading_bot.strategies_new.factory.strategy_registry import register_strategy, StrategyType, AssetClass, MarketRegime, TimeFrame
from trading_bot.core.event_system import EventBus, Event, EventType
from trading_bot.strategies_new.factory.strategy_template import Signal, SignalType

logger = logging.getLogger(__name__)

@register_strategy({
    'asset_class': 'forex',
    'strategy_type': 'trend_following',
    'compatible_market_regimes': ['trending', 'low_volatility'],
    'timeframe': 'swing',
    'regime_compatibility_scores': {
        'trending': 0.95,       # Highest compatibility with trending markets
        'ranging': 0.40,        # Poor compatibility with ranging markets
        'volatile': 0.60,       # Moderate compatibility with volatile markets
        'low_volatility': 0.75, # Good compatibility with low volatility markets
        'all_weather': 0.70     # Good overall compatibility
    },
    'optimal_parameters': {
        'trending': {
            'fast_ma_period': 8,
            'slow_ma_period': 21,
            'signal_ma_period': 9,
            'adx_period': 14,
            'adx_threshold': 25,
            'atr_period': 14,
            'atr_multiplier': 3.0
        },
        'low_volatility': {
            'fast_ma_period': 5,
            'slow_ma_period': 15,
            'signal_ma_period': 9,
            'adx_period': 14,
            'adx_threshold': 20,
            'atr_period': 14,
            'atr_multiplier': 2.5
        }
    }
})
class ForexTrendFollowingStrategy(ForexBaseStrategy):
    """
    Forex Trend Following Strategy
    
    This strategy identifies and follows trends in forex pairs using:
    - Multiple MA combinations (EMA, SMA, WMA)
    - ADX for trend strength confirmation
    - MACD for additional signal confirmation
    - ATR for volatility and position sizing
    
    It works best in trending market regimes and has built-in session awareness 
    to adjust trading during optimal forex sessions.
    """
    
    # Default parameters - can be overridden via constructor
    DEFAULT_PARAMS = {
        # Moving average parameters
        'fast_ma_type': 'ema',       # Type of fast MA: 'sma', 'ema', 'wma'
        'slow_ma_type': 'ema',       # Type of slow MA: 'sma', 'ema', 'wma'
        'fast_ma_period': 8,         # Period for fast MA
        'slow_ma_period': 21,        # Period for slow MA
        
        # MACD parameters
        'macd_fast_period': 12,      # Fast period for MACD
        'macd_slow_period': 26,      # Slow period for MACD
        'signal_ma_period': 9,       # Signal line period
        
        # ADX parameters
        'adx_period': 14,            # Period for ADX calculation
        'adx_threshold': 25,         # Minimum ADX for trend confirmation
        
        # ATR parameters
        'atr_period': 14,            # Period for ATR calculation
        'atr_multiplier': 3.0,       # Multiplier for ATR stop loss
        
        # Trading session parameters
        'preferred_sessions': [ForexSession.LONDON, ForexSession.NEWYORK],
        'trade_session_overlaps': True,
        
        # Signal parameters
        'min_trend_duration': 3,     # Minimum bars trend must exist for confirmation
        'profit_target_atr': 2.0,    # Profit target as ATR multiple
        'confidence_threshold': 0.7, # Minimum confidence for trade signals
        
        # Risk parameters
        'max_risk_per_trade_percent': 0.01  # 1% risk per trade
    }
    
    def __init__(self, name: str = "ForexTrendFollowingStrategy", 
                parameters: Optional[Dict[str, Any]] = None, 
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize Forex Trend Following Strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters (will be merged with DEFAULT_PARAMS)
            metadata: Strategy metadata
        """
        # Merge default parameters with provided parameters
        merged_params = self.DEFAULT_PARAMS.copy()
        if parameters:
            merged_params.update(parameters)
            
        # Initialize base class
        super().__init__(name, merged_params, metadata)
        
        # Initialize strategy-specific properties
        self.last_signals = {}
        self.trend_directions = {}
        self.trend_durations = {}
        
        logger.info(f"Initialized {name} with parameters: {merged_params}")
    
    def register_events(self, event_bus: EventBus) -> None:
        """
        Register strategy events with the event bus.
        
        Args:
            event_bus: Event bus to register with
        """
        # Register for market data updates
        event_bus.register(EventType.MARKET_DATA_UPDATED, self._on_market_data_updated)
        
        # Register for timeframe completed events
        event_bus.register(EventType.TIMEFRAME_COMPLETED, self._on_timeframe_completed)
        
        # Store reference to event bus for publishing
        self.event_bus = event_bus
        
        logger.info(f"{self.name} registered with event bus")
    
    def _on_market_data_updated(self, event: Event) -> None:
        """Handle market data updated events."""
        # Implementation details...
        pass
    
    def _on_timeframe_completed(self, event: Event) -> None:
        """
        Handle timeframe completed events.
        
        This is when we'll generate new trading signals based on the 
        completed candle data.
        """
        data = event.data
        timeframe = data.get('timeframe')
        symbol = data.get('symbol')
        
        # Only process our target timeframe
        if timeframe != self.parameters.get('timeframe'):
            return
        
        # Generate signals for the completed timeframe
        history = data.get('history')
        if history is not None and isinstance(history, pd.DataFrame):
            signals = self.generate_signals({symbol: history})
            
            # Publish signals if we have any
            if signals and self.event_bus:
                signal_event = Event(
                    event_type=EventType.STRATEGY_SIGNAL_GENERATED,
                    data={
                        'strategy': self.name,
                        'signals': signals,
                        'timestamp': datetime.now().isoformat()
                    }
                )
                self.event_bus.publish(signal_event)
    
    def calculate_indicators(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Calculate technical indicators for the strategy.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol for the data
            
        Returns:
            Dictionary of calculated indicators
        """
        # Get parameters
        fast_ma_period = self.parameters['fast_ma_period']
        slow_ma_period = self.parameters['slow_ma_period']
        fast_ma_type = self.parameters['fast_ma_type']
        slow_ma_type = self.parameters['slow_ma_type']
        adx_period = self.parameters['adx_period']
        atr_period = self.parameters['atr_period']
        
        # Calculate forex-specific indicators from base class
        forex_indicators = self.calculate_forex_indicators(data, symbol)
        
        # Calculate moving averages
        if fast_ma_type == 'sma':
            data['fast_ma'] = data['close'].rolling(window=fast_ma_period).mean()
        elif fast_ma_type == 'ema':
            data['fast_ma'] = data['close'].ewm(span=fast_ma_period, adjust=False).mean()
        elif fast_ma_type == 'wma':
            weights = np.arange(1, fast_ma_period + 1)
            data['fast_ma'] = data['close'].rolling(window=fast_ma_period).apply(
                lambda x: np.sum(weights * x) / weights.sum(), raw=True
            )
        
        if slow_ma_type == 'sma':
            data['slow_ma'] = data['close'].rolling(window=slow_ma_period).mean()
        elif slow_ma_type == 'ema':
            data['slow_ma'] = data['close'].ewm(span=slow_ma_period, adjust=False).mean()
        elif slow_ma_type == 'wma':
            weights = np.arange(1, slow_ma_period + 1)
            data['slow_ma'] = data['close'].rolling(window=slow_ma_period).apply(
                lambda x: np.sum(weights * x) / weights.sum(), raw=True
            )
        
        # Calculate MACD
        macd_fast = data['close'].ewm(span=self.parameters['macd_fast_period'], adjust=False).mean()
        macd_slow = data['close'].ewm(span=self.parameters['macd_slow_period'], adjust=False).mean()
        data['macd'] = macd_fast - macd_slow
        data['macd_signal'] = data['macd'].ewm(span=self.parameters['signal_ma_period'], adjust=False).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Merge with forex indicators
        indicators = {
            **forex_indicators,
            'fast_ma': data['fast_ma'].iloc[-1],
            'slow_ma': data['slow_ma'].iloc[-1],
            'ma_cross': (data['fast_ma'] > data['slow_ma']).iloc[-1],
            'ma_cross_prev': (data['fast_ma'] > data['slow_ma']).iloc[-2] if len(data) > 2 else False,
            'macd': data['macd'].iloc[-1],
            'macd_signal': data['macd_signal'].iloc[-1],
            'macd_histogram': data['macd_histogram'].iloc[-1],
            'macd_cross': (data['macd'] > data['macd_signal']).iloc[-1],
            'macd_cross_prev': (data['macd'] > data['macd_signal']).iloc[-2] if len(data) > 2 else False,
        }
        
        return indicators
    
    def generate_signals(self, universe: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate trading signals for the universe of symbols.
        
        Args:
            universe: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        signals = {}
        
        # Filter universe based on forex criteria
        filtered_universe = self.filter_universe(universe)
        
        # Calculate currency strength
        currency_strength = self.calculate_currency_strength(filtered_universe)
        
        # Check if we're in a preferred trading session
        in_preferred_session = self.is_current_session_active()
        
        # Process each symbol
        for symbol, data in filtered_universe.items():
            # Skip if not enough data
            if len(data) < max(self.parameters['slow_ma_period'], self.parameters['adx_period']):
                continue
            
            # Calculate indicators
            indicators = self.calculate_indicators(data.copy(), symbol)
            
            # Check for trend conditions
            adx_value = indicators['adx']
            trend_strength = adx_value >= self.parameters['adx_threshold']
            
            # MA and MACD cross conditions
            ma_bullish_cross = indicators['ma_cross'] and not indicators['ma_cross_prev']
            ma_bearish_cross = not indicators['ma_cross'] and indicators['ma_cross_prev']
            
            macd_bullish_cross = indicators['macd_cross'] and not indicators['macd_cross_prev']
            macd_bearish_cross = not indicators['macd_cross'] and indicators['macd_cross_prev']
            
            # Current trend direction
            if symbol not in self.trend_directions:
                self.trend_directions[symbol] = 0
                self.trend_durations[symbol] = 0
            
            # Update trend direction and duration
            current_trend = self.trend_directions[symbol]
            
            if indicators['ma_cross']:  # Bullish
                if current_trend <= 0:  # New bullish trend
                    self.trend_directions[symbol] = 1
                    self.trend_durations[symbol] = 1
                else:  # Continuing bullish trend
                    self.trend_durations[symbol] += 1
            elif not indicators['ma_cross']:  # Bearish
                if current_trend >= 0:  # New bearish trend
                    self.trend_directions[symbol] = -1
                    self.trend_durations[symbol] = 1
                else:  # Continuing bearish trend
                    self.trend_durations[symbol] += 1
            
            # Check if trend duration meets minimum requirement
            trend_duration_met = self.trend_durations[symbol] >= self.parameters['min_trend_duration']
            
            # Signal generation logic
            signal = None
            confidence = 0.0
            
            # Extract currency pair components
            base_currency, quote_currency = symbol.split('/') if '/' in symbol else (symbol[:3], symbol[3:])
            
            # Consider currency strength for signal
            base_strength = currency_strength.get(base_currency, 0)
            quote_strength = currency_strength.get(quote_currency, 0)
            relative_strength = base_strength - quote_strength
            
            # Long signal conditions
            if (ma_bullish_cross or macd_bullish_cross) and trend_strength and trend_duration_met:
                # Adjust confidence based on trend strength and duration
                confidence = 0.7 + (min(adx_value, 50) / 100) + (min(self.trend_durations[symbol], 10) / 100)
                
                # Adjust based on currency strength
                if relative_strength > 0:
                    confidence = min(1.0, confidence + 0.1)
                else:
                    confidence = max(0.5, confidence - 0.1)
                
                # Adjust based on trading session
                if not in_preferred_session:
                    confidence *= 0.8
                
                # Create signal if confidence meets threshold
                if confidence >= self.parameters['confidence_threshold']:
                    # Calculate ATR for position sizing and stop-loss
                    atr = indicators['atr']
                    stop_loss = data['close'].iloc[-1] - (atr * self.parameters['atr_multiplier'])
                    take_profit = data['close'].iloc[-1] + (atr * self.parameters['profit_target_atr'])
                    
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.LONG,
                        confidence=confidence,
                        entry_price=data['close'].iloc[-1],
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'strategy': self.name,
                            'indicators': {
                                'adx': adx_value,
                                'trend_duration': self.trend_durations[symbol],
                                'ma_fast': indicators['fast_ma'],
                                'ma_slow': indicators['slow_ma'],
                                'atr': atr
                            }
                        }
                    )
            
            # Short signal conditions
            elif (ma_bearish_cross or macd_bearish_cross) and trend_strength and trend_duration_met:
                # Adjust confidence based on trend strength and duration
                confidence = 0.7 + (min(adx_value, 50) / 100) + (min(self.trend_durations[symbol], 10) / 100)
                
                # Adjust based on currency strength
                if relative_strength < 0:
                    confidence = min(1.0, confidence + 0.1)
                else:
                    confidence = max(0.5, confidence - 0.1)
                
                # Adjust based on trading session
                if not in_preferred_session:
                    confidence *= 0.8
                
                # Create signal if confidence meets threshold
                if confidence >= self.parameters['confidence_threshold']:
                    # Calculate ATR for position sizing and stop-loss
                    atr = indicators['atr']
                    stop_loss = data['close'].iloc[-1] + (atr * self.parameters['atr_multiplier'])
                    take_profit = data['close'].iloc[-1] - (atr * self.parameters['profit_target_atr'])
                    
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SHORT,
                        confidence=confidence,
                        entry_price=data['close'].iloc[-1],
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'strategy': self.name,
                            'indicators': {
                                'adx': adx_value,
                                'trend_duration': self.trend_durations[symbol],
                                'ma_fast': indicators['fast_ma'],
                                'ma_slow': indicators['slow_ma'],
                                'atr': atr
                            }
                        }
                    )
            
            # Store signal if generated
            if signal:
                signals[symbol] = signal
                self.last_signals[symbol] = signal
        
        # Apply session-based adjustments to signals
        adjusted_signals = self.adjust_for_trading_session(signals)
        
        return adjusted_signals
        
    def calculate_position_size(self, signal: Signal, account_balance: float) -> float:
        """
        Calculate position size for the signal based on risk management rules.
        
        Args:
            signal: Trading signal
            account_balance: Current account balance
            
        Returns:
            Position size in units
        """
        # Extract parameters
        max_risk_percent = self.parameters['max_risk_per_trade_percent']
        risk_amount = account_balance * max_risk_percent
        
        # Calculate stop loss distance in pips
        entry_price = signal.entry_price
        stop_loss = signal.stop_loss
        pip_value = self.parameters['pip_value']
        
        # Convert price difference to pips
        if signal.signal_type == SignalType.LONG:
            stop_loss_pips = (entry_price - stop_loss) / pip_value
        else:
            stop_loss_pips = (stop_loss - entry_price) / pip_value
        
        # Calculate position size based on pips
        return self.calculate_position_size_pips(
            symbol=signal.symbol,
            entry_price=entry_price,
            stop_loss_pips=stop_loss_pips,
            risk_amount=risk_amount
        )
