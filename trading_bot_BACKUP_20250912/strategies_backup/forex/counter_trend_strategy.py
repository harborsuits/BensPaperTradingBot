#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Counter-Trend Trading Strategy

This module implements a counter-trend strategy for forex markets,
identifying potential reversal points when trends are exhausting themselves.
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

class ForexCounterTrendStrategy(ForexBaseStrategy):
    """Counter-trend strategy for forex markets.
    
    This strategy identifies potential trend reversal points by:
    1. Detecting overbought/oversold conditions with oscillators
    2. Identifying price exhaustion at Bollinger Band extremes
    3. Recognizing divergence between price and momentum indicators
    4. Using Fibonacci retracements for reversal zones
    """
    
    # Default strategy parameters
    DEFAULT_PARAMETERS = {
        # Oscillator parameters
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'rsi_extreme_overbought': 80,
        'rsi_extreme_oversold': 20,
        
        # Bollinger Band parameters
        'bb_period': 20,
        'bb_std_dev': 2.0,
        'bb_squeeze_threshold': 0.5,
        
        # Divergence parameters
        'use_divergence': True,
        'divergence_lookback': 10,
        'min_divergence_percentage': 3.0,
        
        # Fibonacci parameters
        'use_fibonacci': True,
        'fib_lookback': 50,
        'fib_levels': [0.382, 0.5, 0.618, 0.786],
        
        # Confirmation parameters
        'min_confirmations': 2,
        'confirmation_timeout_bars': 3,
        'candlestick_confirmation': True,
        'use_volume_confirmation': True,
        
        # Trade management parameters  
        'stop_loss_atr_multiple': 1.0,  # Tighter than trend-following
        'take_profit_atr_multiple': 2.0,
        'use_partial_exits': True,
        'partial_exit_levels': [0.382, 0.618],
        'partial_exit_sizes': [0.3, 0.3],  # 30% at each level
        
        # Trend threshold parameters
        'min_trend_strength': 25,  # Minimum ADX for trend confirmation
        'max_trend_strength': 50,  # Avoid extremely strong trends
        
        # Risk parameters
        'max_risk_per_trade_percent': 0.008,  # 0.8% risk per trade (lower)
        
        # Session preferences
        'trading_sessions': [ForexSession.LONDON, ForexSession.NEWYORK],
    }
    
    def __init__(self, name: str = "Forex Counter-Trend", 
                 parameters: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the forex counter-trend strategy.
        
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
        self.current_signals = {}
        self.pending_setups = {}
        self.active_reversals = {}
        self.signal_confirmations = {}
        
        logger.info(f"Initialized {self.name} strategy")
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_time: datetime) -> Dict[str, Signal]:
        """
        Generate trade signals for counter-trend reversals.
        
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
            required_lookback = max(
                self.parameters['rsi_period'],
                self.parameters['bb_period'],
                self.parameters['fib_lookback']
            ) + 10
            
            if len(ohlcv) < required_lookback:
                logger.debug(f"Insufficient data for {symbol}, skipping")
                continue
            
            # Calculate indicators
            indicators = self._calculate_reversal_indicators(ohlcv)
            
            # Evaluate potential reversal setups
            signal = self._evaluate_reversal_setup(symbol, ohlcv, indicators, current_time)
            
            if signal:
                signals[symbol] = signal
                # Also store in current signals
                self.current_signals[symbol] = signal
        
        # Publish event with active reversal setups
        if self.active_reversals:
            event_data = {
                'strategy_name': self.name,
                'active_reversals': self.active_reversals,
                'reversal_count': len(self.active_reversals),
                'timestamp': current_time.isoformat()
            }
            
            event = Event(
                event_type=EventType.SIGNAL_GENERATED,
                source=self.name,
                data=event_data,
                metadata={'strategy_type': 'forex', 'category': 'counter_trend'}
            )
            self.event_bus.publish(event)
        
        return signals
    
    def _calculate_reversal_indicators(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for reversal detection.
        
        Args:
            ohlcv: DataFrame with OHLCV price data
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        # Calculate RSI
        rsi_period = self.parameters['rsi_period']
        delta = ohlcv['close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        
        rs = avg_gain / avg_loss
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        bb_period = self.parameters['bb_period']
        bb_std_dev = self.parameters['bb_std_dev']
        
        indicators['bb_middle'] = ohlcv['close'].rolling(window=bb_period).mean()
        indicators['bb_std'] = ohlcv['close'].rolling(window=bb_period).std()
        indicators['bb_upper'] = indicators['bb_middle'] + (indicators['bb_std'] * bb_std_dev)
        indicators['bb_lower'] = indicators['bb_middle'] - (indicators['bb_std'] * bb_std_dev)
        
        # Calculate BB width for squeeze detection
        indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']
        indicators['bb_width_percentile'] = indicators['bb_width'].rolling(window=100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=True
        )
        
        # Calculate Stochastic oscillator for divergence
        k_period = 14
        d_period = 3
        
        low_min = ohlcv['low'].rolling(window=k_period).min()
        high_max = ohlcv['high'].rolling(window=k_period).max()
        
        indicators['stoch_k'] = 100 * ((ohlcv['close'] - low_min) / (high_max - low_min))
        indicators['stoch_d'] = indicators['stoch_k'].rolling(window=d_period).mean()
        
        # Calculate ATR for volatility
        atr_period = 14
        high_low = ohlcv['high'] - ohlcv['low']
        high_close = np.abs(ohlcv['high'] - ohlcv['close'].shift())
        low_close = np.abs(ohlcv['low'] - ohlcv['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        indicators['atr'] = true_range.rolling(atr_period).mean()
        
        # Calculate ADX for trend strength
        adx_period = 14
        
        # Directional movement
        up_move = ohlcv['high'] - ohlcv['high'].shift()
        down_move = ohlcv['low'].shift() - ohlcv['low']
        
        pos_dm = up_move.copy()
        pos_dm[up_move <= down_move] = 0
        pos_dm[up_move <= 0] = 0
        
        neg_dm = down_move.copy()
        neg_dm[down_move <= up_move] = 0
        neg_dm[down_move <= 0] = 0
        
        # Smooth the indicators
        tr_smooth = true_range.rolling(window=adx_period).mean()
        pos_di = 100 * (pos_dm.rolling(window=adx_period).mean() / tr_smooth)
        neg_di = 100 * (neg_dm.rolling(window=adx_period).mean() / tr_smooth)
        
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        indicators['adx'] = dx.rolling(window=adx_period).mean()
        indicators['plus_di'] = pos_di
        indicators['minus_di'] = neg_di
        
        # Calculate short-term and medium-term trend
        indicators['ema5'] = ohlcv['close'].ewm(span=5, adjust=False).mean()
        indicators['ema20'] = ohlcv['close'].ewm(span=20, adjust=False).mean()
        
        # Fibonacci retracement calculations if enabled
        if self.parameters['use_fibonacci']:
            fib_lookback = self.parameters['fib_lookback']
            indicators['recent_high'] = ohlcv['high'].rolling(window=fib_lookback).max()
            indicators['recent_low'] = ohlcv['low'].rolling(window=fib_lookback).min()
            
            # Calculate Fibonacci levels (simplified for common use case)
            recent_range = indicators['recent_high'] - indicators['recent_low']
            
            # Store Fibonacci levels for both up and down trends
            for level in self.parameters['fib_levels']:
                # For downtrend retracements (high to low)
                indicators[f'fib_down_{int(level*1000)}'] = indicators['recent_high'] - (recent_range * level)
                
                # For uptrend retracements (low to high)
                indicators[f'fib_up_{int(level*1000)}'] = indicators['recent_low'] + (recent_range * level)
        
        return indicators
    
    def _evaluate_reversal_setup(self, 
                               symbol: str, 
                               ohlcv: pd.DataFrame, 
                               indicators: Dict[str, Any],
                               current_time: datetime) -> Optional[Signal]:
        """
        Evaluate potential trend reversal setup.
        
        Args:
            symbol: Currency pair symbol
            ohlcv: DataFrame with OHLCV price data
            indicators: Dictionary of technical indicators
            current_time: Current timestamp
            
        Returns:
            Signal object if a valid reversal setup is found
        """
        # Current values
        current_price = ohlcv['close'].iloc[-1]
        current_rsi = indicators['rsi'].iloc[-1]
        
        # Get trend strength
        adx = indicators['adx'].iloc[-1]
        min_trend = self.parameters['min_trend_strength']
        max_trend = self.parameters['max_trend_strength']
        
        # Skip if trend is too weak or too strong
        if adx < min_trend or adx > max_trend:
            return None
        
        # Determine the current trend direction
        ema_short = indicators['ema5'].iloc[-1]
        ema_medium = indicators['ema20'].iloc[-1]
        plus_di = indicators['plus_di'].iloc[-1]
        minus_di = indicators['minus_di'].iloc[-1]
        
        # Trend direction: 1 for up, -1 for down
        trend_direction = 1 if (ema_short > ema_medium and plus_di > minus_di) else -1
        
        # Count the confirmation signals
        confirmations = self._count_reversal_confirmations(symbol, ohlcv, indicators, trend_direction)
        
        # Check if we have enough confirmations
        min_confirmations = self.parameters['min_confirmations']
        
        if confirmations < min_confirmations:
            # Store as a pending setup if we have at least one confirmation
            if confirmations > 0:
                self.pending_setups[symbol] = {
                    'timestamp': current_time.isoformat(),
                    'price': current_price,
                    'trend_direction': trend_direction,
                    'confirmations': confirmations,
                    'needed': min_confirmations
                }
            return None
        
        # Calculate the counter-trend direction (opposite of trend)
        signal_direction = -trend_direction
        
        # Calculate stop loss and take profit levels
        atr = indicators['atr'].iloc[-1]
        stop_atr = self.parameters['stop_loss_atr_multiple']
        take_profit_atr = self.parameters['take_profit_atr_multiple']
        
        # For counter-trend, place stop beyond the recent extreme
        if signal_direction > 0:  # Buy signal (against downtrend)
            recent_low = ohlcv['low'].iloc[-5:].min()
            stop_loss = recent_low - (atr * 0.5)  # Just below recent low
        else:  # Sell signal (against uptrend)
            recent_high = ohlcv['high'].iloc[-5:].max()
            stop_loss = recent_high + (atr * 0.5)  # Just above recent high
        
        # Take profit based on ATR
        take_profit = current_price + (signal_direction * atr * take_profit_atr)
        
        # Calculate partial exit levels if enabled
        partial_exits = []
        if self.parameters['use_partial_exits']:
            price_range = abs(take_profit - current_price)
            
            for idx, level in enumerate(self.parameters['partial_exit_levels']):
                exit_price = current_price + (signal_direction * price_range * level)
                exit_size = self.parameters['partial_exit_sizes'][idx]
                
                partial_exits.append({
                    'level': level,
                    'price': exit_price,
                    'size': exit_size
                })
        
        # Calculate confidence based on number of confirmations
        # Start with base confidence
        base_confidence = 0.4  # Counter-trend is inherently less confident
        
        # Add confidence based on number of confirmations (diminishing returns)
        confirmation_boost = min(0.4, confirmations * 0.1)  # Max +0.4 from confirmations
        
        # Add confidence if RSI is in extreme territory against the trend
        rsi_boost = 0
        if (trend_direction > 0 and current_rsi > self.parameters['rsi_extreme_overbought']) or \
           (trend_direction < 0 and current_rsi < self.parameters['rsi_extreme_oversold']):
            rsi_boost = 0.1
        
        # Combine confidence factors
        confidence = min(0.9, base_confidence + confirmation_boost + rsi_boost)
        
        # Create signal
        signal = Signal(
            symbol=symbol,
            signal_type=SignalType.MARKET_OPEN,
            direction=signal_direction,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'strategy': self.name,
                'setup_type': 'counter_trend_reversal',
                'trend_direction': trend_direction,
                'signal_direction': signal_direction,
                'confirmations': confirmations,
                'atr': atr,
                'rsi': current_rsi,
                'adx': adx,
                'partial_exits': partial_exits
            }
        )
        
        # Register this as an active reversal
        self.active_reversals[symbol] = {
            'entry_time': current_time.isoformat(),
            'entry_price': current_price,
            'direction': signal_direction,
            'confirmations': confirmations,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
        
        # Reset the confirmation count for this symbol
        self.signal_confirmations[symbol] = 0
        
        return signal
    
    def _count_reversal_confirmations(self, 
                                    symbol: str, 
                                    ohlcv: pd.DataFrame, 
                                    indicators: Dict[str, Any],
                                    trend_direction: int) -> int:
        """
        Count the number of reversal confirmation signals.
        
        Args:
            symbol: Currency pair symbol
            ohlcv: DataFrame with OHLCV price data
            indicators: Dictionary of technical indicators
            trend_direction: Current trend direction (1 for up, -1 for down)
            
        Returns:
            Number of confirmation signals detected
        """
        confirmations = 0
        
        # Current price and indicator values
        current_price = ohlcv['close'].iloc[-1]
        
        # 1. RSI Confirmation - Check for overbought/oversold against the trend
        rsi = indicators['rsi'].iloc[-1]
        if (trend_direction > 0 and rsi > self.parameters['rsi_overbought']) or \
           (trend_direction < 0 and rsi < self.parameters['rsi_oversold']):
            confirmations += 1
        
        # 2. Bollinger Band Confirmation - Price reaching or exceeding the bands
        bb_upper = indicators['bb_upper'].iloc[-1]
        bb_lower = indicators['bb_lower'].iloc[-1]
        
        if (trend_direction > 0 and current_price >= bb_upper) or \
           (trend_direction < 0 and current_price <= bb_lower):
            confirmations += 1
        
        # 3. Bollinger Band Width - Check for potential squeeze
        bb_width_percentile = indicators['bb_width_percentile'].iloc[-1]
        if bb_width_percentile < self.parameters['bb_squeeze_threshold']:
            confirmations += 0.5  # Half point for this confirmation
        
        # 4. Divergence Confirmation - Check for divergence against the trend
        if self.parameters['use_divergence']:
            # Hidden divergence (price and oscillator moving in opposite directions)
            lookback = self.parameters['divergence_lookback']
            
            # Get price change
            price_change = (current_price - ohlcv['close'].iloc[-lookback]) / ohlcv['close'].iloc[-lookback] * 100
            
            # Get oscillator change
            rsi_change = indicators['rsi'].iloc[-1] - indicators['rsi'].iloc[-lookback]
            
            # Check for divergence
            price_up = price_change > 0
            rsi_up = rsi_change > 0
            
            # If price and RSI are moving in opposite directions
            if (price_up and not rsi_up) or (not price_up and rsi_up):
                # Only count if the change is significant
                if abs(price_change) > self.parameters['min_divergence_percentage']:
                    confirmations += 1
        
        # 5. Fibonacci Level Confirmation
        if self.parameters['use_fibonacci']:
            fib_confirmed = False
            
            # Loop through Fibonacci levels
            for level in self.parameters['fib_levels']:
                if trend_direction > 0:
                    # In uptrend, check if price is near a Fibonacci retracement level
                    fib_level = indicators[f'fib_down_{int(level*1000)}'].iloc[-1]
                    # If price is within 0.5 ATR of the Fibonacci level
                    if abs(current_price - fib_level) < (indicators['atr'].iloc[-1] * 0.5):
                        fib_confirmed = True
                        break
                else:
                    # In downtrend, check if price is near a Fibonacci retracement level
                    fib_level = indicators[f'fib_up_{int(level*1000)}'].iloc[-1]
                    # If price is within 0.5 ATR of the Fibonacci level
                    if abs(current_price - fib_level) < (indicators['atr'].iloc[-1] * 0.5):
                        fib_confirmed = True
                        break
            
            if fib_confirmed:
                confirmations += 1
        
        # 6. Candlestick Pattern Confirmation
        if self.parameters['candlestick_confirmation']:
            if trend_direction > 0:
                # Look for bearish reversal patterns
                
                # Bearish engulfing
                if (ohlcv['open'].iloc[-1] > ohlcv['close'].iloc[-2] and
                    ohlcv['close'].iloc[-1] < ohlcv['open'].iloc[-2]):
                    confirmations += 1
                
                # Shooting star
                elif (ohlcv['high'].iloc[-1] - max(ohlcv['open'].iloc[-1], ohlcv['close'].iloc[-1]) >
                     2 * abs(ohlcv['open'].iloc[-1] - ohlcv['close'].iloc[-1]) and
                     min(ohlcv['open'].iloc[-1], ohlcv['close'].iloc[-1]) - ohlcv['low'].iloc[-1] <
                     0.2 * (ohlcv['high'].iloc[-1] - ohlcv['low'].iloc[-1])):
                    confirmations += 1
            else:
                # Look for bullish reversal patterns
                
                # Bullish engulfing
                if (ohlcv['open'].iloc[-1] < ohlcv['close'].iloc[-2] and
                    ohlcv['close'].iloc[-1] > ohlcv['open'].iloc[-2]):
                    confirmations += 1
                
                # Hammer
                elif (min(ohlcv['open'].iloc[-1], ohlcv['close'].iloc[-1]) - ohlcv['low'].iloc[-1] >
                     2 * abs(ohlcv['open'].iloc[-1] - ohlcv['close'].iloc[-1]) and
                     ohlcv['high'].iloc[-1] - max(ohlcv['open'].iloc[-1], ohlcv['close'].iloc[-1]) <
                     0.2 * (ohlcv['high'].iloc[-1] - ohlcv['low'].iloc[-1])):
                    confirmations += 1
        
        # 7. Volume Confirmation
        if self.parameters['use_volume_confirmation']:
            # Check for volume spike
            avg_volume = ohlcv['volume'].iloc[-5:].mean()
            current_volume = ohlcv['volume'].iloc[-1]
            
            if current_volume > avg_volume * 1.5:
                confirmations += 0.5  # Half point for volume confirmation
        
        return confirmations
    
    def get_compatibility_score(self, market_regime: MarketRegime) -> float:
        """
        Calculate compatibility score with the given market regime.
        
        Args:
            market_regime: The current market regime
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        # Counter-trend strategies work best during reversals and ranges
        compatibility_map = {
            # Best regimes for counter-trend
            MarketRegime.VOLATILE_REVERSAL: 0.95,  # Excellent for counter-trend
            MarketRegime.RANGING: 0.85,           # Very good for counter-trend
            
            # Medium compatibility
            MarketRegime.CHOPPY: 0.75,            # Good for counter-trend
            MarketRegime.TRENDING_UP: 0.50,       # Medium - can work at exhaustion
            MarketRegime.TRENDING_DOWN: 0.50,     # Medium - can work at exhaustion
            
            # Worst regime for counter-trend
            MarketRegime.VOLATILE_BREAKOUT: 0.35, # Poor for counter-trend
            
            # Default for unknown regimes
            MarketRegime.UNKNOWN: 0.60            # Above average compatibility
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
        if market_regime == MarketRegime.VOLATILE_REVERSAL:
            # For volatile reversals, focus on quick entries and tighter stops
            optimized_params['rsi_overbought'] = 75          # More extreme threshold
            optimized_params['rsi_oversold'] = 25            # More extreme threshold
            optimized_params['min_confirmations'] = 2        # Fewer confirmations required
            optimized_params['stop_loss_atr_multiple'] = 0.8 # Tighter stops
            optimized_params['take_profit_atr_multiple'] = 2.5 # Larger targets
            
        elif market_regime == MarketRegime.RANGING:
            # For ranging markets, maximize mean reversion
            optimized_params['rsi_overbought'] = 65          # Less extreme for ranges
            optimized_params['rsi_oversold'] = 35            # Less extreme for ranges
            optimized_params['bb_std_dev'] = 1.8             # Tighter bands
            optimized_params['min_confirmations'] = 3        # More confirmations
            optimized_params['use_partial_exits'] = True     # Take partial profits
            
        elif market_regime == MarketRegime.CHOPPY:
            # For choppy markets, focus on oscillations
            optimized_params['rsi_overbought'] = 60          # Even less extreme
            optimized_params['rsi_oversold'] = 40            # Even less extreme
            optimized_params['bb_std_dev'] = 1.5             # Even tighter bands
            optimized_params['stop_loss_atr_multiple'] = 1.2 # Slightly wider stops
            optimized_params['take_profit_atr_multiple'] = 1.5 # Smaller targets
            
        elif market_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            # For trending markets, be very selective and use stronger confirmations
            optimized_params['rsi_overbought'] = 80          # More extreme
            optimized_params['rsi_oversold'] = 20            # More extreme
            optimized_params['min_confirmations'] = 4        # Require more confirmations
            optimized_params['min_divergence_percentage'] = 5.0 # Stronger divergence
            optimized_params['stop_loss_atr_multiple'] = 0.7 # Very tight stops
            
        elif market_regime == MarketRegime.VOLATILE_BREAKOUT:
            # For volatile breakouts, be extremely conservative
            optimized_params['rsi_overbought'] = 85          # Very extreme
            optimized_params['rsi_oversold'] = 15            # Very extreme
            optimized_params['min_confirmations'] = 5        # Many confirmations
            optimized_params['stop_loss_atr_multiple'] = 0.5 # Extremely tight stops
            optimized_params['max_risk_per_trade_percent'] = 0.005 # Half risk
        
        # Log the optimization
        logger.info(f"Optimized {self.name} for {market_regime} regime")
        
        return optimized_params
