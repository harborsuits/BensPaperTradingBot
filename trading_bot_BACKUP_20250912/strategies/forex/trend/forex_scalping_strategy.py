#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Scalping Strategy

This module implements a scalping strategy for forex markets,
focusing on very short-term trades with tight stops and small profit targets.
Optimized for high-liquidity trading sessions.
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
    'regime_compatibility_scores': {}
})
class ForexScalpingStrategy(ForexBaseStrategy):
    """
    Scalping strategy for forex markets.
    
    This strategy focuses on quick, short-term trades with:
    1. Very short holding periods (minutes to hours)
    2. Tight stop losses to minimize risk
    3. Small profit targets (5-15 pips)
    4. High win rate approach
    5. Session-specific optimization (best during high liquidity periods)
    """
    
    # Default strategy parameters
    DEFAULT_PARAMETERS = {
        # Basic parameters
        'timeframe': TimeFrame.MINUTE_5,        # 5-minute timeframe for scalping
        'profit_target_pips': 8,          # Take profit at 8 pips
        'stop_loss_pips': 5,              # Stop loss at 5 pips
        'risk_per_trade_pct': 0.5,        # Risk 0.5% per trade
        
        # Session preferences (highest liquidity periods)
        'trading_sessions': [ForexSession.LONDON, ForexSession.OVERLAP_LONDON_NEWYORK],
        'avoid_news_releases': True,      # Avoid trading during news
        
        # Entry/exit parameters
        'ema_fast_period': 8,             # Fast EMA period
        'ema_medium_period': 13,          # Medium EMA period
        'ema_slow_period': 21,            # Slow EMA period
        'rsi_period': 7,                  # RSI period for scalping
        'rsi_overbought': 70,             # RSI overbought level
        'rsi_oversold': 30,               # RSI oversold level
        'stoch_k_period': 5,              # Stochastic %K period
        'stoch_d_period': 3,              # Stochastic %D period
        'stoch_overbought': 80,           # Stochastic overbought level
        'stoch_oversold': 20,             # Stochastic oversold level
        
        # Spread and execution parameters
        'max_spread_pips': 2.0,           # Maximum allowed spread
        'min_volatility_pips': 3.0,       # Minimum volatility required (ATR)
        'max_volatility_pips': 15.0,      # Maximum volatility allowed (ATR)
        
        # Additional filters
        'min_volume': 0.8,                # Minimum volume as multiple of average
        'require_spread_narrowing': True, # Require spread to be narrowing
    }
    
    def __init__(self, name: str = "Forex Scalping Strategy", 
                 parameters: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the forex scalping strategy.
        
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
        self.spread_history = {}       # Track spread history for narrowing detection
        self.last_trades = {}          # Track last trade times to prevent overtrading
        
        logger.info(f"Initialized {self.name} strategy with timeframe {self.parameters['timeframe']}")
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_time: datetime):
        """
        Generate trade signals for scalping opportunities.
        
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
            required_bars = max(
                self.parameters['ema_slow_period'],
                self.parameters['rsi_period'],
                self.parameters['stoch_k_period']
            ) + 10  # Add buffer
            
            if len(ohlcv) < required_bars:
                logger.warning(f"Not enough data for {symbol} to generate scalping signals")
                continue
            
            # Skip if the spread is too wide
            if not self._check_spread(symbol, ohlcv):
                continue
                
            # Skip if outside preferred sessions
            if not self._is_valid_trading_session(current_time):
                logger.debug(f"Current time {current_time} is outside preferred trading sessions")
                continue
            
            # Skip if recent trade (prevent overtrading)
            if self._is_recent_trade(symbol, current_time):
                logger.debug(f"Skipping {symbol} due to recent trade")
                continue
            
            # Calculate technical indicators
            indicators = self._calculate_scalping_indicators(ohlcv)
            
            # Check for scalping opportunities
            signal = self._evaluate_scalping_opportunity(symbol, ohlcv, indicators, current_time)
            
            if signal:
                # Store the signal
                signals[symbol] = signal
                self.current_signals[symbol] = signal
                
                # Store trade time to prevent overtrading
                self.last_trades[symbol] = current_time
                
                # Emit the signal event
                self.emit_strategy_event({
                    'symbol': symbol,
                    'signal_type': signal.signal_type.name,
                    'confidence': signal.confidence,
                    'entry_price': signal.metadata.get('entry_price'),
                    'stop_loss': signal.metadata.get('stop_loss'),
                    'take_profit': signal.metadata.get('take_profit'),
                    'pip_target': self.parameters['profit_target_pips'],
                    'pip_risk': self.parameters['stop_loss_pips']
                })
                
                logger.info(
                    f"Generated {signal.signal_type.name} scalping signal for {symbol} "
                    f"with confidence {signal.confidence:.2f}, target {self.parameters['profit_target_pips']} pips"
                )
        
        return signals
    
    def _check_spread(self, symbol: str, ohlcv: pd.DataFrame) -> bool:
        """
        Check if the spread is acceptable for scalping.
        
        Args:
            symbol: Currency pair symbol
            ohlcv: OHLCV DataFrame
            
        Returns:
            True if spread is acceptable, False otherwise
        """
        # In real implementation, get actual spread from broker data
        # For simulation, we'll use a simple approach based on high-low range
        
        # Get the most recent candle
        latest_candle = ohlcv.iloc[-1]
        
        # Estimate spread from the high-low range (just an approximation)
        # In a real implementation, you would get this from broker data
        estimated_spread = (latest_candle['high'] - latest_candle['low']) * 0.1
        estimated_spread_pips = estimated_spread / self.parameters['pip_value']
        
        # Store spread history for narrowing detection if requested
        if self.parameters['require_spread_narrowing']:
            if symbol not in self.spread_history:
                self.spread_history[symbol] = []
            
            # Keep last 5 spread measurements
            self.spread_history[symbol].append(estimated_spread_pips)
            if len(self.spread_history[symbol]) > 5:
                self.spread_history[symbol].pop(0)
            
            # Check if spread is narrowing (current less than average of previous)
            if len(self.spread_history[symbol]) >= 3:
                prev_avg = sum(self.spread_history[symbol][:-1]) / (len(self.spread_history[symbol]) - 1)
                if estimated_spread_pips >= prev_avg:
                    logger.debug(f"Spread for {symbol} not narrowing: {estimated_spread_pips:.2f} >= {prev_avg:.2f}")
                    return False
        
        # Check against maximum allowed spread
        if estimated_spread_pips > self.parameters['max_spread_pips']:
            logger.debug(f"Spread too wide for {symbol}: {estimated_spread_pips:.2f} pips")
            return False
            
        return True
    
    def _is_valid_trading_session(self, current_time: datetime) -> bool:
        """
        Check if the current time is within preferred trading sessions.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            True if within preferred sessions, False otherwise
        """
        preferred_sessions = self.parameters.get('trading_sessions', [])
        
        # If no preferred sessions specified, allow trading anytime
        if not preferred_sessions:
            return True
            
        # Get the current forex session
        current_sessions = self.get_active_forex_sessions(current_time)
        
        # Check if any current session is in our preferred sessions
        for session in current_sessions:
            if session in preferred_sessions:
                return True
                
        return False
    
    def _is_recent_trade(self, symbol: str, current_time: datetime) -> bool:
        """
        Check if we've recently traded this symbol (avoid overtrading).
        
        Args:
            symbol: Currency pair symbol
            current_time: Current timestamp
            
        Returns:
            True if recent trade, False otherwise
        """
        if symbol not in self.last_trades:
            return False
            
        # Get minimum time between trades (default 5 minutes for scalping)
        min_time_between_trades = timedelta(minutes=5)
        
        # Check if enough time has passed since last trade
        time_since_last_trade = current_time - self.last_trades[symbol]
        
        return time_since_last_trade < min_time_between_trades
    
    def _calculate_scalping_indicators(self, ohlcv: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate technical indicators for scalping strategy.
        
        Args:
            ohlcv: OHLCV DataFrame
            
        Returns:
            Dictionary of calculated indicators
        """
        # Copy parameters to local variables for readability
        ema_fast = self.parameters['ema_fast_period']
        ema_medium = self.parameters['ema_medium_period']
        ema_slow = self.parameters['ema_slow_period']
        rsi_period = self.parameters['rsi_period']
        stoch_k = self.parameters['stoch_k_period']
        stoch_d = self.parameters['stoch_d_period']
        
        # Initialize indicators dictionary
        indicators = {}
        
        # Calculate EMAs
        close = ohlcv['close']
        indicators['ema_fast'] = close.ewm(span=ema_fast, adjust=False).mean()
        indicators['ema_medium'] = close.ewm(span=ema_medium, adjust=False).mean()
        indicators['ema_slow'] = close.ewm(span=ema_slow, adjust=False).mean()
        
        # EMA slopes (rate of change)
        indicators['ema_fast_slope'] = indicators['ema_fast'].diff(3) / 3
        indicators['ema_medium_slope'] = indicators['ema_medium'].diff(3) / 3
        indicators['ema_slow_slope'] = indicators['ema_slow'].diff(3) / 3
        
        # Calculate RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        
        # Avoid division by zero
        loss = loss.replace(0, 0.00001)
        
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        low_min = ohlcv['low'].rolling(window=stoch_k).min()
        high_max = ohlcv['high'].rolling(window=stoch_k).max()
        
        # Avoid division by zero
        range_diff = high_max - low_min
        range_diff = range_diff.replace(0, 0.00001)
        
        indicators['stoch_k'] = 100 * ((close - low_min) / range_diff)
        indicators['stoch_d'] = indicators['stoch_k'].rolling(window=stoch_d).mean()
        
        # Volume analysis if available
        if 'volume' in ohlcv.columns:
            indicators['volume'] = ohlcv['volume']
            indicators['volume_ma'] = indicators['volume'].rolling(window=20).mean()
            indicators['volume_ratio'] = indicators['volume'] / indicators['volume_ma']
        else:
            # If volume data is missing, use placeholder values
            indicators['volume_ratio'] = pd.Series(1.0, index=ohlcv.index)
        
        # Calculate ATR for volatility assessment
        high = ohlcv['high']
        low = ohlcv['low']
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        indicators['atr'] = tr.rolling(window=14).mean()
        indicators['atr_pips'] = indicators['atr'] / self.parameters['pip_value']
        
        # Price-EMA relationships
        indicators['close_vs_ema_fast'] = (close / indicators['ema_fast'] - 1) * 10000  # Basis points
        indicators['close_vs_ema_medium'] = (close / indicators['ema_medium'] - 1) * 10000
        indicators['close_vs_ema_slow'] = (close / indicators['ema_slow'] - 1) * 10000
        
        # Check if EMAs are properly aligned for trend
        indicators['ema_aligned_bullish'] = (indicators['ema_fast'] > indicators['ema_medium']) & \
                                         (indicators['ema_medium'] > indicators['ema_slow'])
        indicators['ema_aligned_bearish'] = (indicators['ema_fast'] < indicators['ema_medium']) & \
                                         (indicators['ema_medium'] < indicators['ema_slow'])
        
        return indicators
    
    def _evaluate_scalping_opportunity(self, symbol: str, ohlcv: pd.DataFrame, 
                                     indicators: Dict[str, pd.Series], 
                                     current_time: datetime) -> Optional[Signal]:
        """
        Evaluate market conditions for scalping opportunities.
        
        Args:
            symbol: Currency pair symbol
            ohlcv: OHLCV DataFrame
            indicators: Dictionary of technical indicators
            current_time: Current timestamp
            
        Returns:
            Signal object if opportunity found, None otherwise
        """
        # Get the latest values of our indicators
        current_close = ohlcv['close'].iloc[-1]
        ema_fast = indicators['ema_fast'].iloc[-1]
        ema_medium = indicators['ema_medium'].iloc[-1]
        ema_slow = indicators['ema_slow'].iloc[-1]
        
        fast_slope = indicators['ema_fast_slope'].iloc[-1]
        medium_slope = indicators['ema_medium_slope'].iloc[-1]
        
        rsi = indicators['rsi'].iloc[-1]
        stoch_k = indicators['stoch_k'].iloc[-1]
        stoch_d = indicators['stoch_d'].iloc[-1]
        
        volume_ratio = indicators['volume_ratio'].iloc[-1]
        atr_pips = indicators['atr_pips'].iloc[-1]
        
        ema_aligned_bullish = indicators['ema_aligned_bullish'].iloc[-1]
        ema_aligned_bearish = indicators['ema_aligned_bearish'].iloc[-1]
        
        # Check for invalid ATR (too low or too high volatility)
        if atr_pips < self.parameters['min_volatility_pips']:
            logger.debug(f"Volatility too low for {symbol}: {atr_pips:.2f} pips")
            return None
            
        if atr_pips > self.parameters['max_volatility_pips']:
            logger.debug(f"Volatility too high for {symbol}: {atr_pips:.2f} pips")
            return None
        
        # Check volume if required
        if volume_ratio < self.parameters['min_volume']:
            logger.debug(f"Volume too low for {symbol}: {volume_ratio:.2f}")
            return None
        
        # Initialize signal variables
        signal_type = SignalType.FLAT
        confidence = 0.0
        
        # Long scalping opportunity
        long_conditions = [
            # EMA alignment and slopes
            ema_aligned_bullish,
            fast_slope > 0,
            medium_slope > 0,
            current_close > ema_fast,
            
            # RSI not overbought
            rsi < self.parameters['rsi_overbought'],
            rsi > 40,  # Not too weak
            
            # Stochastic conditions
            stoch_k > stoch_d,  # Stochastic crossing up
            stoch_k > 20 and stoch_k < 80  # Not extreme
        ]
        
        # Short scalping opportunity
        short_conditions = [
            # EMA alignment and slopes
            ema_aligned_bearish,
            fast_slope < 0,
            medium_slope < 0,
            current_close < ema_fast,
            
            # RSI not oversold
            rsi > self.parameters['rsi_oversold'],
            rsi < 60,  # Not too strong
            
            # Stochastic conditions
            stoch_k < stoch_d,  # Stochastic crossing down
            stoch_k < 80 and stoch_k > 20  # Not extreme
        ]
        
        # Count how many conditions are met
        long_count = sum(long_conditions)
        short_count = sum(short_conditions)
        
        # Strong scalping signals are when most conditions are met
        # For long signals
        if long_count >= 6 and long_count > short_count:
            signal_type = SignalType.LONG
            # Scale confidence based on number of conditions met
            confidence = 0.6 + (long_count - 6) * 0.05
            # Boost confidence if volume is high
            if volume_ratio > 1.5:
                confidence += 0.1
        # For short signals
        elif short_count >= 6 and short_count > long_count:
            signal_type = SignalType.SHORT
            # Scale confidence based on number of conditions met
            confidence = 0.6 + (short_count - 6) * 0.05
            # Boost confidence if volume is high
            if volume_ratio > 1.5:
                confidence += 0.1
        
        # If confidence is high enough, create a signal
        if confidence >= 0.6:
            # Calculate stop loss and take profit in pips
            stop_loss_pips = self.parameters['stop_loss_pips']
            take_profit_pips = self.parameters['profit_target_pips']
            
            # Convert pips to price
            pip_value = self.parameters['pip_value']
            
            # Set entry, stop loss and take profit prices
            entry_price = current_close
            
            if signal_type == SignalType.LONG:
                stop_loss = entry_price - (stop_loss_pips * pip_value)
                take_profit = entry_price + (take_profit_pips * pip_value)
            else:  # SHORT
                stop_loss = entry_price + (stop_loss_pips * pip_value)
                take_profit = entry_price - (take_profit_pips * pip_value)
            
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
                    'rsi': rsi,
                    'stoch_k': stoch_k,
                    'stoch_d': stoch_d,
                    'volume_ratio': volume_ratio,
                    'atr_pips': atr_pips,
                    'ema_alignment': 'bullish' if ema_aligned_bullish else 'bearish',
                    'risk_reward_ratio': take_profit_pips / stop_loss_pips
                }
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
        # Scalping strategies perform best in specific market regimes
        # They excel in range-bound markets with moderate volatility
        compatibility_map = {
            # Trending regimes - moderate for scalping if volatility is controlled
            MarketRegime.BULL_TREND: 0.60,      # Can work in gentle bull trends
            MarketRegime.BEAR_TREND: 0.60,      # Can work in gentle bear trends
            
            # Range-bound regimes - excellent for scalping
            MarketRegime.CONSOLIDATION: 0.95,   # Optimal environment for scalping
            
            # Volatile regimes - challenging for scalping
            MarketRegime.HIGH_VOLATILITY: 0.25, # Too much volatility, stops easily hit
            
            # Low volatility regimes - good for scalping but smaller targets
            MarketRegime.LOW_VOLATILITY: 0.75,  # Low volatility good for tight stops
            
            # Default for unknown regimes
            MarketRegime.UNKNOWN: 0.50          # Neutral stance on unknown conditions
        }
        
        # Return the compatibility score
        return compatibility_map.get(market_regime, 0.50)
    
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
        
        # Adjust parameters based on market regime
        if market_regime == MarketRegime.BULL_TREND:
            # For bull trends, slightly wider targets, focus on long entries
            optimized_params['profit_target_pips'] = 10      # Aim for larger moves
            optimized_params['stop_loss_pips'] = 5           # Keep tight stops
            optimized_params['rsi_oversold'] = 35            # More sensitive to oversold for longs
            optimized_params['min_volume'] = 1.0             # Require more volume confirmation
            
        elif market_regime == MarketRegime.BEAR_TREND:
            # For bear trends, slightly wider targets, focus on short entries
            optimized_params['profit_target_pips'] = 10      # Aim for larger moves
            optimized_params['stop_loss_pips'] = 5           # Keep tight stops
            optimized_params['rsi_overbought'] = 65          # More sensitive to overbought for shorts
            optimized_params['min_volume'] = 1.0             # Require more volume confirmation
            
        elif market_regime == MarketRegime.CONSOLIDATION:
            # Ideal for scalping - tight ranges with predictable bounces
            optimized_params['profit_target_pips'] = 8       # Standard target
            optimized_params['stop_loss_pips'] = 4           # Tighter stops in range
            optimized_params['rsi_overbought'] = 70          # Standard overbought
            optimized_params['rsi_oversold'] = 30            # Standard oversold
            optimized_params['ema_fast_period'] = 8          # Standard settings
            
        elif market_regime == MarketRegime.HIGH_VOLATILITY:
            # Very challenging - wider stops required, focus on strong signals
            optimized_params['profit_target_pips'] = 15      # Larger targets in volatile markets
            optimized_params['stop_loss_pips'] = 10          # Wider stops needed
            optimized_params['max_spread_pips'] = 2.5        # Allow slightly wider spreads
            optimized_params['min_volume'] = 1.5             # Require strong volume confirmation
            optimized_params['trading_sessions'] = [ForexSession.LONDON_NEWYORK_OVERLAP]  # Only trade highest liquidity
            
        elif market_regime == MarketRegime.LOW_VOLATILITY:
            # Good for scalping but with smaller targets
            optimized_params['profit_target_pips'] = 5       # Smaller targets
            optimized_params['stop_loss_pips'] = 3           # Very tight stops
            optimized_params['max_spread_pips'] = 1.5        # Need tighter spreads
            optimized_params['min_volatility_pips'] = 2.0    # Lower minimum volatility requirement
            
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
            'timeframe': self.parameters['timeframe'].name,
            'stop_loss_pips': self.parameters['stop_loss_pips'],
            'take_profit_pips': self.parameters['profit_target_pips'],
            **data
        }
        
        # Create and publish event
        event = Event(
            event_type=EventType.SIGNAL_GENERATED,
            source=self.name,
            data=event_data,
            metadata={'strategy_type': 'forex', 'category': 'scalping'}
        )
        self.event_bus.publish(event)
