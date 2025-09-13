#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Prop Trend Following Strategy

This module implements a trend following strategy for forex markets that 
incorporates the enhanced prop trading rules required for prop trading.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime

from trading_bot.strategies.forex.base.forex_base_strategy import ForexBaseStrategy, ForexSession
from trading_bot.strategies.forex.enhanced_prop_trading_mixin import EnhancedPropTradingMixin
from trading_bot.strategies.factory.strategy_registry import register_strategy
from trading_bot.core.event_system import EventBus, Event, EventType
from trading_bot.strategies_new.factory.strategy_template import Signal, SignalType

logger = logging.getLogger(__name__)

@register_strategy({
    'asset_class': 'forex',
    'strategy_type': 'enhanced_prop_trend_following',
    'compatible_market_regimes': ['trending', 'low_volatility'],
    'timeframe': 'swing',
    'regime_compatibility_scores': {
        'trending': 0.95,       # Highest compatibility with trending markets
        'ranging': 0.40,        # Poor compatibility with ranging markets
        'volatile': 0.60,       # Moderate compatibility with volatile markets
        'low_volatility': 0.75, # Good compatibility with low volatility markets
        'all_weather': 0.70     # Good overall compatibility
    }
})
class EnhancedPropTrendFollowingStrategy(EnhancedPropTradingMixin, ForexBaseStrategy):
    """
    Enhanced Proprietary Trading Trend Following Strategy
    
    This strategy identifies and follows trends in forex pairs using:
    - Multiple MA combinations (EMA, SMA, WMA)
    - ADX for trend strength confirmation
    - MACD for additional signal confirmation
    - ATR for volatility and position sizing
    
    It strictly adheres to enhanced prop trading rules:
    - Maximum 1-2% daily loss limit
    - Maximum 5% drawdown limit
    - 0.5-1% risk per trade
    - Minimum 2:1 reward-risk ratio
    - Partial take-profits at defined levels
    - Trailing stops after partial exits
    - Time-based exit rules
    - Session-aware trading
    - Focus on major pairs
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
        'preferred_sessions': ['london', 'newyork'],
        'trade_session_overlaps': True,
        
        # Signal parameters
        'min_trend_duration': 3,     # Minimum bars trend must exist for confirmation
        'profit_target_atr': 2.5,    # Profit target as ATR multiple (increased for 2:1 R:R)
        'confidence_threshold': 0.7, # Minimum confidence for trade signals
        
        # Enhanced prop-specific parameters
        'risk_per_trade_percent': 0.007,     # 0.7% risk per trade
        'max_daily_loss_percent': 0.015,     # 1.5% max daily loss
        'max_drawdown_percent': 0.05,        # 5% max drawdown
        'min_reward_risk_ratio': 2.0,        # Minimum 2:1 reward-to-risk
        'scale_out_levels': [0.5, 0.75],     # Take partial profits at 50% and 75% of target
        'trailing_activation_percent': 0.4,  # Activate trailing stops at 40% to target
        'max_trade_duration_hours': 72,      # Max trade duration of 72 hours
        'max_concurrent_positions': 3,       # Max 3 positions open at once
    }
    
    def __init__(self, name: str = "EnhancedPropTrendFollowingStrategy", 
                parameters: Optional[Dict[str, Any]] = None, 
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize Enhanced Proprietary Trend Following Strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters (will be merged with DEFAULT_PARAMS)
            metadata: Strategy metadata
        """
        # Initialize trend durations dictionary
        self.trend_durations = {}
        self.last_signals = {}
        self.account_info = {'balance': 0.0, 'starting_balance': 0.0}
        self.current_positions = []
        
        # Initialize with base parameters
        super().__init__(name=name, parameters=parameters, metadata=metadata)
        
        # Set minimum reward-risk ratio to ensure proper prop compliance
        if self.parameters['profit_target_atr'] / self.parameters['atr_multiplier'] < self.parameters['min_reward_risk_ratio']:
            self.parameters['profit_target_atr'] = self.parameters['atr_multiplier'] * self.parameters['min_reward_risk_ratio']
            logger.info(f"Adjusted profit target to ensure {self.parameters['min_reward_risk_ratio']}:1 reward-risk ratio")
    
    def register_events(self, event_bus: EventBus):
        """
        Register strategy events with the event bus.
        
        Args:
            event_bus: Event bus to register with
        """
        super().register_events(event_bus)
        
        # Register for account updates to track P&L and drawdown
        event_bus.subscribe(EventType.ACCOUNT_UPDATED, self._on_account_updated)
        
        # Register for trade execution events to track trade history
        event_bus.subscribe(EventType.TRADE_EXECUTED, self._on_trade_executed)
        
        # Register for day completed events to reset daily tracking
        event_bus.subscribe(EventType.TRADING_DAY_COMPLETED, self._on_trading_day_completed)
        
        # Register for market data updates to check positions
        event_bus.subscribe(EventType.MARKET_DATA_UPDATED, self._on_market_data_updated)
    
    def _on_account_updated(self, event: Event):
        """Handle account updated events."""
        account_data = event.data.get('account_data', {})
        
        # Update account information
        self.account_info['balance'] = account_data.get('balance', 0.0)
        
        # Initialize starting balance if not set
        if self.account_info['starting_balance'] == 0.0:
            self.account_info['starting_balance'] = self.account_info['balance']
            
        # Update current positions
        self.current_positions = account_data.get('positions', [])
    
    def _on_trade_executed(self, event: Event):
        """Handle trade executed events."""
        trade_data = event.data.get('trade_data', {})
        
        # Only process trades from this strategy
        if trade_data.get('strategy') != self.name:
            return
            
        # Update trade record with result
        self.update_trade_record(trade_data)
    
    def _on_trading_day_completed(self, event: Event):
        """Handle trading day completed events."""
        # Reset daily tracking metrics
        self.reset_daily_tracking()
    
    def _on_market_data_updated(self, event: Event):
        """
        Handle market data updated events to process trade management.
        
        This checks for:
        - Partial exit opportunities
        - Trailing stop adjustments
        - Time-based exits
        """
        market_data = event.data.get('market_data', {})
        symbols = market_data.get('symbols', [])
        
        # Process each current position
        for position in self.current_positions:
            # Skip if not a position from this strategy
            if position.get('strategy') != self.name:
                continue
                
            symbol = position.get('symbol')
            position_id = position.get('position_id')
            
            # Skip if we don't have market data for this symbol
            if symbol not in symbols:
                continue
                
            # Get current price
            current_price = market_data.get('prices', {}).get(symbol, 0)
            if current_price <= 0:
                continue
                
            # Check for partial exit opportunities
            exit_orders = self.process_partial_exits(position, current_price)
            for order in exit_orders:
                # In a real implementation, this would execute the order
                logger.info(f"Partial exit triggered: {order}")
                
            # Check for trailing stop adjustments
            stop_update = self.process_trailing_stops(position, current_price)
            if stop_update:
                # In a real implementation, this would update the stop loss
                new_stop = stop_update.get('new_stop_loss')
                logger.info(f"Trailing stop updated for {symbol}: {new_stop:.5f}")
                
            # Check for time-based exits
            if self.check_time_based_exits(position):
                # In a real implementation, this would close the position
                logger.info(f"Time-based exit triggered for {symbol}")
    
    def calculate_indicators(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Calculate technical indicators for the strategy.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol for the data
            
        Returns:
            Dictionary of calculated indicators
        """
        # Extract parameters
        fast_ma_type = self.parameters['fast_ma_type']
        slow_ma_type = self.parameters['slow_ma_type']
        fast_period = self.parameters['fast_ma_period']
        slow_period = self.parameters['slow_ma_period']
        macd_fast = self.parameters['macd_fast_period']
        macd_slow = self.parameters['macd_slow_period']
        macd_signal = self.parameters['signal_ma_period']
        adx_period = self.parameters['adx_period']
        atr_period = self.parameters['atr_period']
        
        # Calculate moving averages
        if fast_ma_type == 'sma':
            data['fast_ma'] = data['close'].rolling(window=fast_period).mean()
        elif fast_ma_type == 'ema':
            data['fast_ma'] = data['close'].ewm(span=fast_period, adjust=False).mean()
        elif fast_ma_type == 'wma':
            # Weighted MA calculation
            weights = np.arange(1, fast_period + 1)
            data['fast_ma'] = data['close'].rolling(window=fast_period).apply(
                lambda x: np.sum(weights * x) / weights.sum(), raw=True)
        
        if slow_ma_type == 'sma':
            data['slow_ma'] = data['close'].rolling(window=slow_period).mean()
        elif slow_ma_type == 'ema':
            data['slow_ma'] = data['close'].ewm(span=slow_period, adjust=False).mean()
        elif slow_ma_type == 'wma':
            # Weighted MA calculation
            weights = np.arange(1, slow_period + 1)
            data['slow_ma'] = data['close'].rolling(window=slow_period).apply(
                lambda x: np.sum(weights * x) / weights.sum(), raw=True)
            
        # Calculate MACD
        data['macd_fast'] = data['close'].ewm(span=macd_fast, adjust=False).mean()
        data['macd_slow'] = data['close'].ewm(span=macd_slow, adjust=False).mean()
        data['macd'] = data['macd_fast'] - data['macd_slow']
        data['macd_signal'] = data['macd'].ewm(span=macd_signal, adjust=False).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']
        
        # Calculate ADX
        # This is a simplified ADX calculation for brevity
        data['tr1'] = abs(data['high'] - data['low'])
        data['tr2'] = abs(data['high'] - data['close'].shift())
        data['tr3'] = abs(data['low'] - data['close'].shift())
        data['tr'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
        data['atr'] = data['tr'].rolling(window=atr_period).mean()
        
        # Simple +DI and -DI calculation
        data['+dm'] = np.where((data['high'] - data['high'].shift()) > 
                               (data['low'].shift() - data['low']),
                               np.maximum(data['high'] - data['high'].shift(), 0), 0)
        data['-dm'] = np.where((data['low'].shift() - data['low']) > 
                               (data['high'] - data['high'].shift()),
                               np.maximum(data['low'].shift() - data['low'], 0), 0)
        data['+di'] = 100 * (data['+dm'].rolling(window=adx_period).mean() / 
                             data['tr'].rolling(window=adx_period).mean())
        data['-di'] = 100 * (data['-dm'].rolling(window=adx_period).mean() / 
                             data['tr'].rolling(window=adx_period).mean())
        data['dx'] = 100 * (abs(data['+di'] - data['-di']) / 
                           (data['+di'] + data['-di']).replace(0, 1))
        data['adx'] = data['dx'].rolling(window=adx_period).mean()
        
        # Return the latest values
        return {
            'fast_ma': data['fast_ma'].iloc[-1],
            'slow_ma': data['slow_ma'].iloc[-1],
            'macd': data['macd'].iloc[-1],
            'macd_signal': data['macd_signal'].iloc[-1],
            'macd_hist': data['macd_hist'].iloc[-1],
            'adx': data['adx'].iloc[-1],
            '+di': data['+di'].iloc[-1],
            '-di': data['-di'].iloc[-1],
            'atr': data['atr'].iloc[-1]
        }
    
    def generate_signals(self, universe: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate trading signals for the universe of symbols.
        
        Args:
            universe: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Skip signal generation if we're in a mandatory break period
        if self.check_in_mandatory_break():
            logger.info("In mandatory break period, skipping signal generation")
            return {}
        
        signals = {}
        
        # Check if we've hit daily loss limit
        if not self.validate_daily_loss_limit(self.account_info['balance']):
            logger.warning("Daily loss limit hit, skipping signal generation")
            return {}
            
        # Check if we've exceeded drawdown limit
        if not self.validate_drawdown_limit(
            self.account_info['balance'], 
            self.account_info['starting_balance']):
            logger.warning("Drawdown limit exceeded, skipping signal generation")
            return {}
            
        # Check if we have too many positions open
        if not self.validate_concurrent_positions(self.current_positions):
            logger.info("Maximum concurrent positions reached, skipping signal generation")
            return {}
            
        # Initialize trend durations for new symbols
        for symbol in universe:
            if symbol not in self.trend_durations:
                self.trend_durations[symbol] = 0
                
        # Check if we're in preferred trading sessions
        in_preferred_session = self.is_current_session_active(
            self.parameters['preferred_sessions'])
                
        # Calculate currency strength (from forex base strategy)
        currency_strength = self.calculate_currency_strength(universe)
        
        # Process each symbol in the universe
        for symbol, data in universe.items():
            # Skip if not enough data
            if len(data) < max(
                self.parameters['slow_ma_period'],
                self.parameters['macd_slow_period'],
                self.parameters['adx_period']):
                continue
                
            # Skip symbols with News events if configured
            if self.parameters['news_avoidance_minutes'] > 0 and self.should_avoid_news_events(symbol, datetime.now()):
                logger.info(f"Skipping {symbol} due to high impact news")
                continue
                
            # Focus on major pairs if configured
            if (self.parameters['focus_on_major_pairs'] and 
                symbol not in self.PROP_RECOMMENDED_PAIRS):
                continue
                
            # Calculate indicators
            indicators = self.calculate_indicators(data, symbol)
            
            # Extract indicator values
            adx_value = indicators['adx']
            adx_threshold = self.parameters['adx_threshold']
            trend_strength = adx_value > adx_threshold
            
            fast_ma = indicators['fast_ma']
            slow_ma = indicators['slow_ma']
            macd = indicators['macd']
            macd_signal = indicators['macd_signal']
            atr = indicators['atr']
            
            # Determine trend direction
            if fast_ma > slow_ma:
                # Bullish trend
                if self.trend_durations.get(symbol, 0) < 0:
                    self.trend_durations[symbol] = 1
                else:
                    self.trend_durations[symbol] += 1
            elif fast_ma < slow_ma:
                # Bearish trend
                if self.trend_durations.get(symbol, 0) > 0:
                    self.trend_durations[symbol] = -1
                else:
                    self.trend_durations[symbol] -= 1
            
            # Trend duration check
            trend_duration_met = abs(self.trend_durations.get(symbol, 0)) >= self.parameters['min_trend_duration']
            
            # Calculate relative currency strength
            base, quote = symbol.split('/') if '/' in symbol else (symbol[:3], symbol[3:])
            base_strength = currency_strength.get(base, 0)
            quote_strength = currency_strength.get(quote, 0)
            relative_strength = base_strength - quote_strength
            
            # Identify potential signal conditions
            ma_bullish_cross = fast_ma > slow_ma and data['close'].iloc[-1] > fast_ma
            ma_bearish_cross = fast_ma < slow_ma and data['close'].iloc[-1] < fast_ma
            
            macd_bullish_cross = macd > macd_signal and macd > 0
            macd_bearish_cross = macd < macd_signal and macd < 0
            
            signal = None
            
            # Long signal conditions
            if (ma_bullish_cross or macd_bullish_cross) and trend_strength and trend_duration_met:
                # Adjust confidence based on trend strength and duration
                confidence = 0.7 + (min(adx_value, 50) / 100) + (min(abs(self.trend_durations[symbol]), 10) / 100)
                
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
                    # Calculate stop-loss and take-profit levels with ATR
                    stop_loss = data['close'].iloc[-1] - (atr * self.parameters['atr_multiplier'])
                    take_profit = data['close'].iloc[-1] + (atr * self.parameters['profit_target_atr'])
                    
                    # Ensure reward-risk ratio meets prop requirements
                    entry_price = data['close'].iloc[-1]
                    reward = take_profit - entry_price
                    risk = entry_price - stop_loss
                    
                    if reward / risk < self.parameters['min_reward_risk_ratio']:
                        take_profit = entry_price + (risk * self.parameters['min_reward_risk_ratio'])
                        logger.info(f"Adjusted take profit to ensure {self.parameters['min_reward_risk_ratio']}:1 reward-risk ratio")
                    
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.LONG,
                        confidence=confidence,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'strategy': self.name,
                            'indicators': {
                                'adx': adx_value,
                                'trend_duration': self.trend_durations[symbol],
                                'ma_fast': fast_ma,
                                'ma_slow': slow_ma,
                                'atr': atr
                            }
                        }
                    )
            
            # Short signal conditions
            elif (ma_bearish_cross or macd_bearish_cross) and trend_strength and trend_duration_met:
                # Adjust confidence based on trend strength and duration
                confidence = 0.7 + (min(adx_value, 50) / 100) + (min(abs(self.trend_durations[symbol]), 10) / 100)
                
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
                    # Calculate stop-loss and take-profit levels with ATR
                    stop_loss = data['close'].iloc[-1] + (atr * self.parameters['atr_multiplier'])
                    take_profit = data['close'].iloc[-1] - (atr * self.parameters['profit_target_atr'])
                    
                    # Ensure reward-risk ratio meets prop requirements
                    entry_price = data['close'].iloc[-1]
                    reward = entry_price - take_profit
                    risk = stop_loss - entry_price
                    
                    if reward / risk < self.parameters['min_reward_risk_ratio']:
                        take_profit = entry_price - (risk * self.parameters['min_reward_risk_ratio'])
                        logger.info(f"Adjusted take profit to ensure {self.parameters['min_reward_risk_ratio']}:1 reward-risk ratio")
                    
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SHORT,
                        confidence=confidence,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            'strategy': self.name,
                            'indicators': {
                                'adx': adx_value,
                                'trend_duration': self.trend_durations[symbol],
                                'ma_fast': fast_ma,
                                'ma_slow': slow_ma,
                                'atr': atr
                            }
                        }
                    )
            
            # Store signal if generated
            if signal:
                # Validate signal against prop trading rules
                if self.validate_prop_trading_rules(
                    signal, 
                    self.account_info['balance'],
                    self.account_info['starting_balance'],
                    self.current_positions
                ):
                    signals[symbol] = signal
                    self.last_signals[symbol] = signal
        
        # Apply session-based adjustments to signals
        adjusted_signals = self.adjust_for_trading_session(signals)
        
        return adjusted_signals
        
    def calculate_position_size(self, signal: Signal, account_balance: float) -> float:
        """
        Calculate position size for the signal based on prop risk management rules.
        
        Args:
            signal: Trading signal
            account_balance: Current account balance
            
        Returns:
            Position size in units
        """
        # Use prop trading position sizing
        return self.calculate_prop_position_size(
            account_balance=account_balance,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            symbol=signal.symbol
        )
