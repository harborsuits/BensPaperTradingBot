#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prop Trend Following Strategy

This module implements a trend following strategy for forex markets
that adheres to proprietary trading firm rules and restrictions.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime

from trading_bot.strategies.forex.base.forex_base_strategy import ForexBaseStrategy, ForexSession
from trading_bot.strategies.forex.prop_trading_rules_mixin import PropTradingRulesMixin
from trading_bot.strategies.factory.strategy_registry import register_strategy
from trading_bot.core.event_system import EventBus, Event, EventType
from trading_bot.strategies_new.factory.strategy_template import Signal, SignalType

logger = logging.getLogger(__name__)

@register_strategy({
    'asset_class': 'forex',
    'strategy_type': 'prop_trend_following',
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
class PropTrendFollowingStrategy(PropTradingRulesMixin, ForexBaseStrategy):
    """
    Proprietary Trading Trend Following Strategy
    
    This strategy identifies and follows trends in forex pairs using:
    - Multiple MA combinations (EMA, SMA, WMA)
    - ADX for trend strength confirmation
    - MACD for additional signal confirmation
    - ATR for volatility and position sizing
    
    It strictly adheres to prop trading rules:
    - Maximum 1-2% daily loss limit
    - Maximum 5% drawdown limit
    - 0.5-1% risk per trade
    - Minimum 2:1 reward-risk ratio
    - Mandatory breaks after hitting loss limits
    - Focus on major pairs during optimal sessions
    - Avoidance of high-impact news events
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
        'profit_target_atr': 2.5,    # Profit target as ATR multiple (increased from original)
        'confidence_threshold': 0.7, # Minimum confidence for trade signals
        
        # Prop-specific parameters (these will be merged with PropTradingRulesMixin defaults)
        'risk_per_trade_percent': 0.005,  # 0.5% risk per trade (conservative prop approach)
        'max_daily_loss_percent': 0.01,   # 1% max daily loss
        'max_drawdown_percent': 0.05,     # 5% max drawdown
    }
    
    def __init__(self, name: str = "PropTrendFollowingStrategy", 
                parameters: Optional[Dict[str, Any]] = None, 
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize Proprietary Trend Following Strategy.
        
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
                
            # Calculate forex-specific indicators
            indicators = self.calculate_forex_indicators(data, symbol)
            
            # Calculate additional trend following indicators
            indicators.update(self.calculate_indicators(data, symbol))
            
            # Skip symbols with News events if configured
            if self.parameters['avoid_high_impact_news'] and self.is_high_impact_news_time(symbol):
                logger.info(f"Skipping {symbol} due to high impact news")
                continue
                
            # Focus on major pairs if configured
            if (self.parameters['focus_on_major_pairs'] and 
                symbol not in self.PROP_RECOMMENDED_PAIRS):
                continue
                
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
    
    def process_partial_exits(self, position_data: Dict, current_price: float) -> List[Dict]:
        """
        Process partial exit strategy based on prop trading rules.
        
        Args:
            position_data: Current position data
            current_price: Current market price
            
        Returns:
            List of exit orders to execute
        """
        exit_orders = []
        
        # Skip if no scale-out levels defined
        if not self.parameters.get('scale_out_levels'):
            return exit_orders
            
        # Extract position information
        entry_price = position_data.get('entry_price', 0)
        take_profit = position_data.get('take_profit', 0)
        position_size = position_data.get('position_size', 0)
        position_type = position_data.get('position_type', '')
        symbol = position_data.get('symbol', '')
        
        # Calculate profit targets based on scale-out levels
        for idx, level in enumerate(self.parameters['scale_out_levels']):
            # Calculate price target
            if position_type == 'LONG':
                price_distance = take_profit - entry_price
                target_price = entry_price + (price_distance * level)
                
                # Check if price has reached this target
                if current_price >= target_price:
                    # Calculate exit size (equal portions)
                    exit_size = position_size / (len(self.parameters['scale_out_levels']) + 1)
                    
                    # Add to exit orders if we haven't exited this level yet
                    level_key = f"level_{idx}"
                    if not position_data.get(level_key, False):
                        exit_orders.append({
                            'symbol': symbol,
                            'size': exit_size,
                            'price': current_price,
                            'type': 'PARTIAL_TAKE_PROFIT',
                            'level': level_key
                        })
                        
            elif position_type == 'SHORT':
                price_distance = entry_price - take_profit
                target_price = entry_price - (price_distance * level)
                
                # Check if price has reached this target
                if current_price <= target_price:
                    # Calculate exit size (equal portions)
                    exit_size = position_size / (len(self.parameters['scale_out_levels']) + 1)
                    
                    # Add to exit orders if we haven't exited this level yet
                    level_key = f"level_{idx}"
                    if not position_data.get(level_key, False):
                        exit_orders.append({
                            'symbol': symbol,
                            'size': exit_size,
                            'price': current_price,
                            'type': 'PARTIAL_TAKE_PROFIT',
                            'level': level_key
                        })
        
        return exit_orders
