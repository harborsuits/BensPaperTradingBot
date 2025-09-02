#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Prop Range Trading Strategy

This module implements a forex range trading strategy that incorporates the 
enhanced prop trading rules required for proprietary trading.
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
    'strategy_type': 'enhanced_prop_range',
    'compatible_market_regimes': ['ranging', 'low_volatility'],
    'timeframe': 'intraday',
    'regime_compatibility_scores': {
        'trending': 0.40,       # Poor compatibility with trending markets
        'ranging': 0.95,        # Excellent compatibility with ranging markets
        'volatile': 0.50,       # Moderate compatibility with volatile markets
        'low_volatility': 0.80, # Good compatibility with low volatility markets
        'all_weather': 0.65     # Good overall compatibility
    }
})
class EnhancedPropRangeStrategy(EnhancedPropTradingMixin, ForexBaseStrategy):
    """
    Enhanced Proprietary Forex Range Trading Strategy
    
    This strategy identifies and trades within established ranges:
    - Support and resistance boundaries
    - Bollinger Bands as dynamic ranges
    - RSI for identifying overbought/oversold conditions
    - Volume confirmation for range validity
    
    It strictly adheres to enhanced prop trading rules:
    - Maximum 1-2% daily loss limit
    - Maximum 5% drawdown limit
    - 0.5-1% risk per trade
    - Minimum 2:1 reward-risk ratio
    - Partial take-profits at defined levels
    - Trailing stops after partial exits
    - Time-based exit rules
    """
    
    # Default parameters specific to range trading
    DEFAULT_PARAMS = {
        # Range detection parameters
        'lookback_periods': 20,        # Lookback for range identification
        'range_confirmation_bars': 5,  # Consecutive bars for range confirmation
        'range_retest_factor': 0.8,    # Minimum retest factor for range edges
        
        # Technical indicators
        'bb_period': 20,               # Bollinger Bands period
        'bb_std': 2.0,                 # Bollinger Bands standard deviation
        'rsi_period': 14,              # RSI period
        'rsi_overbought': 70,          # RSI overbought threshold
        'rsi_oversold': 30,            # RSI oversold threshold
        
        # Entry/exit parameters
        'inner_range_buffer': 0.1,     # Buffer from range edges (10%)
        'stop_beyond_range_pips': 10,  # Pips beyond range for stop loss
        'target_range_factor': 0.7,    # Target distance as % of range height
        
        # Session parameters
        'preferred_sessions': ['london', 'newyork'],
        'avoid_high_volatility_sessions': True,  # Avoid highest volatility times
        
        # Enhanced prop-specific parameters
        'risk_per_trade_percent': 0.006,     # 0.6% risk per trade
        'max_daily_loss_percent': 0.015,     # 1.5% max daily loss
        'max_drawdown_percent': 0.05,        # 5% max drawdown
        'scale_out_levels': [0.5, 0.75],     # Take partial profits at 50% and 75% of target
        'trailing_activation_percent': 0.5,  # Activate trailing stops at 50% to target
        'max_trade_duration_hours': 48,      # Max trade duration of 48 hours
        'max_concurrent_positions': 3,       # Max 3 positions open at once
    }
    
    def __init__(self, name: str = "EnhancedPropRangeStrategy", 
                parameters: Optional[Dict[str, Any]] = None, 
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize Enhanced Proprietary Range Trading Strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Initialize tracking variables
        self.identified_ranges = {}
        self.last_signals = {}
        self.account_info = {'balance': 0.0, 'starting_balance': 0.0}
        self.current_positions = []
        
        # Initialize with base parameters
        super().__init__(name=name, parameters=parameters, metadata=metadata)
    
    def register_events(self, event_bus: EventBus):
        """Register strategy events with the event bus."""
        super().register_events(event_bus)
        
        # Register for account updates
        event_bus.subscribe(EventType.ACCOUNT_UPDATED, self._on_account_updated)
        
        # Register for trade execution events
        event_bus.subscribe(EventType.TRADE_EXECUTED, self._on_trade_executed)
        
        # Register for day completed events
        event_bus.subscribe(EventType.TRADING_DAY_COMPLETED, self._on_trading_day_completed)
        
        # Register for market data updates
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
        Handle market data updates.
        Process trade management including partial exits and trailing stops.
        """
        market_data = event.data.get('market_data', {})
        symbols = market_data.get('symbols', [])
        
        # Process each current position
        for position in self.current_positions:
            # Skip if not a position from this strategy
            if position.get('strategy') != self.name:
                continue
                
            symbol = position.get('symbol')
            
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
                logger.info(f"Partial exit triggered: {order}")
                
            # Check for trailing stop adjustments
            stop_update = self.process_trailing_stops(position, current_price)
            if stop_update:
                new_stop = stop_update.get('new_stop_loss')
                logger.info(f"Trailing stop updated for {symbol}: {new_stop:.5f}")
                
            # Check for time-based exits
            if self.check_time_based_exits(position):
                logger.info(f"Time-based exit triggered for {symbol}")
    
    def identify_range(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """
        Identify a trading range for a symbol.
        
        Args:
            symbol: Symbol to identify range for
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with range information or None if no valid range found
        """
        if len(data) < self.parameters['lookback_periods']:
            return None
            
        # Get recent data for range identification
        lookback = self.parameters['lookback_periods']
        recent_data = data.iloc[-lookback:]
        
        # Calculate Bollinger Bands
        bb_period = self.parameters['bb_period']
        bb_std = self.parameters['bb_std']
        
        if len(recent_data) < bb_period:
            return None
            
        # Calculate simple moving average
        sma = recent_data['close'].rolling(window=bb_period).mean()
        
        # Calculate standard deviation
        rolling_std = recent_data['close'].rolling(window=bb_period).std()
        
        # Calculate upper and lower bands
        upper_band = sma + (rolling_std * bb_std)
        lower_band = sma - (rolling_std * bb_std)
        
        # Check if price is contained within bands
        last_n_bars = 5  # Last 5 bars for current assessment
        if len(recent_data) < last_n_bars:
            return None
            
        current_data = recent_data.iloc[-last_n_bars:]
        
        # Check if most closing prices are within bands
        if (sum(current_data['close'] <= upper_band.iloc[-last_n_bars]) < last_n_bars * 0.8 or
            sum(current_data['close'] >= lower_band.iloc[-last_n_bars]) < last_n_bars * 0.8):
            return None
            
        # Calculate range parameters
        range_high = upper_band.iloc[-1]
        range_low = lower_band.iloc[-1]
        range_height = range_high - range_low
        
        # Calculate RSI
        rsi_period = self.parameters['rsi_period']
        if len(recent_data) < rsi_period + 1:
            return None
            
        delta = recent_data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        
        rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        # Check for range-bound criteria
        range_bound = True
        
        # Calculate "touches" of upper and lower bands
        upper_touches = sum(recent_data['high'] >= upper_band * 0.98)
        lower_touches = sum(recent_data['low'] <= lower_band * 1.02)
        
        # Require minimum number of touches for valid range
        if upper_touches < 2 or lower_touches < 2:
            range_bound = False
            
        # Check for range retest factor
        range_retest_factor = self.parameters['range_retest_factor']
        
        # Upper band retests
        upper_retests = 0
        for i in range(1, len(recent_data)):
            if recent_data['high'].iloc[i] >= upper_band.iloc[i] * range_retest_factor:
                upper_retests += 1
                
        # Lower band retests
        lower_retests = 0
        for i in range(1, len(recent_data)):
            if recent_data['low'].iloc[i] <= lower_band.iloc[i] * (2 - range_retest_factor):
                lower_retests += 1
        
        # Check retest requirements
        if upper_retests < 2 or lower_retests < 2:
            range_bound = False
            
        # Return range information if valid
        if range_bound:
            return {
                'high': range_high,
                'low': range_low,
                'height': range_height,
                'middle': (range_high + range_low) / 2,
                'rsi': current_rsi,
                'upper_band': upper_band.iloc[-1],
                'lower_band': lower_band.iloc[-1],
                'timestamp': datetime.now()
            }
            
        return None
    
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
            
        # Check if we're in preferred trading sessions
        in_preferred_session = self.is_current_session_active(
            self.parameters['preferred_sessions'])
                
        if not in_preferred_session:
            logger.info("Not in preferred trading session, reducing signal confidence")
        
        # Process each symbol in the universe
        for symbol, data in universe.items():
            # Skip if not enough data
            if len(data) < 30:  # Need at least 30 bars for range identification
                continue
                
            # Skip if not a major pair and we're focusing on majors
            if (self.parameters['focus_on_major_pairs'] and 
                symbol not in self.PROP_RECOMMENDED_PAIRS):
                continue
                
            # Skip symbols with News events if configured
            if self.parameters['news_avoidance_minutes'] > 0 and self.should_avoid_news_events(symbol, datetime.now()):
                logger.info(f"Skipping {symbol} due to high impact news")
                continue
                
            # Identify or update range
            current_range = self.identify_range(symbol, data)
            
            if current_range:
                # Store range
                self.identified_ranges[symbol] = current_range
                
                # Get current price
                current_price = data['close'].iloc[-1]
                
                # Get current RSI
                current_rsi = current_range['rsi']
                
                # Check for potential signals
                signal = None
                
                # Calculate inner range buffer
                buffer_factor = self.parameters['inner_range_buffer']
                inner_range_high = current_range['high'] - (current_range['height'] * buffer_factor)
                inner_range_low = current_range['low'] + (current_range['height'] * buffer_factor)
                
                # LONG signal near lower range bound
                if (current_price <= inner_range_low and 
                    current_rsi <= self.parameters['rsi_oversold']):
                    
                    # Base confidence on proximity to range low and RSI
                    proximity_factor = (inner_range_low - current_price) / current_range['height']
                    rsi_factor = (self.parameters['rsi_oversold'] - current_rsi) / self.parameters['rsi_oversold']
                    
                    confidence = 0.7 + min(proximity_factor * 0.5, 0.15) + min(rsi_factor * 0.5, 0.15)
                    
                    # Adjust for session
                    if not in_preferred_session:
                        confidence *= 0.8
                    
                    # Only generate signal if confidence is high enough
                    if confidence >= 0.7:
                        # Calculate stop loss below the range
                        pip_value = self.parameters.get('pip_value', 0.0001)
                        stop_loss = current_range['low'] - (self.parameters['stop_beyond_range_pips'] * pip_value)
                        
                        # Calculate take profit toward the middle/upper part of range
                        target_distance = current_range['height'] * self.parameters['target_range_factor']
                        take_profit = current_price + target_distance
                        
                        # Ensure profit target is below the upper range
                        take_profit = min(take_profit, inner_range_high)
                        
                        # Ensure reward-risk ratio meets prop requirements
                        risk = current_price - stop_loss
                        reward = take_profit - current_price
                        
                        if reward / risk < self.parameters['min_reward_risk_ratio']:
                            take_profit = current_price + (risk * self.parameters['min_reward_risk_ratio'])
                            # Ensure take profit is still within range
                            take_profit = min(take_profit, inner_range_high)
                        
                        signal = Signal(
                            symbol=symbol,
                            signal_type=SignalType.LONG,
                            confidence=confidence,
                            entry_price=current_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            metadata={
                                'strategy': self.name,
                                'range_high': current_range['high'],
                                'range_low': current_range['low'],
                                'rsi': current_rsi
                            }
                        )
                
                # SHORT signal near upper range bound
                elif (current_price >= inner_range_high and 
                      current_rsi >= self.parameters['rsi_overbought']):
                    
                    # Base confidence on proximity to range high and RSI
                    proximity_factor = (current_price - inner_range_high) / current_range['height']
                    rsi_factor = (current_rsi - self.parameters['rsi_overbought']) / (100 - self.parameters['rsi_overbought'])
                    
                    confidence = 0.7 + min(proximity_factor * 0.5, 0.15) + min(rsi_factor * 0.5, 0.15)
                    
                    # Adjust for session
                    if not in_preferred_session:
                        confidence *= 0.8
                    
                    # Only generate signal if confidence is high enough
                    if confidence >= 0.7:
                        # Calculate stop loss above the range
                        pip_value = self.parameters.get('pip_value', 0.0001)
                        stop_loss = current_range['high'] + (self.parameters['stop_beyond_range_pips'] * pip_value)
                        
                        # Calculate take profit toward the middle/lower part of range
                        target_distance = current_range['height'] * self.parameters['target_range_factor']
                        take_profit = current_price - target_distance
                        
                        # Ensure profit target is above the lower range
                        take_profit = max(take_profit, inner_range_low)
                        
                        # Ensure reward-risk ratio meets prop requirements
                        risk = stop_loss - current_price
                        reward = current_price - take_profit
                        
                        if reward / risk < self.parameters['min_reward_risk_ratio']:
                            take_profit = current_price - (risk * self.parameters['min_reward_risk_ratio'])
                            # Ensure take profit is still within range
                            take_profit = max(take_profit, inner_range_low)
                        
                        signal = Signal(
                            symbol=symbol,
                            signal_type=SignalType.SHORT,
                            confidence=confidence,
                            entry_price=current_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            metadata={
                                'strategy': self.name,
                                'range_high': current_range['high'],
                                'range_low': current_range['low'],
                                'rsi': current_rsi
                            }
                        )
                
                # Store signal if generated and validated against prop rules
                if signal and self.validate_prop_trading_rules(
                    signal, 
                    self.account_info['balance'],
                    self.account_info['starting_balance'],
                    self.current_positions
                ):
                    signals[symbol] = signal
                    self.last_signals[symbol] = signal
        
        return signals
    
    def calculate_position_size(self, signal: Signal, account_balance: float) -> float:
        """
        Calculate position size based on prop risk management rules.
        
        Args:
            signal: Trading signal
            account_balance: Current account balance
            
        Returns:
            Position size in standard lots
        """
        # Use prop trading position sizing
        return self.calculate_prop_position_size(
            account_balance=account_balance,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            symbol=signal.symbol
        )
