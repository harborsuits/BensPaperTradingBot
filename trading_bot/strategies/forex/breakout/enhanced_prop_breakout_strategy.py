#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Prop Breakout Strategy

This module implements a forex breakout strategy that incorporates the 
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
    'strategy_type': 'enhanced_prop_breakout',
    'compatible_market_regimes': ['ranging', 'volatile'],
    'timeframe': 'intraday',
    'regime_compatibility_scores': {
        'trending': 0.65,      # Moderate compatibility with trending markets
        'ranging': 0.90,       # High compatibility with ranging markets
        'volatile': 0.80,      # Good compatibility with volatile markets
        'low_volatility': 0.40, # Poor compatibility with low volatility
        'all_weather': 0.70    # Good overall compatibility
    }
})
class EnhancedPropBreakoutStrategy(EnhancedPropTradingMixin, ForexBaseStrategy):
    """
    Enhanced Proprietary Forex Breakout Strategy
    
    This strategy identifies and trades breakouts from key levels:
    - Support and resistance levels
    - Price channels
    - Key psychological levels
    - Prior session high/lows
    
    It strictly adheres to enhanced prop trading rules:
    - Maximum 1-2% daily loss limit
    - Maximum 5% drawdown limit
    - 0.5-1% risk per trade
    - Minimum 2:1 reward-risk ratio
    - Partial take-profits at defined levels
    - Trailing stops after partial exits
    - Time-based exit rules
    """
    
    # Default parameters specific to breakout trading
    DEFAULT_PARAMS = {
        # Breakout detection parameters
        'lookback_periods': 20,        # Lookback for support/resistance
        'breakout_threshold_pips': 5,  # Minimum pips to confirm breakout
        'consolidation_atr_mult': 0.5, # Max ATR multiplier for consolidation
        
        # Filter parameters
        'min_consolidation_bars': 5,   # Minimum bars in consolidation
        'volume_surge_factor': 1.5,    # Volume increase factor for confirmation
        'false_breakout_filter': True, # Use false breakout filter
        
        # Session parameters
        'london_open_breakout': True,  # Trade London session open breakouts
        'use_session_high_low': True,  # Use prior session high/lows
        'preferred_sessions': ['london', 'newyork'],
        
        # Technical parameters
        'atr_period': 14,              # ATR calculation period
        'atr_stop_multiplier': 1.5,    # ATR multiplier for stop loss
        'profit_target_mult': 3.0,     # Profit target multiplier relative to stop
        
        # Enhanced prop-specific parameters
        'risk_per_trade_percent': 0.007,     # 0.7% risk per trade
        'max_daily_loss_percent': 0.015,     # 1.5% max daily loss
        'max_drawdown_percent': 0.05,        # 5% max drawdown
        'scale_out_levels': [0.5, 0.8],      # Take partial profits at 50% and 80% of target
        'trailing_activation_percent': 0.5,  # Activate trailing stops at 50% to target
        'max_trade_duration_hours': 24,      # Max trade duration of 24 hours
        'max_concurrent_positions': 2,       # Max 2 positions open at once
    }
    
    def __init__(self, name: str = "EnhancedPropBreakoutStrategy", 
                parameters: Optional[Dict[str, Any]] = None, 
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize Enhanced Proprietary Breakout Strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Initialize tracking variables
        self.support_resistance_levels = {}
        self.consolidation_zones = {}
        self.session_high_lows = {}
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
        
        # Register for session start/end events
        event_bus.subscribe(EventType.SESSION_STARTED, self._on_session_started)
        event_bus.subscribe(EventType.SESSION_ENDED, self._on_session_ended)
        
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
    
    def _on_session_started(self, event: Event):
        """Handle session start events."""
        session_data = event.data.get('session_data', {})
        session = session_data.get('session')
        
        # Special handling for London session if enabled
        if session == 'london' and self.parameters['london_open_breakout']:
            logger.info("London session started - monitoring for breakouts")
    
    def _on_session_ended(self, event: Event):
        """
        Handle session end events.
        Store session high/lows for breakout levels.
        """
        session_data = event.data.get('session_data', {})
        session = session_data.get('session')
        symbols = session_data.get('symbols', [])
        
        # Store session high/lows for each symbol
        for symbol in symbols:
            if symbol not in self.session_high_lows:
                self.session_high_lows[symbol] = {}
                
            high = session_data.get('high', {}).get(symbol)
            low = session_data.get('low', {}).get(symbol)
            
            if high is not None and low is not None:
                self.session_high_lows[symbol][session] = {
                    'high': high,
                    'low': low,
                    'timestamp': datetime.now()
                }
    
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
    
    def update_support_resistance_levels(self, symbol: str, data: pd.DataFrame):
        """
        Update support and resistance levels for a symbol.
        
        Args:
            symbol: The symbol to update levels for
            data: DataFrame with OHLCV data
        """
        if symbol not in self.support_resistance_levels:
            self.support_resistance_levels[symbol] = {'support': [], 'resistance': []}
        
        # Get lookback window
        lookback = self.parameters['lookback_periods']
        if len(data) < lookback:
            return
            
        # Focus on recent data
        recent_data = data.iloc[-lookback:]
        
        # Find peaks and troughs
        highs = []
        lows = []
        
        for i in range(1, len(recent_data) - 1):
            # Potential high
            if (recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and 
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1]):
                highs.append(recent_data['high'].iloc[i])
                
            # Potential low
            if (recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1] and 
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1]):
                lows.append(recent_data['low'].iloc[i])
        
        # Group nearby levels
        grouped_highs = self._group_nearby_levels(highs)
        grouped_lows = self._group_nearby_levels(lows)
        
        # Store levels
        self.support_resistance_levels[symbol]['resistance'] = grouped_highs
        self.support_resistance_levels[symbol]['support'] = grouped_lows
    
    def _group_nearby_levels(self, levels, pip_threshold=10):
        """
        Group levels that are within a threshold of each other.
        
        Args:
            levels: List of price levels
            pip_threshold: Maximum distance in pips to consider levels as the same
            
        Returns:
            List of grouped levels
        """
        if not levels:
            return []
            
        # Sort levels
        sorted_levels = sorted(levels)
        
        # Group nearby levels
        grouped = []
        current_group = [sorted_levels[0]]
        
        for i in range(1, len(sorted_levels)):
            # Convert price difference to pips
            pip_value = self.parameters.get('pip_value', 0.0001)
            pips_diff = abs(sorted_levels[i] - current_group[0]) / pip_value
            
            if pips_diff <= pip_threshold:
                # Add to current group
                current_group.append(sorted_levels[i])
            else:
                # Create new group
                grouped.append(sum(current_group) / len(current_group))
                current_group = [sorted_levels[i]]
        
        # Add last group
        if current_group:
            grouped.append(sum(current_group) / len(current_group))
            
        return grouped
    
    def update_consolidation_zones(self, symbol: str, data: pd.DataFrame):
        """
        Update consolidation zones for a symbol.
        
        Args:
            symbol: The symbol to update zones for
            data: DataFrame with OHLCV data
        """
        if symbol not in self.consolidation_zones:
            self.consolidation_zones[symbol] = []
        
        # Calculate ATR
        atr_period = self.parameters['atr_period']
        if len(data) < atr_period + 5:
            return
            
        # Calculate ATR
        tr1 = abs(data['high'] - data['low'])
        tr2 = abs(data['high'] - data['close'].shift())
        tr3 = abs(data['low'] - data['close'].shift())
        tr = pd.DataFrame([tr1, tr2, tr3]).max()
        atr = tr.rolling(window=atr_period).mean()
        
        # Get recent data
        lookback = min(20, len(data) - atr_period)
        recent_data = data.iloc[-(lookback+atr_period):]
        recent_atr = atr.iloc[-lookback:]
        
        # Look for consolidation (low volatility periods)
        max_range_mult = self.parameters['consolidation_atr_mult']
        min_bars = self.parameters['min_consolidation_bars']
        
        # Find periods where price is consolidating
        consolidation_start = None
        for i in range(lookback):
            current_range = recent_data['high'].iloc[i+atr_period] - recent_data['low'].iloc[i+atr_period]
            
            if current_range <= recent_atr.iloc[i] * max_range_mult:
                # Potential consolidation
                if consolidation_start is None:
                    consolidation_start = i + atr_period
            else:
                # End of consolidation
                if consolidation_start is not None:
                    consolidation_length = i + atr_period - consolidation_start
                    
                    if consolidation_length >= min_bars:
                        # Calculate zone boundaries
                        zone_start = recent_data.index[consolidation_start]
                        zone_end = recent_data.index[i + atr_period - 1]
                        zone_high = recent_data['high'].iloc[consolidation_start:i+atr_period].max()
                        zone_low = recent_data['low'].iloc[consolidation_start:i+atr_period].min()
                        
                        # Add to zones
                        self.consolidation_zones[symbol].append({
                            'start': zone_start,
                            'end': zone_end,
                            'high': zone_high,
                            'low': zone_low,
                            'timestamp': datetime.now()
                        })
                    
                    consolidation_start = None
    
    def detect_breakout(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect breakouts from support/resistance or consolidation zones.
        
        Args:
            symbol: Symbol to check for breakouts
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with breakout information or None if no breakout
        """
        if len(data) < 3:
            return None
            
        current_price = data['close'].iloc[-1]
        current_high = data['high'].iloc[-1]
        current_low = data['low'].iloc[-1]
        current_volume = data['volume'].iloc[-1] if 'volume' in data else None
        
        breakout_info = None
        
        # Check support/resistance breakouts
        if symbol in self.support_resistance_levels:
            levels = self.support_resistance_levels[symbol]
            
            # Check resistance breakout
            for resistance in levels['resistance']:
                if (current_high > resistance and 
                    data['high'].iloc[-2] <= resistance):
                    # Potential bullish breakout
                    pip_value = self.parameters.get('pip_value', 0.0001)
                    breakout_pips = (current_high - resistance) / pip_value
                    
                    if breakout_pips >= self.parameters['breakout_threshold_pips']:
                        breakout_info = {
                            'type': 'resistance',
                            'direction': 'bullish',
                            'level': resistance,
                            'strength': breakout_pips / self.parameters['breakout_threshold_pips']
                        }
                        break
            
            # Check support breakout if no resistance breakout found
            if not breakout_info:
                for support in levels['support']:
                    if (current_low < support and 
                        data['low'].iloc[-2] >= support):
                        # Potential bearish breakout
                        pip_value = self.parameters.get('pip_value', 0.0001)
                        breakout_pips = (support - current_low) / pip_value
                        
                        if breakout_pips >= self.parameters['breakout_threshold_pips']:
                            breakout_info = {
                                'type': 'support',
                                'direction': 'bearish',
                                'level': support,
                                'strength': breakout_pips / self.parameters['breakout_threshold_pips']
                            }
                            break
        
        # Check consolidation breakouts if no S/R breakout found
        if not breakout_info and symbol in self.consolidation_zones:
            zones = self.consolidation_zones[symbol]
            
            # Check recent zones (last 24 hours)
            recent_time = datetime.now() - timedelta(hours=24)
            recent_zones = [z for z in zones if z['timestamp'] > recent_time]
            
            for zone in recent_zones:
                # Bullish breakout
                if current_high > zone['high']:
                    pip_value = self.parameters.get('pip_value', 0.0001)
                    breakout_pips = (current_high - zone['high']) / pip_value
                    
                    if breakout_pips >= self.parameters['breakout_threshold_pips']:
                        # Check for volume confirmation if available
                        volume_confirmed = True
                        if current_volume is not None and 'volume' in data:
                            avg_volume = data['volume'].iloc[-10:-1].mean()
                            volume_confirmed = current_volume >= avg_volume * self.parameters['volume_surge_factor']
                            
                        if volume_confirmed:
                            breakout_info = {
                                'type': 'consolidation',
                                'direction': 'bullish',
                                'level': zone['high'],
                                'strength': breakout_pips / self.parameters['breakout_threshold_pips']
                            }
                            break
                
                # Bearish breakout
                elif current_low < zone['low']:
                    pip_value = self.parameters.get('pip_value', 0.0001)
                    breakout_pips = (zone['low'] - current_low) / pip_value
                    
                    if breakout_pips >= self.parameters['breakout_threshold_pips']:
                        # Check for volume confirmation if available
                        volume_confirmed = True
                        if current_volume is not None and 'volume' in data:
                            avg_volume = data['volume'].iloc[-10:-1].mean()
                            volume_confirmed = current_volume >= avg_volume * self.parameters['volume_surge_factor']
                            
                        if volume_confirmed:
                            breakout_info = {
                                'type': 'consolidation',
                                'direction': 'bearish',
                                'level': zone['low'],
                                'strength': breakout_pips / self.parameters['breakout_threshold_pips']
                            }
                            break
        
        # Check session breakouts if no other breakout found
        if not breakout_info and symbol in self.session_high_lows and self.parameters['use_session_high_low']:
            sessions = self.session_high_lows[symbol]
            
            # Check each session
            for session, levels in sessions.items():
                # Skip if session is not recent (last 24 hours)
                if levels['timestamp'] < datetime.now() - timedelta(hours=24):
                    continue
                    
                # Bullish breakout of session high
                if current_high > levels['high']:
                    pip_value = self.parameters.get('pip_value', 0.0001)
                    breakout_pips = (current_high - levels['high']) / pip_value
                    
                    if breakout_pips >= self.parameters['breakout_threshold_pips']:
                        breakout_info = {
                            'type': 'session_high',
                            'direction': 'bullish',
                            'level': levels['high'],
                            'session': session,
                            'strength': breakout_pips / self.parameters['breakout_threshold_pips']
                        }
                        break
                
                # Bearish breakout of session low
                elif current_low < levels['low']:
                    pip_value = self.parameters.get('pip_value', 0.0001)
                    breakout_pips = (levels['low'] - current_low) / pip_value
                    
                    if breakout_pips >= self.parameters['breakout_threshold_pips']:
                        breakout_info = {
                            'type': 'session_low',
                            'direction': 'bearish',
                            'level': levels['low'],
                            'session': session,
                            'strength': breakout_pips / self.parameters['breakout_threshold_pips']
                        }
                        break
        
        return breakout_info
    
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
            if len(data) < 30:  # Need at least 30 bars for support/resistance
                continue
                
            # Skip if not a major pair and we're focusing on majors
            if (self.parameters['focus_on_major_pairs'] and 
                symbol not in self.PROP_RECOMMENDED_PAIRS):
                continue
                
            # Skip symbols with News events if configured
            if self.parameters['news_avoidance_minutes'] > 0 and self.should_avoid_news_events(symbol, datetime.now()):
                logger.info(f"Skipping {symbol} due to high impact news")
                continue
                
            # Update support/resistance levels
            self.update_support_resistance_levels(symbol, data)
            
            # Update consolidation zones
            self.update_consolidation_zones(symbol, data)
            
            # Detect breakout
            breakout = self.detect_breakout(symbol, data)
            
            if breakout:
                # Calculate ATR for stop loss and take profit
                atr = self._calculate_atr(data)
                current_price = data['close'].iloc[-1]
                
                # Generate signal based on breakout direction
                if breakout['direction'] == 'bullish':
                    # Base confidence on breakout strength and session
                    confidence = min(0.7 + (breakout['strength'] * 0.1), 0.9)
                    
                    # Adjust for session
                    if not in_preferred_session:
                        confidence *= 0.8
                    
                    # Only generate signal if confidence is high enough
                    if confidence >= 0.7:
                        # Set stop loss below the breakout level
                        stop_distance = atr * self.parameters['atr_stop_multiplier']
                        stop_loss = current_price - stop_distance
                        
                        # Ensure stop is below breakout level
                        stop_loss = min(stop_loss, breakout['level'] - (self.parameters.get('pip_value', 0.0001) * 3))
                        
                        # Set take profit based on R:R ratio
                        risk = current_price - stop_loss
                        take_profit = current_price + (risk * self.parameters['profit_target_mult'])
                        
                        signal = Signal(
                            symbol=symbol,
                            signal_type=SignalType.LONG,
                            confidence=confidence,
                            entry_price=current_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            metadata={
                                'strategy': self.name,
                                'breakout_type': breakout['type'],
                                'breakout_level': breakout['level'],
                                'atr': atr
                            }
                        )
                        
                elif breakout['direction'] == 'bearish':
                    # Base confidence on breakout strength and session
                    confidence = min(0.7 + (breakout['strength'] * 0.1), 0.9)
                    
                    # Adjust for session
                    if not in_preferred_session:
                        confidence *= 0.8
                    
                    # Only generate signal if confidence is high enough
                    if confidence >= 0.7:
                        # Set stop loss above the breakout level
                        stop_distance = atr * self.parameters['atr_stop_multiplier']
                        stop_loss = current_price + stop_distance
                        
                        # Ensure stop is above breakout level
                        stop_loss = max(stop_loss, breakout['level'] + (self.parameters.get('pip_value', 0.0001) * 3))
                        
                        # Set take profit based on R:R ratio
                        risk = stop_loss - current_price
                        take_profit = current_price - (risk * self.parameters['profit_target_mult'])
                        
                        signal = Signal(
                            symbol=symbol,
                            signal_type=SignalType.SHORT,
                            confidence=confidence,
                            entry_price=current_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            metadata={
                                'strategy': self.name,
                                'breakout_type': breakout['type'],
                                'breakout_level': breakout['level'],
                                'atr': atr
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
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = None) -> float:
        """Calculate Average True Range."""
        if period is None:
            period = self.parameters['atr_period']
            
        if len(data) < period:
            return 0.0
            
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.DataFrame([tr1, tr2, tr3]).max()
        atr = tr.rolling(period).mean()
        
        return atr.iloc[-1]
    
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
