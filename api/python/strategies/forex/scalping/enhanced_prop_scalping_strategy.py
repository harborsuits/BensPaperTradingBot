#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Prop Scalping Strategy

This module implements a forex scalping strategy that incorporates the 
enhanced prop trading rules required for proprietary trading.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta

from trading_bot.strategies.forex.base.forex_base_strategy import ForexBaseStrategy, ForexSession
from trading_bot.strategies.forex.enhanced_prop_trading_mixin import EnhancedPropTradingMixin
from trading_bot.strategies.factory.strategy_registry import register_strategy
from trading_bot.core.event_system import EventBus, Event, EventType
from trading_bot.strategies_new.factory.strategy_template import Signal, SignalType

logger = logging.getLogger(__name__)

@register_strategy({
    'asset_class': 'forex',
    'strategy_type': 'enhanced_prop_scalping',
    'compatible_market_regimes': ['low_volatility', 'ranging'],
    'timeframe': 'intraday',
    'regime_compatibility_scores': {
        'trending': 0.60,        # Moderate compatibility with trending markets
        'ranging': 0.85,         # High compatibility with ranging markets
        'volatile': 0.45,        # Poor compatibility with volatile markets
        'low_volatility': 0.90,  # Excellent compatibility with low volatility
        'all_weather': 0.65      # Good overall compatibility
    }
})
class EnhancedPropScalpingStrategy(EnhancedPropTradingMixin, ForexBaseStrategy):
    """
    Enhanced Proprietary Forex Scalping Strategy
    
    This strategy aims to capture small price movements in forex markets using:
    - Fast moving averages (EMA)
    - Bollinger Bands for volatility
    - RSI for overbought/oversold conditions
    - Support/resistance levels
    
    It strictly adheres to enhanced prop trading rules:
    - Maximum 1-2% daily loss limit
    - Maximum 5% drawdown limit
    - Conservative 0.5% risk per trade
    - Minimum 2:1 reward-risk ratio
    - Partial take-profits at defined levels
    - Trailing stops after partial exits
    - Time-based exit rules
    """
    
    # Default parameters specific to scalping
    DEFAULT_PARAMS = {
        # Technical parameters
        'fast_ema': 5,                 # Fast EMA period
        'medium_ema': 10,              # Medium EMA period
        'slow_ema': 20,                # Slow EMA period
        'bb_period': 20,               # Bollinger Bands period
        'bb_std': 2.0,                 # Bollinger Bands standard deviation
        'rsi_period': 7,               # RSI period
        'rsi_overbought': 70,          # RSI overbought threshold
        'rsi_oversold': 30,            # RSI oversold threshold
        
        # Trading parameters
        'min_volatility_pips': 3,      # Minimum volatility in pips
        'max_volatility_pips': 15,     # Maximum volatility in pips
        'max_spread_pips': 1.0,        # Maximum acceptable spread in pips
        
        # Execution parameters
        'use_limit_orders': True,      # Use limit orders for entry
        'use_immediate_stops': True,   # Place stop loss immediately
        'trail_stop': True,            # Use trailing stops
        
        # Session parameters
        'preferred_sessions': ['london', 'newyork', 'london_newyork_overlap'],
        'avoid_session_open_minutes': 15,  # Avoid first N minutes of session
        
        # Timeframe settings
        'primary_timeframe': '5min',   # Primary analysis timeframe
        'confirmation_timeframe': '1min', # Confirmation timeframe
        
        # Scalping-specific exit rules
        'quick_profit_target_pips': 10, # Quick profit target in pips
        'max_trade_duration_minutes': 60, # Maximum time in trade
        
        # Enhanced prop-specific parameters
        'risk_per_trade_percent': 0.005,    # 0.5% risk per trade (conservative)
        'max_daily_loss_percent': 0.01,     # 1% max daily loss
        'max_drawdown_percent': 0.05,       # 5% max drawdown
        'max_concurrent_positions': 2,      # Max 2 positions for scalping
        'scale_out_levels': [0.5, 0.8],     # Take partial profits at 50% and 80% of target
        'trailing_activation_percent': 0.3,  # Activate trailing stops at 30% to target
        'max_trade_duration_hours': 2,      # Max trade duration of 2 hours
    }
    
    def __init__(self, name: str = "EnhancedPropScalpingStrategy", 
                parameters: Optional[Dict[str, Any]] = None, 
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize Enhanced Proprietary Scalping Strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Initialize tracking variables
        self.last_signals = {}
        self.account_info = {'balance': 0.0, 'starting_balance': 0.0}
        self.current_positions = []
        self.trade_start_times = {}
        self.session_start_times = {}
        
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
        
        # Register for session start events
        event_bus.subscribe(EventType.SESSION_STARTED, self._on_session_started)
        
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
            
        # Store start time for new trades
        if trade_data.get('action') == 'OPEN':
            self.trade_start_times[trade_data.get('position_id')] = datetime.now()
            
        # Update trade record with result
        self.update_trade_record(trade_data)
    
    def _on_trading_day_completed(self, event: Event):
        """Handle trading day completed events."""
        # Reset daily tracking metrics
        self.reset_daily_tracking()
    
    def _on_session_started(self, event: Event):
        """Handle forex session start events."""
        session_data = event.data.get('session_data', {})
        session = session_data.get('session')
        
        if session:
            self.session_start_times[session] = datetime.now()
    
    def _on_market_data_updated(self, event: Event):
        """
        Handle market data updates.
        Process trade management including partial exits, trailing stops,
        and time-based exits which are critical for scalping.
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
                
            # Check for time-based exits (critical for scalping)
            if position_id in self.trade_start_times:
                start_time = self.trade_start_times[position_id]
                elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
                
                max_duration = self.parameters['max_trade_duration_minutes']
                if elapsed_minutes >= max_duration:
                    logger.info(
                        f"Time-based exit for {symbol}: {elapsed_minutes:.1f} minutes exceeded limit"
                    )
                    # In a real implementation, this would close the position
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
    
    def is_near_session_start(self, session: str) -> bool:
        """
        Check if we're within the avoid period after session start.
        
        Args:
            session: The forex session to check
            
        Returns:
            True if we're near the session start, False otherwise
        """
        if session not in self.session_start_times:
            return False
            
        session_start = self.session_start_times[session]
        avoid_minutes = self.parameters['avoid_session_open_minutes']
        
        elapsed_minutes = (datetime.now() - session_start).total_seconds() / 60
        return elapsed_minutes < avoid_minutes
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate scalping-specific technical indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of calculated indicators
        """
        # Extract parameters
        fast_ema = self.parameters['fast_ema']
        medium_ema = self.parameters['medium_ema'] 
        slow_ema = self.parameters['slow_ema']
        bb_period = self.parameters['bb_period']
        bb_std = self.parameters['bb_std']
        rsi_period = self.parameters['rsi_period']
        
        # Skip if not enough data
        if len(data) < max(slow_ema, bb_period, rsi_period) + 10:
            return {}
            
        # Calculate EMAs
        data['ema_fast'] = data['close'].ewm(span=fast_ema, adjust=False).mean()
        data['ema_medium'] = data['close'].ewm(span=medium_ema, adjust=False).mean()
        data['ema_slow'] = data['close'].ewm(span=slow_ema, adjust=False).mean()
        
        # Calculate Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=bb_period).mean()
        data['bb_std'] = data['close'].rolling(window=bb_period).std()
        data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * bb_std)
        data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * bb_std)
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        
        rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate ATR for volatility measurement
        data['atr'] = self._calculate_atr(data, 14)
        
        # Calculate volatility in pips
        pip_value = self.parameters.get('pip_value', 0.0001)
        data['volatility_pips'] = data['atr'] / pip_value
        
        # Return the last values of indicators
        return {
            'ema_fast': data['ema_fast'].iloc[-1],
            'ema_medium': data['ema_medium'].iloc[-1],
            'ema_slow': data['ema_slow'].iloc[-1],
            'bb_upper': data['bb_upper'].iloc[-1],
            'bb_middle': data['bb_middle'].iloc[-1],
            'bb_lower': data['bb_lower'].iloc[-1],
            'rsi': data['rsi'].iloc[-1],
            'volatility_pips': data['volatility_pips'].iloc[-1],
            'atr': data['atr'].iloc[-1],
            'current_price': data['close'].iloc[-1]
        }
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.DataFrame([tr1, tr2, tr3]).max()
        atr = tr.rolling(period).mean()
        
        return atr
    
    def generate_signals(self, universe: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate scalping signals for the universe of symbols.
        
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
            logger.info("Not in preferred trading session, skipping signal generation")
            return {}
        
        # Check if we're near a session start (high volatility period)
        for session in self.parameters['preferred_sessions']:
            if self.is_near_session_start(session):
                logger.info(f"Avoiding trading near {session} session start")
                return {}
        
        # Process each symbol in the universe
        for symbol, data in universe.items():
            # Skip if not enough data
            if len(data) < 30:  # Need at least 30 bars for indicators
                continue
                
            # Skip if not a major pair and we're focusing on majors
            if (self.parameters['focus_on_major_pairs'] and 
                symbol not in self.PROP_RECOMMENDED_PAIRS):
                continue
                
            # Skip symbols with News events if configured
            if self.parameters['news_avoidance_minutes'] > 0 and self.should_avoid_news_events(symbol, datetime.now()):
                logger.info(f"Skipping {symbol} due to high impact news")
                continue
                
            # Calculate indicators
            indicators = self.calculate_indicators(data)
            if not indicators:
                continue
                
            # Skip if volatility is too low or too high
            volatility_pips = indicators['volatility_pips']
            if (volatility_pips < self.parameters['min_volatility_pips'] or 
                volatility_pips > self.parameters['max_volatility_pips']):
                continue
                
            # Extract indicator values
            ema_fast = indicators['ema_fast']
            ema_medium = indicators['ema_medium']
            ema_slow = indicators['ema_slow']
            bb_upper = indicators['bb_upper']
            bb_lower = indicators['bb_lower']
            rsi = indicators['rsi']
            atr = indicators['atr']
            current_price = indicators['current_price']
            
            signal = None
            pip_value = self.parameters.get('pip_value', 0.0001)
            
            # LONG signal conditions
            if (ema_fast > ema_medium and 
                current_price > ema_slow and
                current_price < bb_upper and
                rsi < 70 and rsi > 30):
                
                # Calculate base confidence
                confidence = 0.7
                
                # Adjust for RSI
                if rsi < 60:  # Not overbought
                    confidence += 0.05
                    
                # Adjust for Bollinger Band position
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                if 0.3 < bb_position < 0.7:  # In middle of bands
                    confidence += 0.05
                    
                # Only generate signal if confidence is high enough
                if confidence >= 0.7:
                    # Calculate stop loss and take profit
                    stop_loss = current_price - (atr * 1.5)
                    
                    # Use fixed pip target for scalping
                    target_pips = self.parameters['quick_profit_target_pips']
                    take_profit = current_price + (target_pips * pip_value)
                    
                    # Ensure reward-risk ratio meets prop requirements
                    entry_price = current_price
                    reward = take_profit - entry_price
                    risk = entry_price - stop_loss
                    
                    if reward / risk < self.parameters['min_reward_risk_ratio']:
                        take_profit = entry_price + (risk * self.parameters['min_reward_risk_ratio'])
                    
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
                                'ema_fast': ema_fast,
                                'ema_medium': ema_medium,
                                'ema_slow': ema_slow,
                                'rsi': rsi,
                                'volatility_pips': volatility_pips
                            }
                        }
                    )
            
            # SHORT signal conditions
            elif (ema_fast < ema_medium and 
                  current_price < ema_slow and
                  current_price > bb_lower and
                  rsi > 30 and rsi < 70):
                
                # Calculate base confidence
                confidence = 0.7
                
                # Adjust for RSI
                if rsi > 40:  # Not oversold
                    confidence += 0.05
                    
                # Adjust for Bollinger Band position
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                if 0.3 < bb_position < 0.7:  # In middle of bands
                    confidence += 0.05
                    
                # Only generate signal if confidence is high enough
                if confidence >= 0.7:
                    # Calculate stop loss and take profit
                    stop_loss = current_price + (atr * 1.5)
                    
                    # Use fixed pip target for scalping
                    target_pips = self.parameters['quick_profit_target_pips']
                    take_profit = current_price - (target_pips * pip_value)
                    
                    # Ensure reward-risk ratio meets prop requirements
                    entry_price = current_price
                    reward = entry_price - take_profit
                    risk = stop_loss - entry_price
                    
                    if reward / risk < self.parameters['min_reward_risk_ratio']:
                        take_profit = entry_price - (risk * self.parameters['min_reward_risk_ratio'])
                    
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
                                'ema_fast': ema_fast,
                                'ema_medium': ema_medium,
                                'ema_slow': ema_slow,
                                'rsi': rsi,
                                'volatility_pips': volatility_pips
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
        
        return signals
        
    def calculate_position_size(self, signal: Signal, account_balance: float) -> float:
        """
        Calculate position size for the signal based on prop risk management rules.
        
        Args:
            signal: Trading signal
            account_balance: Current account balance
            
        Returns:
            Position size in units (standard lots)
        """
        # For scalping, use a more conservative position sizing
        original_risk_percent = self.parameters['risk_per_trade_percent']
        self.parameters['risk_per_trade_percent'] = min(original_risk_percent, 0.005)  # Max 0.5%
        
        # Use prop trading position sizing
        position_size = self.calculate_prop_position_size(
            account_balance=account_balance,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            symbol=signal.symbol
        )
        
        # Restore original risk percent
        self.parameters['risk_per_trade_percent'] = original_risk_percent
        
        return position_size
