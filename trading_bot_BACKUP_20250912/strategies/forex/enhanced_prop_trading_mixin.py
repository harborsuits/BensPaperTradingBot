#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Prop Trading Rules Mixin

This module provides an enhanced mixin class that enforces proprietary trading firm rules
for all forex trading strategies, with additional features for trade management.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta, time
from enum import Enum

from trading_bot.strategies_new.factory.strategy_template import Signal, SignalType

logger = logging.getLogger(__name__)

class EnhancedPropTradingMixin:
    """
    Enhanced mixin class to enforce proprietary trading firm rules for all forex strategies.
    
    This mixin enforces:
    1. Risk Management:
       - Max daily loss: 1-2% of account balance
       - Max drawdown: 5% maximum
       - Position sizing: 0.5-1% risk per trade
       - Leverage control: 10:1 or 20:1 maximum
       
    2. Trade Selection:
       - Minimum 2:1 reward-to-risk ratio
       - Trade with the trend when possible
       - Focus on major pairs for liquidity
       - Avoid high-impact news events
       
    3. Entry/Exit Rules:
       - Clear technical triggers
       - Momentum confirmation
       - Structure-based stop losses
       - Dynamic take profits
       
    4. Trade Management:
       - Scaling out at key levels
       - Trailing stops after partial profits
       - Time-based exit rules
       - Limited concurrent positions
       
    5. Psychological Discipline:
       - No revenge trading
       - Consistency over profits
       - Mandatory breaks after hitting loss limits
    """
    
    # Default prop trading parameters
    DEFAULT_ENHANCED_PROP_PARAMS = {
        # Risk Management
        'max_daily_loss_percent': 0.01,     # 1% max daily loss
        'max_drawdown_percent': 0.05,       # 5% max drawdown
        'risk_per_trade_percent': 0.005,    # 0.5% risk per trade (conservative)
        'max_leverage': 20,                 # 20:1 max leverage
        
        # Trade Selection
        'min_reward_risk_ratio': 2.0,       # Minimum 2:1 reward-to-risk
        'trade_with_trend': True,           # Prefer trend-following trades
        'focus_on_major_pairs': True,       # Prioritize major pairs
        'news_avoidance_minutes': 15,       # Avoid trading 15 min before/after news
        
        # Position Management
        'max_concurrent_positions': 3,      # Max 3 trades open at once
        'max_correlated_positions': 2,      # Max 2 correlated currency positions
        'scale_out_levels': [0.5, 0.75],    # Take partial profits at these levels
        
        # Dynamic Exit Management
        'use_trailing_stops': True,         # Enable trailing stops
        'trailing_activation_percent': 0.5, # Activate trailing at 50% to target
        'time_based_exits': True,           # Enable time-based exits
        'max_trade_duration_hours': 48,     # Max trade duration in hours
        
        # Session-Specific Parameters
        'london_session_hours': (7, 16),    # London session hours (UTC)
        'newyork_session_hours': (12, 21),  # New York session hours (UTC)
        'asian_session_hours': (0, 9),      # Asian session hours (UTC)
        'session_volatility_adjustment': True, # Adjust for session volatility
        
        # Psychological Rules
        'enforce_break_after_loss': True,   # Take mandatory break after hitting daily loss
        'break_duration_minutes': 60,       # 60 minute break after hitting daily loss
        'track_win_rate': True,             # Track win rate for strategy evaluation
    }
    
    # Major forex pairs with good liquidity for prop trading
    PROP_RECOMMENDED_PAIRS = [
        'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 
        'USD/CAD', 'USD/CHF', 'NZD/USD'
    ]
    
    def __init__(self, *args, **kwargs):
        """Initialize with prop trading parameters."""
        # Call the parent class's __init__ if it exists
        super().__init__(*args, **kwargs)
        
        # Merge default prop parameters with strategy parameters
        if hasattr(self, 'parameters'):
            self.parameters = {**self.DEFAULT_ENHANCED_PROP_PARAMS, **self.parameters}
        else:
            self.parameters = self.DEFAULT_ENHANCED_PROP_PARAMS.copy()
            
        # Initialize tracking variables
        self.daily_pnl = 0.0
        self.daily_trades = []
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.last_loss_time = None
        self.on_mandatory_break = False
        self.trade_history = []
        self.trade_start_times = {}
        self.partial_exits = {}
        
    # === RISK MANAGEMENT METHODS ===
    
    def validate_daily_loss_limit(self, account_balance: float) -> bool:
        """
        Check if the strategy has hit the daily loss limit.
        
        Args:
            account_balance: Current account balance
            
        Returns:
            True if within limits, False if daily loss limit hit
        """
        max_daily_loss = account_balance * self.parameters['max_daily_loss_percent']
        
        # Check if we're exceeding the daily loss limit
        if abs(self.daily_pnl) > max_daily_loss and self.daily_pnl < 0:
            logger.warning(
                f"Daily loss limit hit: {self.daily_pnl:.2f} exceeds {max_daily_loss:.2f}. " 
                f"Trading halted for this strategy today."
            )
            
            # Set mandatory break if enabled
            if self.parameters['enforce_break_after_loss']:
                self.last_loss_time = datetime.now()
                self.on_mandatory_break = True
                
            return False
            
        return True
        
    def validate_drawdown_limit(self, account_balance: float, 
                              starting_balance: float) -> bool:
        """
        Check if the strategy has exceeded the maximum drawdown limit.
        
        Args:
            account_balance: Current account balance
            starting_balance: Starting account balance
            
        Returns:
            True if within limits, False if drawdown limit exceeded
        """
        # Calculate current drawdown percentage
        drawdown_percent = (starting_balance - account_balance) / starting_balance
        self.current_drawdown = max(self.current_drawdown, drawdown_percent)
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Check if exceeding maximum drawdown
        if drawdown_percent > self.parameters['max_drawdown_percent']:
            logger.warning(
                f"Maximum drawdown limit exceeded: {drawdown_percent:.2%} exceeds " 
                f"{self.parameters['max_drawdown_percent']:.2%}. Trading halted."
            )
            return False
            
        return True
    
    def validate_reward_risk_ratio(self, entry_price: float, take_profit: float, 
                                stop_loss: float, signal_type: SignalType) -> bool:
        """
        Validate that the trade meets minimum reward-to-risk ratio.
        
        Args:
            entry_price: Entry price
            take_profit: Take profit level
            stop_loss: Stop loss level
            signal_type: Type of signal (LONG/SHORT)
            
        Returns:
            True if reward-to-risk ratio is acceptable, False otherwise
        """
        min_ratio = self.parameters['min_reward_risk_ratio']
        
        if signal_type == SignalType.LONG:
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:  # SHORT
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
            
        # Avoid division by zero
        if risk <= 0:
            logger.warning("Invalid risk calculation: risk must be positive")
            return False
            
        ratio = reward / risk
        
        if ratio < min_ratio:
            logger.info(
                f"Trade rejected: Reward-to-risk ratio {ratio:.2f} below minimum {min_ratio:.2f}"
            )
            return False
            
        return True
    
    # === POSITION SIZING METHODS ===
        
    def calculate_prop_position_size(self, account_balance: float, entry_price: float,
                                  stop_loss: float, symbol: str) -> float:
        """
        Calculate position size based on prop trading risk parameters.
        
        Args:
            account_balance: Current account balance
            entry_price: Entry price
            stop_loss: Stop loss price
            symbol: Trading symbol
            
        Returns:
            Position size in standard lots
        """
        # Risk amount based on account size
        risk_amount = account_balance * self.parameters['risk_per_trade_percent']
        
        # Calculate risk in pips
        pip_value = 0.0001  # Standard pip value (adjust for JPY pairs in real implementation)
        stop_distance_pips = abs(entry_price - stop_loss) / pip_value
        
        # Calculate lot size (1 standard lot = 100,000 units)
        # For a real implementation, need proper pip value calculation per pair
        if stop_distance_pips <= 0:
            logger.warning("Invalid stop distance: must be greater than zero")
            return 0
            
        standard_lots = risk_amount / (stop_distance_pips * 10)  # Approximate calculation
        
        # Enforce maximum leverage rules
        max_position_value = account_balance * self.parameters['max_leverage']
        position_value = standard_lots * 100000  # Value of position in account currency
        
        if position_value > max_position_value:
            logger.info(
                f"Position size reduced due to leverage constraints: " 
                f"{position_value:.2f} > {max_position_value:.2f}"
            )
            standard_lots = max_position_value / 100000
            
        return standard_lots
    
    # === POSITION MANAGEMENT METHODS ===
        
    def validate_concurrent_positions(self, current_positions: List[Dict]) -> bool:
        """
        Validate that we're not exceeding maximum concurrent positions.
        
        Args:
            current_positions: List of current open positions
            
        Returns:
            True if within limits, False if too many positions are open
        """
        if len(current_positions) >= self.parameters['max_concurrent_positions']:
            logger.info(
                f"Maximum concurrent positions reached: {len(current_positions)} positions open"
            )
            return False
            
        # Check for correlated positions
        currencies = []
        for position in current_positions:
            symbol = position.get('symbol', '')
            if '/' in symbol:
                base, quote = symbol.split('/')
                currencies.extend([base, quote])
        
        # Count occurrences of each currency
        currency_counts = {}
        for currency in currencies:
            if currency not in currency_counts:
                currency_counts[currency] = 0
            currency_counts[currency] += 1
            
        # Check if any currency is overexposed
        for currency, count in currency_counts.items():
            if count > self.parameters['max_correlated_positions']:
                logger.info(
                    f"Too many correlated positions with {currency}: {count} occurrences"
                )
                return False
                
        return True
        
    def check_in_mandatory_break(self) -> bool:
        """
        Check if we're in a mandatory break period after hitting loss limits.
        
        Returns:
            True if in mandatory break, False otherwise
        """
        if not self.on_mandatory_break or not self.last_loss_time:
            return False
            
        # Check if break period has passed
        elapsed_minutes = (datetime.now() - self.last_loss_time).total_seconds() / 60
        if elapsed_minutes >= self.parameters['break_duration_minutes']:
            logger.info("Mandatory break period ended, resuming trading")
            self.on_mandatory_break = False
            return False
            
        logger.info(
            f"In mandatory break period: {int(elapsed_minutes)} of " 
            f"{self.parameters['break_duration_minutes']} minutes elapsed"
        )
        return True
    
    # === TRAILING STOP AND PARTIAL EXIT METHODS ===
    
    def process_partial_exits(self, position_data: Dict, current_price: float) -> List[Dict]:
        """
        Process partial exit strategy for all forex strategies.
        
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
        position_id = position_data.get('position_id', '')
        entry_price = position_data.get('entry_price', 0)
        take_profit = position_data.get('take_profit', 0)
        position_size = position_data.get('position_size', 0)
        position_type = position_data.get('position_type', '')
        symbol = position_data.get('symbol', '')
        
        # Initialize tracking for this position if not exists
        if position_id not in self.partial_exits:
            self.partial_exits[position_id] = {'levels_taken': []}
            
        # Calculate profit targets based on scale-out levels
        for idx, level in enumerate(self.parameters['scale_out_levels']):
            # Skip if this level already taken
            if idx in self.partial_exits[position_id]['levels_taken']:
                continue
                
            # Calculate price target
            if position_type == 'LONG':
                price_distance = take_profit - entry_price
                target_price = entry_price + (price_distance * level)
                
                # Check if price has reached this target
                if current_price >= target_price:
                    # Calculate exit size (equal portions)
                    exit_size = position_size / (len(self.parameters['scale_out_levels']) + 1)
                    
                    # Add to exit orders
                    exit_orders.append({
                        'position_id': position_id,
                        'symbol': symbol,
                        'size': exit_size,
                        'price': current_price,
                        'type': 'PARTIAL_TAKE_PROFIT',
                        'level': idx
                    })
                    
                    # Mark this level as taken
                    self.partial_exits[position_id]['levels_taken'].append(idx)
                        
            elif position_type == 'SHORT':
                price_distance = entry_price - take_profit
                target_price = entry_price - (price_distance * level)
                
                # Check if price has reached this target
                if current_price <= target_price:
                    # Calculate exit size (equal portions)
                    exit_size = position_size / (len(self.parameters['scale_out_levels']) + 1)
                    
                    # Add to exit orders
                    exit_orders.append({
                        'position_id': position_id,
                        'symbol': symbol,
                        'size': exit_size,
                        'price': current_price,
                        'type': 'PARTIAL_TAKE_PROFIT',
                        'level': idx
                    })
                    
                    # Mark this level as taken
                    self.partial_exits[position_id]['levels_taken'].append(idx)
        
        return exit_orders
    
    def process_trailing_stops(self, position_data: Dict, current_price: float) -> Dict:
        """
        Process trailing stops for all forex strategies.
        
        Args:
            position_data: Current position data
            current_price: Current market price
            
        Returns:
            Updated stop loss level if trailing stop is triggered
        """
        if not self.parameters['use_trailing_stops']:
            return {}
            
        # Extract position information
        position_id = position_data.get('position_id', '')
        entry_price = position_data.get('entry_price', 0)
        current_stop = position_data.get('stop_loss', 0)
        take_profit = position_data.get('take_profit', 0)
        position_type = position_data.get('position_type', '')
        
        # Check if we've taken partial profits (more aggressive trailing after)
        has_partial_exits = (position_id in self.partial_exits and 
                             len(self.partial_exits[position_id]['levels_taken']) > 0)
        
        # Determine trailing activation point
        activation_percent = self.parameters['trailing_activation_percent']
        
        # For LONG positions
        if position_type == 'LONG':
            # Calculate profit percentage
            if entry_price <= 0:
                return {}
                
            price_distance = take_profit - entry_price
            profit_achieved = current_price - entry_price
            profit_percent = profit_achieved / price_distance if price_distance > 0 else 0
            
            # More aggressive trailing if we've already taken partial profits
            if has_partial_exits:
                activation_percent *= 0.8
            
            # Check if we've reached activation threshold
            if profit_percent >= activation_percent:
                # Calculate new stop level
                trailing_distance = (current_price - entry_price) * 0.5
                new_stop = entry_price + trailing_distance
                
                # Only update if it's higher than current stop
                if new_stop > current_stop:
                    return {'new_stop_loss': new_stop}
                    
        # For SHORT positions
        elif position_type == 'SHORT':
            # Calculate profit percentage
            if entry_price <= 0:
                return {}
                
            price_distance = entry_price - take_profit
            profit_achieved = entry_price - current_price
            profit_percent = profit_achieved / price_distance if price_distance > 0 else 0
            
            # More aggressive trailing if we've already taken partial profits
            if has_partial_exits:
                activation_percent *= 0.8
            
            # Check if we've reached activation threshold
            if profit_percent >= activation_percent:
                # Calculate new stop level
                trailing_distance = (entry_price - current_price) * 0.5
                new_stop = entry_price - trailing_distance
                
                # Only update if it's lower than current stop
                if new_stop < current_stop:
                    return {'new_stop_loss': new_stop}
                    
        return {}
    
    def check_time_based_exits(self, position_data: Dict) -> bool:
        """
        Check if a position should be exited based on time limits.
        
        Args:
            position_data: Current position data
            
        Returns:
            True if position should be exited, False otherwise
        """
        if not self.parameters['time_based_exits']:
            return False
            
        position_id = position_data.get('position_id', '')
        
        # Skip if we don't have a start time
        if position_id not in self.trade_start_times:
            return False
            
        # Check time elapsed
        start_time = self.trade_start_times[position_id]
        elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
        
        # Check if we've exceeded maximum duration
        if elapsed_hours >= self.parameters['max_trade_duration_hours']:
            logger.info(
                f"Time-based exit for position {position_id}: " 
                f"{elapsed_hours:.1f} hours exceed maximum of "
                f"{self.parameters['max_trade_duration_hours']} hours"
            )
            return True
            
        return False
        
    # === NEWS AND SESSION METHODS ===
    
    def should_avoid_news_events(self, symbol: str, current_time: datetime) -> bool:
        """
        Check if we should avoid trading due to upcoming or recent news.
        
        Args:
            symbol: Trading symbol
            current_time: Current datetime
            
        Returns:
            True if trading should be avoided, False otherwise
        """
        # This would integrate with a forex economic calendar
        # For now, return a placeholder implementation
        return False
    
    def is_current_session_active(self, target_sessions: List) -> bool:
        """
        Check if the current time is within any of the specified forex sessions.
        
        Args:
            target_sessions: List of sessions to check
            
        Returns:
            True if current time is within any target session
        """
        current_time = datetime.utcnow().time()
        
        # Check each session
        for session in target_sessions:
            if session == 'london':
                start_hour, end_hour = self.parameters['london_session_hours']
                session_start = time(start_hour, 0)
                session_end = time(end_hour, 0)
                
                # Check if current time is within session
                if session_start <= current_time <= session_end:
                    return True
                    
            elif session == 'newyork':
                start_hour, end_hour = self.parameters['newyork_session_hours']
                session_start = time(start_hour, 0)
                session_end = time(end_hour, 0)
                
                # Check if current time is within session
                if session_start <= current_time <= session_end:
                    return True
                    
            elif session == 'asian':
                start_hour, end_hour = self.parameters['asian_session_hours']
                session_start = time(start_hour, 0)
                session_end = time(end_hour, 0)
                
                # Check if current time is within session
                if session_start <= current_time <= session_end:
                    return True
        
        return False
    
    # === COMPREHENSIVE VALIDATION AND TRACKING METHODS ===
    
    def validate_prop_trading_rules(self, signal: Signal, account_balance: float,
                                 starting_balance: float, 
                                 current_positions: List[Dict]) -> bool:
        """
        Comprehensive validation of all prop trading rules.
        
        Args:
            signal: Trading signal
            account_balance: Current account balance
            starting_balance: Starting account balance
            current_positions: List of current open positions
            
        Returns:
            True if all rules pass, False otherwise
        """
        # Check if we're in a mandatory break period
        if self.check_in_mandatory_break():
            return False
            
        # Validate daily loss limit
        if not self.validate_daily_loss_limit(account_balance):
            return False
            
        # Validate drawdown limit
        if not self.validate_drawdown_limit(account_balance, starting_balance):
            return False
            
        # Validate reward-to-risk ratio
        if not self.validate_reward_risk_ratio(
            signal.entry_price, signal.take_profit, signal.stop_loss, signal.signal_type):
            return False
            
        # Validate maximum concurrent positions
        if not self.validate_concurrent_positions(current_positions):
            return False
            
        # Check for news events if applicable
        if self.parameters['news_avoidance_minutes'] > 0 and self.should_avoid_news_events(
            signal.symbol, datetime.now()):
            logger.info(f"Trade avoided due to nearby news event for {signal.symbol}")
            return False
            
        # Check if we're focusing on major pairs only
        if (self.parameters['focus_on_major_pairs'] and 
            signal.symbol not in self.PROP_RECOMMENDED_PAIRS):
            logger.info(
                f"Trade avoided: {signal.symbol} not in recommended major pairs list"
            )
            return False
            
        return True
        
    def update_trade_record(self, trade_result: Dict) -> None:
        """
        Update trade history and performance metrics.
        
        Args:
            trade_result: Dictionary with trade outcome information
        """
        self.trade_history.append(trade_result)
        
        # Update daily P&L
        pnl = trade_result.get('pnl', 0)
        self.daily_pnl += pnl
        
        # Track win rate if enabled
        if self.parameters['track_win_rate'] and len(self.trade_history) > 0:
            winning_trades = sum(1 for trade in self.trade_history if trade.get('pnl', 0) > 0)
            win_rate = winning_trades / len(self.trade_history)
            
            logger.info(f"Current win rate: {win_rate:.2%} ({winning_trades}/{len(self.trade_history)})")
            
        # For new positions, store start time
        if trade_result.get('action') == 'OPEN':
            position_id = trade_result.get('position_id')
            if position_id:
                self.trade_start_times[position_id] = datetime.now()
                logger.debug(f"Started tracking time for position {position_id}")
        
    def reset_daily_tracking(self) -> None:
        """Reset daily tracking variables for a new trading day."""
        self.daily_pnl = 0.0
        self.daily_trades = []
        self.on_mandatory_break = False
        
        # Clean up old position tracking data
        for position_id in list(self.trade_start_times.keys()):
            if position_id not in [p.get('position_id') for p in self.current_positions]:
                del self.trade_start_times[position_id]
                
        for position_id in list(self.partial_exits.keys()):
            if position_id not in [p.get('position_id') for p in self.current_positions]:
                del self.partial_exits[position_id]
