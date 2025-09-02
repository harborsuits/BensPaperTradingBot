#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Contextual Strategy

This module implements an enhanced contextual strategy with dynamic risk management,
advanced entry filters, adaptive exit strategies, and regime change buffers.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EnhancedContextualStrategy")

# Event types for communication
class EventType:
    """Constants for event types"""
    MARKET_REGIME_CHANGE = "market_regime_change"
    VOLATILITY_UPDATE = "volatility_update"
    CORRELATION_UPDATE = "correlation_update"
    TRADE_EXECUTED = "trade_executed"
    TRADE_CLOSED = "trade_closed"

class Event:
    """Simple event class for the system"""
    def __init__(self, event_type, data):
        self.event_type = event_type
        self.data = data
        self.timestamp = datetime.now()

class EventBus:
    """Event bus for communication between components"""
    def __init__(self):
        self.subscribers = {}
        self.event_history = []
        
    def subscribe(self, event_type, callback):
        """Subscribe to an event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def publish(self, event):
        """Publish an event to subscribers"""
        logger.debug(f"Publishing event: {event.event_type}")
        self.event_history.append(event)
        
        if event.event_type in self.subscribers:
            for callback in self.subscribers[event.event_type]:
                callback(event)

class EnhancedContextualStrategy:
    """
    Enhanced strategy that adapts to market regimes with advanced features:
    - Dynamic risk management based on market regimes and account balances
    - Advanced entry filters using technical indicators and pattern recognition
    - Adaptive exit strategies with regime-specific parameters
    - Buffer periods after regime changes to avoid false signals
    """
    
    def __init__(self, event_bus=None, account_balance=5000.0):
        """
        Initialize the enhanced contextual strategy.
        
        Args:
            event_bus: Event bus for communication
            account_balance: Initial account balance
        """
        self.event_bus = event_bus or EventBus()
        self.account_balance = account_balance
        
        # Current context tracks market and trading conditions
        self.current_context = {
            'market_regime': 'unknown',
            'volatility_state': 'medium',
            'last_regime_change': datetime.now() - timedelta(days=30),
            'bars_since_regime_change': 100,  # Start high to allow immediate trading
            'successful_patterns': {},
            'market_conditions': {}
        }
        
        # Dictionary to track performance by regime
        self.regime_performance = {
            'trending_up': {'wins': 0, 'losses': 0, 'profit': 0, 'drawdown': 0},
            'trending_down': {'wins': 0, 'losses': 0, 'profit': 0, 'drawdown': 0},
            'ranging': {'wins': 0, 'losses': 0, 'profit': 0, 'drawdown': 0},
            'breakout': {'wins': 0, 'losses': 0, 'profit': 0, 'drawdown': 0},
            'unknown': {'wins': 0, 'losses': 0, 'profit': 0, 'drawdown': 0}
        }
        
        # Strategy selection with optimized parameters for each regime
        self.regime_strategies = {
            'trending_up': {
                'id': 'trend_following', 
                'name': 'Trend Following',
                'tp_sl_ratio': 3.0,  # Higher reward:risk in trending markets
                'use_trailing_stop': True,
                'trailing_activation': 0.5,  # Activate trailing stop after 50% of target reached
                'trailing_distance': 0.4,    # Trail by 40% of original stop distance
                'min_strength': 0.6,         # Minimum trend strength to enter
                'entry_filters': ['trend_confirmation', 'momentum', 'volatility_check'],
                'exit_filters': ['trend_weakening', 'overbought']
            },
            'trending_down': {
                'id': 'downtrend', 
                'name': 'Downtrend Strategy',
                'tp_sl_ratio': 3.0,
                'use_trailing_stop': True,
                'trailing_activation': 0.5,
                'trailing_distance': 0.4,
                'min_strength': 0.6,
                'entry_filters': ['trend_confirmation', 'momentum', 'volatility_check'],
                'exit_filters': ['trend_weakening', 'oversold']
            },
            'ranging': {
                'id': 'mean_reversion', 
                'name': 'Mean Reversion',
                'tp_sl_ratio': 1.5,  # Lower reward:risk in ranging markets
                'use_trailing_stop': False,
                'min_strength': 0.5,
                'entry_filters': ['range_boundaries', 'oversold_overbought'],
                'exit_filters': ['approaching_mean']
            },
            'breakout': {
                'id': 'breakout', 
                'name': 'Breakout Strategy',
                'tp_sl_ratio': 2.5,
                'use_trailing_stop': True,
                'trailing_activation': 0.3,  # Earlier trailing for breakouts
                'trailing_distance': 0.5,    # Wider trail distance due to volatility
                'min_strength': 0.7,
                'entry_filters': ['volume_confirmation', 'breakout_strength'],
                'exit_filters': ['momentum_shift']
            },
            'unknown': {
                'id': 'balanced', 
                'name': 'Balanced Strategy',
                'tp_sl_ratio': 2.0,
                'use_trailing_stop': False,
                'min_strength': 0.0,
                'entry_filters': [],
                'exit_filters': []
            }
        }
        
        # Buffer settings
        self.regime_change_buffer = 3     # Bars to wait after regime change
        self.min_bars_between_trades = 6  # Prevent overtrading
        self.last_trade_bar = {}          # Track last trade by symbol
        
        # Pattern recognition database
        self.pattern_database = {
            'trending_up': [],
            'trending_down': [],
            'ranging': [],
            'breakout': []
        }
        
        # Subscribe to context events
        self.event_bus.subscribe(EventType.MARKET_REGIME_CHANGE, self.handle_regime_change)
        self.event_bus.subscribe(EventType.VOLATILITY_UPDATE, self.handle_volatility_update)
        self.event_bus.subscribe(EventType.CORRELATION_UPDATE, self.handle_correlation_update)
        self.event_bus.subscribe(EventType.TRADE_CLOSED, self.handle_trade_closed)
    
    def handle_regime_change(self, event):
        """Handle market regime change events."""
        regime = event.data.get('regime', 'unknown')
        symbol = event.data.get('symbol', 'unknown')
        
        # Record previous regime for transition analysis
        prev_regime = self.current_context.get('market_regime', 'unknown')
        
        # Update context with new regime
        self.current_context['market_regime'] = regime
        self.current_context['last_regime_change'] = datetime.now()
        self.current_context['bars_since_regime_change'] = 0
        self.current_context['previous_regime'] = prev_regime
        
        logger.info(f"Strategy updated regime to: {regime} (from {prev_regime}) for {symbol}")
        
        # Reset per-symbol tracking when regime changes
        if symbol in self.last_trade_bar:
            # Keep track but reset the counter to allow trades after buffer period
            self.last_trade_bar[symbol] = -self.regime_change_buffer
    
    def handle_volatility_update(self, event):
        """Handle volatility update events."""
        volatility = event.data.get('volatility_state', 'medium')
        symbol = event.data.get('symbol', 'unknown')
        
        self.current_context['volatility_state'] = volatility
        self.current_context['market_conditions'][symbol] = {
            **self.current_context['market_conditions'].get(symbol, {}),
            'volatility': volatility,
            'volatility_value': event.data.get('volatility_value', 0)
        }
        
        logger.debug(f"Strategy updated volatility to: {volatility} for {symbol}")
    
    def handle_correlation_update(self, event):
        """Handle correlation update events."""
        correlation_matrix = event.data.get('correlation_matrix', {})
        self.current_context['correlation_matrix'] = correlation_matrix
        logger.debug(f"Strategy updated correlation matrix")
    
    def handle_trade_closed(self, event):
        """Handle trade closed events to update performance statistics and patterns."""
        trade_data = event.data
        symbol = trade_data.get('symbol', 'unknown')
        regime = trade_data.get('regime', 'unknown')
        pnl = trade_data.get('pnl', 0)
        win = pnl > 0
        
        # Update regime performance
        if regime in self.regime_performance:
            if win:
                self.regime_performance[regime]['wins'] += 1
                self.regime_performance[regime]['profit'] += pnl
            else:
                self.regime_performance[regime]['losses'] += 1
                self.regime_performance[regime]['drawdown'] = min(
                    self.regime_performance[regime]['drawdown'],
                    -pnl
                )
        
        # Update pattern database for successful trades
        if win and 'entry_conditions' in trade_data:
            pattern = trade_data['entry_conditions']
            if regime in self.pattern_database and pattern:
                # Check if this pattern exists
                pattern_exists = False
                for p in self.pattern_database[regime]:
                    if self._pattern_similarity(p['pattern'], pattern) > 0.8:
                        # Update existing pattern stats
                        p['occurrences'] += 1
                        p['profit'] += pnl
                        p['avg_profit'] = p['profit'] / p['occurrences']
                        pattern_exists = True
                        break
                
                if not pattern_exists:
                    # Add new pattern
                    self.pattern_database[regime].append({
                        'pattern': pattern,
                        'occurrences': 1,
                        'profit': pnl,
                        'avg_profit': pnl
                    })
                    
        logger.debug(f"Updated performance for {regime} regime: Trade on {symbol} {'won' if win else 'lost'} {pnl:.2f}")
    
    def _pattern_similarity(self, pattern1, pattern2):
        """Calculate similarity between two patterns (simple implementation)."""
        # For this simple version, just check key indicators
        # In a complete implementation, this would use more sophisticated methods
        if not pattern1 or not pattern2:
            return 0
        
        similar_keys = 0
        total_keys = 0
        
        # Compare common keys
        for key in set(pattern1.keys()) & set(pattern2.keys()):
            total_keys += 1
            if isinstance(pattern1[key], bool) and pattern1[key] == pattern2[key]:
                similar_keys += 1
            elif isinstance(pattern1[key], (int, float)) and abs(pattern1[key] - pattern2[key]) < 0.1:
                similar_keys += 1
        
        if total_keys == 0:
            return 0
            
        return similar_keys / total_keys
    
    def select_strategy(self, symbol, market_data):
        """
        Select strategy based on current context with dynamic parameters.
        
        Args:
            symbol: Trading symbol
            market_data: Dataframe with market data for the symbol
            
        Returns:
            Strategy dict with parameters and signals
        """
        # Get current market regime and volatility
        regime = self.current_context.get('market_regime', 'unknown')
        volatility = self.current_context.get('volatility_state', 'medium')
        bars_since_change = self.current_context.get('bars_since_regime_change', 0)
        
        # Increment bars since regime change
        self.current_context['bars_since_regime_change'] = bars_since_change + 1
        
        # Get base strategy for this regime
        strategy = self.regime_strategies.get(regime, self.regime_strategies['unknown']).copy()
        
        # Adjust for volatility
        if volatility == 'high':
            if regime == 'ranging':
                # In high volatility ranging markets, use breakout strategy instead of mean reversion
                strategy = self.regime_strategies.get('breakout', self.regime_strategies['unknown']).copy()
                strategy['name'] += ' (High Vol)'
            else:
                # Reduce reward:risk ratio in high volatility trending markets for safety
                strategy['tp_sl_ratio'] *= 0.8
                # Increase trailing distance in high volatility
                if 'trailing_distance' in strategy:
                    strategy['trailing_distance'] *= 1.2
        elif volatility == 'low':
            # In low volatility, we can use tighter stops
            strategy['tp_sl_ratio'] *= 1.2
            # Tighten trailing distance in low volatility
            if 'trailing_distance' in strategy:
                strategy['trailing_distance'] *= 0.8
        
        # Check if we're still in buffer period after regime change
        if bars_since_change < self.regime_change_buffer:
            strategy['skip_trading'] = True
            strategy['skip_reason'] = f"Waiting for regime confirmation ({bars_since_change}/{self.regime_change_buffer} bars)"
        else:
            strategy['skip_trading'] = False
        
        # Check if we've traded too recently to avoid overtrading
        if symbol in self.last_trade_bar:
            bars_since_trade = bars_since_change - self.last_trade_bar[symbol]
            if bars_since_trade < self.min_bars_between_trades:
                strategy['skip_trading'] = True
                strategy['skip_reason'] = f"Avoiding overtrading ({bars_since_trade}/{self.min_bars_between_trades} bars since last trade)"
        
        # Apply pattern recognition to enhance strategy selection
        if regime in self.pattern_database and len(self.pattern_database[regime]) > 0:
            # Find the most profitable patterns for this regime
            profitable_patterns = sorted(
                self.pattern_database[regime], 
                key=lambda p: p['avg_profit'] * p['occurrences'],
                reverse=True
            )
            
            if len(profitable_patterns) > 0:
                top_pattern = profitable_patterns[0]
                logger.debug(f"Using top pattern for {regime} regime: {top_pattern['avg_profit']:.2f} avg profit over {top_pattern['occurrences']} occurrences")
                
                # Incorporate pattern parameters into strategy
                pattern = top_pattern['pattern']
                if 'entry_filters' in pattern:
                    strategy['entry_filters'].extend([f for f in pattern['entry_filters'] if f not in strategy['entry_filters']])
                if 'tp_sl_ratio' in pattern and pattern['tp_sl_ratio'] > 0:
                    strategy['tp_sl_ratio'] = pattern['tp_sl_ratio']
        
        # Calculate entry signals if we're not skipping trading
        if not strategy.get('skip_trading', False):
            signals = self._calculate_entry_signals(symbol, market_data, strategy)
            strategy.update(signals)
        
        return strategy
    
    def _calculate_entry_signals(self, symbol, data, strategy):
        """
        Calculate entry signals based on the selected strategy and current market data.
        
        Args:
            symbol: Trading symbol
            data: Market data DataFrame
            strategy: Selected strategy with parameters
            
        Returns:
            Dict with entry signals and conditions
        """
        if len(data) < 50:  # Need at least 50 bars for proper signals
            return {'signal': 'none', 'direction': 'none', 'entry_score': 0, 'entry_conditions': {}}
        
        # Get latest data and indicators
        recent = data.iloc[-20:].copy()
        latest = data.iloc[-1]
        regime = self.current_context.get('market_regime', 'unknown')
        
        # Initialize conditions and score
        entry_score = 0
        entry_conditions = {
            'close': latest['Close'],
            'regime': regime,
            'direction': 'none'
        }
        
        # Determine basic direction based on regime
        if regime == 'trending_up':
            direction = 'buy'
        elif regime == 'trending_down':
            direction = 'sell'
        elif regime == 'ranging':
            # For ranging markets, buy low/sell high based on RSI
            if 'RSI' in data.columns:
                rsi = latest['RSI']
                if rsi < 30:
                    direction = 'buy'  # Oversold
                    entry_conditions['oversold'] = True
                elif rsi > 70:
                    direction = 'sell'  # Overbought
                    entry_conditions['overbought'] = True
                else:
                    # Use mean reversion based on recent price action
                    if latest['Close'] < recent['Close'].mean():
                        direction = 'buy'
                    else:
                        direction = 'sell'
            else:
                # Default to mean reversion without RSI
                if latest['Close'] < recent['Close'].mean():
                    direction = 'buy'
                else:
                    direction = 'sell'
        elif regime == 'breakout':
            # For breakouts, use momentum direction
            if 'EMA20' in data.columns and 'EMA50' in data.columns:
                if latest['EMA20'] > latest['EMA50']:
                    direction = 'buy'
                else:
                    direction = 'sell'
            else:
                # Default to price momentum without EMAs
                direction = 'buy' if recent['Close'].pct_change().mean() > 0 else 'sell'
        else:
            # Unknown regime, use simple mean reversion
            direction = 'buy' if latest['Close'] < recent['Close'].mean() else 'sell'
        
        entry_conditions['direction'] = direction
        
        # Apply entry filters from strategy
        entry_filters = strategy.get('entry_filters', [])
        
        # Trend confirmation filter
        if 'trend_confirmation' in entry_filters:
            if 'EMA20' in data.columns and 'EMA50' in data.columns:
                if direction == 'buy':
                    if latest['EMA20'] > latest['EMA50']:
                        entry_score += 1
                        entry_conditions['trend_confirmed'] = True
                else:  # 'sell'
                    if latest['EMA20'] < latest['EMA50']:
                        entry_score += 1
                        entry_conditions['trend_confirmed'] = True
        
        # Momentum filter
        if 'momentum' in entry_filters:
            if 'RSI' in data.columns:
                momentum_aligned = (direction == 'buy' and latest['RSI'] > 50) or \
                                  (direction == 'sell' and latest['RSI'] < 50)
                if momentum_aligned:
                    entry_score += 1
                    entry_conditions['momentum_aligned'] = True
        
        # Volatility check
        if 'volatility_check' in entry_filters:
            if 'ATR14' in data.columns:
                volatility = latest['ATR14'] / latest['Close']
                right_volatility = ((direction == 'buy' or direction == 'sell') and volatility > 0.0005) or \
                                   (regime == 'ranging' and volatility < 0.001)
                if right_volatility:
                    entry_score += 1
                    entry_conditions['volatility_appropriate'] = True
                    entry_conditions['volatility'] = volatility
        
        # Range boundaries
        if 'range_boundaries' in entry_filters:
            if 'BBUpper' in data.columns and 'BBLower' in data.columns:
                near_boundary = (direction == 'buy' and latest['Close'] < latest['BBLower'] * 1.01) or \
                               (direction == 'sell' and latest['Close'] > latest['BBUpper'] * 0.99)
                if near_boundary:
                    entry_score += 1
                    entry_conditions['near_range_boundary'] = True
        
        # Volume confirmation for breakouts
        if 'volume_confirmation' in entry_filters and 'Volume' in data.columns:
            volume_increasing = recent['Volume'].iloc[-1] > recent['Volume'].mean() * 1.5
            if volume_increasing:
                entry_score += 1
                entry_conditions['high_volume'] = True
        
        # Determine overall signal
        min_score = len(entry_filters) * 0.6  # Need at least 60% of filters to pass
        if entry_score >= min_score and entry_score > 0:
            signal = 'buy' if direction == 'buy' else 'sell'
        else:
            signal = 'none'
        
        return {
            'signal': signal,
            'direction': direction,
            'entry_score': entry_score,
            'entry_conditions': entry_conditions
        }
    
    def calculate_position_size(self, symbol, entry_price, stop_loss_pips, account_balance=None):
        """
        Calculate position size based on account balance, risk profile, and market context.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_pips: Stop loss in pips
            account_balance: Optional override of account balance
            
        Returns:
            Dict with position size and risk information
        """
        # Use provided account balance or current balance
        balance = account_balance if account_balance is not None else self.account_balance
        
        # Progressive risk scaling parameters - balanced but still aggressive
        risk_table = {
            500: 0.25,     # 25% risk up to $500 (aggressive but survivable)
            1000: 0.22,    # 22% risk up to $1,000
            2500: 0.20,    # 20% risk up to $2,500
            5000: 0.18,    # 18% risk up to $5,000
            7000: 0.16,    # 16% risk up to $7,000
            10000: 0.15,   # 15% risk up to $10,000
            15000: 0.12,   # 12% risk up to $15,000
            20000: 0.10,   # 10% risk up to $20,000
            24999: 0.08,   # 8% risk up to $24,999
            25000: 0.05,   # 5% risk at $25,000 (PDT threshold)
            35000: 0.05,   # 5% risk up to $35,000
            50000: 0.04,   # 4% risk up to $50,000
            100000: 0.03,  # 3% risk up to $100,000
            250000: 0.025, # 2.5% risk up to $250,000
            500000: 0.02,  # 2% risk up to $500,000
            1000000: 0.015 # 1.5% risk above $1,000,000
        }
        
        # Determine base risk percentage from table
        risk_percentage = 0.01  # Default minimum
        for threshold, risk in sorted(risk_table.items()):
            if balance <= threshold:
                risk_percentage = risk
                break
        
        # Adjust for market regime
        regime = self.current_context.get('market_regime', 'unknown')
        if regime == 'trending_up' or regime == 'trending_down':
            # Slightly higher risk in clear trends
            risk_percentage *= 1.1
        elif regime == 'ranging':
            # Slightly reduced risk in ranges
            risk_percentage *= 0.9
        elif regime == 'breakout':
            # Higher risk for breakout opportunities
            risk_percentage *= 1.2
        elif regime == 'unknown':
            # Lower risk when regime is unclear
            risk_percentage *= 0.7
        
        # Adjust for volatility
        volatility = self.current_context.get('volatility_state', 'medium')
        if volatility == 'high':
            # Reduce risk in high volatility
            risk_percentage *= 0.8
        elif volatility == 'low':
            # Slightly increase risk in low volatility
            risk_percentage *= 1.1
        
        # Calculate risk amount
        risk_amount = balance * risk_percentage
        
        # Calculate pip value (simplified for major forex pairs)
        pip_value = 0.0001  # Standard pip value for 4-decimal forex pairs
        
        # For non-standard pip values (JPY pairs, etc.), adjust here
        if symbol.endswith('JPY'):
            pip_value = 0.01
        
        # Calculate position size
        position_size = risk_amount / (stop_loss_pips * pip_value * 10000)
        
        # Limit position size to avoid excessive leverage
        max_position = balance * 0.2 / entry_price  # 5:1 max effective leverage
        position_size = min(position_size, max_position)
        
        # Return position sizing information
        return {
            'position_size': position_size,
            'risk_percentage': risk_percentage * 100,
            'risk_amount': risk_amount,
            'stop_loss_pips': stop_loss_pips,
            'pip_value': pip_value
        }
    
    def update_balance(self, new_balance):
        """
        Update account balance after trades.
        
        Args:
            new_balance: New account balance
        """
        self.account_balance = new_balance
