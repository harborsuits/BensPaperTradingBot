#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Strategy Comparison Test: Contextual vs Static

This script compares the performance of trading strategies with and without 
contextual awareness (no visualization dependencies required).
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SimpleStrategyTest")

# Mock classes for testing without dependencies
class Event:
    """Mock Event class for demonstration"""
    def __init__(self, event_type, data):
        self.event_type = event_type
        self.data = data
        self.timestamp = datetime.now()
        
class EventType:
    """Mock EventType constants"""
    MARKET_REGIME_CHANGE = "market_regime_change"
    VOLATILITY_UPDATE = "volatility_update"
    
class EventBus:
    """Simplified EventBus for demonstration"""
    def __init__(self):
        self.subscribers = {}
        self.event_history = []
        
    def subscribe(self, event_type, callback):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def publish(self, event):
        logger.debug(f"Publishing event: {event.event_type}")
        self.event_history.append(event)
        
        if event.event_type in self.subscribers:
            for callback in self.subscribers[event.event_type]:
                callback(event)

# Simple trade simulation
class TradeSimulator:
    """Simple trade simulator for backtesting."""
    
    def __init__(self, data, initial_balance=5000.0):
        """
        Initialize the trade simulator.
        
        Args:
            data: Dictionary of symbol -> DataFrame with market data
            initial_balance: Starting account balance
        """
        self.data = data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trades = []
        self.equity_curve = []
        self.positions = {}
    
    def reset(self):
        """Reset the simulator to initial state."""
        self.balance = self.initial_balance
        self.trades = []
        self.equity_curve = []
        self.positions = {}
    
    def execute_trade(self, 
                    symbol: str, 
                    entry_time: datetime, 
                    direction: str, 
                    position_size: float, 
                    risk_amount: float, 
                    stop_loss_pips: float = 20,
                    take_profit_pips: float = 40,
                    max_bars: int = 100,
                    strategy_id: str = None,
                    strategy_name: str = None,
                    use_trailing_stop: bool = False,
                    tp_sl_ratio: float = 2.0):
        """
        Execute a simulated trade with enhanced exit strategies.
        
        Args:
            symbol: Trading symbol
            entry_time: Entry timestamp
            direction: 'buy' or 'sell'
            position_size: Position size in lots
            risk_amount: Amount risked on the trade
            stop_loss_pips: Stop loss in pips
            take_profit_pips: Take profit in pips (can be automatically calculated from tp_sl_ratio)
            max_bars: Maximum bars to hold the trade
            strategy_id: Strategy ID
            strategy_name: Strategy name
            use_trailing_stop: Whether to use trailing stops
            tp_sl_ratio: Take profit to stop loss ratio (for dynamic TP calculation)
            
        Returns:
            Trade result dictionary
        """
        if symbol not in self.data:
            logger.warning(f"Symbol {symbol} not found in data")
            return None
        
        # Find entry bar
        df = self.data[symbol]
        entry_idx = -1
        
        for i, time_idx in enumerate(df.index):
            if time_idx >= entry_time:
                entry_idx = i
                break
        
        if entry_idx == -1 or entry_idx >= len(df) - 1:
            logger.warning(f"Entry time {entry_time} not found in data")
            return None
        
        # Entry price
        entry_price = df.iloc[entry_idx]['close']
        
        # Calculate pip value (simplified for major forex pairs)
        pip_value = 0.0001  # Standard pip for forex
        
        # Dynamically calculate take profit based on tp_sl_ratio if provided
        if tp_sl_ratio > 0:
            take_profit_pips = stop_loss_pips * tp_sl_ratio
        
        # Calculate stop loss and take profit levels
        if direction == 'buy':
            stop_loss = entry_price - (stop_loss_pips * pip_value)
            take_profit = entry_price + (take_profit_pips * pip_value)
        else:  # 'sell'
            stop_loss = entry_price + (stop_loss_pips * pip_value)
            take_profit = entry_price - (take_profit_pips * pip_value)
        
        # Initialize trailing stop variables
        trailing_stop = stop_loss
        highest_price = entry_price if direction == 'buy' else entry_price
        lowest_price = entry_price if direction == 'sell' else entry_price
        trailing_activated = False
        trailing_distance_pips = stop_loss_pips * 0.5  # 50% of original stop
        
        # Track trade through subsequent bars
        exit_idx = -1
        exit_price = entry_price
        exit_reason = 'max_bars'
        
        for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(df))):
            bar = df.iloc[i]
            
            # Update highest/lowest prices for trailing stops
            if use_trailing_stop:
                if direction == 'buy':
                    # For buy trades, update highest price
                    if bar['high'] > highest_price:
                        highest_price = bar['high']
                        # Move trailing stop if we've moved in profit by at least half the stop distance
                        if highest_price > entry_price + (stop_loss_pips * pip_value * 0.5):
                            trailing_activated = True
                            # New trailing stop is highest price minus trailing distance
                            new_stop = highest_price - (trailing_distance_pips * pip_value)
                            # Only move the stop up, never down
                            if new_stop > trailing_stop:
                                trailing_stop = new_stop
                else:  # 'sell'
                    # For sell trades, update lowest price
                    if bar['low'] < lowest_price:
                        lowest_price = bar['low']
                        # Move trailing stop if we've moved in profit by at least half the stop distance
                        if lowest_price < entry_price - (stop_loss_pips * pip_value * 0.5):
                            trailing_activated = True
                            # New trailing stop is lowest price plus trailing distance
                            new_stop = lowest_price + (trailing_distance_pips * pip_value)
                            # Only move the stop down, never up
                            if new_stop < trailing_stop or not trailing_activated:
                                trailing_stop = new_stop
            
            # Check for stop loss (use trailing stop if activated)
            if direction == 'buy':
                stop_level = trailing_stop if trailing_activated else stop_loss
                if bar['low'] <= stop_level:
                    exit_price = stop_level
                    exit_idx = i
                    exit_reason = 'trailing_stop' if trailing_activated else 'stop_loss'
                    break
            else:  # 'sell'
                stop_level = trailing_stop if trailing_activated else stop_loss
                if bar['high'] >= stop_level:
                    exit_price = stop_level
                    exit_idx = i
                    exit_reason = 'trailing_stop' if trailing_activated else 'stop_loss'
                    break
            
            # Check for take profit
            if direction == 'buy':
                if bar['high'] >= take_profit:
                    exit_price = take_profit
                    exit_idx = i
                    exit_reason = 'take_profit'
                    break
            else:  # 'sell'
                if bar['low'] <= take_profit:
                    exit_price = take_profit
                    exit_idx = i
                    exit_reason = 'take_profit'
                    break
        
        # If we haven't exited yet, exit at the last processed bar
        if exit_idx == -1:
            exit_idx = min(entry_idx + max_bars, len(df) - 1)
            exit_price = df.iloc[exit_idx]['close']
        
        # Calculate profit/loss
        if direction == 'buy':
            pips_gained = (exit_price - entry_price) / pip_value
        else:  # 'sell'
            pips_gained = (entry_price - exit_price) / pip_value
        
        # Calculate PnL
        if exit_reason == 'stop_loss':
            pnl = -risk_amount
        elif exit_reason == 'take_profit':
            pnl = risk_amount * (take_profit_pips / stop_loss_pips)
        else:
            # Calculate based on actual pips gained/lost
            risk_per_pip = risk_amount / stop_loss_pips
            pnl = pips_gained * risk_per_pip
        
        # Update balance
        self.balance += pnl
        
        # Create trade record
        trade = {
            'symbol': symbol,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': df.index[exit_idx],
            'exit_price': exit_price,
            'direction': direction,
            'position_size': position_size,
            'risk_amount': risk_amount,
            'risk_percent': (risk_amount / (self.balance - pnl)) * 100,
            'stop_loss_pips': stop_loss_pips,
            'take_profit_pips': take_profit_pips,
            'pnl': pnl,
            'pips': pips_gained,
            'exit_reason': exit_reason,
            'bars_held': exit_idx - entry_idx,
            'strategy_id': strategy_id,
            'strategy_name': strategy_name,
            'balance_after': self.balance
        }
        
        # Add to trade history
        self.trades.append(trade)
        
        # Update equity curve
        self.equity_curve.append({
            'time': df.index[exit_idx],
            'balance': self.balance
        })
        
        return trade
    
    def get_results(self):
        """Get performance results."""
        if not self.trades:
            return {
                'initial_balance': self.initial_balance,
                'final_balance': self.balance,
                'total_return': 0,
                'win_rate': 0,
                'profit_factor': 0
            }
        
        # Calculate performance metrics
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        losses = sum(1 for t in self.trades if t['pnl'] <= 0)
        
        profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        loss = sum(abs(t['pnl']) for t in self.trades if t['pnl'] <= 0)
        
        win_rate = wins / len(self.trades) if len(self.trades) > 0 else 0
        profit_factor = profit / loss if loss > 0 else float('inf')
        
        # Calculate max drawdown
        peak = self.initial_balance
        drawdown = 0
        max_drawdown = 0
        
        for trade in self.trades:
            if trade['balance_after'] > peak:
                peak = trade['balance_after']
            else:
                drawdown = (peak - trade['balance_after']) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_return': (self.balance / self.initial_balance - 1) * 100,
            'total_trades': len(self.trades),
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }

class ContextualStrategy:
    """Strategy that adapts to market regimes with dynamic risk management."""
    
    def __init__(self, event_bus, account_balance=5000.0):
        """
        Initialize the contextual strategy.
        
        Args:
            event_bus: Event bus for communication
            account_balance: Initial account balance
        """
        self.event_bus = event_bus
        self.account_balance = account_balance
        
        # Current context
        self.current_context = {
            'market_regime': 'unknown',
            'volatility_state': 'medium',
            'last_regime_change': datetime.now() - timedelta(days=30),  # Initialize with past date
            'bars_since_regime_change': 100  # Start with high value to allow immediate trading
        }
        
        # Strategy selection based on regime with optimized TP/SL ratios
        self.regime_strategies = {
            'trending_up': {
                'id': 'trend_following', 
                'name': 'Trend Following',
                'tp_sl_ratio': 3.0,  # Higher reward:risk in trending markets
                'use_trailing_stop': True,
                'min_strength': 0.6,  # Minimum trend strength to enter
                'entry_filters': ['trend_confirmation', 'volatility_check']
            },
            'trending_down': {
                'id': 'downtrend', 
                'name': 'Downtrend Strategy',
                'tp_sl_ratio': 3.0,
                'use_trailing_stop': True,
                'min_strength': 0.6,
                'entry_filters': ['trend_confirmation', 'volatility_check']
            },
            'ranging': {
                'id': 'mean_reversion', 
                'name': 'Mean Reversion',
                'tp_sl_ratio': 1.5,  # Lower reward:risk in ranging markets
                'use_trailing_stop': False,
                'min_strength': 0.5,
                'entry_filters': ['range_boundaries']
            },
            'breakout': {
                'id': 'breakout', 
                'name': 'Breakout Strategy',
                'tp_sl_ratio': 2.5,
                'use_trailing_stop': True,
                'min_strength': 0.7,
                'entry_filters': ['volume_confirmation']
            },
            'unknown': {
                'id': 'balanced', 
                'name': 'Balanced Strategy',
                'tp_sl_ratio': 2.0,
                'use_trailing_stop': False,
                'min_strength': 0.0,  # No minimum strength required
                'entry_filters': []
            }
        }
        
        # Regime change buffer (bars to wait after regime change before trading)
        self.regime_change_buffer = 3
        
        # Minimum bars between trades (to avoid overtrading)
        self.min_bars_between_trades = 6
        self.last_trade_bar = {}
        
        # Subscribe to context events
        self.event_bus.subscribe(EventType.MARKET_REGIME_CHANGE, self.handle_regime_change)
        self.event_bus.subscribe(EventType.VOLATILITY_UPDATE, self.handle_volatility_update)
    
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
        
        logger.debug(f"Contextual strategy updated regime to: {regime} (from {prev_regime})")
    
    def handle_volatility_update(self, event):
        """Handle volatility update events."""
        volatility = event.data.get('volatility_state', 'medium')
        self.current_context['volatility_state'] = volatility
        logger.debug(f"Contextual strategy updated volatility to: {volatility}")
    
    def select_strategy(self, symbol, market_data):
        """Select strategy based on current context with dynamic parameters."""
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
                # In high volatility ranging markets, use range breakout instead of mean reversion
                strategy = self.regime_strategies.get('breakout', self.regime_strategies['unknown']).copy()
                strategy['name'] += ' (High Vol)'
            else:
                # Widen stops in high volatility trending markets
                strategy['tp_sl_ratio'] *= 0.8  # Reduce reward:risk in high volatility
        elif volatility == 'low':
            # In low volatility, can use tighter stops
            strategy['tp_sl_ratio'] *= 1.2  # Increase reward:risk in low volatility
        
        # Check if we're still in buffer period after regime change
        if bars_since_change < self.regime_change_buffer:
            strategy['skip_trading'] = True
            strategy['skip_reason'] = f"Waiting for regime confirmation ({bars_since_change}/{self.regime_change_buffer} bars)"
        else:
            strategy['skip_trading'] = False
        
        # Check if we've traded too recently
        if symbol in self.last_trade_bar:
            bars_since_trade = bars_since_change - self.last_trade_bar[symbol]
            if bars_since_trade < self.min_bars_between_trades:
                strategy['skip_trading'] = True
                strategy['skip_reason'] = f"Avoiding overtrading ({bars_since_trade}/{self.min_bars_between_trades} bars since last trade)"
        
        return strategy
    
    def calculate_position_size(self, symbol, entry_price, stop_loss_pips):
        """Calculate position size based on account balance and context."""
        # Starting risk table - more balanced approach
        risk_table = {
            500: 0.25,     # 25% risk up to $500 (still aggressive but survivable)
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
        
        # Determine risk percentage
        risk_percentage = 0.02  # Default minimum
        
        for threshold, risk in sorted(risk_table.items()):
            if self.account_balance <= threshold:
                risk_percentage = risk
                break
        
        # Adjust for volatility
        volatility = self.current_context.get('volatility_state', 'medium')
        if volatility == 'high':
            risk_percentage *= 0.8  # Reduce risk in high volatility
        elif volatility == 'low':
            risk_percentage *= 1.1  # Slightly increase risk in low volatility
        
        # Calculate risk amount
        risk_amount = self.account_balance * risk_percentage
        
        # Calculate pip value (simplified for major forex pairs)
        pip_value = 0.0001
        
        # Calculate position size
        position_size = risk_amount / (stop_loss_pips * pip_value * 10000)
        
        return {
            'position_size': position_size,
            'risk_percentage': risk_percentage * 100,
            'risk_amount': risk_amount
        }
    
    def update_balance(self, new_balance):
        """Update account balance."""
        self.account_balance = new_balance

class StaticStrategy:
    """Strategy with fixed parameters."""
    
    def __init__(self, account_balance=5000.0):
        """
        Initialize the static strategy.
        
        Args:
            account_balance: Initial account balance
        """
        self.account_balance = account_balance
        
        # Fixed strategy
        self.static_strategy = {'id': 'balanced', 'name': 'Balanced Strategy'}
        
        # Fixed risk percentage (2% of account)
        self.risk_percentage = 0.02
    
    def select_strategy(self, symbol, market_data):
        """Return fixed strategy."""
        return self.static_strategy
    
    def calculate_position_size(self, symbol, entry_price, stop_loss_pips):
        """Calculate position size using fixed risk percentage."""
        # Calculate risk amount
        risk_amount = self.account_balance * self.risk_percentage
        
        # Calculate pip value
        pip_value = 0.0001
        
        # Calculate position size
        position_size = risk_amount / (stop_loss_pips * pip_value * 10000)
        
        return {
            'position_size': position_size,
            'risk_percentage': self.risk_percentage * 100,
            'risk_amount': risk_amount
        }
    
    def update_balance(self, new_balance):
        """Update account balance."""
        self.account_balance = new_balance

def run_comparison_test():
    """Compare contextual vs static strategies."""
    # Initialize event bus
    event_bus = EventBus()
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data/market_data'):
        os.makedirs('data/market_data', exist_ok=True)
    
    # Test symbols
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    
    # Load test data
    market_data = {}
    for symbol in symbols:
        file_path = f'data/market_data/{symbol}_1h.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            market_data[symbol] = df
            logger.info(f"Loaded data for {symbol}: {len(df)} rows")
    
    # Check if we have data
    if not market_data:
        logger.error("No market data found. Please run simple_regime_test.py first.")
        return None
    
    # Initialize strategies
    contextual_strategy = ContextualStrategy(event_bus)
    static_strategy = StaticStrategy()
    
    # Initialize trade simulators
    contextual_simulator = TradeSimulator(market_data)
    static_simulator = TradeSimulator(market_data)
    
    # Run simulation
    logger.info("Starting strategy comparison test...")
    
    # Define market regimes for testing
    # These would normally come from the MarketRegimeDetector
    regime_periods = [
        {'symbol': 'EURUSD', 'start': market_data['EURUSD'].index[0], 
         'end': market_data['EURUSD'].index[len(market_data['EURUSD'])//4], 'regime': 'ranging'},
        {'symbol': 'EURUSD', 'start': market_data['EURUSD'].index[len(market_data['EURUSD'])//4], 
         'end': market_data['EURUSD'].index[len(market_data['EURUSD'])//2], 'regime': 'trending_up'},
        {'symbol': 'EURUSD', 'start': market_data['EURUSD'].index[len(market_data['EURUSD'])//2], 
         'end': market_data['EURUSD'].index[3*len(market_data['EURUSD'])//4], 'regime': 'ranging'},
        {'symbol': 'EURUSD', 'start': market_data['EURUSD'].index[3*len(market_data['EURUSD'])//4], 
         'end': market_data['EURUSD'].index[-1], 'regime': 'trending_down'}
    ]
    
    # Simulate each regime
    for regime_period in regime_periods:
        symbol = regime_period['symbol']
        start = regime_period['start']
        end = regime_period['end']
        regime = regime_period['regime']
        
        logger.info(f"Processing {regime} regime from {start} to {end} for {symbol}")
        
        # Publish regime change event
        event_bus.publish(Event(
            event_type=EventType.MARKET_REGIME_CHANGE,
            data={
                'symbol': symbol,
                'regime': regime,
                'previous_regime': 'unknown',
                'confidence': 0.85
            }
        ))
        
        # Process each trading day during this regime
        df = market_data[symbol]
        current_idx = df.index.get_indexer([start], method='nearest')[0]
        end_idx = df.index.get_indexer([end], method='nearest')[0]
        
        while current_idx < end_idx:
            # Current time
            current_time = df.index[current_idx]
            
            # Simulate volatility update occasionally
            if current_idx % 24 == 0:  # Once per day
                # Determine volatility based on regime
                if regime == 'ranging' and current_idx >= len(df)//2:
                    volatility = 'high'
                elif regime == 'trending_up':
                    volatility = 'medium'
                elif regime == 'trending_down':
                    volatility = 'high' if current_idx % 48 == 0 else 'medium'
                else:
                    volatility = 'medium'
                
                # Publish volatility update
                event_bus.publish(Event(
                    event_type=EventType.VOLATILITY_UPDATE,
                    data={
                        'symbol': symbol,
                        'volatility_state': volatility,
                        'previous_state': 'medium'
                    }
                ))
            
            # Every 12 hours, make a trading decision
            if current_idx % 12 == 0:
                # Get current price data
                price = df.iloc[current_idx]['close']
                
                # Make trading decisions
            
            # 1. Contextual strategy
            context_strategy = contextual_strategy.select_strategy(symbol, None)
            context_position = contextual_strategy.calculate_position_size(symbol, price, 20)
            
            # Check if trading should be skipped due to regime change buffer or other filters
            should_skip = context_strategy.get('skip_trading', False)
            skip_reason = context_strategy.get('skip_reason', '')
            
            # Generate trade signal
            if regime == 'trending_up':
                direction = 'buy'
            elif regime == 'trending_down': 
                direction = 'sell'
            elif regime == 'ranging':
                # For ranging markets, buy low/sell high based on recent movement
                recent_bars = df.iloc[max(0, current_idx-5):current_idx+1]
                if len(recent_bars) >= 5:
                    recent_change = (recent_bars['close'].iloc[-1] - recent_bars['close'].iloc[0]) / recent_bars['close'].iloc[0]
                    direction = 'sell' if recent_change > 0.0005 else 'buy'  # Mean reversion
                else:
                    direction = 'buy' if np.random.random() > 0.5 else 'sell'
            else:
                direction = 'buy' if np.random.random() > 0.5 else 'sell'
            
            # Implement entry filters
            entry_filters = context_strategy.get('entry_filters', [])
            entry_score = 0
            
            # Simple trend confirmation filter
            if 'trend_confirmation' in entry_filters:
                trend_direction = 1 if direction == 'buy' else -1
                recent_bars = df.iloc[max(0, current_idx-10):current_idx+1]
                if len(recent_bars) >= 10:
                    ema5 = recent_bars['close'].ewm(span=5).mean().iloc[-1]
                    ema10 = recent_bars['close'].ewm(span=10).mean().iloc[-1]
                    if (trend_direction > 0 and ema5 > ema10) or (trend_direction < 0 and ema5 < ema10):
                        entry_score += 1
            
            # Only execute trade if not skipped and passes entry filters
            if not should_skip and (len(entry_filters) == 0 or entry_score > 0):
                # Record that we made a trade at this bar
                contextual_strategy.last_trade_bar[symbol] = contextual_strategy.current_context.get('bars_since_regime_change', 0)
                
                # Execute contextual trade with dynamic parameters
                contextual_trade = contextual_simulator.execute_trade(
                    symbol=symbol,
                    entry_time=current_time,
                    direction=direction,
                    position_size=context_position['position_size'],
                    risk_amount=context_position['risk_amount'],
                    stop_loss_pips=20,
                    use_trailing_stop=context_strategy.get('use_trailing_stop', False),
                    tp_sl_ratio=context_strategy.get('tp_sl_ratio', 2.0),
                    strategy_id=context_strategy['id'],
                    strategy_name=context_strategy['name']
                )
                
                if contextual_trade:
                    # Update balance
                    contextual_strategy.update_balance(contextual_simulator.balance)
                    logger.debug(f"Contextual trade executed: {direction} {symbol} at {current_time} - Strategy: {context_strategy['name']}")
            else:
                if should_skip:
                    logger.debug(f"Skipped contextual trade: {skip_reason}")
                else:
                    logger.debug(f"Filtered contextual trade: Failed entry filters")
            
            # 2. Static strategy - keep this simple for comparison
            static_strategy_info = static_strategy.select_strategy(symbol, None)
            static_position = static_strategy.calculate_position_size(symbol, price, 20)
            
            # Execute static trade with fixed parameters
            static_trade = static_simulator.execute_trade(
                symbol=symbol,
                entry_time=current_time,
                direction='buy' if np.random.random() > 0.5 else 'sell',  # Random direction
                position_size=static_position['position_size'],
                risk_amount=static_position['risk_amount'],
                stop_loss_pips=20,
                take_profit_pips=40,
                strategy_id=static_strategy_info['id'],
                strategy_name=static_strategy_info['name']
            )
            
            if static_trade:
                # Update balance
                static_strategy.update_balance(static_simulator.balance)
            
            # Move to next bar
            current_idx += 1
    
    # Get results
    contextual_results = contextual_simulator.get_results()
    static_results = static_simulator.get_results()
    
    # Log results
    logger.info("\n=== STRATEGY COMPARISON RESULTS ===")
    logger.info("\nContextual Strategy:")
    logger.info(f"Initial Balance: ${contextual_results['initial_balance']:.2f}")
    logger.info(f"Final Balance: ${contextual_results['final_balance']:.2f}")
    logger.info(f"Total Return: {contextual_results['total_return']:.2f}%")
    logger.info(f"Win Rate: {contextual_results['win_rate']:.2f}%")
    logger.info(f"Profit Factor: {contextual_results['profit_factor']:.2f}")
    logger.info(f"Max Drawdown: {contextual_results['max_drawdown']:.2f}%")
    logger.info(f"Total Trades: {contextual_results['total_trades']}")
    
    logger.info("\nStatic Strategy:")
    logger.info(f"Initial Balance: ${static_results['initial_balance']:.2f}")
    logger.info(f"Final Balance: ${static_results['final_balance']:.2f}")
    logger.info(f"Total Return: {static_results['total_return']:.2f}%")
    logger.info(f"Win Rate: {static_results['win_rate']:.2f}%")
    logger.info(f"Profit Factor: {static_results['profit_factor']:.2f}")
    logger.info(f"Max Drawdown: {static_results['max_drawdown']:.2f}%")
    logger.info(f"Total Trades: {static_results['total_trades']}")
    
    # Performance comparison
    outperformance = contextual_results['total_return'] - static_results['total_return']
    logger.info(f"\nContextual Outperformance: {outperformance:.2f}%")
    
    # Risk table demonstration
    logger.info("\nProgressive Risk Scaling in Action:")
    contextual_trades = contextual_results['trades']
    if contextual_trades:
        # Group into balance ranges
        balance_ranges = [5000, 10000, 15000, 20000, 25000, 30000, 50000, 100000, float('inf')]
        range_labels = ["$0-$5K", "$5K-$10K", "$10K-$15K", "$15K-$20K", "$20K-$25K", 
                       "$25K-$30K", "$30K-$50K", "$50K-$100K", ">$100K"]
        
        risk_by_range = {label: [] for label in range_labels}
        
        for i, trade in enumerate(contextual_trades):
            balance_before = trade['balance_after'] - trade['pnl']
            risk_pct = trade['risk_percent']
            
            # Find the appropriate balance range
            for j, upper in enumerate(balance_ranges):
                if balance_before < upper:
                    risk_by_range[range_labels[j]].append(risk_pct)
                    break
        
        # Output average risk per range
        logger.info("Average Risk Percentage by Account Balance:")
        for label, risks in risk_by_range.items():
            if risks:
                avg_risk = sum(risks) / len(risks)
                logger.info(f"{label}: {avg_risk:.2f}% average risk ({len(risks)} trades)")
                
                if "25K-" in label:
                    logger.info(f"  ** PDT THRESHOLD CROSSED - Risk drops dramatically **")
    
    return {
        'contextual': contextual_results,
        'static': static_results
    }

if __name__ == "__main__":
    run_comparison_test()
