"""
Enhanced Trade Executor

This module provides comprehensive trade execution capabilities with position management,
risk control, and detailed trade tracking.
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger("enhanced_trade_executor")

# Import strategy classes
from trading_bot.strategies.trend_following import TrendFollowingStrategy
from trading_bot.strategies.momentum import MomentumStrategy
from trading_bot.strategies.mean_reversion import MeanReversionStrategy
from trading_bot.strategies.breakout_swing import BreakoutSwingStrategy
from trading_bot.strategies.volatility_breakout import VolatilityBreakoutStrategy
from trading_bot.strategies.option_spreads import OptionSpreadsStrategy

class EnhancedTradeExecutor:
    """
    Enhanced trade executor with comprehensive trade execution capabilities,
    position management, and detailed trade tracking.
    """
    
    def __init__(self, initial_capital=100000.0, commission_rate=0.001, slippage=0.001):
        """
        Initialize the enhanced trade executor.
        
        Args:
            initial_capital: Initial capital amount
            commission_rate: Commission rate as a percentage of trade value
            slippage: Slippage as a percentage of price
        """
        self.cash = initial_capital
        self.positions = {}
        self.trade_history = []
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.portfolio_values = []
        
        logger.info(f"Enhanced trade executor initialized with ${initial_capital:.2f}")
    
    def execute_signals(self, signals, market_data, portfolio_value):
        """
        Execute trades based on generated signals.
        
        Args:
            signals: List of trade signals
            market_data: Current market data
            portfolio_value: Current portfolio value
            
        Returns:
            List of executed trades
        """
        logger.info(f"Executing {len(signals)} signals")
        executed_trades = []
        
        for signal in signals:
            # Get strategy and position sizing information
            strategy_type = signal['signal_type']
            strategy_instance = self._get_strategy_instance(strategy_type)
            
            if not strategy_instance:
                logger.warning(f"No strategy instance found for {strategy_type}")
                continue
            
            # Calculate position size
            position_size = strategy_instance.calculate_position_size(signal, portfolio_value)
            
            # Check if position already exists for this symbol
            symbol = signal['symbol']
            direction = signal['direction']
            
            # Check if we're already in a position for this symbol
            if symbol in self.positions:
                existing_position = self.positions[symbol]
                
                # If same direction, skip or add to position
                if existing_position['direction'] == direction:
                    logger.info(f"Already in {direction} position for {symbol}, skipping")
                    continue
                
                # If opposite direction, close existing position first
                logger.info(f"Closing existing {existing_position['direction']} position for {symbol} before opening {direction} position")
                self._close_position(symbol, market_data)
            
            # Now execute the new position
            if direction in ['long', 'short']:
                trade = self._execute_equity_trade(signal, position_size, market_data)
            elif strategy_type == 'option_spreads':
                trade = self._execute_option_trade(signal, position_size, market_data)
            else:
                logger.warning(f"Unsupported direction: {direction}")
                continue
            
            if trade:
                executed_trades.append(trade)
                logger.info(f"Executed {direction} trade for {symbol}: {position_size} units at {signal['entry_price']}")
        
        # Update portfolio value
        self._update_portfolio_value(market_data)
        
        return executed_trades
    
    def manage_positions(self, market_data):
        """
        Manage existing positions (take profits, stop losses).
        
        Args:
            market_data: Current market data
            
        Returns:
            List of closed positions
        """
        logger.info("Managing existing positions")
        closed_positions = []
        
        for symbol, position in list(self.positions.items()):
            # Skip if symbol not in market data
            if symbol not in market_data:
                logger.warning(f"Symbol {symbol} not in market data, skipping position management")
                continue
            
            current_price = market_data[symbol]['close'][-1]
            
            # Check for stop loss
            if position['direction'] == 'long' and current_price <= position['stop_loss']:
                logger.info(f"Stop loss triggered for long position in {symbol}: {current_price} <= {position['stop_loss']}")
                trade = self._close_position(symbol, market_data, reason='stop_loss')
                if trade:
                    closed_positions.append(trade)
            
            elif position['direction'] == 'short' and current_price >= position['stop_loss']:
                logger.info(f"Stop loss triggered for short position in {symbol}: {current_price} >= {position['stop_loss']}")
                trade = self._close_position(symbol, market_data, reason='stop_loss')
                if trade:
                    closed_positions.append(trade)
            
            # Check for take profit
            elif position['direction'] == 'long' and current_price >= position['target']:
                logger.info(f"Take profit triggered for long position in {symbol}: {current_price} >= {position['target']}")
                trade = self._close_position(symbol, market_data, reason='take_profit')
                if trade:
                    closed_positions.append(trade)
            
            elif position['direction'] == 'short' and current_price <= position['target']:
                logger.info(f"Take profit triggered for short position in {symbol}: {current_price} <= {position['target']}")
                trade = self._close_position(symbol, market_data, reason='take_profit')
                if trade:
                    closed_positions.append(trade)
            
            # For option positions, check days to expiration
            elif position['strategy_type'] == 'option_spreads':
                # Close option positions at 50% profit target or 21 days to expiration
                if 'days_to_expiration' in position:
                    position['days_to_expiration'] -= 1
                    
                    if position['days_to_expiration'] <= 21:
                        logger.info(f"Closing option position in {symbol} due to approaching expiration: {position['days_to_expiration']} days left")
                        trade = self._close_position(symbol, market_data, reason='dte_close')
                        if trade:
                            closed_positions.append(trade)
                
                # Check for 50% profit target on credit spreads
                if position['strategy_type'] in ['put_credit_spread', 'call_credit_spread', 'iron_condor']:
                    current_value = self._calculate_current_spread_value(position, market_data)
                    initial_credit = position['credit_received']
                    
                    if current_value <= initial_credit * 0.5:
                        logger.info(f"Closing option position in {symbol} at 50% profit target: {current_value} <= {initial_credit * 0.5}")
                        trade = self._close_position(symbol, market_data, reason='profit_target')
                        if trade:
                            closed_positions.append(trade)
        
        # Update portfolio value
        self._update_portfolio_value(market_data)
        
        return closed_positions
    
    def _execute_equity_trade(self, signal, position_size, market_data):
        """
        Execute an equity trade (long or short).
        
        Args:
            signal: Trade signal
            position_size: Position size to trade
            market_data: Current market data
            
        Returns:
            Trade record
        """
        symbol = signal['symbol']
        direction = signal['direction']
        entry_price = signal['entry_price']
        
        # Apply slippage to entry price
        if direction == 'long':
            adjusted_entry = entry_price * (1 + self.slippage)
        else:  # short
            adjusted_entry = entry_price * (1 - self.slippage)
        
        # Calculate trade value and commission
        trade_value = position_size * adjusted_entry
        commission = trade_value * self.commission_rate
        
        # Check if we have enough cash for the trade
        if direction == 'long' and trade_value + commission > self.cash:
            # Adjust position size based on available cash
            original_size = position_size
            position_size = int((self.cash - commission) / adjusted_entry)
            
            logger.warning(f"Insufficient cash for full position. Adjusted size from {original_size} to {position_size}")
            
            if position_size <= 0:
                logger.error(f"Insufficient cash for trade: {self.cash} < {trade_value + commission}")
                return None
            
            # Recalculate trade values
            trade_value = position_size * adjusted_entry
            commission = trade_value * self.commission_rate
        
        # Update cash balance
        if direction == 'long':
            self.cash -= (trade_value + commission)
        else:  # short
            self.cash += (trade_value - commission)
        
        # Create position record
        position = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': adjusted_entry,
            'entry_date': market_data.get('date', datetime.now().strftime('%Y-%m-%d')),
            'size': position_size,
            'stop_loss': signal['stop_loss'],
            'target': signal['target'],
            'strategy_type': signal['signal_type'],
            'commission': commission,
            'trade_value': trade_value
        }
        
        # Save position
        self.positions[symbol] = position
        
        # Create trade record
        trade = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': adjusted_entry,
            'size': position_size,
            'date': market_data.get('date', datetime.now().strftime('%Y-%m-%d')),
            'strategy': signal['signal_type'],
            'confidence': signal.get('confidence', 0.5),
            'commission': commission,
            'trade_value': trade_value
        }
        
        # Save to trade history
        self.trade_history.append(trade)
        
        return trade
    
    def _execute_option_trade(self, signal, position_size, market_data):
        """
        Execute an option trade (spreads, strangles).
        
        Args:
            signal: Trade signal
            position_size: Position size to trade
            market_data: Current market data
            
        Returns:
            Trade record
        """
        symbol = signal['symbol']
        strategy_type = signal.get('strategy_type')
        
        # Different handling based on option strategy
        if strategy_type in ['put_credit_spread', 'call_credit_spread']:
            # Credit received per spread
            credit_received = signal['credit_received'] * 100  # Convert to dollars
            
            # Total premium received
            total_credit = credit_received * position_size
            
            # Commission (higher for options)
            commission = total_credit * self.commission_rate * 2  # 2 legs
            
            # Max risk calculation
            max_risk = signal['max_risk'] * 100 * position_size  # Convert to dollars
            
            # Update cash balance (add credit received)
            self.cash += (total_credit - commission)
            
            # Create position record
            position = {
                'symbol': symbol,
                'direction': signal['direction'],
                'strategy_type': strategy_type,
                'entry_date': market_data.get('date', datetime.now().strftime('%Y-%m-%d')),
                'short_strike': signal['short_strike'],
                'long_strike': signal['long_strike'],
                'size': position_size,
                'credit_received': signal['credit_received'],
                'max_risk': signal['max_risk'],
                'days_to_expiration': signal['days_to_expiration'],
                'days_to_expiration_initial': signal['days_to_expiration'],  # Store initial DTE
                'probability_otm': signal.get('probability_otm', 0.5),
                'commission': commission
            }
            
        elif strategy_type == 'iron_condor':
            # Total credit from both spreads
            credit_received = signal['credit_received'] * 100  # Convert to dollars
            
            # Total premium received
            total_credit = credit_received * position_size
            
            # Commission (higher for options)
            commission = total_credit * self.commission_rate * 4  # 4 legs
            
            # Max risk calculation
            max_risk = signal['max_risk'] * 100 * position_size  # Convert to dollars
            
            # Update cash balance (add credit received)
            self.cash += (total_credit - commission)
            
            # Create position record
            position = {
                'symbol': symbol,
                'direction': 'neutral',
                'strategy_type': 'iron_condor',
                'entry_date': market_data.get('date', datetime.now().strftime('%Y-%m-%d')),
                'call_short_strike': signal['call_short_strike'],
                'call_long_strike': signal['call_long_strike'],
                'put_short_strike': signal['put_short_strike'],
                'put_long_strike': signal['put_long_strike'],
                'size': position_size,
                'credit_received': signal['credit_received'],
                'max_risk': signal['max_risk'],
                'days_to_expiration': signal['days_to_expiration'],
                'days_to_expiration_initial': signal['days_to_expiration'],  # Store initial DTE
                'call_probability_otm': signal.get('call_probability_otm', 0.5),
                'put_probability_otm': signal.get('put_probability_otm', 0.5),
                'commission': commission
            }
            
        elif strategy_type == 'long_strangle':
            # Total cost for the strangle
            total_cost = signal['total_cost'] * 100 * position_size  # Convert to dollars
            
            # Commission (higher for options)
            commission = total_cost * self.commission_rate * 2  # 2 legs
            
            # Check if we have enough cash
            if total_cost + commission > self.cash:
                # Adjust position size based on available cash
                original_size = position_size
                position_size = max(1, int((self.cash - commission) / (signal['total_cost'] * 100)))
                
                logger.warning(f"Insufficient cash for full position. Adjusted size from {original_size} to {position_size}")
                
                if position_size <= 0:
                    logger.error(f"Insufficient cash for trade: {self.cash} < {total_cost + commission}")
                    return None
                
                # Recalculate costs
                total_cost = signal['total_cost'] * 100 * position_size
                commission = total_cost * self.commission_rate * 2
            
            # Update cash balance (subtract cost)
            self.cash -= (total_cost + commission)
            
            # Create position record
            position = {
                'symbol': symbol,
                'direction': 'volatile',
                'strategy_type': 'long_strangle',
                'entry_date': market_data.get('date', datetime.now().strftime('%Y-%m-%d')),
                'call_strike': signal['call_strike'],
                'put_strike': signal['put_strike'],
                'size': position_size,
                'total_cost': signal['total_cost'],
                'max_risk': signal['total_cost'] * 100 * position_size,  # Limited to premium paid
                'days_to_expiration': signal['days_to_expiration'],
                'days_to_expiration_initial': signal['days_to_expiration'],  # Store initial DTE
                'commission': commission
            }
        
        else:
            logger.error(f"Unsupported option strategy type: {strategy_type}")
            return None
        
        # Save position
        self.positions[symbol] = position
        
        # Create trade record
        trade = {
            'symbol': symbol,
            'direction': signal['direction'],
            'strategy_type': strategy_type,
            'size': position_size,
            'date': market_data.get('date', datetime.now().strftime('%Y-%m-%d')),
            'strategy': 'option_spreads',
            'confidence': signal.get('confidence', 0.5),
            'commission': commission
        }
        
        # Add strategy-specific details
        for key in position:
            if key not in trade:
                trade[key] = position[key]
        
        # Save to trade history
        self.trade_history.append(trade)
        
        return trade
    
    def _close_position(self, symbol, market_data, reason='manual'):
        """
        Close an existing position.
        
        Args:
            symbol: Symbol to close
            market_data: Current market data
            reason: Reason for closing
            
        Returns:
            Trade record
        """
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return None
        
        position = self.positions[symbol]
        adjusted_exit = None
        pnl = 0
        
        # Handle based on position type
        if position['strategy_type'] in ['trend_following', 'momentum', 'mean_reversion', 'breakout_swing', 'volatility_breakout']:
            # Equity position
            current_price = market_data[symbol]['close'][-1]
            
            # Apply slippage to exit price
            if position['direction'] == 'long':
                adjusted_exit = current_price * (1 - self.slippage)
            else:  # short
                adjusted_exit = current_price * (1 + self.slippage)
            
            # Calculate trade value and commission
            trade_value = position['size'] * adjusted_exit
            commission = trade_value * self.commission_rate
            
            # Calculate profit/loss
            if position['direction'] == 'long':
                pnl = (adjusted_exit - position['entry_price']) * position['size'] - commission - position['commission']
            else:  # short
                pnl = (position['entry_price'] - adjusted_exit) * position['size'] - commission - position['commission']
            
            # Update cash balance
            if position['direction'] == 'long':
                self.cash += (trade_value - commission)
            else:  # short
                self.cash -= (trade_value + commission)
            
        elif position['strategy_type'] in ['put_credit_spread', 'call_credit_spread', 'iron_condor', 'long_strangle']:
            # Option position
            current_value = self._calculate_current_spread_value(position, market_data)
            
            if position['strategy_type'] in ['put_credit_spread', 'call_credit_spread', 'iron_condor']:
                # For credit spreads, PnL = Initial credit - Current value - Commissions
                initial_credit = position['credit_received'] * 100 * position['size']
                exit_cost = current_value * 100 * position['size']
                commission = exit_cost * self.commission_rate * (2 if position['strategy_type'] != 'iron_condor' else 4)
                
                pnl = initial_credit - exit_cost - commission - position['commission']
                
                # Update cash balance (subtract cost to close)
                self.cash -= (exit_cost + commission)
                
            else:  # long_strangle
                # For long options, PnL = Current value - Initial cost - Commissions
                initial_cost = position['total_cost'] * 100 * position['size']
                exit_value = current_value * 100 * position['size']
                commission = exit_value * self.commission_rate * 2
                
                pnl = exit_value - initial_cost - commission - position['commission']
                
                # Update cash balance (add exit value)
                self.cash += (exit_value - commission)
        else:
            logger.error(f"Unsupported strategy type for closing: {position['strategy_type']}")
            return None
        
        # Create close trade record
        trade = {
            'symbol': symbol,
            'direction': 'close_' + position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': adjusted_exit,
            'size': position['size'],
            'entry_date': position['entry_date'],
            'exit_date': market_data.get('date', datetime.now().strftime('%Y-%m-%d')),
            'strategy': position['strategy_type'],
            'pnl': pnl,
            'pnl_percent': (pnl / position['trade_value'] * 100) if 'trade_value' in position else None,
            'reason': reason,
            'commission': commission if 'commission' in locals() else 0
        }
        
        # Log the trade
        if pnl > 0:
            logger.info(f"Closed {position['direction']} position in {symbol} with profit: ${pnl:.2f} ({trade.get('pnl_percent', 0):.2f}%)")
        else:
            logger.info(f"Closed {position['direction']} position in {symbol} with loss: ${pnl:.2f} ({trade.get('pnl_percent', 0):.2f}%)")
        
        # Remove the position
        del self.positions[symbol]
        
        # Save to trade history
        self.trade_history.append(trade)
        
        return trade
    
    def _update_portfolio_value(self, market_data):
        """
        Calculate and update current portfolio value.
        
        Args:
            market_data: Current market data
            
        Returns:
            Current portfolio value
        """
        # Start with cash
        portfolio_value = self.cash
        
        # Add value of open positions
        for symbol, position in self.positions.items():
            if symbol not in market_data:
                logger.warning(f"Symbol {symbol} not in market data for portfolio valuation")
                continue
            
            if position['strategy_type'] in ['trend_following', 'momentum', 'mean_reversion', 'breakout_swing', 'volatility_breakout']:
                # Equity position
                current_price = market_data[symbol]['close'][-1]
                position_value = position['size'] * current_price
                
                # Add value based on direction
                if position['direction'] == 'long':
                    portfolio_value += position_value
                else:  # short
                    portfolio_value += (2 * position['trade_value'] - position_value)
                
            elif position['strategy_type'] in ['put_credit_spread', 'call_credit_spread', 'iron_condor', 'long_strangle']:
                # Option position - add current value estimation
                if position['strategy_type'] in ['put_credit_spread', 'call_credit_spread', 'iron_condor']:
                    # For credit spreads, add (initial credit - current value)
                    initial_credit = position['credit_received'] * 100 * position['size']
                    current_value = self._calculate_current_spread_value(position, market_data) * 100 * position['size']
                    portfolio_value += (initial_credit - current_value)
                
                else:  # long_strangle
                    # For long options, add current value
                    current_value = self._calculate_current_spread_value(position, market_data) * 100 * position['size']
                    portfolio_value += current_value
        
        # Record portfolio value
        self.portfolio_values.append({
            'date': market_data.get('date', datetime.now().strftime('%Y-%m-%d')),
            'value': portfolio_value
        })
        
        return portfolio_value
    
    def _calculate_current_spread_value(self, position, market_data):
        """
        Estimate the current value of an option spread.
        
        Args:
            position: Option position
            market_data: Current market data
            
        Returns:
            Current value of the option spread
        """
        # This would connect to options pricing model or market data
        # For simplicity, we'll use a time decay approximation
        
        strategy_type = position['strategy_type']
        
        if strategy_type in ['put_credit_spread', 'call_credit_spread']:
            # Simple time decay approximation
            initial_credit = position['credit_received']
            days_passed = position.get('days_to_expiration_initial', position['days_to_expiration']) - position['days_to_expiration']
            total_days = position.get('days_to_expiration_initial', position['days_to_expiration'])
            
            # Value decays faster as expiration approaches (simplified theta curve)
            if days_passed / total_days < 0.5:
                decay_factor = days_passed / total_days * 0.6  # Slower decay initially
            else:
                decay_factor = 0.3 + (days_passed / total_days - 0.5) * 1.4  # Faster decay near expiration
            
            current_value = initial_credit * (1 - decay_factor)
            
            # Adjust based on price movement
            underlying_price = market_data[position['symbol']]['close'][-1]
            
            if strategy_type == 'put_credit_spread' and underlying_price < position['short_strike']:
                # Put spread losing value as price drops below short strike
                intrinsic_value = max(0, position['short_strike'] - underlying_price)
                current_value = min(position['short_strike'] - position['long_strike'], intrinsic_value)
            
            elif strategy_type == 'call_credit_spread' and underlying_price > position['short_strike']:
                # Call spread losing value as price rises above short strike
                intrinsic_value = max(0, underlying_price - position['short_strike'])
                current_value = min(position['long_strike'] - position['short_strike'], intrinsic_value)
                
            return max(0, current_value)
            
        elif strategy_type == 'iron_condor':
            # Combine put and call spread calculations
            initial_credit = position['credit_received']
            days_passed = position.get('days_to_expiration_initial', position['days_to_expiration']) - position['days_to_expiration']
            total_days = position.get('days_to_expiration_initial', position['days_to_expiration'])
            
            # Value decays faster as expiration approaches
            if days_passed / total_days < 0.5:
                decay_factor = days_passed / total_days * 0.6
            else:
                decay_factor = 0.3 + (days_passed / total_days - 0.5) * 1.4
            
            current_value = initial_credit * (1 - decay_factor)
            
            # Adjust based on price movement
            underlying_price = market_data[position['symbol']]['close'][-1]
            
            if underlying_price < position['put_short_strike']:
                # Put side losing value
                intrinsic_value = max(0, position['put_short_strike'] - underlying_price)
                put_value = min(position['put_short_strike'] - position['put_long_strike'], intrinsic_value)
                current_value = put_value
            
            elif underlying_price > position['call_short_strike']:
                # Call side losing value
                intrinsic_value = max(0, underlying_price - position['call_short_strike'])
                call_value = min(position['call_long_strike'] - position['call_short_strike'], intrinsic_value)
                current_value = call_value
                
            return max(0, current_value)
            
        elif strategy_type == 'long_strangle':
            # For long strangle, value increases with volatility and directional movement
            initial_cost = position['total_cost']
            days_passed = position.get('days_to_expiration_initial', position['days_to_expiration']) - position['days_to_expiration']
            total_days = position.get('days_to_expiration_initial', position['days_to_expiration'])
            
            # Time decay
            time_decay_factor = (1 - days_passed / total_days) ** 0.7  # Less aggressive decay for long options
            
            # Start with time-decayed value
            current_value = initial_cost * time_decay_factor
            
            # Adjust based on price movement
            underlying_price = market_data[position['symbol']]['close'][-1]
            entry_price = market_data[position['symbol']].get('entry_price', underlying_price)
            
            # Price movement factors
            call_value = max(0, underlying_price - position['call_strike'])
            put_value = max(0, position['put_strike'] - underlying_price)
            
            # Add intrinsic value
            current_value = max(current_value, call_value + put_value)
            
            return current_value
        
        # Default
        return 0
    
    def _get_strategy_instance(self, strategy_type):
        """
        Return the appropriate strategy instance based on strategy type.
        
        Args:
            strategy_type: Strategy type
            
        Returns:
            Strategy instance
        """
        if strategy_type == 'trend_following':
            return TrendFollowingStrategy()
        elif strategy_type == 'momentum':
            return MomentumStrategy()
        elif strategy_type == 'mean_reversion':
            return MeanReversionStrategy()
        elif strategy_type == 'breakout_swing':
            return BreakoutSwingStrategy()
        elif strategy_type == 'volatility_breakout':
            return VolatilityBreakoutStrategy()
        elif strategy_type == 'option_spreads':
            return OptionSpreadsStrategy()
        
        logger.error(f"Unknown strategy type: {strategy_type}")
        return None

def calculate_position_sizes(signals, portfolio_value, strategy_allocations):
    """
    Calculate position sizes for a group of signals based on strategy allocations and risk parameters.
    
    Args:
        signals: List of trade signals from different strategies
        portfolio_value: Current portfolio value
        strategy_allocations: Dictionary of strategy allocations (percentages)
        
    Returns:
        Dictionary mapping signal IDs to position sizes
    """
    position_sizes = {}
    
    # Group signals by strategy
    strategy_signals = {}
    for signal in signals:
        strategy = signal['signal_type']
        if strategy not in strategy_signals:
            strategy_signals[strategy] = []
        strategy_signals[strategy].append(signal)
    
    # Process each strategy
    for strategy, signals_list in strategy_signals.items():
        # Skip if no allocation for this strategy
        if strategy not in strategy_allocations:
            logger.warning(f"No allocation for strategy {strategy}")
            continue
        
        # Calculate capital allocated to this strategy
        strategy_capital = portfolio_value * (strategy_allocations[strategy] / 100.0)
        
        # Get strategy instance
        strategy_instance = None
        if strategy == 'trend_following':
            strategy_instance = TrendFollowingStrategy()
        elif strategy == 'momentum':
            strategy_instance = MomentumStrategy()
        elif strategy == 'mean_reversion':
            strategy_instance = MeanReversionStrategy()
        elif strategy == 'breakout_swing':
            strategy_instance = BreakoutSwingStrategy()
        elif strategy == 'volatility_breakout':
            strategy_instance = VolatilityBreakoutStrategy()
        elif strategy == 'option_spreads':
            strategy_instance = OptionSpreadsStrategy()
            
        if not strategy_instance:
            logger.error(f"Unknown strategy: {strategy}")
            continue
        
        # If strategy has multiple signals, divide capital
        capital_per_signal = strategy_capital / len(signals_list)
        
        # Calculate position size for each signal
        for signal in signals_list:
            # Get position size using strategy's position sizing logic
            position_size = strategy_instance.calculate_position_size(signal, capital_per_signal)
            
            # Store position size
            signal_id = f"{signal['symbol']}_{signal['direction']}_{signal.get('signal_id', id(signal))}"
            position_sizes[signal_id] = position_size
    
    return position_sizes 