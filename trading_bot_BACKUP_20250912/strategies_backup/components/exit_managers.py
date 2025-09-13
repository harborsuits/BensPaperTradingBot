"""
Exit Manager Components

Implementation of various exit manager components for the modular strategy system.
These components determine when and how to exit positions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta

from trading_bot.strategies.base_strategy import SignalType
from trading_bot.strategies.modular_strategy_system import (
    StrategyComponent, ComponentType, ExitManagerComponent
)

logger = logging.getLogger(__name__)

class TrailingStopExitManager(ExitManagerComponent):
    """Manages exits using trailing stops."""
    
    def __init__(self, 
                component_id: Optional[str] = None,
                atr_period: int = 14,
                initial_stop_multiplier: float = 3.0,
                trailing_stop_multiplier: float = 2.0,
                breakeven_threshold: float = 1.0,
                max_days_in_trade: int = 20):
        """
        Initialize trailing stop exit manager
        
        Args:
            component_id: Unique component ID
            atr_period: ATR calculation period
            initial_stop_multiplier: Initial stop distance as ATR multiple
            trailing_stop_multiplier: Trailing stop distance as ATR multiple
            breakeven_threshold: Profit threshold to move stop to breakeven (in ATR units)
            max_days_in_trade: Maximum days to hold a position
        """
        super().__init__(component_id)
        self.parameters = {
            'atr_period': atr_period,
            'initial_stop_multiplier': initial_stop_multiplier,
            'trailing_stop_multiplier': trailing_stop_multiplier,
            'breakeven_threshold': breakeven_threshold,
            'max_days_in_trade': max_days_in_trade
        }
        self.description = f"Trailing Stop ({trailing_stop_multiplier}x ATR)"
        self.position_stops = {}
    
    def calculate_exits(self, positions: Dict[str, Dict[str, Any]], 
                      data: Dict[str, pd.DataFrame], 
                      context: Dict[str, Any]) -> Dict[str, bool]:
        """
        Calculate exit signals for open positions
        
        Args:
            positions: Dictionary of open positions
            data: Market data
            context: Processing context
            
        Returns:
            Dictionary of symbol -> exit flag
        """
        exits = {}
        
        # Get parameters
        atr_period = self.parameters['atr_period']
        initial_mult = self.parameters['initial_stop_multiplier']
        trailing_mult = self.parameters['trailing_stop_multiplier']
        breakeven_threshold = self.parameters['breakeven_threshold']
        max_days = self.parameters['max_days_in_trade']
        
        # Current time
        current_time = context.get('current_time', datetime.now())
        
        for symbol, position in positions.items():
            # Skip if no data available
            if symbol not in data:
                exits[symbol] = False
                continue
            
            df = data[symbol]
            if len(df) < atr_period + 1:
                exits[symbol] = False
                continue
            
            # Calculate ATR if needed
            if 'atr' not in df.columns:
                # Calculate True Range
                high_low = df['high'] - df['low']
                high_close_prev = abs(df['high'] - df['close'].shift(1))
                low_close_prev = abs(df['low'] - df['close'].shift(1))
                tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
                
                # Calculate ATR
                df['atr'] = tr.rolling(window=atr_period).mean()
            
            # Get latest price and ATR
            current_price = df['close'].iloc[-1]
            current_atr = df['atr'].iloc[-1]
            
            if pd.isna(current_atr) or current_atr <= 0:
                exits[symbol] = False
                continue
            
            # Get position details
            entry_price = position.get('entry_price', current_price)
            position_type = position.get('position_type', 'long')
            entry_time = position.get('entry_time', current_time)
            
            # Check if position is new
            is_new_position = symbol not in self.position_stops
            
            # Initialize stop for new positions
            if is_new_position:
                if position_type.lower() == 'long':
                    stop_price = entry_price - (current_atr * initial_mult)
                else:  # Short position
                    stop_price = entry_price + (current_atr * initial_mult)
                
                self.position_stops[symbol] = {
                    'stop_price': stop_price,
                    'highest_profit': 0.0,
                    'entry_price': entry_price
                }
            
            # Get current stop
            stop_info = self.position_stops[symbol]
            stop_price = stop_info['stop_price']
            highest_profit = stop_info['highest_profit']
            
            # Calculate current profit (in ATR units)
            if position_type.lower() == 'long':
                current_profit = (current_price - entry_price) / current_atr
                price_better_than_entry = current_price > entry_price
            else:  # Short position
                current_profit = (entry_price - current_price) / current_atr
                price_better_than_entry = current_price < entry_price
            
            # Update highest profit
            if current_profit > highest_profit:
                highest_profit = current_profit
                stop_info['highest_profit'] = highest_profit
                
                # Update trailing stop
                if position_type.lower() == 'long':
                    new_stop = current_price - (current_atr * trailing_mult)
                    # Only move stop up, never down
                    if new_stop > stop_price:
                        stop_price = new_stop
                else:  # Short position
                    new_stop = current_price + (current_atr * trailing_mult)
                    # Only move stop down, never up
                    if new_stop < stop_price:
                        stop_price = new_stop
                
                stop_info['stop_price'] = stop_price
            
            # Move stop to breakeven if profit exceeds threshold
            if (price_better_than_entry and 
                current_profit >= breakeven_threshold and 
                ((position_type.lower() == 'long' and stop_price < entry_price) or
                 (position_type.lower() == 'short' and stop_price > entry_price))):
                
                stop_price = entry_price
                stop_info['stop_price'] = stop_price
            
            # Check if stop is hit
            if position_type.lower() == 'long':
                stop_hit = current_price <= stop_price
            else:  # Short position
                stop_hit = current_price >= stop_price
            
            # Check if max days exceeded
            days_in_trade = (current_time - entry_time).days
            time_exit = days_in_trade >= max_days
            
            # Exit if stop hit or time expired
            exits[symbol] = stop_hit or time_exit
            
            # If exiting, clean up
            if exits[symbol]:
                if symbol in self.position_stops:
                    del self.position_stops[symbol]
        
        return exits

class TakeProfitExitManager(ExitManagerComponent):
    """Manages exits using take profit levels."""
    
    def __init__(self, 
                component_id: Optional[str] = None,
                profit_targets: List[float] = [1.0, 2.0, 3.0],
                exit_portions: List[float] = [0.33, 0.33, 0.34],
                atr_period: int = 14,
                use_atr_scaling: bool = True):
        """
        Initialize take profit exit manager
        
        Args:
            component_id: Unique component ID
            profit_targets: List of profit targets
            exit_portions: List of position portions to exit at each target
            atr_period: ATR calculation period
            use_atr_scaling: Scale profit targets by ATR if True
        """
        super().__init__(component_id)
        assert len(profit_targets) == len(exit_portions), "Profit targets and exit portions must have the same length"
        assert sum(exit_portions) <= 1.0, "Sum of exit portions cannot exceed 1.0"
        
        self.parameters = {
            'profit_targets': profit_targets,
            'exit_portions': exit_portions,
            'atr_period': atr_period,
            'use_atr_scaling': use_atr_scaling
        }
        targets_str = ', '.join([f"{t:.1f}%" for t in profit_targets])
        self.description = f"Take Profit ({targets_str})"
        
        # Track partial exits
        self.partial_exits = {}
    
    def calculate_exits(self, positions: Dict[str, Dict[str, Any]], 
                      data: Dict[str, pd.DataFrame], 
                      context: Dict[str, Any]) -> Dict[str, Union[bool, float]]:
        """
        Calculate exit signals for open positions
        
        Args:
            positions: Dictionary of open positions
            data: Market data
            context: Processing context
            
        Returns:
            Dictionary of symbol -> exit flag or portion
        """
        exits = {}
        
        # Get parameters
        profit_targets = self.parameters['profit_targets']
        exit_portions = self.parameters['exit_portions']
        atr_period = self.parameters['atr_period']
        use_atr = self.parameters['use_atr_scaling']
        
        # Get current time
        current_time = context.get('current_time', datetime.now())
        
        for symbol, position in positions.items():
            # Skip if no data available
            if symbol not in data:
                exits[symbol] = False
                continue
            
            df = data[symbol]
            if len(df) < atr_period + 1 and use_atr:
                exits[symbol] = False
                continue
            
            # Get position details
            entry_price = position.get('entry_price')
            if entry_price is None:
                exits[symbol] = False
                continue
            
            position_type = position.get('position_type', 'long')
            current_price = df['close'].iloc[-1]
            
            # Initialize partial exit tracking if needed
            if symbol not in self.partial_exits:
                self.partial_exits[symbol] = {
                    'targets_hit': [False] * len(profit_targets),
                    'total_exited': 0.0
                }
            
            partial_info = self.partial_exits[symbol]
            
            # Calculate current profit percentage
            if position_type.lower() == 'long':
                profit_pct = (current_price / entry_price - 1) * 100
            else:  # Short position
                profit_pct = (entry_price / current_price - 1) * 100
            
            # Scale by ATR if enabled
            if use_atr:
                if 'atr' not in df.columns:
                    # Calculate True Range
                    high_low = df['high'] - df['low']
                    high_close_prev = abs(df['high'] - df['close'].shift(1))
                    low_close_prev = abs(df['low'] - df['close'].shift(1))
                    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
                    
                    # Calculate ATR
                    df['atr'] = tr.rolling(window=atr_period).mean()
                
                # Get latest ATR as percentage of price
                current_atr = df['atr'].iloc[-1]
                atr_pct = (current_atr / current_price) * 100
                
                # Scale profit target by ATR
                scaled_targets = [target * atr_pct for target in profit_targets]
            else:
                scaled_targets = profit_targets
            
            # Check for new targets hit
            exit_portion = 0.0
            for i, (target, portion) in enumerate(zip(scaled_targets, exit_portions)):
                # Skip already hit targets
                if partial_info['targets_hit'][i]:
                    continue
                
                # Check if target is hit
                if profit_pct >= target:
                    # Mark target as hit
                    partial_info['targets_hit'][i] = True
                    # Add portion to exit
                    exit_portion += portion
            
            # Calculate total exit amount
            if exit_portion > 0:
                # Update total exited
                total_exited = partial_info['total_exited'] + exit_portion
                
                # Check if this would exceed 100%
                if total_exited >= 0.999:  # Allow for floating point errors
                    # Full exit
                    exits[symbol] = True
                    
                    # Clean up
                    if symbol in self.partial_exits:
                        del self.partial_exits[symbol]
                else:
                    # Partial exit
                    exits[symbol] = exit_portion
                    partial_info['total_exited'] = total_exited
            else:
                # No exit
                exits[symbol] = False
        
        return exits

class TimeBasedExitManager(ExitManagerComponent):
    """Manages exits based on time in trade and time of day."""
    
    def __init__(self, 
                component_id: Optional[str] = None,
                max_days_in_trade: int = 10,
                max_hours_in_trade: int = 0,
                market_close_exit: bool = True,
                market_close_time: str = "15:45",
                weekend_exit: bool = True):
        """
        Initialize time-based exit manager
        
        Args:
            component_id: Unique component ID
            max_days_in_trade: Maximum days to hold a position
            max_hours_in_trade: Maximum hours to hold an intraday position
            market_close_exit: Exit positions before market close
            market_close_time: Market close time (HH:MM)
            weekend_exit: Exit positions before weekend
        """
        super().__init__(component_id)
        self.parameters = {
            'max_days_in_trade': max_days_in_trade,
            'max_hours_in_trade': max_hours_in_trade,
            'market_close_exit': market_close_exit,
            'market_close_time': market_close_time,
            'weekend_exit': weekend_exit
        }
        self.description = f"Time-Based Exit (Max {max_days_in_trade}d)"
    
    def calculate_exits(self, positions: Dict[str, Dict[str, Any]], 
                      data: Dict[str, pd.DataFrame], 
                      context: Dict[str, Any]) -> Dict[str, bool]:
        """
        Calculate exit signals for open positions
        
        Args:
            positions: Dictionary of open positions
            data: Market data
            context: Processing context
            
        Returns:
            Dictionary of symbol -> exit flag
        """
        exits = {}
        
        # Get parameters
        max_days = self.parameters['max_days_in_trade']
        max_hours = self.parameters['max_hours_in_trade']
        close_exit = self.parameters['market_close_exit']
        close_time_str = self.parameters['market_close_time']
        weekend_exit = self.parameters['weekend_exit']
        
        # Parse market close time
        close_hour, close_min = map(int, close_time_str.split(':'))
        
        # Get current time
        current_time = context.get('current_time', datetime.now())
        
        # Check if today is Friday (4) and close to weekend
        is_friday = current_time.weekday() == 4
        
        # Check if close to market close
        current_time_obj = current_time.time()
        market_close_time = datetime.now().replace(
            hour=close_hour, minute=close_min, second=0, microsecond=0
        ).time()
        
        near_market_close = close_exit and current_time_obj >= market_close_time
        near_weekend = weekend_exit and is_friday and current_time_obj >= market_close_time
        
        for symbol, position in positions.items():
            # Get position entry time
            entry_time = position.get('entry_time')
            if entry_time is None:
                exits[symbol] = False
                continue
            
            # Calculate time in trade
            time_delta = current_time - entry_time
            days_in_trade = time_delta.days
            hours_in_trade = time_delta.total_seconds() / 3600
            
            # Check exit conditions
            max_days_exit = max_days > 0 and days_in_trade >= max_days
            max_hours_exit = max_hours > 0 and hours_in_trade >= max_hours
            
            # Determine if should exit
            exits[symbol] = (
                max_days_exit or max_hours_exit or near_market_close or near_weekend
            )
        
        return exits

class TechnicalExitManager(ExitManagerComponent):
    """Manages exits based on technical indicators."""
    
    def __init__(self, 
                component_id: Optional[str] = None,
                indicators: Dict[str, Dict[str, Any]] = None,
                exit_on_reversal: bool = True,
                use_confirmation: bool = True):
        """
        Initialize technical indicator-based exit manager
        
        Args:
            component_id: Unique component ID
            indicators: Dictionary of indicators to use
            exit_on_reversal: Exit on signal reversal
            use_confirmation: Require multiple indicators to confirm
        """
        super().__init__(component_id)
        self.parameters = {
            'indicators': indicators or {
                'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'bbands': {'period': 20, 'std_dev': 2.0}
            },
            'exit_on_reversal': exit_on_reversal,
            'use_confirmation': use_confirmation
        }
        
        indicators_str = ', '.join(indicators.keys() if indicators else self.parameters['indicators'].keys())
        self.description = f"Technical Exit ({indicators_str})"
    
    def calculate_exits(self, positions: Dict[str, Dict[str, Any]], 
                      data: Dict[str, pd.DataFrame], 
                      context: Dict[str, Any]) -> Dict[str, bool]:
        """
        Calculate exit signals for open positions
        
        Args:
            positions: Dictionary of open positions
            data: Market data
            context: Processing context
            
        Returns:
            Dictionary of symbol -> exit flag
        """
        exits = {}
        
        # Get parameters
        indicators = self.parameters['indicators']
        exit_on_reversal = self.parameters['exit_on_reversal']
        use_confirmation = self.parameters['use_confirmation']
        
        for symbol, position in positions.items():
            # Skip if no data available
            if symbol not in data:
                exits[symbol] = False
                continue
            
            df = data[symbol]
            if len(df) < 30:  # Reasonable minimum for most indicators
                exits[symbol] = False
                continue
            
            # Get position type
            position_type = position.get('position_type', 'long')
            is_long = position_type.lower() == 'long'
            
            # Track indicator signals
            exit_signals = []
            
            # Process each indicator
            for ind_name, ind_params in indicators.items():
                if ind_name == 'rsi':
                    # Calculate RSI if needed
                    period = ind_params.get('period', 14)
                    overbought = ind_params.get('overbought', 70)
                    oversold = ind_params.get('oversold', 30)
                    
                    if 'rsi' not in df.columns:
                        # Calculate price changes
                        delta = df['close'].diff()
                        
                        # Separate gains and losses
                        gain = delta.copy()
                        loss = delta.copy()
                        gain[gain < 0] = 0
                        loss[loss > 0] = 0
                        loss = -loss  # Make loss positive
                        
                        # Calculate average gain and loss
                        avg_gain = gain.rolling(window=period).mean()
                        avg_loss = loss.rolling(window=period).mean()
                        
                        # Calculate RS and RSI
                        rs = avg_gain / avg_loss
                        df['rsi'] = 100 - (100 / (1 + rs))
                    
                    # Get last two RSI values
                    last_rsi = df['rsi'].iloc[-1]
                    prev_rsi = df['rsi'].iloc[-2]
                    
                    # Check for exit signals
                    if is_long and (last_rsi > overbought or (exit_on_reversal and last_rsi < prev_rsi and prev_rsi > overbought)):
                        exit_signals.append(True)
                    elif not is_long and (last_rsi < oversold or (exit_on_reversal and last_rsi > prev_rsi and prev_rsi < oversold)):
                        exit_signals.append(True)
                    else:
                        exit_signals.append(False)
                
                elif ind_name == 'macd':
                    # Calculate MACD if needed
                    fast = ind_params.get('fast', 12)
                    slow = ind_params.get('slow', 26)
                    signal_period = ind_params.get('signal', 9)
                    
                    if 'macd' not in df.columns:
                        # Calculate EMAs
                        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
                        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
                        
                        # Calculate MACD line
                        df['macd'] = ema_fast - ema_slow
                        
                        # Calculate signal line
                        df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
                        
                        # Calculate histogram
                        df['macd_hist'] = df['macd'] - df['macd_signal']
                    
                    # Get last two values
                    last_macd = df['macd'].iloc[-1]
                    last_signal = df['macd_signal'].iloc[-1]
                    last_hist = df['macd_hist'].iloc[-1]
                    
                    prev_macd = df['macd'].iloc[-2]
                    prev_signal = df['macd_signal'].iloc[-2]
                    prev_hist = df['macd_hist'].iloc[-2]
                    
                    # Check for exit signals
                    if is_long and ((last_macd < last_signal and prev_macd > prev_signal) or (exit_on_reversal and last_hist < 0 and prev_hist > 0)):
                        exit_signals.append(True)
                    elif not is_long and ((last_macd > last_signal and prev_macd < prev_signal) or (exit_on_reversal and last_hist > 0 and prev_hist < 0)):
                        exit_signals.append(True)
                    else:
                        exit_signals.append(False)
                
                elif ind_name == 'bbands':
                    # Calculate Bollinger Bands if needed
                    period = ind_params.get('period', 20)
                    std_dev = ind_params.get('std_dev', 2.0)
                    
                    if 'bb_middle' not in df.columns:
                        # Calculate middle band (SMA)
                        df['bb_middle'] = df['close'].rolling(window=period).mean()
                        
                        # Calculate standard deviation
                        std = df['close'].rolling(window=period).std()
                        
                        # Calculate upper and lower bands
                        df['bb_upper'] = df['bb_middle'] + (std * std_dev)
                        df['bb_lower'] = df['bb_middle'] - (std * std_dev)
                    
                    # Get values
                    last_price = df['close'].iloc[-1]
                    last_upper = df['bb_upper'].iloc[-1]
                    last_lower = df['bb_lower'].iloc[-1]
                    
                    # Check for exit signals
                    if is_long and last_price > last_upper:
                        exit_signals.append(True)
                    elif not is_long and last_price < last_lower:
                        exit_signals.append(True)
                    else:
                        exit_signals.append(False)
            
            # Determine final exit decision
            if use_confirmation:
                # Require multiple indicators to confirm
                exit_count = sum(1 for signal in exit_signals if signal)
                exits[symbol] = exit_count >= 2
            else:
                # Exit if any indicator signals
                exits[symbol] = any(exit_signals)
        
        return exits

class CompositeExitManager(ExitManagerComponent):
    """Combines multiple exit managers."""
    
    def __init__(self, 
                component_id: Optional[str] = None,
                exit_managers: List[ExitManagerComponent] = None,
                require_confirmation: bool = False,
                priority_manager: Optional[ExitManagerComponent] = None):
        """
        Initialize composite exit manager
        
        Args:
            component_id: Unique component ID
            exit_managers: List of exit managers
            require_confirmation: Require multiple managers to confirm exit
            priority_manager: Manager whose exit signals always take precedence
        """
        super().__init__(component_id)
        self.exit_managers = exit_managers or []
        self.parameters = {
            'require_confirmation': require_confirmation
        }
        self.priority_manager = priority_manager
        
        manager_names = [m.description for m in self.exit_managers]
        self.description = f"Composite Exit ({', '.join(manager_names)})"
    
    def add_exit_manager(self, manager: ExitManagerComponent) -> None:
        """
        Add an exit manager
        
        Args:
            manager: Exit manager component
        """
        self.exit_managers.append(manager)
        
        # Update description
        manager_names = [m.description for m in self.exit_managers]
        self.description = f"Composite Exit ({', '.join(manager_names)})"
    
    def calculate_exits(self, positions: Dict[str, Dict[str, Any]], 
                      data: Dict[str, pd.DataFrame], 
                      context: Dict[str, Any]) -> Dict[str, Union[bool, float]]:
        """
        Calculate exit signals using multiple managers
        
        Args:
            positions: Dictionary of open positions
            data: Market data
            context: Processing context
            
        Returns:
            Dictionary of symbol -> exit flag or portion
        """
        if not self.exit_managers:
            return {symbol: False for symbol in positions}
        
        # Get exit signals from each manager
        all_exits = {}
        for manager in self.exit_managers:
            manager_exits = manager.calculate_exits(positions, data, context)
            all_exits[manager] = manager_exits
        
        # Process priority manager first if specified
        if self.priority_manager and self.priority_manager in all_exits:
            priority_exits = all_exits[self.priority_manager]
            
            # Filter positions for other managers
            remaining_positions = {}
            for symbol, position in positions.items():
                # Skip positions that priority manager wants to exit
                if symbol in priority_exits and priority_exits[symbol]:
                    continue
                remaining_positions[symbol] = position
        else:
            priority_exits = {}
            remaining_positions = positions
        
        # Process remaining positions
        require_confirmation = self.parameters['require_confirmation']
        final_exits = {}
        
        for symbol, position in positions.items():
            # First apply priority exits
            if symbol in priority_exits and priority_exits[symbol]:
                final_exits[symbol] = priority_exits[symbol]
                continue
            
            # Skip if not in remaining positions
            if symbol not in remaining_positions:
                final_exits[symbol] = False
                continue
            
            # Collect exit signals
            exit_signals = []
            partial_exits = []
            
            for manager, exits in all_exits.items():
                if manager == self.priority_manager:
                    continue
                    
                if symbol in exits:
                    if isinstance(exits[symbol], bool):
                        exit_signals.append(exits[symbol])
                    elif exits[symbol] > 0:  # Partial exit
                        partial_exits.append(exits[symbol])
            
            # Determine final exit
            if require_confirmation:
                # Require multiple confirmations
                exit_count = sum(1 for signal in exit_signals if signal)
                threshold = max(2, len(exit_signals) // 2)  # At least 2 or half
                
                if exit_count >= threshold:
                    final_exits[symbol] = True
                elif partial_exits:
                    # Use largest partial exit
                    final_exits[symbol] = max(partial_exits)
                else:
                    final_exits[symbol] = False
            else:
                # Exit if any manager signals
                if any(exit_signals):
                    final_exits[symbol] = True
                elif partial_exits:
                    # Use sum of partial exits, capped at 1.0
                    total_exit = sum(partial_exits)
                    final_exits[symbol] = min(total_exit, 1.0)
                else:
                    final_exits[symbol] = False
        
        return final_exits
