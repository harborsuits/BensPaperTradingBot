#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Spread Engine Module

This module provides the foundation for options strategies that exploit time decay
and differences between expiration cycles, such as Calendar Spreads and Diagonal Spreads.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
from enum import Enum

from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.core.events import Event, EventType, EventBus
from trading_bot.strategies_new.options.base.options_base_strategy import OptionsBaseStrategy, OptionsSession
from trading_bot.strategies_new.options.base.spread_types import OptionType
from trading_bot.strategies_new.options.base.spread_analyzer import SpreadAnalyzer
from trading_bot.strategies_new.options.base.spread_manager import SpreadManager

# Configure logging
logger = logging.getLogger(__name__)

class TimeSpreadType(Enum):
    """Types of time-based option spreads supported by the engine."""
    CALENDAR = "calendar_spread"  # Same strike, different expirations
    DIAGONAL = "diagonal_spread"  # Different strikes, different expirations
    DOUBLE_CALENDAR = "double_calendar"  # Calendar spread with both calls and puts


class TimeSpreadPosition:
    """Represents a time spread position with options at different expirations."""
    
    def __init__(self, 
                spread_type: TimeSpreadType,
                front_leg: Dict[str, Any],
                back_leg: Dict[str, Any],
                quantity: int = 1,
                position_id: Optional[str] = None):
        """
        Initialize a time spread position.
        
        Args:
            spread_type: Type of time spread
            front_leg: Front-month (shorter-term) option details
            back_leg: Back-month (longer-term) option details
            quantity: Number of spreads in the position
            position_id: Optional unique identifier
        """
        self.spread_type = spread_type
        self.front_leg = front_leg
        self.back_leg = back_leg
        self.quantity = quantity
        self.position_id = position_id or str(uuid.uuid4())
        
        # Position tracking
        self.entry_time = datetime.now()
        self.exit_time = None
        self.status = "open"
        
        # Risk metrics
        self.max_profit = None  # Hard to define for calendar spreads
        self.max_loss = self._calculate_max_loss()
        self.optimal_price = self._calculate_optimal_price()
        
        logger.info(f"Created {spread_type.value} position with ID: {self.position_id}")
    
    def _calculate_max_loss(self) -> float:
        """Calculate maximum potential loss for this time spread."""
        # For a calendar spread, max loss is typically the net debit paid
        if self.front_leg['action'] == 'sell' and self.back_leg['action'] == 'buy':
            return (self.back_leg['entry_price'] - self.front_leg['entry_price']) * 100 * self.quantity
        else:
            return (self.front_leg['entry_price'] - self.back_leg['entry_price']) * 100 * self.quantity
    
    def _calculate_optimal_price(self) -> float:
        """Calculate the optimal underlying price at front-month expiration."""
        # For a calendar, optimal price is typically near the strike price
        return self.front_leg['strike']
    
    def update_prices(self, front_leg_price: float, back_leg_price: float):
        """
        Update current prices for both legs of the spread.
        
        Args:
            front_leg_price: Current price of front-month option
            back_leg_price: Current price of back-month option
        """
        self.front_leg['current_price'] = front_leg_price
        self.back_leg['current_price'] = back_leg_price
    
    def close_position(self, front_leg_price: float, back_leg_price: float, exit_reason: str):
        """
        Close the position and record exit details.
        
        Args:
            front_leg_price: Exit price for front-month option
            back_leg_price: Exit price for back-month option
            exit_reason: Reason for closing the position
        """
        if self.status != "open":
            logger.warning(f"Attempted to close position {self.position_id} that is already {self.status}")
            return
        
        self.exit_time = datetime.now()
        self.front_leg['exit_price'] = front_leg_price
        self.back_leg['exit_price'] = back_leg_price
        self.status = "closed"
        
        # Calculate P&L
        if self.front_leg['action'] == 'sell' and self.back_leg['action'] == 'buy':
            entry_net_debit = self.back_leg['entry_price'] - self.front_leg['entry_price']
            exit_net_debit = back_leg_price - front_leg_price
            pnl = (entry_net_debit - exit_net_debit) * 100 * self.quantity
        else:
            entry_net_debit = self.front_leg['entry_price'] - self.back_leg['entry_price']
            exit_net_debit = front_leg_price - back_leg_price
            pnl = (entry_net_debit - exit_net_debit) * 100 * self.quantity
        
        self.profit_loss = pnl
        
        logger.info(f"Closed {self.spread_type.value} position {self.position_id}, P&L: ${pnl:.2f}, reason: {exit_reason}")


class TimeSpreadEngine(OptionsBaseStrategy):
    """
    Base engine for time-based options spread strategies.
    
    This engine provides the foundation for strategies like Calendar Spreads
    and Diagonal Spreads, which profit from time decay differences between
    option expiration cycles.
    """
    
    def __init__(self, session: OptionsSession, data_pipeline: DataPipeline, 
                 parameters: Dict[str, Any] = None):
        """
        Initialize the time spread engine.
        
        Args:
            session: Options trading session
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        super().__init__(session, data_pipeline, parameters)
        
        # Default parameters specific to time spreads
        default_time_params = {
            # Calendar Spread parameters
            'calendar_min_days_between': 20,   # Minimum days between front and back month
            'calendar_max_days_between': 60,   # Maximum days between front and back month
            'calendar_front_dte_min': 15,      # Minimum DTE for front month
            'calendar_front_dte_max': 45,      # Maximum DTE for front month
            
            # Strike selection parameters
            'use_atm_strikes': True,           # Use strikes near the money
            'delta_target': 0.50,              # Target delta (near ATM)
            
            # IV parameters
            'prefer_high_iv': True,            # Prefer high IV environments for calendars
            'min_iv_skew': 0.02,               # Minimum IV difference between months
            
            # Risk parameters
            'max_loss_per_trade': 300,         # Maximum dollar loss per trade
            'profit_target_pct': 0.50,         # Take profit at this percentage of max potential
            'stop_loss_pct': 0.50,             # Stop loss at this percentage of max loss
            
            # Management parameters
            'manage_front_expiry': True,       # Whether to actively manage near front expiry
            'days_before_expiry_exit': 5,      # Exit N days before front expiry if not managed
        }
        
        # Update parameters with defaults for time spreads
        for key, value in default_time_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Initialize specialized components
        self.spread_analyzer = SpreadAnalyzer()
        self.spread_manager = SpreadManager()
        
        # Strategy state
        self.positions = []  # List of TimeSpreadPosition objects
        self.iv_skew = {}  # Track IV skew between months
        
        logger.info(f"Initialized TimeSpreadEngine for {session.symbol}")
    
    def construct_calendar_spread(self, option_chain: pd.DataFrame, 
                              underlying_price: float,
                              option_type: OptionType = OptionType.CALL) -> Optional[TimeSpreadPosition]:
        """
        Construct a Calendar spread from the option chain.
        
        Args:
            option_chain: Option chain data
            underlying_price: Current price of the underlying asset
            option_type: Type of options to use (CALL or PUT)
            
        Returns:
            TimeSpreadPosition if successful, None otherwise
        """
        if option_chain is None or option_chain.empty:
            return None
        
        # Filter option chain for liquidity, open interest, etc.
        filtered_chain = self.filter_option_chains(option_chain)
        
        # Get unique expiration dates
        expirations = sorted(filtered_chain['expiration_date'].unique())
        if len(expirations) < 2:
            logger.warning("Need at least 2 expiration cycles for calendar spread")
            return None
        
        # Find target strike near the money
        target_strike = self._find_atm_strike(filtered_chain, underlying_price)
        if target_strike is None:
            logger.warning("Could not find suitable ATM strike")
            return None
        
        # Filter chains for the specific option type and strike
        type_filtered = filtered_chain[
            (filtered_chain['option_type'] == option_type.value) &
            (filtered_chain['strike'] == target_strike)
        ]
        
        # Get front and back month options
        today = datetime.now().date()
        front_dte_min = self.parameters['calendar_front_dte_min']
        front_dte_max = self.parameters['calendar_front_dte_max']
        min_days_between = self.parameters['calendar_min_days_between']
        max_days_between = self.parameters['calendar_max_days_between']
        
        # Find suitable expirations
        valid_front_expirations = []
        for exp in expirations:
            exp_date = datetime.strptime(exp, '%Y-%m-%d').date() if isinstance(exp, str) else exp
            dte = (exp_date - today).days
            if front_dte_min <= dte <= front_dte_max:
                valid_front_expirations.append(exp)
        
        if not valid_front_expirations:
            logger.warning("No valid front-month expirations found")
            return None
        
        # For each valid front expiration, try to find a suitable back month
        for front_exp in valid_front_expirations:
            front_exp_date = datetime.strptime(front_exp, '%Y-%m-%d').date() if isinstance(front_exp, str) else front_exp
            front_dte = (front_exp_date - today).days
            
            valid_back_expirations = []
            for back_exp in expirations:
                if back_exp <= front_exp:
                    continue
                    
                back_exp_date = datetime.strptime(back_exp, '%Y-%m-%d').date() if isinstance(back_exp, str) else back_exp
                back_dte = (back_exp_date - today).days
                
                days_between = back_dte - front_dte
                if min_days_between <= days_between <= max_days_between:
                    valid_back_expirations.append(back_exp)
            
            if not valid_back_expirations:
                continue
            
            # Get options for front and back month
            front_option = type_filtered[type_filtered['expiration_date'] == front_exp]
            
            if front_option.empty:
                continue
                
            # Pick the back month with highest IV skew if available
            best_back_exp = None
            best_iv_skew = 0
            
            for back_exp in valid_back_expirations:
                back_option = type_filtered[type_filtered['expiration_date'] == back_exp]
                
                if back_option.empty:
                    continue
                
                if 'implied_volatility' in front_option.columns and 'implied_volatility' in back_option.columns:
                    front_iv = front_option['implied_volatility'].iloc[0]
                    back_iv = back_option['implied_volatility'].iloc[0]
                    iv_skew = back_iv - front_iv
                    
                    if iv_skew > best_iv_skew:
                        best_iv_skew = iv_skew
                        best_back_exp = back_exp
                else:
                    best_back_exp = back_exp
                    break
            
            if best_back_exp is None:
                continue
            
            back_option = type_filtered[type_filtered['expiration_date'] == best_back_exp]
            
            # Check min IV skew if IV data is available
            if 'implied_volatility' in front_option.columns and 'implied_volatility' in back_option.columns:
                front_iv = front_option['implied_volatility'].iloc[0]
                back_iv = back_option['implied_volatility'].iloc[0]
                iv_skew = back_iv - front_iv
                
                if iv_skew < self.parameters['min_iv_skew']:
                    logger.info(f"IV skew ({iv_skew:.4f}) below minimum threshold")
                    continue
            
            # Create calendar spread
            front_leg = {
                'leg_id': 'front_leg',
                'option_type': option_type.value,
                'strike': target_strike,
                'expiration': front_exp,
                'action': 'sell',
                'entry_price': front_option['bid'].iloc[0],  # Sell at bid
                'dte': front_dte
            }
            
            back_leg = {
                'leg_id': 'back_leg',
                'option_type': option_type.value,
                'strike': target_strike,
                'expiration': best_back_exp,
                'action': 'buy',
                'entry_price': back_option['ask'].iloc[0],  # Buy at ask
                'dte': (datetime.strptime(best_back_exp, '%Y-%m-%d').date() - today).days if isinstance(best_back_exp, str) else (best_back_exp - today).days
            }
            
            # Calculate debit
            net_debit = back_leg['entry_price'] - front_leg['entry_price']
            
            # Ensure debit is reasonable 
            if net_debit <= 0:
                logger.warning(f"Calendar spread has zero or negative debit: {net_debit:.2f}")
                continue
                
            if net_debit * 100 > self.parameters['max_loss_per_trade']:
                logger.warning(f"Calendar spread debit (${net_debit * 100:.2f}) exceeds max loss per trade")
                continue
            
            # Create position
            return TimeSpreadPosition(
                spread_type=TimeSpreadType.CALENDAR,
                front_leg=front_leg,
                back_leg=back_leg,
                quantity=1
            )
        
        logger.warning("Could not construct valid Calendar spread")
        return None
    
    def _find_atm_strike(self, option_chain: pd.DataFrame, underlying_price: float) -> Optional[float]:
        """
        Find the strike price closest to the current underlying price (ATM).
        
        Args:
            option_chain: Option chain data
            underlying_price: Current price of the underlying asset
            
        Returns:
            Strike price or None if not found
        """
        if option_chain is None or option_chain.empty or 'strike' not in option_chain.columns:
            return None
            
        # Get unique strikes
        strikes = sorted(option_chain['strike'].unique())
        
        # Find strike closest to underlying price
        closest_strike = min(strikes, key=lambda x: abs(x - underlying_price))
        
        return closest_strike
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for the strategy.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        if data.empty or len(data) < 20:
            return indicators
        
        # Calculate historical volatility (20-day)
        if len(data) >= 20:
            data['returns'] = data['close'].pct_change()
            indicators['hist_volatility_20d'] = data['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Calculate recent price momentum
        if 'close' in data.columns:
            indicators['price_5d_change'] = data['close'].pct_change(periods=5) * 100
            indicators['price_20d_change'] = data['close'].pct_change(periods=20) * 100
            
            # Calculate moving averages
            indicators['ma_20'] = data['close'].rolling(window=20).mean()
            indicators['ma_50'] = data['close'].rolling(window=50).mean()
            
            # Calculate trend indicators
            indicators['trend_direction'] = np.where(
                indicators['ma_20'] > indicators['ma_50'], 
                'up', 
                np.where(indicators['ma_20'] < indicators['ma_50'], 'down', 'neutral')
            )
            
            # Calculate price location relative to recent range
            if len(data) >= 20:
                recent_high = data['high'].rolling(window=20).max()
                recent_low = data['low'].rolling(window=20).min()
                price_range = recent_high - recent_low
                
                # Avoid division by zero
                price_range = np.where(price_range == 0, 0.0001, price_range)
                
                # Where is price in the recent range (0-1)
                indicators['price_in_range'] = (data['close'] - recent_low) / price_range
                
                # Ideal for calendar spreads is when price is near the middle of range
                indicators['range_suitability'] = 1 - abs(indicators['price_in_range'] - 0.5) * 2
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for time spreads based on market conditions.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            "entry": False,
            "spread_type": None,
            "option_type": None,
            "exit_positions": [],
            "signal_strength": 0.0
        }
        
        if data.empty or not indicators:
            return signals
        
        # Get current IV percentile if available
        iv_percentile = None
        if self.session.current_iv is not None and self.session.symbol in self.iv_history:
            iv_metrics = self.calculate_implied_volatility_metrics(self.iv_history[self.session.symbol])
            iv_percentile = iv_metrics.get('iv_percentile')
        
        # Calendar Spread logic - best when IV is high and price is in middle of range
        if self.parameters['prefer_high_iv'] and iv_percentile and iv_percentile > 60:
            # High IV environment is good for selling premium
            if 'range_suitability' in indicators:
                range_suit = indicators['range_suitability'].iloc[-1]
                
                if range_suit > 0.7:  # Price is near middle of range
                    # Decide between put and call calendars
                    if 'trend_direction' in indicators:
                        trend = indicators['trend_direction'].iloc[-1]
                        
                        if trend == 'up':
                            # In uptrend, prefer call calendars
                            signals["entry"] = True
                            signals["spread_type"] = TimeSpreadType.CALENDAR
                            signals["option_type"] = OptionType.CALL
                            signals["signal_strength"] = range_suit * 0.8
                        elif trend == 'down':
                            # In downtrend, prefer put calendars
                            signals["entry"] = True
                            signals["spread_type"] = TimeSpreadType.CALENDAR
                            signals["option_type"] = OptionType.PUT
                            signals["signal_strength"] = range_suit * 0.8
                        else:
                            # In neutral trend, look at volatility for bias
                            if 'hist_volatility_20d' in indicators:
                                vol = indicators['hist_volatility_20d'].iloc[-1]
                                if vol > 0.2:  # Higher volatility
                                    signals["entry"] = True
                                    signals["spread_type"] = TimeSpreadType.CALENDAR
                                    signals["option_type"] = OptionType.PUT  # Puts often have higher IV in high vol
                                    signals["signal_strength"] = range_suit * 0.7
                                else:
                                    signals["entry"] = True
                                    signals["spread_type"] = TimeSpreadType.CALENDAR
                                    signals["option_type"] = OptionType.CALL
                                    signals["signal_strength"] = range_suit * 0.7
        
        # Check for exits
        today = datetime.now().date()
        for position in self.positions:
            if position.status == "open":
                # Exit if approaching front month expiration
                front_exp = datetime.strptime(position.front_leg['expiration'], '%Y-%m-%d').date() if isinstance(position.front_leg['expiration'], str) else position.front_leg['expiration']
                days_to_expiry = (front_exp - today).days
                
                if days_to_expiry <= self.parameters['days_before_expiry_exit']:
                    signals["exit_positions"].append(position.position_id)
                
                # Exit if trend changes significantly against our position
                if 'trend_direction' in indicators:
                    trend = indicators['trend_direction'].iloc[-1]
                    
                    if (position.front_leg['option_type'] == 'call' and trend == 'down') or \
                       (position.front_leg['option_type'] == 'put' and trend == 'up'):
                        signals["exit_positions"].append(position.position_id)
        
        return signals
    
    def _execute_signals(self):
        """Execute trading signals generated by the strategy."""
        if not self.signals:
            return
        
        # Handle entry signals
        if self.signals.get("entry", False) and "spread_type" in self.signals:
            spread_type = self.signals["spread_type"]
            option_type = self.signals.get("option_type", OptionType.CALL)
            underlying_price = self.session.current_price or self.market_data['close'].iloc[-1]
            
            # Construct the appropriate spread
            position = None
            if spread_type == TimeSpreadType.CALENDAR:
                position = self.construct_calendar_spread(self.session.option_chain, underlying_price, option_type)
            # Could add others like diagonal spread here
            
            if position:
                # Add position to active positions
                self.positions.append(position)
                logger.info(f"Opened {spread_type.value} position {position.position_id}")
        
        # Handle exit signals
        for position_id in self.signals.get("exit_positions", []):
            for i, position in enumerate(self.positions):
                if position.position_id == position_id and position.status == "open":
                    # Get current prices for both legs
                    # In a real implementation, would get actual prices from option chain
                    # For now, just set placeholder values
                    front_exit_price = position.front_leg['entry_price'] * 0.5  # Placeholder
                    back_exit_price = position.back_leg['entry_price'] * 1.1    # Placeholder
                    
                    # Close the position
                    position.close_position(front_exit_price, back_exit_price, "signal_generated")
                    logger.info(f"Closed position {position_id} based on signals")
                    break
