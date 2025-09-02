#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spread Manager Module

This module handles the execution and management of vertical spread positions,
including entry/exit logic, position tracking, and event-driven integration.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
import uuid
import pandas as pd

from trading_bot.core.events import Event, EventType, EventBus
from trading_bot.strategies_new.options.base.spread_types import (
    OptionType, VerticalSpreadType, OptionContract, VerticalSpread
)

# Configure logging
logger = logging.getLogger(__name__)

class SpreadPosition:
    """Represents an active spread position with management attributes."""
    
    def __init__(self, 
                spread: VerticalSpread, 
                position_id: str = None,
                max_loss_pct: float = 0.5,
                profit_target_pct: float = 0.5,
                max_days_to_hold: int = 30,
                entry_price: float = None):
        """
        Initialize a spread position.
        
        Args:
            spread: The vertical spread
            position_id: Unique identifier for this position
            max_loss_pct: Maximum loss percentage to trigger stop loss (e.g., 0.5 = 50%)
            profit_target_pct: Profit target percentage (e.g., 0.5 = 50%)
            max_days_to_hold: Maximum days to hold the position
            entry_price: Initial net premium (debit paid or credit received)
        """
        self.spread = spread
        self.position_id = position_id or str(uuid.uuid4())
        
        # Risk management parameters
        self.max_loss_pct = max_loss_pct
        self.profit_target_pct = profit_target_pct
        self.max_days_to_hold = max_days_to_hold
        
        # Position tracking
        self.entry_price = entry_price or abs(spread.net_premium)
        self.current_price = self.entry_price
        self.entry_time = datetime.now()
        self.exit_time = None
        self.exit_price = None
        self.status = "open"
        
        # Calculated exit thresholds
        self._calculate_exit_thresholds()
        
        # Position metadata
        self.metadata = {
            "underlying_price_at_entry": None,
            "exit_reason": None,
            "days_held": 0,
            "profit_loss": 0.0,
            "profit_loss_pct": 0.0
        }
        
        logger.info(f"Initialized {spread.spread_type.value} position {self.position_id}")
    
    def _calculate_exit_thresholds(self):
        """Calculate stop loss and profit target levels."""
        spread_type = self.spread.spread_type
        
        if VerticalSpreadType.is_credit(spread_type):
            # For credit spreads (BEAR_CALL_SPREAD, BULL_PUT_SPREAD)
            initial_credit = abs(self.entry_price)
            self.stop_loss = initial_credit + (self.spread.width - initial_credit) * self.max_loss_pct
            self.profit_target = initial_credit * (1 - self.profit_target_pct)
        else:
            # For debit spreads (BULL_CALL_SPREAD, BEAR_PUT_SPREAD)
            initial_debit = abs(self.entry_price)
            self.stop_loss = initial_debit * (1 - self.max_loss_pct)
            max_profit = self.spread.width - initial_debit
            self.profit_target = initial_debit + max_profit * self.profit_target_pct
    
    def update_current_price(self, price: float):
        """
        Update the current price of the spread.
        
        Args:
            price: Current price of the spread
        """
        self.current_price = price
        self._update_profit_loss()
    
    def _update_profit_loss(self):
        """Update profit/loss metrics based on current price."""
        spread_type = self.spread.spread_type
        
        if VerticalSpreadType.is_credit(spread_type):
            # For credit spreads, profit when price decreases
            self.metadata["profit_loss"] = abs(self.entry_price - self.current_price) * 100 * self.spread.quantity
            self.metadata["profit_loss_pct"] = (self.entry_price - self.current_price) / self.entry_price
        else:
            # For debit spreads, profit when price increases
            self.metadata["profit_loss"] = (self.current_price - self.entry_price) * 100 * self.spread.quantity
            self.metadata["profit_loss_pct"] = (self.current_price - self.entry_price) / self.entry_price
    
    def check_exit_conditions(self, current_date: date, underlying_price: float) -> Tuple[bool, str]:
        """
        Check if any exit conditions are met.
        
        Args:
            current_date: Current date for time-based exits
            underlying_price: Current price of the underlying asset
            
        Returns:
            Tuple of (should_exit, reason)
        """
        if self.status != "open":
            return False, "Position already closed"
        
        # Update days held
        days_held = (current_date - self.entry_time.date()).days
        self.metadata["days_held"] = days_held
        
        # Check max days to hold
        if days_held >= self.max_days_to_hold:
            return True, "max_days_reached"
        
        # Check stop loss
        if VerticalSpreadType.is_credit(self.spread.spread_type):
            # For credit spreads, higher price = loss
            if self.current_price >= self.stop_loss:
                return True, "stop_loss"
        else:
            # For debit spreads, lower price = loss
            if self.current_price <= self.stop_loss:
                return True, "stop_loss"
        
        # Check profit target
        if VerticalSpreadType.is_credit(self.spread.spread_type):
            # For credit spreads, lower price = profit
            if self.current_price <= self.profit_target:
                return True, "profit_target"
        else:
            # For debit spreads, higher price = profit
            if self.current_price >= self.profit_target:
                return True, "profit_target"
        
        # Check days to expiration
        days_to_expiration = (self.spread.long_option.expiration - current_date).days
        if days_to_expiration <= 5:
            return True, "near_expiration"
        
        # Check approaching max loss
        if self.metadata["profit_loss_pct"] <= -0.8:
            return True, "approaching_max_loss"
        
        return False, None
    
    def close_position(self, exit_price: float, exit_reason: str):
        """
        Close the position and record exit details.
        
        Args:
            exit_price: Price at which the position was closed
            exit_reason: Reason for closing the position
        """
        if self.status != "open":
            logger.warning(f"Attempted to close position {self.position_id} that is already {self.status}")
            return
        
        self.exit_price = exit_price
        self.exit_time = datetime.now()
        self.status = "closed"
        self.metadata["exit_reason"] = exit_reason
        
        # Update final P&L
        spread_type = self.spread.spread_type
        if VerticalSpreadType.is_credit(spread_type):
            # For credit spreads, profit when price decreases
            self.metadata["profit_loss"] = (self.entry_price - self.exit_price) * 100 * self.spread.quantity
            self.metadata["profit_loss_pct"] = (self.entry_price - self.exit_price) / self.entry_price
        else:
            # For debit spreads, profit when price increases
            self.metadata["profit_loss"] = (self.exit_price - self.entry_price) * 100 * self.spread.quantity
            self.metadata["profit_loss_pct"] = (self.exit_price - self.entry_price) / self.entry_price
        
        logger.info(f"Closed {self.spread.spread_type.value} position {self.position_id}: "
                  f"P&L ${self.metadata['profit_loss']:.2f} ({self.metadata['profit_loss_pct']:.2%}), "
                  f"Exit reason: {exit_reason}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary for storage/serialization."""
        position_dict = {
            "position_id": self.position_id,
            "spread": self.spread.to_dict(),
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "status": self.status,
            "stop_loss": self.stop_loss,
            "profit_target": self.profit_target,
            "max_loss_pct": self.max_loss_pct,
            "profit_target_pct": self.profit_target_pct,
            "max_days_to_hold": self.max_days_to_hold
        }
        
        # Add metadata
        position_dict.update(self.metadata)
        
        return position_dict

class SpreadManager:
    """
    Manages vertical spread positions including entry/exit execution,
    position tracking, and event-driven updates.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the spread manager with parameters.
        
        Args:
            parameters: Configuration parameters for position management
        """
        # Default parameters
        self.default_params = {
            # Position management
            'max_positions': 5,              # Maximum number of concurrent positions
            'max_positions_per_direction': 3, # Maximum positions in same direction
            'position_size_pct': 0.02,       # Position size as percentage of account
            
            # Risk management
            'max_loss_pct': 0.5,             # Maximum loss percentage to trigger stop loss
            'profit_target_pct': 0.5,        # Profit target percentage
            'max_days_to_hold': 21,          # Maximum days to hold a position
            
            # Execution parameters
            'use_limit_orders': True,        # Use limit orders instead of market
            'limit_order_offset': 0.05,      # Limit order price offset (5%)
            'early_mgmt_days': 5,            # Days before expiration to start managing more aggressively
        }
        
        # Override defaults with provided parameters
        self.parameters = self.default_params.copy()
        if parameters:
            self.parameters.update(parameters)
        
        # Position tracking
        self.active_positions = {}
        self.closed_positions = []
        
        # Performance tracking
        self.performance = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "break_even_trades": 0,
            "total_profit_loss": 0.0,
            "average_profit_loss": 0.0,
            "average_holding_days": 0.0,
            "win_rate": 0.0
        }
        
        # Event bus
        self.event_bus = EventBus.get_instance()
        
        logger.info(f"Initialized SpreadManager with max {self.parameters['max_positions']} positions")
    
    def can_open_position(self, spread_type: VerticalSpreadType) -> bool:
        """
        Check if a new position can be opened.
        
        Args:
            spread_type: Type of vertical spread
            
        Returns:
            True if a new position can be opened, False otherwise
        """
        # Check total positions limit
        if len(self.active_positions) >= self.parameters['max_positions']:
            return False
        
        # Check direction-specific limits
        is_bullish = VerticalSpreadType.is_bullish(spread_type)
        
        # Count existing positions in the same direction
        direction_count = sum(
            1 for pos in self.active_positions.values()
            if VerticalSpreadType.is_bullish(pos.spread.spread_type) == is_bullish
        )
        
        return direction_count < self.parameters['max_positions_per_direction']
    
    def open_position(self, 
                    spread: VerticalSpread, 
                    underlying_price: float,
                    entry_price: float = None) -> Optional[str]:
        """
        Open a new spread position.
        
        Args:
            spread: The vertical spread to open
            underlying_price: Current price of the underlying asset
            entry_price: Entry price (if different from current spread price)
            
        Returns:
            Position ID if successful, None otherwise
        """
        if not self.can_open_position(spread.spread_type):
            logger.warning(f"Cannot open new {spread.spread_type.value} position: position limits reached")
            return None
        
        # Create the position
        position = SpreadPosition(
            spread=spread,
            max_loss_pct=self.parameters['max_loss_pct'],
            profit_target_pct=self.parameters['profit_target_pct'],
            max_days_to_hold=self.parameters['max_days_to_hold'],
            entry_price=entry_price
        )
        
        # Store underlying price
        position.metadata["underlying_price_at_entry"] = underlying_price
        
        # Add to active positions
        self.active_positions[position.position_id] = position
        
        # Update performance stats
        self.performance["total_trades"] += 1
        
        # Publish event
        self._publish_position_event("open", position)
        
        logger.info(f"Opened {spread.spread_type.value} position {position.position_id}: "
                  f"width: ${spread.width:.2f}, premium: ${abs(spread.net_premium):.2f}")
        
        return position.position_id
    
    def update_positions(self, 
                       option_chain: pd.DataFrame, 
                       underlying_price: float,
                       current_date: date = None):
        """
        Update all active positions and check for exit conditions.
        
        Args:
            option_chain: Current option chain data
            underlying_price: Current price of the underlying asset
            current_date: Current date (defaults to today)
        """
        if not current_date:
            current_date = datetime.now().date()
        
        positions_to_close = []
        
        for position_id, position in list(self.active_positions.items()):
            # Skip if position already closed
            if position.status != "open":
                continue
            
            # Update current spread price
            current_price = self._get_current_spread_price(position.spread, option_chain)
            if current_price:
                position.update_current_price(current_price)
            
            # Check exit conditions
            should_exit, exit_reason = position.check_exit_conditions(current_date, underlying_price)
            if should_exit:
                positions_to_close.append((position_id, exit_reason))
        
        # Close positions that met exit conditions
        for position_id, exit_reason in positions_to_close:
            self.close_position(position_id, exit_reason)
    
    def close_position(self, position_id: str, exit_reason: str = "manual") -> bool:
        """
        Close a specific position.
        
        Args:
            position_id: ID of position to close
            exit_reason: Reason for closing the position
            
        Returns:
            True if position was closed, False otherwise
        """
        if position_id not in self.active_positions:
            logger.warning(f"Position {position_id} not found")
            return False
        
        position = self.active_positions[position_id]
        
        # Skip if already closed
        if position.status != "open":
            return False
        
        # Close the position
        position.close_position(position.current_price, exit_reason)
        
        # Move to closed positions list
        self.closed_positions.append(position)
        del self.active_positions[position_id]
        
        # Update performance statistics
        profit_loss = position.metadata["profit_loss"]
        self.performance["total_profit_loss"] += profit_loss
        
        if profit_loss > 0:
            self.performance["winning_trades"] += 1
        elif profit_loss < 0:
            self.performance["losing_trades"] += 1
        else:
            self.performance["break_even_trades"] += 1
            
        # Update win rate
        total = self.performance["total_trades"]
        if total > 0:
            self.performance["win_rate"] = self.performance["winning_trades"] / total
            self.performance["average_profit_loss"] = self.performance["total_profit_loss"] / total
        
        # Update average holding days
        total_days = sum(pos.metadata["days_held"] for pos in self.closed_positions)
        if self.closed_positions:
            self.performance["average_holding_days"] = total_days / len(self.closed_positions)
        
        # Publish event
        self._publish_position_event("close", position)
        
        return True
    
    def _get_current_spread_price(self, 
                                spread: VerticalSpread, 
                                option_chain: pd.DataFrame) -> Optional[float]:
        """
        Calculate the current price of a spread from the option chain.
        
        Args:
            spread: The vertical spread
            option_chain: Current option chain data
            
        Returns:
            Current spread price or None if not available
        """
        if option_chain is None or option_chain.empty:
            return None
        
        # Filter for the specific options in the spread
        long_option = spread.long_option
        short_option = spread.short_option
        
        # Find the long option in the chain
        long_option_row = option_chain[
            (option_chain['option_type'] == long_option.option_type.value) &
            (option_chain['strike'] == long_option.strike) &
            (option_chain['expiration'] == long_option.expiration)
        ]
        
        # Find the short option in the chain
        short_option_row = option_chain[
            (option_chain['option_type'] == short_option.option_type.value) &
            (option_chain['strike'] == short_option.strike) &
            (option_chain['expiration'] == short_option.expiration)
        ]
        
        # If either option not found, return None
        if long_option_row.empty or short_option_row.empty:
            return None
        
        # Calculate mid prices
        long_mid = (long_option_row['bid'].iloc[0] + long_option_row['ask'].iloc[0]) / 2
        short_mid = (short_option_row['bid'].iloc[0] + short_option_row['ask'].iloc[0]) / 2
        
        # Calculate spread price based on spread type
        if spread.spread_type == VerticalSpreadType.BULL_CALL_SPREAD:
            return long_mid - short_mid
        elif spread.spread_type == VerticalSpreadType.BEAR_CALL_SPREAD:
            return short_mid - long_mid
        elif spread.spread_type == VerticalSpreadType.BULL_PUT_SPREAD:
            return short_mid - long_mid
        elif spread.spread_type == VerticalSpreadType.BEAR_PUT_SPREAD:
            return long_mid - short_mid
        
        return None
    
    def _publish_position_event(self, action: str, position: SpreadPosition):
        """
        Publish position event to the event bus.
        
        Args:
            action: Action type ('open' or 'close')
            position: The position object
        """
        event_data = {
            "action": action,
            "position_id": position.position_id,
            "spread_type": position.spread.spread_type.value,
            "position_data": position.to_dict()
        }
        
        event = Event(
            event_type=EventType.OPTION_POSITION_UPDATE,
            data=event_data
        )
        
        self.event_bus.publish(event)
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get a summary of all positions and performance."""
        return {
            "active_positions": len(self.active_positions),
            "closed_positions": len(self.closed_positions),
            "performance": self.performance,
            "positions": {
                "active": [pos.to_dict() for pos in self.active_positions.values()],
                "closed": [pos.to_dict() for pos in self.closed_positions[-10:]]  # Last 10 closed
            }
        }
