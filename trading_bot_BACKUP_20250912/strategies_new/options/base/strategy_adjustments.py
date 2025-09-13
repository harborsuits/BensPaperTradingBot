#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Adjustments Module

This module provides standard adjustment methods that can be used by all option strategies.
It includes methods for rolling positions, defending threatened positions, and partial position
closing to lock in profits or reduce risk.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import pandas as pd

# Import necessary components
from trading_bot.core.events import Event, EventType
from trading_bot.strategies_new.options.base.spread_types import OptionType
from trading_bot.strategies_new.options.base.complex_spread_engine import ComplexSpreadType
from trading_bot.strategies_new.options.base.vertical_spread_engine import VerticalSpreadType
from trading_bot.strategies_new.options.base.volatility_spread_engine import VolatilitySpreadType
from trading_bot.strategies_new.options.base.time_spread_engine import TimeSpreadType

# Configure logging
logger = logging.getLogger(__name__)

class StrategyAdjustments:
    """
    Provides standardized adjustment methods for option positions that can be mixed into
    any strategy class.
    
    This class is designed to be used via multiple inheritance. It expects the parent class
    to have the following attributes:
    - session: Options trading session with option chain access
    - spread_manager: For managing spreads
    - parameters: Dict of strategy parameters
    - positions: List of open positions
    - filter_option_chains: Method to filter option chains
    """
    
    def roll_position(self, position, target_expiry=None, new_strikes=None, adjustment_type="standard"):
        """
        Roll a position to a new expiration date or new strikes.
        
        Args:
            position: The position to roll
            target_expiry: Target expiration date (default: next monthly expiration)
            new_strikes: New strike prices for the legs (default: similar delta to current)
            adjustment_type: Type of adjustment ("standard", "defensive", "offensive")
            
        Returns:
            bool: Success or failure
        """
        try:
            logger.info(f"Rolling position {position.position_id} with adjustment type {adjustment_type}")
            
            # Get current option chain
            option_chain = self.session.option_chain
            if option_chain is None or option_chain.empty:
                logger.error("Cannot roll position - option chain data not available")
                return False
                
            # Get current underlying price
            underlying_price = self.session.current_price
            if not underlying_price:
                logger.error("Cannot roll position - underlying price not available")
                return False
            
            # Determine current expiry
            if hasattr(position, 'legs') and position.legs:
                current_expiry = position.legs[0].get('expiry_date')
            elif hasattr(position, 'call_leg') and position.call_leg:
                current_expiry = position.call_leg.get('expiry_date')
            else:
                logger.error("Cannot determine current expiration date for position")
                return False
                
            # Determine target expiry if not provided
            if target_expiry is None:
                target_expiry = self.session.get_next_monthly_expiration(after_date=current_expiry)
                
            # Close the current position
            success = self.spread_manager.close_position(position.position_id, "roll")
            if not success:
                logger.error(f"Failed to close position {position.position_id} for rolling")
                return False
                
            # Filter the option chain for the new target expiry
            new_chain = self.filter_option_chains(option_chain)
            new_chain = new_chain[new_chain['expiry_date'] == target_expiry]
            
            if new_chain.empty:
                logger.error(f"No option chain data available for target expiry {target_expiry}")
                return False
                
            # Construct a new position based on strategy type
            new_position = None
            
            # Identify strategy type based on position attributes
            if hasattr(position, 'spread_type'):
                if isinstance(position.spread_type, ComplexSpreadType):
                    # Handle complex spread types
                    if position.spread_type == ComplexSpreadType.IRON_CONDOR:
                        new_position = self.construct_iron_condor(new_chain, underlying_price)
                    elif position.spread_type == ComplexSpreadType.BUTTERFLY:
                        new_position = self.construct_butterfly(new_chain, underlying_price)
                    elif position.spread_type == ComplexSpreadType.IRON_BUTTERFLY:
                        new_position = self.construct_iron_butterfly(new_chain, underlying_price)
                    
                elif isinstance(position.spread_type, VerticalSpreadType):
                    # Handle vertical spread types
                    if position.spread_type == VerticalSpreadType.BULL_CALL_SPREAD:
                        new_position = self.construct_bull_call_spread(new_chain, underlying_price)
                    elif position.spread_type == VerticalSpreadType.BEAR_PUT_SPREAD:
                        new_position = self.construct_bear_put_spread(new_chain, underlying_price)
                    elif position.spread_type == VerticalSpreadType.BULL_PUT_SPREAD:
                        new_position = self.construct_bull_put_spread(new_chain, underlying_price)
                    
                elif isinstance(position.spread_type, VolatilitySpreadType):
                    # Handle volatility spread types
                    if position.spread_type == VolatilitySpreadType.LONG_STRADDLE:
                        new_position = self.construct_straddle(new_chain, underlying_price, is_long=True)
                    elif position.spread_type == VolatilitySpreadType.SHORT_STRADDLE:
                        new_position = self.construct_straddle(new_chain, underlying_price, is_long=False)
                    elif position.spread_type == VolatilitySpreadType.LONG_STRANGLE:
                        new_position = self.construct_strangle(new_chain, underlying_price, is_long=True)
                    elif position.spread_type == VolatilitySpreadType.SHORT_STRANGLE:
                        new_position = self.construct_strangle(new_chain, underlying_price, is_long=False)
                    
                elif isinstance(position.spread_type, TimeSpreadType):
                    # Handle time spread types
                    if position.spread_type == TimeSpreadType.CALENDAR_SPREAD:
                        new_position = self.construct_calendar_spread(new_chain, underlying_price)
                    elif position.spread_type == TimeSpreadType.DIAGONAL_SPREAD:
                        new_position = self.construct_diagonal_spread(new_chain, underlying_price)
            
            if new_position:
                # Adjust quantity from the original position
                new_position['quantity'] = position.quantity
                
                # Add to positions
                self.positions.append(new_position)
                logger.info(f"Successfully rolled to new position with expiry {target_expiry}")
                return True
            else:
                logger.error("Failed to construct new position for roll")
                return False
                
        except Exception as e:
            logger.error(f"Error rolling position: {str(e)}")
            return False
    
    def defend_position(self, position, defense_type="standard"):
        """
        Apply defensive adjustments to a position that's under threat.
        
        Args:
            position: The position to defend
            defense_type: Type of defense ("standard", "aggressive", "conservative") 
            
        Returns:
            bool: Success or failure
        """
        try:
            logger.info(f"Applying {defense_type} defensive adjustment to position {position.position_id}")
            
            # Get current underlying price
            underlying_price = self.session.current_price
            if not underlying_price:
                logger.error("Cannot defend position - underlying price not available")
                return False
            
            # Get the option chain
            option_chain = self.session.option_chain
            if option_chain is None or option_chain.empty:
                logger.error("Cannot defend position - option chain data not available")
                return False
            
            # Identify which side is threatened based on strategy type
            threatened_side = None
            
            # For complex spreads
            if hasattr(position, 'spread_type') and isinstance(position.spread_type, ComplexSpreadType):
                # Iron Condor - determine if call or put side is threatened
                if position.spread_type == ComplexSpreadType.IRON_CONDOR:
                    # Extract short legs to determine threatened side
                    short_call = next((leg for leg in position.legs 
                                      if leg['option_type'] == OptionType.CALL and leg['action'] == 'sell'), None)
                    short_put = next((leg for leg in position.legs 
                                     if leg['option_type'] == OptionType.PUT and leg['action'] == 'sell'), None)
                    
                    if short_call and short_put:
                        short_call_strike = short_call['strike']
                        short_put_strike = short_put['strike']
                        
                        # Determine which side is threatened
                        spread_width = short_call_strike - short_put_strike
                        
                        if underlying_price > (short_call_strike - spread_width * 0.25):
                            threatened_side = "call"
                        elif underlying_price < (short_put_strike + spread_width * 0.25):
                            threatened_side = "put"
            
            # For vertical spreads
            elif hasattr(position, 'spread_type') and isinstance(position.spread_type, VerticalSpreadType):
                if position.spread_type == VerticalSpreadType.BULL_CALL_SPREAD:
                    # Threatened if price drops well below long call
                    long_call_strike = position.legs[0]['strike'] if hasattr(position, 'legs') else None
                    if long_call_strike and underlying_price < (long_call_strike * 0.97):
                        threatened_side = "long_call"
                        
                elif position.spread_type == VerticalSpreadType.BEAR_PUT_SPREAD:
                    # Threatened if price rises well above long put
                    long_put_strike = position.legs[0]['strike'] if hasattr(position, 'legs') else None
                    if long_put_strike and underlying_price > (long_put_strike * 1.03):
                        threatened_side = "long_put"
                        
                elif position.spread_type == VerticalSpreadType.BULL_PUT_SPREAD:
                    # Threatened if price drops close to short put
                    short_put_strike = position.legs[1]['strike'] if hasattr(position, 'legs') else None
                    if short_put_strike and underlying_price < (short_put_strike * 1.03):
                        threatened_side = "short_put"
            
            # Apply the appropriate defensive adjustment based on threatened side and defense type
            if threatened_side:
                logger.info(f"Detected threat to {threatened_side} side of position {position.position_id}")
                
                if defense_type == "standard":
                    # For standard defense, buy back threatened short options
                    if threatened_side == "call":
                        logger.info(f"Buying back threatened short call(s)")
                        # In real implementation, execute closing order for short call
                        return True
                    elif threatened_side == "put":
                        logger.info(f"Buying back threatened short put(s)")
                        # In real implementation, execute closing order for short put
                        return True
                    elif threatened_side in ["long_call", "long_put"]:
                        logger.info(f"Closing threatened position to limit losses")
                        # In real implementation, execute closing order for entire spread
                        return self.spread_manager.close_position(position.position_id, "defensive_exit")
                
                elif defense_type == "aggressive":
                    # For aggressive defense, roll the threatened side to further OTM
                    if threatened_side == "call":
                        logger.info(f"Rolling call spread to higher strikes")
                        # In real implementation, execute roll to higher strikes
                        return self.roll_position(position, adjustment_type="defensive")
                    elif threatened_side == "put":
                        logger.info(f"Rolling put spread to lower strikes")
                        # In real implementation, execute roll to lower strikes
                        return self.roll_position(position, adjustment_type="defensive")
                
                elif defense_type == "conservative":
                    # For conservative defense, close partial position
                    logger.info(f"Closing 50% of position to reduce risk")
                    return self.close_partial_position(position, close_percentage=50)
            else:
                logger.info("No immediate threat to position detected - no defensive action taken")
                return False
                
        except Exception as e:
            logger.error(f"Error defending position: {str(e)}")
            return False
    
    def close_partial_position(self, position, close_percentage=50):
        """
        Close a portion of a position to lock in partial profits or reduce risk.
        
        Args:
            position: The position to partially close
            close_percentage: Percentage of the position to close (default 50%)
            
        Returns:
            bool: Success or failure
        """
        try:
            logger.info(f"Closing {close_percentage}% of position {position.position_id}")
            
            # Validate the close percentage
            if close_percentage <= 0 or close_percentage >= 100:
                logger.error(f"Invalid close percentage: {close_percentage}%")
                return False
                
            # Calculate the number of contracts to close
            contracts_to_close = max(1, int(position.quantity * close_percentage / 100))
            
            if contracts_to_close >= position.quantity:
                # If closing all or more than we have, close the entire position
                logger.info(f"Closing entire position {position.position_id}")
                return self.spread_manager.close_position(position.position_id, "partial_profit_taking")
                
            # Calculate the exit prices for each leg (in a real implementation)
            exit_prices = {}
            
            # Get current market prices for legs (in a real implementation)
            # For demonstration, just use placeholder values
            if hasattr(position, 'legs'):
                for leg in position.legs:
                    # In a real implementation, would get actual market prices
                    if leg['action'] == 'buy':
                        # Long options typically lose value over time (simplified)
                        exit_prices[leg['leg_id']] = leg['entry_price'] * 0.8  # Arbitrary example
                    else:
                        # Short options can be bought back for less than collected premium (profit)
                        exit_prices[leg['leg_id']] = leg['entry_price'] * 0.4  # Arbitrary example
            
            # In a real implementation, would execute the actual closing orders
            # For now, just update the position quantity
            old_quantity = position.quantity
            position.quantity = old_quantity - contracts_to_close
            
            logger.info(f"Closed {contracts_to_close} contracts of position {position.position_id}")
            logger.info(f"Remaining position quantity: {position.quantity} contracts")
            
            return True
            
        except Exception as e:
            logger.error(f"Error partially closing position: {str(e)}")
            return False
    
    def add_to_position(self, position, add_percentage=50, adjustment_type="standard"):
        """
        Add contracts to an existing position (average down/up).
        
        Args:
            position: The position to add to
            add_percentage: Percentage of the current position to add (default 50%)
            adjustment_type: Type of adjustment ("standard", "defensive", "offensive")
            
        Returns:
            bool: Success or failure
        """
        try:
            logger.info(f"Adding {add_percentage}% to position {position.position_id}")
            
            # Validate the add percentage
            if add_percentage <= 0:
                logger.error(f"Invalid add percentage: {add_percentage}%")
                return False
                
            # Calculate the number of contracts to add
            contracts_to_add = max(1, int(position.quantity * add_percentage / 100))
            
            # Get current option chain
            option_chain = self.session.option_chain
            if option_chain is None or option_chain.empty:
                logger.error("Cannot add to position - option chain data not available")
                return False
                
            # Get current underlying price
            underlying_price = self.session.current_price
            if not underlying_price:
                logger.error("Cannot add to position - underlying price not available")
                return False
                
            # Filter the option chain for current expiry
            if hasattr(position, 'legs') and position.legs:
                current_expiry = position.legs[0].get('expiry_date')
            elif hasattr(position, 'call_leg') and position.call_leg:
                current_expiry = position.call_leg.get('expiry_date')
            else:
                logger.error("Cannot determine current expiration date for position")
                return False
                
            filtered_chain = self.filter_option_chains(option_chain)
            filtered_chain = filtered_chain[filtered_chain['expiry_date'] == current_expiry]
            
            if filtered_chain.empty:
                logger.error(f"No option chain data available for current expiry {current_expiry}")
                return False
            
            # In a real implementation, would execute the actual orders to add to position
            # For now, just update the position quantity
            position.quantity += contracts_to_add
            
            logger.info(f"Added {contracts_to_add} contracts to position {position.position_id}")
            logger.info(f"New position quantity: {position.quantity} contracts")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding to position: {str(e)}")
            return False
