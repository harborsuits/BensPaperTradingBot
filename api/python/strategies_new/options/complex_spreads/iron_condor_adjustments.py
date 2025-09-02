#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iron Condor Adjustment Methods

This module contains the adjustment methods for the Iron Condor strategy.
These methods are used to manage risk and adjust positions as market conditions change.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

from trading_bot.strategies_new.options.base.spread_types import OptionType
from trading_bot.strategies_new.options.base.complex_spread_engine import ComplexSpreadType

# Configure logging
logger = logging.getLogger(__name__)

class IronCondorAdjustments:
    """
    Provides adjustment methods for Iron Condor positions that can be mixed into the
    IronCondorStrategy class.
    """
    
    def _roll_iron_condor(self, position):
        """
        Roll an existing Iron Condor position to a new expiration date.
        
        Args:
            position: The Iron Condor position to roll
        """
        try:
            logger.info(f"Rolling Iron Condor position {position.position_id} to new expiration")
            
            # Get current option chain
            option_chain = self.session.option_chain
            if option_chain is None or option_chain.empty:
                logger.error("Cannot roll position - option chain data not available")
                return
                
            # Get current underlying price
            underlying_price = self.session.current_price
            if not underlying_price:
                logger.error("Cannot roll position - underlying price not available")
                return
                
            # Determine which side to roll (call or put or both)
            # This depends on which side is threatened or about to expire
            current_expiry = position.legs[0].get('expiry_date')
            days_to_expiry = (current_expiry - datetime.now().date()).days
            
            # For roll to next month
            target_expiry = self.session.get_next_monthly_expiration()
            
            # Get the short strikes from current position
            short_call = next((leg for leg in position.legs 
                              if leg['option_type'] == OptionType.CALL and leg['action'] == 'sell'), None)
            short_put = next((leg for leg in position.legs 
                             if leg['option_type'] == OptionType.PUT and leg['action'] == 'sell'), None)
                             
            if not short_call or not short_put:
                logger.error("Cannot identify short options in Iron Condor position")
                return
                
            # Close the current position
            self.spread_manager.close_position(position.position_id, "roll")
            
            # Construct a new Iron Condor with similar parameters but new expiration
            # Preserve the same approximate delta values for the short options
            self.parameters["call_spread_delta_short"] = short_call.get('delta', 0.16)
            self.parameters["put_spread_delta_short"] = abs(short_put.get('delta', 0.16))
            
            # Filter the option chain for the new target expiry
            new_chain = self.filter_option_chains(option_chain)
            new_chain = new_chain[new_chain['expiry_date'] == target_expiry]
            
            if new_chain.empty:
                logger.error(f"No option chain data available for target expiry {target_expiry}")
                return
                
            # Construct the new Iron Condor
            new_iron_condor = self.construct_iron_condor(new_chain, underlying_price)
            
            if new_iron_condor:
                # Add position with same quantity as original
                new_iron_condor['quantity'] = position.quantity
                self.positions.append(new_iron_condor)
                logger.info(f"Successfully rolled Iron Condor to new expiry {target_expiry}")
            else:
                logger.error("Failed to construct new Iron Condor for roll")
                
        except Exception as e:
            logger.error(f"Error rolling Iron Condor: {str(e)}")
    
    def _defend_iron_condor(self, position):
        """
        Apply defensive adjustments to an Iron Condor position that's under threat.
        
        Args:
            position: The Iron Condor position to defend
        """
        try:
            logger.info(f"Applying defensive adjustment to Iron Condor position {position.position_id}")
            
            # Get current underlying price
            underlying_price = self.session.current_price
            if not underlying_price:
                logger.error("Cannot defend position - underlying price not available")
                return
                
            # Determine which side is threatened (call side or put side)
            short_call = next((leg for leg in position.legs 
                              if leg['option_type'] == OptionType.CALL and leg['action'] == 'sell'), None)
            short_put = next((leg for leg in position.legs 
                             if leg['option_type'] == OptionType.PUT and leg['action'] == 'sell'), None)
                             
            if not short_call or not short_put:
                logger.error("Cannot identify short options in Iron Condor position")
                return
                
            short_call_strike = short_call['strike']
            short_put_strike = short_put['strike']
            
            # Determine which side is threatened
            call_side_threatened = underlying_price > (short_call_strike - (short_call_strike - short_put_strike) * 0.25)
            put_side_threatened = underlying_price < (short_put_strike + (short_call_strike - short_put_strike) * 0.25)
            
            # Get the option chain
            option_chain = self.session.option_chain
            if option_chain is None or option_chain.empty:
                logger.error("Cannot defend position - option chain data not available")
                return
                
            # Apply the appropriate defensive adjustment based on which side is threatened
            if call_side_threatened:
                logger.info(f"Defending call side of Iron Condor at strike {short_call_strike}")
                # Strategy: Buy back the threatened short call
                #           OR roll the call spread higher
                
                # For demonstration, we'll buy back the short call
                # In a real implementation, would execute the actual order
                logger.info(f"Defensive action: Buying back short call at strike {short_call_strike}")
                
            elif put_side_threatened:
                logger.info(f"Defending put side of Iron Condor at strike {short_put_strike}")
                # Strategy: Buy back the threatened short put
                #           OR roll the put spread lower
                
                # For demonstration, we'll buy back the short put
                # In a real implementation, would execute the actual order
                logger.info(f"Defensive action: Buying back short put at strike {short_put_strike}")
                
            else:
                logger.info("No immediate threat to Iron Condor position - no defensive action taken")
                
        except Exception as e:
            logger.error(f"Error defending Iron Condor: {str(e)}")
    
    def _close_partial_iron_condor(self, position, close_percentage=50):
        """
        Close a portion of an Iron Condor position to lock in partial profits.
        
        Args:
            position: The Iron Condor position to partially close
            close_percentage: Percentage of the position to close (default 50%)
        """
        try:
            logger.info(f"Closing {close_percentage}% of Iron Condor position {position.position_id}")
            
            # Validate the close percentage
            if close_percentage <= 0 or close_percentage >= 100:
                logger.error(f"Invalid close percentage: {close_percentage}%")
                return
                
            # Calculate the number of contracts to close
            contracts_to_close = max(1, int(position.quantity * close_percentage / 100))
            
            if contracts_to_close >= position.quantity:
                # If closing all or more than we have, close the entire position
                logger.info(f"Closing entire Iron Condor position {position.position_id}")
                self.spread_manager.close_position(position.position_id, "partial_profit_taking")
                return
                
            # Calculate the exit prices for each leg (in a real implementation)
            exit_prices = {}
            for leg in position.legs:
                # In a real implementation, would get actual market prices
                # For demo, just use some assumed values based on entry price
                if leg['action'] == 'buy':
                    # Long options typically lose value over time
                    exit_prices[leg['leg_id']] = leg['entry_price'] * 0.8  # Arbitrary example
                else:
                    # Short options can be bought back for less than collected premium (profit)
                    exit_prices[leg['leg_id']] = leg['entry_price'] * 0.4  # Arbitrary example
            
            # Create a new position with reduced quantity
            new_quantity = position.quantity - contracts_to_close
            
            # Record the partial close (in reality, would execute the actual orders)
            logger.info(f"Closed {contracts_to_close} contracts of Iron Condor {position.position_id}")
            
            # Update the position quantity
            position.quantity = new_quantity
            logger.info(f"Remaining position quantity: {new_quantity} contracts")
            
        except Exception as e:
            logger.error(f"Error partially closing Iron Condor: {str(e)}")
