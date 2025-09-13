#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complex Spread Factory for Options Strategies

This module implements a factory pattern for creating complex option spreads
such as iron condors, butterflies, and other multi-leg option strategies.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum

from trading_bot.strategies.options.complex_spreads.components.market_analysis import ComplexSpreadMarketAnalyzer
from trading_bot.strategies.options.complex_spreads.components.option_selection import ComplexSpreadOptionSelector
from trading_bot.strategies.options.complex_spreads.components.risk_management import ComplexSpreadRiskManager

logger = logging.getLogger(__name__)

class SpreadType(Enum):
    """Enumeration of complex spread types."""
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"
    CONDOR = "condor"
    IRON_BUTTERFLY = "iron_butterfly"
    DOUBLE_CALENDAR = "double_calendar"
    BROKEN_WING_BUTTERFLY = "broken_wing_butterfly"

class SpreadDirection(Enum):
    """Direction of the spread (bullish, bearish, neutral)."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

class ComplexSpreadFactory:
    """
    Factory class for creating complex option spreads.
    
    This class provides methods to create various option spread strategies
    based on market conditions, option chain data, and strategy parameters.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the complex spread factory.
        
        Args:
            parameters: Strategy parameters
        """
        self.params = parameters or {}
        self.market_analyzer = ComplexSpreadMarketAnalyzer(parameters)
        self.option_selector = ComplexSpreadOptionSelector(parameters)
        self.risk_manager = ComplexSpreadRiskManager(parameters)
        self.logger = logger
        
    def create_complex_spread(self, 
                            symbol: str, 
                            current_price: float,
                            option_chain: Any,
                            market_data: Any,
                            spread_type: Union[str, SpreadType],
                            direction: Union[str, SpreadDirection] = SpreadDirection.NEUTRAL,
                            account_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a complex spread trade configuration.
        
        Args:
            symbol: Underlying symbol
            current_price: Current price of the underlying
            option_chain: Option chain data
            market_data: Market data for analysis
            spread_type: Type of spread to create
            direction: Direction of the spread (bullish, bearish, neutral)
            account_info: Account information for position sizing
            
        Returns:
            Dictionary with spread trade details or None if no valid spread found
        """
        try:
            # Normalize inputs
            if isinstance(spread_type, str):
                spread_type = SpreadType(spread_type)
                
            if isinstance(direction, str):
                direction = SpreadDirection(direction)
                
            # Analyze market conditions
            iv_surface = self.market_analyzer.calculate_implied_volatility_surface(
                option_chain, current_price
            )
            
            price_levels = self.market_analyzer.detect_price_levels(market_data)
            
            # Select appropriate spread based on type and direction
            spread = None
            
            if spread_type == SpreadType.IRON_CONDOR:
                spread = self.option_selector.find_iron_condor_options(
                    symbol, option_chain, current_price, iv_surface, price_levels
                )
                
            elif spread_type == SpreadType.BUTTERFLY:
                spread = self.option_selector.find_butterfly_options(
                    symbol, option_chain, current_price, iv_surface, price_levels
                )
                
            # Add other spread types as needed
            
            # If no valid spread found, return None
            if not spread:
                return None
                
            # Calculate position size if account info provided
            if account_info:
                position_size = self.risk_manager.calculate_position_size(spread, account_info)
                spread['quantity'] = position_size
                
            # Calculate exit conditions
            position_size = spread.get('quantity', 1)
            exit_conditions = self.risk_manager.calculate_exit_conditions(spread, position_size)
            spread['exit_conditions'] = exit_conditions
            
            # Calculate additional metrics with price data
            additional_metrics = self.option_selector.evaluate_spread_metrics(
                spread, current_price, market_data
            )
            spread.update(additional_metrics)
            
            # Add timestamp and metadata
            spread['timestamp'] = datetime.now().isoformat()
            spread['parameters'] = {k: v for k, v in self.params.items() if k in [
                'target_dte', 'short_call_delta', 'short_put_delta', 
                'call_spread_width', 'put_spread_width', 'profit_target_pct'
            ]}
            
            return spread
            
        except Exception as e:
            self.logger.error(f"Error creating complex spread for {symbol}: {e}")
            return None
            
    def create_iron_condor(self, 
                         symbol: str, 
                         current_price: float,
                         option_chain: Any,
                         market_data: Any,
                         account_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create an iron condor spread.
        
        Args:
            symbol: Underlying symbol
            current_price: Current price of the underlying
            option_chain: Option chain data
            market_data: Market data for analysis
            account_info: Account information for position sizing
            
        Returns:
            Dictionary with iron condor details or None if no valid spread found
        """
        return self.create_complex_spread(
            symbol=symbol,
            current_price=current_price,
            option_chain=option_chain,
            market_data=market_data,
            spread_type=SpreadType.IRON_CONDOR,
            direction=SpreadDirection.NEUTRAL,
            account_info=account_info
        )
        
    def create_butterfly(self, 
                       symbol: str, 
                       current_price: float,
                       option_chain: Any,
                       market_data: Any,
                       account_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a butterfly spread.
        
        Args:
            symbol: Underlying symbol
            current_price: Current price of the underlying
            option_chain: Option chain data
            market_data: Market data for analysis
            account_info: Account information for position sizing
            
        Returns:
            Dictionary with butterfly details or None if no valid spread found
        """
        return self.create_complex_spread(
            symbol=symbol,
            current_price=current_price,
            option_chain=option_chain,
            market_data=market_data,
            spread_type=SpreadType.BUTTERFLY,
            direction=SpreadDirection.NEUTRAL,
            account_info=account_info
        )
        
    def create_best_complex_spread(self,
                                 symbol: str,
                                 current_price: float,
                                 option_chain: Any,
                                 market_data: Any,
                                 account_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create the best complex spread based on market conditions.
        
        Args:
            symbol: Underlying symbol
            current_price: Current price of the underlying
            option_chain: Option chain data
            market_data: Market data for analysis
            account_info: Account information for position sizing
            
        Returns:
            Dictionary with spread details or None if no valid spread found
        """
        try:
            # Analyze market regime to determine best strategy
            regime_analysis = self.market_analyzer.analyze_market_regime(market_data)
            
            if not regime_analysis['is_suitable']:
                self.logger.info(f"Current market regime {regime_analysis['regime']} not suitable for complex spreads")
                return None
                
            # Determine best strategy based on regime analysis
            recommendation = regime_analysis['recommendation']
            
            if recommendation == 'iron_condor':
                return self.create_iron_condor(
                    symbol, current_price, option_chain, market_data, account_info
                )
                
            elif recommendation == 'butterfly':
                return self.create_butterfly(
                    symbol, current_price, option_chain, market_data, account_info
                )
                
            # Default to iron condor if no clear recommendation
            return self.create_iron_condor(
                symbol, current_price, option_chain, market_data, account_info
            )
            
        except Exception as e:
            self.logger.error(f"Error creating best complex spread for {symbol}: {e}")
            return None
            
    def create_spread_for_earnings(self,
                                 symbol: str,
                                 current_price: float,
                                 option_chain: Any,
                                 market_data: Any,
                                 days_to_earnings: int,
                                 account_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a complex spread optimized for earnings events.
        
        Args:
            symbol: Underlying symbol
            current_price: Current price of the underlying
            option_chain: Option chain data
            market_data: Market data for analysis
            days_to_earnings: Days until earnings announcement
            account_info: Account information for position sizing
            
        Returns:
            Dictionary with spread details or None if no valid spread found
        """
        # Customize parameters for earnings
        earnings_params = self.params.copy()
        
        # For earnings, typically want closer to the event
        earnings_params['target_dte'] = max(days_to_earnings + 5, 21)  # At least 5 days after earnings
        earnings_params['min_dte'] = max(days_to_earnings + 3, 14)     # At least 3 days after earnings
        
        # Typically want wider wings for earnings to accommodate bigger moves
        earnings_params['call_spread_width'] = self.params.get('call_spread_width', 5) * 1.5
        earnings_params['put_spread_width'] = self.params.get('put_spread_width', 5) * 1.5
        
        # Adjust deltas for earnings (typically more conservative)
        earnings_params['short_call_delta'] = max(self.params.get('short_call_delta', -0.16) * 0.75, -0.12)
        earnings_params['short_put_delta'] = min(self.params.get('short_put_delta', 0.16) * 0.75, 0.12)
        
        # Create temporary factory with earnings parameters
        earnings_factory = ComplexSpreadFactory(earnings_params)
        
        # Create iron condor with adjusted parameters
        return earnings_factory.create_iron_condor(
            symbol, current_price, option_chain, market_data, account_info
        )
        
    def adjust_iron_condor(self,
                         position: Dict[str, Any],
                         current_price: float,
                         option_chain: Any,
                         market_data: Any,
                         adjustment_type: str) -> Dict[str, Any]:
        """
        Generate adjustments for an existing iron condor position.
        
        Args:
            position: Current position data
            current_price: Current price of the underlying
            option_chain: Option chain data
            market_data: Market data for analysis
            adjustment_type: Type of adjustment ('roll_out', 'defend_call_side', 'defend_put_side')
            
        Returns:
            Dictionary with adjustment details or None if no valid adjustment
        """
        try:
            # Validate position and adjustment type
            if not position or 'strategy_type' not in position or position['strategy_type'] != 'iron_condor':
                self.logger.warning("Invalid position for iron condor adjustment")
                return None
                
            if adjustment_type not in ['roll_out', 'defend_call_side', 'defend_put_side']:
                self.logger.warning(f"Unsupported adjustment type: {adjustment_type}")
                return None
                
            # Extract necessary data
            symbol = position.get('symbol', '')
            dte = position.get('dte', 0)
            
            # Analyze market conditions for new spread components
            iv_surface = self.market_analyzer.calculate_implied_volatility_surface(
                option_chain, current_price
            )
            
            # Create adjustment based on type
            adjustment = {
                'position_id': position.get('position_id', ''),
                'symbol': symbol,
                'strategy_type': 'iron_condor',
                'adjustment_type': adjustment_type,
                'current_price': current_price,
                'timestamp': datetime.now().isoformat(),
                'original_legs': {
                    'short_call': {
                        'strike': position.get('short_call_strike', 0),
                        'symbol': position.get('short_call_symbol', '')
                    },
                    'long_call': {
                        'strike': position.get('long_call_strike', 0),
                        'symbol': position.get('long_call_symbol', '')
                    },
                    'short_put': {
                        'strike': position.get('short_put_strike', 0),
                        'symbol': position.get('short_put_symbol', '')
                    },
                    'long_put': {
                        'strike': position.get('long_put_strike', 0),
                        'symbol': position.get('long_put_symbol', '')
                    }
                },
                'new_legs': {}
            }
            
            # Create specific adjustment
            if adjustment_type == 'roll_out':
                # Roll all legs to a later expiration
                new_dte = dte + 30
                target_dte_params = self.params.copy()
                target_dte_params['target_dte'] = new_dte
                target_dte_params['min_dte'] = new_dte - 7
                target_dte_params['max_dte'] = new_dte + 7
                
                # Create temporary factory with new DTE parameters
                roll_factory = ComplexSpreadFactory(target_dte_params)
                
                # Find new iron condor with similar strikes but further expiration
                new_spread = roll_factory.create_iron_condor(
                    symbol, current_price, option_chain, market_data
                )
                
                if new_spread:
                    adjustment['new_legs'] = {
                        'short_call': {
                            'strike': new_spread.get('short_call_strike', 0),
                            'symbol': new_spread.get('short_call_symbol', '')
                        },
                        'long_call': {
                            'strike': new_spread.get('long_call_strike', 0),
                            'symbol': new_spread.get('long_call_symbol', '')
                        },
                        'short_put': {
                            'strike': new_spread.get('short_put_strike', 0),
                            'symbol': new_spread.get('short_put_symbol', '')
                        },
                        'long_put': {
                            'strike': new_spread.get('long_put_strike', 0),
                            'symbol': new_spread.get('long_put_symbol', '')
                        }
                    }
                    adjustment['new_expiration'] = new_spread.get('expiration')
                    adjustment['net_debit_credit'] = new_spread.get('total_credit', 0) - position.get('current_value', 0)
                    
                else:
                    self.logger.warning("Could not find suitable roll for iron condor")
                    return None
                    
            elif adjustment_type == 'defend_call_side':
                # Roll just the call spread further out and up
                # This is a simplified version - in real implementation, would need 
                # to find specific new call options based on current chain
                adjustment['new_legs'] = {
                    'short_call': {
                        'action': 'roll_up_and_out',
                        'new_delta': -0.12  # More conservative delta
                    },
                    'long_call': {
                        'action': 'roll_up_and_out'
                    }
                }
                
            elif adjustment_type == 'defend_put_side':
                # Roll just the put spread further out and down
                adjustment['new_legs'] = {
                    'short_put': {
                        'action': 'roll_down_and_out',
                        'new_delta': 0.12  # More conservative delta
                    },
                    'long_put': {
                        'action': 'roll_down_and_out'
                    }
                }
                
            return adjustment
                
        except Exception as e:
            self.logger.error(f"Error adjusting iron condor: {e}")
            return None
