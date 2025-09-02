#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complex Spread Engine Module

This module provides the foundation for multi-leg options spread strategies
like Iron Condors and Butterflies, supporting strategies that use more than
two options contracts in their construction.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
from enum import Enum

from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.core.events import Event, EventType, EventBus
from trading_bot.strategies_new.options.base.options_base_strategy import OptionsBaseStrategy, OptionsSession
from trading_bot.strategies_new.options.base.spread_types import OptionType, VerticalSpreadType
from trading_bot.strategies_new.options.base.spread_analyzer import SpreadAnalyzer
from trading_bot.strategies_new.options.base.spread_manager import SpreadManager

# Configure logging
logger = logging.getLogger(__name__)

class ComplexSpreadType(Enum):
    """Types of complex option spreads supported by the engine."""
    IRON_CONDOR = "iron_condor"  # Bull put spread + bear call spread
    IRON_BUTTERFLY = "iron_butterfly"  # Bull put spread + bear call spread with same short strikes
    BUTTERFLY = "butterfly"  # Long 1 at lower strike, short 2 at middle strike, long 1 at higher strike
    CONDOR = "condor"  # Long 1 at lowest strike, short 1 at lower-mid, short 1 at upper-mid, long 1 at highest
    BROKEN_WING_BUTTERFLY = "broken_wing_butterfly"  # Asymmetric butterfly with uneven wings


class ComplexSpreadPosition:
    """Represents a complex spread position with multiple option legs."""
    
    def __init__(self, 
                spread_type: ComplexSpreadType,
                legs: List[Dict[str, Any]],
                quantity: int = 1,
                position_id: Optional[str] = None):
        """
        Initialize a complex spread position.
        
        Args:
            spread_type: Type of complex spread
            legs: List of option legs, each containing contract details
            quantity: Number of spreads in the position
            position_id: Optional unique identifier
        """
        self.spread_type = spread_type
        self.legs = legs
        self.quantity = quantity
        self.position_id = position_id or str(uuid.uuid4())
        
        # Position tracking
        self.entry_time = datetime.now()
        self.exit_time = None
        self.entry_prices = {leg['leg_id']: leg.get('entry_price') for leg in legs}
        self.exit_prices = {}
        self.status = "open"
        
        # Risk metrics
        self.max_profit = self._calculate_max_profit()
        self.max_loss = self._calculate_max_loss()
        self.breakeven_points = self._calculate_breakeven_points()
        
        logger.info(f"Created {spread_type.value} position with {len(legs)} legs, ID: {self.position_id}")
    
    def _calculate_max_profit(self) -> float:
        """Calculate maximum potential profit for this complex spread."""
        # For Iron Condor and Iron Butterfly
        if self.spread_type in [ComplexSpreadType.IRON_CONDOR, ComplexSpreadType.IRON_BUTTERFLY]:
            # Max profit for iron condor/butterfly is the net credit received
            net_credit = 0.0
            for leg in self.legs:
                if leg['action'] == 'buy':
                    net_credit -= leg['entry_price'] * 100  # Convert to dollars (100 shares per contract)
                else:  # 'sell'
                    net_credit += leg['entry_price'] * 100
                    
            return net_credit * self.quantity
            
        # For regular Butterfly
        elif self.spread_type == ComplexSpreadType.BUTTERFLY:
            # Get the strikes sorted
            call_legs = [leg for leg in self.legs if leg['option_type'] == OptionType.CALL]
            put_legs = [leg for leg in self.legs if leg['option_type'] == OptionType.PUT]
            
            # If it's a call butterfly
            if len(call_legs) >= 3:
                strikes = sorted([leg['strike'] for leg in call_legs])
                # Max profit = middle strike - lower strike - net debit
                if len(strikes) >= 3:
                    net_debit = 0.0
                    for leg in call_legs:
                        if leg['action'] == 'buy':
                            net_debit += leg['entry_price'] * 100
                        else:  # 'sell'
                            net_debit -= leg['entry_price'] * 100
                    
                    # For butterfly, max profit occurs at middle strike at expiration
                    return ((strikes[1] - strikes[0]) * 100 - net_debit) * self.quantity
            
            # If it's a put butterfly
            if len(put_legs) >= 3:
                strikes = sorted([leg['strike'] for leg in put_legs])
                # Max profit = middle strike - lower strike - net debit
                if len(strikes) >= 3:
                    net_debit = 0.0
                    for leg in put_legs:
                        if leg['action'] == 'buy':
                            net_debit += leg['entry_price'] * 100
                        else:  # 'sell'
                            net_debit -= leg['entry_price'] * 100
                    
                    # For butterfly, max profit occurs at middle strike at expiration
                    return ((strikes[1] - strikes[0]) * 100 - net_debit) * self.quantity
        
        # For Condor
        elif self.spread_type == ComplexSpreadType.CONDOR:
            # Implementation similar to butterfly but with different middle strikes
            # For simplicity, return placeholder
            return 0.0
        
        # Default case
        logger.warning(f"Max profit calculation not implemented for {self.spread_type.value}")
        return 0.0
    
    def _calculate_max_loss(self) -> float:
        """Calculate maximum potential loss for this complex spread."""
        # For Iron Condor
        if self.spread_type == ComplexSpreadType.IRON_CONDOR:
            # Get call and put legs
            call_legs = [leg for leg in self.legs if leg['option_type'] == OptionType.CALL]
            put_legs = [leg for leg in self.legs if leg['option_type'] == OptionType.PUT]
            
            # Calculate max loss for call spread
            call_max_loss = 0.0
            if len(call_legs) >= 2:
                call_strikes = sorted([(leg['strike'], leg['action']) for leg in call_legs])
                # Width of the spread - net credit
                call_width = abs(call_strikes[1][0] - call_strikes[0][0])
                
                call_net_credit = 0.0
                for leg in call_legs:
                    if leg['action'] == 'buy':
                        call_net_credit -= leg['entry_price']
                    else:  # 'sell'
                        call_net_credit += leg['entry_price']
                
                call_max_loss = (call_width - call_net_credit) * 100
            
            # Calculate max loss for put spread
            put_max_loss = 0.0
            if len(put_legs) >= 2:
                put_strikes = sorted([(leg['strike'], leg['action']) for leg in put_legs])
                # Width of the spread - net credit
                put_width = abs(put_strikes[1][0] - put_strikes[0][0])
                
                put_net_credit = 0.0
                for leg in put_legs:
                    if leg['action'] == 'buy':
                        put_net_credit -= leg['entry_price']
                    else:  # 'sell'
                        put_net_credit += leg['entry_price']
                
                put_max_loss = (put_width - put_net_credit) * 100
            
            # Max loss is the larger of the two sides' max loss
            return max(call_max_loss, put_max_loss) * self.quantity
        
        # For Iron Butterfly
        elif self.spread_type == ComplexSpreadType.IRON_BUTTERFLY:
            # Similar to iron condor but with middle strikes the same
            # For simplicity, use similar calculation as iron condor
            # This is a simplified approach; actual calculation would be more complex
            
            # Get call and put legs
            call_legs = [leg for leg in self.legs if leg['option_type'] == OptionType.CALL]
            put_legs = [leg for leg in self.legs if leg['option_type'] == OptionType.PUT]
            
            # Calculate max loss for a side
            max_loss = 0.0
            if len(call_legs) >= 2 and len(put_legs) >= 2:
                call_strikes = sorted([leg['strike'] for leg in call_legs])
                put_strikes = sorted([leg['strike'] for leg in put_legs])
                
                # Width of the wider spread
                width = max(abs(call_strikes[1] - call_strikes[0]), abs(put_strikes[1] - put_strikes[0]))
                
                # Calculate net credit
                net_credit = 0.0
                for leg in self.legs:
                    if leg['action'] == 'buy':
                        net_credit -= leg['entry_price']
                    else:  # 'sell'
                        net_credit += leg['entry_price']
                
                max_loss = (width - net_credit) * 100 * self.quantity
            
            return max_loss
        
        # For Butterfly
        elif self.spread_type == ComplexSpreadType.BUTTERFLY:
            # For a butterfly, max loss is typically the net debit paid
            net_debit = 0.0
            for leg in self.legs:
                if leg['action'] == 'buy':
                    net_debit += leg['entry_price'] * 100
                else:  # 'sell'
                    net_debit -= leg['entry_price'] * 100
            
            return net_debit * self.quantity
        
        # Default case
        logger.warning(f"Max loss calculation not implemented for {self.spread_type.value}")
        return 0.0
    
    def _calculate_breakeven_points(self) -> List[float]:
        """Calculate breakeven points for this complex spread."""
        breakeven_points = []
        
        # For Iron Condor
        if self.spread_type == ComplexSpreadType.IRON_CONDOR:
            # Get call and put legs
            call_legs = [leg for leg in self.legs if leg['option_type'] == OptionType.CALL]
            put_legs = [leg for leg in self.legs if leg['option_type'] == OptionType.PUT]
            
            # Sort legs by strike
            call_legs = sorted(call_legs, key=lambda x: x['strike'])
            put_legs = sorted(put_legs, key=lambda x: x['strike'])
            
            # Calculate net credit
            net_credit = 0.0
            for leg in self.legs:
                if leg['action'] == 'buy':
                    net_credit -= leg['entry_price']
                else:  # 'sell'
                    net_credit += leg['entry_price']
            
            # For a typical iron condor with 2 call legs and 2 put legs
            if len(call_legs) >= 2 and len(put_legs) >= 2:
                # Lower breakeven point (put side)
                # Short put strike - net credit
                short_put = next((leg for leg in put_legs if leg['action'] == 'sell'), None)
                if short_put:
                    lower_breakeven = short_put['strike'] - net_credit
                    breakeven_points.append(lower_breakeven)
                
                # Upper breakeven point (call side)
                # Short call strike + net credit
                short_call = next((leg for leg in call_legs if leg['action'] == 'sell'), None)
                if short_call:
                    upper_breakeven = short_call['strike'] + net_credit
                    breakeven_points.append(upper_breakeven)
        
        # For Iron Butterfly
        elif self.spread_type == ComplexSpreadType.IRON_BUTTERFLY:
            # For an iron butterfly, typically the short put and short call have the same strike
            short_legs = [leg for leg in self.legs if leg['action'] == 'sell']
            if len(short_legs) >= 2:
                middle_strike = short_legs[0]['strike']  # Both short strikes should be the same
                
                # Calculate net credit
                net_credit = 0.0
                for leg in self.legs:
                    if leg['action'] == 'buy':
                        net_credit -= leg['entry_price']
                    else:  # 'sell'
                        net_credit += leg['entry_price']
                
                # Lower breakeven = middle strike - net credit
                lower_breakeven = middle_strike - net_credit
                # Upper breakeven = middle strike + net credit
                upper_breakeven = middle_strike + net_credit
                
                breakeven_points = [lower_breakeven, upper_breakeven]
        
        # For Butterfly
        elif self.spread_type == ComplexSpreadType.BUTTERFLY:
            # For a call butterfly with long strikes at A and C, and short strike at B
            call_legs = [leg for leg in self.legs if leg['option_type'] == OptionType.CALL]
            put_legs = [leg for leg in self.legs if leg['option_type'] == OptionType.PUT]
            
            # Process call butterfly
            if len(call_legs) >= 3:
                strikes = sorted([leg['strike'] for leg in call_legs])
                if len(strikes) >= 3:
                    # Calculate net debit
                    net_debit = 0.0
                    for leg in call_legs:
                        if leg['action'] == 'buy':
                            net_debit += leg['entry_price']
                        else:  # 'sell'
                            net_debit -= leg['entry_price']
                    
                    # Lower breakeven = middle strike - net debit
                    lower_breakeven = strikes[1] - net_debit
                    # Upper breakeven = middle strike + net debit
                    upper_breakeven = strikes[1] + net_debit
                    
                    breakeven_points = [lower_breakeven, upper_breakeven]
            
            # Process put butterfly
            elif len(put_legs) >= 3:
                strikes = sorted([leg['strike'] for leg in put_legs])
                if len(strikes) >= 3:
                    # Calculate net debit
                    net_debit = 0.0
                    for leg in put_legs:
                        if leg['action'] == 'buy':
                            net_debit += leg['entry_price']
                        else:  # 'sell'
                            net_debit -= leg['entry_price']
                    
                    # Lower breakeven = middle strike - net debit
                    lower_breakeven = strikes[1] - net_debit
                    # Upper breakeven = middle strike + net debit
                    upper_breakeven = strikes[1] + net_debit
                    
                    breakeven_points = [lower_breakeven, upper_breakeven]
        
        return breakeven_points
    
    def update_prices(self, current_prices: Dict[str, float]):
        """
        Update current prices for all legs of the spread.
        
        Args:
            current_prices: Dictionary mapping leg IDs to current prices
        """
        for leg in self.legs:
            leg_id = leg['leg_id']
            if leg_id in current_prices:
                leg['current_price'] = current_prices[leg_id]
    
    def close_position(self, exit_prices: Dict[str, float], exit_reason: str):
        """
        Close the position and record exit details.
        
        Args:
            exit_prices: Dictionary mapping leg IDs to exit prices
            exit_reason: Reason for closing the position
        """
        if self.status != "open":
            logger.warning(f"Attempted to close position {self.position_id} that is already {self.status}")
            return
        
        self.exit_time = datetime.now()
        self.exit_prices = exit_prices
        self.status = "closed"
        
        # Calculate P&L
        # Implementation depends on specific spread type
        
        logger.info(f"Closed {self.spread_type.value} position {self.position_id}, reason: {exit_reason}")


class ComplexSpreadEngine(OptionsBaseStrategy):
    """
    Base engine for complex multi-leg options spread strategies.
    
    This engine provides the foundation for strategies like Iron Condors,
    Butterflies, and other multi-leg option spreads. It extends the
    OptionsBaseStrategy with functionality specific to complex spreads.
    """
    
    def __init__(self, session: OptionsSession, data_pipeline: DataPipeline, 
                 parameters: Dict[str, Any] = None):
        """
        Initialize the complex spread engine.
        
        Args:
            session: Options trading session
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        super().__init__(session, data_pipeline, parameters)
        
        # Default parameters specific to complex spreads
        default_complex_params = {
            # Iron Condor parameters
            'ic_width_call_side': 5,         # Width of call side in points/dollars
            'ic_width_put_side': 5,          # Width of put side in points/dollars
            'ic_short_delta_call': 0.30,     # Target delta for short call
            'ic_short_delta_put': -0.30,     # Target delta for short put
            
            # Butterfly parameters
            'butterfly_width': 5,            # Width between strikes in butterfly
            'butterfly_center_delta': 0.50,  # Target delta for center strike (ATM)
            
            # General complex spread parameters
            'max_leg_bid_ask_spread': 0.20,  # Maximum acceptable bid-ask spread for any leg
            'min_credit_received': 0.20,     # Minimum credit received as percentage of width
            'target_probability_profit': 0.65, # Target probability of profit
            
            # Risk parameters
            'max_loss_per_trade': 500,       # Maximum dollar loss per trade
            'profit_target_pct': 0.50,       # Take profit at this percentage of max profit
            'stop_loss_pct': 0.50,           # Stop loss at this percentage of max loss
            
            # Execution parameters
            'leg_execution_sequence': 'simultaneous', # How to execute legs: 'simultaneous', 'sequential'
        }
        
        # Update parameters with defaults for complex spreads
        for key, value in default_complex_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Initialize specialized components
        self.spread_analyzer = SpreadAnalyzer()
        self.spread_manager = SpreadManager()
        
        # Strategy state
        self.positions = []  # List of ComplexSpreadPosition objects
        self.market_state = "unknown"  # Can be 'trending', 'ranging', 'volatile', etc.
        
        logger.info(f"Initialized ComplexSpreadEngine for {session.symbol}")
    
    def construct_iron_condor(self, option_chain: pd.DataFrame, 
                            underlying_price: float) -> Optional[ComplexSpreadPosition]:
        """
        Construct an Iron Condor spread from the option chain.
        
        Args:
            option_chain: Option chain data
            underlying_price: Current price of the underlying asset
            
        Returns:
            ComplexSpreadPosition if successful, None otherwise
        """
        if option_chain is None or option_chain.empty:
            return None
        
        # Filter option chain for liquidity, open interest, etc.
        filtered_chain = self.filter_option_chains(option_chain)
        
        # Select expiration date
        expiration = self.select_expiration(filtered_chain)
        if not expiration:
            logger.warning("No suitable expiration found for Iron Condor")
            return None
        
        # Get options for selected expiration only
        exp_options = filtered_chain[filtered_chain['expiration_date'] == expiration]
        calls = exp_options[exp_options['option_type'] == 'call']
        puts = exp_options[exp_options['option_type'] == 'put']
        
        # Get target deltas for short strikes
        short_call_delta = self.parameters['ic_short_delta_call']
        short_put_delta = self.parameters['ic_short_delta_put']
        
        # Sort options to find closest to target deltas
        if 'delta' in calls.columns and 'delta' in puts.columns:
            calls['delta_diff'] = (calls['delta'] - short_call_delta).abs()
            puts['delta_diff'] = (puts['delta'] - short_put_delta).abs()
            
            calls = calls.sort_values('delta_diff')
            puts = puts.sort_values('delta_diff')
            
            # Get short strikes
            if not calls.empty and not puts.empty:
                short_call = calls.iloc[0]
                short_put = puts.iloc[0]
                
                # Get long strikes based on width
                call_width = self.parameters['ic_width_call_side']
                put_width = self.parameters['ic_width_put_side']
                
                long_call_strike = short_call['strike'] + call_width
                long_put_strike = short_put['strike'] - put_width
                
                # Find corresponding long options
                long_calls = calls[calls['strike'] >= long_call_strike].sort_values('strike')
                long_puts = puts[puts['strike'] <= long_put_strike].sort_values('strike', ascending=False)
                
                if not long_calls.empty and not long_puts.empty:
                    long_call = long_calls.iloc[0]
                    long_put = long_puts.iloc[0]
                    
                    # Create legs
                    legs = [
                        {
                            'leg_id': 'short_call',
                            'option_type': 'call',
                            'strike': short_call['strike'],
                            'expiration': expiration,
                            'action': 'sell',
                            'delta': short_call['delta'],
                            'entry_price': short_call['bid']  # Sell at bid price
                        },
                        {
                            'leg_id': 'long_call',
                            'option_type': 'call',
                            'strike': long_call['strike'],
                            'expiration': expiration,
                            'action': 'buy',
                            'delta': long_call['delta'],
                            'entry_price': long_call['ask']  # Buy at ask price
                        },
                        {
                            'leg_id': 'short_put',
                            'option_type': 'put',
                            'strike': short_put['strike'],
                            'expiration': expiration,
                            'action': 'sell',
                            'delta': short_put['delta'],
                            'entry_price': short_put['bid']  # Sell at bid price
                        },
                        {
                            'leg_id': 'long_put',
                            'option_type': 'put',
                            'strike': long_put['strike'],
                            'expiration': expiration,
                            'action': 'buy',
                            'delta': long_put['delta'],
                            'entry_price': long_put['ask']  # Buy at ask price
                        }
                    ]
                    
                    # Calculate credit received
                    net_credit = (short_call['bid'] - long_call['ask']) + (short_put['bid'] - long_put['ask'])
                    
                    # Check if credit is sufficient
                    min_credit = self.parameters['min_credit_received'] * (call_width + put_width)
                    if net_credit < min_credit:
                        logger.warning(f"Iron Condor credit ({net_credit:.2f}) below minimum ({min_credit:.2f})")
                        return None
                    
                    # Create position
                    return ComplexSpreadPosition(
                        spread_type=ComplexSpreadType.IRON_CONDOR,
                        legs=legs,
                        quantity=1
                    )
        
        logger.warning("Could not construct valid Iron Condor spread")
        return None
    
    def construct_butterfly(self, option_chain: pd.DataFrame, 
                         underlying_price: float) -> Optional[ComplexSpreadPosition]:
        """
        Construct a Butterfly spread from the option chain.
        
        Args:
            option_chain: Option chain data
            underlying_price: Current price of the underlying asset
            
        Returns:
            ComplexSpreadPosition if successful, None otherwise
        """
        # Implementation for butterfly construction
        # Similar to Iron Condor but with different structure
        # For now, return None as placeholder
        return None
    
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
        
        # Calculate market regime indicators
        if 'close' in data.columns:
            # 20-day moving average
            indicators['ma_20'] = data['close'].rolling(window=20).mean()
            
            # 50-day moving average
            indicators['ma_50'] = data['close'].rolling(window=50).mean()
            
            # 20-day Bollinger Bands
            ma_20 = indicators['ma_20']
            std_20 = data['close'].rolling(window=20).std()
            indicators['upper_band'] = ma_20 + (std_20 * 2)
            indicators['lower_band'] = ma_20 - (std_20 * 2)
            
            # Bollinger Band width (volatility indicator)
            indicators['bb_width'] = (indicators['upper_band'] - indicators['lower_band']) / ma_20
            
            # Determine market regime based on BB width
            bb_width = indicators['bb_width'].iloc[-1] if not indicators['bb_width'].empty else 0
            
            if bb_width > 0.1:  # Highly volatile (exact threshold would need calibration)
                self.market_state = "volatile"
            elif bb_width < 0.05:  # Low volatility
                if indicators['ma_20'].iloc[-1] > indicators['ma_50'].iloc[-1]:
                    self.market_state = "trending_up"
                elif indicators['ma_20'].iloc[-1] < indicators['ma_50'].iloc[-1]:
                    self.market_state = "trending_down"
                else:
                    self.market_state = "ranging"
            else:  # Medium volatility
                self.market_state = "normal"
                
            indicators['market_state'] = self.market_state
            
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for complex spreads based on market conditions.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            "entry": False,
            "spread_type": None,
            "exit_positions": [],
            "signal_strength": 0.0
        }
        
        if data.empty or not indicators:
            return signals
        
        # Get current market state
        market_state = indicators.get('market_state', 'unknown')
        
        # Iron Condor signals - ideal for ranging markets
        if market_state in ['ranging', 'normal']:
            if self.can_enter_position(ComplexSpreadType.IRON_CONDOR):
                signals["entry"] = True
                signals["spread_type"] = ComplexSpreadType.IRON_CONDOR
                signals["signal_strength"] = 0.7
        
        # Butterfly signals - ideal for precise price targets
        elif market_state == 'ranging' and 'hist_volatility_20d' in indicators:
            # Low volatility is ideal for butterflies
            volatility = indicators['hist_volatility_20d'].iloc[-1]
            if volatility < 0.2 and self.can_enter_position(ComplexSpreadType.BUTTERFLY):
                signals["entry"] = True
                signals["spread_type"] = ComplexSpreadType.BUTTERFLY
                signals["signal_strength"] = 0.65
        
        # Check for exits
        for position in self.positions:
            # Exit if market state changes dramatically
            if (position.spread_type == ComplexSpreadType.IRON_CONDOR and 
                market_state in ['trending_up', 'trending_down', 'volatile']):
                signals["exit_positions"].append(position.position_id)
        
        return signals
    
    def can_enter_position(self, spread_type: ComplexSpreadType) -> bool:
        """
        Check if a new position of the given type can be entered.
        
        Args:
            spread_type: Type of complex spread
            
        Returns:
            True if a position can be entered, False otherwise
        """
        # Count existing positions of this type
        count = sum(1 for pos in self.positions if pos.spread_type == spread_type)
        
        # Get maximum number of positions of this type
        max_positions = self.parameters.get(f'max_{spread_type.value}_positions', 1)
        
        return count < max_positions
    
    def _execute_signals(self):
        """Execute trading signals generated by the strategy."""
        if not self.signals:
            return
        
        # Handle entry signals
        if self.signals.get("entry", False) and "spread_type" in self.signals:
            spread_type = self.signals["spread_type"]
            underlying_price = self.session.current_price or self.market_data['close'].iloc[-1]
            
            # Construct the appropriate spread
            position = None
            if spread_type == ComplexSpreadType.IRON_CONDOR:
                position = self.construct_iron_condor(self.session.option_chain, underlying_price)
            elif spread_type == ComplexSpreadType.BUTTERFLY:
                position = self.construct_butterfly(self.session.option_chain, underlying_price)
            
            if position:
                # Add position to active positions
                self.positions.append(position)
                logger.info(f"Opened {spread_type.value} position {position.position_id}")
        
        # Handle exit signals
        for position_id in self.signals.get("exit_positions", []):
            for i, position in enumerate(self.positions):
                if position.position_id == position_id:
                    # Get current prices for each leg
                    exit_prices = {}
                    
                    # In a real implementation, would get actual prices from option chain
                    # For now, just set some placeholder values for demonstration
                    for leg in position.legs:
                        if leg['action'] == 'buy':
                            # When exiting a long leg, sell at bid
                            exit_prices[leg['leg_id']] = leg['entry_price'] * 1.05  # Placeholder
                        else:
                            # When exiting a short leg, buy at ask
                            exit_prices[leg['leg_id']] = leg['entry_price'] * 0.95  # Placeholder
                    
                    # Close the position
                    position.close_position(exit_prices, "signal_generated")
                    
                    # Remove from active positions
                    self.positions.pop(i)
                    break
