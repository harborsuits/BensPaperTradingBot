#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volatility Spread Engine Module

This module provides the foundation for options strategies that exploit volatility,
such as Straddles and Strangles, which benefit from significant price movement
in either direction.
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
from trading_bot.strategies_new.options.base.spread_types import OptionType
from trading_bot.strategies_new.options.base.spread_analyzer import SpreadAnalyzer
from trading_bot.strategies_new.options.base.spread_manager import SpreadManager

# Configure logging
logger = logging.getLogger(__name__)

class VolatilitySpreadType(Enum):
    """Types of volatility-based option spreads supported by the engine."""
    LONG_STRADDLE = "long_straddle"    # Long call + long put at same strike
    SHORT_STRADDLE = "short_straddle"  # Short call + short put at same strike
    LONG_STRANGLE = "long_strangle"    # Long call + long put at different strikes
    SHORT_STRANGLE = "short_strangle"  # Short call + short put at different strikes


class VolatilitySpreadPosition:
    """Represents a volatility spread position with call and put options."""
    
    def __init__(self, 
                spread_type: VolatilitySpreadType,
                call_leg: Dict[str, Any],
                put_leg: Dict[str, Any],
                quantity: int = 1,
                position_id: Optional[str] = None):
        """
        Initialize a volatility spread position.
        
        Args:
            spread_type: Type of volatility spread
            call_leg: Call option details
            put_leg: Put option details
            quantity: Number of spreads in the position
            position_id: Optional unique identifier
        """
        self.spread_type = spread_type
        self.call_leg = call_leg
        self.put_leg = put_leg
        self.quantity = quantity
        self.position_id = position_id or str(uuid.uuid4())
        
        # Position tracking
        self.entry_time = datetime.now()
        self.exit_time = None
        self.status = "open"
        
        # Risk metrics
        self.max_profit = None  # Unlimited for long straddle/strangle
        self.max_loss = self._calculate_max_loss()
        self.breakeven_points = self._calculate_breakeven_points()
        
        logger.info(f"Created {spread_type.value} position with ID: {self.position_id}")
    
    def _calculate_max_loss(self) -> float:
        """Calculate maximum potential loss for this volatility spread."""
        is_long = self.spread_type in [VolatilitySpreadType.LONG_STRADDLE, VolatilitySpreadType.LONG_STRANGLE]
        
        if is_long:
            # For long straddle/strangle, max loss is the premium paid
            return (self.call_leg['entry_price'] + self.put_leg['entry_price']) * 100 * self.quantity
        else:
            # For short straddle/strangle, max loss is theoretically unlimited
            # But we'll estimate based on a large move (e.g., 3 standard deviations)
            # For now, return a placeholder
            return float('inf')
    
    def _calculate_breakeven_points(self) -> List[float]:
        """Calculate breakeven points for this volatility spread."""
        is_long = self.spread_type in [VolatilitySpreadType.LONG_STRADDLE, VolatilitySpreadType.LONG_STRANGLE]
        
        # Calculate total premium
        total_premium = self.call_leg['entry_price'] + self.put_leg['entry_price']
        
        if self.spread_type in [VolatilitySpreadType.LONG_STRADDLE, VolatilitySpreadType.SHORT_STRADDLE]:
            # For straddles, strike prices are the same
            strike = self.call_leg['strike']
            
            if is_long:
                # For long straddle: strike ± total_premium
                lower_breakeven = strike - total_premium
                upper_breakeven = strike + total_premium
            else:
                # For short straddle: strike ± total_premium
                lower_breakeven = strike - total_premium
                upper_breakeven = strike + total_premium
                
            return [lower_breakeven, upper_breakeven]
        else:
            # For strangles, strike prices are different
            call_strike = self.call_leg['strike']
            put_strike = self.put_leg['strike']
            
            if is_long:
                # For long strangle: put_strike - put_premium, call_strike + call_premium
                lower_breakeven = put_strike - self.put_leg['entry_price']
                upper_breakeven = call_strike + self.call_leg['entry_price']
            else:
                # For short strangle: put_strike - put_premium, call_strike + call_premium
                lower_breakeven = put_strike - self.put_leg['entry_price']
                upper_breakeven = call_strike + self.call_leg['entry_price']
                
            return [lower_breakeven, upper_breakeven]
    
    def update_prices(self, call_price: float, put_price: float):
        """
        Update current prices for both legs of the spread.
        
        Args:
            call_price: Current price of call option
            put_price: Current price of put option
        """
        self.call_leg['current_price'] = call_price
        self.put_leg['current_price'] = put_price
    
    def close_position(self, call_price: float, put_price: float, exit_reason: str):
        """
        Close the position and record exit details.
        
        Args:
            call_price: Exit price for call option
            put_price: Exit price for put option
            exit_reason: Reason for closing the position
        """
        if self.status != "open":
            logger.warning(f"Attempted to close position {self.position_id} that is already {self.status}")
            return
        
        self.exit_time = datetime.now()
        self.call_leg['exit_price'] = call_price
        self.put_leg['exit_price'] = put_price
        self.status = "closed"
        
        # Calculate P&L
        is_long = self.spread_type in [VolatilitySpreadType.LONG_STRADDLE, VolatilitySpreadType.LONG_STRANGLE]
        
        if is_long:
            # For long straddle/strangle
            call_pnl = (call_price - self.call_leg['entry_price']) * 100 * self.quantity
            put_pnl = (put_price - self.put_leg['entry_price']) * 100 * self.quantity
        else:
            # For short straddle/strangle
            call_pnl = (self.call_leg['entry_price'] - call_price) * 100 * self.quantity
            put_pnl = (self.put_leg['entry_price'] - put_price) * 100 * self.quantity
        
        total_pnl = call_pnl + put_pnl
        self.profit_loss = total_pnl
        
        logger.info(f"Closed {self.spread_type.value} position {self.position_id}, P&L: ${total_pnl:.2f}, reason: {exit_reason}")


class VolatilitySpreadEngine(OptionsBaseStrategy):
    """
    Base engine for volatility-based options spread strategies.
    
    This engine provides the foundation for strategies like Straddles and Strangles,
    which profit from significant price movement in either direction.
    """
    
    def __init__(self, session: OptionsSession, data_pipeline: DataPipeline, 
                 parameters: Dict[str, Any] = None):
        """
        Initialize the volatility spread engine.
        
        Args:
            session: Options trading session
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        super().__init__(session, data_pipeline, parameters)
        
        # Default parameters specific to volatility spreads
        default_vol_params = {
            # General volatility parameters
            'expected_move_factor': 1.5,   # Expected move as factor of historical volatility
            'min_volatility': 0.20,        # Minimum historical volatility
            'iv_rank_threshold': 30,       # IV rank below this is low (good for long vol)
            
            # Strategy specific parameters
            'strangle_strike_width': 1.0,  # Width of strangle strikes as ATR multiple
            'use_historical_vol': True,    # Use historical vol for expected move
            'use_iv_percentile': True,     # Use IV percentile for strategy selection
            
            # Risk parameters
            'max_loss_per_trade': 1000,    # Maximum dollar loss per trade
            'profit_target_pct': 0.50,     # Take profit at this percentage of max potential
            'stop_loss_pct': 0.50,         # Stop loss at this percentage of max loss
            
            # Position sizing
            'account_risk_pct': 0.02,      # Risk percentage of account per trade
            'position_sizing_method': 'risk_based',  # "fixed" or "risk_based"
            
            # DTE parameters
            'min_dte': 30,                 # Minimum days to expiration
            'max_dte': 60,                 # Maximum days to expiration
            'target_dte': 45,              # Target days to expiration
        }
        
        # Update parameters with defaults for volatility spreads
        for key, value in default_vol_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Initialize specialized components
        self.spread_analyzer = SpreadAnalyzer()
        self.spread_manager = SpreadManager()
        
        # Strategy state
        self.positions = []  # List of VolatilitySpreadPosition objects
        self.historical_volatility = None
        self.expected_move = None
        
        logger.info(f"Initialized VolatilitySpreadEngine for {session.symbol}")
    
    def construct_straddle(self, option_chain: pd.DataFrame, 
                        underlying_price: float,
                        is_long: bool = True) -> Optional[VolatilitySpreadPosition]:
        """
        Construct a Straddle from the option chain.
        
        Args:
            option_chain: Option chain data
            underlying_price: Current price of the underlying asset
            is_long: Whether to create a long straddle (True) or short straddle (False)
            
        Returns:
            VolatilitySpreadPosition if successful, None otherwise
        """
        if option_chain is None or option_chain.empty:
            return None
        
        # Filter option chain for liquidity, open interest, etc.
        filtered_chain = self.filter_option_chains(option_chain)
        
        # Select expiration date
        expiration = self.select_expiration(filtered_chain)
        if not expiration:
            logger.warning("No suitable expiration found for Straddle")
            return None
        
        # Get options for selected expiration only
        exp_options = filtered_chain[filtered_chain['expiration_date'] == expiration]
        calls = exp_options[exp_options['option_type'] == 'call']
        puts = exp_options[exp_options['option_type'] == 'put']
        
        # Find strike closest to ATM (at-the-money)
        strikes = sorted(exp_options['strike'].unique())
        atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))
        
        # Get ATM call and put
        atm_calls = calls[calls['strike'] == atm_strike]
        atm_puts = puts[puts['strike'] == atm_strike]
        
        if atm_calls.empty or atm_puts.empty:
            logger.warning(f"No ATM options found at strike {atm_strike}")
            return None
        
        # Get first ATM call and put
        atm_call = atm_calls.iloc[0]
        atm_put = atm_puts.iloc[0]
        
        # Create leg details
        if is_long:
            # Long straddle - buy both options
            call_leg = {
                'leg_id': 'call_leg',
                'option_type': 'call',
                'strike': atm_strike,
                'expiration': expiration,
                'action': 'buy',
                'entry_price': atm_call['ask'],  # Buy at ask
                'delta': atm_call.get('delta', 0.5)
            }
            
            put_leg = {
                'leg_id': 'put_leg',
                'option_type': 'put',
                'strike': atm_strike,
                'expiration': expiration,
                'action': 'buy',
                'entry_price': atm_put['ask'],  # Buy at ask
                'delta': atm_put.get('delta', -0.5)
            }
            
            spread_type = VolatilitySpreadType.LONG_STRADDLE
        else:
            # Short straddle - sell both options
            call_leg = {
                'leg_id': 'call_leg',
                'option_type': 'call',
                'strike': atm_strike,
                'expiration': expiration,
                'action': 'sell',
                'entry_price': atm_call['bid'],  # Sell at bid
                'delta': atm_call.get('delta', 0.5)
            }
            
            put_leg = {
                'leg_id': 'put_leg',
                'option_type': 'put',
                'strike': atm_strike,
                'expiration': expiration,
                'action': 'sell',
                'entry_price': atm_put['bid'],  # Sell at bid
                'delta': atm_put.get('delta', -0.5)
            }
            
            spread_type = VolatilitySpreadType.SHORT_STRADDLE
        
        # Calculate total premium
        total_premium = call_leg['entry_price'] + put_leg['entry_price']
        
        # Check against risk parameters
        max_risk = total_premium * 100  # For long straddle
        if is_long and max_risk > self.parameters['max_loss_per_trade']:
            logger.warning(f"Straddle premium (${max_risk:.2f}) exceeds max loss per trade")
            return None
        
        # Create position
        return VolatilitySpreadPosition(
            spread_type=spread_type,
            call_leg=call_leg,
            put_leg=put_leg,
            quantity=1
        )
    
    def construct_strangle(self, option_chain: pd.DataFrame, 
                        underlying_price: float,
                        is_long: bool = True) -> Optional[VolatilitySpreadPosition]:
        """
        Construct a Strangle from the option chain.
        
        Args:
            option_chain: Option chain data
            underlying_price: Current price of the underlying asset
            is_long: Whether to create a long strangle (True) or short strangle (False)
            
        Returns:
            VolatilitySpreadPosition if successful, None otherwise
        """
        if option_chain is None or option_chain.empty:
            return None
        
        # Filter option chain for liquidity, open interest, etc.
        filtered_chain = self.filter_option_chains(option_chain)
        
        # Select expiration date
        expiration = self.select_expiration(filtered_chain)
        if not expiration:
            logger.warning("No suitable expiration found for Strangle")
            return None
        
        # Get options for selected expiration only
        exp_options = filtered_chain[filtered_chain['expiration_date'] == expiration]
        calls = exp_options[exp_options['option_type'] == 'call']
        puts = exp_options[exp_options['option_type'] == 'put']
        
        # Determine width based on volatility
        width_factor = self.parameters['strangle_strike_width']
        
        # Use ATR if available, otherwise just use a percentage of price
        if hasattr(self, 'market_data') and not self.market_data.empty:
            # Calculate ATR if not already done
            if 'atr' not in self.market_data.columns:
                high = self.market_data['high']
                low = self.market_data['low']
                close = self.market_data['close']
                
                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                
                tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
                atr = tr.rolling(window=14).mean().iloc[-1]
            else:
                atr = self.market_data['atr'].iloc[-1]
                
            width = atr * width_factor
        else:
            # Fallback to percentage of price
            width = underlying_price * 0.03 * width_factor
        
        # Find OTM strikes
        call_strike = None
        put_strike = None
        
        strikes = sorted(exp_options['strike'].unique())
        for strike in strikes:
            if strike > underlying_price and call_strike is None:
                # First strike above underlying for call
                if strike >= underlying_price + width:
                    call_strike = strike
            
            if strike < underlying_price and (put_strike is None or strike > put_strike):
                # Highest strike below underlying for put
                if strike <= underlying_price - width:
                    put_strike = strike
        
        if call_strike is None or put_strike is None:
            logger.warning(f"Could not find suitable OTM strikes for strangle")
            return None
        
        # Get OTM options
        otm_calls = calls[calls['strike'] == call_strike]
        otm_puts = puts[puts['strike'] == put_strike]
        
        if otm_calls.empty or otm_puts.empty:
            logger.warning(f"No options found at selected strikes {call_strike}/{put_strike}")
            return None
        
        # Get first OTM call and put
        otm_call = otm_calls.iloc[0]
        otm_put = otm_puts.iloc[0]
        
        # Create leg details
        if is_long:
            # Long strangle - buy both options
            call_leg = {
                'leg_id': 'call_leg',
                'option_type': 'call',
                'strike': call_strike,
                'expiration': expiration,
                'action': 'buy',
                'entry_price': otm_call['ask'],  # Buy at ask
                'delta': otm_call.get('delta', 0.3)
            }
            
            put_leg = {
                'leg_id': 'put_leg',
                'option_type': 'put',
                'strike': put_strike,
                'expiration': expiration,
                'action': 'buy',
                'entry_price': otm_put['ask'],  # Buy at ask
                'delta': otm_put.get('delta', -0.3)
            }
            
            spread_type = VolatilitySpreadType.LONG_STRANGLE
        else:
            # Short strangle - sell both options
            call_leg = {
                'leg_id': 'call_leg',
                'option_type': 'call',
                'strike': call_strike,
                'expiration': expiration,
                'action': 'sell',
                'entry_price': otm_call['bid'],  # Sell at bid
                'delta': otm_call.get('delta', 0.3)
            }
            
            put_leg = {
                'leg_id': 'put_leg',
                'option_type': 'put',
                'strike': put_strike,
                'expiration': expiration,
                'action': 'sell',
                'entry_price': otm_put['bid'],  # Sell at bid
                'delta': otm_put.get('delta', -0.3)
            }
            
            spread_type = VolatilitySpreadType.SHORT_STRANGLE
        
        # Calculate total premium
        total_premium = call_leg['entry_price'] + put_leg['entry_price']
        
        # Check against risk parameters
        max_risk = total_premium * 100  # For long strangle
        if is_long and max_risk > self.parameters['max_loss_per_trade']:
            logger.warning(f"Strangle premium (${max_risk:.2f}) exceeds max loss per trade")
            return None
        
        # Create position
        return VolatilitySpreadPosition(
            spread_type=spread_type,
            call_leg=call_leg,
            put_leg=put_leg,
            quantity=1
        )
    
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
            hist_vol_20d = data['returns'].rolling(window=20).std() * np.sqrt(252)
            indicators['hist_volatility_20d'] = hist_vol_20d
            
            # Store for future reference
            self.historical_volatility = hist_vol_20d.iloc[-1]
            
            # Calculate expected move
            expected_move_factor = self.parameters['expected_move_factor']
            dte = self.parameters['target_dte']
            current_price = data['close'].iloc[-1]
            
            # Expected move = price * volatility * sqrt(dte/365) * factor
            expected_move = current_price * self.historical_volatility * np.sqrt(dte/365) * expected_move_factor
            indicators['expected_move'] = expected_move
            self.expected_move = expected_move
        
        # Volatility indicators for strategy selection
        if 'close' in data.columns:
            # Calculate recent price movement
            indicators['price_5d_change'] = data['close'].pct_change(periods=5) * 100
            indicators['price_20d_change'] = data['close'].pct_change(periods=20) * 100
            
            # Calculate Bollinger Band width as volatility indicator
            ma_20 = data['close'].rolling(window=20).mean()
            std_20 = data['close'].rolling(window=20).std()
            upper_band = ma_20 + (std_20 * 2)
            lower_band = ma_20 - (std_20 * 2)
            indicators['bb_width'] = (upper_band - lower_band) / ma_20
            
            # Calculate volatility trend (increasing or decreasing)
            if len(hist_vol_20d) > 5:
                vol_5d_ago = hist_vol_20d.iloc[-6] if len(hist_vol_20d) > 5 else hist_vol_20d.iloc[0]
                current_vol = hist_vol_20d.iloc[-1]
                
                indicators['vol_trend'] = 'increasing' if current_vol > vol_5d_ago else 'decreasing'
                indicators['vol_change_pct'] = (current_vol / vol_5d_ago - 1) * 100
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for volatility spreads based on market conditions.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            "entry": False,
            "spread_type": None,
            "is_long": None,
            "exit_positions": [],
            "signal_strength": 0.0
        }
        
        if data.empty or not indicators:
            return signals
        
        # Get IV metrics if available
        iv_percentile = None
        iv_rank = None
        
        if self.session.current_iv is not None and self.session.symbol in self.iv_history:
            iv_metrics = self.calculate_implied_volatility_metrics(self.iv_history[self.session.symbol])
            iv_percentile = iv_metrics.get('iv_percentile')
            iv_rank = iv_metrics.get('iv_rank')
        
        # Check historical volatility
        if 'hist_volatility_20d' in indicators:
            hist_vol = indicators['hist_volatility_20d'].iloc[-1]
            min_vol = self.parameters['min_volatility']
            
            # Decide between long and short volatility strategies
            is_long = True
            
            if iv_percentile is not None and self.parameters['use_iv_percentile']:
                # Low IV percentile is good for long vol strategies
                if iv_percentile < self.parameters['iv_rank_threshold']:
                    is_long = True
                    
                    # Low IV + increasing volatility is a strong signal for long vol
                    if indicators.get('vol_trend') == 'increasing':
                        signals["entry"] = True
                        signals["spread_type"] = VolatilitySpreadType.LONG_STRADDLE
                        signals["is_long"] = True
                        signals["signal_strength"] = 0.8
                else:
                    # High IV is better for short vol strategies, but more risk
                    is_long = False
            
            # If no strong signal yet, check volatility change
            if not signals["entry"] and 'vol_change_pct' in indicators:
                vol_change = indicators['vol_change_pct']
                
                if is_long and vol_change > 20:
                    # Significant increase in volatility, prefer strangle for cheaper entry
                    signals["entry"] = True
                    signals["spread_type"] = VolatilitySpreadType.LONG_STRANGLE
                    signals["is_long"] = True
                    signals["signal_strength"] = 0.7
                elif not is_long and vol_change < -20:
                    # Significant decrease in volatility, potential short vol opportunity
                    signals["entry"] = True
                    signals["spread_type"] = VolatilitySpreadType.SHORT_STRANGLE
                    signals["is_long"] = False
                    signals["signal_strength"] = 0.65
        
        # Upcoming events often increase volatility
        # In a real implementation, would check for upcoming earnings, etc.
        
        # Check for exits
        for position in self.positions:
            if position.status == "open":
                # Exit if volatility regime changes significantly
                if 'vol_trend' in indicators:
                    vol_trend = indicators['vol_trend']
                    
                    is_long_position = position.spread_type in [
                        VolatilitySpreadType.LONG_STRADDLE, 
                        VolatilitySpreadType.LONG_STRANGLE
                    ]
                    
                    # Exit long vol positions if volatility is decreasing significantly
                    if is_long_position and vol_trend == 'decreasing' and indicators.get('vol_change_pct', 0) < -15:
                        signals["exit_positions"].append(position.position_id)
                    
                    # Exit short vol positions if volatility is increasing significantly
                    if not is_long_position and vol_trend == 'increasing' and indicators.get('vol_change_pct', 0) > 15:
                        signals["exit_positions"].append(position.position_id)
        
        return signals
    
    def _execute_signals(self):
        """Execute trading signals generated by the strategy."""
        if not self.signals:
            return
        
        # Handle entry signals
        if self.signals.get("entry", False) and "spread_type" in self.signals:
            spread_type = self.signals["spread_type"]
            is_long = self.signals.get("is_long", True)
            underlying_price = self.session.current_price or self.market_data['close'].iloc[-1]
            
            # Construct the appropriate spread
            position = None
            if spread_type in [VolatilitySpreadType.LONG_STRADDLE, VolatilitySpreadType.SHORT_STRADDLE]:
                position = self.construct_straddle(self.session.option_chain, underlying_price, is_long)
            elif spread_type in [VolatilitySpreadType.LONG_STRANGLE, VolatilitySpreadType.SHORT_STRANGLE]:
                position = self.construct_strangle(self.session.option_chain, underlying_price, is_long)
            
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
                    call_exit_price = position.call_leg['entry_price'] * 1.1  # Placeholder
                    put_exit_price = position.put_leg['entry_price'] * 1.1    # Placeholder
                    
                    # Close the position
                    position.close_position(call_exit_price, put_exit_price, "signal_generated")
                    logger.info(f"Closed position {position_id} based on signals")
