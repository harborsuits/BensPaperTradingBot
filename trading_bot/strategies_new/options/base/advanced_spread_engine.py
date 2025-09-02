#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Spread Engine Module

This module provides the foundation for advanced multi-leg options strategies,
including Ratio Spreads, Jade Lizards, Broken Wing Butterflies, and Iron Butterflies.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union
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

class AdvancedSpreadType(Enum):
    """Types of advanced option spreads supported by the engine."""
    RATIO_CALL_SPREAD = "ratio_call_spread"  # Unbalanced ratio of long/short calls
    RATIO_PUT_SPREAD = "ratio_put_spread"    # Unbalanced ratio of long/short puts
    JADE_LIZARD = "jade_lizard"              # Short put + short call spread
    BROKEN_WING_BUTTERFLY = "broken_wing_butterfly"  # Asymmetric butterfly
    IRON_BUTTERFLY = "iron_butterfly"        # Bull put + bear call spread


class OptionLeg:
    """Represents a single leg in a multi-leg options position."""
    
    def __init__(self, 
                option_type: str,
                strike: float,
                expiration: str,
                action: str,
                quantity: int = 1,
                entry_price: Optional[float] = None,
                delta: Optional[float] = None,
                leg_id: Optional[str] = None):
        """
        Initialize an option leg.
        
        Args:
            option_type: 'call' or 'put'
            strike: Strike price
            expiration: Expiration date
            action: 'buy' or 'sell'
            quantity: Number of contracts
            entry_price: Option premium
            delta: Option delta (for risk calculation)
            leg_id: Optional unique identifier
        """
        self.option_type = option_type
        self.strike = strike
        self.expiration = expiration
        self.action = action
        self.quantity = quantity
        self.entry_price = entry_price
        self.delta = delta
        self.leg_id = leg_id or str(uuid.uuid4())
        
        # Additional tracking
        self.current_price = entry_price
        self.exit_price = None
        
    def __repr__(self):
        return (f"{self.action.upper()} {self.quantity} {self.option_type.upper()} "
                f"@{self.strike} EXP:{self.expiration}")


class AdvancedSpreadPosition:
    """Represents a complex multi-leg options position."""
    
    def __init__(self, 
                spread_type: AdvancedSpreadType,
                legs: List[OptionLeg],
                position_id: Optional[str] = None):
        """
        Initialize an advanced spread position.
        
        Args:
            spread_type: Type of advanced spread
            legs: List of option legs in the position
            position_id: Optional unique identifier
        """
        self.spread_type = spread_type
        self.legs = legs
        self.position_id = position_id or str(uuid.uuid4())
        
        # Position tracking
        self.entry_time = datetime.now()
        self.exit_time = None
        self.status = "open"
        
        # Risk metrics
        self.max_profit = self._calculate_max_profit()
        self.max_loss = self._calculate_max_loss()
        self.breakeven_points = self._calculate_breakeven_points()
        
        logger.info(f"Created {spread_type.value} position with ID: {self.position_id}")
    
    def _calculate_max_profit(self) -> Optional[float]:
        """Calculate maximum potential profit for this advanced spread."""
        # Implementation varies by spread type
        if self.spread_type == AdvancedSpreadType.RATIO_CALL_SPREAD:
            # For ratio call spreads, max profit depends on the ratio
            return None  # Complex calculation, implement later
        elif self.spread_type == AdvancedSpreadType.RATIO_PUT_SPREAD:
            return None  # Complex calculation, implement later
        else:
            return None  # To be implemented for other spread types
    
    def _calculate_max_loss(self) -> float:
        """Calculate maximum potential loss for this advanced spread."""
        # Implementation varies by spread type
        if self.spread_type == AdvancedSpreadType.RATIO_CALL_SPREAD:
            # For ratio call spreads with more shorts than longs, risk can be unlimited
            return float('inf')  # Simplified, would need more complex calculation
        else:
            return 0.0  # To be implemented for other spread types
    
    def _calculate_breakeven_points(self) -> List[float]:
        """Calculate breakeven points for this advanced spread."""
        # Implementation varies by spread type
        return []  # To be implemented for each spread type
    
    def update_prices(self, leg_prices: Dict[str, float]):
        """
        Update current prices for all legs of the spread.
        
        Args:
            leg_prices: Dictionary mapping leg_id to current price
        """
        for leg in self.legs:
            if leg.leg_id in leg_prices:
                leg.current_price = leg_prices[leg.leg_id]
    
    def close_position(self, leg_prices: Dict[str, float], exit_reason: str):
        """
        Close the position and record exit details.
        
        Args:
            leg_prices: Dictionary mapping leg_id to exit price
            exit_reason: Reason for closing the position
        """
        if self.status != "open":
            logger.warning(f"Attempted to close position {self.position_id} that is already {self.status}")
            return
        
        self.exit_time = datetime.now()
        total_pnl = 0.0
        
        for leg in self.legs:
            if leg.leg_id in leg_prices:
                leg.exit_price = leg_prices[leg.leg_id]
                
                # Calculate P&L for this leg
                if leg.action == 'buy':
                    leg_pnl = (leg.exit_price - leg.entry_price) * 100 * leg.quantity
                else:  # 'sell'
                    leg_pnl = (leg.entry_price - leg.exit_price) * 100 * leg.quantity
                
                total_pnl += leg_pnl
            else:
                logger.warning(f"No exit price provided for leg {leg.leg_id}")
        
        self.status = "closed"
        self.profit_loss = total_pnl
        
        logger.info(f"Closed {self.spread_type.value} position {self.position_id}, P&L: ${total_pnl:.2f}, reason: {exit_reason}")


class AdvancedSpreadEngine(OptionsBaseStrategy):
    """
    Base engine for advanced multi-leg options spread strategies.
    
    This engine provides the foundation for complex strategies like Ratio Spreads,
    Jade Lizards, Broken Wing Butterflies, and Iron Butterflies.
    """
    
    def __init__(self, session: OptionsSession, data_pipeline: DataPipeline, 
                 parameters: Dict[str, Any] = None):
        """
        Initialize the advanced spread engine.
        
        Args:
            session: Options trading session
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        super().__init__(session, data_pipeline, parameters)
        
        # Default parameters specific to advanced spreads
        default_adv_params = {
            # Ratio spread parameters
            'ratio_spread_ratio': 1.5,      # Ratio for ratio spreads (e.g., 1:1.5)
            'ratio_strike_width': 1.0,      # Width between strikes as ATR multiple
            
            # Risk parameters
            'max_loss_per_trade': 1000,     # Maximum dollar loss per trade
            'profit_target_pct': 0.50,      # Take profit at this percentage of max potential
            'stop_loss_pct': 0.50,          # Stop loss at this percentage of max loss
            
            # Position sizing
            'account_risk_pct': 0.02,       # Risk percentage of account per trade
            'position_sizing_method': 'risk_based',  # "fixed" or "risk_based"
            
            # DTE parameters
            'min_dte': 30,                  # Minimum days to expiration
            'max_dte': 60,                  # Maximum days to expiration
            'target_dte': 45,               # Target days to expiration
            
            # Strategy-specific parameters to be added later
        }
        
        # Update parameters with defaults for advanced spreads
        for key, value in default_adv_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Initialize specialized components
        self.spread_analyzer = SpreadAnalyzer()
        self.spread_manager = SpreadManager()
        
        # Strategy state
        self.positions = []  # List of AdvancedSpreadPosition objects
        
        logger.info(f"Initialized AdvancedSpreadEngine for {session.symbol}")
    
    def construct_ratio_spread(self, option_chain: pd.DataFrame, 
                            underlying_price: float,
                            option_type: str = 'call') -> Optional[AdvancedSpreadPosition]:
        """
        Construct a Ratio Spread from the option chain.
        
        Args:
            option_chain: Option chain data
            underlying_price: Current price of the underlying asset
            option_type: 'call' or 'put'
            
        Returns:
            AdvancedSpreadPosition if successful, None otherwise
        """
        if option_chain is None or option_chain.empty:
            return None
        
        # Filter option chain for liquidity, open interest, etc.
        filtered_chain = self.filter_option_chains(option_chain)
        
        # Select expiration date
        expiration = self.select_expiration(filtered_chain)
        if not expiration:
            logger.warning("No suitable expiration found for Ratio Spread")
            return None
        
        # Get options for selected expiration only
        exp_options = filtered_chain[filtered_chain['expiration_date'] == expiration]
        type_options = exp_options[exp_options['option_type'] == option_type]
        
        if type_options.empty:
            logger.warning(f"No {option_type} options found for selected expiration")
            return None
        
        # Ratio spread parameters
        ratio = self.parameters['ratio_spread_ratio']
        long_qty = 1
        short_qty = int(long_qty * ratio) if ratio > 1 else 1
        
        if short_qty <= long_qty:
            logger.warning(f"Invalid ratio: {ratio}. Short quantity must be greater than long quantity.")
            return None
        
        # Select strikes based on option type
        if option_type == 'call':
            # For call ratio spread, buy lower strike, sell higher strike
            strikes = sorted(type_options['strike'].unique())
            
            # Find strike closest to ATM
            atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))
            atm_index = strikes.index(atm_strike)
            
            # Determine strike width based on ATR or price percentage
            width_factor = self.parameters['ratio_strike_width']
            
            if hasattr(self, 'market_data') and not self.market_data.empty:
                if 'atr' in self.market_data.columns:
                    atr = self.market_data['atr'].iloc[-1]
                    width = atr * width_factor
                else:
                    width = underlying_price * 0.03 * width_factor
            else:
                width = underlying_price * 0.03 * width_factor
            
            # Find suitable OTM strike
            long_strike = atm_strike
            short_strike = None
            
            for strike in strikes[atm_index:]:
                if strike >= long_strike + width:
                    short_strike = strike
                    break
            
            if short_strike is None:
                logger.warning("Could not find suitable OTM strike for call ratio spread")
                return None
            
            # Get option data for selected strikes
            long_options = type_options[type_options['strike'] == long_strike]
            short_options = type_options[type_options['strike'] == short_strike]
            
            if long_options.empty or short_options.empty:
                logger.warning(f"No options found at selected strikes {long_strike}/{short_strike}")
                return None
            
            # Create option legs
            long_leg = OptionLeg(
                option_type=option_type,
                strike=long_strike,
                expiration=expiration,
                action='buy',
                quantity=long_qty,
                entry_price=long_options['ask'].iloc[0],  # Buy at ask
                delta=long_options.get('delta', [0.5])[0],
                leg_id='long_leg'
            )
            
            short_leg = OptionLeg(
                option_type=option_type,
                strike=short_strike,
                expiration=expiration,
                action='sell',
                quantity=short_qty,
                entry_price=short_options['bid'].iloc[0],  # Sell at bid
                delta=short_options.get('delta', [0.3])[0],
                leg_id='short_leg'
            )
            
            # Create position
            spread_type = AdvancedSpreadType.RATIO_CALL_SPREAD
            
        else:  # option_type == 'put'
            # For put ratio spread, buy higher strike, sell lower strike
            strikes = sorted(type_options['strike'].unique(), reverse=True)
            
            # Find strike closest to ATM
            atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))
            atm_index = strikes.index(atm_strike)
            
            # Determine strike width based on ATR or price percentage
            width_factor = self.parameters['ratio_strike_width']
            
            if hasattr(self, 'market_data') and not self.market_data.empty:
                if 'atr' in self.market_data.columns:
                    atr = self.market_data['atr'].iloc[-1]
                    width = atr * width_factor
                else:
                    width = underlying_price * 0.03 * width_factor
            else:
                width = underlying_price * 0.03 * width_factor
            
            # Find suitable OTM strike
            long_strike = atm_strike
            short_strike = None
            
            for strike in strikes[atm_index:]:
                if strike <= long_strike - width:
                    short_strike = strike
                    break
            
            if short_strike is None:
                logger.warning("Could not find suitable OTM strike for put ratio spread")
                return None
            
            # Get option data for selected strikes
            long_options = type_options[type_options['strike'] == long_strike]
            short_options = type_options[type_options['strike'] == short_strike]
            
            if long_options.empty or short_options.empty:
                logger.warning(f"No options found at selected strikes {long_strike}/{short_strike}")
                return None
            
            # Create option legs
            long_leg = OptionLeg(
                option_type=option_type,
                strike=long_strike,
                expiration=expiration,
                action='buy',
                quantity=long_qty,
                entry_price=long_options['ask'].iloc[0],  # Buy at ask
                delta=long_options.get('delta', [-0.5])[0],
                leg_id='long_leg'
            )
            
            short_leg = OptionLeg(
                option_type=option_type,
                strike=short_strike,
                expiration=expiration,
                action='sell',
                quantity=short_qty,
                entry_price=short_options['bid'].iloc[0],  # Sell at bid
                delta=short_options.get('delta', [-0.3])[0],
                leg_id='short_leg'
            )
            
            # Create position
            spread_type = AdvancedSpreadType.RATIO_PUT_SPREAD
        
        # Check if the debit/credit is acceptable
        net_debit = (long_leg.entry_price * long_leg.quantity) - (short_leg.entry_price * short_leg.quantity)
        
        if net_debit > 0:  # This is a debit spread
            max_risk = net_debit * 100  # Convert to dollars
            if max_risk > self.parameters['max_loss_per_trade']:
                logger.warning(f"Ratio spread debit (${max_risk:.2f}) exceeds max loss per trade")
                return None
        
        # Create advanced spread position
        return AdvancedSpreadPosition(
            spread_type=spread_type,
            legs=[long_leg, short_leg],
            position_id=None  # Auto-generated
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
            indicators['hist_volatility_20d'] = data['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Calculate volatility percentile if we have enough data
        if len(data) >= 252:  # Full year of data
            hist_vol = indicators['hist_volatility_20d']
            vol_percentile = (hist_vol.iloc[-1] - hist_vol.min()) / (hist_vol.max() - hist_vol.min())
            indicators['vol_percentile'] = vol_percentile
        
        # Calculate price momentum indicators
        if 'close' in data.columns:
            # Calculate price momentum
            indicators['price_5d_change'] = data['close'].pct_change(periods=5) * 100
            indicators['price_20d_change'] = data['close'].pct_change(periods=20) * 100
            
            # Calculate moving averages
            indicators['ma_20'] = data['close'].rolling(window=20).mean()
            indicators['ma_50'] = data['close'].rolling(window=50).mean()
            indicators['ma_200'] = data['close'].rolling(window=200).mean()
            
            # Calculate trend strength
            indicators['adx'] = self._calculate_adx(data, 14)
            
            # Add RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
        
        return indicators
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX) for trend strength."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Plus Directional Movement
        plus_dm = high.diff()
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > low.shift(1) - low), 0)
        
        # Minus Directional Movement
        minus_dm = low.shift(1) - low
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > high - high.shift(1)), 0)
        
        # Directional Indicators
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Directional Index
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        
        # Average Directional Index
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for advanced spreads based on market conditions.
        
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
        
        # Get current market conditions
        current_price = data['close'].iloc[-1]
        
        # Get IV metrics if available
        iv_percentile = None
        
        if self.session.current_iv is not None and self.session.symbol in self.iv_history:
            iv_metrics = self.calculate_implied_volatility_metrics(self.iv_history[self.session.symbol])
            iv_percentile = iv_metrics.get('iv_percentile')
        
        # Ratio spread signal logic
        # Ratio call spreads work well in moderately bullish markets with high IV
        # Ratio put spreads work well in moderately bearish markets with high IV
        
        if 'adx' in indicators and indicators['adx'].iloc[-1] > 20:  # Strong trend
            if 'ma_20' in indicators and 'ma_50' in indicators:
                # Check trend direction
                bullish = indicators['ma_20'].iloc[-1] > indicators['ma_50'].iloc[-1]
                bearish = indicators['ma_20'].iloc[-1] < indicators['ma_50'].iloc[-1]
                
                # High IV environment is good for ratio spreads
                high_iv = iv_percentile is not None and iv_percentile > 60
                
                if bullish and high_iv:
                    # Bullish trend with high IV - good for ratio call spreads
                    if 'rsi' in indicators and indicators['rsi'].iloc[-1] < 70:  # Not overbought
                        signals["entry"] = True
                        signals["spread_type"] = AdvancedSpreadType.RATIO_CALL_SPREAD
                        signals["option_type"] = 'call'
                        signals["signal_strength"] = 0.7
                
                elif bearish and high_iv:
                    # Bearish trend with high IV - good for ratio put spreads
                    if 'rsi' in indicators and indicators['rsi'].iloc[-1] > 30:  # Not oversold
                        signals["entry"] = True
                        signals["spread_type"] = AdvancedSpreadType.RATIO_PUT_SPREAD
                        signals["option_type"] = 'put'
                        signals["signal_strength"] = 0.7
        
        # Check for exits
        for position in self.positions:
            if position.status == "open":
                # Exit logic based on position type
                if position.spread_type == AdvancedSpreadType.RATIO_CALL_SPREAD:
                    # Exit call ratio spread if trend turns bearish
                    if 'ma_20' in indicators and 'ma_50' in indicators:
                        if indicators['ma_20'].iloc[-1] < indicators['ma_50'].iloc[-1]:
                            signals["exit_positions"].append(position.position_id)
                    
                    # Exit if RSI indicates overbought conditions
                    if 'rsi' in indicators and indicators['rsi'].iloc[-1] > 70:
                        signals["exit_positions"].append(position.position_id)
                
                elif position.spread_type == AdvancedSpreadType.RATIO_PUT_SPREAD:
                    # Exit put ratio spread if trend turns bullish
                    if 'ma_20' in indicators and 'ma_50' in indicators:
                        if indicators['ma_20'].iloc[-1] > indicators['ma_50'].iloc[-1]:
                            signals["exit_positions"].append(position.position_id)
                    
                    # Exit if RSI indicates oversold conditions
                    if 'rsi' in indicators and indicators['rsi'].iloc[-1] < 30:
                        signals["exit_positions"].append(position.position_id)
        
        return signals
    
    def _execute_signals(self):
        """Execute trading signals generated by the strategy."""
        if not self.signals:
            return
        
        # Handle entry signals
        if self.signals.get("entry", False) and "spread_type" in self.signals:
            spread_type = self.signals["spread_type"]
            option_type = self.signals.get("option_type", 'call')
            underlying_price = self.session.current_price or self.market_data['close'].iloc[-1]
            
            # Construct the appropriate spread
            position = None
            if spread_type in [AdvancedSpreadType.RATIO_CALL_SPREAD, AdvancedSpreadType.RATIO_PUT_SPREAD]:
                position = self.construct_ratio_spread(self.session.option_chain, underlying_price, option_type)
            # Additional spread types would be implemented here
            
            if position:
                # Add position to active positions
                self.positions.append(position)
                logger.info(f"Opened {spread_type.value} position {position.position_id}")
        
        # Handle exit signals
        for position_id in self.signals.get("exit_positions", []):
            for position in self.positions:
                if position.position_id == position_id and position.status == "open":
                    # Get current prices for all legs
                    # In a real implementation, would get actual prices from option chain
                    # For now, just set placeholder values
                    leg_prices = {}
                    for leg in position.legs:
                        if leg.action == 'buy':
                            # For long options, assume a slight loss on exit
                            leg_prices[leg.leg_id] = leg.entry_price * 0.9
                        else:
                            # For short options, assume a slight gain on exit
                            leg_prices[leg.leg_id] = leg.entry_price * 0.9
                    
                    # Close the position
                    position.close_position(leg_prices, "signal_generated")
                    logger.info(f"Closed position {position_id} based on signals")
