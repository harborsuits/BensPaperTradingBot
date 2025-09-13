#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iron Butterfly Strategy

A professional-grade iron butterfly implementation that leverages the modular,
event-driven architecture. This strategy combines a bull put spread and a bear call
spread with the short options at the same strike price, profiting from low volatility
and range-bound markets.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np

from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.core.events import Event, EventType, EventBus
from trading_bot.strategies_new.options.base.options_base_strategy import OptionsSession
from trading_bot.strategies_new.options.base.spread_types import OptionType
from trading_bot.strategies_new.options.base.complex_spread_engine import ComplexSpreadEngine
from trading_bot.strategies_new.factory.registry import register_strategy
from trading_bot.strategies_new.validation.account_aware_mixin import AccountAwareMixin

# Configure logging
logger = logging.getLogger(__name__)

class IronButterflyPosition:
    """Represents an iron butterfly position with four options legs."""
    
    def __init__(self, 
                lower_put_leg: Dict[str, Any],
                middle_put_leg: Dict[str, Any],
                middle_call_leg: Dict[str, Any],
                upper_call_leg: Dict[str, Any],
                quantity: int = 1,
                position_id: Optional[str] = None):
        """
        Initialize an iron butterfly position.
        
        Args:
            lower_put_leg: Long put option details (lower strike)
            middle_put_leg: Short put option details (middle strike)
            middle_call_leg: Short call option details (middle strike)
            upper_call_leg: Long call option details (upper strike)
            quantity: Number of spreads in the position
            position_id: Optional unique identifier
        """
        self.lower_put_leg = lower_put_leg
        self.middle_put_leg = middle_put_leg
        self.middle_call_leg = middle_call_leg
        self.upper_call_leg = upper_call_leg
        self.quantity = quantity
        self.position_id = position_id or str(uuid.uuid4())
        
        # Position tracking
        self.entry_time = datetime.now()
        self.exit_time = None
        self.status = "open"
        
        # Risk metrics
        self.middle_strike = middle_put_leg['strike']
        self.net_credit = self._calculate_net_credit()
        self.max_profit = self.net_credit * 100 * self.quantity
        self.max_loss = self._calculate_max_loss()
        self.breakeven_points = self._calculate_breakeven_points()
        
        logger.info(f"Created iron butterfly position with ID: {self.position_id}")
    
    def _calculate_net_credit(self) -> float:
        """Calculate net credit received for the iron butterfly."""
        # For an iron butterfly:
        # - Buy lower strike put
        # - Sell middle strike put
        # - Sell middle strike call
        # - Buy upper strike call
        credit = 0.0
        
        # Long lower put (debit)
        credit -= self.lower_put_leg['entry_price']
        
        # Short middle put (credit)
        credit += self.middle_put_leg['entry_price']
        
        # Short middle call (credit)
        credit += self.middle_call_leg['entry_price']
        
        # Long upper call (debit)
        credit -= self.upper_call_leg['entry_price']
        
        return credit
    
    def _calculate_max_loss(self) -> float:
        """Calculate maximum potential loss for the iron butterfly."""
        # Max loss is the difference between strike prices minus the net credit
        wing_width = self.upper_call_leg['strike'] - self.middle_call_leg['strike']
        max_loss = (wing_width - self.net_credit) * 100 * self.quantity
        return max_loss
    
    def _calculate_breakeven_points(self) -> List[float]:
        """Calculate breakeven points for the iron butterfly."""
        # Lower breakeven = middle strike - net credit
        # Upper breakeven = middle strike + net credit
        lower_breakeven = self.middle_strike - self.net_credit
        upper_breakeven = self.middle_strike + self.net_credit
        
        return [lower_breakeven, upper_breakeven]
    
    def close_position(self, leg_prices: Dict[str, float], exit_reason: str):
        """
        Close the position and record exit details.
        
        Args:
            leg_prices: Dictionary with exit prices for each leg
            exit_reason: Reason for closing the position
        """
        if self.status != "open":
            logger.warning(f"Attempted to close position {self.position_id} that is already {self.status}")
            return
        
        self.exit_time = datetime.now()
        
        # Record exit prices
        self.lower_put_leg['exit_price'] = leg_prices.get('lower_put', 0)
        self.middle_put_leg['exit_price'] = leg_prices.get('middle_put', 0)
        self.middle_call_leg['exit_price'] = leg_prices.get('middle_call', 0)
        self.upper_call_leg['exit_price'] = leg_prices.get('upper_call', 0)
        
        # Calculate exit debit
        exit_debit = (
            leg_prices.get('lower_put', 0) - 
            leg_prices.get('middle_put', 0) - 
            leg_prices.get('middle_call', 0) + 
            leg_prices.get('upper_call', 0)
        )
        
        # Calculate P&L
        pnl = (self.net_credit - exit_debit) * 100 * self.quantity
        
        self.status = "closed"
        self.profit_loss = pnl
        self.exit_reason = exit_reason
        
        logger.info(f"Closed iron butterfly position {self.position_id}, P&L: ${pnl:.2f}, reason: {exit_reason}")


@register_strategy(
    name="IronButterflyStrategy",
    market_type="options",
    description="A strategy that combines a bull put spread and a bear call spread with the short options at the same strike price, profiting from low volatility and range-bound markets",
    timeframes=["1d", "1w"],
    parameters={
        "width_pct": {"description": "Width between strikes as % of underlying price", "type": "float"},
        "use_atm_middle": {"description": "Whether middle strike should be near ATM", "type": "boolean"},
        "target_days_to_expiry": {"description": "Ideal DTE for iron butterfly entry", "type": "integer"},
        "min_reward_to_risk": {"description": "Minimum reward-to-risk ratio", "type": "float"}
    }
)
class IronButterflyStrategy(ComplexSpreadEngine, AccountAwareMixin):
    """
    Iron Butterfly Strategy
    
    This strategy combines a bull put spread and a bear call spread with the short
    options at the same strike price. It profits most when the underlying price
    is at the short strike price at expiration, with defined risk and reward.
    
    Features:
    - Adapts the legacy Iron Butterfly implementation to the new architecture
    - Uses event-driven signal generation and position management
    - Leverages the ComplexSpreadEngine for core multi-leg spread mechanics
    - Implements custom filtering and volatility analysis
    """
    
    def __init__(self, session: OptionsSession, data_pipeline: DataPipeline, 
                 parameters: Dict[str, Any] = None):
        """
        Initialize the Iron Butterfly strategy.
        
        Args:
            session: Options trading session
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        # Initialize base engine
        super().__init__(session, data_pipeline, parameters)
        # Initialize account awareness functionality
        AccountAwareMixin.__init__(self)
        
        # Strategy-specific default parameters
        default_params = {
            # Strategy identification
            'strategy_name': 'Iron Butterfly',
            'strategy_id': 'iron_butterfly',
            
            # Iron Butterfly specific parameters
            'wing_width_pct': 3.0,          # Width between strikes as % of underlying price
            'use_atm_middle': True,         # Middle strike near ATM
            'target_days_to_expiry': 30,    # Ideal DTE for iron butterfly entry
            
            # Market condition preferences
            'min_stability_score': 75,      # Minimum score for price stability (0-100)
            'prefer_high_iv': True,         # Iron butterflies benefit from high IV environments
            'min_iv_percentile': 60,        # Minimum IV percentile for entry
            'max_iv_percentile': 85,        # Maximum IV percentile for entry (avoid extreme IV)
            
            # Risk parameters
            'max_risk_per_trade_pct': 2.0,  # Max risk as % of account
            'target_profit_pct': 50,        # Target profit as % of max profit
            'stop_loss_pct': 100,           # Stop loss as % of max profit (100% = full loss)
            'max_positions': 2,             # Maximum concurrent positions
            'min_credit_to_width_ratio': 0.15,  # Minimum credit as % of wing width
            
            # Exit parameters
            'days_before_expiry_exit': 5,   # Exit this many days before expiry
        }
        
        # Update with default parameters
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Strategy state
        self.positions = []  # List of IronButterflyPosition objects
        
        # Register for market events
        self.register_events()
        
        logger.info(f"Initialized Iron Butterfly Strategy for {session.symbol}")
    
    def construct_iron_butterfly(self, option_chain: pd.DataFrame, underlying_price: float) -> Optional[IronButterflyPosition]:
        """
        Construct an Iron Butterfly spread from the option chain.
        
        Args:
            option_chain: Filtered option chain data
            underlying_price: Current price of the underlying asset
            
        Returns:
            IronButterflyPosition if successful, None otherwise
        """
        if option_chain is None or option_chain.empty:
            return None
        
        # Apply filters for liquidity, open interest, etc.
        filtered_chain = self.filter_option_chains(option_chain)
        
        if filtered_chain is None or filtered_chain.empty:
            logger.warning("No suitable options found for iron butterfly")
            return None
        
        # Select expiration based on target days to expiry
        target_dte = self.parameters['target_days_to_expiry']
        expiration = self.select_expiration_by_dte(filtered_chain, target_dte)
        
        if not expiration:
            logger.warning(f"No suitable expiration found near {target_dte} DTE")
            return None
        
        # Filter for selected expiration
        exp_options = filtered_chain[filtered_chain['expiration_date'] == expiration]
        calls = exp_options[exp_options['option_type'] == 'call']
        puts = exp_options[exp_options['option_type'] == 'put']
        
        if calls.empty or puts.empty:
            logger.warning("Missing call or put options for the selected expiration")
            return None
        
        # Find middle strike (ATM or near ATM)
        strikes = sorted(exp_options['strike'].unique())
        
        if len(strikes) < 3:
            logger.warning("Not enough strikes available for iron butterfly")
            return None
        
        # Middle strike selection
        if self.parameters['use_atm_middle']:
            # Find strike closest to current price
            middle_strike = min(strikes, key=lambda x: abs(x - underlying_price))
        else:
            # Use technical analysis to determine a precise target
            # For example, use pivot points or support/resistance
            if 'pivot' in self.indicators:
                pivot = self.indicators['pivot']
                middle_strike = min(strikes, key=lambda x: abs(x - pivot))
            else:
                middle_strike = min(strikes, key=lambda x: abs(x - underlying_price))
        
        # Determine wing width based on parameters
        wing_width_pct = self.parameters['wing_width_pct']
        wing_width = underlying_price * (wing_width_pct / 100)
        
        # Find strikes for wings
        lower_strike = None
        upper_strike = None
        
        for strike in strikes:
            if strike < middle_strike and (lower_strike is None or strike > lower_strike):
                if middle_strike - strike <= wing_width * 1.2:  # Allow some flexibility
                    lower_strike = strike
            
            if strike > middle_strike and (upper_strike is None or strike < upper_strike):
                if strike - middle_strike <= wing_width * 1.2:  # Allow some flexibility
                    upper_strike = strike
        
        if lower_strike is None or upper_strike is None:
            logger.warning(f"Could not find suitable wing strikes for iron butterfly around {middle_strike}")
            return None
        
        # Get option contracts for each strike
        middle_puts = puts[puts['strike'] == middle_strike]
        middle_calls = calls[calls['strike'] == middle_strike]
        lower_puts = puts[puts['strike'] == lower_strike]
        upper_calls = calls[calls['strike'] == upper_strike]
        
        if middle_puts.empty or middle_calls.empty or lower_puts.empty or upper_calls.empty:
            logger.warning("Missing options for one or more iron butterfly legs")
            return None
        
        # Get first option for each strike/type
        lower_put = lower_puts.iloc[0]
        middle_put = middle_puts.iloc[0]
        middle_call = middle_calls.iloc[0]
        upper_call = upper_calls.iloc[0]
        
        # Create legs for iron butterfly
        # Buy lower put
        lower_put_leg = {
            'option_type': 'put',
            'strike': lower_strike,
            'expiration': expiration,
            'action': 'buy',
            'quantity': 1,
            'entry_price': lower_put['ask'],  # Buy at ask
            'delta': lower_put.get('delta', -0.10)
        }
        
        # Sell middle put
        middle_put_leg = {
            'option_type': 'put',
            'strike': middle_strike,
            'expiration': expiration,
            'action': 'sell',
            'quantity': 1,
            'entry_price': middle_put['bid'],  # Sell at bid
            'delta': middle_put.get('delta', -0.50)
        }
        
        # Sell middle call
        middle_call_leg = {
            'option_type': 'call',
            'strike': middle_strike,
            'expiration': expiration,
            'action': 'sell',
            'quantity': 1,
            'entry_price': middle_call['bid'],  # Sell at bid
            'delta': middle_call.get('delta', 0.50)
        }
        
        # Buy upper call
        upper_call_leg = {
            'option_type': 'call',
            'strike': upper_strike,
            'expiration': expiration,
            'action': 'buy',
            'quantity': 1,
            'entry_price': upper_call['ask'],  # Buy at ask
            'delta': upper_call.get('delta', 0.10)
        }
        
        # Create iron butterfly position
        position = IronButterflyPosition(
            lower_put_leg=lower_put_leg,
            middle_put_leg=middle_put_leg,
            middle_call_leg=middle_call_leg,
            upper_call_leg=upper_call_leg
        )
        
        # Check if credit-to-width ratio meets minimum threshold
        wing_width = upper_strike - middle_strike  # Same as middle_strike - lower_strike
        credit_to_width_ratio = position.net_credit / wing_width
        min_ratio = self.parameters['min_credit_to_width_ratio']
        
        if credit_to_width_ratio < min_ratio:
            logger.info(f"Iron butterfly rejected: credit/width ratio {credit_to_width_ratio:.3f} below minimum {min_ratio:.3f}")
            return None
        
        logger.info(f"Constructed iron butterfly with strikes {lower_strike}/{middle_strike}/{upper_strike}, credit: ${position.net_credit * 100:.2f}")
        
        return position
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicators for iron butterfly signals.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        if data.empty or len(data) < 30:
            return indicators
        
        # Calculate HV (Historical Volatility)
        if 'close' in data.columns:
            log_returns = np.log(data['close'] / data['close'].shift(1)).dropna()
            if len(log_returns) > 20:
                hv_20 = log_returns.rolling(window=20).std() * np.sqrt(252) * 100
                indicators['hv_20'] = hv_20.iloc[-1]
                indicators['hv_20_percentile'] = self.calculate_percentile(hv_20.iloc[-1], hv_20.dropna())
        
        # Calculate price stability metrics
        if 'close' in data.columns and len(data) > 20:
            # Standard deviation of returns over 10 and 20 days
            returns_10 = data['close'].pct_change().rolling(10).std().iloc[-1] * 100
            returns_20 = data['close'].pct_change().rolling(20).std().iloc[-1] * 100
            indicators['volatility_10d'] = returns_10
            indicators['volatility_20d'] = returns_20
            
            # Range as percentage of price
            atr = self.calculate_atr(data, period=14)
            if atr is not None and 'close' in data.columns and data['close'].iloc[-1] > 0:
                indicators['atr_pct'] = (atr / data['close'].iloc[-1]) * 100
            
            # Trend strength - ADX
            adx = self.calculate_adx(data, period=14)
            indicators['adx'] = adx
            
            # RSI to check for extremes
            rsi = self.calculate_rsi(data['close'], period=14)
            indicators['rsi'] = rsi.iloc[-1] if not rsi.empty else None
            
            # Calculate price channel
            indicators['upper_channel'] = data['high'].rolling(20).max().iloc[-1]
            indicators['lower_channel'] = data['low'].rolling(20).min().iloc[-1]
            indicators['channel_width_pct'] = ((indicators['upper_channel'] - indicators['lower_channel']) / 
                                              data['close'].iloc[-1]) * 100
            
            # Pivot points
            if len(data) >= 20:
                pivot = (data['high'].iloc[-2] + data['low'].iloc[-2] + data['close'].iloc[-2]) / 3
                indicators['pivot'] = pivot
        
        # Calculate stability score (0-100)
        stability_score = 0
        if 'adx' in indicators and indicators['adx'] is not None:
            # Lower ADX suggests less trend strength, better for iron butterfly
            adx_score = max(0, 100 - indicators['adx'] * 2)
            stability_score += adx_score * 0.4  # 40% weight
        
        if 'atr_pct' in indicators and indicators['atr_pct'] is not None:
            # Lower ATR % is better for iron butterfly
            atr_score = max(0, 100 - indicators['atr_pct'] * 10)
            stability_score += atr_score * 0.3  # 30% weight
        
        if 'channel_width_pct' in indicators and indicators['channel_width_pct'] is not None:
            # Narrower channel is better for iron butterfly
            channel_score = max(0, 100 - indicators['channel_width_pct'] * 5)
            stability_score += channel_score * 0.3  # 30% weight
        
        indicators['stability_score'] = min(100, stability_score)
        
        # Trend analysis
        if len(data) >= 50 and 'close' in data.columns:
            indicators['sma_20'] = data['close'].rolling(20).mean().iloc[-1]
            indicators['sma_50'] = data['close'].rolling(50).mean().iloc[-1]
            indicators['trend'] = 'neutral'
            
            if indicators['sma_20'] > indicators['sma_50'] * 1.02:
                indicators['trend'] = 'uptrend'
            elif indicators['sma_20'] < indicators['sma_50'] * 0.98:
                indicators['trend'] = 'downtrend'
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for iron butterfly based on market conditions.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            "entry": False,
            "exit_positions": [],
            "signal_strength": 0.0
        }
        
        if data.empty or not indicators:
            return signals
        
        # Check if we already have too many positions
        if len(self.positions) >= self.parameters['max_positions']:
            logger.info("Maximum number of positions reached, no new entries.")
        else:
            # Check IV environment - Iron Butterfly works best in high IV environments
            iv_check_passed = True
            if self.session.current_iv is not None:
                iv_metrics = self.calculate_implied_volatility_metrics(self.iv_history[self.session.symbol])
                iv_percentile = iv_metrics.get('iv_percentile', 50)
                
                min_iv = self.parameters['min_iv_percentile']
                max_iv = self.parameters['max_iv_percentile']
                
                if iv_percentile < min_iv or iv_percentile > max_iv:
                    iv_check_passed = False
                    logger.info(f"IV filter failed: current IV percentile {iv_percentile:.2f} outside range [{min_iv}-{max_iv}]")
            
            # Check stability score
            if 'stability_score' in indicators and iv_check_passed:
                stability_score = indicators['stability_score']
                min_score = self.parameters['min_stability_score']
                
                if stability_score >= min_score:
                    # Neutral trend check
                    trend_check = True
                    if 'trend' in indicators:
                        trend = indicators['trend']
                        if trend != 'neutral':
                            # In strong trends, iron butterflies can be riskier
                            # But they can still work if placed strategically
                            trend_check = False
                            logger.info(f"Warning: Current trend is '{trend}', not ideal for iron butterfly")
                    
                    # Check price in relation to channel
                    price_check = True
                    if ('upper_channel' in indicators and 
                        'lower_channel' in indicators and 
                        'close' in data.columns):
                        current_price = data['close'].iloc[-1]
                        upper_channel = indicators['upper_channel']
                        lower_channel = indicators['lower_channel']
                        channel_mid = (upper_channel + lower_channel) / 2
                        
                        # Preferably, price should be near the middle of the channel
                        deviation_from_mid = abs(current_price - channel_mid) / (upper_channel - lower_channel)
                        if deviation_from_mid > 0.4:  # More than 40% deviation from midpoint
                            price_check = False
                            logger.info(f"Price position check failed: price too far from channel midpoint")
                    
                    # All checks passed?
                    if price_check and (trend_check or stability_score > 85):  # High stability can override trend concerns
                        # Generate entry signal
                        signals["entry"] = True
                        signals["signal_strength"] = stability_score / 100.0
                        
                        logger.info(f"Iron butterfly entry signal: stability score {stability_score}, "
                                    f"strength {signals['signal_strength']:.2f}")
                    else:
                        logger.info(f"Iron butterfly entry rejected: price position or trend not ideal")
                else:
                    logger.info(f"Stability score {stability_score} below threshold {min_score}")
        
        # Check for exits on existing positions
        for position in self.positions:
            if position.status == "open":
                exit_signal = self._check_exit_conditions(position, data, indicators)
                if exit_signal:
                    signals["exit_positions"].append(position.position_id)
        
        return signals
    
    def _check_exit_conditions(self, position, data: pd.DataFrame, indicators: Dict[str, Any]) -> bool:
        """
        Check exit conditions for an open iron butterfly position.
        
        Args:
            position: The position to check
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Boolean indicating whether to exit
        """
        # Get current market data
        if data.empty:
            return False
        
        # 1. Days to expiration check
        if hasattr(position, 'middle_put_leg') and 'expiration' in position.middle_put_leg:
            expiry_date = position.middle_put_leg.get('expiration')
            if isinstance(expiry_date, str):
                expiry_date = datetime.strptime(expiry_date, '%Y-%m-%d').date()
            
            days_to_expiry = (expiry_date - datetime.now().date()).days
            
            # Exit ahead of expiration to avoid pin risk
            if days_to_expiry <= self.parameters['days_before_expiry_exit']:
                logger.info(f"Exit signal: approaching expiration ({days_to_expiry} days left)")
                return True
        
        # 2. Profit target check
        if hasattr(position, 'max_profit') and position.max_profit > 0:
            # In a real implementation, would get actual current prices from option chain
            # For now, we'll estimate based on underlying price
            current_price = data['close'].iloc[-1]
            middle_strike = position.middle_strike
            
            # Simplified profit calculation - closer to middle strike means more profit
            price_to_strike_ratio = abs(current_price - middle_strike) / middle_strike
            estimated_profit_pct = max(0, 1 - price_to_strike_ratio * 10)  # Rough approximation
            
            target_profit_pct = self.parameters['target_profit_pct'] / 100
            
            if estimated_profit_pct >= target_profit_pct:
                logger.info(f"Exit signal: profit target reached (est. {estimated_profit_pct:.1%})")
                return True
        
        # 3. Stop loss check
        if hasattr(position, 'max_profit') and position.max_profit > 0:
            # Similar simplified estimation for losses
            current_price = data['close'].iloc[-1]
            wing_width = position.upper_call_leg['strike'] - position.middle_call_leg['strike']
            
            # Calculate how far outside our wings we are (if at all)
            outside_wings = False
            lower_wing = position.lower_put_leg['strike']
            upper_wing = position.upper_call_leg['strike']
            
            if current_price <= lower_wing or current_price >= upper_wing:
                outside_wings = True
            
            stop_loss_pct = self.parameters['stop_loss_pct'] / 100
            
            if outside_wings and stop_loss_pct < 1.0:  # If stop loss is less than 100%
                logger.info(f"Exit signal: price moved outside wings, stop loss triggered")
                return True
        
        # 4. Stability deterioration check
        if 'stability_score' in indicators:
            stability_score = indicators['stability_score']
            if stability_score < 40:  # Major deterioration in stability
                logger.info(f"Exit signal: stability score dropped to {stability_score}")
                return True
        
        # 5. Trend change check
        if 'trend' in indicators:
            trend = indicators['trend']
            if trend != 'neutral' and 'adx' in indicators and indicators['adx'] > 25:
                # Strong trend developing - bad for iron butterfly
                logger.info(f"Exit signal: strong {trend} developing, ADX={indicators['adx']}")
                return True
        
        return False
    
    def _execute_signals(self):

        # Verify account has sufficient buying power
        buying_power = self.get_buying_power()
        if buying_power <= 0:
            logger.warning("Trade execution aborted: Insufficient buying power")
            return
        """
        Execute Iron Butterfly specific trading signals.
        """
        if not self.signals:
            return
        
        # Handle entry signals
        if self.signals.get("entry", False):
            # Construct and open iron butterfly position
            underlying_price = self.session.current_price
            
            if underlying_price and self.session.option_chain is not None:
                # Use the construct_iron_butterfly method for position creation
                iron_butterfly_position = self.construct_iron_butterfly(
                    self.session.option_chain, 
                    underlying_price
                )
                
                if iron_butterfly_position:
                    # Determine position size
                    account_value = self.session.account_value
                    max_risk_pct = self.parameters['max_risk_per_trade_pct'] / 100
                    max_risk_amount = account_value * max_risk_pct
                    
                    # Calculate number of spreads based on max risk
                    if iron_butterfly_position.max_loss > 0:
                        num_spreads = int(max_risk_amount / iron_butterfly_position.max_loss)
                        num_spreads = max(1, num_spreads)  # At least 1 spread
                        iron_butterfly_position.quantity = num_spreads
                    
                    # Add position
                    self.positions.append(iron_butterfly_position)
                    logger.info(f"Opened iron butterfly position {iron_butterfly_position.position_id} "
                               f"with {num_spreads} spread(s)")
                else:
                    logger.warning("Failed to construct valid iron butterfly")
        
        # Handle exit signals
        for position_id in self.signals.get("exit_positions", []):
            for i, position in enumerate(self.positions):
                if position.position_id == position_id and position.status == "open":
                    # Get current prices for all legs (simplified example)
                    leg_prices = {
                        'lower_put': position.lower_put_leg['entry_price'] * 0.5,   # Placeholder
                        'middle_put': position.middle_put_leg['entry_price'] * 0.5,  # Placeholder
                        'middle_call': position.middle_call_leg['entry_price'] * 0.5, # Placeholder
                        'upper_call': position.upper_call_leg['entry_price'] * 0.5,  # Placeholder
                    }
                    
                    # Close the position
                    position.close_position(leg_prices, "signal_generated")
                    logger.info(f"Closed iron butterfly position {position_id} based on signals")
    
    def run_strategy(self):
        """
        Run the iron butterfly strategy cycle.
        """
        # Check if we have necessary data
        if not self.session.current_price or self.session.option_chain is None:
            logger.warning("Missing current price or option chain data, skipping strategy execution")
            return
        
        # Get market data
        data = self.get_historical_data(self.session.symbol, '1d', 90)
        
        if data is not None and not data.empty:
            # Calculate indicators
            self.indicators = self.calculate_indicators(data)
            
            # Generate signals
            self.signals = self.generate_signals(data, self.indicators)
            
            # Execute signals
            self._execute_signals()
        else:
            logger.warning("Insufficient market data for iron butterfly analysis")
    
    def register_events(self):
        """Register for events relevant to Iron Butterfly."""
        # Register for common event types from the parent class
        super().register_events()
        
        # Add iron butterfly specific event subscriptions
        EventBus.subscribe(EventType.EARNINGS_ANNOUNCEMENT, self.on_event)
        EventBus.subscribe(EventType.IMPLIED_VOLATILITY_CHANGE, self.on_event)
        EventBus.subscribe(EventType.MARKET_REGIME_CHANGE, self.on_event)
    
    def on_event(self, event: Event):
        """
        Process incoming events for the Iron Butterfly strategy.
        
        Args:
            event: The event to process
        """
        # Let parent class handle common events first
        super().on_event(event)
        
        # Add additional Iron Butterfly specific event handling
        if event.type == EventType.EARNINGS_ANNOUNCEMENT and self.session.symbol in event.data.get('symbols', []):
            # Close positions before earnings (high volatility event)
            days_to_earnings = event.data.get('days_to_announcement', 0)
            
            if days_to_earnings <= 5:  # Within 5 days of earnings
                logger.info(f"Earnings announcement approaching for {self.session.symbol}, closing iron butterfly positions")
                
                for position in self.positions:
                    if position.status == "open":
                        self.signals.setdefault("exit_positions", []).append(position.position_id)
                
                # Execute the exits
                self._execute_signals()
        
        elif event.type == EventType.IMPLIED_VOLATILITY_CHANGE:
            symbol = event.data.get('symbol')
            magnitude = event.data.get('magnitude', 0)
            direction = event.data.get('direction', '')
            
            if symbol == self.session.symbol and abs(magnitude) > 15:  # Significant IV change
                logger.info(f"Significant IV {direction} of {magnitude}% detected")
                
                # For IV drop, consider closing positions to take profit
                if direction == 'drop' and magnitude > 20:
                    for position in self.positions:
                        if position.status == "open":
                            logger.info(f"IV drop detected, closing iron butterfly position {position.position_id} to capture profit")
                            self.signals.setdefault("exit_positions", []).append(position.position_id)
                    
                    # Execute the exits
                    self._execute_signals()
        
        elif event.type == EventType.MARKET_REGIME_CHANGE:
            new_regime = event.data.get('new_regime', '')
            
            if new_regime == 'high_volatility' or new_regime == 'trending':
                logger.info(f"Market regime changed to {new_regime}, closing all iron butterfly positions")
                
                # Close all positions in regimes unfavorable to iron butterflies
                for position in self.positions:
                    if position.status == "open":
                        self.signals.setdefault("exit_positions", []).append(position.position_id)
                
                # Execute the exits
                self._execute_signals()
