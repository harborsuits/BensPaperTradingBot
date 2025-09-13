#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Butterfly Spread Strategy

A professional-grade butterfly spread implementation that leverages the modular,
event-driven architecture. This strategy profits when the underlying price is 
near the middle strike at expiration, with limited risk.
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

@register_strategy(
    name="ButterflySpreadStrategy",
    market_type="options",
    description="A strategy that combines a bull call spread and a bear call spread with a shared middle strike price, profiting most when the underlying price is at the middle strike at expiration",
    timeframes=["1d", "1w"],
    parameters={
        "option_type": {"description": "Option type for butterfly (call or put)", "type": "string"},
        "width_pct": {"description": "Width between strikes as % of underlying price", "type": "float"},
        "use_atm_middle": {"description": "Whether middle strike should be near ATM", "type": "boolean"},
        "target_days_to_expiry": {"description": "Ideal DTE for butterfly entry", "type": "integer"}
    }
)
class ButterflySpreadStrategy(ComplexSpreadEngine, AccountAwareMixin):
    """
    Butterfly Spread Strategy
    
    This strategy combines a bull call spread and a bear call spread with a shared
    middle strike price. It has limited risk and limited reward, profiting most
    when the underlying price is at the middle strike at expiration.
    
    Features:
    - Adapts the legacy Butterfly implementation to the new architecture
    - Uses event-driven signal generation and position management
    - Leverages the ComplexSpreadEngine for core multi-leg spread mechanics
    - Implements custom filtering and precise target price analysis
    """
    
    def __init__(self, session: OptionsSession, data_pipeline: DataPipeline, 
                 parameters: Dict[str, Any] = None):
        """
        Initialize the Butterfly Spread strategy.
        
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
            'strategy_name': 'Butterfly Spread',
            'strategy_id': 'butterfly_spread',
            
            # Butterfly specific parameters
            'option_type': 'call',         # 'call' or 'put' butterfly
            'width_pct': 2.5,              # Width between strikes as % of underlying price
            'use_atm_middle': True,        # Middle strike near ATM
            'target_days_to_expiry': 30,   # Ideal DTE for butterfly entry
            
            # Market condition preferences
            'min_precision_score': 70,     # Minimum score for precision price target (0-100)
            'prefer_low_iv': True,         # Butterflies often perform better in lower IV
            'max_iv_percentile': 60,       # Maximum IV percentile for entry
            
            # Risk parameters
            'max_risk_per_trade_pct': 1.0, # Max risk as % of account
            'target_profit_pct': 80,       # Target profit as % of max profit
            'stop_loss_pct': 75,           # Stop loss as % of max loss
            'max_positions': 3,            # Maximum concurrent positions
            'min_reward_to_risk': 3.0,     # Minimum reward-to-risk ratio
            
            # Advanced butterfly parameters
            'entry_days_before_expiry': 45,  # Enter this many days before expiry
            'exit_days_before_expiry': 5,    # Exit this many days before expiry to avoid gamma risk
        }
        
        # Update with default parameters
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Register for market events
        self.register_events()
        
        logger.info(f"Initialized Butterfly Spread Strategy for {session.symbol}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for the Butterfly Spread strategy.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary of calculated indicators
        """
        # Get base indicators from parent class
        indicators = super().calculate_indicators(data)
        
        if data.empty or len(data) < 50:
            return indicators
        
        # Calculate indicators specific to identifying precise price targets
        
        # Recent price consolidation (standard deviation of close prices)
        if len(data) >= 20:
            price_std = data['close'].rolling(window=20).std()
            price_mean = data['close'].rolling(window=20).mean()
            indicators['price_volatility'] = price_std / price_mean * 100  # As percentage
        
        # Support and resistance levels
        if len(data) >= 50:
            # Simple pivot points
            high = data['high'].iloc[-1]
            low = data['low'].iloc[-1]
            close = data['close'].iloc[-1]
            
            pivot = (high + low + close) / 3
            resistance1 = 2 * pivot - low
            support1 = 2 * pivot - high
            
            indicators['pivot'] = pivot
            indicators['resistance1'] = resistance1
            indicators['support1'] = support1
            
            # Check if current price is near pivot point
            price_to_pivot_pct = abs(close - pivot) / close * 100
            indicators['price_to_pivot_pct'] = price_to_pivot_pct
        
        # Price channeling - identify range-bound markets with clear boundaries
        if len(data) >= 50:
            # Upper and lower channels using recent highs and lows
            upper_channel = data['high'].rolling(window=20).max()
            lower_channel = data['low'].rolling(window=20).min()
            mid_channel = (upper_channel + lower_channel) / 2
            
            # Channel width as percentage
            channel_width = (upper_channel - lower_channel) / mid_channel * 100
            indicators['channel_width_pct'] = channel_width
            
            # Where is price in the channel (0 = bottom, 1 = top)
            current_price = data['close'].iloc[-1]
            channel_position = (current_price - lower_channel.iloc[-1]) / (upper_channel.iloc[-1] - lower_channel.iloc[-1])
            indicators['channel_position'] = channel_position
            
            # Calculate channeling score - higher means better defined channel
            channeling_score = 50  # Default score
            
            # Narrower channels indicate better precision for butterfly placement
            if channel_width.iloc[-1] < 10:  # Narrow channel
                channeling_score += 20
            elif channel_width.iloc[-1] > 20:  # Wide channel
                channeling_score -= 20
            
            # Price near middle of channel is ideal for butterfly
            if 0.4 <= channel_position <= 0.6:  # Near middle
                channeling_score += 20
            
            # Consistent channel width indicates stable range
            channel_width_std = channel_width.rolling(window=10).std().iloc[-1]
            if channel_width_std < 2:  # Low variation in channel width
                channeling_score += 10
            
            # Check if recent price action respects the channel
            price_respects_channel = True
            for i in range(5):
                if i < len(data):
                    if (data['high'].iloc[-i-1] > upper_channel.iloc[-i-1] * 1.02 or
                        data['low'].iloc[-i-1] < lower_channel.iloc[-i-1] * 0.98):
                        price_respects_channel = False
                        break
            
            if price_respects_channel:
                channeling_score += 10
            
            # Calculate precision score (0-100)
            precision_score = max(0, min(100, channeling_score))
            indicators['precision_score'] = precision_score
        
        return indicators
    
    def construct_butterfly(self, option_chain: pd.DataFrame, underlying_price: float) -> Optional[Dict]:
        """
        Construct a Butterfly spread from the option chain.
        
        Args:
            option_chain: Filtered option chain data
            underlying_price: Current price of the underlying asset
            
        Returns:
            Butterfly position details or None if not possible
        """
        if option_chain is None or option_chain.empty:
            return None
        
        # Apply filters for liquidity, open interest, etc.
        filtered_chain = self.filter_option_chains(option_chain)
        
        if filtered_chain is None or filtered_chain.empty:
            logger.warning("No suitable options found for butterfly spread")
            return None
        
        # Determine option type (call or put)
        option_type = self.parameters['option_type']
        type_filtered = filtered_chain[filtered_chain['option_type'] == option_type]
        
        if type_filtered.empty:
            logger.warning(f"No {option_type} options available for butterfly")
            return None
        
        # Select expiration based on target days to expiry
        target_dte = self.parameters['target_days_to_expiry']
        expiration = self.select_expiration_by_dte(type_filtered, target_dte)
        
        if not expiration:
            logger.warning(f"No suitable expiration found near {target_dte} DTE")
            return None
        
        # Filter for selected expiration
        exp_options = type_filtered[type_filtered['expiration_date'] == expiration]
        
        # Find ATM strike for middle strike of butterfly
        strikes = sorted(exp_options['strike'].unique())
        
        if len(strikes) < 3:
            logger.warning("Not enough strikes available for butterfly spread")
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
        width_pct = self.parameters['width_pct']
        width_amount = underlying_price * (width_pct / 100)
        
        # Find strikes for wings
        lower_strike = None
        upper_strike = None
        
        for strike in strikes:
            if strike < middle_strike and (lower_strike is None or strike > lower_strike):
                if middle_strike - strike <= width_amount * 1.2:  # Allow some flexibility
                    lower_strike = strike
            
            if strike > middle_strike and (upper_strike is None or strike < upper_strike):
                if strike - middle_strike <= width_amount * 1.2:  # Allow some flexibility
                    upper_strike = strike
        
        if lower_strike is None or upper_strike is None:
            logger.warning(f"Could not find suitable wing strikes for butterfly around {middle_strike}")
            return None
        
        # Get option contracts
        lower_options = exp_options[exp_options['strike'] == lower_strike]
        middle_options = exp_options[exp_options['strike'] == middle_strike]
        upper_options = exp_options[exp_options['strike'] == upper_strike]
        
        if lower_options.empty or middle_options.empty or upper_options.empty:
            logger.warning("Missing options for one or more butterfly legs")
            return None
        
        # Get first option for each strike
        lower_option = lower_options.iloc[0]
        middle_option = middle_options.iloc[0]
        upper_option = upper_options.iloc[0]
        
        # Create legs for butterfly
        if option_type == 'call':
            # Call butterfly: Buy lower call, sell 2 middle calls, buy upper call
            lower_leg = {
                'option_type': 'call',
                'strike': lower_strike,
                'expiration': expiration,
                'action': 'buy',
                'quantity': 1,
                'price': lower_option['ask'],  # Buy at ask
                'delta': lower_option.get('delta', 0.7)
            }
            
            middle_leg = {
                'option_type': 'call',
                'strike': middle_strike,
                'expiration': expiration,
                'action': 'sell',
                'quantity': 2,  # Sell 2 contracts
                'price': middle_option['bid'],  # Sell at bid
                'delta': middle_option.get('delta', 0.5)
            }
            
            upper_leg = {
                'option_type': 'call',
                'strike': upper_strike,
                'expiration': expiration,
                'action': 'buy',
                'quantity': 1,
                'price': upper_option['ask'],  # Buy at ask
                'delta': upper_option.get('delta', 0.3)
            }
        else:  # put butterfly
            # Put butterfly: Buy lower put, sell 2 middle puts, buy upper put
            lower_leg = {
                'option_type': 'put',
                'strike': lower_strike,
                'expiration': expiration,
                'action': 'buy',
                'quantity': 1,
                'price': lower_option['ask'],  # Buy at ask
                'delta': lower_option.get('delta', -0.7)
            }
            
            middle_leg = {
                'option_type': 'put',
                'strike': middle_strike,
                'expiration': expiration,
                'action': 'sell',
                'quantity': 2,  # Sell 2 contracts
                'price': middle_option['bid'],  # Sell at bid
                'delta': middle_option.get('delta', -0.5)
            }
            
            upper_leg = {
                'option_type': 'put',
                'strike': upper_strike,
                'expiration': expiration,
                'action': 'buy',
                'quantity': 1,
                'price': upper_option['ask'],  # Buy at ask
                'delta': upper_option.get('delta', -0.3)
            }
        
        # Calculate net debit
        net_debit = (lower_leg['price'] - 2 * middle_leg['price'] + upper_leg['price'])
        
        # Maximum profit potential
        max_profit = middle_strike - lower_strike - net_debit
        
        # Maximum risk
        max_loss = net_debit
        
        # Reward to risk ratio
        reward_risk_ratio = max_profit / max_loss if max_loss > 0 else 0
        
        # Check if reward-to-risk ratio meets minimum threshold
        min_reward_risk = self.parameters['min_reward_to_risk']
        if reward_risk_ratio < min_reward_risk:
            logger.info(f"Butterfly rejected: reward/risk ratio {reward_risk_ratio:.2f} below minimum {min_reward_risk:.2f}")
            return None
        
        # Create the butterfly position
        butterfly = {
            'lower_leg': lower_leg,
            'middle_leg': middle_leg,
            'upper_leg': upper_leg,
            'spread_type': 'butterfly',
            'option_type': option_type,
            'net_debit': net_debit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'reward_risk_ratio': reward_risk_ratio,
            'middle_strike': middle_strike,
            'expiration': expiration
        }
        
        logger.info(f"Constructed {option_type} butterfly spread with strikes {lower_strike}/{middle_strike}/{upper_strike}")
        
        return butterfly
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for Butterfly Spreads based on market conditions.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            "entry": False,
            "exit_positions": [],
            "signal_strength": 0.0,
            "butterfly_type": None
        }
        
        if data.empty or not indicators:
            return signals
        
        # Check if we already have too many positions
        if len(self.positions) >= self.parameters['max_positions']:
            logger.info("Maximum number of positions reached, no new entries.")
            # Still check for exits
        else:
            # Check IV environment (butterflies often work better in lower IV)
            iv_check_passed = True
            if self.parameters['prefer_low_iv'] and self.session.current_iv is not None:
                iv_metrics = self.calculate_implied_volatility_metrics(self.iv_history[self.session.symbol])
                iv_percentile = iv_metrics.get('iv_percentile', 50)
                
                max_iv = self.parameters['max_iv_percentile']
                
                if iv_percentile > max_iv:
                    iv_check_passed = False
                    logger.info(f"IV filter failed: current IV percentile {iv_percentile:.2f} above max {max_iv}")
            
            # Check precision score for butterfly placement
            if 'precision_score' in indicators and iv_check_passed:
                precision_score = indicators['precision_score']
                min_score = self.parameters['min_precision_score']
                
                if precision_score >= min_score:
                    # Generate entry signal
                    signals["entry"] = True
                    signals["signal_strength"] = precision_score / 100.0
                    
                    # Determine if we should use call or put butterfly
                    if 'channel_position' in indicators:
                        channel_pos = indicators['channel_position']
                        
                        if channel_pos < 0.4:
                            # Price in lower part of channel, prefer put butterfly
                            signals["butterfly_type"] = "put"
                        elif channel_pos > 0.6:
                            # Price in upper part of channel, prefer call butterfly
                            signals["butterfly_type"] = "call"
                        else:
                            # Price in middle of channel, use strategy default
                            signals["butterfly_type"] = self.parameters['option_type']
                    else:
                        signals["butterfly_type"] = self.parameters['option_type']
                    
                    logger.info(f"Butterfly entry signal: precision score {precision_score}, "
                                f"type {signals['butterfly_type']}, strength {signals['signal_strength']:.2f}")
                else:
                    logger.info(f"Precision score {precision_score} below threshold {min_score}")
        
        # Check for exits on existing positions
        for position in self.positions:
            if position.status == "open":
                exit_signal = self._check_exit_conditions(position, data, indicators)
                if exit_signal:
                    signals["exit_positions"].append(position.position_id)
        
        return signals
    
    def _check_exit_conditions(self, position, data: pd.DataFrame, indicators: Dict[str, Any]) -> bool:
        """
        Check exit conditions for an open butterfly position.
        
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
        
        current_price = data['close'].iloc[-1]
        
        # Extract position details
        if hasattr(position, 'middle_strike') and hasattr(position, 'expiration'):
            middle_strike = position.middle_strike
            expiry_date = position.expiration
            
            if isinstance(expiry_date, str):
                expiry_date = datetime.strptime(expiry_date, '%Y-%m-%d').date()
            
            # Days to expiration check
            days_to_expiry = (expiry_date - datetime.now().date()).days
            exit_days = self.parameters['exit_days_before_expiry']
            
            if days_to_expiry <= exit_days:
                logger.info(f"Exit signal: approaching expiration ({days_to_expiry} days left)")
                return True
            
            # Distance from middle strike
            if middle_strike:
                distance_pct = abs(current_price - middle_strike) / middle_strike * 100
                
                # If price moves too far from middle strike with limited time left
                if distance_pct > 7 and days_to_expiry < 14:
                    logger.info(f"Exit signal: price {distance_pct:.2f}% away from middle strike with {days_to_expiry} days left")
                    return True
        
        # Check if precision score has deteriorated significantly
        if 'precision_score' in indicators:
            precision_score = indicators['precision_score']
            if precision_score < 30:  # Major deterioration in price precision
                logger.info(f"Exit signal: precision score dropped to {precision_score}")
                return True
        
        # Check for significant change in volatility
        if self.session.current_iv is not None and hasattr(position, 'entry_iv'):
            entry_iv = position.entry_iv
            current_iv = self.session.current_iv
            
            iv_change_pct = (current_iv / entry_iv - 1) * 100
            if iv_change_pct > 30:  # IV increased by more than 30%
                logger.info(f"Exit signal: IV increased by {iv_change_pct:.2f}%")
                return True
        
        # Profit target and stop loss are handled by the base ComplexSpreadEngine
        
        return False
    
    def filter_option_chains(self, option_chain: pd.DataFrame) -> pd.DataFrame:
        """
        Apply custom filters for Butterfly Spread option selection.
        
        Args:
            option_chain: Option chain data
            
        Returns:
            Filtered option chain
        """
        # Apply base filters first
        filtered_chain = super().filter_option_chains(option_chain)
        
        if filtered_chain is None or filtered_chain.empty:
            return filtered_chain
        
        # We need good liquidity for all legs
        if 'open_interest' in filtered_chain.columns and 'volume' in filtered_chain.columns:
            min_oi = 100
            min_volume = 10
            filtered_chain = filtered_chain[
                (filtered_chain['open_interest'] >= min_oi) |
                (filtered_chain['volume'] >= min_volume)
            ]
        
        # For butterflies, additional expiration filtering
        if 'expiration_date' in filtered_chain.columns:
            # Filter for expirations that are at least our minimum DTE
            today = datetime.now().date()
            min_days = self.parameters['exit_days_before_expiry'] + 5  # Minimum buffer
            max_days = self.parameters['entry_days_before_expiry']
            
            filtered_expirations = []
            
            for expiry in filtered_chain['expiration_date'].unique():
                if isinstance(expiry, str):
                    expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
                else:
                    expiry_date = expiry
                
                days_to_expiry = (expiry_date - today).days
                
                if min_days <= days_to_expiry <= max_days:
                    filtered_expirations.append(expiry)
            
            if filtered_expirations:
                filtered_chain = filtered_chain[filtered_chain['expiration_date'].isin(filtered_expirations)]
        
        return filtered_chain
    
    def _execute_signals(self):

        # Verify account has sufficient buying power
        buying_power = self.get_buying_power()
        if buying_power <= 0:
            logger.warning("Trade execution aborted: Insufficient buying power")
            return
        """
        Execute Butterfly Spread specific trading signals.
        """
        if not self.signals:
            return
        
        # Handle entry signals
        if self.signals.get("entry", False):
            # Override option type if specified in signal
            if "butterfly_type" in self.signals and self.signals["butterfly_type"]:
                original_type = self.parameters['option_type']
                self.parameters['option_type'] = self.signals["butterfly_type"]
                
                logger.info(f"Setting butterfly type to {self.signals['butterfly_type']} based on signal")
            
            # Construct and open butterfly position
            underlying_price = self.session.current_price
            
            if underlying_price and self.session.option_chain is not None:
                butterfly = self.construct_butterfly(self.session.option_chain, underlying_price)
                
                if butterfly:
                    # Add butterfly position
                    position_id = self.spread_manager.open_position(butterfly)
                    logger.info(f"Opened butterfly position {position_id}")
                else:
                    logger.warning("Failed to construct valid butterfly spread")
            
            # Restore original option type if it was overridden
            if "butterfly_type" in self.signals and self.signals["butterfly_type"]:
                self.parameters['option_type'] = original_type
        
        # Handle exit signals
        for position_id in self.signals.get("exit_positions", []):
            for position in self.positions:
                if position.position_id == position_id and position.status == "open":
                    self.spread_manager.close_position(position_id, "signal_generated")
                    logger.info(f"Closed butterfly position {position_id} based on signals")
    
    def register_events(self):
        """Register for events relevant to Butterfly Spreads."""
        # Register for common event types from the parent class
        super().register_events()
        
        # Add any Butterfly Spread specific event subscriptions
        EventBus.subscribe(EventType.VOLATILITY_SPIKE, self.on_event)
        EventBus.subscribe(EventType.PRICE_CHANNEL_BREAKOUT, self.on_event)
    
    def on_event(self, event: Event):
        """
        Process incoming events for the Butterfly Spread strategy.
        
        Args:
            event: The event to process
        """
        # Let parent class handle common events first
        super().on_event(event)
        
        # Add additional Butterfly Spread specific event handling
        if event.type == EventType.VOLATILITY_SPIKE:
            spike_pct = event.data.get('percentage', 0)
            
            if spike_pct > 15:
                logger.info(f"Volatility spike of {spike_pct}% detected, adjusting Butterfly parameters")
                # Consider closing positions during volatility spikes
                for position in self.positions:
                    if position.status == "open":
                        self.spread_manager.close_position(position.position_id, "volatility_spike")
        
        elif event.type == EventType.PRICE_CHANNEL_BREAKOUT:
            symbol = event.data.get('symbol')
            if symbol == self.session.symbol:
                logger.info(f"Price channel breakout for {symbol}, closing butterfly positions")
                # Close positions since price breakout will likely move away from middle strike
                for position in self.positions:
                    if position.status == "open":
                        self.spread_manager.close_position(position.position_id, "channel_breakout")
