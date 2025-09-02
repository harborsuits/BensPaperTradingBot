#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Broken Wing Butterfly Strategy

A professional-grade Broken Wing Butterfly implementation that leverages the modular,
event-driven architecture. This is an asymmetric butterfly spread with uneven wing widths,
typically designed to provide a better risk-reward ratio than a traditional butterfly.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np

from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.core.events import Event, EventType, EventBus
from trading_bot.strategies_new.options.base.options_base_strategy import OptionsSession
from trading_bot.strategies_new.options.base.spread_types import OptionType
from trading_bot.strategies_new.options.base.complex_spread_engine import ComplexSpreadEngine, ComplexSpreadType
from trading_bot.strategies_new.factory.registry import register_strategy
from trading_bot.strategies_new.validation.account_aware_mixin import AccountAwareMixin

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="BrokenWingButterflyStrategy",
    market_type="options",
    description="An asymmetric butterfly spread with uneven wing widths, providing directional bias and often reducing debit cost or creating a credit position",
    timeframes=["1d", "1w"],
    parameters={
        "option_type": {"description": "Option type for BWB (call or put)", "type": "string"},
        "direction": {"description": "Directional bias (bullish or bearish)", "type": "string"},
        "balanced_wing": {"description": "Which wing is balanced (lower or upper)", "type": "string"},
        "target_days_to_expiry": {"description": "Target DTE for entry", "type": "integer"}
    }
)
class BrokenWingButterflyStrategy(ComplexSpreadEngine, AccountAwareMixin):
    """
    Broken Wing Butterfly Strategy
    
    A broken wing butterfly is an asymmetric butterfly spread with uneven wing widths.
    The wider wing typically gives the spread a directional bias and often reduces the
    debit cost or even creates a credit position. The trade-off is a potentially uncapped
    risk on one side.
    
    Features:
    - Adapts the legacy broken wing butterfly implementation to the new architecture
    - Uses event-driven signal generation and position management
    - Leverages the ComplexSpreadEngine for core complex spread mechanics
    - Implements custom filtering, technical analysis, and directional bias scoring
    """
    
    def __init__(self, session: OptionsSession, data_pipeline: DataPipeline, 
                 parameters: Dict[str, Any] = None):
        """
        Initialize the Broken Wing Butterfly strategy.
        
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
            'strategy_name': 'Broken Wing Butterfly',
            'strategy_id': 'broken_wing_butterfly',
            
            # Broken Wing Butterfly specific parameters
            'option_type': 'call',          # 'call' or 'put'
            'direction': 'bullish',         # 'bullish' (put BWB) or 'bearish' (call BWB)
            'balanced_wing': 'lower',       # 'lower' or 'upper' - which wing is balanced
            'bias_score_threshold': 65,     # Minimum directional bias score (0-100)
            'target_days_to_expiry': 35,    # Target DTE for entry
            'atm_strike_percent': 2.0,      # ATM strike within +/- this % of current price
            'narrow_wing_width': 1,         # Strike width of the balanced wing (in strikes)
            'wide_wing_width': 2,           # Strike width of the unbalanced/broken wing (in strikes)
            'max_debit_cost': 0.15,         # Maximum debit as % of underlying price
            'prefer_credit': True,          # Prefer credit BWB when possible
            
            # Market condition preferences
            'min_iv_percentile': 40,        # Minimum IV percentile for entry
            'max_iv_percentile': 90,        # Maximum IV percentile for entry
            'min_option_volume': 50,        # Minimum option volume for liquidity
            
            # Risk parameters
            'max_risk_per_trade_pct': 1.5,  # Max risk as % of account
            'target_profit_pct': 50,        # Target profit as % of max profit
            'stop_loss_pct': 75,            # Stop loss as % of max potential loss
            'max_positions': 3,             # Maximum concurrent positions
            
            # Exit parameters
            'days_before_expiry_exit': 5,   # Exit when reaching this DTE
            'wing_breach_adjustment': True, # Adjust position if price breaches broken wing
        }
        
        # Update with default parameters
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Set spread type
        self.spread_type = ComplexSpreadType.BROKEN_WING_BUTTERFLY
        
        # Register for market events
        self.register_events()
        
        logger.info(f"Initialized Broken Wing Butterfly Strategy for {session.symbol}")
    
    def construct_broken_wing_butterfly(self, option_chain: pd.DataFrame, underlying_price: float) -> Optional[Dict[str, Any]]:
        """
        Construct a broken wing butterfly from the option chain.
        
        Args:
            option_chain: Filtered option chain data
            underlying_price: Current price of the underlying asset
            
        Returns:
            Broken wing butterfly position if successful, None otherwise
        """
        if option_chain is None or option_chain.empty:
            logger.warning("Empty option chain provided")
            return None
        
        # Apply filters for liquidity, open interest, etc.
        filtered_chain = self.filter_option_chains(option_chain)
        
        if filtered_chain is None or filtered_chain.empty:
            logger.warning("No suitable options found after filtering")
            return None
        
        # Extract target parameters
        target_dte = self.parameters['target_days_to_expiry']
        option_type = self.parameters['option_type']
        direction = self.parameters['direction']
        balanced_wing = self.parameters['balanced_wing']
        narrow_wing_width = self.parameters['narrow_wing_width']
        wide_wing_width = self.parameters['wide_wing_width']
        
        # Select expiration based on target days to expiry
        expiration = self.select_expiration_by_dte(filtered_chain, target_dte)
        
        if not expiration:
            logger.warning(f"No suitable expiration found near {target_dte} DTE")
            return None
        
        # Filter options for this expiration
        exp_options = filtered_chain[filtered_chain['expiration_date'] == expiration]
        
        # Filter by option type
        options = exp_options[exp_options['option_type'] == option_type]
        
        if options.empty:
            logger.warning(f"No {option_type} options available for {expiration}")
            return None
        
        # Get available strikes
        available_strikes = sorted(options['strike'].unique())
        
        if len(available_strikes) < 3:
            logger.warning(f"Insufficient strikes available ({len(available_strikes)}), need at least 3")
            return None
        
        # Find closest strike to ATM
        atm_percent = self.parameters['atm_strike_percent'] / 100
        atm_range_low = underlying_price * (1 - atm_percent)
        atm_range_high = underlying_price * (1 + atm_percent)
        
        atm_strike = None
        atm_strike_diff = float('inf')
        
        for strike in available_strikes:
            if atm_range_low <= strike <= atm_range_high:
                strike_diff = abs(strike - underlying_price)
                if strike_diff < atm_strike_diff:
                    atm_strike = strike
                    atm_strike_diff = strike_diff
        
        if atm_strike is None:
            # If no strike in range, use closest available
            atm_strike = min(available_strikes, key=lambda s: abs(s - underlying_price))
        
        # Find strike index of ATM strike
        if atm_strike not in available_strikes:
            logger.warning(f"ATM strike {atm_strike} not in available strikes")
            return None
        
        atm_index = available_strikes.index(atm_strike)
        
        # Select strikes based on direction and balanced wing
        middle_strike_index = atm_index
        
        # For bullish broken wing butterfly (typically a put BWB)
        if direction == 'bullish':
            if balanced_wing == 'lower':
                # Lower balanced, upper broken - typically for bullish BWB
                lower_strike_index = max(0, middle_strike_index - narrow_wing_width)
                upper_strike_index = min(len(available_strikes) - 1, middle_strike_index + wide_wing_width)
            else:  # balanced_wing == 'upper'
                # Upper balanced, lower broken
                lower_strike_index = max(0, middle_strike_index - wide_wing_width)
                upper_strike_index = min(len(available_strikes) - 1, middle_strike_index + narrow_wing_width)
        else:  # direction == 'bearish'
            if balanced_wing == 'lower':
                # Lower balanced, upper broken - but for bearish BWB
                lower_strike_index = max(0, middle_strike_index - narrow_wing_width)
                upper_strike_index = min(len(available_strikes) - 1, middle_strike_index + wide_wing_width)
            else:  # balanced_wing == 'upper'
                # Upper balanced, lower broken
                lower_strike_index = max(0, middle_strike_index - wide_wing_width)
                upper_strike_index = min(len(available_strikes) - 1, middle_strike_index + narrow_wing_width)
        
        # Get actual strikes
        lower_strike = available_strikes[lower_strike_index]
        middle_strike = available_strikes[middle_strike_index]
        upper_strike = available_strikes[upper_strike_index]
        
        # Ensure valid butterfly structure
        if not (lower_strike < middle_strike < upper_strike):
            logger.warning(f"Invalid strike configuration: {lower_strike}, {middle_strike}, {upper_strike}")
            return None
        
        # Get options for each strike
        lower_options = options[options['strike'] == lower_strike]
        middle_options = options[options['strike'] == middle_strike]
        upper_options = options[options['strike'] == upper_strike]
        
        if lower_options.empty or middle_options.empty or upper_options.empty:
            logger.warning("Missing options for one of the selected strikes")
            return None
        
        # Get specific options
        lower_option = lower_options.iloc[0]
        middle_option = middle_options.iloc[0]
        upper_option = upper_options.iloc[0]
        
        # Determine ratios based on balanced wing
        middle_ratio = 2
        outer_ratio = 1
        
        # Create leg objects
        lower_leg = {
            'option_type': option_type,
            'strike': lower_strike,
            'expiration': expiration,
            'action': 'buy',
            'quantity': outer_ratio,
            'entry_price': lower_option['ask'],  # Buy at ask
            'delta': lower_option.get('delta', 0),
            'gamma': lower_option.get('gamma', 0),
            'theta': lower_option.get('theta', 0),
            'vega': lower_option.get('vega', 0)
        }
        
        middle_leg = {
            'option_type': option_type,
            'strike': middle_strike,
            'expiration': expiration,
            'action': 'sell',
            'quantity': middle_ratio,
            'entry_price': middle_option['bid'],  # Sell at bid
            'delta': middle_option.get('delta', 0),
            'gamma': middle_option.get('gamma', 0),
            'theta': middle_option.get('theta', 0),
            'vega': middle_option.get('vega', 0)
        }
        
        upper_leg = {
            'option_type': option_type,
            'strike': upper_strike,
            'expiration': expiration,
            'action': 'buy',
            'quantity': outer_ratio,
            'entry_price': upper_option['ask'],  # Buy at ask
            'delta': upper_option.get('delta', 0),
            'gamma': upper_option.get('gamma', 0),
            'theta': upper_option.get('theta', 0),
            'vega': upper_option.get('vega', 0)
        }
        
        # Calculate net debit/credit
        net_cost = (lower_leg['entry_price'] * outer_ratio) - \
                  (middle_leg['entry_price'] * middle_ratio) + \
                  (upper_leg['entry_price'] * outer_ratio)
        
        # Check if cost is acceptable
        max_debit = underlying_price * self.parameters['max_debit_cost']
        
        # If we prefer credit BWB, check if this is a credit or acceptable debit
        if self.parameters['prefer_credit'] and net_cost > 0 and net_cost > max_debit:
            logger.info(f"BWB rejected: net debit ${net_cost:.2f} exceeds maximum ${max_debit:.2f}")
            return None
        
        # Calculate max profit, max loss and risk/reward
        # For broken wing, these calculations are more complex due to asymmetry
        if option_type == 'call':
            # For call BWB
            if balanced_wing == 'lower':
                # Lower wing balanced: limited loss on downside, potentially uncapped on upside
                max_profit = (middle_strike - lower_strike) * 100 - (net_cost * 100)
                max_loss = net_cost * 100 if net_cost > 0 else 0
            else:
                # Upper wing balanced: potentially uncapped loss on downside, limited on upside
                max_profit = (upper_strike - middle_strike) * 100 - (net_cost * 100)
                max_loss = net_cost * 100 if net_cost > 0 else 0
        else:
            # For put BWB
            if balanced_wing == 'lower':
                # Lower wing balanced: potentially uncapped loss on downside, limited on upside
                max_profit = (middle_strike - lower_strike) * 100 - (net_cost * 100)
                max_loss = net_cost * 100 if net_cost > 0 else 0
            else:
                # Upper wing balanced: limited loss on downside, potentially uncapped on upside
                max_profit = (upper_strike - middle_strike) * 100 - (net_cost * 100)
                max_loss = net_cost * 100 if net_cost > 0 else 0
        
        # Avoid division by zero
        if max_loss <= 0:
            risk_reward = float('inf')
        else:
            risk_reward = max_profit / max_loss
        
        # Create position object
        position = {
            'position_id': str(uuid.uuid4()),
            'lower_leg': lower_leg,
            'middle_leg': middle_leg,
            'upper_leg': upper_leg,
            'entry_time': datetime.now(),
            'spread_type': self.spread_type,
            'option_type': option_type,
            'direction': direction,
            'balanced_wing': balanced_wing,
            'net_cost': net_cost,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'risk_reward': risk_reward,
            'status': 'open',
            'quantity': 1,  # Will be adjusted later based on risk parameters
            'pnl': 0.0
        }
        
        # Calculate Greeks for the full position
        position['net_delta'] = (
            lower_leg['delta'] * outer_ratio + 
            middle_leg['delta'] * middle_ratio + 
            upper_leg['delta'] * outer_ratio
        )
        
        position['net_gamma'] = (
            lower_leg.get('gamma', 0) * outer_ratio + 
            middle_leg.get('gamma', 0) * middle_ratio + 
            upper_leg.get('gamma', 0) * outer_ratio
        )
        
        position['net_theta'] = (
            lower_leg.get('theta', 0) * outer_ratio + 
            middle_leg.get('theta', 0) * middle_ratio + 
            upper_leg.get('theta', 0) * outer_ratio
        )
        
        position['net_vega'] = (
            lower_leg.get('vega', 0) * outer_ratio + 
            middle_leg.get('vega', 0) * middle_ratio + 
            upper_leg.get('vega', 0) * outer_ratio
        )
        
        # Calculate breakeven points
        if option_type == 'call':
            position['lower_breakeven'] = middle_strike - (max_profit / 100)
            position['upper_breakeven'] = upper_strike if balanced_wing == 'upper' else None
        else:  # put
            position['lower_breakeven'] = lower_strike if balanced_wing == 'lower' else None
            position['upper_breakeven'] = middle_strike + (max_profit / 100)
        
        logger.info(f"Constructed {direction} {option_type} broken wing butterfly: strikes {lower_strike}/{middle_strike}/{upper_strike}, "
                   f"cost: ${net_cost * 100:.2f}, max profit: ${max_profit:.2f}, "
                   f"max loss: ${max_loss:.2f}, risk/reward: {risk_reward:.2f}")
        
        return position
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicators for broken wing butterfly signals.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        if data.empty or len(data) < 30:
            return indicators
        
        # Calculate price-based indicators
        if 'close' in data.columns:
            # Moving averages for trend identification
            indicators['sma_20'] = data['close'].rolling(20).mean().iloc[-1]
            indicators['sma_50'] = data['close'].rolling(50).mean().iloc[-1]
            indicators['sma_200'] = data['close'].rolling(200).mean().iloc[-1]
            indicators['ema_8'] = data['close'].ewm(span=8, adjust=False).mean().iloc[-1]
            indicators['ema_21'] = data['close'].ewm(span=21, adjust=False).mean().iloc[-1]
            
            # Momentum indicators
            indicators['rsi_14'] = self.calculate_rsi(data['close'], period=14).iloc[-1]
            indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = self.calculate_macd(data['close'])
            
            # Current price and relative position
            current_price = data['close'].iloc[-1]
            indicators['current_price'] = current_price
            indicators['price_vs_sma20'] = (current_price / indicators['sma_20'] - 1) * 100  # as percentage
            indicators['price_vs_sma50'] = (current_price / indicators['sma_50'] - 1) * 100
            indicators['price_vs_sma200'] = (current_price / indicators['sma_200'] - 1) * 100
            
            # Trend detection
            indicators['trend'] = 'neutral'
            if (indicators['sma_20'] > indicators['sma_50'] and 
                indicators['price_vs_sma20'] > 0 and 
                indicators['price_vs_sma50'] > 0):
                indicators['trend'] = 'uptrend'
            elif (indicators['sma_20'] < indicators['sma_50'] and 
                  indicators['price_vs_sma20'] < 0 and 
                  indicators['price_vs_sma50'] < 0):
                indicators['trend'] = 'downtrend'
            
            # Volatility measures
            indicators['atr'] = self.calculate_atr(data, period=14)
            indicators['atr_pct'] = (indicators['atr'] / current_price) * 100 if indicators['atr'] else None
            
            indicators['bollinger_upper'], indicators['bollinger_middle'], indicators['bollinger_lower'] = \
                self.calculate_bollinger_bands(data['close'], window=20)
            
            # Price channel
            indicators['price_channel_high'] = data['high'].rolling(20).max().iloc[-1]
            indicators['price_channel_low'] = data['low'].rolling(20).min().iloc[-1]
            
            # Historical volatility (annualized 20-day standard deviation)
            try:
                log_returns = np.log(data['close'] / data['close'].shift(1))
                indicators['hv_20'] = log_returns.rolling(20).std() * np.sqrt(252) * 100
                indicators['hv_20_current'] = indicators['hv_20'].iloc[-1]
            except Exception as e:
                logger.warning(f"Error calculating historical volatility: {e}")
            
            # Calculate true range-based position within recent range (percentile)
            try:
                highest_high = data['high'].rolling(20).max()
                lowest_low = data['low'].rolling(20).min()
                price_range = highest_high - lowest_low
                relative_position = (current_price - lowest_low) / price_range
                indicators['range_position'] = relative_position.iloc[-1]
            except Exception as e:
                logger.warning(f"Error calculating range position: {e}")
        
        # Get implied volatility data from session
        if self.session.current_iv is not None:
            indicators['current_iv'] = self.session.current_iv
            
            # IV percentile calculation
            iv_metrics = self.calculate_implied_volatility_metrics(self.iv_history[self.session.symbol])
            indicators['iv_percentile'] = iv_metrics.get('iv_percentile', 50)
            indicators['iv_rank'] = iv_metrics.get('iv_rank', 50)
        
        # Directional and stability scores
        # For BWB, we want to evaluate directional bias and mean-reversion potential
        directional_score = 50  # Neutral starting point
        direction_preference = self.parameters['direction']
        
        # Adjust based on trend
        if direction_preference == 'bullish':
            if indicators['trend'] == 'uptrend':
                directional_score += 20
            elif indicators['trend'] == 'downtrend':
                directional_score -= 25
        else:  # bearish
            if indicators['trend'] == 'downtrend':
                directional_score += 20
            elif indicators['trend'] == 'uptrend':
                directional_score -= 25
        
        # Adjust based on RSI
        rsi = indicators['rsi_14']
        if direction_preference == 'bullish':
            if rsi < 30:  # Oversold
                directional_score += 15  # Bullish potential
            elif rsi > 70:  # Overbought
                directional_score -= 15
        else:  # bearish
            if rsi > 70:  # Overbought
                directional_score += 15  # Bearish potential
            elif rsi < 30:  # Oversold
                directional_score -= 15
        
        # Adjust based on MACD
        macd_hist = indicators['macd_hist']
        if direction_preference == 'bullish' and macd_hist > 0:
            directional_score += 10
        elif direction_preference == 'bearish' and macd_hist < 0:
            directional_score += 10
        
        # Adjust based on price vs moving averages
        if direction_preference == 'bullish':
            if indicators['price_vs_sma20'] > 0 and indicators['price_vs_sma50'] > 0:
                directional_score += 10
            elif indicators['price_vs_sma20'] < -3 and indicators['price_vs_sma50'] < -3:
                directional_score += 5  # Potential for mean reversion
        else:  # bearish
            if indicators['price_vs_sma20'] < 0 and indicators['price_vs_sma50'] < 0:
                directional_score += 10
            elif indicators['price_vs_sma20'] > 3 and indicators['price_vs_sma50'] > 3:
                directional_score += 5  # Potential for mean reversion
        
        # Adjust based on range position
        range_pos = indicators.get('range_position', 0.5)
        if direction_preference == 'bullish' and range_pos < 0.3:  # Near bottom of range
            directional_score += 15
        elif direction_preference == 'bearish' and range_pos > 0.7:  # Near top of range
            directional_score += 15
        
        # Ensure score is within valid range
        indicators['directional_bias_score'] = max(0, min(100, directional_score))
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for broken wing butterfly based on market conditions.
        
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
            # Check IV environment - BWB can work in various IV environments depending on structure
            iv_check_passed = True
            if 'iv_percentile' in indicators:
                iv_percentile = indicators['iv_percentile']
                min_iv = self.parameters['min_iv_percentile']
                max_iv = self.parameters['max_iv_percentile']
                
                if iv_percentile < min_iv or iv_percentile > max_iv:
                    iv_check_passed = False
                    logger.info(f"IV filter failed: current IV percentile {iv_percentile:.2f} outside range [{min_iv}-{max_iv}]")
            
            # Check directional bias score
            if 'directional_bias_score' in indicators and iv_check_passed:
                directional_score = indicators['directional_bias_score']
                threshold = self.parameters['bias_score_threshold']
                
                if directional_score >= threshold:
                    # Calculate signal strength (0.0 to 1.0)
                    signal_strength = directional_score / 100.0
                    
                    # Generate entry signal
                    signals["entry"] = True
                    signals["signal_strength"] = signal_strength
                    
                    logger.info(f"Broken wing butterfly entry signal: directional bias score {directional_score}, "
                               f"strength {signal_strength:.2f}, direction preference: {self.parameters['direction']}")
                else:
                    logger.info(f"Directional bias score {directional_score} below threshold {threshold}")
        
        # Check for exits on existing positions
        for position in self.positions:
            if position.get('status') == "open":
                exit_signal = self._check_exit_conditions(position, data, indicators)
                if exit_signal:
                    signals["exit_positions"].append(position['position_id'])
        
        return signals
    
    def _check_exit_conditions(self, position: Dict[str, Any], data: pd.DataFrame, indicators: Dict[str, Any]) -> bool:
        """
        Check exit conditions for an open broken wing butterfly position.
        
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
        
        # 1. Expiration approach
        if 'lower_leg' in position and 'expiration' in position['lower_leg']:
            expiry_date = position['lower_leg'].get('expiration')
            if isinstance(expiry_date, str):
                expiry_date = datetime.strptime(expiry_date, '%Y-%m-%d').date()
            
            days_to_expiry = (expiry_date - datetime.now().date()).days
            
            # Exit ahead of expiration to avoid gamma risk and pin risk
            if days_to_expiry <= self.parameters['days_before_expiry_exit']:
                logger.info(f"Exit signal: approaching expiration ({days_to_expiry} days left)")
                return True
        
        # 2. Profit target check
        current_price = data['close'].iloc[-1]
        
        # Calculate theoretical profit percentage
        profit_pct = self._calculate_theoretical_profit(position, current_price)
        
        if profit_pct is not None:
            target_profit_pct = self.parameters['target_profit_pct'] / 100
            
            if profit_pct >= target_profit_pct:
                logger.info(f"Exit signal: profit target reached (est. {profit_pct:.1%})")
                return True
        
        # 3. Stop loss check
        if profit_pct is not None:
            stop_loss_pct = -self.parameters['stop_loss_pct'] / 100
            
            if profit_pct <= stop_loss_pct:
                logger.info(f"Exit signal: stop loss triggered (est. {profit_pct:.1%})")
                return True
        
        # 4. Check for wing breach (price moving outside the wings)
        option_type = position.get('option_type')
        direction = position.get('direction')
        balanced_wing = position.get('balanced_wing')
        
        if 'lower_leg' in position and 'middle_leg' in position and 'upper_leg' in position:
            lower_strike = position['lower_leg'].get('strike', 0)
            middle_strike = position['middle_leg'].get('strike', 0)
            upper_strike = position['upper_leg'].get('strike', 0)
            
            # Check price in relation to strikes
            if option_type == 'call':
                if balanced_wing == 'lower':
                    # Price moving above the broken wing is risky
                    if current_price > upper_strike * 1.02:  # 2% above upper strike
                        logger.info(f"Exit signal: price above broken upper wing ({upper_strike})")
                        return True
                else:  # balanced_wing == 'upper'
                    # Price moving below the broken wing is risky
                    if current_price < lower_strike * 0.98:  # 2% below lower strike
                        logger.info(f"Exit signal: price below broken lower wing ({lower_strike})")
                        return True
            else:  # put
                if balanced_wing == 'lower':
                    # Price moving below the broken wing is risky
                    if current_price < lower_strike * 0.98:  # 2% below lower strike
                        logger.info(f"Exit signal: price below broken lower wing ({lower_strike})")
                        return True
                else:  # balanced_wing == 'upper'
                    # Price moving above the broken wing is risky
                    if current_price > upper_strike * 1.02:  # 2% above upper strike
                        logger.info(f"Exit signal: price above broken upper wing ({upper_strike})")
                        return True
        
        # 5. Check for directional bias change - exit if original bias is invalidated
        if 'directional_bias_score' in indicators:
            original_direction = position.get('direction')
            current_score = indicators['directional_bias_score']
            
            # If the direction is now the opposite of our position with a strong signal
            if (original_direction == 'bullish' and current_score < 35) or \
               (original_direction == 'bearish' and current_score > 65):
                logger.info(f"Exit signal: directional bias reversed, original: {original_direction}, score: {current_score}")
                return True
        
        return False
    
    def _calculate_theoretical_profit(self, position: Dict[str, Any], current_price: float) -> Optional[float]:
        """
        Calculate theoretical profit percentage for a broken wing butterfly.
        
        Args:
            position: The position to evaluate
            current_price: Current price of the underlying
        
        Returns:
            Estimated profit percentage or None if calculation not possible
        """
        if 'lower_leg' not in position or 'middle_leg' not in position or 'upper_leg' not in position:
            return None
        
        # Extract position details
        lower_leg = position['lower_leg']
        middle_leg = position['middle_leg']
        upper_leg = position['upper_leg']
        net_cost = position.get('net_cost', 0)
        max_profit = position.get('max_profit', 0)
        max_loss = position.get('max_loss', 0)
        
        # Get key parameters
        option_type = position.get('option_type')
        balanced_wing = position.get('balanced_wing')
        
        # Get strikes
        lower_strike = lower_leg.get('strike')
        middle_strike = middle_leg.get('strike')
        upper_strike = upper_leg.get('strike')
        
        # Get expiration for time value calculation
        expiry = lower_leg.get('expiration')
        if isinstance(expiry, str):
            expiry = datetime.strptime(expiry, '%Y-%m-%d').date()
        
        days_to_expiry = max(1, (expiry - datetime.now().date()).days)
        original_dte = self.parameters['target_days_to_expiry']
        time_decay_factor = days_to_expiry / original_dte
        
        # Simplified theoretical valuation based on current price, expiry, and strikes
        # In a real implementation, would use Black-Scholes or other pricing model
        
        # Intrinsic values at current price
        if option_type == 'call':
            lower_intrinsic = max(0, current_price - lower_strike)
            middle_intrinsic = max(0, current_price - middle_strike)
            upper_intrinsic = max(0, current_price - upper_strike)
        else:  # put
            lower_intrinsic = max(0, lower_strike - current_price)
            middle_intrinsic = max(0, middle_strike - current_price)
            upper_intrinsic = max(0, upper_strike - current_price)
        
        # Time values (simplified approximation)
        lower_time_value = (lower_leg.get('entry_price', 0) - lower_intrinsic) * time_decay_factor
        middle_time_value = (middle_leg.get('entry_price', 0) - middle_intrinsic) * time_decay_factor
        upper_time_value = (upper_leg.get('entry_price', 0) - upper_intrinsic) * time_decay_factor
        
        # Current theoretical values
        lower_value = lower_intrinsic + lower_time_value
        middle_value = middle_intrinsic + middle_time_value
        upper_value = upper_intrinsic + upper_time_value
        
        # Calculate current cost to close
        # If we bought, we sell; if we sold, we buy
        outer_ratio = 1
        middle_ratio = 2
        current_close_value = (lower_value * -outer_ratio) + (middle_value * middle_ratio) + (upper_value * -outer_ratio)
        
        # Original cost
        original_cost = net_cost
        
        # Current P&L
        current_pnl = current_close_value - original_cost
        
        # Express as percentage of max profit/loss
        if original_cost > 0:  # We paid a debit
            if current_pnl > 0:  # Profit
                profit_pct = current_pnl / max_profit if max_profit > 0 else 0
            else:  # Loss
                profit_pct = current_pnl / max_loss if max_loss > 0 else 0
        else:  # We received a credit
            if current_pnl > 0:  # Profit
                profit_pct = current_pnl / abs(original_cost) if original_cost != 0 else 0
            else:  # Loss
                profit_pct = current_pnl / max_loss if max_loss > 0 else 0
        
        return profit_pct
    
    def _execute_signals(self):

        # Verify account has sufficient buying power
        buying_power = self.get_buying_power()
        if buying_power <= 0:
            logger.warning("Trade execution aborted: Insufficient buying power")
            return
        """
        Execute Broken Wing Butterfly specific trading signals.
        """
        if not self.signals:
            return
        
        # Handle entry signals
        if self.signals.get("entry", False):
            # Construct and open broken wing butterfly position
            underlying_price = self.session.current_price
            
            if underlying_price and self.session.option_chain is not None:
                # Use the construct_broken_wing_butterfly method for position creation
                bwb_position = self.construct_broken_wing_butterfly(
                    self.session.option_chain, 
                    underlying_price
                )
                
                if bwb_position:
                    # Determine position size based on risk
                    account_value = self.session.account_value
                    max_risk_pct = self.parameters['max_risk_per_trade_pct'] / 100
                    max_risk_amount = account_value * max_risk_pct
                    
                    # Calculate position size based on max risk
                    position_max_loss = bwb_position.get('max_loss', float('inf'))
                    
                    if position_max_loss < float('inf') and position_max_loss > 0:
                        # Number of BWBs to trade
                        num_spreads = int(max_risk_amount / position_max_loss)
                        num_spreads = max(1, min(5, num_spreads))  # At least 1, at most 5 spreads
                        bwb_position['quantity'] = num_spreads
                    else:
                        bwb_position['quantity'] = 1  # Default to 1 if risk can't be calculated
                    
                    # Add position
                    self.positions.append(bwb_position)
                    
                    # Log entry details
                    option_type = bwb_position.get('option_type', 'call')
                    direction = bwb_position.get('direction', 'bullish')
                    balanced_wing = bwb_position.get('balanced_wing', 'lower')
                    lower_strike = bwb_position.get('lower_leg', {}).get('strike', 0)
                    middle_strike = bwb_position.get('middle_leg', {}).get('strike', 0)
                    upper_strike = bwb_position.get('upper_leg', {}).get('strike', 0)
                    net_cost = bwb_position.get('net_cost', 0)
                    quantity = bwb_position.get('quantity', 1)
                    
                    logger.info(f"Opened {direction} {option_type} broken wing butterfly position {bwb_position['position_id']}: "  
                              f"strikes {lower_strike}/{middle_strike}/{upper_strike}, "  
                              f"{'debit' if net_cost > 0 else 'credit'}: ${abs(net_cost) * 100 * quantity:.2f}, "
                              f"quantity: {quantity}, balanced wing: {balanced_wing}")
                else:
                    logger.warning("Failed to construct valid broken wing butterfly")
        
        # Handle exit signals
        for position_id in self.signals.get("exit_positions", []):
            for i, position in enumerate(self.positions):
                if position.get('position_id') == position_id and position.get('status') == "open":
                    # In a real implementation, would close with actual market prices
                    # Here we'll use a simplified approach with theoretical values
                    
                    # Mark position as closed
                    position['status'] = "closed"
                    position['exit_time'] = datetime.now()
                    
                    # Calculate P&L (simplified)
                    current_price = self.session.current_price
                    profit_pct = self._calculate_theoretical_profit(position, current_price)
                    
                    if profit_pct is not None:
                        max_profit = position.get('max_profit', 0)
                        quantity = position.get('quantity', 1)
                        pnl = profit_pct * max_profit * quantity
                        position['pnl'] = pnl
                        
                        logger.info(f"Closed broken wing butterfly position {position_id}, P&L: ${pnl:.2f}")
                    else:
                        logger.info(f"Closed broken wing butterfly position {position_id}, P&L calculation not available")
    
    def run_strategy(self):
        """
        Run the broken wing butterfly strategy cycle.
        """
        # Check if we have necessary data
        if not self.session.current_price or self.session.option_chain is None:
            logger.warning("Missing current price or option chain data, skipping strategy execution")
            return
        
        # Get market data
        data = self.get_historical_data(self.session.symbol, '1d', 60)
        
        if data is not None and not data.empty:
            # Calculate indicators
            self.indicators = self.calculate_indicators(data)
            
            # Generate signals
            self.signals = self.generate_signals(data, self.indicators)
            
            # Execute signals
            self._execute_signals()
        else:
            logger.warning("Insufficient market data for broken wing butterfly analysis")
    
    def register_events(self):
        """Register for events relevant to Broken Wing Butterfly strategy."""
        # Register for common event types from the parent class
        super().register_events()
        
        # Add BWB specific event subscriptions
        EventBus.subscribe(EventType.EARNINGS_ANNOUNCEMENT, self.on_event)
        EventBus.subscribe(EventType.IMPLIED_VOLATILITY_CHANGE, self.on_event)
        EventBus.subscribe(EventType.MARKET_REGIME_CHANGE, self.on_event)
    
    def on_event(self, event: Event):
        """
        Process incoming events for the Broken Wing Butterfly strategy.
        
        Args:
            event: The event to process
        """
        # Let parent class handle common events first
        super().on_event(event)
        
        # Add additional BWB specific event handling
        if event.type == EventType.EARNINGS_ANNOUNCEMENT and self.session.symbol in event.data.get('symbols', []):
            # Earnings announcements can significantly impact BWB positions
            days_to_earnings = event.data.get('days_to_announcement', 0)
            
            if days_to_earnings <= 5:  # Within 5 days of earnings
                logger.info(f"Earnings announcement approaching for {self.session.symbol}, evaluating BWB positions")
                
                # Check position type - for earnings-neutral BWBs, might be ok to hold through earnings
                # But directional BWBs should be closed
                for position in self.positions:
                    if position.get('status') == "open":
                        direction = position.get('direction', 'unknown')
                        
                        # Close directional BWBs, especially those with broken wings that have unlimited risk
                        if direction in ['bullish', 'bearish']:
                            logger.info(f"Closing {direction} BWB position {position.get('position_id')} ahead of earnings")
                            self.signals.setdefault("exit_positions", []).append(position.get('position_id'))
                
                # Execute the exits
                self._execute_signals()
        
        elif event.type == EventType.IMPLIED_VOLATILITY_CHANGE:
            symbol = event.data.get('symbol')
            magnitude = event.data.get('magnitude', 0)
            direction = event.data.get('direction', '')
            
            if symbol == self.session.symbol and abs(magnitude) > 15:  # Significant IV change
                logger.info(f"Significant IV {direction} of {magnitude}% detected")
                
                # BWB response depends on whether it was established for vega positive or negative
                # Typically BWBs are slightly vega positive
                if direction == 'crash' and magnitude > 30:  # Major IV crash
                    logger.info(f"Significant IV crash may impact BWB positions")
                    
                    # Close positions if there's a major IV crash, especially vega positive BWBs
                    for position in self.positions:
                        if position.get('status') == "open":
                            if position.get('net_vega', 0) > 0:  # Vega positive position
                                logger.info(f"Closing vega positive BWB {position.get('position_id')} due to IV crash")
                                self.signals.setdefault("exit_positions", []).append(position.get('position_id'))
                
                # Execute the exits if needed
                if "exit_positions" in self.signals and self.signals["exit_positions"]:
                    self._execute_signals()
        
        elif event.type == EventType.MARKET_REGIME_CHANGE:
            new_regime = event.data.get('new_regime', '')
            old_regime = event.data.get('old_regime', '')
            
            # BWB is sensitive to market regime changes, especially involving volatility
            regime_pairs = [(old_regime, new_regime)]
            
            # Problematic regime transitions for BWB
            problematic_transitions = [
                ('neutral-low-volatility', 'bearish-high-volatility'),
                ('bullish-low-volatility', 'bearish-high-volatility'),
                ('neutral-low-volatility', 'crash'),
                ('bullish-low-volatility', 'crash')
            ]
            
            for old, new in regime_pairs:
                if (old, new) in problematic_transitions:
                    logger.info(f"Unfavorable market regime change from {old} to {new}, evaluating BWB positions")
                    
                    # Evaluate each position based on its direction
                    for position in self.positions:
                        if position.get('status') == "open":
                            direction = position.get('direction', 'unknown')
                            balanced_wing = position.get('balanced_wing', 'unknown')
                            
                            # Critical scenarios requiring immediate exit
                            if (direction == 'bullish' and new in ['bearish-high-volatility', 'crash']) or \
                               (direction == 'bearish' and new in ['bullish-high-volatility']) or \
                               (balanced_wing == 'lower' and new in ['crash']) or \
                               (balanced_wing == 'upper' and new in ['crash']):
                                logger.info(f"Closing BWB position {position.get('position_id')} due to unfavorable regime change to {new}")
                                self.signals.setdefault("exit_positions", []).append(position.get('position_id'))
                    
                    # Execute the exits
                    if "exit_positions" in self.signals and self.signals["exit_positions"]:
                        self._execute_signals()
