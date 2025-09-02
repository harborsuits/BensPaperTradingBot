#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Double Calendar Strategy

A professional-grade double calendar implementation that leverages the modular,
event-driven architecture. This is a time spread strategy that combines a call calendar
spread with a put calendar spread to create a non-directional strategy that benefits
from time decay and volatility expansion.
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
from trading_bot.strategies_new.options.base.time_spread_engine import TimeSpreadEngine, TimeSpreadType
from trading_bot.strategies_new.factory.registry import register_strategy
from trading_bot.strategies_new.validation.account_aware_mixin import AccountAwareMixin

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="DoubleCalendarStrategy",
    market_type="options",
    description="A strategy that combines both a call calendar spread and a put calendar spread with the same strikes to create a non-directional strategy benefiting from time decay and volatility expansion",
    timeframes=["1d", "1w"],
    parameters={
        "near_dte_target": {"description": "Target DTE for near month expiration", "type": "integer"},
        "far_dte_target": {"description": "Target DTE for far month expiration", "type": "integer"},
        "min_dte_diff": {"description": "Minimum DTE difference between near and far expiry", "type": "integer"},
        "range_confidence": {"description": "Standard deviation multiplier for expected price range", "type": "float"}
    }
)
class DoubleCalendarStrategy(TimeSpreadEngine, AccountAwareMixin):
    """
    Double Calendar Strategy
    
    This strategy combines both a call calendar spread and a put calendar spread with the same
    or similar strikes to create a non-directional strategy that benefits from time decay
    and volatility expansion while the underlying stays within a range. It's essentially two
    calendar spreads combined, typically with the strikes near the current price.
    
    Features:
    - Adapts the legacy double calendar implementation to the new architecture
    - Uses event-driven signal generation and position management
    - Leverages the TimeSpreadEngine for core time spread mechanics
    - Implements custom filtering, technical/volatility analysis, and range-bound detection
    """
    
    def __init__(self, session: OptionsSession, data_pipeline: DataPipeline, 
                 parameters: Dict[str, Any] = None):
        """
        Initialize the Double Calendar strategy.
        
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
            'strategy_name': 'Double Calendar',
            'strategy_id': 'double_calendar',
            
            # Double Calendar specific parameters
            'atm_strike_percent': 2.0,       # Select strike within +/- this % of current price
            'near_dte_target': 30,           # Target DTE for near month expiration
            'far_dte_target': 60,            # Target DTE for far month expiration
            'min_dte_diff': 15,              # Minimum DTE difference between near and far expiry
            'max_dte_diff': 45,              # Maximum DTE difference between near and far expiry
            
            # Range prediction parameters
            'range_periods': 20,             # Look-back periods for expected range calculation
            'range_confidence': 1.0,         # Standard deviation multiplier for range estimate
            'range_bound_score_threshold': 65, # Min score for range-bound market (0-100)
            
            # Market condition preferences
            'min_iv_percentile': 45,         # Minimum IV percentile for entry
            'max_iv_percentile': 90,         # Maximum IV percentile for entry (lower is better for buying far legs)
            'min_iv_term_structure': 1.0,    # Minimum IV term structure ratio (far/near)
            'iv_time_preference': 'contango', # Preferred IV term structure: 'contango' or 'backwardation'
            
            # Risk parameters
            'max_risk_per_trade_pct': 1.5,   # Max risk as % of account
            'target_profit_pct': 40,         # Target profit as % of max risk
            'stop_loss_pct': 60,             # Stop loss as % of max risk
            'max_positions': 2,              # Maximum concurrent positions
            
            # Exit parameters
            'near_days_before_expiry_exit': 7, # Exit when near expiration reaches this DTE
            'max_wing_width_pct': 10.0,      # Maximum width between strikes as % of underlying price
            'max_vega_risk': 0.15,           # Maximum vega risk allowed as % of account value
        }
        
        # Update with default parameters
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Set spread type
        self.spread_type = TimeSpreadType.DOUBLE_CALENDAR
        
        # Register for market events
        self.register_events()
        
        logger.info(f"Initialized Double Calendar Strategy for {session.symbol}")
    
    def construct_double_calendar(self, option_chain: pd.DataFrame, underlying_price: float) -> Optional[Dict[str, Any]]:
        """
        Construct a double calendar from the option chain.
        
        Args:
            option_chain: Filtered option chain data
            underlying_price: Current price of the underlying asset
            
        Returns:
            Double calendar position if successful, None otherwise
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
        near_dte_target = self.parameters['near_dte_target']
        far_dte_target = self.parameters['far_dte_target']
        min_dte_diff = self.parameters['min_dte_diff']
        max_dte_diff = self.parameters['max_dte_diff']
        atm_percent = self.parameters['atm_strike_percent'] / 100
        
        # Define ATM strike range
        atm_range_low = underlying_price * (1 - atm_percent)
        atm_range_high = underlying_price * (1 + atm_percent)
        
        # Find expiration dates closest to targets
        expirations = sorted(filtered_chain['expiration_date'].unique())
        
        if len(expirations) < 2:
            logger.warning("Not enough expiration dates available for calendar spread")
            return None
        
        # Convert expirations to datetime if they're strings
        expiration_dates = []
        for exp in expirations:
            if isinstance(exp, str):
                try:
                    exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                    expiration_dates.append(exp_date)
                except ValueError:
                    continue
            elif isinstance(exp, (date, datetime)):
                if isinstance(exp, datetime):
                    expiration_dates.append(exp.date())
                else:
                    expiration_dates.append(exp)
        
        # Sort expiration dates
        expiration_dates = sorted(expiration_dates)
        
        # Calculate DTE for each expiration
        today = datetime.now().date()
        dte_values = [(exp_date, (exp_date - today).days) for exp_date in expiration_dates]
        
        # Find near and far expirations
        near_expiry = None
        far_expiry = None
        near_dte = 0
        far_dte = 0
        
        for exp_date, dte in dte_values:
            # Find near expiration
            if near_expiry is None and abs(dte - near_dte_target) < abs(near_dte - near_dte_target):
                near_expiry = exp_date
                near_dte = dte
            
            # Find far expiration
            if far_expiry is None and abs(dte - far_dte_target) < abs(far_dte - far_dte_target):
                far_expiry = exp_date
                far_dte = dte
            
            # Update near expiration if this one is closer to target
            if near_expiry and abs(dte - near_dte_target) < abs(near_dte - near_dte_target) and dte >= 7:  # At least 7 DTE
                near_expiry = exp_date
                near_dte = dte
            
            # Update far expiration if this one is closer to target
            if far_expiry and abs(dte - far_dte_target) < abs(far_dte - far_dte_target) and dte > near_dte + min_dte_diff:
                far_expiry = exp_date
                far_dte = dte
        
        # Validate expiration selection
        if not near_expiry or not far_expiry:
            logger.warning("Could not find suitable near and far expirations")
            return None
        
        if near_expiry >= far_expiry:
            logger.warning(f"Near expiry ({near_expiry}) must be before far expiry ({far_expiry})")
            return None
        
        dte_diff = far_dte - near_dte
        if dte_diff < min_dte_diff:
            logger.warning(f"DTE difference ({dte_diff}) is less than minimum required ({min_dte_diff})")
            return None
        
        if dte_diff > max_dte_diff:
            logger.warning(f"DTE difference ({dte_diff}) is greater than maximum allowed ({max_dte_diff})")
            return None
        
        # Get options for both expirations
        near_options = filtered_chain[filtered_chain['expiration_date'] == near_expiry.strftime('%Y-%m-%d')]
        far_options = filtered_chain[filtered_chain['expiration_date'] == far_expiry.strftime('%Y-%m-%d')]
        
        if near_options.empty or far_options.empty:
            logger.warning("Missing options for one of the selected expirations")
            return None
        
        # Find ATM strike for calls and puts
        near_calls = near_options[near_options['option_type'] == 'call']
        near_puts = near_options[near_options['option_type'] == 'put']
        far_calls = far_options[far_options['option_type'] == 'call']
        far_puts = far_options[far_options['option_type'] == 'put']
        
        if near_calls.empty or near_puts.empty or far_calls.empty or far_puts.empty:
            logger.warning("Missing required option types for double calendar")
            return None
        
        # Find strikes within ATM range
        atm_near_calls = near_calls[(near_calls['strike'] >= atm_range_low) & (near_calls['strike'] <= atm_range_high)]
        atm_near_puts = near_puts[(near_puts['strike'] >= atm_range_low) & (near_puts['strike'] <= atm_range_high)]
        
        if atm_near_calls.empty or atm_near_puts.empty:
            logger.warning("No near-term options available within ATM range")
            return None
        
        # Find the strike closest to current price for call calendar
        call_strike_diff = float('inf')
        call_strike = None
        
        for strike in sorted(atm_near_calls['strike'].unique()):
            # Check if same strike is available for far expiration
            matching_far_calls = far_calls[far_calls['strike'] == strike]
            if not matching_far_calls.empty:
                diff = abs(strike - underlying_price)
                if diff < call_strike_diff:
                    call_strike = strike
                    call_strike_diff = diff
        
        # Find the strike closest to current price for put calendar
        put_strike_diff = float('inf')
        put_strike = None
        
        for strike in sorted(atm_near_puts['strike'].unique()):
            # Check if same strike is available for far expiration
            matching_far_puts = far_puts[far_puts['strike'] == strike]
            if not matching_far_puts.empty:
                diff = abs(strike - underlying_price)
                if diff < put_strike_diff:
                    put_strike = strike
                    put_strike_diff = diff
        
        if call_strike is None or put_strike is None:
            logger.warning("Could not find matching strikes for both call and put calendars")
            return None
        
        # Get specific option contracts
        near_call = near_calls[near_calls['strike'] == call_strike].iloc[0]
        far_call = far_calls[far_calls['strike'] == call_strike].iloc[0]
        near_put = near_puts[near_puts['strike'] == put_strike].iloc[0]
        far_put = far_puts[far_puts['strike'] == put_strike].iloc[0]
        
        # Create leg objects
        near_call_leg = {
            'option_type': 'call',
            'strike': call_strike,
            'expiration': near_expiry,
            'action': 'sell',
            'quantity': 1,
            'entry_price': near_call['bid'],  # Sell at bid
            'delta': near_call.get('delta', 0),
            'gamma': near_call.get('gamma', 0),
            'theta': near_call.get('theta', 0),
            'vega': near_call.get('vega', 0)
        }
        
        far_call_leg = {
            'option_type': 'call',
            'strike': call_strike,
            'expiration': far_expiry,
            'action': 'buy',
            'quantity': 1,
            'entry_price': far_call['ask'],  # Buy at ask
            'delta': far_call.get('delta', 0),
            'gamma': far_call.get('gamma', 0),
            'theta': far_call.get('theta', 0),
            'vega': far_call.get('vega', 0)
        }
        
        near_put_leg = {
            'option_type': 'put',
            'strike': put_strike,
            'expiration': near_expiry,
            'action': 'sell',
            'quantity': 1,
            'entry_price': near_put['bid'],  # Sell at bid
            'delta': near_put.get('delta', 0),
            'gamma': near_put.get('gamma', 0),
            'theta': near_put.get('theta', 0),
            'vega': near_put.get('vega', 0)
        }
        
        far_put_leg = {
            'option_type': 'put',
            'strike': put_strike,
            'expiration': far_expiry,
            'action': 'buy',
            'quantity': 1,
            'entry_price': far_put['ask'],  # Buy at ask
            'delta': far_put.get('delta', 0),
            'gamma': far_put.get('gamma', 0),
            'theta': far_put.get('theta', 0),
            'vega': far_put.get('vega', 0)
        }
        
        # Calculate net debit
        net_debit = (far_call_leg['entry_price'] - near_call_leg['entry_price'] + 
                     far_put_leg['entry_price'] - near_put_leg['entry_price'])
        
        # Calculate Greeks for the full position
        net_delta = (near_call_leg['delta'] + far_call_leg['delta'] + 
                    near_put_leg['delta'] + far_put_leg['delta'])
        
        net_gamma = (near_call_leg.get('gamma', 0) + far_call_leg.get('gamma', 0) + 
                    near_put_leg.get('gamma', 0) + far_put_leg.get('gamma', 0))
        
        net_theta = (near_call_leg.get('theta', 0) + far_call_leg.get('theta', 0) + 
                    near_put_leg.get('theta', 0) + far_put_leg.get('theta', 0))
        
        net_vega = (near_call_leg.get('vega', 0) + far_call_leg.get('vega', 0) + 
                   near_put_leg.get('vega', 0) + far_put_leg.get('vega', 0))
        
        # Calculate max risk (typically the net debit paid)
        max_risk = net_debit * 100
        
        # Create position object
        position = {
            'position_id': str(uuid.uuid4()),
            'near_call_leg': near_call_leg,
            'far_call_leg': far_call_leg,
            'near_put_leg': near_put_leg,
            'far_put_leg': far_put_leg,
            'entry_time': datetime.now(),
            'spread_type': self.spread_type,
            'net_debit': net_debit,
            'call_strike': call_strike,
            'put_strike': put_strike,
            'near_dte': near_dte,
            'far_dte': far_dte,
            'max_risk': max_risk,
            'status': 'open',
            'quantity': 1,  # Will be adjusted later based on risk parameters
            'pnl': 0.0,
            'net_delta': net_delta,
            'net_gamma': net_gamma,
            'net_theta': net_theta,
            'net_vega': net_vega
        }
        
        # Calculate expected profit range based on the strikes and underlying price
        # This is simplified; in reality would use more sophisticated modeling
        position['profit_range_low'] = min(call_strike, put_strike) * 0.95
        position['profit_range_high'] = max(call_strike, put_strike) * 1.05
        
        logger.info(f"Constructed double calendar: call strike {call_strike}, put strike {put_strike}, "
                   f"near DTE {near_dte}, far DTE {far_dte}, net debit: ${net_debit * 100:.2f}")
        
        return position
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicators for double calendar signals.
        
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
            
            # Price channel and range
            range_periods = self.parameters['range_periods']
            indicators['price_channel_high'] = data['high'].rolling(range_periods).max().iloc[-1]
            indicators['price_channel_low'] = data['low'].rolling(range_periods).min().iloc[-1]
            indicators['price_range'] = indicators['price_channel_high'] - indicators['price_channel_low']
            indicators['price_range_pct'] = (indicators['price_range'] / current_price) * 100
            
            # Calculate position within range
            if indicators['price_range'] > 0:
                indicators['range_position'] = (current_price - indicators['price_channel_low']) / indicators['price_range']
            else:
                indicators['range_position'] = 0.5
            
            # Historical volatility (annualized 20-day standard deviation)
            try:
                log_returns = np.log(data['close'] / data['close'].shift(1))
                indicators['hv_20'] = log_returns.rolling(20).std() * np.sqrt(252) * 100
                indicators['hv_20_current'] = indicators['hv_20'].iloc[-1]
            except Exception as e:
                logger.warning(f"Error calculating historical volatility: {e}")
            
            # Calculate expected price range for double calendar timeframe
            expected_range_days = self.parameters['near_dte_target']  # Use near DTE as timeframe
            confidence = self.parameters['range_confidence']
            
            # If we have historical volatility, use it to calculate expected range
            if 'hv_20_current' in indicators:
                hv = indicators['hv_20_current'] / 100  # Convert from percentage
                # Expected 1 standard deviation move over the period
                expected_move = current_price * hv * np.sqrt(expected_range_days / 252) * confidence
                indicators['expected_range_low'] = current_price - expected_move
                indicators['expected_range_high'] = current_price + expected_move
                indicators['expected_range_pct'] = (expected_move * 2 / current_price) * 100
            else:
                # Fallback - use recent range as proxy
                indicators['expected_range_low'] = current_price * 0.95
                indicators['expected_range_high'] = current_price * 1.05
                indicators['expected_range_pct'] = 10.0
        
        # Get implied volatility data from session
        if self.session.current_iv is not None:
            indicators['current_iv'] = self.session.current_iv
            
            # IV percentile calculation
            iv_metrics = self.calculate_implied_volatility_metrics(self.iv_history[self.session.symbol])
            indicators['iv_percentile'] = iv_metrics.get('iv_percentile', 50)
            indicators['iv_rank'] = iv_metrics.get('iv_rank', 50)
            
            # Term structure information if available
            if hasattr(self.session, 'iv_term_structure') and self.session.iv_term_structure is not None:
                indicators['iv_term_structure'] = self.session.iv_term_structure
                
                # Calculate implied volatility at near and far DTEs
                near_dte = self.parameters['near_dte_target']
                far_dte = self.parameters['far_dte_target']
                
                # Get IVs for our target DTEs (simplistic approach - in real implementation would interpolate)
                near_iv = self.session.iv_term_structure.get(near_dte, self.session.current_iv)
                far_iv = self.session.iv_term_structure.get(far_dte, self.session.current_iv)
                
                indicators['near_iv'] = near_iv
                indicators['far_iv'] = far_iv
                indicators['iv_term_structure_ratio'] = far_iv / near_iv if near_iv > 0 else 1.0
        
        # Calculate range-bound market score (0-100)
        # Double Calendar works best in range-bound markets
        range_score = 50  # Neutral starting point
        
        # Adjust based on trend strength
        if indicators['trend'] == 'neutral':
            range_score += 25  # Neutral trend is good for range-bound strategies
        else:
            # Check if trend is weak (close to flat)
            if abs(indicators['price_vs_sma20']) < 1.5 and abs(indicators['price_vs_sma50']) < 2.0:
                range_score += 15  # Weak trend is still acceptable
            else:
                range_score -= 20  # Strong trend is bad for range-bound strategies
        
        # Adjust based on historical volatility vs current ATR
        if 'hv_20_current' in indicators and 'atr_pct' in indicators:
            expected_daily_vol = indicators['hv_20_current'] / np.sqrt(252)
            actual_daily_vol = indicators['atr_pct']
            
            # If actual volatility is lower than expected, market may be consolidating
            if actual_daily_vol < expected_daily_vol * 0.8:
                range_score += 15  # Consolidation is good for range-bound strategies
            elif actual_daily_vol > expected_daily_vol * 1.5:
                range_score -= 15  # Expanding volatility could indicate breakout
        
        # Adjust based on RSI (middle RSI values are good for range-bound strategies)
        rsi = indicators['rsi_14']
        if 40 <= rsi <= 60:
            range_score += 15  # Middle RSI values indicate balance between buyers and sellers
        elif 30 <= rsi < 40 or 60 < rsi <= 70:
            range_score += 5   # Still somewhat balanced
        elif rsi < 30 or rsi > 70:
            range_score -= 10  # Extreme RSI could indicate trend continuation
        
        # Adjust based on range position
        range_pos = indicators.get('range_position', 0.5)
        if 0.3 <= range_pos <= 0.7:
            range_score += 10  # Price in middle of range is good
        elif range_pos < 0.2 or range_pos > 0.8:
            range_score -= 15  # Price near edge of range could break out
        
        # Adjust based on IV term structure
        iv_ratio = indicators.get('iv_term_structure_ratio', 1.0)
        iv_time_preference = self.parameters['iv_time_preference']
        
        if iv_time_preference == 'contango':
            # For contango (future IV > near IV), higher ratio is better
            if iv_ratio >= 1.1:
                range_score += 10  # Strong contango is good
            elif iv_ratio < 0.95:
                range_score -= 15  # Backwardation is bad for this preference
        else:  # backwardation
            # For backwardation (future IV < near IV), lower ratio is better
            if iv_ratio <= 0.9:
                range_score += 10  # Strong backwardation is good
            elif iv_ratio > 1.05:
                range_score -= 15  # Contango is bad for this preference
        
        # Ensure score is within valid range
        indicators['range_bound_score'] = max(0, min(100, range_score))
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for double calendar based on market conditions.
        
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
            # Check IV environment
            iv_check_passed = True
            if 'iv_percentile' in indicators:
                iv_percentile = indicators['iv_percentile']
                min_iv = self.parameters['min_iv_percentile']
                max_iv = self.parameters['max_iv_percentile']
                
                if iv_percentile < min_iv or iv_percentile > max_iv:
                    iv_check_passed = False
                    logger.info(f"IV filter failed: current IV percentile {iv_percentile:.2f} outside range [{min_iv}-{max_iv}]")
            
            # Check IV term structure if available
            iv_term_check_passed = True
            if 'iv_term_structure_ratio' in indicators:
                iv_ratio = indicators['iv_term_structure_ratio']
                min_ratio = self.parameters['min_iv_term_structure']
                
                if iv_ratio < min_ratio:
                    iv_term_check_passed = False
                    logger.info(f"IV term structure filter failed: ratio {iv_ratio:.2f} below minimum {min_ratio:.2f}")
            
            # Check range-bound market score
            if 'range_bound_score' in indicators and iv_check_passed and iv_term_check_passed:
                range_score = indicators['range_bound_score']
                threshold = self.parameters['range_bound_score_threshold']
                
                if range_score >= threshold:
                    # Calculate signal strength (0.0 to 1.0)
                    signal_strength = range_score / 100.0
                    
                    # Generate entry signal
                    signals["entry"] = True
                    signals["signal_strength"] = signal_strength
                    
                    logger.info(f"Double calendar entry signal: range-bound score {range_score}, "
                               f"strength {signal_strength:.2f}, IV percentile: {indicators.get('iv_percentile', 'N/A')}")
                else:
                    logger.info(f"Range-bound score {range_score} below threshold {threshold}")
        
        # Check for exits on existing positions
        for position in self.positions:
            if position.get('status') == "open":
                exit_signal = self._check_exit_conditions(position, data, indicators)
                if exit_signal:
                    signals["exit_positions"].append(position['position_id'])
        
        return signals
    
    def _check_exit_conditions(self, position: Dict[str, Any], data: pd.DataFrame, indicators: Dict[str, Any]) -> bool:
        """
        Check exit conditions for an open double calendar position.
        
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
        
        # 1. Expiration approach for near-term options
        if 'near_call_leg' in position and 'expiration' in position['near_call_leg']:
            expiry_date = position['near_call_leg'].get('expiration')
            if isinstance(expiry_date, str):
                expiry_date = datetime.strptime(expiry_date, '%Y-%m-%d').date()
            
            days_to_expiry = (expiry_date - datetime.now().date()).days
            
            # Exit ahead of near expiration
            if days_to_expiry <= self.parameters['near_days_before_expiry_exit']:
                logger.info(f"Exit signal: approaching near expiration ({days_to_expiry} days left)")
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
        
        # 4. Check if price moves outside the expected profit range
        if 'profit_range_low' in position and 'profit_range_high' in position:
            profit_range_low = position['profit_range_low']
            profit_range_high = position['profit_range_high']
            
            # Add buffer for range evaluation
            range_buffer = 0.02  # 2% buffer
            extended_low = profit_range_low * (1 - range_buffer)
            extended_high = profit_range_high * (1 + range_buffer)
            
            # If price moves significantly outside expected range with momentum
            if current_price < extended_low:
                # Check for continuing downward momentum
                if indicators.get('trend', 'neutral') == 'downtrend' and indicators.get('rsi_14', 50) < 35:
                    logger.info(f"Exit signal: price below profit range with downtrend momentum")
                    return True
            elif current_price > extended_high:
                # Check for continuing upward momentum
                if indicators.get('trend', 'neutral') == 'uptrend' and indicators.get('rsi_14', 50) > 65:
                    logger.info(f"Exit signal: price above profit range with uptrend momentum")
                    return True
        
        # 5. Check for significant range expansion/volatility spike
        if 'atr_pct' in indicators and 'near_call_leg' in position:
            entry_time = position.get('entry_time')
            if entry_time:
                # Convert to datetime if it's a string
                if isinstance(entry_time, str):
                    entry_time = datetime.strptime(entry_time, '%Y-%m-%d %H:%M:%S')
                
                days_in_trade = (datetime.now() - entry_time).days
                
                # If we've been in the trade for a while and volatility spikes
                if days_in_trade >= 7:  # At least a week in the trade
                    current_atr_pct = indicators['atr_pct']
                    
                    # Check if current ATR is much higher than expected
                    if 'expected_range_pct' in indicators:
                        expected_daily_range = indicators['expected_range_pct'] / np.sqrt(position['near_dte'])
                        
                        if current_atr_pct > expected_daily_range * 2:  # Volatility doubled
                            logger.info(f"Exit signal: volatility spike detected (ATR: {current_atr_pct:.1f}% vs expected: {expected_daily_range:.1f}%)")
                            return True
        
        # 6. Range-bound market score deterioration
        if 'range_bound_score' in indicators:
            current_score = indicators['range_bound_score']
            threshold = self.parameters['range_bound_score_threshold']
            
            # If the market is no longer range-bound
            if current_score < threshold * 0.7:  # Below 70% of entry threshold
                logger.info(f"Exit signal: range-bound score deteriorated to {current_score} (threshold: {threshold})")
                return True
        
        return False
    
    def _calculate_theoretical_profit(self, position: Dict[str, Any], current_price: float) -> Optional[float]:
        """
        Calculate theoretical profit percentage for a double calendar.
        
        Args:
            position: The position to evaluate
            current_price: Current price of the underlying
        
        Returns:
            Estimated profit percentage or None if calculation not possible
        """
        if ('near_call_leg' not in position or 'far_call_leg' not in position or
            'near_put_leg' not in position or 'far_put_leg' not in position):
            return None
        
        # Extract position details
        near_call_leg = position['near_call_leg']
        far_call_leg = position['far_call_leg']
        near_put_leg = position['near_put_leg']
        far_put_leg = position['far_put_leg']
        net_debit = position.get('net_debit', 0)
        max_risk = position.get('max_risk', 0)
        
        # Get expiration for time value calculation
        near_expiry = near_call_leg.get('expiration')
        far_expiry = far_call_leg.get('expiration')
        
        if isinstance(near_expiry, str):
            near_expiry = datetime.strptime(near_expiry, '%Y-%m-%d').date()
        if isinstance(far_expiry, str):
            far_expiry = datetime.strptime(far_expiry, '%Y-%m-%d').date()
        
        # Calculate days to expiry
        today = datetime.now().date()
        near_dte = max(1, (near_expiry - today).days)
        far_dte = max(1, (far_expiry - today).days)
        original_near_dte = position.get('near_dte', 30)
        original_far_dte = position.get('far_dte', 60)
        
        # Simplified theta decay model - in reality would use option pricing model
        near_time_decay_factor = near_dte / original_near_dte
        far_time_decay_factor = far_dte / original_far_dte
        
        # Call strikes
        call_strike = position.get('call_strike', near_call_leg.get('strike', 0))
        
        # Put strikes
        put_strike = position.get('put_strike', near_put_leg.get('strike', 0))
        
        # Calculate theoretical current values of each leg
        # Simplified approach - in reality would use option pricing model
        
        # Intrinsic values
        near_call_intrinsic = max(0, current_price - call_strike)
        far_call_intrinsic = max(0, current_price - call_strike)
        near_put_intrinsic = max(0, put_strike - current_price)
        far_put_intrinsic = max(0, put_strike - current_price)
        
        # Time values - simplified estimation based on time decay
        near_call_time_value = max(0, (near_call_leg.get('entry_price', 0) - near_call_intrinsic) * near_time_decay_factor)
        far_call_time_value = max(0, (far_call_leg.get('entry_price', 0) - far_call_intrinsic) * far_time_decay_factor)
        near_put_time_value = max(0, (near_put_leg.get('entry_price', 0) - near_put_intrinsic) * near_time_decay_factor)
        far_put_time_value = max(0, (far_put_leg.get('entry_price', 0) - far_put_intrinsic) * far_time_decay_factor)
        
        # Current theoretical values
        near_call_value = near_call_intrinsic + near_call_time_value
        far_call_value = far_call_intrinsic + far_call_time_value
        near_put_value = near_put_intrinsic + near_put_time_value
        far_put_value = far_put_intrinsic + far_put_time_value
        
        # Cost to close position
        # We sold near, bought far - so to close, we buy near and sell far
        current_cost = (near_call_value - far_call_value + near_put_value - far_put_value)
        
        # Calculate P&L
        current_pnl = (net_debit + current_cost) * 100  # Convert to dollars
        
        # Express as percentage of max risk
        if max_risk > 0:
            profit_pct = current_pnl / max_risk
        else:
            profit_pct = 0
        
        return profit_pct
    
    def _execute_signals(self):

        # Verify account has sufficient buying power
        buying_power = self.get_buying_power()
        if buying_power <= 0:
            logger.warning("Trade execution aborted: Insufficient buying power")
            return
        """
        Execute Double Calendar specific trading signals.
        """
        if not self.signals:
            return
        
        # Handle entry signals
        if self.signals.get("entry", False):
            # Construct and open double calendar position
            underlying_price = self.session.current_price
            
            if underlying_price and self.session.option_chain is not None:
                # Use the construct_double_calendar method for position creation
                calendar_position = self.construct_double_calendar(
                    self.session.option_chain, 
                    underlying_price
                )
                
                if calendar_position:
                    # Determine position size based on risk
                    account_value = self.session.account_value
                    max_risk_pct = self.parameters['max_risk_per_trade_pct'] / 100
                    max_risk_amount = account_value * max_risk_pct
                    
                    # Calculate position size based on max risk
                    position_max_risk = calendar_position.get('max_risk', float('inf'))
                    
                    if position_max_risk < float('inf') and position_max_risk > 0:
                        # Number of calendars to trade
                        num_spreads = int(max_risk_amount / position_max_risk)
                        num_spreads = max(1, min(3, num_spreads))  # At least 1, at most 3 spreads
                        calendar_position['quantity'] = num_spreads
                    else:
                        calendar_position['quantity'] = 1  # Default to 1 if risk can't be calculated
                    
                    # Add position
                    self.positions.append(calendar_position)
                    
                    # Log entry details
                    call_strike = calendar_position.get('call_strike', 0)
                    put_strike = calendar_position.get('put_strike', 0)
                    near_dte = calendar_position.get('near_dte', 0)
                    far_dte = calendar_position.get('far_dte', 0)
                    net_debit = calendar_position.get('net_debit', 0)
                    quantity = calendar_position.get('quantity', 1)
                    
                    logger.info(f"Opened double calendar position {calendar_position['position_id']}: "
                              f"call strike {call_strike}, put strike {put_strike}, "
                              f"near/far DTE: {near_dte}/{far_dte}, "
                              f"debit: ${net_debit * 100 * quantity:.2f}, quantity: {quantity}")
                else:
                    logger.warning("Failed to construct valid double calendar")
        
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
                        max_risk = position.get('max_risk', 0)
                        quantity = position.get('quantity', 1)
                        pnl = profit_pct * max_risk * quantity
                        position['pnl'] = pnl
                        
                        logger.info(f"Closed double calendar position {position_id}, P&L: ${pnl:.2f}")
                    else:
                        logger.info(f"Closed double calendar position {position_id}, P&L calculation not available")
    
    def run_strategy(self):
        """
        Run the double calendar strategy cycle.
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
            logger.warning("Insufficient market data for double calendar analysis")
    
    def register_events(self):
        """Register for events relevant to Double Calendar strategy."""
        # Register for common event types from the parent class
        super().register_events()
        
        # Add double calendar specific event subscriptions
        EventBus.subscribe(EventType.EARNINGS_ANNOUNCEMENT, self.on_event)
        EventBus.subscribe(EventType.IMPLIED_VOLATILITY_CHANGE, self.on_event)
        EventBus.subscribe(EventType.MARKET_REGIME_CHANGE, self.on_event)
    
    def on_event(self, event: Event):
        """
        Process incoming events for the Double Calendar strategy.
        
        Args:
            event: The event to process
        """
        # Let parent class handle common events first
        super().on_event(event)
        
        # Add additional Double Calendar specific event handling
        if event.type == EventType.EARNINGS_ANNOUNCEMENT and self.session.symbol in event.data.get('symbols', []):
            # Earnings announcements can significantly impact double calendar positions
            days_to_earnings = event.data.get('days_to_announcement', 0)
            
            if days_to_earnings <= 5:  # Within 5 days of earnings
                logger.info(f"Earnings announcement approaching for {self.session.symbol}, evaluating double calendar positions")
                
                # Double calendars can be tricky around earnings - sometimes they benefit from IV expansion
                # But if earnings are between the two expiration dates, it could be problematic
                for position in self.positions:
                    if position.get('status') == "open":
                        # Check if earnings is between our two expiration dates
                        near_expiry = position.get('near_call_leg', {}).get('expiration')
                        far_expiry = position.get('far_call_leg', {}).get('expiration')
                        
                        if isinstance(near_expiry, str):
                            near_expiry = datetime.strptime(near_expiry, '%Y-%m-%d').date()
                        if isinstance(far_expiry, str):
                            far_expiry = datetime.strptime(far_expiry, '%Y-%m-%d').date()
                        
                        earnings_date = datetime.now().date() + timedelta(days=days_to_earnings)
                        
                        # If earnings is between our expirations, definitely close the position
                        if near_expiry <= earnings_date <= far_expiry:
                            logger.info(f"Closing double calendar position {position.get('position_id')} - earnings between expirations")
                            self.signals.setdefault("exit_positions", []).append(position.get('position_id'))
                        # Otherwise close if position is already profitable
                        else:
                            current_price = self.session.current_price
                            profit_pct = self._calculate_theoretical_profit(position, current_price)
                            
                            if profit_pct and profit_pct > 0.15:  # 15% of max profit
                                logger.info(f"Closing profitable double calendar {position.get('position_id')} ahead of earnings")
                                self.signals.setdefault("exit_positions", []).append(position.get('position_id'))
                
                # Execute the exits
                self._execute_signals()
        
        elif event.type == EventType.IMPLIED_VOLATILITY_CHANGE:
            symbol = event.data.get('symbol')
            magnitude = event.data.get('magnitude', 0)
            direction = event.data.get('direction', '')
            
            if symbol == self.session.symbol and abs(magnitude) > 10:  # Significant IV change
                logger.info(f"Significant IV {direction} of {magnitude}% detected")
                
                # Double calendar response depends on IV expectations
                if direction == 'spike' and magnitude > 20:  # Major IV spike
                    logger.info(f"Significant IV spike may benefit double calendar positions")
                    
                    # For positions that are already profitable, consider locking in gains
                    for position in self.positions:
                        if position.get('status') == "open":
                            current_price = self.session.current_price
                            profit_pct = self._calculate_theoretical_profit(position, current_price)
                            
                            if profit_pct and profit_pct > 0.25:  # 25% of max profit
                                logger.info(f"Closing profitable double calendar {position.get('position_id')} after IV spike")
                                self.signals.setdefault("exit_positions", []).append(position.get('position_id'))
                
                elif direction == 'crash' and magnitude > 30:  # Major IV crash
                    logger.info(f"Significant IV crash may impact double calendar positions")
                    
                    # Significant IV collapse could hurt double calendars
                    for position in self.positions:
                        if position.get('status') == "open":
                            logger.info(f"Evaluating double calendar {position.get('position_id')} after IV crash")
                            
                            # If we detect a loss, better to exit
                            current_price = self.session.current_price
                            profit_pct = self._calculate_theoretical_profit(position, current_price)
                            
                            if profit_pct and profit_pct < -0.15:  # 15% loss
                                logger.info(f"Closing unprofitable double calendar {position.get('position_id')} after IV crash")
                                self.signals.setdefault("exit_positions", []).append(position.get('position_id'))
                
                # Execute the exits if needed
                if "exit_positions" in self.signals and self.signals["exit_positions"]:
                    self._execute_signals()
        
        elif event.type == EventType.MARKET_REGIME_CHANGE:
            new_regime = event.data.get('new_regime', '')
            
            # Double calendar is sensitive to market regime changes involving volatility
            if new_regime in ['bearish-high-volatility', 'bullish-high-volatility', 'crash']:
                logger.info(f"Market regime changed to {new_regime}, evaluating double calendar positions")
                
                # In high volatility regimes, consider exiting if position is profitable
                for position in self.positions:
                    if position.get('status') == "open":
                        current_price = self.session.current_price
                        profit_pct = self._calculate_theoretical_profit(position, current_price)
                        
                        # If already profitable, lock in gains in unstable regime
                        if profit_pct and profit_pct > 0.2:  # 20% of max profit
                            logger.info(f"Closing profitable double calendar {position.get('position_id')} due to regime change to {new_regime}")
                            self.signals.setdefault("exit_positions", []).append(position.get('position_id'))
                        # If in a truly dangerous regime, exit even if not profitable
                        elif new_regime == 'crash':
                            logger.info(f"Closing double calendar {position.get('position_id')} due to market crash regime")
                            self.signals.setdefault("exit_positions", []).append(position.get('position_id'))
                
                # Execute the exits
                if "exit_positions" in self.signals and self.signals["exit_positions"]:
                    self._execute_signals()
