#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagonal Spread Strategy

A professional-grade diagonal spread implementation that leverages the modular,
event-driven architecture.
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
    name="DiagonalSpreadStrategy",
    market_type="options",
    description="A strategy that involves options of the same type but with different strikes and expiration dates, combining features of both vertical and calendar spreads",
    timeframes=["1d", "1w"],
    parameters={
        "option_type": {"description": "Option type (call or put)", "type": "string"},
        "is_long": {"description": "Whether to use a long or short diagonal spread", "type": "boolean"},
        "front_month_dte_target": {"description": "Target days to expiration for front month", "type": "integer"},
        "back_month_dte_target": {"description": "Target days to expiration for back month", "type": "integer"}
    }
)
class DiagonalSpreadStrategy(TimeSpreadEngine, AccountAwareMixin):
    """
    Diagonal Spread Strategy
    
    This strategy involves options of the same type (calls or puts) but with different 
    strikes and different expiration dates. It combines features of both vertical 
    spreads and calendar spreads.
    
    Features:
    - Adapts the legacy Diagonal Spread implementation to the new architecture
    - Uses event-driven signal generation and position management
    - Leverages the TimeSpreadEngine for core multi-expiry spread mechanics
    - Implements custom filtering and price trend/volatility analysis
    """
    
    def __init__(self, session: OptionsSession, data_pipeline: DataPipeline, 
                 parameters: Dict[str, Any] = None):
        """
        Initialize the Diagonal Spread strategy.
        
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
            'strategy_name': 'Diagonal Spread',
            'strategy_id': 'diagonal_spread',
            
            # Diagonal Spread specific parameters
            'option_type': 'call',                # 'call' or 'put'
            'is_long': True,                      # Long diagonal by default
            'front_month_delta_target': 0.50,     # Target delta for front month
            'back_month_delta_target': 0.50,      # Target delta for back month
            'front_month_dte_target': 30,         # Target DTE for front month
            'back_month_dte_target': 60,          # Target DTE for back month
            'min_dte_difference': 20,             # Minimum DTE difference between months
            'strike_differential': 2,             # Number of strikes difference
            
            # Market condition preferences
            'min_directional_score': 65,          # Minimum score for directional bias (0-100)
            'min_iv_differential': 2.0,           # Minimum IV differential between months (percentage points)
            'max_iv_ratio': 1.2,                  # Maximum ratio of back month IV to front month IV
            
            # Risk parameters
            'max_risk_per_trade_pct': 2.0,        # Max risk as % of account
            'target_profit_pct': 50,              # Target profit as % of max profit
            'stop_loss_pct': 65,                  # Stop loss as % of max profit
            'max_positions': 2,                   # Maximum concurrent positions
            
            # Exit parameters
            'front_month_exit_dte': 7,            # Exit when front month reaches this DTE
        }
        
        # Update with default parameters
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Set spread type based on parameters
        if self.parameters['option_type'] == 'call':
            self.spread_type = TimeSpreadType.CALL_DIAGONAL
        else:
            self.spread_type = TimeSpreadType.PUT_DIAGONAL
        
        # Register for market events
        self.register_events()
        
        logger.info(f"Initialized Diagonal Spread Strategy for {session.symbol}")
    
    def construct_diagonal_spread(self, option_chain: pd.DataFrame, underlying_price: float) -> Optional[Dict[str, Any]]:
        """
        Construct a diagonal spread from the option chain.
        
        Args:
            option_chain: Filtered option chain data
            underlying_price: Current price of the underlying asset
            
        Returns:
            Diagonal spread position if successful, None otherwise
        """
        if option_chain is None or option_chain.empty:
            logger.warning("Empty option chain provided")
            return None
        
        # Apply filters for liquidity, open interest, etc.
        filtered_chain = self.filter_option_chains(option_chain)
        
        if filtered_chain is None or filtered_chain.empty:
            logger.warning("No suitable options found after filtering")
            return None
        
        # Determine option type based on parameters
        option_type = self.parameters['option_type'].lower()
        is_long = self.parameters['is_long']
        
        # Extract parameters for selection
        front_month_dte_target = self.parameters['front_month_dte_target']
        back_month_dte_target = self.parameters['back_month_dte_target']
        min_dte_difference = self.parameters['min_dte_difference']
        front_month_delta_target = self.parameters['front_month_delta_target']
        back_month_delta_target = self.parameters['back_month_delta_target']
        
        # Filter by option type
        chain_by_type = filtered_chain[filtered_chain['option_type'] == option_type]
        
        if chain_by_type.empty:
            logger.warning(f"No {option_type} options found in chain")
            return None
        
        # Get unique expirations
        expirations = sorted(chain_by_type['expiration_date'].unique())
        
        if len(expirations) < 2:
            logger.warning("Need at least two different expirations for diagonal spread")
            return None
        
        # Find suitable front and back month expirations
        front_expiry = None
        back_expiry = None
        
        # Calculate days to expiry for each expiration
        for expiry in expirations:
            if isinstance(expiry, str):
                expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
            else:
                expiry_date = expiry
            
            dte = (expiry_date - datetime.now().date()).days
            
            # Front month selection
            if front_expiry is None or abs(dte - front_month_dte_target) < abs((front_expiry - datetime.now().date()).days - front_month_dte_target):
                front_expiry = expiry_date
            
            # Back month selection
            if dte >= front_month_dte_target + min_dte_difference:
                if back_expiry is None or abs(dte - back_month_dte_target) < abs((back_expiry - datetime.now().date()).days - back_month_dte_target):
                    back_expiry = expiry_date
        
        if front_expiry is None or back_expiry is None:
            logger.warning("Could not find suitable expiration pair for diagonal spread")
            return None
        
        # Convert to string format for filtering
        front_expiry_str = front_expiry.strftime('%Y-%m-%d')
        back_expiry_str = back_expiry.strftime('%Y-%m-%d')
        
        # Get options for each expiration
        front_options = chain_by_type[chain_by_type['expiration_date'] == front_expiry_str]
        back_options = chain_by_type[chain_by_type['expiration_date'] == back_expiry_str]
        
        if front_options.empty or back_options.empty:
            logger.warning("Missing options for one of the selected expirations")
            return None
        
        # Find strikes based on delta targets
        front_strike = None
        back_strike = None
        
        # For front month
        for _, option in front_options.iterrows():
            if 'delta' in option:
                delta = option['delta']
                if option_type == 'call':
                    # For calls, delta is positive
                    delta_diff = abs(delta - front_month_delta_target)
                else:
                    # For puts, delta is negative
                    delta_diff = abs(delta - (-front_month_delta_target))
                
                if front_strike is None or delta_diff < front_delta_diff:
                    front_strike = option['strike']
                    front_delta_diff = delta_diff
        
        # For back month
        for _, option in back_options.iterrows():
            if 'delta' in option:
                delta = option['delta']
                if option_type == 'call':
                    delta_diff = abs(delta - back_month_delta_target)
                else:
                    delta_diff = abs(delta - (-back_month_delta_target))
                
                if back_strike is None or delta_diff < back_delta_diff:
                    back_strike = option['strike']
                    back_delta_diff = delta_diff
        
        if front_strike is None or back_strike is None:
            logger.warning("Could not find suitable strikes based on delta targets")
            return None
        
        # Adjust for strike differential if needed
        strike_diff = self.parameters['strike_differential']
        available_back_strikes = sorted(back_options['strike'].unique())
        
        if option_type == 'call':
            # For call diagonals, back month strike is typically higher for long diagonals
            target_back_strike = None
            for strike in available_back_strikes:
                if is_long and strike > front_strike:
                    steps = round((strike - front_strike) / (available_back_strikes[1] - available_back_strikes[0]))
                    if steps == strike_diff:
                        target_back_strike = strike
                        break
                elif not is_long and strike < front_strike:
                    steps = round((front_strike - strike) / (available_back_strikes[1] - available_back_strikes[0]))
                    if steps == strike_diff:
                        target_back_strike = strike
                        break
            
            if target_back_strike:
                back_strike = target_back_strike
        else:
            # For put diagonals, back month strike is typically lower for long diagonals
            target_back_strike = None
            for strike in available_back_strikes:
                if is_long and strike < front_strike:
                    steps = round((front_strike - strike) / (available_back_strikes[1] - available_back_strikes[0]))
                    if steps == strike_diff:
                        target_back_strike = strike
                        break
                elif not is_long and strike > front_strike:
                    steps = round((strike - front_strike) / (available_back_strikes[1] - available_back_strikes[0]))
                    if steps == strike_diff:
                        target_back_strike = strike
                        break
            
            if target_back_strike:
                back_strike = target_back_strike
        
        # Get specific option contracts
        front_contract = front_options[front_options['strike'] == front_strike].iloc[0]
        back_contract = back_options[back_options['strike'] == back_strike].iloc[0]
        
        # Create leg objects
        front_leg = {
            'option_type': option_type,
            'strike': front_strike,
            'expiration': front_expiry_str,
            'action': 'sell' if is_long else 'buy',  # For long diagonal, we sell front month
            'quantity': 1,
            'entry_price': front_contract['bid'] if is_long else front_contract['ask'],
            'delta': front_contract.get('delta', front_month_delta_target if option_type == 'call' else -front_month_delta_target),
            'gamma': front_contract.get('gamma', 0),
            'theta': front_contract.get('theta', 0),
            'vega': front_contract.get('vega', 0)
        }
        
        back_leg = {
            'option_type': option_type,
            'strike': back_strike,
            'expiration': back_expiry_str,
            'action': 'buy' if is_long else 'sell',  # For long diagonal, we buy back month
            'quantity': 1,
            'entry_price': back_contract['ask'] if is_long else back_contract['bid'],
            'delta': back_contract.get('delta', back_month_delta_target if option_type == 'call' else -back_month_delta_target),
            'gamma': back_contract.get('gamma', 0),
            'theta': back_contract.get('theta', 0),
            'vega': back_contract.get('vega', 0)
        }
        
        # Calculate net debit/credit
        if is_long:
            # Long diagonal: Buy back month, sell front month
            net_debit = back_leg['entry_price'] - front_leg['entry_price']
        else:
            # Short diagonal: Sell back month, buy front month
            net_debit = front_leg['entry_price'] - back_leg['entry_price']
        
        # Create position object
        position = {
            'position_id': str(uuid.uuid4()),
            'front_leg': front_leg,
            'back_leg': back_leg,
            'entry_time': datetime.now(),
            'quantity': 1,  # Will be adjusted later based on risk parameters
            'spread_type': self.spread_type,
            'is_long': is_long,
            'net_debit': net_debit,
            'status': 'open',
            'pnl': 0.0
        }
        
        # Calculate Greeks for the full position
        position['net_delta'] = back_leg['delta'] - front_leg['delta'] if is_long else front_leg['delta'] - back_leg['delta']
        position['net_gamma'] = back_leg.get('gamma', 0) - front_leg.get('gamma', 0) if is_long else front_leg.get('gamma', 0) - back_leg.get('gamma', 0)
        position['net_theta'] = back_leg.get('theta', 0) - front_leg.get('theta', 0) if is_long else front_leg.get('theta', 0) - back_leg.get('theta', 0)
        position['net_vega'] = back_leg.get('vega', 0) - front_leg.get('vega', 0) if is_long else front_leg.get('vega', 0) - back_leg.get('vega', 0)
        
        logger.info(f"Constructed diagonal spread: {option_type} {front_strike}/{back_strike} "
                  f"{front_expiry_str}/{back_expiry_str}, net debit: ${net_debit:.2f}")
        
        return position
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicators for diagonal spread signals.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        if data.empty or len(data) < 30:
            return indicators
        
        # Calculate basic price indicators
        if 'close' in data.columns:
            # Moving averages
            indicators['sma_20'] = data['close'].rolling(20).mean().iloc[-1]
            indicators['sma_50'] = data['close'].rolling(50).mean().iloc[-1]
            indicators['sma_200'] = data['close'].rolling(200).mean().iloc[-1]
            
            # Momentum/trend indicators
            indicators['rsi_14'] = self.calculate_rsi(data['close'], period=14).iloc[-1]
            indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = self.calculate_macd(data['close'])
            
            # Volatility measures
            indicators['bollinger_upper'], indicators['bollinger_middle'], indicators['bollinger_lower'] = \
                self.calculate_bollinger_bands(data['close'], window=20)
            indicators['atr'] = self.calculate_atr(data, period=14)
            indicators['atr_pct'] = (indicators['atr'] / data['close'].iloc[-1]) * 100 if indicators['atr'] else None
            
            # Price vs moving averages
            current_price = data['close'].iloc[-1]
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
            
            # Calculate rate of change
            roc_5 = ((data['close'].iloc[-1] / data['close'].iloc[-6]) - 1) * 100  # 5-day rate of change
            roc_10 = ((data['close'].iloc[-1] / data['close'].iloc[-11]) - 1) * 100  # 10-day rate of change
            indicators['roc_5'] = roc_5
            indicators['roc_10'] = roc_10
        
        # Calculate volume indicators
        if 'volume' in data.columns:
            indicators['volume_sma_20'] = data['volume'].rolling(20).mean().iloc[-1]
            indicators['volume_ratio'] = data['volume'].iloc[-1] / indicators['volume_sma_20']
        
        # Calculate IV skew if available
        # This would come from the option chain data in a real implementation
        if self.session.iv_skew is not None:
            indicators['iv_skew'] = self.session.iv_skew
        
        # Calculate IV term structure if available
        # In a diagonal, we want to see a steeper IV term structure
        if self.session.iv_term_structure is not None:
            indicators['iv_term_structure'] = self.session.iv_term_structure
        
        # Calculate directional score (0-100) for bias indication
        directional_score = 50  # Neutral starting point
        
        # Adjust based on trend indicators
        option_type = self.parameters['option_type']
        is_long = self.parameters['is_long']
        
        # Determine what indicates a "good" setup based on option type and position
        if (option_type == 'call' and is_long) or (option_type == 'put' and not is_long):
            # Want bullish bias
            if indicators['trend'] == 'uptrend':
                directional_score += 15
            elif indicators['trend'] == 'downtrend':
                directional_score -= 15
            
            if indicators['rsi_14'] > 60:
                directional_score += 10
            elif indicators['rsi_14'] < 40:
                directional_score -= 10
            
            if indicators['macd_hist'] > 0:
                directional_score += 10
            else:
                directional_score -= 5
            
            if indicators.get('roc_5', 0) > 1 and indicators.get('roc_10', 0) > 2:
                directional_score += 15
        else:  # Put long or Call short - want bearish bias
            # Want bearish bias
            if indicators['trend'] == 'downtrend':
                directional_score += 15
            elif indicators['trend'] == 'uptrend':
                directional_score -= 15
            
            if indicators['rsi_14'] < 40:
                directional_score += 10
            elif indicators['rsi_14'] > 60:
                directional_score -= 10
            
            if indicators['macd_hist'] < 0:
                directional_score += 10
            else:
                directional_score -= 5
            
            if indicators.get('roc_5', 0) < -1 and indicators.get('roc_10', 0) < -2:
                directional_score += 15
        
        # Adjust based on volume confirmation
        if 'volume_ratio' in indicators:
            if indicators['volume_ratio'] > 1.3 and indicators['trend'] != 'neutral':
                # Higher volume in the direction of the trend is positive
                directional_score += 5
        
        # Ensure score is within valid range
        indicators['directional_score'] = max(0, min(100, directional_score))
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for diagonal spreads based on market conditions.
        
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
            # Check IV environment - Diagonals benefit from certain IV conditions
            iv_check_passed = True
            if self.session.current_iv is not None and self.session.iv_term_structure is not None:
                # For diagonals, we want to see term structure differences
                min_iv_differential = self.parameters['min_iv_differential']
                max_iv_ratio = self.parameters['max_iv_ratio']
                
                # Example term structure: 30 day IV vs 60 day IV
                front_iv = self.session.iv_term_structure.get('30day', self.session.current_iv)
                back_iv = self.session.iv_term_structure.get('60day', self.session.current_iv * 1.05)  # Default slight increase
                
                iv_differential = back_iv - front_iv
                iv_ratio = back_iv / front_iv if front_iv > 0 else 1.0
                
                if iv_differential < min_iv_differential or iv_ratio > max_iv_ratio:
                    iv_check_passed = False
                    logger.info(f"IV term structure not ideal: differential {iv_differential:.2f}%, ratio {iv_ratio:.2f}")
            
            # Check directional bias score
            if 'directional_score' in indicators and iv_check_passed:
                directional_score = indicators['directional_score']
                min_score = self.parameters['min_directional_score']
                
                if directional_score >= min_score:
                    # Directional bias is aligned with our strategy
                    option_type = self.parameters['option_type']
                    desired_trend = ""
                    
                    if (option_type == 'call' and self.parameters['is_long']) or \
                       (option_type == 'put' and not self.parameters['is_long']):
                        desired_trend = "bullish"
                    else:
                        desired_trend = "bearish"
                    
                    # Technical agreement check
                    trend_agreement = False
                    if desired_trend == "bullish" and indicators.get('trend') == 'uptrend':
                        trend_agreement = True
                    elif desired_trend == "bearish" and indicators.get('trend') == 'downtrend':
                        trend_agreement = True
                    
                    if trend_agreement:
                        # Generate entry signal with confidence based on score
                        signals["entry"] = True
                        signals["signal_strength"] = directional_score / 100.0
                        
                        logger.info(f"Diagonal spread entry signal: {desired_trend} bias with score {directional_score}, "
                                   f"strength {signals['signal_strength']:.2f}")
                    else:
                        logger.info(f"Diagonal spread entry not optimal: {desired_trend} bias needed, "
                                   f"but trend is {indicators.get('trend', 'unknown')}")
                else:
                    logger.info(f"Directional score {directional_score} below threshold {min_score}")
        
        # Check for exits on existing positions
        for position in self.positions:
            if position.get('status', 'closed') == "open":
                exit_signal = self._check_exit_conditions(position, data, indicators)
                if exit_signal:
                    signals["exit_positions"].append(position['position_id'])
        
        return signals
    
    def _check_exit_conditions(self, position: Dict[str, Any], data: pd.DataFrame, indicators: Dict[str, Any]) -> bool:
        """
        Check exit conditions for an open diagonal spread position.
        
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
        
        # 1. Front month expiration approach
        if 'front_leg' in position and 'expiration' in position['front_leg']:
            expiry_date = position['front_leg'].get('expiration')
            if isinstance(expiry_date, str):
                expiry_date = datetime.strptime(expiry_date, '%Y-%m-%d').date()
            
            days_to_expiry = (expiry_date - datetime.now().date()).days
            
            # Exit when front month gets too close to expiration to avoid gamma risk
            if days_to_expiry <= self.parameters['front_month_exit_dte']:
                logger.info(f"Exit signal: front month approaching expiration ({days_to_expiry} days left)")
                return True
        
        # 2. Profit target check
        # In a real implementation, would calculate based on actual option prices
        # Here we'll use a simplified approach based on the underlying price
        current_price = data['close'].iloc[-1]
        
        # Calculate theoretical profit percentage (simplified)
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
        
        # 4. Directional bias change check
        option_type = position.get('front_leg', {}).get('option_type')
        is_long = position.get('is_long', True)
        desired_trend = ""
        
        if (option_type == 'call' and is_long) or (option_type == 'put' and not is_long):
            desired_trend = "bullish"
        else:
            desired_trend = "bearish"
        
        if ('trend' in indicators and 
            ((desired_trend == "bullish" and indicators['trend'] == 'downtrend') or 
             (desired_trend == "bearish" and indicators['trend'] == 'uptrend'))):
            
            # Trend has reversed against our position
            if 'directional_score' in indicators and indicators['directional_score'] < 40:
                logger.info(f"Exit signal: trend reversed to {indicators['trend']} with low directional score")
                return True
        
        # 5. IV environment deterioration
        if self.session.iv_term_structure is not None:
            front_iv = self.session.iv_term_structure.get('30day', self.session.current_iv)
            back_iv = self.session.iv_term_structure.get('60day', self.session.current_iv * 1.05)
            iv_differential = back_iv - front_iv
            
            # If IV term structure flattens significantly, consider exit
            if iv_differential < 0.5:  # Very flat term structure
                logger.info(f"Exit signal: IV term structure flattened (differential: {iv_differential:.2f}%)")
                return True
        
        return False
    
    def _calculate_theoretical_profit(self, position: Dict[str, Any], current_price: float) -> Optional[float]:
        """
        Calculate theoretical profit percentage for a diagonal spread.
        
        Args:
            position: The position to evaluate
            current_price: Current price of the underlying
        
        Returns:
            Estimated profit percentage or None if calculation not possible
        """
        if 'front_leg' not in position or 'back_leg' not in position:
            return None
        
        # Extract position details
        front_leg = position['front_leg']
        back_leg = position['back_leg']
        net_debit = position.get('net_debit', 0)
        
        if net_debit == 0:  # Avoid division by zero
            return None
        
        # Calculate theoretical current values of each leg
        front_strike = front_leg.get('strike')
        back_strike = back_leg.get('strike')
        option_type = front_leg.get('option_type')
        is_long = position.get('is_long', True)
        
        # Calculate days to expiration for time value
        front_expiry = front_leg.get('expiration')
        back_expiry = back_leg.get('expiration')
        
        if isinstance(front_expiry, str):
            front_expiry = datetime.strptime(front_expiry, '%Y-%m-%d').date()
        if isinstance(back_expiry, str):
            back_expiry = datetime.strptime(back_expiry, '%Y-%m-%d').date()
        
        front_dte = max(1, (front_expiry - datetime.now().date()).days)
        back_dte = max(1, (back_expiry - datetime.now().date()).days)
        
        # Very simplified pricing model - in a real implementation would use Black-Scholes
        # This is just a rough approximation based on intrinsic value + time value
        
        # Front month
        front_intrinsic = max(0, current_price - front_strike) if option_type == 'call' else max(0, front_strike - current_price)
        front_time_value = (front_leg.get('entry_price', 0) - front_intrinsic) * (front_dte / self.parameters['front_month_dte_target'])
        front_value = front_intrinsic + front_time_value
        
        # Back month
        back_intrinsic = max(0, current_price - back_strike) if option_type == 'call' else max(0, back_strike - current_price)
        back_time_value = (back_leg.get('entry_price', 0) - back_intrinsic) * (back_dte / self.parameters['back_month_dte_target'])
        back_value = back_intrinsic + back_time_value
        
        # Calculate net value based on position type
        if is_long:  # Long diagonal: bought back month, sold front month
            current_value = back_value - front_value
        else:  # Short diagonal: sold back month, bought front month
            current_value = front_value - back_value
        
        # Calculate profit/loss percentage
        profit_loss = current_value - net_debit
        profit_pct = profit_loss / abs(net_debit)
        
        return profit_pct
    
    def _execute_signals(self):

        # Verify account has sufficient buying power
        buying_power = self.get_buying_power()
        if buying_power <= 0:
            logger.warning("Trade execution aborted: Insufficient buying power")
            return
        """
        Execute Diagonal Spread specific trading signals.
        """
        if not self.signals:
            return
        
        # Handle entry signals
        if self.signals.get("entry", False):
            # Construct and open diagonal spread position
            underlying_price = self.session.current_price
            
            if underlying_price and self.session.option_chain is not None:
                # Use the construct_diagonal_spread method for position creation
                diagonal_position = self.construct_diagonal_spread(
                    self.session.option_chain, 
                    underlying_price
                )
                
                if diagonal_position:
                    # Determine position size based on risk
                    account_value = self.session.account_value
                    max_risk_pct = self.parameters['max_risk_per_trade_pct'] / 100
                    max_risk_amount = account_value * max_risk_pct
                    
                    # Calculate number of spreads based on net debit and max risk
                    net_debit = diagonal_position.get('net_debit', 0)
                    if net_debit > 0:
                        # For options, multiply by 100 (contract multiplier)
                        cost_per_spread = net_debit * 100
                        num_spreads = int(max_risk_amount / cost_per_spread)
                        num_spreads = max(1, min(5, num_spreads))  # At least 1, at most 5 spreads
                        diagonal_position['quantity'] = num_spreads
                    
                    # Add position
                    self.positions.append(diagonal_position)
                    logger.info(f"Opened diagonal spread position {diagonal_position['position_id']} "
                               f"with {num_spreads} spread(s)")
                else:
                    logger.warning("Failed to construct valid diagonal spread")
        
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
                        net_debit = position.get('net_debit', 0)
                        pnl = profit_pct * net_debit * 100 * position.get('quantity', 1)
                        position['pnl'] = pnl
                        
                        logger.info(f"Closed diagonal spread position {position_id}, P&L: ${pnl:.2f}")
                    else:
                        logger.info(f"Closed diagonal spread position {position_id}, P&L calculation not available")
    
    def run_strategy(self):
        """
        Run the diagonal spread strategy cycle.
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
            logger.warning("Insufficient market data for diagonal spread analysis")
    
    def register_events(self):
        """Register for events relevant to Diagonal Spreads."""
        # Register for common event types from the parent class
        super().register_events()
        
        # Add diagonal spread specific event subscriptions
        EventBus.subscribe(EventType.EARNINGS_ANNOUNCEMENT, self.on_event)
        EventBus.subscribe(EventType.IMPLIED_VOLATILITY_CHANGE, self.on_event)
        EventBus.subscribe(EventType.MARKET_REGIME_CHANGE, self.on_event)
    
    def on_event(self, event: Event):
        """
        Process incoming events for the Diagonal Spread strategy.
        
        Args:
            event: The event to process
        """
        # Let parent class handle common events first
        super().on_event(event)
        
        # Add additional Diagonal Spread specific event handling
        if event.type == EventType.EARNINGS_ANNOUNCEMENT and self.session.symbol in event.data.get('symbols', []):
            # Earnings announcements can significantly impact diagonal spreads
            days_to_earnings = event.data.get('days_to_announcement', 0)
            
            if days_to_earnings <= 5:  # Within 5 days of earnings
                logger.info(f"Earnings announcement approaching for {self.session.symbol}, evaluate diagonal positions")
                
                # Decide what to do based on position type
                for position in self.positions:
                    if position.get('status') == "open":
                        option_type = position.get('front_leg', {}).get('option_type')
                        is_long = position.get('is_long', True)
                        
                        # Long diagonals with back month calls can benefit from IV expansion pre-earnings
                        # Short diagonals or put diagonals may be at higher risk
                        if not (option_type == 'call' and is_long):
                            logger.info(f"Closing diagonal position {position.get('position_id')} ahead of earnings")
                            self.signals.setdefault("exit_positions", []).append(position.get('position_id'))
                
                # Execute the exits if needed
                if "exit_positions" in self.signals and self.signals["exit_positions"]:
                    self._execute_signals()
        
        elif event.type == EventType.IMPLIED_VOLATILITY_CHANGE:
            symbol = event.data.get('symbol')
            magnitude = event.data.get('magnitude', 0)
            direction = event.data.get('direction', '')
            
            if symbol == self.session.symbol and abs(magnitude) > 15:  # Significant IV change
                logger.info(f"Significant IV {direction} of {magnitude}% detected")
                
                # Effect depends on position type
                for position in self.positions:
                    if position.get('status') == "open":
                        is_long = position.get('is_long', True)
                        
                        # For long diagonals, IV drop can be harmful
                        # For short diagonals, IV drop can be beneficial
                        if (is_long and direction == 'drop' and magnitude > 20) or \
                           (not is_long and direction == 'spike' and magnitude > 20):
                            logger.info(f"Unfavorable IV change, closing diagonal position {position.get('position_id')}")
                            self.signals.setdefault("exit_positions", []).append(position.get('position_id'))
                
                # Execute the exits if needed
                if "exit_positions" in self.signals and self.signals["exit_positions"]:
                    self._execute_signals()
        
        elif event.type == EventType.MARKET_REGIME_CHANGE:
            new_regime = event.data.get('new_regime', '')
            
            # Regime changes can affect diagonal spreads differently based on type
            if new_regime in ['high_volatility', 'trending']:
                logger.info(f"Market regime changed to {new_regime}, evaluating diagonal positions")
                
                # Check each position to see if it aligns with the new regime
                for position in self.positions:
                    if position.get('status') == "open":
                        option_type = position.get('front_leg', {}).get('option_type')
                        is_long = position.get('is_long', True)
                        
                        # Determine if position is still suitable in the new regime
                        regime_compatible = True
                        
                        if new_regime == 'high_volatility':
                            # Long diagonals generally benefit from vol expansion
                            # Short diagonals are at risk in high vol environments
                            if not is_long:
                                regime_compatible = False
                        
                        elif new_regime == 'trending':
                            # In trending markets, diagonal directionality becomes more important
                            # If trend direction doesn't match our bias, close position
                            trend_direction = event.data.get('trend_direction', '')
                            
                            if ((option_type == 'call' and is_long) or (option_type == 'put' and not is_long)):
                                # Bullish position - needs uptrend
                                if trend_direction != 'up':
                                    regime_compatible = False
                            else:
                                # Bearish position - needs downtrend
                                if trend_direction != 'down':
                                    regime_compatible = False
                        
                        if not regime_compatible:
                            logger.info(f"Position not compatible with new regime, closing diagonal {position.get('position_id')}")
                            self.signals.setdefault("exit_positions", []).append(position.get('position_id'))
                
                # Execute the exits if needed
                if "exit_positions" in self.signals and self.signals["exit_positions"]:
                    self._execute_signals()
