#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ratio Spread Strategy

A professional-grade ratio spread implementation that leverages the modular,
event-driven architecture. This strategy involves trading options with unbalanced
quantities of long and short positions.
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
from trading_bot.strategies_new.options.base.advanced_spread_engine import AdvancedSpreadEngine, AdvancedSpreadType
from trading_bot.strategies_new.factory.registry import register_strategy
from trading_bot.strategies_new.validation.account_aware_mixin import AccountAwareMixin

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="RatioSpreadStrategy",
    market_type="options",
    description="A strategy that involves buying and selling options of the same type and expiration but at different strikes and with unbalanced quantities, typically to generate income with defined risk",
    timeframes=["1d", "1w"],
    parameters={
        "option_type": {"description": "Option type (call or put)", "type": "string"},
        "ratio": {"description": "Ratio of short to long options (e.g., 2 for 1:2)", "type": "integer"},
        "is_long_first": {"description": "Whether to buy the first leg and sell the second", "type": "boolean"},
        "target_days_to_expiry": {"description": "Ideal DTE for ratio spread entry", "type": "integer"}
    }
)
class RatioSpreadStrategy(AdvancedSpreadEngine, AccountAwareMixin):
    """
    Ratio Spread Strategy
    
    This strategy involves buying and selling options of the same type (calls or puts)
    and same expiration, but at different strikes and with unbalanced quantities.
    Typically it involves buying one option and selling multiple options at a further 
    OTM strike.
    
    Features:
    - Adapts the legacy Ratio Spread implementation to the new architecture
    - Uses event-driven signal generation and position management
    - Leverages the AdvancedSpreadEngine for core advanced spread mechanics
    - Implements custom filtering and directional/volatility analysis
    """
    
    def __init__(self, session: OptionsSession, data_pipeline: DataPipeline, 
                 parameters: Dict[str, Any] = None):
        """
        Initialize the Ratio Spread strategy.
        
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
            'strategy_name': 'Ratio Spread',
            'strategy_id': 'ratio_spread',
            
            # Ratio Spread specific parameters
            'option_type': 'call',                # 'call' or 'put'
            'ratio': 2,                           # Typical ratio is 1:2 (buy 1, sell 2)
            'is_long_first': True,                # True = buy first leg, sell second leg
            'first_leg_delta_target': 0.60,       # Target delta for first leg (long)
            'second_leg_delta_target': 0.30,      # Target delta for second leg (short)
            'target_days_to_expiry': 30,          # Ideal DTE for ratio spread entry
            
            # Market condition preferences
            'directional_bias_threshold': 70,     # Minimum directional bias score (0-100)
            'volatility_percentile_min': 30,      # Minimum IV percentile for entry
            'volatility_percentile_max': 70,      # Maximum IV percentile for entry 
            
            # Risk parameters
            'max_risk_per_trade_pct': 2.0,        # Max risk as % of account
            'target_profit_pct': 50,              # Target profit as % of max profit
            'stop_loss_pct': 100,                 # Stop loss as % of credit received or debit paid
            'max_positions': 2,                   # Maximum concurrent positions
            
            # Exit parameters
            'days_before_expiry_exit': 7,         # Exit when reaching this DTE
        }
        
        # Update with default parameters
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Set spread type based on parameters
        if self.parameters['option_type'] == 'call':
            self.spread_type = AdvancedSpreadType.CALL_RATIO_SPREAD
        else:
            self.spread_type = AdvancedSpreadType.PUT_RATIO_SPREAD
        
        # Register for market events
        self.register_events()
        
        logger.info(f"Initialized Ratio Spread Strategy for {session.symbol}")
    
    def construct_ratio_spread(self, option_chain: pd.DataFrame, underlying_price: float) -> Optional[Dict[str, Any]]:
        """
        Construct a ratio spread from the option chain.
        
        Args:
            option_chain: Filtered option chain data
            underlying_price: Current price of the underlying asset
            
        Returns:
            Ratio spread position if successful, None otherwise
        """
        if option_chain is None or option_chain.empty:
            logger.warning("Empty option chain provided")
            return None
        
        # Apply filters for liquidity, open interest, etc.
        filtered_chain = self.filter_option_chains(option_chain)
        
        if filtered_chain is None or filtered_chain.empty:
            logger.warning("No suitable options found after filtering")
            return None
        
        # Determine option type and other parameters
        option_type = self.parameters['option_type'].lower()
        is_long_first = self.parameters['is_long_first']
        ratio = self.parameters['ratio']
        target_dte = self.parameters['target_days_to_expiry']
        first_delta_target = self.parameters['first_leg_delta_target']
        second_delta_target = self.parameters['second_leg_delta_target']
        
        # Filter by option type
        chain_by_type = filtered_chain[filtered_chain['option_type'] == option_type]
        
        if chain_by_type.empty:
            logger.warning(f"No {option_type} options found in chain")
            return None
        
        # Select expiration based on target days to expiry
        expiration = self.select_expiration_by_dte(chain_by_type, target_dte)
        
        if not expiration:
            logger.warning(f"No suitable expiration found near {target_dte} DTE")
            return None
        
        # Filter for selected expiration
        exp_options = chain_by_type[chain_by_type['expiration_date'] == expiration]
        
        if exp_options.empty:
            logger.warning("No options available for the selected expiration")
            return None
        
        # Find strikes based on delta targets
        first_strike = None
        second_strike = None
        first_delta_diff = float('inf')
        second_delta_diff = float('inf')
        
        for _, option in exp_options.iterrows():
            if 'delta' in option:
                delta = option['delta']
                if option_type == 'call':
                    # For calls, delta is positive
                    delta_diff_first = abs(delta - first_delta_target)
                    delta_diff_second = abs(delta - second_delta_target)
                else:
                    # For puts, delta is negative
                    delta_diff_first = abs(delta - (-first_delta_target))
                    delta_diff_second = abs(delta - (-second_delta_target))
                
                if delta_diff_first < first_delta_diff:
                    first_strike = option['strike']
                    first_delta_diff = delta_diff_first
                
                if delta_diff_second < second_delta_diff:
                    second_strike = option['strike']
                    second_delta_diff = delta_diff_second
        
        if first_strike is None or second_strike is None:
            logger.warning("Could not find suitable strikes based on delta targets")
            return None
        
        # For call ratio spreads, typically first_strike < second_strike
        # For put ratio spreads, typically first_strike > second_strike
        is_strike_order_correct = False
        if option_type == 'call' and first_strike < second_strike:
            is_strike_order_correct = True
        elif option_type == 'put' and first_strike > second_strike:
            is_strike_order_correct = True
        
        if not is_strike_order_correct:
            logger.warning(f"Strike order is not optimal for {option_type} ratio spread")
            # Swap strikes if needed
            first_strike, second_strike = second_strike, first_strike
        
        # Get specific option contracts
        first_options = exp_options[exp_options['strike'] == first_strike]
        second_options = exp_options[exp_options['strike'] == second_strike]
        
        if first_options.empty or second_options.empty:
            logger.warning("Missing options for one of the selected strikes")
            return None
        
        first_option = first_options.iloc[0]
        second_option = second_options.iloc[0]
        
        # Create leg objects
        first_leg = {
            'option_type': option_type,
            'strike': first_strike,
            'expiration': expiration,
            'action': 'buy' if is_long_first else 'sell',
            'quantity': 1,  # Always 1 for first leg
            'entry_price': first_option['ask'] if is_long_first else first_option['bid'],
            'delta': first_option.get('delta', first_delta_target if option_type == 'call' else -first_delta_target),
            'gamma': first_option.get('gamma', 0),
            'theta': first_option.get('theta', 0),
            'vega': first_option.get('vega', 0)
        }
        
        second_leg = {
            'option_type': option_type,
            'strike': second_strike,
            'expiration': expiration,
            'action': 'sell' if is_long_first else 'buy',
            'quantity': ratio,  # Ratio for second leg (typically 2 or 3)
            'entry_price': second_option['bid'] if is_long_first else second_option['ask'],
            'delta': second_option.get('delta', second_delta_target if option_type == 'call' else -second_delta_target),
            'gamma': second_option.get('gamma', 0),
            'theta': second_option.get('theta', 0),
            'vega': second_option.get('vega', 0)
        }
        
        # Calculate net debit/credit
        if is_long_first:
            # Long first leg, short second leg (at ratio)
            net_debit = first_leg['entry_price'] - (second_leg['entry_price'] * ratio)
        else:
            # Short first leg, long second leg (at ratio)
            net_debit = (second_leg['entry_price'] * ratio) - first_leg['entry_price']
        
        # Create position object
        position = {
            'position_id': str(uuid.uuid4()),
            'first_leg': first_leg,
            'second_leg': second_leg,
            'entry_time': datetime.now(),
            'spread_type': self.spread_type,
            'is_long_first': is_long_first,
            'ratio': ratio,
            'net_debit': net_debit,
            'status': 'open',
            'quantity': 1,  # Overall position quantity (will be adjusted later)
            'pnl': 0.0
        }
        
        # Calculate Greeks for the full position
        position['net_delta'] = first_leg['delta'] - (second_leg['delta'] * ratio) if is_long_first else (second_leg['delta'] * ratio) - first_leg['delta']
        position['net_gamma'] = first_leg.get('gamma', 0) - (second_leg.get('gamma', 0) * ratio) if is_long_first else (second_leg.get('gamma', 0) * ratio) - first_leg.get('gamma', 0)
        position['net_theta'] = first_leg.get('theta', 0) - (second_leg.get('theta', 0) * ratio) if is_long_first else (second_leg.get('theta', 0) * ratio) - first_leg.get('theta', 0)
        position['net_vega'] = first_leg.get('vega', 0) - (second_leg.get('vega', 0) * ratio) if is_long_first else (second_leg.get('vega', 0) * ratio) - first_leg.get('vega', 0)
        
        # Calculate max risk and breakeven points
        if is_long_first:
            # For long call ratio spreads
            if option_type == 'call':
                # Max risk is usually the debit paid
                position['max_risk'] = max(0, net_debit * 100)
                
                # Breakeven points
                width = second_leg['strike'] - first_leg['strike']
                lower_be = first_leg['strike'] + net_debit
                upper_be = second_leg['strike'] + (net_debit / (ratio - 1)) if ratio > 1 else float('inf')
                position['breakeven_points'] = [lower_be, upper_be]
            
            # For long put ratio spreads
            else:
                position['max_risk'] = max(0, net_debit * 100)
                
                width = first_leg['strike'] - second_leg['strike']
                upper_be = first_leg['strike'] - net_debit
                lower_be = second_leg['strike'] - (net_debit / (ratio - 1)) if ratio > 1 else 0
                position['breakeven_points'] = [lower_be, upper_be]
        else:
            # For short call ratio spreads
            if option_type == 'call':
                width = second_leg['strike'] - first_leg['strike']
                position['max_risk'] = ((width * 100 * ratio) - (first_leg['entry_price'] * 100)) if net_debit < 0 else float('inf')
                
                # Breakeven points
                credit = -net_debit if net_debit < 0 else 0
                be_point = first_leg['strike'] + credit
                position['breakeven_points'] = [be_point]
            
            # For short put ratio spreads
            else:
                width = first_leg['strike'] - second_leg['strike']
                position['max_risk'] = ((width * 100 * ratio) - (first_leg['entry_price'] * 100)) if net_debit < 0 else float('inf')
                
                # Breakeven points
                credit = -net_debit if net_debit < 0 else 0
                be_point = first_leg['strike'] - credit
                position['breakeven_points'] = [be_point]
        
        # Log detailed information about the ratio spread
        leg_type = "Long" if is_long_first else "Short"
        logger.info(f"Constructed {leg_type} {ratio}:1 {option_type} ratio spread: {first_strike}/{second_strike}, "
                   f"net {'debit' if net_debit > 0 else 'credit'}: ${abs(net_debit) * 100:.2f}")
        
        return position
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicators for ratio spread signals.
        
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
            
            # Price vs moving averages
            current_price = data['close'].iloc[-1]
            indicators['price_vs_sma20'] = (current_price / indicators['sma_20'] - 1) * 100  # as percentage
            indicators['price_vs_sma50'] = (current_price / indicators['sma_50'] - 1) * 100
            indicators['price_vs_sma200'] = (current_price / indicators['sma_200'] - 1) * 100
            
            # Trend detection based on price vs MAs and MA crossovers
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
            indicators['roc_5'] = ((current_price / data['close'].iloc[-6]) - 1) * 100  # 5-day rate of change
            indicators['roc_10'] = ((current_price / data['close'].iloc[-11]) - 1) * 100  # 10-day rate of change
            indicators['roc_20'] = ((current_price / data['close'].iloc[-21]) - 1) * 100  # 20-day rate of change
            
            # Volatility measures
            indicators['atr'] = self.calculate_atr(data, period=14)
            indicators['atr_pct'] = (indicators['atr'] / current_price) * 100 if indicators['atr'] else None
            
            try:
                # Historical volatility (20-day)
                log_returns = np.log(data['close'] / data['close'].shift(1))
                indicators['hv_20'] = log_returns.rolling(20).std() * np.sqrt(252) * 100
                indicators['hv_20_current'] = indicators['hv_20'].iloc[-1]
            except Exception as e:
                logger.warning(f"Error calculating historical volatility: {e}")
        
        # Calculate volume-based indicators
        if 'volume' in data.columns:
            indicators['volume_sma_20'] = data['volume'].rolling(20).mean().iloc[-1]
            indicators['volume_ratio'] = data['volume'].iloc[-1] / indicators['volume_sma_20']
        
        # Get implied volatility data from session
        if self.session.current_iv is not None:
            indicators['current_iv'] = self.session.current_iv
            
            # IV percentile calculation
            iv_metrics = self.calculate_implied_volatility_metrics(self.iv_history[self.session.symbol])
            indicators['iv_percentile'] = iv_metrics.get('iv_percentile', 50)
            indicators['iv_rank'] = iv_metrics.get('iv_rank', 50)
        
        # Calculate directional bias score (0-100)
        directional_score = 50  # Neutral starting point
        option_type = self.parameters['option_type']
        is_long_first = self.parameters['is_long_first']
        
        # Determine desired directional bias based on position type
        bullish_bias = False
        bearish_bias = False
        
        if (option_type == 'call' and is_long_first) or (option_type == 'put' and not is_long_first):
            bullish_bias = True  # We want a bullish market
        else:
            bearish_bias = True  # We want a bearish market
        
        # Adjust directional score based on indicators
        if bullish_bias:
            # Bullish trend confirmation
            if indicators['trend'] == 'uptrend':
                directional_score += 15
            elif indicators['trend'] == 'downtrend':
                directional_score -= 20
            
            # Momentum indicators for bullish bias
            if indicators['rsi_14'] > 60:
                directional_score += 10
            elif indicators['rsi_14'] < 40:
                directional_score -= 10
            
            if indicators['macd_hist'] > 0 and indicators['macd'] > indicators['macd_signal']:
                directional_score += 10
            elif indicators['macd_hist'] < 0 and indicators['macd'] < indicators['macd_signal']:
                directional_score -= 10
            
            # Rate of change for bullish momentum
            if indicators.get('roc_5', 0) > 0 and indicators.get('roc_10', 0) > 0:
                directional_score += 10
                if indicators.get('roc_5', 0) > indicators.get('roc_10', 0):  # Acceleration
                    directional_score += 5
            elif indicators.get('roc_5', 0) < 0 and indicators.get('roc_10', 0) < 0:
                directional_score -= 15
        
        elif bearish_bias:
            # Bearish trend confirmation
            if indicators['trend'] == 'downtrend':
                directional_score += 15
            elif indicators['trend'] == 'uptrend':
                directional_score -= 20
            
            # Momentum indicators for bearish bias
            if indicators['rsi_14'] < 40:
                directional_score += 10
            elif indicators['rsi_14'] > 60:
                directional_score -= 10
            
            if indicators['macd_hist'] < 0 and indicators['macd'] < indicators['macd_signal']:
                directional_score += 10
            elif indicators['macd_hist'] > 0 and indicators['macd'] > indicators['macd_signal']:
                directional_score -= 10
            
            # Rate of change for bearish momentum
            if indicators.get('roc_5', 0) < 0 and indicators.get('roc_10', 0) < 0:
                directional_score += 10
                if indicators.get('roc_5', 0) < indicators.get('roc_10', 0):  # Acceleration
                    directional_score += 5
            elif indicators.get('roc_5', 0) > 0 and indicators.get('roc_10', 0) > 0:
                directional_score -= 15
        
        # Volume confirmation
        if 'volume_ratio' in indicators:
            if (bullish_bias and indicators['trend'] == 'uptrend' and indicators['volume_ratio'] > 1.2) or \
               (bearish_bias and indicators['trend'] == 'downtrend' and indicators['volume_ratio'] > 1.2):
                directional_score += 5
        
        # Ensure score is within valid range
        indicators['directional_bias_score'] = max(0, min(100, directional_score))
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for ratio spreads based on market conditions.
        
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
        
        option_type = self.parameters['option_type']
        is_long_first = self.parameters['is_long_first']
        
        # Check if we already have too many positions
        if len(self.positions) >= self.parameters['max_positions']:
            logger.info("Maximum number of positions reached, no new entries.")
        else:
            # Check IV environment - Ratio spreads have specific IV preferences
            iv_check_passed = True
            if 'iv_percentile' in indicators:
                iv_percentile = indicators['iv_percentile']
                min_iv = self.parameters['volatility_percentile_min']
                max_iv = self.parameters['volatility_percentile_max']
                
                if iv_percentile < min_iv or iv_percentile > max_iv:
                    iv_check_passed = False
                    logger.info(f"IV filter failed: current IV percentile {iv_percentile:.2f} outside range [{min_iv}-{max_iv}]")
            
            # Check directional bias score
            if 'directional_bias_score' in indicators and iv_check_passed:
                directional_score = indicators['directional_bias_score']
                threshold = self.parameters['directional_bias_threshold']
                
                if directional_score >= threshold:
                    # Calculate signal strength
                    signal_strength = directional_score / 100.0
                    
                    # Confirm trend alignment
                    trend_aligned = False
                    
                    if (option_type == 'call' and is_long_first and indicators.get('trend') == 'uptrend') or \
                       (option_type == 'put' and not is_long_first and indicators.get('trend') == 'uptrend') or \
                       (option_type == 'put' and is_long_first and indicators.get('trend') == 'downtrend') or \
                       (option_type == 'call' and not is_long_first and indicators.get('trend') == 'downtrend'):
                        trend_aligned = True
                    
                    # Generate entry signal if all conditions met
                    if trend_aligned:
                        signals["entry"] = True
                        signals["signal_strength"] = signal_strength
                        logger.info(f"Ratio spread entry signal: directional score {directional_score}, "
                                   f"option type {option_type}, {'long' if is_long_first else 'short'} first leg, "
                                   f"strength {signal_strength:.2f}")
                    else:
                        logger.info(f"Ratio spread not aligned with current market trend")
                else:
                    logger.info(f"Directional score {directional_score} below threshold {threshold}")
        
        # Check for exits on existing positions
        for position in self.positions:
            if position.get('status') == "open":
                exit_signal = self._check_exit_conditions(position, data, indicators)
                if exit_signal:
                    signals["exit_positions"].append(position['position_id'])
        
        return signals
    
    def _check_exit_conditions(self, position: Dict[str, Any], data: pd.DataFrame, indicators: Dict[str, Any]) -> bool:
        """
        Check exit conditions for an open ratio spread position.
        
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
        if 'first_leg' in position and 'expiration' in position['first_leg']:
            expiry_date = position['first_leg'].get('expiration')
            if isinstance(expiry_date, str):
                expiry_date = datetime.strptime(expiry_date, '%Y-%m-%d').date()
            
            days_to_expiry = (expiry_date - datetime.now().date()).days
            
            # Exit ahead of expiration to avoid gamma risk
            if days_to_expiry <= self.parameters['days_before_expiry_exit']:
                logger.info(f"Exit signal: approaching expiration ({days_to_expiry} days left)")
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
        
        # 4. Trend change / directional bias loss
        option_type = position.get('first_leg', {}).get('option_type')
        is_long_first = position.get('is_long_first', True)
        current_trend = indicators.get('trend', 'neutral')
        
        # Define what unfavorable trend condition would be for this position
        unfavorable_trend = False
        
        if (option_type == 'call' and is_long_first and current_trend == 'downtrend') or \
           (option_type == 'put' and not is_long_first and current_trend == 'downtrend') or \
           (option_type == 'put' and is_long_first and current_trend == 'uptrend') or \
           (option_type == 'call' and not is_long_first and current_trend == 'uptrend'):
            unfavorable_trend = True
        
        if unfavorable_trend and indicators.get('directional_bias_score', 50) < 30:
            logger.info(f"Exit signal: unfavorable trend ({current_trend}) with low directional score")
            return True
        
        # 5. Specific for ratio spreads - check for rapid underlying movement
        # Ratio spreads can have unlimited risk to one side, so watch for rapid moves
        
        # For call ratio spreads, need to watch for upside risk if ratio > 1
        if option_type == 'call' and position.get('ratio', 1) > 1:
            second_leg_strike = position.get('second_leg', {}).get('strike', 0)
            
            # If price moves significantly above the short strike, risk increases
            if current_price > second_leg_strike * 1.05:  # 5% above short strike
                # Check for continuing momentum
                if indicators.get('roc_5', 0) > 2 and indicators.get('rsi_14', 50) > 70:
                    logger.info(f"Exit signal: price momentum above short call strike, increasing risk")
                    return True
        
        # For put ratio spreads, need to watch for downside risk if ratio > 1
        elif option_type == 'put' and position.get('ratio', 1) > 1:
            second_leg_strike = position.get('second_leg', {}).get('strike', 0)
            
            # If price moves significantly below the short strike, risk increases
            if current_price < second_leg_strike * 0.95:  # 5% below short strike
                # Check for continuing momentum
                if indicators.get('roc_5', 0) < -2 and indicators.get('rsi_14', 50) < 30:
                    logger.info(f"Exit signal: price momentum below short put strike, increasing risk")
                    return True
        
        return False
    
    def _calculate_theoretical_profit(self, position: Dict[str, Any], current_price: float) -> Optional[float]:
        """
        Calculate theoretical profit percentage for a ratio spread.
        
        Args:
            position: The position to evaluate
            current_price: Current price of the underlying
        
        Returns:
            Estimated profit percentage or None if calculation not possible
        """
        if 'first_leg' not in position or 'second_leg' not in position:
            return None
        
        # Extract position details
        first_leg = position['first_leg']
        second_leg = position['second_leg']
        ratio = position.get('ratio', 1)
        is_long_first = position.get('is_long_first', True)
        net_debit = position.get('net_debit', 0)
        
        if abs(net_debit) < 0.001:  # Avoid division by zero
            return None
        
        # Calculate theoretical current values of each leg
        first_strike = first_leg.get('strike')
        second_strike = second_leg.get('strike')
        option_type = first_leg.get('option_type')
        
        # Get expiration for time value calculation
        expiry = first_leg.get('expiration')
        if isinstance(expiry, str):
            expiry = datetime.strptime(expiry, '%Y-%m-%d').date()
        
        days_to_expiry = max(1, (expiry - datetime.now().date()).days)
        original_dte = self.parameters['target_days_to_expiry']
        time_decay_factor = days_to_expiry / original_dte
        
        # Very simplified pricing model - in a real implementation would use Black-Scholes
        # This is just a rough approximation based on intrinsic value + time value
        
        # First leg
        first_intrinsic = max(0, current_price - first_strike) if option_type == 'call' else max(0, first_strike - current_price)
        first_time_value = (first_leg.get('entry_price', 0) - first_intrinsic) * time_decay_factor
        first_value = first_intrinsic + first_time_value
        
        # Second leg
        second_intrinsic = max(0, current_price - second_strike) if option_type == 'call' else max(0, second_strike - current_price)
        second_time_value = (second_leg.get('entry_price', 0) - second_intrinsic) * time_decay_factor
        second_value = second_intrinsic + second_time_value
        
        # Calculate net value based on position type
        if is_long_first:  # Long first leg, short second leg (at ratio)
            current_value = first_value - (second_value * ratio)
        else:  # Short first leg, long second leg (at ratio)
            current_value = (second_value * ratio) - first_value
        
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
        Execute Ratio Spread specific trading signals.
        """
        if not self.signals:
            return
        
        # Handle entry signals
        if self.signals.get("entry", False):
            # Construct and open ratio spread position
            underlying_price = self.session.current_price
            
            if underlying_price and self.session.option_chain is not None:
                # Use the construct_ratio_spread method for position creation
                ratio_position = self.construct_ratio_spread(
                    self.session.option_chain, 
                    underlying_price
                )
                
                if ratio_position:
                    # Determine position size based on risk
                    account_value = self.session.account_value
                    max_risk_pct = self.parameters['max_risk_per_trade_pct'] / 100
                    max_risk_amount = account_value * max_risk_pct
                    
                    # Calculate position size based on max risk
                    position_max_risk = ratio_position.get('max_risk', 100)  # Default to $100 if not calculated
                    
                    if position_max_risk > 0:
                        # Determine number of spreads
                        num_spreads = int(max_risk_amount / position_max_risk)
                        num_spreads = max(1, min(5, num_spreads))  # At least 1, at most 5 spreads
                        ratio_position['quantity'] = num_spreads
                    
                    # Add position to tracking
                    self.positions.append(ratio_position)
                    
                    # Log entry details
                    option_type = ratio_position.get('first_leg', {}).get('option_type', '')
                    is_long_first = ratio_position.get('is_long_first', True)
                    ratio = ratio_position.get('ratio', 1)
                    first_strike = ratio_position.get('first_leg', {}).get('strike', 0)
                    second_strike = ratio_position.get('second_leg', {}).get('strike', 0)
                    
                    logger.info(f"Opened {option_type} ratio spread position {ratio_position['position_id']}: "
                               f"{'long' if is_long_first else 'short'} first leg, ratio {ratio}, "
                               f"strikes {first_strike}/{second_strike}, quantity {num_spreads}")
                else:
                    logger.warning("Failed to construct valid ratio spread")
        
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
                        pnl = profit_pct * abs(net_debit) * 100 * position.get('quantity', 1)
                        position['pnl'] = pnl
                        
                        logger.info(f"Closed ratio spread position {position_id}, P&L: ${pnl:.2f}")
                    else:
                        logger.info(f"Closed ratio spread position {position_id}, P&L calculation not available")
    
    def run_strategy(self):
        """
        Run the ratio spread strategy cycle.
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
            logger.warning("Insufficient market data for ratio spread analysis")
    
    def register_events(self):
        """Register for events relevant to Ratio Spreads."""
        # Register for common event types from the parent class
        super().register_events()
        
        # Add ratio spread specific event subscriptions
        EventBus.subscribe(EventType.EARNINGS_ANNOUNCEMENT, self.on_event)
        EventBus.subscribe(EventType.IMPLIED_VOLATILITY_CHANGE, self.on_event)
        EventBus.subscribe(EventType.MARKET_REGIME_CHANGE, self.on_event)
    
    def on_event(self, event: Event):
        """
        Process incoming events for the Ratio Spread strategy.
        
        Args:
            event: The event to process
        """
        # Let parent class handle common events first
        super().on_event(event)
        
        # Add additional Ratio Spread specific event handling
        if event.type == EventType.EARNINGS_ANNOUNCEMENT and self.session.symbol in event.data.get('symbols', []):
            # Earnings announcements can significantly impact ratio spreads
            days_to_earnings = event.data.get('days_to_announcement', 0)
            
            if days_to_earnings <= 5:  # Within 5 days of earnings
                logger.info(f"Earnings announcement approaching for {self.session.symbol}, evaluating ratio spread positions")
                
                # Ratio spreads with uncovered short options can be risky during earnings
                for position in self.positions:
                    if position.get('status') == "open" and position.get('ratio', 1) > 1:
                        logger.info(f"Closing ratio spread position {position.get('position_id')} ahead of earnings due to increased risk")
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
                
                # Ratio spreads can be sensitive to IV changes depending on structure
                for position in self.positions:
                    if position.get('status') == "open":
                        option_type = position.get('first_leg', {}).get('option_type', '')
                        is_long_first = position.get('is_long_first', True)
                        ratio = position.get('ratio', 1)
                        
                        # Significant risk for ratio spreads with multiple short options
                        if (ratio > 1) and ((direction == 'spike' and magnitude > 25) or 
                                           (direction == 'drop' and magnitude > 30)):
                            logger.info(f"Closing ratio spread {position.get('position_id')} due to significant IV change")
                            self.signals.setdefault("exit_positions", []).append(position.get('position_id'))
                
                # Execute the exits if needed
                if "exit_positions" in self.signals and self.signals["exit_positions"]:
                    self._execute_signals()
        
        elif event.type == EventType.MARKET_REGIME_CHANGE:
            new_regime = event.data.get('new_regime', '')
            
            # Market regime changes can significantly impact ratio spreads
            if new_regime in ['high_volatility', 'crash']:
                logger.info(f"Market regime changed to {new_regime}, evaluating ratio spread positions")
                
                # In high volatility or crash regimes, ratio spreads with uncovered shorts can be very risky
                for position in self.positions:
                    if position.get('status') == "open":
                        ratio = position.get('ratio', 1)
                        
                        if ratio > 1:
                            logger.info(f"Closing ratio spread position {position.get('position_id')} due to regime change to {new_regime}")
                            self.signals.setdefault("exit_positions", []).append(position.get('position_id'))
                
                # Execute the exits if needed
                if "exit_positions" in self.signals and self.signals["exit_positions"]:
                    self._execute_signals()
