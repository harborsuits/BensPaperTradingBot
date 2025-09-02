#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jade Lizard Strategy

A professional-grade jade lizard implementation that leverages the modular,
event-driven architecture. This strategy combines a short put with a short call spread
to collect premium in neutral to slightly bullish market conditions.
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
    name="JadeLizardStrategy",
    market_type="options",
    description="A strategy that combines a short put with a short call spread to collect premium in neutral to slightly bullish market conditions with declining volatility",
    timeframes=["1d", "1w"],
    parameters={
        "short_put_delta_target": {"description": "Target delta for short put", "type": "float"},
        "short_call_delta_target": {"description": "Target delta for short call", "type": "float"},
        "call_spread_width_pct": {"description": "Width of call spread as % of underlying price", "type": "float"},
        "target_days_to_expiry": {"description": "Ideal DTE for entry", "type": "integer"}
    }
)
class JadeLizardStrategy(AdvancedSpreadEngine, AccountAwareMixin):
    """
    Jade Lizard Strategy
    
    This strategy combines a short put with a short call spread (short call at a lower strike, 
    long call at a higher strike). It's a premium collection strategy that benefits from 
    neutral to slightly bullish market conditions with declining volatility.
    
    Features:
    - Adapts the legacy Jade Lizard implementation to the new architecture
    - Uses event-driven signal generation and position management
    - Leverages the AdvancedSpreadEngine for core advanced spread mechanics
    - Implements custom filtering and technical/volatility analysis
    """
    
    def __init__(self, session: OptionsSession, data_pipeline: DataPipeline, 
                 parameters: Dict[str, Any] = None):
        """
        Initialize the Jade Lizard strategy.
        
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
            'strategy_name': 'Jade Lizard',
            'strategy_id': 'jade_lizard',
            
            # Jade Lizard specific parameters
            'short_put_delta_target': -0.30,     # Target delta for short put
            'short_call_delta_target': 0.30,     # Target delta for short call
            'long_call_delta_target': 0.15,      # Target delta for long call
            'call_spread_width_pct': 2.0,        # Width of call spread as % of underlying price
            'target_days_to_expiry': 30,         # Ideal DTE for entry
            'min_credit_to_max_risk': 0.20,      # Min credit received as % of max risk
            
            # Market condition preferences
            'min_neutral_score': 65,            # Minimum score for neutral market bias (0-100)
            'prefer_high_iv': True,             # Jade lizards benefit from high IV environments
            'min_iv_percentile': 60,            # Minimum IV percentile for entry
            'max_iv_percentile': 90,            # Maximum IV percentile for entry
            
            # Risk parameters
            'max_risk_per_trade_pct': 2.0,      # Max risk as % of account
            'target_profit_pct': 50,            # Target profit as % of max profit (credit received)
            'stop_loss_pct': 200,               # Stop loss as % of credit received
            'max_positions': 2,                 # Maximum concurrent positions
            
            # Exit parameters
            'days_before_expiry_exit': 7,       # Exit when reaching this DTE
            'delta_adjustment_threshold': 0.50, # Adjust short put if delta exceeds this value
        }
        
        # Update with default parameters
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Set spread type
        self.spread_type = AdvancedSpreadType.JADE_LIZARD
        
        # Register for market events
        self.register_events()
        
        logger.info(f"Initialized Jade Lizard Strategy for {session.symbol}")
    
    def construct_jade_lizard(self, option_chain: pd.DataFrame, underlying_price: float) -> Optional[Dict[str, Any]]:
        """
        Construct a jade lizard from the option chain.
        
        Args:
            option_chain: Filtered option chain data
            underlying_price: Current price of the underlying asset
            
        Returns:
            Jade lizard position if successful, None otherwise
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
        short_put_delta_target = self.parameters['short_put_delta_target']
        short_call_delta_target = self.parameters['short_call_delta_target']
        long_call_delta_target = self.parameters['long_call_delta_target']
        call_spread_width_pct = self.parameters['call_spread_width_pct']
        
        # Select expiration based on target days to expiry
        expiration = self.select_expiration_by_dte(filtered_chain, target_dte)
        
        if not expiration:
            logger.warning(f"No suitable expiration found near {target_dte} DTE")
            return None
        
        # Split into puts and calls for the selected expiration
        exp_options = filtered_chain[filtered_chain['expiration_date'] == expiration]
        calls = exp_options[exp_options['option_type'] == 'call']
        puts = exp_options[exp_options['option_type'] == 'put']
        
        if calls.empty or puts.empty:
            logger.warning("Missing call or put options for the selected expiration")
            return None
        
        # Find strikes based on delta targets
        short_put_strike = None
        short_call_strike = None
        long_call_strike = None
        
        short_put_delta_diff = float('inf')
        short_call_delta_diff = float('inf')
        long_call_delta_diff = float('inf')
        
        # Find short put
        for _, put in puts.iterrows():
            if 'delta' in put:
                delta_diff = abs(put['delta'] - short_put_delta_target)
                if delta_diff < short_put_delta_diff:
                    short_put_strike = put['strike']
                    short_put_delta_diff = delta_diff
        
        # Find short call
        for _, call in calls.iterrows():
            if 'delta' in call:
                delta_diff = abs(call['delta'] - short_call_delta_target)
                if delta_diff < short_call_delta_diff:
                    short_call_strike = call['strike']
                    short_call_delta_diff = delta_diff
        
        if short_put_strike is None or short_call_strike is None:
            logger.warning("Could not find suitable strikes for short put and call")
            return None
        
        # Ensure short put is below short call
        if short_put_strike >= short_call_strike:
            logger.warning(f"Invalid strike configuration: short put ({short_put_strike}) >= short call ({short_call_strike})")
            return None
        
        # Find long call based on width or delta
        if self.parameters.get('use_delta_for_long_call', False):
            # Use delta target
            for _, call in calls.iterrows():
                if 'delta' in call and call['strike'] > short_call_strike:
                    delta_diff = abs(call['delta'] - long_call_delta_target)
                    if delta_diff < long_call_delta_diff:
                        long_call_strike = call['strike']
                        long_call_delta_diff = delta_diff
        else:
            # Use width percentage
            call_spread_width = underlying_price * (call_spread_width_pct / 100)
            target_long_call_strike = short_call_strike + call_spread_width
            
            # Find closest available strike
            available_call_strikes = sorted([call['strike'] for _, call in calls.iterrows() if call['strike'] > short_call_strike])
            if available_call_strikes:
                long_call_strike = min(available_call_strikes, key=lambda x: abs(x - target_long_call_strike))
        
        if long_call_strike is None:
            logger.warning("Could not find suitable strike for long call")
            return None
        
        # Get specific option contracts
        short_puts = puts[puts['strike'] == short_put_strike]
        short_calls = calls[calls['strike'] == short_call_strike]
        long_calls = calls[calls['strike'] == long_call_strike]
        
        if short_puts.empty or short_calls.empty or long_calls.empty:
            logger.warning("Missing options for one of the selected strikes")
            return None
        
        short_put = short_puts.iloc[0]
        short_call = short_calls.iloc[0]
        long_call = long_calls.iloc[0]
        
        # Create leg objects
        short_put_leg = {
            'option_type': 'put',
            'strike': short_put_strike,
            'expiration': expiration,
            'action': 'sell',
            'quantity': 1,
            'entry_price': short_put['bid'],  # Sell at bid
            'delta': short_put.get('delta', short_put_delta_target),
            'gamma': short_put.get('gamma', 0),
            'theta': short_put.get('theta', 0),
            'vega': short_put.get('vega', 0)
        }
        
        short_call_leg = {
            'option_type': 'call',
            'strike': short_call_strike,
            'expiration': expiration,
            'action': 'sell',
            'quantity': 1,
            'entry_price': short_call['bid'],  # Sell at bid
            'delta': short_call.get('delta', short_call_delta_target),
            'gamma': short_call.get('gamma', 0),
            'theta': short_call.get('theta', 0),
            'vega': short_call.get('vega', 0)
        }
        
        long_call_leg = {
            'option_type': 'call',
            'strike': long_call_strike,
            'expiration': expiration,
            'action': 'buy',
            'quantity': 1,
            'entry_price': long_call['ask'],  # Buy at ask
            'delta': long_call.get('delta', long_call_delta_target),
            'gamma': long_call.get('gamma', 0),
            'theta': long_call.get('theta', 0),
            'vega': long_call.get('vega', 0)
        }
        
        # Calculate net credit
        net_credit = short_put_leg['entry_price'] + short_call_leg['entry_price'] - long_call_leg['entry_price']
        
        if net_credit <= 0:
            logger.warning(f"Negative credit for jade lizard: ${net_credit:.2f}, rejecting")
            return None
        
        # Calculate max risk (for a jade lizard, max risk is short put strike minus net credit received)
        max_risk = (short_put_strike - net_credit) * 100
        
        # Check if credit to max risk ratio meets minimum threshold
        credit_to_risk_ratio = (net_credit * 100) / max_risk if max_risk > 0 else 0
        min_ratio = self.parameters['min_credit_to_max_risk']
        
        if credit_to_risk_ratio < min_ratio:
            logger.info(f"Jade lizard rejected: credit/risk ratio {credit_to_risk_ratio:.3f} below minimum {min_ratio:.3f}")
            return None
        
        # Create position object
        position = {
            'position_id': str(uuid.uuid4()),
            'short_put_leg': short_put_leg,
            'short_call_leg': short_call_leg,
            'long_call_leg': long_call_leg,
            'entry_time': datetime.now(),
            'spread_type': self.spread_type,
            'net_credit': net_credit,
            'max_risk': max_risk,
            'credit_to_risk_ratio': credit_to_risk_ratio,
            'status': 'open',
            'quantity': 1,  # Will be adjusted later based on risk parameters
            'pnl': 0.0
        }
        
        # Calculate Greeks for the full position
        position['net_delta'] = (
            short_put_leg['delta'] + 
            short_call_leg['delta'] + 
            long_call_leg['delta']
        )
        
        position['net_gamma'] = (
            short_put_leg.get('gamma', 0) + 
            short_call_leg.get('gamma', 0) + 
            long_call_leg.get('gamma', 0)
        )
        
        position['net_theta'] = (
            short_put_leg.get('theta', 0) + 
            short_call_leg.get('theta', 0) + 
            long_call_leg.get('theta', 0)
        )
        
        position['net_vega'] = (
            short_put_leg.get('vega', 0) + 
            short_call_leg.get('vega', 0) + 
            long_call_leg.get('vega', 0)
        )
        
        # Calculate breakeven point (typically the short put strike minus the net credit)
        position['breakeven_point'] = short_put_strike - net_credit
        
        logger.info(f"Constructed jade lizard: put {short_put_strike} / call {short_call_strike}/{long_call_strike}, "
                   f"credit: ${net_credit * 100:.2f}, max risk: ${max_risk:.2f}, "
                   f"credit/risk ratio: {credit_to_risk_ratio:.3f}")
        
        return position
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicators for jade lizard signals.
        
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
        
        # Get implied volatility data from session
        if self.session.current_iv is not None:
            indicators['current_iv'] = self.session.current_iv
            
            # IV percentile calculation
            iv_metrics = self.calculate_implied_volatility_metrics(self.iv_history[self.session.symbol])
            indicators['iv_percentile'] = iv_metrics.get('iv_percentile', 50)
            indicators['iv_rank'] = iv_metrics.get('iv_rank', 50)
        
        # Market regime identification (Combine trend and volatility metrics)
        regime = 'neutral'
        if indicators.get('trend') == 'uptrend' and indicators.get('hv_20_current', 20) < 20:
            regime = 'bullish-low-volatility'  # Good for jade lizard
        elif indicators.get('trend') == 'uptrend' and indicators.get('hv_20_current', 20) >= 20:
            regime = 'bullish-high-volatility'
        elif indicators.get('trend') == 'downtrend' and indicators.get('hv_20_current', 20) < 20:
            regime = 'bearish-low-volatility'
        elif indicators.get('trend') == 'downtrend' and indicators.get('hv_20_current', 20) >= 20:
            regime = 'bearish-high-volatility'
        elif indicators.get('trend') == 'neutral' and indicators.get('hv_20_current', 20) >= 20:
            regime = 'neutral-high-volatility'  # Potentially good for jade lizard
        elif indicators.get('trend') == 'neutral' and indicators.get('hv_20_current', 20) < 20:
            regime = 'neutral-low-volatility'
            
        indicators['market_regime'] = regime
        
        # Calculate neutral market bias score (0-100)
        # Jade Lizard works best in neutral to slightly bullish markets
        neutral_score = 50  # Neutral starting point
        
        # Adjust based on trend
        if indicators['trend'] == 'neutral':
            neutral_score += 20
        elif indicators['trend'] == 'uptrend' and abs(indicators['price_vs_sma20']) < 3:
            # Slight uptrend is good for jade lizard
            neutral_score += 15
        elif indicators['trend'] == 'downtrend':
            neutral_score -= 25
        
        # Adjust based on RSI (middle values are good)
        rsi = indicators['rsi_14']
        if 40 <= rsi <= 60:
            neutral_score += 15  # Neutral RSI is good
        elif 30 <= rsi < 40 or 60 < rsi <= 70:
            neutral_score += 5   # Slightly extended but still reasonable
        elif rsi < 30 or rsi > 70:
            neutral_score -= 15  # Overbought/oversold is bad for neutral strategies
        
        # Adjust based on volatility
        if indicators.get('atr_pct') is not None:
            atr_pct = indicators['atr_pct']
            if atr_pct < 1.5:
                neutral_score += 10  # Low relative volatility is good
            elif atr_pct > 3.0:
                neutral_score -= 15  # High relative volatility is risky
        
        # Adjust based on IV percentile
        if 'iv_percentile' in indicators:
            iv_percentile = indicators['iv_percentile']
            if 60 <= iv_percentile <= 90:
                neutral_score += 15  # High IV is good for selling premium
            elif iv_percentile > 90:
                neutral_score -= 5   # Extremely high IV could be risky
            elif iv_percentile < 30:
                neutral_score -= 15  # Low IV is not good for premium selling
        
        # Ensure score is within valid range
        indicators['neutral_bias_score'] = max(0, min(100, neutral_score))
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for jade lizard based on market conditions.
        
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
            # Check IV environment - Jade Lizard works best in high IV environments
            iv_check_passed = True
            if self.parameters['prefer_high_iv'] and 'iv_percentile' in indicators:
                iv_percentile = indicators['iv_percentile']
                min_iv = self.parameters['min_iv_percentile']
                max_iv = self.parameters['max_iv_percentile']
                
                if iv_percentile < min_iv or iv_percentile > max_iv:
                    iv_check_passed = False
                    logger.info(f"IV filter failed: current IV percentile {iv_percentile:.2f} outside range [{min_iv}-{max_iv}]")
            
            # Check neutral market bias score
            if 'neutral_bias_score' in indicators and iv_check_passed:
                neutral_score = indicators['neutral_bias_score']
                threshold = self.parameters['min_neutral_score']
                
                if neutral_score >= threshold:
                    # Check if market regime is favorable
                    regime = indicators.get('market_regime', 'unknown')
                    favorable_regimes = ['neutral-high-volatility', 'bullish-low-volatility', 'neutral-low-volatility']
                    
                    if regime in favorable_regimes:
                        # Calculate signal strength
                        signal_strength = neutral_score / 100.0
                        
                        # Generate entry signal
                        signals["entry"] = True
                        signals["signal_strength"] = signal_strength
                        
                        logger.info(f"Jade lizard entry signal: neutral score {neutral_score}, "
                                   f"market regime {regime}, strength {signal_strength:.2f}")
                    else:
                        logger.info(f"Jade lizard entry skipped: unfavorable market regime {regime}")
                else:
                    logger.info(f"Neutral score {neutral_score} below threshold {threshold}")
        
        # Check for exits on existing positions
        for position in self.positions:
            if position.get('status') == "open":
                exit_signal = self._check_exit_conditions(position, data, indicators)
                if exit_signal:
                    signals["exit_positions"].append(position['position_id'])
        
        return signals
    
    def _check_exit_conditions(self, position: Dict[str, Any], data: pd.DataFrame, indicators: Dict[str, Any]) -> bool:
        """
        Check exit conditions for an open jade lizard position.
        
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
        if 'short_put_leg' in position and 'expiration' in position['short_put_leg']:
            expiry_date = position['short_put_leg'].get('expiration')
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
        
        # 4. Check for price breach of short put strike
        if 'short_put_leg' in position:
            short_put_strike = position['short_put_leg'].get('strike', 0)
            
            # If price moves significantly below the short put strike and is still dropping
            if current_price < short_put_strike * 0.98:  # 2% below short put strike
                # Check for continuing downward momentum
                if indicators.get('trend') == 'downtrend' and indicators.get('rsi_14', 50) < 35:
                    logger.info(f"Exit signal: price breached short put strike with continuing downtrend")
                    return True
        
        # 5. Check for delta adjustment threshold
        # Jade Lizard's short put can become very risky as delta approaches -1
        # Get current delta of short put (would be calculated dynamically in a real system)
        if 'short_put_leg' in position:
            # Simplified approach to estimate current delta
            short_put_strike = position['short_put_leg'].get('strike', 0)
            original_delta = position['short_put_leg'].get('delta', -0.30)
            
            # As stock price approaches or goes below short put strike, delta increases
            delta_adjustment = 0
            if current_price < short_put_strike * 1.05:  # Within 5% of strike
                moneyness = (current_price / short_put_strike) - 1
                # Simple delta estimation - in a real system would use option pricing model
                delta_adjustment = max(0, -0.5 * moneyness)  # Increase delta as price falls below strike
            
            estimated_current_delta = original_delta - delta_adjustment
            
            if abs(estimated_current_delta) > self.parameters['delta_adjustment_threshold']:
                logger.info(f"Exit signal: short put delta threshold exceeded (est. {estimated_current_delta:.2f})")
                return True
        
        # 6. Market regime change check
        if indicators.get('market_regime') in ['bearish-high-volatility', 'bearish-low-volatility']:
            # Jade Lizard is vulnerable in bearish markets
            if indicators.get('neutral_bias_score', 50) < 30:
                logger.info(f"Exit signal: unfavorable market regime with low neutral score")
                return True
        
        return False
    
    def _calculate_theoretical_profit(self, position: Dict[str, Any], current_price: float) -> Optional[float]:
        """
        Calculate theoretical profit percentage for a jade lizard.
        
        Args:
            position: The position to evaluate
            current_price: Current price of the underlying
        
        Returns:
            Estimated profit percentage or None if calculation not possible
        """
        if 'short_put_leg' not in position or 'short_call_leg' not in position or 'long_call_leg' not in position:
            return None
        
        # Extract position details
        short_put_leg = position['short_put_leg']
        short_call_leg = position['short_call_leg']
        long_call_leg = position['long_call_leg']
        net_credit = position.get('net_credit', 0)
        
        if net_credit <= 0:  # Avoid division by zero or negative credit
            return None
        
        # Calculate theoretical current values of each leg
        short_put_strike = short_put_leg.get('strike')
        short_call_strike = short_call_leg.get('strike')
        long_call_strike = long_call_leg.get('strike')
        
        # Get expiration for time value calculation
        expiry = short_put_leg.get('expiration')
        if isinstance(expiry, str):
            expiry = datetime.strptime(expiry, '%Y-%m-%d').date()
        
        days_to_expiry = max(1, (expiry - datetime.now().date()).days)
        original_dte = self.parameters['target_days_to_expiry']
        time_decay_factor = days_to_expiry / original_dte
        
        # Very simplified pricing model - in a real implementation would use Black-Scholes
        # This is just a rough approximation
        
        # Short put leg
        short_put_intrinsic = max(0, short_put_strike - current_price)
        short_put_time_value = (short_put_leg.get('entry_price', 0) - short_put_intrinsic) * time_decay_factor
        short_put_value = short_put_intrinsic + short_put_time_value
        
        # Short call leg
        short_call_intrinsic = max(0, current_price - short_call_strike)
        short_call_time_value = (short_call_leg.get('entry_price', 0) - short_call_intrinsic) * time_decay_factor
        short_call_value = short_call_intrinsic + short_call_time_value
        
        # Long call leg
        long_call_intrinsic = max(0, current_price - long_call_strike)
        long_call_time_value = (long_call_leg.get('entry_price', 0) - long_call_intrinsic) * time_decay_factor
        long_call_value = long_call_intrinsic + long_call_time_value
        
        # Calculate current theoretical value of the position
        # For short legs, we pay to close; for long legs, we receive when we close
        current_cost_to_close = short_put_value + short_call_value - long_call_value
        
        # Original credit received
        original_credit = net_credit
        
        # Calculate profit as percentage of original credit
        profit = original_credit - current_cost_to_close
        profit_pct = profit / original_credit
        
        return profit_pct
    
    def _execute_signals(self):

        # Verify account has sufficient buying power
        buying_power = self.get_buying_power()
        if buying_power <= 0:
            logger.warning("Trade execution aborted: Insufficient buying power")
            return
        """
        Execute Jade Lizard specific trading signals.
        """
        if not self.signals:
            return
        
        # Handle entry signals
        if self.signals.get("entry", False):
            # Construct and open jade lizard position
            underlying_price = self.session.current_price
            
            if underlying_price and self.session.option_chain is not None:
                # Use the construct_jade_lizard method for position creation
                jade_position = self.construct_jade_lizard(
                    self.session.option_chain, 
                    underlying_price
                )
                
                if jade_position:
                    # Determine position size based on risk
                    account_value = self.session.account_value
                    max_risk_pct = self.parameters['max_risk_per_trade_pct'] / 100
                    max_risk_amount = account_value * max_risk_pct
                    
                    # Calculate position size based on max risk
                    position_max_risk = jade_position.get('max_risk', float('inf'))
                    
                    if position_max_risk < float('inf'):
                        # Number of jade lizards to trade
                        num_spreads = int(max_risk_amount / position_max_risk)
                        num_spreads = max(1, min(5, num_spreads))  # At least 1, at most 5 spreads
                        jade_position['quantity'] = num_spreads
                    
                    # Add position
                    self.positions.append(jade_position)
                    
                    # Log entry details
                    short_put_strike = jade_position.get('short_put_leg', {}).get('strike', 0)
                    short_call_strike = jade_position.get('short_call_leg', {}).get('strike', 0)
                    long_call_strike = jade_position.get('long_call_leg', {}).get('strike', 0)
                    net_credit = jade_position.get('net_credit', 0)
                    
                    logger.info(f"Opened jade lizard position {jade_position['position_id']}: "
                              f"put {short_put_strike} / call {short_call_strike}/{long_call_strike}, "
                              f"credit: ${net_credit * 100 * num_spreads:.2f}, quantity: {num_spreads}")
                else:
                    logger.warning("Failed to construct valid jade lizard")
        
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
                        net_credit = position.get('net_credit', 0)
                        quantity = position.get('quantity', 1)
                        pnl = profit_pct * (net_credit * 100) * quantity
                        position['pnl'] = pnl
                        
                        logger.info(f"Closed jade lizard position {position_id}, P&L: ${pnl:.2f}")
                    else:
                        logger.info(f"Closed jade lizard position {position_id}, P&L calculation not available")
    
    def run_strategy(self):
        """
        Run the jade lizard strategy cycle.
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
            logger.warning("Insufficient market data for jade lizard analysis")
    
    def register_events(self):
        """Register for events relevant to Jade Lizard strategy."""
        # Register for common event types from the parent class
        super().register_events()
        
        # Add jade lizard specific event subscriptions
        EventBus.subscribe(EventType.EARNINGS_ANNOUNCEMENT, self.on_event)
        EventBus.subscribe(EventType.IMPLIED_VOLATILITY_CHANGE, self.on_event)
        EventBus.subscribe(EventType.MARKET_REGIME_CHANGE, self.on_event)
    
    def on_event(self, event: Event):
        """
        Process incoming events for the Jade Lizard strategy.
        
        Args:
            event: The event to process
        """
        # Let parent class handle common events first
        super().on_event(event)
        
        # Add additional Jade Lizard specific event handling
        if event.type == EventType.EARNINGS_ANNOUNCEMENT and self.session.symbol in event.data.get('symbols', []):
            # Earnings announcements can significantly impact jade lizard positions
            days_to_earnings = event.data.get('days_to_announcement', 0)
            
            if days_to_earnings <= 5:  # Within 5 days of earnings
                logger.info(f"Earnings announcement approaching for {self.session.symbol}, closing jade lizard positions")
                
                # Close all open jade lizard positions ahead of earnings
                for position in self.positions:
                    if position.get('status') == "open":
                        logger.info(f"Closing jade lizard position {position.get('position_id')} ahead of earnings")
                        self.signals.setdefault("exit_positions", []).append(position.get('position_id'))
                
                # Execute the exits
                self._execute_signals()
        
        elif event.type == EventType.IMPLIED_VOLATILITY_CHANGE:
            symbol = event.data.get('symbol')
            magnitude = event.data.get('magnitude', 0)
            direction = event.data.get('direction', '')
            
            if symbol == self.session.symbol and abs(magnitude) > 15:  # Significant IV change
                logger.info(f"Significant IV {direction} of {magnitude}% detected")
                
                # For jade lizard, we benefit from IV contraction (as a net premium seller)
                if direction == 'spike' and magnitude > 25:
                    logger.info(f"Significant IV spike may increase risk for jade lizard positions")
                    
                    # Consider closing positions if IV increases dramatically
                    for position in self.positions:
                        if position.get('status') == "open":
                            # Look at the profitability of the position
                            current_price = self.session.current_price
                            profit_pct = self._calculate_theoretical_profit(position, current_price)
                            
                            # If position is already profitable, lock in gains
                            if profit_pct and profit_pct > 0.25:  # 25% of max profit
                                logger.info(f"Closing profitable jade lizard {position.get('position_id')} due to IV spike")
                                self.signals.setdefault("exit_positions", []).append(position.get('position_id'))
                
                # Execute the exits if needed
                if "exit_positions" in self.signals and self.signals["exit_positions"]:
                    self._execute_signals()
        
        elif event.type == EventType.MARKET_REGIME_CHANGE:
            new_regime = event.data.get('new_regime', '')
            
            # Jade lizard is vulnerable in bearish high volatility regimes
            if new_regime in ['bearish-high-volatility', 'crash']:
                logger.info(f"Market regime changed to {new_regime}, evaluating jade lizard positions")
                
                # Close all positions in unfavorable regime
                for position in self.positions:
                    if position.get('status') == "open":
                        logger.info(f"Closing jade lizard position {position.get('position_id')} due to regime change to {new_regime}")
                        self.signals.setdefault("exit_positions", []).append(position.get('position_id'))
                
                # Execute the exits
                if "exit_positions" in self.signals and self.signals["exit_positions"]:
                    self._execute_signals()
