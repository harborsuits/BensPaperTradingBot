#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Covered Call Strategy Implementation

This strategy implements a covered call options strategy following the standardized
template system, ensuring compatibility with the backtester and trading dashboard.

A covered call is created by:
1. Buying (or already owning) shares of the underlying stock
2. Selling call options against the owned shares (typically 1 contract per 100 shares)
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Union

from trading_bot.strategies.base.options_base_strategy import OptionsBaseStrategy
from trading_bot.strategies.factory.strategy_registry import register_strategy, StrategyType, AssetClass, MarketRegime, TimeFrame
from trading_bot.market.market_data import MarketData
from trading_bot.market.universe import Universe
from trading_bot.market.option_chains import OptionChains

logger = logging.getLogger(__name__)

@register_strategy({
    'asset_class': AssetClass.OPTIONS.value,
    'strategy_type': StrategyType.INCOME.value,
    'timeframe': TimeFrame.SWING.value,
    'compatible_market_regimes': [MarketRegime.RANGING.value, MarketRegime.LOW_VOLATILITY.value],
    'description': 'Covered Call strategy for income generation in neutral to slightly bullish markets',
    'risk_level': 'low',
    'typical_holding_period': '30-45 days'
})
class CoveredCallStrategy(OptionsBaseStrategy):
    """
    Covered Call Options Strategy
    
    This strategy involves owning shares of the underlying stock and selling
    call options against that position. It's a popular income-generating strategy
    that slightly reduces downside risk while capping potential upside gains.
    
    Key characteristics:
    - Income generation through option premium collection
    - Partial downside protection (by the amount of premium received)
    - Capped upside potential (limited to strike price plus premium)
    - Most suitable in neutral to slightly bullish markets
    """
    
    # Default parameters for covered call strategy
    DEFAULT_PARAMS = {
        'strategy_name': 'covered_call',
        'strategy_version': '1.0.0',
        'asset_class': 'options',
        'strategy_type': 'income',
        'timeframe': 'swing',
        'market_regime': 'neutral',
        
        # Stock selection criteria
        'min_stock_price': 30.0,              # Minimum stock price to consider
        'max_stock_price': 500.0,             # Maximum stock price to consider
        'min_option_volume': 100,             # Minimum option volume
        'min_option_open_interest': 200,      # Minimum option open interest
        'min_iv_percentile': 40,              # Minimum IV percentile
        'max_iv_percentile': 80,              # Maximum IV percentile
        
        # Stock quality criteria
        'min_dividend_yield': 0.0,            # Minimum dividend yield (0 = not required)
        'min_market_cap': 2000000000,         # Minimum market cap ($2B by default)
        'quality_score_min': 60,              # Minimum quality score (0-100 scale)
        
        # Option parameters
        'target_dte': 30,                     # Target days to expiration (monthly cycle)
        'min_dte': 25,                        # Minimum days to expiration
        'max_dte': 40,                        # Maximum days to expiration
        'delta_target': 0.30,                 # Target delta for call options
        'delta_range': 0.05,                  # Acceptable range around target delta
        
        # Pricing and premium parameters
        'min_premium_yield': 0.008,           # Minimum premium as % of stock price (0.8%)
        'annualized_target_return': 0.12,     # Target annualized return from premiums (12%)
        
        # Risk management parameters
        'max_position_size_percent': 0.05,    # Maximum position size as % of portfolio
        'stock_stop_loss_percent': 0.15,      # Stock stop loss percentage (15%)
        'position_size_adjustment': 'equal',  # Position sizing method: 'equal', 'volatility', 'beta'
        
        # Exit parameters
        'profit_target_percent': 80,          # Exit at this percentage of max profit
        'roll_when_dte': 5,                   # Roll options with this many days left
        'min_roll_credit': 0.002,             # Minimum credit to collect when rolling (0.2%)
        'close_on_ex_dividend_day': True,     # Whether to close before ex-dividend date to avoid assignment
    }
    
    def __init__(self, 
                 strategy_id: str = None, 
                 name: str = None,
                 parameters: Dict[str, Any] = None):
        """
        Initialize the Covered Call strategy.
        
        Args:
            strategy_id: Unique identifier for this strategy instance
            name: Human-readable name of the strategy
            parameters: Strategy-specific parameters that override the defaults
        """
        super().__init__(strategy_id, name, parameters)
        
        # Strategy-specific tracking
        self.covered_call_positions = {}     # Track current covered call positions
        self.stock_positions = {}            # Track underlying stock positions
        self.dividend_calendar = {}          # Track upcoming dividend dates
    
    def generate_signals(self, market_data: MarketData, 
                         option_chains: Optional[OptionChains] = None) -> List[Dict[str, Any]]:
        """
        Generate covered call signals based on market data and option chains.
        
        Args:
            market_data: Market data for analysis
            option_chains: Option chain data
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        # Get universe of tradable stocks
        universe = self.define_universe(market_data)
        symbols = universe.get_symbols()
        
        for symbol in symbols:
            # Check if this stock meets our neutral-to-bullish criteria
            if not self._is_suitable_for_covered_call(symbol, market_data):
                continue
                
            # Get option chain for this symbol
            if not option_chains or not option_chains.has_symbol(symbol):
                continue
                
            chain = option_chains.get_chain(symbol)
            if not chain:
                continue
                
            # Find appropriate call options for covered call
            covered_call = self._find_covered_call_option(symbol, chain, market_data)
            if not covered_call:
                continue
                
            # Create signal
            signal = self._create_covered_call_signal(symbol, covered_call, market_data)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _is_suitable_for_covered_call(self, symbol: str, market_data: MarketData) -> bool:
        """
        Check if a stock is suitable for a covered call.
        
        Args:
            symbol: Stock symbol
            market_data: Market data
            
        Returns:
            True if stock is suitable for covered call, False otherwise
        """
        # Get historical data
        historical_data = market_data.get_historical_data(
            symbol, 
            period=90  # 3 months of data
        )
        
        if historical_data is None or len(historical_data) < 50:
            return False
            
        # Convert to DataFrame if it's not already
        if not isinstance(historical_data, pd.DataFrame):
            historical_data = pd.DataFrame(historical_data)
        
        # Calculate technical indicators
        historical_data['sma_20'] = historical_data['close'].rolling(window=20).mean()
        historical_data['sma_50'] = historical_data['close'].rolling(window=50).mean()
        
        # Calculate volatility
        historical_data['returns'] = historical_data['close'].pct_change()
        volatility = historical_data['returns'].std() * np.sqrt(252)  # Annualized
        
        # Calculate distance from 20-day SMA
        current_price = historical_data['close'].iloc[-1]
        sma_20 = historical_data['sma_20'].iloc[-1]
        distance_from_sma = abs(current_price - sma_20) / sma_20
        
        # Check if current price is near 20-day SMA (consolidating/neutral)
        is_near_sma = distance_from_sma <= 0.03  # Within 3% of 20-day SMA
        
        # Check if price is above 50-day SMA (longer-term uptrend)
        is_above_sma50 = current_price > historical_data['sma_50'].iloc[-1]
        
        # Check volatility (not too volatile, not too flat)
        has_good_volatility = 0.15 <= volatility <= 0.35
        
        # Check volume trend (consistent or increasing)
        volume_trend = historical_data['volume'].iloc[-20:].mean() / historical_data['volume'].iloc[-40:-20].mean()
        has_good_volume = volume_trend >= 0.8  # At least 80% of previous period's volume
        
        # Check for earnings upcoming (avoid selling calls right before earnings)
        # This would normally use an earnings calendar API
        # For now, we'll just assume no earnings is coming up
        no_earnings_soon = True
        
        # Combine criteria (at least 3 out of 5 should be true)
        criteria_met = sum([is_near_sma, is_above_sma50, has_good_volatility, has_good_volume, no_earnings_soon])
        return criteria_met >= 3
    
    def _find_covered_call_option(self, symbol: str, chain: Dict[str, Any], 
                                market_data: MarketData) -> Optional[Dict[str, Any]]:
        """
        Find appropriate call option for a covered call strategy.
        
        Args:
            symbol: Stock symbol
            chain: Option chain data
            market_data: Market data
            
        Returns:
            Dictionary with call option details if found, None otherwise
        """
        # Get stock price
        quote = market_data.get_latest_quote(symbol)
        if not quote:
            return None
            
        stock_price = quote.get('price', 0)
        if stock_price <= 0:
            return None
            
        # Get parameters
        target_dte = self.parameters.get('target_dte', 30)
        min_dte = self.parameters.get('min_dte', 25)
        max_dte = self.parameters.get('max_dte', 40)
        delta_target = self.parameters.get('delta_target', 0.30)
        delta_range = self.parameters.get('delta_range', 0.05)
        
        # Find appropriate expiration date
        today = date.today()
        expirations = chain.get('expirations', [])
        best_expiry = None
        best_dte = float('inf')
        
        for exp in expirations:
            exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
            dte = (exp_date - today).days
            
            if min_dte <= dte <= max_dte:
                if abs(dte - target_dte) < abs(best_dte - target_dte):
                    best_expiry = exp
                    best_dte = dte
        
        if not best_expiry:
            return None
            
        # Get calls for the selected expiration
        calls = [opt for opt in chain.get('calls', []) if opt.get('expiration') == best_expiry]
        if not calls:
            return None
            
        # Filter for minimum volume and open interest
        min_volume = self.parameters.get('min_option_volume', 100)
        min_oi = self.parameters.get('min_option_open_interest', 200)
        
        calls = [opt for opt in calls if 
                 opt.get('volume', 0) >= min_volume and 
                 opt.get('open_interest', 0) >= min_oi]
        
        if not calls:
            return None
            
        # Sort by strike
        calls.sort(key=lambda x: x.get('strike', 0))
        
        # Find OTM call with delta closest to target
        otm_calls = [call for call in calls if call.get('strike', 0) > stock_price]
        if not otm_calls:
            return None
            
        # Find call with delta closest to target_delta
        best_call = min(otm_calls, key=lambda x: abs(abs(x.get('delta', 0)) - delta_target))
        
        # Ensure bid price is reasonable
        if best_call.get('bid', 0) <= 0:
            return None
            
        # Calculate metrics
        strike = best_call.get('strike', 0)
        premium = best_call.get('bid', 0)  # Use bid for selling
        
        # Calculate yield metrics
        premium_yield = premium / stock_price
        annualized_yield = premium_yield * (365 / best_dte)
        
        # Check minimum premium yield
        min_premium_yield = self.parameters.get('min_premium_yield', 0.008)
        if premium_yield < min_premium_yield:
            return None
            
        # Return covered call details
        return {
            'symbol': symbol,
            'stock_price': stock_price,
            'call_option': best_call,
            'strike': strike,
            'premium': premium,
            'expiration': best_expiry,
            'dte': best_dte,
            'delta': best_call.get('delta', 0),
            'premium_yield': premium_yield,
            'annualized_yield': annualized_yield,
            'upside_potential': (strike - stock_price) / stock_price,
            'total_return': premium_yield + max(0, (strike - stock_price) / stock_price),
        }
    
    def _create_covered_call_signal(self, symbol: str, covered_call: Dict[str, Any], 
                                  market_data: MarketData) -> Optional[Dict[str, Any]]:
        """
        Create a signal for a covered call trade.
        
        Args:
            symbol: Stock symbol
            covered_call: Covered call details
            market_data: Market data
            
        Returns:
            Signal dictionary
        """
        if not covered_call:
            return None
            
        # Extract covered call details
        stock_price = covered_call.get('stock_price', 0)
        strike = covered_call.get('strike', 0)
        premium = covered_call.get('premium', 0)
        expiration = covered_call.get('expiration', '')
        dte = covered_call.get('dte', 0)
        premium_yield = covered_call.get('premium_yield', 0)
        annualized_yield = covered_call.get('annualized_yield', 0)
        
        # Check for existing position
        has_stock = symbol in self.stock_positions
        already_has_call = symbol in self.covered_call_positions
        
        # If we already have a covered call on this stock, skip it
        if already_has_call:
            return None
            
        # Create the appropriate signal (buy-write or sell call against existing stock)
        action = "BUY_WRITE" if not has_stock else "SELL_CALL"
        
        # Calculate stop loss for stock
        stop_loss_pct = self.parameters.get('stock_stop_loss_percent', 0.15)
        stop_loss = stock_price * (1 - stop_loss_pct)
        
        # Create base options signal
        signal = self.create_options_signal(
            symbol=symbol,
            action=action,
            option_type="CALL",
            strike=strike,
            expiration=expiration,
            premium=premium,
            stock_price=stock_price,
            reason=f"Covered call with {premium_yield:.1%} premium yield",
            strength=min(1.0, annualized_yield / 0.25),  # Normalized strength based on yield
            stop_loss=stop_loss,
            take_profit=None  # For covered calls, we typically hold to expiration
        )
        
        # Add covered call-specific details
        signal.update({
            'strategy_subtype': 'covered_call',
            'stock_quantity': 100,  # Covered call requires 100 shares per contract
            'premium_yield': premium_yield,
            'annualized_yield': annualized_yield,
            'dte': dte,
            'max_profit': (strike - stock_price) + premium,
            'max_profit_percent': ((strike - stock_price) + premium) / stock_price,
            'breakeven': stock_price - premium,
            'stock_stop_loss': stop_loss,
            'stock_stop_loss_percent': stop_loss_pct,
            'option_symbol': covered_call.get('call_option', {}).get('symbol'),
        })
        
        return signal
    
    def position_sizing(self, signal: Dict[str, Any], account_info: Dict[str, Any]) -> float:
        """
        Calculate position size for a covered call.
        
        Args:
            signal: Trading signal dictionary
            account_info: Account information including equity, margin, etc.
            
        Returns:
            Number of contracts to trade
        """
        account_value = account_info.get('equity', 0)
        if account_value <= 0:
            return 0
            
        # Extract parameters
        max_position_pct = self.parameters.get('max_position_size_percent', 0.05)
        position_sizing_method = self.parameters.get('position_size_adjustment', 'equal')
        
        # Maximum amount to allocate to this position
        max_amount = account_value * max_position_pct
        
        # Calculate position size based on stock price (100 shares per contract)
        stock_price = signal.get('stock_price', 0)
        if stock_price <= 0:
            return 0
            
        # Cost per contract (stock cost - premium received)
        contract_cost = (stock_price * 100) - (signal.get('premium', 0) * 100)
        
        # Calculate how many contracts we can afford
        max_contracts = int(max_amount / contract_cost)
        
        # Apply adjustments based on sizing method
        if position_sizing_method == 'volatility':
            # Reduce position size for higher volatility stocks
            volatility = signal.get('volatility', 0.3)  # Default to medium if not provided
            volatility_factor = 0.3 / volatility  # Normalize around 0.3
            max_contracts = int(max_contracts * min(1.5, max(0.5, volatility_factor)))
        
        elif position_sizing_method == 'beta':
            # Adjust position size based on stock beta
            beta = signal.get('beta', 1.0)  # Default to 1.0 if not provided
            beta_factor = 1.0 / max(0.5, min(2.0, beta))  # Limit range to avoid extremes
            max_contracts = int(max_contracts * beta_factor)
        
        return max(1, max_contracts)  # At least 1 contract
    
    def on_exit_signal(self, position: Dict[str, Any], market_data=None, option_chains=None) -> Optional[Dict[str, Any]]:
        """
        Generate an exit signal for a covered call position using comprehensive exit criteria.
        
        Args:
            position: Current position data
            market_data: Optional market data for additional analysis
            option_chains: Optional option chain data for Greeks and pricing
            
        Returns:
            Exit signal if conditions are met, None otherwise
        """
        # Extract position details
        symbol = position.get('symbol')
        action = position.get('action')
        entry_time = position.get('timestamp')
        current_time = datetime.now()
        expiration = position.get('expiration')
        
        # Exit if required data is missing
        if not symbol or not entry_time or not expiration:
            return None
            
        # Parse expiration date
        expiry_date = datetime.strptime(expiration, '%Y-%m-%d').date()
        today = date.today()
        dte_remaining = (expiry_date - today).days
        
        # Check comprehensive exit criteria
        should_exit, reason = self._should_exit_position(position, market_data, option_chains)
        if should_exit:
            return self.create_signal(
                symbol=symbol,
                action="CLOSE_POSITION",
                reason=reason,
                strength=1.0
            )
        
        # Check if we should roll the position instead of closing
        should_roll, roll_reason = self._should_roll_position(position, market_data, option_chains)
        if should_roll:
            return self.create_signal(
                symbol=symbol,
                action="ROLL_COVERED_CALL",
                reason=roll_reason,
                strength=1.0
            )
        
        # No exit signal needed
        return None
        
    def _should_exit_position(self, position: Dict[str, Any], market_data=None, option_chains=None) -> Tuple[bool, str]:
        """
        Comprehensive evaluation of whether a position should be exited based on multiple criteria.
        
        Args:
            position: Current position data
            market_data: Optional market data for additional analysis
            option_chains: Optional option chain data for Greeks analysis
            
        Returns:
            Tuple of (should_exit: bool, reason: str)
        """
        symbol = position.get('symbol')
        entry_time = position.get('timestamp')
        current_time = datetime.now()
        expiration = position.get('expiration')
        
        # Parse expiration date
        expiry_date = datetime.strptime(expiration, '%Y-%m-%d').date() if expiration else None
        today = date.today()
        
        # Check profit target
        profit_target_pct = self.parameters.get('profit_target_percent', 80) / 100
        current_option_price = position.get('current_option_price', 0)
        original_premium = position.get('premium', 0)
        
        if current_option_price > 0 and original_premium > 0:
            profit_pct = (original_premium - current_option_price) / original_premium
            
            if profit_pct >= profit_target_pct:
                return True, f"Profit target reached ({profit_pct:.1%} of premium)"
        
        # Check stock stop loss
        stock_stop_loss = position.get('stock_stop_loss')
        current_stock_price = position.get('current_stock_price')
        
        if stock_stop_loss and current_stock_price and current_stock_price <= stock_stop_loss:
            return True, f"Stock hit stop loss at {stock_stop_loss:.2f}"
        
        # Check rapid stock price increase (risk of assignment)
        entry_stock_price = position.get('stock_price', 0)
        strike_price = position.get('strike', 0)
        
        if all([entry_stock_price > 0, current_stock_price > 0, strike_price > 0]):
            # If stock price is significantly above strike, consider closing to avoid assignment
            if current_stock_price > strike_price * 1.02:  # 2% above strike
                deep_itm_exit_threshold = self.parameters.get('deep_itm_exit_threshold', 0.02)
                itm_pct = (current_stock_price - strike_price) / strike_price
                
                if itm_pct > deep_itm_exit_threshold:
                    return True, f"Call deep ITM ({itm_pct:.1%} above strike), high assignment risk"
        
        # Check for upcoming ex-dividend date
        if self.parameters.get('close_on_ex_dividend_day', True):
            ex_div_date = self.dividend_calendar.get(symbol)
            if ex_div_date:
                days_to_ex_div = (ex_div_date - today).days
                
                # If ex-dividend is within 3 days and call is in the money
                if days_to_ex_div <= 3 and current_stock_price > position.get('strike', 0):
                    return True, f"Close before ex-dividend ({days_to_ex_div} days away) to avoid assignment"
        
        # Check for upcoming earnings (if enabled)
        if market_data and self.parameters.get('exit_before_earnings', True):
            days_to_earnings = self._get_days_to_earnings(symbol, market_data)
            earnings_exit_days = self.parameters.get('earnings_exit_days', 5)
            
            if days_to_earnings is not None and days_to_earnings <= earnings_exit_days:
                return True, f"Upcoming earnings in {days_to_earnings} days"
        
        # Check sentiment change
        if market_data:
            sentiment_exit, sentiment_reason = self._check_sentiment_exit(symbol, market_data)
            if sentiment_exit:
                return True, sentiment_reason
        
        # Check option Greeks if available
        if option_chains:
            greek_exit, greek_reason = self._check_greek_thresholds(position, option_chains)
            if greek_exit:
                return True, greek_reason
        
        # Check risk-adjusted metrics
        if market_data:
            metrics_exit, metrics_reason = self._check_risk_adjusted_metrics(position, market_data)
            if metrics_exit:
                return True, metrics_reason
        
        return False, ""
    
    def _should_roll_position(self, position: Dict[str, Any], market_data=None, option_chains=None) -> Tuple[bool, str]:
        """
        Determine if a covered call position should be rolled to a new expiration.
        
        Args:
            position: Current position data
            market_data: Optional market data for additional analysis
            option_chains: Optional option chain data for evaluating roll opportunities
            
        Returns:
            Tuple of (should_roll: bool, reason: str)
        """
        # Extract position details
        expiration = position.get('expiration')
        
        # Parse expiration date
        if not expiration:
            return False, ""
            
        expiry_date = datetime.strptime(expiration, '%Y-%m-%d').date()
        today = date.today()
        dte_remaining = (expiry_date - today).days
        
        # Check DTE-based roll criteria
        roll_dte = self.parameters.get('roll_when_dte', 5)
        if dte_remaining <= roll_dte:
            return True, f"DTE threshold reached ({dte_remaining} days remaining)"
        
        # Check for theta decay acceleration point
        theta_roll_point = self.parameters.get('theta_acceleration_roll_dte', 21)
        if dte_remaining <= theta_roll_point and dte_remaining > roll_dte:
            # Only roll at theta acceleration if we have a favorable roll opportunity
            if option_chains:
                roll_credit = self._evaluate_roll_opportunity(position, option_chains)
                min_roll_credit = self.parameters.get('min_roll_credit', 0.002) * position.get('stock_price', 0)
                
                if roll_credit >= min_roll_credit:
                    return True, f"Favorable roll at theta acceleration point, credit: ${roll_credit:.2f}"
        
        return False, ""
    
    def _check_sentiment_exit(self, symbol: str, market_data) -> Tuple[bool, str]:
        """
        Check if market sentiment has changed enough to warrant exiting the position.
        
        Args:
            symbol: Stock symbol
            market_data: Market data containing sentiment information
            
        Returns:
            Tuple of (exit: bool, reason: str)
        """
        sentiment = None
        if hasattr(market_data, 'get_sentiment'):
            sentiment = market_data.get_sentiment(symbol)
        
        # If sentiment turned strongly bearish for a covered call (we own the underlying)
        if sentiment and sentiment.get('score', 0) < -0.6:
            # For covered calls, strongly negative sentiment is a danger to the underlying
            return True, f"Sentiment turned strongly bearish ({sentiment.get('score', 0):.2f})"
        
        # Check earnings announcements coming soon
        earnings_soon = False
        if hasattr(market_data, 'is_earnings_soon'):
            earnings_soon = market_data.is_earnings_soon(symbol, days=5)
            
        if earnings_soon and self.parameters.get('exit_before_earnings', True):
            return True, "Earnings announcement approaching"
            
        return False, ""
    
    def _get_days_to_earnings(self, symbol: str, market_data) -> Optional[int]:
        """
        Get the number of days until the next earnings announcement.
        
        Args:
            symbol: Stock symbol
            market_data: Market data containing earnings information
            
        Returns:
            Number of days to next earnings or None if not available
        """
        if hasattr(market_data, 'get_earnings_date'):
            earnings_date = market_data.get_earnings_date(symbol)
            if earnings_date:
                today = date.today()
                if isinstance(earnings_date, str):
                    earnings_date = datetime.strptime(earnings_date, '%Y-%m-%d').date()
                return (earnings_date - today).days
        return None
    
    def _check_greek_thresholds(self, position: Dict[str, Any], option_chains) -> Tuple[bool, str]:
        """
        Check if option Greeks have crossed exit thresholds.
        
        Args:
            position: Current position data
            option_chains: Option chain data for Greeks analysis
            
        Returns:
            Tuple of (exit: bool, reason: str)
        """
        symbol = position.get('symbol')
        option_symbol = position.get('option_symbol')
        
        if not option_symbol or not option_chains or not hasattr(option_chains, 'get_option_greeks'):
            return False, ""
        
        # Get current Greeks
        greeks = option_chains.get_option_greeks(option_symbol)
        if not greeks:
            return False, ""
        
        # Check delta
        delta = greeks.get('delta', 0)
        max_delta = self.parameters.get('max_delta_threshold', -0.85)  # Negative for short call
        if delta < max_delta:  # More negative than threshold
            return True, f"Delta threshold exceeded ({delta:.2f})"
        
        # Check theta decay (for covered calls, we actually want high theta)
        theta = greeks.get('theta', 0)
        min_theta = self.parameters.get('min_theta_value', 0.05)
        if abs(theta) < min_theta:  # Theta too small (not decaying enough)
            return True, f"Theta decay too slow ({theta:.2f})"
        
        # Check implied volatility
        iv = greeks.get('iv', 0)
        entry_iv = position.get('entry_iv', 0)
        
        if entry_iv > 0 and iv > 0:
            iv_change_pct = (iv - entry_iv) / entry_iv
            max_iv_increase = self.parameters.get('max_iv_increase_pct', 0.30)  # 30% increase
            
            if iv_change_pct > max_iv_increase:
                return True, f"IV increased significantly ({iv_change_pct:.1%})"
        
        return False, ""
    
    def _check_risk_adjusted_metrics(self, position: Dict[str, Any], market_data) -> Tuple[bool, str]:
        """
        Check if risk-adjusted metrics have deteriorated to exit thresholds.
        
        Args:
            position: Current position data
            market_data: Market data for performance analysis
            
        Returns:
            Tuple of (exit: bool, reason: str)
        """
        # If we don't have a way to get position performance, skip this check
        if not hasattr(self, '_get_position_performance'):
            return False, ""
        
        # Get historical performance of this position
        performance = self._get_position_performance(position)
        
        # Calculate Sharpe ratio (if we have enough data points)
        if len(performance) >= 20:  # Need enough data for meaningful calculation
            returns = [p.get('return', 0) for p in performance]
            sharpe = self._calculate_sharpe(returns)
            
            if sharpe < self.parameters.get('min_sharpe_ratio', 0.5):
                return True, f"Sharpe ratio below threshold ({sharpe:.2f})"
        
        # Calculate max drawdown
        if performance:
            drawdown = self._calculate_max_drawdown(performance)
            if drawdown > self.parameters.get('max_drawdown_percent', 0.10):
                return True, f"Maximum drawdown exceeded ({drawdown:.1%})"
        
        return False, ""
    
    def _calculate_sharpe(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """
        Calculate the Sharpe ratio for a list of returns.
        
        Args:
            returns: List of returns
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Sharpe ratio
        """
        if not returns:
            return 0.0
            
        mean_return = sum(returns) / len(returns)
        excess_returns = [r - (risk_free_rate / 252) for r in returns]  # Daily risk-free rate
        std_dev = np.std(excess_returns) if len(excess_returns) > 1 else 0.001
        
        if std_dev == 0:
            return 0.0
            
        sharpe = (mean_return * 252) / (std_dev * np.sqrt(252))  # Annualized
        return sharpe
    
    def _calculate_max_drawdown(self, performance: List[Dict[str, Any]]) -> float:
        """
        Calculate the maximum drawdown from a list of performance data.
        
        Args:
            performance: List of performance data dictionaries
            
        Returns:
            Maximum drawdown as a percentage
        """
        if not performance:
            return 0.0
            
        values = [p.get('value', 0) for p in performance]
        max_dd = 0.0
        peak = values[0]
        
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
            
        return max_dd
    
    def _evaluate_roll_opportunity(self, position: Dict[str, Any], option_chains) -> float:
        """
        Evaluate the potential credit from rolling a covered call position.
        
        Args:
            position: Current position data
            option_chains: Option chain data
            
        Returns:
            Net credit from rolling the position
        """
        symbol = position.get('symbol')
        current_strike = position.get('strike', 0)
        current_expiration = position.get('expiration')
        
        if not symbol or not current_strike or not current_expiration or not option_chains:
            return 0.0
            
        # Get current option value
        option_symbol = position.get('option_symbol')
        current_price = 0.0
        
        if option_symbol and hasattr(option_chains, 'get_option_price'):
            current_price = option_chains.get_option_price(option_symbol)
        else:
            current_price = position.get('current_option_price', 0)
        
        # Find new expiration
        if not hasattr(option_chains, 'get_expirations'):
            return 0.0
            
        expirations = option_chains.get_expirations(symbol)
        if not expirations:
            return 0.0
            
        # Convert current expiration to date
        current_exp_date = datetime.strptime(current_expiration, '%Y-%m-%d').date()
        
        # Find next monthly expiration (30-45 days out)
        target_dte = self.parameters.get('target_dte', 30)
        today = date.today()
        
        best_expiry = None
        best_dte_diff = float('inf')
        
        for exp in expirations:
            exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
            if exp_date <= current_exp_date:
                continue  # Skip expirations before or equal to current
                
            dte = (exp_date - today).days
            if dte >= target_dte:  # Only consider expirations at least target_dte days out
                dte_diff = abs(dte - target_dte)
                if dte_diff < best_dte_diff:
                    best_expiry = exp
                    best_dte_diff = dte_diff
        
        if not best_expiry:
            return 0.0
            
        # Find new call at same strike or slightly higher
        if not hasattr(option_chains, 'get_chain'):
            return 0.0
            
        chain = option_chains.get_chain(symbol, expiration=best_expiry)
        if not chain or 'calls' not in chain:
            return 0.0
            
        calls = chain['calls']
        new_strike = current_strike
        
        # Optionally move to a higher strike if stock price increased
        current_stock_price = position.get('current_stock_price', 0)
        if current_stock_price > current_strike * 1.05:  # Stock price up at least 5%
            # Find higher strike that's still OTM or slightly ITM
            for strike in sorted([call.get('strike', 0) for call in calls]):
                if strike >= current_stock_price:
                    new_strike = strike
                    break
        
        # Find the call option at the selected strike
        new_call = None
        for call in calls:
            if call.get('strike', 0) == new_strike:
                new_call = call
                break
                
        if not new_call:
            return 0.0
            
        # Calculate net credit
        new_price = new_call.get('bid', 0)  # Use bid price for selling
        net_credit = new_price - current_price
        
        return net_credit * 100  # Convert to dollar amount (per contract)

# Strategy registered via decorator - no need for explicit registration
