#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iron Condor Strategy Implementation

This strategy implements an iron condor options strategy following the standardized
template system, ensuring compatibility with the backtester and trading dashboard.

An iron condor is created by:
1. Selling an out-of-the-money put (short put)
2. Buying a further out-of-the-money put (long put)
3. Selling an out-of-the-money call (short call)
4. Buying a further out-of-the-money call (long call)

All options have the same expiration date.
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

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
    'description': 'Iron Condor strategy for income generation in neutral markets',
    'risk_level': 'medium',
    'typical_holding_period': '30-45 days'
})
class IronCondorStrategy(OptionsBaseStrategy):
    """
    Iron Condor Options Strategy
    
    This strategy involves selling an out-of-the-money put spread and an out-of-the-money call spread
    with the same expiration date. It's a market-neutral strategy that profits from low volatility
    and time decay when the underlying stays within a certain price range.
    
    Key characteristics:
    - Limited risk and limited reward
    - Maximum profit when underlying expires between short strikes
    - Maximum loss when underlying moves beyond long strikes
    - Benefits from time decay and decreasing volatility
    - Typically used in neutral or range-bound markets
    """
    
    # Default parameters for iron condor strategy
    DEFAULT_PARAMS = {
        'strategy_name': 'iron_condor',
        'strategy_version': '1.0.0',
        'asset_class': 'options',
        'strategy_type': 'income',
        'timeframe': 'swing',
        'market_regime': 'neutral',
        
        # Stock selection criteria
        'min_stock_price': 50.0,              # Minimum stock price to consider
        'max_stock_price': 500.0,             # Maximum stock price to consider
        'min_option_volume': 200,             # Minimum option volume
        'min_option_open_interest': 300,      # Minimum option open interest
        'min_iv_percentile': 40,              # Minimum IV percentile
        'max_iv_percentile': 70,              # Maximum IV percentile for entry
        
        # Technical analysis parameters
        'min_historical_days': 252,           # Days of historical data required
        'atr_period': 14,                     # Period for ATR calculation
        'bollinger_period': 20,               # Period for Bollinger Bands
        'bollinger_std': 2.0,                 # Standard deviations for Bollinger Bands
        
        # Option parameters
        'target_dte': 45,                     # Target days to expiration
        'min_dte': 30,                        # Minimum days to expiration
        'max_dte': 60,                        # Maximum days to expiration
        'call_spread_width': 5,               # Width of call spread (strike difference)
        'put_spread_width': 5,                # Width of put spread (strike difference)
        'short_call_delta': -0.16,            # Target delta for short call (negative)
        'short_put_delta': 0.16,              # Target delta for short put (positive)
        'min_wing_distance': 1.0,             # Minimum distance between short strikes (ATRs)
        
        # Credit and return parameters
        'min_credit_collected': 0.80,         # Minimum credit to collect
        'min_reward_risk_ratio': 0.25,        # Minimum credit to width ratio
        'max_buying_power_pct': 0.05,         # Maximum buying power reduction as % of account
        
        # Risk management parameters
        'profit_target_pct': 50,              # Take profit at % of max profit
        'stop_loss_pct': 200,                 # Stop loss at % of credit received
        'max_loss_pct': 15,                   # Maximum account loss per trade
        
        # Exit parameters
        'early_profit_days': 14,              # If 50% profit reached before this many days, take it
        'dte_exit_threshold': 21,             # Exit when DTE reaches this value regardless of P/L
        'gamma_risk_threshold': 0.05,         # Exit if gamma risk becomes too high
        'iv_exit_threshold': 20,              # Exit if IV percentile drops below this value
    }
    
    def __init__(self, 
                 strategy_id: str = None, 
                 name: str = None,
                 parameters: Dict[str, Any] = None):
        """
        Initialize the Iron Condor strategy.
        
        Args:
            strategy_id: Unique identifier for this strategy instance
            name: Human-readable name of the strategy
            parameters: Strategy-specific parameters that override the defaults
        """
        super().__init__(strategy_id, name, parameters)
        
        # Strategy-specific tracking
        self.iron_condor_positions = {}      # Track current iron condor positions
        self.position_performance = {}       # Track performance metrics for each position
        
    def define_universe(self, market_data: MarketData) -> Universe:
        """
        Define the universe of tradable assets for iron condor strategy.
        
        Args:
            market_data: Market data for analysis
            
        Returns:
            Universe object with selected symbols
        """
        universe = Universe()
        
        # Get all available symbols
        all_symbols = market_data.get_all_symbols()
        
        # Filter by price range
        min_price = self.parameters.get('min_stock_price', 50.0)
        max_price = self.parameters.get('max_stock_price', 500.0)
        
        filtered_symbols = []
        for symbol in all_symbols:
            quote = market_data.get_latest_quote(symbol)
            if not quote:
                continue
                
            price = quote.get('price', 0)
            if min_price <= price <= max_price:
                filtered_symbols.append(symbol)
        
        # Get historical data and calculate technical indicators
        lookback_days = self.parameters.get('min_historical_days', 252)
        
        for symbol in filtered_symbols:
            # Get historical data
            historical_data = market_data.get_historical_data(symbol, period=lookback_days)
            if not historical_data or len(historical_data) < lookback_days:
                continue
                
            # Convert to DataFrame if not already
            if not isinstance(historical_data, pd.DataFrame):
                historical_data = pd.DataFrame(historical_data)
            
            # Calculate Average True Range (ATR)
            atr_period = self.parameters.get('atr_period', 14)
            historical_data['tr1'] = abs(historical_data['high'] - historical_data['low'])
            historical_data['tr2'] = abs(historical_data['high'] - historical_data['close'].shift())
            historical_data['tr3'] = abs(historical_data['low'] - historical_data['close'].shift())
            historical_data['tr'] = historical_data[['tr1', 'tr2', 'tr3']].max(axis=1)
            historical_data['atr'] = historical_data['tr'].rolling(window=atr_period).mean()
            
            # Calculate Bollinger Bands
            bb_period = self.parameters.get('bollinger_period', 20)
            bb_std = self.parameters.get('bollinger_std', 2.0)
            historical_data['sma'] = historical_data['close'].rolling(window=bb_period).mean()
            historical_data['std'] = historical_data['close'].rolling(window=bb_period).std()
            historical_data['upper_band'] = historical_data['sma'] + (bb_std * historical_data['std'])
            historical_data['lower_band'] = historical_data['sma'] - (bb_std * historical_data['std'])
            
            # Calculate volatility percentile
            historical_data['daily_return'] = historical_data['close'].pct_change()
            rolling_vol = historical_data['daily_return'].rolling(window=20).std() * np.sqrt(252)  # Annualized
            current_vol = rolling_vol.iloc[-1]
            
            # Calculate historical volatility percentile (last 252 trading days)
            vol_percentile = percentileofscore(rolling_vol.dropna().values, current_vol)
            
            # Check if the stock is in a range-bound pattern
            # 1. Price is within Bollinger Bands
            price = historical_data['close'].iloc[-1]
            upper_band = historical_data['upper_band'].iloc[-1]
            lower_band = historical_data['lower_band'].iloc[-1]
            sma = historical_data['sma'].iloc[-1]
            
            price_within_bands = lower_band < price < upper_band
            
            # 2. Price is near the middle of the range (not trending)
            price_position = (price - lower_band) / (upper_band - lower_band)
            neutral_position = 0.4 <= price_position <= 0.6
            
            # 3. ATR is relatively low (low volatility)
            atr = historical_data['atr'].iloc[-1]
            atr_pct = atr / price
            low_atr = atr_pct < 0.02  # ATR less than 2% of price
            
            # Check IV percentile range if available
            iv_ok = True
            iv_percentile = market_data.get_iv_percentile(symbol) if hasattr(market_data, 'get_iv_percentile') else None
            
            if iv_percentile is not None:
                min_iv = self.parameters.get('min_iv_percentile', 40)
                max_iv = self.parameters.get('max_iv_percentile', 70)
                iv_ok = min_iv <= iv_percentile <= max_iv
            
            # Check option liquidity if available
            liquidity_ok = True
            if hasattr(market_data, 'get_option_volume'):
                option_volume = market_data.get_option_volume(symbol)
                option_oi = market_data.get_option_open_interest(symbol)
                
                min_volume = self.parameters.get('min_option_volume', 200)
                min_oi = self.parameters.get('min_option_open_interest', 300)
                
                liquidity_ok = option_volume >= min_volume and option_oi >= min_oi
            
            # Add to universe if criteria met
            if price_within_bands and (neutral_position or low_atr) and iv_ok and liquidity_ok:
                universe.add_symbol(symbol)
        
        return universe
    
    def generate_signals(self, market_data: MarketData, 
                        option_chains: Optional[OptionChains] = None) -> List[Dict[str, Any]]:
        """
        Generate iron condor signals based on market data and option chains.
        
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
            # Get option chain for this symbol
            if not option_chains or not option_chains.has_symbol(symbol):
                continue
                
            chain = option_chains.get_chain(symbol)
            if not chain:
                continue
                
            # Find appropriate options for iron condor
            iron_condor = self._find_iron_condor(symbol, chain, market_data)
            if not iron_condor:
                continue
                
            # Create signal
            signal = self._create_iron_condor_signal(symbol, iron_condor, market_data)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _find_iron_condor(self, symbol: str, chain: Dict[str, Any], 
                        market_data: MarketData) -> Optional[Dict[str, Any]]:
        """
        Find appropriate options for an iron condor spread.
        
        Args:
            symbol: Stock symbol
            chain: Option chain data
            market_data: Market data
            
        Returns:
            Dictionary with iron condor details if found, None otherwise
        """
        # Get stock price
        quote = market_data.get_latest_quote(symbol)
        if not quote:
            return None
            
        stock_price = quote.get('price', 0)
        if stock_price <= 0:
            return None
            
        # Get parameters
        target_dte = self.parameters.get('target_dte', 45)
        min_dte = self.parameters.get('min_dte', 30)
        max_dte = self.parameters.get('max_dte', 60)
        call_spread_width = self.parameters.get('call_spread_width', 5)
        put_spread_width = self.parameters.get('put_spread_width', 5)
        short_call_delta = self.parameters.get('short_call_delta', -0.16)  # Negative
        short_put_delta = self.parameters.get('short_put_delta', 0.16)   # Positive
        
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
            
        # Get calls and puts for the selected expiration
        calls = [opt for opt in chain.get('calls', []) if opt.get('expiration') == best_expiry]
        puts = [opt for opt in chain.get('puts', []) if opt.get('expiration') == best_expiry]
        
        if not calls or not puts:
            return None
            
        # Filter for minimum volume and open interest
        min_volume = self.parameters.get('min_option_volume', 200)
        min_oi = self.parameters.get('min_option_open_interest', 300)
        
        calls = [opt for opt in calls if 
                opt.get('volume', 0) >= min_volume and 
                opt.get('open_interest', 0) >= min_oi]
        
        puts = [opt for opt in puts if 
               opt.get('volume', 0) >= min_volume and 
               opt.get('open_interest', 0) >= min_oi]
        
        if not calls or not puts:
            return None
            
        # Sort by strike
        calls.sort(key=lambda x: x.get('strike', 0))
        puts.sort(key=lambda x: x.get('strike', 0))
        
        # Find appropriate strikes based on delta
        otm_calls = [call for call in calls if call.get('strike', 0) > stock_price]
        otm_puts = [put for put in puts if put.get('strike', 0) < stock_price]
        
        if not otm_calls or not otm_puts:
            return None
            
        # Find short call with delta closest to target delta
        short_call = min(otm_calls, key=lambda x: abs(x.get('delta', 0) - short_call_delta))
        short_call_strike = short_call.get('strike', 0)
        
        # Find short put with delta closest to target delta
        short_put = min(otm_puts, key=lambda x: abs(x.get('delta', 0) - short_put_delta))
        short_put_strike = short_put.get('strike', 0)
        
        # Check wing distance (ensure shorts aren't too close together)
        min_wing_distance_atr = self.parameters.get('min_wing_distance', 1.0)
        
        # Get ATR if available
        atr = 0
        if hasattr(market_data, 'get_atr'):
            atr = market_data.get_atr(symbol, period=14)
        else:
            # Rough estimate - 2% of stock price
            atr = stock_price * 0.02
            
        min_wing_distance = atr * min_wing_distance_atr
        if (short_call_strike - short_put_strike) < min_wing_distance:
            return None
            
        # Find long call (higher strike)
        long_call_strike = short_call_strike + call_spread_width
        long_call = None
        
        for call in calls:
            if call.get('strike', 0) == long_call_strike:
                long_call = call
                break
                
        if not long_call:
            return None
            
        # Find long put (lower strike)
        long_put_strike = short_put_strike - put_spread_width
        long_put = None
        
        for put in puts:
            if put.get('strike', 0) == long_put_strike:
                long_put = put
                break
                
        if not long_put:
            return None
            
        # Calculate spread details
        short_call_price = short_call.get('bid', 0)  # Use bid price when selling
        long_call_price = long_call.get('ask', 0)   # Use ask price when buying
        short_put_price = short_put.get('bid', 0)   # Use bid price when selling
        long_put_price = long_put.get('ask', 0)     # Use ask price when buying
        
        # Calculate credit received
        call_spread_credit = short_call_price - long_call_price
        put_spread_credit = short_put_price - long_put_price
        total_credit = call_spread_credit + put_spread_credit
        
        # Check minimum credit
        min_credit = self.parameters.get('min_credit_collected', 0.80)
        if total_credit < min_credit:
            return None
            
        # Calculate max profit and loss
        max_profit = total_credit * 100  # Credit received (per contract)
        call_spread_width_dollars = (long_call_strike - short_call_strike) * 100
        put_spread_width_dollars = (short_put_strike - long_put_strike) * 100
        max_loss_call_side = call_spread_width_dollars - (call_spread_credit * 100)
        max_loss_put_side = put_spread_width_dollars - (put_spread_credit * 100)
        max_loss = max(max_loss_call_side, max_loss_put_side)
        
        # Calculate reward to risk ratio
        reward_risk_ratio = max_profit / max_loss if max_loss > 0 else 0
        min_reward_risk = self.parameters.get('min_reward_risk_ratio', 0.25)
        
        if reward_risk_ratio < min_reward_risk:
            return None
            
        # Calculate probability of profit
        pop = None
        if all([short_call.get('delta'), short_put.get('delta')]):
            # Approximate POP using delta
            pop = (1 - abs(short_call.get('delta', 0))) + (1 - abs(short_put.get('delta', 0))) - 1
            pop = min(max(pop, 0), 1)  # Bound between 0 and 1
        
        return {
            'symbol': symbol,
            'stock_price': stock_price,
            'expiration': best_expiry,
            'dte': best_dte,
            'short_call': short_call,
            'long_call': long_call,
            'short_put': short_put,
            'long_put': long_put,
            'short_call_strike': short_call_strike,
            'long_call_strike': long_call_strike,
            'short_put_strike': short_put_strike,
            'long_put_strike': long_put_strike,
            'call_spread_width': call_spread_width,
            'put_spread_width': put_spread_width,
            'call_spread_credit': call_spread_credit,
            'put_spread_credit': put_spread_credit,
            'total_credit': total_credit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'reward_risk_ratio': reward_risk_ratio,
            'probability_of_profit': pop,
            'atr': atr,
        }
    
    def _create_iron_condor_signal(self, symbol: str, iron_condor: Dict[str, Any], 
                                 market_data: MarketData) -> Optional[Dict[str, Any]]:
        """
        Create a signal for an iron condor trade.
        
        Args:
            symbol: Stock symbol
            iron_condor: Iron condor details
            market_data: Market data
            
        Returns:
            Signal dictionary
        """
        if not iron_condor:
            return None
            
        # Extract iron condor details
        stock_price = iron_condor.get('stock_price', 0)
        expiration = iron_condor.get('expiration')
        dte = iron_condor.get('dte', 0)
        short_call_strike = iron_condor.get('short_call_strike', 0)
        long_call_strike = iron_condor.get('long_call_strike', 0)
        short_put_strike = iron_condor.get('short_put_strike', 0)
        long_put_strike = iron_condor.get('long_put_strike', 0)
        total_credit = iron_condor.get('total_credit', 0)
        max_profit = iron_condor.get('max_profit', 0)
        max_loss = iron_condor.get('max_loss', 0)
        reward_risk_ratio = iron_condor.get('reward_risk_ratio', 0)
        pop = iron_condor.get('probability_of_profit', 0)
        
        # Check if we already have an active iron condor for this symbol
        if symbol in self.iron_condor_positions:
            return None
            
        # Create base options signal
        signal = self.create_options_signal(
            symbol=symbol,
            action="IRON_CONDOR",
            option_type="SPREAD",
            strike=short_call_strike,  # Using short call strike as reference
            expiration=expiration,
            premium=total_credit,
            stock_price=stock_price,
            reason=f"Iron condor with {total_credit:.2f} credit and {pop:.1%} probability of profit",
            strength=min(1.0, reward_risk_ratio * 2.0),  # Normalized strength based on R/R
            stop_loss=total_credit * self.parameters.get('stop_loss_pct', 200) / 100, 
            take_profit=total_credit * self.parameters.get('profit_target_pct', 50) / 100
        )
        
        # Add iron condor-specific details
        signal.update({
            'spread_type': 'iron_condor',
            'short_call_strike': short_call_strike,
            'long_call_strike': long_call_strike,
            'short_put_strike': short_put_strike,
            'long_put_strike': long_put_strike,
            'call_wing_width': long_call_strike - short_call_strike,
            'put_wing_width': short_put_strike - long_put_strike,
            'short_call_delta': iron_condor.get('short_call', {}).get('delta', 0),
            'short_put_delta': iron_condor.get('short_put', {}).get('delta', 0),
            'max_profit': max_profit,
            'max_loss': max_loss,
            'reward_risk_ratio': reward_risk_ratio,
            'probability_of_profit': pop,
            'dte': dte,
            'short_call_symbol': iron_condor.get('short_call', {}).get('symbol'),
            'long_call_symbol': iron_condor.get('long_call', {}).get('symbol'),
            'short_put_symbol': iron_condor.get('short_put', {}).get('symbol'),
            'long_put_symbol': iron_condor.get('long_put', {}).get('symbol'),
        })
        
        return signal
    
    def position_sizing(self, signal: Dict[str, Any], account_info: Dict[str, Any]) -> float:
        """
        Calculate position size for an iron condor spread.
        
        Args:
            signal: Trading signal dictionary
            account_info: Account information including equity, margin, etc.
            
        Returns:
            Number of spreads to trade
        """
        account_value = account_info.get('equity', 0)
        if account_value <= 0:
            return 0
            
        # Extract parameters
        max_position_pct = self.parameters.get('max_buying_power_pct', 0.05)
        max_loss_pct = self.parameters.get('max_loss_pct', 0.15) / 100
        
        # Calculate maximum dollar amount to risk
        max_risk_amount = account_value * max_loss_pct
        
        # Calculate risk per spread
        max_loss = signal.get('max_loss', 0)
        if max_loss <= 0:
            return 0
            
        # Calculate position size based on risk
        risk_based_size = int(max_risk_amount / max_loss)
        
        # Calculate position size based on buying power
        buying_power = account_info.get('buying_power', account_value)
        max_buying_power = buying_power * max_position_pct
        
        # For iron condors, the buying power reduction is typically the width of the wider wing minus credit
        call_wing_width = signal.get('call_wing_width', 0) * 100  # Convert to dollars
        put_wing_width = signal.get('put_wing_width', 0) * 100
        wider_wing = max(call_wing_width, put_wing_width)
        premium = signal.get('premium', 0) * 100
        
        buying_power_reduction = wider_wing - premium
        if buying_power_reduction <= 0:
            buying_power_reduction = max_loss  # Fallback
            
        buying_power_based_size = int(max_buying_power / buying_power_reduction)
        
        # Take the smaller of the two calculations
        position_size = min(risk_based_size, buying_power_based_size)
        
        # Minimum of 1 contract
        return max(1, position_size)
        
    def on_exit_signal(self, position: Dict[str, Any], market_data=None, option_chains=None) -> Optional[Dict[str, Any]]:
        """
        Generate an exit signal for an iron condor position using comprehensive exit criteria.
        
        Args:
            position: Current position data
            market_data: Optional market data for additional analysis
            option_chains: Optional option chain data for Greeks and pricing
            
        Returns:
            Exit signal if conditions are met, None otherwise
        """
        # Extract position details
        symbol = position.get('symbol')
        entry_time = position.get('timestamp')
        current_time = datetime.now()
        expiration = position.get('expiration')
        
        # Exit if required data is missing
        if not symbol or not entry_time or not expiration:
            return None
            
        # Check comprehensive exit criteria
        should_exit, reason = self._should_exit_position(position, market_data, option_chains)
        if should_exit:
            return self.create_signal(
                symbol=symbol,
                action="CLOSE_POSITION",
                reason=reason,
                strength=1.0
            )
        
        # No exit signal needed
        return None
        
    def _should_exit_position(self, position: Dict[str, Any], market_data=None, option_chains=None) -> Tuple[bool, str]:
        """
        Comprehensive evaluation of whether an iron condor position should be exited based on multiple criteria.
        
        Args:
            position: Current position data
            market_data: Optional market data for additional analysis
            option_chains: Optional option chain data for Greeks analysis
            
        Returns:
            Tuple of (should_exit: bool, reason: str)
        """
        symbol = position.get('symbol')
        expiration = position.get('expiration')
        entry_time = position.get('timestamp')
        current_time = datetime.now()
        
        # Parse expiration date
        if not expiration:
            return False, ""
            
        expiry_date = datetime.strptime(expiration, '%Y-%m-%d').date()
        today = date.today()
        dte_remaining = (expiry_date - today).days
        
        # Check DTE-based exit
        dte_exit = self.parameters.get('dte_exit_threshold', 21)
        if dte_remaining <= dte_exit:
            return True, f"DTE threshold reached ({dte_remaining} days remaining)"
        
        # Check profit target
        profit_target_pct = self.parameters.get('profit_target_pct', 50) / 100
        original_credit = position.get('premium', 0) * 100  # Convert to dollars
        current_value = position.get('current_value', 0) * 100
        
        if original_credit > 0 and current_value >= 0:
            profit = original_credit - current_value
            profit_pct = profit / original_credit
            
            # Check early profit threshold
            days_held = (current_time - entry_time).days
            early_profit_days = self.parameters.get('early_profit_days', 14)
            
            if profit_pct >= profit_target_pct:
                if days_held < early_profit_days:
                    return True, f"Early profit target reached ({profit_pct:.1%} in {days_held} days)"
                else:
                    return True, f"Profit target reached ({profit_pct:.1%})"
        
        # Check stop loss
        stop_loss_pct = self.parameters.get('stop_loss_pct', 200) / 100
        if original_credit > 0 and current_value > 0:
            loss = current_value - original_credit
            if loss > 0:  # We have a loss
                loss_pct = loss / original_credit
                if loss_pct >= stop_loss_pct:
                    return True, f"Stop loss triggered ({loss_pct:.1%} of credit)"
        
        # Check stock price approaching short strikes
        current_stock_price = position.get('current_stock_price', 0)
        short_call_strike = position.get('short_call_strike', 0)
        short_put_strike = position.get('short_put_strike', 0)
        
        if all([current_stock_price > 0, short_call_strike > 0, short_put_strike > 0]):
            # Calculate how close price is to short strikes
            call_distance_pct = (short_call_strike - current_stock_price) / current_stock_price
            put_distance_pct = (current_stock_price - short_put_strike) / current_stock_price
            
            # If price is within 2% of either short strike, consider exiting
            danger_threshold = 0.02  # 2%
            if call_distance_pct < danger_threshold:
                return True, f"Stock price approaching short call strike ({call_distance_pct:.1%} away)"
                
            if put_distance_pct < danger_threshold:
                return True, f"Stock price approaching short put strike ({put_distance_pct:.1%} away)"
        
        # Check IV changes
        if option_chains and hasattr(market_data, 'get_iv_percentile'):
            current_iv_percentile = market_data.get_iv_percentile(symbol)
            iv_exit_threshold = self.parameters.get('iv_exit_threshold', 20)
            
            if current_iv_percentile is not None and current_iv_percentile < iv_exit_threshold:
                return True, f"IV percentile dropped below threshold ({current_iv_percentile} < {iv_exit_threshold})"
        
        # Check sentiment change
        if market_data:
            sentiment_exit, sentiment_reason = self._check_sentiment_exit(symbol, market_data)
            if sentiment_exit:
                return True, sentiment_reason
        
        # Check gamma risk if available
        if option_chains:
            gamma_exit, gamma_reason = self._check_gamma_risk(position, option_chains)
            if gamma_exit:
                return True, gamma_reason
        
        # Check probability of profit changes
        if option_chains:
            pop_exit, pop_reason = self._check_pop_changes(position, option_chains)
            if pop_exit:
                return True, pop_reason
        
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
        # Get current sentiment if available
        sentiment = None
        if hasattr(market_data, 'get_sentiment'):
            sentiment = market_data.get_sentiment(symbol)
        
        # For iron condors, both strong bullish or bearish sentiment is a warning
        if sentiment:
            score = sentiment.get('score', 0)
            if abs(score) > 0.7:  # Strong directional bias (either way)
                direction = "bullish" if score > 0 else "bearish"
                return True, f"Strong {direction} sentiment detected ({score:.2f})"
        
        # Check for upcoming earnings or significant events
        has_upcoming_event = False
        if hasattr(market_data, 'has_upcoming_event'):
            has_upcoming_event = market_data.has_upcoming_event(symbol, days=7)
            
        if has_upcoming_event:
            return True, "Upcoming significant event detected"
            
        return False, ""
    
    def _check_gamma_risk(self, position: Dict[str, Any], option_chains) -> Tuple[bool, str]:
        """
        Check if gamma risk has increased beyond acceptable levels.
        
        Args:
            position: Position details
            option_chains: Option chain data
            
        Returns:
            Tuple of (exit: bool, reason: str)
        """
        # Extract option symbols
        short_call_symbol = position.get('short_call_symbol')
        short_put_symbol = position.get('short_put_symbol')
        
        if not short_call_symbol or not short_put_symbol or not hasattr(option_chains, 'get_option_greeks'):
            return False, ""
        
        # Get current Greeks
        short_call_greeks = option_chains.get_option_greeks(short_call_symbol)
        short_put_greeks = option_chains.get_option_greeks(short_put_symbol)
        
        if not short_call_greeks or not short_put_greeks:
            return False, ""
        
        # Check gamma (we want low gamma for iron condors)
        short_call_gamma = abs(short_call_greeks.get('gamma', 0))
        short_put_gamma = abs(short_put_greeks.get('gamma', 0))
        
        gamma_threshold = self.parameters.get('gamma_risk_threshold', 0.05)
        max_gamma = max(short_call_gamma, short_put_gamma)
        
        if max_gamma > gamma_threshold:
            return True, f"Gamma risk too high ({max_gamma:.4f})"
            
        return False, ""
    
    def _check_pop_changes(self, position: Dict[str, Any], option_chains) -> Tuple[bool, str]:
        """
        Check if probability of profit has significantly decreased.
        
        Args:
            position: Position details
            option_chains: Option chain data
            
        Returns:
            Tuple of (exit: bool, reason: str)
        """
        # Get original probability of profit
        original_pop = position.get('probability_of_profit', 0)
        if original_pop <= 0:
            return False, ""
        
        # Calculate current probability of profit
        short_call_symbol = position.get('short_call_symbol')
        short_put_symbol = position.get('short_put_symbol')
        
        if not short_call_symbol or not short_put_symbol or not hasattr(option_chains, 'get_option_greeks'):
            return False, ""
        
        short_call_greeks = option_chains.get_option_greeks(short_call_symbol)
        short_put_greeks = option_chains.get_option_greeks(short_put_symbol)
        
        if not short_call_greeks or not short_put_greeks:
            return False, ""
        
        # Calculate current POP using delta approximation
        short_call_delta = abs(short_call_greeks.get('delta', 0))
        short_put_delta = abs(short_put_greeks.get('delta', 0))
        
        if short_call_delta > 0 and short_put_delta > 0:
            current_pop = (1 - short_call_delta) + (1 - short_put_delta) - 1
            current_pop = min(max(current_pop, 0), 1)  # Bound between 0 and 1
            
            # Check for significant POP decrease
            pop_decrease_threshold = 0.15  # 15 percentage points
            if (original_pop - current_pop) > pop_decrease_threshold:
                return True, f"Probability of profit decreased significantly ({current_pop:.1%} from {original_pop:.1%})"
                
        return False, ""
