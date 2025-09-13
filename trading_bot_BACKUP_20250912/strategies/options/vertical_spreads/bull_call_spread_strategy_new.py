#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bull Call Spread Strategy Implementation

This strategy implements a bull call spread options strategy following the standardized
template system, ensuring compatibility with the backtester and trading dashboard.

A bull call spread is created by:
1. Buying a call option at a lower strike price
2. Selling a call option at a higher strike price
3. Using the same expiration date for both options
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
    'strategy_type': StrategyType.MOMENTUM.value,
    'timeframe': TimeFrame.SWING.value,
    'compatible_market_regimes': [MarketRegime.TRENDING.value],
    'description': 'Bull Call Spread options strategy for moderately bullish market movement',
    'risk_level': 'medium',
    'typical_holding_period': '30-45 days'
})
class BullCallSpreadStrategy(OptionsBaseStrategy):
    """
    Bull Call Spread Options Strategy
    
    This strategy involves buying a call option at a lower strike price and selling a call option
    at a higher strike price with the same expiration date. This creates a debit spread that
    profits from moderately bullish movements while capping both the maximum profit and loss.
    
    Key characteristics:
    - Limited risk (max loss = net premium paid)
    - Limited profit (max profit = difference between strikes - net premium paid)
    - Requires less capital than buying calls outright
    - Benefits from moderately bullish price movement
    - Mitigates time decay impact compared to single calls
    - Breakeven point is at lower strike plus net debit
    - Maximum profit achieved when price rises above higher strike at expiration
    """
    
    # Default parameters for bull call spread strategy
    DEFAULT_PARAMS = {
        'strategy_name': 'bull_call_spread',
        'strategy_version': '2.0.0',
        'asset_class': 'options',
        'strategy_type': 'directional',
        'timeframe': 'swing',
        'market_regime': 'trending',
        
        # Stock selection criteria
        'min_stock_price': 20.0,              # Minimum stock price to consider
        'max_stock_price': 500.0,             # Maximum stock price to consider
        'min_option_volume': 50,              # Minimum option volume
        'min_option_open_interest': 100,      # Minimum option open interest
        'min_iv_percentile': 30,              # Minimum IV percentile
        'max_iv_percentile': 60,              # Maximum IV percentile for entry
        
        # Technical analysis parameters
        'min_historical_days': 252,           # Days of historical data required
        'min_up_trend_days': 5,               # Days the stock should be in an uptrend
        'trend_indicator': 'ema_20_50',       # Indicator to determine trend
        
        # Option parameters
        'target_dte': 45,                     # Target days to expiration
        'min_dte': 30,                        # Minimum days to expiration
        'max_dte': 60,                        # Maximum days to expiration
        'spread_width': 5,                    # Target width between strikes
        'strike_selection_method': 'delta',   # 'delta', 'otm_percentage', or 'price_range'
        'long_call_delta': 0.70,              # Target delta for long call
        'short_call_delta': 0.30,             # Target delta for short call
        'otm_percentage': 0.05,               # Alternative: % OTM for long call
        'short_call_otm_extra': 0.05,         # Extra % OTM for short call
        
        # Risk management parameters
        'max_position_size_percent': 0.05,    # Maximum position size as % of portfolio
        'max_num_positions': 10,              # Maximum number of positions
        'max_risk_per_trade': 0.02,           # Maximum risk per trade as % of portfolio
        'take_profit_pct': 0.50,              # Take profit at % of max profit
        'stop_loss_percent': 0.50,            # Stop loss at % of max loss
        
        # Exit parameters
        'dte_exit_threshold': 21,             # Exit when DTE reaches this value
        'profit_target_percent': 50,          # Exit at this percentage of max profit
        'loss_limit_percent': 75,             # Exit at this percentage of max loss
        'ema_cross_exit': True,               # Exit on bearish EMA cross
    }
    
    def __init__(self, 
                 strategy_id: str = None, 
                 name: str = None,
                 parameters: Dict[str, Any] = None):
        """
        Initialize the Bull Call Spread strategy.
        
        Args:
            strategy_id: Unique identifier for this strategy instance
            name: Human-readable name of the strategy
            parameters: Strategy-specific parameters that override the defaults
        """
        super().__init__(strategy_id, name, parameters)
        
        # Strategy-specific tracking
        self.spread_trades = {}          # Track current spread positions
        self.spread_performance = {}     # Track performance of spread trades
        
    def generate_signals(self, market_data: MarketData, 
                        option_chains: Optional[OptionChains] = None) -> List[Dict[str, Any]]:
        """
        Generate bull call spread signals based on market data and option chains.
        
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
            # Check if this stock meets our trend criteria
            if not self._is_bullish_trend(symbol, market_data):
                continue
                
            # Get option chain for this symbol
            if not option_chains or not option_chains.has_symbol(symbol):
                continue
                
            chain = option_chains.get_chain(symbol)
            if not chain:
                continue
                
            # Find appropriate options for bull call spread
            spread = self._find_bull_call_spread(symbol, chain, market_data)
            if not spread:
                continue
                
            # Create signal
            signal = self._create_bull_call_spread_signal(symbol, spread, market_data)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _is_bullish_trend(self, symbol: str, market_data: MarketData) -> bool:
        """
        Check if a stock is in a bullish trend based on technical indicators.
        
        Args:
            symbol: Stock symbol
            market_data: Market data
            
        Returns:
            True if stock is in a bullish trend, False otherwise
        """
        # Get historical data
        historical_data = market_data.get_historical_data(
            symbol, 
            period=self.parameters.get('min_historical_days', 252)
        )
        
        if historical_data is None or len(historical_data) < 50:
            return False
            
        # Convert to DataFrame if it's not already
        if not isinstance(historical_data, pd.DataFrame):
            historical_data = pd.DataFrame(historical_data)
        
        # Use the specified trend indicator
        trend_indicator = self.parameters.get('trend_indicator', 'ema_20_50')
        
        if trend_indicator == 'ema_20_50':
            # Calculate EMAs
            historical_data['ema_20'] = historical_data['close'].ewm(span=20).mean()
            historical_data['ema_50'] = historical_data['close'].ewm(span=50).mean()
            
            # Check for bullish alignment
            last_n_days = self.parameters.get('min_up_trend_days', 5)
            is_bullish = (
                historical_data['ema_20'].iloc[-1] > historical_data['ema_50'].iloc[-1] and
                all(historical_data['ema_20'].iloc[-i] > historical_data['ema_50'].iloc[-i] 
                   for i in range(1, last_n_days + 1))
            )
            return is_bullish
            
        elif trend_indicator == 'price_above_sma':
            # Calculate SMA
            sma_period = self.parameters.get('sma_period', 50)
            historical_data['sma'] = historical_data['close'].rolling(window=sma_period).mean()
            
            # Check if price is above SMA
            last_n_days = self.parameters.get('min_up_trend_days', 5)
            is_bullish = all(historical_data['close'].iloc[-i] > historical_data['sma'].iloc[-i] 
                           for i in range(1, last_n_days + 1))
            return is_bullish
            
        elif trend_indicator == 'higher_highs_lows':
            # Check for higher highs and higher lows
            last_n_days = self.parameters.get('min_up_trend_days', 5)
            highs = historical_data['high'].iloc[-last_n_days:].values
            lows = historical_data['low'].iloc[-last_n_days:].values
            
            # Simple check for consecutive higher highs and higher lows
            is_bullish = all(highs[i] >= highs[i-1] for i in range(1, len(highs))) and \
                         all(lows[i] >= lows[i-1] for i in range(1, len(lows)))
            return is_bullish
            
        # Default to False if indicator is not recognized
        return False
    
    def _find_bull_call_spread(self, symbol: str, chain: Dict[str, Any], 
                             market_data: MarketData) -> Optional[Dict[str, Any]]:
        """
        Find appropriate options for a bull call spread.
        
        Args:
            symbol: Stock symbol
            chain: Option chain data
            market_data: Market data
            
        Returns:
            Dictionary with spread details if found, None otherwise
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
        spread_width = self.parameters.get('spread_width', 5)
        strike_method = self.parameters.get('strike_selection_method', 'delta')
        
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
        min_volume = self.parameters.get('min_option_volume', 50)
        min_oi = self.parameters.get('min_option_open_interest', 100)
        
        calls = [opt for opt in calls if 
                 opt.get('volume', 0) >= min_volume and 
                 opt.get('open_interest', 0) >= min_oi]
        
        if not calls:
            return None
            
        # Sort by strike
        calls.sort(key=lambda x: x.get('strike', 0))
        
        # Find appropriate strikes based on selection method
        long_call = None
        short_call = None
        
        if strike_method == 'delta':
            target_long_delta = self.parameters.get('long_call_delta', 0.70)
            target_short_delta = self.parameters.get('short_call_delta', 0.30)
            
            # Find call with delta closest to target_long_delta
            long_call = min(calls, key=lambda x: abs(abs(x.get('delta', 0)) - target_long_delta))
            
            # Find call with delta closest to target_short_delta
            short_call = min(calls, key=lambda x: abs(abs(x.get('delta', 0)) - target_short_delta))
            
        elif strike_method == 'otm_percentage':
            otm_pct = self.parameters.get('otm_percentage', 0.05)
            short_otm_extra = self.parameters.get('short_call_otm_extra', 0.05)
            
            # Calculate target strikes
            long_target_strike = stock_price * (1 + otm_pct)
            short_target_strike = stock_price * (1 + otm_pct + short_otm_extra)
            
            # Find calls with strikes closest to targets
            long_call = min(calls, key=lambda x: abs(x.get('strike', 0) - long_target_strike))
            short_call = min(calls, key=lambda x: abs(x.get('strike', 0) - short_target_strike))
            
        elif strike_method == 'price_range':
            # Use a fixed spread width
            atm_call = min(calls, key=lambda x: abs(x.get('strike', 0) - stock_price))
            atm_index = calls.index(atm_call)
            
            if atm_index >= 0 and atm_index + 1 < len(calls):
                long_call = atm_call
                
                # Find short call that is spread_width strikes away
                short_index = atm_index
                while short_index + 1 < len(calls) and \
                      calls[short_index].get('strike', 0) < long_call.get('strike', 0) + spread_width:
                    short_index += 1
                    
                if short_index < len(calls):
                    short_call = calls[short_index]
        
        # Ensure we have both legs and the spread is valid
        if not long_call or not short_call:
            return None
            
        long_strike = long_call.get('strike', 0)
        short_strike = short_call.get('strike', 0)
        
        # Ensure the long call is cheaper than the short call
        if long_strike >= short_strike:
            return None
            
        # Calculate spread details
        long_price = long_call.get('ask', 0)
        short_price = short_call.get('bid', 0)
        
        if long_price <= 0 or short_price <= 0:
            return None
            
        net_debit = long_price - short_price
        max_profit = (short_strike - long_strike) - net_debit
        max_loss = net_debit
        
        if max_profit <= 0 or max_loss <= 0:
            return None
            
        risk_reward = max_profit / max_loss
        breakeven = long_strike + net_debit
        
        return {
            'symbol': symbol,
            'stock_price': stock_price,
            'expiration': best_expiry,
            'dte': best_dte,
            'long_call': long_call,
            'short_call': short_call,
            'long_strike': long_strike,
            'short_strike': short_strike,
            'long_price': long_price,
            'short_price': short_price,
            'net_debit': net_debit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'risk_reward': risk_reward,
            'breakeven': breakeven,
        }
    
    def _create_bull_call_spread_signal(self, symbol: str, spread: Dict[str, Any], 
                                      market_data: MarketData) -> Optional[Dict[str, Any]]:
        """
        Create a signal for a bull call spread trade.
        
        Args:
            symbol: Stock symbol
            spread: Spread details
            market_data: Market data
            
        Returns:
            Signal dictionary
        """
        if not spread:
            return None
            
        # Extract spread details
        stock_price = spread.get('stock_price', 0)
        expiration = spread.get('expiration')
        dte = spread.get('dte', 0)
        long_strike = spread.get('long_strike', 0)
        short_strike = spread.get('short_strike', 0)
        net_debit = spread.get('net_debit', 0)
        max_profit = spread.get('max_profit', 0)
        max_loss = spread.get('max_loss', 0)
        risk_reward = spread.get('risk_reward', 0)
        breakeven = spread.get('breakeven', 0)
        
        # Check minimum risk/reward ratio
        min_risk_reward = self.parameters.get('min_risk_reward', 1.5)
        if risk_reward < min_risk_reward:
            return None
            
        # Check if we have an active spread for this symbol already
        if symbol in self.spread_trades:
            return None
            
        # Create base options signal
        signal = self.create_options_signal(
            symbol=symbol,
            action="BULL_CALL_SPREAD",
            option_type="SPREAD",
            strike=long_strike,  # Using long strike as reference
            expiration=expiration,
            premium=net_debit,
            stock_price=stock_price,
            reason="Bullish trend detected",
            strength=min(1.0, risk_reward / 3.0),  # Normalized strength based on R/R
            stop_loss=net_debit * (1 + self.parameters.get('stop_loss_percent', 0.5)), 
            take_profit=net_debit * (1 - self.parameters.get('take_profit_pct', 0.5))
        )
        
        # Add spread-specific details
        signal.update({
            'spread_type': 'bull_call_spread',
            'long_strike': long_strike,
            'short_strike': short_strike,
            'long_price': spread.get('long_price', 0),
            'short_price': spread.get('short_price', 0),
            'spread_width': short_strike - long_strike,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'risk_reward': risk_reward,
            'breakeven': breakeven,
            'dte': dte,
            'long_option_symbol': spread.get('long_call', {}).get('symbol'),
            'short_option_symbol': spread.get('short_call', {}).get('symbol'),
        })
        
        return signal
    
    def position_sizing(self, signal: Dict[str, Any], account_info: Dict[str, Any]) -> float:
        """
        Calculate position size for a bull call spread.
        
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
        max_position_pct = self.parameters.get('max_position_size_percent', 0.05)
        max_risk_pct = self.parameters.get('max_risk_per_trade', 0.02)
        
        # Maximum amount to allocate to this trade
        max_amount = account_value * max_position_pct
        
        # Calculate position size based on spread cost
        net_debit = signal.get('premium', 0)
        if net_debit <= 0:
            return 0
            
        # Each option contract is for 100 shares
        spread_cost = net_debit * 100
        
        # Calculate how many spreads we can afford
        max_spreads = int(max_amount / spread_cost)
        
        # Calculate position size based on risk
        max_risk_amount = account_value * max_risk_pct
        max_loss = signal.get('max_loss', 0)
        
        if max_loss > 0:
            max_loss_total = max_loss * 100  # Convert to dollar amount per spread
            risk_based_spreads = int(max_risk_amount / max_loss_total)
            # Take the smaller of the two limits
            max_spreads = min(max_spreads, risk_based_spreads)
        
        return max_spreads
    
    def on_exit_signal(self, position: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate an exit signal for a bull call spread position.
        
        Args:
            position: Current position data
            
        Returns:
            Exit signal if conditions are met, None otherwise
        """
        # Check exit conditions
        symbol = position.get('symbol')
        entry_time = position.get('timestamp')
        current_time = datetime.now()
        
        # Early exit if required data is missing
        if not symbol or not entry_time:
            return None
            
        # Calculate days elapsed since entry
        days_held = (current_time - entry_time).days
        
        # Check DTE-based exit
        dte_remaining = position.get('dte', 0) - days_held
        dte_exit = self.parameters.get('dte_exit_threshold', 21)
        
        if dte_remaining <= dte_exit:
            return self.create_signal(
                symbol=symbol,
                action="EXIT",
                reason=f"DTE threshold reached ({dte_remaining} days remaining)",
                strength=1.0
            )
        
        # Check profit target
        current_value = position.get('current_value', 0)
        entry_value = position.get('premium', 0)
        max_profit = position.get('max_profit', 0)
        profit_target_pct = self.parameters.get('profit_target_percent', 50) / 100
        
        if max_profit > 0 and entry_value > 0:
            current_profit = entry_value - current_value
            profit_pct = current_profit / max_profit
            
            if profit_pct >= profit_target_pct:
                return self.create_signal(
                    symbol=symbol,
                    action="EXIT",
                    reason=f"Profit target reached ({profit_pct:.1%} of max profit)",
                    strength=1.0
                )
        
        # Check stop loss
        max_loss = position.get('max_loss', 0)
        loss_limit_pct = self.parameters.get('loss_limit_percent', 75) / 100
        
        if max_loss > 0 and entry_value > 0:
            current_loss = current_value - entry_value
            loss_pct = current_loss / max_loss
            
            if loss_pct >= loss_limit_pct:
                return self.create_signal(
                    symbol=symbol,
                    action="EXIT",
                    reason=f"Stop loss triggered ({loss_pct:.1%} of max loss)",
                    strength=1.0
                )
        
        # No exit signal needed
        return None

# Strategy registered via decorator - no need for explicit registration
