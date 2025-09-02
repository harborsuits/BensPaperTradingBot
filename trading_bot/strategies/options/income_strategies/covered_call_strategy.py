#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Covered Call Options Strategy Module

This module implements a covered call options strategy that generates income by selling
call options against a long stock position.

A covered call position is created by:
1. Owning or buying shares of the underlying stock
2. Selling call options against those shares (typically 1 call per 100 shares)

This creates an income-generating position that slightly reduces downside risk
while capping potential upside gains.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Tuple, Optional
import math

from trading_bot.strategies.options.base.options_base_strategy import OptionsBaseStrategy
from trading_bot.market.universe import Universe
from trading_bot.market.market_data import MarketData
from trading_bot.market.option_chains import OptionChains
from trading_bot.orders.order_manager import OrderManager
from trading_bot.orders.order import Order, OrderType, OrderAction, OrderStatus
from trading_bot.utils.option_utils import get_atm_strike, calculate_max_loss, annualize_returns
from trading_bot.risk.position_sizer import PositionSizer
from trading_bot.signals.volatility_signals import VolatilitySignals
from trading_bot.signals.technical_signals import TechnicalSignals
from trading_bot.accounts.account_data import AccountData

logger = logging.getLogger(__name__)

class CoveredCallStrategy(OptionsBaseStrategy):
    """
    Covered Call Options Strategy
    
    This strategy involves owning or buying shares of the underlying stock and selling
    call options against that position. It's a popular income-generating strategy
    that slightly reduces downside risk while capping potential upside gains.
    
    Key characteristics:
    - Income generation through option premium collection
    - Partial downside protection (by the amount of premium received)
    - Capped upside potential (limited to strike price plus premium)
    - Most suitable in neutral to slightly bullish markets
    - Typically implemented using monthly expiration cycles
    - Common implementations include "buy-write" (simultaneous stock purchase and call sale)
        or managing existing stock positions
    
    Attributes:
        params (Dict[str, Any]): Dictionary of strategy parameters
        name (str): Strategy name, defaults to 'covered_call'
        version (str): Strategy version, defaults to '1.0.0'
    """
    
    # Default parameters for covered call strategy
    DEFAULT_PARAMS = {
        # Strategy identification
        'strategy_name': 'covered_call',
        'strategy_version': '1.0.0',
        
        # Universe selection criteria
        'min_stock_price': 30.0,              # Minimum stock price to consider
        'max_stock_price': 500.0,             # Maximum stock price to consider
        'min_option_volume': 100,             # Minimum option volume
        'min_option_open_interest': 200,      # Minimum option open interest
        'min_iv_percentile': 40,              # Minimum IV percentile
        'max_iv_percentile': 80,              # Maximum IV percentile
        
        # Technical analysis parameters 
        'min_historical_days': 252,           # Days of historical data required
        'trend_indicator': 'ema_20_50',       # Indicator to determine trend
        
        # Stock selection criteria
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
        'max_num_positions': 20,              # Maximum number of positions
        'stock_stop_loss_percent': 0.15,      # Stock stop loss percentage (15%)
        'position_size_adjustment': 'equal',  # Position sizing method: 'equal', 'volatility', 'beta'
        
        # Exit parameters
        'profit_target_percent': 80,          # Exit at this percentage of max profit
        'roll_when_dte': 5,                   # Roll options with this many days left
        'min_roll_credit': 0.002,             # Minimum credit to collect when rolling (0.2%)
        'close_on_ex_dividend_day': True,     # Whether to close before ex-dividend date to avoid assignment
    }
    
    def __init__(self, strategy_id: str = None, params: Dict[str, Any] = None):
        """
        Initialize the Covered Call strategy.
        
        Args:
            strategy_id: Unique identifier for this strategy instance
            params: Strategy parameters that override the defaults
        """
        # Initialize with default parameters, overridden by any provided params
        all_params = self.DEFAULT_PARAMS.copy()
        if params:
            all_params.update(params)
            
        super().__init__(strategy_id=strategy_id, 
                         name=all_params.get('strategy_name', 'covered_call'),
                         parameters=all_params)
        
        self.logger = logging.getLogger(f"{__name__}.{self.strategy_id}")
        self.logger.info(f"Initialized Covered Call Strategy: {self.strategy_id}")
    
    def generate_signals(self, market_data: MarketData, option_chains: OptionChains) -> List[Dict[str, Any]]:
        """
        Generate covered call trade signals based on market data and option chains.
        
        Args:
            market_data: Market data for underlying assets
            option_chains: Option chain data for potential trades
            
        Returns:
            List of trade signals
        """
        signals = []
        
        # Get universe of stocks that meet our criteria
        universe = self._get_eligible_stocks(market_data)
        
        for symbol in universe:
            # Get current stock data
            stock_data = market_data.get_latest_quote(symbol)
            if not stock_data:
                continue
                
            stock_price = stock_data.get('price')
            if not stock_price:
                continue
            
            # Get eligible option chains for the stock
            eligible_options = self._get_eligible_call_options(symbol, stock_price, option_chains)
            
            if not eligible_options or len(eligible_options) == 0:
                self.logger.debug(f"No eligible call options found for {symbol}")
                continue
            
            # Find the best call option based on our criteria
            best_call = self._select_best_call_option(eligible_options, stock_price)
            if not best_call:
                continue
                
            # Calculate the trade metrics
            premium = best_call.get('bid')
            strike = best_call.get('strike')
            expiration = best_call.get('expiration')
            dte = best_call.get('dte')
            
            # Premium yield and potential return calculations
            premium_yield = premium / stock_price
            max_profit = premium + (strike - stock_price if strike > stock_price else 0)
            max_profit_percent = max_profit / stock_price
            annualized_return = annualize_returns(max_profit_percent, dte)
            
            # Generate signal if metrics meet our thresholds
            min_premium_yield = self.parameters.get('min_premium_yield')
            target_return = self.parameters.get('annualized_target_return')
            
            if premium_yield >= min_premium_yield and annualized_return >= target_return:
                signal = {
                    'symbol': symbol,
                    'strategy': self.name,
                    'action': 'SELL_CALL',
                    'stock_price': stock_price,
                    'option_symbol': best_call.get('symbol'),
                    'option_expiration': expiration,
                    'option_strike': strike,
                    'option_premium': premium,
                    'option_delta': best_call.get('delta'),
                    'premium_yield': premium_yield,
                    'annualized_return': annualized_return,
                    'signal_time': datetime.now(),
                    'signal_strength': self._calculate_signal_strength(premium_yield, annualized_return),
                    'quantity': self._calculate_position_size(symbol, stock_price),
                }
                signals.append(signal)
        
        return signals
    
    def _get_eligible_stocks(self, market_data: MarketData) -> List[str]:
        """
        Get list of stocks that meet our selection criteria.
        
        Args:
            market_data: Market data for filtering stocks
            
        Returns:
            List of eligible stock symbols
        """
        # Implementation would filter stocks based on:
        # - Price range, volume, market cap
        # - Technical indicators and trend
        # - Fundamental quality metrics
        # - Volatility and IV percentile
        # For simplicity, we're returning a placeholder
        
        return market_data.get_symbols_by_criteria(
            min_price=self.parameters.get('min_stock_price'),
            max_price=self.parameters.get('max_stock_price'),
            min_market_cap=self.parameters.get('min_market_cap'),
            min_volume=100000,  # Assume reasonable volume needed
            min_dividend_yield=self.parameters.get('min_dividend_yield')
        )
    
    def _get_eligible_call_options(self, symbol: str, stock_price: float, option_chains: OptionChains) -> List[Dict[str, Any]]:
        """
        Get eligible call options for covered call strategy.
        
        Args:
            symbol: Stock symbol
            stock_price: Current stock price
            option_chains: Option chain data
            
        Returns:
            List of eligible call options
        """
        # Get strategy parameters
        target_dte = self.parameters.get('target_dte')
        min_dte = self.parameters.get('min_dte')
        max_dte = self.parameters.get('max_dte')
        delta_target = self.parameters.get('delta_target')
        delta_range = self.parameters.get('delta_range')
        min_option_volume = self.parameters.get('min_option_volume')
        min_option_open_interest = self.parameters.get('min_option_open_interest')
        
        # Get all call options for this symbol
        all_calls = option_chains.get_calls(symbol)
        if not all_calls:
            return []
        
        # Filter for options that meet our criteria
        eligible_calls = []
        for option in all_calls:
            # Basic option data
            expiration = option.get('expiration')
            if not expiration:
                continue
                
            # Calculate days to expiration
            expiry_date = datetime.strptime(expiration, '%Y-%m-%d').date()
            today = date.today()
            dte = (expiry_date - today).days
            
            # Check if DTE is within our target range
            if dte < min_dte or dte > max_dte:
                continue
                
            # Check delta range
            delta = option.get('delta')
            if not delta or delta < (delta_target - delta_range) or delta > (delta_target + delta_range):
                continue
                
            # Check liquidity metrics
            volume = option.get('volume', 0)
            open_interest = option.get('open_interest', 0)
            if volume < min_option_volume or open_interest < min_option_open_interest:
                continue
                
            # Check bid-ask spread is reasonable (typically < 10% of option price)
            bid = option.get('bid', 0)
            ask = option.get('ask', 0)
            if bid <= 0 or ask <= 0:
                continue
                
            spread_percent = (ask - bid) / ((bid + ask) / 2)
            if spread_percent > 0.10:  # 10% maximum spread
                continue
                
            # Add DTE to the option data for later use
            option['dte'] = dte
            eligible_calls.append(option)
        
        return eligible_calls
    
    def _select_best_call_option(self, eligible_options: List[Dict[str, Any]], stock_price: float) -> Dict[str, Any]:
        """
        Select the best call option for the covered call strategy.
        
        Args:
            eligible_options: List of eligible call options
            stock_price: Current stock price
            
        Returns:
            Best call option for the strategy
        """
        if not eligible_options:
            return None
            
        # Sort by annualized return (premium yield adjusted for DTE)
        for option in eligible_options:
            premium = option.get('bid', 0)
            strike = option.get('strike', 0)
            dte = option.get('dte', 30)
            
            # Calculate premium yield
            premium_yield = premium / stock_price
            
            # Calculate max profit percentage and annualized return
            max_profit = premium + max(0, strike - stock_price)
            max_profit_percent = max_profit / stock_price
            annualized = annualize_returns(max_profit_percent, dte)
            
            # Add these calculations to the option data
            option['premium_yield'] = premium_yield
            option['annualized_return'] = annualized
        
        # Sort by annualized return in descending order
        sorted_options = sorted(eligible_options, key=lambda x: x.get('annualized_return', 0), reverse=True)
        
        # Return the option with the highest annualized return
        return sorted_options[0] if sorted_options else None
    
    def _calculate_signal_strength(self, premium_yield: float, annualized_return: float) -> float:
        """
        Calculate the strength of the signal based on expected returns.
        
        Args:
            premium_yield: Premium as a percentage of stock price
            annualized_return: Annualized return of the covered call
            
        Returns:
            Signal strength value between 0 and 1
        """
        # Weight the premium yield and annualized return to get a signal strength
        min_premium = self.parameters.get('min_premium_yield', 0.005)
        target_premium = min_premium * 2
        
        min_return = self.parameters.get('annualized_target_return', 0.10)
        target_return = min_return * 1.5
        
        premium_strength = min(1.0, (premium_yield - min_premium) / (target_premium - min_premium))
        return_strength = min(1.0, (annualized_return - min_return) / (target_return - min_return))
        
        # Combined signal strength (weighted average)
        signal_strength = (premium_strength * 0.4) + (return_strength * 0.6)
        return max(0.0, min(1.0, signal_strength))
    
    def _calculate_position_size(self, symbol: str, stock_price: float) -> int:
        """
        Calculate the appropriate position size based on our risk parameters.
        
        Args:
            symbol: Stock symbol
            stock_price: Current stock price
            
        Returns:
            Number of shares to buy (will be a multiple of 100 for options)
        """
        # Get account equity and position sizing limits
        account_equity = self.get_account_equity()
        max_position_pct = self.parameters.get('max_position_size_percent', 0.05)
        
        # Calculate maximum dollar amount for this position
        max_position_value = account_equity * max_position_pct
        
        # Calculate number of shares, rounded down to nearest 100
        max_shares = int(max_position_value / stock_price)
        shares = (max_shares // 100) * 100  # Round down to nearest 100 for option contracts
        
        return shares if shares >= 100 else 0
    
    def get_account_equity(self) -> float:
        """
        Get the current account equity (placeholder method).
        
        Returns:
            Current account equity
        """
        # In a real implementation, this would get the actual account equity
        # For now, we'll return a placeholder value
        return 100000.0
    
    def on_exit_signal(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an exit signal based on current position and market conditions.
        
        Args:
            position: Current position data
            
        Returns:
            Exit signal if appropriate, None otherwise
        """
        # Exit logic for a covered call would include:
        # 1. Exit if stock drops below stop loss
        # 2. Exit if option can be bought back for significant profit (e.g., 80% of premium)
        # 3. Roll the option near expiration if conditions are favorable
        # 4. Close before ex-dividend date if ITM and at risk of assignment
        
        # This is a placeholder implementation
        return {
            'symbol': position.get('symbol'),
            'strategy': self.name,
            'action': 'EXIT',
            'signal_time': datetime.now(),
            'reason': 'profit_target_reached'
        }

# Register strategy with the registry
from trading_bot.core.strategy_registry import StrategyRegistry
try:
    StrategyRegistry.register("covered_call", CoveredCallStrategy)
except Exception as e:
    logger.error(f"Failed to register CoveredCallStrategy: {e}")
