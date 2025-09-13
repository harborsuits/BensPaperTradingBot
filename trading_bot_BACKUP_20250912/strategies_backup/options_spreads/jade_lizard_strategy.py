#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jade Lizard Strategy Module

This module implements a Jade Lizard options strategy, which combines a short put with
a short call spread (bear call spread) to collect premium with no upside risk.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Tuple, Optional

from trading_bot.strategies.strategy_template import StrategyOptimizable
from trading_bot.market.universe import Universe
from trading_bot.market.market_data import MarketData
from trading_bot.market.option_chains import OptionChains
from trading_bot.orders.order_manager import OrderManager
from trading_bot.orders.order import Order, OrderType, OrderAction, OrderStatus
from trading_bot.utils.option_utils import get_atm_strike, calculate_max_loss, annualize_returns
from trading_bot.risk.position_sizer import PositionSizer
from trading_bot.signals.volatility_signals import VolatilitySignals
from trading_bot.signals.technical_signals import TechnicalSignals

logger = logging.getLogger(__name__)

class JadeLizardStrategy(StrategyOptimizable):
    """
    Jade Lizard Strategy
    
    A Jade Lizard is an options strategy that combines a short put with a short call spread
    (bear call spread). It is typically used in neutral to moderately bullish market conditions
    to collect premium while having no upside risk.
    
    Key characteristics:
    - Composed of three legs: short put, short call, and long call (higher strike)
    - Collects premium from all three legs
    - No upside risk (call spread credit >= width of call spread)
    - Limited downside risk (breakeven = short put strike - total premium received)
    - Maximum profit = total premium received
    - Benefits from time decay (theta positive)
    - Performs best when the underlying stays around the short put strike
    """
    
    # Default parameters specific to Jade Lizard strategy
    DEFAULT_PARAMS = {
        # Strategy identification
        'strategy_name': 'jade_lizard',
        'strategy_version': '1.0.0',
        
        # Universe selection criteria
        'min_stock_price': 30.0,               # Minimum stock price to consider
        'max_stock_price': 500.0,              # Maximum stock price to consider
        'min_option_volume': 200,              # Minimum option volume
        'min_option_open_interest': 500,       # Minimum option open interest
        'min_adv': 500000,                     # Minimum average daily volume (500K)
        'max_bid_ask_spread_pct': 0.05,        # Maximum bid-ask spread as % of price (5%)
        
        # Volatility parameters
        'min_iv_rank': 40,                     # Minimum IV rank (40%)
        'max_iv_rank': 100,                    # Maximum IV rank (100%)
        
        # Market condition parameters
        'min_price_to_sma_ratio': 0.90,        # Price should be at least 90% of 50-day SMA
        'max_price_to_sma_ratio': 1.10,        # Price should be at most 110% of 50-day SMA
        'neutral_to_bullish_indicators': True,  # Look for neutral to bullish indicators
        
        # Option parameters
        'target_dte': 45,                      # Target days to expiration (30-60 DTE)
        'min_dte': 30,                         # Minimum days to expiration
        'max_dte': 60,                         # Maximum days to expiration
        
        # Strike selection
        'short_put_delta': 0.30,               # Target delta for short put (0.25-0.35)
        'short_call_delta': 0.20,              # Target delta for short call (0.15-0.25)
        'call_spread_width': 5.0,              # Width between short call and long call strikes
        'strike_selection_method': 'delta',    # 'delta' or 'otm_percentage'
        'otm_percentage_put': 0.05,            # 5% OTM for put if using percentage method
        'otm_percentage_call': 0.07,           # 7% OTM for call if using percentage method
        
        # Credit requirements
        'min_total_credit': 1.0,               # Minimum credit to collect
        'min_credit_to_risk_ratio': 0.15,      # Minimum credit as % of risk (15%)
        'call_spread_credit_threshold': 0.90,  # Call spread credit must be at least 90% of width
        
        # Risk management parameters
        'max_position_size_percent': 0.05,     # Maximum position size as % of portfolio (5%)
        'max_risk_per_trade_percent': 0.02,    # Maximum risk per trade as % of portfolio (2%)
        'max_margin_usage_percent': 0.15,      # Maximum margin usage as % of account (15%)
        
        # Exit parameters
        'profit_target_percent': 50,           # Exit at 50% of max profit
        'max_loss_percent': 100,               # Exit at 100% of max loss (double the credit)
        'dte_exit_threshold': 14,              # Exit when DTE reaches 14 days
        
        # Adjustment parameters
        'enable_adjustments': False,           # Whether to adjust positions
        'adjustment_threshold': 0.80,          # Adjust when short put delta reaches 0.80
    }
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Jade Lizard strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters (will be merged with DEFAULT_PARAMS)
            metadata: Strategy metadata
        """
        # Start with default parameters
        jade_lizard_params = self.DEFAULT_PARAMS.copy()
        
        # Override with provided parameters
        if parameters:
            jade_lizard_params.update(parameters)
        
        # Initialize the parent class
        super().__init__(name=name, parameters=jade_lizard_params, metadata=metadata)
        
        logger.info(f"Initialized Jade Lizard strategy: {name}")
    
    def define_universe(self, market_data: MarketData) -> Universe:
        """
        Define the universe of stocks suitable for the Jade Lizard strategy.
        
        Args:
            market_data: Market data instance
            
        Returns:
            Universe of symbols
        """
        universe = Universe()
        
        # Filter by price range
        price_df = market_data.get_latest_prices()
        filtered_symbols = price_df[(price_df['close'] >= self.params['min_stock_price']) & 
                                  (price_df['close'] <= self.params['max_stock_price'])].index.tolist()
        
        universe.add_symbols(filtered_symbols)
        
        # Filter by price relative to 50-day SMA
        sma_df = market_data.get_technical_indicators()['sma_50']
        
        for symbol in list(universe.get_symbols()):
            if symbol not in sma_df.index:
                universe.remove_symbol(symbol)
                continue
                
            price = price_df.loc[symbol, 'close']
            sma = sma_df.loc[symbol, 'sma_50']
            
            price_to_sma_ratio = price / sma
            
            if (price_to_sma_ratio < self.params['min_price_to_sma_ratio'] or
                price_to_sma_ratio > self.params['max_price_to_sma_ratio']):
                universe.remove_symbol(symbol)
                
        # Filter by liquidity criteria
        for symbol in list(universe.get_symbols()):
            # Check average daily volume
            volume_data = market_data.get_historical_data(symbol, 'volume', days=30)
            if volume_data is not None and len(volume_data) > 0:
                avg_volume = volume_data.mean()
                if avg_volume < self.params['min_adv']:
                    universe.remove_symbol(symbol)
                    continue
            else:
                universe.remove_symbol(symbol)
                continue
        
        # Filter by IV rank
        vol_signals = VolatilitySignals(market_data)
        
        for symbol in list(universe.get_symbols()):
            iv_rank = vol_signals.get_iv_rank(symbol)
            
            if iv_rank is None or iv_rank < self.params['min_iv_rank'] or iv_rank > self.params['max_iv_rank']:
                universe.remove_symbol(symbol)
        
        logger.info(f"Jade Lizard universe contains {len(universe.get_symbols())} symbols")
        return universe
    
    def check_selection_criteria(self, symbol: str, market_data: MarketData, 
                               option_chains: OptionChains) -> bool:
        """
        Check if the symbol meets the criteria for the Jade Lizard strategy.
        
        Args:
            symbol: Symbol to check
            market_data: Market data instance
            option_chains: Option chains instance
            
        Returns:
            True if symbol meets the criteria, False otherwise
        """
        # Check if we have options data
        chains = option_chains.get_option_chain(symbol)
        if chains is None or chains.empty:
            logger.debug(f"{symbol} has no option chains available")
            return False
        
        # Check if we have suitable expiration dates
        target_dte = self.params['target_dte']
        min_dte = self.params['min_dte']
        max_dte = self.params['max_dte']
        
        expirations = option_chains.get_expiration_dates(symbol)
        suitable_expiration = False
        
        for exp in expirations:
            exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
            dte = (exp_date - date.today()).days
            
            if min_dte <= dte <= max_dte:
                suitable_expiration = True
                break
        
        if not suitable_expiration:
            logger.debug(f"{symbol} does not have suitable expiration dates")
            return False
        
        # Check option liquidity
        for exp in expirations:
            exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
            dte = (exp_date - date.today()).days
            
            if min_dte <= dte <= max_dte:
                # Check put options
                puts = option_chains.get_puts(symbol, exp)
                if puts.empty:
                    continue
                
                # Check if any puts meet volume/OI criteria
                put_liquidity = puts[(puts['volume'] >= self.params['min_option_volume']) & 
                                     (puts['open_interest'] >= self.params['min_option_open_interest'])]
                
                if put_liquidity.empty:
                    continue
                
                # Check call options
                calls = option_chains.get_calls(symbol, exp)
                if calls.empty:
                    continue
                
                # Check if any calls meet volume/OI criteria
                call_liquidity = calls[(calls['volume'] >= self.params['min_option_volume']) & 
                                      (calls['open_interest'] >= self.params['min_option_open_interest'])]
                
                if call_liquidity.empty:
                    continue
                
                # If we've made it here, we have at least one suitable expiration with liquid options
                return True
        
        logger.debug(f"{symbol} does not have liquid options for Jade Lizard")
        return False
    
    def select_option_contracts(self, symbol: str, market_data: MarketData,
                              option_chains: OptionChains) -> Dict[str, Any]:
        """
        Select the appropriate option contracts for the Jade Lizard strategy.
        
        Args:
            symbol: The stock symbol
            market_data: Market data instance
            option_chains: Option chains instance
            
        Returns:
            Dict with selected option contracts and trade details
        """
        # Get current price
        current_price = market_data.get_latest_price(symbol)
        if current_price is None:
            logger.error(f"Unable to get current price for {symbol}")
            return {}
        
        # Select expiration
        target_dte = self.params['target_dte']
        min_dte = self.params['min_dte']
        max_dte = self.params['max_dte']
        
        expirations = option_chains.get_expiration_dates(symbol)
        selected_expiration = None
        dte = 0
        
        for exp in expirations:
            exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
            current_dte = (exp_date - date.today()).days
            
            if min_dte <= current_dte <= max_dte:
                # Choose the expiration closest to target DTE
                if selected_expiration is None or abs(current_dte - target_dte) < abs(dte - target_dte):
                    selected_expiration = exp
                    dte = current_dte
        
        if selected_expiration is None:
            logger.error(f"No suitable expiration found for {symbol}")
            return {}
        
        # Get puts and calls for selected expiration
        puts = option_chains.get_puts(symbol, selected_expiration)
        calls = option_chains.get_calls(symbol, selected_expiration)
        
        if puts.empty or calls.empty:
            logger.error(f"No options available for {symbol} at {selected_expiration}")
            return {}
        
        # Select strikes based on delta or OTM percentage
        if self.params['strike_selection_method'] == 'delta':
            # Short put selection
            short_put = self._select_short_put_by_delta(puts, self.params['short_put_delta'])
            if short_put is None:
                logger.error(f"Could not find suitable short put for {symbol}")
                return {}
            
            # Short call selection
            short_call = self._select_short_call_by_delta(calls, self.params['short_call_delta'])
            if short_call is None:
                logger.error(f"Could not find suitable short call for {symbol}")
                return {}
        else:
            # Strike selection by OTM percentage
            short_put = self._select_short_put_by_percentage(puts, current_price, self.params['otm_percentage_put'])
            if short_put is None:
                logger.error(f"Could not find suitable short put for {symbol}")
                return {}
            
            short_call = self._select_short_call_by_percentage(calls, current_price, self.params['otm_percentage_call'])
            if short_call is None:
                logger.error(f"Could not find suitable short call for {symbol}")
                return {}
        
        # Select long call (higher strike) for bear call spread
        long_call = self._select_long_call(calls, short_call['strike'], self.params['call_spread_width'])
        if long_call is None:
            logger.error(f"Could not find suitable long call for {symbol}")
            return {}
        
        # Calculate credits and max loss
        put_credit = short_put['bid']
        call_spread_width = long_call['strike'] - short_call['strike']
        call_spread_credit = short_call['bid'] - long_call['ask']
        total_credit = put_credit + call_spread_credit
        
        # Check if call spread credit is sufficient (no upside risk)
        if call_spread_credit < call_spread_width * self.params['call_spread_credit_threshold']:
            logger.debug(f"Call spread credit ({call_spread_credit:.2f}) is insufficient for {symbol}")
            return {}
        
        # Calculate max risk (if stock goes to zero)
        max_risk = short_put['strike'] * 100 - total_credit * 100
        
        # Check minimum credit requirements
        if total_credit < self.params['min_total_credit']:
            logger.debug(f"Total credit ({total_credit:.2f}) is below minimum for {symbol}")
            return {}
        
        if total_credit / (short_put['strike']) < self.params['min_credit_to_risk_ratio']:
            logger.debug(f"Credit/risk ratio is below minimum for {symbol}")
            return {}
        
        # Create option identifiers
        short_put_id = f"{symbol}_{selected_expiration}_{short_put['strike']}_P"
        short_call_id = f"{symbol}_{selected_expiration}_{short_call['strike']}_C"
        long_call_id = f"{symbol}_{selected_expiration}_{long_call['strike']}_C"
        
        # Return trade details
        return {
            'symbol': symbol,
            'strategy': 'jade_lizard',
            'expiration': selected_expiration,
            'dte': dte,
            'short_put': short_put,
            'short_call': short_call,
            'long_call': long_call,
            'short_put_id': short_put_id,
            'short_call_id': short_call_id,
            'long_call_id': long_call_id,
            'put_credit': put_credit,
            'call_spread_credit': call_spread_credit,
            'total_credit': total_credit,
            'call_spread_width': call_spread_width,
            'max_risk': max_risk,
            'risk_reward_ratio': max_risk / (total_credit * 100),
            'current_price': current_price,
            'timestamp': datetime.now().isoformat()
        }
    
    def calculate_position_size(self, trade_details: Dict[str, Any], 
                              position_sizer: PositionSizer) -> int:
        """
        Calculate the number of Jade Lizard spreads to trade.
        
        Args:
            trade_details: Details of the selected Jade Lizard
            position_sizer: Position sizer instance
            
        Returns:
            Number of spreads to trade
        """
        # Get portfolio value
        portfolio_value = position_sizer.get_portfolio_value()
        
        # Calculate max risk for this trade
        max_risk_per_trade = portfolio_value * self.params['max_risk_per_trade_percent']
        
        # Calculate risk per spread
        risk_per_spread = trade_details['max_risk']
        
        # Calculate number of spreads
        if risk_per_spread <= 0:
            return 0
        
        num_spreads = int(max_risk_per_trade / risk_per_spread)
        
        # Check against maximum position size
        max_position_size = portfolio_value * self.params['max_position_size_percent']
        position_cost = trade_details['short_put']['strike'] * 100 * num_spreads
        
        if position_cost > max_position_size:
            num_spreads = int(max_position_size / (trade_details['short_put']['strike'] * 100))
        
        # Check margin requirements
        margin_requirement = trade_details['short_put']['strike'] * 100 * num_spreads * 0.20  # Approximate margin
        max_margin = portfolio_value * self.params['max_margin_usage_percent']
        
        if margin_requirement > max_margin:
            num_spreads = int(max_margin / (trade_details['short_put']['strike'] * 100 * 0.20))
        
        # Ensure at least 1 spread if we're trading
        num_spreads = max(1, num_spreads)
        
        logger.info(f"Jade Lizard position size for {trade_details['symbol']}: {num_spreads} spreads")
        return num_spreads
    
    def prepare_entry_orders(self, trade_details: Dict[str, Any], 
                           num_spreads: int) -> List[Order]:
        """
        Prepare orders for entering the Jade Lizard position.
        
        Args:
            trade_details: Details of the selected Jade Lizard
            num_spreads: Number of spreads to trade
            
        Returns:
            List of orders
        """
        if num_spreads <= 0:
            return []
        
        symbol = trade_details['symbol']
        orders = []
        
        # Create a unique trade ID
        trade_id = f"jl_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Short put order
        short_put_order = Order(
            symbol=symbol,
            option_symbol=trade_details['short_put_id'],
            order_type=OrderType.LIMIT,
            action=OrderAction.SELL,
            quantity=num_spreads,
            limit_price=trade_details['short_put']['bid'],
            trade_id=trade_id,
            order_details={
                'strategy': 'jade_lizard',
                'leg': 'short_put',
                'expiration': trade_details['expiration'],
                'strike': trade_details['short_put']['strike'],
                'trade_details': trade_details
            }
        )
        orders.append(short_put_order)
        
        # Short call order
        short_call_order = Order(
            symbol=symbol,
            option_symbol=trade_details['short_call_id'],
            order_type=OrderType.LIMIT,
            action=OrderAction.SELL,
            quantity=num_spreads,
            limit_price=trade_details['short_call']['bid'],
            trade_id=trade_id,
            order_details={
                'strategy': 'jade_lizard',
                'leg': 'short_call',
                'expiration': trade_details['expiration'],
                'strike': trade_details['short_call']['strike'],
                'trade_details': trade_details
            }
        )
        orders.append(short_call_order)
        
        # Long call order
        long_call_order = Order(
            symbol=symbol,
            option_symbol=trade_details['long_call_id'],
            order_type=OrderType.LIMIT,
            action=OrderAction.BUY,
            quantity=num_spreads,
            limit_price=trade_details['long_call']['ask'],
            trade_id=trade_id,
            order_details={
                'strategy': 'jade_lizard',
                'leg': 'long_call',
                'expiration': trade_details['expiration'],
                'strike': trade_details['long_call']['strike'],
                'trade_details': trade_details
            }
        )
        orders.append(long_call_order)
        
        logger.info(f"Created Jade Lizard entry orders for {symbol}: {num_spreads} spreads")
        return orders
    
    def check_exit_conditions(self, position: Dict[str, Any], 
                            market_data: MarketData,
                            option_chains: OptionChains) -> bool:
        """
        Check if exit conditions are met for an existing Jade Lizard position.
        
        Args:
            position: Current position
            market_data: Market data instance
            option_chains: Option chains instance
            
        Returns:
            True if exit conditions are met, False otherwise
        """
        if 'trade_details' not in position:
            logger.error("Invalid position data for exit check")
            return False
        
        trade_details = position['trade_details']
        symbol = trade_details['symbol']
        expiration = trade_details['expiration']
        
        # Check DTE
        exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
        current_dte = (exp_date - date.today()).days
        
        if current_dte <= self.params['dte_exit_threshold']:
            logger.info(f"Exiting {symbol} Jade Lizard: DTE {current_dte} <= threshold {self.params['dte_exit_threshold']}")
            return True
        
        # Get current price of underlying
        current_price = market_data.get_latest_price(symbol)
        if current_price is None:
            logger.error(f"Unable to get current price for {symbol}")
            return False
        
        # Check if options data is available
        try:
            # Get current option prices
            short_put_price = option_chains.get_option_price(
                symbol=symbol,
                expiration=expiration,
                strike=trade_details['short_put']['strike'],
                option_type='put',
                price_type='ask'  # Ask price to close short position
            )
            
            short_call_price = option_chains.get_option_price(
                symbol=symbol,
                expiration=expiration,
                strike=trade_details['short_call']['strike'],
                option_type='call',
                price_type='ask'  # Ask price to close short position
            )
            
            long_call_price = option_chains.get_option_price(
                symbol=symbol,
                expiration=expiration,
                strike=trade_details['long_call']['strike'],
                option_type='call',
                price_type='bid'  # Bid price to close long position
            )
            
            if None in [short_put_price, short_call_price, long_call_price]:
                logger.error(f"Missing option price data for {symbol} Jade Lizard")
                return False
            
            # Calculate current value of position
            initial_credit = trade_details['total_credit']
            current_cost = short_put_price + short_call_price - long_call_price
            
            # Calculate profit/loss percentage
            profit_pct = (initial_credit - current_cost) / initial_credit * 100
            
            # Check profit target
            if profit_pct >= self.params['profit_target_percent']:
                logger.info(f"Exiting {symbol} Jade Lizard: Profit target reached ({profit_pct:.2f}%)")
                return True
            
            # Check max loss
            if profit_pct <= -self.params['max_loss_percent']:
                logger.info(f"Exiting {symbol} Jade Lizard: Max loss reached ({profit_pct:.2f}%)")
                return True
            
            # Check price breaching short put strike (high risk)
            if current_price < trade_details['short_put']['strike'] * 0.95:
                logger.info(f"Exiting {symbol} Jade Lizard: Price below short put strike")
                return True
            
        except Exception as e:
            logger.error(f"Error checking exit conditions for {symbol} Jade Lizard: {str(e)}")
            return False
        
        return False
    
    def prepare_exit_orders(self, position: Dict[str, Any]) -> List[Order]:
        """
        Prepare orders for exiting the Jade Lizard position.
        
        Args:
            position: Current position
            
        Returns:
            List of orders
        """
        if 'trade_details' not in position:
            logger.error("Invalid position data for exit orders")
            return []
        
        trade_details = position['trade_details']
        symbol = trade_details['symbol']
        num_spreads = position.get('quantity', 0)
        
        if num_spreads <= 0:
            return []
        
        orders = []
        
        # Create a unique trade ID
        trade_id = f"jl_exit_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Buy to close short put
        buy_put_order = Order(
            symbol=symbol,
            option_symbol=trade_details['short_put_id'],
            order_type=OrderType.MARKET,
            action=OrderAction.BUY,
            quantity=num_spreads,
            trade_id=trade_id,
            order_details={
                'strategy': 'jade_lizard',
                'leg': 'short_put_exit',
                'expiration': trade_details['expiration'],
                'strike': trade_details['short_put']['strike'],
                'original_trade_id': position.get('trade_id', '')
            }
        )
        orders.append(buy_put_order)
        
        # Buy to close short call
        buy_call_order = Order(
            symbol=symbol,
            option_symbol=trade_details['short_call_id'],
            order_type=OrderType.MARKET,
            action=OrderAction.BUY,
            quantity=num_spreads,
            trade_id=trade_id,
            order_details={
                'strategy': 'jade_lizard',
                'leg': 'short_call_exit',
                'expiration': trade_details['expiration'],
                'strike': trade_details['short_call']['strike'],
                'original_trade_id': position.get('trade_id', '')
            }
        )
        orders.append(buy_call_order)
        
        # Sell to close long call
        sell_call_order = Order(
            symbol=symbol,
            option_symbol=trade_details['long_call_id'],
            order_type=OrderType.MARKET,
            action=OrderAction.SELL,
            quantity=num_spreads,
            trade_id=trade_id,
            order_details={
                'strategy': 'jade_lizard',
                'leg': 'long_call_exit',
                'expiration': trade_details['expiration'],
                'strike': trade_details['long_call']['strike'],
                'original_trade_id': position.get('trade_id', '')
            }
        )
        orders.append(sell_call_order)
        
        logger.info(f"Created Jade Lizard exit orders for {symbol}: {num_spreads} spreads")
        return orders
    
    def prepare_adjustment_orders(self, position: Dict[str, Any],
                                market_data: MarketData,
                                option_chains: OptionChains) -> List[Order]:
        """
        Prepare orders for adjusting the Jade Lizard position if needed.
        
        Args:
            position: Current position
            market_data: Market data instance
            option_chains: Option chains instance
            
        Returns:
            List of adjustment orders
        """
        # Only proceed if adjustments are enabled
        if not self.params['enable_adjustments']:
            return []
        
        if 'trade_details' not in position:
            logger.error("Invalid position data for adjustment")
            return []
        
        trade_details = position['trade_details']
        symbol = trade_details['symbol']
        num_spreads = position.get('quantity', 0)
        
        if num_spreads <= 0:
            return []
        
        # Get current price
        current_price = market_data.get_latest_price(symbol)
        if current_price is None:
            logger.error(f"Unable to get current price for {symbol}")
            return []
        
        # Check if price is approaching short put strike
        short_put_strike = trade_details['short_put']['strike']
        
        # If price is above short put strike, no adjustment needed
        if current_price > short_put_strike:
            return []
        
        # Calculate short put delta
        try:
            # This is simplified - in a real system you would get the actual delta from the options chain
            short_put_delta = self._estimate_put_delta(short_put_strike, current_price, trade_details['dte'])
            
            # Check if delta exceeds threshold
            if abs(short_put_delta) < self.params['adjustment_threshold']:
                return []
            
            # For a Jade Lizard adjustment, we can:
            # 1. Roll down the call spread to collect more premium
            # 2. Roll the put to a lower strike
            # Here we'll implement a simple roll down of the call spread
            
            # Find new strikes for call spread
            new_expiration = trade_details['expiration']  # Keep same expiration for simplicity
            calls = option_chains.get_calls(symbol, new_expiration)
            
            if calls.empty:
                logger.error(f"No call options available for {symbol} adjustment")
                return []
            
            # Find new short call with higher premium
            new_short_call = self._select_short_call_by_delta(calls, self.params['short_call_delta'] * 1.5)
            if new_short_call is None:
                logger.error(f"Could not find suitable new short call for {symbol} adjustment")
                return []
            
            # Find new long call
            new_long_call = self._select_long_call(calls, new_short_call['strike'], self.params['call_spread_width'])
            if new_long_call is None:
                logger.error(f"Could not find suitable new long call for {symbol} adjustment")
                return []
            
            # Calculate credit from rolling the call spread
            old_call_spread_value = (short_call_ask - long_call_bid)  # Cost to close existing call spread
            new_call_spread_credit = new_short_call['bid'] - new_long_call['ask']  # Credit from new call spread
            net_credit = new_call_spread_credit - old_call_spread_value
            
            # Only adjust if we can collect a worthwhile credit
            if net_credit < 0.10:
                logger.info(f"Adjustment credit too small for {symbol} Jade Lizard")
                return []
            
            # Create adjustment orders
            orders = []
            trade_id = f"jl_adjust_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Close existing call spread
            buy_call_order = Order(
                symbol=symbol,
                option_symbol=trade_details['short_call_id'],
                order_type=OrderType.LIMIT,
                action=OrderAction.BUY,
                quantity=num_spreads,
                limit_price=short_call_ask,
                trade_id=trade_id,
                order_details={
                    'strategy': 'jade_lizard',
                    'leg': 'short_call_close',
                    'adjustment': True,
                    'original_trade_id': position.get('trade_id', '')
                }
            )
            orders.append(buy_call_order)
            
            sell_call_order = Order(
                symbol=symbol,
                option_symbol=trade_details['long_call_id'],
                order_type=OrderType.LIMIT,
                action=OrderAction.SELL,
                quantity=num_spreads,
                limit_price=long_call_bid,
                trade_id=trade_id,
                order_details={
                    'strategy': 'jade_lizard',
                    'leg': 'long_call_close',
                    'adjustment': True,
                    'original_trade_id': position.get('trade_id', '')
                }
            )
            orders.append(sell_call_order)
            
            # Open new call spread
            new_short_call_id = f"{symbol}_{new_expiration}_{new_short_call['strike']}_C"
            new_long_call_id = f"{symbol}_{new_expiration}_{new_long_call['strike']}_C"
            
            new_short_call_order = Order(
                symbol=symbol,
                option_symbol=new_short_call_id,
                order_type=OrderType.LIMIT,
                action=OrderAction.SELL,
                quantity=num_spreads,
                limit_price=new_short_call['bid'],
                trade_id=trade_id,
                order_details={
                    'strategy': 'jade_lizard',
                    'leg': 'new_short_call',
                    'adjustment': True,
                    'original_trade_id': position.get('trade_id', '')
                }
            )
            orders.append(new_short_call_order)
            
            new_long_call_order = Order(
                symbol=symbol,
                option_symbol=new_long_call_id,
                order_type=OrderType.LIMIT,
                action=OrderAction.BUY,
                quantity=num_spreads,
                limit_price=new_long_call['ask'],
                trade_id=trade_id,
                order_details={
                    'strategy': 'jade_lizard',
                    'leg': 'new_long_call',
                    'adjustment': True,
                    'original_trade_id': position.get('trade_id', '')
                }
            )
            orders.append(new_long_call_order)
            
            logger.info(f"Created adjustment orders for {symbol} Jade Lizard")
            return orders
            
        except Exception as e:
            logger.error(f"Error calculating adjustment for {symbol} Jade Lizard: {str(e)}")
            return []
        
        return []
    
    def get_optimization_params(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            'short_put_delta': [0.25, 0.30, 0.35],
            'short_call_delta': [0.15, 0.20, 0.25],
            'call_spread_width': [5.0, 7.5, 10.0],
            'target_dte': [30, 45, 60],
            'profit_target_percent': [40, 50, 60],
            'min_iv_rank': [30, 40, 50],
            'strike_selection_method': ['delta', 'otm_percentage'],
            'otm_percentage_put': [0.03, 0.05, 0.07],
            'otm_percentage_call': [0.05, 0.07, 0.10],
            'min_credit_to_risk_ratio': [0.10, 0.15, 0.20],
            'dte_exit_threshold': [7, 14, 21]
        }
    
    def evaluate_performance(self, backtest_results: Dict[str, Any]) -> float:
        """
        Evaluate the performance of the strategy in a backtest.
        
        Args:
            backtest_results: Dictionary of backtest results
            
        Returns:
            Performance score (higher is better)
        """
        # Extract key metrics
        total_return = backtest_results.get('total_return', 0)
        max_drawdown = backtest_results.get('max_drawdown', 0)
        sharpe_ratio = backtest_results.get('sharpe_ratio', 0)
        profit_factor = backtest_results.get('profit_factor', 0)
        win_rate = backtest_results.get('win_rate', 0)
        
        # Avoid division by zero
        if max_drawdown == 0:
            max_drawdown = 0.0001
        
        # Calculate a weighted score
        return_to_drawdown = total_return / abs(max_drawdown)
        
        # Weighted score (adjust weights based on priorities)
        score = (
            0.3 * return_to_drawdown + 
            0.3 * sharpe_ratio + 
            0.2 * profit_factor + 
            0.2 * win_rate
        )
        
        return score
    
    # Helper methods
    def _select_short_put_by_delta(self, puts: pd.DataFrame, target_delta: float) -> Dict[str, Any]:
        """
        Select short put based on delta.
        
        Args:
            puts: DataFrame of put options
            target_delta: Target delta (absolute value)
            
        Returns:
            Selected put option or None
        """
        if 'delta' not in puts.columns:
            logger.error("Delta column not found in put options data")
            return None
        
        # Get absolute delta values
        puts['abs_delta'] = puts['delta'].abs()
        
        # Find put with delta closest to target
        puts['delta_diff'] = abs(puts['abs_delta'] - target_delta)
        best_put = puts.sort_values('delta_diff').iloc[0]
        
        return best_put.to_dict()
    
    def _select_short_call_by_delta(self, calls: pd.DataFrame, target_delta: float) -> Dict[str, Any]:
        """
        Select short call based on delta.
        
        Args:
            calls: DataFrame of call options
            target_delta: Target delta
            
        Returns:
            Selected call option or None
        """
        if 'delta' not in calls.columns:
            logger.error("Delta column not found in call options data")
            return None
        
        # Find call with delta closest to target
        calls['delta_diff'] = abs(calls['delta'] - target_delta)
        best_call = calls.sort_values('delta_diff').iloc[0]
        
        return best_call.to_dict()
    
    def _select_short_put_by_percentage(self, puts: pd.DataFrame, current_price: float, 
                                       otm_percentage: float) -> Dict[str, Any]:
        """
        Select short put based on OTM percentage.
        
        Args:
            puts: DataFrame of put options
            current_price: Current price of underlying
            otm_percentage: Target OTM percentage
            
        Returns:
            Selected put option or None
        """
        target_strike = current_price * (1 - otm_percentage)
        
        # Find put with strike closest to target
        puts['strike_diff'] = abs(puts['strike'] - target_strike)
        best_put = puts.sort_values('strike_diff').iloc[0]
        
        return best_put.to_dict()
    
    def _select_short_call_by_percentage(self, calls: pd.DataFrame, current_price: float, 
                                        otm_percentage: float) -> Dict[str, Any]:
        """
        Select short call based on OTM percentage.
        
        Args:
            calls: DataFrame of call options
            current_price: Current price of underlying
            otm_percentage: Target OTM percentage
            
        Returns:
            Selected call option or None
        """
        target_strike = current_price * (1 + otm_percentage)
        
        # Find call with strike closest to target
        calls['strike_diff'] = abs(calls['strike'] - target_strike)
        best_call = calls.sort_values('strike_diff').iloc[0]
        
        return best_call.to_dict()
    
    def _select_long_call(self, calls: pd.DataFrame, short_strike: float, width: float) -> Dict[str, Any]:
        """
        Select long call based on width from short call.
        
        Args:
            calls: DataFrame of call options
            short_strike: Strike of short call
            width: Target width between strikes
            
        Returns:
            Selected call option or None
        """
        target_strike = short_strike + width
        
        # Find all calls with higher strike
        higher_calls = calls[calls['strike'] > short_strike]
        
        if higher_calls.empty:
            return None
        
        # Find call with strike closest to target
        higher_calls['strike_diff'] = abs(higher_calls['strike'] - target_strike)
        best_call = higher_calls.sort_values('strike_diff').iloc[0]
        
        return best_call.to_dict()
    
    def _estimate_put_delta(self, strike: float, current_price: float, dte: int) -> float:
        """
        Estimate put delta based on moneyness and DTE.
        This is a simplified calculation - in production, use actual option greeks.
        
        Args:
            strike: Strike price
            current_price: Current price of underlying
            dte: Days to expiration
            
        Returns:
            Estimated delta (negative value)
        """
        # Moneyness (K/S - 1)
        moneyness = strike / current_price - 1
        
        # Time factor (shorter time = higher delta)
        time_factor = max(0.1, min(1.0, dte / 365))
        
        # Calculate approximate delta
        if moneyness >= 0:  # ITM put
            delta = -0.5 - 0.5 * moneyness / (0.2 * time_factor)
            return max(-1.0, min(-0.5, delta))
        else:  # OTM put
            delta = -0.5 * np.exp(moneyness / (0.2 * time_factor))
            return max(-0.5, min(0, delta)) 