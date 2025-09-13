#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iron Butterfly Strategy Module

This module implements an iron butterfly options strategy that profits from
low volatility markets when price stays near the strike price at expiration.
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

class IronButterflyStrategy(StrategyOptimizable):
    """
    Iron Butterfly Strategy
    
    An Iron Butterfly is an options strategy that combines a bull put spread with a bear call spread,
    where both spreads share the same short strike. This creates a position that profits when the 
    underlying price stays near the middle strike at expiration.
    
    Key characteristics:
    - Limited risk (defined by the wing width minus the premium received)
    - Limited profit (maximum profit achieved when price settles at the middle strike)
    - Profits in low volatility environments when price stays near the strike
    - Maximum profit occurs at the short strike price
    - Typical profit zone is narrower than an iron condor but with higher potential return
    
    This strategy performs best in range-bound markets with low expected volatility.
    """
    
    # Default parameters specific to Iron Butterfly strategy
    DEFAULT_PARAMS = {
        # Strategy identification
        'strategy_name': 'iron_butterfly',
        'strategy_version': '1.0.0',
        
        # Universe selection criteria
        'min_stock_price': 30.0,              # Minimum stock price to consider
        'max_stock_price': 500.0,             # Maximum stock price to consider
        'min_option_volume': 200,             # Minimum option volume
        'min_option_open_interest': 500,      # Minimum option open interest
        'min_adv': 500000,                    # Minimum average daily volume (500K)
        'max_bid_ask_spread_pct': 0.05,       # Maximum bid-ask spread as % of price (5%)
        
        # Volatility parameters
        'min_iv_rank': 30,                    # Minimum IV rank (30%)
        'max_iv_rank': 80,                    # Maximum IV rank (80%)
        'prefer_low_historical_volatility': True,  # Prefer stocks with low historical volatility
        
        # Market condition parameters
        'range_days': 20,                     # Days to check for range-bound behavior
        'max_price_movement_pct': 10,         # Maximum price movement % in range period
        
        # Option parameters
        'target_dte': 30,                     # Target days to expiration (21-45 DTE)
        'min_dte': 21,                        # Minimum days to expiration
        'max_dte': 45,                        # Maximum days to expiration
        
        # Strike selection
        'middle_strike_method': 'atm',        # 'atm' (at-the-money) or 'delta' 
        'middle_strike_delta': 0.50,          # Target delta for middle strike if using delta method
        'wing_width': 2,                      # Number of strikes between middle and wing strikes
        
        # Entry and credit parameters
        'min_credit': 0.50,                   # Minimum credit to collect
        'min_credit_to_width_ratio': 0.15,    # Minimum credit as % of wing width
        
        # Risk management parameters
        'max_position_size_percent': 0.02,    # Maximum position size as % of portfolio (2%)
        'max_risk_per_trade_percent': 0.01,   # Maximum risk per trade as % of portfolio (1%)
        'max_num_positions': 5,               # Maximum number of concurrent positions
        'max_correlation': 0.70,              # Maximum correlation between positions
        
        # Exit parameters
        'profit_target_percent': 50,          # Exit at 50% of max profit
        'max_loss_percent': 100,              # Exit at 100% of max loss (wing width - credit)
        'dte_exit_threshold': 7,              # Exit when DTE reaches 7 days
        
        # Adjustment parameters
        'enable_adjustments': False,          # Whether to adjust positions
        'adjustment_threshold': 0.35,         # Adjust when price is within 35% of wing strikes
    }
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Iron Butterfly strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters (will be merged with DEFAULT_PARAMS)
            metadata: Strategy metadata
        """
        # Start with default parameters
        butterfly_params = self.DEFAULT_PARAMS.copy()
        
        # Override with provided parameters
        if parameters:
            butterfly_params.update(parameters)
        
        # Initialize the parent class
        super().__init__(name=name, parameters=butterfly_params, metadata=metadata)
        
        logger.info(f"Initialized Iron Butterfly strategy: {name}")
    
    def define_universe(self, market_data: MarketData) -> Universe:
        """
        Define the universe of stocks suitable for the Iron Butterfly strategy.
        
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
        
        # Filter by price movement (range-bound behavior)
        for symbol in list(universe.get_symbols()):
            # Get historical prices for range period
            hist_prices = market_data.get_historical_data(symbol, 'close', days=self.params['range_days'])
            if hist_prices is None or len(hist_prices) < self.params['range_days'] * 0.8:  # Allow for some missing days
                universe.remove_symbol(symbol)
                continue
                
            # Calculate price range
            price_range_pct = (hist_prices.max() - hist_prices.min()) / hist_prices.min() * 100
            
            # Remove if price movement is too large
            if price_range_pct > self.params['max_price_movement_pct']:
                universe.remove_symbol(symbol)
                continue
                
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
                continue
                
            # For iron butterflies, we prefer low historical volatility
            if self.params['prefer_low_historical_volatility']:
                hist_vol = vol_signals.get_historical_volatility(symbol, days=30)
                if hist_vol is None:
                    continue
                    
                # Compare to market volatility
                market_vol = vol_signals.get_market_volatility(days=30)
                if market_vol is not None and hist_vol > market_vol * 1.2:  # 20% more volatile than market
                    universe.remove_symbol(symbol)
        
        logger.info(f"Iron Butterfly universe contains {len(universe.get_symbols())} symbols")
        return universe
    
    def check_selection_criteria(self, symbol: str, market_data: MarketData, 
                              option_chains: OptionChains) -> bool:
        """
        Check if the symbol meets the selection criteria for the Iron Butterfly strategy.
        
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
                
                # Check call options
                calls = option_chains.get_calls(symbol, exp)
                if calls.empty:
                    continue
                
                # Check if any options meet volume/OI criteria
                liquid_puts = puts[(puts['volume'] >= self.params['min_option_volume']) & 
                                 (puts['open_interest'] >= self.params['min_option_open_interest'])]
                
                liquid_calls = calls[(calls['volume'] >= self.params['min_option_volume']) & 
                                   (calls['open_interest'] >= self.params['min_option_open_interest'])]
                
                if not liquid_puts.empty and not liquid_calls.empty:
                    # If we've made it here, we have at least one suitable expiration with liquid options
                    return True
        
        logger.debug(f"{symbol} does not have liquid options for Iron Butterfly")
        return False
    
    def select_option_contracts(self, symbol: str, market_data: MarketData,
                             option_chains: OptionChains) -> Dict[str, Any]:
        """
        Select the appropriate option contracts for the Iron Butterfly strategy.
        
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
        
        # Select middle strike (where we'll sell both put and call)
        if self.params['middle_strike_method'] == 'atm':
            # Find the strike closest to current price
            middle_strike = self._find_atm_strike(current_price, puts['strike'].unique())
        else:  # 'delta' method
            # Find the strike with delta closest to 0.50
            middle_strike = self._find_delta_strike(puts, calls, self.params['middle_strike_delta'])
        
        if middle_strike is None:
            logger.error(f"Could not find suitable middle strike for {symbol}")
            return {}
        
        # Get all available strikes
        all_strikes = sorted(list(set(puts['strike'].unique()) | set(calls['strike'].unique())))
        
        # Find index of middle strike
        try:
            middle_idx = all_strikes.index(middle_strike)
        except ValueError:
            logger.error(f"Middle strike {middle_strike} not found in available strikes for {symbol}")
            return {}
        
        # Determine wing width
        wing_width = self.params['wing_width']
        
        # Ensure we have enough strikes on both sides
        if middle_idx < wing_width or middle_idx + wing_width >= len(all_strikes):
            logger.debug(f"Not enough strikes around middle strike for {symbol}")
            return {}
        
        # Select put wing strike (lower)
        lower_wing_strike = all_strikes[middle_idx - wing_width]
        
        # Select call wing strike (higher)
        upper_wing_strike = all_strikes[middle_idx + wing_width]
        
        # Get option contracts
        short_put = puts[puts['strike'] == middle_strike].iloc[0].to_dict()
        long_put = puts[puts['strike'] == lower_wing_strike].iloc[0].to_dict()
        short_call = calls[calls['strike'] == middle_strike].iloc[0].to_dict()
        long_call = calls[calls['strike'] == upper_wing_strike].iloc[0].to_dict()
        
        # Calculate credits and max loss
        put_spread_credit = short_put['bid'] - long_put['ask']
        call_spread_credit = short_call['bid'] - long_call['ask']
        total_credit = put_spread_credit + call_spread_credit
        
        # Verify minimum credit requirements
        if total_credit < self.params['min_credit']:
            logger.debug(f"Total credit ({total_credit:.2f}) is below minimum for {symbol}")
            return {}
        
        # Calculate max loss and profit potential
        wing_width_points = middle_strike - lower_wing_strike  # Same as upper_wing - middle
        max_loss = (wing_width_points * 100) - (total_credit * 100)
        max_profit = total_credit * 100
        
        # Check credit to width ratio
        credit_to_width_ratio = total_credit / wing_width_points
        if credit_to_width_ratio < self.params['min_credit_to_width_ratio']:
            logger.debug(f"Credit to width ratio ({credit_to_width_ratio:.2f}) is below minimum for {symbol}")
            return {}
        
        # Create option identifiers
        short_put_id = f"{symbol}_{selected_expiration}_{middle_strike}_P"
        long_put_id = f"{symbol}_{selected_expiration}_{lower_wing_strike}_P"
        short_call_id = f"{symbol}_{selected_expiration}_{middle_strike}_C"
        long_call_id = f"{symbol}_{selected_expiration}_{upper_wing_strike}_C"
        
        # Return trade details
        return {
            'symbol': symbol,
            'strategy': 'iron_butterfly',
            'expiration': selected_expiration,
            'dte': dte,
            'current_price': current_price,
            'middle_strike': middle_strike,
            'lower_wing_strike': lower_wing_strike,
            'upper_wing_strike': upper_wing_strike,
            'short_put': short_put,
            'long_put': long_put,
            'short_call': short_call,
            'long_call': long_call,
            'short_put_id': short_put_id,
            'long_put_id': long_put_id,
            'short_call_id': short_call_id,
            'long_call_id': long_call_id,
            'put_spread_credit': put_spread_credit,
            'call_spread_credit': call_spread_credit,
            'total_credit': total_credit,
            'wing_width_points': wing_width_points,
            'max_loss': max_loss,
            'max_profit': max_profit,
            'risk_reward_ratio': max_loss / max_profit if max_profit > 0 else float('inf'),
            'timestamp': datetime.now().isoformat()
        }
    
    def calculate_position_size(self, trade_details: Dict[str, Any], 
                             position_sizer: PositionSizer) -> int:
        """
        Calculate the number of iron butterflies to trade.
        
        Args:
            trade_details: Details of the selected iron butterfly
            position_sizer: Position sizer instance
            
        Returns:
            Number of iron butterflies to trade
        """
        # Get portfolio value
        portfolio_value = position_sizer.get_portfolio_value()
        
        # Calculate max risk for this trade
        max_risk_per_trade = portfolio_value * self.params['max_risk_per_trade_percent']
        
        # Calculate risk per butterfly
        risk_per_butterfly = trade_details['max_loss']
        
        # Calculate number of butterflies
        if risk_per_butterfly <= 0:
            return 0
        
        num_butterflies = int(max_risk_per_trade / risk_per_butterfly)
        
        # Check against maximum position size
        max_position_size = portfolio_value * self.params['max_position_size_percent']
        # For butterflies, margin requirement is approximately the max loss
        position_cost = trade_details['max_loss'] * num_butterflies
        
        if position_cost > max_position_size:
            num_butterflies = int(max_position_size / trade_details['max_loss'])
        
        # Ensure at least 1 butterfly if we're trading
        num_butterflies = max(1, num_butterflies)
        
        logger.info(f"Iron Butterfly position size for {trade_details['symbol']}: {num_butterflies} butterflies")
        return num_butterflies
    
    def prepare_entry_orders(self, trade_details: Dict[str, Any], 
                          num_butterflies: int) -> List[Order]:
        """
        Prepare orders for executing the iron butterfly.
        
        Args:
            trade_details: Details of the selected iron butterfly
            num_butterflies: Number of butterflies to trade
            
        Returns:
            List of orders to execute
        """
        if num_butterflies <= 0:
            return []
            
        symbol = trade_details['symbol']
        orders = []
        
        # Create a unique trade ID
        trade_id = f"ifly_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # In a real system, an iron butterfly could be executed as a complex order
        # Here we'll create individual leg orders for demonstration
        
        # 1. Sell Put at middle strike (short put)
        short_put_order = Order(
            symbol=symbol,
            option_symbol=trade_details['short_put_id'],
            order_type=OrderType.LIMIT,
            action=OrderAction.SELL,
            quantity=num_butterflies,
            limit_price=trade_details['short_put']['bid'],
            trade_id=trade_id,
            order_details={
                'strategy': 'iron_butterfly',
                'leg': 'short_put',
                'expiration': trade_details['expiration'],
                'strike': trade_details['middle_strike'],
                'trade_details': trade_details
            }
        )
        orders.append(short_put_order)
        
        # 2. Buy Put at lower strike (long put)
        long_put_order = Order(
            symbol=symbol,
            option_symbol=trade_details['long_put_id'],
            order_type=OrderType.LIMIT,
            action=OrderAction.BUY,
            quantity=num_butterflies,
            limit_price=trade_details['long_put']['ask'],
            trade_id=trade_id,
            order_details={
                'strategy': 'iron_butterfly',
                'leg': 'long_put',
                'expiration': trade_details['expiration'],
                'strike': trade_details['lower_wing_strike'],
                'trade_details': trade_details
            }
        )
        orders.append(long_put_order)
        
        # 3. Sell Call at middle strike (short call)
        short_call_order = Order(
            symbol=symbol,
            option_symbol=trade_details['short_call_id'],
            order_type=OrderType.LIMIT,
            action=OrderAction.SELL,
            quantity=num_butterflies,
            limit_price=trade_details['short_call']['bid'],
            trade_id=trade_id,
            order_details={
                'strategy': 'iron_butterfly',
                'leg': 'short_call',
                'expiration': trade_details['expiration'],
                'strike': trade_details['middle_strike'],
                'trade_details': trade_details
            }
        )
        orders.append(short_call_order)
        
        # 4. Buy Call at upper strike (long call)
        long_call_order = Order(
            symbol=symbol,
            option_symbol=trade_details['long_call_id'],
            order_type=OrderType.LIMIT,
            action=OrderAction.BUY,
            quantity=num_butterflies,
            limit_price=trade_details['long_call']['ask'],
            trade_id=trade_id,
            order_details={
                'strategy': 'iron_butterfly',
                'leg': 'long_call',
                'expiration': trade_details['expiration'],
                'strike': trade_details['upper_wing_strike'],
                'trade_details': trade_details
            }
        )
        orders.append(long_call_order)
        
        logger.info(f"Created Iron Butterfly entry orders for {symbol}: {num_butterflies} butterflies")
        return orders
    
    def check_exit_conditions(self, position: Dict[str, Any], 
                          market_data: MarketData,
                          option_chains: OptionChains) -> bool:
        """
        Check if exit conditions are met for an existing Iron Butterfly position.
        
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
            logger.info(f"Exiting {symbol} Iron Butterfly: DTE {current_dte} <= threshold {self.params['dte_exit_threshold']}")
            return True
        
        # Get current price of underlying
        current_price = market_data.get_latest_price(symbol)
        if current_price is None:
            logger.error(f"Unable to get current price for {symbol}")
            return False
        
        # Check if price is getting close to wing strikes (risk management)
        lower_wing_strike = trade_details['lower_wing_strike']
        upper_wing_strike = trade_details['upper_wing_strike']
        middle_strike = trade_details['middle_strike']
        
        # Calculate price distance from wings as percentage of wing width
        wing_width = middle_strike - lower_wing_strike  # Same as upper_wing - middle
        
        if current_price <= lower_wing_strike:
            # Price at or below lower wing - max loss on put side
            logger.info(f"Exiting {symbol} Iron Butterfly: Price at or below lower wing strike")
            return True
        
        if current_price >= upper_wing_strike:
            # Price at or above upper wing - max loss on call side
            logger.info(f"Exiting {symbol} Iron Butterfly: Price at or above upper wing strike")
            return True
        
        # Check current option prices and P/L
        try:
            # Get current option prices
            short_put_price = option_chains.get_option_price(
                symbol=symbol,
                expiration=expiration,
                strike=middle_strike,
                option_type='put',
                price_type='ask'  # Ask price to buy back short position
            )
            
            long_put_price = option_chains.get_option_price(
                symbol=symbol,
                expiration=expiration,
                strike=lower_wing_strike,
                option_type='put',
                price_type='bid'  # Bid price to sell long position
            )
            
            short_call_price = option_chains.get_option_price(
                symbol=symbol,
                expiration=expiration,
                strike=middle_strike,
                option_type='call',
                price_type='ask'  # Ask price to buy back short position
            )
            
            long_call_price = option_chains.get_option_price(
                symbol=symbol,
                expiration=expiration,
                strike=upper_wing_strike,
                option_type='call',
                price_type='bid'  # Bid price to sell long position
            )
            
            if None in [short_put_price, long_put_price, short_call_price, long_call_price]:
                logger.error(f"Missing option price data for {symbol}")
                return False
            
            # Calculate current cost to close position
            current_cost = short_put_price - long_put_price + short_call_price - long_call_price
            
            # Calculate P/L as percentage of max profit
            initial_credit = trade_details['total_credit']
            current_profit = initial_credit - current_cost
            profit_percent = (current_profit / initial_credit) * 100
            
            # Check profit target
            if profit_percent >= self.params['profit_target_percent']:
                logger.info(f"Exiting {symbol} Iron Butterfly: Profit target reached ({profit_percent:.2f}%)")
                return True
            
            # Check max loss
            loss_percent = -profit_percent
            if loss_percent >= self.params['max_loss_percent']:
                logger.info(f"Exiting {symbol} Iron Butterfly: Max loss reached ({loss_percent:.2f}%)")
                return True
            
        except Exception as e:
            logger.error(f"Error checking exit conditions for {symbol}: {str(e)}")
            return False
        
        return False
    
    def prepare_exit_orders(self, position: Dict[str, Any]) -> List[Order]:
        """
        Prepare orders for exiting the Iron Butterfly position.
        
        Args:
            position: Current position details
            
        Returns:
            List of orders to execute
        """
        if 'trade_details' not in position:
            logger.error("Invalid position data for exit orders")
            return []
        
        trade_details = position['trade_details']
        symbol = trade_details['symbol']
        num_butterflies = position.get('quantity', 0)
        
        if num_butterflies <= 0:
            return []
        
        orders = []
        
        # Create a unique trade ID
        trade_id = f"ifly_exit_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 1. Buy to close short put
        buy_put_order = Order(
            symbol=symbol,
            option_symbol=trade_details['short_put_id'],
            order_type=OrderType.MARKET,
            action=OrderAction.BUY,
            quantity=num_butterflies,
            trade_id=trade_id,
            order_details={
                'strategy': 'iron_butterfly',
                'leg': 'short_put_exit',
                'expiration': trade_details['expiration'],
                'strike': trade_details['middle_strike'],
                'original_trade_id': position.get('trade_id', '')
            }
        )
        orders.append(buy_put_order)
        
        # 2. Sell to close long put
        sell_put_order = Order(
            symbol=symbol,
            option_symbol=trade_details['long_put_id'],
            order_type=OrderType.MARKET,
            action=OrderAction.SELL,
            quantity=num_butterflies,
            trade_id=trade_id,
            order_details={
                'strategy': 'iron_butterfly',
                'leg': 'long_put_exit',
                'expiration': trade_details['expiration'],
                'strike': trade_details['lower_wing_strike'],
                'original_trade_id': position.get('trade_id', '')
            }
        )
        orders.append(sell_put_order)
        
        # 3. Buy to close short call
        buy_call_order = Order(
            symbol=symbol,
            option_symbol=trade_details['short_call_id'],
            order_type=OrderType.MARKET,
            action=OrderAction.BUY,
            quantity=num_butterflies,
            trade_id=trade_id,
            order_details={
                'strategy': 'iron_butterfly',
                'leg': 'short_call_exit',
                'expiration': trade_details['expiration'],
                'strike': trade_details['middle_strike'],
                'original_trade_id': position.get('trade_id', '')
            }
        )
        orders.append(buy_call_order)
        
        # 4. Sell to close long call
        sell_call_order = Order(
            symbol=symbol,
            option_symbol=trade_details['long_call_id'],
            order_type=OrderType.MARKET,
            action=OrderAction.SELL,
            quantity=num_butterflies,
            trade_id=trade_id,
            order_details={
                'strategy': 'iron_butterfly',
                'leg': 'long_call_exit',
                'expiration': trade_details['expiration'],
                'strike': trade_details['upper_wing_strike'],
                'original_trade_id': position.get('trade_id', '')
            }
        )
        orders.append(sell_call_order)
        
        logger.info(f"Created Iron Butterfly exit orders for {symbol}: {num_butterflies} butterflies")
        return orders
    
    def prepare_adjustment_orders(self, position: Dict[str, Any],
                               market_data: MarketData,
                               option_chains: OptionChains) -> List[Order]:
        """
        Prepare orders for adjusting the Iron Butterfly position if necessary.
        
        Args:
            position: Current position details
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
        num_butterflies = position.get('quantity', 0)
        
        if num_butterflies <= 0:
            return []
        
        # Get current price
        current_price = market_data.get_latest_price(symbol)
        if current_price is None:
            logger.error(f"Unable to get current price for {symbol}")
            return []
        
        # Get position details
        middle_strike = trade_details['middle_strike']
        lower_wing_strike = trade_details['lower_wing_strike']
        upper_wing_strike = trade_details['upper_wing_strike']
        expiration = trade_details['expiration']
        
        # Calculate adjustment thresholds
        wing_width = middle_strike - lower_wing_strike
        adjustment_threshold = self.params['adjustment_threshold']
        
        lower_adjustment_price = lower_wing_strike + (wing_width * adjustment_threshold)
        upper_adjustment_price = upper_wing_strike - (wing_width * adjustment_threshold)
        
        orders = []
        
        # Check if price is nearing the lower wing (bearish move)
        if current_price <= lower_adjustment_price:
            try:
                # For a bearish move, we can consider rolling the entire butterfly lower
                # or adding a put ratio spread to offset potential losses
                
                # Here we'll implement a simplified adjustment: 
                # Buy additional long puts at a lower strike to reduce potential losses
                
                # Get available strikes below the current lower wing
                puts = option_chains.get_puts(symbol, expiration)
                if puts.empty:
                    return []
                
                # Find the next strike down
                available_strikes = sorted(puts['strike'].unique())
                lower_strikes = [s for s in available_strikes if s < lower_wing_strike]
                
                if not lower_strikes:
                    logger.debug(f"No lower strikes available for {symbol} adjustment")
                    return []
                
                new_protective_strike = max(lower_strikes)  # The highest of the lower strikes
                
                # Get the option contract
                protective_put = puts[puts['strike'] == new_protective_strike].iloc[0]
                protective_put_id = f"{symbol}_{expiration}_{new_protective_strike}_P"
                
                # Create the order for the additional put
                protective_put_order = Order(
                    symbol=symbol,
                    option_symbol=protective_put_id,
                    order_type=OrderType.LIMIT,
                    action=OrderAction.BUY,
                    quantity=num_butterflies,
                    limit_price=protective_put['ask'],
                    trade_id=f"ifly_adjust_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    order_details={
                        'strategy': 'iron_butterfly',
                        'leg': 'protective_put',
                        'adjustment': True,
                        'expiration': expiration,
                        'strike': new_protective_strike,
                        'original_trade_id': position.get('trade_id', '')
                    }
                )
                orders.append(protective_put_order)
                
                logger.info(f"Created downside adjustment orders for {symbol} Iron Butterfly")
            
            except Exception as e:
                logger.error(f"Error creating downside adjustment for {symbol}: {str(e)}")
                return []
        
        # Check if price is nearing the upper wing (bullish move)
        elif current_price >= upper_adjustment_price:
            try:
                # For a bullish move, we can consider rolling the entire butterfly higher
                # or adding a call ratio spread to offset potential losses
                
                # Here we'll implement a simplified adjustment:
                # Buy additional long calls at a higher strike to reduce potential losses
                
                # Get available strikes above the current upper wing
                calls = option_chains.get_calls(symbol, expiration)
                if calls.empty:
                    return []
                
                # Find the next strike up
                available_strikes = sorted(calls['strike'].unique())
                higher_strikes = [s for s in available_strikes if s > upper_wing_strike]
                
                if not higher_strikes:
                    logger.debug(f"No higher strikes available for {symbol} adjustment")
                    return []
                
                new_protective_strike = min(higher_strikes)  # The lowest of the higher strikes
                
                # Get the option contract
                protective_call = calls[calls['strike'] == new_protective_strike].iloc[0]
                protective_call_id = f"{symbol}_{expiration}_{new_protective_strike}_C"
                
                # Create the order for the additional call
                protective_call_order = Order(
                    symbol=symbol,
                    option_symbol=protective_call_id,
                    order_type=OrderType.LIMIT,
                    action=OrderAction.BUY,
                    quantity=num_butterflies,
                    limit_price=protective_call['ask'],
                    trade_id=f"ifly_adjust_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    order_details={
                        'strategy': 'iron_butterfly',
                        'leg': 'protective_call',
                        'adjustment': True,
                        'expiration': expiration,
                        'strike': new_protective_strike,
                        'original_trade_id': position.get('trade_id', '')
                    }
                )
                orders.append(protective_call_order)
                
                logger.info(f"Created upside adjustment orders for {symbol} Iron Butterfly")
            
            except Exception as e:
                logger.error(f"Error creating upside adjustment for {symbol}: {str(e)}")
                return []
        
        return orders
    
    def get_optimization_params(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            'target_dte': [21, 30, 45],
            'wing_width': [1, 2, 3],
            'middle_strike_method': ['atm', 'delta'],
            'middle_strike_delta': [0.45, 0.50, 0.55],
            'min_iv_rank': [25, 30, 35, 40],
            'max_iv_rank': [70, 80, 90],
            'profit_target_percent': [30, 40, 50, 60],
            'max_loss_percent': [80, 100, 120],
            'dte_exit_threshold': [5, 7, 10, 14],
            'adjustment_threshold': [0.25, 0.35, 0.45]
        }
    
    def evaluate_performance(self, backtest_results: Dict[str, Any]) -> float:
        """
        Evaluate the performance of the Iron Butterfly strategy in a backtest.
        
        Args:
            backtest_results: Results from backtesting
            
        Returns:
            Performance score (higher is better)
        """
        # Extract key metrics
        total_return = backtest_results.get('total_return', 0)
        max_drawdown = backtest_results.get('max_drawdown', 0)
        sharpe_ratio = backtest_results.get('sharpe_ratio', 0)
        win_rate = backtest_results.get('win_rate', 0)
        avg_trade_duration = backtest_results.get('avg_trade_duration', 0)  # in days
        
        # Avoid division by zero
        if max_drawdown == 0:
            max_drawdown = 0.0001
        
        # Calculate risk-adjusted return
        return_to_drawdown = total_return / abs(max_drawdown)
        
        # For Iron Butterfly, we prioritize win rate and shorter trade durations
        # due to the strategy's narrow profit zone
        
        # Normalize trade duration (shorter is better)
        duration_score = 1.0 / (1.0 + avg_trade_duration / 30.0)  # Normalizes around 30 days
        
        # Weighted score calculation
        score = (
            0.35 * return_to_drawdown +  # Higher weight on risk-adjusted return
            0.25 * sharpe_ratio +         # Moderate weight on risk-adjusted performance
            0.25 * win_rate +             # High weight on win rate for premium collection
            0.15 * duration_score         # Some weight on trade duration
        )
        
        return score
    
    # Helper methods
    def _find_atm_strike(self, current_price: float, available_strikes: List[float]) -> Optional[float]:
        """
        Find the strike price closest to the current price.
        
        Args:
            current_price: Current price of the underlying
            available_strikes: List of available strike prices
            
        Returns:
            Strike price closest to current price
        """
        if not available_strikes:
            return None
        
        # Find strike closest to current price
        return min(available_strikes, key=lambda s: abs(s - current_price))
    
    def _find_delta_strike(self, puts: pd.DataFrame, calls: pd.DataFrame, target_delta: float) -> Optional[float]:
        """
        Find a strike with delta closest to the target.
        
        Args:
            puts: DataFrame of put options
            calls: DataFrame of call options
            target_delta: Target delta value (typically around 0.50 for ATM)
            
        Returns:
            Strike with delta closest to target
        """
        if 'delta' not in puts.columns or 'delta' not in calls.columns:
            logger.error("Delta information not available in options data")
            return None
        
        # For put options, convert to absolute delta for comparison
        puts['abs_delta'] = puts['delta'].abs()
        
        # Find puts with delta close to target
        puts_filtered = puts.copy()
        puts_filtered['delta_diff'] = abs(puts_filtered['abs_delta'] - target_delta)
        best_put = puts_filtered.sort_values('delta_diff').iloc[0]
        
        # Find calls with delta close to target
        calls_filtered = calls.copy()
        calls_filtered['delta_diff'] = abs(calls_filtered['delta'] - target_delta)
        best_call = calls_filtered.sort_values('delta_diff').iloc[0]
        
        # Return the better match between put and call
        if best_put['delta_diff'] <= best_call['delta_diff']:
            return best_put['strike']
        else:
            return best_call['strike'] 