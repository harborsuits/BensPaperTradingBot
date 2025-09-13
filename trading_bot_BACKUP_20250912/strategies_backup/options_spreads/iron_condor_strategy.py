#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iron Condor Strategy Module

This module implements an iron condor options strategy that profits from 
neutral market movements by collecting premium with defined risk on both sides.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Tuple, Optional, Set

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

class IronCondorStrategy(StrategyOptimizable):
    """
    Iron Condor Strategy
    
    This strategy involves simultaneously selling an OTM put spread and an OTM call spread
    with the same expiration date, creating a range where the strategy profits if the
    underlying price stays between the short strikes at expiration.
    
    Key characteristics:
    - Limited risk (max loss = wing width - net premium received)
    - Limited profit (max profit = net premium received)
    - Benefits from time decay
    - Profits from neutral to range-bound price movement
    - Defined risk-reward ratio on both sides of the market
    """
    
    # ======================== 1. STRATEGY PHILOSOPHY ========================
    # Collect premium in neutral markets by selling an out-of-the-money put spread and 
    # an out-of-the-money call spread simultaneously, defining risk on both sides while 
    # profiting from time decay and rangebound behavior.
    
    # ======================== 2. DEFAULT PARAMETERS ========================
    DEFAULT_PARAMS = {
        # Strategy identification
        'strategy_name': 'iron_condor',
        'strategy_version': '1.0.0',
        
        # Universe selection criteria
        'min_stock_price': 50.0,              # Minimum stock price to consider
        'max_stock_price': 2000.0,            # Maximum stock price to consider
        'min_option_volume': 500,             # Minimum option volume
        'min_option_open_interest': 1000,     # Minimum option open interest
        'min_adv': 1000000,                   # Minimum average daily volume (1M)
        'max_bid_ask_spread_pct': 0.001,      # Maximum bid-ask spread as % of price (0.1%)
        
        # Volatility parameters
        'min_iv_rank': 30,                    # Minimum IV rank (30%)
        'max_iv_rank': 60,                    # Maximum IV rank (60%)
        
        # Market condition parameters
        'range_days': 30,                     # Days to check for range-bound behavior
        'max_trend_strength': 0.1,            # Maximum trend strength (10%)
        
        # Option parameters
        'target_dte': 35,                     # Target days to expiration (25-45 DTE)
        'min_dte': 25,                        # Minimum days to expiration
        'max_dte': 45,                        # Maximum days to expiration
        'entry_dte_min': 20,                  # Minimum DTE for entry
        
        # Strike selection
        'short_put_buffer_pct': 0.06,         # OTM buffer % for short put (5-8%)
        'short_call_buffer_pct': 0.06,        # OTM buffer % for short call (5-8%)
        'short_put_delta': 0.20,              # Target delta for short put (0.15-0.25)
        'short_call_delta': 0.20,             # Target delta for short call (0.15-0.25)
        'strike_selection_method': 'delta',   # 'delta' or 'otm_percentage'
        'wing_width': 2,                      # Number of strikes between short and long (2-3)
        
        # Entry and credit parameters
        'min_credit': 0.50,                   # Minimum credit to collect (per condor)
        'min_credit_per_width_pct': 0.10,     # Minimum credit as % of wing width (10%)
        
        # Risk management parameters
        'max_position_size_percent': 0.02,    # Maximum position size as % of portfolio (1-2%)
        'max_num_positions': 5,               # Maximum number of concurrent positions (3-5)
        'max_risk_per_trade': 0.01,           # Maximum risk per trade as % of portfolio
        'max_margin_usage_percent': 0.10,     # Maximum margin usage as % of account (10%)
        
        # Exit parameters
        'profit_target_percent': 60,          # Exit at this percentage of max credit (50-75%)
        'stop_loss_multiplier': 1.5,          # Stop loss at 1.5x max credit
        'dte_exit_threshold': 10,             # Exit when DTE reaches this value (7-10 days)
        
        # Adjustment parameters
        'enable_adjustments': False,          # Whether to adjust positions
        'adjustment_threshold': 0.70,         # Adjust when short strike is breached by 70%
    }
    
    # ======================== 3. UNIVERSE DEFINITION ========================
    def define_universe(self, market_data: MarketData) -> Universe:
        """Define the universe of stocks to trade based on criteria."""
        universe = Universe()
        
        # Filter by price range
        price_df = market_data.get_latest_prices()
        filtered_symbols = price_df[(price_df['close'] >= self.params['min_stock_price']) & 
                                   (price_df['close'] <= self.params['max_stock_price'])].index.tolist()
        
        universe.add_symbols(filtered_symbols)
        
        # Filter by volume and liquidity criteria
        option_chains = OptionChains()
        for symbol in universe.get_symbols():
            # Check ADV (Average Daily Volume)
            if not self._check_adv(symbol, market_data):
                universe.remove_symbol(symbol)
                continue
                
            # Check if options meet volume and open interest criteria
            if not self._check_option_liquidity(symbol, option_chains):
                universe.remove_symbol(symbol)
                continue
                
            # Check option bid-ask spreads
            if not self._check_option_spreads(symbol, option_chains):
                universe.remove_symbol(symbol)
                continue
        
        # Filter by market condition (range-bound)
        for symbol in list(universe.get_symbols()):
            if not self._is_range_bound(symbol, market_data):
                universe.remove_symbol(symbol)
                continue
        
        logger.info(f"Iron Condor universe contains {len(universe.get_symbols())} symbols")
        return universe
    
    # ======================== 4. SELECTION CRITERIA ========================
    def check_selection_criteria(self, symbol: str, market_data: MarketData, 
                                option_chains: OptionChains) -> bool:
        """
        Check if the symbol meets the selection criteria for the strategy.
        
        Parameters:
            symbol: Symbol to check
            market_data: Market data instance
            option_chains: Option chains instance
            
        Returns:
            bool: True if symbol meets all criteria, False otherwise
        """
        # Check IV rank is in the desired range
        vol_signals = VolatilitySignals(market_data)
        iv_rank = vol_signals.get_iv_rank(symbol)
        
        if iv_rank is None:
            logger.debug(f"{symbol} has no IV rank data")
            return False
            
        if not (self.params['min_iv_rank'] <= iv_rank <= self.params['max_iv_rank']):
            logger.debug(f"{symbol} IV rank {iv_rank:.2f}% outside range")
            return False
        
        # Check for range-bound behavior
        if not self._is_range_bound(symbol, market_data):
            logger.debug(f"{symbol} is not range-bound")
            return False
        
        # Check if we have appropriate option chains
        try:
            chains = option_chains.get_option_chain(symbol)
            if chains is None or chains.empty:
                logger.debug(f"{symbol} has no option chains available")
                return False
                
            # Check if we have options with suitable expiration
            available_expirations = chains['expiration_date'].unique()
            valid_expiration = False
            
            for exp in available_expirations:
                exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                dte = (exp_date - date.today()).days
                
                if self.params['min_dte'] <= dte <= self.params['max_dte']:
                    valid_expiration = True
                    break
                    
            if not valid_expiration:
                logger.debug(f"{symbol} has no options in the desired DTE range")
                return False
                
        except Exception as e:
            logger.error(f"Error checking option chains for {symbol}: {str(e)}")
            return False
            
        logger.info(f"{symbol} meets all selection criteria for iron condor")
        return True
    
    # ======================== 5. OPTION SELECTION ========================
    def select_option_contract(self, symbol: str, market_data: MarketData,
                              option_chains: OptionChains) -> Dict[str, Any]:
        """
        Select the appropriate option contracts for the iron condor.
        
        Parameters:
            symbol: The stock symbol
            market_data: Market data instance
            option_chains: Option chains instance
            
        Returns:
            Dict with selected option contracts
        """
        # Get current price
        current_price = market_data.get_latest_price(symbol)
        if current_price is None:
            logger.error(f"Unable to get current price for {symbol}")
            return {}
            
        # Find appropriate expiration
        target_expiration = self._select_expiration(symbol, option_chains)
        if not target_expiration:
            logger.error(f"No suitable expiration found for {symbol}")
            return {}
            
        # Get options for the selected expiration
        put_options = option_chains.get_puts(symbol, target_expiration)
        call_options = option_chains.get_calls(symbol, target_expiration)
        
        if put_options.empty or call_options.empty:
            logger.error(f"No options available for {symbol} at {target_expiration}")
            return {}
        
        # Get available strikes for reference
        all_strikes = sorted(list(set(put_options['strike'].unique()).union(set(call_options['strike'].unique()))))
        strike_diff = all_strikes[1] - all_strikes[0] if len(all_strikes) >= 2 else 1.0
            
        # Select strikes based on the configured method
        if self.params['strike_selection_method'] == 'delta':
            short_put, long_put = self._select_put_strikes_by_delta(put_options, current_price, all_strikes)
            short_call, long_call = self._select_call_strikes_by_delta(call_options, current_price, all_strikes)
        else:  # Default to otm_percentage
            short_put, long_put = self._select_put_strikes_by_otm_percentage(put_options, current_price, all_strikes)
            short_call, long_call = self._select_call_strikes_by_otm_percentage(call_options, current_price, all_strikes)
            
        if not short_put or not long_put or not short_call or not long_call:
            logger.error(f"Could not select appropriate strikes for {symbol}")
            return {}
            
        # Calculate the credit and max profit/loss
        put_credit = short_put['bid'] - long_put['ask']
        call_credit = short_call['bid'] - long_call['ask']
        total_credit = put_credit + call_credit
        
        # Calculate max loss for each wing
        put_wing_width = short_put['strike'] - long_put['strike']
        call_wing_width = long_call['strike'] - short_call['strike']
        put_max_loss = put_wing_width - put_credit
        call_max_loss = call_wing_width - call_credit
        
        # Overall max loss is the greater of the two wings' max loss
        max_loss = max(put_max_loss, call_max_loss)
        
        # Check if the credit meets minimum requirements
        if total_credit < self.params['min_credit']:
            logger.debug(f"Credit of {total_credit:.2f} for {symbol} is below minimum {self.params['min_credit']}")
            return {}
            
        # Check if credit as percentage of width is within acceptable range
        avg_wing_width = (put_wing_width + call_wing_width) / 2
        credit_percent = total_credit / avg_wing_width
        
        if credit_percent < self.params['min_credit_per_width_pct']:
            logger.debug(f"Credit percentage {credit_percent:.2f}% for {symbol} is below minimum")
            return {}
            
        # Return the selected options and trade details
        return {
            'symbol': symbol,
            'strategy': 'iron_condor',
            'expiration': target_expiration,
            'dte': (datetime.strptime(target_expiration, '%Y-%m-%d').date() - date.today()).days,
            'short_put': short_put,
            'long_put': long_put,
            'short_call': short_call,
            'long_call': long_call,
            'short_put_contract': f"{symbol}_{target_expiration}_{short_put['strike']}_P",
            'long_put_contract': f"{symbol}_{target_expiration}_{long_put['strike']}_P",
            'short_call_contract': f"{symbol}_{target_expiration}_{short_call['strike']}_C",
            'long_call_contract': f"{symbol}_{target_expiration}_{long_call['strike']}_C",
            'put_credit': put_credit,
            'call_credit': call_credit,
            'total_credit': total_credit,
            'put_wing_width': put_wing_width,
            'call_wing_width': call_wing_width,
            'max_loss': max_loss,
            'risk_reward_ratio': max_loss / total_credit if total_credit > 0 else 0,
            'price': current_price,
            'timestamp': datetime.now().isoformat()
        }
    
    # ======================== 6. POSITION SIZING ========================
    def calculate_position_size(self, trade_details: Dict[str, Any], 
                               position_sizer: PositionSizer) -> int:
        """
        Calculate the number of condors to trade based on risk parameters.
        
        Parameters:
            trade_details: Details of the selected iron condor
            position_sizer: Position sizer instance
            
        Returns:
            int: Number of condors to trade
        """
        # Calculate max risk per condor
        max_loss_per_condor = trade_details['max_loss'] * 100  # Convert to dollars (per contract)
        
        # Get portfolio value
        portfolio_value = position_sizer.get_portfolio_value()
        
        # Calculate max risk for this trade based on portfolio percentage
        max_risk_dollars = portfolio_value * self.params['max_risk_per_trade']
        
        # Calculate number of condors
        if max_loss_per_condor <= 0:
            return 0
            
        num_condors = int(max_risk_dollars / max_loss_per_condor)
        
        # Check against max position size
        max_position_dollars = portfolio_value * self.params['max_position_size_percent']
        position_risk = max_loss_per_condor * num_condors
        
        if position_risk > max_position_dollars:
            num_condors = int(max_position_dollars / max_loss_per_condor)
            
        # Check margin requirements (approximate)
        # For iron condor, margin is typically the width of the largest wing minus total credit
        margin_req = max(trade_details['put_wing_width'], trade_details['call_wing_width']) * 100 * num_condors
        max_margin = portfolio_value * self.params['max_margin_usage_percent']
        
        if margin_req > max_margin:
            num_condors = int(max_margin / (max(trade_details['put_wing_width'], trade_details['call_wing_width']) * 100))
            
        # Ensure at least 1 condor if we're trading
        num_condors = max(1, num_condors)
        
        logger.info(f"Iron Condor position size for {trade_details['symbol']}: {num_condors} condors")
        return num_condors
    
    # ======================== 7. ENTRY EXECUTION ========================
    def prepare_entry_orders(self, trade_details: Dict[str, Any], 
                            num_condors: int) -> List[Order]:
        """
        Prepare orders for executing the iron condor.
        
        Parameters:
            trade_details: Details of the selected iron condor
            num_condors: Number of condors to trade
            
        Returns:
            List of orders to execute
        """
        if num_condors <= 0:
            return []
            
        symbol = trade_details['symbol']
        orders = []
        
        # Ideal implementation would use a combo/spread order for the full iron condor
        # Example for combo order (if supported by broker API):
        # condor_order = Order(
        #     symbol=symbol,
        #     order_type=OrderType.LIMIT,
        #     action=OrderAction.SELL,  # Selling an iron condor
        #     quantity=num_condors,
        #     limit_price=trade_details['total_credit'],
        #     option_spread_type="IRON_CONDOR",
        #     ...other details...
        # )
        # orders.append(condor_order)
        
        # Since we're implementing individual leg orders:
        
        # Create short put order (sell to open)
        short_put_order = Order(
            symbol=symbol,
            option_symbol=trade_details['short_put_contract'],
            order_type=OrderType.LIMIT,
            action=OrderAction.SELL,
            quantity=num_condors,
            limit_price=trade_details['short_put']['bid'],
            trade_id=f"ic_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            order_details={
                'strategy': 'iron_condor',
                'leg': 'short_put',
                'expiration': trade_details['expiration'],
                'strike': trade_details['short_put']['strike'],
                'trade_details': trade_details
            }
        )
        orders.append(short_put_order)
        
        # Create long put order (buy to open)
        long_put_order = Order(
            symbol=symbol,
            option_symbol=trade_details['long_put_contract'],
            order_type=OrderType.LIMIT,
            action=OrderAction.BUY,
            quantity=num_condors,
            limit_price=trade_details['long_put']['ask'],
            trade_id=f"ic_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            order_details={
                'strategy': 'iron_condor',
                'leg': 'long_put',
                'expiration': trade_details['expiration'],
                'strike': trade_details['long_put']['strike'],
                'trade_details': trade_details
            }
        )
        orders.append(long_put_order)
        
        # Create short call order (sell to open)
        short_call_order = Order(
            symbol=symbol,
            option_symbol=trade_details['short_call_contract'],
            order_type=OrderType.LIMIT,
            action=OrderAction.SELL,
            quantity=num_condors,
            limit_price=trade_details['short_call']['bid'],
            trade_id=f"ic_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            order_details={
                'strategy': 'iron_condor',
                'leg': 'short_call',
                'expiration': trade_details['expiration'],
                'strike': trade_details['short_call']['strike'],
                'trade_details': trade_details
            }
        )
        orders.append(short_call_order)
        
        # Create long call order (buy to open)
        long_call_order = Order(
            symbol=symbol,
            option_symbol=trade_details['long_call_contract'],
            order_type=OrderType.LIMIT,
            action=OrderAction.BUY,
            quantity=num_condors,
            limit_price=trade_details['long_call']['ask'],
            trade_id=f"ic_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            order_details={
                'strategy': 'iron_condor',
                'leg': 'long_call',
                'expiration': trade_details['expiration'],
                'strike': trade_details['long_call']['strike'],
                'trade_details': trade_details
            }
        )
        orders.append(long_call_order)
        
        logger.info(f"Created iron condor orders for {symbol}: {num_condors} condors")
        return orders 

    # ======================== 8. EXIT CONDITIONS ========================
    def check_exit_conditions(self, position: Dict[str, Any], 
                             market_data: MarketData) -> bool:
        """
        Check if exit conditions are met for an existing position.
        
        Parameters:
            position: The current position
            market_data: Market data instance
            
        Returns:
            bool: True if exit conditions are met
        """
        if not position or 'trade_details' not in position:
            logger.error("Invalid position data for exit check")
            return False
            
        trade_details = position.get('trade_details', {})
        symbol = trade_details.get('symbol')
        
        if not symbol:
            return False
            
        # Check if DTE is below threshold
        current_dte = trade_details.get('dte', 0)
        if current_dte <= self.params['dte_exit_threshold']:
            logger.info(f"Exiting {symbol} iron condor: DTE {current_dte} <= threshold {self.params['dte_exit_threshold']}")
            return True
            
        # Get current price of the underlying
        current_price = market_data.get_latest_price(symbol)
        if current_price is None:
            return False
            
        # Check for profit target
        current_value = position.get('current_value', 0)
        entry_value = position.get('entry_value', 0)
        
        if entry_value > 0:
            # For a credit spread, entry_value is negative (credit received)
            # and current_value is the cost to close (debit paid)
            max_credit = trade_details.get('total_credit', 0) * 100  # Convert to dollars
            profit = max_credit - abs(current_value)
            
            if max_credit > 0:
                profit_pct = (profit / max_credit) * 100
                
                # If we've reached our target profit percentage
                if profit_pct >= self.params['profit_target_percent']:
                    logger.info(f"Exiting {symbol} iron condor: Profit target reached {profit_pct:.2f}%")
                    return True
                
            # Check for stop loss - if debit to close exceeds credit * stop_loss_multiplier
            if abs(current_value) > max_credit * self.params['stop_loss_multiplier']:
                logger.info(f"Exiting {symbol} iron condor: Stop loss triggered")
                return True
            
        # Check if price is approaching short strikes (within 5%)
        short_put_strike = trade_details.get('short_put', {}).get('strike', 0)
        short_call_strike = trade_details.get('short_call', {}).get('strike', 0)
        
        # Calculate distance to short strikes as percentage
        if short_put_strike > 0:
            distance_to_put = (current_price - short_put_strike) / short_put_strike
            if distance_to_put < 0.03:  # Within 3% of short put
                logger.info(f"Exiting {symbol} iron condor: Price approaching short put strike")
                return True
                
        if short_call_strike > 0:
            distance_to_call = (short_call_strike - current_price) / short_call_strike
            if distance_to_call < 0.03:  # Within 3% of short call
                logger.info(f"Exiting {symbol} iron condor: Price approaching short call strike")
                return True
        
        return False
    
    # ======================== 9. EXIT EXECUTION ========================
    def prepare_exit_orders(self, position: Dict[str, Any]) -> List[Order]:
        """
        Prepare orders to close an existing position.
        
        Parameters:
            position: The position to close
            
        Returns:
            List of orders to execute
        """
        orders = []
        
        if not position or 'legs' not in position:
            logger.error("Invalid position data for exit orders")
            return orders
            
        legs = position.get('legs', [])
        
        for leg in legs:
            if not leg or 'status' not in leg or leg['status'] != OrderStatus.FILLED:
                continue
                
            # Determine action to close the position
            close_action = OrderAction.BUY if leg.get('action') == OrderAction.SELL else OrderAction.SELL
            
            close_order = Order(
                symbol=leg.get('symbol', ''),
                option_symbol=leg.get('option_symbol', ''),
                order_type=OrderType.MARKET,
                action=close_action,
                quantity=leg.get('quantity', 0),
                trade_id=f"close_{leg.get('trade_id', '')}",
                order_details={
                    'strategy': 'iron_condor',
                    'leg': 'exit_' + leg.get('order_details', {}).get('leg', ''),
                    'closing_order': True,
                    'original_order_id': leg.get('order_id', '')
                }
            )
            orders.append(close_order)
            
        logger.info(f"Created exit orders for iron condor position")
        return orders
    
    # ======================== 10. ADJUSTMENT EXECUTION ========================
    def prepare_adjustment_orders(self, position: Dict[str, Any], 
                                market_data: MarketData,
                                option_chains: OptionChains) -> List[Order]:
        """
        Prepare orders to adjust an existing position that's approaching a wing breach.
        
        Parameters:
            position: The position to adjust
            market_data: Market data instance
            option_chains: Option chains instance
            
        Returns:
            List of orders to execute the adjustment
        """
        if not self.params['enable_adjustments']:
            return []
            
        if not position or 'trade_details' not in position:
            logger.error("Invalid position data for adjustment")
            return []
            
        trade_details = position.get('trade_details', {})
        symbol = trade_details.get('symbol')
        expiration = trade_details.get('expiration')
        
        if not symbol or not expiration:
            return []
        
        # Get current price
        current_price = market_data.get_latest_price(symbol)
        if current_price is None:
            return []
        
        # Check which wing needs adjustment (put or call)
        short_put_strike = trade_details.get('short_put', {}).get('strike', 0)
        short_call_strike = trade_details.get('short_call', {}).get('strike', 0)
        
        # Calculate thresholds for adjustment
        put_threshold = short_put_strike * (1 + self.params['adjustment_threshold'])
        call_threshold = short_call_strike * (1 - self.params['adjustment_threshold'])
        
        adjustment_orders = []
        
        # TODO: Implement wing adjustment logic based on which threshold is breached
        # - If price is approaching short put, consider rolling put wing down
        # - If price is approaching short call, consider rolling call wing up
        # This would involve closing the current wing and opening a new one at different strikes
        
        # For illustration, here's the structure of an adjustment:
        # if current_price < put_threshold:
        #     # Close existing put wing
        #     # ...
        #     # Open new put wing at lower strikes
        #     # ...
        # elif current_price > call_threshold:
        #     # Close existing call wing
        #     # ...
        #     # Open new call wing at higher strikes
        #     # ...
        
        logger.info(f"Created adjustment orders for {symbol} iron condor")
        return adjustment_orders
    
    # ======================== HELPER METHODS ========================
    def _check_adv(self, symbol: str, market_data: MarketData) -> bool:
        """Check if a symbol meets the Average Daily Volume criteria."""
        try:
            # Get daily volume data for the last 20 trading days
            volume_data = market_data.get_historical_data(symbol, days=20, fields=['volume'])
            
            if volume_data is None or len(volume_data) < 20:
                return False
                
            # Calculate average daily volume
            adv = volume_data['volume'].mean()
            
            return adv >= self.params['min_adv']
            
        except Exception as e:
            logger.error(f"Error checking ADV for {symbol}: {str(e)}")
            return False
    
    def _check_option_liquidity(self, symbol: str, option_chains: OptionChains) -> bool:
        """Check if options for a symbol meet liquidity criteria."""
        try:
            chains = option_chains.get_option_chain(symbol)
            if chains is None or chains.empty:
                return False
                
            # Check volume and open interest criteria
            volume_ok = (chains['volume'] >= self.params['min_option_volume']).any()
            oi_ok = (chains['open_interest'] >= self.params['min_option_open_interest']).any()
            
            return volume_ok and oi_ok
            
        except Exception as e:
            logger.error(f"Error checking option liquidity for {symbol}: {str(e)}")
            return False
    
    def _check_option_spreads(self, symbol: str, option_chains: OptionChains) -> bool:
        """Check if options for a symbol have acceptable bid-ask spreads."""
        try:
            chains = option_chains.get_option_chain(symbol)
            if chains is None or chains.empty:
                return False
                
            # Calculate bid-ask spread as percentage of option price
            chains['spread_pct'] = (chains['ask'] - chains['bid']) / ((chains['bid'] + chains['ask']) / 2)
            
            # Check if there are enough options with acceptable spreads
            acceptable_spreads = (chains['spread_pct'] <= self.params['max_bid_ask_spread_pct'])
            
            # Consider it liquid if at least 50% of options have acceptable spreads
            return acceptable_spreads.mean() >= 0.5
            
        except Exception as e:
            logger.error(f"Error checking option spreads for {symbol}: {str(e)}")
            return False
    
    def _is_range_bound(self, symbol: str, market_data: MarketData) -> bool:
        """Check if a symbol is trading in a range-bound pattern."""
        try:
            # Get historical price data
            price_data = market_data.get_historical_data(
                symbol, 
                days=self.params['range_days'], 
                fields=['close', 'high', 'low']
            )
            
            if price_data is None or len(price_data) < self.params['range_days']:
                return False
                
            # Calculate range metrics
            high = price_data['high'].max()
            low = price_data['low'].min()
            last_price = price_data['close'].iloc[-1]
            
            # Calculate range as percentage of current price
            range_pct = (high - low) / last_price
            
            # Calculate linear regression slope to measure trend strength
            x = np.arange(len(price_data))
            y = price_data['close'].values
            slope, _, _, _, _ = np.polyfit(x, y, 1, full=True)
            
            # Normalize slope to percentage change over the period
            trend_strength = abs(slope[0] * len(x) / price_data['close'].mean())
            
            # Criteria for range-bound:
            # 1. Range is not too wide (e.g., < 20%)
            # 2. Trend strength is low
            # 3. Price is not at extremes of the range
            return (
                range_pct < 0.20 and
                trend_strength < self.params['max_trend_strength'] and
                (last_price > low * 1.02) and  # Not at the bottom
                (last_price < high * 0.98)     # Not at the top
            )
            
        except Exception as e:
            logger.error(f"Error checking range-bound behavior for {symbol}: {str(e)}")
            return False
    
    def _select_expiration(self, symbol: str, option_chains: OptionChains) -> str:
        """Select the appropriate expiration date."""
        try:
            chains = option_chains.get_option_chain(symbol)
            if chains is None or chains.empty:
                return ""
                
            available_expirations = chains['expiration_date'].unique()
            target_dte = self.params['target_dte']
            min_dte = self.params['min_dte']
            max_dte = self.params['max_dte']
            
            closest_exp = ""
            closest_diff = float('inf')
            
            for exp in available_expirations:
                exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                dte = (exp_date - date.today()).days
                
                if min_dte <= dte <= max_dte:
                    diff = abs(dte - target_dte)
                    if diff < closest_diff:
                        closest_diff = diff
                        closest_exp = exp
                        
            return closest_exp
            
        except Exception as e:
            logger.error(f"Error selecting expiration for {symbol}: {str(e)}")
            return ""
    
    def _select_put_strikes_by_delta(self, put_options: pd.DataFrame, current_price: float, all_strikes: List[float]) -> Tuple[Dict, Dict]:
        """
        Select put strikes based on delta targets. For iron condor:
        - Short put at delta ~0.15-0.25
        - Long put farther OTM based on wing_width
        """
        if 'delta' not in put_options.columns:
            logger.warning("Delta data not available, falling back to OTM percentage method")
            return self._select_put_strikes_by_otm_percentage(put_options, current_price, all_strikes)
            
        # For puts, delta is negative, so take absolute value
        put_options['abs_delta'] = put_options['delta'].abs()
        
        # Find short put with delta closest to target
        short_put_options = put_options.copy()
        short_put_options['delta_diff'] = abs(short_put_options['abs_delta'] - self.params['short_put_delta'])
        short_put_options = short_put_options.sort_values('delta_diff')
        
        if short_put_options.empty:
            return None, None
            
        short_put = short_put_options.iloc[0].to_dict()
        
        # Find long put farther OTM based on wing width
        short_put_index = all_strikes.index(short_put['strike']) if short_put['strike'] in all_strikes else -1
        
        if short_put_index < 0 or short_put_index < self.params['wing_width']:
            # Fallback to closest OTM put
            long_put_options = put_options[put_options['strike'] < short_put['strike']].copy()
            if long_put_options.empty:
                return short_put, None
            long_put_options = long_put_options.sort_values('strike', ascending=False)
            long_put = long_put_options.iloc[0].to_dict()
        else:
            # Use wing width to determine long put strike
            long_put_strike = all_strikes[short_put_index - self.params['wing_width']]
            long_put_options = put_options[put_options['strike'] == long_put_strike]
            if long_put_options.empty:
                return short_put, None
            long_put = long_put_options.iloc[0].to_dict()
        
        return short_put, long_put
    
    def _select_call_strikes_by_delta(self, call_options: pd.DataFrame, current_price: float, all_strikes: List[float]) -> Tuple[Dict, Dict]:
        """
        Select call strikes based on delta targets. For iron condor:
        - Short call at delta ~0.15-0.25
        - Long call farther OTM based on wing_width
        """
        if 'delta' not in call_options.columns:
            logger.warning("Delta data not available, falling back to OTM percentage method")
            return self._select_call_strikes_by_otm_percentage(call_options, current_price, all_strikes)
            
        # Find short call with delta closest to target
        short_call_options = call_options.copy()
        short_call_options['delta_diff'] = abs(short_call_options['delta'] - self.params['short_call_delta'])
        short_call_options = short_call_options.sort_values('delta_diff')
        
        if short_call_options.empty:
            return None, None
            
        short_call = short_call_options.iloc[0].to_dict()
        
        # Find long call farther OTM based on wing width
        short_call_index = all_strikes.index(short_call['strike']) if short_call['strike'] in all_strikes else -1
        
        if short_call_index < 0 or short_call_index + self.params['wing_width'] >= len(all_strikes):
            # Fallback to closest OTM call
            long_call_options = call_options[call_options['strike'] > short_call['strike']].copy()
            if long_call_options.empty:
                return short_call, None
            long_call_options = long_call_options.sort_values('strike')
            long_call = long_call_options.iloc[0].to_dict()
        else:
            # Use wing width to determine long call strike
            long_call_strike = all_strikes[short_call_index + self.params['wing_width']]
            long_call_options = call_options[call_options['strike'] == long_call_strike]
            if long_call_options.empty:
                return short_call, None
            long_call = long_call_options.iloc[0].to_dict()
        
        return short_call, long_call
    
    def _select_put_strikes_by_otm_percentage(self, put_options: pd.DataFrame, current_price: float, all_strikes: List[float]) -> Tuple[Dict, Dict]:
        """
        Select put strikes based on OTM percentage. For iron condor:
        - Short put strike ~5-8% below current price
        - Long put farther OTM based on wing_width
        """
        # Calculate target short put strike
        short_put_target = current_price * (1 - self.params['short_put_buffer_pct'])
        
        # Find closest short put strike
        put_options['short_strike_diff'] = abs(put_options['strike'] - short_put_target)
        put_options = put_options.sort_values('short_strike_diff')
        
        if put_options.empty:
            return None, None
            
        short_put = put_options.iloc[0].to_dict()
        
        # Find long put farther OTM based on wing width
        short_put_index = all_strikes.index(short_put['strike']) if short_put['strike'] in all_strikes else -1
        
        if short_put_index < 0 or short_put_index < self.params['wing_width']:
            # Fallback to closest OTM put
            long_put_options = put_options[put_options['strike'] < short_put['strike']].copy()
            if long_put_options.empty:
                return short_put, None
            long_put_options = long_put_options.sort_values('strike', ascending=False)
            long_put = long_put_options.iloc[0].to_dict()
        else:
            # Use wing width to determine long put strike
            long_put_strike = all_strikes[short_put_index - self.params['wing_width']]
            long_put_options = put_options[put_options['strike'] == long_put_strike]
            if long_put_options.empty:
                return short_put, None
            long_put = long_put_options.iloc[0].to_dict()
        
        return short_put, long_put
    
    def _select_call_strikes_by_otm_percentage(self, call_options: pd.DataFrame, current_price: float, all_strikes: List[float]) -> Tuple[Dict, Dict]:
        """
        Select call strikes based on OTM percentage. For iron condor:
        - Short call strike ~5-8% above current price
        - Long call farther OTM based on wing_width
        """
        # Calculate target short call strike
        short_call_target = current_price * (1 + self.params['short_call_buffer_pct'])
        
        # Find closest short call strike
        call_options['short_strike_diff'] = abs(call_options['strike'] - short_call_target)
        call_options = call_options.sort_values('short_strike_diff')
        
        if call_options.empty:
            return None, None
            
        short_call = call_options.iloc[0].to_dict()
        
        # Find long call farther OTM based on wing width
        short_call_index = all_strikes.index(short_call['strike']) if short_call['strike'] in all_strikes else -1
        
        if short_call_index < 0 or short_call_index + self.params['wing_width'] >= len(all_strikes):
            # Fallback to closest OTM call
            long_call_options = call_options[call_options['strike'] > short_call['strike']].copy()
            if long_call_options.empty:
                return short_call, None
            long_call_options = long_call_options.sort_values('strike')
            long_call = long_call_options.iloc[0].to_dict()
        else:
            # Use wing width to determine long call strike
            long_call_strike = all_strikes[short_call_index + self.params['wing_width']]
            long_call_options = call_options[call_options['strike'] == long_call_strike]
            if long_call_options.empty:
                return short_call, None
            long_call = long_call_options.iloc[0].to_dict()
        
        return short_call, long_call
    
    # ======================== OPTIMIZATION METHODS ========================
    def get_optimization_params(self) -> Dict[str, Any]:
        """Define parameters that can be optimized and their ranges."""
        return {
            'target_dte': {'type': 'int', 'min': 25, 'max': 60, 'step': 5},
            'short_put_delta': {'type': 'float', 'min': 0.15, 'max': 0.30, 'step': 0.05},
            'short_call_delta': {'type': 'float', 'min': 0.15, 'max': 0.30, 'step': 0.05},
            'short_put_buffer_pct': {'type': 'float', 'min': 0.04, 'max': 0.09, 'step': 0.01},
            'short_call_buffer_pct': {'type': 'float', 'min': 0.04, 'max': 0.09, 'step': 0.01},
            'wing_width': {'type': 'int', 'min': 1, 'max': 3, 'step': 1},
            'profit_target_percent': {'type': 'int', 'min': 50, 'max': 80, 'step': 5},
            'min_iv_rank': {'type': 'int', 'min': 20, 'max': 40, 'step': 5},
            'max_iv_rank': {'type': 'int', 'min': 50, 'max': 70, 'step': 5},
        }
        
    def evaluate_performance(self, backtest_results: Dict[str, Any]) -> float:
        """
        Evaluate strategy performance for optimization.
        
        Parameters:
            backtest_results: Results from backtest
            
        Returns:
            float: Performance score
        """
        # Calculate performance score based on multiple factors
        if 'sharpe_ratio' not in backtest_results or 'max_drawdown' not in backtest_results:
            return 0.0
            
        sharpe = backtest_results.get('sharpe_ratio', 0)
        max_dd = abs(backtest_results.get('max_drawdown', 0))
        win_rate = backtest_results.get('win_rate', 0)
        avg_holding_period = backtest_results.get('avg_holding_period', 0)
        
        # Penalize high drawdowns
        if max_dd > 0.25:  # 25% drawdown
            sharpe = sharpe * (1 - (max_dd - 0.25))
            
        # Reward high win rates
        if win_rate > 0.5:
            sharpe = sharpe * (1 + (win_rate - 0.5))
            
        # Consider holding period - prefer trades that reach profit target faster
        target_holding_period = 15  # days
        if avg_holding_period < target_holding_period:
            sharpe = sharpe * (1 + 0.1 * (target_holding_period - avg_holding_period) / target_holding_period)
            
        return max(0, sharpe)

# TODOs for implementation and optimization
"""
TODO: Implement more sophisticated range-bound detection
TODO: Add volatility skew analysis to balance put and call wing widths
TODO: Implement support/resistance level detection for better strike selection
TODO: Add correlation analysis to ensure diverse positions across underlyings
TODO: Develop more robust adjustment logic for breached wings
TODO: Consider dynamic wing width based on IV rank
TODO: Implement early gamma risk detection to avoid assignment
TODO: Add advanced backtesting with expected move analysis
TODO: Consider machine learning model to predict successful condor setups
TODO: Add outlier protection for high-impact event risk
""" 