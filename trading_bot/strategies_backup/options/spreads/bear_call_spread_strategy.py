#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bear Call Spread Strategy Module

This module implements a bear call spread options strategy that profits from 
moderate bearish price movements with defined risk and reward.
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

class BearCallSpreadStrategy(StrategyOptimizable):
    """
    Bear Call Spread Strategy
    
    This strategy involves selling a call option at a lower strike price and buying a call option
    at a higher strike price with the same expiration date. This creates a credit spread that
    profits from neutral to bearish price movements with defined risk and reward.
    
    Key characteristics:
    - Limited risk (max loss = difference between strikes - net premium received)
    - Limited profit (max profit = net premium received)
    - Benefits from time decay and/or bearish price movement
    - Requires less margin than selling naked calls
    - Defined risk-reward ratio
    """
    
    # ======================== 1. STRATEGY PHILOSOPHY ========================
    # Collect premium by selling a call option at a lower strike while buying a higher-strike call to 
    # define risk, profiting from neutral-to-bearish moves while avoiding unlimited risk of naked calls.
    
    # ======================== 2. DEFAULT PARAMETERS ========================
    DEFAULT_PARAMS = {
        # Strategy identification
        'strategy_name': 'bear_call_spread',
        'strategy_version': '1.0.0',
        
        # Universe selection criteria (Liquid large-caps or ETFs)
        'min_stock_price': 20.0,              # Minimum stock price to consider
        'max_stock_price': 1000.0,            # Maximum stock price to consider
        'min_option_volume': 500,             # Minimum option volume
        'min_option_open_interest': 500,      # Minimum option open interest
        'min_adv': 500000,                    # Minimum average daily volume
        'max_bid_ask_spread_pct': 0.015,      # Maximum bid-ask spread as % of price (0.015 = 1.5%)
        
        # Volatility parameters
        'min_iv_percentile': 30,              # Minimum IV percentile
        'max_iv_percentile': 60,              # Maximum IV percentile for entry
        
        # Technical analysis parameters
        'min_historical_days': 252,           # Days of historical data required
        'trend_indicator': 'ema_20',          # Indicator to determine trend
        'momentum_lookback_days': 10,         # Lookback window for momentum confirmation
        'momentum_threshold': -0.03,          # Price decline threshold (-0.03 = -3%)
        
        # Option parameters
        'target_dte': 35,                     # Target days to expiration (~30-40 DTE)
        'min_dte': 25,                        # Minimum days to expiration
        'max_dte': 45,                        # Maximum days to expiration
        'spread_width_pct': 0.04,             # Width between strikes as % of stock price (~3-5%)
        'strike_selection_method': 'delta',   # 'delta' or 'otm_percentage'
        'short_call_delta': 0.35,             # Target delta for short call (0.30-0.40)
        'long_call_delta': 0.15,              # Target delta for long call (0.10-0.20)
        'short_otm_percentage': 0.00,         # Alternative: % OTM for short call (0 = ATM)
        'long_otm_extra': 0.04,               # Extra % OTM for long call
        
        # Entry and credit parameters
        'min_credit': 0.25,                   # Minimum credit to collect (per spread)
        'target_credit_percent': 0.15,        # Target credit as % of spread width (15%)
        
        # Risk management parameters
        'max_position_size_percent': 0.02,    # Maximum position size as % of portfolio (1-2%)
        'max_num_positions': 5,               # Maximum number of concurrent positions (3-5)
        'max_risk_per_trade': 0.01,           # Maximum risk per trade as % of portfolio
        
        # Exit parameters
        'profit_target_percent': 60,          # Exit at this percentage of max profit (50-75%)
        'loss_limit_percent': 150,            # Exit if loss exceeds threshold (% of credit)
        'dte_exit_threshold': 10,             # Exit when DTE reaches this value (7-10 days)
        'use_trailing_stop': False,           # Whether to use trailing stop
        'trailing_stop_activation': 0.3,      # Activate trailing stop after % of max profit
        'trailing_stop_distance': 0.15,       # Trailing stop as % of position value
        
        # Rolling parameters
        'enable_rolling': False,              # Whether to roll positions
        'roll_when_dte': 7,                   # Roll when DTE reaches this value
        'roll_min_credit': 0.10,              # Minimum credit to collect when rolling
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
                
        # Filter by technical criteria
        tech_signals = TechnicalSignals(market_data)
        symbols_to_remove = []
        
        for symbol in universe.get_symbols():
            # Check if stock is in a downtrend (below 20-day EMA)
            if not self._has_bearish_trend(symbol, tech_signals):
                symbols_to_remove.append(symbol)
                continue
                
            # Check for recent price decline (momentum confirmation)
            if not self._has_negative_momentum(symbol, market_data):
                symbols_to_remove.append(symbol)
                continue
                
        for symbol in symbols_to_remove:
            universe.remove_symbol(symbol)
            
        logger.info(f"Bear Call Spread universe contains {len(universe.get_symbols())} symbols")
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
        # Check if we have enough historical data
        if not market_data.has_min_history(symbol, self.params['min_historical_days']):
            logger.debug(f"{symbol} doesn't have enough historical data")
            return False
        
        # Check implied volatility is in the desired range
        vol_signals = VolatilitySignals(market_data)
        iv_percentile = vol_signals.get_iv_percentile(symbol)
        
        if iv_percentile is None:
            logger.debug(f"{symbol} has no IV percentile data")
            return False
            
        if not (self.params['min_iv_percentile'] <= iv_percentile <= self.params['max_iv_percentile']):
            logger.debug(f"{symbol} IV percentile {iv_percentile:.2f}% outside range")
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
            
        # Check if the stock is in a downtrend
        if not self._has_bearish_trend(symbol, TechnicalSignals(market_data)):
            logger.debug(f"{symbol} does not have a bearish trend")
            return False
            
        # Check for recent price decline (momentum confirmation)
        if not self._has_negative_momentum(symbol, market_data):
            logger.debug(f"{symbol} does not have recent bearish momentum")
            return False
            
        logger.info(f"{symbol} meets all selection criteria for bear call spread")
        return True
    
    # ======================== 5. OPTION SELECTION ========================
    def select_option_contract(self, symbol: str, market_data: MarketData,
                              option_chains: OptionChains) -> Dict[str, Any]:
        """
        Select the appropriate option contracts for the bear call spread.
        
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
            
        # Get call options for the selected expiration
        call_options = option_chains.get_calls(symbol, target_expiration)
        if call_options.empty:
            logger.error(f"No call options available for {symbol} at {target_expiration}")
            return {}
            
        # Select strikes based on the configured method
        if self.params['strike_selection_method'] == 'delta':
            short_call, long_call = self._select_strikes_by_delta(call_options, current_price)
        else:  # Default to otm_percentage
            short_call, long_call = self._select_strikes_by_otm_percentage(call_options, current_price)
            
        if not short_call or not long_call:
            logger.error(f"Could not select appropriate strikes for {symbol}")
            return {}
            
        # Calculate the credit and max profit/loss
        credit = short_call['bid'] - long_call['ask']
        max_profit = credit
        max_loss = (long_call['strike'] - short_call['strike']) - credit
        
        # Check if the credit meets minimum requirements
        if credit < self.params['min_credit']:
            logger.debug(f"Credit of {credit:.2f} for {symbol} is below minimum {self.params['min_credit']}")
            return {}
            
        # Check if credit as percentage of width is within target range
        width = long_call['strike'] - short_call['strike']
        credit_percent = credit / width
        
        if credit_percent < self.params['target_credit_percent']:
            logger.debug(f"Credit percentage {credit_percent:.2f}% for {symbol} is below target")
            return {}
            
        # Return the selected options and trade details
        return {
            'symbol': symbol,
            'strategy': 'bear_call_spread',
            'expiration': target_expiration,
            'dte': (datetime.strptime(target_expiration, '%Y-%m-%d').date() - date.today()).days,
            'short_call': short_call,
            'long_call': long_call,
            'short_call_contract': f"{symbol}_{target_expiration}_{short_call['strike']}_C",
            'long_call_contract': f"{symbol}_{target_expiration}_{long_call['strike']}_C",
            'credit': credit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': short_call['strike'] + credit,
            'risk_reward_ratio': max_loss / max_profit if max_profit > 0 else 0,
            'price': current_price,
            'timestamp': datetime.now().isoformat()
        }
    
    # ======================== 6. POSITION SIZING ========================
    def calculate_position_size(self, trade_details: Dict[str, Any], 
                               position_sizer: PositionSizer) -> int:
        """
        Calculate the number of spreads to trade based on risk parameters.
        
        Parameters:
            trade_details: Details of the selected option spread
            position_sizer: Position sizer instance
            
        Returns:
            int: Number of spreads to trade
        """
        # Calculate max risk per spread
        max_loss_per_spread = trade_details['max_loss'] * 100  # Convert to dollars (per contract)
        
        # Get portfolio value
        portfolio_value = position_sizer.get_portfolio_value()
        
        # Calculate max risk for this trade based on portfolio percentage
        max_risk_dollars = portfolio_value * self.params['max_risk_per_trade']
        
        # Calculate number of spreads
        if max_loss_per_spread <= 0:
            return 0
            
        num_spreads = int(max_risk_dollars / max_loss_per_spread)
        
        # Check against max position size
        max_position_dollars = portfolio_value * self.params['max_position_size_percent']
        position_risk = max_loss_per_spread * num_spreads
        
        if position_risk > max_position_dollars:
            num_spreads = int(max_position_dollars / max_loss_per_spread)
            
        # Ensure at least 1 spread if we're trading
        num_spreads = max(1, num_spreads)
        
        logger.info(f"Bear Call Spread position size for {trade_details['symbol']}: {num_spreads} spreads")
        return num_spreads
    
    # ======================== 7. ENTRY EXECUTION ========================
    def prepare_entry_orders(self, trade_details: Dict[str, Any], 
                            num_spreads: int) -> List[Order]:
        """
        Prepare orders for executing the bear call spread.
        
        Parameters:
            trade_details: Details of the selected spread
            num_spreads: Number of spreads to trade
            
        Returns:
            List of orders to execute
        """
        if num_spreads <= 0:
            return []
            
        symbol = trade_details['symbol']
        orders = []
        
        # Create short call order (sell to open)
        short_call_order = Order(
            symbol=symbol,
            option_symbol=trade_details['short_call_contract'],
            order_type=OrderType.LIMIT,
            action=OrderAction.SELL,
            quantity=num_spreads,
            limit_price=trade_details['short_call']['bid'],
            trade_id=f"bear_call_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            order_details={
                'strategy': 'bear_call_spread',
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
            quantity=num_spreads,
            limit_price=trade_details['long_call']['ask'],
            trade_id=f"bear_call_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            order_details={
                'strategy': 'bear_call_spread',
                'leg': 'long_call',
                'expiration': trade_details['expiration'],
                'strike': trade_details['long_call']['strike'],
                'trade_details': trade_details
            }
        )
        orders.append(long_call_order)
        
        logger.info(f"Created bear call spread orders for {symbol}: {num_spreads} spreads")
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
            logger.info(f"Exiting {symbol} bear call spread: DTE {current_dte} <= threshold {self.params['dte_exit_threshold']}")
            return True
            
        # Check for profit target
        current_value = position.get('current_value', 0)
        entry_value = position.get('entry_value', 0)
        
        if entry_value > 0:
            # For a credit spread, entry_value is negative (credit received)
            # and current_value is the cost to close (debit paid)
            max_credit = trade_details.get('credit', 0) * 100  # Convert to dollars
            profit = max_credit - current_value
            
            if max_credit > 0:
                profit_pct = (profit / max_credit) * 100
                
                # If we've reached our target profit percentage
                if profit_pct >= self.params['profit_target_percent']:
                    logger.info(f"Exiting {symbol} bear call spread: Profit target reached {profit_pct:.2f}%")
                    return True
                
            # Check for loss limit
            loss = current_value - max_credit
            if loss > 0:
                loss_pct = (loss / max_credit) * 100
                
                if loss_pct >= self.params['loss_limit_percent']:
                    logger.info(f"Exiting {symbol} bear call spread: Loss limit reached {loss_pct:.2f}%")
                    return True
                    
        # Check if the underlying price has moved significantly against our position
        current_price = market_data.get_latest_price(symbol)
        if current_price:
            short_call_strike = trade_details.get('short_call', {}).get('strike', 0)
            
            # If price moves substantially above short strike, consider exiting
            if current_price > short_call_strike * 1.02:  # 2% above short strike
                logger.info(f"Exiting {symbol} bear call spread: Price moved against position")
                return True
                
        # Implement trailing stop if enabled
        if self.params['use_trailing_stop'] and entry_value > 0 and 'highest_profit' in position:
            # Only activate trailing stop after reaching activation threshold
            activation_threshold = max_credit * (self.params['trailing_stop_activation'] / 100)
            
            if profit >= activation_threshold:
                # Calculate trailing stop level
                trailing_stop = profit * (1 - self.params['trailing_stop_distance'])
                
                # If current profit drops below trailing stop level, exit
                if position['highest_profit'] - profit > trailing_stop:
                    logger.info(f"Exiting {symbol} bear call spread: Trailing stop triggered")
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
                    'strategy': 'bear_call_spread',
                    'leg': 'exit_' + leg.get('order_details', {}).get('leg', ''),
                    'closing_order': True,
                    'original_order_id': leg.get('order_id', '')
                }
            )
            orders.append(close_order)
            
        logger.info(f"Created exit orders for bear call spread position")
        return orders
    
    # ======================== 10. ROLL EXECUTION ========================
    def prepare_roll_orders(self, position: Dict[str, Any], 
                           market_data: MarketData,
                           option_chains: OptionChains) -> List[Order]:
        """
        Prepare orders to roll a position to a new expiration.
        
        Parameters:
            position: The position to roll
            market_data: Market data instance
            option_chains: Option chains instance
            
        Returns:
            List of orders to execute the roll
        """
        if not self.params['enable_rolling']:
            return []
            
        if not position or 'trade_details' not in position:
            logger.error("Invalid position data for roll")
            return []
            
        trade_details = position.get('trade_details', {})
        symbol = trade_details.get('symbol')
        current_dte = trade_details.get('dte', 0)
        
        # Only roll if DTE is at or below roll threshold
        if current_dte > self.params['roll_when_dte']:
            return []
            
        # Find a new expiration further out
        new_expiration = self._select_roll_expiration(symbol, option_chains)
        if not new_expiration:
            logger.error(f"No suitable roll expiration found for {symbol}")
            return []
            
        # Create exit orders for current position
        exit_orders = self.prepare_exit_orders(position)
        
        # Get current price
        current_price = market_data.get_latest_price(symbol)
        if current_price is None:
            return exit_orders  # Only exit the current position
            
        # Get call options for the new expiration
        call_options = option_chains.get_calls(symbol, new_expiration)
        if call_options.empty:
            return exit_orders  # Only exit the current position
            
        # Select strikes similar to current position
        short_call_strike = trade_details.get('short_call', {}).get('strike', 0)
        long_call_strike = trade_details.get('long_call', {}).get('strike', 0)
        
        # Find closest strikes in the new expiration
        try:
            # For short call
            new_short_call = call_options.iloc[
                (call_options['strike'] - short_call_strike).abs().argsort()[:1]
            ].to_dict('records')[0]
            
            # For long call
            new_long_call = call_options.iloc[
                (call_options['strike'] - long_call_strike).abs().argsort()[:1]
            ].to_dict('records')[0]
            
            # Calculate the new credit
            new_credit = new_short_call['bid'] - new_long_call['ask']
            
            # Check if roll meets minimum credit requirement
            if new_credit < self.params['roll_min_credit']:
                logger.info(f"Roll for {symbol} does not meet minimum credit requirement")
                return exit_orders  # Only exit the current position
                
            # Create entry orders for the new position
            quantity = position.get('legs', [{}])[0].get('quantity', 1)
            
            # Create new trade details for the roll
            new_trade_details = {
                'symbol': symbol,
                'strategy': 'bear_call_spread',
                'expiration': new_expiration,
                'dte': (datetime.strptime(new_expiration, '%Y-%m-%d').date() - date.today()).days,
                'short_call': new_short_call,
                'long_call': new_long_call,
                'short_call_contract': f"{symbol}_{new_expiration}_{new_short_call['strike']}_C",
                'long_call_contract': f"{symbol}_{new_expiration}_{new_long_call['strike']}_C",
                'credit': new_credit,
                'price': current_price,
                'timestamp': datetime.now().isoformat()
            }
            
            roll_orders = []
            
            # Create short call order for the new expiration
            short_call_order = Order(
                symbol=symbol,
                option_symbol=new_trade_details['short_call_contract'],
                order_type=OrderType.LIMIT,
                action=OrderAction.SELL,
                quantity=quantity,
                limit_price=new_short_call['bid'],
                trade_id=f"roll_bear_call_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                order_details={
                    'strategy': 'bear_call_spread',
                    'leg': 'short_call',
                    'expiration': new_expiration,
                    'strike': new_short_call['strike'],
                    'trade_details': new_trade_details,
                    'roll': True
                }
            )
            roll_orders.append(short_call_order)
            
            # Create long call order for the new expiration
            long_call_order = Order(
                symbol=symbol,
                option_symbol=new_trade_details['long_call_contract'],
                order_type=OrderType.LIMIT,
                action=OrderAction.BUY,
                quantity=quantity,
                limit_price=new_long_call['ask'],
                trade_id=f"roll_bear_call_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                order_details={
                    'strategy': 'bear_call_spread',
                    'leg': 'long_call',
                    'expiration': new_expiration,
                    'strike': new_long_call['strike'],
                    'trade_details': new_trade_details,
                    'roll': True
                }
            )
            roll_orders.append(long_call_order)
            
            logger.info(f"Created roll orders for {symbol} bear call spread to {new_expiration}")
            
            # Combine exit orders and roll orders
            return exit_orders + roll_orders
            
        except Exception as e:
            logger.error(f"Error creating roll orders for {symbol}: {str(e)}")
            return exit_orders  # Only exit the current position
    
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
    
    def _has_bearish_trend(self, symbol: str, tech_signals: TechnicalSignals) -> bool:
        """Check if a symbol is in a bearish trend."""
        # Check if price is below the 20-day EMA
        return tech_signals.is_below_ema(symbol, period=20)
    
    def _has_negative_momentum(self, symbol: str, market_data: MarketData) -> bool:
        """Check if a symbol has recent negative price momentum."""
        try:
            # Get historical price data
            lookback = self.params['momentum_lookback_days']
            price_data = market_data.get_historical_data(symbol, days=lookback+5, fields=['close'])
            
            if price_data is None or len(price_data) < lookback:
                return False
                
            # Calculate price change over the lookback period
            start_price = price_data['close'].iloc[-lookback]
            end_price = price_data['close'].iloc[-1]
            
            price_change = (end_price / start_price) - 1
            
            # Check if price change is below threshold
            return price_change <= self.params['momentum_threshold']
            
        except Exception as e:
            logger.error(f"Error checking momentum for {symbol}: {str(e)}")
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
    
    def _select_roll_expiration(self, symbol: str, option_chains: OptionChains) -> str:
        """Select an appropriate expiration date for rolling a position."""
        try:
            chains = option_chains.get_option_chain(symbol)
            if chains is None or chains.empty:
                return ""
                
            available_expirations = chains['expiration_date'].unique()
            target_dte = self.params['target_dte']
            
            # Convert expirations to dates and filter for future dates
            future_exps = []
            for exp in available_expirations:
                exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                dte = (exp_date - date.today()).days
                if dte >= self.params['min_dte']:
                    future_exps.append((exp, dte))
                    
            if not future_exps:
                return ""
                
            # Sort by closest to target DTE
            future_exps.sort(key=lambda x: abs(x[1] - target_dte))
            
            return future_exps[0][0]
            
        except Exception as e:
            logger.error(f"Error selecting roll expiration for {symbol}: {str(e)}")
            return ""
    
    def _select_strikes_by_delta(self, call_options: pd.DataFrame, current_price: float) -> Tuple[Dict, Dict]:
        """Select strikes based on delta targets."""
        if 'delta' not in call_options.columns:
            logger.warning("Delta data not available, falling back to OTM percentage method")
            return self._select_strikes_by_otm_percentage(call_options, current_price)
            
        # Find short call with delta closest to target
        short_call_options = call_options.copy()
        short_call_options['delta_diff'] = abs(short_call_options['delta'] - self.params['short_call_delta'])
        short_call_options = short_call_options.sort_values('delta_diff')
        
        if short_call_options.empty:
            return None, None
            
        short_call = short_call_options.iloc[0].to_dict()
        
        # Find long call with delta closest to target (and higher strike than short call)
        long_call_options = call_options[call_options['strike'] > short_call['strike']].copy()
        long_call_options['delta_diff'] = abs(long_call_options['delta'] - self.params['long_call_delta'])
        long_call_options = long_call_options.sort_values('delta_diff')
        
        if long_call_options.empty:
            return short_call, None
            
        long_call = long_call_options.iloc[0].to_dict()
        
        return short_call, long_call
    
    def _select_strikes_by_otm_percentage(self, call_options: pd.DataFrame, current_price: float) -> Tuple[Dict, Dict]:
        """Select strikes based on OTM percentage."""
        # Calculate target strike prices
        short_call_target = current_price * (1 + self.params['short_otm_percentage'])
        long_call_target = current_price * (1 + self.params['short_otm_percentage'] + self.params['long_otm_extra'])
        
        # Find closest short call strike
        call_options['short_strike_diff'] = abs(call_options['strike'] - short_call_target)
        call_options = call_options.sort_values('short_strike_diff')
        
        if call_options.empty:
            return None, None
            
        short_call = call_options.iloc[0].to_dict()
        
        # Find long call with strike higher than short call
        long_call_options = call_options[call_options['strike'] > short_call['strike']].copy()
        long_call_options['long_strike_diff'] = abs(long_call_options['strike'] - long_call_target)
        long_call_options = long_call_options.sort_values('long_strike_diff')
        
        if long_call_options.empty:
            return short_call, None
            
        long_call = long_call_options.iloc[0].to_dict()
        
        # Ensure the spread width is reasonable
        spread_width = long_call['strike'] - short_call['strike']
        if spread_width < 1.0 or spread_width > current_price * 0.10:  # 10% max width
            logger.warning(f"Selected strikes create an unusual spread width: {spread_width}")
            
        return short_call, long_call

    # ======================== OPTIMIZATION METHODS ========================
    def get_optimization_params(self) -> Dict[str, Any]:
        """Define parameters that can be optimized and their ranges."""
        return {
            'target_dte': {'type': 'int', 'min': 20, 'max': 60, 'step': 5},
            'short_call_delta': {'type': 'float', 'min': 0.30, 'max': 0.45, 'step': 0.05},
            'long_call_delta': {'type': 'float', 'min': 0.10, 'max': 0.25, 'step': 0.05},
            'short_otm_percentage': {'type': 'float', 'min': 0.00, 'max': 0.03, 'step': 0.01},
            'long_otm_extra': {'type': 'float', 'min': 0.02, 'max': 0.06, 'step': 0.01},
            'profit_target_percent': {'type': 'int', 'min': 40, 'max': 80, 'step': 5},
            'loss_limit_percent': {'type': 'int', 'min': 100, 'max': 200, 'step': 10},
            'min_iv_percentile': {'type': 'int', 'min': 20, 'max': 40, 'step': 5},
            'max_iv_percentile': {'type': 'int', 'min': 50, 'max': 70, 'step': 5},
        }
        
    def evaluate_performance(self, backtest_results: Dict[str, Any]) -> float:
        """
        Evaluate strategy performance for optimization.
        
        Parameters:
            backtest_results: Results from backtest
            
        Returns:
            float: Performance score
        """
        # Calculate Sharpe ratio with a penalty for max drawdown
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
TODO: Implement more sophisticated trend detection methods beyond simple EMAs
TODO: Add relative strength/weakness indicators to select strongest bearish candidates
TODO: Enhance volatility analysis to adapt spread width with VIX levels
TODO: Improve exit rules with dynamic time-based profit targets (higher targets for longer DTE)
TODO: Add correlation analysis to avoid too many similar positions
TODO: Consider spread adjustments for adverse moves instead of simple exits
TODO: Implement more advanced rolling logic based on volatility changes
TODO: Add sector rotation analysis to focus on weakest sectors
TODO: Explore calendar/diagonal variants for high volatility environments
TODO: Consider machine learning model for optimal strike selection based on historical spread performance
""" 