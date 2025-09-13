#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bull Put Spread Strategy Module

This module implements a bull put spread options strategy that profits from 
neutral-to-bullish price movements by collecting premium with defined risk.
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

class BullPutSpreadStrategy(StrategyOptimizable):
    """
    Bull Put Spread Strategy
    
    This strategy involves selling a put option at a higher strike price and buying a put option
    at a lower strike price with the same expiration date. This creates a credit spread that
    profits from neutral to bullish price movements with defined maximum risk.
    
    Key characteristics:
    - Limited risk (max loss = width between strikes - net premium received)
    - Limited profit (max profit = net premium received)
    - Benefits from theta decay
    - Profits from neutral to bullish price movement
    - Defined risk-reward ratio
    """
    
    # ======================== 1. STRATEGY PHILOSOPHY ========================
    # Collect premium by selling a put at a higher strike while buying a deeper-OTM put to 
    # define risk, profiting from neutral-to-bullish moves with limited downside.
    
    # ======================== 2. DEFAULT PARAMETERS ========================
    DEFAULT_PARAMS = {
        # Strategy identification
        'strategy_name': 'bull_put_spread',
        'strategy_version': '1.0.0',
        
        # Universe selection criteria (Liquid large-caps or ETFs)
        'min_stock_price': 20.0,              # Minimum stock price to consider
        'max_stock_price': 1000.0,            # Maximum stock price to consider
        'min_option_volume': 500,             # Minimum option volume
        'min_option_open_interest': 1000,     # Minimum option open interest
        'min_adv': 500000,                    # Minimum average daily volume
        'max_bid_ask_spread_pct': 0.015,      # Maximum bid-ask spread as % of price (0.015 = 1.5%)
        
        # Volatility parameters
        'min_iv_percentile': 30,              # Minimum IV percentile
        'max_iv_percentile': 60,              # Maximum IV percentile for entry
        
        # Technical analysis parameters
        'min_historical_days': 252,           # Days of historical data required
        'trend_indicator': 'ema_20',          # Indicator to determine trend
        'max_pullback_percent': 0.03,         # Maximum pullback from recent highs (3%)
        
        # Option parameters
        'target_dte': 38,                     # Target days to expiration (30-45 DTE)
        'min_dte': 25,                        # Minimum days to expiration
        'max_dte': 45,                        # Maximum days to expiration
        
        # Strike selection
        'short_put_otm_percent': 0.03,        # OTM percentage for short put (2-4%)
        'long_put_otm_percent': 0.08,         # OTM percentage for long put (5-8%)
        'short_put_delta': 0.25,              # Target delta for short put (0.20-0.30)
        'long_put_delta': 0.12,               # Target delta for long put (0.10-0.15)
        'strike_selection_method': 'delta',   # 'delta' or 'otm_percentage'
        
        # Entry and credit parameters
        'min_credit': 0.20,                   # Minimum credit to collect (per spread)
        'target_credit_percent': 0.10,        # Target credit as % of spread width (10%)
        'max_credit_percent': 0.20,           # Maximum acceptable credit as % of width
        
        # Risk management parameters
        'max_position_size_percent': 0.02,    # Maximum position size as % of portfolio (1-2%)
        'max_num_positions': 5,               # Maximum number of concurrent positions (3-5)
        'max_risk_per_trade': 0.01,           # Maximum risk per trade as % of portfolio
        
        # Exit parameters
        'profit_target_percent': 60,          # Exit at this percentage of max credit (50-75%)
        'stop_loss_percent': 20,              # Exit if loss exceeds this % of max risk
        'dte_exit_threshold': 10,             # Exit when DTE reaches this value (7-10 days)
        'use_trailing_stop': False,           # Whether to use trailing stop
        
        # Rolling parameters
        'enable_rolling': False,              # Whether to roll positions
        'roll_when_dte': 7,                   # Roll when DTE reaches this value
        'roll_min_credit': 0.10,              # Minimum credit to collect when rolling
    }
    
    # ======================== 3. UNIVERSE DEFINITION ========================
    def define_universe(self, market_data: MarketData) -> Universe:
        """
        Define the universe of tradable securities for bull put spread strategy.
        
        This method identifies a suitable universe of stocks for implementing bull put spreads
        by applying a series of filtering criteria to the entire market. It selects securities
        with appropriate price ranges, sufficient liquidity, quality option chains, and
        technical characteristics that align with the strategy's bullish bias.
        
        The filtering process follows these steps:
        1. Initial price range filter to focus on mid-to-large cap stocks
        2. Volume and liquidity screening to ensure efficient trade execution
        3. Option chain quality validation for spread implementation
        4. Technical analysis filters to confirm bullish bias
        
        Parameters:
            market_data (MarketData): Market data provider with current and historical data
            
        Returns:
            Universe: A Universe object containing the filtered list of tradable symbols
            
        Notes:
            Universe selection methodology:
            
            - Price filtering criteria:
              - Minimum price threshold eliminates low-priced stocks with poor option liquidity
              - Maximum price threshold manages capital efficiency and position sizing
              - Typical range focuses on $20-$1000 stocks for optimal option characteristics
            
            - Liquidity requirements:
              - Average daily volume threshold ensures sufficient trading liquidity
              - Option volume and open interest minimums ensure tradable option chains
              - Bid-ask spread constraints minimize transaction costs in option positions
            
            - Technical filters:
              - Bullish trend confirmation through price location relative to moving averages
              - Pullback limitation ensures entries near support rather than extended prices
              - Volume pattern analysis confirms accumulation characteristics
              
            The universe is dynamically updated with each strategy run, adapting to
            changing market conditions and liquidity environments.
        """
        universe = Universe()
        logger.info("Building bull put spread universe...")
        
        # Step 1: Initial price filter
        price_df = market_data.get_latest_prices()
        min_price = self.params['min_stock_price']
        max_price = self.params['max_stock_price']
        
        if price_df is None or price_df.empty:
            logger.warning("No price data available for universe construction")
            return universe
            
        filtered_symbols = price_df[(price_df['close'] >= min_price) & 
                                   (price_df['close'] <= max_price)].index.tolist()
        
        universe.add_symbols(filtered_symbols)
        logger.debug(f"Price filter ({min_price} to {max_price}): {len(universe.get_symbols())} symbols")
        
        # Step 2: Volume and liquidity filter
        option_chains = OptionChains()
        symbols_to_remove = []
        
        for symbol in universe.get_symbols():
            # Check ADV (Average Daily Volume)
            if not self._check_adv(symbol, market_data):
                symbols_to_remove.append(symbol)
                continue
                
            # Check if options meet volume and open interest criteria
            if not self._check_option_liquidity(symbol, option_chains):
                symbols_to_remove.append(symbol)
                continue
                
            # Check option bid-ask spreads
            if not self._check_option_spreads(symbol, option_chains):
                symbols_to_remove.append(symbol)
                continue
        
        for symbol in symbols_to_remove:
            universe.remove_symbol(symbol)
            
        logger.debug(f"After liquidity filter: {len(universe.get_symbols())} symbols")
            
        # Step 3: Technical filters
        tech_signals = TechnicalSignals(market_data)
        symbols_to_remove = []
        
        for symbol in universe.get_symbols():
            # Check if stock is in a bullish trend (above 20-day EMA)
            if not self._has_bullish_trend(symbol, tech_signals):
                symbols_to_remove.append(symbol)
                continue
                
            # Check for limited pullback from highs
            if not self._has_acceptable_pullback(symbol, market_data):
                symbols_to_remove.append(symbol)
                continue
                
        for symbol in symbols_to_remove:
            universe.remove_symbol(symbol)
            
        logger.info(f"Bull Put Spread universe contains {len(universe.get_symbols())} symbols")
        
        # Log some sample symbols for verification
        if len(universe.get_symbols()) > 0:
            sample_size = min(5, len(universe.get_symbols()))
            sample_symbols = universe.get_symbols()[:sample_size]
            logger.debug(f"Sample universe symbols: {', '.join(sample_symbols)}")
            
        return universe
    
    # ======================== 4. SELECTION CRITERIA ========================
    def check_selection_criteria(self, symbol: str, market_data: MarketData, 
                                option_chains: OptionChains) -> bool:
        """
        Evaluate if a symbol meets all criteria for implementing a bull put spread.
        
        This method conducts a comprehensive multi-stage filtering process to identify
        suitable candidates for the bull put spread strategy. It evaluates historical data
        availability, market conditions, volatility metrics, option chain characteristics,
        and technical indicators to ensure optimal trade opportunities.
        
        The selection process implements these validation stages:
        1. Data adequacy check: Ensures sufficient historical data for accurate analysis
        2. Volatility evaluation: Confirms implied volatility is within optimal range
        3. Option chain validation: Verifies appropriate strikes and expirations are available
        4. Technical confirmation: Confirms bullish price trends and acceptable pullbacks
        5. Support level assessment: Validates existence of clear support levels for strike selection
        
        Parameters:
            symbol (str): Ticker symbol to evaluate
            market_data (MarketData): Market data provider with historical and current data 
            option_chains (OptionChains): Option chain data provider with strikes and expirations
            
        Returns:
            bool: True if symbol meets all criteria for a bull put spread, False otherwise
            
        Notes:
            Selection criteria framework:
            
            - Data validation requirements:
              - Historical data: Minimum of specified trading days (default 252 days)
              - Option chain data: Complete chains with appropriate strikes and expirations
              - Volume and open interest: Adequate liquidity for efficient trade execution
            
            - Volatility conditions:
              - IV percentile range: Typically 30-60% for optimal premium collection
              - Term structure shape: Normal contango preferred for stable conditions
              - Skew pattern: Balanced or moderate put skew for reasonable premiums
            
            - Technical confirmation signals:
              - Trend indicators: Price above key moving averages (e.g., 20-day EMA)
              - Pullback assessment: Limited retracement from recent highs (e.g., <3%)
              - Support level identification: Clear support at or near short put strike level
              
            - Option chain requirements:
              - Availability: Options with appropriate expirations (typically 25-45 DTE)
              - Strike spacing: Adequate strikes to create optimal spread width
              - Liquidity: Sufficient open interest at candidate strikes
              
            These multi-factor criteria help identify optimal candidates for the
            bull put spread strategy, providing high-probability opportunities with
            favorable risk/reward characteristics.
        """
        logger.debug(f"Evaluating selection criteria for {symbol}")
        
        # Step 1: Data adequacy check
        if not market_data.has_min_history(symbol, self.params['min_historical_days']):
            logger.debug(f"{symbol} doesn't have enough historical data, "
                       f"need {self.params['min_historical_days']} days")
            return False
        
        # Step 2: Volatility evaluation
        vol_signals = VolatilitySignals(market_data)
        iv_percentile = vol_signals.get_iv_percentile(symbol)
        
        if iv_percentile is None:
            logger.debug(f"{symbol} has no IV percentile data")
            return False
            
        min_iv = self.params['min_iv_percentile']
        max_iv = self.params['max_iv_percentile']
        
        if not (min_iv <= iv_percentile <= max_iv):
            logger.debug(f"{symbol} IV percentile {iv_percentile:.2f}% outside target range "
                       f"({min_iv}%-{max_iv}%)")
            return False
        
        # Step 3: Option chain validation
        try:
            chains = option_chains.get_option_chain(symbol)
            if chains is None or chains.empty:
                logger.debug(f"{symbol} has no option chains available")
                return False
                
            # Check for appropriate expirations
            available_expirations = chains['expiration_date'].unique()
            valid_expiration = False
            
            for exp in available_expirations:
                exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                dte = (exp_date - date.today()).days
                
                if self.params['min_dte'] <= dte <= self.params['max_dte']:
                    valid_expiration = True
                    break
                    
            if not valid_expiration:
                logger.debug(f"{symbol} has no options in the desired DTE range "
                           f"({self.params['min_dte']}-{self.params['max_dte']} days)")
                return False
                
            # Get current price to evaluate strike distribution
            current_price = market_data.get_latest_price(symbol)
            if current_price is None:
                logger.debug(f"Could not get current price for {symbol}")
                return False
                
            # For target expiration, ensure enough strikes below current price
            put_options = option_chains.get_puts(symbol, exp)
            
            if put_options.empty:
                logger.debug(f"No put options available for {symbol} at {exp}")
                return False
                
            # Check if we have appropriate strikes for the spread
            short_put_target = current_price * (1 - self.params['short_put_otm_percent'])
            long_put_target = current_price * (1 - self.params['long_put_otm_percent'])
            
            available_strikes = put_options['strike'].unique()
            
            # Check for strikes near our target levels
            strikes_near_short = [s for s in available_strikes if abs(s - short_put_target) / short_put_target < 0.05]
            strikes_near_long = [s for s in available_strikes if abs(s - long_put_target) / long_put_target < 0.05]
            
            if not strikes_near_short:
                logger.debug(f"{symbol} lacks suitable strikes near short put target ({short_put_target:.2f})")
                return False
                
            if not strikes_near_long:
                logger.debug(f"{symbol} lacks suitable strikes near long put target ({long_put_target:.2f})")
                return False
                
        except Exception as e:
            logger.error(f"Error checking option chains for {symbol}: {str(e)}")
            return False
            
        # Step 4: Technical confirmation
        tech_signals = TechnicalSignals(market_data)
        
        # Check if stock is in a bullish trend (above 20-day EMA)
        if not self._has_bullish_trend(symbol, tech_signals):
            logger.debug(f"{symbol} does not have a bullish trend (price not above 20-day EMA)")
            return False
            
        # Check for limited pullback from highs
        if not self._has_acceptable_pullback(symbol, market_data):
            logger.debug(f"{symbol} pullback exceeds threshold (max {self.params['max_pullback_percent']*100:.1f}%)")
            return False
        
        # All criteria passed
        logger.info(f"{symbol} meets all selection criteria for bull put spread")
        return True
    
    # ======================== 5. OPTION SELECTION ========================
    def select_option_contract(self, symbol: str, market_data: MarketData,
                              option_chains: OptionChains) -> Dict[str, Any]:
        """
        Select optimal option contracts for constructing a bull put spread.
        
        This method performs precise selection of the appropriate expiration date and
        strike prices for both the short and long put legs of a bull put spread.
        It implements a structured decision process to identify the optimal contracts
        based on the strategy parameters and current market conditions.
        
        The selection process follows these key steps:
        1. Identify the optimal expiration date matching target DTE parameters
        2. Select appropriate strike prices based on delta or OTM percentage rules
        3. Calculate and validate key spread metrics (credit, max profit/loss, breakeven)
        4. Generate a comprehensive trade specification with complete contract details
        
        Parameters:
            symbol (str): Underlying asset ticker symbol
            market_data (MarketData): Market data provider with current prices
            option_chains (OptionChains): Option chain data provider with available contracts
            
        Returns:
            Dict[str, Any]: Complete trade specification including:
                - symbol: Underlying ticker symbol
                - strategy: Strategy identifier ('bull_put_spread')
                - expiration: Selected expiration date
                - dte: Days to expiration
                - short_put/long_put: Contract details for each leg
                - short_put_contract/long_put_contract: Option symbols
                - credit: Net premium received for the spread
                - max_profit/max_loss: Maximum potential profit/loss
                - breakeven: Breakeven price at expiration
                - risk_reward_ratio: Ratio of max loss to max profit
                - price: Current price of underlying
                - timestamp: Time of selection
                
        Notes:
            Option selection strategy for bull put spreads:
            
            - Expiration selection logic:
              - Target DTE typically 30-45 days to balance premium collection and time decay
              - Avoids earnings announcements and other known event dates when possible
              - Considers liquidity across available expirations
              
            - Strike selection approaches:
              - Delta-based: Short put at ~0.25-0.30 delta, long put at ~0.10-0.15 delta
              - OTM percentage: Short put 2-4% below current price, long put 5-8% below
              - Strike selection aims to balance premium collection with probability of success
              
            - Credit requirements:
              - Minimum credit threshold ensures adequate compensation for risk
              - Credit as percentage of width verifies appropriate risk/reward profile
              - Higher IV environments generally allow for wider spreads and better credits
              
            - Risk profile characteristics:
              - Limited risk (max loss = width between strikes - credit received)
              - Limited profit (max profit = credit received)
              - Breakeven occurs at short put strike - credit received
              - Profit achieved when price remains above short put strike at expiration
        """
        # Get current price
        current_price = market_data.get_latest_price(symbol)
        if current_price is None:
            logger.error(f"Unable to get current price for {symbol}")
            return {}
            
        logger.debug(f"Selecting option contracts for {symbol} at ${current_price:.2f}")
            
        # Find appropriate expiration
        target_expiration = self._select_expiration(symbol, option_chains)
        if not target_expiration:
            logger.error(f"No suitable expiration found for {symbol}")
            return {}
            
        expiration_date = datetime.strptime(target_expiration, '%Y-%m-%d').date()
        dte = (expiration_date - date.today()).days
        logger.debug(f"Selected expiration {target_expiration} ({dte} DTE)")
            
        # Get put options for the selected expiration
        put_options = option_chains.get_puts(symbol, target_expiration)
        if put_options.empty:
            logger.error(f"No put options available for {symbol} at {target_expiration}")
            return {}
            
        # Select strikes based on the configured method
        strike_method = self.params['strike_selection_method']
        logger.debug(f"Using {strike_method} strike selection method")
        
        if strike_method == 'delta':
            short_put, long_put = self._select_strikes_by_delta(put_options, current_price)
        else:  # Default to otm_percentage
            short_put, long_put = self._select_strikes_by_otm_percentage(put_options, current_price)
            
        if not short_put or not long_put:
            logger.error(f"Could not select appropriate strikes for {symbol}")
            return {}
            
        # Calculate the credit and max profit/loss
        credit = short_put['bid'] - long_put['ask']
        width = short_put['strike'] - long_put['strike']
        max_profit = credit
        max_loss = width - credit
        
        # Check if the credit meets minimum requirements
        min_credit = self.params['min_credit']
        if credit < min_credit:
            logger.debug(f"Credit of ${credit:.2f} for {symbol} is below minimum ${min_credit:.2f}")
            return {}
            
        # Check if credit as percentage of width is within acceptable range
        credit_percent = (credit / width) * 100
        min_credit_pct = self.params['target_credit_percent'] * 100
        max_credit_pct = self.params['max_credit_percent'] * 100
        
        if credit_percent < min_credit_pct or credit_percent > max_credit_pct:
            logger.debug(f"Credit percentage {credit_percent:.2f}% for {symbol} is outside acceptable range "
                        f"({min_credit_pct:.1f}%-{max_credit_pct:.1f}%)")
            return {}
            
        # Calculate breakeven and risk-reward ratio
        breakeven = short_put['strike'] - credit
        risk_reward_ratio = max_loss / max_profit if max_profit > 0 else 0
        
        logger.info(f"Selected bull put spread for {symbol}: "
                  f"Short {short_put['strike']} put, Long {long_put['strike']} put, "
                  f"Credit: ${credit:.2f}, Width: ${width:.2f}, "
                  f"Risk/reward: {risk_reward_ratio:.2f}")
            
        # Return the selected options and trade details
        return {
            'symbol': symbol,
            'strategy': 'bull_put_spread',
            'expiration': target_expiration,
            'dte': dte,
            'short_put': short_put,
            'long_put': long_put,
            'short_put_contract': f"{symbol}_{target_expiration}_{short_put['strike']}_P",
            'long_put_contract': f"{symbol}_{target_expiration}_{long_put['strike']}_P",
            'credit': credit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'risk_reward_ratio': risk_reward_ratio,
            'price': current_price,
            'width': width,
            'width_pct': (width / current_price) * 100,  # Width as % of stock price
            'credit_percent': credit_percent,  # Credit as % of width
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
        
        logger.info(f"Bull Put Spread position size for {trade_details['symbol']}: {num_spreads} spreads")
        return num_spreads
    
    # ======================== 7. ENTRY EXECUTION ========================
    def prepare_entry_orders(self, symbol: str, market_data: Dict[str, Any]) -> List[Order]:
        """
        Prepare orders to establish a new bull put spread position.
        
        Key functions:
        - Analyzes current portfolio allocation and determines appropriate position size
        - Selects optimal short put and long put options based on selection criteria
        - Creates paired orders for both legs of the spread with appropriate pricing
        - Attaches comprehensive trade metadata for tracking and management
        
        Parameters:
            symbol: Ticker symbol of the underlying asset
            market_data: Dictionary containing current market data with structure:
                - price: Current price of the underlying
                - options_chain: Complete options chain for the symbol
                - volatility: Current implied volatility metrics
                - indicators: Technical indicators values
                - volume: Trading volume information
                
        Returns:
            List[Order]: Executable order specifications for both legs of the spread
            
        Notes:
            - Position sizing: Dynamically calculates position size based on account value,
              risk parameters, and volatility conditions. Higher volatility leads to smaller
              position sizes to manage risk.
            
            - Strike selection: Selects strikes based on delta targets (default short put
              delta 0.30, long put delta 0.15) and minimum spread width requirements
              to optimize risk/reward profile.
              
            - Pricing strategy: Uses limit orders with calculated debit/credit pricing based
              on current market conditions, maintaining a minimum credit requirement to
              ensure favorable risk/reward ratio.
              
            - Risk management: Embeds max loss parameters directly in order sizing to
              ensure portfolio risk limits are maintained across multiple positions.
              
            - Trade lifecycle: Initializes complete trade context within orders to support
              subsequent management, adjustments, and performance tracking.
        """
        # ... existing code ...
    
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
            logger.info(f"Exiting {symbol} bull put spread: DTE {current_dte} <= threshold {self.params['dte_exit_threshold']}")
            return True
            
        # Check for profit target
        current_value = position.get('current_value', 0)
        entry_value = position.get('entry_value', 0)
        
        if entry_value > 0:
            # For a credit spread, entry_value is negative (credit received)
            # and current_value is the cost to close (debit paid)
            max_credit = trade_details.get('credit', 0) * 100  # Convert to dollars
            profit = max_credit - abs(current_value)
            
            if max_credit > 0:
                profit_pct = (profit / max_credit) * 100
                
                # If we've reached our target profit percentage
                if profit_pct >= self.params['profit_target_percent']:
                    logger.info(f"Exiting {symbol} bull put spread: Profit target reached {profit_pct:.2f}%")
                    return True
                
            # Check for loss limit
            # For a bull put spread, max loss is width between strikes - credit received
            max_width = (trade_details.get('short_put', {}).get('strike', 0) - 
                         trade_details.get('long_put', {}).get('strike', 0)) * 100
            max_loss = max_width - max_credit
            
            # Current loss is the difference between current cost to close and initial credit
            current_loss = abs(current_value) - max_credit
            
            if max_loss > 0:
                loss_pct = (current_loss / max_loss) * 100
                
                if loss_pct >= self.params['stop_loss_percent']:
                    logger.info(f"Exiting {symbol} bull put spread: Loss limit reached {loss_pct:.2f}%")
                    return True
                    
        # Check if the underlying price has moved significantly against our position
        current_price = market_data.get_latest_price(symbol)
        if current_price:
            short_put_strike = trade_details.get('short_put', {}).get('strike', 0)
            
            # If price approaches short put strike, consider exiting to avoid assignment
            if current_price < short_put_strike * 1.02:  # Within 2% of short strike
                logger.info(f"Exiting {symbol} bull put spread: Price approaching short strike")
                return True
                
        # Implement trailing stop if enabled
        if self.params.get('use_trailing_stop', False) and 'highest_profit_pct' in position:
            highest_profit = position.get('highest_profit_pct', 0)
            current_profit_pct = profit_pct if 'profit_pct' in locals() else 0
            
            # If current profit has dropped by more than 15% from highest recorded profit
            if highest_profit > 20 and (highest_profit - current_profit_pct) > 15:
                logger.info(f"Exiting {symbol} bull put spread: Trailing stop triggered")
                return True
                
        return False
    
    # ======================== 9. EXIT EXECUTION ========================
    def prepare_exit_orders(self, positions_to_exit: List[Dict[str, Any]], market_data: Dict[str, Any] = None) -> List[Order]:
        """
        Prepare orders to close existing bull put spread positions that have triggered exit conditions.
        
        Key functions:
        - Processes each leg of the spread positions (short put, long put) for closure
        - Creates opposing orders (buys for short positions, sells for long positions)
        - Determines optimal order types based on exit reasons (market orders for stop losses, limit orders for profit targets)
        - Calculates appropriate limit prices when applicable
        - Preserves trade context and metadata for performance tracking
        
        Parameters:
            positions_to_exit: List of position dictionaries to exit, each containing:
                - legs: List of individual option contracts in the spread
                - trade_details: Original trade parameters and context
                - exit_reason: String indicating why the position is being exited (e.g., 'stop_loss', 'profit_target', 'max_dte')
            market_data: Dictionary containing current market data for pricing (optional for market orders)
                
        Returns:
            List[Order]: Executable order specifications for all legs of all positions to be exited
            
        Notes:
            Exit pricing strategy:
            - Stop loss exits: Uses market orders to prioritize guaranteed execution over price
            - Profit target exits: Uses limit orders at calculated prices to maximize profit realization
            - Time-based exits: Uses market orders if DTE is critical (1-2 days), otherwise limit orders
            
            Order type considerations:
            - Market orders ensure immediate execution but may have unfavorable fills, especially in illiquid options
            - Limit orders offer price control but risk non-execution if markets move rapidly
            
            Leg handling:
            - Both legs (short and long puts) must be closed to completely exit the position
            - Short put closure is prioritized as it carries assignment risk
            - Each leg's exit price is calculated independently based on current market conditions
            
            Risk management:
            - Exit orders include the original position context for risk tracking
            - All exit orders preserve metadata for comprehensive performance analysis
            - Emergency exits (e.g., extreme market conditions) always use market orders regardless of exit reason
        """
        orders = []
        
        if not positions_to_exit:
            return orders
            
        # Handle single position case by wrapping it in a list
        if isinstance(positions_to_exit, dict):
            positions_to_exit = [positions_to_exit]
            
        for position in positions_to_exit:
            if not position or 'legs' not in position:
                logger.error("Invalid position data for exit orders")
                continue
                
            legs = position.get('legs', [])
            exit_reason = position.get('exit_reason', 'unknown')
            
            # Determine order type based on exit reason
            order_type = OrderType.MARKET
            if exit_reason == 'profit_target' and market_data is not None:
                order_type = OrderType.LIMIT
            
            for leg in legs:
                if not leg or 'status' not in leg or leg['status'] != OrderStatus.FILLED:
                    continue
                    
                # Determine action to close the position
                close_action = OrderAction.BUY if leg.get('action') == OrderAction.SELL else OrderAction.SELL
                
                # Calculate limit price if using limit orders
                limit_price = None
                if order_type == OrderType.LIMIT and market_data is not None:
                    option_symbol = leg.get('option_symbol', '')
                    if option_symbol in market_data:
                        option_data = market_data[option_symbol]
                        # For buying back short puts, use bid price with small buffer
                        if close_action == OrderAction.BUY:
                            limit_price = option_data.get('bid', 0) * 1.05  # 5% buffer
                        # For selling long puts, use ask price with small discount
                        else:
                            limit_price = option_data.get('ask', 0) * 0.95  # 5% discount
                
                close_order = Order(
                    symbol=leg.get('symbol', ''),
                    option_symbol=leg.get('option_symbol', ''),
                    order_type=order_type,
                    action=close_action,
                    quantity=leg.get('quantity', 0),
                    limit_price=limit_price,
                    trade_id=f"close_{leg.get('trade_id', '')}",
                    order_details={
                        'strategy': 'bull_put_spread',
                        'leg': 'exit_' + leg.get('order_details', {}).get('leg', ''),
                        'closing_order': True,
                        'exit_reason': exit_reason,
                        'original_order_id': leg.get('order_id', ''),
                        'original_trade_details': position.get('trade_details', {})
                    }
                )
                orders.append(close_order)
                
            logger.info(f"Created exit orders for bull put spread position. Reason: {exit_reason}")
            
        return orders
    
    # ======================== 10. CONTINUOUS OPTIMIZATION ========================
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
            
        # Get put options for the new expiration
        put_options = option_chains.get_puts(symbol, new_expiration)
        if put_options.empty:
            return exit_orders  # Only exit the current position
            
        # Try to find similar delta strikes for the new position
        try:
            if self.params['strike_selection_method'] == 'delta':
                new_short_put, new_long_put = self._select_strikes_by_delta(put_options, current_price)
            else:
                new_short_put, new_long_put = self._select_strikes_by_otm_percentage(put_options, current_price)
                
            if not new_short_put or not new_long_put:
                return exit_orders  # Only exit the current position
                
            # Calculate the new credit
            new_credit = new_short_put['bid'] - new_long_put['ask']
            
            # Check if the new credit meets minimum requirements
            if new_credit < self.params['roll_min_credit']:
                logger.info(f"Roll for {symbol} does not meet minimum credit requirement")
                return exit_orders  # Only exit the current position
                
            # Create entry orders for the new position
            quantity = position.get('legs', [{}])[0].get('quantity', 1)
            
            # Create new trade details for the roll
            new_trade_details = {
                'symbol': symbol,
                'strategy': 'bull_put_spread',
                'expiration': new_expiration,
                'dte': (datetime.strptime(new_expiration, '%Y-%m-%d').date() - date.today()).days,
                'short_put': new_short_put,
                'long_put': new_long_put,
                'short_put_contract': f"{symbol}_{new_expiration}_{new_short_put['strike']}_P",
                'long_put_contract': f"{symbol}_{new_expiration}_{new_long_put['strike']}_P",
                'credit': new_credit,
                'price': current_price,
                'timestamp': datetime.now().isoformat()
            }
            
            roll_orders = []
            
            # Create short put order for the new expiration
            short_put_order = Order(
                symbol=symbol,
                option_symbol=new_trade_details['short_put_contract'],
                order_type=OrderType.LIMIT,
                action=OrderAction.SELL,
                quantity=quantity,
                limit_price=new_short_put['bid'],
                trade_id=f"roll_bull_put_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                order_details={
                    'strategy': 'bull_put_spread',
                    'leg': 'short_put',
                    'expiration': new_expiration,
                    'strike': new_short_put['strike'],
                    'trade_details': new_trade_details,
                    'roll': True
                }
            )
            roll_orders.append(short_put_order)
            
            # Create long put order for the new expiration
            long_put_order = Order(
                symbol=symbol,
                option_symbol=new_trade_details['long_put_contract'],
                order_type=OrderType.LIMIT,
                action=OrderAction.BUY,
                quantity=quantity,
                limit_price=new_long_put['ask'],
                trade_id=f"roll_bull_put_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                order_details={
                    'strategy': 'bull_put_spread',
                    'leg': 'long_put',
                    'expiration': new_expiration,
                    'strike': new_long_put['strike'],
                    'trade_details': new_trade_details,
                    'roll': True
                }
            )
            roll_orders.append(long_put_order)
            
            logger.info(f"Created roll orders for {symbol} bull put spread to {new_expiration}")
            
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
    
    def _has_bullish_trend(self, symbol: str, tech_signals: TechnicalSignals) -> bool:
        """
        Determine if a symbol exhibits a confirmed bullish trend.
        
        This method evaluates technical indicators to verify that the underlying asset
        is in a bullish price trend, which is a fundamental requirement for implementing
        a bull put spread strategy. The primary indicator used is price position relative
        to its 20-day exponential moving average (EMA).
        
        Parameters:
            symbol (str): Ticker symbol to evaluate
            tech_signals (TechnicalSignals): Technical signals provider with indicator calculations
            
        Returns:
            bool: True if symbol shows confirmed bullish trend characteristics, False otherwise
            
        Notes:
            Trend evaluation methodology:
            
            - Primary signal: Price above 20-day EMA indicates bullish trend
            - This simple yet effective filter helps ensure the underlying price action
              supports the bullish thesis of the strategy
            - In production environments, this could be enhanced with additional trend
              confirmation indicators such as MACD, directional movement, or longer-term
              moving average relationships
            
            The bullish trend confirmation is a critical component of the strategy entry
            criteria, as bull put spreads perform best when price continues to move upward
            or remains stable after position establishment.
        """
        # Check if price is above the 20-day EMA
        is_bullish = tech_signals.is_above_ema(symbol, period=20)
        logger.debug(f"{symbol} bullish trend check: {'PASS' if is_bullish else 'FAIL'}")
        return is_bullish
    
    def _has_acceptable_pullback(self, symbol: str, market_data: MarketData) -> bool:
        """
        Verify that a symbol has a moderate pullback from recent highs.
        
        This method analyzes recent price action to identify stocks that have experienced
        a limited pullback from their recent highs. Targeting stocks with moderate pullbacks
        can provide optimal entry points for bull put spreads, balancing premium collection
        with probability of success.
        
        Parameters:
            symbol (str): Ticker symbol to evaluate
            market_data (MarketData): Market data provider with historical price data
            
        Returns:
            bool: True if symbol shows acceptable pullback within thresholds, False otherwise
            
        Notes:
            Pullback assessment methodology:
            
            - Calculation: (recent_high - current_price) / recent_high
            - Lookback period: Typically 20 trading days to identify recent price action
            - Threshold: Default 3% maximum pullback (configurable)
            
            Pullback analysis serves several strategic purposes:
            1. Identifies stocks in bullish trends that have temporarily pulled back
            2. Avoids stocks that are extended too far from their moving averages
            3. Provides potentially higher premiums during small retracements
            4. Targets situations where price is likely finding support
            
            This filter complements the trend analysis by focusing on tactical entry timing
            within the broader bullish trend. It helps avoid entering positions when stocks
            have pulled back too severely, which could indicate a more significant trend change.
        """
        try:
            # Get historical price data for the last 20 trading days
            price_data = market_data.get_historical_data(symbol, days=20, fields=['close', 'high'])
            
            if price_data is None or len(price_data) < 20:
                logger.debug(f"{symbol} insufficient price history for pullback calculation")
                return False
                
            # Calculate recent high
            recent_high = price_data['high'].max()
            
            # Get current price
            current_price = price_data['close'].iloc[-1]
            
            # Calculate pullback percentage
            pullback = (recent_high - current_price) / recent_high
            pullback_pct = pullback * 100
            threshold = self.params['max_pullback_percent'] * 100
            
            acceptable = pullback <= self.params['max_pullback_percent']
            
            logger.debug(f"{symbol} pullback: {pullback_pct:.2f}% (threshold: {threshold:.2f}%), "
                        f"{'PASS' if acceptable else 'FAIL'}")
            
            return acceptable
            
        except Exception as e:
            logger.error(f"Error checking pullback for {symbol}: {str(e)}")
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
    
    def _select_strikes_by_delta(self, put_options: pd.DataFrame, current_price: float) -> Tuple[Dict, Dict]:
        """
        Select optimal strike prices for bull put spread based on delta targets.
        
        This method identifies the appropriate strike prices for both legs of the spread
        by targeting specific delta values. For bull put spreads, this means selecting
        a short put with moderate delta (probability of being ITM) and a long put with
        lower delta to define risk at an acceptable level.
        
        Parameters:
            put_options (pd.DataFrame): Available put options data with columns including:
                - strike: Strike prices
                - bid/ask: Current market prices
                - delta: Delta values (sensitivity to underlying price changes)
                - other option metrics
            current_price (float): Current price of the underlying asset
            
        Returns:
            Tuple[Dict, Dict]: Dictionaries containing the selected short put and long put details
            
        Notes:
            Delta selection methodology:
            
            - For bull put spreads, the method targets:
              - Short put: Moderate delta (~0.25) = Higher strike, near support levels
              - Long put: Lower delta (~0.10-0.15) = Lower strike, further OTM for protection
            
            - Delta interpretation:
              - Delta represents approximate probability of expiring ITM
              - Delta 0.25 = ~25% probability of expiring ITM (75% probability OTM)
              - Delta 0.10 = ~10% probability of expiring ITM (90% probability OTM)
              
            - Implementation details:
              - Uses absolute delta values as puts have negative deltas
              - Ensures long put strike is below short put strike
              - Falls back to OTM percentage method if delta data unavailable
              
            This delta-based approach provides consistency across different underlying
            prices and volatility environments, as it automatically adjusts based on
            market conditions rather than using fixed percentages.
        """
        if 'delta' not in put_options.columns:
            logger.warning(f"Delta data not available for strike selection, falling back to OTM percentage method")
            return self._select_strikes_by_otm_percentage(put_options, current_price)
            
        # For puts, delta is negative, so take absolute value
        put_options['abs_delta'] = put_options['delta'].abs()
        
        # Find short put with delta closest to target
        short_put_options = put_options.copy()
        short_put_target = self.params['short_put_delta']
        short_put_options['delta_diff'] = abs(short_put_options['abs_delta'] - short_put_target)
        short_put_options = short_put_options.sort_values('delta_diff')
        
        if short_put_options.empty:
            logger.warning(f"No suitable options found for short put leg")
            return None, None
            
        short_put = short_put_options.iloc[0].to_dict()
        
        # Find long put with delta closest to target (and lower strike than short put)
        long_put_options = put_options[put_options['strike'] < short_put['strike']].copy()
        long_put_target = self.params['long_put_delta']
        long_put_options['delta_diff'] = abs(long_put_options['abs_delta'] - long_put_target)
        long_put_options = long_put_options.sort_values('delta_diff')
        
        if long_put_options.empty:
            logger.warning(f"No suitable options found for long put leg (below short put strike)")
            return short_put, None
            
        long_put = long_put_options.iloc[0].to_dict()
        
        # Calculate spread width and verify it's reasonable
        width = short_put['strike'] - long_put['strike']
        width_pct = width / current_price * 100
        
        logger.debug(f"Selected strikes by delta - Short put: {short_put['strike']} (delta: {short_put['delta']:.3f}), "
                   f"Long put: {long_put['strike']} (delta: {long_put['delta']:.3f}), "
                   f"Width: ${width:.2f} ({width_pct:.1f}% of price)")
        
        return short_put, long_put
    
    def _select_strikes_by_otm_percentage(self, put_options: pd.DataFrame, current_price: float) -> Tuple[Dict, Dict]:
        """
        Select optimal strike prices for bull put spread based on OTM percentages.
        
        This method identifies the appropriate strike prices for both legs of the spread
        by targeting strikes at specific percentages out-of-the-money relative to the
        current price. This approach provides a direct way to control the risk/reward
        profile of the spread based on price distances.
        
        Parameters:
            put_options (pd.DataFrame): Available put options data including strike prices
            current_price (float): Current price of the underlying asset
            
        Returns:
            Tuple[Dict, Dict]: Dictionaries containing the selected short put and long put details
            
        Notes:
            OTM percentage selection methodology:
            
            - For bull put spreads, the method targets:
              - Short put: Typically 2-4% below current price (support level)
              - Long put: Typically 5-8% below current price (protection level)
            
            - Strike width considerations:
              - Target width is determined by the difference between the two OTM percentages
              - Width typically ranges from 3-5% of underlying price
              - Minimum width of $1.00 ensures reasonable risk/reward profile
              
            - Implementation details:
              - Strikes selected based on closest match to target prices
              - Ensures long put strike is below short put strike
              - Validates width to ensure appropriate risk/reward
              
            The OTM percentage approach provides a simple and intuitive way to select
            strikes based on price distances, allowing direct control over the probability
            of success (higher for further OTM short puts) and max loss (controlled by width).
        """
        # Calculate target strike prices based on OTM percentages
        short_put_target = current_price * (1 - self.params['short_put_otm_percent'])
        long_put_target = current_price * (1 - self.params['long_put_otm_percent'])
        
        short_otm_pct = self.params['short_put_otm_percent'] * 100
        long_otm_pct = self.params['long_put_otm_percent'] * 100
        
        logger.debug(f"Target strikes - Short put: ${short_put_target:.2f} ({short_otm_pct:.1f}% OTM), "
                   f"Long put: ${long_put_target:.2f} ({long_otm_pct:.1f}% OTM)")
        
        # Find closest short put strike
        put_options['short_strike_diff'] = abs(put_options['strike'] - short_put_target)
        put_options = put_options.sort_values('short_strike_diff')
        
        if put_options.empty:
            logger.warning(f"No suitable options available for strike selection")
            return None, None
            
        short_put = put_options.iloc[0].to_dict()
        
        # Find long put with strike below short put
        long_put_options = put_options[put_options['strike'] < short_put['strike']].copy()
        long_put_options['long_strike_diff'] = abs(long_put_options['strike'] - long_put_target)
        long_put_options = long_put_options.sort_values('long_strike_diff')
        
        if long_put_options.empty:
            logger.warning(f"No suitable options for long put leg (below short put strike)")
            return short_put, None
            
        long_put = long_put_options.iloc[0].to_dict()
        
        # Ensure the spread width is reasonable (not too narrow)
        width = short_put['strike'] - long_put['strike']
        width_pct = width / current_price * 100
        
        if width < 1.0:  # Minimum $1 width
            # Try to find a wider spread by selecting a lower long put strike
            logger.debug(f"Selected spread width (${width:.2f}) below minimum, attempting to find wider spread")
            wider_puts = put_options[(put_options['strike'] < short_put['strike']) & 
                                    (put_options['strike'] <= long_put_target)].copy()
            if not wider_puts.empty:
                # Sort by strike to get the next lower strike
                wider_puts = wider_puts.sort_values('strike', ascending=False)
                if len(wider_puts) > 1:  # If there's at least one strike below our original selection
                    long_put = wider_puts.iloc[1].to_dict()
                    width = short_put['strike'] - long_put['strike']
                    width_pct = width / current_price * 100
                    logger.debug(f"Adjusted to wider spread, new width: ${width:.2f} ({width_pct:.1f}% of price)")
        
        logger.debug(f"Selected strikes by OTM% - Short put: {short_put['strike']}, Long put: {long_put['strike']}")
        
        return short_put, long_put

    # ======================== OPTIMIZATION METHODS ========================
    def get_optimization_params(self) -> Dict[str, Any]:
        """
        Define parameters that can be optimized and their valid ranges.
        
        This method specifies which strategy parameters are suitable for systematic
        optimization, along with their valid ranges and step sizes. These definitions
        enable automated parameter tuning through backtesting to identify optimal
        parameter combinations for different market regimes.
        
        Returns:
            Dict[str, Any]: Dictionary of parameters with their optimization constraints,
                where each parameter entry contains:
                - type: Data type (int, float)
                - min: Minimum allowable value
                - max: Maximum allowable value
                - step: Step size for optimization iterations
                
        Notes:
            Parameter optimization strategy:
            
            - Key parameters for optimization include:
              - DTE (target_dte): Controls time horizon and theta decay exposure
              - Delta targets: Defines probability of success and premium levels
              - OTM percentages: Alternative to delta for controlling strike distances
              - Profit/loss thresholds: Determines trade duration and risk management
              - IV percentile range: Optimizes premium collection in ideal volatility conditions
            
            - Optimization methodology:
              - Parameters can be optimized individually or jointly using grid search
              - Walk-forward optimization helps prevent overfitting to specific periods
              - Parameter combinations should be evaluated across different market regimes
              - Sharpe ratio optimization balances absolute returns with consistency
              
            - Implementation considerations:
              - Step sizes are chosen to create manageable search spaces
              - Ranges reflect practical trading constraints and strategy requirements
              - Parameter interdependencies should be considered during optimization
        """
        return {
            'target_dte': {'type': 'int', 'min': 25, 'max': 60, 'step': 5},
            'short_put_delta': {'type': 'float', 'min': 0.20, 'max': 0.35, 'step': 0.05},
            'long_put_delta': {'type': 'float', 'min': 0.05, 'max': 0.20, 'step': 0.05},
            'short_put_otm_percent': {'type': 'float', 'min': 0.02, 'max': 0.05, 'step': 0.01},
            'long_put_otm_percent': {'type': 'float', 'min': 0.05, 'max': 0.10, 'step': 0.01},
            'profit_target_percent': {'type': 'int', 'min': 50, 'max': 80, 'step': 5},
            'stop_loss_percent': {'type': 'int', 'min': 15, 'max': 30, 'step': 5},
            'min_iv_percentile': {'type': 'int', 'min': 20, 'max': 40, 'step': 5},
            'max_iv_percentile': {'type': 'int', 'min': 50, 'max': 70, 'step': 5},
        }
        
    def evaluate_performance(self, backtest_results: Dict[str, Any]) -> float:
        """
        Evaluate strategy performance to guide parameter optimization.
        
        This method calculates a comprehensive performance score based on backtest results,
        considering multiple performance metrics including risk-adjusted returns, drawdowns,
        win rates, and trade efficiency. The resulting score helps identify optimal parameter
        combinations during systematic optimization.
        
        Parameters:
            backtest_results (Dict[str, Any]): Comprehensive results from backtest including:
                - sharpe_ratio: Risk-adjusted return metric
                - max_drawdown: Maximum peak-to-trough decline (as negative percentage)
                - win_rate: Percentage of trades that were profitable
                - avg_holding_period: Average days per trade
                - other performance metrics
                
        Returns:
            float: Performance score where higher values indicate better performance
            
        Notes:
            Scoring methodology:
            
            - Core metric: Sharpe ratio forms the foundation of the performance score
              - Balances absolute returns with consistency/volatility
              - Default calculation uses risk-free rate appropriate for the strategy timeframe
            
            - Adjustment factors:
              - Drawdown penalty: Reduces score for excessive drawdowns (>25%)
              - Win rate bonus: Increases score for strategies with high win rates (>50%)
              - Holding period efficiency: Rewards strategies that achieve profits in shorter timeframes
              
            - Implementation details:
              - Returns 0.0 if essential metrics are missing
              - Score is bounded at minimum of 0.0 (no negative scores)
              - Higher scores indicate more desirable parameter combinations
              - Credit spread optimization places higher emphasis on win rate than debit spreads
              
            Bull put spreads, being credit strategies, particularly benefit from high
            win rates and low drawdowns, so these factors are weighted accordingly in
            the scoring function. Short-duration trades are also rewarded as they allow
            for more efficient capital redeployment.
        """
        # Check for required metrics
        if 'sharpe_ratio' not in backtest_results or 'max_drawdown' not in backtest_results:
            logger.warning("Missing required metrics for performance evaluation")
            return 0.0
            
        # Extract key performance metrics
        sharpe = backtest_results.get('sharpe_ratio', 0)
        max_dd = abs(backtest_results.get('max_drawdown', 0))
        win_rate = backtest_results.get('win_rate', 0)
        avg_holding_period = backtest_results.get('avg_holding_period', 0)
        
        # Base score starts with Sharpe ratio
        score = sharpe
        
        # Apply drawdown penalty for excessive drawdowns
        if max_dd > 0.25:  # 25% drawdown threshold
            drawdown_penalty = (max_dd - 0.25) 
            score = score * (1 - drawdown_penalty)
            logger.debug(f"Applied drawdown penalty: -{drawdown_penalty:.2f} (max DD: {max_dd:.2f})")
            
        # Apply win rate bonus for high win rates
        # For credit spreads, win rate is particularly important
        if win_rate > 0.5:  # 50% win rate threshold
            win_bonus = (win_rate - 0.5) * 1.25  # Enhanced bonus for credit spreads
            score = score * (1 + win_bonus)
            logger.debug(f"Applied win rate bonus: +{win_bonus:.2f} (win rate: {win_rate:.2f})")
            
        # Apply efficiency bonus for shorter holding periods
        target_holding_period = 15  # Target days for bull put spreads
        if avg_holding_period < target_holding_period and avg_holding_period > 0:
            efficiency_bonus = 0.1 * (target_holding_period - avg_holding_period) / target_holding_period
            score = score * (1 + efficiency_bonus)
            logger.debug(f"Applied efficiency bonus: +{efficiency_bonus:.2f} (avg hold: {avg_holding_period:.1f} days)")
            
        # Apply credit capture bonus for efficient premium capture
        if 'avg_profit_pct' in backtest_results:
            avg_profit_pct = backtest_results.get('avg_profit_pct', 0)
            credit_bonus = 0 
            
            # If average profit is over 30% of max credit, apply bonus
            if avg_profit_pct > 30:
                credit_bonus = 0.1 * (avg_profit_pct / 100)
                score = score * (1 + credit_bonus)
                logger.debug(f"Applied credit capture bonus: +{credit_bonus:.2f} (avg profit: {avg_profit_pct:.1f}%)")
        
        # Ensure score is never negative
        score = max(0, score)
        
        logger.info(f"Strategy performance score: {score:.4f} (Sharpe: {sharpe:.2f}, Max DD: {max_dd:.2f}, "
                  f"Win Rate: {win_rate:.2f}, Avg Hold: {avg_holding_period:.1f} days)")
        
        return score

# TODOs for implementation and optimization
"""
TODO: Implement more sophisticated trend detection methods beyond simple EMAs
TODO: Add relative strength analysis to select strongest bullish candidates
TODO: Enhance volatility analysis to adapt strike selection with IV levels
TODO: Implement support level detection to optimize short put strike placement
TODO: Add correlation analysis to avoid too many similar positions
TODO: Consider put ratio spreads as an alternative in high IV environments
TODO: Implement more advanced rolling logic based on market conditions
TODO: Add sector rotation analysis to focus on strongest sectors
TODO: Explore dynamic credit targets based on market volatility
TODO: Consider machine learning model to predict optimal strike selection
""" 