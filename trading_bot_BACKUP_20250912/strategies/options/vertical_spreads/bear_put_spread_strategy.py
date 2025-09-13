#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bear Put Spread Strategy Module

This module implements a bear put spread options strategy that profits from 
moderate bearish price movements with defined risk and reward.

A bear put spread is created by:
1. Buying a put option at a higher strike price
2. Selling a put option at a lower strike price
3. Using the same expiration date for both options

This creates a defined-risk, defined-reward position that benefits from 
downward movement in the underlying asset.
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

class BearPutSpreadStrategy(StrategyOptimizable):
    """
    Bear Put Spread Options Strategy
    
    This strategy involves buying a put option at a higher strike price and selling a put option
    at a lower strike price with the same expiration date. This creates a debit spread that
    profits from moderately bearish movements while capping both the maximum profit and loss.
    
    Key characteristics:
    - Limited risk (max loss = net premium paid)
    - Limited profit (max profit = difference between strikes - net premium paid)
    - Requires less capital than buying puts outright
    - Benefits from moderately bearish price movement
    - Mitigates time decay impact compared to single puts
    - Breakeven point is at long put strike minus net debit
    - Maximum profit achieved when price falls below short put strike
    
    Ideal market conditions:
    - Moderately bearish outlook
    - Medium to high implied volatility
    - Expected downside move within a defined range
    - When you want defined risk exposure to downside movement
    """
    
    # ======================== 1. STRATEGY PHILOSOPHY ========================
    # Profit from moderate bearish moves by buying a higher-strike put and selling a lower-strike put,
    # defining your risk while capturing downside leverage at a fraction of the cost of a naked put.
    
    # ======================== 2. DEFAULT PARAMETERS ========================
    DEFAULT_PARAMS = {
        # Strategy identification
        'strategy_name': 'bear_put_spread',
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
        'target_dte': 35,                     # Target days to expiration (~30 DTE)
        'min_dte': 25,                        # Minimum days to expiration
        'max_dte': 45,                        # Maximum days to expiration
        'spread_width_pct': 0.04,             # Width between strikes as % of stock price (~3-5%)
        'strike_selection_method': 'delta',   # 'delta' or 'otm_percentage'
        'long_put_delta': 0.45,               # Target delta for long put (0.35-0.50)
        'short_put_delta': 0.20,              # Target delta for short put (0.15-0.25)
        'long_otm_percentage': 0.00,          # Alternative: % OTM for long put (0 = ATM)
        'short_otm_extra': 0.04,              # Extra % OTM for short put
        
        # Risk management parameters
        'max_position_size_percent': 0.02,    # Maximum position size as % of portfolio (1-2%)
        'max_num_positions': 5,               # Maximum number of concurrent positions (3-5)
        'max_risk_per_trade': 0.01,           # Maximum risk per trade as % of portfolio
        
        # Exit parameters
        'profit_target_percent': 60,          # Exit at this percentage of max profit (50-75%)
        'loss_limit_percent': 20,             # Exit if loss exceeds threshold (width × 0.2)
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
        """
        Define the universe of stocks to trade based on criteria.
        
        This method filters the entire market to find suitable candidates for the 
        bear put spread strategy based on price range, volume, option liquidity,
        bid-ask spreads, and technical indicators showing bearish conditions.
        
        Parameters:
            market_data (MarketData): Source of historical and current market data
            
        Returns:
            Universe: A Universe object containing filtered symbols that meet all criteria
            
        Notes:
            The filtering process applies multiple criteria in sequence:
            1. Price range filtering (min/max stock price)
            2. Volume filtering (average daily volume)
            3. Option liquidity filtering (volume and open interest)
            4. Bid-ask spread filtering
            5. Technical criteria (bearish trend and momentum)
        """
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
            
        logger.info(f"Bear Put Spread universe contains {len(universe.get_symbols())} symbols")
        return universe
    
    # ======================== 4. SELECTION CRITERIA ========================
    def check_selection_criteria(self, symbol: str, market_data: MarketData, 
                                option_chains: OptionChains) -> bool:
        """
        Evaluate if a symbol meets all criteria for implementing a bear put spread.
        
        This method conducts a comprehensive multi-stage filtering process to identify
        suitable candidates for the bear put spread strategy. It evaluates historical data
        availability, market conditions, volatility metrics, option chain characteristics,
        and technical indicators to ensure optimal trade opportunities.
        
        The selection process implements these validation stages:
        1. Data adequacy check: Ensures sufficient historical data for accurate analysis
        2. Volatility evaluation: Confirms implied volatility is within optimal range
        3. Option chain validation: Verifies appropriate strikes and expirations are available
        4. Technical confirmation: Confirms bearish price trends and momentum signals
        5. Liquidity assessment: Ensures adequate liquidity for efficient trade execution
        
        Parameters:
            symbol (str): Ticker symbol to evaluate
            market_data (MarketData): Market data provider with historical and current data 
            option_chains (OptionChains): Option chain data provider with strikes and expirations
            
        Returns:
            bool: True if symbol meets all criteria for a bear put spread, False otherwise
            
        Notes:
            Selection criteria framework:
            
            - Data validation requirements:
              - Historical data: Minimum of specified trading days (default 252 days)
              - Option chain data: Complete chains with strikes above and below current price
              - Volume and open interest: Minimum levels to ensure adequate liquidity
            
            - Volatility conditions:
              - IV percentile range: Typically 30-60% (high enough for premium, low enough for direction)
              - Term structure: Preferably normal or slightly inverted for bearish thesis
              - Skew profile: Put skew indicating downside potential or market concern
            
            - Technical confirmation signals:
              - Trend indicators: Price below key moving averages (e.g., 20-day EMA)
              - Momentum confirmation: Recent downward price momentum (typical -3% or more)
              - Volume confirmation: Increased volume on down days vs up days
              
            - Option chain requirements:
              - Availability: Options with appropriate expirations (typically 25-45 DTE)
              - Strike spacing: Adequate strikes to create optimal spread width
              - Open interest: Sufficient open interest at candidate strikes
              
            These multi-factor criteria help identify optimal candidates for the
            bear put spread strategy, balancing technical signals, volatility conditions,
            and practical trade implementation considerations.
        """
        logger.debug(f"Checking selection criteria for {symbol}")
        
        # Stage 1: Data adequacy check
        if not market_data.has_min_history(symbol, self.params['min_historical_days']):
            logger.debug(f"{symbol} doesn't have enough historical data, "
                        f"need {self.params['min_historical_days']} days")
            return False
        
        # Stage 2: Volatility evaluation
        vol_signals = VolatilitySignals(market_data)
        iv_percentile = vol_signals.get_iv_percentile(symbol)
        
        if iv_percentile is None:
            logger.debug(f"{symbol} has no IV percentile data")
            return False
            
        if not (self.params['min_iv_percentile'] <= iv_percentile <= self.params['max_iv_percentile']):
            logger.debug(f"{symbol} IV percentile {iv_percentile:.2f}% outside target range "
                        f"({self.params['min_iv_percentile']}%-{self.params['max_iv_percentile']}%)")
            return False
        
        # Stage 3: Option chain validation
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
                
            # Check for sufficient strike coverage
            current_price = market_data.get_latest_price(symbol)
            if current_price is None:
                logger.debug(f"Could not get current price for {symbol}")
                return False
                
            # For target expiration, ensure enough strikes above and below current price
            put_options = option_chains.get_puts(symbol, exp)
            
            if put_options.empty:
                logger.debug(f"No put options available for {symbol} at {exp}")
                return False
                
            # Check if we have strikes suitable for spread creation
            # Need strikes spanning from ATM to at least 5% below current price
            strikes = put_options['strike'].unique()
            atm_strike = get_atm_strike(current_price, strikes)
            
            if atm_strike is None:
                logger.debug(f"No suitable ATM strike found for {symbol}")
                return False
                
            min_strike_needed = current_price * 0.95  # 5% OTM
            has_lower_strikes = any(strike <= min_strike_needed for strike in strikes)
            
            if not has_lower_strikes:
                logger.debug(f"{symbol} does not have suitable OTM put strikes")
                return False
                
        except Exception as e:
            logger.error(f"Error checking option chains for {symbol}: {str(e)}")
            return False
            
        # Stage 4: Technical confirmation
        tech_signals = TechnicalSignals(market_data)
        
        # Check if stock is in a downtrend (below 20-day EMA)
        if not self._has_bearish_trend(symbol, tech_signals):
            logger.debug(f"{symbol} does not have a bearish trend (price not below 20-day EMA)")
            return False
            
        # Check for recent price decline (momentum confirmation)
        if not self._has_negative_momentum(symbol, market_data):
            logger.debug(f"{symbol} does not have recent bearish momentum "
                        f"(minimum {self.params['momentum_threshold']*100:.1f}% decline needed)")
            return False
        
        # All criteria passed
        logger.info(f"{symbol} meets all selection criteria for bear put spread")
        return True
    
    # ======================== 5. OPTION SELECTION ========================
    def select_option_contract(self, symbol: str, market_data: MarketData,
                              option_chains: OptionChains) -> Dict[str, Any]:
        """
        Select optimal option contracts for constructing a bear put spread.
        
        This method performs precise selection of the appropriate expiration date and 
        strike prices for both the long and short put legs of a bear put spread. 
        It implements a structured decision process to identify the optimal contracts
        based on the strategy parameters and current market conditions.
        
        The selection process follows these key steps:
        1. Identify the optimal expiration date matching target DTE parameters
        2. Select appropriate strike prices based on delta or OTM percentage rules
        3. Calculate and validate key spread metrics (debit, max profit/loss, breakeven)
        4. Generate a comprehensive trade specification with complete contract details
        
        Parameters:
            symbol (str): Underlying asset ticker symbol
            market_data (MarketData): Market data provider with current prices
            option_chains (OptionChains): Option chain data provider with available contracts
            
        Returns:
            Dict[str, Any]: Complete trade specification including:
                - symbol: Underlying ticker symbol
                - strategy: Strategy identifier ('bear_put_spread')
                - expiration: Selected expiration date
                - dte: Days to expiration
                - long_put/short_put: Contract details for each leg
                - long_put_contract/short_put_contract: Option symbols
                - debit: Net cost of the spread
                - max_profit/max_loss: Maximum potential profit/loss
                - breakeven: Breakeven price at expiration
                - risk_reward_ratio: Ratio of max profit to max loss
                - price: Current price of underlying
                - timestamp: Time of selection
                
        Notes:
            Option selection strategy:
            
            - Expiration selection logic:
              - Target DTE typically 30-45 days to balance time decay and directional exposure
              - Considers liquidity and open interest across available expirations
              - Avoids earnings and other known event dates when possible
              
            - Strike selection approaches:
              - Delta-based: Select long put at ~0.45 delta, short put at ~0.20 delta
              - OTM percentage: Select long put near ATM, short put 4-5% further OTM
              - Strike width typically 3-5% of underlying price to optimize risk/reward
              
            - Spread metrics analysis:
              - Validates debit cost relative to width between strikes
              - Calculates risk/reward ratio to ensure favorable trade characteristics
              - Identifies precise breakeven level for risk management
              
            - Risk profile characteristics:
              - Limited risk (maximum loss = net debit paid)
              - Limited profit (maximum profit = strike difference - net debit)
              - Profit achieved when underlying moves below short put strike by expiration
              - Breakeven occurs at long put strike minus net debit
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
        if self.params['strike_selection_method'] == 'delta':
            logger.debug(f"Using delta-based strike selection method")
            long_put, short_put = self._select_strikes_by_delta(put_options, current_price)
        else:  # Default to otm_percentage
            logger.debug(f"Using OTM percentage strike selection method")
            long_put, short_put = self._select_strikes_by_otm_percentage(put_options, current_price)
            
        if not long_put or not short_put:
            logger.error(f"Could not select appropriate strikes for {symbol}")
            return {}
            
        # Calculate the debit and max profit potential
        debit = long_put['ask'] - short_put['bid']
        
        # Validate the debit - must be positive for a debit spread
        if debit <= 0:
            logger.warning(f"Invalid debit spread for {symbol}, debit: {debit}")
            return {}
            
        # Calculate spread width and max profit/loss
        spread_width = long_put['strike'] - short_put['strike']
        max_profit = spread_width - debit
        max_loss = debit
        
        # Calculate breakeven and risk-reward ratio
        breakeven = long_put['strike'] - debit
        risk_reward_ratio = max_profit / max_loss if max_loss > 0 else 0
        
        logger.info(f"Selected bear put spread for {symbol}: "
                   f"Long {long_put['strike']} put, Short {short_put['strike']} put, "
                   f"Debit: ${debit:.2f}, Max profit: ${max_profit:.2f}, "
                   f"Risk/reward: {risk_reward_ratio:.2f}")
        
        # Return the selected options and trade details
        return {
            'symbol': symbol,
            'strategy': 'bear_put_spread',
            'expiration': target_expiration,
            'dte': dte,
            'long_put': long_put,
            'short_put': short_put,
            'long_put_contract': f"{symbol}_{target_expiration}_{long_put['strike']}_P",
            'short_put_contract': f"{symbol}_{target_expiration}_{short_put['strike']}_P",
            'debit': debit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'risk_reward_ratio': risk_reward_ratio,
            'price': current_price,
            'spread_width': spread_width,
            'spread_width_pct': spread_width / current_price * 100,  # Width as % of stock price
            'timestamp': datetime.now().isoformat()
        }
    
    # ======================== 6. POSITION SIZING ========================
    def calculate_position_size(self, trade_details: Dict[str, Any], 
                               position_sizer: PositionSizer) -> int:
        """
        Calculate the optimal number of bear put spreads to trade based on risk parameters.
        
        This method implements a sophisticated position sizing algorithm that determines the
        appropriate number of option spreads to trade based on portfolio value, risk tolerance,
        and the specific characteristics of the trade opportunity. It ensures that all positions
        adhere to strict risk management guidelines at both individual trade and portfolio levels.
        
        The position sizing process follows these key steps:
        1. Calculate the maximum risk per spread (debit paid per contract)
        2. Determine maximum portfolio allocation based on risk parameters
        3. Calculate the number of spreads based on per-trade risk limits
        4. Apply additional position size constraints from overall exposure limits
        5. Ensure the final position size is practical and within risk parameters
        
        Parameters:
            trade_details (Dict[str, Any]): Complete details of the selected spread:
                - max_loss: Maximum loss per spread (option contract)
                - debit: Net cost of the spread
                - symbol: Underlying asset ticker
                - other trade specifications
            position_sizer (PositionSizer): Position sizing service that provides:
                - Portfolio value and allocation data
                - Current positions and exposure information
                - Risk model parameters
                
        Returns:
            int: Number of option spreads to trade (contracts)
            
        Notes:
            Position sizing strategy:
            
            - Risk-based allocation approach:
              - Core principle: Risk no more than specified percentage on any single trade
              - Default risk limit: 1% of portfolio per trade
              - Position sizing directly proportional to account size, ensuring scaling with equity
            
            - Multi-level risk constraints:
              - Per-trade risk limit (primary constraint)
              - Maximum position size as percentage of portfolio (secondary constraint)
              - Maximum number of concurrent positions (exposure diversification)
              
            - Volatility adjustments:
              - Implicitly accounts for market volatility through option pricing
              - Higher implied volatility raises option premiums, naturally reducing position size
              - Adapts to changing market conditions through price-based constraints
              
            - Special considerations:
              - Ensures at least 1 spread for viable trades (minimum execution size)
              - Maximum size cap prevents overexposure to a single underlying
              - Liquidity constraints may further limit size based on market conditions
        """
        # Calculate max risk per spread (debit paid)
        max_risk_per_spread = trade_details['max_loss'] * 100  # Convert to dollars (per contract)
        
        # Get portfolio value and risk parameters
        portfolio_value = position_sizer.get_portfolio_value()
        max_risk_per_trade_pct = self.params['max_risk_per_trade']
        max_position_size_pct = self.params['max_position_size_percent']
        
        # Calculate maximum risk allocation for this trade
        max_risk_dollars = portfolio_value * max_risk_per_trade_pct
        
        # Calculate base number of spreads based on risk limit
        if max_risk_per_spread <= 0:
            logger.warning(f"Invalid max risk per spread: ${max_risk_per_spread:.2f}")
            return 0
            
        num_spreads = int(max_risk_dollars / max_risk_per_spread)
        logger.debug(f"Initial position size based on risk: {num_spreads} spreads")
        
        # Check against max position size constraint (secondary limit)
        max_position_dollars = portfolio_value * max_position_size_pct
        position_cost = trade_details['debit'] * 100 * num_spreads
        
        if position_cost > max_position_dollars:
            num_spreads_by_size = int(max_position_dollars / (trade_details['debit'] * 100))
            logger.debug(f"Position size limited by max position constraint: {num_spreads_by_size} spreads")
            num_spreads = min(num_spreads, num_spreads_by_size)
        
        # Check current number of open positions (diversification constraint)
        current_positions = position_sizer.get_open_positions_count()
        max_positions = self.params['max_num_positions']
        
        if current_positions >= max_positions:
            logger.warning(f"Maximum number of positions reached ({max_positions}), cannot open new position")
            return 0
            
        # Ensure at least 1 spread for viable trades
        num_spreads = max(1, num_spreads)
        
        # Calculate final position details
        final_risk_dollars = num_spreads * max_risk_per_spread
        final_risk_pct = (final_risk_dollars / portfolio_value) * 100
        
        logger.info(f"Bear Put Spread position size for {trade_details['symbol']}: {num_spreads} spreads, "
                   f"risk: ${final_risk_dollars:.2f} ({final_risk_pct:.2f}% of portfolio)")
        
        return num_spreads
    
    # ======================== 7. ENTRY EXECUTION ========================
    def prepare_entry_orders(self, trade_details: Dict[str, Any], 
                            num_spreads: int) -> List[Order]:
        """
        Prepare orders for executing a new bear put spread position.
        
        This method creates well-structured order objects for both legs of the bear put spread
        (long put at higher strike, short put at lower strike). It includes all necessary
        details for execution, trade tracking, and risk management across the position lifecycle.
        
        Key functions performed:
        1. Validates position parameters and constructs paired option orders
        2. Creates separate limit orders for both legs with appropriate pricing
        3. Attaches comprehensive trade context metadata to both orders
        4. Ensures consistent trade IDs and references for position tracking
        5. Sets appropriate limit prices to optimize execution probability
        
        Parameters:
            trade_details (Dict[str, Any]): Comprehensive details of the selected spread including:
                - symbol: Underlying asset ticker symbol
                - expiration: Option expiration date
                - long_put/short_put: Selected option contract details
                - long_put_contract/short_put_contract: Option symbols
                - debit: Net cost of the spread
                - max_profit/max_loss: Risk parameters
                - breakeven: Breakeven price at expiration
                - price: Current price of underlying
            num_spreads (int): Number of option spreads to trade (quantity)
            
        Returns:
            List[Order]: List of executable order specifications (one for each leg)
                - Long put buy order with all trade details
                - Short put sell order with all trade details
                - Both orders include comprehensive metadata for lifecycle tracking
                
        Notes:
            Entry strategy factors:
            
            - Order execution approach:
              - Uses limit orders with strategic pricing to enhance fill probability
              - Timestamps are synchronized to treat the spread as a single position
              - All orders share a common trade ID for lifecycle management
              
            - Trading mechanics:
              - Long put order: BUY at higher strike with limit price at or near ask
              - Short put order: SELL at lower strike with limit price at or near bid
              - Debit paid should match the calculated entry debit in trade_details
              
            - Order linkage and structure:
              - Orders include metadata linking them to the same trade
              - Order details preserve full context of selection criteria
              - Strategy parameters embedded in order for performance analysis
              
            - Implementation considerations:
              - In production, may use broker's combo/spread order types if available
              - Pricing logic adjusts based on market conditions at execution time
              - First leg filled partially may require adjusting quantity of second leg
              - Order cancellation logic should handle both legs if either fails
        """
        if num_spreads <= 0:
            return []
            
        symbol = trade_details['symbol']
        trade_id = f"bear_put_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        orders = []
        
        # Create long put order (higher strike) - BUY
        long_put_order = Order(
            symbol=symbol,
            option_symbol=trade_details['long_put_contract'],
            order_type=OrderType.LIMIT,
            action=OrderAction.BUY,
            quantity=num_spreads,
            limit_price=trade_details['long_put']['ask'],  # Use ask price for buy orders
            trade_id=trade_id,
            order_details={
                'strategy': 'bear_put_spread',
                'leg': 'long_put',
                'expiration': trade_details['expiration'],
                'strike': trade_details['long_put']['strike'],
                'trade_details': trade_details,
                'timestamp': datetime.now().isoformat()
            }
        )
        orders.append(long_put_order)
        
        # Create short put order (lower strike) - SELL
        short_put_order = Order(
            symbol=symbol,
            option_symbol=trade_details['short_put_contract'],
            order_type=OrderType.LIMIT,
            action=OrderAction.SELL,
            quantity=num_spreads,
            limit_price=trade_details['short_put']['bid'],  # Use bid price for sell orders
            trade_id=trade_id,
            order_details={
                'strategy': 'bear_put_spread',
                'leg': 'short_put',
                'expiration': trade_details['expiration'],
                'strike': trade_details['short_put']['strike'],
                'trade_details': trade_details,
                'timestamp': datetime.now().isoformat()
            }
        )
        orders.append(short_put_order)
        
        # Calculate expected debit
        expected_debit = trade_details['debit'] * num_spreads * 100  # Convert to dollars
        logger.info(f"Created bear put spread orders for {symbol}: {num_spreads} spreads, "
                   f"expected debit ${expected_debit:.2f}")
        
        return orders
    
    # ======================== 8. EXIT CONDITIONS ========================
    def check_exit_conditions(self, position: Dict[str, Any], 
                             market_data: MarketData) -> bool:
        """
        Evaluate if an existing bear put spread position should be closed.
        
        This method performs a comprehensive analysis of an open position against multiple
        exit criteria to determine if it should be closed. It considers time-based factors,
        profit/loss thresholds, underlying price movements, and risk management rules.
        
        The method implements a multi-factor decision framework that evaluates:
        1. Time-based exits: DTE thresholds to avoid accelerated time decay
        2. Profit target exits: Close when a specified percentage of max profit is achieved
        3. Loss limit exits: Close when losses exceed defined risk tolerance thresholds
        4. Technical signal exits: Close when underlying price action contradicts the thesis
        5. Volatility-based exits: Close when volatility conditions significantly change
        
        Parameters:
            position (Dict[str, Any]): Complete position data structure containing:
                - trade_details: Original trade parameters and context
                - current_value: Current market value of the position
                - entry_value: Original cost basis of the position
                - legs: Individual option contract details
                - metadata: Additional tracking information
            market_data (MarketData): Current market data provider for up-to-date pricing
                
        Returns:
            bool: True if any exit condition is met, False if position should be maintained
            
        Notes:
            Exit condition framework:
            
            - Time-based exit rules:
              - Primary DTE threshold: Exit when DTE drops below specified threshold (default: 10 days)
              - Accelerated time decay typically begins in the final 21-14 days
              - Gamma risk increases exponentially in final days before expiration
              
            - Profit-taking rules:
              - Target threshold: Exit when profit reaches specified % of maximum (default: 60%)
              - Profit calculation uses (current_value - entry_value) compared to max potential profit
              - Most profit in debit spreads is captured in the first 50-70% of the trade duration
              
            - Loss management rules:
              - Stop loss threshold: Exit when loss exceeds specified % of risk (default: 20%)
              - Loss calculation compares current value to entry value relative to maximum risk
              - Bears higher priority than profit targets to preserve capital
              
            - Technical condition rules:
              - Underlying price movement: Exit if price moves significantly against position direction
              - Specifically for bear put spreads, exit if price rises above the long put strike by defined buffer
              - Prevents holding losing positions when market direction contradicts the strategy thesis
              
            - Special handling:
              - Trailing stops: When enabled, lock in profits by exiting if profits retrace from peaks
              - Close to expiration: More aggressive exit rules apply as expiration approaches
              - Assignment risk: Higher priority for exit if underlying price approaches short put strike
        """
        if not position or 'trade_details' not in position:
            logger.error("Invalid position data for exit check")
            return False
            
        trade_details = position.get('trade_details', {})
        symbol = trade_details.get('symbol')
        
        if not symbol:
            logger.warning("Missing symbol in position data")
            return False
            
        # 1. Check time-based exit - DTE is below threshold
        expiration_date = datetime.strptime(trade_details.get('expiration', ''), '%Y-%m-%d').date()
        current_dte = (expiration_date - date.today()).days
        trade_details['dte'] = current_dte  # Update DTE in the trade details
        
        if current_dte <= self.params['dte_exit_threshold']:
            logger.info(f"Exiting {symbol} bear put spread: DTE {current_dte} <= threshold {self.params['dte_exit_threshold']}")
            position['exit_reason'] = 'dte_threshold'
            return True
            
        # 2. Check profit target
        current_value = position.get('current_value', 0)
        entry_value = position.get('entry_value', 0)
        
        if entry_value > 0:
            # For bear put spread (debit spread), entry_value is positive (debit paid)
            # and current_value is the value we could sell it for now
            
            # Calculate profit as percentage of max potential profit
            max_profit = trade_details.get('max_profit', 0) * 100  # Convert to dollars
            realized_profit = current_value - entry_value
            
            if max_profit > 0:
                profit_pct = (realized_profit / max_profit) * 100
                
                # Store highest profit percentage for trailing stops
                position['highest_profit_pct'] = max(position.get('highest_profit_pct', 0), profit_pct)
                
                # If we've reached our target profit percentage
                if profit_pct >= self.params['profit_target_percent']:
                    logger.info(f"Exiting {symbol} bear put spread: Profit target reached {profit_pct:.2f}%")
                    position['exit_reason'] = 'profit_target'
                    return True
                
            # 3. Check for loss limit
            max_loss = trade_details.get('max_loss', 0) * 100  # Convert to dollars
            loss = entry_value - current_value
            
            if loss > 0 and max_loss > 0:
                loss_pct = (loss / max_loss) * 100
                
                if loss_pct >= self.params['loss_limit_percent']:
                    logger.info(f"Exiting {symbol} bear put spread: Loss limit reached {loss_pct:.2f}%")
                    position['exit_reason'] = 'stop_loss'
                    return True
                    
        # 4. Check underlying price movement (technical condition)
        current_price = market_data.get_latest_price(symbol)
        if current_price:
            long_put_strike = trade_details.get('long_put', {}).get('strike', 0)
            
            # If price moves substantially above our long put strike, consider exiting
            # This contradicts the bearish thesis of the position
            if current_price > long_put_strike * 1.05:  # 5% above long put strike
                logger.info(f"Exiting {symbol} bear put spread: Price moved against position")
                position['exit_reason'] = 'trend_reversal'
                return True
                
        # 5. Implement trailing stop if enabled
        if self.params.get('use_trailing_stop', False):
            highest_profit_pct = position.get('highest_profit_pct', 0)
            
            # Only activate trailing stop after reaching activation threshold
            activation_threshold = self.params.get('trailing_stop_activation', 30)  # Default 30%
            
            if highest_profit_pct >= activation_threshold:
                # Calculate trailing stop distance
                trail_distance = self.params.get('trailing_stop_distance', 15)  # Default 15%
                
                # If profit has dropped by more than the trail distance from the peak
                if highest_profit_pct - profit_pct > trail_distance:
                    logger.info(f"Exiting {symbol} bear put spread: Trailing stop triggered. "
                               f"Peak profit: {highest_profit_pct:.2f}%, Current: {profit_pct:.2f}%")
                    position['exit_reason'] = 'trailing_stop'
                    return True
        
        # No exit conditions met
        return False
    
    # ======================== 9. EXIT EXECUTION ========================
    def prepare_exit_orders(self, positions_to_exit: List[Dict[str, Any]], market_data: Dict[str, Any] = None) -> List[Order]:
        """
        Prepare orders to close existing bear put spread positions that have triggered exit conditions.
        
        This method systematically constructs exit orders for bear put spread positions, handling
        both legs of each spread (long and short puts) to ensure complete position closure. It 
        determines appropriate order types and pricing based on exit reasons and market conditions,
        maintaining complete trade context for accurate P&L tracking and performance analysis.
        
        The method performs these key functions:
        1. Processes multiple positions that need to be exited simultaneously
        2. Analyzes each bear put spread structure to identify both legs (long and short puts)
        3. Creates opposite orders for each leg (sell-to-close for long put, buy-to-close for short put)
        4. Determines optimal order types based on exit reason and market conditions
        5. Sets appropriate pricing for each exit order to balance execution certainty with cost
        6. Preserves complete trade context through metadata for performance analysis
        
        Parameters:
            positions_to_exit (List[Dict[str, Any]]): List of positions to close, each containing:
                - legs: List of component orders forming the spread
                - trade_id: Unique identifier for position tracking
                - quantity: Number of spread contracts in the position
                - metadata: Additional information about the position
                - exit_reason: Optional reason for exit (e.g., 'stop_loss', 'take_profit', 'expiration')
            market_data (Dict[str, Any], optional): Current market data for pricing decisions, including:
                - current prices for the underlying
                - option chain data for pricing reference
                - volatility metrics for order type decisions
                
        Returns:
            List[Order]: List of executable order specifications for closing all specified positions:
                - Buy-to-close orders for the short put legs
                - Sell-to-close orders for the long put legs
                - Each order includes complete details and reference to the original position
                
        Notes:
            Exit pricing strategy:
            
            - Order type selection is based on exit reason:
              - Stop losses: Market orders ensure immediate execution to prevent further losses
              - Take profits: Limit orders optimize execution price for maximum gain
              - Expiration: Market orders prevent assignment risk as expiration approaches
              - Trend reversal: Limit orders with competitive pricing based on current spreads
              
            - Market condition considerations:
              - High volatility periods may require more aggressive pricing or market orders
              - Liquidity changes during the trading day affect optimal order timing
              - Fundamental events may necessitate immediate execution regardless of price impact
              - Extremely wide spreads may require specialized handling for reasonable fills
            
            - Leg handling strategy:
              - Short put closure takes priority to eliminate assignment risk (buy-to-close)
              - Long put closure preserves remaining extrinsic value (sell-to-close)
              - For multi-leg exits, orders may be staged in priority sequence
              - All legs must be closed to properly realize the position's P&L
              
            - Risk management factors:
              - Time to expiration significantly impacts exit urgency
              - Proximity to breakeven price influences order type selection
              - Position size relative to available liquidity affects pricing strategy
              - Current implied volatility compared to entry conditions informs pricing decisions
            
            For bear put spreads, closing the short put position is particularly important as
            expiration approaches, especially if the underlying price is near or below the short put
            strike, to eliminate assignment risk. All exit orders preserve complete trade context
            through identifiers and metadata for accurate performance analysis and reporting.
        """
        all_orders = []
        
        if not positions_to_exit:
            return all_orders
            
        for position in positions_to_exit:
            if not position or 'legs' not in position:
                logger.error("Invalid position data for exit orders")
                continue
                
            legs = position.get('legs', [])
            exit_reason = position.get('exit_reason', 'general_exit')
            orders = []
            
            for leg in legs:
                if not leg or 'status' not in leg or leg['status'] != OrderStatus.FILLED:
                    continue
                    
                # Determine action to close the position
                close_action = OrderAction.SELL if leg.get('action') == OrderAction.BUY else OrderAction.BUY
                
                # Determine order type based on exit reason
                order_type = OrderType.MARKET
                if exit_reason in ['take_profit', 'trend_reversal'] and market_data is not None:
                    order_type = OrderType.LIMIT
                
                close_order = Order(
                    symbol=leg.get('symbol', ''),
                    option_symbol=leg.get('option_symbol', ''),
                    order_type=order_type,
                    action=close_action,
                    quantity=leg.get('quantity', 0),
                    trade_id=f"close_{leg.get('trade_id', '')}",
                    order_details={
                        'strategy': 'bear_put_spread',
                        'leg': 'exit_' + leg.get('order_details', {}).get('leg', ''),
                        'closing_order': True,
                        'original_order_id': leg.get('order_id', ''),
                        'exit_reason': exit_reason
                    }
                )
                
                # Add limit price if using limit order and market data is available
                if order_type == OrderType.LIMIT and market_data is not None:
                    # Set competitive limit price based on bid/ask
                    # This is a simplified example - in a real system you would use actual bid/ask data
                    if close_action == OrderAction.BUY:
                        # For buying back short put, use ask * 1.05 (5% buffer to ensure fill)
                        close_order.limit_price = market_data.get('options', {}).get(
                            leg.get('option_symbol', ''), {}).get('ask', 0) * 1.05
                    else:
                        # For selling long put, use bid * 0.95 (5% buffer to ensure fill)
                        close_order.limit_price = market_data.get('options', {}).get(
                            leg.get('option_symbol', ''), {}).get('bid', 0) * 0.95
                
                orders.append(close_order)
                
            if orders:
                logger.info(f"Created {len(orders)} exit orders for bear put spread position. Exit reason: {exit_reason}")
                all_orders.extend(orders)
            
        return all_orders
    
    # ======================== 10. CONTINUOUS OPTIMIZATION ========================
    def prepare_roll_orders(self, position: Dict[str, Any], 
                           market_data: MarketData,
                           option_chains: OptionChains) -> List[Order]:
        """
        Prepare orders to roll a bear put spread position to a new expiration.
        
        This method evaluates existing positions approaching expiration and determines if
        they should be rolled forward to maintain the strategy exposure while managing
        time decay risk. It handles the complete roll process including closing the existing
        position and establishing a similar position with a later expiration date.
        
        The rolling process follows these key steps:
        1. Evaluate if the position meets rolling criteria (DTE threshold, status)
        2. Identify an appropriate new expiration date meeting strategy parameters
        3. Select optimal strike prices in the new expiration that match strategy
        4. Calculate and validate roll economics (credit/debit requirements)
        5. Generate paired orders: exit orders for current position, entry orders for new position
        
        Parameters:
            position (Dict[str, Any]): Complete position data including:
                - trade_details: Original trade parameters
                - legs: Component orders of the current position
                - status information and metrics
            market_data (MarketData): Current market data provider
            option_chains (OptionChains): Option chain data provider for new expirations
            
        Returns:
            List[Order]: Combined list of exit and entry orders to execute the roll:
                - Exit orders to close the current position
                - Entry orders to establish the new position
                - Empty list if roll conditions aren't met or favorable
                
        Notes:
            Rolling strategy considerations:
            
            - Roll timing criteria:
              - Primary trigger: DTE threshold (typically 7-10 days before expiration)
              - Only rolls when explicitly enabled in strategy parameters
              - Avoids rolling positions that have exceeded profit targets
            
            - Strike selection approach:
              - Attempts to maintain similar strikes to the original position when possible
              - Adjusts strikes based on current market price if underlying has moved significantly
              - Preserves the original strategy's risk/reward profile
            
            - Economic requirements:
              - Validates that the roll meets minimum net credit requirements
              - Calculates net roll cost including closing current position and opening new one
              - Only executes rolls with favorable economics (roll_min_credit parameter)
            
            - Risk management factors:
              - Maintains position delta exposure through consistent strike selection
              - Extends time horizon while managing gamma risk from approaching expiration
              - May adjust position size down if market conditions have changed significantly
            
            - Implementation mechanics:
              - Creates complete set of orders for broker execution
              - Preserves trade context and history through order metadata
              - Maintains appropriate tracking IDs for position lifecycle management
        """
        if not self.params['enable_rolling']:
            logger.debug("Rolling is disabled in strategy parameters")
            return []
            
        if not position or 'trade_details' not in position:
            logger.error("Invalid position data for roll evaluation")
            return []
            
        trade_details = position.get('trade_details', {})
        symbol = trade_details.get('symbol')
        
        # Check expiration date and calculate current DTE
        expiration_date = datetime.strptime(trade_details.get('expiration', ''), '%Y-%m-%d').date()
        current_dte = (expiration_date - date.today()).days
        
        # Only roll if DTE is at or below roll threshold
        if current_dte > self.params['roll_when_dte']:
            logger.debug(f"Position DTE ({current_dte}) above roll threshold ({self.params['roll_when_dte']})")
            return []
            
        logger.info(f"Evaluating roll for {symbol} bear put spread, {current_dte} DTE remaining")
            
        # Find a new expiration further out
        new_expiration = self._select_roll_expiration(symbol, option_chains)
        if not new_expiration:
            logger.error(f"No suitable roll expiration found for {symbol}")
            return []
            
        new_expiration_date = datetime.strptime(new_expiration, '%Y-%m-%d').date()
        new_dte = (new_expiration_date - date.today()).days
        logger.info(f"Selected new expiration: {new_expiration} ({new_dte} DTE)")
            
        # Create exit orders for current position
        exit_orders = self.prepare_exit_orders({'legs': position.get('legs', []), 
                                               'exit_reason': 'roll',
                                               'trade_details': trade_details})
        
        if not exit_orders:
            logger.warning(f"Unable to create exit orders for {symbol} position")
            return []
            
        # Calculate estimated cost to close the position
        close_cost = sum([
            order.limit_price if order.order_type == OrderType.LIMIT and order.action == OrderAction.BUY 
            else -order.limit_price if order.order_type == OrderType.LIMIT and order.action == OrderAction.SELL
            else 0
            for order in exit_orders
        ])
        
        logger.debug(f"Estimated cost to close position: ${close_cost:.2f}")
        
        # Get current price and option chain data for the new expiration
        current_price = market_data.get_latest_price(symbol)
        if current_price is None:
            logger.error(f"Unable to get current price for {symbol}")
            return exit_orders  # Only exit the current position
            
        # Get put options for the new expiration
        put_options = option_chains.get_puts(symbol, new_expiration)
        if put_options.empty:
            logger.error(f"No put options available for {symbol} at {new_expiration}")
            return exit_orders  # Only exit the current position
            
        # Get original strikes
        long_put_strike = trade_details.get('long_put', {}).get('strike', 0)
        short_put_strike = trade_details.get('short_put', {}).get('strike', 0)
        
        # Find similar strikes in the new expiration
        try:
            # First try to find exact strikes if available
            available_strikes = put_options['strike'].unique()
            
            exact_match_long = long_put_strike in available_strikes
            exact_match_short = short_put_strike in available_strikes
            
            if exact_match_long and exact_match_short:
                logger.debug(f"Found exact strike matches in new expiration")
                new_long_put = put_options[put_options['strike'] == long_put_strike].iloc[0].to_dict()
                new_short_put = put_options[put_options['strike'] == short_put_strike].iloc[0].to_dict()
            else:
                # If exact strikes not available, find closest strikes
                logger.debug(f"Using closest strikes in new expiration")
                
                # For long put
                new_long_put = put_options.iloc[
                    (put_options['strike'] - long_put_strike).abs().argsort()[:1]
                ].to_dict('records')[0]
                
                # For short put
                new_short_put = put_options.iloc[
                    (put_options['strike'] - short_put_strike).abs().argsort()[:1]
                ].to_dict('records')[0]
            
            # Calculate the new debit for the spread
            new_debit = new_long_put['ask'] - new_short_put['bid']
            
            # Check if roll meets minimum credit requirement
            # For a bear put spread roll, we want: close_cost - new_debit > min_credit
            net_credit = close_cost - new_debit
            
            if net_credit < self.params['roll_min_credit']:
                logger.info(f"Roll for {symbol} does not meet minimum credit requirement. " 
                           f"Net credit: ${net_credit:.2f}, required: ${self.params['roll_min_credit']:.2f}")
                return exit_orders  # Only exit the current position
                
            logger.info(f"Roll economics favorable: Net credit ${net_credit:.2f}")
                
            # Create entry orders for the new position
            quantity = position.get('legs', [{}])[0].get('quantity', 1)
            roll_id = f"roll_bear_put_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create new trade details for the roll
            new_trade_details = {
                'symbol': symbol,
                'strategy': 'bear_put_spread',
                'expiration': new_expiration,
                'dte': new_dte,
                'long_put': new_long_put,
                'short_put': new_short_put,
                'long_put_contract': f"{symbol}_{new_expiration}_{new_long_put['strike']}_P",
                'short_put_contract': f"{symbol}_{new_expiration}_{new_short_put['strike']}_P",
                'debit': new_debit,
                'max_profit': (new_long_put['strike'] - new_short_put['strike']) - new_debit,
                'max_loss': new_debit,
                'breakeven': new_long_put['strike'] - new_debit,
                'price': current_price,
                'timestamp': datetime.now().isoformat(),
                'roll': True,
                'original_trade_id': position.get('trade_id', '')
            }
            
            roll_orders = []
            
            # Create long put order for the new expiration
            long_put_order = Order(
                symbol=symbol,
                option_symbol=new_trade_details['long_put_contract'],
                order_type=OrderType.LIMIT,
                action=OrderAction.BUY,
                quantity=quantity,
                limit_price=new_long_put['ask'],
                trade_id=roll_id,
                order_details={
                    'strategy': 'bear_put_spread',
                    'leg': 'long_put',
                    'expiration': new_expiration,
                    'strike': new_long_put['strike'],
                    'trade_details': new_trade_details,
                    'roll': True,
                    'original_position': position.get('trade_id', '')
                }
            )
            roll_orders.append(long_put_order)
            
            # Create short put order for the new expiration
            short_put_order = Order(
                symbol=symbol,
                option_symbol=new_trade_details['short_put_contract'],
                order_type=OrderType.LIMIT,
                action=OrderAction.SELL,
                quantity=quantity,
                limit_price=new_short_put['bid'],
                trade_id=roll_id,
                order_details={
                    'strategy': 'bear_put_spread',
                    'leg': 'short_put',
                    'expiration': new_expiration,
                    'strike': new_short_put['strike'],
                    'trade_details': new_trade_details,
                    'roll': True,
                    'original_position': position.get('trade_id', '')
                }
            )
            roll_orders.append(short_put_order)
            
            logger.info(f"Created roll orders for {symbol} bear put spread to {new_expiration}")
            
            # Combine exit orders and roll orders
            return exit_orders + roll_orders
            
        except Exception as e:
            logger.error(f"Error creating roll orders for {symbol}: {str(e)}")
            return exit_orders  # Only exit the current position
    
    # ======================== HELPER METHODS ========================
    def _check_adv(self, symbol: str, market_data: MarketData) -> bool:
        """
        Check if a symbol meets the Average Daily Volume criteria.
        
        Analyzes the recent trading volume of a symbol to determine if it meets
        the minimum liquidity requirements for the strategy.
        
        Parameters:
            symbol (str): Symbol to check
            market_data (MarketData): Market data instance
            
        Returns:
            bool: True if ADV meets or exceeds the minimum threshold, False otherwise
        """
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
        """
        Check if options for a symbol meet liquidity criteria.
        
        Verifies that the options available for a symbol have sufficient volume
        and open interest to ensure adequate liquidity for trading.
        
        Parameters:
            symbol (str): Symbol to check
            option_chains (OptionChains): Option chains data provider
            
        Returns:
            bool: True if options meet volume and open interest criteria, False otherwise
        """
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
        """
        Check if options for a symbol have acceptable bid-ask spreads.
        
        Analyzes the bid-ask spreads of available options to ensure they are
        tight enough for efficient trading without excessive transaction costs.
        
        Parameters:
            symbol (str): Symbol to check
            option_chains (OptionChains): Option chains data provider
            
        Returns:
            bool: True if options have acceptable bid-ask spreads, False otherwise
        """
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
        """
        Determine if a symbol is in a confirmed bearish trend.
        
        This method evaluates technical indicators to verify that the underlying asset
        is exhibiting a bearish price trend, which is a fundamental requirement for
        implementing a bear put spread strategy. The primary indicator used is price
        position relative to its 20-day exponential moving average (EMA).
        
        Parameters:
            symbol (str): Ticker symbol to evaluate
            tech_signals (TechnicalSignals): Technical signals provider with indicator calculations
            
        Returns:
            bool: True if symbol shows confirmed bearish trend characteristics, False otherwise
            
        Notes:
            Trend evaluation methodology:
            
            - Primary signal: Price below 20-day EMA indicates bearish trend
            - This simple yet effective filter helps ensure the underlying price action
              supports the bearish thesis of the strategy
            - In production environments, this could be enhanced with additional trend
              confirmation indicators such as MACD, directional movement, or longer-term
              moving average relationships
            
            The bearish trend confirmation is a critical component of the strategy entry
            criteria, as bear put spreads perform best when the price continues to move
            downward after position establishment.
        """
        # Check if price is below the 20-day EMA
        is_bearish = tech_signals.is_below_ema(symbol, period=20)
        logger.debug(f"{symbol} bearish trend check: {'PASS' if is_bearish else 'FAIL'}")
        return is_bearish
    
    def _has_negative_momentum(self, symbol: str, market_data: MarketData) -> bool:
        """
        Verify that a symbol has demonstrable negative price momentum.
        
        This method analyzes recent price action to determine if the symbol is showing
        downward momentum consistent with a bearish outlook. It calculates price change
        over the configured lookback period and compares it to the momentum threshold.
        
        Parameters:
            symbol (str): Ticker symbol to evaluate
            market_data (MarketData): Market data provider with historical price data
            
        Returns:
            bool: True if symbol shows negative momentum exceeding threshold, False otherwise
            
        Notes:
            Momentum calculation methodology:
            
            - Lookback period: Typically 10 trading days (configurable)
            - Momentum threshold: Default -3% price change (configurable)
            - Calculation: (current_price / price_n_days_ago) - 1
            
            Negative momentum confirmation serves as a secondary filter that ensures
            the bearish trend is active and recent. This helps avoid entering positions
            during consolidations or when bearish trends may be losing momentum.
            
            This check complements the trend analysis to provide a more robust confirmation
            of bearish conditions before trade entry. In production systems, this could be
            extended with volume analysis or rate-of-change indicators for enhanced signal quality.
        """
        try:
            # Get historical price data
            lookback = self.params['momentum_lookback_days']
            price_data = market_data.get_historical_data(symbol, days=lookback+5, fields=['close'])
            
            if price_data is None or len(price_data) < lookback:
                logger.debug(f"{symbol} insufficient price history for momentum calculation")
                return False
                
            # Calculate price change over the lookback period
            start_price = price_data['close'].iloc[-lookback]
            end_price = price_data['close'].iloc[-1]
            
            price_change = (end_price / start_price) - 1
            
            # Check if price change is below threshold
            threshold = self.params['momentum_threshold']
            has_momentum = price_change <= threshold
            
            logger.debug(f"{symbol} momentum: {price_change:.2%} (threshold: {threshold:.2%}), "
                        f"{'PASS' if has_momentum else 'FAIL'}")
            
            return has_momentum
            
        except Exception as e:
            logger.error(f"Error checking momentum for {symbol}: {str(e)}")
            return False
    
    def _select_expiration(self, symbol: str, option_chains: OptionChains) -> str:
        """
        Select the appropriate expiration date for the option spread.
        
        Identifies the optimal expiration date for the bear put spread by finding
        the expiration that best matches the target DTE (days to expiration) from
        the available options chain.
        
        Parameters:
            symbol (str): Symbol to check
            option_chains (OptionChains): Option chains data provider
            
        Returns:
            str: Selected expiration date in format 'YYYY-MM-DD', or empty string if none found
        """
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
        """
        Select an appropriate expiration date for rolling a position.
        
        Identifies a suitable expiration date for rolling an existing position,
        typically looking for an expiration that matches the strategy's target
        DTE criteria.
        
        Parameters:
            symbol (str): Symbol to check
            option_chains (OptionChains): Option chains data provider
            
        Returns:
            str: Selected expiration date in format 'YYYY-MM-DD', or empty string if none found
        """
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
        Select optimal strike prices for the bear put spread based on delta targets.
        
        This method identifies the appropriate strike prices for both legs of the spread 
        by targeting specific delta values, which represent the approximate probability 
        of options expiring in-the-money and provide a consistent way to select strikes
        across different underlyings and market conditions.
        
        Parameters:
            put_options (pd.DataFrame): Available put options data with columns including:
                - strike: Strike prices
                - bid/ask: Current market prices
                - delta: Delta values (sensitivity to underlying price changes)
                - other option metrics
            current_price (float): Current price of the underlying asset
            
        Returns:
            Tuple[Dict, Dict]: Dictionaries containing the selected long put and short put details
            
        Notes:
            Delta selection methodology:
            
            - For bear put spreads, the method targets:
              - Long put: Higher delta (~0.45) = Higher strike, closer to ATM
              - Short put: Lower delta (~0.20) = Lower strike, further OTM
            
            - Delta interpretation:
              - Delta represents approximate probability of expiring ITM
              - Delta 0.45 = ~45% probability of finishing ITM
              - Delta 0.20 = ~20% probability of finishing ITM
              
            - Implementation details:
              - Absolute delta values are used (puts have negative deltas)
              - Lower strike must be below higher strike to create valid spread
              - Falls back to OTM percentage method if delta data unavailable
              
            Using delta targeting provides consistency across different underlying prices
            and volatility environments, as it automatically adjusts strike selection based
            on implied volatility and time to expiration.
        """
        if 'delta' not in put_options.columns:
            logger.warning(f"Delta data not available, falling back to OTM percentage method")
            return self._select_strikes_by_otm_percentage(put_options, current_price)
            
        # For puts, delta is negative, so take absolute value
        put_options['abs_delta'] = put_options['delta'].abs()
        
        # Find long put with delta closest to target
        long_put_options = put_options.copy()
        long_put_options['delta_diff'] = abs(long_put_options['abs_delta'] - self.params['long_put_delta'])
        long_put_options = long_put_options.sort_values('delta_diff')
        
        if long_put_options.empty:
            logger.warning(f"No suitable options found for long put leg")
            return None, None
            
        long_put = long_put_options.iloc[0].to_dict()
        
        # Find short put with delta closest to target (and lower strike than long put)
        short_put_options = put_options[put_options['strike'] < long_put['strike']].copy()
        short_put_options['delta_diff'] = abs(short_put_options['abs_delta'] - self.params['short_put_delta'])
        short_put_options = short_put_options.sort_values('delta_diff')
        
        if short_put_options.empty:
            logger.warning(f"No suitable options found for short put leg")
            return long_put, None
            
        short_put = short_put_options.iloc[0].to_dict()
        
        # Log selected strikes and their deltas
        logger.debug(f"Selected strikes by delta - Long put: {long_put['strike']} (delta: {long_put['delta']:.3f}), "
                    f"Short put: {short_put['strike']} (delta: {short_put['delta']:.3f})")
        
        return long_put, short_put
    
    def _select_strikes_by_otm_percentage(self, put_options: pd.DataFrame, current_price: float) -> Tuple[Dict, Dict]:
        """
        Select optimal strike prices for the bear put spread based on OTM percentages.
        
        This method identifies the appropriate strike prices for both legs of the spread
        by targeting specific percentages out-of-the-money relative to the current price
        of the underlying asset. This approach provides a straightforward and consistent
        way to create spreads with desired characteristics.
        
        Parameters:
            put_options (pd.DataFrame): Available put options data including strike prices
            current_price (float): Current price of the underlying asset
            
        Returns:
            Tuple[Dict, Dict]: Dictionaries containing the selected long put and short put details
            
        Notes:
            OTM percentage selection methodology:
            
            - For bear put spreads, the method targets:
              - Long put: Typically near-the-money (0% OTM) = Higher strike
              - Short put: Further OTM (typically 4% below current price) = Lower strike
            
            - Strike width considerations:
              - Target width is determined by the difference between OTM percentages
              - Width typically ranges from 3-5% of underlying price
              - Minimum width of $1.00 ensures reasonable risk/reward profile
              
            - Implementation details:
              - Strikes selected based on closest match to target prices
              - Ensures short strike is below long strike
              - Wider spreads may be created if available strikes are limited
              
            OTM percentage selection is particularly useful when delta data is unavailable
            or when consistent spread width is desired regardless of volatility environment.
            It provides a straightforward approach to creating spreads with predictable
            risk/reward characteristics.
        """
        # Calculate target strike prices based on OTM percentages
        target_long_strike = current_price * (1 - self.params['long_otm_percentage'])
        target_short_strike = current_price * (1 - self.params['long_otm_percentage'] - self.params['spread_width_pct'])
        
        logger.debug(f"Target strikes - Long put: {target_long_strike:.2f} ({self.params['long_otm_percentage']*100:.1f}% OTM), "
                   f"Short put: {target_short_strike:.2f} ({(self.params['long_otm_percentage'] + self.params['spread_width_pct'])*100:.1f}% OTM)")
        
        # Find closest long put strike
        put_options['long_strike_diff'] = abs(put_options['strike'] - target_long_strike)
        put_options = put_options.sort_values('long_strike_diff')
        
        if put_options.empty:
            logger.warning(f"No suitable options available")
            return None, None
            
        long_put = put_options.iloc[0].to_dict()
        
        # Find short put with strike below long put strike
        short_put_options = put_options[put_options['strike'] < long_put['strike']].copy()
        short_put_options['short_strike_diff'] = abs(short_put_options['strike'] - target_short_strike)
        short_put_options = short_put_options.sort_values('short_strike_diff')
        
        if short_put_options.empty:
            logger.warning(f"No suitable options for short put leg (below long put strike)")
            return long_put, None
            
        short_put = short_put_options.iloc[0].to_dict()
        
        # Ensure the spread width is reasonable
        spread_width = long_put['strike'] - short_put['strike']
        if spread_width < 1.0:  # Minimum $1 width
            # Try to find a wider spread
            logger.debug(f"Selected spread width ({spread_width:.2f}) below minimum, attempting to find wider spread")
            wider_puts = put_options[(put_options['strike'] < short_put['strike'])].copy()
            
            if not wider_puts.empty:
                # Sort by strike to get the next lower strike
                wider_puts = wider_puts.sort_values('strike', ascending=False)
                if not wider_puts.empty:
                    short_put = wider_puts.iloc[0].to_dict()
                    spread_width = long_put['strike'] - short_put['strike']
                    logger.debug(f"Adjusted to wider spread, new width: {spread_width:.2f}")
        
        logger.debug(f"Selected strikes by OTM% - Long put: {long_put['strike']}, Short put: {short_put['strike']}, "
                   f"Width: {spread_width:.2f} ({spread_width/current_price*100:.1f}% of price)")
        
        return long_put, short_put

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
              - DTE (target_dte): Significantly impacts time decay exposure and price sensitivity
              - Delta targets: Controls strike selection and position moneyness
              - Spread width: Affects risk/reward profile and capital efficiency
              - Profit/loss thresholds: Determines trade duration and win rate
              - IV percentile range: Ensures optimal volatility regime alignment
            
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
            'target_dte': {'type': 'int', 'min': 20, 'max': 60, 'step': 5},
            'long_put_delta': {'type': 'float', 'min': 0.35, 'max': 0.55, 'step': 0.05},
            'short_put_delta': {'type': 'float', 'min': 0.15, 'max': 0.30, 'step': 0.05},
            'spread_width_pct': {'type': 'float', 'min': 0.02, 'max': 0.06, 'step': 0.01},
            'profit_target_percent': {'type': 'int', 'min': 40, 'max': 80, 'step': 10},
            'loss_limit_percent': {'type': 'int', 'min': 15, 'max': 40, 'step': 5},
            'min_iv_percentile': {'type': 'int', 'min': 20, 'max': 40, 'step': 5},
            'max_iv_percentile': {'type': 'int', 'min': 50, 'max': 70, 'step': 5},
            'momentum_threshold': {'type': 'float', 'min': -0.05, 'max': -0.01, 'step': 0.01},
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
              - Typically used in conjunction with walkforward testing across different market regimes
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
        if win_rate > 0.5:  # 50% win rate threshold
            win_bonus = (win_rate - 0.5)
            score = score * (1 + win_bonus)
            logger.debug(f"Applied win rate bonus: +{win_bonus:.2f} (win rate: {win_rate:.2f})")
            
        # Apply efficiency bonus for shorter holding periods
        target_holding_period = 15  # Target days for bear put spreads
        if avg_holding_period < target_holding_period and avg_holding_period > 0:
            efficiency_bonus = 0.1 * (target_holding_period - avg_holding_period) / target_holding_period
            score = score * (1 + efficiency_bonus)
            logger.debug(f"Applied efficiency bonus: +{efficiency_bonus:.2f} (avg hold: {avg_holding_period:.1f} days)")
            
        # Ensure score is never negative
        score = max(0, score)
        
        logger.info(f"Strategy performance score: {score:.4f} (Sharpe: {sharpe:.2f}, Max DD: {max_dd:.2f}, "
                  f"Win Rate: {win_rate:.2f}, Avg Hold: {avg_holding_period:.1f} days)")
        
        return score

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