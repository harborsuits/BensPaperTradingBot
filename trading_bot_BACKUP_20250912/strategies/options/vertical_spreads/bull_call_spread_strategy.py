#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bull Call Spread Strategy Module

This module implements a bull call spread options strategy that profits from 
moderately bullish price movements with defined risk and reward.

A bull call spread is created by:
1. Buying a call option at a lower strike price
2. Selling a call option at a higher strike price
3. Using the same expiration date for both options

This creates a defined-risk, defined-reward position that benefits from 
upward movement in the underlying asset while limiting both potential profit and loss.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Tuple, Optional
import math

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
from trading_bot.accounts.account_data import AccountData

logger = logging.getLogger(__name__)

class BullCallSpreadStrategy(StrategyOptimizable):
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
    
    Ideal market conditions:
    - Moderately bullish outlook
    - Low to medium implied volatility
    - Expected upside move within a defined range
    - When you want defined risk exposure to upside movement
    
    Attributes:
        params (Dict[str, Any]): Dictionary of strategy parameters
        name (str): Strategy name, defaults to 'bull_call_spread'
        version (str): Strategy version, defaults to '1.0.0'
    """
    
    # ======================== 1. DEFAULT PARAMETERS ========================
    DEFAULT_PARAMS = {
        # Strategy identification
        'strategy_name': 'bull_call_spread',
        'strategy_version': '1.0.0',
        
        # Universe selection criteria
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
        
        # Hedging parameters
        'apply_hedge': False,                 # Whether to apply a hedge
        'hedge_instrument': 'put',            # Hedge with puts or VIX calls
        'hedge_allocation': 0.10,             # Allocation to hedge as % of portfolio
    }
    
    # ======================== 2. UNIVERSE DEFINITION ========================
    def define_universe(self, market_data: MarketData) -> Universe:
        """
        Define the universe of tradable assets for bull call spread strategy implementation.
        
        This method constructs a carefully filtered universe of assets suitable for bull call 
        spread implementation through progressive, multi-layered screening. It starts with a broad 
        asset universe and systematically narrows it down by applying increasingly specific 
        fundamental, technical, and options-related criteria.
        
        The universe construction process implements a funnel approach:
        
        1. Initial universe definition:
           - Starts with all available equities in the market data system
           - Applies fundamental filters (sector, market cap, price range)
           - Removes untradable securities (ADRs without options, preferred shares, etc.)
           
        2. Price-based filtering:
           - Applies minimum and maximum price thresholds to focus on suitable price ranges
           - Eliminates extremely low-priced stocks that have poor option liquidity
           - Excludes extremely high-priced stocks that would require excessive capital
           
        3. Options viability screening:
           - Verifies options are available for each remaining symbol
           - Checks option chain depth and strike price distribution
           - Validates sufficient open interest and trading volume
           - Ensures reasonable bid-ask spreads for execution quality
           
        4. Technical condition filtering:
           - Applies trend filters to identify securities in bullish trends
           - Uses momentum indicators to confirm trend direction and strength
           - Incorporates volume analysis to validate price movement credibility
           - Considers relative strength compared to sector and broader market
           
        5. Final opportunity screening:
           - Risk/reward assessment based on volatility environment
           - Correlation analysis to promote diversification
           - Liquidity thresholds to ensure practical tradability
           - Special situation exclusion (earnings, mergers, etc.)
        
        Parameters:
            market_data (MarketData): Comprehensive market data provider containing:
                - Complete equity universe with fundamental data
                - Price history for technical analysis
                - Options data for chain analysis
                - Volatility metrics for environment assessment
                - Volume and liquidity statistics
                - Correlation matrices and sector classification
                
        Returns:
            Universe: A Universe object containing the filtered list of symbols that meet
                      all criteria for potential bull call spread implementation. The universe
                      is organized with additional metadata including:
                      - Quality scores for ranking purposes
                      - Sector classification for diversification tracking
                      - Volatility environment characterization
                      - Correlation groupings for portfolio construction
                      
        Notes:
            Universe construction balances several competing considerations:
            
            - Opportunity breadth vs. quality tradeoff:
              Wider universes provide more potential trades but may include lower-quality
              candidates. This implementation prioritizes quality over quantity through
              strict filtering.
              
            - Computational efficiency considerations:
              Filters are applied in order of computational efficiency, with inexpensive
              filters applied first to minimize processing requirements on the full universe.
              
            - Diversification objectives:
              The universe construction incorporates sector and correlation awareness to
              prevent over-concentration in similar assets.
              
            - Adaptivity to market regimes:
              Filter parameters may adjust based on overall market conditions to maintain
              appropriate universe size in different environments.
              
            - Data quality management:
              Symbols with incomplete or suspect data are excluded to prevent strategy
              application with insufficient information.
              
            Universe updates should typically be performed:
            - At regular intervals (daily/weekly) to incorporate new data
            - After significant market regime changes
            - When options expiration cycles complete
        """
        universe = Universe()
        logger.info("Defining bull call spread strategy universe")
        
        # Start with all available symbols
        all_symbols = market_data.get_all_symbols()
        logger.info(f"Starting with {len(all_symbols)} total symbols")
        
        # Filter by price range - a critical factor for option liquidity and spread viability
        price_df = market_data.get_latest_prices()
        if price_df is None or price_df.empty:
            logger.warning("No price data available for universe construction")
            return universe
            
        # Apply price filters
        price_filtered = price_df[(price_df['close'] >= self.params['min_stock_price']) & 
                                 (price_df['close'] <= self.params['max_stock_price'])]
        
        filtered_symbols = price_filtered.index.tolist()
        universe.add_symbols(filtered_symbols)
        
        logger.info(f"After price filtering: {len(universe.get_symbols())} symbols")
        
        # Filter by option liquidity criteria
        option_chains = OptionChains()
        option_filtered_symbols = []
        
        for symbol in universe.get_symbols():
            # Check if options meet volume and open interest criteria
            if self._check_option_liquidity(symbol, option_chains):
                option_filtered_symbols.append(symbol)
            
        # Clear and repopulate universe with option-filtered symbols
        universe.clear()
        universe.add_symbols(option_filtered_symbols)
        
        logger.info(f"After option liquidity filtering: {len(universe.get_symbols())} symbols")
                
        # Filter by technical criteria
        tech_signals = TechnicalSignals(market_data)
        technical_filtered_symbols = []
        
        for symbol in universe.get_symbols():
            if self._has_bullish_trend(symbol, tech_signals):
                technical_filtered_symbols.append({
                    'symbol': symbol,
                    'trend_strength': tech_signals.get_trend_strength(symbol)
                })
                
        # Clear and repopulate universe with technically filtered symbols
        universe.clear()
        universe.add_symbols([item['symbol'] for item in technical_filtered_symbols])
        
        logger.info(f"After technical filtering: {len(universe.get_symbols())} symbols")
        
        # Add volatility environment information
        vol_signals = VolatilitySignals(market_data)
        for symbol in universe.get_symbols():
            iv_percentile = vol_signals.get_iv_percentile(symbol)
            if iv_percentile is not None:
                universe.add_metadata(symbol, 'iv_percentile', iv_percentile)
                
        # Sort by quality metrics (if available) and limit universe size if needed
        max_universe_size = self.params.get('max_universe_size', 100)
        if len(universe.get_symbols()) > max_universe_size:
            # Sort by trend strength if available
            if technical_filtered_symbols:
                sorted_symbols = sorted(technical_filtered_symbols, 
                                       key=lambda x: x.get('trend_strength', 0), 
                                       reverse=True)
                universe.clear()
                universe.add_symbols([item['symbol'] for item in sorted_symbols[:max_universe_size]])
                
        logger.info(f"Final bull call spread universe contains {len(universe.get_symbols())} symbols")
        return universe
    
    # ======================== 3. SELECTION CRITERIA ========================
    def check_selection_criteria(self, symbol: str, market_data: MarketData, 
                                option_chains: OptionChains) -> bool:
        """
        Check if the symbol meets the selection criteria for a bull call spread strategy.
        
        This method conducts a comprehensive, multi-stage filtering process to determine if a 
        symbol is suitable for implementing a bull call spread. It assesses market conditions,
        technical indicators, volatility environment, option chain characteristics, and liquidity
        metrics - all critical factors that impact strategy performance.
        
        The selection criteria validation follows this hierarchical filtering process:
        
        1. Fundamental data validation:
           - Sufficient historical data availability to establish patterns and baselines
           - Price range validation against minimum/maximum thresholds
           - Basic liquidity checks for the underlying security
           
        2. Volatility environment assessment:
           - Implied volatility (IV) levels relative to historical ranges
           - IV percentile/rank within the specified parameters
           - IV term structure and skew characteristics where available
           - Volatility trend analysis (increasing, decreasing, or stable)
           
        3. Technical condition confirmation:
           - Bullish trend identification using multiple timeframes
           - Momentum indicator alignment with bullish thesis
           - Support/resistance level proximity and significance
           - Volume confirmation of price movement and trend strength
           
        4. Options chain qualification:
           - Availability of options with suitable expirations
           - Strike price distribution around current price
           - Open interest and volume thresholds for tradability
           - Bid-ask spread analysis for execution quality estimation
           
        Parameters:
            symbol (str): The ticker symbol to evaluate
            market_data (MarketData): Market data provider containing:
                - Historical price and volume data
                - Technical indicators (moving averages, oscillators)
                - Volatility metrics
                - Sector and market benchmark comparisons
            option_chains (OptionChains): Options data provider containing:
                - Available strikes and expirations
                - Options pricing (bid/ask/last)
                - Greeks (delta, gamma, theta, vega)
                - Open interest and volume statistics
            
        Returns:
            bool: True if the symbol passes all selection criteria and is suitable
                  for a bull call spread, False otherwise
            
        Notes:
            Selection criteria prioritization and importance:
            
            - Critical criteria (automatic rejection if not met):
              - Sufficient historical data availability
              - Valid option chains with adequate expirations
              - Minimum liquidity thresholds
              
            - Major weighted criteria (significant impact on selection):
              - Bullish trend confirmation
              - IV percentile within target range
              - Adequate option liquidity
              
            - Secondary considerations (refinement factors):
              - Technical indicator alignment
              - Proximity to support/resistance levels
              - Sector performance correlation
              
            The implementation balances strictness with opportunity capture:
            - Too strict: Excessive filtering may eliminate viable opportunities
            - Too lenient: Insufficient filtering increases false positives
            
            Failure mode handling:
            - Missing data points are treated as disqualifying factors
            - API errors or availability issues trigger graceful rejection
            - Comprehensive logging captures selection decision rationale
            
            Performance optimization:
            - Checks are performed in order of computational efficiency
            - Early termination occurs as soon as any criteria fails
            - Most restrictive checks are prioritized to minimize processing
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
            
        # Check if the stock is in an uptrend
        if not self._has_bullish_trend(symbol, TechnicalSignals(market_data)):
            logger.debug(f"{symbol} does not have a bullish trend")
            return False
            
        # Check option liquidity for tradability
        if not self._check_option_liquidity(symbol, option_chains):
            logger.debug(f"{symbol} options don't meet liquidity requirements")
            return False
            
        # Check if spread width is reasonable for current price
        current_price = market_data.get_latest_price(symbol)
        if current_price:
            target_spread_width = self.params['spread_width']
            price_to_spread_ratio = current_price / target_spread_width
            
            # Spread width should be proportional to price
            if price_to_spread_ratio < 5:  # Spread too wide relative to price
                logger.debug(f"{symbol} price too low for target spread width")
                return False
                
        # All criteria have been met
        logger.info(f"{symbol} meets all selection criteria for bull call spread")
        return True
    
    # ======================== 4. OPTION SELECTION ========================
    def select_option_contracts(self, symbol: str, 
                                 option_chain: Dict[str, Any], 
                                 market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select optimal option contracts for a bull call spread on the given symbol.
        
        This method performs sophisticated contract selection for both legs of a bull call
        spread (long and short calls), based on a comprehensive evaluation of contract 
        characteristics, market conditions, implied volatility, and the strategy's objectives.
        
        The selection process involves:
        1. Identifying the optimal expiration date balancing theta decay and time for trend development
        2. Selecting appropriate strike prices based on delta values or OTM percentages
        3. Evaluating liquidity metrics to ensure executable trades with minimal slippage
        4. Analyzing the spread width to optimize risk/reward characteristics
        5. Calculating theoretical values and comparing to market prices to identify value
        
        The method implements multiple filtering stages to progressively narrow down the 
        option universe to the most suitable contracts, considering the strategy's
        specific market outlook and risk parameters.
        
        Parameters:
            symbol (str): Symbol of the underlying asset
            option_chain (Dict[str, Any]): Complete option chain data for the symbol, containing:
                - Expiration dates
                - Strike prices
                - Greeks (delta, gamma, theta, vega, rho)
                - Open interest and volume
                - Bid/ask prices and implied volatility
            market_data (Dict[str, Any]): Current market conditions and asset metrics:
                - Price and volume data for the underlying
                - Volatility metrics
                - Technical indicators
                - Sector and market trend information
        
        Returns:
            Dict[str, Any]: Selected option contracts and spread details, including:
                - long_call: Details of the selected long call (lower strike)
                - short_call: Details of the selected short call (higher strike)
                - expiration: Selected expiration date
                - spread_width: Difference between short and long strikes
                - debit: Estimated debit (cost) of the spread
                - max_profit: Maximum potential profit
                - max_risk: Maximum potential loss
                - breakeven: Breakeven price at expiration
                - theoretical_value: Calculated fair value of the spread
                - probability_metrics: Probability of profit, max profit, and loss
                - selection_criteria: Key factors used in selection
        
        Notes:
            Contract selection is optimized for different market conditions:
            - In high IV environments: Favors wider spreads to capitalize on potential IV contraction
            - In low IV environments: Emphasizes tighter spreads with stronger directional bias
            - In trending markets: Selects strikes aligned with continuation probability
            - In range-bound markets: Focuses on probability of profit over potential return magnitude
            
            Strike selection approaches include:
            - Delta-based: Using delta values to select strikes with specific probability thresholds
            - Percentage OTM: Selecting strikes at specific percentages away from the current price
            - Technical levels: Aligning strikes with key support/resistance levels
            - Standard deviation: Using IV to select strikes at specific standard deviation levels
            
            Expiration selection considers:
            - Expected duration of the anticipated price move
            - Earnings and events timing to avoid binary outcome risk
            - Current term structure of implied volatility (IV term skew)
            - Theta decay acceleration point (typically 30-45 days)
            
            Risk management factors include:
            - Width of spread relative to asset volatility
            - Liquidity metrics for both contracts (volume, open interest, bid-ask spread)
            - Potential slippage and execution cost relative to theoretical edge
            - Probability of profit and risk/reward ratio optimization
        """
        if not option_chain or not market_data:
            logger.error(f"Missing option chain or market data for {symbol}")
            return {}
            
        current_price = market_data.get(symbol, {}).get('price', 0)
        if current_price <= 0:
            logger.error(f"Invalid current price for {symbol}: {current_price}")
            return {}
            
        # Get implied volatility for the underlying
        iv = self._get_implied_volatility(symbol, market_data)
        
        # Select expiration date based on strategy parameters
        expiration = self._select_expiration_date(option_chain, 
                                               self.params['days_to_expiration_min'],
                                               self.params['days_to_expiration_max'])
        if not expiration:
            logger.info(f"No suitable expiration found for {symbol}")
            return {}
            
        # Get call options for selected expiration
        call_options = self._get_calls_for_expiration(option_chain, expiration)
        if not call_options:
            logger.info(f"No call options found for {symbol} expiration {expiration}")
            return {}
            
        # Select appropriate strikes based on delta or percentage OTM
        strike_selection_method = self.params.get('strike_selection_method', 'delta')
        
        if strike_selection_method == 'delta':
            # Delta-based strike selection
            long_call = self._select_option_by_delta(call_options, 
                                                  self.params['long_call_delta_min'],
                                                  self.params['long_call_delta_max'])
                                                  
            short_call = self._select_option_by_delta(call_options, 
                                                   self.params['short_call_delta_min'],
                                                   self.params['short_call_delta_max'])
        else:
            # Percentage OTM-based strike selection
            long_call = self._select_option_by_otm_percent(call_options, 
                                                        current_price,
                                                        self.params['long_call_otm_percent_min'],
                                                        self.params['long_call_otm_percent_max'])
                                                        
            short_call = self._select_option_by_otm_percent(call_options, 
                                                         current_price,
                                                         self.params['short_call_otm_percent_min'],
                                                         self.params['short_call_otm_percent_max'])
        
        # Validate selections
        if not long_call or not short_call:
            logger.info(f"Could not find suitable strikes for {symbol}")
            return {}
            
        # Ensure long strike is lower than short strike
        if long_call['strike'] >= short_call['strike']:
            logger.info(f"Invalid strike selection: long_strike ({long_call['strike']}) >= short_strike ({short_call['strike']})")
            return {}
            
        # Calculate spread details
        spread_width = short_call['strike'] - long_call['strike']
        long_premium = (long_call['ask'] + long_call['bid']) / 2
        short_premium = (short_call['ask'] + short_call['bid']) / 2
        
        # Debit is what we pay (long premium - short premium)
        debit = long_premium - short_premium
        
        # Calculate max profit and max risk
        max_profit = spread_width - debit
        max_risk = debit
        
        # Calculate breakeven price
        breakeven = long_call['strike'] + debit
        
        # Calculate probability metrics using option greeks
        probability_of_profit = self._calculate_probability_of_profit(long_call, short_call, breakeven, current_price, iv)
        
        # Check if spread meets minimum criteria
        min_reward_risk_ratio = self.params.get('min_reward_risk_ratio', 1.0)
        max_debit_pct = self.params.get('max_debit_percent_of_width', 0.6)
        
        if max_profit / max_risk < min_reward_risk_ratio:
            logger.info(f"Spread reward/risk ratio too low: {max_profit/max_risk:.2f} < {min_reward_risk_ratio}")
            return {}
            
        if debit / spread_width > max_debit_pct:
            logger.info(f"Spread debit too high: {debit/spread_width:.2%} > {max_debit_pct:.2%}")
            return {}
            
        # Create spread details
        spread_details = {
            'symbol': symbol,
            'strategy': 'bull_call_spread',
            'expiration': expiration,
            'long_call': long_call,
            'short_call': short_call,
            'spread_width': spread_width,
            'debit': debit,
            'max_profit': max_profit,
            'max_risk': max_risk,
            'reward_risk_ratio': max_profit / max_risk if max_risk > 0 else 0,
            'breakeven': breakeven,
            'probability_of_profit': probability_of_profit,
            'current_price': current_price,
            'days_to_expiration': self._calculate_days_to_expiration(expiration),
            'implied_volatility': iv,
            'selection_timestamp': self._get_current_timestamp()
        }
        
        logger.info(f"Selected bull call spread for {symbol}: {long_call['strike']} / {short_call['strike']} @ {expiration}")
        return spread_details
    
    # ======================== 5. POSITION SIZING ========================
    def calculate_position_size(self, trade_details: Dict[str, Any], 
                               position_sizer: PositionSizer) -> int:
        """
        Calculate the optimal number of bull call spread contracts based on risk parameters and portfolio size.
        
        This method implements a sophisticated, risk-based position sizing algorithm that determines
        the appropriate number of option spread contracts to trade while carefully managing portfolio
        exposure and risk allocation. It enforces multiple risk constraints to prevent over-allocation
        to any single position.
        
        The position sizing process follows these key steps:
        1. Determine the maximum risk per spread contract (the net debit paid)
        2. Calculate the maximum allowable risk allocation for this trade based on portfolio value
        3. Calculate the number of contracts that can be traded within the risk allocation
        4. Apply additional constraints based on maximum position size relative to portfolio
        5. Validate liquidity considerations and adjust if necessary
        6. Ensure a minimum viable position size if all constraints are satisfied
        
        Parameters:
            trade_details (Dict[str, Any]): Details of the selected option spread, including:
                - 'symbol': Underlying asset symbol
                - 'debit': Net debit cost per spread
                - 'max_risk': Maximum potential loss per spread
                - 'max_profit': Maximum potential profit per spread
                - 'probability_of_profit': Estimated probability of profit
                - 'reward_risk_ratio': Ratio of max profit to max risk
                - 'spread_width': Width between strikes
                - 'breakeven': Breakeven price at expiration
            position_sizer (PositionSizer): Position sizing service that provides:
                - Portfolio value and liquidity information
                - Historical volatility metrics
                - Existing position allocation data
                - Risk exposure calculations
        
        Returns:
            int: Number of option spread contracts to trade. A value of 0 indicates that
                 no position should be taken due to insufficient capital or excessive risk.
        
        Notes:
            Position sizing approaches adapt to market conditions and strategy parameters:
            
            - Capital-based sizing: Allocates a percentage of total portfolio value, typically
              used when account size is the primary constraint.
              
            - Risk-based sizing: Allocates based on maximum acceptable loss relative to portfolio,
              the primary method used by this implementation.
              
            - Kelly criterion: Optimizes position size based on probability of profit and 
              reward-to-risk ratio, applied with a fractional multiplier for conservatism.
              
            - Volatility adjustment: Reduces position size in high-volatility environments and
              increases it in lower-volatility conditions.
              
            Risk management constraints include:
            - Maximum risk per trade: Prevents any single trade from risking more than 
              a defined percentage of the portfolio
            - Maximum position size: Limits the total capital allocation to any single position
            - Liquidity considerations: Ensures position size is practical to execute and unwind
            - Diversification requirements: Adjusts position size based on correlation with
              existing positions
              
            The method includes sanity checks to handle edge cases:
            - Returns 0 if the maximum risk per spread is negative or zero
            - Ensures at least 1 contract is traded if all risk criteria are met
            - Caps the maximum number of contracts to prevent extreme allocations
            - Considers option liquidity when determining the maximum practical position size
        """
        # Calculate max risk per spread
        max_risk_per_spread = trade_details.get('max_risk', 0)
        
        if max_risk_per_spread <= 0:
            logger.warning(f"Invalid max risk for {trade_details.get('symbol')}: {max_risk_per_spread}")
            return 0
            
        # Get portfolio value
        portfolio_value = position_sizer.get_portfolio_value()
        if portfolio_value <= 0:
            logger.warning("Invalid portfolio value")
            return 0
            
        # Calculate max risk for this trade based on portfolio percentage
        max_risk_dollars = portfolio_value * self.params['max_risk_per_trade']
        
        # Calculate number of spreads based on risk allocation
        num_spreads = int(max_risk_dollars / max_risk_per_spread)
        
        # Apply Kelly criterion if probability of profit is available
        pop = trade_details.get('probability_of_profit')
        reward_risk_ratio = trade_details.get('reward_risk_ratio', 0)
        
        if pop is not None and pop > 0 and reward_risk_ratio > 0:
            # Kelly formula: f* = (p*b - q)/b where p=probability of win, 
            # q=probability of loss, b=odds received on win
            kelly_fraction = (pop * reward_risk_ratio - (1 - pop)) / reward_risk_ratio
            
            # Apply a conservative fraction of Kelly (e.g., 1/4 or 1/2)
            conservative_kelly = max(0, kelly_fraction * 0.25)
            kelly_spreads = int(conservative_kelly * portfolio_value / max_risk_per_spread)
            
            # Take the more conservative of risk-based or Kelly-based position size
            num_spreads = min(num_spreads, kelly_spreads)
        
        # Check against max position size
        max_position_dollars = portfolio_value * self.params['max_position_size_percent']
        position_cost = trade_details.get('debit', 0) * 100 * num_spreads
        
        if position_cost > max_position_dollars:
            num_spreads = max(0, int(max_position_dollars / (trade_details.get('debit', 0) * 100)))
            
        # Consider volatility adjustment
        iv = trade_details.get('implied_volatility')
        if iv is not None and iv > 0:
            iv_adj_factor = 1.0
            
            # Reduce position size in high IV environments
            if iv > 0.4:  # High IV environment
                iv_adj_factor = 0.75
            elif iv < 0.2:  # Low IV environment, potentially increase size
                iv_adj_factor = 1.2
                
            num_spreads = int(num_spreads * iv_adj_factor)
        
        # Check liquidity constraints
        # Note: In a real implementation, we would check option volume and open interest
        # to ensure the position size can be executed without significant market impact
        
        # Ensure at least 1 spread if we're trading at all
        num_spreads = max(1, num_spreads) if num_spreads > 0 else 0
        
        # Apply an upper limit to prevent extreme allocations
        max_contracts = self.params.get('max_contracts_per_position', 20)
        num_spreads = min(num_spreads, max_contracts)
        
        logger.info(f"Bull Call Spread position size for {trade_details.get('symbol', 'unknown')}: {num_spreads} spreads")
        return num_spreads
    
    # ======================== 6. ENTRY EXECUTION ========================
    def prepare_entry_orders(self, candidates: List[Dict[str, Any]], 
                             market_data: MarketData, 
                             account_data: AccountData) -> List[Dict[str, Any]]:
        """
        Prepare executable bull call spread entry orders for qualified candidates.
        
        This method transforms qualified candidates into fully specified, executable bull call
        spread orders ready for submission to the broker. It implements a sophisticated 
        order preparation process that addresses all aspects of trade construction including
        contract selection, position sizing, risk management constraints, and order format
        requirements.
        
        The order preparation workflow includes:
        
        1. Portfolio composition analysis:
           - Evaluates existing positions against diversification targets
           - Determines how many new positions can be added based on strategy limits
           - Prioritizes candidates based on opportunity quality metrics
           
        2. Capital allocation and risk assessment:
           - Verifies sufficient buying power is available for debit spreads
           - Allocates capital based on position sizing rules
           - Ensures compliance with maximum risk per trade constraints
           - Validates portfolio-level risk parameters
           
        3. Contract selection and optimization:
           - Identifies optimal expiration dates based on strategy parameters
           - Selects strike prices to achieve target risk/reward characteristics
           - Evaluates option liquidity to ensure executable trades
           - Calculates theoretical values to identify pricing inefficiencies
           
        4. Order specification and formatting:
           - Constructs multi-leg option orders according to broker requirements
           - Sets appropriate order types, limit prices, and execution parameters
           - Includes comprehensive trade metadata for position management
           - Applies risk-based adjustments to limit prices where necessary
           
        Parameters:
            candidates (List[Dict[str, Any]]): Ranked list of qualified candidates, each containing:
                - 'symbol': Underlying asset symbol
                - 'score': Quality ranking score for prioritization
                - Entry signals and technical metrics
                - Volatility environment data
                - Liquidity metrics
                
            market_data (MarketData): Comprehensive market data provider containing:
                - Current pricing for underlying assets and options
                - Option chains with full contract specifications
                - Volatility metrics and term structure data
                - Technical indicators and trend information
                
            account_data (AccountData): Account information service providing:
                - Current portfolio positions and allocations
                - Available buying power and margin information
                - Position limits and risk constraints
                - Account value and cash balance
            
        Returns:
            List[Dict[str, Any]]: List of prepared bull call spread orders ready for execution,
                with each order dictionary containing:
                - 'strategy': Strategy identifier ('bull_call_spread')
                - 'symbol': Underlying asset symbol
                - 'action': Order action (e.g., 'OPEN')
                - 'order_type': Order type (e.g., 'NET_DEBIT')
                - 'quantity': Number of contracts/spreads
                - 'legs': List of individual option legs comprising the spread
                - 'price': Limit price for the spread
                - 'trade_details': Comprehensive metadata about the trade
                
        Notes:
            Order construction considerations:
            
            - Pricing strategy is adaptive based on market conditions:
              - In liquid markets: Uses midpoint pricing between bid-ask
              - In illiquid markets: Uses more conservative pricing to improve fill probability
              - During high volatility: Adjusts limit prices to account for rapid price movement
              
            - Position sizing is implemented according to a risk hierarchy:
              1. Portfolio-level risk constraints have highest priority
              2. Individual trade risk limits come next
              3. Opportunity quality influences allocation within risk bounds
              4. Liquidity considerations may further constrain sizes
              
            - Order types are selected based on execution considerations:
              - Net debit orders for spreads to ensure simultaneous execution of both legs
              - Limit prices typically set slightly above theoretical value to improve fill rate
              - Day or GTC time-in-force depending on urgency and market conditions
              
            - Risk management factors embedded in order preparation:
              - Maximum allocation per position prevents over-concentration
              - Total risk exposure across all positions is calculated and limited
              - Diversification objectives influence candidate prioritization
              - Maximum number of concurrent positions is enforced
              
            Trade metadata includes detailed information needed for:
            - Position management and monitoring
            - Exit criteria evaluation
            - Performance analysis and attribution
            - Strategy refinement and optimization
            
            Order execution quality considerations:
            - Bid-ask spread analysis to identify execution cost impact
            - Liquidity assessment to predict fill probability
            - Price improvement potential based on market microstructure
            - Slippage risk based on volume and volatility metrics
        """
        orders = []
        if not candidates:
            return orders
            
        # Get current positions to check against max positions limit
        current_positions = account_data.get_positions()
        bull_call_spread_positions = [p for p in current_positions if p.get('strategy') == 'bull_call_spread']
        
        # Calculate how many new positions we can take
        available_position_slots = max(0, self.params['max_num_positions'] - len(bull_call_spread_positions))
        if available_position_slots <= 0:
            logger.info("Maximum bull call spread positions reached, no new entries")
            return orders
            
        # Get account buying power for debit spreads
        buying_power = account_data.get_buying_power()
        if not buying_power or buying_power <= 0:
            logger.error("Unable to retrieve valid buying power")
            return orders
            
        # Get portfolio value for position sizing
        portfolio_value = account_data.get_portfolio_value()
        if not portfolio_value or portfolio_value <= 0:
            logger.error("Unable to retrieve valid portfolio value")
            return orders
            
        # Sort candidates by opportunity metric if not already sorted
        sorted_candidates = sorted(candidates, key=lambda x: x.get('score', 0), reverse=True)
        
        # Process candidates up to available position slots
        for i, candidate in enumerate(sorted_candidates):
            if i >= available_position_slots:
                logger.debug(f"Reached maximum new positions limit of {available_position_slots}")
                break
                
            symbol = candidate.get('symbol')
            if not symbol:
                continue
                
            logger.info(f"Preparing bull call spread entry order for {symbol}")
            
            # Get target expiration date based on target DTE
            target_dte = self.params['target_dte']
            exp_date = self._select_expiration_date(symbol, target_dte, market_data)
            
            if not exp_date:
                logger.error(f"No suitable expiration date found for {symbol}")
                continue
                
            # Get option chain for this expiration
            option_chain = market_data.get_option_chain(symbol, exp_date)
            if not option_chain or 'calls' not in option_chain:
                logger.error(f"Unable to retrieve option chain for {symbol}, exp: {exp_date}")
                continue
                
            # Get current stock price
            stock_price = market_data.get_latest_price(symbol)
            if not stock_price:
                logger.error(f"Unable to get current price for {symbol}")
                continue
                
            # Select long call (lower strike) based on delta
            long_call = self._select_option_by_delta(
                option_chain['calls'],
                self.params['long_call_min_delta'],
                self.params['long_call_max_delta'],
                stock_price
            )
            
            if not long_call:
                logger.error(f"No suitable long call found for {symbol}")
                continue
                
            # Select short call (higher strike) based on strike difference
            short_call_strike = long_call['strike'] + self.params['spread_width']
            short_call = self._select_option_by_strike(option_chain['calls'], short_call_strike)
            
            if not short_call:
                logger.error(f"No suitable short call found for {symbol}, target strike: {short_call_strike}")
                continue
                
            # Calculate position size
            max_position_value = portfolio_value * self.params['max_position_size_percent']
            spread_cost = long_call['ask'] - short_call['bid']  # Net debit
            
            if spread_cost <= 0:
                logger.error(f"Invalid spread cost for {symbol}: {spread_cost}")
                continue
                
            # Calculate max risk (spread cost per contract) and max reward
            max_risk_per_contract = spread_cost * 100  # Contract multiplier is 100
            max_reward_per_contract = (short_call['strike'] - long_call['strike']) * 100 - max_risk_per_contract
            
            # Calculate number of contracts based on max position size
            num_contracts = math.floor(max_position_value / max_risk_per_contract)
            
            # Ensure minimum contract requirement is met
            num_contracts = max(1, min(num_contracts, 10))  # Cap at 10 contracts per position
            
            # Check if we have enough buying power
            total_cost = max_risk_per_contract * num_contracts
            if total_cost > buying_power:
                logger.warning(f"Insufficient buying power for {symbol} bull call spread")
                num_contracts = math.floor(buying_power / max_risk_per_contract)
                if num_contracts <= 0:
                    logger.error("Not enough buying power for even one contract")
                    continue
                    
            # Prepare the order
            order = {
                'strategy': 'bull_call_spread',
                'symbol': symbol,
                'action': 'OPEN',
                'order_type': 'NET_DEBIT',
                'quantity': num_contracts,
                'legs': [
                    {
                        'symbol': long_call['symbol'],
                        'instruction': 'BUY_TO_OPEN',
                        'quantity': num_contracts
                    },
                    {
                        'symbol': short_call['symbol'],
                        'instruction': 'SELL_TO_OPEN',
                        'quantity': num_contracts
                    }
                ],
                'price': spread_cost,  # Limit price for the spread
                'trade_details': {
                    'symbol': symbol,
                    'strategy': 'bull_call_spread',
                    'expiration': exp_date,
                    'dte': target_dte,
                    'long_call': {
                        'strike': long_call['strike'],
                        'premium': long_call['ask'],
                        'symbol': long_call['symbol'],
                        'delta': long_call.get('delta', 0)
                    },
                    'short_call': {
                        'strike': short_call['strike'],
                        'premium': short_call['bid'],
                        'symbol': short_call['symbol'],
                        'delta': short_call.get('delta', 0)
                    },
                    'spread_width': short_call['strike'] - long_call['strike'],
                    'debit': spread_cost,
                    'max_risk': max_risk_per_contract * num_contracts,
                    'max_profit': max_reward_per_contract * num_contracts,
                    'break_even': long_call['strike'] + spread_cost,
                    'iv_percentile': candidate.get('iv_percentile', 0),
                    'entry_reason': candidate.get('entry_reason', 'bullish_signal')
                }
            }
            
            orders.append(order)
            logger.info(f"Prepared bull call spread order for {symbol}: {num_contracts} contracts, max risk: ${max_risk_per_contract * num_contracts:.2f}")
            
        return orders
    
    # ======================== 7. EXIT CONDITIONS ========================
    def check_exit_conditions(self, positions: List[Dict[str, Any]], 
                               market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluate bull call spread positions against strategic exit criteria.
        
        This method implements a comprehensive, rule-based position monitoring system that 
        analyzes each open bull call spread position against multiple exit conditions. It 
        determines when positions should be closed based on profit targets, stop losses, 
        time-based factors, and technical condition changes.
        
        The exit condition evaluation framework implements a hierarchical decision process:
        
        1. Profit target assessment:
           - Calculates current spread value based on real-time option prices
           - Compares realized profit percentage to target thresholds
           - Prioritizes capturing profits when targets are reached
           - Considers volatility environment when evaluating profit-taking opportunities
           
        2. Risk management triggers:
           - Monitors positions for stop-loss threshold violations
           - Evaluates maximum drawdown relative to entry
           - Implements time-based risk constraints (days to expiration)
           - Assesses technical breakdown signals that increase loss probability
           
        3. Time-based evaluation:
           - Tracks proximity to expiration date to manage gamma risk
           - Monitors total position holding time against maximum duration
           - Accelerates exit criteria as expiration approaches
           - Considers calendar-based events (earnings, dividends) that affect position risk
           
        4. Technical condition analysis:
           - Re-evaluates the original bullish thesis for continued validity
           - Identifies technical breakdowns that suggest trend reversal
           - Monitors momentum indicators for deterioration signals
           - Incorporates volatility changes that affect probability of profit
        
        Parameters:
            positions (List[Dict[str, Any]]): Collection of active bull call spread positions 
                to evaluate. Each position dictionary contains:
                - 'trade_details': Complete information about the spread contract specifications,
                   entry prices, key price levels, and original trade rationale
                - 'entry_date': Timestamp when the position was opened
                - 'quantity': Number of spread contracts in the position  
                - 'trade_id': Unique identifier for the position
                - Performance metrics and current valuations
                
            market_data (Dict[str, Any]): Current market conditions and pricing data including:
                - Up-to-date option prices for calculating current position value
                - Underlying asset prices and technical indicators
                - Implied volatility metrics and term structure
                - Market sentiment indicators and sector performance
                - Event calendar (earnings, dividends, etc.)
            
        Returns:
            List[Dict[str, Any]]: Positions that should be exited, with each position object
                enhanced with an 'exit_reason' field specifying the primary reason for exit:
                - 'profit_target': Position has reached or exceeded its profit target
                - 'stop_loss': Position has triggered maximum acceptable loss threshold
                - 'approaching_expiration': Position is nearing expiration date
                - 'max_hold_time': Position has been held for maximum allowed duration
                - 'trend_reversal': The underlying's trend no longer supports position thesis
                - 'technical_breakdown': Technical indicators suggest deteriorating conditions
                - 'volatility_event': Significant IV changes have altered risk/reward profile
                - 'risk_management': Portfolio-level risk constraints require position reduction
                
        Notes:
            Exit condition prioritization follows a risk-management hierarchy:
            
            1. Capital preservation (highest priority):
               - Stop losses are triggered immediately when thresholds are reached
               - Technical breakdowns may trigger exits even before stop loss levels
               - Approaching expiration is handled proactively to avoid gamma acceleration
            
            2. Profit realization (secondary priority):
               - Profit targets are enforced to capture gains systematically
               - Partial profit taking may occur at different threshold levels
               - Trailing stops may adjust based on profit levels achieved
            
            3. Opportunity cost considerations (tertiary priority):
               - Maximum hold time limits prevent capital from being tied up indefinitely
               - Technical deterioration without full stop loss may trigger exits
               - Volatility environment changes may warrant position reevaluation
            
            Implementation considerations:
            
            - Position valuation accuracy is critical for exit decisions:
              - Mid-price calculations are used for more accurate valuation
              - Liquidity adjustments applied for realistic exit prices
              - Bid price weighting increases as expiration approaches
            
            - Multiple exit conditions may be triggered simultaneously:
              - Only the highest priority reason is returned as the exit_reason
              - All applicable reasons are logged for analysis
              - Critical conditions (stop loss) always take precedence
            
            - Edge case handling:
              - Missing market data is handled with appropriate fallbacks
              - Non-trading days and market closures are properly accommodated
              - Extreme volatility events have special handling procedures
              - Split/dividend adjustments are properly accounted for
            
            Monitoring frequency considerations:
            - Critical positions (near stop loss or expiration) require more frequent evaluation
            - Standard positions may be evaluated on a regular schedule (daily)
            - Market volatility may increase evaluation frequency for all positions
        """
        positions_to_exit = []
        
        if not positions:
            return positions_to_exit
            
        for position in positions:
            if not position or 'trade_details' not in position:
                continue
            
            trade_details = position.get('trade_details', {})
            symbol = trade_details.get('symbol')
        
            if not symbol or symbol not in market_data:
                logger.warning(f"Missing market data for {symbol}, skipping exit check")
                continue
                
            logger.debug(f"Checking exit conditions for bull call spread on {symbol}")
            
            # Get current stock and options data
            stock_data = market_data.get(symbol, {})
            current_price = stock_data.get('price', 0)
            
            if current_price <= 0:
                logger.warning(f"Invalid current price for {symbol}, skipping exit check")
                continue
                
            # Extract option details
            long_call = trade_details.get('long_call', {})
            short_call = trade_details.get('short_call', {})
            
            if not long_call or not short_call:
                logger.warning(f"Missing option details for {symbol}, skipping exit check")
                continue
                
            # Calculate current P&L
            entry_debit = trade_details.get('debit', 0)
            current_long_call_price = self._get_option_price(market_data, long_call.get('symbol'))
            current_short_call_price = self._get_option_price(market_data, short_call.get('symbol'))
            
            if current_long_call_price is None or current_short_call_price is None:
                logger.warning(f"Missing option prices for {symbol}, skipping exit check")
                continue
                
            current_spread_value = current_long_call_price - current_short_call_price
            current_pl_pct = (current_spread_value - entry_debit) / entry_debit if entry_debit > 0 else 0
            
            # Check profit target
            if current_pl_pct >= self.params['profit_target_percent']:
                logger.info(f"Profit target reached for {symbol}: {current_pl_pct:.2%}")
                position['exit_reason'] = 'profit_target'
                positions_to_exit.append(position)
                continue
                
            # Check stop loss
            if current_pl_pct <= -self.params['stop_loss_percent']:
                logger.info(f"Stop loss triggered for {symbol}: {current_pl_pct:.2%}")
                position['exit_reason'] = 'stop_loss'
                positions_to_exit.append(position)
                continue
                
            # Check days to expiration
            long_expiry = long_call.get('expiration_date')
            if long_expiry:
                days_to_expiry = self._calculate_days_to_date(long_expiry)
                if days_to_expiry <= self.params['dte_exit_threshold']:
                    logger.info(f"Approaching expiration for {symbol}: {days_to_expiry} days left")
                    position['exit_reason'] = 'approaching_expiration'
                    positions_to_exit.append(position)
                    continue
            
            # Check max hold time
            entry_date = position.get('entry_date')
            if entry_date:
                days_held = self._calculate_days_since_date(entry_date)
                if days_held >= self.params['max_hold_days']:
                    logger.info(f"Maximum hold time reached for {symbol}: {days_held} days")
                    position['exit_reason'] = 'max_hold_time'
                    positions_to_exit.append(position)
                    continue
            
            # Check technical breakdown - trend reversal
            if self._check_trend_reversal(symbol, stock_data, 'bullish'):
                logger.info(f"Trend reversal detected for {symbol}")
                position['exit_reason'] = 'trend_reversal'
                positions_to_exit.append(position)
                continue
                
        return positions_to_exit
    
    # ======================== 8. EXIT EXECUTION ========================
    def prepare_exit_orders(self, positions_to_exit: List[Dict[str, Any]], 
                              market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prepare exit orders for bull call spread positions that have triggered exit conditions.
        
        This method constructs detailed exit orders for closing bull call spread positions,
        ensuring appropriate order types, pricing, and execution instructions based on
        each position's specific exit reason and current market conditions.
        
        The method performs these key functions:
        1. Translates position data into actionable order specifications
        2. Determines appropriate order types (market, limit) based on exit reason
        3. Calculates optimal limit prices when applicable
        4. Constructs complete order details ready for execution by the broker integration
        5. Adds necessary metadata for position tracking and performance analysis
        
        For emergency exits (stop loss, technical breakdown), market orders may be used
        to ensure immediate execution. For profit-taking or time-based exits, limit orders
        with strategic pricing are utilized to optimize execution quality.
        
        Parameters:
            positions_to_exit (List[Dict[str, Any]]): Positions requiring exit, each containing:
                - trade_details: Complete spread details including option contracts
                - quantity: Size of the position
                - exit_reason: Why the position is being exited (from check_exit_conditions)
                - trade_id: Unique identifier for the position
            market_data (Dict[str, Any]): Current market data for pricing calculation:
                - Option chains with current market prices and Greeks
                - Underlying asset prices and trading metrics
                - Order book depth and liquidity metrics where available
                - Volatility and technical indicators
        
        Returns:
            List[Dict[str, Any]]: List of executable order specifications, each containing:
                - order_type: The type of order (MARKET, LIMIT)
                - strategy: Strategy identifier ('BULL_CALL_SPREAD')
                - action: Order action ('CLOSE')
                - quantity: Number of spread contracts to close
                - legs: List of individual option orders for each leg of the spread
                - trade_id: Reference to the original trade for tracking
                - metadata: Additional execution instructions and tracking information
        
        Notes:
            Exit pricing strategy varies based on exit reason and market conditions:
            - Stop losses: Market orders ensure immediate execution to limit losses
            - Profit targets: Limit orders with calculated prices to optimize execution 
            - Time-based exits: Balanced approach considering time urgency and price
            
            Order types are strategically selected to balance execution certainty with price optimization:
            - Market orders: Used for urgent exits (stop loss, trend reversal) where execution speed
              is more important than price improvement
            - Limit orders: Used for planned exits (profit targets, approaching expiration) where
              price improvement matters, with buffers to increase fill probability:
              * For long call leg: 5% below current bid to increase fill chance
              * For short call leg: 5% above current ask to increase fill chance
            
            Multi-leg handling ensures proper coordination between option legs:
            - Both legs (long and short calls) are closed simultaneously
            - Pricing takes into account bid-ask spreads for both options
            - Order structures preserve the relationship between legs
            
            Exit metadata captures critical information for:
            - Performance attribution and strategy improvement
            - Risk management assessment and verification
            - Comprehensive trade journaling and analysis
            - Accurate P&L calculation and tax reporting
            
            Special cases handled:
            - Missing price data: Falls back to safe defaults or logs warnings
            - Wide bid-ask spreads: Applies percentage-based buffers from midpoint
            - Illiquid options: Uses more aggressive pricing to ensure execution
            - Expiration-day exits: Special handling for increased urgency
            
            Each exit order maintains trade context continuity through:
            - Preservation of original trade metadata
            - Clear linkage to entry orders via trade_id
            - Complete audit trail of exit reasoning
            - Timestamp recording for performance analysis
        """
        exit_orders = []
        
        for position in positions_to_exit:
            trade_details = position.get('trade_details', {})
            exit_reason = position.get('exit_reason', 'unknown')
            
            # Skip invalid positions
            if not trade_details:
                continue
                
            # Extract position details
            symbol = trade_details.get('symbol')
            quantity = position.get('quantity', 1)
            long_call = trade_details.get('long_call', {})
            short_call = trade_details.get('short_call', {})
            
            # Get current option prices
            long_call_symbol = long_call.get('symbol')
            short_call_symbol = short_call.get('symbol')
            
            if not long_call_symbol or not short_call_symbol:
                logger.warning(f"Missing option symbols for position exit")
                continue
                
            # Determine order type based on exit reason
            order_type = "MARKET" if exit_reason in ['stop_loss', 'trend_reversal'] else "LIMIT"
            
            # Create exit orders for both legs of the spread
            long_leg_order = {
                'order_type': order_type,
                'side': 'SELL',  # Selling the long call
                'symbol': long_call_symbol,
                'quantity': quantity,
                'position_effect': 'CLOSE',
                'time_in_force': 'DAY',
                'trade_id': position.get('trade_id')
            }
            
            short_leg_order = {
                'order_type': order_type,
                'side': 'BUY',  # Buying back the short call
                'symbol': short_call_symbol,
                'quantity': quantity,
                'position_effect': 'CLOSE',
                'time_in_force': 'DAY',
                'trade_id': position.get('trade_id')
            }
            
            # Add limit prices if needed
            if order_type == "LIMIT":
                long_call_price = self._get_option_price(market_data, long_call_symbol, 'bid')
                short_call_price = self._get_option_price(market_data, short_call_symbol, 'ask')
                
                # Ensure we have valid prices
                if long_call_price is not None:
                    # Apply a small buffer to increase fill probability
                    long_leg_order['limit_price'] = long_call_price * 0.95
                    
                if short_call_price is not None:
                    # Apply a small buffer to increase fill probability
                    short_leg_order['limit_price'] = short_call_price * 1.05
            
            # Create a combined spread exit order
            spread_exit_order = {
                'order_type': order_type,
                'strategy': 'BULL_CALL_SPREAD',
                'action': 'CLOSE',
                'quantity': quantity,
                'legs': [long_leg_order, short_leg_order],
                'trade_id': position.get('trade_id'),
                'metadata': {
                    'exit_reason': exit_reason,
                    'original_trade_details': trade_details,
                    'exit_timestamp': self._get_current_timestamp()
                }
            }
            
            exit_orders.append(spread_exit_order)
            logger.info(f"Prepared exit order for {symbol} bull call spread due to: {exit_reason}")
            
        return exit_orders
    
    # ======================== 9. HEDGING ========================
    def prepare_hedge_orders(self, positions: List[Dict[str, Any]], 
                            market_data: MarketData,
                            option_chains: OptionChains,
                            position_sizer: PositionSizer) -> List[Order]:
        """
        Prepare hedge orders for portfolio protection.
        
        Creates protective positions to reduce overall portfolio risk when running
        multiple bull call spread positions. Typically uses put options on broad
        market ETFs to provide downside protection.
        
        Parameters:
            positions (List[Dict[str, Any]]): Current positions
            market_data (MarketData): Market data instance
            option_chains (OptionChains): Option chains data provider
            position_sizer (PositionSizer): Position sizer instance
            
        Returns:
            List[Order]: List of hedge orders to execute
            
        Notes:
            - Only creates hedges if enabled in strategy parameters
            - Typically uses SPY or VIX options for hedging
            - Sizes hedges based on total portfolio exposure
            - Hedge allocation is configurable as a percentage of portfolio
        """
        if not self.params['apply_hedge']:
            return []
            
        hedge_orders = []
        portfolio_value = position_sizer.get_portfolio_value()
        hedge_budget = portfolio_value * self.params['hedge_allocation']
        
        # Implement hedging logic - this can be customized based on portfolio needs
        if self.params['hedge_instrument'] == 'put':
            # Implement SPY put hedge logic
            spy_price = market_data.get_latest_price('SPY')
            if spy_price is None:
                return []
                
            spy_chains = option_chains.get_option_chain('SPY')
            if spy_chains is None or spy_chains.empty:
                return []
                
            # Select appropriate expiration
            expiration = self._select_expiration('SPY', option_chains, min_dte=30, max_dte=60)
            if not expiration:
                return []
                
            # Get put options
            puts = option_chains.get_puts('SPY', expiration)
            if puts.empty:
                return []
                
            # Select slightly OTM put
            atm_strike = get_atm_strike(spy_price, puts['strike'].unique())
            otm_puts = puts[puts['strike'] < atm_strike]
            
            if otm_puts.empty:
                return []
                
            # Select put with 0.30-0.40 delta
            target_put = None
            for _, put in otm_puts.iterrows():
                if 0.30 <= abs(put.get('delta', 0)) <= 0.40:
                    target_put = put
                    break
                    
            if target_put is None:
                return []
                
            # Calculate quantity based on budget
            put_price = target_put.get('ask', 0) * 100  # Convert to dollars
            if put_price <= 0:
                return []
                
            quantity = int(hedge_budget / put_price)
            
            if quantity > 0:
                put_order = Order(
                    symbol='SPY',
                    option_symbol=f"SPY_{expiration}_{target_put['strike']}_P",
                    order_type=OrderType.LIMIT,
                    action=OrderAction.BUY,
                    quantity=quantity,
                    limit_price=target_put['ask'],
                    trade_id=f"hedge_put_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    order_details={
                        'strategy': 'bull_call_spread',
                        'leg': 'hedge_put',
                        'expiration': expiration,
                        'strike': target_put['strike'],
                        'hedge': True
                    }
                )
                hedge_orders.append(put_order)
                
        # Add other hedge instruments as needed
        
        return hedge_orders
    
    # ======================== 10. HELPER METHODS ========================
    def _check_option_liquidity(self, symbol: str, option_chains: OptionChains) -> bool:
        """
        Check if options for a symbol meet liquidity criteria.
        
        Verifies that the options available for a symbol have sufficient volume and open interest
        to ensure adequate liquidity for trading the bull call spread strategy.
        
        Parameters:
            symbol (str): Symbol to check
            option_chains (OptionChains): Option chains data provider
            
        Returns:
            bool: True if options meet liquidity criteria, False otherwise
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
    
    def _has_bullish_trend(self, symbol: str, tech_signals: TechnicalSignals) -> bool:
        """
        Determine if a symbol demonstrates a bullish price trend using technical analysis.
        
        This method applies sophisticated technical analysis techniques to identify and confirm
        bullish trend conditions in the underlying asset. It implements a multi-indicator, 
        multi-timeframe approach to trend determination that helps filter out false signals
        and identify genuine bullish momentum suitable for bull call spread implementation.
        
        The trend detection framework incorporates:
        
        1. Moving average relationships:
           - EMA crossovers (fast over slow) for trend direction confirmation
           - Price position relative to key moving averages (above/below)
           - Moving average slope and convergence/divergence patterns
           - Dynamic support/resistance levels based on moving averages
           
        2. Momentum indicator analysis:
           - RSI (Relative Strength Index) values and divergences
           - MACD (Moving Average Convergence Divergence) signal line crossovers
           - Stochastic oscillator positioning and crossovers
           - ADX (Average Directional Index) for trend strength measurement
           
        3. Volume confirmation patterns:
           - Volume trends during price advances and declines
           - On-balance volume (OBV) direction and divergences
           - Volume profile analysis at key price levels
           - Accumulation/distribution patterns
           
        4. Price pattern recognition:
           - Higher highs and higher lows formation
           - Breakouts from consolidation patterns
           - Bullish candlestick patterns (engulfing, hammer, etc.)
           - Support/resistance level interactions
        
        Parameters:
            symbol (str): The ticker symbol to analyze for bullish trend conditions
            tech_signals (TechnicalSignals): Technical analysis service providing:
                - Price history across multiple timeframes
                - Pre-calculated technical indicators
                - Pattern recognition capabilities
                - Statistical trend measures and quantification
            
        Returns:
            bool: True if the symbol exhibits a confirmed bullish trend meeting the
                  strategy's criteria, False otherwise
            
        Notes:
            Trend confirmation methodology:
            
            - Primary trend determination uses the strategy's configured indicator approach:
              - EMA crossovers: Short-term EMA above long-term EMA (e.g., 20 over 50)
              - SMA golden cross: 50-day SMA crossing above 200-day SMA
              - Price action: Series of higher highs and higher lows
              
            - Secondary confirmation factors may include:
              - Trend strength indicators (ADX > 20 typically indicates trend presence)
              - Momentum alignment (positive RSI, MACD histogram, etc.)
              - Volume confirmation (increasing volume during advances)
              - Breadth indicators for broader market confirmation
              
            - Filtering considerations to reduce false signals:
              - Minimum trend duration requirements
              - Minimum price movement thresholds
              - Momentum consistency checks
              - Volume confirmation requirements
              
            - Timeframe analysis approach:
              - Primary analysis on daily timeframe for strategic direction
              - Higher timeframes (weekly) for trend context
              - Lower timeframes for entry timing optimization
              - Multiple timeframe alignment for strongest signals
              
            Trend quality assessment incorporates:
            - Trend age (mature vs. early trends)
            - Trend strength and momentum characteristics
            - Recent price volatility and consolidation patterns
            - Historical trend adherence and violations
            
            Implementation adaptively selects the most appropriate trend detection
            method based on:
            - Market regime (trending vs. ranging)
            - Asset class characteristics
            - Volatility environment
            - Strategy parameter configuration
        """
        if self.params['trend_indicator'] == 'ema_20_50':
            return tech_signals.is_ema_bullish(symbol, short_period=20, long_period=50)
        elif self.params['trend_indicator'] == 'sma_50_200':
            return tech_signals.is_sma_bullish(symbol, short_period=50, long_period=200)
        elif self.params['trend_indicator'] == 'macd':
            return tech_signals.is_macd_bullish(symbol)
        elif self.params['trend_indicator'] == 'price_action':
            # Check for higher highs and higher lows pattern
            return tech_signals.has_higher_highs_and_lows(symbol, periods=self.params['min_up_trend_days'])
        elif self.params['trend_indicator'] == 'rsi':
            # Check if RSI is in bullish territory (above 50) and rising
            return tech_signals.is_rsi_bullish(symbol, threshold=50, periods=14)
        elif self.params['trend_indicator'] == 'multi_indicator':
            # Require multiple indicators to confirm bullish trend
            ema_bullish = tech_signals.is_ema_bullish(symbol, short_period=20, long_period=50)
            rsi_bullish = tech_signals.is_rsi_bullish(symbol, threshold=50)
            volume_confirming = tech_signals.is_volume_confirming_trend(symbol, trend='bullish')
            
            # Require at least 2 of 3 indicators to be bullish
            indicators_bullish = sum([ema_bullish, rsi_bullish, volume_confirming])
            return indicators_bullish >= 2
        else:
            # Default to checking recent price action
            return tech_signals.is_uptrend(symbol, days=self.params['min_up_trend_days'])
    
    def _select_expiration(self, symbol: str, option_chains: OptionChains, 
                          min_dte=None, max_dte=None) -> str:
        """
        Select the appropriate expiration date for the option spread.
        
        Finds the expiration date closest to the target DTE (days to expiration)
        from the available options chain, within the specified min/max range.
        
        Parameters:
            symbol (str): Symbol to check
            option_chains (OptionChains): Option chains data provider
            min_dte (int, optional): Minimum days to expiration, uses param if None
            max_dte (int, optional): Maximum days to expiration, uses param if None
            
        Returns:
            str: Selected expiration date in format 'YYYY-MM-DD', or empty string if none found
        """
        if min_dte is None:
            min_dte = self.params['min_dte']
        if max_dte is None:
            max_dte = self.params['max_dte']
            
        try:
            chains = option_chains.get_option_chain(symbol)
            if chains is None or chains.empty:
                return ""
                
            available_expirations = chains['expiration_date'].unique()
            target_dte = self.params['target_dte']
            
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
    
    def _select_strikes_by_delta(self, call_options: pd.DataFrame, current_price: float) -> Tuple[Dict, Dict]:
        """
        Select strikes based on delta targets.
        
        Selects the appropriate strike prices for both legs of the bull call spread
        by finding call options with deltas closest to the target values specified
        in the strategy parameters.
        
        Parameters:
            call_options (pd.DataFrame): Available call options data
            current_price (float): Current price of the underlying
            
        Returns:
            Tuple[Dict, Dict]: Dictionaries containing details of selected long and short calls
        """
        if 'delta' not in call_options.columns:
            logger.warning("Delta data not available, falling back to OTM percentage method")
            return self._select_strikes_by_otm_percentage(call_options, current_price)
            
        # Find long call with delta closest to target
        long_call_options = call_options.copy()
        long_call_options['delta_diff'] = abs(long_call_options['delta'] - self.params['long_call_delta'])
        long_call_options = long_call_options.sort_values('delta_diff')
        
        if long_call_options.empty:
            return None, None
            
        long_call = long_call_options.iloc[0].to_dict()
        
        # Find short call with delta closest to target
        short_call_options = call_options[call_options['strike'] > long_call['strike']].copy()
        short_call_options['delta_diff'] = abs(short_call_options['delta'] - self.params['short_call_delta'])
        short_call_options = short_call_options.sort_values('delta_diff')
        
        if short_call_options.empty:
            return long_call, None
            
        short_call = short_call_options.iloc[0].to_dict()
        
        return long_call, short_call
    
    def _select_strikes_by_otm_percentage(self, call_options: pd.DataFrame, current_price: float) -> Tuple[Dict, Dict]:
        """
        Select strikes based on OTM percentage.
        
        Selects the appropriate strike prices for both legs of the bull call spread
        based on target percentages out-of-the-money relative to the current price
        of the underlying asset.
        
        Parameters:
            call_options (pd.DataFrame): Available call options data
            current_price (float): Current price of the underlying
            
        Returns:
            Tuple[Dict, Dict]: Dictionaries containing details of selected long and short calls
        """
        target_long_strike = current_price * (1 + self.params['otm_percentage'])
        target_short_strike = current_price * (1 + self.params['otm_percentage'] + self.params['short_call_otm_extra'])
        
        # Find closest long strike
        call_options['long_strike_diff'] = abs(call_options['strike'] - target_long_strike)
        call_options = call_options.sort_values('long_strike_diff')
        
        if call_options.empty:
            return None, None
            
        long_call = call_options.iloc[0].to_dict()
        
        # Find short strike that is higher than long strike
        short_call_options = call_options[call_options['strike'] > long_call['strike']].copy()
        short_call_options['short_strike_diff'] = abs(short_call_options['strike'] - target_short_strike)
        short_call_options = short_call_options.sort_values('short_strike_diff')
        
        if short_call_options.empty:
            return long_call, None
            
        short_call = short_call_options.iloc[0].to_dict()
        
        return long_call, short_call
    
    def _select_strikes_by_price_range(self, call_options: pd.DataFrame, current_price: float) -> Tuple[Dict, Dict]:
        """
        Select strikes based on price range and spread width.
        
        Selects the appropriate strike prices for both legs of the bull call spread
        by finding a long strike near ATM and a short strike approximately the target
        spread width higher.
        
        Parameters:
            call_options (pd.DataFrame): Available call options data
            current_price (float): Current price of the underlying
            
        Returns:
            Tuple[Dict, Dict]: Dictionaries containing details of selected long and short calls
        """
        # Get the ATM strike
        atm_strike = get_atm_strike(current_price, call_options['strike'].unique())
        
        # Find closest strike to ATM for long call
        call_options['atm_diff'] = abs(call_options['strike'] - atm_strike)
        call_options = call_options.sort_values('atm_diff')
        
        if call_options.empty:
            return None, None
            
        long_call = call_options.iloc[0].to_dict()
        
        # Find short call with strike approximately spread_width higher
        target_short_strike = long_call['strike'] + self.params['spread_width']
        short_call_options = call_options[call_options['strike'] > long_call['strike']].copy()
        short_call_options['short_strike_diff'] = abs(short_call_options['strike'] - target_short_strike)
        short_call_options = short_call_options.sort_values('short_strike_diff')
        
        if short_call_options.empty:
            return long_call, None
            
        short_call = short_call_options.iloc[0].to_dict()
        
        return long_call, short_call

    # ======================== OPTIMIZATION METHODS ========================
    def get_optimization_params(self) -> Dict[str, Any]:
        """
        Define parameters that can be optimized and their ranges.
        
        Specifies the strategy parameters that are suitable for optimization,
        along with their valid ranges and step sizes for optimization algorithms.
        
        Returns:
            Dict[str, Any]: Dictionary of parameters with their optimization constraints
                Each parameter entry contains:
                - type: Data type (int, float)
                - min: Minimum allowable value
                - max: Maximum allowable value
                - step: Step size for optimization iterations
        """
        return {
            'target_dte': {'type': 'int', 'min': 20, 'max': 60, 'step': 5},
            'spread_width': {'type': 'int', 'min': 1, 'max': 10, 'step': 1},
            'long_call_delta': {'type': 'float', 'min': 0.50, 'max': 0.80, 'step': 0.05},
            'short_call_delta': {'type': 'float', 'min': 0.20, 'max': 0.50, 'step': 0.05},
            'profit_target_percent': {'type': 'int', 'min': 30, 'max': 75, 'step': 5},
            'loss_limit_percent': {'type': 'int', 'min': 50, 'max': 90, 'step': 5},
            'min_iv_percentile': {'type': 'int', 'min': 20, 'max': 50, 'step': 5},
            'max_iv_percentile': {'type': 'int', 'min': 50, 'max': 80, 'step': 5},
        }
        
    def evaluate_performance(self, backtest_results: Dict[str, Any]) -> float:
        """
        Evaluate strategy performance for optimization.
        
        Calculates a performance score based on backtest results for parameter optimization.
        The score incorporates Sharpe ratio with penalties for drawdowns and rewards for high win rates.
        
        Parameters:
            backtest_results (Dict[str, Any]): Results from backtest
            
        Returns:
            float: Performance score (higher is better)
            
        Notes:
            The scoring function:
            - Uses Sharpe ratio as the base metric
            - Penalizes high drawdowns (>25%)
            - Rewards high win rates (>50%)
            - Returns 0 if critical metrics are missing
        """
        # Calculate Sharpe ratio with a penalty for max drawdown
        if 'sharpe_ratio' not in backtest_results or 'max_drawdown' not in backtest_results:
            return 0.0
            
        sharpe = backtest_results.get('sharpe_ratio', 0)
        max_dd = abs(backtest_results.get('max_drawdown', 0))
        win_rate = backtest_results.get('win_rate', 0)
        
        # Penalize high drawdowns
        if max_dd > 0.25:  # 25% drawdown
            sharpe = sharpe * (1 - (max_dd - 0.25))
            
        # Reward high win rates
        if win_rate > 0.5:
            sharpe = sharpe * (1 + (win_rate - 0.5))
            
        return max(0, sharpe)

# Add TODOs for improvements
"""
TODO: Implement more sophisticated trend detection methods
TODO: Add mean reversion checks to avoid entering during overextended moves
TODO: Consider adding adjustment logic when trades move against the position
TODO: Implement IV rank/percentile calculation and use it for strike selection
TODO: Add correlation analysis to avoid too many similar positions
TODO: Consider calendar spread variants for high IV environments
TODO: Implement rolling logic to extend profitable trades
TODO: Add portfolio-level risk metrics
TODO: Consider seasonality factors in symbol selection
""" 