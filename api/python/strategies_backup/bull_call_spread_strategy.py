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
        'stop_loss_pct': 0.50,                # Stop loss at % of max loss
        
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
        Define the universe of stocks to trade based on criteria.
        
        This method filters the available stocks based on:
        1. Price range set in parameters
        2. Option liquidity (volume and open interest)
        3. Technical criteria (bullish trend)
        
        Parameters:
            market_data (MarketData): Market data instance containing price data
            
        Returns:
            Universe: A Universe object containing the filtered symbols
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
            # Check if options meet volume and open interest criteria
            if not self._check_option_liquidity(symbol, option_chains):
                universe.remove_symbol(symbol)
                
        # Filter by technical criteria
        tech_signals = TechnicalSignals(market_data)
        symbols_to_remove = []
        
        for symbol in universe.get_symbols():
            if not self._has_bullish_trend(symbol, tech_signals):
                symbols_to_remove.append(symbol)
                
        for symbol in symbols_to_remove:
            universe.remove_symbol(symbol)
            
        logger.info(f"Bull Call Spread universe contains {len(universe.get_symbols())} symbols")
        return universe
    
    # ======================== 3. SELECTION CRITERIA ========================
    def check_selection_criteria(self, symbol: str, market_data: MarketData, 
                                option_chains: OptionChains) -> bool:
        """
        Check if the symbol meets the selection criteria for the strategy.
        
        Evaluates a symbol against multiple criteria:
        - Sufficient historical data
        - IV percentile within desired range
        - Available option chains with suitable expirations
        - Bullish trend confirmation
        
        Parameters:
            symbol (str): Symbol to check
            market_data (MarketData): Market data instance
            option_chains (OptionChains): Option chains instance
            
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
            
        # Check if the stock is in an uptrend
        if not self._has_bullish_trend(symbol, TechnicalSignals(market_data)):
            logger.debug(f"{symbol} does not have a bullish trend")
            return False
            
        logger.info(f"{symbol} meets all selection criteria for bull call spread")
        return True
    
    # ======================== 4. OPTION SELECTION ========================
    def select_option_contract(self, symbol: str, market_data: MarketData,
                              option_chains: OptionChains) -> Dict[str, Any]:
        """
        Select the appropriate option contracts for the bull call spread.
        
        Identifies the optimal call options for both legs of the spread based on:
        1. Finding appropriate expiration date
        2. Selecting strike prices based on parameters (delta, OTM percentage, or price range)
        3. Calculating the spread's risk/reward characteristics
        
        Parameters:
            symbol (str): The stock symbol
            market_data (MarketData): Market data instance
            option_chains (OptionChains): Option chains instance
            
        Returns:
            Dict[str, Any]: Dictionary containing selected option contracts and trade details,
                           or empty dict if no suitable contracts found
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
            long_call, short_call = self._select_strikes_by_delta(call_options, current_price)
        elif self.params['strike_selection_method'] == 'otm_percentage':
            long_call, short_call = self._select_strikes_by_otm_percentage(call_options, current_price)
        else:  # Default to price_range
            long_call, short_call = self._select_strikes_by_price_range(call_options, current_price)
            
        if not long_call or not short_call:
            logger.error(f"Could not select appropriate strikes for {symbol}")
            return {}
            
        # Calculate the debit and max profit potential
        debit = long_call['ask'] - short_call['bid']
        max_profit = (short_call['strike'] - long_call['strike']) - debit
        max_loss = debit
        
        if debit <= 0:
            logger.warning(f"Invalid debit spread for {symbol}, debit: {debit}")
            return {}
            
        # Return the selected options and trade details
        return {
            'symbol': symbol,
            'strategy': 'bull_call_spread',
            'expiration': target_expiration,
            'dte': (datetime.strptime(target_expiration, '%Y-%m-%d').date() - date.today()).days,
            'long_call': long_call,
            'short_call': short_call,
            'long_call_contract': f"{symbol}_{target_expiration}_{long_call['strike']}_C",
            'short_call_contract': f"{symbol}_{target_expiration}_{short_call['strike']}_C",
            'debit': debit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': long_call['strike'] + debit,
            'risk_reward_ratio': max_profit / max_loss if max_loss > 0 else 0,
            'price': current_price,
            'timestamp': datetime.now().isoformat()
        }
    
    # ======================== 5. POSITION SIZING ========================
    def calculate_position_size(self, trade_details: Dict[str, Any], 
                               position_sizer: PositionSizer) -> int:
        """
        Calculate the number of spreads to trade based on risk parameters.
        
        Determines the optimal position size by:
        1. Calculating max risk per spread
        2. Determining max risk amount based on portfolio size
        3. Calculating number of spreads within risk limits
        4. Ensuring position size is within overall portfolio allocation limits
        
        Parameters:
            trade_details (Dict[str, Any]): Details of the selected option spread
            position_sizer (PositionSizer): Position sizer instance for portfolio info
            
        Returns:
            int: Number of spreads to trade (0 if invalid risk calculation)
        """
        # Calculate max risk per spread
        max_risk_per_spread = trade_details['max_loss'] * 100  # Convert to dollars (per contract)
        
        # Get portfolio value
        portfolio_value = position_sizer.get_portfolio_value()
        
        # Calculate max risk for this trade based on portfolio percentage
        max_risk_dollars = portfolio_value * self.params['max_risk_per_trade']
        
        # Calculate number of spreads
        if max_risk_per_spread <= 0:
            return 0
            
        num_spreads = int(max_risk_dollars / max_risk_per_spread)
        
        # Check against max position size
        max_position_dollars = portfolio_value * self.params['max_position_size_percent']
        position_cost = trade_details['debit'] * 100 * num_spreads
        
        if position_cost > max_position_dollars:
            num_spreads = int(max_position_dollars / (trade_details['debit'] * 100))
            
        # Ensure at least 1 spread if we're trading
        num_spreads = max(1, num_spreads)
        
        logger.info(f"Bull Call Spread position size for {trade_details['symbol']}: {num_spreads} spreads")
        return num_spreads
    
    # ======================== 6. ENTRY EXECUTION ========================
    def prepare_entry_orders(self, trade_details: Dict[str, Any], 
                            num_spreads: int) -> List[Order]:
        """
        Prepare orders for executing the bull call spread.
        
        Creates the necessary orders for both legs of the spread:
        1. Long call order at lower strike
        2. Short call order at higher strike
        
        Parameters:
            trade_details (Dict[str, Any]): Details of the selected spread
            num_spreads (int): Number of spreads to trade
            
        Returns:
            List[Order]: List of orders to execute, empty if num_spreads <= 0
        """
        if num_spreads <= 0:
            return []
            
        symbol = trade_details['symbol']
        orders = []
        
        # Create long call order
        long_call_order = Order(
            symbol=symbol,
            option_symbol=trade_details['long_call_contract'],
            order_type=OrderType.LIMIT,
            action=OrderAction.BUY,
            quantity=num_spreads,
            limit_price=trade_details['long_call']['ask'],
            trade_id=f"bull_call_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            order_details={
                'strategy': 'bull_call_spread',
                'leg': 'long_call',
                'expiration': trade_details['expiration'],
                'strike': trade_details['long_call']['strike'],
                'trade_details': trade_details
            }
        )
        orders.append(long_call_order)
        
        # Create short call order
        short_call_order = Order(
            symbol=symbol,
            option_symbol=trade_details['short_call_contract'],
            order_type=OrderType.LIMIT,
            action=OrderAction.SELL,
            quantity=num_spreads,
            limit_price=trade_details['short_call']['bid'],
            trade_id=f"bull_call_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            order_details={
                'strategy': 'bull_call_spread',
                'leg': 'short_call',
                'expiration': trade_details['expiration'],
                'strike': trade_details['short_call']['strike'],
                'trade_details': trade_details
            }
        )
        orders.append(short_call_order)
        
        logger.info(f"Created bull call spread orders for {symbol}: {num_spreads} spreads")
        return orders
    
    # ======================== 7. EXIT CONDITIONS ========================
    def check_exit_conditions(self, position: Dict[str, Any], 
                             market_data: MarketData) -> bool:
        """
        Check if exit conditions are met for an existing position.
        
        Evaluates position against multiple exit criteria:
        1. Days to expiration threshold
        2. Profit target reached
        3. Loss limit reached
        4. Trend reversal (if enabled)
        
        Parameters:
            position (Dict[str, Any]): The current position data
            market_data (MarketData): Market data instance
            
        Returns:
            bool: True if exit conditions are met, False otherwise
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
            logger.info(f"Exiting {symbol} bull call spread: DTE {current_dte} <= threshold {self.params['dte_exit_threshold']}")
            return True
            
        # Check for profit target
        current_value = position.get('current_value', 0)
        entry_value = position.get('entry_value', 0)
        
        if entry_value > 0:
            profit_pct = (current_value - entry_value) / abs(entry_value) * 100
            max_profit_pct = (trade_details.get('max_profit', 0) / trade_details.get('debit', 1)) * 100
            
            # If we've reached our target profit percentage
            if profit_pct >= (max_profit_pct * (self.params['profit_target_percent'] / 100)):
                logger.info(f"Exiting {symbol} bull call spread: Profit target reached {profit_pct:.2f}%")
                return True
                
        # Check for loss limit
        if entry_value > 0:
            loss_pct = (entry_value - current_value) / abs(entry_value) * 100
            
            if loss_pct >= self.params['loss_limit_percent']:
                logger.info(f"Exiting {symbol} bull call spread: Loss limit reached {loss_pct:.2f}%")
                return True
                
        # Check if trend has reversed
        if self.params['ema_cross_exit']:
            if not self._has_bullish_trend(symbol, TechnicalSignals(market_data)):
                logger.info(f"Exiting {symbol} bull call spread: Bearish trend detected")
                return True
                
        return False
    
    # ======================== 8. EXIT EXECUTION ========================
    def prepare_exit_orders(self, position: Dict[str, Any]) -> List[Order]:
        """
        Prepare orders to close an existing position.
        
        Creates the necessary orders to close both legs of the spread:
        1. Sell order for the long call
        2. Buy order for the short call
        
        Parameters:
            position (Dict[str, Any]): The position to close
            
        Returns:
            List[Order]: List of orders to execute
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
            close_action = OrderAction.SELL if leg.get('action') == OrderAction.BUY else OrderAction.BUY
            
            close_order = Order(
                symbol=leg.get('symbol', ''),
                option_symbol=leg.get('option_symbol', ''),
                order_type=OrderType.MARKET,
                action=close_action,
                quantity=leg.get('quantity', 0),
                trade_id=f"close_{leg.get('trade_id', '')}",
                order_details={
                    'strategy': 'bull_call_spread',
                    'leg': 'exit_' + leg.get('order_details', {}).get('leg', ''),
                    'closing_order': True,
                    'original_order_id': leg.get('order_id', '')
                }
            )
            orders.append(close_order)
            
        logger.info(f"Created exit orders for bull call spread position")
        return orders
    
    # ======================== 9. HEDGING ========================
    def prepare_hedge_orders(self, positions: List[Dict[str, Any]], 
                            market_data: MarketData,
                            option_chains: OptionChains,
                            position_sizer: PositionSizer) -> List[Order]:
        """
        Prepare hedge orders for portfolio protection.
        
        Creates protective positions to reduce overall portfolio risk:
        1. Evaluates current portfolio exposure 
        2. Decides on hedge approach based on parameters
        3. Typically uses SPY puts for broad market protection
        
        Parameters:
            positions (List[Dict[str, Any]]): Current positions
            market_data (MarketData): Market data instance
            option_chains (OptionChains): Option chains instance
            position_sizer (PositionSizer): Position sizer instance
            
        Returns:
            List[Order]: List of hedge orders to execute
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
        
        Verifies that options have sufficient volume and open interest
        according to the parameters.
        
        Parameters:
            symbol (str): Symbol to check
            option_chains (OptionChains): Option chains data
            
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
        Check if a symbol is in a bullish trend.
        
        Uses the configured trend indicator to determine if the symbol
        is showing bullish characteristics.
        
        Parameters:
            symbol (str): Symbol to check
            tech_signals (TechnicalSignals): Technical signals instance
            
        Returns:
            bool: True if symbol is in a bullish trend, False otherwise
        """
        if self.params['trend_indicator'] == 'ema_20_50':
            return tech_signals.is_ema_bullish(symbol, short_period=20, long_period=50)
        elif self.params['trend_indicator'] == 'sma_50_200':
            return tech_signals.is_sma_bullish(symbol, short_period=50, long_period=200)
        else:
            # Default to checking recent price action
            return tech_signals.is_uptrend(symbol, days=self.params['min_up_trend_days'])
    
    def _select_expiration(self, symbol: str, option_chains: OptionChains, 
                          min_dte=None, max_dte=None) -> str:
        """
        Select the appropriate expiration date.
        
        Finds the expiration date closest to the target DTE within
        the specified min/max range.
        
        Parameters:
            symbol (str): Symbol to get expiration for
            option_chains (OptionChains): Option chains data
            min_dte (int, optional): Minimum days to expiration
            max_dte (int, optional): Maximum days to expiration
            
        Returns:
            str: Selected expiration date in 'YYYY-MM-DD' format, empty if none found
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
        
        Finds call option strikes with deltas closest to the target values.
        
        Parameters:
            call_options (pd.DataFrame): DataFrame of call options
            current_price (float): Current price of the underlying
            
        Returns:
            Tuple[Dict, Dict]: Tuple of (long call data, short call data),
                              returns (None, None) if suitable options not found
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
        
        Finds call option strikes at specified percentages out-of-the-money.
        
        Parameters:
            call_options (pd.DataFrame): DataFrame of call options
            current_price (float): Current price of the underlying
            
        Returns:
            Tuple[Dict, Dict]: Tuple of (long call data, short call data),
                              returns (None, None) if suitable options not found
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
        
        Finds a strike close to at-the-money for the long call, and
        another strike approximately spread_width higher for the short call.
        
        Parameters:
            call_options (pd.DataFrame): DataFrame of call options
            current_price (float): Current price of the underlying
            
        Returns:
            Tuple[Dict, Dict]: Tuple of (long call data, short call data),
                              returns (None, None) if suitable options not found
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
        
        Specifies which parameters can be optimized during backtesting,
        along with their types, allowed ranges, and step sizes.
        
        Returns:
            Dict[str, Any]: Dictionary of parameter specifications for optimization
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