"""
Straddle Options Strategy

A straddle is an options strategy that involves buying both a call and put option
with the same strike price and expiration date. This strategy is used when a trader
believes the underlying asset will experience significant volatility but is unsure
about the direction.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from trading_bot.strategies.strategy_base import StrategyOptimizable
from trading_bot.config.straddle_config import DEFAULT_CONFIG
from trading_bot.utils.straddle_utils import (
    calculate_straddle_cost,
    calculate_breakeven_points,
    calculate_max_loss,
    calculate_max_profit,
    calculate_probability_of_profit,
    calculate_iv_rank,
    score_straddle_opportunity,
    get_optimal_dte_for_straddle,
    estimate_position_size
)
from trading_bot.utils.option_utils import (
    get_option_chain,
    calculate_implied_volatility,
    get_historical_volatility,
    get_nearest_expiration_date,
    get_atm_options
)
from trading_bot.utils.market_utils import (
    get_current_price,
    get_iv_history
)

logger = logging.getLogger(__name__)

class StraddleStrategy(StrategyOptimizable):
    """
    Options straddle strategy implementation.
    
    The strategy looks for high implied volatility rank situations where
    the underlying security is expected to move significantly, but the
    direction is uncertain.
    """
    
    def __init__(self, config=None):
        """
        Initialize the straddle strategy with configuration parameters.
        
        Args:
            config (dict, optional): Configuration dictionary to override defaults
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        super().__init__(config=self.config)
        
        # Initialize strategy-specific attributes
        self.watchlist = self.config['watchlist']
        self.entry_criteria = self.config['entry_criteria']
        self.strike_selection = self.config['strike_selection']
        self.risk_management = self.config['risk_management']
        self.positions = []
        
        # Weight factors for scoring opportunities
        self.weight_factors = {
            'iv_rank': 0.25,
            'iv_vs_historical_ratio': 0.15,
            'bid_ask_spread': 0.15,
            'option_volume': 0.15,
            'days_to_expiration': 0.10,
            'probability_of_profit': 0.20
        }
        
        logger.info("Straddle strategy initialized with %d securities in watchlist", 
                   len(self.watchlist))
    
    def scan_for_setups(self):
        """
        Scan the watchlist for potential straddle setup opportunities.
        
        Returns:
            list: List of dictionaries containing setup details
        """
        setups = []
        
        for ticker in self.watchlist:
            try:
                # Get current price and volatility metrics
                current_price = get_current_price(ticker)
                
                # Get historical volatility data
                historical_data = get_historical_volatility(ticker, lookback_days=252)
                
                # Get implied volatility history
                iv_history = get_iv_history(ticker, lookback_days=252)
                current_iv = iv_history[-1] if iv_history else None
                
                if current_iv is None:
                    logger.warning(f"No IV data available for {ticker}, skipping")
                    continue
                    
                # Calculate IV rank
                historical_iv_low = min(iv_history) if iv_history else current_iv
                historical_iv_high = max(iv_history) if iv_history else current_iv
                iv_rank = calculate_iv_rank(current_iv, historical_iv_low, historical_iv_high)
                
                # Check if IV rank meets minimum threshold
                if iv_rank < self.entry_criteria['min_iv_rank']:
                    logger.debug(f"{ticker} IV rank {iv_rank:.2f} below minimum {self.entry_criteria['min_iv_rank']}")
                    continue
                
                # Get historical volatility average
                hist_vol_avg = np.mean(historical_data) if len(historical_data) > 0 else current_iv
                iv_hist_ratio = current_iv / hist_vol_avg if hist_vol_avg > 0 else 1.0
                
                # Get option chain
                expiration_date = get_optimal_expiration_date(
                    ticker, 
                    self.strike_selection['min_days_to_expiration'],
                    self.strike_selection['max_days_to_expiration'],
                    self.strike_selection['target_days_to_expiration']
                )
                
                if not expiration_date:
                    logger.warning(f"No suitable expiration date found for {ticker}")
                    continue
                
                days_to_expiration = (expiration_date - datetime.now()).days
                
                # Get ATM options (call and put)
                atm_options = get_atm_options(ticker, expiration_date)
                
                if not atm_options or 'call' not in atm_options or 'put' not in atm_options:
                    logger.warning(f"Unable to get ATM options for {ticker}")
                    continue
                
                call_option = atm_options['call']
                put_option = atm_options['put']
                
                # Check option volume
                option_volume = min(call_option['volume'], put_option['volume'])
                if option_volume < self.entry_criteria['min_option_volume']:
                    logger.debug(f"{ticker} option volume {option_volume} below minimum {self.entry_criteria['min_option_volume']}")
                    continue
                
                # Calculate bid-ask spread percentage
                call_spread = (call_option['ask'] - call_option['bid']) / call_option['mid'] * 100
                put_spread = (put_option['ask'] - put_option['bid']) / put_option['mid'] * 100
                avg_spread = (call_spread + put_spread) / 2
                
                if avg_spread > self.entry_criteria['max_bid_ask_spread']:
                    logger.debug(f"{ticker} bid-ask spread {avg_spread:.2f}% above maximum {self.entry_criteria['max_bid_ask_spread']}%")
                    continue
                
                # Calculate straddle cost
                straddle_cost = calculate_straddle_cost(
                    call_option['ask'], 
                    put_option['ask']
                ) / 100  # Convert to per-share cost
                
                # Calculate probability of profit
                pop = calculate_probability_of_profit(
                    current_price,
                    call_option['strike'],
                    days_to_expiration,
                    current_iv,
                    straddle_cost
                )
                
                if pop < self.entry_criteria['min_probability_of_profit']:
                    logger.debug(f"{ticker} probability of profit {pop:.2f}% below minimum {self.entry_criteria['min_probability_of_profit']}%")
                    continue
                
                # Score the opportunity
                score = score_straddle_opportunity(
                    iv_rank,
                    iv_hist_ratio,
                    avg_spread,
                    option_volume,
                    days_to_expiration,
                    pop,
                    self.weight_factors
                )
                
                if score < self.entry_criteria['min_score']:
                    logger.debug(f"{ticker} score {score:.2f} below minimum {self.entry_criteria['min_score']}")
                    continue
                
                # Calculate breakeven points
                lower_breakeven, upper_breakeven = calculate_breakeven_points(
                    call_option['strike'], 
                    straddle_cost
                )
                
                # Calculate max loss and estimated max profit
                max_loss = calculate_max_loss(straddle_cost * 100)  # Per contract
                max_profit = calculate_max_profit(
                    straddle_cost, 
                    call_option, 
                    put_option, 
                    current_price
                )
                
                # Create setup details
                setup = {
                    'ticker': ticker,
                    'timestamp': datetime.now(),
                    'price': current_price,
                    'iv_rank': iv_rank,
                    'iv_historical_ratio': iv_hist_ratio,
                    'expiration_date': expiration_date,
                    'days_to_expiration': days_to_expiration,
                    'strike': call_option['strike'],
                    'call_option': call_option,
                    'put_option': put_option,
                    'straddle_cost': straddle_cost * 100,  # Per contract
                    'breakeven_lower': lower_breakeven,
                    'breakeven_upper': upper_breakeven,
                    'bid_ask_spread': avg_spread,
                    'option_volume': option_volume,
                    'probability_of_profit': pop,
                    'max_loss': max_loss,
                    'max_profit': max_profit,
                    'score': score
                }
                
                setups.append(setup)
                logger.info(f"Found straddle setup for {ticker} with score {score:.2f}")
                
            except Exception as e:
                logger.error(f"Error scanning {ticker} for straddle setups: {e}")
        
        # Sort setups by score (descending)
        setups.sort(key=lambda x: x['score'], reverse=True)
        return setups
    
    def get_optimal_expiration_date(self, ticker, min_days, max_days, target_days):
        """
        Get the optimal expiration date for a straddle based on the configuration.
        
        Args:
            ticker (str): Ticker symbol
            min_days (int): Minimum days to expiration
            max_days (int): Maximum days to expiration
            target_days (int): Target days to expiration
            
        Returns:
            datetime: The optimal expiration date
        """
        try:
            # Get available expiration dates
            expiration_dates = get_nearest_expiration_date(
                ticker, 
                min_days=min_days,
                max_days=max_days
            )
            
            if not expiration_dates:
                return None
            
            # Get volatility term structure
            volatility_term_structure = {}
            for exp_date in expiration_dates:
                days = (exp_date - datetime.now()).days
                # Get ATM IV for this expiration
                atm_options = get_atm_options(ticker, exp_date)
                if atm_options and 'call' in atm_options:
                    volatility_term_structure[days] = atm_options['call']['iv']
            
            # Get optimal DTE based on volatility term structure
            optimal_dte = get_optimal_dte_for_straddle(
                volatility_term_structure, 
                target_days
            )
            
            # Find the expiration date closest to the optimal DTE
            optimal_date = min(
                expiration_dates,
                key=lambda date: abs((date - datetime.now()).days - optimal_dte)
            )
            
            return optimal_date
            
        except Exception as e:
            logger.error(f"Error getting optimal expiration date for {ticker}: {e}")
            return None
    
    def rank_opportunities(self, setups):
        """
        Rank the setup opportunities by score.
        
        Args:
            setups (list): List of setup dictionaries
            
        Returns:
            list: Ranked list of setups
        """
        # Already sorted in scan_for_setups, but we can add additional criteria here
        return setups
    
    def calculate_position_size(self, setup, account_value):
        """
        Calculate the optimal position size for a straddle setup.
        
        Args:
            setup (dict): Setup details
            account_value (float): Current account value
            
        Returns:
            int: Number of contracts to trade
        """
        max_position_size_pct = self.risk_management['max_position_size'] / 100
        max_loss_pct = self.risk_management['max_loss_percent'] / 100
        
        contracts = estimate_position_size(
            account_value,
            max_position_size_pct,
            setup['straddle_cost'],
            max_loss_pct * 100  # Convert to percentage
        )
        
        return contracts
    
    def check_portfolio_allocation(self, new_position_cost, account_value):
        """
        Check if adding a new position would exceed portfolio allocation limits.
        
        Args:
            new_position_cost (float): Cost of the new position
            account_value (float): Current account value
            
        Returns:
            bool: True if allocation is within limits, False otherwise
        """
        # Calculate current allocation percentage
        current_allocation = sum(pos['cost'] for pos in self.positions) / account_value * 100
        
        # Calculate new allocation percentage if position is added
        new_allocation = current_allocation + (new_position_cost / account_value * 100)
        
        # Check if new allocation would exceed maximum
        return new_allocation <= self.risk_management['max_portfolio_allocation']
    
    def execute_entry(self, setup, account_value):
        """
        Execute entry into a straddle position.
        
        Args:
            setup (dict): Setup details
            account_value (float): Current account value
            
        Returns:
            dict: Position details if entry successful, None otherwise
        """
        try:
            # Calculate position size
            contracts = self.calculate_position_size(setup, account_value)
            
            if contracts <= 0:
                logger.warning(f"Position size calculation resulted in 0 contracts for {setup['ticker']}")
                return None
            
            # Calculate total cost of position
            position_cost = setup['straddle_cost'] * contracts
            
            # Check if adding this position would exceed portfolio allocation
            if not self.check_portfolio_allocation(position_cost, account_value):
                logger.warning(f"Adding position for {setup['ticker']} would exceed maximum portfolio allocation")
                return None
            
            # Create position details
            position = {
                'ticker': setup['ticker'],
                'strategy_type': 'straddle',
                'entry_date': datetime.now(),
                'expiration_date': setup['expiration_date'],
                'days_to_expiration': setup['days_to_expiration'],
                'strike': setup['strike'],
                'contracts': contracts,
                'call_option': setup['call_option'],
                'put_option': setup['put_option'],
                'entry_price': setup['price'],
                'cost': position_cost,
                'max_loss': setup['max_loss'] * contracts,
                'stop_loss': None,  # Straddles typically don't use stop losses
                'target_profit': position_cost * (self.risk_management['target_profit_percent'] / 100),
                'status': 'active',
                'pnl': 0
            }
            
            # Add position to positions list
            self.positions.append(position)
            
            logger.info(f"Entered straddle position for {setup['ticker']}: {contracts} contracts at strike {setup['strike']}")
            
            return position
            
        except Exception as e:
            logger.error(f"Error executing entry for {setup['ticker']}: {e}")
            return None
    
    def check_exit_conditions(self, position, current_data):
        """
        Check if any exit conditions are met for a position.
        
        Args:
            position (dict): Position details
            current_data (dict): Current market data
            
        Returns:
            tuple: (should_exit, reason)
        """
        ticker = position['ticker']
        current_price = current_data.get('price', get_current_price(ticker))
        
        # Get current option prices
        call_price = current_data.get('call_price')
        put_price = current_data.get('put_price')
        
        if call_price is None or put_price is None:
            # Fetch current option prices if not provided
            option_chain = get_option_chain(
                ticker, 
                position['expiration_date']
            )
            
            for option in option_chain.get('calls', []):
                if option['strike'] == position['strike']:
                    call_price = option['bid']  # Use bid for selling
                    break
            
            for option in option_chain.get('puts', []):
                if option['strike'] == position['strike']:
                    put_price = option['bid']  # Use bid for selling
                    break
        
        if call_price is None or put_price is None:
            logger.warning(f"Could not get current option prices for {ticker} straddle")
            return False, None
        
        # Calculate current position value
        current_value = (call_price + put_price) * position['contracts'] * 100
        
        # Calculate profit/loss
        pnl = current_value - position['cost']
        pnl_percent = (pnl / position['cost']) * 100
        
        # Update position PnL
        position['pnl'] = pnl
        
        # Check profit target
        if pnl >= position['target_profit']:
            return True, f"Profit target reached: {pnl_percent:.2f}%"
        
        # Check days to expiration exit threshold
        current_date = datetime.now()
        days_remaining = (position['expiration_date'] - current_date).days
        
        if days_remaining <= self.risk_management['exit_dte']:
            return True, f"Exit DTE threshold reached: {days_remaining} days remaining"
        
        # Check max loss
        if pnl <= -position['max_loss'] * (self.risk_management['max_loss_percent'] / 100):
            return True, f"Maximum loss reached: {pnl_percent:.2f}%"
        
        return False, None
    
    def execute_exit(self, position, exit_reason, current_data=None):
        """
        Execute exit from a straddle position.
        
        Args:
            position (dict): Position details
            exit_reason (str): Reason for exiting the position
            current_data (dict, optional): Current market data
            
        Returns:
            dict: Updated position details
        """
        try:
            ticker = position['ticker']
            
            # Get current option prices if not provided
            if current_data is None or 'call_price' not in current_data or 'put_price' not in current_data:
                current_data = {}
                
                # Fetch current option prices
                option_chain = get_option_chain(
                    ticker, 
                    position['expiration_date']
                )
                
                for option in option_chain.get('calls', []):
                    if option['strike'] == position['strike']:
                        current_data['call_price'] = option['bid']  # Use bid for selling
                        break
                
                for option in option_chain.get('puts', []):
                    if option['strike'] == position['strike']:
                        current_data['put_price'] = option['bid']  # Use bid for selling
                        break
            
            # Calculate exit value
            exit_value = (current_data.get('call_price', 0) + current_data.get('put_price', 0)) * position['contracts'] * 100
            
            # Calculate final P&L
            pnl = exit_value - position['cost']
            pnl_percent = (pnl / position['cost']) * 100
            
            # Update position
            position['exit_date'] = datetime.now()
            position['exit_price'] = current_data.get('price', get_current_price(ticker))
            position['pnl'] = pnl
            position['pnl_percent'] = pnl_percent
            position['exit_reason'] = exit_reason
            position['status'] = 'closed'
            
            logger.info(f"Exited straddle position for {ticker}: {pnl_percent:.2f}% profit/loss")
            
            return position
            
        except Exception as e:
            logger.error(f"Error executing exit for {position['ticker']}: {e}")
            return position
    
    def run(self, account_value):
        """
        Run the strategy to scan for setups and manage positions.
        
        Args:
            account_value (float): Current account value
            
        Returns:
            dict: Results containing new setups and position updates
        """
        results = {
            'new_setups': [],
            'position_updates': [],
            'new_positions': [],
            'closed_positions': []
        }
        
        try:
            # Scan for new setups
            setups = self.scan_for_setups()
            ranked_setups = self.rank_opportunities(setups)
            results['new_setups'] = ranked_setups
            
            # Check exit conditions for existing positions
            positions_to_remove = []
            
            for position in self.positions:
                if position['status'] == 'closed':
                    continue
                
                # Get current market data
                ticker = position['ticker']
                current_data = {
                    'price': get_current_price(ticker)
                }
                
                # Check if position should be exited
                should_exit, reason = self.check_exit_conditions(position, current_data)
                
                if should_exit:
                    updated_position = self.execute_exit(position, reason, current_data)
                    results['closed_positions'].append(updated_position)
                else:
                    results['position_updates'].append(position)
            
            # Enter new positions if opportunities exist
            for setup in ranked_setups:
                # Check if we're already at max portfolio allocation
                current_allocation = sum(pos['cost'] for pos in self.positions if pos['status'] == 'active') / account_value * 100
                if current_allocation >= self.risk_management['max_portfolio_allocation']:
                    logger.info("Maximum portfolio allocation reached, not entering new positions")
                    break
                
                position = self.execute_entry(setup, account_value)
                if position:
                    results['new_positions'].append(position)
            
            return results
            
        except Exception as e:
            logger.error(f"Error running straddle strategy: {e}")
            return results
    
    def optimize(self, historical_data, optimization_params=None):
        """
        Optimize strategy parameters based on historical data.
        
        Args:
            historical_data (dict): Historical market and options data
            optimization_params (dict, optional): Parameters to optimize
            
        Returns:
            dict: Optimized parameters
        """
        # Implementation of optimization logic
        # This would typically use backtesting to find optimal parameter values
        # For now, return the current parameters
        return self.config
    
    def get_strategy_metrics(self):
        """
        Calculate strategy performance metrics.
        
        Returns:
            dict: Performance metrics
        """
        if not self.positions:
            return {
                'win_rate': 0,
                'average_profit': 0,
                'average_loss': 0,
                'profit_factor': 0,
                'total_pnl': 0
            }
        
        closed_positions = [p for p in self.positions if p['status'] == 'closed']
        
        if not closed_positions:
            return {
                'win_rate': 0,
                'average_profit': 0,
                'average_loss': 0,
                'profit_factor': 0,
                'total_pnl': 0
            }
        
        winning_trades = [p for p in closed_positions if p['pnl'] > 0]
        losing_trades = [p for p in closed_positions if p['pnl'] <= 0]
        
        total_profit = sum(p['pnl'] for p in winning_trades) if winning_trades else 0
        total_loss = abs(sum(p['pnl'] for p in losing_trades)) if losing_trades else 0
        total_pnl = total_profit - total_loss
        
        win_rate = len(winning_trades) / len(closed_positions) * 100 if closed_positions else 0
        average_profit = total_profit / len(winning_trades) if winning_trades else 0
        average_loss = total_loss / len(losing_trades) if losing_trades else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
        
        return {
            'win_rate': win_rate,
            'average_profit': average_profit,
            'average_loss': average_loss,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl
        } 