"""
Straddle Options Strategy Implementation

This module implements a straddle options strategy, which involves buying both a call and put option 
at the same strike price and expiration date. This strategy profits from significant price movements 
in either direction, making it suitable for volatile markets or before major events.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Tuple, Optional, Union

from trading_bot.strategies.strategy_template import StrategyOptimizable
from trading_bot.utils.straddle_utils import (
    calculate_straddle_cost, 
    calculate_straddle_breakeven_points,
    calculate_max_loss,
    calculate_expected_move,
    calculate_probability_of_profit,
    calculate_risk_reward_ratio,
    score_straddle_opportunity,
    find_optimal_strike
)
from trading_bot.utils.options_utils import (
    get_option_chain,
    filter_option_chain,
    calculate_iv_rank
)
from trading_bot.data.market_data import MarketData
from trading_bot.execution.order_manager import OrderManager
from trading_bot.execution.position_manager import PositionManager
from trading_bot.risk.risk_manager import RiskManager
from trading_bot.config.straddle_config import STRADDLE_CONFIG

logger = logging.getLogger(__name__)

class StraddleStrategy(StrategyOptimizable):
    """
    Options Straddle Strategy Implementation
    
    A straddle involves buying both a call and put option at the same strike price and expiration date.
    This strategy profits when there is a significant price movement in either direction.
    """
    
    def __init__(self, 
                 market_data: MarketData,
                 order_manager: OrderManager,
                 position_manager: PositionManager,
                 risk_manager: RiskManager,
                 config: Dict = None):
        """
        Initialize the straddle strategy.
        
        Args:
            market_data: Market data provider
            order_manager: Order execution manager
            position_manager: Position tracking manager
            risk_manager: Risk management service
            config: Strategy configuration parameters (optional)
        """
        super().__init__(market_data, order_manager, position_manager, risk_manager)
        
        # Load configuration
        self.config = config or STRADDLE_CONFIG
        
        # Strategy state
        self.watchlist = self.config.get('watchlist', [])
        self.opportunities = {}
        self.active_positions = {}
        self.historical_performance = pd.DataFrame()
        self.is_optimization_enabled = self.config.get('optimization', {}).get('enabled', False)
        
        logger.info(f"Straddle strategy initialized with {len(self.watchlist)} tickers in watchlist")
        
    def scan_for_opportunities(self) -> Dict[str, Dict]:
        """
        Scan the watchlist for straddle opportunities.
        
        Returns:
            Dictionary of opportunities with ticker as key
        """
        opportunities = {}
        
        for ticker in self.watchlist:
            try:
                # Get current stock data
                stock_data = self.market_data.get_latest_price(ticker)
                stock_price = stock_data.get('close', stock_data.get('last', 0))
                
                if stock_price <= 0:
                    logger.warning(f"Invalid stock price for {ticker}: {stock_price}")
                    continue
                
                # Get historical volatility data
                historical_vol = self.market_data.get_historical_volatility(ticker)
                
                # Get option chain data
                option_chains = self.get_filtered_option_chains(ticker)
                
                if not option_chains:
                    logger.warning(f"No suitable option chains found for {ticker}")
                    continue
                    
                # Find the best expiration and strike for straddle
                best_expiry, best_strike, score, details = self.find_best_straddle(
                    ticker, stock_price, historical_vol, option_chains
                )
                
                if best_expiry and best_strike and score > 0:
                    opportunities[ticker] = {
                        'ticker': ticker,
                        'stock_price': stock_price,
                        'expiration': best_expiry,
                        'strike': best_strike,
                        'score': score,
                        'details': details
                    }
                    
                    logger.info(f"Found straddle opportunity for {ticker}: "
                                f"Strike=${best_strike}, Expiry={best_expiry}, Score={score:.2f}")
                
            except Exception as e:
                logger.error(f"Error scanning {ticker} for straddle opportunities: {str(e)}")
        
        self.opportunities = opportunities
        return opportunities
                
    def get_filtered_option_chains(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """
        Get and filter option chains for the given ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Dictionary of filtered option chains by expiration date
        """
        # Get raw option chain data
        raw_chains = self.market_data.get_option_chain(ticker)
        
        if not raw_chains:
            logger.warning(f"No option chain data available for {ticker}")
            return {}
            
        # Get filter criteria from config
        min_dte = self.config.get('entry_criteria', {}).get('min_days_to_expiration', 20)
        max_dte = self.config.get('entry_criteria', {}).get('max_days_to_expiration', 60)
        min_volume = self.config.get('entry_criteria', {}).get('min_option_volume', 10)
        max_spread_pct = self.config.get('entry_criteria', {}).get('max_bid_ask_spread', 0.10)
        
        # Filter chains by date and other criteria
        filtered_chains = {}
        
        for expiry, chain in raw_chains.items():
            # Calculate days to expiration
            try:
                exp_date = datetime.strptime(expiry, '%Y-%m-%d')
                days_to_exp = (exp_date - datetime.now()).days
                
                # Skip if outside desired DTE range
                if days_to_exp < min_dte or days_to_exp > max_dte:
                    continue
                
                # Apply other filters
                filtered_chain = filter_option_chain(
                    chain,
                    min_volume=min_volume,
                    max_spread_pct=max_spread_pct
                )
                
                # Skip if no options left after filtering
                if filtered_chain.empty:
                    continue
                    
                filtered_chains[expiry] = filtered_chain
                
            except Exception as e:
                logger.error(f"Error filtering option chain for {ticker} expiry {expiry}: {str(e)}")
        
        return filtered_chains
    
    def find_best_straddle(self, 
                          ticker: str, 
                          stock_price: float, 
                          historical_vol: Dict, 
                          option_chains: Dict[str, pd.DataFrame]) -> Tuple[str, float, float, Dict]:
        """
        Find the best straddle setup based on the configuration criteria.
        
        Args:
            ticker: Ticker symbol
            stock_price: Current stock price
            historical_vol: Historical volatility data
            option_chains: Filtered option chains by expiration date
            
        Returns:
            Tuple of (expiration, strike, score, details)
        """
        best_score = 0
        best_expiry = None
        best_strike = None
        best_details = {}
        
        # Get scoring weights from config
        scoring_weights = self.config.get('scoring_weights', {})
        
        # Get current IV data
        current_iv = self.market_data.get_current_iv(ticker)
        iv_rank = calculate_iv_rank(
            current_iv, 
            historical_vol.get('iv_52wk_low', 0), 
            historical_vol.get('iv_52wk_high', 1)
        )
        
        # Skip if IV rank is too low (unless override is enabled)
        min_iv_rank = self.config.get('entry_criteria', {}).get('min_iv_rank', 30)
        if iv_rank < min_iv_rank and not self.config.get('entry_criteria', {}).get('ignore_iv_rank', False):
            logger.info(f"Skipping {ticker}: IV rank {iv_rank:.1f} below minimum {min_iv_rank}")
            return None, None, 0, {}
        
        # Calculate IV history ratio
        iv_history_ratio = current_iv / historical_vol.get('iv_median', current_iv)
        
        # Calculate additional common metrics
        min_profit_target = self.config.get('risk_management', {}).get('min_profit_target', 0.5)
        
        for expiry, chain in option_chains.items():
            try:
                # Calculate days to expiration
                exp_date = datetime.strptime(expiry, '%Y-%m-%d')
                days_to_exp = (exp_date - datetime.now()).days
                
                # Calculate expected move
                expected_move = calculate_expected_move(stock_price, chain, days_to_exp)
                
                # Define strike selection method
                strike_method = self.config.get('strike_selection', {}).get('method', 'atm')
                
                # Find candidate strikes based on method
                if strike_method == 'atm':
                    # At-the-money strike
                    candidate_strikes = [round(stock_price / 5) * 5]  # Round to nearest $5
                elif strike_method == 'expected_move':
                    # Strike based on expected move
                    move_pct = self.config.get('strike_selection', {}).get('expected_move_factor', 0.5)
                    expected_target = stock_price * (1 + move_pct * expected_move / stock_price)
                    candidate_strikes = [round(expected_target / 5) * 5]
                elif strike_method == 'delta':
                    # Strike based on delta
                    target_delta = self.config.get('strike_selection', {}).get('target_delta', 0.5)
                    optimal_strike = find_optimal_strike(stock_price, chain, target_delta)
                    candidate_strikes = [optimal_strike]
                else:
                    # Default to ATM
                    candidate_strikes = [round(stock_price / 5) * 5]
                
                # Evaluate each candidate strike
                for strike in candidate_strikes:
                    # Find the call and put options at this strike
                    calls = chain[(chain['option_type'] == 'call') & (chain['strike'] == strike)]
                    puts = chain[(chain['option_type'] == 'put') & (chain['strike'] == strike)]
                    
                    if calls.empty or puts.empty:
                        continue
                    
                    # Get first matching option (should be only one per strike)
                    call_option = calls.iloc[0]
                    put_option = puts.iloc[0]
                    
                    # Calculate straddle cost
                    call_price = (call_option['bid'] + call_option['ask']) / 2
                    put_price = (put_option['bid'] + put_option['ask']) / 2
                    straddle_cost = calculate_straddle_cost(call_price, put_price)
                    
                    # Calculate key metrics
                    max_loss = calculate_max_loss(straddle_cost)
                    breakeven_points = calculate_straddle_breakeven_points(strike, straddle_cost)
                    
                    # Calculate bid-ask spread percentage
                    call_spread = (call_option['ask'] - call_option['bid']) / ((call_option['bid'] + call_option['ask']) / 2)
                    put_spread = (put_option['ask'] - put_option['bid']) / ((put_option['bid'] + put_option['ask']) / 2)
                    avg_spread = (call_spread + put_spread) / 2
                    
                    # Calculate option volume
                    avg_volume = (call_option['volume'] + put_option['volume']) / 2
                    
                    # Calculate probability of profit
                    avg_iv = (call_option['implied_volatility'] + put_option['implied_volatility']) / 2
                    pop = calculate_probability_of_profit(
                        stock_price, strike, straddle_cost, avg_iv, days_to_exp
                    )
                    
                    # Calculate risk/reward ratio
                    risk_reward = calculate_risk_reward_ratio(
                        straddle_cost, expected_move, days_to_exp
                    )
                    
                    # Skip if risk/reward is too low
                    if risk_reward < min_profit_target:
                        continue
                    
                    # Score the opportunity
                    score = score_straddle_opportunity(
                        ticker, stock_price, iv_rank, iv_history_ratio,
                        avg_spread, avg_volume, days_to_exp,
                        pop, scoring_weights
                    )
                    
                    # Check if this is the best score so far
                    if score > best_score:
                        best_score = score
                        best_expiry = expiry
                        best_strike = strike
                        best_details = {
                            'expected_move': expected_move,
                            'straddle_cost': straddle_cost,
                            'max_loss': max_loss,
                            'breakeven_points': breakeven_points,
                            'probability_of_profit': pop,
                            'risk_reward_ratio': risk_reward,
                            'iv_rank': iv_rank,
                            'days_to_expiration': days_to_exp,
                            'call_price': call_price,
                            'put_price': put_price,
                            'call_volume': call_option['volume'],
                            'put_volume': put_option['volume'],
                            'bid_ask_spread': avg_spread
                        }
            
            except Exception as e:
                logger.error(f"Error evaluating straddle for {ticker} expiry {expiry}: {str(e)}")
        
        return best_expiry, best_strike, best_score, best_details
    
    def execute_entries(self) -> List[Dict]:
        """
        Execute entry orders for the highest-scoring opportunities.
        
        Returns:
            List of executed trade details
        """
        executed_trades = []
        
        # Get positions limit and account value
        max_positions = self.config.get('risk_management', {}).get('max_positions', 3)
        current_positions = len(self.active_positions)
        available_positions = max_positions - current_positions
        
        if available_positions <= 0:
            logger.info("Maximum number of positions reached, no new entries")
            return executed_trades
            
        # Get account value
        account_value = self.position_manager.get_account_value()
        max_position_size = self.config.get('risk_management', {}).get('max_position_size', 0.05)
        max_dollars_per_position = account_value * max_position_size
        
        # Sort opportunities by score
        sorted_opportunities = sorted(
            self.opportunities.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Take top N based on available position slots
        top_opportunities = sorted_opportunities[:available_positions]
        
        # Minimum score threshold
        min_score = self.config.get('entry_criteria', {}).get('min_score', 60)
        
        for opportunity in top_opportunities:
            ticker = opportunity['ticker']
            score = opportunity['score']
            
            # Skip if score is below threshold
            if score < min_score:
                logger.info(f"Skipping {ticker}: Score {score:.1f} below minimum {min_score}")
                continue
                
            # Skip if already in this ticker
            if ticker in self.active_positions:
                logger.info(f"Skipping {ticker}: Already have an active position")
                continue
                
            try:
                # Get details for the trade
                details = opportunity['details']
                expiry = opportunity['expiration']
                strike = opportunity['strike']
                
                # Calculate contract counts
                contract_size = 100  # Standard US option contract size
                straddle_cost = details['straddle_cost'] * contract_size
                
                if straddle_cost <= 0:
                    logger.warning(f"Invalid straddle cost for {ticker}: {straddle_cost}")
                    continue
                    
                max_contracts = int(max_dollars_per_position / straddle_cost)
                
                if max_contracts <= 0:
                    logger.warning(f"Insufficient funds for minimum position in {ticker}")
                    continue
                    
                # Get configured position sizing method
                sizing_method = self.config.get('risk_management', {}).get('position_sizing', 'fixed')
                
                if sizing_method == 'fixed':
                    # Use fixed number of contracts
                    contracts = self.config.get('risk_management', {}).get('fixed_contracts', 1)
                    contracts = min(contracts, max_contracts)
                elif sizing_method == 'risk_percent':
                    # Base on risk percentage
                    risk_percent = self.config.get('risk_management', {}).get('risk_percent', 0.01)
                    dollars_to_risk = account_value * risk_percent
                    contracts = min(max_contracts, int(dollars_to_risk / straddle_cost))
                elif sizing_method == 'kelly':
                    # Kelly criterion sizing
                    win_prob = details['probability_of_profit'] / 100
                    profit_ratio = details['risk_reward_ratio']
                    kelly_pct = max(0.1, win_prob - ((1 - win_prob) / profit_ratio))
                    kelly_contracts = int(kelly_pct * max_contracts)
                    # Limit to half-Kelly for safety
                    contracts = min(max_contracts, max(1, int(kelly_contracts / 2)))
                else:
                    # Default to 1 contract
                    contracts = 1
                
                # Execute the trade
                call_order = self.order_manager.place_option_order(
                    ticker=ticker,
                    option_type='call',
                    strike=strike,
                    expiration=expiry,
                    contracts=contracts,
                    order_type='market',
                    direction='buy'
                )
                
                put_order = self.order_manager.place_option_order(
                    ticker=ticker,
                    option_type='put',
                    strike=strike,
                    expiration=expiry,
                    contracts=contracts,
                    order_type='market',
                    direction='buy'
                )
                
                # Track the new position
                if call_order.get('status') == 'filled' and put_order.get('status') == 'filled':
                    entry_time = datetime.now()
                    
                    # Save position details
                    self.active_positions[ticker] = {
                        'ticker': ticker,
                        'strategy': 'straddle',
                        'entry_date': entry_time,
                        'expiration': expiry,
                        'strike': strike,
                        'contracts': contracts,
                        'call_entry_price': call_order.get('fill_price', 0),
                        'put_entry_price': put_order.get('fill_price', 0),
                        'total_cost': (call_order.get('fill_price', 0) + put_order.get('fill_price', 0)) * contracts * 100,
                        'stop_loss': None,  # Will be calculated during monitoring
                        'profit_target': None,  # Will be calculated during monitoring
                        'expected_move': details['expected_move'],
                        'score': score
                    }
                    
                    # Log the entry
                    logger.info(f"Entered straddle position in {ticker}: "
                               f"{contracts} contracts at ${strike} strike, expiring {expiry}")
                    
                    # Add to executed trades list
                    executed_trades.append(self.active_positions[ticker])
                else:
                    # Log the failure
                    logger.error(f"Failed to enter straddle position in {ticker}: "
                                f"Call order status: {call_order.get('status')}, "
                                f"Put order status: {put_order.get('status')}")
            
            except Exception as e:
                logger.error(f"Error executing straddle entry for {ticker}: {str(e)}")
        
        return executed_trades
        
    def monitor_positions(self) -> List[Dict]:
        """
        Monitor active positions and manage exits according to strategy rules.
        
        Returns:
            List of closed position details
        """
        closed_positions = []
        
        # Get current date/time
        current_time = datetime.now()
        
        # Exit parameters from config
        stop_loss_pct = self.config.get('risk_management', {}).get('stop_loss_pct', 0.5)
        profit_target_pct = self.config.get('risk_management', {}).get('profit_target_pct', 1.0)
        max_days_held = self.config.get('risk_management', {}).get('max_days_held', 30)
        time_decay_exit = self.config.get('exit_criteria', {}).get('time_decay_exit_days', 14)
        
        # Check each active position
        for ticker, position in list(self.active_positions.items()):
            try:
                # Get current market data for both call and put
                call_data = self.market_data.get_option_price(
                    ticker, 
                    position['strike'], 
                    position['expiration'], 
                    'call'
                )
                
                put_data = self.market_data.get_option_price(
                    ticker, 
                    position['strike'], 
                    position['expiration'], 
                    'put'
                )
                
                # Calculate current position value
                call_current_price = (call_data.get('bid', 0) + call_data.get('ask', 0)) / 2
                put_current_price = (put_data.get('bid', 0) + put_data.get('ask', 0)) / 2
                current_value = (call_current_price + put_current_price) * position['contracts'] * 100
                
                # Calculate entry cost and P&L
                entry_cost = position['total_cost']
                pnl = current_value - entry_cost
                pnl_pct = pnl / entry_cost if entry_cost > 0 else 0
                
                # Calculate days held
                days_held = (current_time - position['entry_date']).days
                
                # Calculate days to expiration
                exp_date = datetime.strptime(position['expiration'], '%Y-%m-%d')
                days_to_exp = (exp_date - current_time).days
                
                # Determine if we should exit
                exit_reason = None
                
                # Check stop loss
                if pnl_pct <= -stop_loss_pct:
                    exit_reason = "stop_loss"
                
                # Check profit target
                elif pnl_pct >= profit_target_pct:
                    exit_reason = "profit_target"
                
                # Check max days held
                elif days_held >= max_days_held:
                    exit_reason = "max_days_held"
                
                # Check time decay exit
                elif days_to_exp <= time_decay_exit:
                    exit_reason = "time_decay"
                
                # Execute exit if needed
                if exit_reason:
                    # Close call position
                    call_exit = self.order_manager.place_option_order(
                        ticker=ticker,
                        option_type='call',
                        strike=position['strike'],
                        expiration=position['expiration'],
                        contracts=position['contracts'],
                        order_type='market',
                        direction='sell'
                    )
                    
                    # Close put position
                    put_exit = self.order_manager.place_option_order(
                        ticker=ticker,
                        option_type='put',
                        strike=position['strike'],
                        expiration=position['expiration'],
                        contracts=position['contracts'],
                        order_type='market',
                        direction='sell'
                    )
                    
                    # Update position with exit details
                    position['exit_date'] = current_time
                    position['call_exit_price'] = call_exit.get('fill_price', 0)
                    position['put_exit_price'] = put_exit.get('fill_price', 0)
                    position['exit_value'] = (position['call_exit_price'] + position['put_exit_price']) * position['contracts'] * 100
                    position['pnl'] = position['exit_value'] - position['total_cost']
                    position['pnl_pct'] = position['pnl'] / position['total_cost'] if position['total_cost'] > 0 else 0
                    position['exit_reason'] = exit_reason
                    position['days_held'] = days_held
                    
                    # Log the exit
                    logger.info(f"Exited straddle position in {ticker}: "
                               f"P&L: ${position['pnl']:.2f} ({position['pnl_pct']:.2%}), "
                               f"Reason: {exit_reason}")
                    
                    # Add to closed positions
                    closed_positions.append(position)
                    
                    # Remove from active positions
                    del self.active_positions[ticker]
                    
                    # Update historical performance
                    self._update_performance_history(position)
            
            except Exception as e:
                logger.error(f"Error monitoring straddle position for {ticker}: {str(e)}")
        
        return closed_positions
        
    def _update_performance_history(self, closed_position: Dict):
        """
        Update the historical performance database with closed position data.
        
        Args:
            closed_position: Details of the closed position
        """
        # Convert position to Series/DataFrame
        position_data = pd.Series(closed_position)
        
        # Append to historical performance
        self.historical_performance = pd.concat([
            self.historical_performance, 
            pd.DataFrame([position_data])
        ])
        
        # Notify optimization module if enabled
        if self.is_optimization_enabled:
            self._trigger_optimization()
            
    def _trigger_optimization(self):
        """
        Trigger strategy parameter optimization based on historical performance.
        """
        # Check if we have enough data for optimization
        min_trades = self.config.get('optimization', {}).get('min_trades_for_optimization', 20)
        
        if len(self.historical_performance) < min_trades:
            logger.debug(f"Not enough historical trades for optimization: {len(self.historical_performance)}/{min_trades}")
            return
            
        # Check if enough time has passed since last optimization
        last_optimization = getattr(self, 'last_optimization_time', None)
        optimization_interval = self.config.get('optimization', {}).get('optimization_interval_days', 30)
        
        if last_optimization and (datetime.now() - last_optimization).days < optimization_interval:
            logger.debug("Not enough time has passed since last optimization")
            return
            
        # Run optimization
        logger.info("Triggering straddle strategy parameter optimization")
        self.optimize_parameters()
        self.last_optimization_time = datetime.now()
        
    def optimize_parameters(self):
        """
        Optimize strategy parameters based on historical performance.
        """
        if not self.is_optimization_enabled:
            logger.info("Parameter optimization is disabled")
            return
            
        try:
            # Get optimization parameters from config
            lookback_days = self.config.get('optimization', {}).get('lookback_days', 180)
            lookback_date = datetime.now() - timedelta(days=lookback_days)
            
            # Filter historical data to lookback period
            recent_data = self.historical_performance[
                pd.to_datetime(self.historical_performance['entry_date']) >= lookback_date
            ]
            
            if len(recent_data) < 10:
                logger.info(f"Insufficient historical data for optimization: {len(recent_data)} trades")
                return
                
            # Calculate performance metrics
            win_rate = len(recent_data[recent_data['pnl'] > 0]) / len(recent_data)
            avg_profit = recent_data[recent_data['pnl'] > 0]['pnl'].mean()
            avg_loss = recent_data[recent_data['pnl'] <= 0]['pnl'].mean()
            profit_factor = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
            
            logger.info(f"Current performance: Win rate={win_rate:.2%}, Profit factor={profit_factor:.2f}")
            
            # Optimize parameters based on performance
            self._optimize_entry_criteria(recent_data, win_rate, profit_factor)
            self._optimize_exit_criteria(recent_data, win_rate, profit_factor)
            self._optimize_position_sizing(recent_data, win_rate, profit_factor)
            
            logger.info("Strategy parameters successfully optimized")
            
        except Exception as e:
            logger.error(f"Error during parameter optimization: {str(e)}")
    
    def _optimize_entry_criteria(self, historical_data, win_rate, profit_factor):
        """
        Optimize entry criteria parameters based on historical performance.
        
        Args:
            historical_data: DataFrame of historical trade data
            win_rate: Current win rate
            profit_factor: Current profit factor
        """
        # Analyze which entry parameters lead to better performance
        if len(historical_data) < 20:
            return
            
        # Example: Optimize IV rank threshold based on performance
        iv_ranks = historical_data['iv_rank']
        
        # Group by quartiles and analyze performance
        iv_quartiles = pd.qcut(iv_ranks, 4, duplicates='drop')
        performance_by_iv = historical_data.groupby(iv_quartiles)['pnl_pct'].mean()
        
        # Find optimal threshold
        best_quartile = performance_by_iv.idxmax()
        if best_quartile is not None:
            optimal_iv = best_quartile.left
            current_iv = self.config.get('entry_criteria', {}).get('min_iv_rank', 30)
            
            # Update if significantly different
            if abs(optimal_iv - current_iv) > 5:
                logger.info(f"Updating min_iv_rank from {current_iv} to {int(optimal_iv)}")
                self.config['entry_criteria']['min_iv_rank'] = int(optimal_iv)
    
    def _optimize_exit_criteria(self, historical_data, win_rate, profit_factor):
        """
        Optimize exit criteria parameters based on historical performance.
        
        Args:
            historical_data: DataFrame of historical trade data
            win_rate: Current win rate
            profit_factor: Current profit factor
        """
        # Analyze exit performance
        exit_reasons = historical_data['exit_reason'].value_counts()
        performance_by_exit = historical_data.groupby('exit_reason')['pnl_pct'].mean()
        
        # Optimize based on exit reason performance
        if 'stop_loss' in performance_by_exit and 'profit_target' in performance_by_exit:
            # Adjust stop loss and profit target if needed
            stop_loss_pct = self.config.get('risk_management', {}).get('stop_loss_pct', 0.5)
            profit_target_pct = self.config.get('risk_management', {}).get('profit_target_pct', 1.0)
            
            # If stop losses are too frequent and costly, widen them
            if (exit_reasons.get('stop_loss', 0) / len(historical_data) > 0.4 and 
                performance_by_exit['stop_loss'] < -0.3):
                new_stop_loss = min(0.7, stop_loss_pct * 1.2)
                logger.info(f"Widening stop_loss_pct from {stop_loss_pct:.2f} to {new_stop_loss:.2f}")
                self.config['risk_management']['stop_loss_pct'] = new_stop_loss
            
            # If profit targets are rarely hit, lower them
            if (exit_reasons.get('profit_target', 0) / len(historical_data) < 0.2):
                new_profit_target = max(0.5, profit_target_pct * 0.8)
                logger.info(f"Lowering profit_target_pct from {profit_target_pct:.2f} to {new_profit_target:.2f}")
                self.config['risk_management']['profit_target_pct'] = new_profit_target
    
    def _optimize_position_sizing(self, historical_data, win_rate, profit_factor):
        """
        Optimize position sizing parameters based on historical performance.
        
        Args:
            historical_data: DataFrame of historical trade data
            win_rate: Current win rate
            profit_factor: Current profit factor
        """
        # Adjust position sizing based on overall performance
        max_position_size = self.config.get('risk_management', {}).get('max_position_size', 0.05)
        
        # If performance is very good, consider increasing position size
        if win_rate > 0.6 and profit_factor > 2.0:
            new_position_size = min(0.1, max_position_size * 1.2)
            logger.info(f"Increasing max_position_size from {max_position_size:.2f} to {new_position_size:.2f}")
            self.config['risk_management']['max_position_size'] = new_position_size
        
        # If performance is poor, reduce position size
        elif win_rate < 0.4 or profit_factor < 1.0:
            new_position_size = max(0.01, max_position_size * 0.8)
            logger.info(f"Decreasing max_position_size from {max_position_size:.2f} to {new_position_size:.2f}")
            self.config['risk_management']['max_position_size'] = new_position_size
            
    def run(self):
        """
        Run a complete cycle of the strategy.
        """
        # Scan for opportunities
        logger.info("Scanning for straddle opportunities...")
        self.scan_for_opportunities()
        
        # Execute new entries
        logger.info("Executing entry orders...")
        self.execute_entries()
        
        # Monitor and manage existing positions
        logger.info("Monitoring active positions...")
        self.monitor_positions()
        
        # Update strategy state
        if self.is_optimization_enabled:
            self._trigger_optimization()
            
        logger.info(f"Strategy cycle complete: {len(self.opportunities)} opportunities found, "
                   f"{len(self.active_positions)} active positions")
        
    def get_strategy_state(self) -> Dict:
        """
        Get the current state of the strategy.
        
        Returns:
            Dictionary with strategy state information
        """
        return {
            'name': 'Straddle Strategy',
            'active_positions': len(self.active_positions),
            'active_position_details': self.active_positions,
            'current_opportunities': len(self.opportunities),
            'opportunity_details': self.opportunities,
            'historical_trades': len(self.historical_performance),
            'config': self.config
        } 