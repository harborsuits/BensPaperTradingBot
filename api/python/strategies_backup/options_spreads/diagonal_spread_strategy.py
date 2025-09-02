"""
Diagonal Spread Options Strategy Implementation

This module implements a diagonal spread options strategy, which involves buying a longer-dated
option at one strike and selling a shorter-dated option at a different strike to capture 
both time decay and a directional move.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Tuple, Optional, Union

from trading_bot.strategies.strategy_template import StrategyOptimizable
from trading_bot.utils.diagonal_spread_utils import (
    calculate_trend_strength,
    find_diagonal_spread_candidates,
    calculate_diagonal_spread_metrics,
    calculate_diagonal_spread_adjustments,
    calculate_roll_cost,
    analyze_diagonal_spread_performance
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
from trading_bot.config.diagonal_spread_config import DIAGONAL_SPREAD_CONFIG

logger = logging.getLogger(__name__)

class DiagonalSpreadStrategy(StrategyOptimizable):
    """
    Diagonal Spread Options Strategy Implementation
    
    A diagonal spread combines elements of vertical and calendar spreads by using options
    with different strikes and different expiration dates. It's designed to profit from
    both time decay and a directional move in the underlying asset.
    """
    
    def __init__(self, 
                 market_data: MarketData,
                 order_manager: OrderManager,
                 position_manager: PositionManager,
                 risk_manager: RiskManager,
                 config: Dict = None):
        """
        Initialize the diagonal spread strategy.
        
        Args:
            market_data: Market data provider
            order_manager: Order execution manager
            position_manager: Position tracking manager
            risk_manager: Risk management service
            config: Strategy configuration parameters (optional)
        """
        super().__init__(market_data, order_manager, position_manager, risk_manager)
        
        # Load configuration
        self.config = config or DIAGONAL_SPREAD_CONFIG
        
        # Strategy state
        self.universe = self.config["universe"]["symbols"]
        self.active_positions = {}
        self.pending_rolls = {}
        self.opportunities = {}
        self.historical_performance = pd.DataFrame()
        
        logger.info(f"Diagonal spread strategy initialized with {len(self.universe)} tickers in universe")
    
    def scan_for_opportunities(self) -> Dict[str, Dict]:
        """
        Scan the strategy universe for diagonal spread opportunities.
        
        Returns:
            Dictionary of opportunities with ticker as key
        """
        opportunities = {}
        
        for ticker in self.universe:
            try:
                # Skip if already have a position in this ticker
                if ticker in self.active_positions:
                    continue
                
                # Get current price and price history
                current_price = self.market_data.get_latest_price(ticker)
                price_history = self.market_data.get_price_history(
                    ticker, 
                    days=self.config["underlying_criteria"]["trend_period_days"] * 2
                )
                
                if current_price <= 0 or price_history.empty:
                    logger.warning(f"Insufficient price data for {ticker}")
                    continue
                
                # Check volume requirement
                avg_volume = price_history['volume'].mean()
                if avg_volume < self.config["underlying_criteria"]["min_adv"]:
                    logger.debug(f"{ticker} volume {avg_volume:.0f} below minimum {self.config['underlying_criteria']['min_adv']}")
                    continue
                
                # Analyze trend
                trend_strength, trend_direction = calculate_trend_strength(
                    price_history,
                    period_days=self.config["underlying_criteria"]["trend_period_days"]
                )
                
                # Skip if trend is not strong enough or neutral
                if (trend_strength < self.config["underlying_criteria"]["trend_strength_threshold"] or 
                    trend_direction == "neutral"):
                    logger.debug(f"{ticker} trend not suitable: {trend_direction} with strength {trend_strength:.2f}")
                    continue
                
                # Get IV rank
                iv_data = self.market_data.get_historical_iv(ticker)
                current_iv = iv_data[-1] if len(iv_data) > 0 else None
                
                if current_iv is None:
                    logger.warning(f"No IV data available for {ticker}")
                    continue
                
                iv_rank = calculate_iv_rank(current_iv, min(iv_data), max(iv_data)) if len(iv_data) > 2 else 50
                
                # Skip if IV rank is outside desired range
                if (iv_rank < self.config["underlying_criteria"]["min_iv_rank"] or 
                    iv_rank > self.config["underlying_criteria"]["max_iv_rank"]):
                    logger.debug(f"{ticker} IV rank {iv_rank:.1f} outside target range")
                    continue
                
                # Get option chain
                option_chain = self.market_data.get_option_chain(ticker)
                
                if not option_chain:
                    logger.warning(f"No option chain data available for {ticker}")
                    continue
                
                # Find diagonal spread candidates
                candidates = find_diagonal_spread_candidates(
                    ticker, 
                    trend_direction,
                    current_price,
                    option_chain,
                    self.config
                )
                
                if candidates:
                    # Get the best candidate
                    best_candidate = candidates[0]
                    best_candidate["iv_rank"] = iv_rank
                    best_candidate["trend_strength"] = trend_strength
                    best_candidate["current_price"] = current_price
                    
                    opportunities[ticker] = best_candidate
                    logger.info(f"Found diagonal spread opportunity for {ticker}: "
                               f"{trend_direction} {best_candidate['option_type']} diagonal with score {best_candidate['score']:.1f}")
            
            except Exception as e:
                logger.error(f"Error scanning {ticker} for diagonal spread opportunities: {str(e)}")
        
        self.opportunities = opportunities
        return opportunities
    
    def check_position_sizing(self, spread: Dict) -> int:
        """
        Calculate the appropriate position size for a spread opportunity.
        
        Args:
            spread: Spread opportunity details
            
        Returns:
            Number of spreads to trade
        """
        # Get account value
        account_value = self.position_manager.get_account_value()
        
        # Get risk per spread parameter
        max_risk_pct = self.config["risk_management"]["max_risk_per_spread_pct"] / 100.0
        
        # Calculate maximum dollars to risk
        max_dollars_to_risk = account_value * max_risk_pct
        
        # Calculate cost per spread
        cost_per_spread = spread["net_debit"] * 100  # Convert to dollars per spread
        
        # Calculate maximum number of spreads
        max_spreads = int(max_dollars_to_risk / cost_per_spread)
        
        # Ensure at least 1 spread
        return max(1, max_spreads)
    
    def check_sector_limits(self, ticker: str) -> bool:
        """
        Check if adding a position in this ticker would exceed sector limits.
        
        Args:
            ticker: Ticker symbol to check
            
        Returns:
            True if within limits, False if would exceed limits
        """
        # Get sector correlation matrix
        sector_matrix = self.config["risk_management"]["sector_correlation_matrix"]
        max_per_sector = self.config["risk_management"]["max_positions_per_sector"]
        
        # Find which sector this ticker belongs to
        ticker_sector = None
        for sector, tickers in sector_matrix.items():
            if ticker in tickers:
                ticker_sector = sector
                break
        
        if ticker_sector is None:
            # Not in any defined sector, so allow it
            return True
        
        # Count active positions in this sector
        sector_positions = 0
        for pos_ticker in self.active_positions:
            for sector, tickers in sector_matrix.items():
                if sector == ticker_sector and pos_ticker in tickers:
                    sector_positions += 1
        
        return sector_positions < max_per_sector
    
    def execute_entries(self) -> List[Dict]:
        """
        Execute entry orders for diagonal spread opportunities.
        
        Returns:
            List of executed trade details
        """
        executed_trades = []
        
        # Check if we can add more positions
        max_positions = self.config["risk_management"]["max_positions"]
        current_positions = len(self.active_positions)
        
        if current_positions >= max_positions:
            logger.info(f"Already at maximum positions ({max_positions}), no new entries")
            return executed_trades
        
        # Sort opportunities by score
        sorted_opps = sorted(
            self.opportunities.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        # Try to execute top opportunities
        positions_to_add = max_positions - current_positions
        
        for opportunity in sorted_opps[:positions_to_add]:
            ticker = opportunity["ticker"]
            
            # Check sector limits
            if not self.check_sector_limits(ticker):
                logger.info(f"Skipping {ticker}: Would exceed sector position limit")
                continue
            
            # Calculate appropriate position size
            position_size = self.check_position_sizing(opportunity)
            
            try:
                # Prepare order details
                long_leg = opportunity["long_leg"]
                short_leg = opportunity["short_leg"]
                
                # Use combo order if configured
                if self.config["entry_execution"]["use_combo_order"]:
                    # Place combo order
                    combo_order = self.order_manager.place_spread_order(
                        ticker=ticker,
                        legs=[
                            {
                                "option_type": opportunity["option_type"],
                                "strike": long_leg["strike"],
                                "expiration": long_leg["expiry"],
                                "direction": "buy",
                                "contracts": position_size
                            },
                            {
                                "option_type": opportunity["option_type"],
                                "strike": short_leg["strike"],
                                "expiration": short_leg["expiry"],
                                "direction": "sell",
                                "contracts": position_size
                            }
                        ],
                        order_type="net_debit",
                        price=opportunity["net_debit"],
                        strategy_type="diagonal"
                    )
                    
                    if combo_order.get("status") == "filled":
                        # Record the position
                        entry_time = datetime.now()
                        
                        position = {
                            "ticker": ticker,
                            "strategy_type": "diagonal",
                            "direction": opportunity["trend_direction"],
                            "option_type": opportunity["option_type"],
                            "entry_date": entry_time,
                            "entry_price": opportunity["current_price"],
                            "current_price": opportunity["current_price"],
                            "long_leg": long_leg,
                            "short_leg": short_leg,
                            "position_size": position_size,
                            "net_debit": opportunity["net_debit"],
                            "total_cost": opportunity["net_debit"] * position_size * 100,
                            "original_metrics": opportunity,
                            "days_held": 0,
                            "rolls_executed": 0,
                            "status": "active"
                        }
                        
                        # Add to active positions
                        self.active_positions[ticker] = position
                        executed_trades.append(position)
                        
                        logger.info(f"Entered diagonal spread in {ticker}: {position_size} contracts")
                    else:
                        logger.error(f"Failed to enter diagonal spread in {ticker}: {combo_order.get('status')}")
                
                else:
                    # Place leg orders separately
                    long_order = self.order_manager.place_option_order(
                        ticker=ticker,
                        option_type=opportunity["option_type"],
                        strike=long_leg["strike"],
                        expiration=long_leg["expiry"],
                        contracts=position_size,
                        order_type="limit",
                        direction="buy",
                        price=long_leg["ask"]
                    )
                    
                    if long_order.get("status") == "filled":
                        # Only place short leg if long leg fills
                        short_order = self.order_manager.place_option_order(
                            ticker=ticker,
                            option_type=opportunity["option_type"],
                            strike=short_leg["strike"],
                            expiration=short_leg["expiry"],
                            contracts=position_size,
                            order_type="limit",
                            direction="sell",
                            price=short_leg["bid"]
                        )
                        
                        if short_order.get("status") == "filled":
                            # Record the position
                            entry_time = datetime.now()
                            
                            # Calculate actual net debit from fills
                            actual_debit = long_order.get("fill_price", long_leg["ask"]) - short_order.get("fill_price", short_leg["bid"])
                            
                            position = {
                                "ticker": ticker,
                                "strategy_type": "diagonal",
                                "direction": opportunity["trend_direction"],
                                "option_type": opportunity["option_type"],
                                "entry_date": entry_time,
                                "entry_price": opportunity["current_price"],
                                "current_price": opportunity["current_price"],
                                "long_leg": {
                                    **long_leg,
                                    "fill_price": long_order.get("fill_price", long_leg["ask"])
                                },
                                "short_leg": {
                                    **short_leg,
                                    "fill_price": short_order.get("fill_price", short_leg["bid"])
                                },
                                "position_size": position_size,
                                "net_debit": actual_debit,
                                "total_cost": actual_debit * position_size * 100,
                                "original_metrics": opportunity,
                                "days_held": 0,
                                "rolls_executed": 0,
                                "status": "active"
                            }
                            
                            # Add to active positions
                            self.active_positions[ticker] = position
                            executed_trades.append(position)
                            
                            logger.info(f"Entered diagonal spread in {ticker}: {position_size} contracts")
                        else:
                            # Failed to fill short leg, need to close long leg
                            close_long = self.order_manager.place_option_order(
                                ticker=ticker,
                                option_type=opportunity["option_type"],
                                strike=long_leg["strike"],
                                expiration=long_leg["expiry"],
                                contracts=position_size,
                                order_type="market",
                                direction="sell"
                            )
                            
                            logger.warning(f"Failed to fill short leg for {ticker}, closed long leg")
                    else:
                        logger.error(f"Failed to enter long leg for {ticker}: {long_order.get('status')}")
            
            except Exception as e:
                logger.error(f"Error executing entry for {ticker}: {str(e)}")
        
        return executed_trades
    
    def monitor_positions(self) -> List[Dict]:
        """
        Monitor active positions, manage rolls, and execute exits as needed.
        
        Returns:
            List of position updates, including closed positions and rolled positions
        """
        position_updates = []
        current_time = datetime.now()
        
        for ticker, position in list(self.active_positions.items()):
            try:
                # Get current price
                current_price = self.market_data.get_latest_price(ticker)
                
                if current_price <= 0:
                    logger.warning(f"Could not get current price for {ticker}")
                    continue
                
                # Update position data
                days_held = (current_time - position["entry_date"]).days
                position["days_held"] = days_held
                position["current_price"] = current_price
                
                # Check for adjustment needs
                adjustments = calculate_diagonal_spread_adjustments(
                    position,
                    current_price,
                    days_held
                )
                
                # Handle rolling short leg if needed
                if adjustments["should_roll_short"]:
                    # Record in pending rolls
                    self.pending_rolls[ticker] = {
                        "position": position,
                        "recommendations": adjustments
                    }
                    
                    logger.info(f"Rolling short leg for {ticker}: {adjustments['explanation']}")
                    
                    # Get recommendation details
                    roll_recommendations = adjustments["roll_recommendation"]
                    if not roll_recommendations:
                        logger.warning(f"No roll recommendations available for {ticker}")
                        continue
                    
                    # Get new expiration based on target DTE
                    target_dte = roll_recommendations["target_dte"]
                    target_expiry_date = (current_time + timedelta(days=target_dte)).date()
                    
                    # Get option chain
                    option_chain = self.market_data.get_option_chain(ticker)
                    
                    if not option_chain:
                        logger.warning(f"Could not get option chain for {ticker} roll")
                        continue
                    
                    # Find closest expiration to target
                    closest_expiry = None
                    closest_dte_diff = float('inf')
                    
                    for expiry_str in option_chain.keys():
                        try:
                            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
                            dte_diff = abs((expiry_date - target_expiry_date).days)
                            
                            if dte_diff < closest_dte_diff:
                                closest_dte_diff = dte_diff
                                closest_expiry = expiry_str
                        except:
                            continue
                    
                    if not closest_expiry:
                        logger.warning(f"Could not find suitable expiration for {ticker} roll")
                        continue
                    
                    # Determine new strike
                    new_strike = adjustments.get("target_strike")
                    if not new_strike:
                        # Use original strike if no adjustment needed
                        new_strike = position["short_leg"]["strike"]
                    
                    # Calculate roll cost and impact
                    roll_details = calculate_roll_cost(
                        position,
                        closest_expiry,
                        new_strike,
                        option_chain
                    )
                    
                    if not roll_details["success"]:
                        logger.warning(f"Roll calculation failed for {ticker}: {roll_details.get('message')}")
                        continue
                    
                    # Execute the roll
                    # 1. Buy back short leg
                    buy_to_close = self.order_manager.place_option_order(
                        ticker=ticker,
                        option_type=position["option_type"],
                        strike=position["short_leg"]["strike"],
                        expiration=position["short_leg"]["expiry"],
                        contracts=position["position_size"],
                        order_type="market",
                        direction="buy"
                    )
                    
                    if buy_to_close.get("status") != "filled":
                        logger.error(f"Failed to close short leg for {ticker} roll: {buy_to_close.get('status')}")
                        continue
                    
                    # 2. Sell new short leg
                    sell_to_open = self.order_manager.place_option_order(
                        ticker=ticker,
                        option_type=position["option_type"],
                        strike=new_strike,
                        expiration=closest_expiry,
                        contracts=position["position_size"],
                        order_type="market",
                        direction="sell"
                    )
                    
                    if sell_to_open.get("status") != "filled":
                        logger.error(f"Failed to open new short leg for {ticker} roll: {sell_to_open.get('status')}")
                        
                        # Need to adjust position to reflect the closed short leg
                        position["status"] = "long_only"
                        position_updates.append(position)
                        continue
                    
                    # Update position with new short leg
                    old_short = position["short_leg"]
                    
                    position["short_leg"] = {
                        "expiry": closest_expiry,
                        "strike": new_strike,
                        "dte": (datetime.strptime(closest_expiry, "%Y-%m-%d").date() - current_time.date()).days,
                        "fill_price": sell_to_open.get("fill_price", 0)
                    }
                    
                    position["rolls_executed"] += 1
                    
                    # Record roll details
                    roll_info = {
                        "ticker": ticker,
                        "roll_date": current_time,
                        "days_held_before_roll": days_held,
                        "old_expiry": old_short["expiry"],
                        "new_expiry": closest_expiry,
                        "old_strike": old_short["strike"],
                        "new_strike": new_strike,
                        "roll_cost": buy_to_close.get("fill_price", 0) - sell_to_open.get("fill_price", 0),
                        "position": position
                    }
                    
                    position_updates.append(roll_info)
                    logger.info(f"Successfully rolled {ticker} short leg from {old_short['expiry']} to {closest_expiry}")
                    
                    # Remove from pending rolls
                    if ticker in self.pending_rolls:
                        del self.pending_rolls[ticker]
                
                # Check for complete exit
                elif adjustments["should_close_spread"]:
                    self.execute_exit(position, "time_exit", position_updates)
                
                # Update position with current status
                else:
                    # Get current option prices for P&L calculation
                    long_leg_price = self.market_data.get_option_price(
                        ticker, 
                        position["long_leg"]["strike"],
                        position["long_leg"]["expiry"],
                        position["option_type"]
                    )
                    
                    short_leg_price = self.market_data.get_option_price(
                        ticker,
                        position["short_leg"]["strike"],
                        position["short_leg"]["expiry"],
                        position["option_type"]
                    )
                    
                    if long_leg_price and short_leg_price:
                        # Calculate current value (long - short)
                        long_current = (long_leg_price.get("bid", 0) + long_leg_price.get("ask", 0)) / 2
                        short_current = (short_leg_price.get("bid", 0) + short_leg_price.get("ask", 0)) / 2
                        current_value = long_current - short_current
                        
                        # Calculate P&L
                        initial_debit = position["net_debit"]
                        pnl = (current_value - initial_debit) * position["position_size"] * 100
                        pnl_pct = ((current_value / initial_debit) - 1) * 100 if initial_debit > 0 else 0
                        
                        position["current_value"] = current_value
                        position["current_pnl"] = pnl
                        position["current_pnl_pct"] = pnl_pct
                        
                        # Check for profit target exit
                        profit_target_pct = self.config["exit_rules"]["profit_take_pct"]
                        if pnl_pct >= profit_target_pct:
                            self.execute_exit(position, "profit_target", position_updates)
                        
                        # Check for stop loss exit
                        stop_loss_pct = self.config["exit_rules"]["stop_loss_pct"]
                        if pnl_pct <= -stop_loss_pct:
                            self.execute_exit(position, "stop_loss", position_updates)
                
            except Exception as e:
                logger.error(f"Error monitoring position for {ticker}: {str(e)}")
        
        return position_updates
    
    def execute_exit(self, position: Dict, reason: str, updates_list: List) -> bool:
        """
        Execute a complete exit for a diagonal spread position.
        
        Args:
            position: Position to exit
            reason: Reason for exit
            updates_list: List to append the exit details to
            
        Returns:
            True if exit was successful, False otherwise
        """
        ticker = position["ticker"]
        
        try:
            # Close the long leg
            long_exit = self.order_manager.place_option_order(
                ticker=ticker,
                option_type=position["option_type"],
                strike=position["long_leg"]["strike"],
                expiration=position["long_leg"]["expiry"],
                contracts=position["position_size"],
                order_type="market",
                direction="sell"
            )
            
            # If there's still a short leg, buy it back
            if position["status"] != "long_only":
                short_exit = self.order_manager.place_option_order(
                    ticker=ticker,
                    option_type=position["option_type"],
                    strike=position["short_leg"]["strike"],
                    expiration=position["short_leg"]["expiry"],
                    contracts=position["position_size"],
                    order_type="market",
                    direction="buy"
                )
                
                short_exit_status = short_exit.get("status")
                short_exit_price = short_exit.get("fill_price", 0)
            else:
                # No short leg to close
                short_exit_status = "no_short_leg"
                short_exit_price = 0
            
            # Check if both legs were successfully closed
            if long_exit.get("status") == "filled" and (short_exit_status == "filled" or short_exit_status == "no_short_leg"):
                # Update position with exit details
                exit_time = datetime.now()
                
                # Calculate exit value and P&L
                long_exit_price = long_exit.get("fill_price", 0)
                net_exit_value = long_exit_price - short_exit_price
                
                initial_debit = position["net_debit"]
                pnl = (net_exit_value - initial_debit) * position["position_size"] * 100
                pnl_pct = ((net_exit_value / initial_debit) - 1) * 100 if initial_debit > 0 else 0
                
                # Update position with exit details
                position["exit_date"] = exit_time
                position["exit_price"] = position["current_price"]
                position["long_exit_price"] = long_exit_price
                position["short_exit_price"] = short_exit_price
                position["net_exit_value"] = net_exit_value
                position["pnl"] = pnl
                position["pnl_pct"] = pnl_pct
                position["exit_reason"] = reason
                position["status"] = "closed"
                
                # Add to updates list
                updates_list.append(position)
                
                # Add to historical performance
                self._update_performance_history(position)
                
                # Remove from active positions
                if ticker in self.active_positions:
                    del self.active_positions[ticker]
                
                logger.info(f"Exited diagonal spread in {ticker}: "
                           f"P&L ${pnl:.2f} ({pnl_pct:.2f}%), Reason: {reason}")
                
                return True
            else:
                logger.error(f"Failed to exit position in {ticker}: "
                            f"Long status: {long_exit.get('status')}, Short status: {short_exit_status}")
                return False
        
        except Exception as e:
            logger.error(f"Error executing exit for {ticker}: {str(e)}")
            return False
    
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
        
        # Analyze performance
        entry_date = closed_position["entry_date"]
        exit_date = closed_position["exit_date"]
        
        # Get price history for analysis
        price_history = self.market_data.get_price_history(
            closed_position["ticker"],
            days=(exit_date - entry_date).days + 5  # Add buffer days
        )
        
        performance_metrics = analyze_diagonal_spread_performance(
            closed_position,
            entry_date,
            exit_date,
            price_history
        )
        
        logger.info(f"Performance analysis for {closed_position['ticker']}: "
                   f"Days held: {performance_metrics['days_held']}, "
                   f"Annualized return: {performance_metrics['annualized_return']:.2f}%, "
                   f"Theta capture: ${performance_metrics['theta_capture']:.2f}")
    
    def run(self):
        """
        Run a complete cycle of the strategy.
        """
        # Scan for opportunities
        logger.info("Scanning for diagonal spread opportunities...")
        self.scan_for_opportunities()
        
        # Execute new entries
        logger.info("Executing entry orders...")
        self.execute_entries()
        
        # Monitor and manage existing positions
        logger.info("Monitoring active positions...")
        self.monitor_positions()
        
        logger.info(f"Strategy cycle complete: {len(self.opportunities)} opportunities found, "
                   f"{len(self.active_positions)} active positions, "
                   f"{len(self.pending_rolls)} pending rolls")
    
    def get_strategy_state(self) -> Dict:
        """
        Get the current state of the strategy.
        
        Returns:
            Dictionary with strategy state information
        """
        # Aggregate performance metrics if we have historical trades
        performance_metrics = {}
        if not self.historical_performance.empty:
            # Calculate basic metrics
            num_trades = len(self.historical_performance)
            win_rate = (self.historical_performance["pnl"] > 0).mean() * 100
            avg_pnl = self.historical_performance["pnl"].mean()
            avg_pnl_pct = self.historical_performance["pnl_pct"].mean()
            
            # Calculate by exit reason
            by_reason = self.historical_performance.groupby("exit_reason")["pnl"].agg(["count", "mean"])
            
            performance_metrics = {
                "num_trades": num_trades,
                "win_rate": win_rate,
                "avg_pnl": avg_pnl,
                "avg_pnl_pct": avg_pnl_pct,
                "by_exit_reason": by_reason.to_dict()
            }
        
        return {
            "name": "Diagonal Spread Strategy",
            "active_positions": len(self.active_positions),
            "active_position_details": self.active_positions,
            "current_opportunities": len(self.opportunities),
            "opportunity_details": self.opportunities,
            "performance_metrics": performance_metrics,
            "config": self.config
        } 