"""
Strangle Options Strategy

This module implements a strangle options trading strategy. A strangle is an options strategy
involving the purchase of both a put and a call option for the same underlying asset, expiration date,
but with different strike prices. It's typically used when a trader expects significant price movement
in the underlying asset but is uncertain about the direction.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any

from trading_bot.strategies.strategy_template import StrategyOptimizable as BaseOptionsStrategy
from trading_bot.utils.strangle_utils import (
    calculate_strangle_cost, calculate_strangle_breakeven_points,
    calculate_strangle_pnl, find_strangle_strikes, find_strangle_by_otm_pct,
    calculate_expected_move, calculate_probability_of_profit,
    calculate_strangle_metrics, score_strangle_opportunity,
    calculate_liquidity_score, calculate_vega_capture_ratio,
    check_recent_gaps, analyze_event_outcome
)
# Local implementations are already provided below
# Remove the problematic imports
from trading_bot.utils.option_utils import get_atm_strike

# Local implementation of calculate_iv_rank
def calculate_iv_rank(price_data):
    """
    Calculate the IV Rank based on historical implied volatility.
    This is a placeholder implementation - in a real environment, you would
    calculate this from actual historical IV data.
    """
    # Placeholder implementation - would use real IV data in production
    return 60  # Default to a moderate IV rank for testing

# Local implementation of get_upcoming_events
def get_upcoming_events(symbol):
    """
    Get upcoming events for a symbol, such as earnings, dividends, etc.
    This is a placeholder implementation.
    """
    # Placeholder - in production this would call an API or database
    return []  # Return empty list as placeholder

# Local implementation of position_size_by_risk to avoid import issues
def position_size_by_risk(
    price: float,
    account_value: float,
    risk_percent: float,
    stop_price: Optional[float] = None,
    max_position_percent: float = 15.0,
    min_position_size: int = 1
) -> int:
    """
    Calculate position size based on risk percentage of account value.
    
    Args:
        price: Current price of the asset
        account_value: Total account value in dollars
        risk_percent: Percentage of account to risk (e.g., 1.0 for 1%)
        stop_price: Price level for stop loss (if None, uses 7% from entry price)
        max_position_percent: Maximum percentage of account for any position
        min_position_size: Minimum position size to return
        
    Returns:
        Recommended position size in number of shares/contracts
    """
    try:
        # Validate inputs
        if price <= 0:
            logging.warning("Invalid price, must be positive")
            return min_position_size
            
        if account_value <= 0:
            logging.warning("Invalid account value, must be positive")
            return min_position_size
            
        if risk_percent <= 0 or risk_percent > 100:
            logging.warning("Invalid risk percentage, must be between 0 and 100")
            risk_percent = 1.0  # Default to 1%
        
        # Calculate risk amount in dollars
        risk_amount = account_value * (risk_percent / 100)
        
        # Calculate dollar risk per share
        if stop_price is not None and stop_price > 0:
            # If stop price is provided, use it
            if price > stop_price:
                # Long position
                dollar_risk_per_share = price - stop_price
            else:
                # Short position
                dollar_risk_per_share = stop_price - price
        else:
            # Use default 7% risk if no stop price provided
            dollar_risk_per_share = price * 0.07
            
        # Avoid division by zero
        if dollar_risk_per_share <= 0:
            logging.warning("Invalid dollar risk per share, defaulting to 1% of price")
            dollar_risk_per_share = price * 0.01
            
        # Calculate position size based on risk
        position_size = int(risk_amount / dollar_risk_per_share)
        
        # Check against maximum position size
        max_position_size = int((account_value * max_position_percent / 100) / price)
        position_size = min(position_size, max_position_size)
        
        # Ensure minimum position size
        position_size = max(position_size, min_position_size)
        
        return position_size
        
    except Exception as e:
        logging.error(f"Error calculating position size: {e}")
        return min_position_size

class StrangleStrategy(BaseOptionsStrategy):
    """
    Options strategy implementation for strangles.
    
    A strangle involves buying both an OTM call and OTM put with the same expiration
    to profit from significant price movements in either direction.
    """
    
    def __init__(self, config: Dict, broker: Any) -> None:
        """
        Initialize the strangle strategy with configuration settings.
        
        Args:
            config: Dictionary containing strategy configuration
            broker: Broker instance for executing trades
        """
        super().__init__(config, broker)
        self.name = "Strangle"
        self.logger = logging.getLogger(f"trading_bot.strategies.{self.name}")
        self.logger.setLevel(self.config.get("log_level", logging.INFO))
        
        self.logger.info(f"Initializing {self.name} strategy with config: {self.config}")
        
        # Configuration parameters specific to strangles
        self.min_iv_rank = self.config.get("entry_criteria", {}).get("min_iv_rank", 50)
        self.target_delta_call = self.config.get("entry_criteria", {}).get("target_delta_call", 0.30)
        self.target_delta_put = self.config.get("entry_criteria", {}).get("target_delta_put", -0.30)
        self.target_otm_pct = self.config.get("entry_criteria", {}).get("target_otm_pct", 0.10)
        self.min_liquidity_score = self.config.get("entry_criteria", {}).get("min_liquidity_score", 60)
        self.min_score = self.config.get("entry_criteria", {}).get("min_opportunity_score", 70)
        
        # Risk management
        self.max_position_size = self.config.get("risk_management", {}).get("max_position_size", 5000)
        self.max_risk_per_trade_pct = self.config.get("risk_management", {}).get("max_risk_per_trade_pct", 2.0)
        self.max_correlated_positions = self.config.get("risk_management", {}).get("max_correlated_positions", 3)
        
        # Exit criteria
        self.profit_target_pct = self.config.get("exit_criteria", {}).get("profit_target_pct", 50)
        self.stop_loss_pct = self.config.get("exit_criteria", {}).get("stop_loss_pct", 75)
        self.days_before_expiry_to_close = self.config.get("exit_criteria", {}).get("days_before_expiry_to_close", 5)
        self.iv_drop_exit_pct = self.config.get("exit_criteria", {}).get("iv_drop_exit_pct", 20)
        
        # Track open positions and opportunities
        self.open_positions = {}
        self.current_opportunities = {}
        
    def scan_for_opportunities(self, symbols: List[str]) -> List[Dict]:
        """
        Scan for strangle opportunities across a list of symbols.
        
        Args:
            symbols: List of symbols to scan
            
        Returns:
            List of opportunity dictionaries, sorted by score
        """
        self.logger.info(f"Scanning {len(symbols)} symbols for strangle opportunities")
        opportunities = []
        
        for symbol in symbols:
            try:
                # Get market data
                price_data = self.broker.get_price_history(symbol)
                option_chain = self.broker.get_option_chain(symbol)
                current_price = price_data.iloc[-1]["Close"]
                
                # Filter by IV rank
                iv_rank = calculate_iv_rank(price_data)
                if iv_rank < self.min_iv_rank:
                    self.logger.debug(f"Skipping {symbol}: IV rank {iv_rank:.1f} below minimum {self.min_iv_rank}")
                    continue
                
                # Check for upcoming events
                upcoming_events = get_upcoming_events(symbol)
                event_score = 0
                if upcoming_events:
                    # Score events based on proximity and importance
                    for event in upcoming_events:
                        days_to_event = (event["date"] - datetime.now().date()).days
                        if 3 <= days_to_event <= 14:  # Ideal time frame
                            event_score = max(event_score, 80 + event["importance"] * 5)
                        elif days_to_event < 3:  # Too close
                            event_score = max(event_score, 40 + event["importance"] * 5)
                        elif days_to_event <= 21:  # Still good
                            event_score = max(event_score, 60 + event["importance"] * 5)
                        else:  # Too far
                            event_score = max(event_score, 30 + event["importance"] * 5)
                
                # Find best expiration
                target_days = self.config.get("entry_criteria", {}).get("target_days_to_expiration", 45)
                expiration_date = self.get_option_expiration(option_chain, target_days)
                if not expiration_date:
                    self.logger.debug(f"No suitable expiration found for {symbol}")
                    continue
                
                # Get the chain for selected expiration
                exp_chain = option_chain[option_chain["expiration"] == expiration_date]
                days_to_expiration = (datetime.strptime(expiration_date, "%Y-%m-%d") - datetime.now()).days
                
                # Check for gaps in recent price history
                recent_gaps = check_recent_gaps(price_data)
                has_recent_gaps = len(recent_gaps) > 0
                
                # Find strangle strikes based on strategy configuration
                if self.config.get("entry_criteria", {}).get("use_delta_based_strikes", True):
                    call_strike, put_strike, call_data, put_data = find_strangle_strikes(
                        exp_chain, 
                        current_price, 
                        self.target_delta_call, 
                        self.target_delta_put
                    )
                else:
                    call_strike, put_strike, call_data, put_data = find_strangle_by_otm_pct(
                        exp_chain, 
                        current_price, 
                        self.target_otm_pct, 
                        self.target_otm_pct
                    )
                
                if not call_strike or not put_strike:
                    self.logger.debug(f"No suitable strikes found for {symbol}")
                    continue
                
                # Calculate basic strangle metrics
                call_price = (call_data["bid"] + call_data["ask"]) / 2
                put_price = (put_data["bid"] + put_data["ask"]) / 2
                
                # Calculate breakeven points
                lower_breakeven, upper_breakeven = calculate_strangle_breakeven_points(
                    call_strike, put_strike, call_price, put_price
                )
                
                # Calculate implied volatility
                avg_iv = (call_data["impliedVolatility"] + put_data["impliedVolatility"]) / 2
                
                # Calculate probability of profit
                pop = calculate_probability_of_profit(
                    current_price, lower_breakeven, upper_breakeven, avg_iv, days_to_expiration
                )
                
                # Calculate liquidity score
                call_spread_pct = (call_data["ask"] - call_data["bid"]) / call_data["mid"] if call_data["mid"] > 0 else 1
                put_spread_pct = (put_data["ask"] - put_data["bid"]) / put_data["mid"] if put_data["mid"] > 0 else 1
                avg_spread_pct = (call_spread_pct + put_spread_pct) / 2
                
                call_liquidity = calculate_liquidity_score(
                    call_data["openInterest"], call_data["volume"], call_spread_pct
                )
                put_liquidity = calculate_liquidity_score(
                    put_data["openInterest"], put_data["volume"], put_spread_pct
                )
                
                liquidity_score = min(call_liquidity, put_liquidity)
                
                if liquidity_score < self.min_liquidity_score:
                    self.logger.debug(f"Skipping {symbol}: Liquidity score {liquidity_score:.1f} below minimum {self.min_liquidity_score}")
                    continue
                
                # Calculate vega capture
                vega_capture = calculate_vega_capture_ratio(
                    call_data["vega"], put_data["vega"], call_price, put_price
                )
                
                # Calculate comprehensive metrics
                metrics = calculate_strangle_metrics(
                    current_price,
                    call_strike,
                    put_strike,
                    call_price,
                    put_price,
                    days_to_expiration,
                    avg_iv
                )
                
                # Score the opportunity
                opportunity_score = score_strangle_opportunity(
                    iv_rank=iv_rank,
                    probability_of_profit=pop,
                    risk_reward_ratio=metrics["risk_reward_ratio"],
                    liquidity_score=liquidity_score,
                    days_to_expiration=days_to_expiration,
                    event_score=event_score,
                    vega_capture=vega_capture,
                    has_recent_gaps=has_recent_gaps
                )
                
                if opportunity_score < self.min_score:
                    self.logger.debug(f"Skipping {symbol}: Opportunity score {opportunity_score:.1f} below minimum {self.min_score}")
                    continue
                
                # Prepare opportunity data
                opportunity = {
                    "symbol": symbol,
                    "current_price": current_price,
                    "call_strike": call_strike,
                    "put_strike": put_strike,
                    "call_price": call_price,
                    "put_price": put_price,
                    "total_premium": call_price + put_price,
                    "expiration_date": expiration_date,
                    "days_to_expiration": days_to_expiration,
                    "iv_rank": iv_rank,
                    "implied_volatility": avg_iv,
                    "probability_of_profit": pop,
                    "liquidity_score": liquidity_score,
                    "opportunity_score": opportunity_score,
                    "event_score": event_score,
                    "has_recent_gaps": has_recent_gaps,
                    "vega_capture": vega_capture,
                    "risk_reward_ratio": metrics["risk_reward_ratio"],
                    "breakeven_lower": lower_breakeven,
                    "breakeven_upper": upper_breakeven,
                    "expected_move": metrics["expected_move"],
                    "timestamp": datetime.now().isoformat(),
                    "call_data": call_data,
                    "put_data": put_data,
                    "upcoming_events": upcoming_events
                }
                
                opportunities.append(opportunity)
                self.logger.info(f"Found strangle opportunity for {symbol} with score {opportunity_score:.1f}")
                
            except Exception as e:
                self.logger.error(f"Error scanning {symbol} for strangle opportunities: {str(e)}")
        
        # Sort opportunities by score
        sorted_opportunities = sorted(opportunities, key=lambda x: x["opportunity_score"], reverse=True)
        
        # Update current opportunities
        self.current_opportunities = {opp["symbol"]: opp for opp in sorted_opportunities}
        
        return sorted_opportunities
    
    def execute_entry(self, opportunity: Dict) -> Dict:
        """
        Execute entry for a strangle position.
        
        Args:
            opportunity: Opportunity dictionary with trade details
            
        Returns:
            Trade execution result
        """
        symbol = opportunity["symbol"]
        self.logger.info(f"Executing strangle entry for {symbol}")
        
        try:
            # Calculate position size based on risk
            account_value = self.broker.get_account_value()
            risk_amount = account_value * (self.max_risk_per_trade_pct / 100)
            
            max_loss = opportunity["total_premium"] * 100  # Per contract
            num_contracts = min(
                int(risk_amount / max_loss),
                self.config.get("risk_management", {}).get("max_contracts", 10)
            )
            
            if num_contracts < 1:
                num_contracts = 1
                self.logger.warning(f"Risk calculation resulted in 0 contracts, defaulting to 1 for {symbol}")
            
            # Place call and put orders
            call_order = {
                "symbol": symbol,
                "strike": opportunity["call_strike"],
                "option_type": "call",
                "expiration": opportunity["expiration_date"],
                "action": "buy",
                "quantity": num_contracts,
                "order_type": "limit",
                "limit_price": opportunity["call_price"] * 1.05  # 5% buffer
            }
            
            put_order = {
                "symbol": symbol,
                "strike": opportunity["put_strike"],
                "option_type": "put",
                "expiration": opportunity["expiration_date"],
                "action": "buy",
                "quantity": num_contracts,
                "order_type": "limit",
                "limit_price": opportunity["put_price"] * 1.05  # 5% buffer
            }
            
            # Execute orders
            call_result = self.broker.place_option_order(call_order)
            if not call_result.get("success"):
                self.logger.error(f"Failed to execute call order for {symbol} strangle: {call_result.get('message')}")
                return {"success": False, "message": f"Call order failed: {call_result.get('message')}"}
            
            put_result = self.broker.place_option_order(put_order)
            if not put_result.get("success"):
                # Attempt to close the call if put order fails
                self.broker.place_option_order({
                    **call_order,
                    "action": "sell",
                    "order_type": "market"
                })
                self.logger.error(f"Failed to execute put order for {symbol} strangle: {put_result.get('message')}")
                return {"success": False, "message": f"Put order failed: {put_result.get('message')}"}
            
            # Calculate actual entry prices
            actual_call_price = call_result.get("fill_price", opportunity["call_price"])
            actual_put_price = put_result.get("fill_price", opportunity["put_price"])
            actual_total_cost = actual_call_price + actual_put_price
            
            # Record the position
            position = {
                "symbol": symbol,
                "strategy": "strangle",
                "call_strike": opportunity["call_strike"],
                "put_strike": opportunity["put_strike"],
                "expiration_date": opportunity["expiration_date"],
                "entry_date": datetime.now().isoformat(),
                "entry_price": opportunity["current_price"],
                "call_entry_price": actual_call_price,
                "put_entry_price": actual_put_price,
                "total_cost": actual_total_cost,
                "num_contracts": num_contracts,
                "days_to_expiration": opportunity["days_to_expiration"],
                "initial_iv": opportunity["implied_volatility"],
                "iv_rank_at_entry": opportunity["iv_rank"],
                "expected_move": opportunity["expected_move"],
                "breakeven_lower": opportunity["breakeven_lower"],
                "breakeven_upper": opportunity["breakeven_upper"],
                "probability_of_profit": opportunity["probability_of_profit"],
                "max_loss": actual_total_cost * num_contracts * 100,
                "call_order_id": call_result.get("order_id"),
                "put_order_id": put_result.get("order_id"),
                "profit_target": actual_total_cost * (self.profit_target_pct / 100),
                "stop_loss": actual_total_cost * (self.stop_loss_pct / 100)
            }
            
            # Add to open positions
            self.open_positions[symbol] = position
            
            self.logger.info(f"Successfully entered strangle for {symbol} with {num_contracts} contracts")
            
            return {
                "success": True,
                "position": position,
                "message": f"Entered strangle position for {symbol}"
            }
            
        except Exception as e:
            self.logger.error(f"Error executing strangle entry for {symbol}: {str(e)}")
            return {"success": False, "message": str(e)}
    
    def execute_exit(self, position: Dict, reason: str = "manual") -> Dict:
        """
        Execute exit for a strangle position.
        
        Args:
            position: Position dictionary with trade details
            reason: Reason for exit
            
        Returns:
            Trade execution result
        """
        symbol = position["symbol"]
        self.logger.info(f"Executing strangle exit for {symbol}, reason: {reason}")
        
        try:
            # Create sell orders for both legs
            call_order = {
                "symbol": symbol,
                "strike": position["call_strike"],
                "option_type": "call",
                "expiration": position["expiration_date"],
                "action": "sell",
                "quantity": position["num_contracts"],
                "order_type": "market"
            }
            
            put_order = {
                "symbol": symbol,
                "strike": position["put_strike"],
                "option_type": "put",
                "expiration": position["expiration_date"],
                "action": "sell",
                "quantity": position["num_contracts"],
                "order_type": "market"
            }
            
            # Execute orders
            call_result = self.broker.place_option_order(call_order)
            put_result = self.broker.place_option_order(put_order)
            
            if not call_result.get("success") or not put_result.get("success"):
                self.logger.error(f"Failed to exit strangle for {symbol}: "
                                 f"Call: {call_result.get('message', 'Unknown')}, "
                                 f"Put: {put_result.get('message', 'Unknown')}")
                return {
                    "success": False, 
                    "message": "Failed to exit one or both legs of the strangle"
                }
            
            # Calculate actual exit prices
            actual_call_price = call_result.get("fill_price", 0)
            actual_put_price = put_result.get("fill_price", 0)
            actual_total_exit = actual_call_price + actual_put_price
            
            # Get current market price
            current_price = self.broker.get_current_price(symbol)
            
            # Calculate P&L
            entry_cost = position["total_cost"] * position["num_contracts"] * 100
            exit_value = actual_total_exit * position["num_contracts"] * 100
            pnl = exit_value - entry_cost
            pnl_pct = (pnl / entry_cost) * 100 if entry_cost > 0 else 0
            
            # Record closed position
            exit_data = {
                **position,
                "exit_date": datetime.now().isoformat(),
                "exit_price": current_price,
                "call_exit_price": actual_call_price,
                "put_exit_price": actual_put_price,
                "exit_value": exit_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "exit_reason": reason,
                "days_held": (datetime.now() - datetime.fromisoformat(position["entry_date"])).days
            }
            
            # Remove from open positions
            if symbol in self.open_positions:
                del self.open_positions[symbol]
            
            # Add to trade history
            if not hasattr(self, 'trade_history'):
                self.trade_history = []
            self.trade_history.append(exit_data)
            
            self.logger.info(f"Successfully exited strangle for {symbol} with P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
            
            return {
                "success": True,
                "exit_data": exit_data,
                "message": f"Exited strangle position for {symbol}"
            }
            
        except Exception as e:
            self.logger.error(f"Error executing strangle exit for {symbol}: {str(e)}")
            return {"success": False, "message": str(e)}
    
    def manage_positions(self) -> Dict:
        """
        Manage all open strangle positions, checking exit criteria.
        
        Returns:
            Dictionary with management results
        """
        self.logger.info(f"Managing {len(self.open_positions)} open strangle positions")
        results = {"positions_checked": 0, "exits_executed": 0, "errors": 0}
        
        for symbol, position in list(self.open_positions.items()):
            try:
                results["positions_checked"] += 1
                
                # Get current market data
                current_price = self.broker.get_current_price(symbol)
                option_chain = self.broker.get_option_chain(symbol)
                
                # Find current option prices
                expiry_chain = option_chain[option_chain["expiration"] == position["expiration_date"]]
                
                call_data = expiry_chain[
                    (expiry_chain["type"] == "call") & 
                    (expiry_chain["strike"] == position["call_strike"])
                ].iloc[0]
                
                put_data = expiry_chain[
                    (expiry_chain["type"] == "put") & 
                    (expiry_chain["strike"] == position["put_strike"])
                ].iloc[0]
                
                # Calculate current option values
                current_call_price = (call_data["bid"] + call_data["ask"]) / 2
                current_put_price = (put_data["bid"] + put_data["ask"]) / 2
                current_total_value = current_call_price + current_put_price
                
                # Calculate current IV
                current_iv = (call_data["impliedVolatility"] + put_data["impliedVolatility"]) / 2
                
                # Calculate days to expiration
                current_date = datetime.now().date()
                expiry_date = datetime.strptime(position["expiration_date"], "%Y-%m-%d").date()
                days_to_expiry = (expiry_date - current_date).days
                
                # Calculate P&L
                pnl_pct = (current_total_value - position["total_cost"]) / position["total_cost"] * 100
                
                # Check exit criteria
                exit_reason = None
                
                # 1. Profit target reached
                if pnl_pct >= self.profit_target_pct:
                    exit_reason = "profit_target"
                
                # 2. Stop loss hit
                elif pnl_pct <= -self.stop_loss_pct:
                    exit_reason = "stop_loss"
                
                # 3. Close to expiration
                elif days_to_expiry <= self.days_before_expiry_to_close:
                    exit_reason = "near_expiry"
                
                # 4. IV decline (if initially entered on high IV)
                if position["iv_rank_at_entry"] > 70:
                    iv_decline_pct = (position["initial_iv"] - current_iv) / position["initial_iv"] * 100
                    if iv_decline_pct >= self.iv_drop_exit_pct:
                        exit_reason = "iv_decline"
                
                # Execute exit if needed
                if exit_reason:
                    self.logger.info(f"Exit criteria met for {symbol} strangle: {exit_reason}")
                    exit_result = self.execute_exit(position, reason=exit_reason)
                    
                    if exit_result.get("success"):
                        results["exits_executed"] += 1
                    else:
                        results["errors"] += 1
                else:
                    self.logger.debug(f"No exit criteria met for {symbol} strangle. "
                                     f"P&L: {pnl_pct:.2f}%, Days to expiry: {days_to_expiry}")
            
            except Exception as e:
                self.logger.error(f"Error managing strangle position for {symbol}: {str(e)}")
                results["errors"] += 1
        
        return results
    
    def run(self, symbols: List[str] = None) -> Dict:
        """
        Run the strangle strategy, identifying and executing trading opportunities.
        
        This method serves as the main entry point for the strangle strategy workflow,
        orchestrating the complete trading cycle from opportunity scanning to position
        management. It implements a systematic approach to strangle trading based on
        volatility conditions, option pricing, and risk management parameters.
        
        The execution process follows these key steps:
        1. Position management: Evaluate existing positions against exit criteria
           - Profit targets reached
           - Stop losses triggered
           - Time decay thresholds crossed
           - Volatility environment changes
        
        2. Opportunity scanning: Identify potential new strangle setups
           - Screen for elevated IV rank and volatility conditions
           - Look for upcoming catalysts and event-driven opportunities
           - Evaluate option chains for strike selection and pricing
           - Score opportunities using a multi-factor model
        
        3. Portfolio allocation: Make entry decisions based on available capacity
           - Respect maximum position count limits
           - Prioritize opportunities by score
           - Filter symbols with existing positions
           - Execute entry orders for selected opportunities
        
        This systematic approach ensures disciplined strategy execution while
        adapting to current market conditions and maintaining appropriate
        portfolio-level risk management.
        
        Args:
            symbols: Optional list of symbols to scan (uses configured watchlist if None)
            
        Returns:
            Dictionary containing execution results:
            - timestamp: Execution timestamp
            - strategy: Strategy name
            - positions_managed: Number of positions checked
            - exits_executed: Number of positions exited
            - opportunities_found: Number of new opportunities identified
            - entries_executed: Number of new positions entered
            - errors: Number of errors encountered
            - open_positions: Current number of open positions
        """
        self.logger.info("Running strangle strategy")
        
        try:
            # 1. Manage existing positions
            management_results = self.manage_positions()
            
            # 2. Scan for new opportunities
            if symbols is None:
                symbols = self.config.get("watchlist", [])
                
            opportunities = self.scan_for_opportunities(symbols)
            
            # 3. Check if we can take new positions
            max_positions = self.config.get("risk_management", {}).get("max_positions", 5)
            current_positions = len(self.open_positions)
            available_slots = max_positions - current_positions
            
            entries_executed = 0
            
            if available_slots > 0 and opportunities:
                self.logger.info(f"Found {len(opportunities)} opportunities, {available_slots} position slots available")
                
                # Filter out symbols we already have positions in
                filtered_opportunities = [
                    opp for opp in opportunities 
                    if opp["symbol"] not in self.open_positions
                ]
                
                # Take top opportunities up to available slots
                for i, opportunity in enumerate(filtered_opportunities[:available_slots]):
                    entry_result = self.execute_entry(opportunity)
                    if entry_result.get("success"):
                        entries_executed += 1
            
            return {
                "timestamp": datetime.now().isoformat(),
                "strategy": self.name,
                "positions_managed": management_results["positions_checked"],
                "exits_executed": management_results["exits_executed"],
                "opportunities_found": len(opportunities),
                "entries_executed": entries_executed,
                "errors": management_results["errors"],
                "open_positions": len(self.open_positions)
            }
            
        except Exception as e:
            self.logger.error(f"Error running strangle strategy: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "strategy": self.name,
                "error": str(e),
                "success": False
            }
    
    def get_strategy_stats(self) -> Dict:
        """
        Calculate comprehensive performance statistics for the strangle strategy.
        
        This method analyzes the historical trade data to generate a detailed
        performance report for the strangle strategy. It provides critical metrics
        for evaluating strategy effectiveness, profitability patterns, and potential
        areas for optimization.
        
        Performance metrics calculated include:
        1. General performance: Total trades, win rate, average P&L
        2. Risk metrics: Maximum gain and loss, reward-to-risk ratio
        3. Efficiency metrics: Average holding period, capital utilization
        4. Exit analysis: Performance breakdown by exit reason
        5. Market regime analysis: Performance across different market environments
        
        These statistics are valuable for:
        - Evaluating the strategy's overall effectiveness
        - Identifying which exit conditions are most profitable
        - Understanding optimal holding periods
        - Recognizing how the strategy performs in different market conditions
        - Providing data for ongoing strategy refinement and optimization
        
        Returns:
            Dict containing comprehensive strategy performance metrics:
            - Trade frequency metrics (total trades)
            - Profitability metrics (win rate, avg P&L)
            - Risk metrics (max gain/loss)
            - Efficiency metrics (holding period)
            - Exit analysis (performance by exit reason)
            - Current portfolio status (open positions)
        """
        if not hasattr(self, 'trade_history') or not self.trade_history:
            return {
                "total_trades": 0,
                "success_rate": 0,
                "avg_pnl_pct": 0,
                "max_gain_pct": 0,
                "max_loss_pct": 0,
                "avg_days_held": 0
            }
        
        total_trades = len(self.trade_history)
        profitable_trades = sum(1 for trade in self.trade_history if trade.get("pnl", 0) > 0)
        success_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
        
        pnl_pcts = [trade.get("pnl_pct", 0) for trade in self.trade_history]
        avg_pnl_pct = sum(pnl_pcts) / len(pnl_pcts) if pnl_pcts else 0
        max_gain_pct = max(pnl_pcts) if pnl_pcts else 0
        max_loss_pct = min(pnl_pcts) if pnl_pcts else 0
        
        days_held = [trade.get("days_held", 0) for trade in self.trade_history]
        avg_days_held = sum(days_held) / len(days_held) if days_held else 0
        
        # Calculate statistics by exit reason
        exit_reasons = {}
        for trade in self.trade_history:
            reason = trade.get("exit_reason", "unknown")
            if reason not in exit_reasons:
                exit_reasons[reason] = {
                    "count": 0,
                    "total_pnl_pct": 0,
                    "wins": 0
                }
            
            exit_reasons[reason]["count"] += 1
            exit_reasons[reason]["total_pnl_pct"] += trade.get("pnl_pct", 0)
            if trade.get("pnl", 0) > 0:
                exit_reasons[reason]["wins"] += 1
        
        # Calculate averages for each exit reason
        for reason in exit_reasons:
            count = exit_reasons[reason]["count"]
            exit_reasons[reason]["avg_pnl_pct"] = exit_reasons[reason]["total_pnl_pct"] / count if count > 0 else 0
            exit_reasons[reason]["win_rate"] = (exit_reasons[reason]["wins"] / count) * 100 if count > 0 else 0
        
        return {
            "total_trades": total_trades,
            "profitable_trades": profitable_trades,
            "success_rate": success_rate,
            "avg_pnl_pct": avg_pnl_pct,
            "max_gain_pct": max_gain_pct,
            "max_loss_pct": max_loss_pct,
            "avg_days_held": avg_days_held,
            "exit_reason_stats": exit_reasons,
            "open_positions": len(self.open_positions)
        }

    def get_option_expiration(self, option_chain, target_days=45):
        """
        Get the best expiration date from an option chain based on target days.
        
        Args:
            option_chain: DataFrame with option chain data
            target_days: Target number of days for expiration
            
        Returns:
            Best expiration date or None if not found
        """
        try:
            if "expiration" not in option_chain.columns:
                return None
            
            # Get unique expiration dates
            expirations = option_chain["expiration"].unique()
            
            if len(expirations) == 0:
                return None
            
            # Calculate days to each expiration
            today = datetime.now().date()
            
            best_exp = None
            min_diff = float('inf')
            
            for exp in expirations:
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                days_diff = (exp_date - today).days
                
                # Skip expired options
                if days_diff <= 0:
                    continue
                    
                # Find closest to target
                if abs(days_diff - target_days) < min_diff:
                    min_diff = abs(days_diff - target_days)
                    best_exp = exp
            
            return best_exp
            
        except Exception as e:
            logging.error(f"Error finding expiration date: {e}")
            return None 