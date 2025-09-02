#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collar Strategy Module

This module implements a collar strategy that locks in downside protection while capping upside,
by pairing long stock with a protective put and financing it by selling a covered call.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta

from trading_bot.strategies.strategy_template import (
    StrategyTemplate, 
    StrategyOptimizable,
    Signal, 
    SignalType,
    TimeFrame,
    MarketRegime
)

# Setup logging
logger = logging.getLogger(__name__)

class CollarStrategy(StrategyOptimizable):
    """
    Collar Strategy designed to lock in downside protection while capping upside.
    
    This strategy pairs a long stock position with a protective put and finances it
    by selling a covered call, creating a "collar" that limits both potential losses
    and potential gains.
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Collar strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters based on the strategy blueprint
        default_params = {
            # Section 3: Selection Criteria for Underlying
            "trend_filter": {
                "use_sma": True,
                "sma_period": 50,  # 50-day SMA for trend filter
                "min_slope": 0     # Minimum slope for gentle uptrend
            },
            "volatility_regime": {
                "iv_rank_min": 30,  # Minimum IV rank percentile
                "iv_rank_max": 60,  # Maximum IV rank percentile
            },
            "liquidity": {
                "min_adv": 500000,           # Minimum average daily volume
                "min_option_open_interest": 1000,  # Minimum OI for both puts and calls
                "max_bid_ask_spread_pct": 0.15     # Maximum bid-ask spread as percent
            },
            
            # Section 4: Put Strike Selection
            "put_leg": {
                "otm_buffer_pct": 7.0,    # Default OTM buffer percentage (5-8%)
                "delta_target": 0.25,     # Target delta (0.20-0.30 range)
                "iv_adjustment_factor": 0.2  # Additional OTM % per 10% IV rank above 50
            },
            
            # Section 5: Call Strike Selection
            "call_leg": {
                "otm_buffer_pct": 7.0,    # Default OTM buffer percentage (5-10%)
                "delta_target": 0.20,     # Target delta (0.15-0.25 range)
                "min_upside_pct": 5.0     # Minimum upside potential before cap
            },
            
            # Section 6: Expiration Selection
            "expiration": {
                "dte_range": (30, 45),    # Target days to expiration range
                "min_dte": 10,            # Avoid options with less than this DTE
                "weekly_adjustment": False  # Whether to use weeklies for adjustments
            },
            
            # Section 8: Exit & Adjustment Rules
            "exit_rules": {
                "time_exit_dte": 7,       # Close options legs within this many days to expiry
                "early_unwind_threshold": 50.0,  # % of initial net debit to close early
                "roll_when_tested": True   # Roll when stock approaches strikes
            },
            
            # Section 9: Position Sizing & Risk Controls
            "position_sizing": {
                "max_allocation_pct": 5.0,  # Maximum position size as % of portfolio
                "max_positions": 4,         # Maximum concurrent collar positions
                "margin_buffer_pct": 20.0   # Additional cash as % of position for potential assignment
            }
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        # Initialize strategy-specific attributes
        self.active_collars = {}  # Track active collar positions
        self.total_allocation = 0.0  # Track total allocation to collar positions
        
        logger.info(f"Initialized Collar strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            # Selection criteria parameters
            "trend_filter.sma_period": [20, 50, 100, 200],
            "volatility_regime.iv_rank_min": [20, 30, 40],
            "volatility_regime.iv_rank_max": [50, 60, 70],
            
            # Put strike parameters
            "put_leg.otm_buffer_pct": [5.0, 6.0, 7.0, 8.0],
            "put_leg.delta_target": [0.20, 0.25, 0.30],
            
            # Call strike parameters
            "call_leg.otm_buffer_pct": [5.0, 7.0, 10.0],
            "call_leg.delta_target": [0.15, 0.20, 0.25],
            
            # Expiration parameters
            "expiration.dte_range": [(20, 30), (30, 45), (40, 60)],
            "expiration.min_dte": [7, 10, 14],
            
            # Exit rule parameters
            "exit_rules.time_exit_dte": [5, 7, 10],
            "exit_rules.early_unwind_threshold": [40.0, 50.0, 60.0],
            
            # Position sizing parameters
            "position_sizing.max_allocation_pct": [3.0, 5.0, 7.0],
            "position_sizing.max_positions": [3, 4, 5]
        }
    
    # Section 1: Strategy Philosophy
    def _strategy_philosophy(self) -> str:
        """
        Return the strategy philosophy and purpose.
        
        Returns:
            Description of strategy philosophy
        """
        # TODO: Document the downside protection approach
        # TODO: Explain the upside cap and its benefits
        # TODO: Describe the concept of a low-cost hedge and volatility reduction
        
        return """
        This strategy locks in downside protection while capping upside, by pairing long stock with
        a protective put and financing it by selling a covered call. It aims for a low-cost hedge
        that reduces volatility drag while providing defined risk parameters.
        """
    
    # Section 2: Underlying & Option Universe & Timeframe
    def _define_universe(self) -> Dict[str, Any]:
        """
        Define the universe of underlyings and options to trade.
        
        Returns:
            Dictionary of universe specifications
        """
        # TODO: Implement screening for liquid large-caps or ETFs
        # TODO: Define monthly option cycle selection criteria
        # TODO: Specify weekly option use for tactical adjustments
        
        universe = {
            "underlyings": {
                "type": "large_caps_etfs",
                "min_market_cap": 10e9,  # $10B minimum market cap
                "sectors": ["Technology", "Healthcare", "Consumer", "Financials", "Industrials", "ETFs"],
                "examples": ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN"]
            },
            "options": {
                "primary_cycle": "monthly",
                "tactical_cycle": "weekly",
                "contract_types": ["put", "call"]
            },
            "holding_period": {
                "typical_days": (21, 42),  # 3-6 weeks
                "max_days": 60
            }
        }
        
        return universe
    
    # Section 3: Selection Criteria for Underlying
    def _check_selection_criteria(
        self, 
        underlying_data: pd.DataFrame,
        option_chains: Dict[str, pd.DataFrame],
        market_data: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Check if an underlying meets the selection criteria.
        
        Args:
            underlying_data: OHLCV data for the underlying
            option_chains: Put and call option chain data
            market_data: Additional market data dictionary
            
        Returns:
            Dictionary of criteria check results
        """
        # TODO: Implement trend filter (50-day SMA slope ≥ 0 or near support)
        # TODO: Check IV rank is in moderate range (30-60%)
        # TODO: Verify liquidity requirements for stock and both option types
        
        results = {
            "trend_filter_passed": False,
            "iv_regime_passed": False,
            "liquidity_passed": False,
            "all_criteria_passed": False
        }
        
        return results
    
    # Section 4: Put Strike Selection (Protective Leg)
    def _select_optimal_put_strike(
        self, 
        underlying_price: float,
        put_chain: pd.DataFrame,
        iv_rank: float
    ) -> Dict[str, Any]:
        """
        Select the optimal put strike price for the protective leg.
        
        Args:
            underlying_price: Current price of the underlying
            put_chain: Put option chain data
            iv_rank: Current IV rank (0-100)
            
        Returns:
            Dictionary with selected put strike information
        """
        # TODO: Implement OTM buffer calculation (5-8% below spot)
        # TODO: Find options closest to delta target (0.20-0.30)
        # TODO: Adjust strike based on IV regime
        
        # Base OTM buffer percentage
        base_otm_pct = self.parameters["put_leg"]["otm_buffer_pct"]
        
        # Adjust buffer based on IV regime
        iv_adjustment = 0.0
        if iv_rank > 50:
            # In elevated IV, move puts slightly deeper OTM
            iv_adjustment = self.parameters["put_leg"]["iv_adjustment_factor"] * (iv_rank - 50) / 10
        
        total_otm_pct = base_otm_pct + iv_adjustment
        
        # Calculate target strike based on buffer
        target_strike = underlying_price * (1 - total_otm_pct / 100)
        
        # Placeholder for actual strike selection logic
        put_strike_info = {
            "strike": target_strike,
            "delta": self.parameters["put_leg"]["delta_target"],
            "otm_pct": total_otm_pct,
            "iv": 0.0,  # To be filled based on actual option data
            "premium": 0.0  # To be filled based on actual option data
        }
        
        return put_strike_info
    
    # Section 5: Call Strike Selection (Financing Leg)
    def _select_optimal_call_strike(
        self, 
        underlying_price: float,
        call_chain: pd.DataFrame,
        put_premium: float,
        iv_rank: float
    ) -> Dict[str, Any]:
        """
        Select the optimal call strike price for the financing leg.
        
        Args:
            underlying_price: Current price of the underlying
            call_chain: Call option chain data
            put_premium: Premium of the selected put option
            iv_rank: Current IV rank (0-100)
            
        Returns:
            Dictionary with selected call strike information
        """
        # TODO: Implement OTM buffer calculation (5-10% above spot)
        # TODO: Find options closest to delta target (0.15-0.25)
        # TODO: Ensure minimum upside potential
        # TODO: Aim for call premium to offset put cost (minimize net debit)
        
        # Base OTM buffer percentage
        base_otm_pct = self.parameters["call_leg"]["otm_buffer_pct"]
        
        # Ensure minimum upside potential
        min_upside = self.parameters["call_leg"]["min_upside_pct"]
        otm_pct = max(base_otm_pct, min_upside)
        
        # Calculate target strike based on buffer
        target_strike = underlying_price * (1 + otm_pct / 100)
        
        # Placeholder for actual strike selection logic
        call_strike_info = {
            "strike": target_strike,
            "delta": self.parameters["call_leg"]["delta_target"],
            "otm_pct": otm_pct,
            "iv": 0.0,  # To be filled based on actual option data
            "premium": 0.0  # To be filled based on actual option data
        }
        
        return call_strike_info
    
    # Section 6: Expiration Selection
    def _select_optimal_expiration(
        self, 
        option_chains: Dict[str, pd.DataFrame],
        iv_rank: float
    ) -> Dict[str, Any]:
        """
        Select the optimal expiration date for both option legs.
        
        Args:
            option_chains: Dictionary of put and call option chain data
            iv_rank: Current IV rank (0-100)
            
        Returns:
            Dictionary with selected expiration information
        """
        # TODO: Find expirations within optimal DTE range (30-45 days)
        # TODO: Avoid options with < 10 DTE
        # TODO: Consider weekly options for tactical adjustments if needed
        
        min_dte, max_dte = self.parameters["expiration"]["dte_range"]
        
        expiration_info = {
            "expiration_date": None,  # To be filled based on actual option data
            "dte": 0,  # To be filled based on actual option data
            "cycle": "monthly",  # Default to monthly cycle
            "available_weeklys": []  # List of available weekly expirations if needed
        }
        
        return expiration_info
    
    # Section 7: Entry Execution
    def _prepare_entry_orders(
        self, 
        underlying_symbol: str,
        current_price: float,
        put_strike_info: Dict[str, Any],
        call_strike_info: Dict[str, Any],
        expiration_info: Dict[str, Any],
        account_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare the entry orders for all three legs of the collar.
        
        Args:
            underlying_symbol: Symbol of the underlying
            current_price: Current price of the underlying
            put_strike_info: Put strike information
            call_strike_info: Call strike information
            expiration_info: Expiration information
            account_data: Account information
            
        Returns:
            Dictionary with order details for all legs
        """
        # TODO: Calculate position size based on risk parameters
        # TODO: Prepare orders for stock leg (if not already holding)
        # TODO: Prepare orders for put and call legs
        # TODO: Create multi-leg order if supported
        
        equity = account_data["equity"]
        cash_available = account_data["cash_available"]
        existing_shares = account_data.get("existing_positions", {}).get(underlying_symbol, {}).get("shares", 0)
        
        # Calculate max allocation
        max_allocation = equity * (self.parameters["position_sizing"]["max_allocation_pct"] / 100)
        
        # Calculate number of shares
        shares_to_buy = 0
        if existing_shares > 0:
            # Already own shares, use them for the collar
            shares_to_use = existing_shares
        else:
            # Need to buy shares
            shares_to_buy = int(max_allocation / current_price)
            shares_to_use = shares_to_buy
        
        # Ensure we have enough for round lots (100 shares per contract)
        contracts = shares_to_use // 100
        shares_to_use = contracts * 100
        
        if shares_to_buy > 0:
            shares_to_buy = shares_to_use
        
        # Calculate cost of put leg
        put_cost = put_strike_info["premium"] * contracts * 100
        
        # Calculate premium from call leg
        call_premium = call_strike_info["premium"] * contracts * 100
        
        # Calculate net debit/credit
        net_debit = put_cost - call_premium
        
        # Calculate total cash needed
        total_cash_needed = (current_price * shares_to_buy) + net_debit
        
        # Add margin buffer for potential assignment on the short call
        margin_buffer = (current_price * shares_to_use) * (self.parameters["position_sizing"]["margin_buffer_pct"] / 100)
        
        # Check if we have enough cash
        if total_cash_needed + margin_buffer > cash_available:
            # Adjust position size down
            potential_contracts = max(1, int((cash_available - margin_buffer) / ((current_price * 100) + net_debit)))
            contracts = min(contracts, potential_contracts)
            shares_to_use = contracts * 100
            shares_to_buy = shares_to_use if shares_to_buy > 0 else 0
        
        # Prepare orders
        orders = {
            "stock_leg": {
                "action": "BUY" if shares_to_buy > 0 else None,
                "symbol": underlying_symbol,
                "shares": shares_to_buy,
                "order_type": "LIMIT" if shares_to_buy > 0 else None,
                "limit_price": current_price if shares_to_buy > 0 else None,
            },
            "put_leg": {
                "action": "BUY",
                "symbol": underlying_symbol,
                "option_type": "PUT",
                "strike": put_strike_info["strike"],
                "expiration": expiration_info["expiration_date"],
                "contracts": contracts,
                "order_type": "LIMIT",
                "limit_price": put_strike_info["premium"],
            },
            "call_leg": {
                "action": "SELL",
                "symbol": underlying_symbol,
                "option_type": "CALL",
                "strike": call_strike_info["strike"],
                "expiration": expiration_info["expiration_date"],
                "contracts": contracts,
                "order_type": "LIMIT",
                "limit_price": call_strike_info["premium"],
            },
            "combined": {
                "type": "collar",
                "shares_used": shares_to_use,
                "shares_to_buy": shares_to_buy,
                "contracts": contracts,
                "put_cost": put_cost,
                "call_premium": call_premium,
                "net_debit": net_debit,
                "total_cash_needed": total_cash_needed,
                "downside_protection_pct": (current_price - put_strike_info["strike"]) / current_price * 100,
                "upside_potential_pct": (call_strike_info["strike"] - current_price) / current_price * 100
            }
        }
        
        return orders
    
    # Section 8: Exit & Adjustment Rules
    def _check_exit_conditions(
        self, 
        position: Dict[str, Any],
        current_stock_price: float,
        current_put_price: float,
        current_call_price: float,
        dte: int
    ) -> Dict[str, Any]:
        """
        Check if exit conditions are met for a collar position.
        
        Args:
            position: Current position details
            current_stock_price: Current stock price
            current_put_price: Current put option price
            current_call_price: Current call option price
            dte: Days to expiration
            
        Returns:
            Dictionary with exit decision and reason
        """
        # TODO: Check if stock is approaching call strike for potential roll
        # TODO: Check if stock is approaching put strike for potential roll
        # TODO: Check if time-based exit should be triggered
        # TODO: Check if early unwind threshold is met
        
        exit_decision = {
            "exit_collar": False,
            "exit_reason": None,
            "roll_put": False,
            "roll_call": False,
            "partial_exit": False
        }
        
        # Calculate current value of the collar
        entry_stock_price = position["entry_price"]
        entry_put_price = position["put_entry_price"]
        entry_call_price = position["call_entry_price"]
        
        put_strike = position["put_strike"]
        call_strike = position["call_strike"]
        
        initial_net_debit = entry_put_price - entry_call_price
        current_net_debit = current_put_price - current_call_price
        
        # Check time-based exit (approaching expiration)
        if dte <= self.parameters["exit_rules"]["time_exit_dte"]:
            exit_decision["exit_collar"] = True
            exit_decision["exit_reason"] = "time_exit"
            
            # Check if we should roll to next cycle
            if position.get("needs_roll", True):
                exit_decision["roll_put"] = True
                exit_decision["roll_call"] = True
            
            return exit_decision
        
        # Check early unwind threshold
        if initial_net_debit > 0:  # If we paid a debit initially
            debit_reduction_pct = (initial_net_debit - current_net_debit) / initial_net_debit * 100
            if debit_reduction_pct >= self.parameters["exit_rules"]["early_unwind_threshold"]:
                exit_decision["exit_collar"] = True
                exit_decision["exit_reason"] = "profit_target"
                return exit_decision
        
        # Check if stock is testing strikes
        if self.parameters["exit_rules"]["roll_when_tested"]:
            # Calculate proximity to strikes as percentage
            to_call_pct = (call_strike - current_stock_price) / current_stock_price * 100
            to_put_pct = (current_stock_price - put_strike) / current_stock_price * 100
            
            # If within 2% of either strike, consider rolling
            if to_call_pct <= 2.0:
                exit_decision["roll_call"] = True
            
            if to_put_pct <= 2.0:
                exit_decision["roll_put"] = True
        
        return exit_decision
    
    def _prepare_roll_orders(
        self,
        position: Dict[str, Any],
        current_stock_price: float,
        option_chains: Dict[str, pd.DataFrame],
        iv_rank: float,
        roll_put: bool = True,
        roll_call: bool = True
    ) -> Dict[str, Any]:
        """
        Prepare orders to roll a collar to a new expiration.
        
        Args:
            position: Current position details
            current_stock_price: Current stock price
            option_chains: Dictionary of put and call option chain data
            iv_rank: Current IV rank
            roll_put: Whether to roll the put leg
            roll_call: Whether to roll the call leg
            
        Returns:
            Dictionary with roll orders
        """
        # TODO: Select new expiration
        # TODO: Select new strikes based on current price
        # TODO: Create exit and entry orders for the roll
        
        # Select new expiration
        expiration_info = self._select_optimal_expiration(
            option_chains,
            iv_rank
        )
        
        # Select new put strike if rolling put
        put_orders = {}
        if roll_put:
            new_put_strike = self._select_optimal_put_strike(
                current_stock_price,
                option_chains.get("puts", pd.DataFrame()),
                iv_rank
            )
            
            put_orders = {
                "close_current": {
                    "action": "SELL",
                    "symbol": position["symbol"],
                    "option_type": "PUT",
                    "strike": position["put_strike"],
                    "expiration": position["expiration"],
                    "contracts": position["contracts"],
                    "order_type": "LIMIT",
                    # Price to be determined at time of order
                },
                "open_new": {
                    "action": "BUY",
                    "symbol": position["symbol"],
                    "option_type": "PUT",
                    "strike": new_put_strike["strike"],
                    "expiration": expiration_info["expiration_date"],
                    "contracts": position["contracts"],
                    "order_type": "LIMIT",
                    "limit_price": new_put_strike["premium"],
                }
            }
        
        # Select new call strike if rolling call
        call_orders = {}
        if roll_call:
            # Pass 0 as put premium if not rolling put (to avoid influencing call selection)
            put_premium = new_put_strike["premium"] if roll_put else 0
            
            new_call_strike = self._select_optimal_call_strike(
                current_stock_price,
                option_chains.get("calls", pd.DataFrame()),
                put_premium,
                iv_rank
            )
            
            call_orders = {
                "close_current": {
                    "action": "BUY",  # Buy to close the short call
                    "symbol": position["symbol"],
                    "option_type": "CALL",
                    "strike": position["call_strike"],
                    "expiration": position["expiration"],
                    "contracts": position["contracts"],
                    "order_type": "LIMIT",
                    # Price to be determined at time of order
                },
                "open_new": {
                    "action": "SELL",
                    "symbol": position["symbol"],
                    "option_type": "CALL",
                    "strike": new_call_strike["strike"],
                    "expiration": expiration_info["expiration_date"],
                    "contracts": position["contracts"],
                    "order_type": "LIMIT",
                    "limit_price": new_call_strike["premium"],
                }
            }
        
        # Combine roll orders
        roll_orders = {
            "put_leg": put_orders if roll_put else None,
            "call_leg": call_orders if roll_call else None,
            "new_expiration": expiration_info["expiration_date"],
            "new_dte": expiration_info["dte"],
            "new_put_strike": new_put_strike["strike"] if roll_put else position["put_strike"],
            "new_call_strike": new_call_strike["strike"] if roll_call else position["call_strike"],
            "estimated_net_debit": (new_put_strike["premium"] - new_call_strike["premium"]) * position["contracts"] * 100 if roll_put and roll_call else 0.0
        }
        
        return roll_orders
    
    # Section 9: Position Sizing & Risk Controls
    def _check_portfolio_constraints(
        self, 
        new_position_cost: float,
        account_data: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Check if portfolio constraints allow a new collar position.
        
        Args:
            new_position_cost: Cost of the new position (stock + net option debit)
            account_data: Account information
            
        Returns:
            Dictionary of constraint check results
        """
        # TODO: Check allocation limit per position
        # TODO: Check maximum concurrent collars
        # TODO: Ensure margin buffer for potential assignment
        
        equity = account_data["equity"]
        
        # Extract current positions
        current_positions = account_data.get("collar_positions", [])
        
        # Check constraints
        max_allocation_pct = self.parameters["position_sizing"]["max_allocation_pct"]
        max_positions = self.parameters["position_sizing"]["max_positions"]
        
        constraints = {
            "max_positions_ok": len(current_positions) < max_positions,
            "allocation_ok": new_position_cost <= (equity * max_allocation_pct / 100),
            "all_constraints_ok": False
        }
        
        # Check if all constraints are satisfied
        constraints["all_constraints_ok"] = all([
            constraints["max_positions_ok"],
            constraints["allocation_ok"]
        ])
        
        return constraints
    
    # Section 10: Backtesting & Performance Metrics
    def _calculate_performance_metrics(
        self, 
        trade_history: List[Dict[str, Any]],
        account_history: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Calculate performance metrics for the collar strategy.
        
        Args:
            trade_history: List of completed trades
            account_history: DataFrame with account balance history
            benchmark_data: Optional benchmark data for comparison
            
        Returns:
            Dictionary of performance metrics
        """
        # TODO: Calculate net return after option costs
        # TODO: Compute cost of protection
        # TODO: Calculate assignment frequency and outcome
        # TODO: Determine max drawdown vs. unhedged stock
        # TODO: Calculate win rate of collar cycles
        
        metrics = {
            "net_return_pct": 0.0,
            "annualized_return": 0.0,
            "cost_of_protection_pct": 0.0,
            "assignment_frequency": 0.0,
            "max_drawdown_pct": 0.0,
            "unhedged_max_drawdown_pct": 0.0,  # For comparison
            "drawdown_reduction_pct": 0.0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0
        }
        
        if benchmark_data is not None:
            metrics["excess_return_vs_benchmark"] = 0.0
            metrics["downside_protection_vs_benchmark"] = 0.0
        
        return metrics
    
    def calculate_iv_rank(
        self, 
        current_iv: float,
        historical_iv: pd.Series,
        lookback_days: int = 252
    ) -> float:
        """
        Calculate the IV rank (percentile of current IV relative to history).
        
        Args:
            current_iv: Current implied volatility
            historical_iv: Series of historical IV values
            lookback_days: Number of days to look back
            
        Returns:
            IV rank as a percentage (0-100)
        """
        # Use recent history based on lookback period
        recent_iv = historical_iv.tail(lookback_days)
        
        if len(recent_iv) < 10:  # Need enough data points
            logger.warning("Insufficient IV history for proper IV rank calculation")
            return 50.0  # Default to middle
        
        iv_min = recent_iv.min()
        iv_max = recent_iv.max()
        
        if iv_max == iv_min:  # Avoid division by zero
            return 50.0
        
        iv_rank = (current_iv - iv_min) / (iv_max - iv_min) * 100
        
        return iv_rank
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate indicators needed for the collar strategy.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary of calculated indicators for each symbol
        """
        indicators = {}
        
        for symbol, df in data.items():
            try:
                # Calculate 50-day SMA for trend filter
                sma_period = self.parameters["trend_filter"]["sma_period"]
                sma = df['close'].rolling(window=sma_period).mean()
                
                # Calculate SMA slope
                sma_slope = pd.Series(np.gradient(sma), index=sma.index)
                
                # Trend status (SMA slope > min_slope)
                min_slope = self.parameters["trend_filter"]["min_slope"]
                trend_status = sma_slope > min_slope
                
                # Check if price is near support (within 3% of recent lows)
                near_support = False
                if len(df) > 20:
                    recent_low = df['low'].tail(20).min()
                    near_support = (df['close'].iloc[-1] - recent_low) / df['close'].iloc[-1] <= 0.03
                
                # Placeholder for IV data (would come from options data in real implementation)
                # Here we'll just simulate IV for demonstration
                iv_series = pd.Series(np.random.uniform(0.2, 0.4, len(df)), index=df.index)
                iv_current = iv_series.iloc[-1]
                
                # Calculate IV rank
                iv_rank = self.calculate_iv_rank(iv_current, iv_series)
                
                # Store indicators
                indicators[symbol] = {
                    "sma": sma,
                    "sma_slope": sma_slope,
                    "trend_status": trend_status,
                    "near_support": near_support,
                    "iv_series": iv_series,
                    "iv_current": iv_current,
                    "iv_rank": iv_rank,
                    "above_sma": df['close'].iloc[-1] > sma.iloc[-1],
                    "price_to_sma_ratio": df['close'].iloc[-1] / sma.iloc[-1] if not pd.isna(sma.iloc[-1]) else 1.0
                }
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
        
        return indicators
    
    def generate_signals(
        self, 
        data: Dict[str, pd.DataFrame],
        option_chains: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
        account_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Signal]:
        """
        Generate collar signals based on selection criteria.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            option_chains: Optional dictionary of option chains for each symbol
            account_data: Optional account information
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Calculate indicators
        indicators = self.calculate_indicators(data)
        
        # Initialize signals dictionary
        signals = {}
        
        # Mock option chains and account data if not provided
        if option_chains is None:
            option_chains = self._mock_option_chains(data)
        
        if account_data is None:
            account_data = {
                "equity": 100000.0,
                "cash_available": 80000.0,
                "collar_positions": [],
                "existing_positions": {}
            }
        
        # First check portfolio constraints before generating new signals
        portfolio_constraints = self._check_portfolio_constraints(
            10000.0,  # Assume typical position cost for check
            account_data
        )
        
        if not portfolio_constraints["all_constraints_ok"]:
            logger.info("Portfolio constraints not met, skipping signal generation")
            return signals
        
        # Generate signals
        for symbol, symbol_indicators in indicators.items():
            try:
                # Get latest price
                latest_data = data[symbol].iloc[-1]
                latest_price = latest_data['close']
                latest_timestamp = latest_data.name if isinstance(latest_data.name, datetime) else datetime.now()
                
                # Check selection criteria
                selection_results = self._check_selection_criteria(
                    data[symbol], 
                    option_chains.get(symbol, {"puts": pd.DataFrame(), "calls": pd.DataFrame()}),
                    {"iv_rank": symbol_indicators["iv_rank"]}
                )
                
                # Skip if criteria not met
                if not selection_results.get("all_criteria_passed", False):
                    logger.debug(f"Selection criteria not met for {symbol}: {selection_results}")
                    continue
                
                # Select put strike
                put_strike_info = self._select_optimal_put_strike(
                    latest_price,
                    option_chains.get(symbol, {}).get("puts", pd.DataFrame()),
                    symbol_indicators["iv_rank"]
                )
                
                # Select call strike
                call_strike_info = self._select_optimal_call_strike(
                    latest_price,
                    option_chains.get(symbol, {}).get("calls", pd.DataFrame()),
                    put_strike_info["premium"],
                    symbol_indicators["iv_rank"]
                )
                
                # Select expiration
                expiration_info = self._select_optimal_expiration(
                    option_chains.get(symbol, {"puts": pd.DataFrame(), "calls": pd.DataFrame()}),
                    symbol_indicators["iv_rank"]
                )
                
                # Prepare entry orders
                entry_orders = self._prepare_entry_orders(
                    symbol,
                    latest_price,
                    put_strike_info,
                    call_strike_info,
                    expiration_info,
                    account_data
                )
                
                if entry_orders is None or entry_orders["combined"]["contracts"] == 0:
                    logger.debug(f"Could not prepare valid entry orders for {symbol}")
                    continue
                
                # Generate signal
                signals[symbol] = Signal(
                    symbol=symbol,
                    signal_type=SignalType.COLLAR,  # Custom signal type for collar
                    price=latest_price,
                    timestamp=latest_timestamp,
                    confidence=min(0.5 + symbol_indicators["iv_rank"] / 100, 0.9),  # Higher with better IV env
                    stop_loss=put_strike_info["strike"],  # Put strike serves as effective stop
                    take_profit=call_strike_info["strike"],  # Call strike caps upside
                    metadata={
                        "strategy_type": "collar",
                        "iv_rank": symbol_indicators["iv_rank"],
                        "put_strike": put_strike_info["strike"],
                        "put_premium": put_strike_info["premium"],
                        "call_strike": call_strike_info["strike"],
                        "call_premium": call_strike_info["premium"],
                        "expiration": expiration_info.get("expiration_date"),
                        "dte": expiration_info.get("dte"),
                        "shares": entry_orders["combined"]["shares_to_buy"],
                        "contracts": entry_orders["combined"]["contracts"],
                        "net_debit": entry_orders["combined"]["net_debit"],
                        "downside_protection_pct": entry_orders["combined"]["downside_protection_pct"],
                        "upside_potential_pct": entry_orders["combined"]["upside_potential_pct"],
                        "sma": symbol_indicators["sma"].iloc[-1],
                        "sma_slope": symbol_indicators["sma_slope"].iloc[-1],
                        "above_sma": symbol_indicators["above_sma"]
                    }
                )
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    def _mock_option_chains(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Create mock option chains for testing/demo purposes.
        
        Args:
            data: Dictionary of price data
            
        Returns:
            Dictionary of mock option chains
        """
        option_chains = {}
        
        for symbol, df in data.items():
            latest_price = df['close'].iloc[-1]
            
            # Create mock put chain
            put_strikes = [latest_price * (1 - i/100) for i in range(1, 15)]
            put_expirations = [
                datetime.now() + timedelta(days=days) for days in [7, 14, 21, 30, 45, 60, 90]
            ]
            
            put_rows = []
            for strike in put_strikes:
                for exp in put_expirations:
                    dte = (exp - datetime.now()).days
                    delta = 0.5 - (latest_price - strike) / latest_price
                    iv = 0.3 + (0.1 * np.random.random())
                    bid = max(0.01, latest_price * 0.02 * (delta + 0.2) * (dte/30))
                    ask = bid * 1.1
                    
                    put_rows.append({
                        "strike": strike,
                        "expiration": exp,
                        "dte": dte,
                        "option_type": "put",
                        "bid": bid,
                        "ask": ask,
                        "mid": (bid + ask) / 2,
                        "delta": max(0.01, min(0.99, delta)),
                        "gamma": 0.01,
                        "theta": -0.01,
                        "vega": 0.1,
                        "iv": iv,
                        "volume": int(100 * np.random.random()),
                        "open_interest": int(1000 * np.random.random())
                    })
            
            # Create mock call chain
            call_strikes = [latest_price * (1 + i/100) for i in range(1, 15)]
            call_expirations = put_expirations  # Use same expirations
            
            call_rows = []
            for strike in call_strikes:
                for exp in call_expirations:
                    dte = (exp - datetime.now()).days
                    delta = 0.5 - (strike - latest_price) / latest_price
                    iv = 0.3 + (0.1 * np.random.random())
                    bid = max(0.01, latest_price * 0.02 * (delta + 0.2) * (dte/30))
                    ask = bid * 1.1
                    
                    call_rows.append({
                        "strike": strike,
                        "expiration": exp,
                        "dte": dte,
                        "option_type": "call",
                        "bid": bid,
                        "ask": ask,
                        "mid": (bid + ask) / 2,
                        "delta": max(0.01, min(0.99, delta)),
                        "gamma": 0.01,
                        "theta": -0.01,
                        "vega": 0.1,
                        "iv": iv,
                        "volume": int(100 * np.random.random()),
                        "open_interest": int(1000 * np.random.random())
                    })
            
            # Store both chains
            option_chains[symbol] = {
                "puts": pd.DataFrame(put_rows),
                "calls": pd.DataFrame(call_rows)
            }
        
        return option_chains 