#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Married-Put Strategy Module

This module implements a married-put strategy that protects long equity positions
against sharp downside by pairing a stock purchase with a simultaneous long put,
creating a synthetic insurance policy.
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

class MarriedPutStrategy(StrategyOptimizable):
    """
    Married-Put Strategy designed to protect long equity positions against sharp downside
    by pairing a stock purchase with a simultaneous long put option.
    
    This strategy aims to capture upside potential while capping downside losses, effectively
    creating a synthetic insurance policy for equity positions.
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Married-Put strategy.
        
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
                "higher_highs_lookback": 10  # Days to check for higher highs
            },
            "volatility_regime": {
                "iv_rank_min": 20,  # Minimum IV rank percentile
                "iv_rank_max": 50,  # Maximum IV rank percentile
            },
            "liquidity": {
                "min_adv": 500000,  # Minimum average daily volume
                "min_option_open_interest": 500,  # Minimum put open interest
                "max_bid_ask_spread_pct": 0.15  # Maximum bid-ask spread as percent
            },
            
            # Section 4: Put Strike Selection
            "otm_buffer_pct": 7.0,  # Default OTM buffer percentage (5-8%)
            "delta_target": 0.30,  # Target delta (0.25-0.35 range)
            "iv_regime_adjustment": {
                "high_iv_threshold": 75,  # IV rank percentile for high IV
                "low_iv_threshold": 25,   # IV rank percentile for low IV
                "high_iv_buffer_pct": 8.0,  # Deeper OTM in high IV
                "low_iv_buffer_pct": 5.0    # Less OTM in low IV
            },
            
            # Section 5: Expiration Selection
            "dte_range": (30, 45),  # Target days to expiration range
            "min_dte": 15,  # Avoid options with less than this DTE
            "roll_buffer_adjustment": 2.0,  # Wider buffer when rolling (percentage points)
            
            # Section 7: Exit & Hedge Management
            "profit_target_pct": 12.0,  # Target for stock profit (10-15%)
            "put_take_profit_pct": 50.0,  # Take profit on put at 50% of max premium
            "hedge_unwind_iv_threshold": 20,  # Remove put when IV rank below this
            "partial_unwind_drawdown": 25.0,  # Sell half the put at this drawdown percentage
            
            # Section 8: Position Sizing & Risk Controls
            "equity_allocation_pct": 5.0,  # Maximum percent of portfolio per position
            "hedge_cost_max_pct": 1.0,     # Maximum hedge cost as percentage of portfolio
            "max_positions": 4,            # Maximum concurrent married-put positions
            "drawdown_halt_threshold": 3.0,  # Stop new positions if drawdown exceeds this
            
            # Section 10: Continuous Optimization
            "optimization_frequency_days": 90,  # Re-optimize quarterly
            "dynamic_layering_iv_surge": 25.0,  # Add second put if IV increases by this much
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        # Initialize strategy-specific attributes
        self.active_positions = {}  # Track active married-put positions
        self.hedged_shares = 0      # Track total hedged shares
        self.equity_allocation = 0.0  # Track total equity allocation
        self.hedge_allocation = 0.0   # Track total hedge allocation
        
        logger.info(f"Initialized Married-Put strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            # Selection criteria parameters
            "trend_filter.sma_period": [20, 50, 100, 200],
            "volatility_regime.iv_rank_min": [15, 20, 25, 30],
            "volatility_regime.iv_rank_max": [40, 50, 60, 70],
            
            # Put strike selection parameters
            "otm_buffer_pct": [5.0, 6.0, 7.0, 8.0],
            "delta_target": [0.20, 0.25, 0.30, 0.35, 0.40],
            
            # Expiration selection parameters
            "dte_range": [(20, 35), (30, 45), (40, 60)],
            "min_dte": [10, 15, 20],
            
            # Exit & Hedge management parameters
            "profit_target_pct": [8.0, 10.0, 12.0, 15.0],
            "put_take_profit_pct": [40.0, 50.0, 60.0],
            
            # Position sizing parameters
            "equity_allocation_pct": [3.0, 5.0, 7.0],
            "hedge_cost_max_pct": [0.5, 0.75, 1.0, 1.25]
        }
    
    # Section 1: Strategy Philosophy
    def _strategy_philosophy(self) -> str:
        """
        Return the strategy philosophy and purpose.
        
        Returns:
            Description of strategy philosophy
        """
        # TODO: Document the downside protection approach
        # TODO: Define the synthetic insurance policy concept
        # TODO: Explain the trade-off between protection cost and risk reduction
        
        return """
        This strategy protects long equity positions against sharp downside by pairing a stock 
        purchase with a simultaneous long put ("marrying" the put). It aims to capture upside 
        potential while capping downside losses, effectively creating a synthetic insurance policy.
        """
    
    # Section 2: Underlying & Option Universe & Timeframe
    def _define_universe(self) -> Dict[str, Any]:
        """
        Define the universe of underlyings and options to trade.
        
        Returns:
            Dictionary of universe specifications
        """
        # TODO: Implement screening for quality large-caps or ETFs
        # TODO: Create rules for filtering based on stable fundamentals
        # TODO: Define option chain selection logic
        
        universe = {
            "underlyings": {
                "type": "quality_large_caps_etfs",
                "min_market_cap": 10e9,  # $10B minimum market cap
                "sectors": ["Technology", "Healthcare", "Consumer", "Financials", "Industrials", "ETFs"],
                "examples": ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN"]
            },
            "options": {
                "primary_cycle": "monthly",
                "contract_types": ["put"]
            },
            "holding_period": {
                "typical_days": (14, 56),  # 2-8 weeks
                "max_days": 60
            }
        }
        
        return universe
    
    # Section 3: Selection Criteria for Underlying
    def _check_selection_criteria(
        self, 
        underlying_data: pd.DataFrame,
        option_chain: pd.DataFrame,
        market_data: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Check if an underlying meets the selection criteria.
        
        Args:
            underlying_data: OHLCV data for the underlying
            option_chain: Option chain data
            market_data: Additional market data dictionary
            
        Returns:
            Dictionary of criteria check results
        """
        # TODO: Implement trend filter (above 50-day SMA or higher highs)
        # TODO: Check IV rank is in moderate range (20-50%)
        # TODO: Verify liquidity requirements (volume, OI, spread)
        
        results = {
            "trend_filter_passed": False,
            "iv_regime_passed": False,
            "liquidity_passed": False,
            "all_criteria_passed": False
        }
        
        return results
    
    # Section 4: Put Strike Selection
    def _select_optimal_put_strike(
        self, 
        underlying_price: float,
        option_chain: pd.DataFrame,
        iv_rank: float
    ) -> Dict[str, Any]:
        """
        Select the optimal put strike price for protection.
        
        Args:
            underlying_price: Current price of the underlying
            option_chain: Option chain data
            iv_rank: Current IV rank (0-100)
            
        Returns:
            Dictionary with selected put strike information
        """
        # TODO: Implement OTM buffer calculation (5-8% below spot)
        # TODO: Find options closest to delta target (0.25-0.35)
        # TODO: Adjust strike based on IV regime
        
        # Determine buffer based on IV regime
        iv_regime_adj = self.parameters["iv_regime_adjustment"]
        if iv_rank >= iv_regime_adj["high_iv_threshold"]:
            # High IV - use deeper OTM
            buffer_pct = iv_regime_adj["high_iv_buffer_pct"]
        elif iv_rank <= iv_regime_adj["low_iv_threshold"]:
            # Low IV - use less OTM (tighter protection)
            buffer_pct = iv_regime_adj["low_iv_buffer_pct"]
        else:
            # Normal IV - use default buffer
            buffer_pct = self.parameters["otm_buffer_pct"]
        
        # Calculate target strike based on buffer
        target_strike = underlying_price * (1 - buffer_pct / 100)
        
        # Placeholder for actual strike selection logic
        strike_info = {
            "strike": target_strike,
            "delta": self.parameters["delta_target"],
            "otm_pct": buffer_pct,
            "iv": 0.0,  # To be filled
            "premium": 0.0  # To be filled
        }
        
        return strike_info
    
    # Section 5: Expiration Selection
    def _select_optimal_expiration(
        self, 
        option_chain: pd.DataFrame,
        iv_rank: float,
        is_roll: bool = False
    ) -> Dict[str, Any]:
        """
        Select the optimal expiration date for the put option.
        
        Args:
            option_chain: Option chain data
            iv_rank: Current IV rank (0-100)
            is_roll: Whether this is a roll from existing position
            
        Returns:
            Dictionary with selected expiration information
        """
        # TODO: Find expirations within optimal DTE range (30-45 days)
        # TODO: Avoid options with < 15 DTE
        # TODO: Implement roll logic if needed
        
        min_dte, max_dte = self.parameters["dte_range"]
        
        # If this is a roll and we need to adjust buffer
        buffer_adjustment = self.parameters["roll_buffer_adjustment"] if is_roll else 0.0
        
        expiration_info = {
            "expiration_date": None,  # To be filled
            "dte": 0,  # To be filled
            "buffer_adjustment": buffer_adjustment,
            "cycle": "monthly"  # Assume monthly for now
        }
        
        return expiration_info
    
    # Section 6: Entry Execution
    def _prepare_entry_orders(
        self, 
        underlying_symbol: str,
        current_price: float,
        strike_info: Dict[str, Any],
        expiration_info: Dict[str, Any],
        account_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare the entry orders for the stock and put legs.
        
        Args:
            underlying_symbol: Symbol of the underlying
            current_price: Current price of the underlying
            strike_info: Strike information from put strike selection
            expiration_info: Expiration information
            account_data: Account information
            
        Returns:
            Dictionary with order details for both legs
        """
        # TODO: Calculate position size based on risk parameters
        # TODO: Prepare limit order for stock leg
        # TODO: Prepare limit order for put leg
        # TODO: Create multi-leg order if platform supports it
        
        equity = account_data["equity"]
        cash_available = account_data["cash_available"]
        
        # Calculate max equity allocation
        max_equity_allocation = equity * (self.parameters["equity_allocation_pct"] / 100)
        
        # Calculate number of shares
        max_shares = int(max_equity_allocation / current_price)
        
        # Calculate hedge cost per share
        hedge_cost_per_share = strike_info["premium"]
        total_hedge_cost = hedge_cost_per_share * max_shares
        
        # Check if hedge cost within limit
        max_hedge_allocation = equity * (self.parameters["hedge_cost_max_pct"] / 100)
        if total_hedge_cost > max_hedge_allocation:
            # Adjust shares down to meet hedge cost limit
            max_shares = int(max_hedge_allocation / hedge_cost_per_share)
        
        # Check available cash
        total_cost = (current_price * max_shares) + total_hedge_cost
        if total_cost > cash_available:
            # Adjust shares down to meet cash limit
            max_shares = int(cash_available / (current_price + hedge_cost_per_share))
        
        # Calculate number of contracts (standard is 100 shares per contract)
        num_contracts = max(1, max_shares // 100)
        # Adjust shares to match contracts
        shares = num_contracts * 100
        
        orders = {
            "stock_leg": {
                "action": "BUY",
                "symbol": underlying_symbol,
                "shares": shares,
                "order_type": "LIMIT",
                "limit_price": current_price,  # Potentially adjust for slight dip
                "time_in_force": "DAY"
            },
            "put_leg": {
                "action": "BUY",
                "symbol": underlying_symbol,
                "option_type": "PUT",
                "strike": strike_info["strike"],
                "expiration": expiration_info["expiration_date"],
                "contracts": num_contracts,
                "order_type": "LIMIT",
                "limit_price": strike_info["premium"],
                "time_in_force": "DAY"
            },
            "combined": {
                "type": "combo",
                "legs": ["stock_leg", "put_leg"],
                "total_cost": (current_price * shares) + (strike_info["premium"] * num_contracts * 100),
                "equity_allocation": current_price * shares,
                "hedge_allocation": strike_info["premium"] * num_contracts * 100,
                "hedge_cost_pct": (strike_info["premium"] * 100) / current_price,  # As percentage of stock price
                "protection_buffer_pct": strike_info["otm_pct"]
            }
        }
        
        return orders
    
    # Section 7: Exit & Hedge Management
    def _check_exit_conditions(
        self, 
        position: Dict[str, Any],
        current_stock_price: float,
        current_put_price: float,
        iv_rank: float,
        dte: int
    ) -> Dict[str, Any]:
        """
        Check if exit conditions are met for a position.
        
        Args:
            position: Current position details
            current_stock_price: Current stock price
            current_put_price: Current put option price
            iv_rank: Current IV rank
            dte: Days to expiration
            
        Returns:
            Dictionary with exit decision and reason
        """
        # TODO: Check if stock hit profit target
        # TODO: Check if put can be sold at profit
        # TODO: Check if hedge should be unwound due to low IV
        # TODO: Check if partial unwind should occur
        
        exit_decision = {
            "exit_stock": False,
            "exit_put": False,
            "partial_unwind": False,
            "roll_put": False,
            "reason_stock": None,
            "reason_put": None
        }
        
        # Check stock profit target
        entry_stock_price = position["stock_entry_price"]
        stock_profit_pct = (current_stock_price - entry_stock_price) / entry_stock_price * 100
        
        if stock_profit_pct >= self.parameters["profit_target_pct"]:
            exit_decision["exit_stock"] = True
            exit_decision["exit_put"] = True  # Exit both legs
            exit_decision["reason_stock"] = "profit_target"
            exit_decision["reason_put"] = "stock_profit_target"
            return exit_decision
        
        # Check put take profit
        entry_put_price = position["put_entry_price"]
        put_profit_pct = (entry_put_price - current_put_price) / entry_put_price * 100
        
        if put_profit_pct >= self.parameters["put_take_profit_pct"]:
            exit_decision["exit_put"] = True
            exit_decision["reason_put"] = "put_profit_target"
        
        # Check IV rank for hedge unwind
        if iv_rank <= self.parameters["hedge_unwind_iv_threshold"]:
            exit_decision["exit_put"] = True
            exit_decision["reason_put"] = "low_iv_unwind"
        
        # Check for partial unwind based on drawdown
        stock_drawdown_pct = max(0, (entry_stock_price - current_stock_price) / entry_stock_price * 100)
        
        if (stock_drawdown_pct >= self.parameters["partial_unwind_drawdown"] and 
            not position.get("partial_unwind_executed", False)):
            exit_decision["partial_unwind"] = True
        
        # Check if put needs to be rolled (close to expiration)
        if dte <= self.parameters["min_dte"]:
            exit_decision["roll_put"] = True
        
        return exit_decision
    
    def _prepare_roll_hedge(
        self,
        position: Dict[str, Any],
        current_stock_price: float,
        option_chain: pd.DataFrame,
        iv_rank: float
    ) -> Dict[str, Any]:
        """
        Prepare orders to roll a put hedge to a new expiration.
        
        Args:
            position: Current position details
            current_stock_price: Current stock price
            option_chain: Option chain data
            iv_rank: Current IV rank
            
        Returns:
            Dictionary with roll orders
        """
        # TODO: Implement roll logic with equal or wider buffer
        # TODO: Select new expiration cycle
        # TODO: Create exit and entry orders for the roll
        
        # Select new expiration
        expiration_info = self._select_optimal_expiration(
            option_chain,
            iv_rank,
            is_roll=True  # This is a roll
        )
        
        # Select new strike with adjusted buffer
        strike_info = self._select_optimal_put_strike(
            current_stock_price,
            option_chain,
            iv_rank
        )
        
        # Create roll orders
        roll_orders = {
            "exit_current_put": {
                "action": "SELL",
                "symbol": position["symbol"],
                "option_type": "PUT",
                "strike": position["put_strike"],
                "expiration": position["put_expiration"],
                "contracts": position["contracts"],
                "order_type": "LIMIT",
                "limit_price": 0.0  # To be filled with current market price
            },
            "enter_new_put": {
                "action": "BUY",
                "symbol": position["symbol"],
                "option_type": "PUT",
                "strike": strike_info["strike"],
                "expiration": expiration_info["expiration_date"],
                "contracts": position["contracts"],
                "order_type": "LIMIT",
                "limit_price": strike_info["premium"]
            },
            "net_debit": 0.0,  # To be calculated
            "new_protection_level": strike_info["strike"],
            "new_dte": expiration_info["dte"]
        }
        
        return roll_orders
    
    # Section 8: Position Sizing & Risk Controls
    def _check_portfolio_constraints(
        self, 
        new_position_cost: float,
        account_data: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Check if portfolio constraints allow a new position.
        
        Args:
            new_position_cost: Cost of the new position (stock + put)
            account_data: Account information
            
        Returns:
            Dictionary of constraint check results
        """
        # TODO: Check equity allocation limit per position
        # TODO: Check hedge cost limit
        # TODO: Check maximum concurrent positions
        # TODO: Check drawdown threshold
        
        equity = account_data["equity"]
        
        # Extract current positions and allocation
        current_positions = account_data.get("married_put_positions", [])
        current_equity_allocation = sum(p.get("equity_allocation", 0) for p in current_positions)
        current_hedge_allocation = sum(p.get("hedge_allocation", 0) for p in current_positions)
        
        # Calculate new allocations
        new_equity_allocation = current_equity_allocation + new_position_cost * 0.9  # Estimate 90% stock, 10% put
        new_hedge_allocation = current_hedge_allocation + new_position_cost * 0.1  # Estimate
        
        # Check constraints
        max_equity_allocation = equity * (self.parameters["equity_allocation_pct"] / 100) * self.parameters["max_positions"]
        max_hedge_allocation = equity * (self.parameters["hedge_cost_max_pct"] / 100) * self.parameters["max_positions"]
        
        # Check drawdown
        portfolio_drawdown = account_data.get("current_drawdown_pct", 0)
        
        constraints = {
            "max_positions_ok": len(current_positions) < self.parameters["max_positions"],
            "equity_allocation_ok": new_equity_allocation <= max_equity_allocation,
            "hedge_allocation_ok": new_hedge_allocation <= max_hedge_allocation,
            "drawdown_ok": portfolio_drawdown < self.parameters["drawdown_halt_threshold"],
            "all_constraints_ok": False
        }
        
        # Check if all constraints are satisfied
        constraints["all_constraints_ok"] = all([
            constraints["max_positions_ok"],
            constraints["equity_allocation_ok"],
            constraints["hedge_allocation_ok"],
            constraints["drawdown_ok"]
        ])
        
        return constraints
    
    # Section 9: Backtesting & Performance Metrics
    def _calculate_performance_metrics(
        self, 
        trade_history: List[Dict[str, Any]],
        account_history: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Calculate performance metrics for the strategy.
        
        Args:
            trade_history: List of completed trades
            account_history: DataFrame with account balance history
            benchmark_data: Optional benchmark data for comparison
            
        Returns:
            Dictionary of performance metrics
        """
        # TODO: Calculate net return after option costs
        # TODO: Compute cost of hedging
        # TODO: Calculate max drawdown in protected vs. unprotected periods
        # TODO: Measure hedge efficacy
        # TODO: Determine win rate
        
        metrics = {
            "net_return_pct": 0.0,
            "cost_of_hedging_pct": 0.0,
            "protected_max_drawdown_pct": 0.0,
            "unprotected_max_drawdown_pct": 0.0,
            "hedge_efficacy": 0.0,  # Ratio of loss reduction
            "win_rate": 0.0,
            "avg_gain_per_win": 0.0,
            "avg_loss_per_loss": 0.0,
            "profit_factor": 0.0,
            "recovery_factor": 0.0
        }
        
        if benchmark_data is not None:
            metrics["excess_return_vs_benchmark"] = 0.0
            metrics["downside_protection_vs_benchmark"] = 0.0
        
        return metrics
    
    # Section 10: Continuous Optimization
    def _optimize_parameters(
        self, 
        trade_history: List[Dict[str, Any]],
        market_data: Dict[str, Any],
        current_iv_rank: float
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters based on recent performance and market conditions.
        
        Args:
            trade_history: Recent trade history
            market_data: Current market data
            current_iv_rank: Current IV rank (0-100)
            
        Returns:
            Dictionary of optimized parameters
        """
        # TODO: Re-evaluate OTM buffer and DTE window
        # TODO: Implement IV-adaptive strike adjustments
        # TODO: Add dynamic layering logic
        # TODO: Consider ML predictions for drawdown risk
        
        optimized_params = self.parameters.copy()
        
        # IV-adaptive strike adjustments
        if current_iv_rank >= 75:
            # High IV environment - go deeper OTM
            optimized_params["otm_buffer_pct"] = self.parameters["iv_regime_adjustment"]["high_iv_buffer_pct"]
            logger.info(f"High IV environment ({current_iv_rank}), adjusting OTM buffer to {optimized_params['otm_buffer_pct']}%")
        elif current_iv_rank <= 25:
            # Low IV environment - tighter protection
            optimized_params["otm_buffer_pct"] = self.parameters["iv_regime_adjustment"]["low_iv_buffer_pct"]
            logger.info(f"Low IV environment ({current_iv_rank}), adjusting OTM buffer to {optimized_params['otm_buffer_pct']}%")
        
        # Adjust parameters based on recent performance
        if len(trade_history) >= 10:
            # Calculate recent metrics to guide optimization
            recent_trades = trade_history[-10:]
            
            # Simple example: adjust DTE based on average holding period
            avg_holding_period = sum(t.get("holding_period_days", 30) for t in recent_trades) / len(recent_trades)
            
            if avg_holding_period < 20:
                # Trades are closing quickly, use shorter DTE
                optimized_params["dte_range"] = (20, 35)
            elif avg_holding_period > 45:
                # Trades are lasting longer, use longer DTE
                optimized_params["dte_range"] = (40, 60)
        
        return optimized_params
    
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
        Calculate indicators needed for the married-put strategy.
        
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
                
                # Trend status (above/below SMA)
                trend_status = df['close'] > sma
                
                # Check for higher highs
                lookback = self.parameters["trend_filter"]["higher_highs_lookback"]
                has_higher_highs = False
                if len(df) > lookback:
                    recent_highs = df['high'].rolling(window=lookback).max()
                    has_higher_highs = recent_highs.iloc[-1] > recent_highs.iloc[-lookback]
                
                # Placeholder for IV data (would come from options data in real implementation)
                # Here we'll just simulate IV for demonstration
                iv_series = pd.Series(np.random.uniform(0.2, 0.4, len(df)), index=df.index)
                iv_current = iv_series.iloc[-1]
                
                # Calculate IV rank
                iv_rank = self.calculate_iv_rank(iv_current, iv_series)
                
                # Store indicators
                indicators[symbol] = {
                    "sma": sma,
                    "trend_status": trend_status,
                    "has_higher_highs": has_higher_highs,
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
        option_chains: Optional[Dict[str, pd.DataFrame]] = None,
        account_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Signal]:
        """
        Generate married-put signals based on selection criteria.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            option_chains: Optional dictionary of option chains
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
                "married_put_positions": [],
                "current_drawdown_pct": 0.0
            }
        
        # First check portfolio constraints before generating new signals
        portfolio_constraints = self._check_portfolio_constraints(
            20000.0,  # Assume typical position cost for check
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
                    option_chains.get(symbol, pd.DataFrame()),
                    {"iv_rank": symbol_indicators["iv_rank"]}
                )
                
                # Skip if criteria not met
                if not selection_results.get("all_criteria_passed", False):
                    logger.debug(f"Selection criteria not met for {symbol}: {selection_results}")
                    continue
                
                # Select put strike
                strike_info = self._select_optimal_put_strike(
                    latest_price,
                    option_chains.get(symbol, pd.DataFrame()),
                    symbol_indicators["iv_rank"]
                )
                
                # Select expiration
                expiration_info = self._select_optimal_expiration(
                    option_chains.get(symbol, pd.DataFrame()),
                    symbol_indicators["iv_rank"]
                )
                
                # Prepare entry orders
                entry_orders = self._prepare_entry_orders(
                    symbol,
                    latest_price,
                    strike_info,
                    expiration_info,
                    account_data
                )
                
                if entry_orders is None or entry_orders["stock_leg"]["shares"] == 0:
                    logger.debug(f"Could not prepare valid entry orders for {symbol}")
                    continue
                
                # Generate signal
                signals[symbol] = Signal(
                    symbol=symbol,
                    signal_type=SignalType.MARRIED_PUT,  # Custom signal type for married put
                    price=latest_price,
                    timestamp=latest_timestamp,
                    confidence=min(0.7 + (symbol_indicators["iv_rank"] / 100) / 3, 0.9),  # Higher with better IV env
                    stop_loss=strike_info["strike"],  # Put strike serves as effective stop
                    take_profit=latest_price * (1 + self.parameters["profit_target_pct"] / 100),
                    metadata={
                        "strategy_type": "married_put",
                        "iv_rank": symbol_indicators["iv_rank"],
                        "put_strike": strike_info["strike"],
                        "put_premium": strike_info["premium"],
                        "expiration": expiration_info.get("expiration_date"),
                        "dte": expiration_info.get("dte"),
                        "shares": entry_orders["stock_leg"]["shares"],
                        "contracts": entry_orders["put_leg"]["contracts"],
                        "total_cost": entry_orders["combined"]["total_cost"],
                        "hedge_cost_pct": entry_orders["combined"]["hedge_cost_pct"],
                        "protection_buffer_pct": entry_orders["combined"]["protection_buffer_pct"],
                        "sma": symbol_indicators["sma"].iloc[-1],
                        "above_sma": symbol_indicators["above_sma"],
                        "has_higher_highs": symbol_indicators["has_higher_highs"]
                    }
                )
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    def _mock_option_chains(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
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
            
            # Create a range of strikes
            strikes = [latest_price * (1 - i/100) for i in range(1, 15)]
            expirations = [
                datetime.now() + timedelta(days=days) for days in [7, 14, 21, 30, 45, 60, 90]
            ]
            
            # Create mock option data
            rows = []
            for strike in strikes:
                for exp in expirations:
                    dte = (exp - datetime.now()).days
                    delta = 0.5 - (latest_price - strike) / latest_price
                    iv = 0.3 + (0.1 * np.random.random())
                    bid = max(0.01, latest_price * 0.02 * (delta + 0.2) * (dte/30))
                    ask = bid * 1.1
                    
                    rows.append({
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
            
            option_chains[symbol] = pd.DataFrame(rows)
        
        return option_chains 