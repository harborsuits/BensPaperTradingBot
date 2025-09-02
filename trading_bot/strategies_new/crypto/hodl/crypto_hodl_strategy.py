#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto HODL (Buy and Hold) Strategy Module

This module implements a HODL (Buy and Hold) strategy for cryptocurrency trading.
Unlike other active trading strategies, this strategy focuses on long-term accumulation
and holding of crypto assets, with optional features like periodic rebalancing,
dollar-cost averaging, and capital preservation mechanisms.

The strategy is designed to be a baseline for comparison with more active strategies
and to capture the long-term growth potential of crypto assets.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

# Import base strategy
from trading_bot.strategies_new.crypto.base.crypto_base_strategy import CryptoBaseStrategy, CryptoSession
from trading_bot.strategies_new.factory.registry import register_strategy
from trading_bot.event_system.event import Event
from trading_bot.position_management.position import Position

# Configure logger
logger = logging.getLogger(__name__)

@register_strategy(
    name="CryptoHODLStrategy",
    market_type="crypto",
    description="A Buy-and-Hold strategy for cryptocurrencies, focusing on long-term accumulation and holding with optional features like periodic rebalancing and capital preservation.",
    timeframes=["1d", "1w", "1M"],
    parameters={
        # Core parameters
        "initial_allocation_pct": {
            "type": "float",
            "default": 0.5,
            "description": "Initial percentage of available capital to allocate to the asset"
        },
        "max_allocation_pct": {
            "type": "float",
            "default": 0.80,
            "description": "Maximum percentage of total portfolio that can be allocated to this asset"
        },
        "enable_dca": {
            "type": "bool",
            "default": True,
            "description": "Whether to enable dollar-cost averaging for additional purchases"
        },
        "dca_frequency_days": {
            "type": "int",
            "default": 30,
            "description": "Frequency of DCA purchases in days"
        },
        "dca_allocation_pct": {
            "type": "float",
            "default": 0.05,
            "description": "Percentage of available capital to use for each DCA purchase"
        },
        "enable_threshold_buys": {
            "type": "bool",
            "default": True,
            "description": "Whether to enable additional purchases on significant price drops"
        },
        "price_drop_threshold": {
            "type": "float",
            "default": 0.20,
            "description": "Price drop threshold (as percentage) to trigger additional buys"
        },
        "enable_trailing_stop": {
            "type": "bool",
            "default": True,
            "description": "Whether to enable a trailing stop for capital preservation"
        },
        "trailing_stop_pct": {
            "type": "float",
            "default": 0.30,
            "description": "Trailing stop percentage (from highest point)"
        },
        "enable_rebalancing": {
            "type": "bool",
            "default": False,
            "description": "Whether to enable periodic portfolio rebalancing"
        },
        "rebalance_frequency_days": {
            "type": "int",
            "default": 90,
            "description": "Frequency of portfolio rebalancing in days"
        },
        "target_allocation_pct": {
            "type": "float",
            "default": 0.5,
            "description": "Target allocation percentage for rebalancing"
        },
        "enable_take_profit": {
            "type": "bool",
            "default": False,
            "description": "Whether to enable taking partial profits at predefined levels"
        },
        "take_profit_levels": {
            "type": "list",
            "default": [2.0, 3.0, 5.0, 8.0, 13.0], 
            "description": "Multiple levels at which to take partial profits (as multiple of entry price)"
        },
        "take_profit_pct_per_level": {
            "type": "float",
            "default": 0.20,
            "description": "Percentage of position to take as profit at each profit level"
        },
        "enable_reinvestment": {
            "type": "bool",
            "default": True,
            "description": "Whether to reinvest profits from partial take-profit actions"
        },
        "min_reinvestment_days": {
            "type": "int",
            "default": 60,
            "description": "Minimum number of days to wait before reinvesting profits"
        },
        "reinvestment_dip_threshold": {
            "type": "float",
            "default": 0.15,
            "description": "Price dip threshold to trigger profit reinvestment (as percentage from recent high)"
        },
        "capital_preservation_mode": {
            "type": "str",
            "default": "trailing_stop",
            "enum": ["trailing_stop", "fixed_stop", "moving_average", "none"],
            "description": "Method used for capital preservation"
        },
        "ma_periods": {
            "type": "int",
            "default": 200,
            "description": "Periods to use for moving average capital preservation mode"
        },
        "ma_type": {
            "type": "str",
            "default": "sma",
            "enum": ["sma", "ema", "wma"],
            "description": "Type of moving average to use for capital preservation"
        },
        # Performance tracking parameters
        "track_vs_benchmark": {
            "type": "bool",
            "default": True,
            "description": "Whether to track performance against benchmark (e.g., Bitcoin)"
        },
        "benchmark_symbol": {
            "type": "str",
            "default": "BTC-USD",
            "description": "Symbol to use as benchmark for performance comparison"
        },
        "record_drawdowns": {
            "type": "bool",
            "default": True,
            "description": "Whether to record and track drawdowns"
        },
    },
    asset_classes=["crypto"],
    timeframes=["1h", "4h", "1d", "1w"]
)
class CryptoHODLStrategy(CryptoBaseStrategy):
    """
    A cryptocurrency HODL (Buy and Hold) strategy that focuses on long-term
    accumulation and holding with optional features like periodic rebalancing,
    dollar-cost averaging, and capital preservation mechanisms.
    
    This strategy aims to capture the long-term growth potential of crypto assets
    while potentially reducing volatility through disciplined purchase schedules
    and basic risk management.
    """
    
    def __init__(self, session: CryptoSession, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the HODL strategy.
        
        Args:
            session: The trading session
            parameters: Strategy parameters
        """
        super().__init__(session, parameters)
        
        # Initialize strategy-specific state variables
        self.highest_price_seen = 0.0
        self.accumulated_position_value = 0.0
        self.total_investment = 0.0
        self.total_cost_basis = 0.0
        self.avg_entry_price = 0.0
        self.last_dca_date = None
        self.last_rebalance_date = None
        self.profit_taken_levels = set()  # Keeps track of profit levels where we've taken profits
        self.reinvestable_profits = 0.0  # Cash from profits that can be reinvested
        self.last_profit_taking_date = None
        
        # Dictionary to track partial positions for each buy
        self.position_entries = []  # List of {date, price, quantity, cost, type}
        
        # Initialize performance metrics
        self.drawdowns = []  # List of {start_date, end_date, depth_pct, duration_days}
        self.current_drawdown = None
        self.benchmark_start_price = None
        self.benchmark_current_price = None
        
        # Initialize additional buy levels for threshold-based purchases
        if self.parameters["enable_threshold_buys"]:
            self._initialize_threshold_levels()
            
        # Register additional event handlers
        self.register_event_handler("benchmark_updated", self._on_benchmark_updated)
        
        logger.info(f"CryptoHODLStrategy initialized for {self.session.symbol} "
                    f"with initial allocation percentage: {self.parameters['initial_allocation_pct'] * 100:.1f}%")
    
    def _initialize_threshold_levels(self) -> None:
        """Initialize the threshold price levels for additional buys on dips."""
        self.threshold_levels = []
        # These will be populated once we have a base price from the first purchase
        
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicators required for the HODL strategy.
        
        For a HODL strategy, we primarily track simple metrics like:
        - Moving averages (if used for capital preservation)
        - Percent from all-time high (drawdown)
        - Current gain/loss vs initial investment
        - DCA schedules and rebalancing opportunities
        
        Args:
            data: Market data DataFrame with OHLCV data
            
        Returns:
            Dictionary of calculated indicators
        """
        if data.empty:
            return {}
        
        indicators = {}
        
        # Current price and historical stats
        current_price = data["close"].iloc[-1]
        indicators["current_price"] = current_price
        
        # Update highest price seen (for trailing stop and drawdown calculations)
        if current_price > self.highest_price_seen:
            self.highest_price_seen = current_price
        
        # Calculate drawdown
        if self.highest_price_seen > 0:
            drawdown = (self.highest_price_seen - current_price) / self.highest_price_seen
            indicators["drawdown"] = drawdown
            indicators["drawdown_pct"] = drawdown * 100
            
            # Track significant drawdowns if enabled
            if self.parameters["record_drawdowns"]:
                self._track_drawdowns(current_price, drawdown, data.index[-1])
        
        # Calculate basic moving averages if needed for capital preservation
        if self.parameters["capital_preservation_mode"] == "moving_average":
            ma_type = self.parameters["ma_type"]
            periods = self.parameters["ma_periods"]
            
            if ma_type == "sma":
                indicators["ma"] = data["close"].rolling(window=periods).mean()
            elif ma_type == "ema":
                indicators["ma"] = data["close"].ewm(span=periods, adjust=False).mean()
            elif ma_type == "wma":
                weights = np.arange(1, periods + 1)
                indicators["ma"] = data["close"].rolling(window=periods).apply(
                    lambda x: np.sum(weights * x) / weights.sum(), raw=True
                )
                
            indicators["above_ma"] = current_price > indicators["ma"].iloc[-1]
            indicators["ma_value"] = indicators["ma"].iloc[-1]
            indicators["pct_from_ma"] = (current_price / indicators["ma"].iloc[-1] - 1) * 100
            
        # Check for DCA opportunity
        if self.parameters["enable_dca"]:
            indicators["dca_opportunity"] = self._check_dca_opportunity(data.index[-1])
        
        # Check for rebalancing opportunity
        if self.parameters["enable_rebalancing"]:
            indicators["rebalance_opportunity"] = self._check_rebalance_opportunity(data.index[-1])
        
        # Check for threshold buy opportunities (buying dips)
        if self.parameters["enable_threshold_buys"] and self.avg_entry_price > 0:
            indicators["threshold_buy_opportunity"] = self._check_threshold_buy_opportunity(current_price)
        
        # Check for take profit opportunities
        if self.parameters["enable_take_profit"] and self.avg_entry_price > 0:
            indicators["take_profit_opportunity"] = self._check_take_profit_opportunity(current_price)
            
        # Check for reinvestment opportunities
        if (self.parameters["enable_reinvestment"] and self.reinvestable_profits > 0 and 
            self.last_profit_taking_date is not None):
            indicators["reinvest_opportunity"] = self._check_reinvestment_opportunity(
                current_price, data.index[-1]
            )
            
        # Calculate current position performance
        if self.total_investment > 0:
            current_position_value = current_price * sum(entry["quantity"] for entry in self.position_entries)
            gain_loss_pct = (current_position_value / self.total_investment - 1) * 100
            indicators["gain_loss_pct"] = gain_loss_pct
            indicators["current_position_value"] = current_position_value
            
            # Compare to benchmark if enabled
            if (self.parameters["track_vs_benchmark"] and 
                self.benchmark_start_price is not None and 
                self.benchmark_current_price is not None):
                benchmark_gain_pct = (self.benchmark_current_price / self.benchmark_start_price - 1) * 100
                indicators["benchmark_gain_pct"] = benchmark_gain_pct
                indicators["vs_benchmark_pct"] = gain_loss_pct - benchmark_gain_pct
        
        return indicators
    
    def _check_dca_opportunity(self, current_date) -> bool:
        """
        Check if it's time for a scheduled DCA purchase based on elapsed days.
        
        Args:
            current_date: Current date/time
            
        Returns:
            True if it's time for a DCA purchase, False otherwise
        """
        # If this is our first DCA check, initialize the last DCA date
        if self.last_dca_date is None:
            self.last_dca_date = current_date
            return False
        
        # Calculate days since last DCA
        if isinstance(current_date, pd.Timestamp):
            days_since_last_dca = (current_date - self.last_dca_date).days
        else:
            days_since_last_dca = (current_date - self.last_dca_date).total_seconds() / 86400
        
        # Check if we've reached the DCA frequency
        if days_since_last_dca >= self.parameters["dca_frequency_days"]:
            logger.info(f"DCA opportunity detected after {days_since_last_dca:.1f} days")
            return True
        
        return False
    
    def _check_rebalance_opportunity(self, current_date) -> bool:
        """
        Check if it's time for portfolio rebalancing based on elapsed days.
        
        Args:
            current_date: Current date/time
            
        Returns:
            True if it's time for rebalancing, False otherwise
        """
        # If this is our first rebalance check, initialize the last rebalance date
        if self.last_rebalance_date is None:
            self.last_rebalance_date = current_date
            return False
        
        # Calculate days since last rebalance
        if isinstance(current_date, pd.Timestamp):
            days_since_last_rebalance = (current_date - self.last_rebalance_date).days
        else:
            days_since_last_rebalance = (current_date - self.last_rebalance_date).total_seconds() / 86400
        
        # Check if we've reached the rebalance frequency
        if days_since_last_rebalance >= self.parameters["rebalance_frequency_days"]:
            logger.info(f"Rebalance opportunity detected after {days_since_last_rebalance:.1f} days")
            return True
        
        return False
    
    def _check_threshold_buy_opportunity(self, current_price) -> Optional[Dict[str, Any]]:
        """
        Check if the current price has dropped below a threshold level for buying dips.
        
        Args:
            current_price: Current market price
            
        Returns:
            Dictionary with threshold level info if it's a buy opportunity, None otherwise
        """
        if self.avg_entry_price <= 0:
            return None
        
        # Check if price has dropped below threshold from average entry
        price_drop_pct = (self.avg_entry_price - current_price) / self.avg_entry_price
        threshold = self.parameters["price_drop_threshold"]
        
        if price_drop_pct >= threshold:
            # Check if we've already bought at this level
            rounded_drop = round(price_drop_pct / threshold) * threshold
            for level in self.threshold_levels:
                if abs(level["drop_pct"] - rounded_drop) < 0.01 and level["triggered"]:  # 1% tolerance
                    # Already bought at this level
                    return None
            
            # If we get here, this is a new threshold level to buy
            logger.info(f"Threshold buy opportunity: price dropped {price_drop_pct:.2%} from average entry")
            
            # Record this new threshold level if not already in our list
            level_exists = False
            for level in self.threshold_levels:
                if abs(level["drop_pct"] - rounded_drop) < 0.01:  # 1% tolerance
                    level_exists = True
                    level["triggered"] = True
                    break
            
            if not level_exists:
                self.threshold_levels.append({
                    "drop_pct": rounded_drop,
                    "trigger_price": current_price,
                    "triggered": True,
                    "date": datetime.now()
                })
            
            return {
                "drop_pct": price_drop_pct,
                "trigger_price": current_price,
                "entry_price": self.avg_entry_price
            }
        
        return None
    
    def _check_take_profit_opportunity(self, current_price) -> Optional[Dict[str, Any]]:
        """
        Check if the current price has reached a take-profit level.
        
        Args:
            current_price: Current market price
            
        Returns:
            Dictionary with take-profit info if it's a profit-taking opportunity, None otherwise
        """
        if self.avg_entry_price <= 0:
            return None
        
        # Calculate current gain multiple
        gain_multiple = current_price / self.avg_entry_price
        
        # Get profit levels from parameters
        profit_levels = self.parameters["take_profit_levels"]
        
        # Find the highest level that we've hit but haven't taken profits at yet
        triggered_level = None
        for level in sorted(profit_levels, reverse=True):  # Check highest levels first
            if gain_multiple >= level and level not in self.profit_taken_levels:
                triggered_level = level
                break
        
        if triggered_level is not None:
            logger.info(f"Take profit opportunity: price reached {triggered_level}x entry price")
            
            return {
                "level": triggered_level,
                "entry_price": self.avg_entry_price,
                "current_price": current_price,
                "pct_to_take": self.parameters["take_profit_pct_per_level"]
            }
        
        return None
    
    def _check_reinvestment_opportunity(self, current_price, current_date) -> Optional[Dict[str, Any]]:
        """
        Check if we should reinvest profits based on price dips and time elapsed.
        
        Args:
            current_price: Current market price
            current_date: Current date/time
            
        Returns:
            Dictionary with reinvestment info if it's a reinvestment opportunity, None otherwise
        """
        if self.reinvestable_profits <= 0 or self.last_profit_taking_date is None:
            return None
        
        # Calculate days since last profit taking
        if isinstance(current_date, pd.Timestamp):
            days_since_profit = (current_date - self.last_profit_taking_date).days
        else:
            days_since_profit = (current_date - self.last_profit_taking_date).total_seconds() / 86400
        
        # Check if enough time has passed
        if days_since_profit < self.parameters["min_reinvestment_days"]:
            return None
        
        # Check if price has dropped enough from recent high
        if self.highest_price_seen > 0:
            price_drop_pct = (self.highest_price_seen - current_price) / self.highest_price_seen
            if price_drop_pct >= self.parameters["reinvestment_dip_threshold"]:
                logger.info(f"Reinvestment opportunity: price dropped {price_drop_pct:.2%} from high, "
                           f"{days_since_profit:.1f} days since last profit taking")
                
                return {
                    "amount": self.reinvestable_profits,
                    "drop_pct": price_drop_pct,
                    "days_elapsed": days_since_profit
                }
        
        return None
    
    def _track_drawdowns(self, current_price, current_drawdown, current_date) -> None:
        """
        Track significant drawdowns for performance analysis.
        
        Args:
            current_price: Current market price
            current_drawdown: Current drawdown percentage
            current_date: Current date/time
        """
        # Define a significant drawdown threshold (e.g., 10%)
        significant_threshold = 0.10
        
        if current_drawdown >= significant_threshold:
            # We're in a significant drawdown
            if self.current_drawdown is None:
                # Start of a new drawdown
                self.current_drawdown = {
                    "start_date": current_date,
                    "start_price": self.highest_price_seen,
                    "lowest_price": current_price,
                    "depth_pct": current_drawdown,
                    "end_date": None
                }
            else:
                # Update existing drawdown if price is lower
                if current_price < self.current_drawdown["lowest_price"]:
                    self.current_drawdown["lowest_price"] = current_price
                    self.current_drawdown["depth_pct"] = current_drawdown
        elif self.current_drawdown is not None and current_drawdown < significant_threshold / 2:
            # End of a drawdown (requires recovery to half of the threshold)
            self.current_drawdown["end_date"] = current_date
            self.current_drawdown["duration_days"] = (current_date - self.current_drawdown["start_date"]).days \
                if isinstance(current_date, pd.Timestamp) else \
                (current_date - self.current_drawdown["start_date"]).total_seconds() / 86400
            
            # Add to drawdowns list
            self.drawdowns.append(self.current_drawdown)
            logger.info(f"Recorded drawdown: {self.current_drawdown['depth_pct']:.2%} over "
                       f"{self.current_drawdown['duration_days']:.1f} days")
            
            # Reset current drawdown
            self.current_drawdown = None
    
    def _on_benchmark_updated(self, event: Event) -> None:
        """
        Handle benchmark price updates for performance comparison.
        
        Args:
            event: Benchmark updated event
        """
        if not self.parameters["track_vs_benchmark"]:
            return
        
        # Extract benchmark data
        benchmark_data = event.data
        if benchmark_data.get("symbol") != self.parameters["benchmark_symbol"]:
            return
        
        # Update benchmark price
        benchmark_price = benchmark_data.get("price")
        if benchmark_price is not None:
            # If this is our first benchmark price, set it as the starting point
            if self.benchmark_start_price is None and self.total_investment > 0:
                self.benchmark_start_price = benchmark_price
                logger.info(f"Set benchmark start price: {benchmark_price} for {self.parameters['benchmark_symbol']}")
            
            # Update current benchmark price
            self.benchmark_current_price = benchmark_price
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for the HODL strategy.
        
        For a HODL strategy, signals primarily involve:
        - Initial entry for first position
        - Regular DCA entries
        - Threshold-based dip buying entries
        - Rebalancing adjustments (buy/sell to target allocation)
        - Capital preservation exits (trailing stop/MA break)
        - Take profit signals at predefined levels
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            "long_entry": False,
            "short_entry": False,  # Not used in HODL strategy
            "long_exit": False,
            "short_exit": False,  # Not used in HODL strategy
            "signal_strength": 0.5,  # Default neutral strength
            "entry_type": None,
            "exit_type": None,
            "stop_loss": None,
            "take_profit": None
        }
        
        if data.empty or not indicators:
            return signals
        
        current_price = indicators["current_price"]
        
        # Check if we have any positions yet
        if len(self.positions) == 0 and len(self.position_entries) == 0:
            # No positions yet - initial entry signal
            signals["long_entry"] = True
            signals["signal_strength"] = 0.8  # High conviction for initial entry
            signals["entry_type"] = "initial"
            logger.info(f"Initial entry signal for HODL strategy at price {current_price}")
            return signals
        
        # Check DCA opportunity
        if self.parameters["enable_dca"] and indicators.get("dca_opportunity", False):
            signals["long_entry"] = True
            signals["signal_strength"] = 0.7
            signals["entry_type"] = "dca"
            # Update last DCA date when we generate a signal
            self.last_dca_date = data.index[-1]
            logger.info(f"DCA entry signal for HODL strategy at price {current_price}")
            return signals
        
        # Check threshold buy opportunity
        if self.parameters["enable_threshold_buys"] and "threshold_buy_opportunity" in indicators:
            opportunity = indicators["threshold_buy_opportunity"]
            if opportunity is not None:
                signals["long_entry"] = True
                signals["signal_strength"] = 0.6 + min(0.3, opportunity["drop_pct"])
                signals["entry_type"] = "threshold"
                signals["threshold_drop"] = opportunity["drop_pct"]
                logger.info(f"Threshold entry signal for HODL strategy at price {current_price}, " 
                          f"{opportunity['drop_pct']:.2%} below entry")
                return signals
        
        # Check reinvestment opportunity
        if self.parameters["enable_reinvestment"] and "reinvest_opportunity" in indicators:
            opportunity = indicators["reinvest_opportunity"]
            if opportunity is not None:
                signals["long_entry"] = True
                signals["signal_strength"] = 0.6
                signals["entry_type"] = "reinvestment"
                signals["reinvest_amount"] = opportunity["amount"]
                logger.info(f"Profit reinvestment signal for HODL strategy at price {current_price}, " 
                          f"amount: {opportunity['amount']:.2f}")
                return signals
        
        # Check rebalance opportunity
        if self.parameters["enable_rebalancing"] and indicators.get("rebalance_opportunity", False):
            # Calculate if we need to buy or sell to reach target allocation
            # This is simplified; in a real implementation, this would account for the entire portfolio
            if len(self.position_entries) > 0:
                # Placeholder portfolio value - would come from portfolio manager
                total_portfolio_value = 100000.0
                current_position_value = sum(entry["quantity"] for entry in self.position_entries) * current_price
                current_allocation = current_position_value / total_portfolio_value
                target_allocation = self.parameters["target_allocation_pct"]
                
                if abs(current_allocation - target_allocation) > 0.05:  # 5% deviation threshold
                    if current_allocation < target_allocation:
                        # Need to buy more
                        signals["long_entry"] = True
                        signals["entry_type"] = "rebalance"
                        signals["signal_strength"] = 0.6
                        logger.info(f"Rebalance BUY signal: current allocation {current_allocation:.2%} vs " 
                                   f"target {target_allocation:.2%}")
                    else:
                        # Need to sell some
                        signals["long_exit"] = True
                        signals["exit_type"] = "rebalance"
                        signals["signal_strength"] = 0.6
                        logger.info(f"Rebalance SELL signal: current allocation {current_allocation:.2%} vs " 
                                   f"target {target_allocation:.2%}")
                    
                    # Update last rebalance date
                    self.last_rebalance_date = data.index[-1]
            return signals
        
        # Check take-profit opportunity
        if self.parameters["enable_take_profit"] and "take_profit_opportunity" in indicators:
            opportunity = indicators["take_profit_opportunity"]
            if opportunity is not None:
                signals["long_exit"] = True
                signals["exit_type"] = "take_profit"
                signals["exit_percentage"] = opportunity["pct_to_take"]
                signals["profit_level"] = opportunity["level"]
                signals["signal_strength"] = 0.7
                
                # Mark this profit level as taken
                self.profit_taken_levels.add(opportunity["level"])
                self.last_profit_taking_date = data.index[-1]
                
                logger.info(f"Take-profit signal at {opportunity['level']}x entry price, " 
                          f"selling {opportunity['pct_to_take']:.1%} of position")
                return signals
        
        # Check capital preservation conditions
        # 1. Trailing stop
        if self.parameters["enable_trailing_stop"] and self.highest_price_seen > 0:
            trailing_stop_price = self.highest_price_seen * (1 - self.parameters["trailing_stop_pct"])
            if current_price <= trailing_stop_price:
                signals["long_exit"] = True
                signals["exit_type"] = "trailing_stop"
                signals["signal_strength"] = 0.8
                logger.info(f"Trailing stop exit triggered at {current_price}, " 
                          f"{self.parameters['trailing_stop_pct']:.1%} below high of {self.highest_price_seen}")
                return signals
        
        # 2. Moving average break
        if self.parameters["capital_preservation_mode"] == "moving_average" and "above_ma" in indicators:
            # If price was above MA and now crosses below, exit
            if not indicators["above_ma"] and hasattr(self, "was_above_ma") and self.was_above_ma:
                signals["long_exit"] = True
                signals["exit_type"] = "ma_cross"
                signals["signal_strength"] = 0.7
                logger.info(f"MA cross exit triggered: price {current_price} crossed below " 
                          f"MA {indicators['ma_value']:.2f}")
            
            # Store current MA state for next check
            self.was_above_ma = indicators["above_ma"]
        
        return signals
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate position size for HODL strategy trades.
        
        Position sizing varies based on the entry type:
        - Initial entry: Based on initial_allocation_pct of account
        - DCA: Based on dca_allocation_pct
        - Threshold buys: Progressive sizing based on drop magnitude
        - Rebalancing: Based on deviation from target_allocation_pct
        
        Args:
            direction: Direction of trade (always 'long' for HODL)
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Position size in base currency
        """
        if data.empty or not indicators or direction != "long":
            return 0.0
        
        # Get current price and account equity
        current_price = data["close"].iloc[-1]
        account_equity = 10000.0  # Mock value, would come from the broker/exchange in real implementation
        available_cash = account_equity * 0.95  # Conservative estimate, leaving some buffer
        
        entry_type = self.signals.get("entry_type")
        current_asset_value = self.accumulated_position_value
        
        # Size calculation based on entry type
        if entry_type == "initial":
            # Initial position sizing
            allocation_pct = self.parameters["initial_allocation_pct"]
            position_value = available_cash * allocation_pct
            logger.info(f"Initial position sizing: {allocation_pct:.1%} of available cash (${available_cash:.2f})")
        
        elif entry_type == "dca":
            # Regular DCA purchase
            allocation_pct = self.parameters["dca_allocation_pct"]
            position_value = available_cash * allocation_pct
            logger.info(f"DCA position sizing: {allocation_pct:.1%} of available cash (${available_cash:.2f})")
        
        elif entry_type == "threshold":
            # Threshold-based dip buying - size increases with larger dips
            base_allocation = self.parameters["dca_allocation_pct"]
            drop_pct = self.signals.get("threshold_drop", 0.0)
            # Scale allocation based on drop size (larger drops = larger buys)
            scaled_allocation = base_allocation * (1 + drop_pct * 5)  # Up to 2x for a 20% drop
            scaled_allocation = min(scaled_allocation, base_allocation * 3)  # Cap at 3x base
            position_value = available_cash * scaled_allocation
            logger.info(f"Threshold buy sizing: {scaled_allocation:.1%} of available cash " 
                       f"for {drop_pct:.1%} drop (${position_value:.2f})")
        
        elif entry_type == "reinvestment":
            # Reinvesting profits
            position_value = self.signals.get("reinvest_amount", 0.0)
            # Reduce reinvestable profits for next time
            self.reinvestable_profits -= position_value
            self.reinvestable_profits = max(0, self.reinvestable_profits)  # Ensure non-negative
            logger.info(f"Reinvestment sizing: ${position_value:.2f} from accumulated profits")
        
        elif entry_type == "rebalance":
            # Rebalance buying - calculate how much to buy to reach target allocation
            target_allocation = self.parameters["target_allocation_pct"]
            # Mock total portfolio value - would come from portfolio manager
            total_portfolio_value = 100000.0
            target_position_value = total_portfolio_value * target_allocation
            current_position_value = current_asset_value
            position_value = max(0, target_position_value - current_position_value)
            logger.info(f"Rebalance buy sizing: ${position_value:.2f} to reach {target_allocation:.1%} target")
        
        else:
            # Default to a small position if entry type is unknown
            position_value = available_cash * 0.05
            logger.info(f"Default position sizing: 5% of available cash (${available_cash:.2f})")
        
        # Convert position value to quantity in the asset
        if current_price > 0:
            position_quantity = position_value / current_price
        else:
            position_quantity = 0.0
        
        # Record entry for position tracking
        if position_quantity > 0:
            new_entry = {
                "date": data.index[-1],
                "price": current_price,
                "quantity": position_quantity,
                "cost": position_value,
                "type": entry_type
            }
            self.position_entries.append(new_entry)
            
            # Update totals
            self.total_investment += position_value
            self.accumulated_position_value += position_value
            total_quantity = sum(entry["quantity"] for entry in self.position_entries)
            self.avg_entry_price = self.total_investment / total_quantity if total_quantity > 0 else 0.0
            
            # If this is our first purchase and we're tracking vs benchmark, initialize benchmark
            if len(self.position_entries) == 1 and self.parameters["track_vs_benchmark"]:
                # Request benchmark price data if available
                self.event_bus.publish(Event("request_benchmark_data", {
                    "symbol": self.parameters["benchmark_symbol"],
                    "requester": self.session.strategy_id
                }))
        
        return position_quantity
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible the HODL strategy is with the current market regime.
        
        A HODL strategy generally performs best in long-term bull markets and during
        accumulation phases, and relatively worse in prolonged bear markets.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            "trending": 0.85,          # Very good in trending markets (especially uptrends)
            "ranging": 0.65,           # Good in ranging markets (DCA works well)
            "volatile": 0.60,          # Moderate in volatile markets (helps average in during dips)
            "calm": 0.75,              # Good in calm markets
            "breakout": 0.50,          # Moderate during breakouts (doesn't capitalize on momentum)
            "high_volume": 0.70,       # Good in high volume periods
            "low_volume": 0.80,        # Very good in low volume periods (accumulation phases)
            "high_liquidity": 0.75,    # Good in high liquidity markets
            "low_liquidity": 0.65,     # Good in low liquidity markets
            "accumulation": 0.90,      # Excellent during accumulation phases
            "distribution": 0.40,      # Poor during distribution phases
            "bull_trend": 0.95,        # Excellent during bull trends
            "bear_trend": 0.30,        # Poor during bear trends (relies on capital preservation)
        }
        
        return compatibility_map.get(market_regime, 0.70)  # Default to moderate compatibility
