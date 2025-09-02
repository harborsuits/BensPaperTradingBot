"""
Margin Manager Module

Enhanced margin management and exposure control system inspired by EA31337-Libre.
Provides sophisticated margin allocation, multi-instrument exposure monitoring,
and dynamic margin requirement calculations.

This module extends the core RiskManager with additional capabilities for 
professional-grade margin management across multiple instruments.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from datetime import datetime, timedelta
import json
import math

from trading_bot.risk.risk_manager import RiskManager, RiskLevel

logger = logging.getLogger("MarginManager")

class MarginAllocationMethod:
    """Methods for allocating margin across different instruments"""
    EQUAL = "equal"              # Equal margin allocation
    VOLATILITY_WEIGHTED = "vol"  # Allocation based on volatility
    CORRELATION_ADJUSTED = "corr" # Allocation considering correlations 
    PERFORMANCE_WEIGHTED = "perf" # Allocation based on historical performance
    ADAPTIVE = "adaptive"        # Dynamically adjusts based on market conditions
    CUSTOM = "custom"            # Custom allocation based on user-defined rules

class MarginManager:
    """
    Advanced margin and exposure control system that works alongside the RiskManager.
    
    Key capabilities:
    1. Advanced margin allocation across multiple instruments
    2. Correlation-aware position sizing and exposure controls
    3. Dynamic margin requirement calculations
    4. Scenario analysis and stress testing for margin calls
    5. Multi-currency and cross-exchange margin management
    """
    
    def __init__(self, risk_manager: RiskManager, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the margin manager
        
        Args:
            risk_manager: The main risk manager instance
            config: Configuration parameters for margin management
        """
        self.risk_manager = risk_manager
        self.config = config or self._get_default_config()
        
        # Copy some key values from the risk manager for convenience
        self.portfolio_value = risk_manager.portfolio_value
        
        # Initialize margin tracking
        self.margin_used = 0.0
        self.margin_available = self.portfolio_value * self.config["max_margin_utilization"]
        
        # Track instrument-specific margins
        self.instrument_margins = {}
        
        # Correlation matrix for instruments
        self.correlation_matrix = {}
        
        # Historical margin utilization
        self.margin_history = []
        
        # Instrument leverage settings
        self.instrument_leverage = {}
        
        # Default margin allocation method
        self.allocation_method = self.config.get("allocation_method", MarginAllocationMethod.VOLATILITY_WEIGHTED)
        
        # Initialize margin pools (for multi-currency or specialized allocations)
        self.margin_pools = {}
        
        # Risk categories for instruments
        self.risk_categories = {}
        
        logger.info("Margin Manager initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration
        
        Returns:
            Dictionary with default configuration
        """
        return {
            "max_margin_utilization": 0.8,      # Max percentage of portfolio to use as margin
            "margin_warning_level": 0.7,        # Warning level for margin utilization
            "margin_call_level": 0.9,           # Margin call level (reduce positions)
            "max_instrument_margin": 0.25,      # Max margin per instrument (% of total)
            "min_margin_buffer": 0.2,           # Minimum buffer to maintain (% of required)
            "allocation_method": MarginAllocationMethod.VOLATILITY_WEIGHTED,
            "rebalance_frequency": "daily",     # How often to rebalance margin allocations
            "correlation_lookback": 30,         # Days of data for correlation calculation
            "volatility_lookback": 20,          # Days for volatility calculation
            "adaptive_margin": True,            # Dynamically adjust margins based on conditions
            "max_sector_allocation": 0.4,       # Maximum allocation to a single sector
            "stress_test_scenarios": ["2sigma", "market_crash", "liquidity_crisis"],
            "margin_pools": {                   # Optional segregated margin pools
                "default": 1.0,                 # Allocation to default pool (can add others)
            }
        }
    
    def update_portfolio_value(self, new_value: float):
        """
        Update portfolio value and recalculate margin limits
        
        Args:
            new_value: Updated portfolio value
        """
        self.portfolio_value = new_value
        self.margin_available = new_value * self.config["max_margin_utilization"] - self.margin_used
        
        # Record margin utilization
        self.margin_history.append({
            "timestamp": datetime.now(),
            "portfolio_value": new_value,
            "margin_used": self.margin_used,
            "margin_available": self.margin_available,
            "utilization_pct": self.get_margin_utilization() * 100
        })
        
        # Prune history if needed
        if len(self.margin_history) > 1000:
            self.margin_history = self.margin_history[-1000:]
    
    def get_margin_utilization(self) -> float:
        """
        Get current margin utilization as a ratio
        
        Returns:
            Margin utilization ratio (0-1)
        """
        max_margin = self.portfolio_value * self.config["max_margin_utilization"]
        if max_margin <= 0:
            return 0
        
        return self.margin_used / max_margin
    
    def is_margin_available(self, required_margin: float) -> bool:
        """
        Check if margin is available for a new position
        
        Args:
            required_margin: Margin required for new position
            
        Returns:
            True if margin is available, False otherwise
        """
        return required_margin <= self.margin_available
    
    def allocate_margins(self, instruments: List[str], market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Allocate margin across multiple instruments based on selected method
        
        Args:
            instruments: List of instrument symbols
            market_data: Market data for volatility/correlation calculations
            
        Returns:
            Dictionary of instrument -> allocated margin
        """
        max_margin = self.portfolio_value * self.config["max_margin_utilization"]
        allocations = {}
        
        if self.allocation_method == MarginAllocationMethod.EQUAL:
            # Equal allocation across instruments
            per_instrument = max_margin / len(instruments)
            allocations = {symbol: per_instrument for symbol in instruments}
        
        elif self.allocation_method == MarginAllocationMethod.VOLATILITY_WEIGHTED:
            # Calculate volatility for each instrument
            volatilities = {}
            for symbol in instruments:
                if symbol in market_data and 'close' in market_data[symbol]:
                    close_prices = market_data[symbol]['close']
                    if len(close_prices) >= 2:
                        returns = np.diff(close_prices) / close_prices[:-1]
                        volatilities[symbol] = np.std(returns) * np.sqrt(252)  # Annualized
                    else:
                        volatilities[symbol] = 0.2  # Default if not enough data
                else:
                    volatilities[symbol] = 0.2  # Default volatility
            
            # Inverse volatility weighting (lower volatility gets more allocation)
            inv_vols = {s: 1/v if v > 0 else 0 for s, v in volatilities.items()}
            total_inv_vol = sum(inv_vols.values())
            
            if total_inv_vol > 0:
                allocations = {s: (inv_vols[s] / total_inv_vol) * max_margin for s in instruments}
            else:
                # Fall back to equal allocation
                per_instrument = max_margin / len(instruments)
                allocations = {symbol: per_instrument for symbol in instruments}
        
        elif self.allocation_method == MarginAllocationMethod.CORRELATION_ADJUSTED:
            # This would implement a more complex correlation-based allocation
            # For now, use the volatility-weighted method as a fallback
            return self.allocate_margins(instruments, market_data)
        
        elif self.allocation_method == MarginAllocationMethod.PERFORMANCE_WEIGHTED:
            # Implement performance-based weighting using historical returns
            # For demonstration, use equal weighting
            per_instrument = max_margin / len(instruments)
            allocations = {symbol: per_instrument for symbol in instruments}
        
        elif self.allocation_method == MarginAllocationMethod.ADAPTIVE:
            # Dynamically adjust based on market conditions and current positions
            # For now, implement a simple version based on current margin usage
            
            # Get current margin usage by instrument
            used_margins = {symbol: self.instrument_margins.get(symbol, 0) for symbol in instruments}
            total_used = sum(used_margins.values())
            
            # Allocate remaining margin proportionally
            remaining = max_margin - total_used
            if remaining <= 0:
                # No margin left, keep current allocations
                allocations = used_margins
            else:
                # Determine weights for remaining margin
                weights = {}
                for symbol in instruments:
                    # Check if we already have a position
                    if symbol in used_margins and used_margins[symbol] > 0:
                        # Already allocated, don't add more
                        weights[symbol] = 0
                    else:
                        # New instrument, give it a weight
                        weights[symbol] = 1
                
                total_weight = sum(weights.values())
                if total_weight > 0:
                    # Allocate remaining margin by weight
                    for symbol in instruments:
                        allocations[symbol] = used_margins.get(symbol, 0) + (weights.get(symbol, 0) / total_weight) * remaining
                else:
                    # Fall back to current allocations
                    allocations = used_margins
        
        else:  # Custom or unknown
            # Fall back to equal allocation
            per_instrument = max_margin / len(instruments)
            allocations = {symbol: per_instrument for symbol in instruments}
        
        # Enforce instrument limits
        max_instrument_margin = self.portfolio_value * self.config["max_instrument_margin"]
        for symbol in allocations:
            allocations[symbol] = min(allocations[symbol], max_instrument_margin)
        
        return allocations
    
    def calculate_instrument_margin(self, symbol: str, 
                                   position_size: float, 
                                   price: float, 
                                   leverage: Optional[float] = None) -> float:
        """
        Calculate required margin for a specific instrument position
        
        Args:
            symbol: Instrument symbol
            position_size: Size of position (units, shares, contracts)
            price: Current price per unit
            leverage: Optional leverage ratio (defaults to instrument's setting)
            
        Returns:
            Required margin amount
        """
        # Get instrument-specific leverage if not provided
        if leverage is None:
            leverage = self.instrument_leverage.get(symbol, 1.0)
        
        # Different instruments have different margin calculations
        position_value = position_size * price
        
        # For securities with no leverage, full value is required
        if leverage <= 1.0:
            return position_value
        
        # For leveraged instruments, divide by leverage ratio
        # Add a safety buffer based on configuration
        required_margin = position_value / leverage
        buffer_margin = required_margin * self.config["min_margin_buffer"]
        
        return required_margin + buffer_margin
    
    def reserve_margin(self, symbol: str, amount: float) -> bool:
        """
        Reserve margin for a new position
        
        Args:
            symbol: Instrument symbol
            amount: Margin amount to reserve
            
        Returns:
            True if margin was successfully reserved, False otherwise
        """
        if not self.is_margin_available(amount):
            logger.warning(f"Insufficient margin for {symbol}: required={amount}, available={self.margin_available}")
            return False
        
        # Update tracking
        self.margin_used += amount
        self.margin_available -= amount
        
        # Update instrument-specific tracking
        current = self.instrument_margins.get(symbol, 0)
        self.instrument_margins[symbol] = current + amount
        
        logger.info(f"Reserved {amount} margin for {symbol}. Total used: {self.margin_used}")
        return True
    
    def release_margin(self, symbol: str, amount: float):
        """
        Release previously reserved margin
        
        Args:
            symbol: Instrument symbol
            amount: Margin amount to release
        """
        # Update tracking
        self.margin_used = max(0, self.margin_used - amount)
        self.margin_available = (self.portfolio_value * self.config["max_margin_utilization"]) - self.margin_used
        
        # Update instrument-specific tracking
        current = self.instrument_margins.get(symbol, 0)
        self.instrument_margins[symbol] = max(0, current - amount)
        
        logger.info(f"Released {amount} margin for {symbol}. Total used: {self.margin_used}")
    
    def calculate_stress_scenario(self, scenario: str) -> Dict[str, Any]:
        """
        Calculate margin requirements under stress scenarios
        
        Args:
            scenario: Scenario to simulate ("2sigma", "market_crash", etc.)
            
        Returns:
            Dictionary with stress test results
        """
        results = {
            "scenario": scenario,
            "current_margin": self.margin_used,
            "current_portfolio": self.portfolio_value
        }
        
        if scenario == "2sigma":
            # Simulate a 2-sigma market move (roughly 5% down)
            stress_portfolio = self.portfolio_value * 0.95
            stress_margin_call = self.margin_used > (stress_portfolio * self.config["max_margin_utilization"] * self.config["margin_call_level"])
            
            results.update({
                "stressed_portfolio": stress_portfolio,
                "margin_call_triggered": stress_margin_call,
                "margin_utilization": self.margin_used / (stress_portfolio * self.config["max_margin_utilization"])
            })
            
        elif scenario == "market_crash":
            # Simulate a 15% market drop
            stress_portfolio = self.portfolio_value * 0.85
            stress_margin_call = self.margin_used > (stress_portfolio * self.config["max_margin_utilization"] * self.config["margin_call_level"])
            
            results.update({
                "stressed_portfolio": stress_portfolio,
                "margin_call_triggered": stress_margin_call,
                "margin_utilization": self.margin_used / (stress_portfolio * self.config["max_margin_utilization"])
            })
            
        elif scenario == "liquidity_crisis":
            # Simulate a liquidity crisis (20% drop and 50% margin requirement increase)
            stress_portfolio = self.portfolio_value * 0.8
            increased_margin = self.margin_used * 1.5
            stress_margin_call = increased_margin > (stress_portfolio * self.config["max_margin_utilization"] * self.config["margin_call_level"])
            
            results.update({
                "stressed_portfolio": stress_portfolio,
                "increased_margin": increased_margin,
                "margin_call_triggered": stress_margin_call,
                "margin_utilization": increased_margin / (stress_portfolio * self.config["max_margin_utilization"])
            })
            
        else:
            # Unknown scenario
            results["error"] = f"Unknown stress scenario: {scenario}"
        
        return results
    
    def run_all_stress_tests(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all configured stress scenarios
        
        Returns:
            Dictionary of scenario results
        """
        results = {}
        for scenario in self.config["stress_test_scenarios"]:
            results[scenario] = self.calculate_stress_scenario(scenario)
        return results
    
    def check_margin_limits(self) -> Tuple[bool, List[str]]:
        """
        Check if current margin usage violates any limits
        
        Returns:
            Tuple of (limits_violated, list_of_reasons)
        """
        reasons = []
        utilization = self.get_margin_utilization()
        
        # Check overall margin utilization
        if utilization >= self.config["margin_call_level"]:
            reasons.append(f"Margin utilization ({utilization:.2%}) exceeds margin call level ({self.config['margin_call_level']:.2%})")
        
        # Check per-instrument limits
        max_instrument_margin = self.portfolio_value * self.config["max_instrument_margin"]
        for symbol, margin in self.instrument_margins.items():
            if margin > max_instrument_margin:
                reasons.append(f"Instrument {symbol} margin ({margin}) exceeds maximum allowed ({max_instrument_margin})")
        
        # Check sector limits if sector allocations are defined
        if self.risk_categories and self.config.get("max_sector_allocation"):
            sector_margins = {}
            for symbol, margin in self.instrument_margins.items():
                sector = self.risk_categories.get(symbol, "unknown")
                sector_margins[sector] = sector_margins.get(sector, 0) + margin
            
            max_sector_margin = self.portfolio_value * self.config["max_sector_allocation"]
            for sector, margin in sector_margins.items():
                if margin > max_sector_margin:
                    reasons.append(f"Sector {sector} margin ({margin}) exceeds maximum allowed ({max_sector_margin})")
        
        return bool(reasons), reasons
    
    def get_margin_reduction_actions(self) -> List[Dict[str, Any]]:
        """
        Get recommended actions to reduce margin usage if limits are violated
        
        Returns:
            List of recommended actions
        """
        violations, reasons = self.check_margin_limits()
        if not violations:
            return []
        
        actions = []
        
        # Calculate how much we need to reduce by
        max_margin = self.portfolio_value * self.config["max_margin_utilization"] * self.config["margin_warning_level"]
        excess_margin = max(0, self.margin_used - max_margin)
        
        if excess_margin > 0:
            # Sort instruments by margin usage (highest first)
            sorted_instruments = sorted(
                [(symbol, margin) for symbol, margin in self.instrument_margins.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            remaining_to_reduce = excess_margin
            for symbol, margin in sorted_instruments:
                if remaining_to_reduce <= 0:
                    break
                
                # Calculate how much to reduce for this instrument
                # Reduce proportionally to its contribution to excess
                reduction = min(margin, remaining_to_reduce)
                if reduction > 0:
                    actions.append({
                        "action": "reduce_position",
                        "symbol": symbol,
                        "current_margin": margin,
                        "reduce_by": reduction,
                        "reduce_pct": reduction / margin if margin > 0 else 0
                    })
                    remaining_to_reduce -= reduction
        
        return actions
    
    def update_correlation_matrix(self, market_data: Dict[str, Any]):
        """
        Update correlation matrix for instruments based on recent price data
        
        Args:
            market_data: Market data dictionary with price history
        """
        # Extract symbols with sufficient data
        symbols = []
        price_data = {}
        
        for symbol, data in market_data.items():
            if 'close' in data and len(data['close']) >= self.config["correlation_lookback"]:
                symbols.append(symbol)
                price_data[symbol] = data['close'][-self.config["correlation_lookback"]:]
        
        if len(symbols) <= 1:
            # Not enough symbols for correlation
            return
        
        # Create price DataFrame
        df = pd.DataFrame(price_data)
        
        # Calculate returns
        returns = df.pct_change().dropna()
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Convert to dictionary format
        self.correlation_matrix = {}
        for s1 in symbols:
            self.correlation_matrix[s1] = {}
            for s2 in symbols:
                if s1 != s2:
                    self.correlation_matrix[s1][s2] = corr_matrix.loc[s1, s2]
    
    def set_instrument_leverage(self, symbol: str, leverage: float):
        """
        Set leverage ratio for a specific instrument
        
        Args:
            symbol: Instrument symbol
            leverage: Leverage ratio (e.g., 2.0 for 2:1 leverage)
        """
        self.instrument_leverage[symbol] = max(1.0, leverage)  # Minimum leverage is 1.0 (no leverage)
    
    def set_risk_category(self, symbol: str, category: str):
        """
        Set risk category for an instrument (e.g., sector, asset class)
        
        Args:
            symbol: Instrument symbol
            category: Risk category name
        """
        self.risk_categories[symbol] = category
    
    def set_allocation_method(self, method: str):
        """
        Set the margin allocation method
        
        Args:
            method: Allocation method from MarginAllocationMethod
        """
        if hasattr(MarginAllocationMethod, method.upper()):
            self.allocation_method = getattr(MarginAllocationMethod, method.upper())
        else:
            # Try as a direct string value
            valid_methods = [m for m in dir(MarginAllocationMethod) if not m.startswith('_')]
            if method in valid_methods:
                self.allocation_method = method
            else:
                logger.warning(f"Unknown allocation method: {method}. Valid methods: {valid_methods}")
    
    def get_margin_status(self) -> Dict[str, Any]:
        """
        Get comprehensive margin status report
        
        Returns:
            Dictionary with current margin status
        """
        return {
            "portfolio_value": self.portfolio_value,
            "margin_used": self.margin_used,
            "margin_available": self.margin_available,
            "utilization_pct": self.get_margin_utilization() * 100,
            "warning_level": self.config["margin_warning_level"] * 100,
            "call_level": self.config["margin_call_level"] * 100,
            "instrument_margins": self.instrument_margins,
            "allocation_method": self.allocation_method,
            "risk_categories": self.risk_categories,
            "instrument_leverage": self.instrument_leverage
        }
    
    def get_margin_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get margin utilization history
        
        Args:
            days: Number of days of history to return
            
        Returns:
            List of historical margin snapshots
        """
        cutoff = datetime.now() - timedelta(days=days)
        return [entry for entry in self.margin_history if entry["timestamp"] >= cutoff]
