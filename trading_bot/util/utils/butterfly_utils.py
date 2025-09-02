#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Butterfly Spread Strategy Utilities

This module provides utility functions for the butterfly spread options strategy,
including option pricing, spread construction, and position management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import scipy.stats as stats
import logging

# Setup logging
logger = logging.getLogger(__name__)

class OptionLeg:
    """Class representing an option leg in a multi-leg strategy."""
    
    def __init__(
        self,
        symbol: str,
        expiration: datetime,
        strike: float,
        option_type: str,  # 'call' or 'put'
        quantity: int,
        multiplier: int = 100
    ):
        """
        Initialize option leg.
        
        Args:
            symbol: Underlying symbol
            expiration: Option expiration date
            strike: Strike price
            option_type: Option type ('call' or 'put')
            quantity: Number of contracts (negative for short)
            multiplier: Contract multiplier (typically 100 for equity options)
        """
        self.symbol = symbol
        self.expiration = expiration
        self.strike = strike
        self.option_type = option_type.lower()
        self.quantity = quantity  # Negative for short positions
        self.multiplier = multiplier
        self.price = None  # Market price, to be set later
        self.greeks = {}  # Option Greeks, to be set later
    
    def __str__(self):
        """String representation of the option leg."""
        position = "short" if self.quantity < 0 else "long"
        return f"{abs(self.quantity)} {position} {self.symbol} {self.option_type} @ {self.strike} exp {self.expiration.strftime('%Y-%m-%d')}"
    
    def to_dict(self):
        """Convert leg to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "expiration": self.expiration.strftime("%Y-%m-%d"),
            "strike": self.strike,
            "option_type": self.option_type,
            "quantity": self.quantity,
            "multiplier": self.multiplier,
            "price": self.price
        }

class ButterflySpread:
    """Class representing a complete butterfly spread position."""
    
    def __init__(
        self,
        symbol: str,
        expiration: datetime,
        center_strike: float,
        inner_wings_width: float,
        outer_wings_width: float,
        quantity: int = 1,
        option_type: str = "call",  # Use call or put butterfly
        multiplier: int = 100
    ):
        """
        Initialize butterfly spread.
        
        Args:
            symbol: Underlying symbol
            expiration: Option expiration date
            center_strike: Center strike price
            inner_wings_width: Distance from center to inner wings
            outer_wings_width: Distance from inner to outer wings
            quantity: Number of butterfly spreads
            option_type: Option type for all legs ('call' or 'put')
            multiplier: Contract multiplier
        """
        self.symbol = symbol
        self.expiration = expiration
        self.center_strike = center_strike
        self.inner_wings_width = inner_wings_width
        self.outer_wings_width = outer_wings_width
        self.quantity = quantity
        self.option_type = option_type.lower()
        self.multiplier = multiplier
        
        # Calculate all strike prices
        self.lower_inner_strike = center_strike - inner_wings_width
        self.upper_inner_strike = center_strike + inner_wings_width
        self.lower_outer_strike = self.lower_inner_strike - outer_wings_width
        self.upper_outer_strike = self.upper_inner_strike + outer_wings_width
        
        # Create the option legs
        self.legs = self._create_legs()
        
        # Position tracking
        self.entry_price = None
        self.current_price = None
        self.max_profit = None
        self.max_loss = None
        self.breakeven_points = []
    
    def _create_legs(self) -> List[OptionLeg]:
        """Create the four legs of the butterfly spread."""
        legs = []
        
        # Lower outer wing (long)
        legs.append(OptionLeg(
            symbol=self.symbol,
            expiration=self.expiration,
            strike=self.lower_outer_strike,
            option_type=self.option_type,
            quantity=self.quantity,
            multiplier=self.multiplier
        ))
        
        # Lower inner wing (short)
        legs.append(OptionLeg(
            symbol=self.symbol,
            expiration=self.expiration,
            strike=self.lower_inner_strike,
            option_type=self.option_type,
            quantity=-2 * self.quantity,  # Short twice the quantity
            multiplier=self.multiplier
        ))
        
        # Center (long)
        legs.append(OptionLeg(
            symbol=self.symbol,
            expiration=self.expiration,
            strike=self.center_strike,
            option_type=self.option_type,
            quantity=self.quantity,
            multiplier=self.multiplier
        ))
        
        # Upper inner wing (short)
        legs.append(OptionLeg(
            symbol=self.symbol,
            expiration=self.expiration,
            strike=self.upper_inner_strike,
            option_type=self.option_type,
            quantity=-2 * self.quantity,  # Short twice the quantity
            multiplier=self.multiplier
        ))
        
        # Upper outer wing (long)
        legs.append(OptionLeg(
            symbol=self.symbol,
            expiration=self.expiration,
            strike=self.upper_outer_strike,
            option_type=self.option_type,
            quantity=self.quantity,
            multiplier=self.multiplier
        ))
        
        return legs
    
    def __str__(self):
        """String representation of the butterfly spread."""
        return (f"{self.quantity} {self.option_type} butterfly on {self.symbol} "
                f"@ {self.lower_outer_strike}/{self.lower_inner_strike}/"
                f"{self.center_strike}/{self.upper_inner_strike}/{self.upper_outer_strike} "
                f"exp {self.expiration.strftime('%Y-%m-%d')}")
    
    def to_dict(self):
        """Convert spread to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "expiration": self.expiration.strftime("%Y-%m-%d"),
            "center_strike": self.center_strike,
            "inner_wings_width": self.inner_wings_width,
            "outer_wings_width": self.outer_wings_width,
            "quantity": self.quantity,
            "option_type": self.option_type,
            "multiplier": self.multiplier,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
            "breakeven_points": self.breakeven_points,
            "legs": [leg.to_dict() for leg in self.legs]
        }

def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate option price and Greeks using Black-Scholes model.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Volatility (standard deviation of log returns)
        option_type: 'call' or 'put'
    
    Returns:
        Tuple of (option_price, greeks)
    """
    if T <= 0:
        # Handle expired options
        if option_type == 'call':
            return max(0, S - K), {'delta': 1 if S > K else 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        else:
            return max(0, K - S), {'delta': -1 if S < K else 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        delta = stats.norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2) / 100
    else:  # put
        price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        delta = stats.norm.cdf(d1) - 1
        rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2) / 100
    
    gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * stats.norm.pdf(d1) / 100
    theta = (-S * sigma * stats.norm.pdf(d1)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * stats.norm.cdf(d2 if option_type == 'call' else -d2)
    theta = theta / 365  # Daily theta
    
    greeks = {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }
    
    return price, greeks

def price_butterfly_spread(
    butterfly: ButterflySpread,
    underlying_price: float,
    days_to_expiration: int,
    volatility: float,
    risk_free_rate: float = 0.03  # 3% default risk-free rate
) -> Dict[str, Any]:
    """
    Price a butterfly spread and calculate its characteristics.
    
    Args:
        butterfly: ButterflySpread object
        underlying_price: Current price of the underlying
        days_to_expiration: Days to expiration
        volatility: Implied volatility
        risk_free_rate: Risk-free interest rate
    
    Returns:
        Dictionary with pricing information
    """
    # Convert days to years
    T = days_to_expiration / 365.0
    
    total_price = 0
    total_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    # Price each leg
    for leg in butterfly.legs:
        price, greeks = black_scholes_price(
            S=underlying_price,
            K=leg.strike,
            T=T,
            r=risk_free_rate,
            sigma=volatility,
            option_type=leg.option_type
        )
        
        # Store price and Greeks in leg object
        leg.price = price
        leg.greeks = greeks
        
        # Add to totals (considering quantity)
        total_price += price * leg.quantity
        for greek, value in greeks.items():
            total_greeks[greek] += value * leg.quantity
    
    # Calculate maximum profit and loss
    # For a balanced butterfly, max profit occurs when underlying = center strike at expiration
    width = butterfly.inner_wings_width
    max_profit = (width - total_price) * butterfly.multiplier if total_price > 0 else width * butterfly.multiplier
    max_loss = total_price * butterfly.multiplier if total_price > 0 else 0
    
    # Calculate breakeven points
    # Approximation: center_strike ± (width - debit)
    breakeven_lower = butterfly.center_strike - width + abs(total_price)
    breakeven_upper = butterfly.center_strike + width - abs(total_price)
    
    # Update butterfly object
    butterfly.current_price = total_price
    butterfly.max_profit = max_profit
    butterfly.max_loss = max_loss
    butterfly.breakeven_points = [breakeven_lower, breakeven_upper]
    
    return {
        "price": total_price,
        "greeks": total_greeks,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakeven_points": [breakeven_lower, breakeven_upper],
        "profit_loss_ratio": max_profit / max_loss if max_loss > 0 else float('inf')
    }

def calculate_profit_at_price(
    butterfly: ButterflySpread,
    underlying_price: float,
    days_to_expiration: int,
    volatility: float,
    risk_free_rate: float = 0.03
) -> float:
    """
    Calculate the profit/loss at a specific underlying price.
    
    Args:
        butterfly: ButterflySpread object
        underlying_price: Hypothetical price of the underlying
        days_to_expiration: Days to expiration
        volatility: Implied volatility
        risk_free_rate: Risk-free interest rate
    
    Returns:
        Profit/loss amount
    """
    # Price the butterfly at the given underlying price
    current_pricing = price_butterfly_spread(
        butterfly, underlying_price, days_to_expiration, volatility, risk_free_rate
    )
    
    # Calculate profit/loss
    if butterfly.entry_price is None:
        return 0  # Can't calculate P/L without entry price
    
    return (current_pricing["price"] - butterfly.entry_price) * butterfly.multiplier * butterfly.quantity

def generate_profit_curve(
    butterfly: ButterflySpread,
    price_range_pct: float = 0.10,
    num_points: int = 50,
    days_list: List[int] = None,
    volatility: float = 0.20,
    risk_free_rate: float = 0.03
) -> pd.DataFrame:
    """
    Generate profit curve data for different underlying prices and days to expiration.
    
    Args:
        butterfly: ButterflySpread object
        price_range_pct: Price range to analyze as percentage of center strike
        num_points: Number of price points to calculate
        days_list: List of days to expiration to analyze
        volatility: Implied volatility
        risk_free_rate: Risk-free interest rate
    
    Returns:
        DataFrame with profit curve data
    """
    if butterfly.entry_price is None:
        butterfly.entry_price = butterfly.current_price
    
    if days_list is None:
        days_to_exp = (butterfly.expiration - datetime.now()).days
        days_list = [days_to_exp, int(days_to_exp * 0.75), int(days_to_exp * 0.5), int(days_to_exp * 0.25), 1]
        days_list = [d for d in days_list if d > 0]
    
    # Generate price range
    min_price = butterfly.center_strike * (1 - price_range_pct)
    max_price = butterfly.center_strike * (1 + price_range_pct)
    prices = np.linspace(min_price, max_price, num_points)
    
    # Create empty results list
    results = []
    
    # Calculate P/L for each price and DTE combination
    for days in days_list:
        for price in prices:
            profit = calculate_profit_at_price(
                butterfly, price, days, volatility, risk_free_rate
            )
            
            results.append({
                "underlying_price": price,
                "days_to_expiration": days,
                "profit_loss": profit
            })
    
    return pd.DataFrame(results)

def calculate_butterfly_probability(
    butterfly: ButterflySpread,
    current_price: float,
    days_to_expiration: int,
    volatility: float
) -> Dict[str, float]:
    """
    Calculate probabilities of profit for a butterfly spread.
    
    Args:
        butterfly: ButterflySpread object
        current_price: Current underlying price
        days_to_expiration: Days to expiration
        volatility: Implied volatility
    
    Returns:
        Dictionary with probability metrics
    """
    # Calculate expected move based on volatility
    expected_move = current_price * volatility * np.sqrt(days_to_expiration / 365)
    
    # Calculate standard deviation of underlying price at expiration
    std_dev = expected_move / 1.645  # 90% confidence interval
    
    # Get breakeven points
    lower_breakeven, upper_breakeven = butterfly.breakeven_points
    
    # Calculate probability using normal distribution
    prob_below_lower = stats.norm.cdf(lower_breakeven, loc=current_price, scale=std_dev)
    prob_above_upper = 1 - stats.norm.cdf(upper_breakeven, loc=current_price, scale=std_dev)
    prob_between = 1 - prob_below_lower - prob_above_upper
    
    # Calculate probability of max profit (approximation)
    # Assuming max profit occurs in a small range around center strike
    center_range = butterfly.inner_wings_width * 0.1  # 10% of wing width
    prob_max_profit = (stats.norm.cdf(butterfly.center_strike + center_range, loc=current_price, scale=std_dev) - 
                       stats.norm.cdf(butterfly.center_strike - center_range, loc=current_price, scale=std_dev))
    
    return {
        "probability_of_profit": prob_between,
        "probability_of_max_profit": prob_max_profit,
        "probability_below_lower_breakeven": prob_below_lower,
        "probability_above_upper_breakeven": prob_above_upper,
        "expected_move": expected_move
    }

def should_adjust_butterfly(
    butterfly: ButterflySpread,
    current_price: float,
    days_to_expiration: int,
    management_threshold: float = 0.10
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Determine if a butterfly position should be adjusted.
    
    Args:
        butterfly: ButterflySpread object
        current_price: Current underlying price
        days_to_expiration: Days to expiration
        management_threshold: Price movement threshold as percentage
        
    Returns:
        Tuple of (should_adjust, adjustment_type, adjustment_details)
    """
    # Calculate how far price has moved from center strike
    distance_pct = abs(current_price - butterfly.center_strike) / butterfly.center_strike
    
    # Check time-based exit
    if days_to_expiration <= 7:
        return (True, "time_exit", {
            "reason": "Position approaching expiration",
            "action": "Close position"
        })
    
    # Check if underlying has moved significantly toward one of the wings
    if distance_pct > management_threshold:
        direction = "upward" if current_price > butterfly.center_strike else "downward"
        wing_to_adjust = "upper_inner" if direction == "upward" else "lower_inner"
        
        # If price moved significantly but still early in trade
        if days_to_expiration > 14:
            return (True, "recenter", {
                "reason": f"Underlying moved {direction} by {distance_pct:.1%}",
                "action": "Roll entire butterfly to new center strike",
                "new_center_strike": round(current_price, 1)
            })
        # If price moved and getting closer to expiration
        else:
            return (True, "adjust_wing", {
                "reason": f"Underlying moved {direction} by {distance_pct:.1%}",
                "action": f"Widen {wing_to_adjust} wing",
                "wing_to_adjust": wing_to_adjust
            })
    
    # Check profit target
    if butterfly.entry_price is not None and butterfly.current_price is not None:
        current_profit = (butterfly.current_price - butterfly.entry_price) * butterfly.multiplier
        max_possible_profit = butterfly.max_profit
        
        if max_possible_profit > 0 and current_profit >= 0.65 * max_possible_profit:
            return (True, "profit_target", {
                "reason": f"Reached profit target ({current_profit / max_possible_profit:.1%} of max profit)",
                "action": "Close position"
            })
    
    # No adjustment needed
    return (False, "", {})

def create_butterfly_order(
    butterfly: ButterflySpread,
    order_type: str = "limit",
    price_buffer_pct: float = 0.05,
    combo_order: bool = True
) -> Dict[str, Any]:
    """
    Create an order specification for a butterfly spread.
    
    Args:
        butterfly: ButterflySpread object
        order_type: Order type (limit, market)
        price_buffer_pct: Buffer to add to limit price (percentage)
        combo_order: Whether to use a combo order or leg into position
        
    Returns:
        Order specification dictionary
    """
    if butterfly.current_price is None:
        logger.error("Cannot create order: butterfly spread hasn't been priced")
        return {}
    
    limit_price = butterfly.current_price
    if order_type == "limit":
        # Add buffer to limit price
        if limit_price > 0:
            limit_price *= (1 + price_buffer_pct)
        else:
            limit_price *= (1 - price_buffer_pct)
    
    if combo_order:
        # Create a single combo order for the entire butterfly
        return {
            "order_type": "butterfly",
            "symbol": butterfly.symbol,
            "expiration": butterfly.expiration.strftime("%Y-%m-%d"),
            "strikes": [
                butterfly.lower_outer_strike,
                butterfly.lower_inner_strike,
                butterfly.center_strike,
                butterfly.upper_inner_strike,
                butterfly.upper_outer_strike
            ],
            "option_type": butterfly.option_type,
            "quantity": butterfly.quantity,
            "order_spec": order_type,
            "limit_price": limit_price if order_type == "limit" else None
        }
    else:
        # Create individual leg orders
        leg_orders = []
        for leg in butterfly.legs:
            leg_orders.append({
                "symbol": leg.symbol,
                "expiration": leg.expiration.strftime("%Y-%m-%d"),
                "strike": leg.strike,
                "option_type": leg.option_type,
                "quantity": leg.quantity,
                "order_spec": order_type
            })
        
        return {
            "order_type": "legged_butterfly",
            "legs": leg_orders,
            "order_sequence": ["inner_longs", "shorts", "outer_longs"],
            "total_price_limit": limit_price if order_type == "limit" else None
        }

# TODO: Implement function to fetch option chains and calculate implied volatility
# TODO: Implement function to backtest butterfly strategy on historical data
# TODO: Implement function to visualize butterfly P&L profile
# TODO: Implement function to convert butterfly to iron fly or condor
# TODO: Implement ML model for predicting optimal butterfly parameters 