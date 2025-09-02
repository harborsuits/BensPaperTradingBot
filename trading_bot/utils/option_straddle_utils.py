"""
Utility functions for straddle option strategy analysis and calculations.

This module provides tools for evaluating straddle opportunities,
calculating breakeven points, expected returns, and other metrics
specific to the straddle options strategy.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from trading_bot.utils.options_utils import calculate_option_metrics
from trading_bot.utils.volatility_utils import calculate_iv_rank, calculate_expected_move


def calculate_straddle_cost(
    call_price: float, 
    put_price: float, 
    contracts: int = 1
) -> float:
    """
    Calculate the total cost of a straddle position.
    
    Args:
        call_price: Price of the call option
        put_price: Price of the put option
        contracts: Number of contracts (each contract = 100 shares)
        
    Returns:
        Total cost of the straddle position
    """
    return (call_price + put_price) * 100 * contracts


def calculate_straddle_breakeven_points(
    strike_price: float, 
    call_price: float, 
    put_price: float
) -> Tuple[float, float]:
    """
    Calculate the breakeven points for a straddle.
    
    Args:
        strike_price: The strike price for both options
        call_price: Price of the call option
        put_price: Price of the put option
        
    Returns:
        Tuple containing (lower_breakeven, upper_breakeven)
    """
    total_premium = call_price + put_price
    lower_breakeven = strike_price - total_premium
    upper_breakeven = strike_price + total_premium
    
    return (max(0, lower_breakeven), upper_breakeven)


def calculate_straddle_profit(
    strike_price: float,
    call_price: float,
    put_price: float,
    current_price: float
) -> float:
    """
    Calculate the current profit/loss of a straddle position.
    
    Args:
        strike_price: The strike price for both options
        call_price: Entry price of the call option
        put_price: Entry price of the put option
        current_price: Current price of the underlying asset
        
    Returns:
        Current profit/loss of the straddle position per contract
    """
    total_premium = (call_price + put_price) * 100
    
    if current_price > strike_price:
        # Call is in the money
        call_value = (current_price - strike_price) * 100
        put_value = 0  # Put is out of the money
    else:
        # Put is in the money
        put_value = (strike_price - current_price) * 100
        call_value = 0  # Call is out of the money
    
    return call_value + put_value - total_premium


def calculate_straddle_profit_potential(
    strike_price: float,
    call_price: float,
    put_price: float,
    expected_move: float
) -> Tuple[float, float]:
    """
    Calculate profit potential for a straddle based on expected move.
    
    Args:
        strike_price: The strike price for both options
        call_price: Price of the call option
        put_price: Price of the put option
        expected_move: Expected price movement (as a decimal, e.g., 0.10 for 10%)
        
    Returns:
        Tuple of (profit_potential_pct, probability_of_profit)
    """
    total_premium = call_price + put_price
    breakeven_move_pct = total_premium / strike_price
    
    # If expected move exceeds breakeven move, there's profit potential
    if expected_move > breakeven_move_pct:
        # Simple estimation of profit potential
        profit_potential_pct = (expected_move - breakeven_move_pct) / breakeven_move_pct * 100
        
        # Rough probability estimation based on normal distribution
        # This is simplified and assumes the move follows a normal distribution
        probability_of_profit = (expected_move / breakeven_move_pct - 1) * 50
        probability_of_profit = min(max(probability_of_profit, 0), 100)
    else:
        profit_potential_pct = 0
        probability_of_profit = 0
    
    return (profit_potential_pct, probability_of_profit)


def evaluate_straddle_opportunity(
    symbol: str,
    current_price: float,
    strike_price: float,
    call_option: Dict,
    put_option: Dict,
    days_to_expiration: int,
    iv_rank: float,
    historical_volatility: float,
    entry_criteria: Dict
) -> Dict:
    """
    Evaluate a potential straddle opportunity and return a comprehensive analysis.
    
    Args:
        symbol: The ticker symbol
        current_price: Current price of the underlying
        strike_price: Strike price for the straddle
        call_option: Dictionary containing call option data
        put_option: Dictionary containing put option data
        days_to_expiration: Days until option expiration
        iv_rank: Current IV rank (0-100)
        historical_volatility: Historical volatility (annual, decimal)
        entry_criteria: Dictionary of criteria for entry evaluation
        
    Returns:
        Dictionary containing the analysis results and an overall score
    """
    call_price = (call_option['ask'] + call_option['bid']) / 2
    put_price = (put_option['ask'] + put_option['bid']) / 2
    
    # Calculate straddle metrics
    total_cost = calculate_straddle_cost(call_price, put_price)
    breakevens = calculate_straddle_breakeven_points(strike_price, call_price, put_price)
    breakeven_move_pct = (call_price + put_price) / strike_price * 100
    
    # Calculate expected move based on implied volatility
    implied_volatility = (call_option['implied_volatility'] + put_option['implied_volatility']) / 2
    expected_move_pct = calculate_expected_move(implied_volatility, days_to_expiration)
    
    # Calculate profit metrics
    profit_metrics = calculate_straddle_profit_potential(
        strike_price, call_price, put_price, expected_move_pct / 100
    )
    
    # Liquidity check
    call_spread_pct = (call_option['ask'] - call_option['bid']) / call_price * 100
    put_spread_pct = (put_option['ask'] - put_option['bid']) / put_price * 100
    avg_spread_pct = (call_spread_pct + put_spread_pct) / 2
    
    # Score calculation (0-100)
    score_components = []
    
    # IV Rank score (higher is better for straddles)
    iv_rank_score = min(100, iv_rank * 1.5)
    score_components.append(iv_rank_score * 0.25)  # 25% weight
    
    # Expected move vs breakeven move score
    move_ratio = expected_move_pct / breakeven_move_pct
    move_score = min(100, move_ratio * 50)
    score_components.append(move_score * 0.30)  # 30% weight
    
    # Liquidity score (lower spread is better)
    liquidity_score = max(0, 100 - avg_spread_pct * 10)
    score_components.append(liquidity_score * 0.20)  # 20% weight
    
    # IV/HV ratio score (higher is better for straddles)
    iv_hv_ratio = implied_volatility / historical_volatility
    iv_hv_score = min(100, iv_hv_ratio * 50)
    score_components.append(iv_hv_score * 0.25)  # 25% weight
    
    # Overall score
    overall_score = sum(score_components)
    
    # Create result dictionary
    result = {
        "symbol": symbol,
        "current_price": current_price,
        "strike_price": strike_price,
        "days_to_expiration": days_to_expiration,
        "call_price": call_price,
        "put_price": put_price,
        "total_cost": total_cost,
        "breakeven_points": breakevens,
        "breakeven_move_pct": breakeven_move_pct,
        "implied_volatility": implied_volatility,
        "iv_rank": iv_rank,
        "historical_volatility": historical_volatility,
        "iv_hv_ratio": iv_hv_ratio,
        "expected_move_pct": expected_move_pct,
        "expected_vs_breakeven_ratio": move_ratio,
        "avg_bid_ask_spread_pct": avg_spread_pct,
        "profit_potential_pct": profit_metrics[0],
        "probability_of_profit": profit_metrics[1],
        "score": overall_score,
        "pass_criteria": overall_score >= entry_criteria["min_score"] and 
                        avg_spread_pct <= entry_criteria["max_bid_ask_spread_pct"] and
                        min(call_option['volume'], put_option['volume']) >= entry_criteria["min_option_volume"] and
                        iv_rank >= entry_criteria["min_iv_rank"] and
                        iv_hv_ratio >= entry_criteria.get("min_iv_hv_ratio", 1.0)
    }
    
    return result 