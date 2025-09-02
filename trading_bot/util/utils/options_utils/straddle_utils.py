"""
Utility functions for straddle options strategy calculations.

This module provides functions for calculating straddle-related metrics,
including breakeven points, potential profit/loss, probability of profit,
and strategy scoring.
"""

import math
import numpy as np
from typing import Dict, Tuple, List, Optional

from trading_bot.utils.options_utils.pricing import calculate_option_greeks
from trading_bot.utils.options_utils.probability import calculate_probability_above_price, calculate_probability_below_price
from trading_bot.utils.volatility import get_expected_move


def calculate_straddle_cost(call_price: float, put_price: float) -> float:
    """
    Calculate the total cost of a straddle position.
    
    Args:
        call_price: Price of the call option
        put_price: Price of the put option
        
    Returns:
        Total cost of the straddle position
    """
    return call_price + put_price


def calculate_straddle_breakeven_points(strike_price: float, straddle_cost: float) -> Tuple[float, float]:
    """
    Calculate the upper and lower breakeven points for a straddle.
    
    Args:
        strike_price: Strike price of both options
        straddle_cost: Total cost of the straddle position
        
    Returns:
        Tuple of (lower_breakeven, upper_breakeven)
    """
    lower_breakeven = strike_price - straddle_cost
    upper_breakeven = strike_price + straddle_cost
    
    # Ensure lower_breakeven is not negative (would be 0 for practical purposes)
    lower_breakeven = max(0, lower_breakeven)
    
    return lower_breakeven, upper_breakeven


def calculate_straddle_profit(
    current_price: float, 
    strike_price: float, 
    straddle_cost: float
) -> float:
    """
    Calculate the profit/loss of a straddle position at a given current price.
    
    Args:
        current_price: Current price of the underlying asset
        strike_price: Strike price of both options
        straddle_cost: Total cost of the straddle position
        
    Returns:
        Profit or loss as a dollar amount
    """
    intrinsic_value = abs(current_price - strike_price)
    return intrinsic_value - straddle_cost


def calculate_straddle_max_loss(straddle_cost: float) -> float:
    """
    Calculate the maximum possible loss for a straddle position.
    
    Args:
        straddle_cost: Total cost of the straddle position
        
    Returns:
        Maximum possible loss (always the initial cost for a long straddle)
    """
    return straddle_cost


def calculate_probability_of_profit(
    underlying_price: float,
    strike_price: float,
    straddle_cost: float,
    implied_volatility: float,
    days_to_expiration: float
) -> float:
    """
    Calculate the probability of profit for a straddle position.
    
    Args:
        underlying_price: Current price of the underlying asset
        strike_price: Strike price of both options
        straddle_cost: Total cost of the straddle position
        implied_volatility: Annualized implied volatility as a decimal
        days_to_expiration: Number of days until option expiration
        
    Returns:
        Probability of profit as a percentage (0-100)
    """
    lower_breakeven, upper_breakeven = calculate_straddle_breakeven_points(strike_price, straddle_cost)
    
    # Calculate probability of being below lower breakeven
    prob_below = calculate_probability_below_price(
        underlying_price=underlying_price,
        target_price=lower_breakeven,
        implied_volatility=implied_volatility,
        days=days_to_expiration
    )
    
    # Calculate probability of being above upper breakeven
    prob_above = calculate_probability_above_price(
        underlying_price=underlying_price,
        target_price=upper_breakeven,
        implied_volatility=implied_volatility,
        days=days_to_expiration
    )
    
    # Probability of being outside the breakeven range (= probability of profit)
    prob_of_profit = prob_below + prob_above
    
    return prob_of_profit * 100  # Convert to percentage


def calculate_expected_profit(
    underlying_price: float,
    strike_price: float,
    straddle_cost: float,
    implied_volatility: float,
    days_to_expiration: float,
    num_price_points: int = 100
) -> float:
    """
    Calculate the expected profit for a straddle position based on
    the probability distribution of the underlying price at expiration.
    
    Args:
        underlying_price: Current price of the underlying asset
        strike_price: Strike price of both options
        straddle_cost: Total cost of the straddle position
        implied_volatility: Annualized implied volatility as a decimal
        days_to_expiration: Number of days until option expiration
        num_price_points: Number of price points to use in calculation
        
    Returns:
        Expected profit as a percentage of the initial investment
    """
    expected_move = get_expected_move(
        price=underlying_price,
        volatility=implied_volatility,
        days=days_to_expiration
    )
    
    # Create a range of possible prices at expiration
    width = 3 * expected_move  # 3 standard deviations
    min_price = max(0, underlying_price - width)
    max_price = underlying_price + width
    
    price_points = np.linspace(min_price, max_price, num_price_points)
    
    total_weighted_profit = 0
    total_probability = 0
    
    annual_volatility = implied_volatility
    daily_volatility = annual_volatility / math.sqrt(252)
    expiration_volatility = daily_volatility * math.sqrt(days_to_expiration)
    
    # Calculate the expected profit at each potential price point
    for price in price_points:
        # Calculate the log return
        log_return = math.log(price / underlying_price)
        
        # Calculate the probability density at this price
        prob_density = (1 / (price * expiration_volatility * math.sqrt(2 * math.pi))) * \
                       math.exp(-(log_return + (expiration_volatility**2)/2)**2 / (2 * expiration_volatility**2))
        
        # Calculate the profit at this price
        profit = calculate_straddle_profit(price, strike_price, straddle_cost)
        
        # Add the weighted profit to the total
        total_weighted_profit += profit * prob_density
        total_probability += prob_density
    
    # Normalize by the total probability
    if total_probability > 0:
        expected_profit = total_weighted_profit / total_probability
    else:
        expected_profit = 0
    
    # Return as a percentage of initial cost
    return (expected_profit / straddle_cost) * 100


def score_straddle_opportunity(
    underlying_price: float,
    strike_price: float,
    call_price: float,
    put_price: float,
    implied_volatility: float,
    historical_volatility: float,
    days_to_expiration: float,
    iv_rank: float,
    option_volume: int,
    bid_ask_spread_pct: float,
    weight_factors: Dict[str, float] = None
) -> float:
    """
    Score a potential straddle trade opportunity on a scale of 0-100.
    
    Args:
        underlying_price: Current price of the underlying asset
        strike_price: Strike price of both options
        call_price: Price of the call option
        put_price: Price of the put option
        implied_volatility: Annualized implied volatility as a decimal
        historical_volatility: Annualized historical volatility as a decimal
        days_to_expiration: Number of days until option expiration
        iv_rank: Implied volatility rank (0-100)
        option_volume: Combined volume of call and put options
        bid_ask_spread_pct: Bid-ask spread as a percentage of midpoint price
        weight_factors: Dictionary of weight factors for each component score
        
    Returns:
        Overall score for the opportunity (0-100)
    """
    if weight_factors is None:
        weight_factors = {
            "iv_rank": 0.3,
            "breakeven_width": 0.2,
            "expected_profit": 0.2,
            "probability_of_profit": 0.2,
            "liquidity": 0.1
        }
    
    # Calculate straddle metrics
    straddle_cost = calculate_straddle_cost(call_price, put_price)
    lower_breakeven, upper_breakeven = calculate_straddle_breakeven_points(strike_price, straddle_cost)
    breakeven_width = (upper_breakeven - lower_breakeven) / underlying_price * 100  # As percentage of underlying
    
    # Calculate expected profit and probability of profit
    expected_profit_pct = calculate_expected_profit(
        underlying_price=underlying_price,
        strike_price=strike_price,
        straddle_cost=straddle_cost,
        implied_volatility=implied_volatility,
        days_to_expiration=days_to_expiration
    )
    
    prob_of_profit = calculate_probability_of_profit(
        underlying_price=underlying_price,
        strike_price=strike_price,
        straddle_cost=straddle_cost,
        implied_volatility=implied_volatility,
        days_to_expiration=days_to_expiration
    )
    
    # Component scores (0-100 scale)
    iv_rank_score = min(100, iv_rank)
    
    # For breakeven width, a smaller width is better (normalized to 0-100)
    breakeven_width_score = max(0, 100 - breakeven_width)
    
    # Expected profit, higher is better
    expected_profit_score = min(100, expected_profit_pct * 1.5)  # Scale: 67% expected profit = 100 score
    
    # Probability of profit, higher is better
    probability_score = min(100, prob_of_profit * 1.5)  # Scale: 67% probability = 100 score
    
    # Liquidity score based on volume and bid-ask spread
    volume_score = min(100, option_volume / 5)  # Scale: 500 volume = 100 score
    spread_score = max(0, 100 - bid_ask_spread_pct * 10)  # Scale: 0% spread = 100, 10% spread = 0
    liquidity_score = (volume_score + spread_score) / 2
    
    # Combine scores using weight factors
    overall_score = (
        weight_factors["iv_rank"] * iv_rank_score +
        weight_factors["breakeven_width"] * breakeven_width_score +
        weight_factors["expected_profit"] * expected_profit_score +
        weight_factors["probability_of_profit"] * probability_score +
        weight_factors["liquidity"] * liquidity_score
    )
    
    return round(overall_score, 1)


def find_best_straddle_expiration(
    underlying_price: float,
    strike_price: float,
    expirations: List[Dict],
    entry_criteria: Dict,
    weight_factors: Optional[Dict[str, float]] = None
) -> Optional[Dict]:
    """
    Find the best expiration date for a straddle strategy based on scoring.
    
    Args:
        underlying_price: Current price of the underlying asset
        strike_price: Strike price to evaluate
        expirations: List of expiration data dictionaries
        entry_criteria: Dictionary of entry criteria
        weight_factors: Optional dictionary of weight factors for scoring
        
    Returns:
        Dictionary with the best expiration data or None if no expirations meet criteria
    """
    best_score = 0
    best_expiration = None
    
    for expiration in expirations:
        call_price = expiration.get("call_price")
        put_price = expiration.get("put_price")
        implied_volatility = expiration.get("implied_volatility")
        historical_volatility = expiration.get("historical_volatility")
        days_to_expiration = expiration.get("days_to_expiration")
        iv_rank = expiration.get("iv_rank")
        option_volume = expiration.get("option_volume", 0)
        bid_ask_spread_pct = expiration.get("bid_ask_spread_pct", 0)
        
        # Skip if any required data is missing
        if None in (call_price, put_price, implied_volatility, days_to_expiration, iv_rank):
            continue
        
        # Skip if doesn't meet minimum criteria
        if (iv_rank < entry_criteria.get("min_iv_rank", 0) or
            option_volume < entry_criteria.get("min_option_volume", 0) or
            bid_ask_spread_pct > entry_criteria.get("max_bid_ask_spread_pct", 100)):
            continue
        
        # Score the opportunity
        score = score_straddle_opportunity(
            underlying_price=underlying_price,
            strike_price=strike_price,
            call_price=call_price,
            put_price=put_price,
            implied_volatility=implied_volatility,
            historical_volatility=historical_volatility or implied_volatility,
            days_to_expiration=days_to_expiration,
            iv_rank=iv_rank,
            option_volume=option_volume,
            bid_ask_spread_pct=bid_ask_spread_pct,
            weight_factors=weight_factors or entry_criteria.get("weight_factors")
        )
        
        # Update best expiration if this one has a better score
        if score > best_score and score >= entry_criteria.get("min_score", 0):
            best_score = score
            best_expiration = {**expiration, "score": score}
    
    return best_expiration 