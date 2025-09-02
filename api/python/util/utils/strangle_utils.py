"""
Strangle Options Strategy Utilities

This module provides utility functions for analyzing and calculating
metrics for strangle options strategies. A strangle is an options strategy
involving purchasing an out-of-the-money call option and an out-of-the-money
put option with the same expiration date but different strike prices.
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional, Union, Any


def calculate_strangle_cost(call_price: float, put_price: float) -> float:
    """
    Calculate the total cost of a strangle position.
    
    Args:
        call_price: Price of the call option
        put_price: Price of the put option
        
    Returns:
        Total cost of the strangle
    """
    return call_price + put_price


def calculate_strangle_breakeven_points(
    call_strike: float, 
    put_strike: float, 
    call_price: float, 
    put_price: float
) -> Tuple[float, float]:
    """
    Calculate the breakeven points for a strangle position.
    
    Args:
        call_strike: Strike price of the call option
        put_strike: Strike price of the put option
        call_price: Price of the call option
        put_price: Price of the put option
        
    Returns:
        Tuple of (lower_breakeven, upper_breakeven)
    """
    total_premium = call_price + put_price
    lower_breakeven = put_strike - total_premium
    upper_breakeven = call_strike + total_premium
    
    # Lower breakeven can't be negative
    lower_breakeven = max(0, lower_breakeven)
    
    return lower_breakeven, upper_breakeven


def calculate_strangle_max_loss(
    call_price: float, 
    put_price: float, 
    num_contracts: int = 1, 
    contract_multiplier: int = 100
) -> float:
    """
    Calculate the maximum potential loss for a strangle position.
    
    Args:
        call_price: Price of the call option
        put_price: Price of the put option
        num_contracts: Number of contracts
        contract_multiplier: Multiplier for each contract (typically 100)
        
    Returns:
        Maximum potential loss
    """
    return (call_price + put_price) * num_contracts * contract_multiplier


def calculate_strangle_pnl(
    current_price: float,
    call_strike: float,
    put_strike: float,
    call_price: float,
    put_price: float,
    num_contracts: int = 1,
    contract_multiplier: int = 100
) -> float:
    """
    Calculate the profit/loss of a strangle position at a given underlying price.
    
    Args:
        current_price: Current price of the underlying
        call_strike: Strike price of the call option
        put_strike: Strike price of the put option
        call_price: Price paid for the call option
        put_price: Price paid for the put option
        num_contracts: Number of contracts
        contract_multiplier: Multiplier for each contract (typically 100)
        
    Returns:
        Profit/loss amount
    """
    total_premium = call_price + put_price
    
    if current_price > call_strike:
        # Call is in the money
        call_value = current_price - call_strike
        put_value = 0
    elif current_price < put_strike:
        # Put is in the money
        call_value = 0
        put_value = put_strike - current_price
    else:
        # Both options are out of the money
        call_value = 0
        put_value = 0
    
    current_value = call_value + put_value
    profit_loss = (current_value - total_premium) * num_contracts * contract_multiplier
    
    return profit_loss


def find_strangle_strikes(
    option_chain: pd.DataFrame,
    current_price: float, 
    target_delta_call: float = 0.30,
    target_delta_put: float = -0.30,
    tolerance: float = 0.05
) -> Tuple[float, float, Dict, Dict]:
    """
    Find appropriate strike prices for a strangle based on target delta.
    
    Args:
        option_chain: DataFrame containing option chain data
        current_price: Current price of the underlying
        target_delta_call: Target delta for the call option (positive)
        target_delta_put: Target delta for the put option (negative)
        tolerance: Acceptable tolerance range for delta matching
        
    Returns:
        Tuple of (call_strike, put_strike, call_data, put_data)
    """
    # Ensure target deltas have the correct sign
    target_delta_call = abs(target_delta_call)
    target_delta_put = -abs(target_delta_put)
    
    # Filter calls and puts
    calls = option_chain[option_chain["type"] == "call"]
    puts = option_chain[option_chain["type"] == "put"]
    
    # Find closest call to desired delta
    calls_filtered = calls[calls["strike"] > current_price]  # OTM calls
    calls_filtered["delta_diff"] = calls_filtered["delta"].apply(lambda x: abs(x - target_delta_call))
    calls_sorted = calls_filtered.sort_values(by="delta_diff")
    
    if calls_sorted.empty:
        return None, None, {}, {}
    
    # Find closest put to desired delta
    puts_filtered = puts[puts["strike"] < current_price]  # OTM puts
    puts_filtered["delta_diff"] = puts_filtered["delta"].apply(lambda x: abs(x - target_delta_put))
    puts_sorted = puts_filtered.sort_values(by="delta_diff")
    
    if puts_sorted.empty:
        return None, None, {}, {}
    
    # Get best matches
    best_call = calls_sorted.iloc[0]
    best_put = puts_sorted.iloc[0]
    
    # Check if we're within tolerance
    if best_call["delta_diff"] > tolerance or best_put["delta_diff"] > tolerance:
        # If outside tolerance, try to find second best
        if len(calls_sorted) > 1 and len(puts_sorted) > 1:
            second_best_call = calls_sorted.iloc[1]
            second_best_put = puts_sorted.iloc[1]
            
            # See if second choices are better
            if (second_best_call["delta_diff"] < best_call["delta_diff"]):
                best_call = second_best_call
                
            if (second_best_put["delta_diff"] < best_put["delta_diff"]):
                best_put = second_best_put
    
    call_strike = best_call["strike"]
    put_strike = best_put["strike"]
    
    # Convert the Series to dict for easier access
    call_data = best_call.to_dict()
    put_data = best_put.to_dict()
    
    return call_strike, put_strike, call_data, put_data


def find_strangle_by_otm_pct(
    option_chain: pd.DataFrame,
    current_price: float, 
    target_otm_pct_call: float = 0.10,
    target_otm_pct_put: float = 0.10,
    tolerance: float = 0.02
) -> Tuple[float, float, Dict, Dict]:
    """
    Find appropriate strike prices for a strangle based on target out-of-the-money percentage.
    
    Args:
        option_chain: DataFrame containing option chain data
        current_price: Current price of the underlying
        target_otm_pct_call: Target OTM percentage for call (e.g., 0.10 for 10% OTM)
        target_otm_pct_put: Target OTM percentage for put (e.g., 0.10 for 10% OTM)
        tolerance: Acceptable tolerance range for percentage matching
        
    Returns:
        Tuple of (call_strike, put_strike, call_data, put_data)
    """
    # Calculate target strikes
    target_call_strike = current_price * (1 + target_otm_pct_call)
    target_put_strike = current_price * (1 - target_otm_pct_put)
    
    # Filter calls and puts
    calls = option_chain[option_chain["type"] == "call"]
    puts = option_chain[option_chain["type"] == "put"]
    
    # Find closest strikes to targets
    calls["strike_diff"] = calls["strike"].apply(lambda x: abs(x - target_call_strike))
    calls_sorted = calls.sort_values(by="strike_diff")
    
    puts["strike_diff"] = puts["strike"].apply(lambda x: abs(x - target_put_strike))
    puts_sorted = puts.sort_values(by="strike_diff")
    
    if calls_sorted.empty or puts_sorted.empty:
        return None, None, {}, {}
    
    # Get best matches
    best_call = calls_sorted.iloc[0]
    best_put = puts_sorted.iloc[0]
    
    # Calculate actual OTM percentages
    call_otm_pct = (best_call["strike"] / current_price) - 1
    put_otm_pct = 1 - (best_put["strike"] / current_price)
    
    # Check if we're within tolerance
    if (abs(call_otm_pct - target_otm_pct_call) > tolerance or 
        abs(put_otm_pct - target_otm_pct_put) > tolerance):
        # Try to find better matches within tolerance
        for i in range(min(3, len(calls_sorted))):
            call = calls_sorted.iloc[i]
            call_otm = (call["strike"] / current_price) - 1
            if abs(call_otm - target_otm_pct_call) <= tolerance:
                best_call = call
                break
                
        for i in range(min(3, len(puts_sorted))):
            put = puts_sorted.iloc[i]
            put_otm = 1 - (put["strike"] / current_price)
            if abs(put_otm - target_otm_pct_put) <= tolerance:
                best_put = put
                break
    
    call_strike = best_call["strike"]
    put_strike = best_put["strike"]
    
    # Convert the Series to dict for easier access
    call_data = best_call.to_dict()
    put_data = best_put.to_dict()
    
    return call_strike, put_strike, call_data, put_data


def calculate_expected_move(
    current_price: float,
    implied_volatility: float,
    days_to_expiration: int
) -> float:
    """
    Calculate the expected move of the underlying based on implied volatility.
    
    Args:
        current_price: Current price of the underlying
        implied_volatility: Implied volatility (as a decimal, e.g., 0.30 for 30%)
        days_to_expiration: Number of days to expiration
        
    Returns:
        Expected price movement (in dollars)
    """
    # Convert annual IV to the period
    volatility_period = implied_volatility * math.sqrt(days_to_expiration / 365)
    
    # Expected move (1 standard deviation)
    expected_move = current_price * volatility_period
    
    return expected_move


def calculate_probability_of_profit(
    current_price: float,
    lower_breakeven: float,
    upper_breakeven: float,
    implied_volatility: float,
    days_to_expiration: int
) -> float:
    """
    Calculate the probability of profit for a strangle position.
    
    Args:
        current_price: Current price of the underlying
        lower_breakeven: Lower breakeven point
        upper_breakeven: Upper breakeven point
        implied_volatility: Implied volatility (as a decimal)
        days_to_expiration: Number of days to expiration
        
    Returns:
        Probability of profit (as a percentage)
    """
    # Convert annual volatility to the period
    volatility_period = implied_volatility * math.sqrt(days_to_expiration / 365)
    
    # Calculate z-scores for breakeven points
    z_score_lower = (math.log(lower_breakeven / current_price)) / volatility_period if lower_breakeven > 0 else -999
    z_score_upper = (math.log(upper_breakeven / current_price)) / volatility_period
    
    # Calculate probability
    prob_below_lower = norm.cdf(z_score_lower)
    prob_above_upper = 1 - norm.cdf(z_score_upper)
    
    # Probability of profit is the sum of probabilities outside the breakeven range
    probability_of_profit = (prob_below_lower + prob_above_upper) * 100
    
    return probability_of_profit


def calculate_strangle_metrics(
    current_price: float,
    call_strike: float,
    put_strike: float,
    call_price: float,
    put_price: float,
    days_to_expiration: int,
    implied_volatility: float
) -> Dict:
    """
    Calculate comprehensive metrics for a strangle position.
    
    Args:
        current_price: Current price of the underlying
        call_strike: Strike price of the call option
        put_strike: Strike price of the put option
        call_price: Price of the call option
        put_price: Price of the put option
        days_to_expiration: Number of days to expiration
        implied_volatility: Average implied volatility (as a decimal)
        
    Returns:
        Dictionary containing calculated metrics
    """
    # Calculate basic metrics
    total_premium = call_price + put_price
    lower_breakeven, upper_breakeven = calculate_strangle_breakeven_points(
        call_strike, put_strike, call_price, put_price
    )
    
    # Calculate expected move
    expected_move = calculate_expected_move(
        current_price, implied_volatility, days_to_expiration
    )
    
    # Calculate probability of profit
    pop = calculate_probability_of_profit(
        current_price, lower_breakeven, upper_breakeven, implied_volatility, days_to_expiration
    )
    
    # Calculate maximum profit potential (assuming typical 75% target)
    max_profit_potential = total_premium * 0.75
    
    # Calculate risk/reward ratio
    risk = total_premium  # Maximum risk is the premium paid
    reward = max_profit_potential
    risk_reward_ratio = reward / risk if risk > 0 else 0
    
    # Calculate width between strikes as percentage of current price
    strike_width_pct = abs(call_strike - put_strike) / current_price * 100
    
    # Calculate premium as percentage of current price
    premium_pct = total_premium / current_price * 100
    
    return {
        "total_premium": total_premium,
        "premium_pct": premium_pct,
        "max_loss": total_premium * 100,  # Per contract
        "max_profit_potential": max_profit_potential * 100,  # Per contract
        "breakeven_lower": lower_breakeven,
        "breakeven_upper": upper_breakeven,
        "breakeven_range_pct": (upper_breakeven - lower_breakeven) / current_price * 100,
        "expected_move": expected_move,
        "expected_move_pct": expected_move / current_price * 100,
        "probability_of_profit": pop,
        "risk_reward_ratio": risk_reward_ratio,
        "strike_width": call_strike - put_strike,
        "strike_width_pct": strike_width_pct,
        "days_to_expiration": days_to_expiration,
        "implied_volatility": implied_volatility * 100  # Convert to percentage
    }


def score_strangle_opportunity(
    iv_rank: float,
    probability_of_profit: float,
    risk_reward_ratio: float,
    liquidity_score: float,
    days_to_expiration: int,
    event_score: float = 0,
    vega_capture: float = 0,
    has_recent_gaps: bool = False,
    scoring_weights: Dict = None
) -> float:
    """
    Calculate a score for a strangle opportunity based on various factors.
    
    Args:
        iv_rank: Implied volatility rank (0-100)
        probability_of_profit: Probability of profit (0-100)
        risk_reward_ratio: Risk/reward ratio
        liquidity_score: Liquidity score (0-100)
        days_to_expiration: Number of days to expiration
        event_score: Score for upcoming events (0-100)
        vega_capture: Vega capture ratio
        has_recent_gaps: Whether the underlying has had recent price gaps
        scoring_weights: Optional custom weights for each factor
        
    Returns:
        Opportunity score (0-100)
    """
    # Default weights
    if scoring_weights is None:
        scoring_weights = {
            "iv_rank": 0.20,
            "probability_of_profit": 0.20,
            "risk_reward": 0.15,
            "liquidity": 0.15,
            "days_to_expiration": 0.10,
            "event_score": 0.10,
            "vega_capture": 0.05,
            "recent_gaps": 0.05
        }
    
    # Normalize each factor
    
    # IV Rank: Higher is better for strangles (more potential movement)
    iv_rank_score = min(100, iv_rank * 1.2)  # Boost IV rank slightly
    
    # Probability of profit: Higher is better
    pop_score = probability_of_profit
    
    # Risk/reward: Higher is better, cap at 3
    rr_score = min(100, risk_reward_ratio * 33.33)
    
    # Liquidity: Higher is better
    liquidity_score = min(100, liquidity_score)
    
    # Days to expiration: Score based on ideal range
    if days_to_expiration < 14:
        dte_score = days_to_expiration * 5  # Too short, lower score
    elif 14 <= days_to_expiration <= 45:
        dte_score = 100  # Ideal range
    else:
        dte_score = max(0, 100 - (days_to_expiration - 45))  # Decrease for longer durations
        
    # Vega capture: Higher is better for strangles
    vega_score = min(100, vega_capture * 100)
    
    # Recent gaps: Gaps can be favorable for strangles
    gap_score = 70 if has_recent_gaps else 40
    
    # Calculate weighted score
    weighted_score = (
        iv_rank_score * scoring_weights["iv_rank"] +
        pop_score * scoring_weights["probability_of_profit"] +
        rr_score * scoring_weights["risk_reward"] +
        liquidity_score * scoring_weights["liquidity"] +
        dte_score * scoring_weights["days_to_expiration"] +
        event_score * scoring_weights["event_score"] +
        vega_score * scoring_weights["vega_capture"] +
        gap_score * scoring_weights["recent_gaps"]
    )
    
    return round(weighted_score, 1)


def calculate_liquidity_score(
    open_interest: int,
    volume: int,
    bid_ask_spread: float
) -> float:
    """
    Calculate a liquidity score for an option based on open interest, volume, and spread.
    
    Args:
        open_interest: Open interest of the option
        volume: Trading volume of the option
        bid_ask_spread: Bid-ask spread as a percentage of option price
        
    Returns:
        Liquidity score (0-100)
    """
    # Normalize open interest
    oi_score = min(100, open_interest / 50)  # Scale OI, max at 5000
    
    # Normalize volume
    volume_score = min(100, volume / 20)  # Scale volume, max at 2000
    
    # Normalize bid-ask spread
    if bid_ask_spread <= 0.01:  # 1% or less
        spread_score = 100
    elif bid_ask_spread <= 0.03:  # 3% or less
        spread_score = 80
    elif bid_ask_spread <= 0.05:  # 5% or less
        spread_score = 60
    elif bid_ask_spread <= 0.10:  # 10% or less
        spread_score = 40
    else:
        spread_score = max(0, 100 - (bid_ask_spread * 1000))
    
    # Weighted average
    liquidity_score = (oi_score * 0.3) + (volume_score * 0.3) + (spread_score * 0.4)
    
    return round(liquidity_score, 1)


def calculate_vega_capture_ratio(
    call_vega: float,
    put_vega: float,
    call_price: float,
    put_price: float
) -> float:
    """
    Calculate the vega capture ratio for a strangle.
    Vega capture ratio represents how much option premium is attributable to implied volatility.
    Higher values indicate the position captures more volatility premium.
    
    Args:
        call_vega: Vega of the call option
        put_vega: Vega of the put option
        call_price: Price of the call option
        put_price: Price of the put option
        
    Returns:
        Vega capture ratio (0-1)
    """
    total_vega = call_vega + put_vega
    total_premium = call_price + put_price
    
    if total_premium <= 0:
        return 0
    
    # Scale vega to match premium scale
    vega_capture = total_vega / (total_premium * 100)  # Adjust scaling
    
    # Normalize between 0 and 1
    return min(1.0, max(0.0, vega_capture))


def check_recent_gaps(
    price_history: pd.DataFrame,
    days_back: int = 20,
    gap_threshold_pct: float = 2.0
) -> List[Dict]:
    """
    Check for recent price gaps in the underlying's price history.
    
    Args:
        price_history: DataFrame with price history (must have Open, Close columns)
        days_back: Number of days to look back
        gap_threshold_pct: Threshold for considering a gap significant (percentage)
        
    Returns:
        List of dictionaries with gap information
    """
    if len(price_history) < days_back:
        return []
    
    recent_data = price_history.iloc[-days_back:].copy()
    gaps = []
    
    for i in range(1, len(recent_data)):
        prev_close = recent_data.iloc[i-1]["Close"]
        current_open = recent_data.iloc[i]["Open"]
        
        gap_pct = abs(current_open - prev_close) / prev_close * 100
        
        if gap_pct >= gap_threshold_pct:
            gap_info = {
                "date": recent_data.index[i].strftime("%Y-%m-%d") if hasattr(recent_data.index[i], "strftime") else str(recent_data.index[i]),
                "prev_close": prev_close,
                "open": current_open,
                "gap_pct": gap_pct,
                "gap_direction": "up" if current_open > prev_close else "down"
            }
            gaps.append(gap_info)
    
    return gaps


def analyze_event_outcome(
    price_history: pd.DataFrame,
    event_date: str,
    expected_move: float,
    window_days: int = 3
) -> Dict:
    """
    Analyze the outcome of an event relative to expected move.
    
    Args:
        price_history: DataFrame with price history
        event_date: Date of the event
        expected_move: Expected move derived from implied volatility
        window_days: Number of days after event to analyze
        
    Returns:
        Dictionary with analysis results
    """
    # Convert event_date to datetime if it's a string
    if isinstance(event_date, str):
        event_date = pd.to_datetime(event_date)
    
    # Find the price before the event
    pre_event_price = price_history[price_history.index < event_date].iloc[-1]["Close"]
    
    # Find prices after the event within the window
    post_event_data = price_history[
        (price_history.index >= event_date) & 
        (price_history.index <= event_date + pd.Timedelta(days=window_days))
    ]
    
    if post_event_data.empty:
        return {
            "event_date": event_date,
            "pre_event_price": pre_event_price,
            "has_post_event_data": False
        }
    
    # Calculate maximum move in either direction
    post_event_prices = post_event_data["Close"].tolist()
    max_price = max(post_event_prices)
    min_price = min(post_event_prices)
    
    max_up_move = max_price - pre_event_price
    max_down_move = pre_event_price - min_price
    
    max_move = max(max_up_move, max_down_move)
    max_move_pct = max_move / pre_event_price * 100
    
    # Compare to expected move
    expected_move_pct = expected_move / pre_event_price * 100
    move_ratio = max_move_pct / expected_move_pct if expected_move_pct > 0 else 0
    
    # Final price compared to pre-event
    last_price = post_event_data.iloc[-1]["Close"]
    final_move = last_price - pre_event_price
    final_move_pct = final_move / pre_event_price * 100
    
    return {
        "event_date": event_date,
        "pre_event_price": pre_event_price,
        "max_price": max_price,
        "min_price": min_price,
        "last_price": last_price,
        "max_up_move": max_up_move,
        "max_down_move": max_down_move,
        "max_move": max_move,
        "max_move_pct": max_move_pct,
        "expected_move": expected_move,
        "expected_move_pct": expected_move_pct,
        "move_ratio": move_ratio,
        "final_move": final_move,
        "final_move_pct": final_move_pct,
        "exceeded_expected_move": max_move_pct > expected_move_pct
    } 