"""
Utility functions for straddle options strategy calculations.

This module provides helper functions for analyzing, evaluating, and
calculating metrics specific to the straddle options strategy.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def calculate_breakeven_points(strike_price, total_premium):
    """
    Calculate the breakeven points for a straddle position.
    
    Args:
        strike_price (float): The strike price of both the call and put options
        total_premium (float): The total premium paid for both options
        
    Returns:
        tuple: Upper and lower breakeven points (upper_be, lower_be)
    """
    upper_be = strike_price + total_premium
    lower_be = strike_price - total_premium
    if lower_be < 0:
        lower_be = 0
    return (upper_be, lower_be)

def calculate_profit_probability(
    strike_price, 
    current_price, 
    days_to_expiration, 
    iv, 
    premium
):
    """
    Calculate the probability of profit for a straddle position.
    
    Args:
        strike_price (float): The strike price of the straddle
        current_price (float): Current price of the underlying
        days_to_expiration (int): Number of days until expiration
        iv (float): Implied volatility as a decimal (e.g., 0.30 for 30%)
        premium (float): Total premium paid for the straddle
        
    Returns:
        float: Probability of profit as a decimal
    """
    # Calculate breakeven points
    upper_be, lower_be = calculate_breakeven_points(strike_price, premium)
    
    # Convert annual IV to the period of the option
    period_volatility = iv * np.sqrt(days_to_expiration / 365)
    
    # Calculate the probability using the normal distribution
    upper_z = np.log(upper_be / current_price) / period_volatility
    lower_z = np.log(lower_be / current_price) / period_volatility if lower_be > 0 else -999
    
    # Probability is outside the breakeven range
    prob_above = 1 - norm.cdf(upper_z)
    prob_below = norm.cdf(lower_z) if lower_be > 0 else 0
    
    return prob_above + prob_below

def evaluate_straddle_opportunity(
    symbol,
    current_price,
    strike_price,
    call_price,
    put_price,
    days_to_expiration,
    iv_rank,
    historical_volatility,
    avg_true_range,
    iv_percentile,
    upcoming_events=None
):
    """
    Evaluate a potential straddle opportunity and return a score and analysis.
    
    Args:
        symbol (str): The ticker symbol
        current_price (float): Current price of the underlying
        strike_price (float): Strike price of the straddle
        call_price (float): Premium of the call option
        put_price (float): Premium of the put option
        days_to_expiration (int): Days until expiration
        iv_rank (float): Current IV rank (0-100)
        historical_volatility (float): Historical volatility as decimal
        avg_true_range (float): Average True Range of the underlying
        iv_percentile (float): IV percentile (0-100)
        upcoming_events (dict, optional): Dictionary of upcoming events
        
    Returns:
        tuple: (score, analysis_dict) where score is 0-100 and analysis_dict contains the breakdown
    """
    total_premium = call_price + put_price
    premium_percent = (total_premium / current_price) * 100
    
    # Calculate breakevens
    upper_be, lower_be = calculate_breakeven_points(strike_price, total_premium)
    
    # Calculate required move for breakeven
    required_move_pct = (total_premium / current_price) * 100
    
    # Calculate probability of profit
    prob_profit = calculate_profit_probability(
        strike_price, current_price, days_to_expiration, 
        historical_volatility, total_premium
    )
    
    # Base score starts at 50
    score = 50
    
    analysis = {
        "symbol": symbol,
        "current_price": current_price,
        "strike_price": strike_price,
        "call_price": call_price,
        "put_price": put_price,
        "total_premium": total_premium,
        "premium_percent": premium_percent,
        "upper_breakeven": upper_be,
        "lower_breakeven": lower_be,
        "required_move_percent": required_move_pct,
        "days_to_expiration": days_to_expiration,
        "iv_rank": iv_rank,
        "historical_volatility": historical_volatility,
        "iv_percentile": iv_percentile,
        "probability_of_profit": prob_profit,
        "score_components": {}
    }
    
    # Scoring components
    
    # 1. IV Rank (higher is better for straddles)
    iv_score = min(iv_rank, 100)
    score += (iv_score - 50) * 0.2
    analysis["score_components"]["iv_rank_contribution"] = (iv_score - 50) * 0.2
    
    # 2. Days to expiration (prefer 20-40 days)
    if 20 <= days_to_expiration <= 40:
        dte_score = 10
    elif 10 <= days_to_expiration < 20 or 40 < days_to_expiration <= 50:
        dte_score = 5
    else:
        dte_score = 0
    score += dte_score
    analysis["score_components"]["days_to_expiration_contribution"] = dte_score
    
    # 3. Proximity to ATM
    atm_diff_pct = abs((strike_price - current_price) / current_price) * 100
    if atm_diff_pct < 1:
        atm_score = 10
    elif atm_diff_pct < 2:
        atm_score = 5
    else:
        atm_score = 0
    score += atm_score
    analysis["score_components"]["atm_proximity_contribution"] = atm_score
    
    # 4. Premium vs Historical Movement
    # Calculate the historical daily movement
    daily_movement = historical_volatility * current_price / np.sqrt(252)
    expected_movement = daily_movement * np.sqrt(days_to_expiration)
    move_premium_ratio = expected_movement / total_premium
    
    if move_premium_ratio > 1.5:
        value_score = 15
    elif move_premium_ratio > 1.2:
        value_score = 10
    elif move_premium_ratio > 1.0:
        value_score = 5
    else:
        value_score = 0
    score += value_score
    analysis["score_components"]["value_contribution"] = value_score
    
    # 5. Event premium - add points if there's an upcoming event
    event_score = 0
    if upcoming_events and len(upcoming_events) > 0:
        for event_type, event_details in upcoming_events.items():
            if event_details:
                days_to_event = (event_details[0]['date'] - datetime.now().date()).days
                if days_to_event < days_to_expiration:
                    event_score = 15
                    analysis["upcoming_event"] = {
                        "type": event_type,
                        "date": event_details[0]['date'].strftime("%Y-%m-%d"),
                        "days_away": days_to_event
                    }
                    break
    
    score += event_score
    analysis["score_components"]["event_contribution"] = event_score
    
    # Cap the score at 100
    score = min(score, 100)
    analysis["total_score"] = score
    
    return score, analysis

def calculate_expected_move(price, days, volatility):
    """
    Calculate the expected move of a stock over a given time period.
    
    Args:
        price (float): Current stock price
        days (int): Number of days
        volatility (float): Annualized volatility as a decimal
        
    Returns:
        float: Expected dollar move (1 standard deviation)
    """
    return price * volatility * np.sqrt(days / 365)

def calculate_straddle_risk_reward(
    strike_price,
    call_price,
    put_price,
    days_to_expiration,
    historical_volatility,
    current_price
):
    """
    Calculate risk-reward metrics for a straddle position.
    
    Args:
        strike_price (float): Strike price of the options
        call_price (float): Price of the call option
        put_price (float): Price of the put option
        days_to_expiration (int): Days until expiration
        historical_volatility (float): Historical volatility (decimal)
        current_price (float): Current stock price
        
    Returns:
        dict: Risk-reward metrics including max profit potential and risk/reward ratio
    """
    total_premium = call_price + put_price
    max_risk = total_premium  # Maximum risk is the premium paid
    
    # Calculate expected move based on historical volatility
    expected_move = calculate_expected_move(
        current_price, days_to_expiration, historical_volatility
    )
    
    # Estimate profit at expected move
    upper_price = current_price + expected_move
    lower_price = max(0, current_price - expected_move)
    
    upper_profit = max(0, upper_price - strike_price) - total_premium
    lower_profit = max(0, strike_price - lower_price) - total_premium
    
    # Take the higher of the two profit scenarios
    expected_profit = max(upper_profit, lower_profit)
    
    if expected_profit > 0:
        risk_reward_ratio = expected_profit / max_risk
    else:
        risk_reward_ratio = 0.0
    
    return {
        "max_risk": max_risk,
        "expected_move": expected_move,
        "expected_profit": expected_profit,
        "risk_reward_ratio": risk_reward_ratio
    }

def analyze_historical_gaps(price_history, threshold_pct=2.0):
    """
    Analyze historical price gaps to assess potential for gap moves.
    
    Args:
        price_history (DataFrame): DataFrame with daily OHLC data
        threshold_pct (float): Minimum gap percentage to consider significant
        
    Returns:
        dict: Gap analysis statistics
    """
    if price_history is None or len(price_history) < 10:
        return {"gap_frequency": 0, "average_gap_size": 0, "max_gap": 0}
    
    # Calculate overnight gaps
    price_history = price_history.sort_index()
    price_history['prev_close'] = price_history['close'].shift(1)
    price_history['gap_pct'] = ((price_history['open'] - price_history['prev_close']) / 
                               price_history['prev_close'] * 100)
    
    # Filter significant gaps
    significant_gaps = price_history[abs(price_history['gap_pct']) >= threshold_pct]
    
    if len(significant_gaps) == 0:
        return {"gap_frequency": 0, "average_gap_size": 0, "max_gap": 0}
    
    # Calculate statistics
    gap_count = len(significant_gaps)
    total_days = len(price_history)
    gap_frequency = gap_count / total_days
    avg_gap_size = significant_gaps['gap_pct'].abs().mean()
    max_gap = significant_gaps['gap_pct'].abs().max()
    
    return {
        "gap_frequency": gap_frequency,
        "gap_count": gap_count, 
        "days_analyzed": total_days,
        "average_gap_size": avg_gap_size,
        "max_gap": max_gap,
        "gap_details": significant_gaps[['gap_pct']].to_dict('records')
    }

def is_straddle_candidate(
    symbol, 
    iv_rank, 
    volume, 
    option_volume, 
    bid_ask_spread_pct,
    config
):
    """
    Determine if a stock is a candidate for a straddle strategy.
    
    Args:
        symbol (str): Stock symbol
        iv_rank (float): Current IV rank (0-100) 
        volume (int): Average daily trading volume
        option_volume (int): Option contract volume
        bid_ask_spread_pct (float): Bid-ask spread as percentage
        config (dict): Strategy configuration dictionary
        
    Returns:
        bool: True if the stock is a straddle candidate, False otherwise
    """
    # Check criteria based on configuration
    entry_criteria = config["entry_criteria"]
    
    if iv_rank < entry_criteria["iv_rank_threshold"]:
        logger.debug(f"{symbol} rejected: IV rank {iv_rank} below threshold {entry_criteria['iv_rank_threshold']}")
        return False
        
    if volume < entry_criteria["min_average_volume"]:
        logger.debug(f"{symbol} rejected: Volume {volume} below threshold {entry_criteria['min_average_volume']}")
        return False
        
    if option_volume < entry_criteria["min_option_volume"]:
        logger.debug(f"{symbol} rejected: Option volume {option_volume} below threshold {entry_criteria['min_option_volume']}")
        return False
        
    if bid_ask_spread_pct > entry_criteria["max_bid_ask_spread_pct"]:
        logger.debug(f"{symbol} rejected: Bid-ask spread {bid_ask_spread_pct}% above threshold {entry_criteria['max_bid_ask_spread_pct']}%")
        return False
    
    logger.debug(f"{symbol} passed initial screening as straddle candidate")
    return True 