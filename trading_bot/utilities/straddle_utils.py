"""
Utility functions for straddle options strategy.

This module provides helper functions for analyzing and evaluating straddle opportunities.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union

from trading_bot.models.option import Option
from trading_bot.utilities.options_utils import calculate_breakeven_prices


def calculate_straddle_cost(call_option: Option, put_option: Option) -> float:
    """
    Calculate the total cost of a straddle position.
    
    Args:
        call_option: Call option component of the straddle
        put_option: Put option component of the straddle
        
    Returns:
        Total cost of the straddle position (per contract)
    """
    return call_option.ask + put_option.ask


def calculate_straddle_breakevens(call_option: Option, put_option: Option) -> Tuple[float, float]:
    """
    Calculate the upside and downside breakeven prices for a straddle.
    
    Args:
        call_option: Call option component of the straddle
        put_option: Put option component of the straddle
        
    Returns:
        Tuple of (downside_breakeven, upside_breakeven)
    """
    # Strike price is the same for both options in a straddle
    strike = call_option.strike
    straddle_cost = calculate_straddle_cost(call_option, put_option)
    
    # Downside breakeven = Strike - Straddle Cost
    downside_breakeven = strike - straddle_cost
    
    # Upside breakeven = Strike + Straddle Cost
    upside_breakeven = strike + straddle_cost
    
    return (downside_breakeven, upside_breakeven)


def calculate_straddle_max_loss(call_option: Option, put_option: Option) -> float:
    """
    Calculate the maximum loss for a straddle position.
    
    Args:
        call_option: Call option component of the straddle
        put_option: Put option component of the straddle
        
    Returns:
        Maximum loss amount (per contract)
    """
    # Maximum loss in a long straddle is the total premium paid
    return calculate_straddle_cost(call_option, put_option) * 100  # Convert to dollar amount


def calculate_breakeven_move_required(
    stock_price: float, call_option: Option, put_option: Option
) -> Dict[str, float]:
    """
    Calculate the move required to reach breakeven as percentage and absolute value.
    
    Args:
        stock_price: Current stock price
        call_option: Call option component of the straddle
        put_option: Put option component of the straddle
        
    Returns:
        Dictionary with percentage and absolute moves required
    """
    downside_breakeven, upside_breakeven = calculate_straddle_breakevens(call_option, put_option)
    
    # Calculate absolute moves required
    upside_move = upside_breakeven - stock_price
    downside_move = stock_price - downside_breakeven
    
    # Calculate percentage moves
    upside_pct = (upside_move / stock_price) * 100
    downside_pct = (downside_move / stock_price) * 100
    
    # Take the minimum as the required move (whichever direction is closer)
    min_move = min(upside_move, downside_move)
    min_move_pct = min(upside_pct, downside_pct)
    
    return {
        "upside_move": upside_move,
        "downside_move": downside_move,
        "upside_pct": upside_pct,
        "downside_pct": downside_pct,
        "min_move": min_move,
        "min_move_pct": min_move_pct
    }


def evaluate_straddle(
    stock_price: float, 
    call_option: Option, 
    put_option: Option,
    historical_volatility: float,
    iv_rank: float,
    min_volume: int = 100,
    max_spread_pct: float = 5.0
) -> Dict[str, Union[float, bool, str]]:
    """
    Evaluate a straddle opportunity and score it.
    
    Args:
        stock_price: Current stock price
        call_option: Call option component of the straddle
        put_option: Put option component of the straddle
        historical_volatility: Historical volatility (annualized)
        iv_rank: IV rank as a percentage (0-100)
        min_volume: Minimum acceptable volume
        max_spread_pct: Maximum acceptable bid-ask spread percentage
        
    Returns:
        Dictionary with evaluation metrics and score
    """
    straddle_cost = calculate_straddle_cost(call_option, put_option)
    breakeven_moves = calculate_breakeven_move_required(stock_price, call_option, put_option)
    
    # Check for sufficient volume
    sufficient_volume = (call_option.volume >= min_volume and put_option.volume >= min_volume)
    
    # Check bid-ask spread
    call_spread_pct = ((call_option.ask - call_option.bid) / call_option.bid) * 100
    put_spread_pct = ((put_option.ask - put_option.bid) / put_option.bid) * 100
    acceptable_spread = (call_spread_pct <= max_spread_pct and put_spread_pct <= max_spread_pct)
    
    # Calculate expected move based on historical volatility
    days_to_expiry = call_option.days_to_expiration
    expected_move_pct = historical_volatility * np.sqrt(days_to_expiry / 252) * 100
    
    # Calculate the score (0-100)
    score_components = []
    
    # Component 1: IV rank (higher is better for straddles)
    iv_rank_score = iv_rank
    score_components.append(iv_rank_score * 0.3)  # 30% weight
    
    # Component 2: Breakeven move vs expected move
    # If breakeven move is less than expected move, that's good
    move_ratio = expected_move_pct / breakeven_moves["min_move_pct"]
    move_score = min(100, move_ratio * 50)  # Cap at 100
    score_components.append(move_score * 0.4)  # 40% weight
    
    # Component 3: Liquidity score
    liquidity_score = 0
    if sufficient_volume:
        liquidity_score += 50
    
    avg_spread = (call_spread_pct + put_spread_pct) / 2
    spread_score = max(0, 50 * (1 - avg_spread / max_spread_pct))
    liquidity_score += spread_score
    
    score_components.append(liquidity_score * 0.3)  # 30% weight
    
    # Final score
    final_score = sum(score_components)
    
    # Decision
    decision = "STRONG_BUY" if final_score >= 80 else \
              "BUY" if final_score >= 70 else \
              "NEUTRAL" if final_score >= 50 else \
              "AVOID"
    
    return {
        "sufficient_volume": sufficient_volume,
        "acceptable_spread": acceptable_spread,
        "call_spread_pct": call_spread_pct,
        "put_spread_pct": put_spread_pct,
        "straddle_cost": straddle_cost,
        "breakeven_move_pct": breakeven_moves["min_move_pct"],
        "expected_move_pct": expected_move_pct,
        "move_ratio": move_ratio,
        "score": final_score,
        "decision": decision
    }


def calculate_straddle_profit_loss(
    stock_price: float,
    entry_price: float,
    strike: float,
    num_contracts: int = 1
) -> float:
    """
    Calculate profit/loss on a straddle position at a given stock price.
    
    Args:
        stock_price: Current stock price for P/L calculation
        entry_price: Total entry price for the straddle (cost of call + put)
        strike: Strike price of the options
        num_contracts: Number of contracts (each contract = 100 shares)
        
    Returns:
        Profit/loss amount in dollars
    """
    # Calculate intrinsic value of each leg
    call_value = max(0, stock_price - strike)
    put_value = max(0, strike - stock_price)
    
    # Total intrinsic value of the straddle
    straddle_value = call_value + put_value
    
    # Calculate P/L
    profit_loss = (straddle_value - entry_price) * 100 * num_contracts
    
    return profit_loss 