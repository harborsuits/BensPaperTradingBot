#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Straddle Trading Strategy Utilities

This module provides utility functions for the straddle trading strategy,
including option pricing, IV calculations, and event analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import scipy.stats as stats
import logging
from scipy.stats import norm

# Setup logging
logger = logging.getLogger(__name__)

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

def price_straddle(
    current_price: float,
    strike: float,
    days_to_expiration: int,
    volatility: float,
    risk_free_rate: float = 0.03  # 3% default risk-free rate
) -> Dict[str, Any]:
    """
    Calculate straddle price and greeks.
    
    Args:
        current_price: Current price of underlying
        strike: Strike price
        days_to_expiration: Days to expiration
        volatility: Implied volatility
        risk_free_rate: Risk-free interest rate
    
    Returns:
        Dictionary with straddle pricing information
    """
    # Convert days to years
    T = days_to_expiration / 365.0
    
    # Calculate call option price and greeks
    call_price, call_greeks = black_scholes_price(
        S=current_price,
        K=strike,
        T=T,
        r=risk_free_rate,
        sigma=volatility,
        option_type='call'
    )
    
    # Calculate put option price and greeks
    put_price, put_greeks = black_scholes_price(
        S=current_price,
        K=strike,
        T=T,
        r=risk_free_rate,
        sigma=volatility,
        option_type='put'
    )
    
    # Calculate combined values
    straddle_price = call_price + put_price
    
    # Combine greeks
    combined_greeks = {
        'delta': call_greeks['delta'] + put_greeks['delta'],
        'gamma': call_greeks['gamma'] + put_greeks['gamma'],
        'theta': call_greeks['theta'] + put_greeks['theta'],
        'vega': call_greeks['vega'] + put_greeks['vega'],
        'rho': call_greeks['rho'] + put_greeks['rho']
    }
    
    # Calculate breakeven prices
    breakeven_up = strike + straddle_price
    breakeven_down = strike - straddle_price
    breakeven_pct = straddle_price / current_price
    
    # Calculate expected move based on IV
    expected_move_1sd = current_price * volatility * np.sqrt(T)
    
    # Calculate probability metrics
    prob_beyond_breakeven = (
        1 - stats.norm.cdf(breakeven_up, loc=current_price, scale=expected_move_1sd / 1.645) +
        stats.norm.cdf(breakeven_down, loc=current_price, scale=expected_move_1sd / 1.645)
    )
    
    return {
        'call_price': call_price,
        'put_price': put_price,
        'straddle_price': straddle_price,
        'call_greeks': call_greeks,
        'put_greeks': put_greeks,
        'combined_greeks': combined_greeks,
        'breakeven_up': breakeven_up,
        'breakeven_down': breakeven_down,
        'breakeven_pct': breakeven_pct,
        'expected_move_1sd': expected_move_1sd,
        'prob_beyond_breakeven': prob_beyond_breakeven
    }

def calculate_iv_rank(iv_history: List[float], current_iv: float) -> float:
    """
    Calculate IV rank based on historical IV values.
    
    Args:
        iv_history: List of historical IV values
        current_iv: Current IV value
    
    Returns:
        IV rank as a percentage (0-100)
    """
    if not iv_history:
        return 50.0  # Default to middle if no history
    
    min_iv = min(iv_history)
    max_iv = max(iv_history)
    
    if max_iv == min_iv:
        return 50.0  # Default to middle if no range
    
    iv_rank = 100.0 * (current_iv - min_iv) / (max_iv - min_iv)
    return iv_rank

def calculate_iv_percentile(iv_history: List[float], current_iv: float) -> float:
    """
    Calculate IV percentile based on historical IV values.
    
    Args:
        iv_history: List of historical IV values
        current_iv: Current IV value
    
    Returns:
        IV percentile (0-100)
    """
    if not iv_history:
        return 50.0  # Default to middle if no history
    
    iv_percentile = 100.0 * sum(1 for iv in iv_history if iv < current_iv) / len(iv_history)
    return iv_percentile

def estimate_iv_from_historical_vol(
    price_data: pd.DataFrame,
    window: int = 20,
    annualization_factor: float = 252,
    premium_factor: float = 1.2
) -> pd.Series:
    """
    Estimate implied volatility from historical price data.
    
    Args:
        price_data: DataFrame with price data
        window: Rolling window for volatility calculation
        annualization_factor: Factor for annualizing volatility
        premium_factor: Multiplier to convert historical vol to implied vol
    
    Returns:
        Series with estimated IV values
    """
    # Calculate log returns
    log_returns = np.log(price_data['close'] / price_data['close'].shift(1))
    
    # Calculate historical volatility
    hist_vol = log_returns.rolling(window=window).std() * np.sqrt(annualization_factor)
    
    # Estimate IV (typically trades at a premium to historical vol)
    estimated_iv = hist_vol * premium_factor
    
    return estimated_iv

def calculate_straddle_pnl(
    entry_price: float,
    current_price: float,
    strike: float,
    entry_days_to_expiration: int,
    current_days_to_expiration: int,
    entry_iv: float,
    current_iv: float,
    risk_free_rate: float = 0.03
) -> Dict[str, float]:
    """
    Calculate P&L for a straddle position.
    
    Args:
        entry_price: Price paid for the straddle at entry
        current_price: Current price of underlying
        strike: Strike price
        entry_days_to_expiration: DTE at entry
        current_days_to_expiration: Current DTE
        entry_iv: IV at entry
        current_iv: Current IV
        risk_free_rate: Risk-free interest rate
    
    Returns:
        Dictionary with P&L metrics
    """
    # Price straddle at current conditions
    current_straddle = price_straddle(
        current_price=current_price,
        strike=strike,
        days_to_expiration=current_days_to_expiration,
        volatility=current_iv,
        risk_free_rate=risk_free_rate
    )
    
    # Calculate raw P&L
    current_straddle_price = current_straddle['straddle_price']
    pnl = current_straddle_price - entry_price
    pnl_pct = pnl / entry_price
    
    # Calculate contribution from each factor
    
    # 1. Underlying price movement (delta effect)
    # Approximate by calculating what the straddle would be worth at current price but original vol
    theoretical_price_only = price_straddle(
        current_price=current_price,
        strike=strike,
        days_to_expiration=current_days_to_expiration,
        volatility=entry_iv,
        risk_free_rate=risk_free_rate
    )
    
    delta_effect = theoretical_price_only['straddle_price'] - entry_price
    
    # 2. Volatility change (vega effect)
    # Difference between actual current price and theoretical with original vol
    vega_effect = current_straddle_price - theoretical_price_only['straddle_price']
    
    # 3. Time decay (theta effect)
    # Approximate by looking at what straddle would be worth with original price and vol but current time
    theoretical_time_only = price_straddle(
        current_price=current_price,
        strike=strike,
        days_to_expiration=entry_days_to_expiration,
        volatility=entry_iv,
        risk_free_rate=risk_free_rate
    )
    
    theta_effect = theoretical_price_only['straddle_price'] - theoretical_time_only['straddle_price']
    
    return {
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'delta_effect': delta_effect,
        'vega_effect': vega_effect,
        'theta_effect': theta_effect,
        'current_straddle_price': current_straddle_price,
        'current_delta': current_straddle['combined_greeks']['delta'],
        'current_vega': current_straddle['combined_greeks']['vega'],
        'current_theta': current_straddle['combined_greeks']['theta'],
        'current_gamma': current_straddle['combined_greeks']['gamma']
    }

def find_catalyst_events(
    symbol: str,
    events_calendar: pd.DataFrame,
    min_days_ahead: int = 1,
    max_days_ahead: int = 45,
    eligible_event_types: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Find upcoming catalyst events for a symbol.
    
    Args:
        symbol: Ticker symbol to find events for
        events_calendar: DataFrame with events data
        min_days_ahead: Minimum days ahead to consider
        max_days_ahead: Maximum days ahead to consider
        eligible_event_types: List of event types to consider
    
    Returns:
        List of event dictionaries
    """
    if eligible_event_types is None:
        eligible_event_types = ["earnings", "fda_decision", "economic_release", "technical_breakout"]
    
    now = datetime.now()
    min_date = now + timedelta(days=min_days_ahead)
    max_date = now + timedelta(days=max_days_ahead)
    
    # Filter events
    symbol_events = events_calendar[
        (events_calendar['symbol'] == symbol) &
        (events_calendar['datetime'] >= min_date) &
        (events_calendar['datetime'] <= max_date) &
        (events_calendar['event_type'].isin(eligible_event_types))
    ]
    
    # Convert to list of dictionaries
    events = []
    for _, event in symbol_events.iterrows():
        events.append({
            "symbol": symbol,
            "type": event["event_type"],
            "date": event["datetime"],
            "description": event.get("description", ""),
            "importance": event.get("importance", "medium")
        })
    
    return events

def find_atm_strike(
    current_price: float,
    available_strikes: List[float],
    offset: int = 0  # 0 for ATM, +1 for one strike up, -1 for one strike down
) -> float:
    """
    Find the closest strike to current price with optional offset.
    
    Args:
        current_price: Current price of underlying
        available_strikes: List of available strikes
        offset: Offset from ATM in number of strikes
    
    Returns:
        Selected strike price
    """
    if not available_strikes:
        # If no strikes provided, estimate based on price conventions
        if current_price < 10:
            strike_interval = 0.5
        elif current_price < 50:
            strike_interval = 1.0
        elif current_price < 100:
            strike_interval = 2.5
        elif current_price < 200:
            strike_interval = 5.0
        else:
            strike_interval = 10.0
        
        # Round to nearest interval
        base_strike = round(current_price / strike_interval) * strike_interval
        return base_strike + (offset * strike_interval)
    
    # Sort strikes
    sorted_strikes = sorted(available_strikes)
    
    # Find closest strike
    closest_idx = min(range(len(sorted_strikes)), key=lambda i: abs(sorted_strikes[i] - current_price))
    
    # Apply offset
    target_idx = closest_idx + offset
    
    # Ensure index is valid
    if target_idx < 0:
        target_idx = 0
    elif target_idx >= len(sorted_strikes):
        target_idx = len(sorted_strikes) - 1
    
    return sorted_strikes[target_idx]

def generate_straddle_trade_plan(
    symbol: str,
    current_price: float,
    strike: float,
    expiration_date: datetime,
    iv: float,
    straddle_price: float
) -> Dict[str, Any]:
    """
    Generate a complete trade plan for a straddle.
    
    Args:
        symbol: Ticker symbol
        current_price: Current price of underlying
        strike: Strike price for the straddle
        expiration_date: Expiration date
        iv: Implied volatility
        straddle_price: Cost of the straddle
    
    Returns:
        Dictionary with trade plan details
    """
    now = datetime.now()
    days_to_expiration = (expiration_date - now).days
    
    # Calculate key metrics
    breakeven_pct = straddle_price / current_price
    upside_breakeven = strike + straddle_price
    downside_breakeven = strike - straddle_price
    
    # Calculate greeks and expected move
    straddle_info = price_straddle(
        current_price=current_price,
        strike=strike,
        days_to_expiration=days_to_expiration,
        volatility=iv
    )
    
    # Generate trade plan
    trade_plan = {
        "symbol": symbol,
        "strategy": "straddle",
        "current_price": current_price,
        "strike": strike,
        "expiration": expiration_date.strftime("%Y-%m-%d"),
        "days_to_expiration": days_to_expiration,
        "implied_volatility": iv,
        
        # Entry details
        "entry": {
            "straddle_price": straddle_price,
            "call_price": straddle_info["call_price"],
            "put_price": straddle_info["put_price"],
            "breakeven_pct": breakeven_pct * 100,  # Convert to percentage
            "upside_breakeven": upside_breakeven,
            "downside_breakeven": downside_breakeven
        },
        
        # Greeks
        "greeks": {
            "delta": straddle_info["combined_greeks"]["delta"],
            "gamma": straddle_info["combined_greeks"]["gamma"],
            "theta": straddle_info["combined_greeks"]["theta"],
            "vega": straddle_info["combined_greeks"]["vega"]
        },
        
        # Expected move
        "expected_move": {
            "one_std_dev": straddle_info["expected_move_1sd"],
            "one_std_dev_pct": straddle_info["expected_move_1sd"] / current_price * 100,
            "prob_beyond_breakeven": straddle_info["prob_beyond_breakeven"] * 100  # Convert to percentage
        },
        
        # Exit plan
        "exit_plan": {
            "profit_target": straddle_price * 0.65,  # 65% of max gain
            "profit_target_price": straddle_price * 1.65,
            "max_loss": straddle_price,
            "time_exit": (expiration_date - timedelta(days=1)).strftime("%Y-%m-%d")
        }
    }
    
    return trade_plan

def generate_profit_table(
    symbol: str,
    strike: float,
    current_price: float,
    straddle_price: float,
    days_to_expiration: int,
    volatility: float,
    price_points: int = 7,
    price_range_pct: float = 0.15,
    time_points: List[int] = None
) -> pd.DataFrame:
    """
    Generate a P&L table for different price and time scenarios.
    
    Args:
        symbol: Ticker symbol
        strike: Strike price
        current_price: Current price of underlying
        straddle_price: Entry price for the straddle
        days_to_expiration: Days to expiration at entry
        volatility: Implied volatility at entry
        price_points: Number of price points to calculate
        price_range_pct: Price range as percentage of current price
        time_points: List of DTE points to calculate (default: entry, mid-point, expiration)
    
    Returns:
        DataFrame with profit/loss values for different scenarios
    """
    if time_points is None:
        time_points = [days_to_expiration, days_to_expiration // 2, 1]
    
    # Generate price points
    min_price = current_price * (1 - price_range_pct)
    max_price = current_price * (1 + price_range_pct)
    price_steps = np.linspace(min_price, max_price, price_points)
    
    # Create result DataFrame
    results = []
    
    for remaining_days in time_points:
        for price in price_steps:
            # Price straddle at this scenario
            scenario_straddle = price_straddle(
                current_price=price,
                strike=strike,
                days_to_expiration=remaining_days,
                volatility=volatility
            )
            
            # Calculate P&L
            pnl = scenario_straddle["straddle_price"] - straddle_price
            pnl_pct = pnl / straddle_price * 100
            
            results.append({
                "symbol": symbol,
                "price": price,
                "price_change_pct": (price - current_price) / current_price * 100,
                "days_remaining": remaining_days,
                "straddle_value": scenario_straddle["straddle_price"],
                "pnl": pnl,
                "pnl_pct": pnl_pct
            })
    
    return pd.DataFrame(results)

def check_recent_gaps(
    price_data: pd.DataFrame,
    days: int = 3,
    threshold: float = 0.04
) -> List[Dict[str, Any]]:
    """
    Check for significant price gaps in recent days.
    
    Args:
        price_data: DataFrame with OHLC price data
        days: Number of recent days to check
        threshold: Gap threshold as percentage of price
    
    Returns:
        List of detected gaps with details
    """
    if len(price_data) < days + 1:
        return []
    
    recent_data = price_data.iloc[-(days+1):]
    gaps = []
    
    for i in range(1, len(recent_data)):
        prev_close = recent_data.iloc[i-1]['close']
        current_open = recent_data.iloc[i]['open']
        current_date = recent_data.index[i]
        
        gap_pct = (current_open - prev_close) / prev_close
        abs_gap_pct = abs(gap_pct)
        
        if abs_gap_pct > threshold:
            gaps.append({
                "date": current_date,
                "prev_close": prev_close,
                "open": current_open,
                "gap_pct": gap_pct * 100,  # Convert to percentage
                "direction": "up" if gap_pct > 0 else "down"
            })
    
    return gaps

def calculate_straddle_cost(call_price, put_price):
    """
    Calculate the total cost of a straddle position.
    
    Args:
        call_price (float): Price of the call option
        put_price (float): Price of the put option
        
    Returns:
        float: Total cost of the straddle position
    """
    return call_price + put_price

def calculate_straddle_breakeven_points(strike_price, straddle_cost):
    """
    Calculate the upper and lower breakeven points for a straddle.
    
    Args:
        strike_price (float): Strike price of both the call and put options
        straddle_cost (float): Total cost of the straddle position
        
    Returns:
        tuple: (lower_breakeven, upper_breakeven)
    """
    upper_breakeven = strike_price + straddle_cost
    lower_breakeven = strike_price - straddle_cost
    
    # Ensure lower breakeven doesn't go below zero
    lower_breakeven = max(0, lower_breakeven)
    
    return (lower_breakeven, upper_breakeven)

def calculate_max_loss(straddle_cost, contracts=1, contract_multiplier=100):
    """
    Calculate the maximum loss for a straddle position.
    
    Args:
        straddle_cost (float): Total cost of the straddle per contract
        contracts (int): Number of contracts
        contract_multiplier (int): Contract multiplier (typically 100 for US options)
        
    Returns:
        float: Maximum possible loss
    """
    return straddle_cost * contracts * contract_multiplier

def calculate_straddle_pnl(current_price, strike_price, straddle_cost, contracts=1, contract_multiplier=100):
    """
    Calculate the profit/loss of a straddle position at a given price.
    
    Args:
        current_price (float): Current price of the underlying asset
        strike_price (float): Strike price of the straddle
        straddle_cost (float): Total cost of the straddle per contract
        contracts (int): Number of contracts
        contract_multiplier (int): Contract multiplier (typically 100 for US options)
        
    Returns:
        float: Profit/loss of the position
    """
    # Calculate intrinsic value of the straddle
    distance_from_strike = abs(current_price - strike_price)
    intrinsic_value = distance_from_strike
    
    # Calculate P&L
    pnl = (intrinsic_value - straddle_cost) * contracts * contract_multiplier
    
    return pnl

def calculate_expected_move(stock_price, option_chain, days_to_expiration):
    """
    Calculate the expected move of the underlying asset based on option implied volatility.
    
    Args:
        stock_price (float): Current price of the underlying asset
        option_chain (pd.DataFrame): Option chain data with implied volatility
        days_to_expiration (int): Days to expiration
        
    Returns:
        float: Expected move in price over the given time period
    """
    # Find ATM options
    calls = option_chain[option_chain['option_type'] == 'call']
    puts = option_chain[option_chain['option_type'] == 'put']
    
    # Get closest strikes to current price
    call_idx = (calls['strike'] - stock_price).abs().idxmin()
    put_idx = (puts['strike'] - stock_price).abs().idxmin()
    
    # Get implied volatility
    call_iv = calls.loc[call_idx, 'implied_volatility']
    put_iv = puts.loc[put_idx, 'implied_volatility']
    
    # Average the IVs
    avg_iv = (call_iv + put_iv) / 2
    
    # Calculate expected move
    # Expected move = Stock Price * IV * √(DTE/365)
    expected_move = stock_price * avg_iv * np.sqrt(days_to_expiration / 365)
    
    return expected_move

def calculate_probability_of_profit(stock_price, strike_price, straddle_cost, implied_volatility, days_to_expiration):
    """
    Calculate the approximate probability of profit for a straddle position.
    
    Args:
        stock_price (float): Current price of the underlying asset
        strike_price (float): Strike price of the straddle
        straddle_cost (float): Total cost of the straddle
        implied_volatility (float): Implied volatility as a decimal (e.g., 0.30 for 30%)
        days_to_expiration (int): Number of days to expiration
        
    Returns:
        float: Probability of profit as a percentage
    """
    # Calculate breakeven points
    lower_breakeven, upper_breakeven = calculate_straddle_breakeven_points(strike_price, straddle_cost)
    
    # Calculate standard deviation of expected move
    std_dev = stock_price * implied_volatility * np.sqrt(days_to_expiration / 365)
    
    # Calculate z-scores for breakeven points
    z_lower = (lower_breakeven - stock_price) / std_dev
    z_upper = (upper_breakeven - stock_price) / std_dev
    
    # Calculate probability using normal distribution
    from scipy.stats import norm
    
    # Probability that price will be below lower breakeven or above upper breakeven
    prob_below_lower = norm.cdf(z_lower)
    prob_above_upper = 1 - norm.cdf(z_upper)
    
    # Total probability of profit
    probability_of_profit = (prob_below_lower + prob_above_upper) * 100
    
    return probability_of_profit

def calculate_risk_reward_ratio(straddle_cost, expected_move, days_to_expiration, target_days=None):
    """
    Calculate the risk/reward ratio for a straddle position.
    
    Args:
        straddle_cost (float): Total cost of the straddle
        expected_move (float): Expected move of the underlying asset
        days_to_expiration (int): Days to expiration
        target_days (int, optional): Target holding period in days. If None, uses days_to_expiration.
        
    Returns:
        float: Risk/reward ratio
    """
    if target_days is None:
        target_days = days_to_expiration
        
    # Adjust expected move for target holding period
    adjusted_expected_move = expected_move * np.sqrt(target_days / days_to_expiration)
    
    # Risk is the maximum loss (straddle cost)
    risk = straddle_cost
    
    # Reward is the expected profit based on expected move
    reward = max(0, adjusted_expected_move - straddle_cost)
    
    if risk == 0 or reward == 0:
        return 0
        
    # Calculate ratio
    ratio = reward / risk
    
    return ratio

def find_optimal_strike(stock_price, option_chain, target_delta=0.5):
    """
    Find the optimal strike price for a straddle based on target delta.
    
    Args:
        stock_price (float): Current price of the underlying asset
        option_chain (pd.DataFrame): Option chain data with greeks
        target_delta (float): Target absolute delta value (default 0.5 for ATM)
        
    Returns:
        float: Optimal strike price
    """
    calls = option_chain[option_chain['option_type'] == 'call']
    
    # Find the strike with delta closest to the target
    calls['delta_diff'] = abs(abs(calls['delta']) - target_delta)
    optimal_idx = calls['delta_diff'].idxmin()
    optimal_strike = calls.loc[optimal_idx, 'strike']
    
    return optimal_strike

def calculate_iv_history_ratio(current_iv, historical_iv_percentiles):
    """
    Calculate the ratio of current IV to historical IV.
    
    Args:
        current_iv (float): Current implied volatility
        historical_iv_percentiles (dict): Dict containing historical IV percentiles
        
    Returns:
        float: Ratio of current IV to historical median IV
    """
    if 'median' not in historical_iv_percentiles or historical_iv_percentiles['median'] == 0:
        return 1.0
        
    return current_iv / historical_iv_percentiles['median']

def score_straddle_opportunity(ticker, stock_price, iv_rank, iv_history_ratio, 
                               bid_ask_spread, option_volume, days_to_expiration,
                               probability_of_profit, scoring_weights):
    """
    Score a straddle opportunity based on various metrics.
    
    Args:
        ticker (str): Ticker symbol
        stock_price (float): Current price of the underlying asset
        iv_rank (float): IV Rank (0-100)
        iv_history_ratio (float): Ratio of current IV to historical median IV
        bid_ask_spread (float): Bid-ask spread percentage
        option_volume (int): Option volume
        days_to_expiration (int): Days to expiration
        probability_of_profit (float): Probability of profit percentage
        scoring_weights (dict): Weight factors for each metric
        
    Returns:
        float: Composite score (0-100)
    """
    # Normalize metrics to 0-100 scale
    normalized_iv_rank = iv_rank  # Already 0-100
    
    # IV history ratio: higher is better, cap at 3.0 for 100 points
    normalized_iv_ratio = min(100, iv_history_ratio / 3.0 * 100)
    
    # Bid-ask spread: lower is better, 0% = 100 points, 10% = 0 points
    normalized_bid_ask = max(0, 100 - (bid_ask_spread * 10))
    
    # Option volume: higher is better, logarithmic scale, cap at 10,000
    normalized_volume = min(100, np.log10(max(1, option_volume)) / np.log10(10000) * 100)
    
    # Days to expiration: score based on deviation from ideal (45 days)
    ideal_dte = 45
    dte_deviation = abs(days_to_expiration - ideal_dte) / ideal_dte
    normalized_dte = max(0, 100 - (dte_deviation * 100))
    
    # Probability of profit: directly use the percentage
    normalized_pop = probability_of_profit
    
    # Calculate weighted score
    score = (
        scoring_weights.get('iv_rank', 0) * normalized_iv_rank +
        scoring_weights.get('iv_vs_historical_ratio', 0) * normalized_iv_ratio +
        scoring_weights.get('bid_ask_spread', 0) * normalized_bid_ask +
        scoring_weights.get('option_volume', 0) * normalized_volume +
        scoring_weights.get('days_to_expiration', 0) * normalized_dte +
        scoring_weights.get('probability_of_profit', 0) * normalized_pop
    )
    
    logger.debug(f"Scoring {ticker}: IV Rank={iv_rank:.1f}, IV Ratio={iv_history_ratio:.2f}, "
                 f"Bid-Ask={bid_ask_spread:.2f}%, Volume={option_volume}, DTE={days_to_expiration}, "
                 f"POP={probability_of_profit:.1f}%, Score={score:.1f}")
    
    return score

def get_optimal_dte_for_straddle(volatility_term_structure, target_dte=45):
    """
    Determine the optimal days to expiration for a straddle based on
    the volatility term structure and target DTE.
    
    Args:
        volatility_term_structure (dict): Dictionary mapping DTEs to IVs
        target_dte (int): Target days to expiration
        
    Returns:
        int: Recommended DTE for the straddle
    """
    if not volatility_term_structure:
        return target_dte
    
    # Find expirations with volatility skews that favor straddles
    available_dtes = sorted(list(volatility_term_structure.keys()))
    
    if not available_dtes:
        return target_dte
    
    # If target DTE is available, use it
    if target_dte in available_dtes:
        return target_dte
    
    # Otherwise find the closest DTE to the target
    return min(available_dtes, key=lambda x: abs(x - target_dte))

def estimate_position_size(
    account_value, 
    max_position_size_pct, 
    straddle_cost,
    max_loss_pct
):
    """
    Calculate the recommended position size for a straddle.
    
    Args:
        account_value (float): Total account value
        max_position_size_pct (float): Maximum percentage of account to allocate
        straddle_cost (float): Cost per straddle contract
        max_loss_pct (float): Maximum acceptable loss percentage
        
    Returns:
        int: Number of straddle contracts to trade
    """
    # Maximum amount to allocate based on position size limit
    max_allocation = account_value * max_position_size_pct
    
    # Maximum number of contracts based on allocation
    max_contracts = int(max_allocation / straddle_cost)
    
    # Calculate max loss amount
    max_loss_amount = account_value * (max_loss_pct / 100)
    
    # Calculate max contracts based on max loss
    max_contracts_by_risk = int(max_loss_amount / straddle_cost)
    
    # Return the lower of the two limits
    return max(1, min(max_contracts, max_contracts_by_risk))

# TODO: Add function to calculate IV skew metrics for better strike selection
# TODO: Add function to fetch and process option chain data
# TODO: Add function to detect and predict earnings gaps
# TODO: Add function to backtest straddle performance by event type 