"""
Utility functions for diagonal spread options strategy.

This module provides functions for analyzing, evaluating, and
calculating metrics specific to diagonal spread options strategies.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional
from scipy.stats import norm

# TODO: Import option pricing models as needed

def calculate_trend_strength(
    price_data: pd.DataFrame, 
    period_days: int = 20, 
    method: str = "linear_regression"
) -> Tuple[float, str]:
    """
    Calculate the strength and direction of a price trend.
    
    Args:
        price_data: DataFrame with OHLC price data
        period_days: Number of days to analyze
        method: Method to use for trend calculation
        
    Returns:
        Tuple of (trend_strength, trend_direction)
    """
    if len(price_data) < period_days:
        return 0.0, "neutral"
    
    # Use the most recent data
    recent_data = price_data.iloc[-period_days:]
    
    if method == "linear_regression":
        # Use linear regression to calculate trend
        x = np.arange(len(recent_data))
        y = recent_data['close'].values
        
        slope, intercept = np.polyfit(x, y, 1)
        
        # Calculate r-squared to determine trend strength
        y_pred = slope * x + intercept
        y_mean = np.mean(y)
        
        ss_total = np.sum((y - y_mean) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        
        r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
        trend_strength = abs(r_squared)
        
        # Determine direction based on slope
        if slope > 0:
            trend_direction = "bullish"
        elif slope < 0:
            trend_direction = "bearish"
        else:
            trend_direction = "neutral"
            
    elif method == "moving_average":
        # Use moving averages to determine trend
        short_ma = recent_data['close'].rolling(window=5).mean().iloc[-1]
        long_ma = recent_data['close'].rolling(window=20).mean().iloc[-1]
        
        # Calculate trend strength as normalized distance between MAs
        ma_distance = abs(short_ma - long_ma) / long_ma
        trend_strength = min(1.0, ma_distance * 10)  # Scale to 0-1
        
        # Determine direction
        if short_ma > long_ma:
            trend_direction = "bullish"
        elif short_ma < long_ma:
            trend_direction = "bearish"
        else:
            trend_direction = "neutral"
    else:
        # Default to simple method
        first_price = recent_data['close'].iloc[0]
        last_price = recent_data['close'].iloc[-1]
        
        price_change = (last_price - first_price) / first_price
        trend_strength = min(1.0, abs(price_change) * 5)  # Scale to 0-1
        
        if price_change > 0:
            trend_direction = "bullish"
        elif price_change < 0:
            trend_direction = "bearish"
        else:
            trend_direction = "neutral"
    
    return trend_strength, trend_direction

def find_diagonal_spread_candidates(
    ticker: str,
    trend_direction: str,
    current_price: float,
    option_chain: Dict,
    config: Dict
) -> List[Dict]:
    """
    Find candidate diagonal spreads for a given ticker and trend direction.
    
    Args:
        ticker: Symbol to analyze
        trend_direction: "bullish" or "bearish"
        current_price: Current price of the underlying
        option_chain: Option chain data
        config: Strategy configuration
        
    Returns:
        List of candidate diagonal spreads
    """
    candidates = []
    
    # Extract configuration parameters
    spread_config = config["spread_construction"]
    expiration_config = config["expiration_management"]
    
    # Determine option type based on trend direction
    if trend_direction == "bullish":
        option_type = "call"
    elif trend_direction == "bearish":
        option_type = "put"
    else:
        return candidates  # No clear trend, return empty list
    
    # Get current date
    today = datetime.now().date()
    
    # Filter expirations for long leg
    long_leg_expirations = []
    for expiry_str, options in option_chain.items():
        try:
            expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            dte = (expiry - today).days
            
            if expiration_config["entry_long_leg_min_dte"] <= dte <= expiration_config["entry_long_leg_max_dte"]:
                long_leg_expirations.append((expiry_str, dte, options))
        except:
            continue
    
    # Filter expirations for short leg
    short_leg_expirations = []
    for expiry_str, options in option_chain.items():
        try:
            expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            dte = (expiry - today).days
            
            if expiration_config["entry_short_leg_min_dte"] <= dte <= expiration_config["entry_short_leg_max_dte"]:
                short_leg_expirations.append((expiry_str, dte, options))
        except:
            continue
    
    # Sort by DTE
    long_leg_expirations.sort(key=lambda x: x[1])
    short_leg_expirations.sort(key=lambda x: x[1])
    
    # Find long leg options with appropriate delta
    long_leg_candidates = []
    for expiry_str, dte, options in long_leg_expirations:
        options_filtered = []
        
        for option in options:
            if option["option_type"] != option_type:
                continue
                
            delta = abs(option.get("delta", 0))
            
            if spread_config["long_leg_min_delta"] <= delta <= spread_config["long_leg_max_delta"]:
                options_filtered.append({
                    "expiry": expiry_str,
                    "dte": dte,
                    "strike": option["strike"],
                    "bid": option["bid"],
                    "ask": option["ask"],
                    "delta": delta,
                    "volume": option.get("volume", 0),
                    "open_interest": option.get("open_interest", 0),
                    "implied_vol": option.get("implied_volatility", 0)
                })
        
        if options_filtered:
            long_leg_candidates.extend(options_filtered)
    
    # Find short leg options with appropriate delta
    short_leg_candidates = []
    for expiry_str, dte, options in short_leg_expirations:
        options_filtered = []
        
        for option in options:
            if option["option_type"] != option_type:
                continue
                
            delta = abs(option.get("delta", 0))
            
            if spread_config["short_leg_min_delta"] <= delta <= spread_config["short_leg_max_delta"]:
                options_filtered.append({
                    "expiry": expiry_str,
                    "dte": dte,
                    "strike": option["strike"],
                    "bid": option["bid"],
                    "ask": option["ask"],
                    "delta": delta,
                    "volume": option.get("volume", 0),
                    "open_interest": option.get("open_interest", 0),
                    "implied_vol": option.get("implied_volatility", 0)
                })
        
        if options_filtered:
            short_leg_candidates.extend(options_filtered)
    
    # Construct diagonal spreads
    for long_option in long_leg_candidates:
        for short_option in short_leg_candidates:
            # Skip if expiration dates are the same (would be a vertical spread)
            if long_option["expiry"] == short_option["expiry"]:
                continue
                
            # Skip if short option expiration is after long option expiration
            if short_option["dte"] > long_option["dte"]:
                continue
            
            # Calculate strike price difference
            strike_diff_pct = abs(long_option["strike"] - short_option["strike"]) / current_price * 100
            
            # Skip if strikes are too close or too far apart
            target_offset = spread_config["strike_offset_pct"]
            if abs(strike_diff_pct - target_offset) > target_offset:
                continue
            
            # Calculate net debit
            net_debit = long_option["ask"] - short_option["bid"]
            
            # Skip if net debit is negative (should be a debit spread)
            if net_debit <= 0:
                continue
                
            # Calculate metrics for the spread
            spread_metrics = calculate_diagonal_spread_metrics(
                long_option, short_option, current_price, option_type
            )
            
            # Create candidate spread
            candidate = {
                "ticker": ticker,
                "strategy_type": "diagonal",
                "trend_direction": trend_direction,
                "option_type": option_type,
                "long_leg": long_option,
                "short_leg": short_option,
                "net_debit": net_debit,
                "max_profit": spread_metrics["max_profit"],
                "max_loss": spread_metrics["max_loss"],
                "profit_loss_ratio": spread_metrics["profit_loss_ratio"],
                "strike_width_pct": strike_diff_pct,
                "theta_advantage": spread_metrics["theta_advantage"],
                "vega_exposure": spread_metrics["vega_exposure"],
                "score": spread_metrics["score"]
            }
            
            candidates.append(candidate)
    
    # Sort by score
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    return candidates

def calculate_diagonal_spread_metrics(
    long_option: Dict,
    short_option: Dict,
    current_price: float,
    option_type: str
) -> Dict:
    """
    Calculate key metrics for a diagonal spread.
    
    Args:
        long_option: Long option leg details
        short_option: Short option leg details
        current_price: Current price of the underlying
        option_type: "call" or "put"
        
    Returns:
        Dictionary of calculated metrics
    """
    # Calculate net debit (cost of spread)
    net_debit = long_option["ask"] - short_option["bid"]
    
    # Calculate time value differential (theta advantage)
    long_time_value = calculate_time_value(
        option_price=(long_option["bid"] + long_option["ask"]) / 2,
        strike=long_option["strike"],
        current_price=current_price,
        option_type=option_type
    )
    
    short_time_value = calculate_time_value(
        option_price=(short_option["bid"] + short_option["ask"]) / 2,
        strike=short_option["strike"],
        current_price=current_price,
        option_type=option_type
    )
    
    time_value_ratio = short_time_value / long_time_value if long_time_value > 0 else 0
    theta_per_day = (short_time_value / short_option["dte"]) - (long_time_value / long_option["dte"])
    
    # Estimated theta advantage
    theta_advantage = theta_per_day * min(short_option["dte"], 30)  # 30 days max
    
    # Calculate vega exposure (sensitivity to volatility changes)
    vega_exposure = long_option.get("vega", long_option["implied_vol"] * 0.1) - short_option.get("vega", short_option["implied_vol"] * 0.1)
    
    # Estimate max profit and loss
    # This is a simplified calculation - real P/L depends on price movement and vol changes
    if option_type == "call":
        # Bullish diagonal
        max_profit_estimate = (
            abs(long_option["strike"] - short_option["strike"]) * 100 + short_option["bid"] * 100 - long_option["ask"] * 100
        )
        max_loss_estimate = net_debit * 100
    else:
        # Bearish diagonal
        max_profit_estimate = (
            abs(long_option["strike"] - short_option["strike"]) * 100 + short_option["bid"] * 100 - long_option["ask"] * 100
        )
        max_loss_estimate = net_debit * 100
    
    # Ensure max profit is positive
    max_profit_estimate = max(0, max_profit_estimate)
    
    # Calculate profit/loss ratio
    profit_loss_ratio = max_profit_estimate / max_loss_estimate if max_loss_estimate > 0 else 0
    
    # Calculate a composite score (0-100)
    score_components = []
    
    # 1. Profit/Loss ratio (higher is better)
    pl_score = min(100, profit_loss_ratio * 50)
    score_components.append(pl_score * 0.3)  # 30% weight
    
    # 2. Theta advantage (higher is better)
    theta_score = min(100, theta_advantage * 20)
    score_components.append(theta_score * 0.3)  # 30% weight
    
    # 3. Strike width (closer to optimal is better)
    strike_width_pct = abs(long_option["strike"] - short_option["strike"]) / current_price * 100
    optimal_width = 2.0  # 2% width is optimal for balance
    width_deviation = abs(strike_width_pct - optimal_width)
    width_score = max(0, 100 - width_deviation * 20)  # Penalize deviation from optimal
    score_components.append(width_score * 0.2)  # 20% weight
    
    # 4. Liquidity score
    avg_volume = (long_option["volume"] + short_option["volume"]) / 2
    volume_score = min(100, avg_volume / 5)
    score_components.append(volume_score * 0.1)  # 10% weight
    
    # 5. DTE alignment score
    # Best when short leg is ~1/3 of long leg DTE
    optimal_dte_ratio = 0.33
    actual_dte_ratio = short_option["dte"] / long_option["dte"] if long_option["dte"] > 0 else 0
    dte_score = max(0, 100 - abs(actual_dte_ratio - optimal_dte_ratio) * 200)
    score_components.append(dte_score * 0.1)  # 10% weight
    
    # Calculate final score
    final_score = sum(score_components)
    
    return {
        "net_debit": net_debit,
        "max_profit": max_profit_estimate,
        "max_loss": max_loss_estimate,
        "profit_loss_ratio": profit_loss_ratio,
        "theta_advantage": theta_advantage,
        "vega_exposure": vega_exposure,
        "time_value_ratio": time_value_ratio,
        "score": final_score
    }

def calculate_time_value(
    option_price: float,
    strike: float,
    current_price: float,
    option_type: str
) -> float:
    """
    Calculate the time value component of an option's price.
    
    Args:
        option_price: Current option price
        strike: Strike price
        current_price: Current price of the underlying
        option_type: "call" or "put"
        
    Returns:
        Time value component of the option price
    """
    if option_type == "call":
        intrinsic_value = max(0, current_price - strike)
    else:
        intrinsic_value = max(0, strike - current_price)
    
    time_value = max(0, option_price - intrinsic_value)
    return time_value

def calculate_diagonal_spread_adjustments(
    spread: Dict,
    current_price: float,
    days_passed: int
) -> Dict:
    """
    Calculate potential adjustments for a diagonal spread based on price movement.
    
    Args:
        spread: Diagonal spread details
        current_price: Current price of the underlying
        days_passed: Days since entry
        
    Returns:
        Dictionary of potential adjustments
    """
    original_direction = spread["trend_direction"]
    option_type = spread["option_type"]
    
    # Calculate price movement since entry
    entry_price = spread.get("entry_price", current_price)
    price_change_pct = (current_price - entry_price) / entry_price * 100
    
    # Calculate time decay
    short_dte_remaining = spread["short_leg"]["dte"] - days_passed
    long_dte_remaining = spread["long_leg"]["dte"] - days_passed
    
    adjustments = {
        "should_roll_short": False,
        "should_close_spread": False,
        "should_adjust_direction": False,
        "roll_recommendation": None,
        "target_strike": None,
        "explanation": ""
    }
    
    # Check if short leg needs rolling (5-7 DTE)
    if short_dte_remaining <= 7:
        adjustments["should_roll_short"] = True
        adjustments["explanation"] = f"Short leg approaching expiration ({short_dte_remaining} DTE). Consider rolling to avoid gamma risk."
        
        # Recommend target expiration for roll
        target_dte = min(30, long_dte_remaining - 10)  # At least 10 days before long leg
        adjustments["roll_recommendation"] = {
            "target_dte": target_dte,
            "current_short_strike": spread["short_leg"]["strike"]
        }
    
    # Check directional adjustments based on price movement
    if (original_direction == "bullish" and price_change_pct < -3) or \
       (original_direction == "bearish" and price_change_pct > 3):
        # Price moved against our bias
        adjustments["should_adjust_direction"] = True
        adjustments["explanation"] += " Price has moved against the original bias."
        
        if adjustments["should_roll_short"]:
            # Suggest strike adjustment during roll
            if original_direction == "bullish":
                # Moved down, adjust strikes lower
                adjustments["target_strike"] = spread["short_leg"]["strike"] * 0.98
            else:
                # Moved up, adjust strikes higher
                adjustments["target_strike"] = spread["short_leg"]["strike"] * 1.02
    
    # Check if we should close the entire spread
    if (long_dte_remaining <= 14) or \
       (price_change_pct > 10 and original_direction == "bullish") or \
       (price_change_pct < -10 and original_direction == "bearish"):
        # Long leg approaching expiration or price moved strongly in our favor
        adjustments["should_close_spread"] = True
        adjustments["explanation"] += " Consider closing the entire spread due to expiration or target reached."
    
    return adjustments

def calculate_roll_cost(
    spread: Dict,
    new_short_expiry: str,
    new_short_strike: float,
    option_chain: Dict
) -> Dict:
    """
    Calculate the cost and P/L impact of rolling a short option.
    
    Args:
        spread: Current diagonal spread
        new_short_expiry: Target expiration for new short leg
        new_short_strike: Target strike for new short leg
        option_chain: Current option chain data
        
    Returns:
        Dictionary with roll details and cost
    """
    option_type = spread["option_type"]
    current_short = spread["short_leg"]
    
    # Find the new short option in the chain
    new_short_option = None
    if new_short_expiry in option_chain:
        for option in option_chain[new_short_expiry]:
            if (option["option_type"] == option_type and 
                abs(option["strike"] - new_short_strike) < 0.01):
                new_short_option = option
                break
    
    if new_short_option is None:
        return {
            "success": False,
            "message": "Could not find matching option for roll",
            "cost": 0
        }
    
    # Calculate the cost to close the current short
    close_current_cost = current_short["ask"] - current_short["bid"]
    
    # Calculate the credit from selling the new short
    new_short_credit = new_short_option["bid"]
    
    # Net cost of the roll
    roll_cost = close_current_cost - new_short_credit
    
    # Calculate new spread metrics after roll
    new_short_details = {
        "expiry": new_short_expiry,
        "strike": new_short_strike,
        "bid": new_short_option["bid"],
        "ask": new_short_option["ask"],
        "delta": abs(new_short_option.get("delta", 0)),
        "volume": new_short_option.get("volume", 0),
        "open_interest": new_short_option.get("open_interest", 0),
        "implied_vol": new_short_option.get("implied_volatility", 0),
        "dte": (datetime.strptime(new_short_expiry, "%Y-%m-%d").date() - datetime.now().date()).days
    }
    
    # Calculate new spread details
    new_spread_metrics = calculate_diagonal_spread_metrics(
        spread["long_leg"],
        new_short_details,
        spread.get("current_price", 0),
        option_type
    )
    
    return {
        "success": True,
        "original_short_strike": current_short["strike"],
        "original_short_expiry": current_short["expiry"],
        "new_short_strike": new_short_strike,
        "new_short_expiry": new_short_expiry,
        "close_cost": close_current_cost,
        "new_short_credit": new_short_credit,
        "net_roll_cost": roll_cost,
        "new_metrics": new_spread_metrics
    }

def analyze_diagonal_spread_performance(
    spread: Dict,
    entry_date: datetime,
    exit_date: Optional[datetime] = None,
    price_history: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Analyze the performance of a diagonal spread over time.
    
    Args:
        spread: Diagonal spread details
        entry_date: Date of entry
        exit_date: Date of exit (None if still active)
        price_history: DataFrame with price history for the underlying
        
    Returns:
        Dictionary with performance metrics
    """
    # Default values
    metrics = {
        "days_held": 0,
        "price_change_pct": 0,
        "theta_capture": 0,
        "realized_pnl": 0,
        "pnl_explanation": "",
        "annualized_return": 0
    }
    
    if exit_date is None:
        exit_date = datetime.now()
    
    # Calculate days held
    days_held = (exit_date - entry_date).days
    metrics["days_held"] = days_held
    
    if price_history is not None and not price_history.empty:
        # Get entry and exit prices
        entry_price_data = price_history[price_history.index <= entry_date]
        exit_price_data = price_history[price_history.index <= exit_date]
        
        if not entry_price_data.empty and not exit_price_data.empty:
            entry_price = entry_price_data.iloc[-1]["close"]
            exit_price = exit_price_data.iloc[-1]["close"]
            
            # Calculate price change
            price_change_pct = (exit_price - entry_price) / entry_price * 100
            metrics["price_change_pct"] = price_change_pct
            
            # Estimate theta capture (decay of time value)
            # This is a simplified calculation
            short_leg = spread["short_leg"]
            original_time_value = calculate_time_value(
                (short_leg["bid"] + short_leg["ask"]) / 2,
                short_leg["strike"],
                entry_price,
                spread["option_type"]
            )
            
            # Simplified theta capture (proportional to days passed)
            if "exit_short_value" in spread:
                actual_decay = original_time_value - spread["exit_short_value"]
                metrics["theta_capture"] = actual_decay
            else:
                # Estimate based on typical decay curve
                expected_decay_ratio = min(1.0, days_held / short_leg["dte"])
                metrics["theta_capture"] = original_time_value * expected_decay_ratio
    
    # Calculate P&L if available
    if "entry_cost" in spread and "exit_value" in spread:
        pnl = spread["exit_value"] - spread["entry_cost"]
        metrics["realized_pnl"] = pnl
        
        # Calculate annualized return
        if days_held > 0 and spread["entry_cost"] > 0:
            metrics["annualized_return"] = (pnl / spread["entry_cost"]) * (365 / days_held) * 100
        
        # P&L explanation
        direction = spread["trend_direction"]
        if metrics["price_change_pct"] > 0 and direction == "bullish":
            metrics["pnl_explanation"] = "Profitable from directional move and time decay"
        elif metrics["price_change_pct"] < 0 and direction == "bearish":
            metrics["pnl_explanation"] = "Profitable from directional move and time decay"
        elif metrics["realized_pnl"] > 0:
            metrics["pnl_explanation"] = "Profitable primarily from time decay"
        else:
            metrics["pnl_explanation"] = "Loss from adverse price movement exceeding time decay"
    
    return metrics

# TODO: Add function to calculate optimal DTE for different volatility regimes
# TODO: Add function to calculate IV skew impact on diagonal spreads
# TODO: Add function to evaluate multiple diagonal spread variations 