#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Butterfly Spread Strategy Configuration

This file contains the configurable parameters for the butterfly spread options strategy.
Users can modify these settings to customize the strategy behavior.
"""

# Butterfly Spread Strategy Configuration
BUTTERFLY_SPREAD_CONFIG = {
    # Section 1: Strategy Philosophy
    "strategy_description": "Capitalize on low-volatility, range-bound expectations by selling narrowly-spaced wings and hedging with outer wings—maximizing premium capture when the underlying stays near your chosen strike at expiration.",
    
    # Section 2 & 3: Underlying & Selection Criteria
    "eligible_symbols": ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "AMZN", "GOOG"],  # Highly liquid large-caps or ETFs
    "min_underlying_adv": 1000000,  # 1M minimum average daily volume
    "min_option_open_interest": 500,  # Minimum open interest per leg
    "max_bid_ask_spread_pct": 0.10,  # Maximum allowable bid-ask spread as percentage
    "iv_rank_min": 20,  # Minimum IV rank (%)
    "iv_rank_max": 50,  # Maximum IV rank (%)
    "range_days_lookback": 15,  # Days to check for range-bound behavior
    
    # Section 4: Spread Construction
    "center_strike_delta": 0.50,  # Target delta for center strike (0.50 = ATM)
    "center_strike_offset": 0,  # Offset from ATM in # of strikes (0 = ATM)
    "inner_wing_width": 1,  # Width in # of strikes from center to inner wings
    "outer_wing_width": 2,  # Width in # of strikes from center to outer wings
    "target_net_debit": 0.05,  # Target net debit as % of underlying price (negative for credit)
    
    # Section 5: Expiration Selection
    "min_days_to_expiration": 25,  # Minimum DTE for entry
    "max_days_to_expiration": 45,  # Maximum DTE for entry
    "exit_dte_threshold": 8,  # Exit when DTE falls below this value
    "prefer_monthly_expirations": True,  # Prefer monthly cycles over weeklies
    
    # Section 7: Exit & Management Rules
    "profit_take_pct": 65,  # Take profit at % of max potential gain
    "stop_loss_multiplier": 1.5,  # Stop loss at multiple of max potential loss
    "management_threshold_delta": 0.10,  # Delta threshold for adjustment
    "wing_adjustment_enabled": True,  # Enable wing adjustments
    "convert_to_iron_fly_enabled": True,  # Enable conversion to iron fly when appropriate
    
    # Section 8: Position Sizing & Risk
    "risk_pct_per_trade": 1.0,  # Maximum risk per trade as % of account
    "max_concurrent_positions": 4,  # Maximum number of concurrent butterflies
    "max_margin_pct": 5.0,  # Maximum margin requirement as % of account
    "max_correlation_threshold": 0.70,  # Maximum correlation between concurrent positions
    
    # Section 10: Continuous Optimization
    "iv_adjustment_threshold": 20,  # IV rank change threshold for wing adjustment
    "dynamic_recentering": True,  # Whether to allow dynamic re-centering
    "ml_model_enabled": False,  # Enable ML overlay for strike selection
    "auto_optimization_period": 30,  # Number of days between auto-optimization runs
}

# Advanced options (for more experienced users)
ADVANCED_OPTIONS = {
    "use_custom_leg_ratios": False,  # Enable custom leg ratios beyond 1:2:1
    "custom_leg_ratio": [1, 2, 1],  # Default butterfly ratio [outer:inner:outer]
    "use_weekly_cycles": False,  # Use weekly cycles instead of monthlies
    "legging_order": ["inner_longs", "shorts", "outer_longs"],  # Order for legging into position
    "min_liquidity_ratio": 0.5,  # Minimum ratio of volume to open interest
    "black_scholes_model": "european",  # Option pricing model to use
    "adjustment_timing": "immediate",  # When to apply adjustments (immediate/end_of_day)
    "use_volatility_forecast": False,  # Use forecasted volatility instead of current
    "correlation_lookback": 30,  # Days to calculate correlation between underlyings
    "allow_early_exit": True,  # Allow exiting before profit target if conditions change
}

# Performance tracking metrics
PERFORMANCE_METRICS = {
    "track_win_rate": True,
    "track_avg_return": True,
    "track_max_drawdown": True,
    "track_theta_decay": True,
    "track_iv_changes": True,
    "track_adjustment_impact": True
}

# Function to get full configuration
def get_butterfly_config(include_advanced=False):
    """
    Get the butterfly spread configuration.
    
    Args:
        include_advanced: Whether to include advanced options
        
    Returns:
        Combined configuration dictionary
    """
    config = BUTTERFLY_SPREAD_CONFIG.copy()
    
    if include_advanced:
        config.update(ADVANCED_OPTIONS)
    
    config["performance_metrics"] = PERFORMANCE_METRICS
    
    return config 