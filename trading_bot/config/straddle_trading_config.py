#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Straddle Trading Strategy Configuration

This file contains the configurable parameters for the straddle trading strategy.
Users can modify these settings to customize the strategy behavior.
"""

# Straddle Trading Strategy Configuration
STRADDLE_TRADING_CONFIG = {
    # Section 1: Strategy Philosophy
    "strategy_description": "Capture large, rapid moves in either direction around high-volatility events by buying a call and a put at the same strike. You're long pure volatility (vega exposure) and profit when realized moves exceed the combined premium paid.",
    
    # Section 2: Market Universe & Event Selection
    "eligible_symbols": ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "AMZN", "GOOG"],  # Highly liquid large-caps or ETFs
    "min_underlying_liquidity": 1000000,  # 1M minimum average daily volume
    "eligible_catalysts": [
        "earnings", 
        "fda_decision", 
        "economic_release", 
        "technical_breakout",
        "product_launch",
        "analyst_day",
        "conference_presentation"
    ],
    "holding_period_min": 1,  # Minimum holding period in days
    "holding_period_max": 5,  # Maximum holding period in days
    
    # Section 3: Option & Strike Selection
    "strike_atm_offset": 0,  # 0 means exact ATM, 1 means 1 strike above, -1 means 1 strike below
    "min_days_to_expiration": 20,  # Minimum DTE
    "max_days_to_expiration": 30,  # Maximum DTE
    "breakeven_move_target_min": 0.05,  # 5% minimum target for breakeven move
    "breakeven_move_target_max": 0.08,  # 8% maximum target for breakeven move
    
    # Section 4: Greeks & Risk Filters
    "min_vega": 0.15,  # Minimum vega requirement per $1 of underlying
    "max_theta_burn_daily": 0.01,  # Maximum theta burn rate as % of premium per day
    "delta_neutrality_threshold": 0.05,  # Maximum allowed delta imbalance
    "min_iv_rank": 60,  # Minimum IV rank, as a percentage
    "max_iv_percentile": 90,  # Maximum IV percentile, avoid extremely over-priced options
    
    # Section 5: Entry Criteria
    "min_option_open_interest": 1000,  # Minimum open interest for option legs
    "max_bid_ask_spread_pct": 0.10,  # Maximum bid-ask spread as percentage
    "gap_check_days": 3,  # Number of recent days to check for extreme gaps
    "max_gap_percentage": 0.04,  # Maximum allowed gap size as percentage of price
    
    # Section 6: Position Sizing & Risk Controls
    "risk_pct_per_trade": 1.0,  # Maximum risk per trade as % of account
    "max_concurrent_positions": 3,  # Maximum number of concurrent straddles
    "capital_buffer_pct": 120,  # Ensure buffer of at least 120% of total premium
    
    # Section 7: Exit Rules
    "profit_take_pct": 65,  # Take profit at % of max potential gain
    "pre_event_exit_days": 1,  # Exit this many days before the event if not triggered
    "post_event_exit_days": 1,  # Exit within this many days after the event
    "directional_adjustment_threshold": 1.0,  # If move exceeds breakeven by this multiple, consider leg adjustment
    
    # Section 8: Order Execution
    "combo_order_enabled": True,  # Use combo order for simultaneous execution
    "slippage_tolerance_cents": 2,  # Maximum allowed slippage in cents
    "mid_price_buffer_pct": 0.02,  # Target mid-price with this buffer
    
    # Section 9: Performance & Evaluation Metrics
    "track_vega_capture": True,  # Track vega capture ratio
    "track_theta_efficiency": True,  # Track theta burn efficiency
    "track_win_rate": True,  # Track win rate by catalyst type
    "track_max_drawdown": True,  # Track max drawdown
    
    # Section 10: Continuous Optimization
    "iv_recalibration_frequency_days": 30,  # Days between IV threshold recalibrations
    "dynamic_hedging_enabled": False,  # Enable conversion to strangle after large move
    "ml_overlay_enabled": False,  # Enable ML model for entry optimization
}

# Advanced options (for experienced users)
ADVANCED_OPTIONS = {
    "use_delta_weighted_strikes": False,  # Use delta-weighted strike selection instead of price-based
    "target_delta": 0.50,  # Target delta for ATM options
    "vega_target_scaling": True,  # Scale position size based on vega exposure
    "auto_skew_adjustment": False,  # Automatically adjust for skew when selecting strikes
    "gamma_scalping_enabled": False,  # Enable gamma scalping for active management
    "earnings_iv_premium_factor": 1.2,  # Additional factor for earnings IV premium
    "non_earnings_iv_premium_factor": 1.1,  # Additional factor for non-earnings IV premium
    "leverage_factor": 1.0,  # Multiplier for position size (use with extreme caution)
    "rolling_enabled": False,  # Enable automatic rolling to capture more premium
    "option_chain_data_source": "default",  # Source for option chain data
    "event_data_source": "default"  # Source for event calendar data
}

# Event categorization
EVENT_IMPORTANCE = {
    "earnings": "high",
    "fda_decision": "high",
    "product_launch": "medium",
    "economic_release": "variable",  # Depends on specific release
    "technical_breakout": "medium",
    "analyst_day": "medium",
    "conference_presentation": "low"
}

# Function to get full configuration
def get_straddle_config(include_advanced=False):
    """
    Get the straddle trading configuration.
    
    Args:
        include_advanced: Whether to include advanced options
        
    Returns:
        Combined configuration dictionary
    """
    config = STRADDLE_TRADING_CONFIG.copy()
    
    if include_advanced:
        config.update(ADVANCED_OPTIONS)
    
    config["event_importance"] = EVENT_IMPORTANCE
    
    return config 

"""
Configuration for the Straddle options trading strategy.

This file contains all parameters used by the straddle strategy
for entry/exit criteria, risk management, and other settings.
"""

STRADDLE_CONFIG = {
    # Symbols to analyze
    "watchlist": {
        "symbols": ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "AMD", "FB", "GOOG", "SPY", "QQQ"],
        "include_earnings_events": True,
        "include_economic_events": True,
        "include_corporate_events": True,
    },
    
    # Entry criteria
    "entry_criteria": {
        "iv_rank_threshold": 60,  # Only consider stocks with IV rank above this level
        "min_average_volume": 500000,  # Minimum average daily volume
        "min_option_volume": 50,  # Minimum option contract volume
        "max_bid_ask_spread_pct": 10,  # Maximum bid-ask spread as percentage
        "days_before_event": 7,  # Maximum days before event to enter
        "price_check": True,  # Check price history before entering
    },
    
    # Strike selection
    "strike_selection": {
        "strike_selection_method": "ATM",  # ATM, DELTA, CUSTOM
        "strike_offset_pct": 0,  # Offset from current price (0 = ATM)
        "delta_target": 0.50,  # Target delta if using DELTA method
        "min_expiration_days": 14,  # Minimum days to expiration
        "max_expiration_days": 45,  # Maximum days to expiration
    },
    
    # Risk management
    "risk_management": {
        "position_sizing_pct": 5,  # Percent of account per position
        "max_concurrent_positions": 5,  # Maximum number of concurrent positions
        "profit_target_pct": 50,  # Take profit at this percentage gain
        "max_loss_pct": 25,  # Close position at this percentage loss
        "default_account_size": 100000,  # Default account size for calculations
        "close_after_event": True,  # Close position after the event has passed
        "days_after_event": 1,  # Number of days after event to close position
    },
    
    # Gap analysis
    "gap_analysis": {
        "enabled": True,  # Enable gap analysis
        "lookback_days": 30,  # Lookback period for gap analysis
        "gap_threshold_pct": 2.0,  # Minimum gap size to consider
    },

    # Logging and notifications
    "logging": {
        "enable_detailed_logging": True,  # Enable detailed logging
        "save_trade_history": True,  # Save trade history to CSV
        "notification_level": "INFO",  # Level for notifications
    },
    
    # Optimization parameters
    "optimization": {
        "optimization_metric": "sharpe_ratio",  # Metric to optimize for
        "lookback_days": 90,  # Lookback period for optimization
        "enable_auto_optimization": False,  # Enable automatic optimization
        "optimization_frequency": 30,  # Days between optimizations
    },
    
    # Backtest parameters
    "backtest": {
        "start_date": "2020-01-01",
        "end_date": "2023-12-31",
        "initial_capital": 100000,
        "commission_per_contract": 0.65,
        "slippage_model": "percent",
        "slippage_value": 0.05,
    }
} 