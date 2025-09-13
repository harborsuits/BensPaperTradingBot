"""
Configuration for diagonal spread options strategy.

This module contains the default configuration parameters for the diagonal spread strategy,
including selection criteria, entry/exit rules, and risk management settings.
"""

# Default configuration for the diagonal spread strategy
DIAGONAL_SPREAD_CONFIG = {
    # Section 1: Strategy Philosophy - Implemented in strategy class
    
    # Section 2: Underlying & Option Universe & Timeframe
    "universe": {
        "symbols": [
            "SPY",   # S&P 500 ETF
            "QQQ",   # Nasdaq ETF
            "IWM",   # Russell 2000 ETF
            "AAPL",  # Apple
            "MSFT",  # Microsoft
            "NVDA",  # NVIDIA
            "AMD",   # AMD
            "TSLA",  # Tesla
            "AMZN",  # Amazon
            "META"   # Meta
        ],
        "long_leg_min_dte": 60,  # Minimum DTE for long option
        "long_leg_max_dte": 120, # Maximum DTE for long option
        "short_leg_min_dte": 7,  # Minimum DTE for short option
        "short_leg_max_dte": 30, # Maximum DTE for short option
        "target_holding_period_weeks": 4, # Target holding period in weeks
    },
    
    # Section 3: Selection Criteria for Underlying
    "underlying_criteria": {
        "min_adv": 500000,         # Minimum average daily volume
        "min_option_open_interest": 500, # Minimum open interest for options
        "max_bid_ask_spread_pct": 0.15,  # Maximum bid-ask spread as percentage
        "min_iv_rank": 30,         # Minimum IV rank (0-100)
        "max_iv_rank": 60,         # Maximum IV rank (0-100)
        "trend_period_days": 20,   # Number of days to measure trend
        "trend_strength_threshold": 0.6, # Threshold for trend strength (0-1)
    },
    
    # Section 4: Spread Construction
    "spread_construction": {
        "long_leg_min_delta": 0.30, # Minimum delta for long leg
        "long_leg_max_delta": 0.50, # Maximum delta for long leg
        "short_leg_min_delta": 0.15, # Minimum delta for short leg
        "short_leg_max_delta": 0.30, # Maximum delta for short leg
        "strike_offset_pct": 2.0,   # Target percentage offset between strikes
        "max_net_debit_pct": 1.0,   # Maximum net debit as % of account equity
        "preferred_ratio": 1,       # Default leg ratio (usually 1:1)
    },
    
    # Section 5: Expiration & Roll Timing
    "expiration_management": {
        "entry_short_leg_min_dte": 14, # Minimum DTE for short leg at entry
        "entry_short_leg_max_dte": 30, # Maximum DTE for short leg at entry
        "entry_long_leg_min_dte": 60,  # Minimum DTE for long leg at entry
        "entry_long_leg_max_dte": 90,  # Maximum DTE for long leg at entry
        "roll_short_leg_dte": 7,       # DTE to roll/close short leg
        "roll_long_leg_dte": 30,       # DTE to evaluate long leg
    },
    
    # Section 6: Entry Execution
    "entry_execution": {
        "use_combo_order": True,     # Use combo order for both legs
        "target_price_level": "mid", # Target price level: "mid", "bid", "ask"
        "max_slippage_pct": 5.0,     # Maximum allowed slippage percentage
        "max_retry_attempts": 3,     # Max retry attempts for order execution
        "retry_delay_seconds": 30,   # Delay between retry attempts
    },
    
    # Section 7: Exit & Adjustment Rules
    "exit_rules": {
        "profit_take_pct": 60.0,     # Take profit at X% of max theoretical value
        "stop_loss_pct": 100.0,      # Stop loss at X% of initial debit
        "time_exit_short_leg_dte": 7, # Exit short leg when DTE reaches this value
        "time_exit_long_leg_dte": 30, # Exit long leg when DTE reaches this value
        "directional_adjustment_threshold": 2.0, # % move to trigger adjustment
    },
    
    # Section 8: Position Sizing & Risk Controls
    "risk_management": {
        "max_risk_per_spread_pct": 1.0, # Maximum risk per spread as % of account
        "max_positions": 5,             # Maximum number of concurrent positions
        "max_margin_usage_pct": 10.0,   # Maximum margin usage per spread
        "max_positions_per_sector": 1,  # Maximum positions per correlated sector
        "sector_correlation_matrix": {
            "technology": ["AAPL", "MSFT", "NVDA", "AMD"],
            "consumer": ["AMZN", "META", "TSLA"],
            "index": ["SPY", "QQQ", "IWM"]
        }
    },
    
    # Section 9: Backtesting & Performance Metrics - Implemented in strategy class
    
    # Section 10: Continuous Optimization
    "optimization": {
        "enabled": False,                # Whether to enable auto-optimization
        "optimization_frequency_days": 30, # Days between optimization runs
        "iv_adaptive_threshold": 20.0,   # % deviation from median IV to adapt
        "stacked_diagonal_enabled": False, # Enable stacked diagonals
        "use_ml_overlay": False,         # Use ML overlay for entry refinement
        "lookback_period_days": 90,      # Lookback period for optimization
    }
} 