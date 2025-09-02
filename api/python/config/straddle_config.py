"""
Configuration for straddle options strategy.

This module contains the default configuration parameters for the straddle strategy,
including scoring weights, entry/exit criteria, and risk management settings.
"""

# Default configuration for the straddle strategy
STRADDLE_CONFIG = {
    # Watchlist of symbols to analyze
    "watchlist": [
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
    
    # Entry criteria
    "entry_criteria": {
        "min_iv_rank": 60,            # Minimum IV rank (0-100)
        "min_option_volume": 100,     # Minimum option volume
        "max_bid_ask_spread": 0.10,   # Maximum bid-ask spread as decimal
        "min_score": 70,              # Minimum opportunity score (0-100)
        "min_days_to_expiration": 20, # Minimum days to expiration
        "max_days_to_expiration": 60, # Maximum days to expiration
        "ignore_iv_rank": False       # Whether to ignore IV rank requirement
    },
    
    # Strike selection criteria
    "strike_selection": {
        "method": "atm",              # Method: "atm", "delta", "expected_move"
        "target_delta": 0.5,          # Target delta if using delta method
        "expected_move_factor": 0.5,  # Factor for expected move method
    },
    
    # Risk management parameters
    "risk_management": {
        "max_positions": 3,           # Maximum number of concurrent positions
        "max_position_size": 0.05,    # Maximum position size as fraction of account
        "position_sizing": "fixed",   # Position sizing method: "fixed", "risk_percent", "kelly"
        "fixed_contracts": 1,         # Fixed number of contracts if using fixed sizing
        "risk_percent": 0.01,         # Risk percentage if using risk_percent sizing
        "stop_loss_pct": 0.50,        # Stop loss percentage
        "profit_target_pct": 1.0,     # Profit target percentage
        "max_days_held": 30,          # Maximum days to hold a position
        "min_profit_target": 0.5      # Minimum profit target ratio
    },
    
    # Exit criteria
    "exit_criteria": {
        "time_decay_exit_days": 14    # Exit when this many days remain until expiration
    },
    
    # Scoring weights for ranking opportunities
    "scoring_weights": {
        "iv_rank": 0.25,              # Weight for IV rank
        "iv_vs_historical_ratio": 0.15, # Weight for current IV vs historical IV ratio
        "bid_ask_spread": 0.15,       # Weight for bid-ask spread
        "option_volume": 0.15,        # Weight for option volume
        "days_to_expiration": 0.10,   # Weight for days to expiration
        "probability_of_profit": 0.20 # Weight for probability of profit
    },
    
    # Optimization settings
    "optimization": {
        "enabled": False,             # Whether to enable auto-optimization
        "min_trades_for_optimization": 20, # Minimum number of trades before optimizing
        "optimization_interval_days": 30,  # Days between optimization runs
        "lookback_days": 180          # Lookback period for optimization
    }
} 