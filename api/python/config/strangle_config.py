"""
Configuration settings for the strangle options strategy.

This module defines the default configuration parameters used by the
strangle strategy implementation.
"""

STRANGLE_CONFIG = {
    # Symbols to scan for strangle opportunities
    "watchlist": [
        "SPY",  # S&P 500 ETF
        "QQQ",  # Nasdaq ETF
        "IWM",  # Russell 2000 ETF
        "AAPL", # Apple
        "MSFT", # Microsoft
        "NVDA", # NVIDIA
        "AMZN", # Amazon
        "META", # Meta Platforms
        "GOOGL", # Alphabet
        "TSLA"  # Tesla
    ],
    
    # Entry criteria
    "entry_criteria": {
        # Minimum IV rank to consider (0-100)
        "min_iv_rank": 60,
        
        # Minimum option volume
        "min_option_volume": 100,
        
        # Minimum open interest
        "min_open_interest": 500,
        
        # Maximum bid-ask spread as percentage of mid price
        "max_bid_ask_spread_pct": 5.0,
        
        # Minimum days to expiration
        "min_days_to_expiration": 14,
        
        # Maximum days to expiration
        "max_days_to_expiration": 45,
        
        # Price gap threshold for gap analysis (percentage)
        "price_gap_threshold": 3.0,
        
        # Minimum expected event score to consider (0-100)
        "min_event_score": 70
    },
    
    # Strike selection criteria
    "strike_selection": {
        # Selection method: "delta" or "otm_percentage"
        "method": "delta",
        
        # Target delta for call leg (positive value)
        "target_delta_call": 0.16,
        
        # Target delta for put leg (positive value)
        "target_delta_put": 0.16,
        
        # Alternative: OTM percentage for strike selection
        "target_otm_pct_call": 7.0,
        "target_otm_pct_put": 7.0,
        
        # Tolerance for delta or OTM percentage matching
        "delta_tolerance": 0.03,
        "otm_pct_tolerance": 1.0
    },
    
    # Risk management
    "risk_management": {
        # Maximum percentage of account to risk per trade
        "max_risk_per_trade_pct": 2.0,
        
        # Maximum number of open positions
        "max_open_positions": 10,
        
        # Maximum percentage of account in single underlying
        "max_underlying_exposure_pct": 10.0,
        
        # Maximum correlated positions (positions with >0.7 correlation)
        "max_correlated_positions": 3,
        
        # Maximum sector exposure (percentage of account)
        "max_sector_exposure_pct": 25.0,
        
        # Stop loss percentage (based on premium paid)
        "stop_loss_pct": 80.0,
        
        # Trailing stop activation threshold (profit percentage)
        "trailing_stop_activation_pct": 50.0,
        
        # Trailing stop distance (percentage of maximum profit)
        "trailing_stop_distance_pct": 20.0
    },
    
    # Exit criteria
    "exit_criteria": {
        # Profit target as percentage of premium paid
        "profit_target_pct": 50.0,
        
        # Days before expiration to consider rolling/closing
        "days_before_expiration_exit": 7,
        
        # Exit if IV drops below this percentage of entry IV
        "iv_drop_exit_threshold": 0.7,
        
        # Exit if IV spikes above this percentage of entry IV
        "iv_spike_exit_threshold": 1.5,
        
        # Exit if probability of profit falls below this threshold
        "min_probability_to_maintain": 15.0,
        
        # Maximum drawdown percentage before early management
        "max_drawdown_pct": 25.0,
        
        # Exit before events flag (True/False)
        "exit_before_events": True,
        
        # Days before known event to exit
        "days_before_event_exit": 1
    },
    
    # Greeks filters
    "greeks_filters": {
        # Acceptable delta range for entire position
        "delta_neutrality_range": 0.15,
        
        # IV rank range for entry
        "iv_rank": {
            "min": 60,
            "max": 95
        },
        
        # Maximum theta as percentage of premium per day
        "max_theta_pct_daily": 1.5,
        
        # Minimum vega exposure per position
        "min_vega": 0.05,
        
        # Minimum gamma to consider
        "min_gamma": 0.01
    },
    
    # Event-driven settings
    "event_settings": {
        # Prioritize these event types (earnings, FDA, dividends, etc.)
        "priority_event_types": ["earnings", "fed_meeting", "economic_release"],
        
        # Days before event to enter position
        "days_before_event_entry": 5,
        
        # Weight of event surprise history in scoring
        "event_surprise_history_weight": 0.6,
        
        # Adjust IV expectation based on historical events
        "use_historical_iv_behavior": True
    },
    
    # Scoring weights for opportunity evaluation
    "scoring_weights": {
        # IV rank weight in overall score
        "iv_rank": 0.15,
        
        # Probability of profit weight
        "probability_of_profit": 0.20,
        
        # Risk/reward ratio weight
        "risk_reward": 0.15,
        
        # Theta burn rate weight
        "theta_burn": 0.10,
        
        # Premium vs expected move weight
        "premium_vs_expected_move": 0.15,
        
        # Delta neutrality weight
        "delta_neutrality": 0.05,
        
        # Liquidity score weight
        "liquidity": 0.10,
        
        # Event quality weight
        "event_quality": 0.10
    },
    
    # Optimization settings
    "optimization": {
        # Enable auto-optimization
        "enabled": True,
        
        # Minimum trades before optimization
        "min_trades_before_optimization": 20,
        
        # Optimization frequency (in days)
        "optimization_frequency_days": 30,
        
        # Parameters to optimize
        "parameters_to_optimize": [
            "strike_selection.target_delta_call",
            "strike_selection.target_delta_put",
            "entry_criteria.min_iv_rank",
            "exit_criteria.profit_target_pct"
        ],
        
        # Metrics to optimize for
        "optimization_metric": "sharpe_ratio",  # Options: sharpe_ratio, total_return, win_rate
        
        # Weight of recent trades in optimization
        "recent_trade_weight": 0.7
    },
    
    # Logging and notifications
    "logging": {
        # Log level
        "level": "INFO",
        
        # Enable trade notifications
        "trade_notifications": True,
        
        # Log performance metrics
        "log_performance_metrics": True,
        
        # Log trade details
        "log_trade_details": True
    }
} 