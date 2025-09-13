#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk Settings Module

This module defines the RiskSettings class used to configure 
the risk management system.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

@dataclass
class RiskSettings:
    """
    Risk settings for the trading system.
    
    This class contains all parameters related to risk management,
    including position sizing, drawdown limits, and other risk controls.
    """
    
    # Portfolio-level risk limits
    max_portfolio_risk: float = 0.02  # Maximum portfolio risk (2%)
    max_drawdown_percent: float = 0.10  # Maximum allowed drawdown (10%)
    max_position_size_percent: float = 0.05  # Maximum position size (5% of portfolio)
    max_sector_allocation: float = 0.30  # Maximum sector allocation (30%)
    max_strategy_allocation: float = 0.25  # Maximum allocation to single strategy (25%)
    
    # Position-level risk settings
    default_stop_loss_percent: float = 0.05  # Default stop loss (5%)
    default_take_profit_percent: float = 0.10  # Default take profit (10%)
    max_open_positions: int = 20  # Maximum number of open positions
    min_risk_reward_ratio: float = 1.5  # Minimum risk/reward ratio
    
    # Volatility-based settings
    vix_threshold_high: float = 30.0  # High volatility threshold
    vix_threshold_low: float = 15.0  # Low volatility threshold
    position_size_volatility_scalar: float = 0.5  # Reduce position size by this factor in high volatility
    
    # Asset-specific settings
    asset_specific_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Risk metrics
    risk_metrics_window: int = 21  # Window for calculating risk metrics (trading days)
    
    # Circuit breaker settings
    daily_loss_limit_percent: float = 0.03  # Daily loss limit (3%)
    weekly_loss_limit_percent: float = 0.05  # Weekly loss limit (5%)
    monthly_loss_limit_percent: float = 0.08  # Monthly loss limit (8%)
    
    # Monitoring settings
    monitor_interval_seconds: int = 60  # Interval for risk monitoring (seconds)
    
    # Recovery settings
    auto_reduce_exposure_on_limit_hit: bool = True  # Automatically reduce exposure when limits hit
    auto_reduce_percent: float = 0.50  # Percent to reduce exposure by if limits hit
    
    def __post_init__(self):
        """Initialize any derived settings after the dataclass is created."""
        # Create default asset specific settings if none provided
        if not self.asset_specific_settings:
            self.asset_specific_settings = {
                "options": {
                    "max_position_size_percent": 0.02,  # Smaller position sizes for options
                    "default_stop_loss_percent": 0.30,  # Wider stops for options
                },
                "crypto": {
                    "max_position_size_percent": 0.03,  # Smaller positions for crypto
                    "default_stop_loss_percent": 0.15,  # Wider stops for crypto
                }
            }
