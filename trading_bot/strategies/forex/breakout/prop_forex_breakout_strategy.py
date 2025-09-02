#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prop Forex Breakout Strategy

This module implements a forex breakout strategy that adheres to
proprietary trading firm rules and restrictions.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta

from trading_bot.strategies.forex.base.forex_base_strategy import ForexBaseStrategy, ForexSession
from trading_bot.strategies.forex.prop_trading_rules_mixin import PropTradingRulesMixin
from trading_bot.strategies.factory.strategy_registry import register_strategy
from trading_bot.core.event_system import EventBus, Event, EventType
from trading_bot.strategies_new.factory.strategy_template import Signal, SignalType, TimeFrame

logger = logging.getLogger(__name__)

@register_strategy({
    'asset_class': 'forex',
    'strategy_type': 'prop_breakout',
    'compatible_market_regimes': ['ranging', 'volatile'],
    'timeframe': 'intraday',
    'regime_compatibility_scores': {
        'trending': 0.65,        # Moderate compatibility with trending markets
        'ranging': 0.90,         # High compatibility with ranging markets
        'volatile': 0.80,        # Good compatibility with volatile markets
        'low_volatility': 0.40,  # Poor compatibility with low volatility
        'all_weather': 0.70      # Good overall compatibility
    }
})
class PropForexBreakoutStrategy(PropTradingRulesMixin, ForexBaseStrategy):
    """
    Proprietary Forex Breakout Strategy
    
    This strategy identifies and trades breakouts from key levels:
    - Support and resistance levels
    - Price channels
    - Key psychological levels
    - Prior session high/lows
    
    It strictly adheres to prop trading rules:
    - Maximum 1-2% daily loss limit
    - Maximum 5% drawdown limit
    - Conservative 0.5-1% risk per trade
    - Minimum 2:1 reward-risk ratio
    - Avoidance of high-impact news events
    - Focus on major pairs during optimal sessions
    """
    
    # Default parameters specific to breakout trading
    DEFAULT_PARAMS = {
        # Breakout detection parameters
        'lookback_periods': 20,        # Lookback for support/resistance
        'breakout_threshold_pips': 5,  # Minimum pips to confirm breakout
        'consolidation_atr_mult': 0.5, # Max ATR multiplier for consolidation
        
        # Filter parameters
        'min_consolidation_bars': 5,   # Minimum bars in consolidation
        'volume_surge_factor': 1.5,    # Volume increase factor for confirmation
        'false_breakout_filter': True, # Use false breakout filter
        
        # Session parameters
        'london_open_breakout': True,  # Trade London session open breakouts
        'use_session_high_low': True,  # Use prior session high/lows
        'preferred_sessions': [
            ForexSession.LONDON, 
            ForexSession.NEWYORK, 
            ForexSession.OVERLAP_LONDON_NEWYORK
        ],
        
        # Technical parameters
        'atr_period': 14,              # ATR calculation period
        'atr_stop_multiplier': 1.5,    # ATR multiplier for stop loss
        'profit_target_mult': 3.0,     # Profit target multiplier relative to stop
        
        # Confirmation parameters
        'require_momentum_confirmation': True, # Require momentum confirmation
        'require_volume_confirmation': True,   # Require volume confirmation
        
        # Prop-specific parameters (will be merged with PropTradingRulesMixin defaults)
        'risk_per_trade_percent': 0.007,   # 0.7% risk per trade
        'max_daily_loss_percent': 0.015,   # 1.5% max daily loss
        'max_drawdown_percent': 0.05,      # 5% max drawdown
        'scale_out_levels': [0.5, 0.75],   # Take partial profits at 50% and 75% of target
    }
    
    def __init__(self, name: str = "PropForexBreakoutStrategy", 
                parameters: Optional[Dict[str, Any]] = None, 
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize Proprietary Forex Breakout Strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Initialize tracking variables
        self.support_resistance_levels = {}
        self.consolidation_zones = {}
        self.session_high_lows = {}
        self.last_signals = {}
        self.account_info = {'balance': 0.0, 'starting_balance': 0.0}
        self.current_positions = []
        
        # Initialize with base parameters
        super().__init__(name=name, parameters=parameters, metadata=metadata)
