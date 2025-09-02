#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Iron Condor Strategy Implementation

This module implements an enhanced iron condor options strategy with modular
architecture, advanced risk management, and comprehensive event handling.

An iron condor is created by:
1. Selling an out-of-the-money put (short put)
2. Buying a further out-of-the-money put (long put)
3. Selling an out-of-the-money call (short call)
4. Buying a further out-of-the-money call (long call)

All options have the same expiration date.
"""

import logging
import numpy as np
import pandas as pd
from enum import Enum
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

from trading_bot.strategies.base.options_base_strategy import OptionsBaseStrategy
from trading_bot.strategies.factory.strategy_registry import register_strategy, StrategyType, AssetClass, MarketRegime, TimeFrame
from trading_bot.market.market_data import MarketData
from trading_bot.market.universe import Universe
from trading_bot.market.option_chains import OptionChains

# Import modular components
from trading_bot.strategies.options.complex_spreads.components.market_analysis import ComplexSpreadMarketAnalyzer
from trading_bot.strategies.options.complex_spreads.components.option_selection import ComplexSpreadOptionSelector
from trading_bot.strategies.options.complex_spreads.components.risk_management import ComplexSpreadRiskManager
from trading_bot.strategies.options.complex_spreads.components.complex_spread_factory import ComplexSpreadFactory, SpreadType

logger = logging.getLogger(__name__)

class StrategyVariant(Enum):
    """Iron Condor strategy variants."""
    STANDARD = "standard"        # Standard iron condor with equidistant wings
    WIDE = "wide"                # Iron condor with wider wings
    NARROW = "narrow"            # Iron condor with narrower wings
    EARNINGS = "earnings"        # Iron condor optimized for earnings
    CONSERVATIVE = "conservative" # Iron condor with more conservative deltas
    AGGRESSIVE = "aggressive"    # Iron condor with more aggressive deltas
    SKEWED_CALL = "skewed_call"  # Iron condor with wider call spread
    SKEWED_PUT = "skewed_put"    # Iron condor with wider put spread

@register_strategy({
    'asset_class': AssetClass.OPTIONS.value,
    'strategy_type': StrategyType.INCOME.value,
    'timeframe': TimeFrame.SWING.value,
    'compatible_market_regimes': [MarketRegime.RANGING.value, MarketRegime.LOW_VOLATILITY.value],
    'description': 'Enhanced Iron Condor strategy with modular architecture',
    'risk_level': 'medium',
    'typical_holding_period': '30-45 days'
})
class EnhancedIronCondorStrategy:
    """
    Enhanced Iron Condor Options Strategy with modular architecture.
    
    This strategy involves selling an out-of-the-money put spread and an out-of-the-money call spread
    with the same expiration date. It's a market-neutral strategy that profits from low volatility
    and time decay when the underlying stays within a certain price range.
    
    Key enhancements:
    - Modular architecture with separate components
    - Advanced risk management and adaptive position sizing
    - Comprehensive event-driven model with real-time adjustments
    - Multiple strategy variants for different market conditions
    - Detailed performance tracking and analytics
    """
    
    # Default parameters for iron condor strategy
    DEFAULT_PARAMS = {
        'strategy_name': 'enhanced_iron_condor',
        'strategy_version': '2.0.0',
        'asset_class': 'options',
        'strategy_type': 'income',
        'timeframe': 'swing',
        'market_regime': 'neutral',
        'strategy_variant': 'standard',
        
        # Stock selection criteria
        'min_stock_price': 50.0,              # Minimum stock price to consider
        'max_stock_price': 500.0,             # Maximum stock price to consider
        'min_option_volume': 200,             # Minimum option volume
        'min_option_open_interest': 300,      # Minimum option open interest
        'min_iv_percentile': 40,              # Minimum IV percentile
        'max_iv_percentile': 70,              # Maximum IV percentile for entry
        
        # Technical analysis parameters
        'min_historical_days': 252,           # Days of historical data required
        'atr_period': 14,                     # Period for ATR calculation
        'bollinger_period': 20,               # Period for Bollinger Bands
        'bollinger_std': 2.0,                 # Standard deviations for Bollinger Bands
        
        # Option parameters
        'target_dte': 45,                     # Target days to expiration
        'min_dte': 30,                        # Minimum days to expiration
        'max_dte': 60,                        # Maximum days to expiration
        'call_spread_width': 5,               # Width of call spread (strike difference)
        'put_spread_width': 5,                # Width of put spread (strike difference)
        'short_call_delta': -0.16,            # Target delta for short call (negative)
        'short_put_delta': 0.16,              # Target delta for short put (positive)
        'min_credit': 0.40,                   # Minimum credit required
        'min_credit_to_width_ratio': 0.15,    # Minimum ratio of credit to spread width
        
        # Risk management parameters
        'max_risk_per_trade_pct': 2.0,        # Maximum percentage of account at risk per trade
        'max_position_pct': 15.0,             # Maximum percentage of account in position
        'profit_target_pct': 50.0,            # Take profit at this percentage of max profit
        'stop_loss_pct': 100.0,               # Stop loss at this percentage of max loss
        'max_gamma_risk': 0.05,               # Maximum gamma risk
        'max_vega_exposure': 200,             # Maximum vega exposure
        'max_days_to_hold': 45,               # Maximum days to hold position
        
        # Adjustment parameters
        'adjustment_proximity_threshold': 0.02, # Adjust if price within 2% of short strike
        'gamma_risk_threshold': 0.05,         # Adjust if gamma exceeds this threshold
        
        # Exit timing parameters
        'min_dte_exit': 21,                   # Exit if DTE falls below this threshold
        'avoid_earnings': True,               # Avoid earnings during position
        'exit_days_before_earnings': 5,       # Exit this many days before earnings
        
        # Market regime parameters
        'range_width_threshold': 0.15,        # Maximum width of range for range-bound regime
        'range_time_threshold': 0.7,          # Minimum percentage of time in range
        'min_touch_points': 3,                # Minimum number of touch points for range
        
        # Advanced parameters for variants
        'variant_params': {
            'wide': {
                'call_spread_width': 10,
                'put_spread_width': 10,
                'short_call_delta': -0.10,
                'short_put_delta': 0.10
            },
            'narrow': {
                'call_spread_width': 3,
                'put_spread_width': 3,
                'short_call_delta': -0.20,
                'short_put_delta': 0.20
            },
            'earnings': {
                'call_spread_width': 8,
                'put_spread_width': 8,
                'short_call_delta': -0.12,
                'short_put_delta': 0.12
            },
            'conservative': {
                'short_call_delta': -0.10,
                'short_put_delta': 0.10,
                'profit_target_pct': 35.0
            },
            'aggressive': {
                'short_call_delta': -0.25,
                'short_put_delta': 0.25,
                'profit_target_pct': 65.0
            },
            'skewed_call': {
                'call_spread_width': 8,
                'put_spread_width': 5,
                'short_call_delta': -0.14,
                'short_put_delta': 0.16
            },
            'skewed_put': {
                'call_spread_width': 5,
                'put_spread_width': 8,
                'short_call_delta': -0.16,
                'short_put_delta': 0.14
            },
        },
        
        # Operational parameters
        'cache_volatility_data': True,        # Cache volatility calculations
        'avoid_event_window': True,           # Avoid trading around events
        'max_universe_size': 20,              # Maximum number of symbols in universe
    }
    
    def __init__(self, 
                 strategy_id: str = None, 
                 name: str = None,
                 parameters: Dict[str, Any] = None,
                 event_bus: Any = None):
        """
        Initialize the Enhanced Iron Condor strategy.
        
        Args:
            strategy_id: Unique identifier for this strategy instance
            name: Human-readable name of the strategy
            parameters: Strategy-specific parameters that override the defaults
            event_bus: Event bus for publishing and subscribing to events
        """
        # Base initialization
        self.strategy_id = strategy_id or f"enhanced_iron_condor_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.name = name or "Enhanced Iron Condor Strategy"
        
        # Merge default parameters with provided parameters
        self.params = self.DEFAULT_PARAMS.copy()
        if parameters:
            self.params.update(parameters)
            
        # Initialize variant-specific parameters
        self._initialize_variant_params()
        
        # Setup event bus
        self.event_bus = event_bus
        
        # Initialize universe and positions
        self.universe = []
        self.positions = {}
        self.closed_positions = []
        
        # Setup components
        self.volatility_analyzer = ComplexSpreadMarketAnalyzer(self.params)
        self.option_selector = ComplexSpreadOptionSelector(self.params)
        self.risk_manager = ComplexSpreadRiskManager(self.params)
        self.spread_factory = ComplexSpreadFactory(self.params)
        
        # Setup metrics tracking
        self.volatility_metrics = {}
        self.performance_metrics = {}
        self.health_metrics = {
            'last_run_time': None,
            'last_signal_time': None,
            'last_position_check': None,
            'open_position_count': 0,
            'closed_position_count': 0,
            'errors': []
        }
        
        # Initialization
        logger.info(f"Initialized {self.name} (ID: {self.strategy_id}) with variant {self.params['strategy_variant']}")
        if self.event_bus:
            self._subscribe_to_events()
            
    def _initialize_variant_params(self):
        """Initialize parameters based on the selected strategy variant."""
        variant = self.params.get('strategy_variant', 'standard')
        
        # If variant is not standard, update parameters with variant-specific values
        if variant != 'standard' and variant in self.params.get('variant_params', {}):
            variant_params = self.params['variant_params'][variant]
            self.params.update(variant_params)
            
        logger.info(f"Initialized {variant} variant with delta call={self.params['short_call_delta']}, put={self.params['short_put_delta']}")
