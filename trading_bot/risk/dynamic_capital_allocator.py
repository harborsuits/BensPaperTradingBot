#!/usr/bin/env python
"""
Dynamic Capital Allocator

This module implements an advanced capital allocation system that:
1. Dynamically allocates capital across strategies based on performance
2. Adjusts capital based on market regimes
3. Implements the snowball strategy for capital growth
4. Provides risk-adjusted allocation based on strategy volatility
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import json

from trading_bot.core.event_bus import EventBus, Event, get_global_event_bus
from trading_bot.core.constants import EventType

logger = logging.getLogger(__name__)


class DynamicCapitalAllocator:
    """
    Advanced capital allocation system that dynamically allocates
    capital across strategies based on performance, market regimes,
    and risk profiles.
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the dynamic capital allocator.
        
        Args:
            event_bus: Event bus for events
            config: Configuration dictionary
        """
        self.event_bus = event_bus or get_global_event_bus()
        self.config = config or {}
        
        # Account settings
        self.initial_capital = self.config.get("initial_capital", 0.0)
        self.current_capital = self.initial_capital
        self.target_capital = self.config.get("target_capital", self.initial_capital * 3.0)
        
        # Allocation settings
        self.default_allocation = self.config.get("default_allocation", 0.2)  # 20% default
        self.min_strategy_allocation = self.config.get("min_strategy_allocation", 0.05)  # 5% min
        self.max_strategy_allocation = self.config.get("max_strategy_allocation", 0.5)  # 50% max
        
        # Snowball settings
        self.use_snowball = self.config.get("use_snowball", True)
        self.snowball_threshold = self.config.get("snowball_threshold", 0.1)  # 10% profit before snowball
        self.snowball_allocation_factor = self.config.get("snowball_allocation_factor", 0.5)  # Reinvest 50% of profits
        
        # Performance settings
        self.performance_lookback = self.config.get("performance_lookback", 90)  # 90 days
        self.winning_bonus = self.config.get("winning_bonus", 0.2)  # 20% bonus allocation for winning strategies
        self.losing_penalty = self.config.get("losing_penalty", 0.5)  # 50% penalty for losing strategies
        
        # Adaptive allocation
        self.reallocation_frequency = self.config.get("reallocation_frequency", 7)  # days
        self.use_adaptive_allocation = self.config.get("use_adaptive_allocation", True)
        self.allocation_smoothing = self.config.get("allocation_smoothing", 0.2)  # 20% max change per reallocation
        
        # Market regime allocations
        self.regime_allocations = self.config.get("regime_allocations", {
            # Strategy type -> regime -> allocation factor
            "trend_following": {
                "trending": 1.5,
                "ranging": 0.7,
                "volatile": 0.6,
                "low_volatility": 1.0
            },
            "mean_reversion": {
                "trending": 0.6,
                "ranging": 1.5,
                "volatile": 0.8,
                "low_volatility": 1.0
            },
            "breakout": {
                "trending": 1.2,
                "ranging": 0.8,
                "volatile": 1.4,
                "low_volatility": 0.9
            },
            "default": {
                "trending": 1.0,
                "ranging": 1.0,
                "volatile": 0.8,
                "low_volatility": 1.0
            }
        })
        
        # Strategy and allocation tracking
        self.strategy_allocations = {}
        self.strategy_performance = {}
        self.symbol_allocations = {}
        self.market_regimes = {}
        self.last_reallocation = datetime.now() - timedelta(days=self.reallocation_frequency)
        
        # Volatility and risk tracking
        self.strategy_volatility = {}
        self.risk_adjusted_allocations = {}
        self.capital_utilization = 0.0
        
        # Historical allocation tracking
        self.allocation_history = []
        
        # Subscribe to relevant events
        self._subscribe_to_events()
        
        logger.info("Dynamic Capital Allocator initialized")
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events for capital allocation"""
        self.event_bus.subscribe(EventType.ACCOUNT_EQUITY_UPDATED, self._handle_equity_update)
        self.event_bus.subscribe(EventType.MARKET_REGIME_CHANGED, self._handle_regime_change)
        self.event_bus.subscribe(EventType.STRATEGY_PERFORMANCE_UPDATED, self._handle_strategy_performance)
        self.event_bus.subscribe(EventType.TRADE_CLOSED, self._handle_trade_closed)
        self.event_bus.subscribe(EventType.STRATEGY_ADDED, self._handle_strategy_added)
        self.event_bus.subscribe(EventType.STRATEGY_REMOVED, self._handle_strategy_removed)
        
    def _handle_equity_update(self, event: Event):
        """
        Handle account equity update events
        
        Args:
            event: Account equity update event
        """
        new_equity = event.data.get('equity', 0.0)
        if new_equity > 0:
            old_equity = self.current_capital
            self.current_capital = new_equity
            
            # Check if reallocation is needed
            if (datetime.now() - self.last_reallocation).days >= self.reallocation_frequency:
                self._reallocate_capital()
            
            # Handle snowball strategy
            if self.use_snowball and new_equity > old_equity:
                profit = new_equity - self.initial_capital
                if profit > 0 and (profit / self.initial_capital) >= self.snowball_threshold:
                    self._apply_snowball_strategy(profit)
    
    def _handle_regime_change(self, event: Event):
        """
        Handle market regime change events
        
        Args:
            event: Market regime change event
        """
        symbol = event.data.get('symbol')
        regime = event.data.get('current_regime')
        
        if symbol and regime:
            # Store regime for this symbol
            self.market_regimes[symbol] = regime
            
            # Update allocations based on new regime
            self._update_allocations_for_regime(symbol, regime)
    
    def _handle_strategy_performance(self, event: Event):
        """
        Handle strategy performance update events
        
        Args:
            event: Strategy performance update event
        """
        strategy_id = event.data.get('strategy_id')
        performance_metrics = event.data.get('metrics', {})
        timestamp = event.data.get('timestamp', datetime.now())
        
        if strategy_id and performance_metrics:
            # Store performance metrics
            if strategy_id not in self.strategy_performance:
                self.strategy_performance[strategy_id] = []
            
            # Add performance record with timestamp
            self.strategy_performance[strategy_id].append({
                'metrics': performance_metrics,
                'timestamp': timestamp
            })
            
            # Prune old performance records
            cutoff_date = datetime.now() - timedelta(days=self.performance_lookback)
            self.strategy_performance[strategy_id] = [
                record for record in self.strategy_performance[strategy_id]
                if record['timestamp'] > cutoff_date
            ]
            
            # Update strategy volatility
            if 'volatility' in performance_metrics:
                self.strategy_volatility[strategy_id] = performance_metrics['volatility']
            
            # Consider reallocation if adaptive allocation is enabled
            if self.use_adaptive_allocation:
                self._update_strategy_allocation(strategy_id, performance_metrics)
    
    def _handle_trade_closed(self, event: Event):
        """
        Handle trade closed events
        
        Args:
            event: Trade closed event
        """
        strategy_id = event.data.get('strategy')
        symbol = event.data.get('symbol')
        pnl = event.data.get('pnl', 0)
        
        if strategy_id and symbol:
            # Update strategy performance based on trade result
            if strategy_id not in self.strategy_performance:
                self.strategy_performance[strategy_id] = []
            
            # Get previous trades for this strategy
            strategy_trades = [record for record in self.strategy_performance[strategy_id] 
                              if 'metrics' in record and 'trades' in record['metrics']]
            
            if strategy_trades:
                last_record = strategy_trades[-1]
                trades = last_record['metrics'].get('trades', [])
                trades.append({
                    'symbol': symbol,
                    'pnl': pnl,
                    'timestamp': datetime.now()
                })
                
                # Update the record
                last_record['metrics']['trades'] = trades
            else:
                # Create new performance record for this strategy
                self.strategy_performance[strategy_id].append({
                    'metrics': {
                        'trades': [{
                            'symbol': symbol,
                            'pnl': pnl,
                            'timestamp': datetime.now()
                        }]
                    },
                    'timestamp': datetime.now()
                })
    
    def _handle_strategy_added(self, event: Event):
        """
        Handle strategy added events
        
        Args:
            event: Strategy added event
        """
        strategy_id = event.data.get('strategy_id')
        strategy_type = event.data.get('strategy_type', 'default')
        
        if strategy_id:
            # Initialize allocation for new strategy
            self.strategy_allocations[strategy_id] = self.default_allocation
            
            # Store strategy type for regime-based allocation
            if 'strategy_types' not in dir(self):
                self.strategy_types = {}
            self.strategy_types[strategy_id] = strategy_type
            
            # Consider reallocation
            self._reallocate_capital()
    
    def _handle_strategy_removed(self, event: Event):
        """
        Handle strategy removed events
        
        Args:
            event: Strategy removed event
        """
        strategy_id = event.data.get('strategy_id')
        
        if strategy_id and strategy_id in self.strategy_allocations:
            # Remove strategy from allocations
            del self.strategy_allocations[strategy_id]
            
            # Clean up other strategy-related data
            if strategy_id in self.strategy_performance:
                del self.strategy_performance[strategy_id]
            
            if strategy_id in self.strategy_volatility:
                del self.strategy_volatility[strategy_id]
            
            if hasattr(self, 'strategy_types') and strategy_id in self.strategy_types:
                del self.strategy_types[strategy_id]
            
            # Consider reallocation
            self._reallocate_capital()
