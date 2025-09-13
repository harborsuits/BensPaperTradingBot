#!/usr/bin/env python
"""
Enhanced Risk Manager

This module integrates the three major risk management components:
1. Adaptive Risk Manager - Core risk parameters that adjust based on account growth
2. Enhanced Position Sizing - Smart position sizing with Kelly and volatility adjustments
3. Volatility-Adjusted Stops - Dynamic stop loss management based on market conditions
4. Dynamic Capital Allocator - Intelligent capital allocation across strategies

This comprehensive risk framework provides institutional-grade protection while
maximizing capital efficiency across different market conditions.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta

from trading_bot.core.event_bus import EventBus, Event, get_global_event_bus
from trading_bot.core.constants import EventType
from trading_bot.risk.enhanced_position_sizing import EnhancedPositionSizer
from trading_bot.risk.volatility_adjusted_stops import VolatilityAdjustedStops
from trading_bot.risk.dynamic_capital_allocator import DynamicCapitalAllocator

logger = logging.getLogger(__name__)


class EnhancedRiskManager:
    """
    Comprehensive risk management system that integrates multiple risk components
    to provide sophisticated protection while maximizing performance.
    
    This system combines:
    1. Progressive risk management with snowball strategy
    2. Volatility-adjusted position sizing with Kelly criterion
    3. Dynamic stop loss management based on market regimes
    4. Intelligent capital allocation across strategies
    """
    
    def __init__(
        self,
        initial_equity: float,
        target_equity: Optional[float] = None,
        event_bus: Optional[EventBus] = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the enhanced risk manager with all components.
        
        Args:
            initial_equity: Starting account equity
            target_equity: Target account equity where full conservation kicks in
            event_bus: Optional event bus for communication
            config: Configuration parameters for all risk components
        """
        self.event_bus = event_bus or get_global_event_bus()
        self.config = config or {}
        
        # Create individual risk components
        self.adaptive_risk = self._create_adaptive_risk_manager(initial_equity, target_equity)
        self.position_sizer = self._create_position_sizer()
        self.stop_manager = self._create_stop_manager()
        self.capital_allocator = self._create_capital_allocator(initial_equity, target_equity)
        
        # Set up internal state
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.target_equity = target_equity or (initial_equity * 3.0)
        self.strategy_metrics = {}
        self.symbol_metrics = {}
        self.market_regimes = {}
        
        # Set up event subscriptions
        self._subscribe_to_events()
        
        logger.info("Enhanced Risk Manager initialized")
    
    def _create_adaptive_risk_manager(self, initial_equity: float, target_equity: Optional[float]) -> Any:
        """
        Create the adaptive risk manager component.
        
        Args:
            initial_equity: Starting account equity
            target_equity: Target account equity
            
        Returns:
            Configured adaptive risk manager
        """
        from trading_bot.risk.adaptive_risk_manager import AdaptiveRiskManager
        
        adaptive_config = self.config.get('adaptive_risk', {})
        return AdaptiveRiskManager(
            initial_equity=initial_equity,
            target_equity=target_equity,
            config=adaptive_config
        )
    
    def _create_position_sizer(self) -> EnhancedPositionSizer:
        """
        Create the enhanced position sizer component.
        
        Returns:
            Configured position sizer
        """
        position_sizing_config = self.config.get('position_sizing', {})
        return EnhancedPositionSizer(
            event_bus=self.event_bus,
            config=position_sizing_config
        )
    
    def _create_stop_manager(self) -> VolatilityAdjustedStops:
        """
        Create the volatility-adjusted stops component.
        
        Returns:
            Configured stop manager
        """
        stops_config = self.config.get('stops', {})
        return VolatilityAdjustedStops(
            event_bus=self.event_bus,
            config=stops_config
        )
    
    def _create_capital_allocator(self, initial_equity: float, target_equity: Optional[float]) -> DynamicCapitalAllocator:
        """
        Create the dynamic capital allocator component.
        
        Args:
            initial_equity: Starting account equity
            target_equity: Target account equity
            
        Returns:
            Configured capital allocator
        """
        allocation_config = self.config.get('allocation', {})
        allocation_config['initial_capital'] = initial_equity
        allocation_config['target_capital'] = target_equity or (initial_equity * 3.0)
        
        return DynamicCapitalAllocator(
            event_bus=self.event_bus,
            config=allocation_config
        )
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events for risk management"""
        self.event_bus.subscribe(EventType.ACCOUNT_EQUITY_UPDATED, self._handle_equity_update)
        self.event_bus.subscribe(EventType.MARKET_REGIME_CHANGED, self._handle_regime_change)
        self.event_bus.subscribe(EventType.STRATEGY_PERFORMANCE_UPDATED, self._handle_strategy_performance)
        
    def _handle_equity_update(self, event: Event):
        """
        Handle account equity update events
        
        Args:
            event: Account equity update event
        """
        new_equity = event.data.get('equity', 0.0)
        if new_equity > 0:
            self.current_equity = new_equity
            
            # Update all components with new equity
            self.adaptive_risk.update_equity(new_equity)
    
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
    
    def _handle_strategy_performance(self, event: Event):
        """
        Handle strategy performance update events
        
        Args:
            event: Strategy performance update event
        """
        strategy_id = event.data.get('strategy_id')
        performance_metrics = event.data.get('metrics', {})
        
        if strategy_id and performance_metrics:
            # Store metrics
            self.strategy_metrics[strategy_id] = performance_metrics
    
    def calculate_position_size(
        self,
        strategy_id: str,
        symbol: str,
        entry_price: float,
        stop_price: Optional[float] = None,
        account_value: Optional[float] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size using enhanced position sizing.
        
        Args:
            strategy_id: Strategy identifier
            symbol: Trading symbol
            entry_price: Entry price
            stop_price: Optional stop price
            account_value: Optional account value (uses current_equity if None)
            market_data: Optional market data dictionary
            
        Returns:
            Dictionary with position sizing information
        """
        # Use current equity if account value not provided
        account_value = account_value or self.current_equity
        
        # Get strategy-specific metrics
        strategy_metrics = self.strategy_metrics.get(strategy_id, {})
        
        # Get symbol regime if available
        regime = self.market_regimes.get(symbol)
        
        # Get base risk percentage from adaptive risk manager
        base_risk = self.adaptive_risk.get_risk_per_trade(
            strategy_id=strategy_id,
            symbol=symbol,
            performance_metrics=strategy_metrics
        )
        
        # Configure position sizer with base risk
        if 'default_risk_per_trade' not in self.position_sizer.config:
            self.position_sizer.config['default_risk_per_trade'] = base_risk
        else:
            self.position_sizer.config['default_risk_per_trade'] = base_risk
        
        # Get position size from enhanced position sizer
        return self.position_sizer.calculate_position_size(
            account_value=account_value,
            symbol=symbol,
            strategy=strategy_id,
            entry_price=entry_price,
            stop_price=stop_price,
            market_data=market_data,
            regime=regime,
            strategy_metrics=strategy_metrics
        )
    
    def calculate_stop_price(
        self,
        trade_id: str,
        strategy_id: str,
        symbol: str,
        entry_price: float,
        direction: str,
        market_data: Optional[Dict[str, Any]] = None,
        stop_type: str = "atr"
    ) -> Dict[str, Any]:
        """
        Calculate optimal stop price using volatility-adjusted stops.
        
        Args:
            trade_id: Unique trade identifier
            strategy_id: Strategy identifier
            symbol: Trading symbol
            entry_price: Entry price
            direction: Trade direction ('long' or 'short')
            market_data: Optional market data dictionary
            stop_type: Stop type ('fixed', 'atr', 'percent', 'chandelier')
            
        Returns:
            Dictionary with stop information
        """
        # Add regime information to market data if available
        if market_data is None:
            market_data = {}
            
        if symbol in self.market_regimes:
            market_data['regime'] = self.market_regimes[symbol]
        
        # Register stop with the stop manager
        return self.stop_manager.register_stop(
            trade_id=trade_id,
            symbol=symbol,
            strategy=strategy_id,
            entry_price=entry_price,
            direction=direction,
            market_data=market_data,
            stop_type=stop_type
        )
    
    def get_strategy_allocation(self, strategy_id: str) -> float:
        """
        Get current allocation percentage for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Allocation percentage (0.0-1.0)
        """
        return self.capital_allocator.get_strategy_allocation(strategy_id)
    
    def get_all_strategy_allocations(self) -> Dict[str, float]:
        """
        Get all current strategy allocations.
        
        Returns:
            Dictionary of strategy allocations
        """
        return self.capital_allocator.get_all_allocations()
    
    def get_current_risk_parameters(self) -> Dict[str, Any]:
        """
        Get current risk parameters from adaptive risk manager.
        
        Returns:
            Dictionary of current risk parameters
        """
        return self.adaptive_risk.get_current_parameters()
    
    def check_stop_triggered(self, trade_id: str, current_price: float) -> bool:
        """
        Check if a stop has been triggered.
        
        Args:
            trade_id: Trade identifier
            current_price: Current price
            
        Returns:
            True if stop triggered, False otherwise
        """
        return self.stop_manager.check_stop_triggered(trade_id, current_price)
    
    def update_stop(self, trade_id: str, new_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update an existing stop with new parameters.
        
        Args:
            trade_id: Trade identifier
            new_params: New parameters to update
            
        Returns:
            Updated stop info or None if not found
        """
        return self.stop_manager.update_stop(trade_id, new_params)
    
    def get_stop_performance(self) -> Dict[str, Any]:
        """
        Get stop performance metrics.
        
        Returns:
            Dictionary of stop performance metrics
        """
        return self.stop_manager.get_stop_performance()


# Factory function to create an enhanced risk manager
def create_enhanced_risk_manager(
    initial_equity: float,
    target_equity: Optional[float] = None,
    event_bus: Optional[EventBus] = None,
    config: Dict[str, Any] = None
) -> EnhancedRiskManager:
    """
    Factory function to create and configure an enhanced risk manager.
    
    Args:
        initial_equity: Starting account equity
        target_equity: Target account equity
        event_bus: Optional event bus for communication
        config: Configuration parameters
        
    Returns:
        Configured enhanced risk manager
    """
    return EnhancedRiskManager(
        initial_equity=initial_equity,
        target_equity=target_equity,
        event_bus=event_bus,
        config=config
    )
