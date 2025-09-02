"""
Risk Integration Module

This module connects and coordinates the various risk management components
including Snowball Allocation, Position Sizing, Emergency Brakes, and
Adaptive Risk Management.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from trading_bot.risk.position_sizer import PositionSizer
from trading_bot.risk.emergency_brake import EmergencyBrake
from trading_bot.risk.adaptive_risk_manager import AdaptiveRiskManager
from trading_bot.ai_scoring.snowball_allocator import SnowballAllocator

logger = logging.getLogger(__name__)

class IntegratedRiskManager:
    """
    Integrated risk management system that coordinates all risk components.
    
    This class provides a unified interface to access the various risk management
    systems while ensuring they work together coherently:
    
    1. Adaptive Risk Manager - Adjusts risk parameters based on account growth
    2. Snowball Allocator - Handles dynamic strategy allocation based on performance
    3. Position Sizer - Calculates position sizes based on risk parameters
    4. Emergency Brake - Monitors for risk events and can pause trading
    
    The integration ensures that risk decisions are consistent across the system
    and that each component has the information it needs from other components.
    """
    
    def __init__(self, initial_equity: float, config: Dict[str, Any] = None):
        """
        Initialize the integrated risk management system.
        
        Args:
            initial_equity: Starting account equity
            config: Configuration dictionary with settings for all components
        """
        self.config = config or {}
        self.equity = initial_equity
        
        # Extract sub-configurations
        adaptive_config = self.config.get('adaptive_risk', {})
        snowball_config = self.config.get('snowball', {})
        position_sizer_config = self.config.get('position_sizer', {})
        emergency_config = self.config.get('emergency_brake', {})
        
        # Initialize components
        target_equity = adaptive_config.get('target_equity', initial_equity * 3.0)
        
        self.adaptive_risk = AdaptiveRiskManager(
            initial_equity=initial_equity,
            target_equity=target_equity,
            config=adaptive_config
        )
        
        self.snowball = SnowballAllocator(
            config=snowball_config
        )
        
        self.position_sizer = PositionSizer(
            portfolio_value=initial_equity,
            default_risk_percent=position_sizer_config.get('default_risk_percent', 0.01),
            config=position_sizer_config
        )
        
        self.emergency_brake = EmergencyBrake(
            config=emergency_config
        )
        
        # Register callbacks for emergency events
        self.emergency_brake.register_callbacks(
            notify_callback=self._handle_emergency_notification,
            strategy_pause_callback=self._handle_strategy_pause,
            global_pause_callback=self._handle_global_pause
        )
        
        # Start monitoring
        self.emergency_brake.start_monitoring()
        
        # Tracking
        self.paused_strategies = set()
        self.global_pause_active = False
        self.last_update_time = datetime.now()
        
        logger.info(f"Initialized IntegratedRiskManager with equity: ${initial_equity:,.2f}")
        
    def update_equity(self, 
                     new_equity: float, 
                     strategy_equities: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Update account equity across all risk components.
        
        Args:
            new_equity: Current total account equity
            strategy_equities: Optional dictionary mapping strategy IDs to their equity values
            
        Returns:
            Dict with updated risk parameters
        """
        # Update main equity
        self.equity = new_equity
        self.position_sizer.update_portfolio_value(new_equity)
        
        # Update adaptive risk manager and get new parameters
        adaptive_params = self.adaptive_risk.update_equity(new_equity)
        
        # Check for global drawdown
        self.emergency_brake.update_portfolio_equity(new_equity)
        
        # Update strategy-specific metrics if provided
        if strategy_equities:
            for strategy_id, strategy_equity in strategy_equities.items():
                self.emergency_brake.update_strategy_equity(strategy_id, strategy_equity)
        
        # Update heartbeat
        self.emergency_brake.record_heartbeat("risk_manager")
        self.last_update_time = datetime.now()
        
        return adaptive_params
        
    def get_strategy_allocations(self, 
                               current_weights: Dict[str, float],
                               profit_data: Dict[str, float],
                               performance_data: Dict[str, Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate strategy allocations using the snowball system and adaptive limits.
        
        Args:
            current_weights: Current strategy allocations
            profit_data: Profit amounts by strategy
            performance_data: Performance metrics by strategy
            
        Returns:
            Dict mapping strategy IDs to allocation percentages
        """
        # Apply snowball allocation calculation
        allocations = self.snowball.update_allocations(
            current_weights=current_weights,
            profit_data=profit_data,
            total_equity=self.equity
        )
        
        # Apply adaptive limits to each strategy
        limited_allocations = {}
        for strategy_id, allocation in allocations.items():
            # Skip paused strategies
            if strategy_id in self.paused_strategies or self.global_pause_active:
                limited_allocations[strategy_id] = 0.0
                continue
                
            # Get performance metrics if available
            metrics = None
            if performance_data and strategy_id in performance_data:
                metrics = performance_data[strategy_id]
                
            # Get max allocation for this strategy
            max_allocation = self.adaptive_risk.get_max_strategy_allocation(
                strategy_id=strategy_id,
                performance_metrics=metrics
            )
            
            # Apply limit
            limited_allocations[strategy_id] = min(allocation, max_allocation)
            
        # Normalize to ensure allocations sum to 1.0
        total_allocation = sum(limited_allocations.values())
        if total_allocation > 0:
            normalized_allocations = {
                strategy_id: allocation / total_allocation
                for strategy_id, allocation in limited_allocations.items()
            }
        else:
            # If all strategies are paused or have zero allocation
            normalized_allocations = limited_allocations
            
        return normalized_allocations
        
    def calculate_position_size(self,
                              strategy_id: str,
                              symbol: str,
                              entry_price: float,
                              stop_loss: Optional[float] = None,
                              market_data: Optional[Any] = None,
                              performance_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate position size with all risk management rules applied.
        
        Args:
            strategy_id: Strategy identifier
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Optional stop loss price
            market_data: Optional market data for volatility calculation
            performance_metrics: Optional performance metrics
            
        Returns:
            Position size information dictionary
        """
        # Check emergency brake status first
        if self.global_pause_active or strategy_id in self.paused_strategies:
            logger.warning(f"Position sizing requested for paused strategy: {strategy_id}")
            return {
                "symbol": symbol,
                "shares": 0,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "position_value": 0,
                "dollar_risk": 0,
                "risk_percent": 0,
                "portfolio_percent": 0,
                "reason": "Strategy paused"
            }
            
        # Get risk percentage for this trade from adaptive manager
        risk_percent = self.adaptive_risk.get_risk_per_trade(
            strategy_id=strategy_id,
            symbol=symbol,
            performance_metrics=performance_metrics
        )
        
        # Calculate position size
        position = self.position_sizer.calculate_position_size(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            risk_percent=risk_percent,
            market_data=market_data,
            strategy_performance=performance_metrics
        )
        
        # Record heartbeat
        self.emergency_brake.record_heartbeat("position_sizer")
        
        return position
        
    def record_trade_result(self,
                          strategy_id: str,
                          symbol: str,
                          profit: float,
                          slippage_pct: Optional[float] = None) -> bool:
        """
        Record a completed trade result for risk monitoring.
        
        Args:
            strategy_id: Strategy identifier
            symbol: Trading symbol
            profit: Profit/loss amount (positive or negative)
            slippage_pct: Optional slippage percentage
            
        Returns:
            bool: True if trade is within risk parameters, False if it triggered an emergency stop
        """
        # Update emergency brake
        result = self.emergency_brake.record_trade_result(
            strategy_id=strategy_id,
            trade_profit=profit,
            slippage_pct=slippage_pct
        )
        
        # If emergency brake triggered, update paused strategies
        if not result and strategy_id not in self.paused_strategies:
            self.paused_strategies.add(strategy_id)
            logger.warning(f"Strategy {strategy_id} paused due to emergency brake trigger")
            
        return result
        
    def get_max_position_allocation(self,
                                  strategy_id: str,
                                  symbol: str,
                                  performance_metrics: Optional[Dict[str, float]] = None) -> float:
        """
        Get maximum allocation percentage for a single position.
        
        Args:
            strategy_id: Strategy identifier
            symbol: Trading symbol
            performance_metrics: Optional performance metrics
            
        Returns:
            Maximum allocation percentage (0.0-1.0)
        """
        return self.adaptive_risk.get_max_position_allocation(
            strategy_id=strategy_id,
            symbol=symbol,
            performance_metrics=performance_metrics
        )
        
    def set_manual_allocation(self,
                            strategy_id: str,
                            weight: float,
                            duration_secs: int = 86400) -> None:
        """
        Set a manual allocation override for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            weight: Weight to assign
            duration_secs: How long the override should last
        """
        self.snowball.set_manual_allocation(
            strategy=strategy_id,
            weight=weight,
            duration_secs=duration_secs
        )
        
    def clear_manual_allocation(self, strategy_id: Optional[str] = None) -> None:
        """
        Clear manual allocation for a strategy or all strategies.
        
        Args:
            strategy_id: Strategy to clear, or None to clear all
        """
        self.snowball.clear_manual_allocation(strategy=strategy_id)
        
    def reset_strategy_pause(self, strategy_id: str) -> bool:
        """
        Reset emergency pause for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            bool: True if successfully reset
        """
        if strategy_id in self.paused_strategies:
            self.paused_strategies.remove(strategy_id)
            
        self.emergency_brake.reset_strategy_counters(strategy_id)
        return True
        
    def activate_kill_switch(self, reason: str = "Manual activation") -> bool:
        """
        Activate global kill switch to pause all trading.
        
        Args:
            reason: Reason for activation
            
        Returns:
            bool: True if successfully activated
        """
        self.global_pause_active = True
        return self.emergency_brake.activate_kill_switch(reason)
        
    def deactivate_kill_switch(self, reason: str = "Manual deactivation") -> bool:
        """
        Deactivate global kill switch to resume trading.
        
        Args:
            reason: Reason for deactivation
            
        Returns:
            bool: True if successfully deactivated
        """
        self.global_pause_active = False
        return self.emergency_brake.deactivate_kill_switch(reason)
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of all risk management components.
        
        Returns:
            Dict with current status of all risk components
        """
        return {
            "adaptive_risk": self.adaptive_risk.get_current_parameters(),
            "emergency_brake": self.emergency_brake.get_status(),
            "equity": self.equity,
            "paused_strategies": list(self.paused_strategies),
            "global_pause_active": self.global_pause_active,
            "last_update_time": self.last_update_time.isoformat()
        }
        
    def _handle_emergency_notification(self, severity: str, message: str, data: Dict[str, Any]) -> None:
        """Callback handler for emergency notifications."""
        logger.log(
            logging.CRITICAL if severity == "critical" else
            logging.ERROR if severity == "error" else
            logging.WARNING if severity == "warning" else
            logging.INFO,
            f"RISK ALERT ({severity}): {message}"
        )
        
    def _handle_strategy_pause(self, strategy_id: str, reason: str, data: Dict[str, Any]) -> bool:
        """Callback handler for strategy pause events."""
        if strategy_id not in self.paused_strategies:
            self.paused_strategies.add(strategy_id)
            logger.warning(f"Strategy {strategy_id} paused: {reason}")
        return True
        
    def _handle_global_pause(self, reason: str, data: Dict[str, Any]) -> bool:
        """Callback handler for global pause events."""
        self.global_pause_active = True
        logger.critical(f"GLOBAL TRADING PAUSE: {reason}")
        return True
