"""
Adaptive Risk Manager

This module implements a progressive risk management system that
automatically becomes more conservative as account equity grows.
"""

import logging
import math
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class AdaptiveRiskManager:
    """
    Risk management system that adjusts risk parameters based on account growth.
    
    The AdaptiveRiskManager implements a progressive risk management approach where:
    1. Initial risk levels can be aggressive to accelerate account growth
    2. As profitability increases, risk parameters automatically become more conservative
    3. Custom thresholds determine when to reduce risk exposure
    4. Exceptional performance can temporarily override conservative limits
    
    This approach balances the need for faster growth with smaller accounts
    while protecting profits as the account size increases.
    """
    
    def __init__(self, 
                initial_equity: float,
                target_equity: Optional[float] = None,
                config: Dict[str, Any] = None):
        """
        Initialize the AdaptiveRiskManager with equity thresholds and risk parameters.
        
        Args:
            initial_equity: Starting account equity
            target_equity: Target account equity where full conservation kicks in
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Equity thresholds
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.target_equity = target_equity or (initial_equity * 3.0)  # Default 3x growth
        
        # Initial aggressive risk parameters
        self.initial_max_strategy_allocation = self.config.get('initial_max_strategy_allocation', 0.90)  # 90%
        self.initial_max_position_allocation = self.config.get('initial_max_position_allocation', 0.50)  # 50%
        self.initial_base_risk_per_trade = self.config.get('initial_base_risk_per_trade', 0.02)  # 2%
        
        # Conservative risk parameters after reaching target equity
        self.conservative_max_strategy_allocation = self.config.get('conservative_max_strategy_allocation', 0.30)  # 30%
        self.conservative_max_position_allocation = self.config.get('conservative_max_position_allocation', 0.15)  # 15%
        self.conservative_base_risk_per_trade = self.config.get('conservative_base_risk_per_trade', 0.005)  # 0.5%
        
        # Performance thresholds for overriding conservative limits
        self.high_performance_threshold = self.config.get('high_performance_threshold', {
            'sharpe_ratio': 3.0,
            'win_rate': 0.70,
            'profit_factor': 2.5
        })
        
        # Current parameters (will be adjusted based on equity)
        self.current_max_strategy_allocation = self.initial_max_strategy_allocation
        self.current_max_position_allocation = self.initial_max_position_allocation
        self.current_base_risk_per_trade = self.initial_base_risk_per_trade
        
        # Metrics
        self.equity_history = [initial_equity]
        self.high_water_mark = initial_equity
        self.drawdown_threshold = self.config.get('drawdown_threshold', 0.15)  # 15% drawdown triggers conservation
        
        logger.info(f"Initialized AdaptiveRiskManager with initial_equity: ${initial_equity:,.2f}, "
                    f"target_equity: ${self.target_equity:,.2f}")
        logger.info(f"Initial max strategy allocation: {self.initial_max_strategy_allocation:.1%}, "
                    f"Conservative max strategy allocation: {self.conservative_max_strategy_allocation:.1%}")
    
    def update_equity(self, new_equity: float) -> Dict[str, Any]:
        """
        Update current equity and recalculate risk parameters.
        
        Args:
            new_equity: Current account equity
            
        Returns:
            Dictionary of updated risk parameters
        """
        # Store previous values for change detection
        prev_max_strategy = self.current_max_strategy_allocation
        prev_max_position = self.current_max_position_allocation
        prev_risk_per_trade = self.current_base_risk_per_trade
        
        # Update equity tracking
        self.current_equity = new_equity
        self.equity_history.append(new_equity)
        
        # Update high water mark if new equity is higher
        if new_equity > self.high_water_mark:
            self.high_water_mark = new_equity
        
        # Calculate drawdown from high water mark
        drawdown = 0 if self.high_water_mark <= 0 else (self.high_water_mark - new_equity) / self.high_water_mark
        
        # Calculate progress toward target equity (from 0.0 to 1.0+)
        # This is what drives the progressive risk reduction
        equity_growth_ratio = self._calculate_equity_progress(new_equity)
        
        # Apply extra conservation if in drawdown
        conservation_factor = 1.0
        if drawdown > self.drawdown_threshold / 2:
            # Scale conservation exponentially with drawdown
            conservation_factor = 1.0 - min(0.8, (drawdown / self.drawdown_threshold))
            logger.info(f"Adding extra conservation due to drawdown of {drawdown:.1%}, "
                        f"factor: {conservation_factor:.2f}")
        
        # Recalculate risk parameters based on equity ratio and drawdown
        self._recalculate_risk_parameters(equity_growth_ratio, conservation_factor)
        
        # Log changes if significant
        if (abs(prev_max_strategy - self.current_max_strategy_allocation) > 0.02 or
            abs(prev_max_position - self.current_max_position_allocation) > 0.02 or
            abs(prev_risk_per_trade - self.current_base_risk_per_trade) > 0.001):
            logger.info(f"Risk parameters updated - Equity: ${new_equity:,.2f}, "
                        f"Progress: {equity_growth_ratio:.1%}, Drawdown: {drawdown:.1%}")
            logger.info(f"Max strategy allocation: {self.current_max_strategy_allocation:.1%}, "
                        f"Max position allocation: {self.current_max_position_allocation:.1%}, "
                        f"Base risk per trade: {self.current_base_risk_per_trade:.2%}")
        
        # Return current parameters
        return self.get_current_parameters()
    
    def get_max_strategy_allocation(self, 
                                  strategy_id: str = None, 
                                  performance_metrics: Dict[str, float] = None) -> float:
        """
        Get maximum allocation percentage for a strategy, considering performance.
        
        Args:
            strategy_id: Optional strategy identifier
            performance_metrics: Optional performance metrics for the strategy
            
        Returns:
            Maximum allocation percentage (0.0-1.0) for the strategy
        """
        # Start with current general limit
        max_allocation = self.current_max_strategy_allocation
        
        # Apply performance-based adjustments if metrics are provided
        if performance_metrics:
            # Check if this is a high-performing strategy that gets special treatment
            if self._is_high_performance_strategy(performance_metrics):
                # Allow up to initial allocation for exceptional strategies
                # but never exceed 90% regardless of performance
                max_allocation = min(0.90, self.initial_max_strategy_allocation)
                logger.debug(f"High-performance strategy detected - increasing max allocation to {max_allocation:.1%}")
        
        return max_allocation
    
    def get_max_position_allocation(self,
                                 strategy_id: str = None,
                                 symbol: str = None,
                                 performance_metrics: Dict[str, float] = None) -> float:
        """
        Get maximum allocation percentage for a single position.
        
        Args:
            strategy_id: Optional strategy identifier
            symbol: Optional symbol identifier
            performance_metrics: Optional performance metrics
            
        Returns:
            Maximum allocation percentage (0.0-1.0) for the position
        """
        # Start with current general limit
        max_allocation = self.current_max_position_allocation
        
        # Apply performance-based adjustments if metrics are provided
        if performance_metrics:
            # Check if this is a high-performing position that gets special treatment
            if self._is_high_performance_strategy(performance_metrics):
                # Allow up to initial position allocation for exceptional performers
                # but never exceed 50% regardless of performance
                max_allocation = min(0.50, self.initial_max_position_allocation)
                logger.debug(f"High-performance position detected - increasing max allocation to {max_allocation:.1%}")
        
        return max_allocation
    
    def get_risk_per_trade(self,
                          strategy_id: str = None,
                          symbol: str = None,
                          performance_metrics: Dict[str, float] = None) -> float:
        """
        Get risk percentage per trade based on current equity and performance.
        
        Args:
            strategy_id: Optional strategy identifier
            symbol: Optional symbol identifier
            performance_metrics: Optional performance metrics
            
        Returns:
            Risk percentage per trade (0.0-1.0)
        """
        # Start with current general limit
        risk_pct = self.current_base_risk_per_trade
        
        # Apply performance-based adjustments if metrics are provided
        if performance_metrics:
            # Gradually scale risk with performance metrics
            sharpe = performance_metrics.get('sharpe_ratio', 1.0)
            win_rate = performance_metrics.get('win_rate', 0.5)
            
            # Combine metrics for a performance score
            perf_score = (0.6 * min(sharpe / 3.0, 1.0)) + (0.4 * win_rate)
            
            # Scale risk between current and 150% of current based on performance
            risk_adjustment = 1.0 + (perf_score * 0.5)  # 1.0 to 1.5 scaling
            risk_pct = risk_pct * risk_adjustment
            
            logger.debug(f"Performance-based risk adjustment: {risk_adjustment:.2f}, "
                        f"risk_pct: {risk_pct:.2%}")
        
        return risk_pct
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Get current risk parameters.
        
        Returns:
            Dictionary of current risk parameters
        """
        return {
            "equity": self.current_equity,
            "initial_equity": self.initial_equity,
            "target_equity": self.target_equity,
            "high_water_mark": self.high_water_mark,
            "drawdown": 0 if self.high_water_mark <= 0 else (self.high_water_mark - self.current_equity) / self.high_water_mark,
            "max_strategy_allocation": self.current_max_strategy_allocation,
            "max_position_allocation": self.current_max_position_allocation,
            "base_risk_per_trade": self.current_base_risk_per_trade,
            "conservation_level": self._calculate_equity_progress(self.current_equity)
        }
    
    def _calculate_equity_progress(self, equity: float) -> float:
        """
        Calculate progress ratio from initial to target equity.
        
        Args:
            equity: Current equity value
            
        Returns:
            Progress ratio from 0.0 (initial) to 1.0+ (at or beyond target)
        """
        if self.target_equity <= self.initial_equity:
            return 1.0  # Avoid division by zero or negative values
            
        # Calculate linear progress
        linear_progress = (equity - self.initial_equity) / (self.target_equity - self.initial_equity)
        
        # Apply non-linear transformation for smoother progression
        # This uses a sigmoid-like curve that starts slow, accelerates in the middle,
        # and then slows down as it approaches the target
        if linear_progress <= 0:
            return 0.0
        elif linear_progress >= 1.0:
            return 1.0
        else:
            # Sigmoid-like transformation
            x = linear_progress * 6 - 3  # Map 0-1 to -3 to 3 for sigmoid
            sigmoid = 1 / (1 + math.exp(-x))
            # Map 0.05-0.95 to 0-1
            return (sigmoid - 0.05) / 0.9
    
    def _recalculate_risk_parameters(self, equity_progress: float, conservation_factor: float = 1.0):
        """
        Recalculate all risk parameters based on equity progress.
        
        Args:
            equity_progress: Progress ratio from initial to target equity (0.0-1.0+)
            conservation_factor: Extra conservation factor for drawdown (0.0-1.0)
        """
        # Clamp progress to 0.0-1.0 range
        progress = min(1.0, max(0.0, equity_progress))
        
        # Linear interpolation between initial and conservative parameters
        self.current_max_strategy_allocation = self._interpolate(
            self.initial_max_strategy_allocation,
            self.conservative_max_strategy_allocation,
            progress
        ) * conservation_factor
        
        self.current_max_position_allocation = self._interpolate(
            self.initial_max_position_allocation,
            self.conservative_max_position_allocation,
            progress
        ) * conservation_factor
        
        self.current_base_risk_per_trade = self._interpolate(
            self.initial_base_risk_per_trade,
            self.conservative_base_risk_per_trade,
            progress
        ) * conservation_factor
    
    def _interpolate(self, initial: float, target: float, progress: float) -> float:
        """
        Linearly interpolate between two values based on progress.
        
        Args:
            initial: Initial value
            target: Target value
            progress: Progress ratio (0.0-1.0)
            
        Returns:
            Interpolated value
        """
        return initial + (target - initial) * progress
    
    def _is_high_performance_strategy(self, metrics: Dict[str, float]) -> bool:
        """
        Determine if a strategy qualifies as high performance based on metrics.
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            True if the strategy meets high performance criteria
        """
        # Extract metrics with defaults
        sharpe = metrics.get('sharpe_ratio', 0.0)
        win_rate = metrics.get('win_rate', 0.0)
        profit_factor = metrics.get('profit_factor', 1.0)
        
        # Extract thresholds
        sharpe_threshold = self.high_performance_threshold.get('sharpe_ratio', 3.0)
        win_rate_threshold = self.high_performance_threshold.get('win_rate', 0.70)
        profit_factor_threshold = self.high_performance_threshold.get('profit_factor', 2.5)
        
        # Check each criterion
        sharpe_check = sharpe >= sharpe_threshold
        win_rate_check = win_rate >= win_rate_threshold
        profit_factor_check = profit_factor >= profit_factor_threshold
        
        # Strategy must meet at least 2 of 3 criteria
        criteria_met = sum([sharpe_check, win_rate_check, profit_factor_check])
        
        return criteria_met >= 2
