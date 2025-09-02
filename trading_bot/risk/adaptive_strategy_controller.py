"""
Adaptive Strategy Controller

This module connects performance tracking and market regime detection with 
the risk management and allocation systems to create a comprehensive adaptive trading system.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import json
import os

from trading_bot.analytics.performance_tracker import PerformanceTracker
from trading_bot.analytics.market_regime_detector import MarketRegimeDetector, MarketRegime
from trading_bot.ai_scoring.snowball_allocator import SnowballAllocator
from trading_bot.risk.position_sizer import PositionSizer
from trading_bot.risk.emergency_brake import EmergencyBrake
from trading_bot.risk.adaptive_risk_manager import AdaptiveRiskManager

logger = logging.getLogger(__name__)

class AdaptiveStrategyController:
    """
    Integrates performance metrics, market regime detection, and risk management
    to create an adaptive trading system that optimizes strategy selection and 
    parameter settings based on current market conditions.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the adaptive strategy controller.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Initialize components
        self.performance_tracker = PerformanceTracker(
            config=self.config.get('performance_tracker', {})
        )
        
        self.market_regime_detector = MarketRegimeDetector(
            config=self.config.get('market_regime_detector', {})
        )
        
        self.snowball_allocator = SnowballAllocator(
            config=self.config.get('snowball_allocator', {})
        )
        
        self.position_sizer = PositionSizer(
            config=self.config.get('position_sizer', {})
        )
        
        self.emergency_brake = EmergencyBrake(
            config=self.config.get('emergency_brake', {})
        )
        
        self.adaptive_risk_manager = AdaptiveRiskManager(
            config=self.config.get('adaptive_risk_manager', {})
        )
        
        # Strategy registry
        self.strategy_registry = {}  # strategy_id -> strategy metadata
        self.strategy_suitability = {}  # strategy_id -> suitability score (0.0 to 1.0)
        self.active_strategies = set()  # Set of active strategy IDs
        self.paused_strategies = set()  # Set of paused strategy IDs
        
        # Market data cache
        self.symbols_data = {}  # symbol -> market data
        
        # System state
        self.total_equity = self.config.get('initial_equity', 10000.0)
        self.strategy_allocations = {}  # strategy_id -> allocation amount
        self.last_allocation_update = datetime.now()
        self.allocation_frequency = self.config.get('allocation_frequency', 'daily')
        
        # Strategy parameters
        self.strategy_parameters = {}  # strategy_id -> parameters dict
        self.parameter_update_frequency = self.config.get('parameter_update_frequency', 'daily')
        self.last_parameter_update = datetime.now()
        
        # Manual overrides
        self.allocation_overrides = {}  # strategy_id -> override allocation
        self.parameter_overrides = {}  # strategy_id -> override parameters
        self.strategy_status_overrides = {}  # strategy_id -> override status ('active', 'paused')
        
        logger.info(f"Initialized AdaptiveStrategyController with {len(self.strategy_registry)} strategies")
    
    def register_strategy(self, 
                         strategy_id: str, 
                         metadata: Dict[str, Any]) -> None:
        """
        Register a strategy with the controller.
        
        Args:
            strategy_id: Unique strategy identifier
            metadata: Strategy metadata including:
                - name: Strategy name
                - description: Strategy description
                - category: Strategy category (e.g., 'trend_following', 'mean_reversion')
                - symbols: List of traded symbols
                - timeframes: List of timeframes
                - parameters: Default parameters
        """
        self.strategy_registry[strategy_id] = metadata
        self.active_strategies.add(strategy_id)
        self.strategy_suitability[strategy_id] = 0.5  # Default neutral suitability
        
        # Store default parameters
        if 'parameters' in metadata:
            self.strategy_parameters[strategy_id] = metadata['parameters'].copy()
        
        logger.info(f"Registered strategy {strategy_id}: {metadata.get('name', 'Unknown')}")
    
    def deregister_strategy(self, strategy_id: str) -> bool:
        """
        Remove a strategy from the controller.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            True if successful
        """
        if strategy_id in self.strategy_registry:
            del self.strategy_registry[strategy_id]
            
            if strategy_id in self.active_strategies:
                self.active_strategies.remove(strategy_id)
                
            if strategy_id in self.paused_strategies:
                self.paused_strategies.remove(strategy_id)
                
            if strategy_id in self.strategy_suitability:
                del self.strategy_suitability[strategy_id]
                
            if strategy_id in self.strategy_parameters:
                del self.strategy_parameters[strategy_id]
                
            if strategy_id in self.allocation_overrides:
                del self.allocation_overrides[strategy_id]
                
            if strategy_id in self.parameter_overrides:
                del self.parameter_overrides[strategy_id]
                
            logger.info(f"Deregistered strategy {strategy_id}")
            return True
        
        return False
    
    def update_market_data(self, 
                          symbol: str, 
                          data: pd.DataFrame) -> None:
        """
        Update market data for a symbol and recalculate regimes.
        
        Args:
            symbol: Trading symbol
            data: DataFrame with OHLCV market data
        """
        # Store data
        self.symbols_data[symbol] = data
        
        # Update market regime
        self.market_regime_detector.update_prices(symbol, data)
        
        # Update strategy suitabilities
        self._update_strategy_suitabilities()
        
        logger.debug(f"Updated market data for {symbol}")
    
    def record_trade_result(self,
                           strategy_id: str,
                           trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record a trade result and update performance metrics.
        
        Args:
            strategy_id: Strategy identifier
            trade_data: Trade data
            
        Returns:
            Updated performance metrics
        """
        # Check if strategy is registered
        if strategy_id not in self.strategy_registry:
            logger.warning(f"Cannot record trade for unknown strategy {strategy_id}")
            return {}
        
        # Record trade in performance tracker
        metrics = self.performance_tracker.record_trade(strategy_id, trade_data)
        
        # Record trade in emergency brake
        slippage_pct = trade_data.get('slippage_pct') or (
            trade_data.get('slippage', 0) / trade_data.get('entry_price', 1) * 100
            if 'slippage' in trade_data and 'entry_price' in trade_data
            else None
        )
        
        self.emergency_brake.record_trade_result(
            strategy_id=strategy_id,
            trade_profit=trade_data.get('pnl', 0),
            slippage_pct=slippage_pct
        )
        
        # Check if emergency brake triggered
        if strategy_id in self.active_strategies and self.emergency_brake.is_strategy_paused(strategy_id):
            self.active_strategies.remove(strategy_id)
            self.paused_strategies.add(strategy_id)
            logger.warning(f"Emergency brake paused strategy {strategy_id}")
        
        # Update allocations if needed
        self._check_allocation_update()
        
        # Update strategy parameters if needed
        self._check_parameter_update()
        
        return metrics
    
    def update_equity(self, new_equity: float) -> None:
        """
        Update total equity and recalculate allocations.
        
        Args:
            new_equity: New equity value
        """
        previous_equity = self.total_equity
        self.total_equity = new_equity
        
        # Update adaptive risk manager
        self.adaptive_risk_manager.update_equity(new_equity)
        
        # Check if allocation update is needed
        equity_change_pct = abs(new_equity - previous_equity) / previous_equity
        allocation_update_needed = equity_change_pct > 0.05  # 5% change
        
        if allocation_update_needed:
            self._update_allocations()
            
        logger.info(f"Updated equity to {new_equity:.2f} (change: {equity_change_pct:.2%})")
    
    def get_strategy_allocation(self, strategy_id: str) -> float:
        """
        Get current allocation for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Allocation amount
        """
        # Check for manual override
        if strategy_id in self.allocation_overrides:
            return self.allocation_overrides[strategy_id]
        
        return self.strategy_allocations.get(strategy_id, 0.0)
    
    def get_all_allocations(self) -> Dict[str, float]:
        """
        Get all current strategy allocations.
        
        Returns:
            Dict mapping strategy IDs to allocation amounts
        """
        return self.strategy_allocations.copy()
    
    def set_allocation_override(self, 
                               strategy_id: str, 
                               allocation: float) -> None:
        """
        Set a manual override for strategy allocation.
        
        Args:
            strategy_id: Strategy identifier
            allocation: Allocation amount
        """
        if strategy_id not in self.strategy_registry:
            logger.warning(f"Cannot set allocation for unknown strategy {strategy_id}")
            return
        
        self.allocation_overrides[strategy_id] = allocation
        logger.info(f"Set allocation override for {strategy_id}: {allocation:.2f}")
    
    def clear_allocation_override(self, strategy_id: str) -> None:
        """
        Clear a manual allocation override.
        
        Args:
            strategy_id: Strategy identifier
        """
        if strategy_id in self.allocation_overrides:
            del self.allocation_overrides[strategy_id]
            logger.info(f"Cleared allocation override for {strategy_id}")
    
    def get_position_size(self, 
                         strategy_id: str,
                         symbol: str,
                         entry_price: float,
                         stop_loss: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate optimal position size for a trade.
        
        Args:
            strategy_id: Strategy identifier
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Optional stop loss price
            
        Returns:
            Dict with position size information
        """
        # Check if strategy is active
        if strategy_id not in self.active_strategies:
            logger.warning(f"Cannot calculate position size for inactive strategy {strategy_id}")
            return {'size': 0, 'reason': 'strategy_inactive'}
        
        # Check emergency brake
        if self.emergency_brake.should_prevent_new_trades(strategy_id):
            logger.warning(f"Emergency brake preventing new trade for {strategy_id}")
            return {'size': 0, 'reason': 'emergency_brake_active'}
        
        # Get strategy allocation
        allocation = self.get_strategy_allocation(strategy_id)
        
        # Calculate position size using the position sizer
        position_info = self.position_sizer.calculate_position_size(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss
        )
        
        # Adjust position size based on strategy allocation
        equity = self.total_equity
        strategy_equity = equity * allocation
        
        position_pct = position_info.get('position_pct', 0.01)  # Default 1%
        adjusted_size = strategy_equity * position_pct / entry_price
        
        # Apply any market regime adjustments
        regime_data = self.market_regime_detector.get_current_regime(symbol)
        regime_adjustment = 1.0
        
        if regime_data:
            if regime_data.regime == MarketRegime.VOLATILE:
                # Reduce position size in volatile markets
                regime_adjustment = max(0.5, 1.0 - regime_data.strength)
                
            elif regime_data.regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                # Potentially increase position size in trending markets
                strategy_type = self.strategy_registry.get(strategy_id, {}).get('category', '')
                
                if (
                    (regime_data.regime == MarketRegime.TRENDING_UP and 'trend_following' in strategy_type.lower()) or
                    (regime_data.regime == MarketRegime.TRENDING_DOWN and 'trend_following' in strategy_type.lower())
                ):
                    # Increase size for trend following strategies in appropriate trends
                    regime_adjustment = min(1.5, 1.0 + 0.5 * regime_data.strength)
        
        # Final adjusted size
        final_size = adjusted_size * regime_adjustment
        
        return {
            'size': final_size,
            'notional': final_size * entry_price,
            'allocation': allocation,
            'allocation_amount': strategy_equity,
            'position_pct': position_pct,
            'regime_adjustment': regime_adjustment
        }
    
    def get_strategy_parameters(self, 
                               strategy_id: str,
                               symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current parameters for a strategy, adjusted for market conditions.
        
        Args:
            strategy_id: Strategy identifier
            symbol: Optional trading symbol for symbol-specific adjustments
            
        Returns:
            Dict with strategy parameters
        """
        # Check if strategy is registered
        if strategy_id not in self.strategy_registry:
            logger.warning(f"Cannot get parameters for unknown strategy {strategy_id}")
            return {}
        
        # Start with base parameters
        parameters = self.strategy_parameters.get(strategy_id, {}).copy()
        
        # Apply manual overrides
        if strategy_id in self.parameter_overrides:
            for key, value in self.parameter_overrides[strategy_id].items():
                parameters[key] = value
        
        # Apply market regime adjustments if symbol provided
        if symbol:
            strategy_type = self.strategy_registry.get(strategy_id, {}).get('category', '')
            
            if strategy_type:
                regime_params = self.market_regime_detector.get_optimal_parameters(
                    symbol=symbol,
                    strategy_type=strategy_type
                )
                
                # Apply regime-specific adjustments
                for key, value in regime_params.items():
                    parameters[key] = value
        
        return parameters
    
    def set_parameter_override(self, 
                              strategy_id: str, 
                              parameters: Dict[str, Any]) -> None:
        """
        Set manual parameter overrides for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            parameters: Parameter overrides
        """
        if strategy_id not in self.strategy_registry:
            logger.warning(f"Cannot set parameters for unknown strategy {strategy_id}")
            return
        
        if strategy_id not in self.parameter_overrides:
            self.parameter_overrides[strategy_id] = {}
            
        # Update overrides
        for key, value in parameters.items():
            self.parameter_overrides[strategy_id][key] = value
            
        logger.info(f"Set parameter overrides for {strategy_id}: {parameters}")
    
    def clear_parameter_override(self, 
                               strategy_id: str,
                               parameter_names: Optional[List[str]] = None) -> None:
        """
        Clear parameter overrides for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            parameter_names: Optional list of parameter names to clear
                             If None, all overrides for the strategy are cleared
        """
        if strategy_id not in self.parameter_overrides:
            return
            
        if parameter_names is None:
            # Clear all overrides
            del self.parameter_overrides[strategy_id]
            logger.info(f"Cleared all parameter overrides for {strategy_id}")
        else:
            # Clear specific parameters
            for name in parameter_names:
                if name in self.parameter_overrides[strategy_id]:
                    del self.parameter_overrides[strategy_id][name]
            
            # If no overrides left, remove the strategy entry
            if not self.parameter_overrides[strategy_id]:
                del self.parameter_overrides[strategy_id]
                
            logger.info(f"Cleared parameter overrides for {strategy_id}: {parameter_names}")
    
    def pause_strategy(self, strategy_id: str, reason: str = "manual") -> bool:
        """
        Pause a strategy from trading.
        
        Args:
            strategy_id: Strategy identifier
            reason: Reason for pausing
            
        Returns:
            True if successful
        """
        if strategy_id not in self.strategy_registry:
            logger.warning(f"Cannot pause unknown strategy {strategy_id}")
            return False
            
        if strategy_id in self.active_strategies:
            self.active_strategies.remove(strategy_id)
            self.paused_strategies.add(strategy_id)
            
            # Set status override
            self.strategy_status_overrides[strategy_id] = 'paused'
            
            logger.info(f"Paused strategy {strategy_id} ({reason})")
            return True
            
        return False
    
    def resume_strategy(self, strategy_id: str) -> bool:
        """
        Resume a paused strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            True if successful
        """
        if strategy_id not in self.strategy_registry:
            logger.warning(f"Cannot resume unknown strategy {strategy_id}")
            return False
            
        if strategy_id in self.paused_strategies:
            # Check if emergency brake is still active
            if self.emergency_brake.is_strategy_paused(strategy_id):
                logger.warning(f"Cannot resume {strategy_id} - emergency brake still active")
                return False
                
            self.paused_strategies.remove(strategy_id)
            self.active_strategies.add(strategy_id)
            
            # Clear status override
            if strategy_id in self.strategy_status_overrides:
                del self.strategy_status_overrides[strategy_id]
                
            logger.info(f"Resumed strategy {strategy_id}")
            return True
            
        return False
    
    def is_strategy_active(self, strategy_id: str) -> bool:
        """
        Check if a strategy is currently active.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            True if strategy is active
        """
        return strategy_id in self.active_strategies
    
    def get_strategy_status(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get detailed status information for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Dict with status information
        """
        if strategy_id not in self.strategy_registry:
            return {'status': 'unknown', 'active': False}
            
        # Basic status
        status = {}
        
        # Active status
        active = strategy_id in self.active_strategies
        status['active'] = active
        status['status'] = 'active' if active else 'paused'
        
        # Override reason if applicable
        if strategy_id in self.strategy_status_overrides:
            status['status_override'] = self.strategy_status_overrides[strategy_id]
            
        # Emergency brake status
        emergency_active = self.emergency_brake.is_strategy_paused(strategy_id)
        status['emergency_brake_active'] = emergency_active
        
        if emergency_active:
            status['emergency_reason'] = self.emergency_brake.get_pause_reason(strategy_id)
            
        # Allocation
        status['allocation'] = self.get_strategy_allocation(strategy_id)
        status['allocation_override'] = strategy_id in self.allocation_overrides
        
        # Performance metrics
        metrics = self.performance_tracker.get_metrics(strategy_id)
        if metrics:
            status['performance'] = {
                'win_rate': metrics.get('win_rate', 0),
                'profit_factor': metrics.get('profit_factor', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'total_trades': metrics.get('total_trades', 0),
                'net_profit': metrics.get('net_profit', 0)
            }
            
        # Suitability
        status['suitability_score'] = self.strategy_suitability.get(strategy_id, 0)
        
        # Parameter overrides
        if strategy_id in self.parameter_overrides:
            status['parameter_overrides'] = self.parameter_overrides[strategy_id]
            
        return status
    
    def get_all_strategy_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all strategies.
        
        Returns:
            Dict mapping strategy IDs to status information
        """
        return {
            strategy_id: self.get_strategy_status(strategy_id)
            for strategy_id in self.strategy_registry
        }
    
    def get_market_regimes(self) -> Dict[str, Any]:
        """
        Get current market regimes for all tracked symbols.
        
        Returns:
            Dict with market regime information
        """
        regimes = {}
        
        for symbol, regime_data in self.market_regime_detector.get_all_regimes().items():
            regimes[symbol] = {
                'regime': regime_data.regime.value,
                'confidence': regime_data.confidence,
                'strength': regime_data.strength,
                'duration': regime_data.duration,
                'metrics': regime_data.metrics
            }
            
        return regimes
    
    def _update_allocations(self) -> None:
        """
        Update strategy allocations based on performance.
        """
        # Get performance metrics for all strategies
        metrics = self.performance_tracker.get_all_metrics()
        
        # Prepare inputs for snowball allocator
        current_weights = {}
        profit_data = {}
        
        for strategy_id, strategy_metrics in metrics.items():
            # Skip inactive or unknown strategies
            if strategy_id not in self.strategy_registry:
                continue
                
            # Use current allocation as weight if available
            if strategy_id in self.strategy_allocations:
                current_weights[strategy_id] = self.strategy_allocations[strategy_id]
            else:
                # Default to equal weight
                current_weights[strategy_id] = 1.0 / len(self.strategy_registry)
                
            # Extract profit data
            profit = strategy_metrics.get('net_profit', 0)
            profit_data[strategy_id] = profit
        
        # Add any missing strategies with default weights
        for strategy_id in self.strategy_registry:
            if strategy_id not in current_weights:
                current_weights[strategy_id] = 1.0 / len(self.strategy_registry)
                
            if strategy_id not in profit_data:
                profit_data[strategy_id] = 0
        
        # Apply adaptive risk management limits
        adaptive_limits = self.adaptive_risk_manager.get_allocation_limits()
        
        # Update allocations with the snowball allocator
        new_allocations = self.snowball_allocator.update_allocations(
            current_weights=current_weights,
            profit_data=profit_data,
            total_equity=self.total_equity
        )
        
        # Apply manual overrides
        for strategy_id, override in self.allocation_overrides.items():
            if strategy_id in new_allocations:
                new_allocations[strategy_id] = override
        
        # Update allocations
        self.strategy_allocations = new_allocations
        self.last_allocation_update = datetime.now()
        
        logger.info(f"Updated allocations for {len(self.strategy_allocations)} strategies")
    
    def _check_allocation_update(self) -> None:
        """
        Check if allocation update is needed based on frequency.
        """
        now = datetime.now()
        time_diff = (now - self.last_allocation_update).total_seconds()
        
        update_needed = False
        
        if self.allocation_frequency == 'daily' and time_diff >= 24*60*60:
            update_needed = True
        elif self.allocation_frequency == 'weekly' and time_diff >= 7*24*60*60:
            update_needed = True
        elif self.allocation_frequency == 'monthly' and time_diff >= 30*24*60*60:
            update_needed = True
        elif self.allocation_frequency == 'hourly' and time_diff >= 60*60:
            update_needed = True
        
        if update_needed:
            self._update_allocations()
    
    def _update_strategy_suitabilities(self) -> None:
        """
        Update strategy suitability scores based on market regimes.
        """
        for strategy_id, metadata in self.strategy_registry.items():
            strategy_type = metadata.get('category', '')
            symbols = metadata.get('symbols', [])
            
            if not strategy_type or not symbols:
                continue
                
            # Calculate average suitability across all strategy symbols
            suitabilities = []
            
            for symbol in symbols:
                symbol_suitability = self.market_regime_detector.get_strategy_suitability(symbol)
                
                if symbol_suitability and strategy_type.lower() in symbol_suitability:
                    suitabilities.append(symbol_suitability[strategy_type.lower()])
            
            if suitabilities:
                avg_suitability = sum(suitabilities) / len(suitabilities)
                self.strategy_suitability[strategy_id] = avg_suitability
    
    def _check_parameter_update(self) -> None:
        """
        Check if parameter update is needed based on frequency.
        """
        now = datetime.now()
        time_diff = (now - self.last_parameter_update).total_seconds()
        
        update_needed = False
        
        if self.parameter_update_frequency == 'daily' and time_diff >= 24*60*60:
            update_needed = True
        elif self.parameter_update_frequency == 'weekly' and time_diff >= 7*24*60*60:
            update_needed = True
        elif self.parameter_update_frequency == 'hourly' and time_diff >= 60*60:
            update_needed = True
        
        if update_needed:
            self._update_all_parameters()
            self.last_parameter_update = now
    
    def _update_all_parameters(self) -> None:
        """
        Update parameters for all strategies based on market conditions.
        """
        for strategy_id, metadata in self.strategy_registry.items():
            strategy_type = metadata.get('category', '')
            symbols = metadata.get('symbols', [])
            
            if not strategy_type or not symbols:
                continue
                
            # Get base parameters
            base_params = self.strategy_parameters.get(strategy_id, {}).copy()
            
            # Update with market regime adjustments for first symbol
            # (would be better to combine adjustments across symbols in a more sophisticated way)
            if symbols:
                symbol = symbols[0]
                regime_params = self.market_regime_detector.get_optimal_parameters(
                    symbol=symbol,
                    strategy_type=strategy_type
                )
                
                # Apply regime-specific adjustments
                for key, value in regime_params.items():
                    base_params[key] = value
            
            # Store updated parameters
            self.strategy_parameters[strategy_id] = base_params
