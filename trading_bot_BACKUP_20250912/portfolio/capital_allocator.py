"""
Capital Allocation Engine

This module provides the core capital allocation functionality for the trading system,
enabling dynamic allocation of capital across strategies based on performance,
risk-adjusted position sizing, and automated portfolio rebalancing.

Features:
- Performance-based capital allocation
- Risk-adjusted position sizing
- Dynamic portfolio rebalancing
- Drawdown-based capital protection
- Strategy correlation analysis
"""

import logging
import threading
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import math

# Import performance metrics and position management
from trading_bot.accounting.performance_metrics import PerformanceMetrics
from trading_bot.position.position_manager import PositionManager
from trading_bot.accounting.pnl_calculator import PnLCalculator
from trading_bot.brokers.multi_broker_manager import MultiBrokerManager

# Import event system if available
try:
    from trading_bot.event_system import EventBus, Event
except ImportError:
    EventBus = None

logger = logging.getLogger(__name__)


class AllocationMethod:
    """Allocation method types."""
    EQUAL = "equal"                  # Equal allocation to all strategies
    PERFORMANCE = "performance"      # Allocate based on relative performance
    RISK_PARITY = "risk_parity"      # Allocate to equalize risk contribution
    KELLY = "kelly"                  # Kelly criterion allocation
    SHARPE = "sharpe"                # Sharpe ratio-based allocation
    CUSTOM = "custom"                # Custom allocation rules


class CapitalAllocator:
    """
    Engine for dynamic capital allocation, position sizing, and portfolio management.
    
    This class handles the intelligent distribution of trading capital across
    multiple strategies based on performance metrics, risk assessment, and
    custom allocation rules.
    """
    
    def __init__(self,
                 performance_metrics: PerformanceMetrics,
                 position_manager: Optional[PositionManager] = None,
                 pnl_calculator: Optional[PnLCalculator] = None,
                 broker_manager: Optional[MultiBrokerManager] = None,
                 event_bus: Optional[Any] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the capital allocator with required components.
        
        Args:
            performance_metrics: Performance metrics calculator
            position_manager: Optional position manager
            pnl_calculator: Optional PnL calculator
            broker_manager: Optional broker manager
            event_bus: Optional event bus for system events
            config: Optional configuration parameters
        """
        # Core components
        self.performance_metrics = performance_metrics
        self.position_manager = position_manager
        self.pnl_calculator = pnl_calculator
        self.broker_manager = broker_manager
        self.event_bus = event_bus
        
        # Load configuration
        self.config = config or {}
        self.allocation_method = self.config.get('allocation_method', AllocationMethod.PERFORMANCE)
        self.rebalance_frequency = self.config.get('rebalance_frequency_hours', 24)  # hours
        self.min_allocation_pct = self.config.get('min_allocation_pct', 5)  # minimum 5% allocation
        self.max_allocation_pct = self.config.get('max_allocation_pct', 50)  # maximum 50% allocation
        self.drawdown_protection = self.config.get('drawdown_protection', True)
        self.max_drawdown_pct = self.config.get('max_drawdown_pct', 25)  # 25% max drawdown
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)  # 2% default
        self.volatility_window = self.config.get('volatility_window_days', 30)  # 30 days
        
        # Allocation state
        self.strategy_allocations: Dict[str, float] = {}  # strategy_id -> allocation percentage
        self.strategy_max_positions: Dict[str, int] = {}  # strategy_id -> max concurrent positions
        self.strategy_risk_metrics: Dict[str, Dict[str, Any]] = {}  # strategy_id -> risk metrics
        self.total_capital: float = self.config.get('initial_capital', 100000.0)
        self.reserved_capital_pct: float = self.config.get('reserved_capital_pct', 10.0)  # keep 10% in reserve
        
        # System state
        self.last_rebalance_time: Optional[datetime] = None
        self.rebalance_pending: bool = False
        self.active: bool = False
        self._monitoring_thread = None
        self._lock = threading.RLock()
        
        # Initialize allocations
        self._initialize_allocations()
        
        logger.info("Capital Allocator initialized")
    
    def _initialize_allocations(self) -> None:
        """Initialize strategy allocations based on configuration or defaults."""
        with self._lock:
            try:
                # Get active strategies from configuration or performance metrics
                strategy_ids = self.config.get('strategy_ids', None)
                
                if not strategy_ids and hasattr(self.performance_metrics, 'get_strategy_ids'):
                    strategy_ids = self.performance_metrics.get_strategy_ids()
                
                if not strategy_ids:
                    logger.warning("No strategies found for allocation")
                    return
                
                # Check if we have predefined allocations in config
                predefined_allocations = self.config.get('strategy_allocations', None)
                
                if predefined_allocations:
                    # Use predefined allocations
                    total_allocation = sum(predefined_allocations.values())
                    if abs(total_allocation - 100.0) > 0.01:  # Allow small rounding error
                        logger.warning(f"Predefined allocations sum to {total_allocation}%, normalizing to 100%")
                        # Normalize to 100%
                        factor = 100.0 / total_allocation
                        self.strategy_allocations = {
                            strategy_id: alloc * factor
                            for strategy_id, alloc in predefined_allocations.items()
                        }
                    else:
                        self.strategy_allocations = predefined_allocations.copy()
                else:
                    # Default to equal allocation across all strategies
                    equal_allocation = 100.0 / len(strategy_ids)
                    self.strategy_allocations = {
                        strategy_id: equal_allocation for strategy_id in strategy_ids
                    }
                
                # Initialize max positions per strategy
                default_max_positions = self.config.get('default_max_positions', 5)
                predefined_max_positions = self.config.get('strategy_max_positions', {})
                
                for strategy_id in strategy_ids:
                    # Use predefined value or default
                    self.strategy_max_positions[strategy_id] = predefined_max_positions.get(
                        strategy_id, default_max_positions
                    )
                
                logger.info(f"Initialized allocations for {len(strategy_ids)} strategies")
                logger.info(f"Strategy allocations: {json.dumps(self.strategy_allocations)}")
            
            except Exception as e:
                logger.error(f"Error initializing allocations: {str(e)}")
    
    def start(self) -> bool:
        """
        Start the capital allocation engine.
        
        Returns:
            bool: Success status
        """
        if self.active:
            logger.warning("Capital allocator already running")
            return True
        
        try:
            # Get initial capital if broker manager is available
            if self.broker_manager:
                try:
                    accounts = self.broker_manager.get_account_summary()
                    if accounts:
                        # Use the sum of all account balances
                        total_balance = sum(float(acct.get('cash', 0)) for acct in accounts.values())
                        if total_balance > 0:
                            self.total_capital = total_balance
                            logger.info(f"Updated total capital to {self.total_capital} from broker accounts")
                except Exception as e:
                    logger.error(f"Error getting account balances: {str(e)}")
            
            # Register for events if event bus is available
            if self.event_bus:
                self._register_event_handlers()
            
            # Perform initial allocation
            self.update_allocations(force=True)
            
            # Start monitoring thread
            self.active = True
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="CapitalAllocator",
                daemon=True
            )
            self._monitoring_thread.start()
            
            logger.info("Started capital allocation engine")
            return True
        
        except Exception as e:
            self.active = False
            logger.error(f"Error starting capital allocator: {str(e)}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the capital allocation engine.
        
        Returns:
            bool: Success status
        """
        if not self.active:
            logger.warning("Capital allocator already stopped")
            return True
        
        try:
            self.active = False
            
            # Wait for monitoring thread to terminate
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=5.0)
            
            logger.info("Stopped capital allocation engine")
            return True
        
        except Exception as e:
            logger.error(f"Error stopping capital allocator: {str(e)}")
            return False
    
    def _register_event_handlers(self) -> None:
        """Register for relevant system events."""
        if not self.event_bus:
            return
            
        try:
            # Register for trade events
            self.event_bus.subscribe("POSITION_OPENED", self._handle_position_opened)
            self.event_bus.subscribe("POSITION_CLOSED", self._handle_position_closed)
            self.event_bus.subscribe("STRATEGY_PERFORMANCE_UPDATED", self._handle_performance_update)
            self.event_bus.subscribe("ACCOUNT_BALANCE_CHANGED", self._handle_balance_change)
            
            logger.info("Registered event handlers for Capital Allocator")
        except Exception as e:
            logger.error(f"Error registering event handlers: {str(e)}")
    
    def _handle_position_opened(self, event: Any) -> None:
        """Handle position opened events."""
        try:
            data = getattr(event, 'data', event)
            strategy_id = data.get('strategy_id')
            position_id = data.get('position_id')
            
            if strategy_id and position_id:
                logger.debug(f"Position opened: {position_id} for strategy {strategy_id}")
                # Could update strategy usage statistics here
        except Exception as e:
            logger.error(f"Error handling position opened: {str(e)}")
    
    def _handle_position_closed(self, event: Any) -> None:
        """Handle position closed events."""
        try:
            data = getattr(event, 'data', event)
            strategy_id = data.get('strategy_id')
            position_id = data.get('position_id')
            pnl = data.get('pnl')
            
            if strategy_id and pnl is not None:
                logger.debug(f"Position closed: {position_id} for strategy {strategy_id} with P&L {pnl}")
                # Mark for potential reallocation on next cycle
                self.rebalance_pending = True
        except Exception as e:
            logger.error(f"Error handling position closed: {str(e)}")
    
    def _handle_performance_update(self, event: Any) -> None:
        """Handle strategy performance update events."""
        try:
            data = getattr(event, 'data', event)
            strategy_id = data.get('strategy_id')
            
            if strategy_id:
                logger.debug(f"Performance updated for strategy {strategy_id}")
                # Mark for potential reallocation on next cycle
                self.rebalance_pending = True
        except Exception as e:
            logger.error(f"Error handling performance update: {str(e)}")
    
    def _handle_balance_change(self, event: Any) -> None:
        """Handle account balance change events."""
        try:
            data = getattr(event, 'data', event)
            account_id = data.get('account_id')
            balance = data.get('balance')
            
            if account_id and balance is not None:
                logger.debug(f"Balance changed for account {account_id}: {balance}")
                # Update total capital if broker manager is available
                if self.broker_manager:
                    accounts = self.broker_manager.get_account_summary()
                    if accounts:
                        total_balance = sum(float(acct.get('cash', 0)) for acct in accounts.values())
                        if total_balance > 0:
                            with self._lock:
                                self.total_capital = total_balance
                            logger.info(f"Updated total capital to {self.total_capital}")
                        
                # Mark for potential reallocation on next cycle
                self.rebalance_pending = True
        except Exception as e:
            logger.error(f"Error handling balance change: {str(e)}")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop that checks for rebalance conditions."""
        logger.info("Capital allocation monitoring loop started")
        
        while self.active:
            try:
                # Check if it's time to rebalance
                current_time = datetime.now()
                needs_rebalance = False
                
                # Check if forced rebalance is pending
                if self.rebalance_pending:
                    needs_rebalance = True
                    self.rebalance_pending = False
                
                # Check if scheduled rebalance is needed
                elif self.last_rebalance_time:
                    hours_since_rebalance = (current_time - self.last_rebalance_time).total_seconds() / 3600
                    if hours_since_rebalance >= self.rebalance_frequency:
                        needs_rebalance = True
                else:
                    # First run
                    needs_rebalance = True
                
                # Perform rebalance if needed
                if needs_rebalance:
                    logger.info("Scheduled rebalance triggered")
                    self.update_allocations()
                
                # Sleep for a while
                # Check every 5 minutes, or sooner if rebalance is pending
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(600)  # Sleep longer on error
        
        logger.info("Capital allocation monitoring loop terminated")
    
    def update_allocations(self, force: bool = False) -> bool:
        """
        Update strategy allocations based on performance and risk metrics.
        
        Args:
            force: Force update even if conditions don't require it
            
        Returns:
            bool: Success status
        """
        with self._lock:
            try:
                current_time = datetime.now()
                
                # Skip if recent rebalance and not forced
                if not force and self.last_rebalance_time:
                    hours_since_rebalance = (current_time - self.last_rebalance_time).total_seconds() / 3600
                    if hours_since_rebalance < self.rebalance_frequency:
                        return False
                
                # Update strategy risk metrics
                self._update_risk_metrics()
                
                # Calculate new allocations based on selected method
                if self.allocation_method == AllocationMethod.EQUAL:
                    success = self._allocate_equal()
                elif self.allocation_method == AllocationMethod.PERFORMANCE:
                    success = self._allocate_by_performance()
                elif self.allocation_method == AllocationMethod.RISK_PARITY:
                    success = self._allocate_risk_parity()
                elif self.allocation_method == AllocationMethod.KELLY:
                    success = self._allocate_kelly()
                elif self.allocation_method == AllocationMethod.SHARPE:
                    success = self._allocate_by_sharpe()
                elif self.allocation_method == AllocationMethod.CUSTOM:
                    success = self._apply_custom_allocation_rules()
                else:
                    logger.warning(f"Unknown allocation method: {self.allocation_method}")
                    success = False
                
                if success:
                    # Update rebalance time
                    self.last_rebalance_time = current_time
                    
                    # Apply drawdown protection if enabled
                    if self.drawdown_protection:
                        self._apply_drawdown_protection()
                    
                    # Enforce min/max allocation limits
                    self._enforce_allocation_limits()
                    
                    # Emit allocation update event
                    if self.event_bus:
                        self.event_bus.emit("ALLOCATION_UPDATED", {
                            "allocations": self.strategy_allocations,
                            "timestamp": current_time.isoformat()
                        })
                    
                    logger.info(f"Updated allocations: {json.dumps(self.strategy_allocations)}")
                    return True
                else:
                    logger.warning("Failed to update allocations")
                    return False
                
            except Exception as e:
                logger.error(f"Error updating allocations: {str(e)}")
                return False
    
    def _update_risk_metrics(self) -> None:
        """Update risk metrics for all strategies."""
        try:
            # Get all strategy IDs
            strategy_ids = list(self.strategy_allocations.keys())
            
            for strategy_id in strategy_ids:
                # Initialize metrics dictionary if not exists
                if strategy_id not in self.strategy_risk_metrics:
                    self.strategy_risk_metrics[strategy_id] = {}
                
                # Get performance metrics for this strategy
                try:
                    # Trading performance
                    win_rate = self.performance_metrics.get_win_rate(strategy_id)
                    profit_factor = self.performance_metrics.get_profit_factor(strategy_id)
                    avg_win = self.performance_metrics.get_average_win(strategy_id)
                    avg_loss = self.performance_metrics.get_average_loss(strategy_id)
                    max_drawdown = self.performance_metrics.get_max_drawdown_percentage(strategy_id)
                    
                    # Risk metrics
                    daily_returns = self.performance_metrics.get_daily_returns(strategy_id)
                    if len(daily_returns) > 0:
                        volatility = float(np.std(daily_returns))
                        sharpe = self._calculate_sharpe_ratio(daily_returns)
                        sortino = self._calculate_sortino_ratio(daily_returns)
                        
                        # Calculate equity curve metrics
                        drawdown_days = self.performance_metrics.get_drawdown_days(strategy_id)
                        recovery_factor = self.performance_metrics.get_recovery_factor(strategy_id)
                        
                        # Current drawdown
                        current_drawdown = self.performance_metrics.get_current_drawdown_percentage(strategy_id)
                        
                        # Store metrics
                        self.strategy_risk_metrics[strategy_id] = {
                            'win_rate': win_rate,
                            'profit_factor': profit_factor,
                            'avg_win': avg_win,
                            'avg_loss': avg_loss,
                            'max_drawdown': max_drawdown,
                            'current_drawdown': current_drawdown,
                            'volatility': volatility,
                            'sharpe_ratio': sharpe,
                            'sortino_ratio': sortino,
                            'drawdown_days': drawdown_days,
                            'recovery_factor': recovery_factor,
                            'expectancy': (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss)),
                            'updated_at': datetime.now().isoformat()
                        }
                    else:
                        logger.warning(f"No daily returns data for strategy {strategy_id}")
                        # Use defaults for new strategies
                        self.strategy_risk_metrics[strategy_id] = {
                            'win_rate': 0.5,
                            'profit_factor': 1.0,
                            'avg_win': 0.0,
                            'avg_loss': 0.0,
                            'max_drawdown': 0.0,
                            'current_drawdown': 0.0,
                            'volatility': 0.01,
                            'sharpe_ratio': 0.0,
                            'sortino_ratio': 0.0,
                            'drawdown_days': 0,
                            'recovery_factor': 0.0,
                            'expectancy': 0.0,
                            'updated_at': datetime.now().isoformat()
                        }
                except Exception as e:
                    logger.error(f"Error updating risk metrics for {strategy_id}: {str(e)}")
                    # Ensure we have at least default values
                    if strategy_id not in self.strategy_risk_metrics:
                        self.strategy_risk_metrics[strategy_id] = {'updated_at': datetime.now().isoformat()}
            
            logger.debug(f"Updated risk metrics for {len(strategy_ids)} strategies")
        except Exception as e:
            logger.error(f"Error in _update_risk_metrics: {str(e)}")
    
    def _allocate_equal(self) -> bool:
        """
        Allocate capital equally across all strategies.
        
        Returns:
            bool: Success status
        """
        try:
            strategy_ids = list(self.strategy_allocations.keys())
            if not strategy_ids:
                logger.warning("No strategies found for equal allocation")
                return False
            
            # Equal allocation for all strategies
            equal_allocation = 100.0 / len(strategy_ids)
            for strategy_id in strategy_ids:
                self.strategy_allocations[strategy_id] = equal_allocation
            
            logger.info(f"Applied equal allocation ({equal_allocation:.2f}%) to {len(strategy_ids)} strategies")
            return True
        except Exception as e:
            logger.error(f"Error in equal allocation: {str(e)}")
            return False
    
    def _allocate_by_performance(self) -> bool:
        """
        Allocate capital based on relative strategy performance.
        
        Returns:
            bool: Success status
        """
        try:
            strategy_ids = list(self.strategy_allocations.keys())
            if not strategy_ids:
                logger.warning("No strategies found for performance-based allocation")
                return False
            
            # Calculate performance scores
            performance_scores = {}
            total_score = 0.0
            
            for strategy_id in strategy_ids:
                metrics = self.strategy_risk_metrics.get(strategy_id, {})
                
                # Default score is 1.0 (equal weight)
                score = 1.0
                
                # Calculate score based on available metrics
                # We use a weighted combination of several key performance indicators
                if metrics:
                    # 1. Profit factor (most important)
                    profit_factor = metrics.get('profit_factor', 1.0)
                    # Limit profit factor influence
                    profit_factor_score = min(profit_factor, 5.0) / 5.0  # Scale 0-5 to 0-1
                    
                    # 2. Sharpe ratio (risk-adjusted returns)
                    sharpe = metrics.get('sharpe_ratio', 0.0)
                    # Limit Sharpe influence and handle negative values
                    sharpe_score = max(0, min(sharpe, 3.0)) / 3.0  # Scale 0-3 to 0-1
                    
                    # 3. Recovery factor (capital efficiency)
                    recovery = metrics.get('recovery_factor', 0.0)
                    recovery_score = min(recovery, 10.0) / 10.0  # Scale 0-10 to 0-1
                    
                    # 4. Expectancy (average trade profitability)
                    expectancy = metrics.get('expectancy', 0.0)
                    # Scale expectancy score
                    expectancy_score = 0.5  # Default neutral
                    if expectancy > 0:
                        expectancy_score = min(0.5 + (expectancy / 2.0), 1.0)  # Scale up to 1.0
                    elif expectancy < 0:
                        expectancy_score = max(0.5 - (abs(expectancy) / 2.0), 0.0)  # Scale down to 0.0
                    
                    # 5. Drawdown penalty
                    max_dd = metrics.get('max_drawdown', 0.0)
                    dd_score = 1.0 - (min(max_dd, 50.0) / 50.0)  # Penalize for drawdowns up to 50%
                    
                    # Combine scores with weights
                    # Profit factor and expectancy are most important
                    score = (
                        (profit_factor_score * 0.35) +  # 35% weight
                        (sharpe_score * 0.20) +         # 20% weight
                        (recovery_score * 0.10) +       # 10% weight
                        (expectancy_score * 0.25) +     # 25% weight
                        (dd_score * 0.10)               # 10% weight
                    )
                    
                    # Ensure minimal score for all strategies
                    score = max(score, 0.2)  # Minimum score of 0.2
                
                performance_scores[strategy_id] = score
                total_score += score
            
            # Normalize scores to percentages
            if total_score > 0:
                for strategy_id in strategy_ids:
                    self.strategy_allocations[strategy_id] = (performance_scores[strategy_id] / total_score) * 100.0
            else:
                # Fallback to equal allocation if no valid scores
                return self._allocate_equal()
            
            logger.info(f"Applied performance-based allocation to {len(strategy_ids)} strategies")
            return True
        except Exception as e:
            logger.error(f"Error in performance allocation: {str(e)}")
            return False
    
    def _allocate_risk_parity(self) -> bool:
        """
        Allocate capital to equalize risk contribution from each strategy.
        Risk parity: allocate inversely proportional to volatility.
        
        Returns:
            bool: Success status
        """
        try:
            strategy_ids = list(self.strategy_allocations.keys())
            if not strategy_ids:
                logger.warning("No strategies found for risk parity allocation")
                return False
            
            # Calculate inverse volatility values
            inv_volatility = {}
            total_inv_volatility = 0.0
            
            for strategy_id in strategy_ids:
                metrics = self.strategy_risk_metrics.get(strategy_id, {})
                volatility = metrics.get('volatility', 0.01)  # Default to 1% volatility
                
                # Ensure volatility is positive and not too small
                volatility = max(volatility, 0.001)  # Minimum 0.1% volatility
                
                # Inverse volatility (higher vol = lower allocation)
                inv_vol = 1.0 / volatility
                inv_volatility[strategy_id] = inv_vol
                total_inv_volatility += inv_vol
            
            # Normalize to percentages
            if total_inv_volatility > 0:
                for strategy_id in strategy_ids:
                    self.strategy_allocations[strategy_id] = (inv_volatility[strategy_id] / total_inv_volatility) * 100.0
            else:
                # Fallback to equal allocation if issues
                return self._allocate_equal()
            
            logger.info(f"Applied risk parity allocation to {len(strategy_ids)} strategies")
            return True
        except Exception as e:
            logger.error(f"Error in risk parity allocation: {str(e)}")
            return False
    
    def _allocate_kelly(self) -> bool:
        """
        Allocate capital using Kelly criterion for optimal growth.
        Kelly fraction = (p*b - q)/b where p=win rate, q=loss rate, b=win/loss ratio
        
        Returns:
            bool: Success status
        """
        try:
            strategy_ids = list(self.strategy_allocations.keys())
            if not strategy_ids:
                logger.warning("No strategies found for Kelly allocation")
                return False
            
            # Calculate Kelly fractions
            kelly_fractions = {}
            total_kelly = 0.0
            kelly_limit = self.config.get('max_kelly_fraction', 0.5)  # Limit to half-Kelly by default
            
            for strategy_id in strategy_ids:
                metrics = self.strategy_risk_metrics.get(strategy_id, {})
                
                # Get required metrics
                win_rate = metrics.get('win_rate', 0.5)
                avg_win = metrics.get('avg_win', 0.0)  
                avg_loss = abs(metrics.get('avg_loss', 0.0))  # Make positive
                
                # Safety checks
                if avg_loss < 0.01:  # Avoid division by zero
                    avg_loss = 0.01
                
                # Calculate win/loss ratio
                win_loss_ratio = avg_win / avg_loss
                
                # Calculate loss rate
                loss_rate = 1.0 - win_rate
                
                # Classic Kelly formula: (p*b - q)/b
                kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
                
                # Apply safety limits
                kelly = max(0.0, min(kelly, kelly_limit))  # Constrain between 0 and kelly_limit
                
                kelly_fractions[strategy_id] = kelly
                total_kelly += kelly
            
            # Normalize to percentages
            if total_kelly > 0:
                for strategy_id in strategy_ids:
                    self.strategy_allocations[strategy_id] = (kelly_fractions[strategy_id] / total_kelly) * 100.0
            else:
                # Fallback to equal allocation if Kelly values are all zero
                return self._allocate_equal()
            
            logger.info(f"Applied Kelly criterion allocation to {len(strategy_ids)} strategies")
            return True
        except Exception as e:
            logger.error(f"Error in Kelly allocation: {str(e)}")
            return False
    
    def _allocate_by_sharpe(self) -> bool:
        """
        Allocate capital based on relative Sharpe ratios.
        Sharpe ratio = (Return - Risk Free Rate) / Volatility
        
        Returns:
            bool: Success status
        """
        try:
            strategy_ids = list(self.strategy_allocations.keys())
            if not strategy_ids:
                logger.warning("No strategies found for Sharpe ratio allocation")
                return False
            
            # Get Sharpe ratios
            sharpe_values = {}
            total_sharpe = 0.0
            
            for strategy_id in strategy_ids:
                metrics = self.strategy_risk_metrics.get(strategy_id, {})
                
                # Get Sharpe ratio, default to 0 if negative
                sharpe = max(0.0, metrics.get('sharpe_ratio', 0.0))
                
                sharpe_values[strategy_id] = sharpe
                total_sharpe += sharpe
            
            # Normalize to percentages
            if total_sharpe > 0:
                for strategy_id in strategy_ids:
                    self.strategy_allocations[strategy_id] = (sharpe_values[strategy_id] / total_sharpe) * 100.0
            else:
                # Fallback to equal allocation if all Sharpe ratios are zero or negative
                return self._allocate_equal()
            
            logger.info(f"Applied Sharpe ratio allocation to {len(strategy_ids)} strategies")
            return True
        except Exception as e:
            logger.error(f"Error in Sharpe ratio allocation: {str(e)}")
            return False
    
    def _apply_custom_allocation_rules(self) -> bool:
        """
        Apply custom allocation rules defined in the configuration.
        
        Returns:
            bool: Success status
        """
        try:
            # Check if we have custom rules
            custom_rules = self.config.get('custom_allocation_rules', None)
            if not custom_rules:
                logger.warning("No custom allocation rules defined")
                return False
            
            # Custom rules would be defined in configuration
            # This is a placeholder for implementation
            # Example rule: "if strategy X has win_rate > 0.6, increase allocation by 20%"
            
            # For now, just warn and fall back to performance-based allocation
            logger.warning("Custom allocation rules not implemented, using performance-based allocation")
            return self._allocate_by_performance()
            
        except Exception as e:
            logger.error(f"Error in custom allocation: {str(e)}")
            return False
    
    def _apply_drawdown_protection(self) -> None:
        """
        Apply drawdown protection rules to current allocations.
        Reduces allocation to strategies in significant drawdown.
        """
        try:
            for strategy_id, metrics in self.strategy_risk_metrics.items():
                current_drawdown = metrics.get('current_drawdown', 0.0)
                
                # If in significant drawdown, reduce allocation
                if current_drawdown > self.max_drawdown_pct:
                    # Calculate reduction factor based on drawdown severity
                    reduction_factor = 1.0 - ((current_drawdown - self.max_drawdown_pct) / 
                                             (100.0 - self.max_drawdown_pct))
                    
                    # Ensure minimum reduction
                    reduction_factor = max(0.2, reduction_factor)  # Minimum 20% of original allocation
                    
                    # Apply reduction to allocation
                    original_allocation = self.strategy_allocations.get(strategy_id, 0.0)
                    reduced_allocation = original_allocation * reduction_factor
                    
                    logger.info(f"Reducing allocation for {strategy_id} due to {current_drawdown:.2f}% drawdown: "
                               f"{original_allocation:.2f}% -> {reduced_allocation:.2f}%")
                    
                    self.strategy_allocations[strategy_id] = reduced_allocation
            
            # Normalize allocations to 100% after drawdown adjustments
            self._normalize_allocations()
            
        except Exception as e:
            logger.error(f"Error applying drawdown protection: {str(e)}")
    
    def _enforce_allocation_limits(self) -> None:
        """
        Enforce minimum and maximum allocation percentages.
        """
        try:
            # Apply minimum and maximum allocation limits
            limited_allocations = {}
            total_limited = 0.0
            
            for strategy_id, allocation in self.strategy_allocations.items():
                # Apply limits
                if allocation < self.min_allocation_pct:
                    # Round down to zero for very small allocations
                    if allocation < self.min_allocation_pct / 2:
                        limited_allocations[strategy_id] = 0.0
                    else:
                        limited_allocations[strategy_id] = self.min_allocation_pct
                elif allocation > self.max_allocation_pct:
                    limited_allocations[strategy_id] = self.max_allocation_pct
                else:
                    limited_allocations[strategy_id] = allocation
                
                total_limited += limited_allocations[strategy_id]
            
            # Final normalization to ensure sum is 100%
            if total_limited > 0:
                factor = 100.0 / total_limited
                for strategy_id in limited_allocations.keys():
                    limited_allocations[strategy_id] *= factor
            
            # Update allocations
            self.strategy_allocations = limited_allocations
            
        except Exception as e:
            logger.error(f"Error enforcing allocation limits: {str(e)}")
    
    def _normalize_allocations(self) -> None:
        """
        Normalize allocations to ensure they sum to 100%.
        """
        try:
            total_allocation = sum(self.strategy_allocations.values())
            
            if abs(total_allocation - 100.0) > 0.01:  # Check if adjustment needed
                if total_allocation > 0:
                    # Scale all allocations
                    factor = 100.0 / total_allocation
                    for strategy_id in self.strategy_allocations.keys():
                        self.strategy_allocations[strategy_id] *= factor
                else:
                    # Reset to equal if total is zero
                    self._allocate_equal()
                    
        except Exception as e:
            logger.error(f"Error normalizing allocations: {str(e)}")
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """
        Calculate Sharpe ratio for a series of returns.
        
        Args:
            returns: List of return values
            
        Returns:
            float: Sharpe ratio
        """
        try:
            if len(returns) < 2:
                return 0.0
            
            # Calculate annualized return
            mean_return = float(np.mean(returns))
            annualized_return = mean_return * 252  # Trading days in a year
            
            # Calculate annualized volatility
            std_dev = float(np.std(returns))
            annualized_volatility = std_dev * math.sqrt(252)
            
            # Avoid division by zero
            if annualized_volatility < 0.0001:
                return 0.0
            
            # Calculate Sharpe ratio
            sharpe = (annualized_return - self.risk_free_rate) / annualized_volatility
            return sharpe
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """
        Calculate Sortino ratio for a series of returns.
        Similar to Sharpe but penalizes only downside volatility.
        
        Args:
            returns: List of return values
            
        Returns:
            float: Sortino ratio
        """
        try:
            if len(returns) < 2:
                return 0.0
            
            # Calculate annualized return
            mean_return = float(np.mean(returns))
            annualized_return = mean_return * 252  # Trading days in a year
            
            # Calculate downside deviation (only negative returns)
            downside_returns = [r for r in returns if r < 0]
            
            if not downside_returns:
                return 0.0 if mean_return <= 0 else 10.0  # Arbitrary high value if no downside
            
            downside_deviation = float(np.std(downside_returns))
            annualized_downside = downside_deviation * math.sqrt(252)
            
            # Avoid division by zero
            if annualized_downside < 0.0001:
                return 0.0
            
            # Calculate Sortino ratio
            sortino = (annualized_return - self.risk_free_rate) / annualized_downside
            return sortino
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {str(e)}")
            return 0.0
    
    def calculate_position_size(self, 
                              strategy_id: str, 
                              symbol: str, 
                              signal_strength: float = 1.0,
                              entry_price: Optional[float] = None,
                              stop_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate optimal position size for a new trade.
        
        Args:
            strategy_id: ID of the strategy generating the trade
            symbol: Trading symbol
            signal_strength: Optional signal strength (0.0-1.0)
            entry_price: Optional entry price for calculating risk
            stop_price: Optional stop price for calculating risk
            
        Returns:
            Dict with position size information
        """
        with self._lock:
            try:
                # Get strategy allocation
                allocation_pct = self.strategy_allocations.get(strategy_id, 0.0)
                if allocation_pct <= 0:
                    logger.warning(f"No allocation for strategy {strategy_id}")
                    return {'size': 0, 'capital': 0, 'max_risk': 0}
                
                # Calculate available capital for this strategy
                allocated_capital = self.total_capital * (allocation_pct / 100.0)
                
                # Account for reserved capital
                effective_capital = allocated_capital * (1 - self.reserved_capital_pct / 100.0)
                
                # Get active positions for this strategy
                active_positions = 0
                if self.position_manager:
                    strategy_positions = [
                        p for p in self.position_manager.internal_positions.values()
                        if p.get('strategy_id') == strategy_id
                    ]
                    active_positions = len(strategy_positions)
                
                # Get max positions for this strategy
                max_positions = self.strategy_max_positions.get(strategy_id, 5)  # Default to 5
                
                # Calculate capital per position
                if active_positions >= max_positions:
                    # No more positions allowed
                    logger.info(f"Strategy {strategy_id} already has {active_positions}/{max_positions} positions")
                    return {'size': 0, 'capital': 0, 'max_risk': 0}
                
                # Determine remaining positions
                remaining_positions = max_positions - active_positions
                
                # Calculate capital per position
                capital_per_position = effective_capital / max_positions
                
                # Adjust for signal strength
                capital_for_trade = capital_per_position * signal_strength
                
                # Calculate risk amount (default to 2% of allocated capital)
                risk_per_trade_pct = self.config.get('risk_per_trade_pct', 2.0)  # Default 2%
                max_risk_amount = allocated_capital * (risk_per_trade_pct / 100.0)
                
                # Calculate size based on risk if stop loss is provided
                size = 0
                if entry_price is not None and stop_price is not None and entry_price != stop_price:
                    # Calculate risk per share/contract
                    risk_per_unit = abs(entry_price - stop_price)
                    
                    # Calculate size based on risk
                    max_size = max_risk_amount / risk_per_unit
                    
                    # Determine size based on capital and risk
                    size_by_capital = capital_for_trade / entry_price
                    size = min(max_size, size_by_capital)
                else:
                    # Calculate size based on capital only
                    size = capital_for_trade / entry_price if entry_price else 0
                
                # Round size to appropriate precision
                fractional_shares = self.config.get('allow_fractional_shares', False)
                if not fractional_shares and size > 0:
                    size = math.floor(size)
                else:
                    # Round to 6 decimal places for crypto or forex
                    size = round(size, 6)
                
                # Calculate actual capital and risk
                actual_capital = size * entry_price if entry_price else 0
                actual_risk = size * abs(entry_price - stop_price) if entry_price and stop_price else 0
                
                result = {
                    'strategy_id': strategy_id,
                    'symbol': symbol,
                    'size': size,
                    'capital': actual_capital,
                    'max_risk': actual_risk,
                    'allocation_pct': allocation_pct,
                    'allocated_capital': allocated_capital,
                    'max_positions': max_positions,
                    'active_positions': active_positions,
                    'signal_strength': signal_strength
                }
                
                logger.info(f"Calculated position size for {strategy_id}/{symbol}: {size} units, {actual_capital:.2f} capital, {actual_risk:.2f} risk")
                return result
                
            except Exception as e:
                logger.error(f"Error calculating position size: {str(e)}")
                return {'size': 0, 'capital': 0, 'max_risk': 0, 'error': str(e)}
    
    def adjust_position_for_correlation(self, 
                                      strategy_id: str, 
                                      symbol: str, 
                                      base_size: float) -> float:
        """
        Adjust position size based on correlation with existing positions.
        
        Args:
            strategy_id: Strategy ID
            symbol: Symbol for new position
            base_size: Base position size to adjust
            
        Returns:
            float: Adjusted position size
        """
        try:
            # Get current portfolio positions
            if not self.position_manager:
                return base_size  # No adjustment possible without position manager
            
            positions = self.position_manager.internal_positions.values()
            if not positions:
                return base_size  # No existing positions to correlate with
            
            # Get current symbols in portfolio
            current_symbols = [p.get('symbol') for p in positions]
            current_directions = [p.get('direction') for p in positions]
            
            if symbol in current_symbols:
                # Symbol already in portfolio - check if we're hedging or concentrating
                existing_positions = [p for p in positions if p.get('symbol') == symbol]
                existing_directions = [p.get('direction') for p in existing_positions]
                
                # For now, simple adjustment: reduce size if adding to existing direction
                # This is a placeholder for more sophisticated correlation analysis
                if len(existing_positions) > 0:
                    logger.info(f"Symbol {symbol} already has {len(existing_positions)} positions")
                    # Reduce size by 20% for each existing position in same symbol
                    reduction_factor = max(0.6, 1.0 - (0.2 * len(existing_positions)))
                    adjusted_size = base_size * reduction_factor
                    logger.info(f"Reduced position size due to existing positions: {base_size} -> {adjusted_size}")
                    return adjusted_size
            
            # In future versions, implement correlation-based adjustments using market data
            # This would involve calculating correlation matrix between symbols
            # and reducing position sizes for highly correlated assets
            
            return base_size
            
        except Exception as e:
            logger.error(f"Error adjusting for correlation: {str(e)}")
            return base_size  # Return original size on error
    
    def get_strategy_allocation(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get detailed allocation information for a strategy.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Dict with allocation details
        """
        with self._lock:
            try:
                # Get basic allocation
                allocation_pct = self.strategy_allocations.get(strategy_id, 0.0)
                allocated_capital = self.total_capital * (allocation_pct / 100.0)
                
                # Get risk metrics
                risk_metrics = self.strategy_risk_metrics.get(strategy_id, {})
                
                # Get active positions
                active_positions = 0
                current_allocation = 0.0
                current_risk = 0.0
                
                if self.position_manager:
                    strategy_positions = [
                        p for p in self.position_manager.internal_positions.values()
                        if p.get('strategy_id') == strategy_id
                    ]
                    active_positions = len(strategy_positions)
                    
                    # Calculate current allocation
                    total_value = 0.0
                    for position in strategy_positions:
                        qty = float(position.get('quantity', 0))
                        price = float(position.get('current_price', position.get('entry_price', 0)))
                        total_value += qty * price
                    
                    current_allocation = total_value
                    
                    # Calculate current risk if stop loss info available
                    if self.pnl_calculator:
                        current_risk = self.pnl_calculator.calculate_portfolio_risk(strategy_id=strategy_id)
                
                # Get max positions
                max_positions = self.strategy_max_positions.get(strategy_id, 5)  # Default to 5
                
                result = {
                    'strategy_id': strategy_id,
                    'allocation_pct': allocation_pct,
                    'allocated_capital': allocated_capital,
                    'current_allocation': current_allocation,
                    'available_capital': allocated_capital - current_allocation,
                    'active_positions': active_positions,
                    'max_positions': max_positions,
                    'current_risk': current_risk,
                    'risk_metrics': risk_metrics,
                    'updated_at': datetime.now().isoformat()
                }
                
                return result
                
            except Exception as e:
                logger.error(f"Error getting strategy allocation: {str(e)}")
                return {
                    'strategy_id': strategy_id,
                    'allocation_pct': 0.0,
                    'error': str(e)
                }
    
    def get_all_allocations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get allocation details for all strategies.
        
        Returns:
            Dict of strategy_id -> allocation details
        """
        with self._lock:
            try:
                result = {}
                for strategy_id in self.strategy_allocations.keys():
                    result[strategy_id] = self.get_strategy_allocation(strategy_id)
                
                return result
                
            except Exception as e:
                logger.error(f"Error getting all allocations: {str(e)}")
                return {}
    
    def set_allocation_method(self, method: str) -> bool:
        """
        Set the allocation method to use.
        
        Args:
            method: Allocation method from AllocationMethod
            
        Returns:
            bool: Success status
        """
        try:
            # Validate method
            valid_methods = [
                AllocationMethod.EQUAL,
                AllocationMethod.PERFORMANCE,
                AllocationMethod.RISK_PARITY,
                AllocationMethod.KELLY,
                AllocationMethod.SHARPE,
                AllocationMethod.CUSTOM
            ]
            
            if method not in valid_methods:
                logger.warning(f"Invalid allocation method: {method}")
                return False
            
            # Set method
            self.allocation_method = method
            logger.info(f"Set allocation method to {method}")
            
            # Trigger rebalance
            self.rebalance_pending = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting allocation method: {str(e)}")
            return False
    
    def get_allocation_metrics(self) -> Dict[str, Any]:
        """
        Get current allocation performance metrics for dashboard display.
        
        Returns:
            Dict with allocation metrics
        """
        with self._lock:
            try:
                # Get basic metrics
                result = {
                    'total_capital': self.total_capital,
                    'allocation_method': self.allocation_method,
                    'last_rebalance': self.last_rebalance_time.isoformat() if self.last_rebalance_time else None,
                    'strategy_count': len(self.strategy_allocations),
                    'allocations': self.strategy_allocations.copy(),
                    'max_positions': self.strategy_max_positions.copy(),
                    'updated_at': datetime.now().isoformat()
                }
                
                # Calculate allocation performance metrics if performance metrics available
                if hasattr(self.performance_metrics, 'get_portfolio_metrics'):
                    portfolio_metrics = self.performance_metrics.get_portfolio_metrics()
                    result.update(portfolio_metrics)
                
                # Calculate capital utilization
                allocated_capital = 0.0
                if self.position_manager:
                    positions = self.position_manager.internal_positions.values()
                    for position in positions:
                        qty = float(position.get('quantity', 0))
                        price = float(position.get('current_price', position.get('entry_price', 0)))
                        allocated_capital += qty * price
                
                result['allocated_capital'] = allocated_capital
                result['capital_utilization_pct'] = (allocated_capital / self.total_capital * 100.0) if self.total_capital > 0 else 0.0
                
                return result
                
            except Exception as e:
                logger.error(f"Error getting allocation metrics: {str(e)}")
                return {
                    'error': str(e),
                    'updated_at': datetime.now().isoformat()
                }
    
    def get_capital_efficiency(self) -> float:
        """
        Calculate capital efficiency ratio.
        Measures how efficiently capital is being utilized.
        
        Returns:
            float: Capital efficiency ratio (0.0-1.0)
        """
        try:
            # Calculate based on profit per unit of allocated capital
            if not hasattr(self.performance_metrics, 'get_total_realized_pnl'):
                return 0.0
            
            # Get total profit
            total_profit = self.performance_metrics.get_total_realized_pnl()
            
            # Get average allocated capital
            avg_allocated_capital = self.performance_metrics.get_average_allocated_capital()
            if avg_allocated_capital <= 0:
                return 0.0
            
            # Calculate return on allocated capital
            return_on_capital = total_profit / avg_allocated_capital
            
            # Normalize to 0-1 range with reasonable scaling
            # A 20% return is considered very good
            efficiency = min(1.0, max(0.0, return_on_capital / 0.2))
            
            return efficiency
            
        except Exception as e:
            logger.error(f"Error calculating capital efficiency: {str(e)}")
            return 0.0
