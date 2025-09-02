#!/usr/bin/env python3
"""
Optimization Integration

This module integrates the optimization system with the Strategy Lifecycle Manager
and Autonomous Engine, providing a seamless connection between optimization,
strategy lifecycle management, and the event-driven architecture.

Classes:
    OptimizationIntegration: Integrates optimization with other system components
"""

import os
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union

# Import event system
from trading_bot.event_system import EventBus, EventManager, Event, EventType

# Import optimization components
from trading_bot.autonomous.optimization_jobs import (
    OptimizationJob, OptimizationStatus, get_job_store,
    create_optimization_job, PriorityCalculator
)
from trading_bot.autonomous.optimization_scheduler import (
    get_optimization_scheduler, OptimizationEventType
)
from trading_bot.autonomous.optimization_history import (
    get_optimization_history_tracker, OptimizationEffectiveness
)

# Import strategy lifecycle components
from trading_bot.autonomous.strategy_lifecycle_manager import (
    get_lifecycle_manager, StrategyStatus
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationIntegration:
    """
    Integrates optimization with strategy lifecycle and autonomous engine.
    
    This class is responsible for:
    - Connecting optimization events with strategy lifecycle events
    - Managing the promotion of optimized strategies
    - Integrating with the autonomous engine for execution
    """
    
    def __init__(self):
        """Initialize the optimization integration."""
        # Get component instances
        self.event_bus = EventBus()
        self.job_store = get_job_store()
        self.scheduler = get_optimization_scheduler()
        self.history_tracker = get_optimization_history_tracker()
        self.lifecycle_manager = get_lifecycle_manager()
        
        # Initialize state
        self.running = False
        self.lock = threading.RLock()
        
        # Register event handlers
        self._register_event_handlers()
        
        # Register optimization execution callback
        self.scheduler.register_execution_callback(
            self._execute_optimization,
            name="default"
        )
    
    def _register_event_handlers(self) -> None:
        """Register handlers for relevant events."""
        # Optimization events
        self.event_bus.register(
            OptimizationEventType.OPTIMIZATION_COMPLETED,
            self._handle_optimization_completed
        )
        
        # Strategy lifecycle events
        self.event_bus.register(
            EventType.STRATEGY_VERSION_CREATED,
            self._handle_strategy_version_created
        )
        
        self.event_bus.register(
            EventType.STRATEGY_VERSION_PROMOTED,
            self._handle_strategy_version_promoted
        )
        
        self.event_bus.register(
            EventType.STRATEGY_PERFORMANCE_UPDATED,
            self._handle_strategy_performance_updated
        )
    
    def _handle_strategy_version_created(self, event: Event) -> None:
        """
        Handle strategy version creation events.
        
        Args:
            event: Version creation event
        """
        data = event.data
        if not data:
            return
            
        strategy_id = data.get("strategy_id")
        version_id = data.get("version_id")
        
        if not strategy_id or not version_id:
            return
            
        # Schedule initial optimization for future validation
        # This helps establish a baseline for new strategy versions
        scheduled_time = datetime.now() + timedelta(days=1)
        
        # Get performance metrics if available
        performance = data.get("performance_metrics", {})
        if not performance:
            # Try to get from lifecycle manager
            version = self.lifecycle_manager.get_version(strategy_id, version_id)
            if version:
                performance = version.performance or {}
        
        # Only schedule if eligible (non-empty performance metrics)
        if performance:
            self.scheduler.schedule_optimization(
                strategy_id=strategy_id,
                strategy_type=data.get("strategy_type", "unknown"),
                universe=data.get("universe", "unknown"),
                version_id=version_id,
                performance_metrics=performance,
                reason="new_version_baseline",
                scheduled_time=scheduled_time
            )
    
    def _handle_strategy_version_promoted(self, event: Event) -> None:
        """
        Handle strategy version promotion events.
        
        Args:
            event: Version promotion event
        """
        data = event.data
        if not data:
            return
            
        strategy_id = data.get("strategy_id")
        version_id = data.get("version_id")
        
        if not strategy_id or not version_id:
            return
            
        # Schedule optimization to further improve promoted strategy
        # Get the version to check if it was a result of optimization
        version = self.lifecycle_manager.get_version(strategy_id, version_id)
        if not version:
            return
            
        # If this version was a result of optimization, don't schedule immediately
        if version.metadata and version.metadata.get("from_optimization"):
            return
            
        # Get performance metrics
        performance = version.performance or {}
        
        # Schedule optimization for this promoted version
        scheduled_time = datetime.now() + timedelta(hours=12)
        
        self.scheduler.schedule_optimization(
            strategy_id=strategy_id,
            strategy_type=data.get("strategy_type", "unknown"),
            universe=data.get("universe", "unknown"),
            version_id=version_id,
            performance_metrics=performance,
            reason="post_promotion_improvement",
            scheduled_time=scheduled_time
        )
    
    def _handle_strategy_performance_updated(self, event: Event) -> None:
        """
        Handle strategy performance update events.
        
        Args:
            event: Performance update event
        """
        data = event.data
        if not data:
            return
            
        strategy_id = data.get("strategy_id")
        version_id = data.get("version_id")
        performance = data.get("performance", {})
        
        if not strategy_id or not version_id or not performance:
            return
            
        # Check if performance deterioration warrants optimization
        should_optimize = False
        
        # Get previous performance
        version = self.lifecycle_manager.get_version(strategy_id, version_id)
        if not version or not version.performance:
            return
            
        old_performance = version.performance
        
        # Check for significant deterioration
        if "sharpe_ratio" in performance and "sharpe_ratio" in old_performance:
            old_sharpe = old_performance["sharpe_ratio"]
            new_sharpe = performance["sharpe_ratio"]
            
            # If sharpe ratio dropped by more than 15%
            if old_sharpe > 0 and new_sharpe < old_sharpe * 0.85:
                should_optimize = True
        
        if "max_drawdown" in performance and "max_drawdown" in old_performance:
            old_dd = abs(old_performance["max_drawdown"])
            new_dd = abs(performance["max_drawdown"])
            
            # If drawdown increased by more than 20%
            if new_dd > old_dd * 1.2:
                should_optimize = True
        
        if should_optimize:
            # Get strategy metadata
            strategy_metadata = version.metadata or {}
            strategy_type = strategy_metadata.get("strategy_type", "unknown")
            universe = strategy_metadata.get("universe", "unknown")
            
            # Schedule optimization with high priority
            self.scheduler.schedule_optimization(
                strategy_id=strategy_id,
                strategy_type=strategy_type,
                universe=universe,
                version_id=version_id,
                performance_metrics=performance,
                reason="performance_deterioration",
                priority=80  # High priority
            )
    
    def _handle_optimization_completed(self, event: Event) -> None:
        """
        Handle optimization completion events.
        
        Args:
            event: Optimization completion event
        """
        data = event.data
        if not data:
            return
            
        job_id = data.get("job_id")
        if not job_id:
            return
            
        # Get the optimization job
        job = self.job_store.get_job(job_id)
        if not job or job.status != OptimizationStatus.COMPLETED:
            return
            
        # Get results
        results = job.results
        if not results:
            return
            
        strategy_id = job.strategy_id
        old_version_id = job.version_id
        new_parameters = results.get("parameters")
        performance_improvement = results.get("performance_improvement", {})
        
        if not new_parameters or not performance_improvement:
            return
            
        # Get the old version
        old_version = self.lifecycle_manager.get_version(strategy_id, old_version_id)
        if not old_version:
            return
            
        # Create a new version with optimized parameters
        new_version_id = f"{old_version_id}.opt.{datetime.now().strftime('%Y%m%d%H%M')}"
        
        # Combine old parameters with optimized ones
        old_parameters = old_version.parameters or {}
        combined_parameters = {**old_parameters, **new_parameters}
        
        # Create metadata for new version
        metadata = {
            **(old_version.metadata or {}),
            "from_optimization": True,
            "optimization_job_id": job_id,
            "parent_version": old_version_id,
            "strategy_type": old_version.metadata.get("strategy_type", "unknown"),
            "universe": old_version.metadata.get("universe", "unknown")
        }
        
        # Create new version
        try:
            new_version = self.lifecycle_manager.create_version(
                strategy_id=strategy_id,
                version_id=new_version_id,
                parameters=combined_parameters,
                performance=results.get("new_performance", {}),
                status=StrategyStatus.CANDIDATE,
                metadata=metadata
            )
            
            logger.info(
                f"Created optimized version {new_version_id} for {strategy_id} "
                f"from optimization job {job_id}"
            )
            
            # Record optimization history
            self.history_tracker.add_entry(
                strategy_id=strategy_id,
                old_version_id=old_version_id,
                new_version_id=new_version_id,
                optimization_parameters=job.parameters,
                old_metrics=old_version.performance or {},
                new_metrics=results.get("new_performance", {}),
                job_id=job_id
            )
            
            # Check if new version should be promoted
            self._evaluate_for_promotion(
                strategy_id, 
                old_version_id, 
                new_version_id, 
                performance_improvement
            )
            
        except Exception as e:
            logger.error(
                f"Error creating optimized version for {strategy_id}: {str(e)}"
            )
    
    def _evaluate_for_promotion(
        self,
        strategy_id: str,
        old_version_id: str,
        new_version_id: str,
        improvement: Dict[str, float]
    ) -> None:
        """
        Evaluate if optimized version should be promoted.
        
        Args:
            strategy_id: Strategy identifier
            old_version_id: Previous version ID
            new_version_id: New optimized version ID
            improvement: Performance improvement metrics
        """
        # Get versions
        old_version = self.lifecycle_manager.get_version(strategy_id, old_version_id)
        new_version = self.lifecycle_manager.get_version(strategy_id, new_version_id)
        
        if not old_version or not new_version:
            return
            
        # Only auto-promote if old version is ACTIVE or CANDIDATE
        if old_version.status not in (StrategyStatus.ACTIVE, StrategyStatus.CANDIDATE):
            return
            
        # Check if improvement is significant
        significant_improvement = False
        
        # Check Sharpe ratio improvement
        if "sharpe_ratio" in improvement and improvement["sharpe_ratio"] >= 0.15:
            significant_improvement = True
            
        # Check drawdown improvement
        if "max_drawdown" in improvement and improvement["max_drawdown"] >= 0.15:
            significant_improvement = True
            
        # Check win rate improvement
        if "win_rate" in improvement and improvement["win_rate"] >= 0.10:
            significant_improvement = True
        
        if significant_improvement:
            # Promote new version to same status as old version
            try:
                self.lifecycle_manager.update_status(
                    strategy_id=strategy_id,
                    version_id=new_version_id,
                    status=old_version.status
                )
                
                logger.info(
                    f"Auto-promoted optimized version {new_version_id} for {strategy_id} "
                    f"to {old_version.status} status due to significant improvement"
                )
                
                # If old version was ACTIVE, retire it
                if old_version.status == StrategyStatus.ACTIVE:
                    self.lifecycle_manager.update_status(
                        strategy_id=strategy_id,
                        version_id=old_version_id,
                        status=StrategyStatus.RETIRED
                    )
                    
                    logger.info(
                        f"Retired previous version {old_version_id} of {strategy_id} "
                        f"after promoting optimized version"
                    )
            except Exception as e:
                logger.error(
                    f"Error promoting optimized version {new_version_id} for {strategy_id}: {str(e)}"
                )
    
    def _execute_optimization(self, job: OptimizationJob) -> None:
        """
        Execute an optimization job.
        
        This method is called by the scheduler when a job is due.
        
        Args:
            job: Optimization job to execute
        """
        try:
            strategy_id = job.strategy_id
            version_id = job.version_id
            
            logger.info(f"Executing optimization job {job.job_id} for {strategy_id} version {version_id}")
            
            # Get the strategy version
            version = self.lifecycle_manager.get_version(strategy_id, version_id)
            if not version or not version.parameters:
                raise ValueError(f"Strategy version {version_id} not found or has no parameters")
            
            # Extract optimization parameters
            method = job.parameters.get("method", "bayesian")
            iterations = job.parameters.get("iterations", 100)
            target_metric = job.parameters.get("target_metric", "sharpe_ratio")
            
            # This is where we would connect to the actual optimizer
            # For now, we'll simulate optimization with a simple example
            
            # Simulate optimized parameters (in real implementation, this would be from optimizer)
            # In a real implementation, this would call into an optimizer component
            old_params = version.parameters
            old_performance = version.performance or {}
            
            # Simulate optimization process
            # (actual integration would call the enhanced_optimizer here)
            new_params = self._simulate_optimization(old_params)
            new_performance = self._simulate_performance_improvement(old_performance)
            
            # Calculate performance improvement
            improvement = {}
            for metric, new_value in new_performance.items():
                if metric in old_performance:
                    old_value = old_performance[metric]
                    if old_value != 0:
                        improvement[metric] = (new_value - old_value) / abs(old_value)
            
            # Create results
            results = {
                "parameters": new_params,
                "old_performance": old_performance,
                "new_performance": new_performance,
                "performance_improvement": improvement,
                "method": method,
                "iterations": iterations,
                "target_metric": target_metric,
                "optimization_time": (datetime.now() - job.last_updated).total_seconds()
            }
            
            # Update job with results
            job.set_results(results)
            self.job_store.update_job(job)
            
            # Emit event
            self.event_bus.emit(
                OptimizationEventType.OPTIMIZATION_COMPLETED,
                {
                    "job_id": job.job_id,
                    "strategy_id": strategy_id,
                    "version_id": version_id,
                    "results": results
                }
            )
            
            logger.info(
                f"Completed optimization job {job.job_id} for {strategy_id} "
                f"with {len(new_params)} optimized parameters"
            )
            
        except Exception as e:
            logger.error(f"Error executing optimization job {job.job_id}: {str(e)}")
            
            # Update job with error
            job.set_error(str(e))
            self.job_store.update_job(job)
            
            # Emit event
            self.event_bus.emit(
                OptimizationEventType.OPTIMIZATION_FAILED,
                {
                    "job_id": job.job_id,
                    "strategy_id": job.strategy_id,
                    "error": str(e)
                }
            )
    
    def _simulate_optimization(self, old_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate optimization process for testing.
        
        Args:
            old_params: Original parameters
            
        Returns:
            Optimized parameters
        """
        # In a real implementation, this would call the optimizer
        # For now, just modify a few parameters slightly
        import random
        
        new_params = {}
        
        for key, value in old_params.items():
            # Only modify numeric parameters
            if isinstance(value, (int, float)):
                # Adjust by up to ±20%
                if isinstance(value, int):
                    new_params[key] = max(1, int(value * (1 + (random.random() * 0.4 - 0.2))))
                else:
                    new_params[key] = value * (1 + (random.random() * 0.4 - 0.2))
        
        return new_params
    
    def _simulate_performance_improvement(self, old_performance: Dict[str, float]) -> Dict[str, float]:
        """
        Simulate performance improvement for testing.
        
        Args:
            old_performance: Original performance metrics
            
        Returns:
            Improved performance metrics
        """
        # In a real implementation, this would be the result of backtesting with new parameters
        import random
        
        new_performance = {}
        
        for key, value in old_performance.items():
            if key == "max_drawdown":
                # For drawdown (negative value), improvement means less negative
                if value < 0:
                    new_performance[key] = value * (1 - (random.random() * 0.2))
                else:
                    new_performance[key] = value * (1 - (random.random() * 0.2))
            elif key in ("volatility"):
                # For volatility, lower is better
                new_performance[key] = value * (1 - (random.random() * 0.15))
            else:
                # For other metrics (sharpe, sortino, win_rate, etc.), higher is better
                new_performance[key] = value * (1 + (random.random() * 0.25))
        
        return new_performance
    
    def start(self) -> None:
        """Start the optimization integration."""
        with self.lock:
            if self.running:
                logger.warning("Optimization integration is already running")
                return
                
            self.running = True
            
        # Start the scheduler
        self.scheduler.start()
        
        logger.info("Optimization integration started")
    
    def stop(self) -> None:
        """Stop the optimization integration."""
        with self.lock:
            self.running = False
            
        # Stop the scheduler
        self.scheduler.stop()
        
        logger.info("Optimization integration stopped")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """
        Get the current status of the optimization integration.
        
        Returns:
            Status information
        """
        scheduler_status = self.scheduler.get_scheduler_status()
        
        # Get effectiveness stats
        effectiveness_stats = self.history_tracker.get_optimization_effectiveness_stats()
        
        return {
            "running": self.running,
            "scheduler": scheduler_status,
            "history": {
                "total_optimizations": effectiveness_stats["total_optimizations"],
                "effectiveness": effectiveness_stats["effectiveness_percentages"]
            }
        }


# Singleton instance
_integration = None


def get_optimization_integration() -> OptimizationIntegration:
    """
    Get the singleton instance of OptimizationIntegration.
    
    Returns:
        OptimizationIntegration instance
    """
    global _integration
    
    if _integration is None:
        _integration = OptimizationIntegration()
    
    return _integration


if __name__ == "__main__":
    # Start integration when run directly
    integration = get_optimization_integration()
    integration.start()
    
    try:
        import time
        while True:
            time.sleep(60)
            status = integration.get_integration_status()
            print(f"Integration status: {status}")
    except KeyboardInterrupt:
        print("Stopping integration...")
        integration.stop()
