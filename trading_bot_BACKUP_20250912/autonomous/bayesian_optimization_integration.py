#!/usr/bin/env python3
"""
Bayesian Optimization Integration

This module integrates the Bayesian Optimizer with our existing optimization framework.
It provides the connection between our job scheduling system, strategy lifecycle,
and the advanced Bayesian parameter tuning algorithms.

The module follows our established patterns: event-driven architecture, singleton
access, and persistence of optimization state.
"""

import os
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
import time
import numpy as np

# Import optimization components
from trading_bot.autonomous.bayesian_optimizer import (
    BayesianOptimizer, ParameterSpace, ParameterType,
    OptimizationDirection, AcquisitionFunction
)
from trading_bot.autonomous.optimization_jobs import (
    OptimizationJob, OptimizationStatus, OptimizationMethod
)
from trading_bot.autonomous.optimization_scheduler import (
    OptimizationEventType
)

# Import event system
from trading_bot.event_system import EventBus, Event, EventType

# Import strategy lifecycle
from trading_bot.autonomous.strategy_lifecycle_manager import (
    get_lifecycle_manager, StrategyStatus
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BayesianOptimizationManager:
    """
    Manages Bayesian optimization for strategy parameters.
    
    This class integrates the BayesianOptimizer with our optimization framework,
    handling job execution, result tracking, and event integration.
    """
    
    def __init__(self):
        """Initialize the Bayesian optimization manager."""
        self.event_bus = EventBus()
        self.lifecycle_manager = get_lifecycle_manager()
        
        # Optimization state
        self.ongoing_optimizations = {}  # job_id -> optimizer
        self.optimization_results = {}   # job_id -> results
        self.lock = threading.RLock()
        
        # Register for events
        self._register_event_handlers()
        
        # Storage path for persistence
        self.storage_path = os.path.join(
            os.path.expanduser("~"), ".trading_bot", "optimization", "bayesian"
        )
        os.makedirs(self.storage_path, exist_ok=True)
    
    def _register_event_handlers(self):
        """Register for relevant events."""
        # Listen for optimization job events
        self.event_bus.register(
            OptimizationEventType.OPTIMIZATION_STARTED,
            self._handle_optimization_started
        )
        
    def _handle_optimization_started(self, event: Event):
        """
        Handle optimization job start events.
        
        Args:
            event: The optimization started event
        """
        data = event.data
        if not data:
            return
            
        job_id = data.get("job_id")
        strategy_id = data.get("strategy_id")
        version_id = data.get("version_id")
        parameters = data.get("parameters", {})
        
        if not job_id or not strategy_id or not version_id:
            return
            
        # Check if parameters specify Bayesian method
        method = parameters.get("method")
        if method != OptimizationMethod.BAYESIAN.value:
            # Not a Bayesian optimization job
            return
            
        logger.info(
            f"Starting Bayesian optimization job {job_id} for "
            f"strategy {strategy_id} version {version_id}"
        )
        
        # Start optimization in background thread
        threading.Thread(
            target=self._run_optimization,
            args=(job_id, strategy_id, version_id, parameters),
            daemon=True,
            name=f"BayesianOpt-{job_id[:8]}"
        ).start()
    
    def _run_optimization(
        self,
        job_id: str,
        strategy_id: str,
        version_id: str,
        parameters: Dict[str, Any]
    ):
        """
        Run a Bayesian optimization job.
        
        Args:
            job_id: Optimization job ID
            strategy_id: Strategy to optimize
            version_id: Version to optimize
            parameters: Optimization parameters
        """
        try:
            # Get strategy version from lifecycle manager
            version = self.lifecycle_manager.get_version(strategy_id, version_id)
            if not version:
                raise ValueError(f"Strategy version {version_id} not found")
                
            # Extract strategy parameters and parameter space
            strategy_params = version.parameters
            if not strategy_params:
                raise ValueError(f"Strategy version {version_id} has no parameters")
                
            # Get parameter space definition
            param_space = self._get_parameter_space(strategy_id, strategy_params)
            
            # Create optimizer
            iterations = parameters.get("iterations", 100)
            target_metric = parameters.get("target_metric", "sharpe_ratio")
            exploration = parameters.get("exploration", 0.1)
            
            # Determine optimization direction (maximize most metrics except drawdown, volatility)
            direction = OptimizationDirection.MAXIMIZE
            if target_metric in ("max_drawdown", "volatility", "var", "cvar"):
                direction = OptimizationDirection.MINIMIZE
            
            optimizer = BayesianOptimizer(
                parameter_space=param_space,
                direction=direction,
                acquisition_function=AcquisitionFunction.EI,
                n_initial_points=min(5, max(1, iterations // 10)),
                exploration_weight=exploration
            )
            
            # Store optimizer
            with self.lock:
                self.ongoing_optimizations[job_id] = optimizer
            
            # Create objective function that evaluates parameters
            objective_fn = self._create_objective_function(
                strategy_id, version_id, target_metric
            )
            
            # Create progress callback
            def progress_callback(params, value, iteration):
                self._emit_progress_event(
                    job_id, strategy_id, version_id, params, value, 
                    iteration, iterations, target_metric
                )
            
            # Run optimization
            start_time = time.time()
            
            results = optimizer.optimize(
                objective_function=objective_fn,
                n_iterations=iterations,
                callback=progress_callback
            )
            
            # Store results
            with self.lock:
                self.optimization_results[job_id] = results
                if job_id in self.ongoing_optimizations:
                    del self.ongoing_optimizations[job_id]
            
            # Persist results
            self._save_optimization_results(job_id, results)
            
            # Create result summary with old and new performance
            result_summary = self._create_result_summary(
                strategy_id, version_id, results, target_metric
            )
            
            # Emit completion event
            self.event_bus.emit(
                OptimizationEventType.OPTIMIZATION_COMPLETED,
                {
                    "job_id": job_id,
                    "strategy_id": strategy_id,
                    "version_id": version_id,
                    "results": result_summary
                }
            )
            
            logger.info(
                f"Completed Bayesian optimization job {job_id} "
                f"for strategy {strategy_id} in {time.time() - start_time:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"Error in Bayesian optimization job {job_id}: {str(e)}")
            
            # Emit failure event
            self.event_bus.emit(
                OptimizationEventType.OPTIMIZATION_FAILED,
                {
                    "job_id": job_id,
                    "strategy_id": strategy_id,
                    "error": str(e)
                }
            )
            
            # Clean up
            with self.lock:
                if job_id in self.ongoing_optimizations:
                    del self.ongoing_optimizations[job_id]
    
    def _get_parameter_space(
        self,
        strategy_id: str,
        strategy_params: Dict[str, Any]
    ) -> ParameterSpace:
        """
        Create parameter space from strategy parameters.
        
        Args:
            strategy_id: Strategy ID
            strategy_params: Current strategy parameters
            
        Returns:
            Parameter space for optimization
        """
        # Try to load parameter space definition if available
        param_space_path = os.path.join(
            self.storage_path, f"{strategy_id}_param_space.json"
        )
        
        if os.path.exists(param_space_path):
            try:
                with open(param_space_path, 'r') as f:
                    param_space_dict = json.load(f)
                
                return ParameterSpace.from_dict(param_space_dict)
            except Exception as e:
                logger.warning(
                    f"Error loading parameter space for {strategy_id}: {str(e)}. "
                    f"Creating new parameter space."
                )
        
        # Create new parameter space
        # This is a simplified approach - in a real system, we would have
        # strategy-specific parameter space definitions with proper bounds
        
        space = ParameterSpace()
        
        for name, value in strategy_params.items():
            # Skip non-optimizable parameters
            if name.startswith('_') or name in ('id', 'name', 'type', 'universe'):
                continue
                
            if isinstance(value, bool):
                space.add_boolean_parameter(name, default=value)
                
            elif isinstance(value, int):
                # For integer params, create reasonable bounds
                if name.endswith('_period') or name.endswith('_window') or name.endswith('_size'):
                    # Time periods usually between 2 and 100
                    lower = max(2, int(value * 0.5))
                    upper = int(value * 2.0)
                    space.add_integer_parameter(name, lower, upper, default=value)
                elif name.endswith('_days'):
                    # Day parameters usually between 1 and 90
                    lower = max(1, int(value * 0.7))
                    upper = min(90, int(value * 1.5))
                    space.add_integer_parameter(name, lower, upper, default=value)
                else:
                    # Generic integer
                    lower = max(1, int(value * 0.7))
                    upper = int(value * 1.3)
                    space.add_integer_parameter(name, lower, upper, default=value)
                    
            elif isinstance(value, float):
                # For float params, create reasonable bounds
                if name.endswith('_pct') or name.endswith('_ratio') or name.endswith('_threshold'):
                    # Percentage parameters usually between 0 and 1
                    if 0 <= value <= 1:
                        lower = max(0.0, value - 0.2)
                        upper = min(1.0, value + 0.2)
                        space.add_real_parameter(name, lower, upper, default=value)
                    else:
                        # Larger percentages (e.g., 100% = 1.0)
                        lower = max(0.0, value * 0.7)
                        upper = value * 1.3
                        space.add_real_parameter(name, lower, upper, default=value)
                elif value > 100:
                    # Large values might benefit from log scale
                    lower = max(1.0, value * 0.5)
                    upper = value * 2.0
                    space.add_real_parameter(name, lower, upper, default=value, log_scale=True)
                else:
                    # Generic float
                    lower = value * 0.7
                    upper = value * 1.3
                    space.add_real_parameter(name, lower, upper, default=value)
            
            elif isinstance(value, str) and hasattr(value, '__iter__'):
                # For string params that have limited options
                categories = list(set([value]))  # Start with current value
                
                # Could expand with known options for specific params
                if name.endswith('_method'):
                    categories.extend(['sma', 'ema', 'wma', 'linear'])
                elif name.endswith('_type'):
                    categories.extend(['simple', 'exponential', 'adaptive'])
                
                # Only optimize if we have multiple options
                if len(categories) > 1:
                    space.add_categorical_parameter(name, categories, default=value)
        
        # Save parameter space for future use
        try:
            with open(param_space_path, 'w') as f:
                json.dump(space.get_parameters_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving parameter space: {str(e)}")
        
        return space
    
    def _create_objective_function(
        self,
        strategy_id: str,
        version_id: str,
        target_metric: str
    ) -> Callable[[Dict[str, Any]], float]:
        """
        Create objective function for optimization.
        
        This function will evaluate a set of parameters by:
        1. Creating a temporary strategy version with those parameters
        2. Backtesting the strategy
        3. Returning the specified performance metric
        
        Args:
            strategy_id: Strategy to optimize
            version_id: Base version
            target_metric: Metric to optimize
            
        Returns:
            Objective function for optimizer
        """
        # Get base version for reference
        base_version = self.lifecycle_manager.get_version(strategy_id, version_id)
        if not base_version:
            raise ValueError(f"Strategy version {version_id} not found")
            
        # Get base parameters and metadata
        base_params = base_version.parameters.copy() if base_version.parameters else {}
        base_metadata = base_version.metadata.copy() if base_version.metadata else {}
        
        # Reference to self for inner function
        self_ref = self
        
        def objective(parameters: Dict[str, Any]) -> float:
            """
            Objective function that evaluates parameters.
            
            Args:
                parameters: Strategy parameters to evaluate
                
            Returns:
                Performance metric value
            """
            # Create a unique ID for this evaluation
            eval_id = f"eval_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hash(str(parameters)) % 10000}"
            
            try:
                # Combine base parameters with new parameters
                eval_params = base_params.copy()
                eval_params.update(parameters)
                
                # Create temporary version for evaluation
                temp_version_id = f"{version_id}.tmp.{eval_id}"
                
                # Create metadata
                eval_metadata = base_metadata.copy()
                eval_metadata.update({
                    "evaluation_id": eval_id,
                    "base_version": version_id,
                    "optimization_target": target_metric,
                    "temporary": True
                })
                
                # Create temporary version in lifecycle manager
                temp_version = self_ref.lifecycle_manager.create_version(
                    strategy_id=strategy_id,
                    version_id=temp_version_id,
                    parameters=eval_params,
                    performance={},  # Will be filled by evaluation
                    status=StrategyStatus.DEVELOPMENT,
                    metadata=eval_metadata
                )
                
                # Backtest this version
                # In a real implementation, this would call into the backtesting engine
                performance = self_ref._simulate_backtest(strategy_id, eval_params)
                
                # Update version with performance
                temp_version.performance = performance
                
                # Extract target metric
                if target_metric not in performance:
                    logger.warning(f"Target metric {target_metric} not found in performance results")
                    return float('-inf')  # Return worst possible value
                
                metric_value = performance[target_metric]
                
                # Ensure the value is usable for optimization
                if isinstance(metric_value, (int, float)):
                    if np.isnan(metric_value) or np.isinf(metric_value):
                        return float('-inf')
                    return float(metric_value)
                else:
                    logger.warning(f"Non-numeric metric value: {metric_value}")
                    return float('-inf')
                    
            except Exception as e:
                logger.error(f"Error evaluating parameters: {str(e)}")
                return float('-inf')  # Return worst possible value
        
        return objective
    
    def _simulate_backtest(
        self,
        strategy_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Simulate backtesting a strategy with given parameters.
        
        In a real implementation, this would call the actual backtesting engine.
        For now, it returns simulated performance metrics.
        
        Args:
            strategy_id: Strategy to backtest
            parameters: Strategy parameters
            
        Returns:
            Performance metrics
        """
        # This is a simulation for testing
        # In a real system, this would call the actual backtesting engine
        
        # Base performance - slightly randomized
        import random
        
        # Reference performance - each strategy has different baseline
        if 'iron_condor' in strategy_id:
            base_sharpe = 1.2
            base_drawdown = -0.12
            base_win_rate = 0.65
        elif 'strangle' in strategy_id:
            base_sharpe = 1.0
            base_drawdown = -0.15
            base_win_rate = 0.60
        else:
            base_sharpe = 0.9
            base_drawdown = -0.18
            base_win_rate = 0.55
        
        # Calculate "fitness" based on parameter combinations
        # This simulates how some parameter combinations work better than others
        param_influence = 0
        
        # Analyze key parameters that might affect performance
        for name, value in parameters.items():
            if name.endswith('_period') or name.endswith('_window'):
                # Prefer moderate periods (not too short, not too long)
                if isinstance(value, (int, float)):
                    # Periods around 20-40 tend to be better
                    param_influence += 0.02 * (1 - abs(30 - value) / 30)
            
            elif name.endswith('_threshold') or name.endswith('_pct'):
                # Thresholds have sweet spots
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    # Values around 0.4-0.6 might be better
                    param_influence += 0.02 * (1 - abs(0.5 - value) / 0.5)
        
        # Add randomness to simulate noise in backtesting
        noise = random.uniform(-0.1, 0.1)
        
        # Calculate performance metrics with parameter influence
        return {
            "sharpe_ratio": base_sharpe * (1 + param_influence + noise),
            "sortino_ratio": base_sharpe * 1.2 * (1 + param_influence + noise),
            "max_drawdown": base_drawdown * (1 - param_influence - noise * 0.5),
            "win_rate": base_win_rate * (1 + param_influence * 0.5 + noise * 0.3),
            "profit_factor": 1.4 * (1 + param_influence + noise),
            "volatility": 0.12 * (1 - param_influence * 0.5 - noise * 0.2),
            "annualized_return": 0.18 * (1 + param_influence + noise),
            "avg_trade_duration": 5.2 + random.uniform(-0.5, 0.5),
            "max_consecutive_losses": 3 + random.randint(-1, 2)
        }
    
    def _emit_progress_event(
        self,
        job_id: str,
        strategy_id: str,
        version_id: str,
        parameters: Dict[str, Any],
        value: float,
        iteration: int,
        total_iterations: int,
        target_metric: str
    ):
        """
        Emit progress event for optimization monitoring.
        
        Args:
            job_id: Optimization job ID
            strategy_id: Strategy being optimized
            version_id: Version being optimized
            parameters: Current parameters being evaluated
            value: Current objective value
            iteration: Current iteration
            total_iterations: Total iterations planned
            target_metric: Metric being optimized
        """
        self.event_bus.emit(
            "optimization_progress",
            {
                "job_id": job_id,
                "strategy_id": strategy_id,
                "version_id": version_id,
                "iteration": iteration + 1,  # 1-based for display
                "total_iterations": total_iterations,
                "current_parameters": parameters,
                "current_value": value,
                "target_metric": target_metric,
                "progress_pct": (iteration + 1) / total_iterations * 100,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def _create_result_summary(
        self,
        strategy_id: str,
        version_id: str,
        results: Dict[str, Any],
        target_metric: str
    ) -> Dict[str, Any]:
        """
        Create a summary of optimization results.
        
        Args:
            strategy_id: Strategy that was optimized
            version_id: Version that was optimized
            results: Raw optimization results
            target_metric: Metric that was optimized
            
        Returns:
            Results summary for event emission
        """
        # Get original version for comparison
        original_version = self.lifecycle_manager.get_version(strategy_id, version_id)
        
        if not original_version:
            return results
            
        # Get original performance
        original_performance = original_version.parameters or {}
        
        # Get best parameters and performance
        best_parameters = results.get('best_parameters', {})
        best_value = results.get('best_value', 0.0)
        
        # Backtest best parameters to get full performance metrics
        best_performance = self._simulate_backtest(strategy_id, best_parameters)
        
        # Calculate improvement
        improvement = {}
        for metric, new_value in best_performance.items():
            if metric in original_performance:
                old_value = original_performance[metric]
                if old_value != 0:
                    improvement[metric] = (new_value - old_value) / abs(old_value)
        
        return {
            "parameters": best_parameters,
            "old_performance": original_performance,
            "new_performance": best_performance,
            "performance_improvement": improvement,
            "target_metric": target_metric,
            "target_improvement": improvement.get(target_metric, 0.0),
            "optimization_time": results.get('optimization_time', 0.0),
            "n_iterations": results.get('n_iterations', 0)
        }
    
    def _save_optimization_results(self, job_id: str, results: Dict[str, Any]):
        """
        Save optimization results to disk.
        
        Args:
            job_id: Job identifier
            results: Optimization results
        """
        results_path = os.path.join(self.storage_path, f"{job_id}_results.json")
        
        try:
            # Clean up results to make them JSON serializable
            save_results = {
                'best_parameters': results.get('best_parameters', {}),
                'best_value': results.get('best_value', 0.0),
                'n_iterations': results.get('n_iterations', 0),
                'optimization_time': results.get('optimization_time', 0.0),
                'timestamp': datetime.now().isoformat(),
                'parameter_space': results.get('parameter_space', {})
            }
            
            # Save all evaluated parameters
            all_parameters = []
            for p in results.get('all_parameters', []):
                all_parameters.append({
                    'parameters': p.get('parameters', {}),
                    'value': p.get('value', 0.0),
                    'timestamp': p.get('timestamp', '')
                })
            
            save_results['all_parameters'] = all_parameters
            
            # Write to file
            with open(results_path, 'w') as f:
                json.dump(save_results, f, indent=2)
                
            logger.debug(f"Saved optimization results for job {job_id}")
                
        except Exception as e:
            logger.error(f"Error saving optimization results: {str(e)}")
    
    def get_optimization_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of an ongoing optimization.
        
        Args:
            job_id: Optimization job ID
            
        Returns:
            Status information or None if not found
        """
        with self.lock:
            optimizer = self.ongoing_optimizations.get(job_id)
            results = self.optimization_results.get(job_id)
            
        if optimizer:
            # Optimization is ongoing
            return {
                "status": "running",
                "job_id": job_id,
                "best_parameters": optimizer.get_best_parameters()[0],
                "best_value": optimizer.get_best_parameters()[1],
                "iterations_completed": len(optimizer.parameters_history)
            }
        elif results:
            # Optimization is completed
            return {
                "status": "completed",
                "job_id": job_id,
                "best_parameters": results.get('best_parameters', {}),
                "best_value": results.get('best_value', 0.0),
                "iterations_completed": results.get('n_iterations', 0),
                "optimization_time": results.get('optimization_time', 0.0)
            }
        else:
            # Try to load results from disk
            results_path = os.path.join(self.storage_path, f"{job_id}_results.json")
            
            if os.path.exists(results_path):
                try:
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                    
                    return {
                        "status": "completed",
                        "job_id": job_id,
                        "best_parameters": results.get('best_parameters', {}),
                        "best_value": results.get('best_value', 0.0),
                        "iterations_completed": results.get('n_iterations', 0),
                        "optimization_time": results.get('optimization_time', 0.0)
                    }
                except Exception as e:
                    logger.error(f"Error loading optimization results: {str(e)}")
            
            return None


# Singleton instance
_bayesian_manager = None


def get_bayesian_optimization_manager() -> BayesianOptimizationManager:
    """
    Get singleton instance of BayesianOptimizationManager.
    
    Returns:
        BayesianOptimizationManager instance
    """
    global _bayesian_manager
    
    if _bayesian_manager is None:
        _bayesian_manager = BayesianOptimizationManager()
    
    return _bayesian_manager


def register_bayesian_optimizer():
    """
    Register Bayesian optimizer with the optimization framework.
    
    This function should be called during system initialization.
    """
    # Import here to avoid circular imports
    from trading_bot.autonomous.optimization_integration import get_optimization_integration
    
    # Get optimization integration
    integration = get_optimization_integration()
    
    # Get Bayesian manager
    manager = get_bayesian_optimization_manager()
    
    # Register Bayesian optimization execution handler
    integration.scheduler.register_execution_callback(
        lambda job: _handle_bayesian_job(job, manager),
        name="bayesian"
    )
    
    logger.info("Registered Bayesian optimizer with optimization framework")


def _handle_bayesian_job(job, manager):
    """
    Handle a Bayesian optimization job.
    
    Args:
        job: Optimization job
        manager: Bayesian optimization manager
    """
    # Only handle Bayesian optimization jobs
    if job.parameters.get("method") == OptimizationMethod.BAYESIAN.value:
        # Trigger the optimization process via event
        # This allows the manager to handle it through its normal event handling
        EventBus().emit(
            OptimizationEventType.OPTIMIZATION_STARTED,
            {
                "job_id": job.job_id,
                "strategy_id": job.strategy_id,
                "version_id": job.version_id,
                "parameters": job.parameters
            }
        )
        
        # Return True to indicate job was handled
        return True
    
    # Return False to let another handler process the job
    return False


if __name__ == "__main__":
    # For testing
    manager = get_bayesian_optimization_manager()
    
    # Test parameter space creation
    param_space = ParameterSpace()
    param_space.add_real_parameter("threshold", 0.1, 0.9, default=0.5)
    param_space.add_integer_parameter("window", 5, 50, default=20)
    param_space.add_categorical_parameter("method", ["sma", "ema", "wma"], default="sma")
    
    print(f"Parameter space: {param_space.get_parameters_dict()}")
    print(f"Random parameters: {param_space.get_random_parameters(3)}")
