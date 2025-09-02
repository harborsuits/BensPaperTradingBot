#!/usr/bin/env python3
"""
Advanced Optimization Integration

This module integrates all our advanced optimization techniques into a unified framework,
building directly on our existing optimization infrastructure.

Key features:
1. Unified interface for multiple optimization algorithms
2. Intelligent algorithm selection based on task requirements
3. Event-driven architecture for seamless integration
4. Support for cross-algorithm result comparison
5. Configuration management for fine-tuning optimization behavior
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from enum import Enum
import threading

# Import optimization techniques
from trading_bot.autonomous.bayesian_optimizer import (
    BayesianOptimizer, ParameterSpace, ParameterType, OptimizationDirection
)
from trading_bot.autonomous.genetic_optimizer import (
    GeneticOptimizer, SelectionMethod, CrossoverMethod
)
from trading_bot.autonomous.simulated_annealing_optimizer import (
    SimulatedAnnealingOptimizer, CoolingSchedule
)
from trading_bot.autonomous.multi_objective_optimizer import (
    MultiObjectiveOptimizer, MultiObjectiveAlgorithm
)

# Import existing parameter optimizer
from trading_bot.autonomous.parameter_optimization import (
    get_parameter_optimizer, MarketParameters
)

# Import event system
from trading_bot.event_system import EventBus, Event, EventType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationMetric(str, Enum):
    """Key optimization metrics for trading strategies."""
    SHARPE_RATIO = "sharpe_ratio"
    RETURN = "return"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    VOLATILITY = "volatility"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    INFORMATION_RATIO = "information_ratio"
    TRADE_COUNT = "trade_count"
    PROFIT_FACTOR = "profit_factor"
    REGIME_ACCURACY = "regime_accuracy"
    CONSISTENCY = "consistency"


class OptimizationType(str, Enum):
    """Types of optimization problems."""
    SINGLE_OBJECTIVE = "single_objective"
    MULTI_OBJECTIVE = "multi_objective"
    REGIME_BALANCED = "regime_balanced"
    CONSTRAINED = "constrained"


class OptimizationApproach(str, Enum):
    """Optimization approaches/algorithms."""
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    SIMULATED_ANNEALING = "simulated_annealing"
    MULTI_OBJECTIVE = "multi_objective"
    AUTOMATIC = "automatic"  # Let system choose best approach


class AdvancedOptimizationIntegration:
    """
    Integrates advanced optimization techniques with our existing framework.
    
    This class provides a unified interface for different optimization algorithms
    and intelligently selects the most appropriate technique based on the task.
    It builds directly on our parameter optimization system, extending it with
    more sophisticated algorithms.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the advanced optimization integration.
        
        Args:
            config_path: Path to configuration file
        """
        # Get parameter optimizer
        self.parameter_optimizer = get_parameter_optimizer()
        
        # Event system
        self.event_bus = EventBus()
        
        # Optimization jobs
        self.running_jobs = {}
        self.completed_jobs = {}
        self.job_lock = threading.RLock()
        
        # Configuration
        self.config = {
            # Algorithm selection settings
            "auto_selection_criteria": {
                "param_count_threshold": 5,  # Bayesian if under threshold, genetic if over
                "multi_objective_threshold": 2,  # Use multi-objective if >= this many objectives
                "use_simulated_annealing_for_non_convex": True,  # Use SA for likely non-convex problems
            },
            
            # Algorithm-specific default settings
            "bayesian_defaults": {
                "n_iterations": 50,
                "n_initial_points": 10,
                "acquisition_function": "expected_improvement"
            },
            "genetic_defaults": {
                "population_size": 50,
                "n_generations": 20,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "selection_method": "tournament",
                "crossover_method": "uniform",
                "adaptive_mutation": True
            },
            "simulated_annealing_defaults": {
                "initial_temp": 100.0,
                "cooling_rate": 0.95,
                "n_steps_per_temp": 10,
                "min_temp": 1e-10,
                "cooling_schedule": "exponential"
            },
            "multi_objective_defaults": {
                "population_size": 100,
                "n_generations": 30,
                "algorithm": "nsga_ii"
            },
            
            # Market regime optimization settings
            "regime_optimization": {
                "enabled": True,
                "regimes": ["bullish", "bearish", "sideways", "volatile"],
                "objective_weights": {
                    "bullish": {"sharpe_ratio": 0.7, "max_drawdown": 0.3},
                    "bearish": {"max_drawdown": 0.6, "sharpe_ratio": 0.4},
                    "sideways": {"win_rate": 0.5, "sharpe_ratio": 0.5},
                    "volatile": {"max_drawdown": 0.5, "calmar_ratio": 0.5}
                }
            }
        }
        
        # Load configuration
        if config_path:
            self._load_config(config_path)
        
        # Register event handlers
        self._register_event_handlers()
        
        # Paths for storing optimization results
        self.results_path = os.path.join(
            os.path.expanduser("~"), ".trading_bot", "optimization", "advanced"
        )
        os.makedirs(self.results_path, exist_ok=True)
        
        logger.info("Advanced optimization integration initialized")
    
    def _register_event_handlers(self):
        """Register for relevant events."""
        # Handle parameter optimization requests
        self.event_bus.register(
            "optimization_requested",
            self._handle_optimization_request
        )
        
        # Handle verification results (for regime-specific optimization)
        self.event_bus.register(
            "verification_report_generated",
            self._handle_verification_report
        )
        
        logger.info("Registered event handlers for advanced optimization")
    
    def _handle_optimization_request(self, event: Event):
        """
        Handle optimization request events.
        
        Args:
            event: Optimization request event
        """
        data = event.data
        if not data:
            return
        
        # Extract request details
        request_id = data.get("request_id")
        parameter_space = data.get("parameter_space")
        strategy_id = data.get("strategy_id")
        version_id = data.get("version_id")
        objective_function = data.get("objective_function")
        objectives = data.get("objectives", [])
        optimization_type = data.get("optimization_type", OptimizationType.SINGLE_OBJECTIVE)
        approach = data.get("approach", OptimizationApproach.AUTOMATIC)
        
        # Validate required fields
        if not request_id or not parameter_space or not objective_function:
            logger.error("Invalid optimization request: missing required fields")
            return
        
        # Start optimization in background thread
        threading.Thread(
            target=self._run_optimization,
            args=(request_id, parameter_space, objective_function, objectives, 
                  optimization_type, approach, strategy_id, version_id),
            daemon=True,
            name=f"AdvOpt-{request_id[:8]}"
        ).start()
        
        logger.info(f"Started optimization job {request_id} for strategy {strategy_id}")
    
    def _run_optimization(
        self,
        request_id: str,
        parameter_space: ParameterSpace,
        objective_function: Callable,
        objectives: List[Dict[str, Any]],
        optimization_type: Union[OptimizationType, str],
        approach: Union[OptimizationApproach, str],
        strategy_id: Optional[str] = None,
        version_id: Optional[str] = None
    ):
        """
        Run optimization with the appropriate algorithm.
        
        Args:
            request_id: Optimization request ID
            parameter_space: Parameter space to optimize
            objective_function: Function to evaluate parameters
            objectives: List of objective definitions
            optimization_type: Type of optimization problem
            approach: Optimization approach/algorithm
            strategy_id: Strategy ID (optional)
            version_id: Version ID (optional)
        """
        # Convert string enums to enum values if needed
        if isinstance(optimization_type, str):
            optimization_type = OptimizationType(optimization_type)
        
        if isinstance(approach, str):
            approach = OptimizationApproach(approach)
        
        # Record job start
        with self.job_lock:
            self.running_jobs[request_id] = {
                "start_time": datetime.now().isoformat(),
                "parameter_space": parameter_space.to_dict(),
                "optimization_type": optimization_type.value,
                "approach": approach.value,
                "strategy_id": strategy_id,
                "version_id": version_id,
                "status": "running"
            }
        
        # Select appropriate algorithm if automatic
        if approach == OptimizationApproach.AUTOMATIC:
            approach = self._select_algorithm(parameter_space, optimization_type, objectives)
            logger.info(f"Auto-selected algorithm: {approach.value}")
        
        try:
            # Run optimization with selected approach
            if approach == OptimizationApproach.BAYESIAN:
                results = self._run_bayesian_optimization(
                    parameter_space, objective_function, objectives, optimization_type
                )
            elif approach == OptimizationApproach.GENETIC:
                results = self._run_genetic_optimization(
                    parameter_space, objective_function, objectives, optimization_type
                )
            elif approach == OptimizationApproach.SIMULATED_ANNEALING:
                results = self._run_simulated_annealing(
                    parameter_space, objective_function, objectives, optimization_type
                )
            elif approach == OptimizationApproach.MULTI_OBJECTIVE:
                results = self._run_multi_objective_optimization(
                    parameter_space, objective_function, objectives
                )
            else:
                # Fallback to Bayesian
                results = self._run_bayesian_optimization(
                    parameter_space, objective_function, objectives, optimization_type
                )
            
            # Add approach to results
            results["approach"] = approach.value
            
            # Update parameter optimizer with optimized parameters if relevant
            if strategy_id and optimization_type == OptimizationType.REGIME_BALANCED:
                self._apply_regime_parameters(results, strategy_id)
            
            # Record job completion
            with self.job_lock:
                self.running_jobs.pop(request_id, None)
                self.completed_jobs[request_id] = {
                    "start_time": self.running_jobs.get(request_id, {}).get("start_time"),
                    "end_time": datetime.now().isoformat(),
                    "results": results,
                    "status": "completed"
                }
            
            # Save results
            self._save_results(request_id, results)
            
            # Emit completion event
            self.event_bus.emit(Event(
                event_type="optimization_completed",
                data={
                    "request_id": request_id,
                    "results": results,
                    "strategy_id": strategy_id,
                    "version_id": version_id
                }
            ))
            
            logger.info(f"Completed optimization job {request_id}")
            
        except Exception as e:
            logger.error(f"Error in optimization job {request_id}: {str(e)}")
            
            # Record job failure
            with self.job_lock:
                self.running_jobs.pop(request_id, None)
                self.completed_jobs[request_id] = {
                    "start_time": self.running_jobs.get(request_id, {}).get("start_time"),
                    "end_time": datetime.now().isoformat(),
                    "error": str(e),
                    "status": "failed"
                }
            
            # Emit failure event
            self.event_bus.emit(Event(
                event_type="optimization_failed",
                data={
                    "request_id": request_id,
                    "error": str(e),
                    "strategy_id": strategy_id,
                    "version_id": version_id
                }
            ))
    
    def _select_algorithm(
        self,
        parameter_space: ParameterSpace,
        optimization_type: OptimizationType,
        objectives: List[Dict[str, Any]]
    ) -> OptimizationApproach:
        """
        Select the most appropriate optimization algorithm.
        
        Args:
            parameter_space: Parameter space to optimize
            optimization_type: Type of optimization problem
            objectives: Objective definitions
            
        Returns:
            Selected optimization approach
        """
        # Get criteria from config
        param_count_threshold = self.config["auto_selection_criteria"]["param_count_threshold"]
        multi_objective_threshold = self.config["auto_selection_criteria"]["multi_objective_threshold"]
        use_sa_for_non_convex = self.config["auto_selection_criteria"]["use_simulated_annealing_for_non_convex"]
        
        # Count parameters
        param_count = len(parameter_space)
        
        # Count objectives
        objective_count = len(objectives)
        
        # Check for multi-objective problem
        if optimization_type == OptimizationType.MULTI_OBJECTIVE or \
           objective_count >= multi_objective_threshold:
            return OptimizationApproach.MULTI_OBJECTIVE
        
        # Check for regime-balanced optimization
        if optimization_type == OptimizationType.REGIME_BALANCED:
            return OptimizationApproach.MULTI_OBJECTIVE
        
        # Check for likely non-convex problem and use simulated annealing
        non_convex_indicators = ["max_drawdown", "calmar_ratio", "sortino_ratio"]
        objective_names = [obj.get("name") for obj in objectives]
        
        if use_sa_for_non_convex and any(ind in objective_names for ind in non_convex_indicators):
            return OptimizationApproach.SIMULATED_ANNEALING
        
        # For parameter spaces with few parameters, Bayesian works well
        if param_count <= param_count_threshold:
            return OptimizationApproach.BAYESIAN
        else:
            # For larger parameter spaces, genetic algorithms scale better
            return OptimizationApproach.GENETIC
    
    def _run_bayesian_optimization(
        self,
        parameter_space: ParameterSpace,
        objective_function: Callable,
        objectives: List[Dict[str, Any]],
        optimization_type: OptimizationType
    ) -> Dict[str, Any]:
        """
        Run Bayesian optimization.
        
        Args:
            parameter_space: Parameter space to optimize
            objective_function: Function to evaluate parameters
            objectives: Objective definitions
            optimization_type: Type of optimization problem
            
        Returns:
            Optimization results
        """
        # Get default settings
        defaults = self.config["bayesian_defaults"]
        
        # Configure optimizer
        n_iterations = defaults["n_iterations"]
        n_initial_points = defaults["n_initial_points"]
        acquisition_function = defaults["acquisition_function"]
        
        # Determine optimization direction (assume maximization by default)
        minimize = False
        if objectives and objectives[0].get("direction") == "minimize":
            minimize = True
        
        # Create optimizer
        optimizer = BayesianOptimizer(
            parameter_space=parameter_space,
            minimize=minimize,
            acquisition_function=acquisition_function,
            n_initial_points=n_initial_points
        )
        
        # Run optimization
        start_time = time.time()
        results = optimizer.optimize(objective_function, n_iterations=n_iterations)
        
        # Add timing information
        results["optimization_time"] = time.time() - start_time
        
        return results
    
    def _run_genetic_optimization(
        self,
        parameter_space: ParameterSpace,
        objective_function: Callable,
        objectives: List[Dict[str, Any]],
        optimization_type: OptimizationType
    ) -> Dict[str, Any]:
        """
        Run genetic algorithm optimization.
        
        Args:
            parameter_space: Parameter space to optimize
            objective_function: Function to evaluate parameters
            objectives: Objective definitions
            optimization_type: Type of optimization problem
            
        Returns:
            Optimization results
        """
        # Get default settings
        defaults = self.config["genetic_defaults"]
        
        # Configure optimizer
        population_size = defaults["population_size"]
        n_generations = defaults["n_generations"]
        mutation_rate = defaults["mutation_rate"]
        crossover_rate = defaults["crossover_rate"]
        selection_method = defaults["selection_method"]
        crossover_method = defaults["crossover_method"]
        adaptive_mutation = defaults["adaptive_mutation"]
        
        # Determine optimization direction (assume maximization by default)
        minimize = False
        if objectives and objectives[0].get("direction") == "minimize":
            minimize = True
        
        # Create optimizer
        optimizer = GeneticOptimizer(
            parameter_space=parameter_space,
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            selection_method=selection_method,
            crossover_method=crossover_method,
            minimize=minimize,
            adaptive_mutation=adaptive_mutation
        )
        
        # Run optimization
        start_time = time.time()
        results = optimizer.optimize(objective_function, n_generations=n_generations)
        
        # Add timing information
        results["optimization_time"] = time.time() - start_time
        
        return results
    
    def _run_simulated_annealing(
        self,
        parameter_space: ParameterSpace,
        objective_function: Callable,
        objectives: List[Dict[str, Any]],
        optimization_type: OptimizationType
    ) -> Dict[str, Any]:
        """
        Run simulated annealing optimization.
        
        Args:
            parameter_space: Parameter space to optimize
            objective_function: Function to evaluate parameters
            objectives: Objective definitions
            optimization_type: Type of optimization problem
            
        Returns:
            Optimization results
        """
        # Get default settings
        defaults = self.config["simulated_annealing_defaults"]
        
        # Configure optimizer
        initial_temp = defaults["initial_temp"]
        cooling_rate = defaults["cooling_rate"]
        n_steps_per_temp = defaults["n_steps_per_temp"]
        min_temp = defaults["min_temp"]
        cooling_schedule = defaults["cooling_schedule"]
        
        # Determine optimization direction (assume maximization by default)
        minimize = False
        if objectives and objectives[0].get("direction") == "minimize":
            minimize = True
        
        # Create optimizer
        optimizer = SimulatedAnnealingOptimizer(
            parameter_space=parameter_space,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            n_steps_per_temp=n_steps_per_temp,
            min_temp=min_temp,
            cooling_schedule=cooling_schedule,
            minimize=minimize
        )
        
        # Determine number of iterations based on parameter space size
        param_count = len(parameter_space)
        n_iterations = max(50, param_count * 10)  # Scale with parameter count
        
        # Run optimization
        start_time = time.time()
        results = optimizer.optimize(objective_function, n_iterations=n_iterations)
        
        # Add timing information
        results["optimization_time"] = time.time() - start_time
        
        return results
    
    def _run_multi_objective_optimization(
        self,
        parameter_space: ParameterSpace,
        objective_function: Callable,
        objectives: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run multi-objective optimization.
        
        Args:
            parameter_space: Parameter space to optimize
            objective_function: Function that returns multiple objectives
            objectives: Objective definitions
            
        Returns:
            Optimization results
        """
        # Get default settings
        defaults = self.config["multi_objective_defaults"]
        
        # Configure optimizer
        population_size = defaults["population_size"]
        n_generations = defaults["n_generations"]
        algorithm = defaults["algorithm"]
        
        # Extract objective names and directions
        objective_names = [obj.get("name", f"objective_{i}") for i, obj in enumerate(objectives)]
        objective_directions = [obj.get("direction", "maximize") for obj in objectives]
        
        # Extract weights if available
        weights = [obj.get("weight", 1.0) for obj in objectives]
        
        # Create optimizer
        optimizer = MultiObjectiveOptimizer(
            parameter_space=parameter_space,
            objective_names=objective_names,
            objective_directions=objective_directions,
            weights=weights,
            population_size=population_size,
            algorithm=algorithm
        )
        
        # Define wrapper for objective function if needed
        original_objective = objective_function
        
        # Check if we need to create a multi-objective wrapper
        if callable(objective_function) and len(objectives) > 1:
            # Define a wrapper that returns a list of objectives
            def multi_objective_wrapper(params):
                # Call original function
                result = original_objective(params)
                
                # If result is already a list or tuple, return it
                if isinstance(result, (list, tuple)) and len(result) == len(objectives):
                    return result
                
                # If result is a dictionary, extract objectives
                if isinstance(result, dict):
                    return [result.get(name, 0.0) for name in objective_names]
                
                # Otherwise, wrap the single result
                return [result] + [0.0] * (len(objectives) - 1)
            
            objective_function = multi_objective_wrapper
        
        # Run optimization
        start_time = time.time()
        results = optimizer.optimize(objective_function, n_generations=n_generations)
        
        # Add timing information
        results["optimization_time"] = time.time() - start_time
        
        return results
    
    def _handle_verification_report(self, event: Event):
        """
        Handle verification report events for regime-specific optimization.
        
        Args:
            event: Verification report event
        """
        # Check if regime optimization is enabled
        if not self.config["regime_optimization"].get("enabled", True):
            return
        
        # Extract report
        report = event.data.get("report")
        if not report:
            return
        
        # Extract regime-specific accuracy
        regime_accuracy = report.get("regime_accuracy", {})
        
        # Check if we have enough regime data
        if not regime_accuracy:
            return
        
        # Find regimes with low accuracy
        low_accuracy_regimes = []
        for regime, data in regime_accuracy.items():
            accuracy = data.get("accuracy", 0.0)
            if accuracy < 0.7:  # Threshold for triggering optimization
                low_accuracy_regimes.append((regime, accuracy))
        
        # If we have regimes with low accuracy, suggest optimization
        if low_accuracy_regimes:
            regimes_str = ", ".join([f"{r[0]} ({r[1]:.2f})" for r in low_accuracy_regimes])
            logger.info(f"Detected regimes with low accuracy: {regimes_str}")
            
            # Emit event for potential optimization
            self.event_bus.emit(Event(
                event_type="regime_optimization_suggested",
                data={
                    "regimes": [r[0] for r in low_accuracy_regimes],
                    "accuracies": [r[1] for r in low_accuracy_regimes],
                    "report_id": report.get("report_id"),
                    "strategy_id": report.get("strategy_id")
                }
            ))
    
    def _apply_regime_parameters(self, results: Dict[str, Any], strategy_id: str):
        """
        Apply regime-specific optimized parameters.
        
        Args:
            results: Optimization results
            strategy_id: Strategy ID
        """
        # This would contain logic to update the MarketParameters with
        # regime-specific optimized values based on the optimization results.
        # For now, we just log it.
        logger.info(f"Applying regime-specific parameters for strategy {strategy_id}")
    
    def _save_results(self, request_id: str, results: Dict[str, Any]):
        """
        Save optimization results to disk.
        
        Args:
            request_id: Optimization request ID
            results: Optimization results
        """
        try:
            results_path = os.path.join(self.results_path, f"{request_id}.json")
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Saved optimization results to {results_path}")
            
        except Exception as e:
            logger.error(f"Error saving optimization results: {str(e)}")
    
    def _load_config(self, config_path: str):
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    
                    # Update configuration (keeping defaults for missing values)
                    for key, value in loaded_config.items():
                        if key in self.config:
                            if isinstance(self.config[key], dict) and isinstance(value, dict):
                                # Merge dictionaries
                                self.config[key].update(value)
                            else:
                                # Replace value
                                self.config[key] = value
                
                logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
    
    def get_job_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of an optimization job.
        
        Args:
            request_id: Optimization request ID
            
        Returns:
            Job status or None if not found
        """
        with self.job_lock:
            # Check running jobs
            if request_id in self.running_jobs:
                return self.running_jobs[request_id]
            
            # Check completed jobs
            if request_id in self.completed_jobs:
                return self.completed_jobs[request_id]
        
        # Try to load from disk
        results_path = os.path.join(self.results_path, f"{request_id}.json")
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                return {
                    "status": "completed",
                    "results": results
                }
            except Exception as e:
                logger.error(f"Error loading results: {str(e)}")
        
        return None
    
    def get_optimization_approaches(self) -> Dict[str, str]:
        """
        Get available optimization approaches with descriptions.
        
        Returns:
            Dictionary of approaches with descriptions
        """
        return {
            OptimizationApproach.AUTOMATIC.value: 
                "Automatically select the most appropriate algorithm",
            OptimizationApproach.BAYESIAN.value: 
                "Bayesian optimization for efficient exploration of the parameter space",
            OptimizationApproach.GENETIC.value: 
                "Genetic algorithms for large parameter spaces and complex objectives",
            OptimizationApproach.SIMULATED_ANNEALING.value: 
                "Simulated annealing for non-convex objectives with many local optima",
            OptimizationApproach.MULTI_OBJECTIVE.value: 
                "Multi-objective optimization for balancing competing objectives"
        }


# Singleton instance
_advanced_optimizer = None


def get_advanced_optimization() -> AdvancedOptimizationIntegration:
    """
    Get singleton instance of advanced optimization integration.
    
    Returns:
        AdvancedOptimizationIntegration instance
    """
    global _advanced_optimizer
    
    if _advanced_optimizer is None:
        _advanced_optimizer = AdvancedOptimizationIntegration()
    
    return _advanced_optimizer


if __name__ == "__main__":
    # Simple test of the advanced optimization
    optimizer = get_advanced_optimization()
    approaches = optimizer.get_optimization_approaches()
    
    print("Available optimization approaches:")
    for name, description in approaches.items():
        print(f"- {name}: {description}")
