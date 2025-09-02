#!/usr/bin/env python3
"""
Bayesian Optimizer

This module implements a Bayesian Optimization algorithm for efficiently
optimizing strategy parameters. It uses Gaussian Process regression to model
the objective function and an acquisition function to determine the next
parameters to evaluate.

Classes:
    BayesianOptimizer: Main optimizer class implementing Bayesian optimization
    ParameterSpace: Helper class to define and manage parameter search spaces
"""

import os
import json
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from datetime import datetime
from enum import Enum

# Import optional dependencies with fallbacks
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    
try:
    from skopt import gp_minimize, Optimizer as SkOptimizer
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParameterType(str, Enum):
    """Enum for parameter types in the search space."""
    REAL = "real"  # Continuous real values
    INTEGER = "integer"  # Discrete integer values
    CATEGORICAL = "categorical"  # Categorical choices
    BOOLEAN = "boolean"  # Boolean values


class ParameterSpace:
    """
    Defines and manages the parameter search space for optimization.
    
    This class helps define the bounds and types of parameters to be optimized,
    and provides utilities for converting between parameter dictionaries and
    optimization-ready arrays.
    """
    
    def __init__(self):
        """Initialize an empty parameter space."""
        self.parameters = []
        self.param_names = []
        self.param_types = {}
        self.bounds = {}
        self.defaults = {}
        self.categories = {}
    
    def add_real_parameter(
        self, 
        name: str, 
        lower_bound: float, 
        upper_bound: float, 
        default: Optional[float] = None,
        log_scale: bool = False
    ) -> 'ParameterSpace':
        """
        Add a real-valued parameter to the search space.
        
        Args:
            name: Parameter name
            lower_bound: Minimum value
            upper_bound: Maximum value
            default: Default value, if not provided uses the middle of bounds
            log_scale: Whether to optimize in log scale (for parameters with large ranges)
            
        Returns:
            Self for method chaining
        """
        if default is None:
            default = (lower_bound + upper_bound) / 2.0
            
        self.param_names.append(name)
        self.param_types[name] = ParameterType.REAL
        self.bounds[name] = (lower_bound, upper_bound)
        self.defaults[name] = default
        
        self.parameters.append({
            'name': name,
            'type': ParameterType.REAL.value,
            'bounds': (lower_bound, upper_bound),
            'default': default,
            'log_scale': log_scale
        })
        
        return self
    
    def add_integer_parameter(
        self, 
        name: str, 
        lower_bound: int, 
        upper_bound: int, 
        default: Optional[int] = None
    ) -> 'ParameterSpace':
        """
        Add an integer parameter to the search space.
        
        Args:
            name: Parameter name
            lower_bound: Minimum value (inclusive)
            upper_bound: Maximum value (inclusive)
            default: Default value, if not provided uses the middle of bounds
            
        Returns:
            Self for method chaining
        """
        if default is None:
            default = (lower_bound + upper_bound) // 2
            
        self.param_names.append(name)
        self.param_types[name] = ParameterType.INTEGER
        self.bounds[name] = (lower_bound, upper_bound)
        self.defaults[name] = default
        
        self.parameters.append({
            'name': name,
            'type': ParameterType.INTEGER.value,
            'bounds': (lower_bound, upper_bound),
            'default': default
        })
        
        return self
    
    def add_categorical_parameter(
        self, 
        name: str, 
        categories: List[Any], 
        default: Optional[Any] = None
    ) -> 'ParameterSpace':
        """
        Add a categorical parameter to the search space.
        
        Args:
            name: Parameter name
            categories: List of possible values
            default: Default value, if not provided uses the first category
            
        Returns:
            Self for method chaining
        """
        if not categories:
            raise ValueError(f"Categories list for parameter {name} cannot be empty")
            
        if default is None:
            default = categories[0]
        elif default not in categories:
            raise ValueError(f"Default value {default} for parameter {name} must be in categories list")
            
        self.param_names.append(name)
        self.param_types[name] = ParameterType.CATEGORICAL
        self.categories[name] = categories
        self.defaults[name] = default
        
        self.parameters.append({
            'name': name,
            'type': ParameterType.CATEGORICAL.value,
            'categories': categories,
            'default': default
        })
        
        return self
    
    def add_boolean_parameter(
        self, 
        name: str, 
        default: bool = False
    ) -> 'ParameterSpace':
        """
        Add a boolean parameter to the search space.
        
        Args:
            name: Parameter name
            default: Default value
            
        Returns:
            Self for method chaining
        """
        self.param_names.append(name)
        self.param_types[name] = ParameterType.BOOLEAN
        self.categories[name] = [True, False]
        self.defaults[name] = default
        
        self.parameters.append({
            'name': name,
            'type': ParameterType.BOOLEAN.value,
            'categories': [True, False],
            'default': default
        })
        
        return self
    
    def get_parameters_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the parameter space as a structured dictionary.
        
        Returns:
            Dictionary with parameter definitions
        """
        return {
            param['name']: {
                'type': param['type'],
                **{k: v for k, v in param.items() if k not in ['name', 'type']}
            } 
            for param in self.parameters
        }
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get the default parameter values as a dictionary.
        
        Returns:
            Dictionary mapping parameter names to default values
        """
        return self.defaults.copy()
    
    def get_random_parameters(self, count: int = 1) -> List[Dict[str, Any]]:
        """
        Generate random parameter combinations from the search space.
        
        Args:
            count: Number of random parameter sets to generate
            
        Returns:
            List of parameter dictionaries
        """
        result = []
        
        for _ in range(count):
            params = {}
            
            for param in self.parameters:
                name = param['name']
                param_type = param['type']
                
                if param_type == ParameterType.REAL:
                    lower, upper = param['bounds']
                    if param.get('log_scale', False):
                        # Sample in log space for better distribution
                        log_lower = np.log(max(lower, 1e-10))
                        log_upper = np.log(max(upper, 1e-8))
                        value = np.exp(np.random.uniform(log_lower, log_upper))
                    else:
                        value = np.random.uniform(lower, upper)
                
                elif param_type == ParameterType.INTEGER:
                    lower, upper = param['bounds']
                    value = np.random.randint(lower, upper + 1)
                
                elif param_type in (ParameterType.CATEGORICAL, ParameterType.BOOLEAN):
                    categories = param['categories']
                    value = np.random.choice(categories)
                
                params[name] = value
            
            result.append(params)
        
        return result
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate parameters against the defined space.
        
        Args:
            parameters: Parameter dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        for param in self.parameters:
            name = param['name']
            
            # Check if parameter is present
            if name not in parameters:
                return False, f"Parameter '{name}' is missing"
            
            value = parameters[name]
            param_type = param['type']
            
            # Validate by type
            if param_type == ParameterType.REAL:
                if not isinstance(value, (int, float)):
                    return False, f"Parameter '{name}' should be a number"
                
                lower, upper = param['bounds']
                if value < lower or value > upper:
                    return False, f"Parameter '{name}' value {value} is outside bounds [{lower}, {upper}]"
            
            elif param_type == ParameterType.INTEGER:
                if not isinstance(value, int):
                    return False, f"Parameter '{name}' should be an integer"
                
                lower, upper = param['bounds']
                if value < lower or value > upper:
                    return False, f"Parameter '{name}' value {value} is outside bounds [{lower}, {upper}]"
            
            elif param_type in (ParameterType.CATEGORICAL, ParameterType.BOOLEAN):
                categories = param['categories']
                if value not in categories:
                    return False, f"Parameter '{name}' value {value} is not in allowed categories {categories}"
        
        return True, ""
    
    @classmethod
    def from_dict(cls, space_dict: Dict[str, Dict[str, Any]]) -> 'ParameterSpace':
        """
        Create a parameter space from a dictionary representation.
        
        Args:
            space_dict: Dictionary defining parameter space
            
        Returns:
            Initialized ParameterSpace
        """
        space = cls()
        
        for name, param_def in space_dict.items():
            param_type = param_def.get('type')
            
            if param_type == ParameterType.REAL.value:
                space.add_real_parameter(
                    name,
                    lower_bound=param_def['bounds'][0],
                    upper_bound=param_def['bounds'][1],
                    default=param_def.get('default'),
                    log_scale=param_def.get('log_scale', False)
                )
            elif param_type == ParameterType.INTEGER.value:
                space.add_integer_parameter(
                    name,
                    lower_bound=param_def['bounds'][0],
                    upper_bound=param_def['bounds'][1],
                    default=param_def.get('default')
                )
            elif param_type == ParameterType.CATEGORICAL.value:
                space.add_categorical_parameter(
                    name,
                    categories=param_def['categories'],
                    default=param_def.get('default')
                )
            elif param_type == ParameterType.BOOLEAN.value:
                space.add_boolean_parameter(
                    name,
                    default=param_def.get('default', False)
                )
        
        return space
    
    def __len__(self) -> int:
        """Get the number of parameters in the space."""
        return len(self.parameters)


class AcquisitionFunction(str, Enum):
    """Acquisition functions for Bayesian optimization."""
    EI = "expected_improvement"  # Expected Improvement
    PI = "probability_improvement"  # Probability of Improvement
    UCB = "upper_confidence_bound"  # Upper Confidence Bound
    LCB = "lower_confidence_bound"  # Lower Confidence Bound (for minimization)


class OptimizationDirection(str, Enum):
    """Direction of optimization."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class BayesianOptimizer:
    """
    Bayesian optimization for strategy parameters.
    
    This class implements Bayesian optimization using Gaussian Processes to
    model the objective function and acquisition functions to guide the search.
    It provides an efficient way to find optimal parameters with a limited
    number of objective function evaluations.
    """
    
    def __init__(
        self,
        parameter_space: ParameterSpace,
        direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
        acquisition_function: AcquisitionFunction = AcquisitionFunction.EI,
        random_state: Optional[int] = None,
        n_initial_points: int = 5,
        exploration_weight: float = 0.1
    ):
        """
        Initialize the Bayesian optimizer.
        
        Args:
            parameter_space: Definition of parameters to optimize
            direction: Whether to maximize or minimize the objective
            acquisition_function: Function to determine next points to evaluate
            random_state: Random seed for reproducibility
            n_initial_points: Number of random initial points to evaluate
            exploration_weight: Trade-off between exploration and exploitation (0-1)
        """
        self.parameter_space = parameter_space
        self.direction = direction
        self.acquisition_function = acquisition_function
        self.random_state = random_state
        self.n_initial_points = n_initial_points
        self.exploration_weight = exploration_weight
        
        # Initialize history
        self.X_observed = []  # Parameter configurations
        self.y_observed = []  # Observed values
        self.parameters_history = []  # Parameter dictionaries
        
        # Check for scikit-optimize availability
        if not SKOPT_AVAILABLE:
            if not SKLEARN_AVAILABLE:
                raise ImportError(
                    "Bayesian optimization requires scikit-optimize or scikit-learn. "
                    "Please install with: pip install scikit-optimize (or scikit-learn)"
                )
            logger.warning(
                "scikit-optimize not found. Falling back to basic implementation "
                "using scikit-learn. For better performance, install scikit-optimize."
            )
        
        # Initialize optimizer backend
        if SKOPT_AVAILABLE:
            self._initialize_skopt()
        else:
            self._initialize_basic_gp()
    
    def _initialize_skopt(self):
        """Initialize using scikit-optimize backend."""
        # Convert parameter space to skopt dimension format
        dimensions = []
        self.param_names = []
        
        for param in self.parameter_space.parameters:
            name = param['name']
            param_type = param['type']
            self.param_names.append(name)
            
            if param_type == ParameterType.REAL.value:
                lower, upper = param['bounds']
                if param.get('log_scale', False):
                    dim = Real(lower, upper, prior='log-uniform', name=name)
                else:
                    dim = Real(lower, upper, name=name)
                dimensions.append(dim)
            
            elif param_type == ParameterType.INTEGER.value:
                lower, upper = param['bounds']
                dimensions.append(Integer(lower, upper, name=name))
            
            elif param_type in (ParameterType.CATEGORICAL.value, ParameterType.BOOLEAN.value):
                categories = param['categories']
                dimensions.append(Categorical(categories, name=name))
        
        # Create optimizer
        self.optimizer = SkOptimizer(
            dimensions=dimensions,
            random_state=self.random_state,
            acq_func='EI' if self.acquisition_function == AcquisitionFunction.EI else
                     'PI' if self.acquisition_function == AcquisitionFunction.PI else
                     'UCB',
            acq_optimizer='auto',
            n_initial_points=self.n_initial_points,
            base_estimator='GP',
            acq_func_kwargs={'kappa': self.exploration_weight} if 'UCB' in self.acquisition_function else {}
        )
        
        # Set optimization direction
        self.minimize = (self.direction == OptimizationDirection.MINIMIZE)
    
    def _initialize_basic_gp(self):
        """Initialize using basic Gaussian Process implementation."""
        # Create kernel for GP
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
        
        # Create GP regressor
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=self.random_state
        )
        
        # Set optimization direction
        self.minimize = (self.direction == OptimizationDirection.MINIMIZE)
        
        # Prepare parameter transformation functions
        self._setup_param_transformations()
    
    def _setup_param_transformations(self):
        """Set up functions for parameter space transformations."""
        self.bounds = []
        self.is_categorical = []
        self.categories_map = {}
        self.param_names = []
        
        # Process each parameter
        for param in self.parameter_space.parameters:
            name = param['name']
            param_type = param['type']
            self.param_names.append(name)
            
            if param_type in (ParameterType.REAL.value, ParameterType.INTEGER.value):
                self.bounds.append(param['bounds'])
                self.is_categorical.append(False)
            
            elif param_type in (ParameterType.CATEGORICAL.value, ParameterType.BOOLEAN.value):
                categories = param['categories']
                # Map categorical values to indices
                self.categories_map[name] = {val: i for i, val in enumerate(categories)}
                self.bounds.append((0, len(categories) - 1))
                self.is_categorical.append(True)
    
    def _params_dict_to_array(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Convert parameters dictionary to array format for internal use.
        
        Args:
            params: Parameters dictionary
            
        Returns:
            Array representation of parameters
        """
        x = []
        
        for i, name in enumerate(self.param_names):
            value = params[name]
            
            if self.is_categorical[i]:
                # Convert categorical to index
                value = self.categories_map[name][value]
            
            x.append(value)
        
        return np.array(x)
    
    def _params_array_to_dict(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Convert internal parameter array to dictionary.
        
        Args:
            x: Parameter array
            
        Returns:
            Parameters dictionary
        """
        params = {}
        
        for i, name in enumerate(self.param_names):
            value = x[i]
            
            if self.is_categorical[i]:
                # Convert index back to categorical value
                reverse_map = {idx: val for val, idx in self.categories_map[name].items()}
                value = reverse_map[int(value)]
            elif self.parameter_space.param_types[name] == ParameterType.INTEGER:
                value = int(value)
            
            params[name] = value
        
        return params
    
    def _acquisition_function(self, x: np.ndarray) -> float:
        """
        Compute acquisition function value.
        
        Args:
            x: Parameter array
            
        Returns:
            Acquisition function value (higher is better for searching)
        """
        x = x.reshape(1, -1)
        
        # Predict mean and std with GP
        mu, sigma = self.gp.predict(x, return_std=True)
        
        # Get best observed value so far
        if not self.y_observed:
            return 0.0
            
        y_best = min(self.y_observed) if self.minimize else max(self.y_observed)
        
        # Compute acquisition function value based on type
        if self.acquisition_function == AcquisitionFunction.EI:
            # Expected Improvement
            if sigma == 0.0:
                return 0.0
                
            z = (mu - y_best) / sigma
            if self.minimize:
                z = -z
                
            return sigma * (z * norm.cdf(z) + norm.pdf(z))
            
        elif self.acquisition_function == AcquisitionFunction.PI:
            # Probability of Improvement
            if sigma == 0.0:
                return 0.0
                
            z = (mu - y_best) / sigma
            if self.minimize:
                z = -z
                
            return norm.cdf(z)
            
        elif self.acquisition_function == AcquisitionFunction.UCB:
            # Upper Confidence Bound
            kappa = self.exploration_weight
            value = mu + kappa * sigma
            return value if not self.minimize else -value
            
        elif self.acquisition_function == AcquisitionFunction.LCB:
            # Lower Confidence Bound (for minimization)
            kappa = self.exploration_weight
            value = mu - kappa * sigma
            return value if self.minimize else -value
    
    def suggest_next_parameters(self) -> Dict[str, Any]:
        """
        Suggest the next set of parameters to evaluate.
        
        Returns:
            Dictionary of parameter values to try next
        """
        if SKOPT_AVAILABLE:
            # Use scikit-optimize backend
            if not self.X_observed:
                # Start with random points
                return self.parameter_space.get_random_parameters(1)[0]
            
            # Ask for next point
            next_x = self.optimizer.ask()
            
            # Convert to dictionary
            next_params = {}
            for i, name in enumerate(self.param_names):
                param_type = self.parameter_space.param_types[name]
                value = next_x[i]
                
                # Convert value if needed
                if param_type == ParameterType.INTEGER:
                    value = int(value)
                
                next_params[name] = value
            
            return next_params
        else:
            # Use basic implementation
            if len(self.X_observed) < self.n_initial_points:
                # Generate random parameters for initial exploration
                return self.parameter_space.get_random_parameters(1)[0]
            
            # Update GP model with all observed data
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            
            # Flip sign if maximizing (GP always minimizes)
            if not self.minimize:
                y = -y
                
            self.gp.fit(X, y)
            
            # Use random search to optimize acquisition function
            best_x = None
            best_acq = -np.inf
            
            # Try random points to find best acquisition value
            n_random_points = 1000
            random_xs = []
            
            for i in range(len(self.param_names)):
                lower, upper = self.bounds[i]
                
                if self.is_categorical[i]:
                    # For categorical, generate uniform integers
                    random_col = np.random.randint(lower, upper + 1, n_random_points)
                else:
                    # For continuous, generate uniform values
                    random_col = np.random.uniform(lower, upper, n_random_points)
                    
                    # Convert to integer if needed
                    param_name = self.param_names[i]
                    if self.parameter_space.param_types[param_name] == ParameterType.INTEGER:
                        random_col = np.floor(random_col).astype(int)
                
                random_xs.append(random_col)
            
            # Transpose to get array of points
            random_xs = np.column_stack(random_xs)
            
            # Evaluate acquisition function for all random points
            for x in random_xs:
                acq_value = self._acquisition_function(x)
                
                if acq_value > best_acq:
                    best_acq = acq_value
                    best_x = x
            
            # If no point found, return random parameters
            if best_x is None:
                return self.parameter_space.get_random_parameters(1)[0]
            
            # Convert best array back to dictionary
            next_params = self._params_array_to_dict(best_x)
            
            return next_params
    
    def register_result(self, parameters: Dict[str, Any], value: float) -> None:
        """
        Register the result of evaluating a parameter set.
        
        Args:
            parameters: The parameters that were evaluated
            value: The value of the objective function
        """
        if SKOPT_AVAILABLE:
            # Use scikit-optimize backend
            x = []
            
            # Convert params to array in correct order
            for name in self.param_names:
                x.append(parameters[name])
            
            # Register with skopt (negate if maximizing since skopt minimizes)
            self.optimizer.tell(x, -value if not self.minimize else value)
            
        else:
            # Use basic implementation
            x = self._params_dict_to_array(parameters)
            
            # Store observation
            self.X_observed.append(x)
            self.y_observed.append(value)
        
        # Store in history regardless of backend
        self.parameters_history.append({
            'parameters': parameters,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_best_parameters(self) -> Tuple[Dict[str, Any], float]:
        """
        Get the best parameters found so far.
        
        Returns:
            Tuple of (best_parameters, best_value)
        """
        if not self.parameters_history:
            # No evaluations yet, return default parameters
            return self.parameter_space.get_default_parameters(), 0.0
        
        # Find best result
        if self.minimize:
            best_idx = np.argmin([h['value'] for h in self.parameters_history])
        else:
            best_idx = np.argmax([h['value'] for h in self.parameters_history])
        
        best_entry = self.parameters_history[best_idx]
        
        return best_entry['parameters'], best_entry['value']
    
    def optimize(
        self, 
        objective_function: Callable[[Dict[str, Any]], float],
        n_iterations: int = 50,
        n_parallel: int = 1,
        callback: Optional[Callable[[Dict[str, Any], float, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Run the optimization process for a given objective function.
        
        Args:
            objective_function: Function that evaluates parameters
            n_iterations: Number of iterations to run
            n_parallel: Number of parallel evaluations (for batch suggestion)
            callback: Optional callback function after each iteration
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        # If any initial points were provided, use those first
        for iteration in range(n_iterations):
            # Suggest parameters (serially for now)
            suggested_params = self.suggest_next_parameters()
            
            # Evaluate objective
            try:
                value = objective_function(suggested_params)
            except Exception as e:
                logger.error(f"Error evaluating parameters: {str(e)}")
                # Assign a penalty value
                if self.minimize:
                    value = float('inf')
                else:
                    value = float('-inf')
            
            # Register result
            self.register_result(suggested_params, value)
            
            # Call callback if provided
            if callback:
                callback(suggested_params, value, iteration)
            
            logger.info(f"Iteration {iteration+1}/{n_iterations}, Value: {value}")
        
        # Get best result
        best_params, best_value = self.get_best_parameters()
        
        optimization_time = time.time() - start_time
        
        return {
            'best_parameters': best_params,
            'best_value': best_value,
            'n_iterations': n_iterations,
            'optimization_time': optimization_time,
            'all_parameters': self.parameters_history,
            'parameter_space': self.parameter_space.get_parameters_dict()
        }
