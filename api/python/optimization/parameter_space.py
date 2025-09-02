import numpy as np
from typing import Dict, List, Union, Any, Tuple, Optional, Callable, Iterator
import itertools
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ParameterType(Enum):
    """Types of parameters for optimization"""
    DISCRETE = "discrete"    # List of discrete values
    RANGE = "range"          # Continuous range (min, max)
    INTEGER = "integer"      # Integer range (min, max)
    CATEGORICAL = "categorical"  # Categorical values
    BOOLEAN = "boolean"      # True/False

class Parameter:
    """Definition of a single parameter to optimize"""
    
    def __init__(
        self,
        name: str,
        param_type: ParameterType,
        values: Union[List[Any], Tuple[float, float]],
        default: Any = None,
        description: str = ""
    ):
        """
        Initialize parameter definition
        
        Args:
            name: Parameter name
            param_type: Type of parameter
            values: Parameter values (list for discrete/categorical, tuple for range/integer)
            default: Default value
            description: Parameter description
        """
        self.name = name
        self.param_type = param_type
        self.values = values
        self.description = description
        
        # Validate parameter values based on type
        self._validate_values()
        
        # Set default value if not provided
        if default is None:
            if param_type == ParameterType.DISCRETE or param_type == ParameterType.CATEGORICAL:
                self.default = values[0] if values else None
            elif param_type == ParameterType.RANGE:
                self.default = (values[0] + values[1]) / 2
            elif param_type == ParameterType.INTEGER:
                self.default = int((values[0] + values[1]) / 2)
            elif param_type == ParameterType.BOOLEAN:
                self.default = False
        else:
            self.default = default
    
    def _validate_values(self) -> None:
        """Validate parameter values based on type"""
        if self.param_type == ParameterType.DISCRETE or self.param_type == ParameterType.CATEGORICAL:
            if not isinstance(self.values, (list, tuple)) or len(self.values) == 0:
                raise ValueError(f"Parameter {self.name}: Discrete/categorical parameter must have a non-empty list of values")
        
        elif self.param_type == ParameterType.RANGE:
            if not isinstance(self.values, (list, tuple)) or len(self.values) != 2:
                raise ValueError(f"Parameter {self.name}: Range parameter must have a tuple of (min, max)")
            if self.values[0] >= self.values[1]:
                raise ValueError(f"Parameter {self.name}: Range parameter min must be less than max")
        
        elif self.param_type == ParameterType.INTEGER:
            if not isinstance(self.values, (list, tuple)) or len(self.values) != 2:
                raise ValueError(f"Parameter {self.name}: Integer parameter must have a tuple of (min, max)")
            if self.values[0] >= self.values[1]:
                raise ValueError(f"Parameter {self.name}: Integer parameter min must be less than max")
        
        elif self.param_type == ParameterType.BOOLEAN:
            self.values = [True, False]
    
    def sample(self) -> Any:
        """Sample a random value from the parameter space"""
        if self.param_type == ParameterType.DISCRETE or self.param_type == ParameterType.CATEGORICAL:
            return np.random.choice(self.values)
        
        elif self.param_type == ParameterType.RANGE:
            return np.random.uniform(self.values[0], self.values[1])
        
        elif self.param_type == ParameterType.INTEGER:
            return np.random.randint(self.values[0], self.values[1] + 1)
        
        elif self.param_type == ParameterType.BOOLEAN:
            return np.random.choice([True, False])
    
    def get_grid_values(self, num_points: int = 10) -> List[Any]:
        """
        Get evenly spaced values for grid search
        
        Args:
            num_points: Number of points to sample for range/integer parameters
            
        Returns:
            List of parameter values for grid search
        """
        if self.param_type == ParameterType.DISCRETE or self.param_type == ParameterType.CATEGORICAL:
            return self.values
        
        elif self.param_type == ParameterType.RANGE:
            return list(np.linspace(self.values[0], self.values[1], num_points))
        
        elif self.param_type == ParameterType.INTEGER:
            if self.values[1] - self.values[0] + 1 <= num_points:
                # If range is smaller than num_points, use all integers in range
                return list(range(self.values[0], self.values[1] + 1))
            else:
                # Otherwise, sample evenly spaced integers
                return [int(x) for x in np.linspace(self.values[0], self.values[1], num_points)]
        
        elif self.param_type == ParameterType.BOOLEAN:
            return [True, False]
    
    def __repr__(self) -> str:
        if self.param_type == ParameterType.DISCRETE or self.param_type == ParameterType.CATEGORICAL:
            values_str = f"{len(self.values)} values"
        elif self.param_type == ParameterType.RANGE or self.param_type == ParameterType.INTEGER:
            values_str = f"[{self.values[0]}, {self.values[1]}]"
        elif self.param_type == ParameterType.BOOLEAN:
            values_str = "True/False"
        
        return f"Parameter({self.name}, {self.param_type.value}, {values_str})"


class ParameterSpace:
    """Defines the space of parameters to optimize"""
    
    def __init__(self, parameters: List[Parameter] = None):
        """
        Initialize parameter space
        
        Args:
            parameters: List of parameter definitions
        """
        self.parameters = parameters or []
        self._param_dict = {param.name: param for param in self.parameters}
    
    def add_parameter(self, parameter: Parameter) -> None:
        """
        Add a parameter to the space
        
        Args:
            parameter: Parameter definition
        """
        self.parameters.append(parameter)
        self._param_dict[parameter.name] = parameter
    
    def add_discrete_parameter(
        self, 
        name: str, 
        values: List[Any], 
        default: Any = None, 
        description: str = ""
    ) -> None:
        """
        Add a discrete parameter
        
        Args:
            name: Parameter name
            values: List of discrete values
            default: Default value
            description: Parameter description
        """
        param = Parameter(
            name=name,
            param_type=ParameterType.DISCRETE,
            values=values,
            default=default,
            description=description
        )
        self.add_parameter(param)
    
    def add_range_parameter(
        self, 
        name: str, 
        min_value: float, 
        max_value: float, 
        default: float = None, 
        description: str = ""
    ) -> None:
        """
        Add a continuous range parameter
        
        Args:
            name: Parameter name
            min_value: Minimum value
            max_value: Maximum value
            default: Default value
            description: Parameter description
        """
        param = Parameter(
            name=name,
            param_type=ParameterType.RANGE,
            values=(min_value, max_value),
            default=default,
            description=description
        )
        self.add_parameter(param)
    
    def add_integer_parameter(
        self, 
        name: str, 
        min_value: int, 
        max_value: int, 
        default: int = None, 
        description: str = ""
    ) -> None:
        """
        Add an integer parameter
        
        Args:
            name: Parameter name
            min_value: Minimum value
            max_value: Maximum value
            default: Default value
            description: Parameter description
        """
        param = Parameter(
            name=name,
            param_type=ParameterType.INTEGER,
            values=(min_value, max_value),
            default=default,
            description=description
        )
        self.add_parameter(param)
    
    def add_categorical_parameter(
        self, 
        name: str, 
        categories: List[Any], 
        default: Any = None, 
        description: str = ""
    ) -> None:
        """
        Add a categorical parameter
        
        Args:
            name: Parameter name
            categories: List of categories
            default: Default value
            description: Parameter description
        """
        param = Parameter(
            name=name,
            param_type=ParameterType.CATEGORICAL,
            values=categories,
            default=default,
            description=description
        )
        self.add_parameter(param)
    
    def add_boolean_parameter(
        self, 
        name: str, 
        default: bool = False, 
        description: str = ""
    ) -> None:
        """
        Add a boolean parameter
        
        Args:
            name: Parameter name
            default: Default value
            description: Parameter description
        """
        param = Parameter(
            name=name,
            param_type=ParameterType.BOOLEAN,
            values=[True, False],
            default=default,
            description=description
        )
        self.add_parameter(param)
    
    def get_parameter(self, name: str) -> Parameter:
        """
        Get parameter by name
        
        Args:
            name: Parameter name
            
        Returns:
            Parameter definition
        """
        if name not in self._param_dict:
            raise ValueError(f"Parameter '{name}' not found")
        return self._param_dict[name]
    
    def get_default_params(self) -> Dict[str, Any]:
        """
        Get dictionary of default parameter values
        
        Returns:
            Dictionary of parameter names and default values
        """
        return {param.name: param.default for param in self.parameters}
    
    def sample(self) -> Dict[str, Any]:
        """
        Sample a random point from the parameter space
        
        Returns:
            Dictionary of parameter names and sampled values
        """
        return {param.name: param.sample() for param in self.parameters}
    
    def grid_search_iterator(self, num_points: int = 10) -> Iterator[Dict[str, Any]]:
        """
        Iterator for grid search of the parameter space
        
        Args:
            num_points: Number of points to sample for range/integer parameters
            
        Returns:
            Iterator of parameter dictionaries
        """
        # Get grid values for each parameter
        param_values = {}
        for param in self.parameters:
            param_values[param.name] = param.get_grid_values(num_points)
        
        # Names of parameters (to preserve order)
        param_names = [param.name for param in self.parameters]
        
        # Calculate total number of combinations
        total_combinations = 1
        for values in param_values.values():
            total_combinations *= len(values)
        
        logger.info(f"Grid search will evaluate {total_combinations} parameter combinations")
        
        # Create iterator for the Cartesian product of all parameter values
        value_lists = [param_values[name] for name in param_names]
        for combination in itertools.product(*value_lists):
            yield {name: value for name, value in zip(param_names, combination)}
    
    def __len__(self) -> int:
        """Get number of parameters in the space"""
        return len(self.parameters)
    
    def __repr__(self) -> str:
        param_strs = [str(param) for param in self.parameters]
        return f"ParameterSpace({len(self.parameters)} parameters: {', '.join(param_strs)})" 