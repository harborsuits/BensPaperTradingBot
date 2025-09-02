"""
Strategy Adapter for BensBot-EvoTrader Integration

This module provides adapters to convert BensBot trading strategies to EvoTrader's
format for evolutionary optimization, and convert them back for deployment in BensBot.
"""

import copy
import uuid
import json
import logging
import inspect
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger("benbot.research.evotrader.adapter")

class StrategyParameter:
    """Defines a strategy parameter with evolution constraints."""
    
    def __init__(
        self, 
        name: str, 
        default_value: Any,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        is_mutable: bool = True,
        mutation_factor: float = 0.2
    ):
        """
        Initialize a strategy parameter.
        
        Args:
            name: Parameter name
            default_value: Default parameter value
            min_value: Minimum allowed value (for numeric parameters)
            max_value: Maximum allowed value (for numeric parameters)
            is_mutable: Whether this parameter can be mutated during evolution
            mutation_factor: How much the parameter can change during mutation
        """
        self.name = name
        self.default_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.is_mutable = is_mutable
        self.mutation_factor = mutation_factor
        
        # Infer type
        self.param_type = type(default_value)
        
        # Set min/max values based on type if not provided
        if self.param_type == int and min_value is None:
            self.min_value = 1
        if self.param_type == int and max_value is None:
            self.max_value = 200
            
        if self.param_type == float and min_value is None:
            self.min_value = 0.0
        if self.param_type == float and max_value is None:
            self.max_value = 1.0

class BensBotStrategyAdapter:
    """Adapter that converts BensBot strategies to EvoTrader-compatible strategies."""
    
    def __init__(
        self, 
        benbot_strategy: Any,
        strategy_id: str = None,
        parameters: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize a strategy adapter.
        
        Args:
            benbot_strategy: Original BensBot strategy instance
            strategy_id: Unique identifier for this strategy
            parameters: Strategy parameters dictionary
            metadata: Additional metadata like generation, parents, etc.
        """
        # Generate unique ID if not provided
        self.strategy_id = strategy_id or f"strategy_{str(uuid.uuid4())[:8]}"
        
        # Store original strategy
        self.benbot_strategy = benbot_strategy
        self.strategy_type = type(benbot_strategy).__name__ if benbot_strategy else "Unknown"
        
        # Extract parameters or use provided ones
        self.parameters = parameters or {}
        if benbot_strategy and not self.parameters:
            self.parameters = self._extract_parameters_from_benbot(benbot_strategy)
            
        # Setup metadata
        self.metadata = metadata or {
            "generation": 0,
            "parent_ids": [],
            "creation_timestamp": None,
            "performance_history": [],
            "mutation_history": []
        }
        
        # Parameters metadata
        self.param_metadata = {}
        self._setup_parameter_metadata()
        
        logger.debug(f"Created adapter for strategy: {self.strategy_id} ({self.strategy_type})")
    
    def _extract_parameters_from_benbot(self, benbot_strategy) -> Dict[str, Any]:
        """Extract tunable parameters from a BensBot strategy."""
        params = {}
        
        # First check for explicit parameters attribute
        if hasattr(benbot_strategy, 'parameters') and isinstance(benbot_strategy.parameters, dict):
            params = copy.deepcopy(benbot_strategy.parameters)
            logger.debug(f"Extracted {len(params)} parameters from strategy.parameters")
            return params
            
        # Look for parameter_definitions attribute (common in many strategies)
        if hasattr(benbot_strategy, 'parameter_definitions') and isinstance(benbot_strategy.parameter_definitions, dict):
            for name, definition in benbot_strategy.parameter_definitions.items():
                if isinstance(definition, dict) and 'default' in definition:
                    params[name] = definition['default']
            
            logger.debug(f"Extracted {len(params)} parameters from parameter_definitions")
            return params
            
        # Try to extract directly from attributes
        extracted = 0
        for attr_name in dir(benbot_strategy):
            # Skip private attributes, methods, and certain known attributes
            if (attr_name.startswith('_') or 
                callable(getattr(benbot_strategy, attr_name)) or
                attr_name in ['logger', 'name', 'description']):
                continue
                
            attr_value = getattr(benbot_strategy, attr_name)
            if isinstance(attr_value, (int, float, bool, str)):
                params[attr_name] = attr_value
                extracted += 1
        
        logger.debug(f"Extracted {extracted} parameters from attributes")
        return params
    
    def _setup_parameter_metadata(self):
        """Setup metadata for each parameter (constraints, mutation factors, etc.)"""
        # Try to get parameter definitions from strategy if available
        param_defs = {}
        
        if hasattr(self.benbot_strategy, 'parameter_definitions'):
            param_defs = self.benbot_strategy.parameter_definitions
        
        # Process each parameter
        for name, value in self.parameters.items():
            if name in param_defs and isinstance(param_defs[name], dict):
                # Use definitions from strategy
                definition = param_defs[name]
                self.param_metadata[name] = StrategyParameter(
                    name=name,
                    default_value=value,
                    min_value=definition.get('min'),
                    max_value=definition.get('max'),
                    is_mutable=definition.get('mutable', True),
                    mutation_factor=definition.get('mutation_factor', 0.2)
                )
            else:
                # Create default metadata based on parameter type
                if isinstance(value, int):
                    # Integer parameter
                    self.param_metadata[name] = StrategyParameter(
                        name=name,
                        default_value=value,
                        min_value=max(1, int(value * 0.5)) if value > 0 else 1,
                        max_value=int(value * 2) if value > 0 else 100,
                        is_mutable=True,
                        mutation_factor=0.3
                    )
                elif isinstance(value, float):
                    # Float parameter
                    self.param_metadata[name] = StrategyParameter(
                        name=name,
                        default_value=value,
                        min_value=max(0.001, value * 0.5) if value > 0 else 0.001,
                        max_value=value * 2 if value > 0 else 1.0,
                        is_mutable=True,
                        mutation_factor=0.3
                    )
                elif isinstance(value, bool):
                    # Boolean parameter
                    self.param_metadata[name] = StrategyParameter(
                        name=name,
                        default_value=value,
                        is_mutable=True,
                        mutation_factor=0.2
                    )
                elif isinstance(value, str):
                    # String parameter - consider immutable by default
                    self.param_metadata[name] = StrategyParameter(
                        name=name,
                        default_value=value,
                        is_mutable=False,
                        mutation_factor=0
                    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert adapter to a dictionary for serialization."""
        return {
            "strategy_id": self.strategy_id,
            "strategy_type": self.strategy_type,
            "parameters": self.parameters,
            "metadata": self.metadata,
            # Don't include benbot_strategy as it might not be serializable
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], benbot_strategy_class=None):
        """Create an adapter from a dictionary."""
        benbot_strategy = None
        
        # If strategy class is provided, try to instantiate it
        if benbot_strategy_class:
            try:
                # Check if we need to pass parameters to constructor
                sig = inspect.signature(benbot_strategy_class.__init__)
                if 'parameters' in sig.parameters:
                    benbot_strategy = benbot_strategy_class(parameters=data.get('parameters', {}))
                else:
                    # Create instance without parameters
                    benbot_strategy = benbot_strategy_class()
                    
                    # Set parameters as attributes
                    for name, value in data.get('parameters', {}).items():
                        if hasattr(benbot_strategy, name):
                            setattr(benbot_strategy, name, value)
            except Exception as e:
                logger.error(f"Error creating BensBot strategy: {e}")
        
        # Create adapter
        return cls(
            benbot_strategy=benbot_strategy,
            strategy_id=data.get('strategy_id'),
            parameters=data.get('parameters', {}),
            metadata=data.get('metadata', {})
        )
    
    def get_evolutionary_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Get parameters that can be evolved with their constraints.
        
        Returns:
            Dictionary of parameter definitions with constraints
        """
        evolvable_params = {}
        
        for name, metadata in self.param_metadata.items():
            if metadata.is_mutable:
                evolvable_params[name] = {
                    "value": self.parameters.get(name, metadata.default_value),
                    "type": metadata.param_type.__name__,
                    "min": metadata.min_value,
                    "max": metadata.max_value,
                    "mutation_factor": metadata.mutation_factor
                }
        
        return evolvable_params
    
    def apply_parameters(self, updated_parameters: Dict[str, Any]):
        """
        Apply updated parameters to the strategy.
        
        Args:
            updated_parameters: Dictionary of parameters to update
        """
        # Update parameters dictionary
        self.parameters.update(updated_parameters)
        
        # If we have a BensBot strategy, update it too
        if self.benbot_strategy:
            for name, value in updated_parameters.items():
                # Check if it has a parameters dict
                if hasattr(self.benbot_strategy, 'parameters') and isinstance(self.benbot_strategy.parameters, dict):
                    self.benbot_strategy.parameters[name] = value
                
                # Also set as attribute if it exists
                if hasattr(self.benbot_strategy, name):
                    setattr(self.benbot_strategy, name, value)


def convert_to_benbot_strategy(adapted_strategy, strategy_class=None):
    """
    Convert an adapted strategy back to a BensBot strategy.
    
    Args:
        adapted_strategy: EvoTrader adapted strategy
        strategy_class: Optional BensBot strategy class to instantiate
        
    Returns:
        BensBot strategy instance
    """
    # If no class provided, try to use the class from the original strategy
    if not strategy_class and adapted_strategy.benbot_strategy:
        strategy_class = type(adapted_strategy.benbot_strategy)
    
    if not strategy_class:
        logger.error("No strategy class provided or found in adapted strategy")
        return None
    
    try:
        # Create a new instance
        benbot_strategy = strategy_class()
        
        # Apply parameters
        for name, value in adapted_strategy.parameters.items():
            # Update parameters dictionary if it exists
            if hasattr(benbot_strategy, 'parameters') and isinstance(benbot_strategy.parameters, dict):
                benbot_strategy.parameters[name] = value
            
            # Set attribute if it exists
            if hasattr(benbot_strategy, name):
                setattr(benbot_strategy, name, value)
        
        # Add metadata for tracking
        benbot_strategy.evotrader_metadata = adapted_strategy.metadata
        benbot_strategy.evotrader_id = adapted_strategy.strategy_id
        
        return benbot_strategy
        
    except Exception as e:
        logger.error(f"Error converting to BensBot strategy: {e}")
        return None
