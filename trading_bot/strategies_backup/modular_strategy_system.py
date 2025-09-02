"""
Modular Strategy System

Implements a sophisticated, composable strategy architecture inspired by EA31337-Libre.
Key features:
- Strategy composition and building block approach
- Conditional strategy activation based on market conditions
- Parameter optimization framework
- Strategy components with clean interfaces
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Type, Callable
import abc
import logging
import json
import time
from datetime import datetime
from enum import Enum, auto
import uuid
import copy
import inspect

from trading_bot.strategies.base_strategy import Strategy, SignalType, Position
from trading_bot.event_system.event_handler import EventType
from trading_bot.event_system import Event

logger = logging.getLogger(__name__)

class MarketCondition(Enum):
    """Market condition types for conditional strategy activation"""
    TRENDING = auto()           # Strong trend conditions
    RANGING = auto()            # Sideways/range-bound markets
    VOLATILE = auto()           # High volatility environment
    LOW_VOLATILITY = auto()     # Low volatility environment
    BULLISH = auto()            # Strong bullish bias
    BEARISH = auto()            # Strong bearish bias
    BREAKOUT = auto()           # Breakout conditions
    REVERSAL = auto()           # Potential reversal conditions
    HIGH_VOLUME = auto()        # High volume conditions
    LOW_VOLUME = auto()         # Low volume conditions
    NORMAL = auto()             # Default normal conditions
    CUSTOM = auto()             # Custom user-defined condition
    
class ComponentType(Enum):
    """Types of strategy components"""
    SIGNAL_GENERATOR = auto()   # Generates entry/exit signals
    FILTER = auto()             # Filters signals based on conditions
    POSITION_SIZER = auto()     # Handles position sizing logic
    RISK_MANAGER = auto()       # Manages risk parameters
    ENTRY_MANAGER = auto()      # Manages entry execution
    EXIT_MANAGER = auto()       # Manages exit conditions and execution
    INDICATOR = auto()          # Calculates technical indicators
    TIMEFRAME_MANAGER = auto()  # Handles timeframe analysis
    CONDITION_DETECTOR = auto() # Detects specific market conditions
    CUSTOM = auto()             # Custom user-defined component
    
class ActivationRule(Enum):
    """Strategy activation rule types"""
    ALWAYS = auto()             # Always active
    TIME_BASED = auto()         # Active during specific times/days
    CONDITION_BASED = auto()    # Active during specific market conditions
    INDICATOR_BASED = auto()    # Active based on indicator values
    PERFORMANCE_BASED = auto()  # Active based on historical performance
    HYBRID = auto()             # Combination of multiple criteria
    
class OptimizationType(Enum):
    """Types of optimization approaches"""
    GRID_SEARCH = auto()        # Exhaustive grid search
    RANDOM_SEARCH = auto()      # Random parameter sampling
    BAYESIAN = auto()           # Bayesian optimization
    GENETIC = auto()            # Genetic algorithm optimization
    WALK_FORWARD = auto()       # Walk-forward optimization
    MONTE_CARLO = auto()        # Monte Carlo simulation
    ADAPTIVE = auto()           # Adaptive optimization

class StrategyComponent(abc.ABC):
    """Base class for all modular strategy components"""
    
    def __init__(self, component_id: Optional[str] = None, component_type: ComponentType = ComponentType.CUSTOM):
        """
        Initialize a strategy component
        
        Args:
            component_id: Unique identifier (auto-generated if not provided)
            component_type: Type of component
        """
        self.component_id = component_id or f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        self.component_type = component_type
        self.parameters = {}
        self.enabled = True
        self.parent_strategy = None
        self.description = ""
        self.metadata = {}
        
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set component parameters
        
        Args:
            parameters: Dictionary of parameters
        """
        self.parameters.update(parameters)
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get component parameters
        
        Returns:
            Dictionary of current parameters
        """
        return self.parameters.copy()
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about the component's parameters including ranges, types
        
        Returns:
            Dictionary of parameter information
        """
        param_info = {}
        
        # Get signature of the constructor
        signature = inspect.signature(self.__init__)
        
        for param_name, param in signature.parameters.items():
            if param_name not in ['self', 'component_id', 'component_type']:
                param_info[param_name] = {
                    'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any',
                    'default': None if param.default == inspect.Parameter.empty else param.default,
                    'description': '',  # Would come from docstring parsing in a full implementation
                }
        
        # Add any runtime parameters
        for param_name, value in self.parameters.items():
            if param_name not in param_info:
                param_info[param_name] = {
                    'type': str(type(value)),
                    'default': value,
                    'description': '',
                }
        
        return param_info
    
    def set_parent_strategy(self, strategy: Any) -> None:
        """
        Set the parent strategy that owns this component
        
        Args:
            strategy: Parent strategy instance
        """
        self.parent_strategy = strategy
    
    def is_enabled(self) -> bool:
        """
        Check if component is enabled
        
        Returns:
            True if component is enabled, False otherwise
        """
        return self.enabled
    
    def enable(self) -> None:
        """Enable the component"""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable the component"""
        self.enabled = False
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set metadata value
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata value
        
        Args:
            key: Metadata key
            default: Default value if key not found
            
        Returns:
            Metadata value or default
        """
        return self.metadata.get(key, default)
    
    @abc.abstractmethod
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Process data and return result
        
        Args:
            data: Input data
            context: Processing context
            
        Returns:
            Processed result
        """
        pass
        
    def clone(self) -> 'StrategyComponent':
        """
        Create a copy of this component
        
        Returns:
            Cloned component instance
        """
        clone = copy.deepcopy(self)
        clone.component_id = f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        clone.parent_strategy = None
        return clone
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert component to dictionary representation
        
        Returns:
            Dictionary representation
        """
        return {
            'component_id': self.component_id,
            'component_type': self.component_type.name,
            'component_class': self.__class__.__name__,
            'parameters': self.parameters,
            'enabled': self.enabled,
            'description': self.description,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyComponent':
        """
        Create component from dictionary representation
        
        Args:
            data: Dictionary representation
            
        Returns:
            Component instance
        """
        # This would need to use reflection to instantiate the correct class
        # Implementation depends on how components are registered
        raise NotImplementedError("Subclasses should implement from_dict")

class SignalGeneratorComponent(StrategyComponent):
    """Base class for signal generator components"""
    
    def __init__(self, component_id: Optional[str] = None):
        """
        Initialize signal generator
        
        Args:
            component_id: Unique component ID
        """
        super().__init__(component_id, ComponentType.SIGNAL_GENERATOR)
    
    @abc.abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame], context: Dict[str, Any]) -> Dict[str, SignalType]:
        """
        Generate trading signals
        
        Args:
            data: Market data dictionary (symbol -> DataFrame)
            context: Signal generation context
            
        Returns:
            Dictionary of symbol -> signal type
        """
        pass
    
    def process(self, data: Dict[str, pd.DataFrame], context: Dict[str, Any]) -> Dict[str, SignalType]:
        """
        Process data to generate signals
        
        Args:
            data: Market data
            context: Processing context
            
        Returns:
            Generated signals
        """
        return self.generate_signals(data, context)

class FilterComponent(StrategyComponent):
    """Base class for signal filter components"""
    
    def __init__(self, component_id: Optional[str] = None):
        """
        Initialize filter component
        
        Args:
            component_id: Unique component ID
        """
        super().__init__(component_id, ComponentType.FILTER)
    
    @abc.abstractmethod
    def filter_signals(self, signals: Dict[str, SignalType], data: Dict[str, pd.DataFrame], 
                      context: Dict[str, Any]) -> Dict[str, SignalType]:
        """
        Filter trading signals
        
        Args:
            signals: Input signals to filter
            data: Market data
            context: Filtering context
            
        Returns:
            Filtered signals
        """
        pass
    
    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, SignalType]:
        """
        Process signals with this filter
        
        Args:
            data: Tuple of (signals, market_data)
            context: Processing context
            
        Returns:
            Filtered signals
        """
        if isinstance(data, tuple) and len(data) == 2:
            signals, market_data = data
            return self.filter_signals(signals, market_data, context)
        else:
            logger.error(f"Invalid data format for filter component: {type(data)}")
            return {}

class PositionSizerComponent(StrategyComponent):
    """Base class for position sizer components"""
    
    def __init__(self, component_id: Optional[str] = None):
        """
        Initialize position sizer
        
        Args:
            component_id: Unique component ID
        """
        super().__init__(component_id, ComponentType.POSITION_SIZER)
    
    @abc.abstractmethod
    def calculate_position_size(self, symbol: str, signal: SignalType, price: float, 
                               context: Dict[str, Any]) -> float:
        """
        Calculate position size
        
        Args:
            symbol: Trading symbol
            signal: Signal type
            price: Current price
            context: Position sizing context
            
        Returns:
            Position size
        """
        pass
    
    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Process position sizing requests
        
        Args:
            data: Dictionary of symbol -> (signal, price) tuples
            context: Processing context
            
        Returns:
            Dictionary of symbol -> position size
        """
        if not isinstance(data, dict):
            logger.error(f"Invalid data format for position sizer: {type(data)}")
            return {}
        
        position_sizes = {}
        for symbol, (signal, price) in data.items():
            position_sizes[symbol] = self.calculate_position_size(symbol, signal, price, context)
        
        return position_sizes

class ExitManagerComponent(StrategyComponent):
    """Base class for exit manager components"""
    
    def __init__(self, component_id: Optional[str] = None):
        """
        Initialize exit manager
        
        Args:
            component_id: Unique component ID
        """
        super().__init__(component_id, ComponentType.EXIT_MANAGER)
    
    @abc.abstractmethod
    def calculate_exit_parameters(self, symbol: str, position: Position, 
                                data: Dict[str, pd.DataFrame], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate exit parameters for a position
        
        Args:
            symbol: Trading symbol
            position: Current position
            data: Market data
            context: Exit calculation context
            
        Returns:
            Dictionary with exit parameters (stop_loss, take_profit, etc.)
        """
        pass
    
    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Process exit parameter calculations
        
        Args:
            data: Dictionary of symbol -> position
            context: Processing context
            
        Returns:
            Dictionary of symbol -> exit parameters
        """
        if not isinstance(data, dict):
            logger.error(f"Invalid data format for exit manager: {type(data)}")
            return {}
        
        exit_params = {}
        for symbol, position in data.items():
            market_data = context.get('market_data', {})
            exit_params[symbol] = self.calculate_exit_parameters(symbol, position, market_data, context)
        
        return exit_params
