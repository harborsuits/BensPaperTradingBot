"""
Strategy Adapter for BensBot-EvoTrader Integration

This module provides adapters to connect BensBot's trading strategies with
EvoTrader's evolutionary framework. It allows BensBot strategies to be evolved,
mutated, and promoted based on performance metrics.
"""

import copy
import uuid
import json
import logging
import os
from typing import Dict, List, Any, Optional, Tuple, Type, Union

# Import EvoTrader components
from evotrader.core.strategy import Strategy as EvoStrategy
from evotrader.core.challenge_bot import ChallengeBot
from evotrader.utils.evolution import apply_mutation, crossover_strategies

logger = logging.getLogger(__name__)

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
        self.name = name
        self.default_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.is_mutable = is_mutable
        self.mutation_factor = mutation_factor
        

class BensBotStrategyAdapter(EvoStrategy):
    """Adapter that converts BensBot strategies to EvoTrader-compatible strategies."""
    
    def __init__(
        self, 
        strategy_id: str = None,
        benbot_strategy: Any = None,
        parameters: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize a strategy adapter.
        
        Args:
            strategy_id: Unique identifier for this strategy
            benbot_strategy: Original BensBot strategy instance
            parameters: Strategy parameters dictionary
            metadata: Additional metadata like generation, parents, etc.
        """
        # Generate unique ID if not provided
        if strategy_id is None:
            strategy_id = f"strategy_{str(uuid.uuid4())[:8]}"
            
        super().__init__(strategy_id)
        
        # Initialize BensBot strategy reference
        self.benbot_strategy = benbot_strategy
        
        # Copy parameters or initialize defaults
        self.parameters = parameters or {}
        if benbot_strategy and not self.parameters:
            self.parameters = self._extract_parameters_from_benbot(benbot_strategy)
            
        # Add evolutionary metadata
        self.metadata = metadata or {
            "generation": 0,
            "parent_ids": [],
            "creation_timestamp": None,
            "performance_history": [],
            "mutation_history": []
        }
    
    def _extract_parameters_from_benbot(self, benbot_strategy) -> Dict[str, Any]:
        """Extract tunable parameters from a BensBot strategy."""
        params = {}
        # Extract parameters from the strategy
        # This will need to be customized based on BensBot's strategy structure
        if hasattr(benbot_strategy, 'parameters'):
            params = benbot_strategy.parameters.copy()
        else:
            # Extract parameters from attributes
            # This is a generic approach that should be customized
            for attr_name in dir(benbot_strategy):
                if attr_name.startswith('_'):
                    continue
                    
                attr_value = getattr(benbot_strategy, attr_name)
                if isinstance(attr_value, (int, float, bool, str)) and not callable(attr_value):
                    params[attr_name] = attr_value
        
        return params
    
    def apply_parameters_to_benbot(self) -> Any:
        """Apply evolved parameters to the BensBot strategy instance."""
        if self.benbot_strategy is None:
            logger.warning(f"Cannot apply parameters - no BensBot strategy instance")
            return None
            
        # Apply parameters to BensBot strategy
        if hasattr(self.benbot_strategy, 'parameters'):
            # If strategy has a parameters dict, update it
            self.benbot_strategy.parameters.update(self.parameters)
        else:
            # Otherwise set attributes directly
            for param_name, param_value in self.parameters.items():
                if hasattr(self.benbot_strategy, param_name):
                    setattr(self.benbot_strategy, param_name, param_value)
        
        # If BensBot strategy has an initialization method, call it
        if hasattr(self.benbot_strategy, 'initialize') and callable(getattr(self.benbot_strategy, 'initialize')):
            self.benbot_strategy.initialize()
            
        return self.benbot_strategy
    
    def get_parameters(self) -> List[StrategyParameter]:
        """Get list of parameter definitions with evolution constraints."""
        # This should be customized for each strategy type
        # Default implementation returns basic constraints
        params = []
        for name, value in self.parameters.items():
            param = StrategyParameter(
                name=name,
                default_value=value,
                is_mutable=True
            )
            
            # Add reasonable constraints based on value type
            if isinstance(value, (int, float)):
                # Set reasonable min/max bounds
                if value != 0:
                    param.min_value = value * 0.1  # 10% of original
                    param.max_value = value * 10   # 10x original
                else:
                    param.min_value = -10
                    param.max_value = 10
                    
            params.append(param)
        
        return params
    
    def calculate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate trading signal from market data."""
        # Delegate to BensBot strategy if available
        if self.benbot_strategy is None:
            logger.warning(f"Cannot calculate signal - no BensBot strategy instance")
            return {"signal": "none"}
            
        # Apply current parameters to the strategy
        self.apply_parameters_to_benbot()
        
        # Calculate signal using BensBot's strategy
        try:
            if hasattr(self.benbot_strategy, 'calculate_signal') and callable(getattr(self.benbot_strategy, 'calculate_signal')):
                signal = self.benbot_strategy.calculate_signal(market_data)
            elif hasattr(self.benbot_strategy, 'generate_signal') and callable(getattr(self.benbot_strategy, 'generate_signal')):
                signal = self.benbot_strategy.generate_signal(market_data)
            elif hasattr(self.benbot_strategy, 'get_signal') and callable(getattr(self.benbot_strategy, 'get_signal')):
                signal = self.benbot_strategy.get_signal(market_data)
            else:
                signal = {"signal": "none"}
                
            return signal
        except Exception as e:
            logger.error(f"Error calculating signal: {str(e)}")
            return {"signal": "none"}
    
    def generate_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals from market data.
        
        This method is required by the EvoStrategy abstract base class.
        It delegates to our calculate_signal method for consistency.
        
        Args:
            market_data: Current market data snapshot
            
        Returns:
            Signal dictionary
        """
        return self.calculate_signal(market_data)
        
    def clone_with_mutation(self, mutation_rate: float = 0.1) -> 'BensBotStrategyAdapter':
        """Create a mutated clone of this strategy for evolution."""
        # Create new ID for the cloned strategy
        new_id = f"strategy_{str(uuid.uuid4())[:8]}"
        
        # Deep copy the parameters
        new_params = copy.deepcopy(self.parameters)
        
        # Copy metadata and update for new generation
        new_metadata = copy.deepcopy(self.metadata)
        new_metadata["generation"] = new_metadata.get("generation", 0) + 1
        new_metadata["parent_ids"] = [self.strategy_id]
        
        # Create new adapter instance
        cloned_strategy = BensBotStrategyAdapter(
            strategy_id=new_id,
            parameters=new_params,
            metadata=new_metadata
        )
        
        # Apply mutation to the parameters
        apply_mutation(cloned_strategy, mutation_rate)
        
        # Track mutation in history
        mutation_event = {
            "timestamp": None,  # This should be filled with actual timestamp
            "parent_id": self.strategy_id,
            "mutation_rate": mutation_rate
        }
        cloned_strategy.metadata["mutation_history"].append(mutation_event)
        
        # Clone the actual BensBot strategy if available
        if self.benbot_strategy is not None:
            cloned_strategy.benbot_strategy = copy.deepcopy(self.benbot_strategy)
            cloned_strategy.apply_parameters_to_benbot()
        
        return cloned_strategy
    
    def create_hybrid(self, other_strategy: 'BensBotStrategyAdapter', crossover_rate: float = 0.3) -> 'BensBotStrategyAdapter':
        """Create a hybrid strategy by combining parameters with another strategy."""
        # Use EvoTrader's crossover function
        hybrid_strategy = crossover_strategies(self, other_strategy, crossover_rate)
        
        # Update metadata for the hybrid
        hybrid_strategy.metadata["generation"] = max(
            self.metadata.get("generation", 0),
            other_strategy.metadata.get("generation", 0)
        ) + 1
        hybrid_strategy.metadata["parent_ids"] = [self.strategy_id, other_strategy.strategy_id]
        
        # Track crossover in history
        crossover_event = {
            "timestamp": None,  # This should be filled with actual timestamp
            "parent_ids": [self.strategy_id, other_strategy.strategy_id],
            "crossover_rate": crossover_rate
        }
        hybrid_strategy.metadata["mutation_history"].append(crossover_event)
        
        # Create a new BensBot strategy instance if possible
        if self.benbot_strategy is not None:
            hybrid_strategy.benbot_strategy = copy.deepcopy(self.benbot_strategy)
            hybrid_strategy.apply_parameters_to_benbot()
        
        return hybrid_strategy
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary for serialization."""
        return {
            "strategy_id": self.strategy_id,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "strategy_type": self.benbot_strategy.__class__.__name__ if self.benbot_strategy else "Unknown"
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BensBotStrategyAdapter':
        """Create strategy instance from dictionary."""
        return cls(
            strategy_id=data.get("strategy_id"),
            parameters=data.get("parameters", {}),
            metadata=data.get("metadata", {})
        )
    
    def save_to_file(self, filepath: str) -> bool:
        """Save strategy to a JSON file."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving strategy to {filepath}: {str(e)}")
            return False
    
    @classmethod
    def load_from_file(cls, filepath: str) -> Optional['BensBotStrategyAdapter']:
        """Load strategy from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading strategy from {filepath}: {str(e)}")
            return None
