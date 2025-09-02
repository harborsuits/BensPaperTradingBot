#!/usr/bin/env python3
"""
BensBot-EvoTrader Strategy Execution Adapter

This module allows EvoTrader strategies to be executed in the BensBot environment.
Acts as a compatibility layer for strategy execution without modifying either system.
"""

import os
import sys
import logging
import json
import importlib.util
from typing import Dict, List, Any, Optional, Union, Tuple

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import from EvoTrader components
from benbot.evotrader_bridge.data_format_adapter import DataFormatAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('strategy_execution_adapter')

class StrategyExecutionAdapter:
    """
    Allows EvoTrader evolved strategies to be executed within BensBot.
    
    Acts as a bridge between:
    1. EvoTrader's evolutionary strategy creation system
    2. BensBot's strategy execution environment
    """
    
    def __init__(self):
        """Initialize the strategy execution adapter."""
        logger.info("Initializing strategy execution adapter")
        self.data_adapter = DataFormatAdapter()
        
        # Cache for loaded strategy modules
        self.strategy_cache = {}
    
    def load_evotrader_strategy(self, 
                              strategy_path: str, 
                              strategy_id: str = None) -> Dict[str, Any]:
        """
        Load an EvoTrader strategy module.
        
        Args:
            strategy_path: Path to the strategy file
            strategy_id: Optional identifier for the strategy
            
        Returns:
            Dictionary with loaded strategy functions and metadata
        """
        try:
            logger.info(f"Loading EvoTrader strategy from: {strategy_path}")
            
            # Check if file exists
            if not os.path.exists(strategy_path):
                logger.error(f"Strategy file not found: {strategy_path}")
                return {}
            
            # Generate a strategy ID if not provided
            if not strategy_id:
                strategy_id = os.path.basename(strategy_path).replace('.py', '')
            
            # Check cache
            if strategy_id in self.strategy_cache:
                logger.info(f"Using cached strategy: {strategy_id}")
                return self.strategy_cache[strategy_id]
            
            # Import module
            module_name = os.path.basename(strategy_path).replace('.py', '')
            spec = importlib.util.spec_from_file_location(module_name, strategy_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for required functions
            required_functions = ['initialize', 'calculate_signal']
            
            for func_name in required_functions:
                if not hasattr(module, func_name):
                    logger.error(f"Required function '{func_name}' not found in strategy: {strategy_path}")
                    return {}
            
            # Extract strategy functions
            strategy = {
                'id': strategy_id,
                'path': strategy_path,
                'module': module,
                'initialize': getattr(module, 'initialize'),
                'calculate_signal': getattr(module, 'calculate_signal')
            }
            
            # Look for additional helper functions
            for attr_name in dir(module):
                if attr_name.startswith('_') or attr_name in strategy:
                    continue
                
                attr = getattr(module, attr_name)
                if callable(attr):
                    strategy[attr_name] = attr
            
            # Cache strategy
            self.strategy_cache[strategy_id] = strategy
            
            logger.info(f"Successfully loaded EvoTrader strategy: {strategy_id}")
            return strategy
            
        except Exception as e:
            logger.error(f"Error loading EvoTrader strategy: {e}")
            return {}
    
    def create_benbot_strategy_adapter(self, 
                                     evotrader_strategy: Dict[str, Any],
                                     benbot_base_class=None) -> type:
        """
        Create a BensBot-compatible strategy class from an EvoTrader strategy.
        
        Args:
            evotrader_strategy: Loaded EvoTrader strategy
            benbot_base_class: Optional BensBot base strategy class
            
        Returns:
            BensBot-compatible strategy class
        """
        try:
            logger.info(f"Creating BensBot adapter for strategy: {evotrader_strategy.get('id', 'unknown')}")
            
            # Create a context object to hold strategy state
            class StrategyContext:
                pass
            
            strategy_id = evotrader_strategy.get('id', 'unknown')
            
            # Define the adapter class
            class BenBotStrategyAdapter:
                """
                BensBot-compatible adapter for EvoTrader strategies.
                
                This class wraps an EvoTrader strategy and makes it compatible
                with BensBot's strategy interface.
                """
                
                def __init__(
                    self,
                    name: str = None,
                    parameters: Optional[Dict[str, Any]] = None,
                    metadata: Optional[Dict[str, Any]] = None
                ):
                    """
                    Initialize the strategy adapter.
                    
                    Args:
                        name: Strategy name
                        parameters: Strategy parameters
                        metadata: Strategy metadata
                    """
                    self.name = name or f"EvoTrader_{strategy_id}"
                    self.parameters = parameters or {}
                    self.metadata = metadata or {
                        'source': 'evotrader',
                        'strategy_id': strategy_id
                    }
                    
                    # Create context for EvoTrader strategy
                    self.context = StrategyContext()
                    
                    # Initialize strategy
                    if evotrader_strategy.get('initialize'):
                        evotrader_strategy['initialize'](self.context)
                        
                        # If parameters were provided, override context attributes
                        if parameters:
                            for key, value in parameters.items():
                                setattr(self.context, key, value)
                    
                    # Add a reference to the original strategy
                    self.evotrader_strategy = evotrader_strategy
                    
                    # Setup data adapter
                    self.data_adapter = DataFormatAdapter()
                    
                    # Initialize state
                    self.last_signal = 0
                    self.positions = []
                
                def analyze(self, data):
                    """
                    Run strategy analysis on market data.
                    
                    Args:
                        data: Market data in BensBot format
                        
                    Returns:
                        Analysis result
                    """
                    try:
                        # Convert data format
                        evotrader_data = self.data_adapter.benbot_to_evotrader_market_data(data)
                        
                        # Call EvoTrader strategy
                        signal = evotrader_strategy['calculate_signal'](self.context, evotrader_data)
                        
                        # Store last signal
                        self.last_signal = signal
                        
                        # Return compatible result
                        return {
                            'signal': signal,
                            'signal_strength': abs(signal),
                            'direction': 'long' if signal > 0 else 'short' if signal < 0 else 'neutral'
                        }
                        
                    except Exception as e:
                        logger.error(f"Error in strategy analysis: {e}")
                        return {
                            'signal': 0,
                            'signal_strength': 0,
                            'direction': 'neutral',
                            'error': str(e)
                        }
                
                def generate_signal(self, data, context=None):
                    """
                    Generate trading signal.
                    
                    Args:
                        data: Market data in BensBot format
                        context: BensBot execution context
                        
                    Returns:
                        Trading signal
                    """
                    try:
                        # Convert data format
                        evotrader_data = self.data_adapter.benbot_to_evotrader_market_data(data)
                        
                        # Call EvoTrader strategy
                        signal = evotrader_strategy['calculate_signal'](self.context, evotrader_data)
                        
                        # Store last signal
                        self.last_signal = signal
                        
                        # Map signal to BensBot format
                        if hasattr(self.context, 'strategy_type'):
                            strategy_type = self.context.strategy_type
                        else:
                            strategy_type = "unknown"
                        
                        # Return BensBot compatible signal
                        return {
                            'type': 'MARKET',
                            'side': 'BUY' if signal > 0 else 'SELL' if signal < 0 else None,
                            'strength': abs(signal),
                            'strategy_type': strategy_type,
                            'timestamp': data.index[-1] if hasattr(data, 'index') else None,
                            'metadata': {
                                'source': 'evotrader',
                                'strategy_id': strategy_id
                            }
                        }
                        
                    except Exception as e:
                        logger.error(f"Error generating signal: {e}")
                        return {
                            'type': 'MARKET',
                            'side': None,
                            'strength': 0,
                            'error': str(e)
                        }
                
                def process_parameters(self, parameters):
                    """
                    Process strategy parameters.
                    
                    Args:
                        parameters: New parameters
                        
                    Returns:
                        Processed parameters
                    """
                    # Convert parameters if needed
                    evotrader_params = self.data_adapter.benbot_to_evotrader_strategy_params(parameters)
                    
                    # Update context with new parameters
                    for key, value in evotrader_params.items():
                        setattr(self.context, key, value)
                    
                    self.parameters = parameters
                    return parameters
                
                def get_info(self):
                    """Get strategy information."""
                    info = {
                        'name': self.name,
                        'type': getattr(self.context, 'strategy_type', 'unknown'),
                        'parameters': self.parameters,
                        'metadata': self.metadata,
                        'source': 'evotrader'
                    }
                    
                    # Include strategy context attributes
                    context_attributes = {}
                    for attr in dir(self.context):
                        if not attr.startswith('_') and not callable(getattr(self.context, attr)):
                            context_attributes[attr] = getattr(self.context, attr)
                    
                    info['context'] = context_attributes
                    return info
                
                def reset(self):
                    """Reset strategy state."""
                    # Create new context
                    self.context = StrategyContext()
                    
                    # Re-initialize
                    if evotrader_strategy.get('initialize'):
                        evotrader_strategy['initialize'](self.context)
                        
                        # If parameters were provided, override context attributes
                        if self.parameters:
                            for key, value in self.parameters.items():
                                setattr(self.context, key, value)
                    
                    self.last_signal = 0
                    self.positions = []
            
            # If a BensBot base class is provided, inherit from it
            if benbot_base_class:
                # Create a new class that inherits from both
                class BenBotCompatibleStrategy(benbot_base_class, BenBotStrategyAdapter):
                    def __init__(self, *args, **kwargs):
                        benbot_base_class.__init__(self, *args, **kwargs)
                        BenBotStrategyAdapter.__init__(self, *args, **kwargs)
                        
                # Return the compatible class
                adapter_class = BenBotCompatibleStrategy
            else:
                adapter_class = BenBotStrategyAdapter
            
            logger.info(f"Successfully created BensBot adapter for strategy: {strategy_id}")
            return adapter_class
            
        except Exception as e:
            logger.error(f"Error creating BensBot strategy adapter: {e}")
            return None
    
    def create_evotrader_executor(self, 
                                benbot_strategy_class, 
                                evotrader_strategy: Dict[str, Any]) -> object:
        """
        Create an executor that allows a BensBot strategy to be executed by EvoTrader.
        
        Args:
            benbot_strategy_class: BensBot strategy class
            evotrader_strategy: EvoTrader strategy information
            
        Returns:
            Executor object
        """
        try:
            logger.info(f"Creating EvoTrader executor for BensBot strategy")
            
            class EvoTraderExecutor:
                """
                Allows BensBot strategies to be executed within EvoTrader.
                """
                
                def __init__(self, strategy_class, strategy_info):
                    """
                    Initialize the executor.
                    
                    Args:
                        strategy_class: BensBot strategy class
                        strategy_info: EvoTrader strategy information
                    """
                    self.strategy_class = strategy_class
                    self.strategy_info = strategy_info
                    self.strategy_instance = None
                    self.data_adapter = DataFormatAdapter()
                    
                    # Initialize strategy
                    self._initialize_strategy()
                
                def _initialize_strategy(self):
                    """Initialize the strategy instance."""
                    try:
                        # Create instance
                        self.strategy_instance = self.strategy_class(
                            name=self.strategy_info.get('id', 'BenBotStrategy')
                        )
                    except Exception as e:
                        logger.error(f"Error initializing BensBot strategy: {e}")
                        self.strategy_instance = None
                
                def calculate_signal(self, data):
                    """
                    Calculate trading signal.
                    
                    Args:
                        data: Market data in EvoTrader format
                        
                    Returns:
                        Signal value (-1, 0, 1)
                    """
                    try:
                        if not self.strategy_instance:
                            self._initialize_strategy()
                            if not self.strategy_instance:
                                return 0
                        
                        # Convert data to BensBot format
                        benbot_data = self.data_adapter.evotrader_to_benbot_market_data(data)
                        
                        # Call BensBot strategy
                        if hasattr(self.strategy_instance, 'generate_signal'):
                            result = self.strategy_instance.generate_signal(benbot_data)
                        elif hasattr(self.strategy_instance, 'analyze'):
                            result = self.strategy_instance.analyze(benbot_data)
                        else:
                            logger.error("Strategy has no generate_signal or analyze method")
                            return 0
                        
                        # Convert result to simple signal
                        if isinstance(result, dict):
                            if 'signal' in result:
                                return result['signal']
                            elif 'side' in result:
                                side = result['side']
                                if side in ('BUY', 'buy', 'LONG', 'long'):
                                    return 1
                                elif side in ('SELL', 'sell', 'SHORT', 'short'):
                                    return -1
                                else:
                                    return 0
                        elif isinstance(result, (int, float)):
                            return 1 if result > 0 else -1 if result < 0 else 0
                        
                        return 0
                    
                    except Exception as e:
                        logger.error(f"Error calculating signal: {e}")
                        return 0
                
                def get_parameters(self):
                    """Get strategy parameters."""
                    if not self.strategy_instance:
                        return {}
                    
                    if hasattr(self.strategy_instance, 'parameters'):
                        params = self.strategy_instance.parameters
                        return self.data_adapter.benbot_to_evotrader_strategy_params(params)
                    elif hasattr(self.strategy_instance, 'get_parameters'):
                        params = self.strategy_instance.get_parameters()
                        return self.data_adapter.benbot_to_evotrader_strategy_params(params)
                    
                    return {}
                
                def set_parameters(self, parameters):
                    """
                    Set strategy parameters.
                    
                    Args:
                        parameters: New parameters in EvoTrader format
                    """
                    if not self.strategy_instance:
                        return False
                    
                    # Convert to BensBot format
                    benbot_params = self.data_adapter.evotrader_to_benbot_strategy_params(parameters)
                    
                    # Update strategy
                    if hasattr(self.strategy_instance, 'process_parameters'):
                        self.strategy_instance.process_parameters(benbot_params)
                    elif hasattr(self.strategy_instance, 'parameters'):
                        if isinstance(self.strategy_instance.parameters, dict):
                            self.strategy_instance.parameters.update(benbot_params)
                    
                    return True
                
                def reset(self):
                    """Reset strategy state."""
                    if self.strategy_instance and hasattr(self.strategy_instance, 'reset'):
                        self.strategy_instance.reset()
                    else:
                        self._initialize_strategy()
            
            # Create executor
            executor = EvoTraderExecutor(benbot_strategy_class, evotrader_strategy)
            
            logger.info(f"Successfully created EvoTrader executor for BensBot strategy")
            return executor
            
        except Exception as e:
            logger.error(f"Error creating EvoTrader executor: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    adapter = StrategyExecutionAdapter()
    
    # Load example strategy
    strategy_path = os.path.join(project_root, 'evotrader', 'strategies', 'example_strategy.py')
    
    if os.path.exists(strategy_path):
        # Load EvoTrader strategy
        strategy = adapter.load_evotrader_strategy(strategy_path)
        
        if strategy:
            print(f"Loaded strategy: {strategy.get('id')}")
            
            # Create BensBot adapter
            benbot_adapter = adapter.create_benbot_strategy_adapter(strategy)
            
            if benbot_adapter:
                print(f"Created BensBot adapter class")
                
                # Create instance
                instance = benbot_adapter()
                print(f"Adapter info: {instance.get_info()}")
    else:
        print(f"Example strategy not found at: {strategy_path}")
