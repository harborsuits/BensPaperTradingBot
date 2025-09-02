"""
Component Registry

Registry for managing and accessing modular strategy components.
Provides functionality to register, retrieve, and manage components.
"""

import importlib
import inspect
import logging
import json
import os
from typing import Dict, List, Any, Optional, Type, Union, Callable
import uuid

from trading_bot.strategies.modular_strategy_system import (
    StrategyComponent, ComponentType, SignalGeneratorComponent, 
    FilterComponent, PositionSizerComponent, ExitManagerComponent
)

# Import strategy adapter
try:
    from trading_bot.strategies.strategy_adapter import StrategyAdapter, create_strategy_adapter
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Strategy adapter not available")
    StrategyAdapter = None
    create_strategy_adapter = None

logger = logging.getLogger(__name__)

class ComponentRegistry:
    """
    Registry for managing modular strategy components.
    
    Allows components to be registered, retrieved, and managed from a central registry.
    Supports component discovery, metadata management, and serialization.
    """
    
    def __init__(self):
        """Initialize component registry."""
        # Initialize component dictionaries by type
        self.components = {
            ComponentType.SIGNAL_GENERATOR: {},
            ComponentType.FILTER: {},
            ComponentType.POSITION_SIZER: {},
            ComponentType.EXIT_MANAGER: {},
        }
        
        # Component classes by name
        self.component_classes = {}
        
        # Component metadata (parameters, descriptions, etc.)
        self.component_metadata = {}
        
        # Load built-in components
        self._register_builtin_components()
    
    def _register_builtin_components(self) -> None:
        """Register all built-in components from the components directory."""
        try:
            # Import component modules
            from trading_bot.strategies.components import signal_generators
            from trading_bot.strategies.components import filters
            from trading_bot.strategies.components import position_sizers
            from trading_bot.strategies.components import exit_managers
            
            # Register all component classes from each module
            self._register_components_from_module(signal_generators)
            self._register_components_from_module(filters)
            self._register_components_from_module(position_sizers)
            self._register_components_from_module(exit_managers)
            
            logger.info(f"Registered {len(self.component_classes)} built-in component classes")
        except ImportError as e:
            logger.warning(f"Could not import built-in components: {e}")
    
    def _register_components_from_module(self, module) -> None:
        """
        Register all component classes from a module.
        
        Args:
            module: Module containing component classes
        """
        for name, obj in inspect.getmembers(module):
            # Check if it's a class and subclass of StrategyComponent
            if (inspect.isclass(obj) and 
                issubclass(obj, StrategyComponent) and 
                obj != StrategyComponent and
                obj != SignalGeneratorComponent and
                obj != FilterComponent and
                obj != PositionSizerComponent and
                obj != ExitManagerComponent):
                
                # Register the class
                self.register_component_class(name, obj)
    
    def register_component_class(self, name: str, component_class: Type[StrategyComponent]) -> None:
        """Register a component class with the registry
        
        Args:
            name: Component name
            component_class: Component class to register
        """
        component_type = self._get_component_type(component_class)
        
        if component_type is None:
            logger.error(f"Cannot determine component type for {component_class.__name__}")
            return
            
        # Save both the original class and information needed for adaptation
        self.components[component_type][name] = {
            'class': component_class,
            'requires_adapter': not hasattr(component_class, 'generate_signals') or 
                             not hasattr(component_class, 'size_position') or 
                             not hasattr(component_class, 'manage_open_trades')
        }
        logger.info(f"Registered {component_type.name} component: {name}")
    
    def _get_component_type(self, component_class: Type[StrategyComponent]) -> Optional[ComponentType]:
        """
        Determine the component type from a class.
        
        Args:
            component_class: Component class
            
        Returns:
            Component type or None if unknown
        """
        if issubclass(component_class, SignalGeneratorComponent):
            return ComponentType.SIGNAL_GENERATOR
        elif issubclass(component_class, FilterComponent):
            return ComponentType.FILTER
        elif issubclass(component_class, PositionSizerComponent):
            return ComponentType.POSITION_SIZER
        elif issubclass(component_class, ExitManagerComponent):
            return ComponentType.EXIT_MANAGER
        else:
            return None
    
    def create_component_instance(self, component_type: ComponentType, name: str, 
        **kwargs) -> Optional[StrategyComponent]:
        """Create an instance of a component
        
        Args:
            component_type: Type of component to create
            name: Name of component to create
            **kwargs: Additional arguments to pass to component constructor
            
        Returns:
            Component instance or None if component not found
        """
        if name not in self.components[component_type]:
            logger.error(f"Component not found: {component_type.name}/{name}")
            return None
            
        component_info = self.components[component_type][name]
        component_class = component_info['class']
        requires_adapter = component_info.get('requires_adapter', False)
        
        try:
            # Create component instance
            component = component_class(**kwargs)
            
            # Wrap with adapter if needed and adapter is available
            if requires_adapter and 'create_strategy_adapter' in globals() and create_strategy_adapter is not None:
                logger.info(f"Creating adapter for {component_type.name}/{name}")
                return create_strategy_adapter(component)
            
            return component
        except Exception as e:
            logger.error(f"Error creating component {component_type.name}/{name}: {e}")
            return None
    
    def get_component_by_name(self, name: str) -> Optional[Type[StrategyComponent]]:
        """Get a component class by name, searching all component types
        
        Args:
            name: Name of component to find
            
        Returns:
            Component class or None if not found
        """
        for component_type in ComponentType:
            if name in self.components[component_type]:
                return self.components[component_type][name]['class']
                
        return None
    
    def get_all_component_classes(self) -> Dict[str, Type[StrategyComponent]]:
        """Get all registered component classes.
        
        Returns:
            Dictionary of class name -> class
        """
        return self.component_classes
    
    def get_component_metadata(self, class_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a component class.
        
        Args:
            class_name: Component class name
            
        Returns:
            Component metadata or None if not found
        """
        return self.component_metadata.get(class_name)
    
    def get_strategy_class(self, strategy_type: str) -> Type[object]:
        """Get the class for a registered strategy type."""
        return self.components.get(strategy_type, None)
        
    def get_strategy_instance(self, strategy_type: str) -> Any:
        """Get an instance of a strategy, properly wrapped with the adapter if needed.
        
        This ensures all strategies expose a consistent interface regardless of their
        original implementation.
        
        Args:
            strategy_type: The type of strategy to instantiate
            
        Returns:
            A strategy instance wrapped with the adapter if needed
        """
        from trading_bot.strategies.strategy_adapter import create_strategy_adapter
        
        strategy_class = self.get_strategy_class(strategy_type)
        if strategy_class is None:
            return None
            
        try:
            strategy_instance = strategy_class()
            
            # Check if it already implements the required interface
            if (hasattr(strategy_instance, 'generate_signals') and
                hasattr(strategy_instance, 'size_position') and
                hasattr(strategy_instance, 'manage_open_trades')):
                # Already implements the interface
                return strategy_instance
            else:
                # Wrap with adapter
                return create_strategy_adapter(strategy_instance)
                
        except Exception as e:
            logger.error(f"Error instantiating strategy {strategy_type}: {str(e)}")
            return None
    
    def get_all_component_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all component classes.
        
        Returns:
            Dictionary of class name -> metadata
        """
        return self.component_metadata
    
    def remove_component(self, component_type: Union[ComponentType, str], component_id: str) -> bool:
        """
        Remove a component from the registry.
        
        Args:
            component_type: Type of component
            component_id: Component ID
            
        Returns:
            True if the component was removed, False otherwise
        """
        # Convert string type to enum if needed
        if isinstance(component_type, str):
            component_type = ComponentType[component_type]
            
        if component_id in self.components[component_type]:
            del self.components[component_type][component_id]
            return True
        return False
    
    def clear_components(self, component_type: Optional[Union[ComponentType, str]] = None) -> None:
        """
        Clear all components of a specific type or all components.
        
        Args:
            component_type: Type of component, or None to clear all
        """
        if component_type is None:
            # Clear all components
            for comp_type in ComponentType:
                self.components[comp_type] = {}
        else:
            # Convert string type to enum if needed
            if isinstance(component_type, str):
                component_type = ComponentType[component_type]
                
            # Clear components of specific type
            self.components[component_type] = {}
    
    def serialize_component(self, component: StrategyComponent) -> Dict[str, Any]:
        """
        Serialize a component to a dictionary.
        
        Args:
            component: Component instance
            
        Returns:
            Serialized component
        """
        # Get class name
        class_name = component.__class__.__name__
        
        # Serialize component
        serialized = {
            'class_name': class_name,
            'component_id': component.component_id,
            'component_type': component.component_type.name,
            'parameters': component.parameters.copy(),
            'description': component.description
        }
        
        return serialized
    
    def deserialize_component(self, serialized: Dict[str, Any]) -> Optional[StrategyComponent]:
        """
        Deserialize a component from a dictionary.
        
        Args:
            serialized: Serialized component
            
        Returns:
            Component instance or None if deserialization failed
        """
        try:
            # Get component class
            class_name = serialized['class_name']
            component_id = serialized['component_id']
            parameters = serialized.get('parameters', {})
            
            if class_name not in self.component_classes:
                logger.error(f"Component class '{class_name}' not found in registry")
                return None
            
            # Create component instance
            component_class = self.component_classes[class_name]
            component = component_class(component_id=component_id, **parameters)
            
            # Manually set description if provided
            if 'description' in serialized:
                component.description = serialized['description']
            
            return component
        except Exception as e:
            logger.error(f"Error deserializing component: {e}")
            return None
    
    def save_component_registry(self, file_path: str) -> bool:
        """
        Save the component registry to a JSON file.
        
        Args:
            file_path: Path to save the registry
            
        Returns:
            True if the registry was saved successfully, False otherwise
        """
        try:
            # Serialize all components
            serialized_registry = {
                'components': {},
                'metadata': self.component_metadata
            }
            
            for comp_type in ComponentType:
                serialized_registry['components'][comp_type.name] = {}
                
                for comp_id, component in self.components[comp_type].items():
                    serialized_registry['components'][comp_type.name][comp_id] = self.serialize_component(component)
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(serialized_registry, f, indent=2)
            
            logger.info(f"Component registry saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving component registry: {e}")
            return False
    
    def load_component_registry(self, file_path: str) -> bool:
        """
        Load the component registry from a JSON file.
        
        Args:
            file_path: Path to load the registry from
            
        Returns:
            True if the registry was loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Component registry file {file_path} does not exist")
                return False
            
            # Read from file
            with open(file_path, 'r') as f:
                serialized_registry = json.load(f)
            
            # Clear existing components
            self.clear_components()
            
            # Load components
            serialized_components = serialized_registry.get('components', {})
            for type_name, components in serialized_components.items():
                comp_type = ComponentType[type_name]
                
                for comp_id, serialized in components.items():
                    component = self.deserialize_component(serialized)
                    if component:
                        self.components[comp_type][comp_id] = component
            
            # Load metadata if available
            if 'metadata' in serialized_registry:
                self.component_metadata = serialized_registry['metadata']
            
            logger.info(f"Component registry loaded from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading component registry: {e}")
            return False

# Singleton instance
_registry_instance = None

def get_component_registry() -> ComponentRegistry:
    """
    Get the singleton component registry instance.
    
    Returns:
        Component registry instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ComponentRegistry()
    return _registry_instance
