"""
Service Registry - Dependency management system for the trading bot.
Provides a way to register and access services throughout the application,
reducing direct dependencies between components.
"""

import logging
from typing import Dict, Any, Type, Optional, TypeVar, Generic, cast

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ServiceRegistry:
    """
    Central registry for all services in the trading bot system.
    Follows the Service Locator pattern to provide a way to decouple
    components while still allowing them to find and use each other.
    """
    
    # Storage for registered services
    _services: Dict[str, Any] = {}
    
    # Storage for registered service types (for type checking)
    _service_types: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, service_name: str, service_instance: Any, service_type: Optional[Type] = None) -> None:
        """
        Register a service in the registry.
        
        Args:
            service_name: Unique name for the service
            service_instance: Service instance to register
            service_type: Type of the service (for type checking)
        """
        if service_name in cls._services:
            logger.warning(f"Service '{service_name}' already registered. Overwriting.")
        
        cls._services[service_name] = service_instance
        
        if service_type is not None:
            cls._service_types[service_name] = service_type
        else:
            # If no type specified, use the instance's type
            cls._service_types[service_name] = type(service_instance)
            
        logger.debug(f"Registered service: {service_name}")
    
    @classmethod
    def get(cls, service_name: str) -> Any:
        """
        Get a service from the registry.
        
        Args:
            service_name: Name of the service to retrieve
            
        Returns:
            The registered service instance
            
        Raises:
            KeyError: If service is not registered
        """
        if service_name not in cls._services:
            raise KeyError(f"Service '{service_name}' not registered")
            
        return cls._services[service_name]
    
    @classmethod
    def get_typed(cls, service_name: str, expected_type: Type[T]) -> T:
        """
        Get a service with type checking.
        
        Args:
            service_name: Name of the service to retrieve
            expected_type: Expected type of the service
            
        Returns:
            The registered service instance, type checked
            
        Raises:
            KeyError: If service is not registered
            TypeError: If service is not of the expected type
        """
        service = cls.get(service_name)
        
        if not isinstance(service, expected_type):
            raise TypeError(f"Service '{service_name}' is not of type {expected_type.__name__}")
            
        return cast(expected_type, service)
    
    @classmethod
    def has_service(cls, service_name: str) -> bool:
        """
        Check if a service is registered.
        
        Args:
            service_name: Name of the service to check
            
        Returns:
            True if the service is registered, False otherwise
        """
        return service_name in cls._services
    
    @classmethod
    def unregister(cls, service_name: str) -> None:
        """
        Unregister a service.
        
        Args:
            service_name: Name of the service to unregister
            
        Raises:
            KeyError: If service is not registered
        """
        if service_name not in cls._services:
            raise KeyError(f"Service '{service_name}' not registered")
            
        del cls._services[service_name]
        
        if service_name in cls._service_types:
            del cls._service_types[service_name]
            
        logger.debug(f"Unregistered service: {service_name}")
    
    @classmethod
    def reset(cls) -> None:
        """
        Reset the registry, removing all registered services.
        Primarily used for testing.
        """
        cls._services.clear()
        cls._service_types.clear()
        logger.debug("Service registry reset") 