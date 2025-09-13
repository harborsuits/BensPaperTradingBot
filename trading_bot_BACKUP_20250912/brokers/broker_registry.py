"""
Broker Registry

This module provides a central registry for managing broker connections,
allowing easy access to broker clients throughout the application.
"""

import logging
import os
import json
from typing import Dict, List, Any, Optional, Union, Type, Callable
from datetime import datetime

from .brokerage_client import (
    BrokerageClient, 
    BrokerConnectionStatus,
    BROKER_IMPLEMENTATIONS
)
from .connection_monitor import ConnectionMonitor, ConnectionAlert

# Configure logging
logger = logging.getLogger(__name__)

class BrokerRegistry:
    """
    Central registry for managing broker connections.
    
    This class provides a singleton registry for registering, 
    accessing, and monitoring broker clients.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Ensure only one instance of BrokerRegistry exists."""
        if cls._instance is None:
            cls._instance = super(BrokerRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the broker registry.
        
        Args:
            config_dir: Directory containing broker configuration files
        """
        # Skip initialization if already initialized
        if getattr(self, "_initialized", False):
            return
        
        self.brokers: Dict[str, BrokerageClient] = {}
        self.config_dir = config_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs"
        )
        
        # Initialize connection monitor
        self.connection_monitor = ConnectionMonitor(
            alert_callback=self._handle_connection_alert
        )
        
        # Connection alert handlers
        self.alert_handlers: List[Callable[[ConnectionAlert], None]] = []
        
        # Flag to track if auto-connection is enabled
        self.auto_connect = True
        
        # Status tracking
        self.last_connection_check = None
        self.connection_check_result = {}
        
        # Initialize broker clients from configuration
        self._load_broker_configs()
        
        self._initialized = True
        logger.info("Broker registry initialized")
    
    def register_broker(self, 
                      name: str, 
                      broker_client: BrokerageClient,
                      connect: bool = True,
                      monitor: bool = True) -> bool:
        """
        Register a broker client with the registry.
        
        Args:
            name: Name to register the broker as
            broker_client: BrokerageClient instance
            connect: Whether to connect the broker immediately
            monitor: Whether to monitor the broker's connection
            
        Returns:
            bool: True if registration was successful, False otherwise
        """
        if name in self.brokers:
            logger.warning(f"Broker '{name}' already registered, overwriting")
        
        self.brokers[name] = broker_client
        
        # Connect if requested
        if connect and self.auto_connect:
            try:
                broker_client.connect()
            except Exception as e:
                logger.error(f"Failed to connect broker '{name}': {str(e)}")
        
        # Add to connection monitor if requested
        if monitor:
            self.connection_monitor.register_broker(name, broker_client)
        
        logger.info(f"Registered broker '{name}'")
        return True
    
    def unregister_broker(self, name: str) -> bool:
        """
        Unregister a broker client from the registry.
        
        Args:
            name: Name of the broker to unregister
            
        Returns:
            bool: True if unregistration was successful, False otherwise
        """
        if name not in self.brokers:
            logger.warning(f"Broker '{name}' not registered")
            return False
        
        # Disconnect the broker
        try:
            self.brokers[name].disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting broker '{name}': {str(e)}")
        
        # Remove from connection monitor
        self.connection_monitor.unregister_broker(name)
        
        # Remove from registry
        del self.brokers[name]
        
        logger.info(f"Unregistered broker '{name}'")
        return True
    
    def get_broker(self, name: str) -> Optional[BrokerageClient]:
        """
        Get a registered broker client by name.
        
        Args:
            name: Name of the broker to get
            
        Returns:
            BrokerageClient: The broker client, or None if not found
        """
        return self.brokers.get(name)
    
    def get_all_brokers(self) -> Dict[str, BrokerageClient]:
        """
        Get all registered broker clients.
        
        Returns:
            Dict[str, BrokerageClient]: Dictionary of all registered brokers
        """
        return self.brokers.copy()
    
    def start_connection_monitoring(self) -> None:
        """Start connection monitoring for all registered brokers."""
        self.connection_monitor.start_monitoring()
        logger.info("Started broker connection monitoring")
    
    def stop_connection_monitoring(self) -> None:
        """Stop connection monitoring for all registered brokers."""
        self.connection_monitor.stop_monitoring()
        logger.info("Stopped broker connection monitoring")
    
    def check_all_connections(self) -> Dict[str, BrokerConnectionStatus]:
        """
        Check the connection status of all registered brokers.
        
        Returns:
            Dict[str, BrokerConnectionStatus]: Connection status for each broker
        """
        statuses = self.connection_monitor.check_connections()
        self.last_connection_check = datetime.now()
        self.connection_check_result = statuses
        return statuses
    
    def get_connection_alerts(self, include_resolved: bool = False) -> List[ConnectionAlert]:
        """
        Get connection alerts.
        
        Args:
            include_resolved: Whether to include resolved alerts
            
        Returns:
            List[ConnectionAlert]: Connection alerts
        """
        return self.connection_monitor.get_alerts(include_resolved)
    
    def register_alert_handler(self, handler: Callable[[ConnectionAlert], None]) -> None:
        """
        Register a handler for connection alerts.
        
        Args:
            handler: Function to call when a connection alert occurs
        """
        if handler not in self.alert_handlers:
            self.alert_handlers.append(handler)
    
    def unregister_alert_handler(self, handler: Callable[[ConnectionAlert], None]) -> None:
        """
        Unregister a handler for connection alerts.
        
        Args:
            handler: Handler to unregister
        """
        if handler in self.alert_handlers:
            self.alert_handlers.remove(handler)
    
    def create_broker(self, 
                    broker_type: str, 
                    name: str, 
                    config: Dict[str, Any],
                    connect: bool = True,
                    monitor: bool = True) -> Optional[BrokerageClient]:
        """
        Create and register a new broker client.
        
        Args:
            broker_type: Type of broker to create (e.g., "alpaca", "tradier")
            name: Name to register the broker as
            config: Configuration for the broker
            connect: Whether to connect the broker immediately
            monitor: Whether to monitor the broker's connection
            
        Returns:
            BrokerageClient: The created broker client, or None if creation failed
        """
        # Check if broker type is supported
        if broker_type not in BROKER_IMPLEMENTATIONS:
            logger.error(f"Unsupported broker type: {broker_type}")
            return None
        
        # Get broker implementation class
        broker_class = BROKER_IMPLEMENTATIONS[broker_type]
        
        try:
            # Create broker instance
            broker = broker_class(**config)
            
            # Register broker
            self.register_broker(name, broker, connect, monitor)
            
            return broker
        
        except Exception as e:
            logger.error(f"Failed to create broker '{name}' of type '{broker_type}': {str(e)}")
            return None
    
    def disconnect_all(self) -> None:
        """Disconnect all registered brokers."""
        for name, broker in list(self.brokers.items()):
            try:
                broker.disconnect()
                logger.info(f"Disconnected broker '{name}'")
            except Exception as e:
                logger.error(f"Error disconnecting broker '{name}': {str(e)}")
    
    def connect_all(self) -> Dict[str, bool]:
        """
        Connect all registered brokers.
        
        Returns:
            Dict[str, bool]: Connection success status for each broker
        """
        results = {}
        
        for name, broker in self.brokers.items():
            try:
                success = broker.connect()
                results[name] = success
                logger.info(f"Connected broker '{name}'")
            except Exception as e:
                results[name] = False
                logger.error(f"Error connecting broker '{name}': {str(e)}")
        
        return results
    
    def _load_broker_configs(self) -> None:
        """Load broker configurations from the config directory."""
        if not os.path.exists(self.config_dir):
            logger.warning(f"Config directory {self.config_dir} does not exist")
            return
        
        # Look for broker config files
        for filename in os.listdir(self.config_dir):
            if not filename.endswith('.json'):
                continue
            
            if filename.startswith('broker_') or filename == 'brokers.json':
                config_path = os.path.join(self.config_dir, filename)
                self._load_broker_config_file(config_path)
    
    def _load_broker_config_file(self, config_path: str) -> None:
        """
        Load broker configurations from a file.
        
        Args:
            config_path: Path to the configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Handle different config formats
            if isinstance(config, dict):
                if 'brokers' in config:
                    # Multiple brokers in a 'brokers' field
                    brokers_config = config['brokers']
                    if isinstance(brokers_config, list):
                        for broker_config in brokers_config:
                            self._create_broker_from_config(broker_config)
                    elif isinstance(brokers_config, dict):
                        for name, broker_config in brokers_config.items():
                            broker_config['name'] = name
                            self._create_broker_from_config(broker_config)
                else:
                    # Single broker config
                    self._create_broker_from_config(config)
            
            elif isinstance(config, list):
                # List of broker configs
                for broker_config in config:
                    self._create_broker_from_config(broker_config)
            
            logger.info(f"Loaded broker configurations from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading broker config from {config_path}: {str(e)}")
    
    def _create_broker_from_config(self, config: Dict[str, Any]) -> None:
        """
        Create a broker from configuration.
        
        Args:
            config: Broker configuration
        """
        broker_type = config.get('type')
        name = config.get('name')
        
        if not broker_type:
            logger.error("Broker config missing 'type' field")
            return
        
        if not name:
            logger.error("Broker config missing 'name' field")
            return
        
        # Extract broker-specific config
        broker_config = config.copy()
        broker_config.pop('type', None)
        broker_config.pop('name', None)
        
        # Create the broker
        self.create_broker(
            broker_type=broker_type,
            name=name,
            config=broker_config,
            connect=config.get('auto_connect', self.auto_connect),
            monitor=config.get('monitor', True)
        )
    
    def _handle_connection_alert(self, alert: ConnectionAlert) -> None:
        """
        Handle a connection alert from the connection monitor.
        
        Args:
            alert: Connection alert
        """
        # Notify all registered alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in connection alert handler: {str(e)}")
        
        # Log the alert
        if alert.resolved:
            logger.info(f"Connection alert resolved - {alert.broker_name}: {alert.resolution_message}")
        else:
            logger.warning(f"Connection alert - {alert.broker_name}: {alert.message}")


# Singleton instance
_broker_registry = None

def get_broker_registry(config_dir: Optional[str] = None) -> BrokerRegistry:
    """
    Get the singleton broker registry instance.
    
    Args:
        config_dir: Optional directory containing broker configuration files
    
    Returns:
        BrokerRegistry: The broker registry instance
    """
    global _broker_registry
    if _broker_registry is None:
        _broker_registry = BrokerRegistry(config_dir)
    return _broker_registry 