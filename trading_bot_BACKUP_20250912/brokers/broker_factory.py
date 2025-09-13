"""
Broker Factory

Provides factory functions to create broker instances and set up
the Multi-Broker Manager with specific support for Tradier, Alpaca,
TradeStation, and E*TRADE.
"""

import logging
import os
from typing import Dict, Optional, List, Any

from .broker_interface import (
    BrokerInterface, BrokerCredentials, AssetType,
    BrokerAuthenticationError
)
from .multi_broker_manager import MultiBrokerManager
from .credential_store import CredentialStore, AuthMethod
from .trade_audit_log import TradeAuditLog, AuditEventType
from .auth_manager import create_credential_store, create_audit_log

# Import broker implementations
# Note: These may require additional packages to be installed
try:
    from .alpaca_client import AlpacaClient
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    
try:
    from .tradier_client import TradierClient
    TRADIER_AVAILABLE = True
except ImportError:
    TRADIER_AVAILABLE = False

try:
    from .tradestation_client import TradeStationClient
    TRADESTATION_AVAILABLE = True
except ImportError:
    TRADESTATION_AVAILABLE = False
    
try:
    from .etrade_client import ETradeClient
    ETRADE_AVAILABLE = True
except ImportError:
    ETRADE_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


def create_broker_manager(config: Dict[str, Any]) -> MultiBrokerManager:
    """
    Create a fully configured Multi-Broker Manager based on the provided config.
    
    Args:
        config: Configuration dictionary with broker settings
        
    Returns:
        MultiBrokerManager: Configured manager with all available brokers
        
    Example config:
    {
        "max_retries": 3,
        "retry_delay": 2,
        "auto_failover": true,
        "credential_store": {
            "type": "encrypted",  # or "yaml"
            "path": "data/credentials.enc",
            "master_password": "your_password"  # or use env var
        },
        "audit_log": {
            "enabled": true,
            "type": "sqlite",  # or "json"
            "path": "data/trading_audit.db"
        },
        "brokers": {
            "tradier": {
                "enabled": true,
                "api_key": "YOUR_API_KEY",  # or use env var
                "account_id": "YOUR_ACCOUNT_ID",
                "sandbox": true,
                "primary": true
            },
            "alpaca": {
                "enabled": true,
                "api_key": "YOUR_API_KEY",
                "api_secret": "YOUR_API_SECRET",
                "paper_trading": true
            },
            "tradestation": {
                "enabled": false,
                "client_id": "YOUR_CLIENT_ID",
                "client_secret": "YOUR_CLIENT_SECRET",
                "paper_trading": true
            },
            "etrade": {
                "enabled": false,
                "consumer_key": "YOUR_CONSUMER_KEY",
                "consumer_secret": "YOUR_CONSUMER_SECRET",
                "sandbox": true
            }
        },
        "asset_routing": {
            "stock": "alpaca",
            "option": "tradier",
            "future": "tradestation",
            "forex": "tradier",
            "crypto": "alpaca"
        }
    }
    """
    # Create credential store and audit log
    credential_store = create_credential_store(config)
    audit_log = create_audit_log(config)
    
    # Create the manager with provided settings and secure components
    manager = MultiBrokerManager(
        credential_store=credential_store,
        audit_log=audit_log,
        max_retries=config.get('max_retries', 3),
        retry_delay=config.get('retry_delay', 2),
        auto_failover=config.get('auto_failover', True)
    )
    
    # Set up brokers
    brokers_config = config.get('brokers', {})
    
    # Tradier setup
    if brokers_config.get('tradier', {}).get('enabled', False):
        setup_tradier(manager, brokers_config['tradier'])
    
    # Alpaca setup
    if brokers_config.get('alpaca', {}).get('enabled', False):
        setup_alpaca(manager, brokers_config['alpaca'])
    
    # TradeStation setup
    if brokers_config.get('tradestation', {}).get('enabled', False):
        setup_tradestation(manager, brokers_config['tradestation'])
    
    # E*TRADE setup
    if brokers_config.get('etrade', {}).get('enabled', False):
        setup_etrade(manager, brokers_config['etrade'])
    
    # Configure asset routing
    asset_routing = config.get('asset_routing', {})
    for asset_str, broker_id in asset_routing.items():
        try:
            asset_type = AssetType[asset_str.upper()]
            manager.set_asset_routing(asset_type, broker_id)
        except (KeyError, ValueError):
            logger.warning(f"Invalid asset type '{asset_str}' in routing configuration")
    
    # Start connection monitoring if we have any brokers
    if manager.brokers:
        manager.start_monitoring()
        manager.connect_all()
    
    return manager


def setup_tradier(manager: MultiBrokerManager, config: Dict[str, Any]) -> bool:
    """
    Add Tradier broker to the manager.
    
    Args:
        manager: Multi-Broker Manager instance
        config: Tradier configuration
        
    Returns:
        bool: Success status
    """
    if not TRADIER_AVAILABLE:
        logger.warning("Tradier client not available. Make sure the tradier_client.py is in your path.")
        return False
    
    try:
        # Retrieve configuration values
        sandbox = config.get('sandbox', True)  # Default to sandbox mode for safety
        primary = config.get('primary', False)
        broker_id = 'tradier'
        
        # Create Tradier client (credentials will be fetched from the store later)
        client = TradierClient(sandbox_mode=sandbox)
        
        # Register with manager (no direct credentials passed)
        manager.add_broker(
            broker_id, 
            client, 
            None,  # No credentials passed directly (comes from credential store)
            primary,
            AuthMethod.API_KEY  # Specifying the authentication method
        )
        
        logger.info(f"Added Tradier broker to manager{' (primary)' if primary else ''}")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up Tradier broker: {str(e)}")
        return False


def setup_alpaca(manager: MultiBrokerManager, config: Dict[str, Any]) -> bool:
    """
    Add Alpaca broker to the manager.
    
    Args:
        manager: Multi-Broker Manager instance
        config: Alpaca configuration
        
    Returns:
        bool: Success status
    """
    if not ALPACA_AVAILABLE:
        logger.warning("Alpaca client not available. Make sure the alpaca_client.py is in your path.")
        return False
    
    try:
        # Retrieve configuration values
        paper_trading = config.get('paper_trading', True)  # Default to paper trading
        primary = config.get('primary', False)
        broker_id = 'alpaca'
        
        # Create Alpaca client (credentials will be fetched from the store later)
        client = AlpacaClient(paper_trading=paper_trading)
        
        # Register with manager (no direct credentials passed)
        manager.add_broker(
            broker_id, 
            client, 
            None,  # No credentials passed directly (comes from credential store)
            primary,
            AuthMethod.API_KEY  # Specifying the authentication method
        )
        
        logger.info(f"Added Alpaca broker to manager{' (primary)' if primary else ''}")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up Alpaca broker: {str(e)}")
        return False


def setup_tradestation(manager: MultiBrokerManager, config: Dict[str, Any]) -> bool:
    """
    Add TradeStation broker to the manager.
    
    Args:
        manager: Multi-Broker Manager instance
        config: TradeStation configuration
        
    Returns:
        bool: Success status
    """
    if not TRADESTATION_AVAILABLE:
        logger.warning("TradeStation client not available. Make sure the tradestation_client.py is in your path.")
        return False
    
    try:
        # Retrieve configuration values
        paper_trading = config.get('paper_trading', True)  # Default to paper trading
        primary = config.get('primary', False)
        broker_id = 'tradestation'
        
        # Create TradeStation client (credentials will be fetched from the store later)
        client = TradeStationClient(paper_trading=paper_trading)
        
        # Register with manager (no direct credentials passed)
        manager.add_broker(
            broker_id, 
            client, 
            None,  # No credentials passed directly (comes from credential store)
            primary,
            AuthMethod.OAUTH  # Specifying the authentication method
        )
        
        logger.info(f"Added TradeStation broker to manager{' (primary)' if primary else ''}")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up TradeStation broker: {str(e)}")
        return False


def setup_etrade(manager: MultiBrokerManager, config: Dict[str, Any]) -> bool:
    """
    Add E*TRADE broker to the manager.
    
    Args:
        manager: Multi-Broker Manager instance
        config: E*TRADE configuration
        
    Returns:
        bool: Success status
    """
    if not ETRADE_AVAILABLE:
        logger.warning("E*TRADE client not available. Make sure the etrade_client.py is in your path.")
        return False
    
    try:
        # Retrieve configuration values
        sandbox = config.get('sandbox', True)  # Default to sandbox mode
        primary = config.get('primary', False)
        broker_id = 'etrade'
        
        # Create E*TRADE client (credentials will be fetched from the store later)
        client = ETradeClient(sandbox_mode=sandbox)
        
        # Register with manager (no direct credentials passed)
        manager.add_broker(
            broker_id, 
            client, 
            None,  # No credentials passed directly (comes from credential store)
            primary,
            AuthMethod.OAUTH  # Specifying the authentication method
        )
        
        logger.info(f"Added E*TRADE broker to manager{' (primary)' if primary else ''}")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up E*TRADE broker: {str(e)}")
        return False


def load_from_config_file(config_path: str) -> MultiBrokerManager:
    """
    Load broker manager configuration from a JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        MultiBrokerManager: Configured manager with secure credential storage
    """
    import json
    from .auth_manager import initialize_broker_credentials
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Loaded broker configuration from {config_path}")
        
        # Create the manager with secure storage
        manager = create_broker_manager(config)
        
        # Initialize credential store with broker credentials from config
        if manager.credential_store:
            initialize_broker_credentials(manager.credential_store, config)
        
        return manager
    
    except Exception as e:
        logger.error(f"Error loading broker configuration: {str(e)}")
        # Return a default manager
        return MultiBrokerManager()
