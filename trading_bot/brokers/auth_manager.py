"""
Authentication Manager

Provides secure credential management for broker connections 
with support for different storage backends and authentication methods.
"""

import os
import logging
import json
from typing import Dict, Optional, Any, Tuple
from pathlib import Path

from .credential_store import (
    CredentialStore, EncryptedFileStore, YamlFileStore,
    AuthMethod, CredentialFactory
)
from .trade_audit_log import TradeAuditLog, SqliteAuditLog, JsonFileAuditLog
from trading_bot.core.event_bus import EventBus, get_global_event_bus
from trading_bot.core.audit_log_listener import AuditLogListener, create_audit_log_listener

# Configure logging
logger = logging.getLogger(__name__)


def create_credential_store(config: Dict[str, Any]) -> Optional[CredentialStore]:
    """
    Create a credential store based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        CredentialStore: Configured credential store or None if setup fails
    """
    store_config = config.get('credential_store', {})
    store_type = store_config.get('type', 'encrypted')
    
    # Get store path - create default if not specified
    base_dir = config.get('data_dir', os.path.join(os.path.dirname(__file__), '../../data'))
    os.makedirs(base_dir, exist_ok=True)
    
    if store_type == 'encrypted':
        store_path = store_config.get('path', os.path.join(base_dir, 'credentials.enc'))
        
        # Get master password
        master_password = (
            store_config.get('master_password') or 
            os.environ.get('TRADING_BOT_MASTER_PASSWORD')
        )
        
        if not master_password:
            logger.error("No master password provided for encrypted credential store")
            logger.error("Set TRADING_BOT_MASTER_PASSWORD environment variable or specify in config")
            return None
        
        try:
            store = EncryptedFileStore(store_path, master_password)
            logger.info(f"Initialized encrypted credential store at {store_path}")
            return store
        except Exception as e:
            logger.error(f"Error creating encrypted credential store: {str(e)}")
            return None
            
    elif store_type == 'yaml':
        store_path = store_config.get('path', os.path.join(base_dir, 'credentials.yml'))
        
        try:
            store = YamlFileStore(store_path)
            logger.info(f"Initialized YAML credential store at {store_path}")
            return store
        except Exception as e:
            logger.error(f"Error creating YAML credential store: {str(e)}")
            return None
    
    else:
        logger.error(f"Unsupported credential store type: {store_type}")
        return None


def create_audit_log(config: Dict[str, Any]) -> Optional[TradeAuditLog]:
    """
    Create an audit log based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TradeAuditLog: Configured audit log or None if setup fails
    """
    audit_config = config.get('audit_log', {})
    
    if not audit_config.get('enabled', True):
        logger.info("Audit logging is disabled in configuration")
        return None
    
    audit_type = audit_config.get('type', 'sqlite')
    
    # Get store path - create default if not specified
    base_dir = config.get('data_dir', os.path.join(os.path.dirname(__file__), '../../data'))
    os.makedirs(base_dir, exist_ok=True)
    
    if audit_type == 'sqlite':
        db_path = audit_config.get('path', os.path.join(base_dir, 'trading_audit.db'))
        
        try:
            audit_log = SqliteAuditLog(db_path)
            logger.info(f"Initialized SQLite audit log at {db_path}")
            return audit_log
        except Exception as e:
            logger.error(f"Error creating SQLite audit log: {str(e)}")
            return None
            
    elif audit_type == 'json':
        log_dir = audit_config.get('directory', os.path.join(base_dir, 'audit_logs'))
        os.makedirs(log_dir, exist_ok=True)
        
        try:
            audit_log = JsonFileAuditLog(log_dir)
            logger.info(f"Initialized JSON file audit log in {log_dir}")
            return audit_log
        except Exception as e:
            logger.error(f"Error creating JSON file audit log: {str(e)}")
            return None
    
    else:
        logger.error(f"Unsupported audit log type: {audit_type}")
        return None


def initialize_broker_credentials(credential_store: CredentialStore, 
                               config: Dict[str, Any]) -> bool:
    """
    Initialize broker credentials in the credential store.
    
    Args:
        credential_store: Credential store
        config: Broker configuration dictionary
        
    Returns:
        bool: Success status
    """
    brokers_config = config.get('brokers', {})
    success = True
    
    # Tradier credentials
    if brokers_config.get('tradier', {}).get('enabled', False):
        tradier_config = brokers_config['tradier']
        
        api_key = tradier_config.get('api_key') or os.environ.get('TRADIER_API_KEY')
        account_id = tradier_config.get('account_id') or os.environ.get('TRADIER_ACCOUNT_ID')
        
        if api_key:
            credentials = CredentialFactory.create_api_key_credentials(
                api_key=api_key,
                api_secret="",  # Tradier doesn't use an API secret
                additional_params={
                    "account_id": account_id,
                    "sandbox": tradier_config.get('sandbox', True)
                }
            )
            
            try:
                credential_store.store_credentials('tradier', credentials, AuthMethod.API_KEY)
                logger.info("Stored Tradier credentials in credential store")
            except Exception as e:
                logger.error(f"Error storing Tradier credentials: {str(e)}")
                success = False
        else:
            logger.warning("No Tradier API key provided, skipping credential storage")
    
    # Alpaca credentials
    if brokers_config.get('alpaca', {}).get('enabled', False):
        alpaca_config = brokers_config['alpaca']
        
        api_key = alpaca_config.get('api_key') or os.environ.get('ALPACA_API_KEY')
        api_secret = alpaca_config.get('api_secret') or os.environ.get('ALPACA_API_SECRET')
        
        if api_key and api_secret:
            credentials = CredentialFactory.create_api_key_credentials(
                api_key=api_key,
                api_secret=api_secret,
                additional_params={
                    "paper_trading": alpaca_config.get('paper_trading', True)
                }
            )
            
            try:
                credential_store.store_credentials('alpaca', credentials, AuthMethod.API_KEY)
                logger.info("Stored Alpaca credentials in credential store")
            except Exception as e:
                logger.error(f"Error storing Alpaca credentials: {str(e)}")
                success = False
        else:
            logger.warning("Incomplete Alpaca credentials provided, skipping credential storage")
    
    # TradeStation credentials
    if brokers_config.get('tradestation', {}).get('enabled', False):
        ts_config = brokers_config['tradestation']
        
        client_id = ts_config.get('client_id') or os.environ.get('TRADESTATION_CLIENT_ID')
        client_secret = ts_config.get('client_secret') or os.environ.get('TRADESTATION_CLIENT_SECRET')
        access_token = ts_config.get('access_token') or os.environ.get('TRADESTATION_ACCESS_TOKEN')
        refresh_token = ts_config.get('refresh_token') or os.environ.get('TRADESTATION_REFRESH_TOKEN')
        
        if client_id and client_secret:
            credentials = CredentialFactory.create_oauth_credentials(
                client_id=client_id,
                client_secret=client_secret,
                access_token=access_token,
                refresh_token=refresh_token,
                token_expiry=ts_config.get('token_expiry'),
                auth_url="https://api.tradestation.com/v2/authorize",
                token_url="https://api.tradestation.com/v2/security/authorize"
            )
            
            try:
                credential_store.store_credentials('tradestation', credentials, AuthMethod.OAUTH)
                logger.info("Stored TradeStation credentials in credential store")
            except Exception as e:
                logger.error(f"Error storing TradeStation credentials: {str(e)}")
                success = False
        else:
            logger.warning("Incomplete TradeStation credentials provided, skipping credential storage")
    
    # E*TRADE credentials
    if brokers_config.get('etrade', {}).get('enabled', False):
        etrade_config = brokers_config['etrade']
        
        consumer_key = etrade_config.get('consumer_key') or os.environ.get('ETRADE_CONSUMER_KEY')
        consumer_secret = etrade_config.get('consumer_secret') or os.environ.get('ETRADE_CONSUMER_SECRET')
        oauth_token = etrade_config.get('oauth_token') or os.environ.get('ETRADE_OAUTH_TOKEN')
        oauth_token_secret = etrade_config.get('oauth_token_secret') or os.environ.get('ETRADE_OAUTH_TOKEN_SECRET')
        
        if consumer_key and consumer_secret:
            credentials = CredentialFactory.create_oauth_credentials(
                client_id=consumer_key,
                client_secret=consumer_secret,
                access_token=oauth_token,
                refresh_token=oauth_token_secret,  # E*TRADE uses token secret instead of refresh
                token_expiry=etrade_config.get('token_expiry'),
                auth_url="https://us.etrade.com/e/t/etws/authorize",
                token_url="https://api.etrade.com/oauth/access_token"
            )
            
            try:
                credential_store.store_credentials('etrade', credentials, AuthMethod.OAUTH)
                logger.info("Stored E*TRADE credentials in credential store")
            except Exception as e:
                logger.error(f"Error storing E*TRADE credentials: {str(e)}")
                success = False
        else:
            logger.warning("Incomplete E*TRADE credentials provided, skipping credential storage")
    
    return success


def create_default_config() -> Dict[str, Any]:
    """
    Create a default authentication configuration.
    
    Returns:
        Dict: Default configuration
    """
    return {
        "credential_store": {
            "type": "encrypted",
            "path": "data/credentials.enc"
        },
        "audit_log": {
            "enabled": True,
            "type": "sqlite",
            "path": "data/trading_audit.db"
        }
    }


def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save authentication configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        
    Returns:
        bool: Success status
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Saved authentication configuration to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving authentication configuration: {str(e)}")
        return False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load authentication configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        logger.info(f"Loaded authentication configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Authentication configuration file not found at {config_path}")
        logger.info("Creating default configuration")
        config = create_default_config()
        save_config(config, config_path)
        return config
    except Exception as e:
        logger.error(f"Error loading authentication configuration: {str(e)}")
        return create_default_config()


def create_audit_log_listener(audit_log: TradeAuditLog, event_bus: Optional[EventBus] = None) -> Optional[AuditLogListener]:
    """
    Create and register an audit log listener with the event bus.
    
    Args:
        audit_log: Audit log instance to record events
        event_bus: Event bus to listen to (uses global if None)
        
    Returns:
        AuditLogListener: Registered audit log listener or None if creation fails
    """
    if not audit_log:
        logger.warning("Cannot create audit log listener: No audit log provided")
        return None
        
    try:
        listener = AuditLogListener(audit_log, event_bus or get_global_event_bus())
        listener.register()
        logger.info("Audit log listener created and registered with event bus")
        return listener
    except Exception as e:
        logger.error(f"Error creating audit log listener: {str(e)}")
        return None


def initialize_auth_system(config: Dict[str, Any]) -> Tuple[Optional[CredentialStore], Optional[TradeAuditLog], Optional[AuditLogListener]]:
    """
    Initialize the complete authentication and audit logging system.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple containing credential store, audit log, and audit log listener
    """
    # Create credential store
    credential_store = create_credential_store(config)
    
    # Create audit log
    audit_log = create_audit_log(config)
    
    # Initialize broker credentials if credential store exists
    if credential_store:
        initialize_broker_credentials(credential_store, config)
        
    # Create and register audit log listener if audit log exists
    audit_listener = None
    if audit_log:
        audit_listener = create_audit_log_listener(audit_log)
        
    return credential_store, audit_log, audit_listener
