#!/usr/bin/env python3
"""
Credential Manager

Provides secure credential loading and management for broker API keys and secrets.
Supports multiple credential sources with clear separation from configuration.
"""

import os
import sys
import json
import logging
import tempfile
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class CredentialError(Exception):
    """Exception raised for credential-related errors"""
    pass

class CredentialManager:
    """
    Manages secure loading and access to broker credentials
    with support for multiple credential sources (files, env vars, vault)
    """
    
    def __init__(
        self,
        credential_dir: str = "config/credentials",
        env_prefix: str = "BENBOT",
        use_mock_in_dev: bool = False,
        dev_mode: bool = False
    ):
        """
        Initialize the credential manager
        
        Args:
            credential_dir: Directory containing credential files
            env_prefix: Prefix for environment variables (e.g., BENBOT_BROKERS_...)
            use_mock_in_dev: Whether to use mock credentials in development mode
            dev_mode: Whether the system is running in development mode
        """
        self.credential_dir = Path(credential_dir)
        self.env_prefix = env_prefix
        self.use_mock_in_dev = use_mock_in_dev
        self.dev_mode = dev_mode
        
        # Create credentials directory if it doesn't exist
        os.makedirs(self.credential_dir, exist_ok=True)
        
        # Track loaded credential sources
        self._credential_sources = {}
        
        logger.debug(f"Initialized CredentialManager with credential_dir={credential_dir}")
    
    def get_broker_credentials(
        self,
        broker_id: str,
        broker_type: str,
        required_fields: list = None,
        allow_missing: bool = False
    ) -> Dict[str, Any]:
        """
        Get credentials for a specific broker
        
        Args:
            broker_id: Broker identifier
            broker_type: Type of broker (alpaca, tradier, etc.)
            required_fields: List of required credential fields
            allow_missing: Whether to allow missing credentials
            
        Returns:
            Dictionary of credentials
            
        Raises:
            CredentialError: If credentials are missing or invalid
        """
        # Skip for paper broker
        if broker_type == "paper":
            logger.debug(f"Paper broker {broker_id} does not require credentials")
            return {}
            
        credentials = {}
        source = "none"
        
        # Try environment variables first
        env_credentials = self._load_from_env(broker_id)
        if env_credentials:
            credentials = env_credentials
            source = "environment"
            logger.info(f"Using credentials from environment variables for {broker_id}")
        
        # Then try credential file
        if not credentials:
            file_credentials = self._load_from_file(broker_id)
            if file_credentials:
                credentials = file_credentials
                source = "file"
                logger.info(f"Using credentials from file for {broker_id}")
        
        # Use mock credentials in dev mode if enabled and needed
        if not credentials and self.dev_mode and self.use_mock_in_dev:
            mock_credentials = self._get_mock_credentials(broker_type)
            if mock_credentials:
                credentials = mock_credentials
                source = "mock"
                logger.warning(f"Using MOCK credentials for {broker_id} - NOT FOR PRODUCTION USE")
        
        # Store the credential source for tracking
        self._credential_sources[broker_id] = source
        
        # Validate required fields if specified
        if required_fields and not allow_missing:
            missing_fields = [field for field in required_fields if field not in credentials]
            if missing_fields:
                raise CredentialError(
                    f"Missing required credentials for {broker_id}: {', '.join(missing_fields)}"
                )
        
        return credentials
    
    def get_credential_source(self, broker_id: str) -> str:
        """Get the source of credentials for a broker"""
        return self._credential_sources.get(broker_id, "unknown")
    
    def has_live_credentials(self, broker_id: str, broker_type: str) -> bool:
        """Check if broker has actual credentials (not mock)"""
        if broker_type == "paper":
            return False
            
        source = self.get_credential_source(broker_id)
        return source in ["file", "environment", "vault"]
    
    def create_credential_file(
        self,
        broker_id: str,
        credentials: Dict[str, Any],
        overwrite: bool = False
    ) -> bool:
        """
        Create a credential file for a broker
        
        Args:
            broker_id: Broker identifier
            credentials: Credential dictionary
            overwrite: Whether to overwrite existing file
            
        Returns:
            True if file was created, False otherwise
        """
        credential_file = self.credential_dir / f"{broker_id}.json"
        
        # Check if file already exists
        if credential_file.exists() and not overwrite:
            logger.warning(f"Credential file {credential_file} already exists")
            return False
        
        # Write credentials to file
        try:
            with open(credential_file, 'w') as f:
                json.dump(credentials, f, indent=2)
            
            # Set secure permissions
            os.chmod(credential_file, 0o600)
            
            logger.info(f"Created credential file {credential_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create credential file: {str(e)}")
            return False
    
    def create_template_file(self, broker_type: str) -> str:
        """
        Create a template credential file
        
        Args:
            broker_type: Type of broker
            
        Returns:
            Path to created template file
        """
        template = self._get_credential_template(broker_type)
        
        # Create temporary file
        fd, path = tempfile.mkstemp(suffix='.json', prefix=f"{broker_type}_template_")
        
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(template, f, indent=2)
            
            logger.info(f"Created credential template at {path}")
            return path
            
        except Exception as e:
            logger.error(f"Failed to create template: {str(e)}")
            return ""
    
    def _load_from_file(self, broker_id: str) -> Dict[str, Any]:
        """Load credentials from file"""
        credential_file = self.credential_dir / f"{broker_id}.json"
        
        if not credential_file.exists():
            logger.debug(f"Credential file {credential_file} not found")
            return {}
        
        try:
            with open(credential_file, 'r') as f:
                credentials = json.load(f)
            
            logger.debug(f"Loaded credentials from {credential_file}")
            return credentials
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in credential file {credential_file}: {str(e)}")
            return {}
            
        except Exception as e:
            logger.error(f"Failed to load credentials from {credential_file}: {str(e)}")
            return {}
    
    def _load_from_env(self, broker_id: str) -> Dict[str, Any]:
        """Load credentials from environment variables"""
        credentials = {}
        prefix = f"{self.env_prefix}_BROKERS_{broker_id.upper()}_CREDENTIALS_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Extract credential key name (after the prefix)
                cred_key = key[len(prefix):].lower()
                credentials[cred_key] = value
        
        if credentials:
            logger.debug(f"Loaded {len(credentials)} credentials from environment for {broker_id}")
        
        return credentials
    
    def _get_mock_credentials(self, broker_type: str) -> Dict[str, Any]:
        """Get mock credentials for development/testing"""
        if broker_type == "alpaca":
            return {
                "api_key": "ALPACA_MOCK_API_KEY",
                "api_secret": "ALPACA_MOCK_API_SECRET",
                "base_url": "https://paper-api.alpaca.markets"
            }
        elif broker_type == "tradier":
            return {
                "api_key": "TRADIER_MOCK_API_KEY",
                "account_id": "TRADIER_MOCK_ACCOUNT_ID"
            }
        elif broker_type == "interactive_brokers":
            return {
                "tws_host": "127.0.0.1",
                "tws_port": 7497,
                "client_id": 1
            }
        
        return {}
    
    def _get_credential_template(self, broker_type: str) -> Dict[str, Any]:
        """Get template for credential file"""
        if broker_type == "alpaca":
            return {
                "api_key": "YOUR_ALPACA_API_KEY",
                "api_secret": "YOUR_ALPACA_API_SECRET",
                "base_url": "https://paper-api.alpaca.markets"  # or https://api.alpaca.markets for live
            }
        elif broker_type == "tradier":
            return {
                "api_key": "YOUR_TRADIER_API_KEY",
                "account_id": "YOUR_TRADIER_ACCOUNT_ID"
            }
        elif broker_type == "interactive_brokers":
            return {
                "tws_host": "127.0.0.1",
                "tws_port": 7497,  # 7496 for TWS, 7497 for IB Gateway
                "client_id": 1
            }
        
        return {}


# Utility functions

def is_dev_environment() -> bool:
    """Determine if running in development environment"""
    return os.environ.get("BENBOT_ENVIRONMENT", "dev").lower() == "dev"

def is_live_trading_enabled() -> bool:
    """Check if live trading is explicitly enabled via env var"""
    return os.environ.get("BENBOT_ENABLE_LIVE_TRADING", "").lower() == "true"

def safe_to_trade_live() -> bool:
    """Determine if it's safe to trade live based on environment and settings"""
    # Only allow live trading if explicitly enabled in non-dev environments
    if is_dev_environment():
        # Extra safeguard in dev - require explicit override
        return is_live_trading_enabled() and os.environ.get("BENBOT_CONFIRM_LIVE_TRADING_IN_DEV", "").lower() == "i_understand_the_risks"
    
    # In production, require explicit live trading flag
    return is_live_trading_enabled()
