"""
Credential Management Dashboard Component

This module provides a Streamlit dashboard component for managing broker credentials.
It allows users to view, add, edit, and delete broker credentials stored in the
credential store system.
"""

import streamlit as st
import pandas as pd
import os
import json
from typing import Dict, Any, List, Optional, Tuple
import logging

from trading_bot.brokers.credential_store import (
    CredentialStore, AuthMethod, BrokerCredentials,
    EncryptedFileStore, YamlFileStore
)
from trading_bot.brokers.auth_manager import (
    create_credential_store, initialize_broker_credentials, save_config
)

# Configure logging
logger = logging.getLogger(__name__)


def setup_credential_store(config_path: str) -> Tuple[Optional[CredentialStore], Dict[str, Any]]:
    """
    Set up credential store from configuration file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (credential_store, config)
    """
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Create credential store
        credential_store = create_credential_store(config)
        if not credential_store:
            st.error("Failed to initialize credential store. Check logs for details.")
            return None, config
            
        return credential_store, config
    except Exception as e:
        st.error(f"Error setting up credential store: {str(e)}")
        logger.error(f"Error setting up credential store: {str(e)}")
        return None, {}


def render_credential_management(config_path: str = "config/broker_config.json"):
    """
    Render the credential management dashboard
    
    Args:
        config_path: Path to the broker configuration file
    """
    st.title("Broker Credential Management")
    
    st.write("""
    This dashboard allows you to securely manage credentials for all your trading brokers.
    All sensitive information is stored in an encrypted format and is never exposed in plain text.
    """)
    
    # Setup credential store
    credential_store, config = setup_credential_store(config_path)
    if credential_store is None:
        return
    
    # Sidebar for master password management
    with st.sidebar:
        st.header("Security Settings")
        
        # Master password option
        use_env_var = st.checkbox(
            "Use environment variable for master password", 
            value=os.environ.get('TRADING_BOT_MASTER_PASSWORD') is not None,
            help="When enabled, the master password is read from the TRADING_BOT_MASTER_PASSWORD environment variable"
        )
        
        if not use_env_var:
            master_password = st.text_input(
                "Master Password", 
                type="password",
                help="This password is used to encrypt and decrypt your credentials"
            )
            if master_password:
                # Update config with new master password
                if 'credential_store' not in config:
                    config['credential_store'] = {}
                config['credential_store']['master_password'] = master_password
                
                # Save config
                if st.button("Save Master Password"):
                    if save_config(config, config_path):
                        st.success("Master password saved to configuration")
                    else:
                        st.error("Failed to save master password")
    
    # Main credential management interface
    st.header("Manage Broker Credentials")
    
    # Get available brokers
    available_brokers = []
    try:
        available_brokers = credential_store.list_brokers()
    except Exception as e:
        st.error(f"Error listing brokers: {str(e)}")
    
    # First tab: View existing credentials
    tab1, tab2, tab3 = st.tabs(["View Credentials", "Add/Edit Credentials", "Delete Credentials"])
    
    with tab1:
        st.subheader("Existing Broker Credentials")
        
        if not available_brokers:
            st.info("No broker credentials found. Use the 'Add/Edit Credentials' tab to set up your brokers.")
        else:
            # Create a table of broker information (without sensitive data)
            broker_data = []
            
            for broker_id in available_brokers:
                try:
                    auth_method = credential_store.get_auth_method(broker_id)
                    credentials = credential_store.get_credentials(broker_id)
                    
                    # Create a sanitized version of credentials for display
                    sanitized_creds = {"auth_method": auth_method.value}
                    
                    if auth_method == AuthMethod.API_KEY:
                        sanitized_creds["api_key"] = "********" 
                        
                    elif auth_method == AuthMethod.OAUTH:
                        sanitized_creds["client_id"] = "********"
                        sanitized_creds["has_token"] = credentials.access_token is not None
                        
                    elif auth_method == AuthMethod.USERNAME_PASSWORD:
                        sanitized_creds["username"] = credentials.username
                        sanitized_creds["has_password"] = True
                        
                    broker_data.append({
                        "Broker ID": broker_id,
                        "Auth Method": auth_method.value,
                        "Status": "Valid" if credential_store.validate_credentials(broker_id) else "Invalid",
                        "Details": json.dumps(sanitized_creds)
                    })
                    
                except Exception as e:
                    broker_data.append({
                        "Broker ID": broker_id,
                        "Auth Method": "Unknown",
                        "Status": f"Error: {str(e)}",
                        "Details": "{}"
                    })
            
            if broker_data:
                st.dataframe(pd.DataFrame(broker_data))
            else:
                st.info("No broker credentials found")
        
        # Test credentials button
        if available_brokers:
            broker_to_test = st.selectbox(
                "Select broker to test credentials", 
                options=available_brokers,
                help="Test if the stored credentials are valid"
            )
            
            if st.button("Test Credentials"):
                try:
                    if credential_store.validate_credentials(broker_to_test):
                        st.success(f"Credentials for {broker_to_test} are valid")
                    else:
                        st.error(f"Credentials for {broker_to_test} are invalid or expired")
                except Exception as e:
                    st.error(f"Error testing credentials: {str(e)}")
    
    # Second tab: Add/Edit credentials
    with tab2:
        st.subheader("Add or Edit Broker Credentials")
        
        # Broker selection
        broker_id = st.text_input(
            "Broker ID",
            help="Unique identifier for the broker (e.g., 'tradier', 'alpaca')"
        )
        
        # Authentication method selection
        auth_method = st.selectbox(
            "Authentication Method",
            options=[
                ("API Key", AuthMethod.API_KEY.value),
                ("OAuth", AuthMethod.OAUTH.value),
                ("Username/Password", AuthMethod.USERNAME_PASSWORD.value),
                ("Certificate", AuthMethod.CERTIFICATE.value),
                ("Token", AuthMethod.TOKEN.value)
            ],
            format_func=lambda x: x[0],
            help="Select the authentication method required by this broker"
        )
        
        auth_method_value = auth_method[1]
        
        # Credential form based on selected auth method
        credentials_dict = {}
        
        if auth_method_value == AuthMethod.API_KEY.value:
            api_key = st.text_input(
                "API Key", 
                type="password",
                help="Enter the broker's API key"
            )
            api_secret = st.text_input(
                "API Secret",
                type="password",
                help="Enter the broker's API secret (if required)"
            )
            
            credentials_dict = {
                "api_key": api_key,
                "api_secret": api_secret
            }
            
        elif auth_method_value == AuthMethod.OAUTH.value:
            client_id = st.text_input(
                "Client ID", 
                help="OAuth client ID"
            )
            client_secret = st.text_input(
                "Client Secret",
                type="password",
                help="OAuth client secret"
            )
            access_token = st.text_input(
                "Access Token (if available)",
                type="password",
                help="OAuth access token (if already authorized)"
            )
            refresh_token = st.text_input(
                "Refresh Token (if available)",
                type="password",
                help="OAuth refresh token (if already authorized)"
            )
            
            credentials_dict = {
                "client_id": client_id,
                "client_secret": client_secret,
                "access_token": access_token,
                "refresh_token": refresh_token
            }
            
        elif auth_method_value == AuthMethod.USERNAME_PASSWORD.value:
            username = st.text_input(
                "Username", 
                help="Username for broker login"
            )
            password = st.text_input(
                "Password",
                type="password",
                help="Password for broker login"
            )
            
            credentials_dict = {
                "username": username,
                "password": password
            }
            
        elif auth_method_value == AuthMethod.TOKEN.value:
            token = st.text_input(
                "Token", 
                type="password",
                help="Authentication token"
            )
            
            credentials_dict = {
                "token": token
            }
            
        elif auth_method_value == AuthMethod.CERTIFICATE.value:
            cert_path = st.text_input(
                "Certificate Path", 
                help="Path to the certificate file"
            )
            key_path = st.text_input(
                "Key Path",
                help="Path to the private key file"
            )
            
            credentials_dict = {
                "cert_path": cert_path,
                "key_path": key_path
            }
        
        # Additional fields
        with st.expander("Additional Settings"):
            sandbox = st.checkbox(
                "Sandbox/Paper Trading", 
                value=True,
                help="Use sandbox or paper trading environment"
            )
            
            account_id = st.text_input(
                "Account ID (if required)",
                help="Specific account ID for this broker"
            )
            
            credentials_dict["sandbox"] = sandbox
            if account_id:
                credentials_dict["account_id"] = account_id
        
        # Save credentials button
        if st.button("Save Credentials"):
            if not broker_id:
                st.error("Broker ID is required")
            else:
                try:
                    # Check if we have required fields
                    has_required_fields = False
                    
                    if auth_method_value == AuthMethod.API_KEY.value:
                        has_required_fields = bool(credentials_dict.get("api_key"))
                    elif auth_method_value == AuthMethod.OAUTH.value:
                        has_required_fields = bool(credentials_dict.get("client_id") and credentials_dict.get("client_secret"))
                    elif auth_method_value == AuthMethod.USERNAME_PASSWORD.value:
                        has_required_fields = bool(credentials_dict.get("username") and credentials_dict.get("password"))
                    elif auth_method_value == AuthMethod.TOKEN.value:
                        has_required_fields = bool(credentials_dict.get("token"))
                    elif auth_method_value == AuthMethod.CERTIFICATE.value:
                        has_required_fields = bool(credentials_dict.get("cert_path") and credentials_dict.get("key_path"))
                    
                    if not has_required_fields:
                        st.error("Required credential fields are missing")
                    else:
                        # Create and store credentials
                        if auth_method_value == AuthMethod.API_KEY.value:
                            credential_store.store_api_key_credentials(
                                broker_id, 
                                credentials_dict["api_key"],
                                credentials_dict.get("api_secret", "")
                            )
                        elif auth_method_value == AuthMethod.OAUTH.value:
                            credential_store.store_oauth_credentials(
                                broker_id,
                                credentials_dict["client_id"],
                                credentials_dict["client_secret"],
                                credentials_dict.get("access_token", ""),
                                credentials_dict.get("refresh_token", "")
                            )
                        elif auth_method_value == AuthMethod.USERNAME_PASSWORD.value:
                            credential_store.store_username_password_credentials(
                                broker_id,
                                credentials_dict["username"],
                                credentials_dict["password"]
                            )
                        elif auth_method_value == AuthMethod.TOKEN.value:
                            credential_store.store_token_credentials(
                                broker_id,
                                credentials_dict["token"]
                            )
                        elif auth_method_value == AuthMethod.CERTIFICATE.value:
                            credential_store.store_certificate_credentials(
                                broker_id,
                                credentials_dict["cert_path"],
                                credentials_dict["key_path"]
                            )
                            
                        # Also update the broker in the config
                        if 'brokers' not in config:
                            config['brokers'] = {}
                            
                        if broker_id not in config['brokers']:
                            config['brokers'][broker_id] = {}
                            
                        config['brokers'][broker_id]['enabled'] = True
                        config['brokers'][broker_id]['sandbox'] = credentials_dict.get("sandbox", True)
                        
                        if "account_id" in credentials_dict:
                            config['brokers'][broker_id]['account_id'] = credentials_dict["account_id"]
                        
                        # Save updated config
                        if save_config(config, config_path):
                            st.success(f"Credentials for {broker_id} saved successfully")
                        else:
                            st.warning("Credentials saved but failed to update configuration file")
                        
                except Exception as e:
                    st.error(f"Error saving credentials: {str(e)}")
                    logger.error(f"Error saving credentials: {str(e)}")
    
    # Third tab: Delete credentials
    with tab3:
        st.subheader("Delete Broker Credentials")
        
        if not available_brokers:
            st.info("No broker credentials found to delete")
        else:
            broker_to_delete = st.selectbox(
                "Select broker", 
                options=available_brokers,
                help="Select the broker whose credentials you want to delete"
            )
            
            # Confirmation
            confirm = st.checkbox("I understand this action cannot be undone")
            
            if st.button("Delete Credentials", disabled=not confirm):
                try:
                    credential_store.delete_credentials(broker_to_delete)
                    
                    # Also update the config
                    if 'brokers' in config and broker_to_delete in config['brokers']:
                        config['brokers'][broker_to_delete]['enabled'] = False
                        
                        # Save updated config
                        if save_config(config, config_path):
                            st.success(f"Credentials for {broker_to_delete} deleted successfully")
                        else:
                            st.warning("Credentials deleted but failed to update configuration file")
                    else:
                        st.success(f"Credentials for {broker_to_delete} deleted successfully")
                        
                except Exception as e:
                    st.error(f"Error deleting credentials: {str(e)}")
                    logger.error(f"Error deleting credentials: {str(e)}")
    
    st.divider()
    st.caption("🔒 All sensitive information is encrypted using Fernet symmetric encryption")
