"""
Credential Store

Provides secure storage and management for broker credentials with
support for different authentication methods and providers.
"""

import logging
import os
import json
import base64
import yaml
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import threading
from pathlib import Path
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Configure logging
logger = logging.getLogger(__name__)


class AuthMethod(Enum):
    """Types of authentication methods supported"""
    API_KEY = "api_key"
    OAUTH = "oauth"
    USERNAME_PASSWORD = "username_password"
    CERTIFICATE = "certificate"
    TOKEN = "token"
    CUSTOM = "custom"


class CredentialStore:
    """
    Abstract base class for credential storage systems
    
    Implementations should provide secure storage and retrieval
    of broker credentials.
    """
    
    def __init__(self):
        """Initialize the credential store"""
        self._lock = threading.RLock()
    
    def get_credentials(self, broker_id: str) -> Dict[str, Any]:
        """
        Get credentials for a broker
        
        Args:
            broker_id: Broker identifier
            
        Returns:
            Dict: Credentials for the broker
            
        Raises:
            KeyError: If broker_id not found
        """
        raise NotImplementedError("Subclasses must implement get_credentials")
    
    def store_credentials(self, broker_id: str, credentials: Dict[str, Any]) -> bool:
        """
        Store credentials for a broker
        
        Args:
            broker_id: Broker identifier
            credentials: Credentials to store
            
        Returns:
            bool: Success status
        """
        raise NotImplementedError("Subclasses must implement store_credentials")
    
    def delete_credentials(self, broker_id: str) -> bool:
        """
        Delete credentials for a broker
        
        Args:
            broker_id: Broker identifier
            
        Returns:
            bool: Success status
        """
        raise NotImplementedError("Subclasses must implement delete_credentials")
    
    def list_brokers(self) -> List[str]:
        """
        List all broker IDs with stored credentials
        
        Returns:
            List[str]: List of broker IDs
        """
        raise NotImplementedError("Subclasses must implement list_brokers")
    
    def get_auth_method(self, broker_id: str) -> AuthMethod:
        """
        Get the authentication method for a broker
        
        Args:
            broker_id: Broker identifier
            
        Returns:
            AuthMethod: Authentication method
            
        Raises:
            KeyError: If broker_id not found
        """
        raise NotImplementedError("Subclasses must implement get_auth_method")


class EncryptedFileStore(CredentialStore):
    """
    Stores credentials in an encrypted file
    
    Uses Fernet symmetric encryption with a master password.
    """
    
    def __init__(self, file_path: str, master_password: Optional[str] = None):
        """
        Initialize the encrypted file store
        
        Args:
            file_path: Path to the encrypted file
            master_password: Master password for encryption (if None, will try to
                            use TRADING_BOT_MASTER_PASSWORD environment variable)
        """
        super().__init__()
        self.file_path = file_path
        
        # Get master password from argument or environment
        if master_password is None:
            master_password = os.environ.get('TRADING_BOT_MASTER_PASSWORD')
            if master_password is None:
                raise ValueError("Master password required but not provided")
        
        # Generate encryption key from password
        self.key = self._generate_key(master_password)
        self.fernet = Fernet(self.key)
        
        # Create file if it doesn't exist
        if not os.path.exists(file_path):
            self._save_credentials({})
        
        logger.info(f"Initialized EncryptedFileStore with file: {file_path}")
    
    def _generate_key(self, password: str) -> bytes:
        """Generate encryption key from password"""
        # Use password to derive a key
        password_bytes = password.encode()
        salt = b'trading_bot_salt'  # In production, use a secure random salt
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return key
    
    def _load_credentials(self) -> Dict[str, Dict[str, Any]]:
        """Load and decrypt credentials from file"""
        with self._lock:
            try:
                with open(self.file_path, 'rb') as f:
                    encrypted_data = f.read()
                
                if not encrypted_data:
                    return {}
                
                decrypted_data = self.fernet.decrypt(encrypted_data)
                return json.loads(decrypted_data)
            except FileNotFoundError:
                return {}
            except Exception as e:
                logger.error(f"Error loading credentials: {str(e)}")
                return {}
    
    def _save_credentials(self, credentials: Dict[str, Dict[str, Any]]) -> bool:
        """Encrypt and save credentials to file"""
        with self._lock:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
                
                # Encrypt and save
                encrypted_data = self.fernet.encrypt(json.dumps(credentials).encode())
                
                with open(self.file_path, 'wb') as f:
                    f.write(encrypted_data)
                
                return True
            except Exception as e:
                logger.error(f"Error saving credentials: {str(e)}")
                return False
    
    def get_credentials(self, broker_id: str) -> Dict[str, Any]:
        """Get credentials for a broker"""
        credentials = self._load_credentials()
        
        if broker_id not in credentials:
            raise KeyError(f"No credentials found for broker: {broker_id}")
        
        return credentials[broker_id]['credentials']
    
    def store_credentials(self, broker_id: str, credentials: Dict[str, Any],
                        auth_method: AuthMethod = AuthMethod.API_KEY) -> bool:
        """Store credentials for a broker"""
        all_credentials = self._load_credentials()
        
        all_credentials[broker_id] = {
            'credentials': credentials,
            'auth_method': auth_method.value,
            'last_updated': datetime.now().isoformat()
        }
        
        return self._save_credentials(all_credentials)
    
    def delete_credentials(self, broker_id: str) -> bool:
        """Delete credentials for a broker"""
        all_credentials = self._load_credentials()
        
        if broker_id not in all_credentials:
            return False
        
        del all_credentials[broker_id]
        return self._save_credentials(all_credentials)
    
    def list_brokers(self) -> List[str]:
        """List all broker IDs with stored credentials"""
        return list(self._load_credentials().keys())
    
    def get_auth_method(self, broker_id: str) -> AuthMethod:
        """Get the authentication method for a broker"""
        credentials = self._load_credentials()
        
        if broker_id not in credentials:
            raise KeyError(f"No credentials found for broker: {broker_id}")
        
        method_str = credentials[broker_id].get('auth_method', 'api_key')
        
        # Convert string to enum
        try:
            return AuthMethod(method_str)
        except ValueError:
            # Default to API_KEY if unknown method
            return AuthMethod.API_KEY


class YamlFileStore(CredentialStore):
    """
    Stores credentials in a YAML file
    
    Note: This is less secure than EncryptedFileStore and should only be
    used for development or in secure environments.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the YAML file store
        
        Args:
            file_path: Path to the YAML file
        """
        super().__init__()
        self.file_path = file_path
        
        # Create file if it doesn't exist
        if not os.path.exists(file_path):
            self._save_credentials({})
        
        logger.info(f"Initialized YamlFileStore with file: {file_path}")
    
    def _load_credentials(self) -> Dict[str, Dict[str, Any]]:
        """Load credentials from YAML file"""
        with self._lock:
            try:
                with open(self.file_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            except FileNotFoundError:
                return {}
            except Exception as e:
                logger.error(f"Error loading credentials: {str(e)}")
                return {}
    
    def _save_credentials(self, credentials: Dict[str, Dict[str, Any]]) -> bool:
        """Save credentials to YAML file"""
        with self._lock:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
                
                # Save YAML
                with open(self.file_path, 'w') as f:
                    yaml.dump(credentials, f, default_flow_style=False)
                
                return True
            except Exception as e:
                logger.error(f"Error saving credentials: {str(e)}")
                return False
    
    def get_credentials(self, broker_id: str) -> Dict[str, Any]:
        """Get credentials for a broker"""
        credentials = self._load_credentials()
        
        if broker_id not in credentials:
            raise KeyError(f"No credentials found for broker: {broker_id}")
        
        return credentials[broker_id]['credentials']
    
    def store_credentials(self, broker_id: str, credentials: Dict[str, Any],
                        auth_method: AuthMethod = AuthMethod.API_KEY) -> bool:
        """Store credentials for a broker"""
        all_credentials = self._load_credentials()
        
        all_credentials[broker_id] = {
            'credentials': credentials,
            'auth_method': auth_method.value,
            'last_updated': datetime.now().isoformat()
        }
        
        return self._save_credentials(all_credentials)
    
    def delete_credentials(self, broker_id: str) -> bool:
        """Delete credentials for a broker"""
        all_credentials = self._load_credentials()
        
        if broker_id not in all_credentials:
            return False
        
        del all_credentials[broker_id]
        return self._save_credentials(all_credentials)
    
    def list_brokers(self) -> List[str]:
        """List all broker IDs with stored credentials"""
        return list(self._load_credentials().keys())
    
    def get_auth_method(self, broker_id: str) -> AuthMethod:
        """Get the authentication method for a broker"""
        credentials = self._load_credentials()
        
        if broker_id not in credentials:
            raise KeyError(f"No credentials found for broker: {broker_id}")
        
        method_str = credentials[broker_id].get('auth_method', 'api_key')
        
        # Convert string to enum
        try:
            return AuthMethod(method_str)
        except ValueError:
            # Default to API_KEY if unknown method
            return AuthMethod.API_KEY


class CredentialFactory:
    """
    Factory for creating broker credentials
    
    Provides methods to create different types of credentials
    for various authentication methods.
    """
    
    @staticmethod
    def create_api_key_credentials(api_key: str, api_secret: str, 
                                 additional_params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Create API key credentials
        
        Args:
            api_key: API key
            api_secret: API secret
            additional_params: Additional parameters
            
        Returns:
            Dict: API key credentials
        """
        credentials = {
            'api_key': api_key,
            'api_secret': api_secret,
            'auth_method': AuthMethod.API_KEY.value,
            'created_at': datetime.now().isoformat()
        }
        
        if additional_params:
            credentials.update(additional_params)
        
        return credentials
    
    @staticmethod
    def create_oauth_credentials(client_id: str, client_secret: str,
                              access_token: Optional[str] = None,
                              refresh_token: Optional[str] = None,
                              token_expiry: Optional[str] = None,
                              auth_url: Optional[str] = None,
                              token_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Create OAuth credentials
        
        Args:
            client_id: OAuth client ID
            client_secret: OAuth client secret
            access_token: Current access token
            refresh_token: Refresh token
            token_expiry: Token expiry timestamp
            auth_url: Authorization URL
            token_url: Token URL
            
        Returns:
            Dict: OAuth credentials
        """
        return {
            'client_id': client_id,
            'client_secret': client_secret,
            'access_token': access_token,
            'refresh_token': refresh_token,
            'token_expiry': token_expiry,
            'auth_url': auth_url,
            'token_url': token_url,
            'auth_method': AuthMethod.OAUTH.value,
            'created_at': datetime.now().isoformat()
        }
    
    @staticmethod
    def create_username_password_credentials(username: str, password: str,
                                         additional_params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Create username/password credentials
        
        Args:
            username: Username
            password: Password
            additional_params: Additional parameters
            
        Returns:
            Dict: Username/password credentials
        """
        credentials = {
            'username': username,
            'password': password,
            'auth_method': AuthMethod.USERNAME_PASSWORD.value,
            'created_at': datetime.now().isoformat()
        }
        
        if additional_params:
            credentials.update(additional_params)
        
        return credentials
    
    @staticmethod
    def create_certificate_credentials(cert_path: str, key_path: Optional[str] = None,
                                    passphrase: Optional[str] = None) -> Dict[str, Any]:
        """
        Create certificate-based credentials
        
        Args:
            cert_path: Path to certificate file
            key_path: Path to key file
            passphrase: Certificate passphrase
            
        Returns:
            Dict: Certificate credentials
        """
        return {
            'cert_path': cert_path,
            'key_path': key_path,
            'passphrase': passphrase,
            'auth_method': AuthMethod.CERTIFICATE.value,
            'created_at': datetime.now().isoformat()
        }
    
    @staticmethod
    def create_token_credentials(token: str, token_type: Optional[str] = None,
                             token_expiry: Optional[str] = None) -> Dict[str, Any]:
        """
        Create token-based credentials
        
        Args:
            token: Authentication token
            token_type: Token type (e.g., 'Bearer')
            token_expiry: Token expiry timestamp
            
        Returns:
            Dict: Token credentials
        """
        return {
            'token': token,
            'token_type': token_type,
            'token_expiry': token_expiry,
            'auth_method': AuthMethod.TOKEN.value,
            'created_at': datetime.now().isoformat()
        }


class AuthenticatorFactory:
    """
    Factory for creating authenticator instances
    
    Provides methods to create authenticators for different
    authentication methods and credential stores.
    """
    
    @staticmethod
    def create_authenticator(auth_method: AuthMethod, credential_store: CredentialStore,
                          broker_id: str) -> 'BaseAuthenticator':
        """
        Create an authenticator for a broker
        
        Args:
            auth_method: Authentication method
            credential_store: Credential store to use
            broker_id: Broker identifier
            
        Returns:
            BaseAuthenticator: Authenticator instance
            
        Raises:
            ValueError: If auth_method is not supported
        """
        if auth_method == AuthMethod.API_KEY:
            return ApiKeyAuthenticator(credential_store, broker_id)
        elif auth_method == AuthMethod.OAUTH:
            return OAuthAuthenticator(credential_store, broker_id)
        elif auth_method == AuthMethod.USERNAME_PASSWORD:
            return UsernamePasswordAuthenticator(credential_store, broker_id)
        elif auth_method == AuthMethod.CERTIFICATE:
            return CertificateAuthenticator(credential_store, broker_id)
        elif auth_method == AuthMethod.TOKEN:
            return TokenAuthenticator(credential_store, broker_id)
        else:
            raise ValueError(f"Unsupported authentication method: {auth_method}")


class BaseAuthenticator:
    """
    Base class for authenticators
    
    Authenticators are responsible for handling the authentication
    process for a specific broker using a specific authentication method.
    """
    
    def __init__(self, credential_store: CredentialStore, broker_id: str):
        """
        Initialize the authenticator
        
        Args:
            credential_store: Credential store to use
            broker_id: Broker identifier
        """
        self.credential_store = credential_store
        self.broker_id = broker_id
    
    def authenticate(self) -> Dict[str, Any]:
        """
        Authenticate with the broker
        
        Returns:
            Dict: Authentication result with token or session information
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement authenticate")
    
    def refresh(self) -> Dict[str, Any]:
        """
        Refresh authentication if needed
        
        Returns:
            Dict: Refreshed authentication result
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement refresh")
    
    def is_expired(self) -> bool:
        """
        Check if authentication is expired
        
        Returns:
            bool: True if expired, False otherwise
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement is_expired")
    
    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get headers for authentication
        
        Returns:
            Dict: Authentication headers
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement get_auth_headers")


class ApiKeyAuthenticator(BaseAuthenticator):
    """Authenticator for API key authentication"""
    
    def authenticate(self) -> Dict[str, Any]:
        """Authenticate with API key"""
        # API key authentication is usually stateless, just return credentials
        credentials = self.credential_store.get_credentials(self.broker_id)
        return {
            'authenticated': True,
            'credentials': credentials,
            'timestamp': datetime.now().isoformat()
        }
    
    def refresh(self) -> Dict[str, Any]:
        """Refresh API key authentication"""
        # API keys don't typically need refreshing
        return self.authenticate()
    
    def is_expired(self) -> bool:
        """Check if API key is expired"""
        # API keys don't typically expire
        return False
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get headers for API key authentication"""
        credentials = self.credential_store.get_credentials(self.broker_id)
        
        # Common pattern for API key authentication
        return {
            'X-API-Key': credentials.get('api_key', ''),
            'X-API-Secret': credentials.get('api_secret', '')
        }


class OAuthAuthenticator(BaseAuthenticator):
    """Authenticator for OAuth authentication"""
    
    def authenticate(self) -> Dict[str, Any]:
        """Authenticate with OAuth"""
        # Implementation would interact with OAuth endpoints
        # This is a simplified placeholder
        credentials = self.credential_store.get_credentials(self.broker_id)
        
        # Check if we already have a valid token
        if (credentials.get('access_token') and credentials.get('token_expiry') and
            not self._is_token_expired(credentials.get('token_expiry'))):
            return {
                'authenticated': True,
                'access_token': credentials.get('access_token'),
                'token_type': 'Bearer',
                'expires_at': credentials.get('token_expiry'),
                'timestamp': datetime.now().isoformat()
            }
        
        # Otherwise, we would need to perform OAuth flow
        # In a real implementation, this would redirect the user or use refresh token
        logger.warning(f"OAuth authentication for {self.broker_id} needs manual flow")
        
        return {
            'authenticated': False,
            'error': 'OAuth flow required',
            'auth_url': credentials.get('auth_url'),
            'timestamp': datetime.now().isoformat()
        }
    
    def refresh(self) -> Dict[str, Any]:
        """Refresh OAuth token"""
        credentials = self.credential_store.get_credentials(self.broker_id)
        
        if not credentials.get('refresh_token'):
            logger.error(f"No refresh token available for {self.broker_id}")
            return {
                'authenticated': False,
                'error': 'No refresh token available',
                'timestamp': datetime.now().isoformat()
            }
        
        # In a real implementation, this would call the token endpoint
        # This is a simplified placeholder
        logger.warning(f"OAuth token refresh for {self.broker_id} would be implemented here")
        
        return {
            'authenticated': False,
            'error': 'Token refresh not implemented',
            'timestamp': datetime.now().isoformat()
        }
    
    def is_expired(self) -> bool:
        """Check if OAuth token is expired"""
        credentials = self.credential_store.get_credentials(self.broker_id)
        return self._is_token_expired(credentials.get('token_expiry'))
    
    def _is_token_expired(self, expiry_str: Optional[str]) -> bool:
        """Check if a token expiry timestamp indicates expiration"""
        if not expiry_str:
            return True
        
        try:
            expiry = datetime.fromisoformat(expiry_str)
            # Consider token expired if less than 5 minutes remaining
            return datetime.now() + timedelta(minutes=5) >= expiry
        except ValueError:
            return True
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get headers for OAuth authentication"""
        credentials = self.credential_store.get_credentials(self.broker_id)
        
        if self.is_expired():
            # Try to refresh token
            refresh_result = self.refresh()
            if refresh_result.get('authenticated'):
                # Update stored credentials with new token
                credentials['access_token'] = refresh_result.get('access_token')
                credentials['token_expiry'] = refresh_result.get('expires_at')
                self.credential_store.store_credentials(
                    self.broker_id, credentials, AuthMethod.OAUTH
                )
        
        # Return Authorization header with token
        return {
            'Authorization': f"Bearer {credentials.get('access_token', '')}"
        }


class UsernamePasswordAuthenticator(BaseAuthenticator):
    """Authenticator for username/password authentication"""
    
    def authenticate(self) -> Dict[str, Any]:
        """Authenticate with username/password"""
        # Implementation would call login endpoint
        # This is a simplified placeholder
        credentials = self.credential_store.get_credentials(self.broker_id)
        
        logger.warning(f"Username/password auth for {self.broker_id} would be implemented here")
        
        return {
            'authenticated': False,
            'error': 'Username/password auth not implemented',
            'timestamp': datetime.now().isoformat()
        }
    
    def refresh(self) -> Dict[str, Any]:
        """Refresh username/password authentication"""
        # Usually means re-authenticating
        return self.authenticate()
    
    def is_expired(self) -> bool:
        """Check if session is expired"""
        # Without a real implementation, assume expired
        return True
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get headers for username/password authentication"""
        # This might return a session token if available
        return {}


class CertificateAuthenticator(BaseAuthenticator):
    """Authenticator for certificate-based authentication"""
    
    def authenticate(self) -> Dict[str, Any]:
        """Authenticate with certificate"""
        # Certificate auth is typically handled at connection level
        credentials = self.credential_store.get_credentials(self.broker_id)
        
        # Check if certificate files exist
        cert_path = credentials.get('cert_path')
        key_path = credentials.get('key_path')
        
        if not cert_path or not os.path.exists(cert_path):
            return {
                'authenticated': False,
                'error': f"Certificate file not found: {cert_path}",
                'timestamp': datetime.now().isoformat()
            }
        
        if key_path and not os.path.exists(key_path):
            return {
                'authenticated': False,
                'error': f"Key file not found: {key_path}",
                'timestamp': datetime.now().isoformat()
            }
        
        return {
            'authenticated': True,
            'cert_path': cert_path,
            'key_path': key_path,
            'timestamp': datetime.now().isoformat()
        }
    
    def refresh(self) -> Dict[str, Any]:
        """Refresh certificate authentication"""
        # Certificates don't typically need refreshing unless expired
        return self.authenticate()
    
    def is_expired(self) -> bool:
        """Check if certificate is expired"""
        # Would require parsing certificate to check expiry
        # For now, assume not expired
        return False
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get headers for certificate authentication"""
        # Certificate auth typically doesn't use headers
        return {}
    
    def get_cert_tuple(self) -> Optional[tuple]:
        """
        Get cert/key tuple for requests library
        
        Returns:
            Optional[tuple]: (cert_path, key_path) or cert_path
        """
        credentials = self.credential_store.get_credentials(self.broker_id)
        
        cert_path = credentials.get('cert_path')
        key_path = credentials.get('key_path')
        
        if not cert_path:
            return None
        
        if key_path:
            return (cert_path, key_path)
        else:
            return cert_path


class TokenAuthenticator(BaseAuthenticator):
    """Authenticator for token-based authentication"""
    
    def authenticate(self) -> Dict[str, Any]:
        """Authenticate with token"""
        # Token auth is usually stateless, just return token
        credentials = self.credential_store.get_credentials(self.broker_id)
        
        return {
            'authenticated': True,
            'token': credentials.get('token'),
            'token_type': credentials.get('token_type'),
            'expires_at': credentials.get('token_expiry'),
            'timestamp': datetime.now().isoformat()
        }
    
    def refresh(self) -> Dict[str, Any]:
        """Refresh token"""
        # Would need token refresh logic, simplified placeholder
        return self.authenticate()
    
    def is_expired(self) -> bool:
        """Check if token is expired"""
        credentials = self.credential_store.get_credentials(self.broker_id)
        expiry_str = credentials.get('token_expiry')
        
        if not expiry_str:
            # If no expiry, assume not expired
            return False
        
        try:
            expiry = datetime.fromisoformat(expiry_str)
            # Consider token expired if less than 5 minutes remaining
            return datetime.now() + timedelta(minutes=5) >= expiry
        except ValueError:
            return False
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get headers for token authentication"""
        credentials = self.credential_store.get_credentials(self.broker_id)
        
        token = credentials.get('token', '')
        token_type = credentials.get('token_type', 'Bearer')
        
        return {
            'Authorization': f"{token_type} {token}"
        }
