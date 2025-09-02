"""
Tests for the CredentialStore module
"""

import os
import unittest
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from trading_bot.brokers.credential_store import (
    CredentialStore, EncryptedFileStore, YamlFileStore, 
    AuthMethod, CredentialFactory, AuthenticatorFactory,
    ApiKeyAuthenticator, OAuthAuthenticator
)


class TestCredentialStore(unittest.TestCase):
    """Test cases for credential store implementations"""

    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for test files
        self.test_dir = tempfile.TemporaryDirectory()
        self.yaml_file = os.path.join(self.test_dir.name, "test_creds.yml")
        self.encrypted_file = os.path.join(self.test_dir.name, "test_creds.enc")
        
        # Sample test credentials
        self.test_credentials = {
            "api_key": "test_api_key",
            "api_secret": "test_api_secret",
            "auth_method": "api_key"
        }
        
        # Test master password
        self.test_password = "test_password_123!"
        
        # Mock environment variable
        os.environ['TRADING_BOT_MASTER_PASSWORD'] = self.test_password

    def tearDown(self):
        """Clean up after tests"""
        self.test_dir.cleanup()
        if 'TRADING_BOT_MASTER_PASSWORD' in os.environ:
            del os.environ['TRADING_BOT_MASTER_PASSWORD']

    def test_yaml_file_store(self):
        """Test YamlFileStore functionality"""
        # Create store
        store = YamlFileStore(self.yaml_file)
        
        # Store credentials
        self.assertTrue(store.store_credentials("etrade", self.test_credentials))
        
        # List brokers
        self.assertEqual(store.list_brokers(), ["etrade"])
        
        # Get credentials
        creds = store.get_credentials("etrade")
        self.assertEqual(creds["api_key"], "test_api_key")
        
        # Verify file was created
        self.assertTrue(os.path.exists(self.yaml_file))
        
        # Verify file contains expected data
        with open(self.yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        self.assertIn("etrade", data)
        self.assertEqual(data["etrade"]["credentials"]["api_key"], "test_api_key")
        
        # Test delete credentials
        self.assertTrue(store.delete_credentials("etrade"))
        self.assertEqual(store.list_brokers(), [])
        
        # Test get auth method
        store.store_credentials("alpaca", self.test_credentials, AuthMethod.API_KEY)
        self.assertEqual(store.get_auth_method("alpaca"), AuthMethod.API_KEY)

    def test_encrypted_file_store(self):
        """Test EncryptedFileStore functionality"""
        # Create store with explicit password
        store = EncryptedFileStore(self.encrypted_file, self.test_password)
        
        # Store credentials
        self.assertTrue(store.store_credentials("tradier", self.test_credentials))
        
        # List brokers
        self.assertEqual(store.list_brokers(), ["tradier"])
        
        # Get credentials
        creds = store.get_credentials("tradier")
        self.assertEqual(creds["api_key"], "test_api_key")
        
        # Verify file was created
        self.assertTrue(os.path.exists(self.encrypted_file))
        
        # Create a new store instance to test loading from file
        store2 = EncryptedFileStore(self.encrypted_file, self.test_password)
        creds2 = store2.get_credentials("tradier")
        self.assertEqual(creds2["api_key"], "test_api_key")
        
        # Test env var password usage
        with patch.dict(os.environ, {'TRADING_BOT_MASTER_PASSWORD': 'env_password'}):
            store3 = EncryptedFileStore(os.path.join(self.test_dir.name, "env_store.enc"))
            store3.store_credentials("env_broker", self.test_credentials)
            self.assertEqual(store3.get_credentials("env_broker")["api_key"], "test_api_key")

    def test_credential_factory(self):
        """Test CredentialFactory methods"""
        # Test API key credentials
        api_creds = CredentialFactory.create_api_key_credentials(
            "api_key_123", 
            "api_secret_456",
            {"account_id": "demo123"}
        )
        self.assertEqual(api_creds["api_key"], "api_key_123")
        self.assertEqual(api_creds["api_secret"], "api_secret_456")
        self.assertEqual(api_creds["account_id"], "demo123")
        self.assertEqual(api_creds["auth_method"], "api_key")
        
        # Test OAuth credentials
        oauth_creds = CredentialFactory.create_oauth_credentials(
            "client_123",
            "secret_456",
            "access_token_789",
            "refresh_token_abc"
        )
        self.assertEqual(oauth_creds["client_id"], "client_123")
        self.assertEqual(oauth_creds["client_secret"], "secret_456")
        self.assertEqual(oauth_creds["access_token"], "access_token_789")
        self.assertEqual(oauth_creds["refresh_token"], "refresh_token_abc")
        self.assertEqual(oauth_creds["auth_method"], "oauth")
        
        # Test username/password credentials
        up_creds = CredentialFactory.create_username_password_credentials(
            "testuser",
            "testpass",
            {"domain": "test.com"}
        )
        self.assertEqual(up_creds["username"], "testuser")
        self.assertEqual(up_creds["password"], "testpass")
        self.assertEqual(up_creds["domain"], "test.com")
        self.assertEqual(up_creds["auth_method"], "username_password")

    def test_authenticator_factory(self):
        """Test AuthenticatorFactory and authenticators"""
        # Create test store
        store = YamlFileStore(self.yaml_file)
        store.store_credentials("api_broker", 
                              CredentialFactory.create_api_key_credentials("key1", "secret1"), 
                              AuthMethod.API_KEY)
        
        store.store_credentials("oauth_broker",
                              CredentialFactory.create_oauth_credentials(
                                  "client1", "secret1", "token1", "refresh1"),
                              AuthMethod.OAUTH)
        
        # Test creating API key authenticator
        api_auth = AuthenticatorFactory.create_authenticator(
            AuthMethod.API_KEY, store, "api_broker")
        self.assertIsInstance(api_auth, ApiKeyAuthenticator)
        
        # Test API key authentication
        auth_result = api_auth.authenticate()
        self.assertTrue(auth_result["authenticated"])
        
        # Test headers
        headers = api_auth.get_auth_headers()
        self.assertEqual(headers["X-API-Key"], "key1")
        self.assertEqual(headers["X-API-Secret"], "secret1")
        
        # Test OAuth authenticator
        oauth_auth = AuthenticatorFactory.create_authenticator(
            AuthMethod.OAUTH, store, "oauth_broker")
        self.assertIsInstance(oauth_auth, OAuthAuthenticator)


if __name__ == "__main__":
    unittest.main()
