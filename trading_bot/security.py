"""
Security module for the trading bot API.

Provides functionality for:
1. HTTPS enforcement with Flask-Talisman
2. Rate limiting with Flask-Limiter
3. JWT authentication
4. Input validation
5. Secure logging
6. API key management
"""

import os
import logging
import secrets
import json
import re
import time
from functools import wraps
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Callable, Any

from flask import Flask, request, jsonify, g, current_app
from flask_talisman import Talisman
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_jwt_extended import (
    JWTManager, create_access_token, create_refresh_token,
    jwt_required, get_jwt_identity, get_jwt
)
from marshmallow import Schema, fields, ValidationError, validates, validate
import redis
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = logging.getLogger('security')

# ----------- API Key Management -----------

class APIKeyManager:
    """Manages API keys for external services and client authentication"""
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize the API Key Manager.
        
        Args:
            redis_url: Redis connection URL for storing API keys
        """
        self.redis_client = None
        if redis_url:
            self.redis_client = redis.from_url(redis_url)
        
        # If redis is not available, use in-memory storage (not for production)
        self.in_memory_keys = {}
        
        # Load default keys from environment variables
        self._load_env_keys()
    
    def _load_env_keys(self):
        """Load API keys from environment variables"""
        # Add common API keys from environment
        env_keys = {
            'alpaca': {
                'api_key': os.getenv('ALPACA_API_KEY'),
                'api_secret': os.getenv('ALPACA_API_SECRET'),
                'base_url': os.getenv('ALPACA_BASE_URL')
            },
            'tradier': {
                'api_key': os.getenv('TRADIER_API_KEY'),
                'refresh_token': os.getenv('TRADIER_REFRESH_TOKEN')
            }
        }
        
        # Store keys
        for service, keys in env_keys.items():
            if all(keys.values()):
                self.store_service_keys(service, keys)
    
    def generate_client_api_key(self) -> str:
        """Generate a secure random API key for client authentication"""
        return secrets.token_urlsafe(32)
    
    def store_client_key(self, client_id: str, key: str, permissions: List[str], 
                         expires_at: Optional[datetime] = None) -> bool:
        """
        Store a client API key.
        
        Args:
            client_id: Unique ID for the client
            key: The API key
            permissions: List of permission strings
            expires_at: Expiration date (optional)
        
        Returns:
            bool: Success status
        """
        key_data = {
            'key': key,
            'permissions': permissions,
            'created_at': datetime.utcnow().isoformat(),
            'expires_at': expires_at.isoformat() if expires_at else None
        }
        
        key_id = f"client:{client_id}"
        return self._store_key(key_id, key_data)
    
    def store_service_keys(self, service: str, keys: dict) -> bool:
        """
        Store API keys for an external service.
        
        Args:
            service: Service name (e.g., 'alpaca', 'tradier')
            keys: Dictionary of keys and relevant data
        
        Returns:
            bool: Success status
        """
        key_id = f"service:{service}"
        return self._store_key(key_id, keys)
    
    def _store_key(self, key_id: str, data: dict) -> bool:
        """
        Internal method to store a key either in Redis or in-memory.
        
        Args:
            key_id: Key identifier
            data: Key data to store
            
        Returns:
            bool: Success status
        """
        try:
            if self.redis_client:
                self.redis_client.set(key_id, json.dumps(data))
            else:
                self.in_memory_keys[key_id] = data
            logger.info(f"Stored key for {key_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store key for {key_id}: {str(e)}")
            return False
    
    def get_service_keys(self, service: str) -> Optional[dict]:
        """
        Retrieve API keys for a service.
        
        Args:
            service: Service name to retrieve keys for
            
        Returns:
            dict: Key data or None if not found
        """
        key_id = f"service:{service}"
        return self._get_key(key_id)
    
    def get_client_key(self, client_id: str) -> Optional[dict]:
        """
        Retrieve client API key data.
        
        Args:
            client_id: Client ID to retrieve
            
        Returns:
            dict: Key data or None if not found
        """
        key_id = f"client:{client_id}"
        return self._get_key(key_id)
    
    def _get_key(self, key_id: str) -> Optional[dict]:
        """
        Internal method to retrieve a key from Redis or in-memory.
        
        Args:
            key_id: Key identifier
            
        Returns:
            dict: Key data or None if not found
        """
        try:
            if self.redis_client:
                data = self.redis_client.get(key_id)
                return json.loads(data) if data else None
            else:
                return self.in_memory_keys.get(key_id)
        except Exception as e:
            logger.error(f"Failed to retrieve key {key_id}: {str(e)}")
            return None
    
    def validate_client_key(self, api_key: str) -> Optional[dict]:
        """
        Validate a client API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            dict: Client data if valid, None otherwise
        """
        # In a real implementation, we would hash the API keys
        # and not store them in plaintext
        
        # This is a simplified implementation
        if self.redis_client:
            # Would need a more efficient implementation for Redis
            # This is not efficient for production use
            pass
        
        # Check in-memory keys
        for key_id, data in self.in_memory_keys.items():
            if key_id.startswith("client:") and data.get('key') == api_key:
                # Check if expired
                if data.get('expires_at'):
                    expires = datetime.fromisoformat(data['expires_at'])
                    if datetime.utcnow() > expires:
                        logger.warning(f"API key expired: {key_id}")
                        return None
                return data
        
        return None
    
    def revoke_key(self, key_id: str) -> bool:
        """
        Revoke a key by ID.
        
        Args:
            key_id: Key identifier to revoke
            
        Returns:
            bool: Success status
        """
        try:
            if self.redis_client:
                self.redis_client.delete(key_id)
            else:
                if key_id in self.in_memory_keys:
                    del self.in_memory_keys[key_id]
            logger.info(f"Revoked key: {key_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to revoke key {key_id}: {str(e)}")
            return False


# ----------- Validation Schemas -----------

class LoginSchema(Schema):
    """Schema for validating login requests"""
    username = fields.String(required=True, validate=validate.Length(min=3, max=50))
    password = fields.String(required=True, validate=validate.Length(min=8))


class TradeSchema(Schema):
    """Schema for validating trade requests"""
    symbol = fields.String(required=True, validate=validate.Length(min=1, max=10))
    quantity = fields.Float(required=True, validate=validate.Range(min=0.0001))
    side = fields.String(required=True, validate=validate.OneOf(['buy', 'sell']))
    order_type = fields.String(required=True, validate=validate.OneOf(['market', 'limit', 'stop', 'stop_limit']))
    time_in_force = fields.String(required=True, validate=validate.OneOf(['day', 'gtc', 'ioc', 'fok']))
    
    # Optional fields
    limit_price = fields.Float(validate=validate.Range(min=0.0001), allow_none=True)
    stop_price = fields.Float(validate=validate.Range(min=0.0001), allow_none=True)
    client_order_id = fields.String(validate=validate.Length(max=36), allow_none=True)
    
    @validates('symbol')
    def validate_symbol(self, value):
        """Validate symbol format"""
        if not re.match(r'^[A-Z0-9.]{1,10}$', value):
            raise ValidationError("Symbol must be 1-10 uppercase letters, numbers, or dots")


# ----------- Secure Flask Configuration -----------

def configure_security(app: Flask, redis_url: Optional[str] = None):
    """
    Configure security for a Flask application.
    
    Args:
        app: Flask application
        redis_url: Redis URL for rate limiting and token storage
    """
    # Initialize API Key Manager
    app.config['API_KEY_MANAGER'] = APIKeyManager(redis_url)
    
    # HTTPS with Flask-Talisman
    csp = {
        'default-src': '\'self\'',
        'img-src': ['\'self\'', 'data:'],
        'script-src': ['\'self\''],
        'style-src': ['\'self\'', '\'unsafe-inline\''],
    }
    
    Talisman(
        app,
        force_https=True,
        strict_transport_security=True,
        strict_transport_security_preload=True,
        content_security_policy=csp,
        session_cookie_secure=True,
        session_cookie_http_only=True
    )
    
    # Rate Limiting
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"],
        storage_uri=redis_url
    )
    
    # Rate limiting for specific endpoints
    @app.before_request
    def limit_api_endpoints():
        if request.path.startswith('/api/v1/trade'):
            # Stricter limits for trade endpoints
            limiter.limit("30 per minute")(lambda: None)()
        elif request.path.startswith('/api/v1/auth'):
            # Limit auth attempts to prevent brute force
            limiter.limit("10 per minute")(lambda: None)()
    
    # JWT Authentication
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', secrets.token_hex(32))
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
    app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=30)
    jwt = JWTManager(app)
    
    # JWT Token Blocklist
    jwt_blocklist = set()
    
    @jwt.token_in_blocklist_loader
    def check_if_token_revoked(jwt_header, jwt_payload):
        jti = jwt_payload['jti']
        return jti in jwt_blocklist
    
    # API Key Authentication Decorator
    def require_api_key(required_permissions=None):
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                api_key = request.headers.get('X-API-Key', '')
                if not api_key:
                    return jsonify({'error': 'API key is required'}), 401
                
                key_manager = current_app.config['API_KEY_MANAGER']
                client_data = key_manager.validate_client_key(api_key)
                
                if not client_data:
                    return jsonify({'error': 'Invalid API key'}), 401
                
                if required_permissions:
                    user_permissions = client_data.get('permissions', [])
                    if not all(perm in user_permissions for perm in required_permissions):
                        return jsonify({'error': 'Insufficient permissions'}), 403
                
                g.client_data = client_data
                return f(*args, **kwargs)
            return decorated_function
        return decorator
    
    app.config['require_api_key'] = require_api_key
    
    # IP Address Blocking
    blocked_ips = set()
    
    @app.before_request
    def block_ips():
        ip = request.remote_addr
        if ip in blocked_ips:
            return jsonify({'error': 'Access denied'}), 403
    
    # Input Validation Helper
    def validate_request(schema_class):
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                try:
                    schema = schema_class()
                    
                    # Validate request data based on content type
                    if request.is_json:
                        data = request.get_json()
                    else:
                        data = request.form.to_dict()
                    
                    result = schema.load(data)
                    g.validated_data = result
                    return f(*args, **kwargs)
                except ValidationError as err:
                    return jsonify({'error': 'Validation error', 'details': err.messages}), 400
            return decorated_function
        return decorator
    
    app.config['validate_request'] = validate_request
    
    # Secure headers for all responses
    @app.after_request
    def add_security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'no-referrer'
        return response
    
    # Security logging
    @app.before_request
    def log_request_info():
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'method': request.method,
            'path': request.path,
            'ip': request.remote_addr,
            'user_agent': request.user_agent.string,
            'content_length': request.content_length
        }
        
        # Don't log sensitive routes in detail
        if not request.path.startswith('/api/v1/auth'):
            logger.info(f"Request: {json.dumps(log_data)}")
    
    # Make utility functions available
    app.config['jwt_blocklist'] = jwt_blocklist
    app.config['blocked_ips'] = blocked_ips
    
    return {
        'talisman': Talisman,
        'limiter': limiter,
        'jwt': jwt,
        'validate_request': validate_request,
        'require_api_key': require_api_key,
        'jwt_blocklist': jwt_blocklist,
        'blocked_ips': blocked_ips
    }


# ----------- Utility Functions -----------

def get_security_status(app: Flask) -> dict:
    """
    Get the security status of the application.
    
    Args:
        app: Flask application
        
    Returns:
        dict: Security configuration status
    """
    status = {
        'https_enabled': app.config.get('TALISMAN_ENABLED', False),
        'rate_limiting': bool(app.extensions.get('limiter')),
        'jwt_auth': bool(app.extensions.get('flask-jwt-extended')),
        'api_key_auth': bool(app.config.get('API_KEY_MANAGER')),
        'blocked_ips_count': len(app.config.get('blocked_ips', set())),
        'jwt_blocklist_count': len(app.config.get('jwt_blocklist', set()))
    }
    return status

def secure_file_paths(path: str) -> str:
    """
    Make file paths secure to prevent directory traversal.
    
    Args:
        path: File path to secure
        
    Returns:
        str: Secured path
    """
    # Normalize the path and remove any potentially dangerous components
    norm_path = os.path.normpath(path)
    if norm_path.startswith('..') or '/.' in norm_path:
        raise ValueError("Potentially insecure path detected")
    return norm_path 