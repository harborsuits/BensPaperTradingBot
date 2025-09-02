import os
import json
import time
import logging
import uuid
import re
import ipaddress
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
from functools import wraps

import jwt
from flask import Flask, request, jsonify, g, make_response, current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from marshmallow import Schema, fields, ValidationError, validates_schema, validate

from trading_bot.security_utils import (
    hash_password, verify_password, format_password_for_storage, 
    parse_stored_password, validate_password_strength, sanitize_html,
    generate_secure_token, get_secure_env_var, redact_sensitive_data
)

logger = logging.getLogger("SecurityManager")

class EntrySignalSchema(Schema):
    """Schema for validating entry signal requests."""
    symbol = fields.String(required=True)
    entry_price = fields.Float(required=True)
    stop_loss = fields.Float(required=True)
    take_profit = fields.Float(required=True)
    strategy = fields.String(required=True)
    confidence = fields.Float(required=False)
    
    @validates("symbol")
    def validate_symbol(self, value):
        """Validate symbol format."""
        if not value or len(value) > 20:
            raise ValidationError("Symbol must be between 1 and 20 characters")
        
        # Remove any HTML tags
        return sanitize_html(value)
    
    @validates_schema
    def validate_prices(self, data, **kwargs):
        """Validate that entry, stop loss, and take profit prices make sense."""
        if "entry_price" in data and "stop_loss" in data and "take_profit" in data:
            entry = data["entry_price"]
            stop = data["stop_loss"]
            target = data["take_profit"]
            
            # Basic sanity checks
            if entry <= 0 or stop <= 0 or target <= 0:
                raise ValidationError("Prices must be positive")
                
            # For long trades
            if stop < entry < target:
                return
                
            # For short trades
            if stop > entry > target:
                return
                
            raise ValidationError("Price relationship doesn't make sense for either long or short trade")

class ExitSignalSchema(Schema):
    """Schema for validating exit signal requests."""
    symbol = fields.String(required=True)
    exit_price = fields.Float(required=True)
    trade_id = fields.String(required=True)
    
    @validates("trade_id")
    def validate_trade_id(self, value):
        """Validate trade ID format."""
        if not value or len(value) > 64:
            raise ValidationError("Invalid trade ID format")
        
        # Simple UUID validation
        try:
            uuid.UUID(value)
        except ValueError:
            raise ValidationError("Invalid trade ID format")

class PsychologicalAssessmentSchema(Schema):
    """Schema for validating psychological assessment requests."""
    trade_id = fields.String(required=True)
    emotion_rating = fields.Integer(required=True)
    discipline_rating = fields.Integer(required=True)
    confidence_rating = fields.Integer(required=True)
    notes = fields.String(required=False)
    
    @validates("emotion_rating", "discipline_rating", "confidence_rating")
    def validate_rating(self, value):
        """Validate rating values."""
        if not 1 <= value <= 10:
            raise ValidationError("Ratings must be between 1 and 10")
    
    @validates("notes")
    def validate_notes(self, value):
        """Sanitize notes to prevent XSS."""
        if value:
            return sanitize_html(value)

class SecurityManager:
    """Security manager for trading bot API.
    
    This class provides comprehensive security features including:
    - HTTPS enforcement using Flask-Talisman
    - Rate limiting with Flask-Limiter
    - JWT-based and API key authentication
    - Input validation using Marshmallow schemas
    - Request logging and IP whitelisting
    - Credential management
    """
    
    def __init__(
        self, 
        app: Flask, 
        config_path: str = None, 
        jwt_secret: str = None,
        api_tokens_file: str = None
    ):
        """Initialize security manager with Flask app and configuration.
        
        Args:
            app: Flask application to secure
            config_path: Path to security configuration file (JSON/YAML)
            jwt_secret: Secret key for JWT tokens (overrides config)
            api_tokens_file: Path to API tokens file (JSON)
        """
        self.app = app
        self.config_path = config_path
        self.api_tokens_file = api_tokens_file
        
        # Load configuration
        self.config = self._load_config()
        
        # Override config with passed parameters
        if jwt_secret:
            self.config['jwt_secret'] = jwt_secret
        
        # Set up security components
        self._setup_https()
        self._setup_rate_limiting()
        self._setup_error_handlers()
        
        # Load API tokens
        self.api_tokens = self.load_api_tokens()
        
        # Register security-related API endpoints
        self._register_endpoints()
        
        # Set up security-related logging
        self._setup_logging()
        
        # Security metrics
        self.security_metrics = {
            "failed_auth_attempts": 0,
            "rate_limit_hits": 0,
            "validation_failures": 0,
            "suspicious_requests": 0,
            "last_reset": datetime.now().isoformat()
        }
        
        # IP blacklist/whitelist
        self.ip_whitelist = self.config.get("ip_whitelist", [])
        self.ip_blacklist = self.config.get("ip_blacklist", [])
        
        logger.info("Security manager initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load security configuration from file or use defaults.
        
        Returns:
            Dictionary with security configuration
        """
        default_config = {
            "enforce_https": True,
            "jwt_secret": os.urandom(32).hex(),
            "jwt_expiry_minutes": 60,
            "rate_limit_default": "60 per minute",
            "rate_limit_login": "10 per minute",
            "rate_limit_by_user": True,
            "request_logging": True,
            "log_level": "INFO",
            "ip_whitelist": [],
            "ip_blacklist": [],
            "content_security_policy": {
                "default-src": "'self'",
                "script-src": "'self'",
                "style-src": "'self' 'unsafe-inline'",
                "img-src": "'self' data:",
                "font-src": "'self' data:",
                "connect-src": "'self'"
            }
        }
        
        if not self.config_path:
            logger.warning("No security config path provided, using defaults")
            return default_config
            
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.json'):
                    config = json.load(f)
                elif self.config_path.endswith(('.yaml', '.yml')):
                    import yaml
                    config = yaml.safe_load(f)
                else:
                    logger.error(f"Unsupported config file format: {self.config_path}")
                    return default_config
                    
            # Merge with defaults to ensure all required settings exist
            merged_config = {**default_config, **config}
            logger.info(f"Loaded security configuration from {self.config_path}")
            return merged_config
            
        except Exception as e:
            logger.error(f"Failed to load security config: {e}")
            return default_config
    
    def load_api_tokens(self) -> Dict[str, Dict[str, Any]]:
        """Load API tokens from environment variable or file.
        
        Returns:
            Dictionary mapping API tokens to user data
        """
        # First try environment variable
        tokens = {}
        
        env_tokens = get_secure_env_var("TRADING_BOT_API_TOKENS")
        if env_tokens:
            try:
                tokens = json.loads(env_tokens)
                logger.info("Loaded API tokens from environment variable")
                return tokens
            except json.JSONDecodeError:
                logger.error("Failed to parse API tokens from environment variable")
        
        # Then try file
        if self.api_tokens_file:
            try:
                with open(self.api_tokens_file, 'r') as f:
                    tokens = json.load(f)
                logger.info(f"Loaded API tokens from {self.api_tokens_file}")
                return tokens
            except Exception as e:
                logger.error(f"Failed to load API tokens file: {e}")
        
        # If no tokens are found, create a default token if in development
        if os.environ.get('FLASK_ENV') == 'development':
            default_token = generate_secure_token()
            tokens[default_token] = {
                "user_id": "default_user",
                "roles": ["admin"],
                "created_at": datetime.now().isoformat(),
                "description": "Default development token"
            }
            logger.warning(f"Created default development API token: {default_token}")
            
            # Save to file if specified
            if self.api_tokens_file:
                try:
                    with open(self.api_tokens_file, 'w') as f:
                        json.dump(tokens, f, indent=2)
                except Exception as e:
                    logger.error(f"Failed to save default token: {e}")
                    
        return tokens
    
    def _setup_https(self):
        """Set up HTTPS enforcement with Talisman."""
        if self.config.get("enforce_https", True):
            csp = self.config.get("content_security_policy", {})
            
            # Convert dict to Talisman format
            csp_dict = {}
            for key, value in csp.items():
                csp_dict[key] = value
            
            Talisman(
                self.app,
                force_https=True,
                strict_transport_security=True,
                session_cookie_secure=True,
                session_cookie_http_only=True,
                feature_policy=self.config.get("feature_policy", {}),
                content_security_policy=csp_dict
            )
            logger.info("HTTPS enforcement enabled with Talisman")
    
    def _setup_rate_limiting(self):
        """Set up rate limiting with Flask-Limiter."""
        # Use custom key function if rate limiting by user is enabled
        if self.config.get("rate_limit_by_user", True):
            key_func = self._get_rate_limit_key
        else:
            key_func = get_remote_address
        
        self.limiter = Limiter(
            self.app,
            key_func=key_func,
            default_limits=[self.config.get("rate_limit_default", "60 per minute")]
        )
        
        # Register event handler for rate limit exceeded
        @self.limiter.request_filter
        def check_whitelist():
            client_ip = request.remote_addr
            
            # Skip rate limiting for whitelisted IPs
            for ip_net in self.ip_whitelist:
                try:
                    if ipaddress.ip_address(client_ip) in ipaddress.ip_network(ip_net):
                        return True
                except ValueError:
                    continue
            
            return False
            
        @self.limiter.on_breach
        def on_breach(limit):
            self.security_metrics["rate_limit_hits"] += 1
            self.log_security_event("RATE_LIMIT_BREACH", {
                "limit": str(limit),
                "path": request.path,
                "method": request.method,
                "ip": request.remote_addr
            })
            
        logger.info("Rate limiting enabled")
    
    def _setup_logging(self):
        """Set up security-related logging."""
        if not self.config.get("request_logging", True):
            return
            
        # Configure security log level
        log_level = getattr(logging, self.config.get("log_level", "INFO"))
        logger.setLevel(log_level)
        
        # Add request logging
        @self.app.before_request
        def log_request():
            # Skip logging for certain endpoints
            skip_paths = ['/health', '/metrics', '/static/']
            if any(request.path.startswith(path) for path in skip_paths):
                return
                
            # Check IP blacklist
            client_ip = request.remote_addr
            for ip_net in self.ip_blacklist:
                try:
                    if ipaddress.ip_address(client_ip) in ipaddress.ip_network(ip_net):
                        self.log_security_event("BLOCKED_IP", {
                            "ip": client_ip, 
                            "path": request.path
                        })
                        return jsonify({"error": "Access denied"}), 403
                except ValueError:
                    continue
            
            # Log basic request info
            g.request_start_time = time.time()
            
            # Don't log full data for sensitive endpoints
            sensitive_paths = ['/login', '/auth', '/api-token']
            if any(request.path.startswith(path) for path in sensitive_paths):
                logger.info(f"Request: {request.method} {request.path} from {client_ip}")
            else:
                req_data = {}
                if request.is_json:
                    req_data = request.get_json(silent=True) or {}
                elif request.form:
                    req_data = request.form.to_dict()
                    
                # Redact sensitive data
                if req_data:
                    try:
                        redacted = redact_sensitive_data(req_data)
                        logger.info(f"Request: {request.method} {request.path} from {client_ip}, data: {redacted}")
                    except:
                        logger.info(f"Request: {request.method} {request.path} from {client_ip}")
                else:
                    logger.info(f"Request: {request.method} {request.path} from {client_ip}")
                
        @self.app.after_request
        def log_response(response):
            # Skip logging for certain endpoints
            skip_paths = ['/health', '/metrics', '/static/']
            if any(request.path.startswith(path) for path in skip_paths):
                return response
                
            # Calculate request duration
            if hasattr(g, 'request_start_time'):
                duration = time.time() - g.request_start_time
                logger.info(f"Response: {request.method} {request.path} - {response.status_code} in {duration:.3f}s")
            
            return response
            
        logger.info("Request logging enabled")
            
    def _setup_error_handlers(self):
        """Set up custom error handlers for security-related errors."""
        @self.app.errorhandler(401)
        def unauthorized(error):
            self.log_security_event("UNAUTHORIZED", {
                "path": request.path,
                "method": request.method,
                "ip": request.remote_addr
            })
            return jsonify({"error": "Unauthorized access"}), 401
            
        @self.app.errorhandler(403)
        def forbidden(error):
            self.log_security_event("FORBIDDEN", {
                "path": request.path,
                "method": request.method,
                "ip": request.remote_addr
            })
            return jsonify({"error": "Access forbidden"}), 403
            
        @self.app.errorhandler(ValidationError)
        def validation_error(error):
            self.security_metrics["validation_failures"] += 1
            self.log_security_event("VALIDATION_ERROR", {
                "path": request.path,
                "method": request.method,
                "ip": request.remote_addr,
                "error": str(error)
            })
            return jsonify({"error": "Validation error", "details": error.messages}), 400
    
    def generate_jwt_token(self, username: str, expiry_minutes: int = None) -> str:
        """Generate a JWT token for a user.
        
        Args:
            username: Username to include in token
            expiry_minutes: Token expiration time in minutes (overrides config)
            
        Returns:
            JWT token string
        """
        expiry = expiry_minutes or self.config.get("jwt_expiry_minutes", 60)
        
        payload = {
            'username': username,
            'exp': datetime.utcnow() + timedelta(minutes=expiry),
            'iat': datetime.utcnow(),
            'jti': str(uuid.uuid4())
        }
        
        return jwt.encode(
            payload,
            self.config['jwt_secret'],
            algorithm='HS256'
        )
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify a JWT token and return its claims.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Dictionary with token claims
            
        Raises:
            jwt.InvalidTokenError: If token is invalid or expired
        """
        try:
            return jwt.decode(
                token,
                self.config['jwt_secret'],
                algorithms=['HS256']
            )
        except jwt.ExpiredSignatureError:
            raise jwt.InvalidTokenError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise jwt.InvalidTokenError(f"Invalid token: {e}")
    
    def require_auth(self, f: Callable) -> Callable:
        """Decorator to require authentication for Flask routes.
        
        This decorator checks for valid API token or JWT token in:
        1. Authorization header (Bearer token)
        2. X-API-Token header
        3. api_token query parameter
        4. JWT token in cookie
        
        Args:
            f: Flask route function to decorate
            
        Returns:
            Decorated function that checks for authentication
        """
        @wraps(f)
        def decorated(*args, **kwargs):
            token = None
            auth_type = None
            
            # Check Authorization header
            auth_header = request.headers.get('Authorization')
            if auth_header:
                parts = auth_header.split()
                if len(parts) == 2 and parts[0].lower() == 'bearer':
                    token = parts[1]
                    auth_type = 'bearer'
            
            # Check X-API-Token header
            if not token:
                token = request.headers.get('X-API-Token')
                if token:
                    auth_type = 'api-token'
            
            # Check query parameter
            if not token:
                token = request.args.get('api_token')
                if token:
                    auth_type = 'api-token'
            
            # Check cookie for JWT
            if not token:
                token = request.cookies.get('jwt_token')
                if token:
                    auth_type = 'jwt'
            
            # No token found
            if not token:
                self.security_metrics["failed_auth_attempts"] += 1
                self.log_security_event("AUTH_MISSING", {
                    "path": request.path,
                    "method": request.method,
                    "ip": request.remote_addr
                })
                return jsonify({"error": "Authentication required"}), 401
            
            # Verify token based on type
            try:
                if auth_type == 'api-token':
                    # Check against API tokens
                    if token in self.api_tokens:
                        # Set user info for the request
                        g.user = self.api_tokens[token]
                        g.auth_type = 'api-token'
                    else:
                        self.security_metrics["failed_auth_attempts"] += 1
                        self.log_security_event("AUTH_INVALID_TOKEN", {
                            "path": request.path,
                            "method": request.method,
                            "ip": request.remote_addr,
                            "auth_type": auth_type
                        })
                        return jsonify({"error": "Invalid API token"}), 401
                else:
                    # Try as JWT token
                    try:
                        payload = self.verify_jwt_token(token)
                        g.user = {"user_id": payload["username"]}
                        g.auth_type = 'jwt'
                    except jwt.InvalidTokenError as e:
                        self.security_metrics["failed_auth_attempts"] += 1
                        self.log_security_event("AUTH_INVALID_JWT", {
                            "path": request.path,
                            "method": request.method,
                            "ip": request.remote_addr,
                            "error": str(e)
                        })
                        return jsonify({"error": f"Invalid JWT token: {e}"}), 401
                
                # Log successful authentication
                self.log_security_event("AUTH_SUCCESS", {
                    "path": request.path,
                    "method": request.method,
                    "ip": request.remote_addr,
                    "auth_type": auth_type,
                    "user_id": g.user.get("user_id", "unknown")
                })
                
                return f(*args, **kwargs)
            
            except Exception as e:
                self.security_metrics["failed_auth_attempts"] += 1
                self.log_security_event("AUTH_ERROR", {
                    "path": request.path,
                    "method": request.method,
                    "ip": request.remote_addr,
                    "error": str(e)
                })
                return jsonify({"error": f"Authentication error: {str(e)}"}), 401
        
        return decorated
    
    def validate_json(self, schema_class):
        """Decorator to validate JSON request body against a Marshmallow schema.
        
        Args:
            schema_class: Marshmallow Schema class to use for validation
            
        Returns:
            Decorated function that validates request JSON
        """
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # Check Content-Type
                if request.method == 'POST' and not request.is_json:
                    return jsonify({"error": "Expected Content-Type: application/json"}), 415
                
                schema = schema_class()
                json_data = request.get_json(silent=True) or {}
                
                try:
                    # Validate and deserialize input
                    validated_data = schema.load(json_data)
                    
                    # Add validated data to flask.g for handler to use
                    g.validated_data = validated_data
                    
                    return f(*args, **kwargs)
                except ValidationError as err:
                    self.security_metrics["validation_failures"] += 1
                    self.log_security_event("VALIDATION_FAILURE", {
                        "path": request.path,
                        "method": request.method,
                        "ip": request.remote_addr,
                        "errors": err.messages
                    })
                    return jsonify({"error": "Validation error", "details": err.messages}), 400
            
            return decorated_function
        return decorator
    
    def rate_limit(self, limit_value: str):
        """Decorator to apply specific rate limit to a route.
        
        Args:
            limit_value: Rate limit string (e.g., "10 per minute")
            
        Returns:
            Decorated function with rate limit applied
        """
        return self.limiter.limit(limit_value)
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize user input to prevent injection attacks.
        
        Args:
            text: User input to sanitize
            
        Returns:
            Sanitized text
        """
        return sanitize_html(text)
    
    def log_security_event(self, event_type: str, details: Dict[str, Any] = None):
        """Log a security-related event.
        
        Args:
            event_type: Type of security event
            details: Additional details about the event
        """
        event_data = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details or {}
        }
        
        # Add IP and user info if available
        if hasattr(request, 'remote_addr'):
            event_data["ip"] = request.remote_addr
            
        if hasattr(g, 'user'):
            event_data["user"] = g.user.get("user_id", "unknown")
            
        # Log to application logger
        if event_type.startswith(("AUTH_", "BLOCKED_")):
            logger.warning(f"Security event: {event_type} - {event_data}")
        else:
            logger.info(f"Security event: {event_type} - {event_data}")
            
        # For serious events, consider additional alerting here
        high_severity_events = ["BLOCKED_IP", "AUTH_BRUTE_FORCE", "SUSPICIOUS_ACTIVITY"]
        if event_type in high_severity_events:
            # TODO: Implement alerting mechanism (email, Slack, etc.)
            pass
    
    def _get_rate_limit_key(self) -> str:
        """Get key for rate limiting.
        
        If user is authenticated, use user ID, otherwise use IP.
        
        Returns:
            String key for rate limiting
        """
        if hasattr(g, 'user'):
            return g.user.get("user_id", request.remote_addr)
        return request.remote_addr
    
    def _register_endpoints(self):
        """Register security-related endpoints."""
        # Login endpoint
        @self.app.route('/login', methods=['POST'])
        @self.rate_limit(self.config.get("rate_limit_login", "10 per minute"))
        def login():
            data = request.get_json(silent=True) or {}
            username = data.get('username')
            password = data.get('password')
            
            if not username or not password:
                return jsonify({"error": "Username and password required"}), 400
                
            # TODO: Implement actual user authentication against database
            # This is a placeholder - in a real app, check against database
            if username == "admin" and password == "password":  # Replace with real auth
                token = self.generate_jwt_token(username)
                
                self.log_security_event("LOGIN_SUCCESS", {
                    "username": username
                })
                
                resp = jsonify({"token": token})
                resp.set_cookie(
                    'jwt_token', 
                    token, 
                    httponly=True, 
                    secure=self.config.get("enforce_https", True),
                    samesite='Strict',
                    max_age=self.config.get("jwt_expiry_minutes", 60) * 60
                )
                return resp
            
            self.security_metrics["failed_auth_attempts"] += 1
            self.log_security_event("LOGIN_FAILURE", {
                "username": username
            })
            
            return jsonify({"error": "Invalid credentials"}), 401
            
        # Verify authentication endpoint
        @self.app.route('/verify-auth', methods=['GET'])
        @self.require_auth
        def verify_auth():
            return jsonify({
                "authenticated": True,
                "user": g.user.get("user_id", "unknown"),
                "auth_type": g.auth_type
            })
            
        # Security status endpoint (admin only)
        @self.app.route('/security-status', methods=['GET'])
        @self.require_auth
        def security_status():
            # Check if user has admin role
            user = g.user
            if not user.get("roles") or "admin" not in user.get("roles", []):
                return jsonify({"error": "Admin access required"}), 403
                
            return jsonify({
                "security_metrics": self.security_metrics,
                "config": {
                    k: v for k, v in self.config.items() 
                    if k not in ["jwt_secret", "api_tokens"]
                },
                "ip_whitelist": self.ip_whitelist,
                "ip_blacklist": self.ip_blacklist
            })
        
        logger.info("Security endpoints registered")

# Example usage:
"""
from flask import Flask
from trading_bot.security_manager import SecurityManager, EntrySignalSchema

app = Flask(__name__)
security = SecurityManager(app, config_path="config/security.json")

@app.route('/api/entry-signal', methods=['POST'])
@security.require_auth
@security.validate_json(EntrySignalSchema)
def entry_signal():
    # g.validated_data contains validated JSON data
    # Process entry signal
    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(debug=True)
""" 