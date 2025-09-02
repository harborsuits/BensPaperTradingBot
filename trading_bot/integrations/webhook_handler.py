"""
Webhook Handler - Robust implementation for handling incoming webhooks from TradingView and other sources.
"""

import os
import json
import logging
import threading
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List, Union, Tuple
from flask import Flask, request, jsonify, Response

from trading_bot.core.interfaces import WebhookInterface
from trading_bot.core.service_registry import ServiceRegistry

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Exception raised for webhook payload validation errors."""
    pass

class WebhookHandler(WebhookInterface):
    """
    Robust webhook handler for processing incoming alerts from TradingView and other sources.
    Includes validation, error handling, rate limiting, and logging.
    """
    
    def __init__(
        self,
        port: int = 5000,
        path: str = '/webhook',
        auth_token: Optional[str] = None,
        validation_schema: Optional[Dict[str, Any]] = None,
        rate_limit: int = 100,  # Max requests per minute
        register_service: bool = True
    ):
        """
        Initialize the webhook handler.
        
        Args:
            port: Port to run the webhook server on
            path: URL path for the webhook endpoint
            auth_token: Optional authentication token
            validation_schema: Schema for validating payloads
            rate_limit: Maximum number of requests per minute
            register_service: Whether to register with the ServiceRegistry
        """
        self.port = port
        self.path = path.lstrip('/')  # Remove leading slash if present
        self.auth_token = auth_token
        self.validation_schema = validation_schema or self._default_validation_schema()
        self.rate_limit = rate_limit
        
        # Set up Flask app
        self.app = Flask(__name__)
        
        # Add routes
        self._setup_routes()
        
        # Thread for running the server
        self.server_thread = None
        self.running = False
        
        # Rate limiting
        self.request_timestamps: List[float] = []
        
        # Request handlers
        self.handlers: List[Callable[[Dict[str, Any]], None]] = []
        
        # Register with service registry if requested
        if register_service:
            ServiceRegistry.register('webhook_handler', self, WebhookInterface)
    
    def _default_validation_schema(self) -> Dict[str, Any]:
        """
        Create a default validation schema for TradingView alerts.
        
        Returns:
            Dictionary with validation rules
        """
        return {
            'required_fields': ['symbol'],
            'optional_fields': [
                'timestamp', 'close', 'open', 'high', 'low', 
                'volume', 'indicators', 'asset_type'
            ],
            'field_types': {
                'symbol': str,
                'asset_type': str,
                'timestamp': str,
                'close': (int, float),
                'open': (int, float),
                'high': (int, float),
                'low': (int, float),
                'volume': (int, float),
                'indicators': dict
            }
        }
    
    def _setup_routes(self) -> None:
        """Set up Flask routes for the webhook handler."""
        @self.app.route(f'/{self.path}', methods=['POST'])
        def handle_webhook() -> Tuple[Response, int]:
            """Endpoint for handling webhook requests."""
            # Check rate limits
            if not self._check_rate_limit():
                logger.warning("Rate limit exceeded")
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Verify authentication if required
            if self.auth_token:
                auth_header = request.headers.get('Authorization', '')
                if not auth_header.startswith('Bearer ') or auth_header[7:] != self.auth_token:
                    logger.warning("Unauthorized webhook request")
                    return jsonify({'error': 'Unauthorized'}), 401
            
            # Get request data
            try:
                if not request.is_json:
                    logger.warning("Non-JSON payload received")
                    return jsonify({'error': 'JSON payload required'}), 400
                
                data = request.json
                
                # Process the webhook
                try:
                    result = self.process_webhook(data)
                    return jsonify(result), 200
                except ValidationError as e:
                    logger.warning(f"Validation error: {str(e)}")
                    return jsonify({'error': str(e)}), 400
                except Exception as e:
                    logger.error(f"Error processing webhook: {str(e)}")
                    logger.error(traceback.format_exc())
                    return jsonify({'error': 'Internal processing error'}), 500
                    
            except Exception as e:
                logger.error(f"Error handling webhook request: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({'error': 'Internal server error'}), 500
    
    def _check_rate_limit(self) -> bool:
        """
        Check if the request is within rate limits.
        
        Returns:
            True if within limits, False otherwise
        """
        now = datetime.now().timestamp()
        
        # Remove timestamps older than 1 minute
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]
        
        # Check if we're over the limit
        if len(self.request_timestamps) >= self.rate_limit:
            return False
        
        # Add current timestamp
        self.request_timestamps.append(now)
        return True
    
    def _validate_payload(self, data: Dict[str, Any]) -> None:
        """
        Validate webhook payload against schema.
        
        Args:
            data: Webhook payload to validate
            
        Raises:
            ValidationError: If validation fails
        """
        schema = self.validation_schema
        
        # Check required fields
        for field in schema.get('required_fields', []):
            if field not in data:
                raise ValidationError(f"Missing required field: {field}")
        
        # Check field types
        for field, expected_type in schema.get('field_types', {}).items():
            if field in data and not isinstance(data[field], expected_type):
                # Handle tuple of types (any one is valid)
                if isinstance(expected_type, tuple) and any(isinstance(data[field], t) for t in expected_type):
                    continue
                
                raise ValidationError(
                    f"Field '{field}' has incorrect type. Expected {expected_type.__name__}, "
                    f"got {type(data[field]).__name__}"
                )
    
    def process_webhook(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a webhook payload.
        
        Args:
            data: Webhook payload data
            
        Returns:
            Response data
            
        Raises:
            ValidationError: If validation fails
        """
        # Log the received data
        logger.info(f"Received webhook for {data.get('symbol', 'unknown')}")
        logger.debug(f"Webhook data: {json.dumps(data)}")
        
        # Validate the payload
        self._validate_payload(data)
        
        # Call all registered handlers
        for handler in self.handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Error in webhook handler: {str(e)}")
                logger.error(traceback.format_exc())
        
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'message': f"Processed alert for {data.get('symbol', 'unknown')}"
        }
    
    def register_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a handler function to be called when a webhook is received.
        
        Args:
            handler: Function to call with the webhook payload
        """
        if handler not in self.handlers:
            self.handlers.append(handler)
            logger.debug(f"Registered webhook handler: {handler.__name__}")
    
    def unregister_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Unregister a previously registered handler function.
        
        Args:
            handler: Function to unregister
        """
        if handler in self.handlers:
            self.handlers.remove(handler)
            logger.debug(f"Unregistered webhook handler: {handler.__name__}")
    
    def start(self) -> None:
        """Start the webhook server in a separate thread."""
        if self.running:
            logger.warning("Webhook server is already running")
            return
        
        def run_server():
            """Run the Flask server."""
            try:
                self.app.run(host='0.0.0.0', port=self.port, debug=False)
            except Exception as e:
                logger.error(f"Error starting webhook server: {str(e)}")
                self.running = False
        
        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        self.running = True
        
        logger.info(f"Webhook server started on port {self.port}, path: /{self.path}")
    
    def stop(self) -> None:
        """Stop the webhook server."""
        if not self.running:
            logger.warning("Webhook server is not running")
            return
        
        # There's no clean way to stop a Flask server in a thread
        # Typically this would be handled by a more robust server setup
        # For now, just mark as not running
        self.running = False
        logger.info("Webhook server stop requested (will terminate when thread completes)") 