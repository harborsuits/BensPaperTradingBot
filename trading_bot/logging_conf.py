"""
Centralized logging configuration for the Trading Bot system.

This module provides standardized logging setup with:
- Consistent formatting across all components
- Log rotation to prevent excessive disk usage
- JSON structured logging option for easier parsing
- Console and file outputs with configurable levels
- Clear correlation IDs for request tracing

Usage:
    from trading_bot.logging_conf import setup_logging, get_logger
    
    # Setup logging once at application startup
    setup_logging()
    
    # Then get logger instances throughout the codebase
    logger = get_logger(__name__)
    logger.info("This is a log message")
"""

import os
import json
import uuid
import logging
import logging.handlers
from datetime import datetime
from typing import Dict, Any, Optional, Union

# Import settings from our new typed config
from trading_bot.config.typed_settings import load_config, LoggingSettings

# Constants
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
JSON_FORMAT = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": %(message)s}'

# Global request context for correlation IDs
request_context = {}

class RequestContextFilter(logging.Filter):
    """Filter that adds request context information to log records."""
    
    def filter(self, record):
        # Add request_id if available in global context
        record.request_id = request_context.get('request_id', '')
        
        # Add transaction_id if available
        record.transaction_id = request_context.get('transaction_id', '')
        
        # Always process the record
        return True

class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs logs as JSON."""
    
    def format(self, record):
        # Create a dict with basic log record attributes
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage()
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add request context if available
        if hasattr(record, 'request_id') and record.request_id:
            log_data['request_id'] = record.request_id
            
        if hasattr(record, 'transaction_id') and record.transaction_id:
            log_data['transaction_id'] = record.transaction_id
        
        # Add any extra attributes added by the application
        for key, value in record.__dict__.items():
            if key not in ('args', 'asctime', 'created', 'exc_info', 'exc_text', 
                          'filename', 'funcName', 'id', 'levelname', 'levelno',
                          'lineno', 'module', 'msecs', 'message', 'msg', 'name', 
                          'pathname', 'process', 'processName', 'relativeCreated',
                          'stack_info', 'thread', 'threadName', 'request_id',
                          'transaction_id'):
                log_data[key] = value
        
        # Return JSON string
        return json.dumps(log_data)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name, typically __name__
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

def setup_logging(
    settings: Optional[LoggingSettings] = None,
    log_file: Optional[str] = None,
    log_level: Optional[str] = None,
    use_json: bool = False,
    console: Optional[bool] = None
) -> None:
    """
    Setup centralized logging configuration.
    
    Args:
        settings: LoggingSettings object, loaded from config if not provided
        log_file: Override path to log file
        log_level: Override log level
        use_json: Whether to use JSON formatting
        console: Whether to log to console
    """
    # Load settings if not provided
    if settings is None:
        try:
            config = load_config()
            settings = config.logging
        except Exception as e:
            # Fallback to defaults if config loading fails
            print(f"Warning: Failed to load logging config: {str(e)}")
            settings = LoggingSettings()
    
    # Allow overriding specific settings
    level = log_level or settings.level
    file_path = log_file or settings.file_path
    
    if console is None:
        console = settings.console_logging
    
    # Convert string level to numeric
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter based on settings
    if use_json:
        formatter = JSONFormatter(JSON_FORMAT)
    else:
        formatter = logging.Formatter(settings.log_format or DEFAULT_FORMAT)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler if enabled
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if file path is specified
    if file_path:
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Use rotating file handler with configurable size and backup count
        file_handler = logging.handlers.RotatingFileHandler(
            file_path, 
            maxBytes=settings.max_size_mb * 1024 * 1024, 
            backupCount=settings.backup_count
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Add request context filter to all handlers
    context_filter = RequestContextFilter()
    for handler in root_logger.handlers:
        handler.addFilter(context_filter)
    
    # Log startup message
    root_logger.info(f"Logging initialized: level={level}, json={use_json}")

def set_request_context(request_id: Optional[str] = None) -> str:
    """
    Set the current request context for correlation.
    
    Args:
        request_id: Request ID to use, or auto-generate if None
        
    Returns:
        The request ID being used
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    request_context['request_id'] = request_id
    return request_id

def clear_request_context() -> None:
    """Clear the current request context."""
    request_context.clear()

def start_transaction(transaction_name: str) -> str:
    """
    Start a new transaction for logging correlation.
    
    Args:
        transaction_name: Name of the transaction
        
    Returns:
        Transaction ID
    """
    transaction_id = str(uuid.uuid4())
    request_context['transaction_id'] = transaction_id
    request_context['transaction_name'] = transaction_name
    request_context['transaction_start'] = datetime.now().isoformat()
    
    logger = get_logger("transaction")
    logger.info(f"Transaction {transaction_name} started", extra={
        "transaction_id": transaction_id,
        "event": "transaction_start"
    })
    
    return transaction_id

def end_transaction(success: bool = True, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    End the current transaction and log completion.
    
    Args:
        success: Whether the transaction completed successfully
        metadata: Additional metadata to log with the transaction
    """
    transaction_id = request_context.get('transaction_id')
    transaction_name = request_context.get('transaction_name')
    
    if not transaction_id or not transaction_name:
        return
    
    # Calculate duration if we have start time
    duration_ms = None
    if 'transaction_start' in request_context:
        start_time = datetime.fromisoformat(request_context['transaction_start'])
        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
    
    logger = get_logger("transaction")
    logger.info(
        f"Transaction {transaction_name} {'completed' if success else 'failed'}",
        extra={
            "transaction_id": transaction_id,
            "event": "transaction_end",
            "success": success,
            "duration_ms": duration_ms,
            **(metadata or {})
        }
    )
    
    # Don't clear entire context as request might still be ongoing
    if 'transaction_id' in request_context:
        del request_context['transaction_id']
    if 'transaction_name' in request_context:
        del request_context['transaction_name']
    if 'transaction_start' in request_context:
        del request_context['transaction_start']


if __name__ == "__main__":
    # Example usage
    setup_logging(use_json=True)
    logger = get_logger("example")
    
    # Basic logging
    logger.info("This is a basic log message")
    logger.warning("This is a warning message")
    
    # With request context
    set_request_context("example-request-123")
    logger.info("This log is part of a request")
    
    # With transaction
    transaction_id = start_transaction("example_operation")
    logger.info("Processing within transaction")
    
    # With custom fields
    logger.info("Log with custom data", extra={
        "user_id": "user-456",
        "component": "auth",
        "duration_ms": 120
    })
    
    # End transaction
    end_transaction(success=True, metadata={"items_processed": 10})
    
    # Clear context when done
    clear_request_context()
