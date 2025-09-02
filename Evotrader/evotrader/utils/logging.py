"""Comprehensive logging utilities for EvoTrader."""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Configure the logging system for EvoTrader.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file, None for no file logging
        log_format: Custom log format string
        console: Whether to log to console
        
    Returns:
        Logger: Configured root logger
    """
    # Create logs directory if needed
    if log_file and os.path.dirname(log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger("evotrader")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Use default format if none provided
    if not log_format:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    formatter = logging.Formatter(log_format)
    
    # Add file handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add timestamp to first log
    logger.info(f"Logging initialized at {datetime.now().isoformat()}")
    return logger


def get_bot_logger(bot_id: str) -> logging.Logger:
    """
    Get a logger for a specific trading bot.
    
    Args:
        bot_id: Unique identifier for the bot
        
    Returns:
        Logger: Logger configured with bot context
    """
    return logging.getLogger(f"evotrader.bot.{bot_id}")
