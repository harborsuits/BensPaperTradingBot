#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Bot CLI Application

This module provides a unified command-line interface for the trading bot,
consolidating the various entry points into a single, consistent interface.
"""

import argparse
import logging
import sys
import os
from typing import List, Dict, Any, Optional, Callable
import importlib

from trading_bot.config.unified_config import get_config, load_config
from trading_bot.core.event_system import EventBus
from trading_bot.logging_conf import configure_logging

logger = logging.getLogger(__name__)

# Command registry to store all registered commands
COMMAND_REGISTRY = {}

def command(name: str, help_text: str):
    """
    Decorator to register a command.
    
    Args:
        name: Command name
        help_text: Help text for the command
        
    Returns:
        Decorator function
    """
    def decorator(func):
        COMMAND_REGISTRY[name] = {
            'function': func,
            'help': help_text,
            'arguments': getattr(func, 'arguments', [])
        }
        return func
    return decorator

def argument(*args, **kwargs):
    """
    Decorator to add arguments to a command.
    
    Args:
        *args: Positional arguments for argparse.add_argument
        **kwargs: Keyword arguments for argparse.add_argument
        
    Returns:
        Decorator function
    """
    def decorator(func):
        if not hasattr(func, 'arguments'):
            func.arguments = []
        func.arguments.append((args, kwargs))
        return func
    return decorator

def create_cli_app():
    """
    Create the CLI application.
    
    Returns:
        Configured argparse parser
    """
    # Create main parser
    parser = argparse.ArgumentParser(
        description="Trading Bot Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add global arguments
    parser.add_argument(
        '--config', 
        help='Path to configuration file',
        default=None
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level'
    )
    
    parser.add_argument(
        '--env',
        choices=['development', 'staging', 'production'],
        default='development',
        help='Set the environment'
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        dest='command',
        title='Commands',
        description='Available commands',
        help='Command to execute'
    )
    subparsers.required = True
    
    # Register all commands from the registry
    for name, command_info in COMMAND_REGISTRY.items():
        cmd_parser = subparsers.add_parser(name, help=command_info['help'])
        
        # Add command-specific arguments
        for args, kwargs in command_info['arguments']:
            cmd_parser.add_argument(*args, **kwargs)
    
    return parser

def setup_environment(args):
    """
    Set up the environment based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments
    """
    # Configure logging
    configure_logging(level=args.log_level)
    
    # Set environment variable
    os.environ['TRADING_ENV'] = args.env
    
    # Load configuration
    if args.config:
        os.environ['TRADING_CONFIG'] = args.config
    
    load_config()
    
    logger.info(f"Environment set up: {args.env}, log level: {args.log_level}")

def execute_command(args):
    """
    Execute the selected command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Command exit code
    """
    command_name = args.command
    
    if command_name not in COMMAND_REGISTRY:
        logger.error(f"Unknown command: {command_name}")
        return 1
    
    try:
        # Get command function
        command_func = COMMAND_REGISTRY[command_name]['function']
        
        # Execute command
        return command_func(args) or 0
    except Exception as e:
        logger.exception(f"Error executing command {command_name}: {str(e)}")
        return 1

def import_commands():
    """Import all command modules to register commands."""
    try:
        from .commands import register_commands
        register_commands()
    except ImportError as e:
        logger.error(f"Error importing commands: {str(e)}")

def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI application.
    
    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code
    """
    # Import commands
    import_commands()
    
    # Create parser
    parser = create_cli_app()
    
    # Parse arguments
    args = parser.parse_args(argv)
    
    # Set up environment
    setup_environment(args)
    
    # Execute command
    return execute_command(args)

if __name__ == "__main__":
    sys.exit(main())
