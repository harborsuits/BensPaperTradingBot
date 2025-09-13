#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-Line Interface Package

This package provides a unified command-line interface for the trading bot.
"""

from .commands import register_commands
from .cli_app import create_cli_app, main

__all__ = [
    'register_commands',
    'create_cli_app',
    'main'
]
