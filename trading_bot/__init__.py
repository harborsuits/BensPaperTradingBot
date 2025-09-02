#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Bot Framework

A framework for building, testing, and deploying automated trading strategies.
"""

# Package metadata
__version__ = "0.1.0"
__author__ = "Ben Dickinson"

# Import key components for easy access
from trading_bot.orchestration.main_orchestrator import MainOrchestrator
from trading_bot.assistant.benbot_assistant import BenBotAssistant
from trading_bot.core.trading_bot import TradingBot, TradingMode, TradeResult, TradeAction, TradeType

# Initialize logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress verbose logging from libraries
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Export key classes
__all__ = [
    'MainOrchestrator',
    'BenBotAssistant',
    'TradingBot',
    'TradingMode',
    'TradeResult',
    'TradeAction',
    'TradeType',
] 