"""
Utility modules for the trading bot.

Contains helper functions and classes for data processing, 
market analysis, strategy management, and system utilities.
"""

# Define initial list of exported names
__all__ = [
    'strategy_library',
    'market_context_fetcher'
]

"""
Utility functions and classes for the trading bot.

This package provides various utility functions and helper classes
used throughout the trading bot system.
"""

# Try to import LLM client, but don't fail if dependencies are missing
try:
    from trading_bot.utils.llm_client import LLMClient, analyze_with_gpt4, get_llm_client
    __all__.extend(['LLMClient', 'analyze_with_gpt4', 'get_llm_client'])
except ImportError as e:
    import logging
    logging.warning(f"LLM client not available: {e}. Some features may be limited.")

"""
Utilities package
""" 