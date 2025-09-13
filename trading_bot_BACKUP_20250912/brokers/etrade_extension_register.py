"""
E*TRADE Extension Registration

This module handles registration of E*TRADE-specific extensions with the
broker registry, making them available to the trading platform.
"""

import logging
from typing import Optional

from trading_bot.brokers.broker_registry import BrokerRegistry
from trading_bot.brokers.etrade_client import ETradeClient
from trading_bot.brokers.extensions.etrade_extensions import (
    ETradeOptionsExtension,
    ETradePortfolioExtension
)
from trading_bot.event_system import EventBus, Event

# Configure logging
logger = logging.getLogger(__name__)


def register_etrade_extensions(
    broker_registry: BrokerRegistry,
    etrade_client: Optional[ETradeClient] = None,
    broker_id: str = "etrade"
) -> None:
    """
    Register E*TRADE extensions with the broker registry
    
    Args:
        broker_registry: The broker registry instance
        etrade_client: Optional E*TRADE client instance (if None, will attempt to get from registry)
        broker_id: The ID of the E*TRADE broker in the registry
    """
    logger.info(f"Registering E*TRADE extensions for broker ID: {broker_id}")
    
    # Get E*TRADE client if not provided
    if etrade_client is None:
        try:
            etrade_client = broker_registry.get_broker_instance(broker_id)
            if etrade_client is None:
                logger.error(f"Could not find E*TRADE broker with ID: {broker_id}")
                return
        except Exception as e:
            logger.error(f"Error getting E*TRADE client: {str(e)}")
            return
    
    # Create extension instances
    options_extension = ETradeOptionsExtension(etrade_client)
    portfolio_extension = ETradePortfolioExtension(etrade_client)
    
    # Register extensions with the broker registry
    try:
        broker_registry.register_extension(broker_id, options_extension)
        logger.info(f"Registered {options_extension.get_extension_name()} with capabilities: {options_extension.get_capabilities()}")
        
        broker_registry.register_extension(broker_id, portfolio_extension)
        logger.info(f"Registered {portfolio_extension.get_extension_name()} with capabilities: {portfolio_extension.get_capabilities()}")
        
        # Publish event that extensions were registered
        EventBus().publish(Event(
            "broker_extensions_registered",
            {
                "broker_id": broker_id,
                "extensions": [
                    options_extension.get_extension_name(),
                    portfolio_extension.get_extension_name()
                ]
            }
        ))
        
        logger.info(f"Successfully registered all E*TRADE extensions for broker ID: {broker_id}")
    except Exception as e:
        logger.error(f"Error registering E*TRADE extensions: {str(e)}")


# Utility function to check if extension is available
def has_etrade_options_extension(broker_registry: BrokerRegistry, broker_id: str = "etrade") -> bool:
    """
    Check if E*TRADE options extension is available
    
    Args:
        broker_registry: The broker registry instance
        broker_id: The ID of the E*TRADE broker in the registry
        
    Returns:
        bool: True if extension is available
    """
    try:
        extensions = broker_registry.get_broker_extensions(broker_id)
        for ext in extensions:
            if ext.get_extension_name() == "ETradeOptionsExtension":
                return True
        return False
    except Exception:
        return False


def has_etrade_portfolio_extension(broker_registry: BrokerRegistry, broker_id: str = "etrade") -> bool:
    """
    Check if E*TRADE portfolio extension is available
    
    Args:
        broker_registry: The broker registry instance
        broker_id: The ID of the E*TRADE broker in the registry
        
    Returns:
        bool: True if extension is available
    """
    try:
        extensions = broker_registry.get_broker_extensions(broker_id)
        for ext in extensions:
            if ext.get_extension_name() == "ETradePortfolioExtension":
                return True
        return False
    except Exception:
        return False
