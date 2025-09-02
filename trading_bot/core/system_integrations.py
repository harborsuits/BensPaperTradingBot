"""
System Integrations

This module provides integration hooks to connect different components
to the main BensBot trading system.
"""

import logging
from typing import Dict, List, Any, Optional

# Import system components
from trading_bot.core.trading_system import TradingSystem
from trading_bot.analytics.market_regime.bootstrap import setup_market_regime_system

logger = logging.getLogger(__name__)

def initialize_integrations(system: TradingSystem, config: Dict[str, Any]) -> None:
    """
    Initialize all system integrations.
    
    Args:
        system: The main trading system
        config: System configuration
    """
    logger.info("Initializing system integrations")
    
    # Initialize market regime system if enabled
    if config.get("enable_market_regime_system", True):
        logger.info("Initializing market regime system")
        regime_config_path = config.get("market_regime_config_path")
        
        try:
            regime_manager = setup_market_regime_system(system, regime_config_path)
            system.register_component("market_regime_manager", regime_manager)
            logger.info("Market regime system initialized and registered")
        except Exception as e:
            logger.error(f"Failed to initialize market regime system: {str(e)}")
            logger.warning("System will continue without market regime functionality")
    
    # Initialize other integrations here
    
    logger.info("System integrations initialization complete")

def shutdown_integrations(system: TradingSystem) -> None:
    """
    Shut down all system integrations.
    
    Args:
        system: The main trading system
    """
    logger.info("Shutting down system integrations")
    
    # Shut down market regime system
    if hasattr(system, "market_regime_manager"):
        try:
            system.market_regime_manager.shutdown()
            logger.info("Market regime system shut down")
        except Exception as e:
            logger.error(f"Error shutting down market regime system: {str(e)}")
    
    # Shut down other integrations here
    
    logger.info("System integrations shutdown complete")
