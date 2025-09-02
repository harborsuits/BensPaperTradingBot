#!/usr/bin/env python3
"""
Setup Broker Intelligence System

This script demonstrates how to set up and integrate the broker intelligence system
with the existing broker management framework and orchestrator.

Usage:
    python -m trading_bot.scripts.setup_broker_intelligence

This will initialize the broker intelligence system, connect it to the broker manager,
and register the necessary event handlers.
"""

import os
import logging
import argparse
import json
from typing import Dict, Any, List

from trading_bot.brokers.metrics.manager import BrokerMetricsManager
from trading_bot.brokers.intelligence.broker_advisor import BrokerAdvisor
from trading_bot.brokers.intelligence.multi_broker_integration import BrokerIntelligenceEngine
from trading_bot.brokers.intelligence.orchestrator_integration import MultiBrokerAdvisorIntegration
from trading_bot.brokers.intelligence.event_handlers import (
    BrokerMetricsEventHandler, 
    BrokerIntelligenceEventHandler
)
from trading_bot.event_system.event_manager import EventManager
from trading_bot.brokers.multi_broker_manager import MultiBrokerManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("BrokerIntelligenceSetup")


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load broker intelligence configuration"""
    if not config_path:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config",
            "broker_intelligence_config.json"
        )
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Config not found at {config_path}, using defaults")
            return create_default_config()
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return create_default_config()


def create_default_config() -> Dict[str, Any]:
    """Create default configuration for broker intelligence"""
    return {
        "enable_intelligence": True,
        "monitor_interval_seconds": 15,
        "advisor": {
            "factor_weights": {
                "latency": 0.20,
                "reliability": 0.40,
                "execution_quality": 0.25,
                "cost": 0.15
            },
            "circuit_breaker_thresholds": {
                "error_count": 5,
                "error_rate": 0.3,
                "availability_min": 90.0,
                "reset_after_seconds": 300
            },
            "asset_class_weights": {
                "equities": {
                    "latency": 0.15,
                    "reliability": 0.45,
                    "execution_quality": 0.30,
                    "cost": 0.10
                },
                "forex": {
                    "latency": 0.30,
                    "reliability": 0.40,
                    "execution_quality": 0.20,
                    "cost": 0.10
                },
                "options": {
                    "latency": 0.15,
                    "reliability": 0.30,
                    "execution_quality": 0.40,
                    "cost": 0.15
                }
            }
        },
        "risk_thresholds": {
            "drawdown_warning": 3.0,
            "drawdown_critical": 4.0,
            "error_rate_warning": 0.1,
            "error_rate_critical": 0.3
        }
    }


def setup_broker_intelligence(
    event_manager: EventManager,
    multi_broker_manager: MultiBrokerManager,
    config: Dict[str, Any]
) -> MultiBrokerAdvisorIntegration:
    """
    Set up broker intelligence system
    
    Args:
        event_manager: Event manager instance
        multi_broker_manager: Multi-broker manager instance
        config: Intelligence system configuration
        
    Returns:
        Integration instance
    """
    logger.info("Setting up broker intelligence system...")
    
    # Create metrics manager
    metrics_manager = BrokerMetricsManager()
    
    # Create intelligence engine
    intelligence_engine = BrokerIntelligenceEngine(
        metrics_manager=metrics_manager,
        event_system=event_manager,
        config=config
    )
    
    # Create advisor integration
    integration = MultiBrokerAdvisorIntegration(
        intelligence_engine=intelligence_engine,
        event_system=event_manager,
        multi_broker_manager=multi_broker_manager
    )
    
    # Register event handlers
    metrics_handler = BrokerMetricsEventHandler(metrics_manager)
    intelligence_handler = BrokerIntelligenceEventHandler(intelligence_engine)
    
    # Register handlers with event manager
    for event_type in metrics_handler.event_types:
        event_manager.register_handler(event_type, metrics_handler)
    
    for event_type in intelligence_handler.event_types:
        event_manager.register_handler(event_type, intelligence_handler)
    
    # Start monitoring
    intelligence_engine.start_monitoring()
    
    logger.info("Broker intelligence system is now active")
    return integration


def demo_broker_intelligence():
    """Demonstrate broker intelligence setup and usage"""
    # This function would be used when running this script directly
    logger.info("Broker Intelligence System Demo")
    
    # Load configuration
    config = load_config()
    
    # In a real application, these would be passed in or imported
    from trading_bot.event_system.event_manager import get_event_manager
    from trading_bot.brokers.multi_broker_manager import get_broker_manager
    
    event_manager = get_event_manager()
    broker_manager = get_broker_manager()
    
    # Set up intelligence system
    integration = setup_broker_intelligence(
        event_manager=event_manager,
        multi_broker_manager=broker_manager,
        config=config
    )
    
    logger.info("Broker intelligence system setup complete")
    logger.info("Intelligence system is now providing situational awareness to the orchestrator")
    logger.info("All decision-making authority remains with the orchestrator")
    
    # In a real application this would continue running 
    # as part of the main application


def generate_example_config():
    """Generate example configuration file"""
    config = create_default_config()
    
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config",
        "broker_intelligence_config.json"
    )
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Example configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Error saving example config: {str(e)}")


def main():
    """Main entry point for script"""
    parser = argparse.ArgumentParser(description="Set up broker intelligence system")
    parser.add_argument("--create-config", action="store_true", 
                        help="Generate example configuration file")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to custom configuration file")
    
    args = parser.parse_args()
    
    if args.create_config:
        generate_example_config()
        return
    
    # Run demo
    demo_broker_intelligence()


if __name__ == "__main__":
    main()
