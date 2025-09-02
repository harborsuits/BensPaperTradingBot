#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper Broker Factory

Extends the broker factory to support paper trading by creating PaperBroker instances
and integrating them with the broker manager.
"""

import logging
from typing import Dict, Optional, Any, Union

from .broker_interface import BrokerInterface
from .multi_broker_manager import MultiBrokerManager
from .paper_broker import PaperBroker

logger = logging.getLogger(__name__)


def create_paper_broker(
    config: Dict[str, Any],
    data_source: Optional[BrokerInterface] = None
) -> PaperBroker:
    """
    Create a configured paper broker instance.
    
    Args:
        config: Paper broker configuration
        data_source: Optional data source broker for real market data
        
    Returns:
        PaperBroker: Configured paper broker instance
    
    Example config:
    {
        "name": "MyPaperAccount",
        "initial_balance": 100000.0,  
        "base_currency": "USD",
        "slippage_model": {
            "type": "fixed",  # none, fixed, random
            "basis_points": 5  # 0.05% slippage
        },
        "commission_model": {
            "type": "per_share",  # none, fixed, per_share, percentage
            "per_share": 0.005,
            "minimum": 1.0
        }
    }
    """
    # Extract configuration
    name = config.get("name", "PaperBroker")
    initial_balance = float(config.get("initial_balance", 100000.0))
    base_currency = config.get("base_currency", "USD")
    slippage_model = config.get("slippage_model", {"type": "none"})
    commission_model = config.get("commission_model", {"type": "none"})
    
    # Create paper broker
    paper_broker = PaperBroker(
        name=name,
        initial_balance=initial_balance,
        base_currency=base_currency,
        slippage_model=slippage_model,
        commission_model=commission_model,
        data_source=data_source
    )
    
    logger.info(f"Created paper broker '{name}' with {initial_balance} {base_currency}")
    return paper_broker


def add_paper_broker_to_manager(
    manager: MultiBrokerManager,
    config: Dict[str, Any],
    data_source_id: Optional[str] = None
) -> bool:
    """
    Add a paper broker to the manager.
    
    Args:
        manager: Multi-Broker Manager instance
        config: Paper broker configuration
        data_source_id: Optional broker ID to use as data source
        
    Returns:
        bool: Success status
    """
    try:
        # Get data source if specified
        data_source = None
        if data_source_id and data_source_id in manager.get_broker_ids():
            data_source = manager.get_broker(data_source_id)
        
        # Create paper broker
        paper_broker = create_paper_broker(config, data_source)
        
        # Register with manager
        broker_id = config.get("id", "paper")
        primary = config.get("primary", False)
        
        manager.add_broker(
            broker_id,
            paper_broker,
            None,  # No credentials needed
            primary
        )
        
        logger.info(f"Added paper broker to manager with id '{broker_id}'{' (primary)' if primary else ''}")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up paper broker: {str(e)}")
        return False


def mirror_broker_as_paper(
    manager: MultiBrokerManager,
    source_broker_id: str,
    paper_config: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Create a paper broker that mirrors settings from an existing broker.
    
    Args:
        manager: Multi-Broker Manager instance
        source_broker_id: ID of the source broker to mirror
        paper_config: Additional paper-specific configuration
        
    Returns:
        str: Paper broker ID or None if failed
    """
    if source_broker_id not in manager.get_broker_ids():
        logger.error(f"Source broker '{source_broker_id}' not found in manager")
        return None
    
    try:
        # Get source broker
        source_broker = manager.get_broker(source_broker_id)
        
        # Create paper-specific configuration
        config = paper_config or {}
        
        # Set default values
        if "name" not in config:
            config["name"] = f"{source_broker_id.capitalize()}Paper"
        
        if "id" not in config:
            config["id"] = f"{source_broker_id}_paper"
        
        # Create paper broker with source as data provider
        paper_broker = create_paper_broker(config, source_broker)
        
        # Register with manager
        broker_id = config["id"]
        primary = config.get("primary", False)
        
        manager.add_broker(
            broker_id,
            paper_broker,
            None,  # No credentials needed
            primary
        )
        
        logger.info(f"Created paper mirror of '{source_broker_id}' with id '{broker_id}'")
        return broker_id
        
    except Exception as e:
        logger.error(f"Error creating paper mirror: {str(e)}")
        return None
