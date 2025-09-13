#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Strategy interface.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy.
        
        Args:
            name: Name of the strategy
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.is_active = False
        
    @abstractmethod
    def generate_signals(self, data: Any) -> Dict[str, Any]:
        """
        Generate trading signals based on the provided data.
        
        Args:
            data: Market data or other inputs
            
        Returns:
            Dictionary containing signal information
        """
        pass
    
    def activate(self) -> None:
        """Activate the strategy."""
        self.is_active = True
        logger.info(f"Strategy {self.name} activated")
        
    def deactivate(self) -> None:
        """Deactivate the strategy."""
        self.is_active = False
        logger.info(f"Strategy {self.name} deactivated")
        
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"{self.name} Strategy (active: {self.is_active})" 