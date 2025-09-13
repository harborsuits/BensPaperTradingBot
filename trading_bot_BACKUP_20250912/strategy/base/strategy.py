#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Strategy - Foundational class for all trading strategies
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger("Strategy")

class Strategy:
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        self.name = name
        self.config = config or {}
        self.enabled = True
        self.performance_history = []
        self.last_signal = 0.0
        self.last_update_time = None
    
    def generate_signal(self, market_data: Dict[str, Any]) -> float:
        """
        Generate a trading signal between -1.0 and 1.0.
        
        Args:
            market_data: Current market data
            
        Returns:
            float: Signal between -1.0 (strong sell) and 1.0 (strong buy)
        """
        # Base implementation returns neutral
        logger.debug(f"Base generate_signal called for {self.name}")
        return 0.0
    
    def update_performance(self, performance_metric: float) -> None:
        """
        Update strategy performance.
        
        Args:
            performance_metric: Performance metric (e.g., return, profit)
        """
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "performance": performance_metric
        })
        logger.debug(f"Updated performance for {self.name}: {performance_metric}")
        
    def get_average_performance(self, window: int = 10) -> float:
        """
        Get average performance over last n periods.
        
        Args:
            window: Number of periods to average
            
        Returns:
            float: Average performance
        """
        if not self.performance_history:
            return 0.0
            
        # Take last n performance records
        recent = self.performance_history[-window:]
        
        if not recent:
            return 0.0
            
        return sum(r["performance"] for r in recent) / len(recent)
    
    def reset(self) -> None:
        """Reset strategy state."""
        self.performance_history = []
        self.last_signal = 0.0
        self.last_update_time = None
        logger.info(f"Reset state for strategy: {self.name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary for serialization."""
        return {
            "name": self.name,
            "config": self.config,
            "enabled": self.enabled,
            "performance_history": self.performance_history,
            "last_signal": self.last_signal,
            "last_update_time": self.last_update_time.isoformat() if self.last_update_time else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Strategy':
        """Create strategy from dictionary."""
        strategy = cls(data["name"], data["config"])
        strategy.enabled = data["enabled"]
        strategy.performance_history = data["performance_history"]
        strategy.last_signal = data["last_signal"]
        
        if data["last_update_time"]:
            strategy.last_update_time = datetime.fromisoformat(data["last_update_time"])
            
        return strategy 