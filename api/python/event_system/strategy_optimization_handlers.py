#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Optimization Event Handlers

This module contains handlers for strategy optimization events, connecting
the autonomous engine's optimization process with the UI and other system components.
"""

import logging
from typing import Dict, Any, Optional
import json
from datetime import datetime

from trading_bot.event_system import Event, EventHandler, EventBus, EventType

logger = logging.getLogger(__name__)

class StrategyOptimizationTracker:
    """
    Tracks strategy optimization progress and results.
    Acts as a central hub for optimization data that the UI can query.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the optimization tracker.
        
        Args:
            event_bus: Event bus to register handlers with
        """
        self.optimizations = {}  # Stores optimization data by strategy_id
        self.latest_events = {}  # Stores the latest events by type
        self.event_bus = event_bus
        
        # Register with event bus if provided
        if event_bus:
            self._register_handlers()
    
    def _register_handlers(self) -> None:
        """Register event handlers with the event bus."""
        self.event_bus.register_handler(
            EventHandler(
                callback=self._handle_optimization_event,
                event_type="STRATEGY_OPTIMISED",
                name="strategy_optimised_handler"
            )
        )
        
        self.event_bus.register_handler(
            EventHandler(
                callback=self._handle_optimization_event,
                event_type="STRATEGY_EXHAUSTED",
                name="strategy_exhausted_handler"
            )
        )
        
        logger.info("Registered strategy optimization event handlers")
    
    def _handle_optimization_event(self, event: Event) -> None:
        """
        Handle optimization event.
        
        Args:
            event: Event to handle
        """
        try:
            strategy_id = event.data.get("strategy_id")
            if not strategy_id:
                logger.warning(f"Optimization event missing strategy_id: {event.event_type}")
                return
                
            # Store event data
            self.latest_events[event.event_type] = event.data
            
            # Update optimization data
            if strategy_id not in self.optimizations:
                self.optimizations[strategy_id] = {
                    "status": "pending",
                    "history": [],
                    "last_updated": datetime.now().isoformat(),
                    "events": []
                }
                
            optimization = self.optimizations[strategy_id]
            
            # Update status based on event type
            if event.event_type == "STRATEGY_OPTIMISED":
                optimization["status"] = "optimized"
            elif event.event_type == "STRATEGY_EXHAUSTED":
                optimization["status"] = "exhausted"
            
            # Add event to history
            event_record = {
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "data": event.data
            }
            optimization["events"].append(event_record)
            
            # Store optimization data
            if event.event_type == "STRATEGY_OPTIMISED":
                optimization["original_parameters"] = event.data.get("original_parameters", {})
                optimization["optimized_parameters"] = event.data.get("optimized_parameters", {})
                optimization["performance"] = event.data.get("performance", {})
            elif event.event_type == "STRATEGY_EXHAUSTED":
                optimization["parameters"] = event.data.get("parameters", {})
                optimization["performance"] = event.data.get("performance", {})
                optimization["thresholds"] = event.data.get("thresholds", {})
            
            optimization["last_updated"] = datetime.now().isoformat()
            
            logger.info(f"Updated optimization data for strategy {strategy_id}: {event.event_type}")
            
        except Exception as e:
            logger.error(f"Error handling optimization event: {e}")
    
    def get_optimization_data(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get optimization data.
        
        Args:
            strategy_id: Strategy ID to get data for, or None for all
            
        Returns:
            Dictionary with optimization data
        """
        if strategy_id:
            return self.optimizations.get(strategy_id, {})
        return self.optimizations
    
    def get_latest_events(self) -> Dict[str, Any]:
        """
        Get latest events.
        
        Returns:
            Dictionary with latest events by type
        """
        return self.latest_events
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get optimization summary.
        
        Returns:
            Dictionary with optimization summary
        """
        total = len(self.optimizations)
        optimized = sum(1 for data in self.optimizations.values() if data.get("status") == "optimized")
        exhausted = sum(1 for data in self.optimizations.values() if data.get("status") == "exhausted")
        pending = total - optimized - exhausted
        
        return {
            "total": total,
            "optimized": optimized,
            "exhausted": exhausted,
            "pending": pending,
            "last_updated": datetime.now().isoformat()
        }

# Singleton instance for global access
optimization_tracker = None

def get_optimization_tracker(event_bus: Optional[EventBus] = None) -> StrategyOptimizationTracker:
    """
    Get the optimization tracker instance.
    
    Args:
        event_bus: Event bus to register handlers with
        
    Returns:
        StrategyOptimizationTracker instance
    """
    global optimization_tracker
    if optimization_tracker is None:
        optimization_tracker = StrategyOptimizationTracker(event_bus)
    return optimization_tracker
