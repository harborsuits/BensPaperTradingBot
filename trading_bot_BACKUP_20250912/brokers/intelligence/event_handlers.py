#!/usr/bin/env python3
"""
Broker Intelligence Event Handlers

Handlers for broker intelligence events that update metrics and provide
situational awareness to the orchestrator without overriding decisions.
"""

import logging
import threading
import time
from typing import Dict, List, Any, Optional, Callable

from trading_bot.event_system.event_types import EventType, Event
from trading_bot.event_system.event_handler import EventHandler
from trading_bot.brokers.metrics.manager import BrokerMetricsManager
from trading_bot.brokers.intelligence.multi_broker_integration import BrokerIntelligenceEngine

# Configure logging
logger = logging.getLogger(__name__)


class BrokerMetricsEventHandler(EventHandler):
    """
    Event handler for collecting broker performance metrics
    
    Processes broker events and updates metrics accordingly,
    enabling comprehensive performance tracking.
    """
    
    def __init__(self, metrics_manager: BrokerMetricsManager):
        """
        Initialize the handler
        
        Args:
            metrics_manager: Broker metrics manager
        """
        super().__init__()
        self.metrics_manager = metrics_manager
        self.event_types = [
            EventType.BROKER_ORDER_PLACED,
            EventType.BROKER_ORDER_FILLED,
            EventType.BROKER_ORDER_FAILED,
            EventType.BROKER_CONNECTION_ERROR,
            EventType.BROKER_QUOTE_RECEIVED,
            EventType.BROKER_DATA_RECEIVED
        ]
        
        logger.info("Initialized BrokerMetricsEventHandler")
    
    def handle_event(self, event: Event):
        """
        Handle broker events for metric collection
        
        Args:
            event: Broker event to process
        """
        try:
            # Extract broker ID and event details
            broker_id = event.data.get("broker_id")
            if not broker_id:
                return
            
            if event.event_type == EventType.BROKER_ORDER_PLACED:
                self._handle_order_placed(broker_id, event)
                
            elif event.event_type == EventType.BROKER_ORDER_FILLED:
                self._handle_order_filled(broker_id, event)
                
            elif event.event_type == EventType.BROKER_ORDER_FAILED:
                self._handle_order_failed(broker_id, event)
                
            elif event.event_type == EventType.BROKER_CONNECTION_ERROR:
                self._handle_connection_error(broker_id, event)
                
            elif event.event_type == EventType.BROKER_QUOTE_RECEIVED:
                self._handle_quote_received(broker_id, event)
                
            elif event.event_type == EventType.BROKER_DATA_RECEIVED:
                self._handle_data_received(broker_id, event)
                
        except Exception as e:
            logger.error(f"Error handling broker event: {str(e)}")
    
    def _handle_order_placed(self, broker_id: str, event: Event):
        """Handle order placed event"""
        # Extract timing information
        start_time = event.data.get("request_start_time")
        end_time = event.data.get("request_end_time")
        
        if start_time and end_time:
            # Calculate latency in milliseconds
            latency_ms = (end_time - start_time) * 1000
            
            # Record latency metric
            self.metrics_manager.add_latency_metric(
                broker_id=broker_id,
                operation="place_order",
                latency_ms=latency_ms
            )
            
            logger.debug(f"Recorded order placement latency for {broker_id}: {latency_ms:.2f}ms")
    
    def _handle_order_filled(self, broker_id: str, event: Event):
        """Handle order filled event"""
        # Extract execution quality data
        expected_price = event.data.get("expected_price")
        actual_price = event.data.get("execution_price")
        
        if expected_price and actual_price:
            # Calculate slippage
            slippage = (actual_price - expected_price) / expected_price
            
            # Record execution quality metric
            self.metrics_manager.add_execution_quality_metric(
                broker_id=broker_id,
                slippage_pct=slippage * 100,
                order_data=event.data
            )
            
            logger.debug(f"Recorded execution quality for {broker_id}: {slippage:.5f}")
        
        # Record cost metrics if available
        commission = event.data.get("commission")
        if commission is not None:
            self.metrics_manager.add_cost_metric(
                broker_id=broker_id,
                commission=commission,
                other_fees=event.data.get("fees", 0)
            )
    
    def _handle_order_failed(self, broker_id: str, event: Event):
        """Handle order failed event"""
        # Record reliability metric (error)
        error_type = event.data.get("error_type", "unknown")
        error_msg = event.data.get("error_message", "")
        
        self.metrics_manager.add_reliability_metric(
            broker_id=broker_id,
            operation="place_order",
            success=False,
            error_type=error_type,
            error_message=error_msg
        )
        
        logger.debug(f"Recorded order failure for {broker_id}: {error_type}")
    
    def _handle_connection_error(self, broker_id: str, event: Event):
        """Handle connection error event"""
        # Record reliability metric (connection error)
        error_type = event.data.get("error_type", "connection")
        error_msg = event.data.get("error_message", "")
        
        self.metrics_manager.add_reliability_metric(
            broker_id=broker_id,
            operation="connection",
            success=False,
            error_type=error_type,
            error_message=error_msg
        )
        
        logger.debug(f"Recorded connection error for {broker_id}: {error_type}")
    
    def _handle_quote_received(self, broker_id: str, event: Event):
        """Handle quote received event"""
        # Extract timing information
        start_time = event.data.get("request_start_time")
        end_time = event.data.get("request_end_time")
        
        if start_time and end_time:
            # Calculate latency
            latency_ms = (end_time - start_time) * 1000
            
            # Record latency metric
            self.metrics_manager.add_latency_metric(
                broker_id=broker_id,
                operation="get_quote",
                latency_ms=latency_ms
            )
            
            # Record successful operation
            self.metrics_manager.add_reliability_metric(
                broker_id=broker_id,
                operation="get_quote",
                success=True
            )
            
            logger.debug(f"Recorded quote latency for {broker_id}: {latency_ms:.2f}ms")
    
    def _handle_data_received(self, broker_id: str, event: Event):
        """Handle market data received event"""
        # Extract timing information
        start_time = event.data.get("request_start_time")
        end_time = event.data.get("request_end_time")
        
        if start_time and end_time:
            # Calculate latency
            latency_ms = (end_time - start_time) * 1000
            
            # Record latency metric
            self.metrics_manager.add_latency_metric(
                broker_id=broker_id,
                operation="get_data",
                latency_ms=latency_ms
            )
            
            # Record successful operation
            self.metrics_manager.add_reliability_metric(
                broker_id=broker_id,
                operation="get_data",
                success=True
            )
            
            logger.debug(f"Recorded data retrieval latency for {broker_id}: {latency_ms:.2f}ms")


class BrokerIntelligenceEventHandler(EventHandler):
    """
    Event handler for broker intelligence events
    
    Processes intelligence events and provides advisory input
    to the orchestrator without overriding its decisions.
    """
    
    def __init__(self, intelligence_engine: BrokerIntelligenceEngine):
        """
        Initialize the handler
        
        Args:
            intelligence_engine: Broker intelligence engine
        """
        super().__init__()
        self.intelligence_engine = intelligence_engine
        self.event_types = [
            EventType.BROKER_INTELLIGENCE_UPDATE,
            EventType.BROKER_CIRCUIT_BREAKER,
            EventType.BROKER_HEALTH_STATUS_CHANGE,
            EventType.ORCHESTRATOR_ADVISORY_UPDATE,
            EventType.ORCHESTRATOR_ADVISORY_ALERT
        ]
        
        logger.info("Initialized BrokerIntelligenceEventHandler")
    
    def handle_event(self, event: Event):
        """
        Handle broker intelligence events
        
        Args:
            event: Intelligence event to process
        """
        try:
            # Handle specific event types
            if event.event_type == EventType.BROKER_CIRCUIT_BREAKER:
                self._handle_circuit_breaker(event)
                
            elif event.event_type == EventType.BROKER_HEALTH_STATUS_CHANGE:
                self._handle_health_status_change(event)
                
        except Exception as e:
            logger.error(f"Error handling intelligence event: {str(e)}")
    
    def _handle_circuit_breaker(self, event: Event):
        """
        Handle circuit breaker event
        
        Args:
            event: Circuit breaker event
        """
        broker_id = event.data.get("broker_id")
        reason = event.data.get("reason")
        
        if not broker_id:
            return
            
        logger.warning(f"Processing circuit breaker event for {broker_id}: {reason}")
        
        # The engine will handle failover recommendations
        # The orchestrator will make the final decision
    
    def _handle_health_status_change(self, event: Event):
        """
        Handle health status change event
        
        Args:
            event: Health status event
        """
        status = event.data.get("status")
        
        logger.warning(f"Processing health status change: {status}")
        
        # Additional handlers for specific health status changes could go here
        # The orchestrator will make the final decisions
