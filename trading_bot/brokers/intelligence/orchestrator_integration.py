#!/usr/bin/env python3
"""
Broker Intelligence and Orchestrator Integration

Integrates the broker intelligence system with the main orchestrator
while preserving the orchestrator's decision-making authority.
"""

import logging
import threading
from typing import Dict, List, Any, Optional, Callable

from trading_bot.brokers.intelligence.broker_advisor import BrokerSelectionAdvice
from trading_bot.brokers.intelligence.multi_broker_integration import BrokerIntelligenceEngine
from trading_bot.core.event_system import EventSystem, EventType, EventHandler

# Configure logging
logger = logging.getLogger(__name__)


class OrchestratorAdvisor:
    """
    Provides intelligence to the orchestrator without overriding its decisions.
    
    This class is the bridge between the broker intelligence system and
    the main orchestrator, offering advisory input while preserving the
    orchestrator's ultimate decision-making authority.
    """
    
    def __init__(
        self,
        intelligence_engine: BrokerIntelligenceEngine,
        event_system: EventSystem
    ):
        """
        Initialize orchestrator advisor
        
        Args:
            intelligence_engine: Broker intelligence engine
            event_system: System-wide event bus
        """
        self.intelligence_engine = intelligence_engine
        self.event_system = event_system
        
        # Current asset routing table
        self.asset_routing: Dict[str, Dict[str, Any]] = {}
        
        # Register for intelligence events
        self.event_system.register(
            EventType.BROKER_INTELLIGENCE_UPDATE,
            self._handle_intelligence_update
        )
        
        self.event_system.register(
            EventType.BROKER_CIRCUIT_BREAKER,
            self._handle_circuit_breaker_event
        )
        
        self.event_system.register(
            EventType.BROKER_HEALTH_STATUS_CHANGE,
            self._handle_health_status_change
        )
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        logger.info("Initialized OrchestratorAdvisor")
    
    def _handle_intelligence_update(self, event_data: Dict[str, Any]):
        """
        Handle broker intelligence update event
        
        Args:
            event_data: Intelligence update data
        """
        # Extract key information from update
        health_status = event_data.get("health_status")
        recommendations = event_data.get("recommendations", {})
        
        # Prepare advisory report for orchestrator
        advisory_report = {
            "health_status": health_status,
            "recommendations": recommendations,
            "timestamp": event_data.get("timestamp")
        }
        
        # Emit an advisor event for the orchestrator
        self.event_system.emit(
            event_type=EventType.ORCHESTRATOR_ADVISORY_UPDATE,
            data=advisory_report
        )
        
        logger.debug(f"Processed intelligence update: health={health_status}")
    
    def _handle_circuit_breaker_event(self, event_data: Dict[str, Any]):
        """
        Handle broker circuit breaker event
        
        Args:
            event_data: Circuit breaker event data
        """
        broker_id = event_data.get("broker_id")
        reason = event_data.get("reason")
        
        logger.warning(f"Circuit breaker tripped for {broker_id}: {reason}")
        
        # Add advisory note for orchestrator
        advisory = {
            "type": "circuit_breaker",
            "broker_id": broker_id,
            "reason": reason,
            "timestamp": event_data.get("timestamp"),
            "recommendations": self._get_failover_recommendations(broker_id)
        }
        
        # Emit advisory event
        self.event_system.emit(
            event_type=EventType.ORCHESTRATOR_ADVISORY_ALERT,
            data=advisory
        )
    
    def _handle_health_status_change(self, event_data: Dict[str, Any]):
        """
        Handle health status change event
        
        Args:
            event_data: Health status event data
        """
        status = event_data.get("status")
        
        logger.warning(f"Broker health status changed to: {status}")
        
        # Add advisory note for orchestrator
        advisory = {
            "type": "health_status",
            "status": status,
            "drawdown_alerts": event_data.get("drawdown_alerts", {}),
            "timestamp": event_data.get("timestamp")
        }
        
        # Emit advisory event
        self.event_system.emit(
            event_type=EventType.ORCHESTRATOR_ADVISORY_ALERT,
            data=advisory
        )
    
    def _get_failover_recommendations(self, failed_broker_id: str) -> Dict[str, Any]:
        """
        Get failover recommendations when a broker fails
        
        Args:
            failed_broker_id: ID of the failed broker
            
        Returns:
            Dictionary of recommendations
        """
        # Find what assets might be affected
        affected_assets = []
        for asset, routing in self.asset_routing.items():
            if routing.get("broker_id") == failed_broker_id:
                affected_assets.append(asset)
        
        # Get recommendations for affected assets
        recommendations = {}
        for asset in affected_assets:
            asset_class = self.intelligence_engine.get_asset_class(asset)
            
            # Get recommendations for different operation types
            for operation in ["order", "quote", "data"]:
                advice = self.intelligence_engine.get_broker_recommendation(
                    asset=asset,
                    operation_type=operation,
                    current_broker_id=failed_broker_id
                )
                
                recommendations[f"{asset}_{operation}"] = advice.to_dict()
        
        return recommendations
    
    def update_asset_routing(self, asset_routing: Dict[str, Dict[str, Any]]):
        """
        Update current asset routing table
        
        Args:
            asset_routing: Asset routing configuration
        """
        with self.lock:
            self.asset_routing = asset_routing
    
    def get_broker_recommendation(
        self,
        asset: str,
        operation_type: str,
        current_broker_id: Optional[str] = None
    ) -> BrokerSelectionAdvice:
        """
        Get broker recommendation for a specific asset and operation
        
        Args:
            asset: Asset symbol or identifier
            operation_type: Operation type
            current_broker_id: Currently selected broker
            
        Returns:
            Broker selection advice
        """
        return self.intelligence_engine.get_broker_recommendation(
            asset=asset,
            operation_type=operation_type,
            current_broker_id=current_broker_id
        )


class MultiBrokerAdvisorIntegration:
    """
    Integrates the intelligence layer with the MultiBrokerManager
    
    This class connects the broker intelligence system with the
    MultiBrokerManager to provide enhanced capabilities while
    preserving the orchestrator's decision-making authority.
    """
    
    def __init__(
        self,
        intelligence_engine: BrokerIntelligenceEngine,
        event_system: EventSystem,
        multi_broker_manager: Any  # Avoid circular import
    ):
        """
        Initialize integration
        
        Args:
            intelligence_engine: Broker intelligence engine
            event_system: System-wide event bus
            multi_broker_manager: MultiBrokerManager instance
        """
        self.intelligence_engine = intelligence_engine
        self.event_system = event_system
        self.multi_broker_manager = multi_broker_manager
        
        # Create orchestrator advisor
        self.orchestrator_advisor = OrchestratorAdvisor(
            intelligence_engine=intelligence_engine,
            event_system=event_system
        )
        
        # Register for broker events
        self._register_broker_events()
        
        # Initial integration
        self._integrate_broker_capabilities()
        
        logger.info("Initialized MultiBrokerAdvisorIntegration")
    
    def _register_broker_events(self):
        """Register for relevant broker events"""
        # Connect to broker events for metric collection
        event_types = [
            EventType.BROKER_ORDER_PLACED,
            EventType.BROKER_ORDER_FILLED,
            EventType.BROKER_ORDER_FAILED,
            EventType.BROKER_CONNECTION_ERROR,
            EventType.BROKER_QUOTE_RECEIVED,
            EventType.BROKER_DATA_RECEIVED
        ]
        
        for event_type in event_types:
            self.event_system.register(
                event_type=event_type,
                handler=self._handle_broker_event
            )
    
    def _handle_broker_event(self, event_data: Dict[str, Any]):
        """
        Handle broker events for metric collection
        
        Args:
            event_data: Event data
        """
        # Extract broker ID and event details
        broker_id = event_data.get("broker_id")
        if not broker_id:
            return
        
        # Forward to intelligence engine for metric collection
        # (metrics manager handles the actual collection)
    
    def _integrate_broker_capabilities(self):
        """Integrate broker capabilities with intelligence engine"""
        # Get registered brokers from manager
        brokers = self.multi_broker_manager.get_registered_brokers()
        
        # Register each broker with intelligence engine
        for broker_id, broker_data in brokers.items():
            broker_instance = broker_data.get("instance")
            if not broker_instance:
                continue
            
            # Extract capabilities from broker
            capabilities = self._extract_broker_capabilities(broker_instance)
            
            # Register with intelligence engine
            self.intelligence_engine.register_broker(
                broker_id=broker_id,
                broker_instance=broker_instance,
                asset_classes=capabilities.get("asset_classes", []),
                operation_types=capabilities.get("operation_types", []),
                capabilities=capabilities
            )
        
        # Set up asset routing
        asset_routing = self.multi_broker_manager.get_asset_routing()
        self.orchestrator_advisor.update_asset_routing(asset_routing)
    
    def _extract_broker_capabilities(self, broker_instance: Any) -> Dict[str, Any]:
        """
        Extract capabilities from a broker instance
        
        Args:
            broker_instance: Broker interface instance
            
        Returns:
            Dictionary of capabilities
        """
        # Default capabilities based on broker interface
        capabilities = {
            "asset_classes": ["equities"],  # Default, will be extended
            "operation_types": ["order", "quote", "data"],
            "rate_limits": {},
            "trading_limits": {}
        }
        
        # Try to get specific capabilities from broker
        if hasattr(broker_instance, "get_capabilities"):
            try:
                broker_caps = broker_instance.get_capabilities()
                capabilities.update(broker_caps)
            except Exception as e:
                logger.error(f"Error getting broker capabilities: {str(e)}")
        
        return capabilities
    
    def start(self):
        """Start integration services"""
        # Start intelligence engine monitoring
        self.intelligence_engine.start_monitoring()
        logger.info("Started broker intelligence integration")
    
    def stop(self):
        """Stop integration services"""
        # Stop intelligence engine monitoring
        self.intelligence_engine.stop_monitoring()
        logger.info("Stopped broker intelligence integration")
