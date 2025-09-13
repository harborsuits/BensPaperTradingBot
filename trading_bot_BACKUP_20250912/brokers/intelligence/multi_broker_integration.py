#!/usr/bin/env python3
"""
Multi-Broker Intelligence Integration

Integrates the BrokerAdvisor with MultiBrokerManager to provide intelligence
without interfering with the orchestrator's decision-making authority.
"""

import logging
import threading
import time
from typing import Dict, List, Any, Optional, Tuple, Set

from trading_bot.brokers.intelligence.broker_advisor import BrokerAdvisor, BrokerSelectionAdvice
from trading_bot.brokers.metrics.manager import BrokerMetricsManager
from trading_bot.brokers.broker_interface import BrokerInterface
from trading_bot.core.event_system import EventSystem, EventType

# Configure logging
logger = logging.getLogger(__name__)

class BrokerIntelligenceEngine:
    """
    Intelligence engine that integrates broker metrics and advisor with
    the multi-broker manager for enhanced situational awareness.
    
    This class is responsible for:
    1. Collecting and analyzing broker performance metrics
    2. Providing intelligence to the orchestrator
    3. Alerting when broker performance degrades
    4. Suggesting optimal broker selection
    
    All while preserving the orchestrator's decision-making authority.
    """
    
    def __init__(
        self,
        metrics_manager: BrokerMetricsManager,
        event_system: EventSystem,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize broker intelligence engine
        
        Args:
            metrics_manager: Broker metrics manager
            event_system: System-wide event bus
            config: Configuration dictionary
        """
        self.metrics_manager = metrics_manager
        self.event_system = event_system
        self.config = config or {}
        
        # Create broker advisor
        self.advisor = BrokerAdvisor(metrics_manager, self.config.get("advisor", {}))
        
        # Track registered brokers
        self.registered_brokers: Dict[str, Dict[str, Any]] = {}
        
        # Track current recommendations
        self.current_recommendations: Dict[str, BrokerSelectionAdvice] = {}
        
        # For asset routing
        self.asset_class_map: Dict[str, str] = {}
        
        # Risk management state
        self.drawdown_alerts = {}
        self.health_status = "NORMAL"  # NORMAL, CAUTION, CRITICAL
        
        # Monitoring thread
        self.monitor_thread = None
        self.monitor_active = False
        self.monitor_interval = self.config.get("monitor_interval_seconds", 15)
        
        # Risk thresholds
        self.risk_thresholds = self.config.get("risk_thresholds", {
            "drawdown_warning": 3.0,  # Alert at 3% drawdown
            "drawdown_critical": 4.0,  # Critical at 4% drawdown
            "error_rate_warning": 0.1,  # 10% error rate
            "error_rate_critical": 0.3,  # 30% error rate
        })
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        logger.info("Initialized BrokerIntelligenceEngine")

    def register_broker(
        self,
        broker_id: str,
        broker_instance: BrokerInterface,
        asset_classes: List[str],
        operation_types: List[str],
        capabilities: Optional[Dict[str, Any]] = None
    ):
        """
        Register broker with the intelligence engine
        
        Args:
            broker_id: Broker identifier
            broker_instance: Broker interface instance
            asset_classes: Supported asset classes
            operation_types: Supported operations
            capabilities: Additional broker capabilities
        """
        with self.lock:
            # Store broker details
            self.registered_brokers[broker_id] = {
                "instance": broker_instance,
                "asset_classes": asset_classes,
                "operation_types": operation_types,
                "capabilities": capabilities or {},
                "registered_at": time.time()
            }
            
            # Register with advisor
            self.advisor.register_broker_capability(
                broker_id=broker_id,
                asset_classes=asset_classes,
                operation_types=operation_types,
                rate_limits=capabilities.get("rate_limits") if capabilities else None,
                trading_limits=capabilities.get("trading_limits") if capabilities else None
            )
            
            logger.info(f"Registered broker {broker_id} with intelligence engine")
    
    def map_asset_to_class(self, asset: str, asset_class: str):
        """
        Map individual asset to its asset class
        
        Args:
            asset: Asset symbol or identifier
            asset_class: Asset class (e.g., 'equities', 'forex')
        """
        with self.lock:
            self.asset_class_map[asset] = asset_class
    
    def get_asset_class(self, asset: str) -> str:
        """
        Get asset class for an asset
        
        Args:
            asset: Asset symbol or identifier
            
        Returns:
            Asset class name
        """
        with self.lock:
            return self.asset_class_map.get(asset, "unknown")
    
    def start_monitoring(self):
        """Start intelligence monitoring thread"""
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            logger.warning("Intelligence monitoring thread already running")
            return
        
        self.monitor_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="BrokerIntelligenceMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Started intelligence monitoring thread")
    
    def stop_monitoring(self):
        """Stop intelligence monitoring thread"""
        self.monitor_active = False
        if self.monitor_thread is not None:
            try:
                self.monitor_thread.join(timeout=2.0)
            except Exception:
                pass
            self.monitor_thread = None
        logger.info("Stopped intelligence monitoring thread")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitor_active:
            try:
                # Analyze broker metrics
                self._analyze_broker_health()
                
                # Check for risk thresholds
                self._check_risk_thresholds()
                
                # Update recommendations for common asset classes and operations
                self._update_recommendations()
                
                # Generate broker health report
                health_report = self._generate_health_report()
                
                # Emit intelligence update event (for orchestrator awareness)
                self.event_system.emit(
                    event_type=EventType.BROKER_INTELLIGENCE_UPDATE,
                    data={
                        "health_status": self.health_status,
                        "health_report": health_report,
                        "recommendations": {
                            key: rec.to_dict() for key, rec in self.current_recommendations.items()
                        },
                        "drawdown_alerts": self.drawdown_alerts,
                        "timestamp": time.time()
                    }
                )
                
            except Exception as e:
                logger.error(f"Error in intelligence monitoring loop: {str(e)}")
            
            # Wait for next interval
            time.sleep(self.monitor_interval)
    
    def _analyze_broker_health(self):
        """Analyze broker health metrics"""
        for broker_id in self.registered_brokers.keys():
            try:
                # Get recent metrics
                metrics = self.metrics_manager.get_broker_metrics(broker_id)
                
                # Check for potential circuit breaker conditions
                if self._should_trip_circuit_breaker(broker_id, metrics):
                    reason = self._get_circuit_breaker_reason(broker_id, metrics)
                    self.advisor.trip_circuit_breaker(broker_id, reason)
                    
                    # Alert orchestrator via event
                    self.event_system.emit(
                        event_type=EventType.BROKER_CIRCUIT_BREAKER,
                        data={
                            "broker_id": broker_id,
                            "reason": reason,
                            "timestamp": time.time()
                        }
                    )
            except Exception as e:
                logger.error(f"Error analyzing health for broker {broker_id}: {str(e)}")
    
    def _should_trip_circuit_breaker(self, broker_id: str, metrics: Dict[str, Any]) -> bool:
        """Determine if circuit breaker should be tripped"""
        # Already tripped
        if self.advisor.is_circuit_breaker_active(broker_id):
            return False
            
        # Check error rate
        error_rate = metrics["reliability"].get("error_rate", 0)
        error_count = metrics["reliability"].get("errors", 0)
        availability = metrics["reliability"].get("availability", 100)
        
        thresholds = self.advisor.circuit_breaker_thresholds
        
        # Circuit breaker conditions
        if (error_rate >= thresholds.get("error_rate", 0.3) or
            error_count >= thresholds.get("error_count", 5) or
            availability <= thresholds.get("availability_min", 90.0)):
            return True
            
        return False
    
    def _get_circuit_breaker_reason(self, broker_id: str, metrics: Dict[str, Any]) -> str:
        """Get reason for circuit breaker activation"""
        error_rate = metrics["reliability"].get("error_rate", 0)
        error_count = metrics["reliability"].get("errors", 0)
        availability = metrics["reliability"].get("availability", 100)
        
        thresholds = self.advisor.circuit_breaker_thresholds
        
        reasons = []
        
        if error_rate >= thresholds.get("error_rate", 0.3):
            reasons.append(f"High error rate: {error_rate*100:.1f}%")
            
        if error_count >= thresholds.get("error_count", 5):
            reasons.append(f"Too many errors: {error_count}")
            
        if availability <= thresholds.get("availability_min", 90.0):
            reasons.append(f"Low availability: {availability:.1f}%")
            
        return "; ".join(reasons) if reasons else "Unknown reason"
    
    def _check_risk_thresholds(self):
        """Check for portfolio risk thresholds"""
        # This would integrate with position/risk tracking
        # For now, we'll implement a stub that could be expanded
        pass
    
    def _update_recommendations(self):
        """Update broker recommendations for common asset classes and operations"""
        # Common asset classes
        asset_classes = ["equities", "forex", "futures", "options", "crypto"]
        
        # Common operation types
        operations = ["order", "quote", "data"]
        
        for asset_class in asset_classes:
            for operation in operations:
                try:
                    # Only generate recommendation if we have brokers supporting this
                    have_support = False
                    for broker_data in self.registered_brokers.values():
                        if (asset_class in broker_data["asset_classes"] and 
                            operation in broker_data["operation_types"]):
                            have_support = True
                            break
                            
                    if not have_support:
                        continue
                        
                    # Get advice from advisor
                    advice = self.advisor.get_selection_advice(
                        asset_class=asset_class,
                        operation_type=operation
                    )
                    
                    # Store recommendation
                    key = f"{asset_class}_{operation}"
                    self.current_recommendations[key] = advice
                    
                except Exception as e:
                    logger.error(f"Error updating recommendation for {asset_class}/{operation}: {str(e)}")
    
    def _generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive broker health report"""
        report = {
            "brokers": {},
            "overall_status": self.health_status,
            "timestamp": time.time()
        }
        
        for broker_id, broker_data in self.registered_brokers.items():
            try:
                # Get broker metrics
                metrics = self.metrics_manager.get_broker_metrics(broker_id)
                
                # Calculate performance scores
                performance = self.advisor.get_broker_performance_score(
                    broker_id=broker_id,
                    asset_class="overall",
                    operation_type="overall"
                )
                
                # Check circuit breaker
                circuit_breaker_active = self.advisor.is_circuit_breaker_active(broker_id)
                
                # Add to report
                report["brokers"][broker_id] = {
                    "performance_score": performance["overall_score"],
                    "factor_scores": performance["factor_scores"],
                    "metrics": metrics,
                    "circuit_breaker_active": circuit_breaker_active,
                    "supports": {
                        "asset_classes": broker_data["asset_classes"],
                        "operations": broker_data["operation_types"]
                    }
                }
            except Exception as e:
                logger.error(f"Error generating health report for broker {broker_id}: {str(e)}")
                report["brokers"][broker_id] = {
                    "error": str(e),
                    "status": "ERROR"
                }
        
        return report
    
    def get_broker_recommendation(
        self,
        asset: str,
        operation_type: str,
        current_broker_id: Optional[str] = None
    ) -> BrokerSelectionAdvice:
        """
        Get broker recommendation for specific asset and operation
        
        Args:
            asset: Asset symbol or identifier
            operation_type: Operation type
            current_broker_id: Currently selected broker
            
        Returns:
            Broker selection advice
        """
        # Get asset class
        asset_class = self.get_asset_class(asset)
        
        # Generate recommendation
        return self.advisor.get_selection_advice(
            asset_class=asset_class,
            operation_type=operation_type,
            current_broker_id=current_broker_id
        )
    
    def track_drawdown_alert(
        self,
        strategy_id: str,
        drawdown_pct: float,
        timestamp: Optional[float] = None
    ):
        """
        Track drawdown alert for risk management
        
        Args:
            strategy_id: Strategy identifier
            drawdown_pct: Drawdown percentage
            timestamp: Event timestamp
        """
        with self.lock:
            # Add or update drawdown alert
            self.drawdown_alerts[strategy_id] = {
                "drawdown_pct": drawdown_pct,
                "timestamp": timestamp or time.time(),
                "level": self._get_drawdown_level(drawdown_pct)
            }
            
            # Update overall health status based on drawdown levels
            self._update_health_status()
    
    def _get_drawdown_level(self, drawdown_pct: float) -> str:
        """Get severity level for a drawdown percentage"""
        if drawdown_pct >= self.risk_thresholds.get("drawdown_critical", 4.0):
            return "CRITICAL"
        elif drawdown_pct >= self.risk_thresholds.get("drawdown_warning", 3.0):
            return "WARNING"
        else:
            return "NORMAL"
    
    def _update_health_status(self):
        """Update overall health status based on current alerts"""
        # Start with NORMAL status
        new_status = "NORMAL"
        
        # Check for any warnings
        for alert in self.drawdown_alerts.values():
            if alert["level"] == "WARNING" and new_status == "NORMAL":
                new_status = "CAUTION"
            elif alert["level"] == "CRITICAL":
                new_status = "CRITICAL"
                break
                
        # Check circuit breakers
        for broker_id in self.registered_brokers.keys():
            if self.advisor.is_circuit_breaker_active(broker_id):
                # At least CAUTION if any circuit breaker is active
                if new_status == "NORMAL":
                    new_status = "CAUTION"
        
        # Update status
        self.health_status = new_status
        
        # Emit status change event if needed
        if new_status != "NORMAL":
            self.event_system.emit(
                event_type=EventType.BROKER_HEALTH_STATUS_CHANGE,
                data={
                    "status": new_status,
                    "drawdown_alerts": self.drawdown_alerts,
                    "timestamp": time.time()
                }
            )
