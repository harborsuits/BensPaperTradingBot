#!/usr/bin/env python3
"""
Broker Intelligence Advisor

Provides intelligent broker selection recommendations based on performance metrics
while preserving the orchestrator's ultimate decision authority.
"""

import os
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from enum import Enum

from trading_bot.brokers.metrics.base import MetricType, MetricOperation, MetricPeriod
from trading_bot.brokers.metrics.manager import BrokerMetricsManager

# Configure logging
logger = logging.getLogger(__name__)

class BrokerSelectionFactor(Enum):
    """Factors that influence broker selection"""
    LATENCY = "latency"  # Response time and API performance
    RELIABILITY = "reliability"  # Connection stability and error rate
    EXECUTION_QUALITY = "execution_quality"  # Slippage and fill rates
    COST = "cost"  # Commissions and fees
    
    # Special factors
    CIRCUIT_BREAKER = "circuit_breaker"  # Broker circuit breaker status
    CAPACITY = "capacity"  # Broker capacity/rate limits
    MARKET_CONDITIONS = "market_conditions"  # Current market conditions

class BrokerSelectionAdvice:
    """
    Recommendation object for broker selection
    
    Contains multiple levels of recommendations while preserving
    the orchestrator's decision-making authority.
    """
    
    def __init__(
        self,
        asset_class: str,
        operation_type: str,
        primary_broker_id: Optional[str] = None,
        backup_broker_ids: Optional[List[str]] = None,
        blacklisted_broker_ids: Optional[List[str]] = None,
        recommendation_factors: Optional[Dict[str, Dict[str, float]]] = None,
        priority_scores: Optional[Dict[str, float]] = None,
        is_failover_recommended: bool = False,
        advisory_notes: Optional[List[str]] = None
    ):
        """
        Initialize broker selection advice
        
        Args:
            asset_class: Asset class for this advice
            operation_type: Operation type (e.g., order, quote)
            primary_broker_id: Recommended primary broker
            backup_broker_ids: Recommended backup brokers
            blacklisted_broker_ids: Brokers to avoid
            recommendation_factors: Detailed factors behind recommendations
            priority_scores: Overall score for each broker
            is_failover_recommended: Whether failover is recommended
            advisory_notes: Additional notes and context
        """
        self.asset_class = asset_class
        self.operation_type = operation_type
        self.primary_broker_id = primary_broker_id
        self.backup_broker_ids = backup_broker_ids or []
        self.blacklisted_broker_ids = blacklisted_broker_ids or []
        self.recommendation_factors = recommendation_factors or {}
        self.priority_scores = priority_scores or {}
        self.is_failover_recommended = is_failover_recommended
        self.advisory_notes = advisory_notes or []
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "asset_class": self.asset_class,
            "operation_type": self.operation_type,
            "primary_broker_id": self.primary_broker_id,
            "backup_broker_ids": self.backup_broker_ids,
            "blacklisted_broker_ids": self.blacklisted_broker_ids,
            "recommendation_factors": self.recommendation_factors,
            "priority_scores": self.priority_scores,
            "is_failover_recommended": self.is_failover_recommended,
            "advisory_notes": self.advisory_notes,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BrokerSelectionAdvice':
        """Create from dictionary"""
        return cls(
            asset_class=data["asset_class"],
            operation_type=data["operation_type"],
            primary_broker_id=data["primary_broker_id"],
            backup_broker_ids=data["backup_broker_ids"],
            blacklisted_broker_ids=data["blacklisted_broker_ids"],
            recommendation_factors=data["recommendation_factors"],
            priority_scores=data["priority_scores"],
            is_failover_recommended=data["is_failover_recommended"],
            advisory_notes=data["advisory_notes"]
        )
    
    def __str__(self) -> str:
        """String representation"""
        status = "FAILOVER RECOMMENDED" if self.is_failover_recommended else "NORMAL"
        
        return (
            f"BrokerSelectionAdvice [{status}]\n"
            f"Asset Class: {self.asset_class}\n"
            f"Operation: {self.operation_type}\n"
            f"Primary: {self.primary_broker_id}\n"
            f"Backup: {', '.join(self.backup_broker_ids)}\n"
            f"Blacklisted: {', '.join(self.blacklisted_broker_ids)}\n"
            f"Notes: {'; '.join(self.advisory_notes)}"
        )

class BrokerAdvisor:
    """
    Intelligent broker selection advisor
    
    Analyzes broker performance metrics to provide informed recommendations
    to the orchestrator without replacing its decision-making authority.
    """
    
    def __init__(
        self,
        metrics_manager: BrokerMetricsManager,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize broker advisor
        
        Args:
            metrics_manager: Broker metrics manager
            config: Configuration dictionary
        """
        self.metrics_manager = metrics_manager
        self.config = config or {}
        
        # Default weights for different factors
        self.factor_weights = self.config.get("factor_weights", {
            BrokerSelectionFactor.LATENCY.value: 0.20,
            BrokerSelectionFactor.RELIABILITY.value: 0.40,
            BrokerSelectionFactor.EXECUTION_QUALITY.value: 0.25,
            BrokerSelectionFactor.COST.value: 0.15
        })
        
        # Circuit breaker settings
        self.circuit_breaker_thresholds = self.config.get("circuit_breaker_thresholds", {
            "error_count": 5,  # Number of errors before circuit breaker trips
            "error_rate": 0.3,  # Error rate (0-1) before circuit breaker trips
            "availability_min": 90.0,  # Minimum availability percentage
            "reset_after_seconds": 300  # Time before auto-reset attempt (5 min)
        })
        
        # Asset class specific weights
        self.asset_class_weights = self.config.get("asset_class_weights", {})
        
        # Circuit breaker state
        self.circuit_breakers = {}
        
        # Broker capabilities and limits
        self.broker_capabilities = {}
        
        # Recent recommendations
        self.recent_recommendations = {}
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        logger.info("Initialized BrokerAdvisor")
    
    def register_broker_capability(
        self,
        broker_id: str,
        asset_classes: List[str],
        operation_types: List[str],
        rate_limits: Optional[Dict[str, Any]] = None,
        trading_limits: Optional[Dict[str, Any]] = None
    ):
        """
        Register broker capabilities and limits
        
        Args:
            broker_id: Broker identifier
            asset_classes: Supported asset classes
            operation_types: Supported operation types
            rate_limits: API rate limits
            trading_limits: Trading limits
        """
        with self.lock:
            self.broker_capabilities[broker_id] = {
                "asset_classes": asset_classes,
                "operation_types": operation_types,
                "rate_limits": rate_limits or {},
                "trading_limits": trading_limits or {}
            }
            
            logger.info(f"Registered capabilities for broker {broker_id}")
    
    def trip_circuit_breaker(
        self,
        broker_id: str,
        reason: str,
        reset_after_seconds: Optional[int] = None
    ):
        """
        Trip circuit breaker for a broker
        
        Args:
            broker_id: Broker identifier
            reason: Reason for circuit breaker activation
            reset_after_seconds: Time before auto-reset
        """
        with self.lock:
            # Use default reset time if none provided
            if reset_after_seconds is None:
                reset_after_seconds = self.circuit_breaker_thresholds.get("reset_after_seconds", 300)
            
            # Record circuit breaker state
            self.circuit_breakers[broker_id] = {
                "active": True,
                "reason": reason,
                "tripped_at": time.time(),
                "reset_after": reset_after_seconds,
                "reset_time": time.time() + reset_after_seconds
            }
            
            logger.warning(f"Circuit breaker tripped for broker {broker_id}: {reason}")
    
    def reset_circuit_breaker(self, broker_id: str):
        """
        Reset circuit breaker for a broker
        
        Args:
            broker_id: Broker identifier
        """
        with self.lock:
            if broker_id in self.circuit_breakers:
                breaker = self.circuit_breakers[broker_id]
                
                # Only modify if currently active
                if breaker.get("active", False):
                    breaker["active"] = False
                    breaker["reset_at"] = time.time()
                    logger.info(f"Circuit breaker reset for broker {broker_id}")
    
    def is_circuit_breaker_active(self, broker_id: str) -> bool:
        """
        Check if circuit breaker is active for a broker
        
        Args:
            broker_id: Broker identifier
            
        Returns:
            True if circuit breaker is active
        """
        with self.lock:
            if broker_id not in self.circuit_breakers:
                return False
            
            breaker = self.circuit_breakers[broker_id]
            
            # Not active
            if not breaker.get("active", False):
                return False
            
            # Check if it's time for auto-reset
            if time.time() >= breaker.get("reset_time", 0):
                self.reset_circuit_breaker(broker_id)
                return False
            
            return True
    
    def get_broker_performance_score(
        self,
        broker_id: str,
        asset_class: str,
        operation_type: str
    ) -> Dict[str, Any]:
        """
        Calculate performance score for a broker
        
        Args:
            broker_id: Broker identifier
            asset_class: Asset class
            operation_type: Operation type
            
        Returns:
            Dictionary with scores and factors
        """
        # Get broker metrics
        metrics = self.metrics_manager.get_broker_metrics(broker_id, MetricPeriod.HOUR)
        
        # Calculate factor scores (0-100 scale where higher is better)
        latency_score = self._calculate_latency_score(metrics)
        reliability_score = self._calculate_reliability_score(metrics)
        execution_score = self._calculate_execution_score(metrics)
        cost_score = self._calculate_cost_score(metrics)
        
        # Get factor weights (potentially customized for asset class)
        weights = self._get_factor_weights(asset_class)
        
        # Calculate overall score
        overall_score = (
            latency_score * weights[BrokerSelectionFactor.LATENCY.value] +
            reliability_score * weights[BrokerSelectionFactor.RELIABILITY.value] +
            execution_score * weights[BrokerSelectionFactor.EXECUTION_QUALITY.value] +
            cost_score * weights[BrokerSelectionFactor.COST.value]
        )
        
        # Circuit breaker penalty (if active)
        if self.is_circuit_breaker_active(broker_id):
            overall_score *= 0.1  # 90% penalty
        
        return {
            "broker_id": broker_id,
            "overall_score": overall_score,
            "factor_scores": {
                BrokerSelectionFactor.LATENCY.value: latency_score,
                BrokerSelectionFactor.RELIABILITY.value: reliability_score,
                BrokerSelectionFactor.EXECUTION_QUALITY.value: execution_score,
                BrokerSelectionFactor.COST.value: cost_score,
                BrokerSelectionFactor.CIRCUIT_BREAKER.value: 0 if self.is_circuit_breaker_active(broker_id) else 100
            }
        }
    
    def _calculate_latency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate latency score from metrics"""
        # From metrics, extract mean latency
        mean_latency = metrics["latency"].get("mean_ms", 1000)
        
        # Convert to score (lower latency = higher score)
        # Scale: 0ms=100, 1000ms=0
        score = max(0, min(100, 100 - (mean_latency / 10)))
        return score
    
    def _calculate_reliability_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate reliability score from metrics"""
        # Get availability percentage
        availability = metrics["reliability"].get("availability", 0)
        
        # Get error count
        error_count = metrics["reliability"].get("errors", 0)
        
        # Calculate reliability score
        # 75% from availability, 25% from error count
        availability_score = availability  # Already 0-100
        error_score = max(0, min(100, 100 - (error_count * 10)))
        
        return (availability_score * 0.75) + (error_score * 0.25)
    
    def _calculate_execution_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate execution quality score from metrics"""
        # Get average slippage
        avg_slippage = abs(metrics["execution_quality"].get("avg_slippage_pct", 0))
        
        # Convert to score (lower slippage = higher score)
        # Scale: 0%=100, 1%=0
        slippage_score = max(0, min(100, 100 - (avg_slippage * 100)))
        
        return slippage_score
    
    def _calculate_cost_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate cost score from metrics"""
        # Get average commission
        avg_commission = metrics["costs"].get("avg_commission", 10)
        
        # Convert to score (lower cost = higher score)
        # Scale: $0=100, $10=0
        cost_score = max(0, min(100, 100 - (avg_commission * 10)))
        
        return cost_score
    
    def _get_factor_weights(self, asset_class: str) -> Dict[str, float]:
        """Get factor weights for an asset class"""
        # Use asset-specific weights if available, otherwise defaults
        if asset_class in self.asset_class_weights:
            return self.asset_class_weights[asset_class]
        return self.factor_weights
    
    def get_selection_advice(
        self,
        asset_class: str,
        operation_type: str,
        current_broker_id: Optional[str] = None
    ) -> BrokerSelectionAdvice:
        """
        Get broker selection advice
        
        Args:
            asset_class: Asset class
            operation_type: Operation type
            current_broker_id: Currently selected broker
            
        Returns:
            Broker selection advice
        """
        with self.lock:
            # Get all brokers that support this asset class and operation
            eligible_brokers = []
            blacklisted_brokers = []
            
            for broker_id, capabilities in self.broker_capabilities.items():
                if (asset_class in capabilities["asset_classes"] and
                    operation_type in capabilities["operation_types"]):
                    
                    # Check if broker is blacklisted (circuit breaker)
                    if self.is_circuit_breaker_active(broker_id):
                        blacklisted_brokers.append(broker_id)
                    else:
                        eligible_brokers.append(broker_id)
            
            if not eligible_brokers:
                logger.warning(f"No eligible brokers for {asset_class}/{operation_type}")
                
                # Special case: if all brokers are blacklisted, pick the least bad one
                if blacklisted_brokers:
                    advice = BrokerSelectionAdvice(
                        asset_class=asset_class,
                        operation_type=operation_type,
                        primary_broker_id=blacklisted_brokers[0],
                        blacklisted_broker_ids=blacklisted_brokers[1:],
                        is_failover_recommended=False,
                        advisory_notes=[
                            f"All brokers have active circuit breakers",
                            f"Selected least bad option {blacklisted_brokers[0]}"
                        ]
                    )
                    return advice
                else:
                    # No brokers at all
                    advice = BrokerSelectionAdvice(
                        asset_class=asset_class,
                        operation_type=operation_type,
                        is_failover_recommended=False,
                        advisory_notes=[f"No brokers support {asset_class}/{operation_type}"]
                    )
                    return advice
            
            # Score all eligible brokers
            broker_scores = {}
            factor_details = {}
            
            for broker_id in eligible_brokers:
                try:
                    performance = self.get_broker_performance_score(broker_id, asset_class, operation_type)
                    broker_scores[broker_id] = performance["overall_score"]
                    factor_details[broker_id] = performance["factor_scores"]
                except Exception as e:
                    logger.error(f"Error scoring broker {broker_id}: {str(e)}")
                    # Assign a low score on error
                    broker_scores[broker_id] = 10.0
                    factor_details[broker_id] = {factor.value: 10.0 for factor in BrokerSelectionFactor}
            
            # Sort brokers by score (descending)
            sorted_brokers = sorted(broker_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Determine primary and backup brokers
            primary_broker_id = sorted_brokers[0][0] if sorted_brokers else None
            backup_broker_ids = [b[0] for b in sorted_brokers[1:]]
            
            # Determine if failover is recommended
            failover_recommended = False
            advisory_notes = []
            
            if current_broker_id and current_broker_id != primary_broker_id:
                # Current broker is not the best option
                current_score = broker_scores.get(current_broker_id, 0)
                best_score = broker_scores.get(primary_broker_id, 100)
                
                # Only recommend failover if there's a significant difference
                score_difference = best_score - current_score
                
                if score_difference > 20:  # At least 20% better performance
                    failover_recommended = True
                    advisory_notes.append(
                        f"Recommend failover from {current_broker_id} to {primary_broker_id}: "
                        f"{score_difference:.1f}% better performance"
                    )
                else:
                    advisory_notes.append(
                        f"Current broker {current_broker_id} is acceptable: "
                        f"only {score_difference:.1f}% worse than best option"
                    )
            
            # Create and return advice
            advice = BrokerSelectionAdvice(
                asset_class=asset_class,
                operation_type=operation_type,
                primary_broker_id=primary_broker_id,
                backup_broker_ids=backup_broker_ids,
                blacklisted_broker_ids=blacklisted_brokers,
                recommendation_factors=factor_details,
                priority_scores=broker_scores,
                is_failover_recommended=failover_recommended,
                advisory_notes=advisory_notes
            )
            
            # Store in recent recommendations
            key = f"{asset_class}_{operation_type}"
            self.recent_recommendations[key] = advice
            
            return advice
