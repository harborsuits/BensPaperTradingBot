#!/usr/bin/env python3
"""
Strategy Deployment Pipeline

This module creates a standardized workflow for deploying strategies from
the autonomous engine to actual execution with proper risk controls.
It builds upon our successful event-driven architecture.
"""

import logging
import os
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import time
import uuid

# Import risk components
from trading_bot.risk.risk_manager import RiskManager, RiskLevel, StopLossType
from trading_bot.autonomous.risk_integration import AutonomousRiskManager, get_autonomous_risk_manager

# Import autonomous components
from trading_bot.autonomous.autonomous_engine import AutonomousEngine, StrategyCandidate

# Import event system
from trading_bot.event_system import EventBus, Event, EventType

logger = logging.getLogger(__name__)

class DeploymentStatus:
    """Status codes for strategy deployment"""
    PENDING = "pending"
    DEPLOYING = "deploying"  
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    FAILED = "failed"


class StrategyDeployment:
    """
    Represents a deployed strategy with its configuration and current status.
    """
    
    def __init__(self, 
                 strategy_id: str,
                 deployment_id: str,
                 status: str = DeploymentStatus.PENDING,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize a strategy deployment.
        
        Args:
            strategy_id: Original strategy ID
            deployment_id: Unique deployment ID
            status: Deployment status
            config: Deployment configuration
        """
        self.strategy_id = strategy_id
        self.deployment_id = deployment_id
        self.status = status
        self.config = config or {}
        
        # Risk parameters
        self.risk_params = {
            "allocation_percentage": config.get("allocation_percentage", 5.0),
            "risk_level": config.get("risk_level", RiskLevel.MEDIUM),
            "stop_loss_type": config.get("stop_loss_type", StopLossType.VOLATILITY)
        }
        
        # Tracking
        self.deploy_time = datetime.now().isoformat()
        self.last_update_time = self.deploy_time
        self.metadata = config.get("metadata", {})
        
        # Performance
        self.performance = {
            "trades": 0,
            "profit_loss": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "last_updated": self.deploy_time
        }
        
        # Status history
        self.status_history = [{
            "status": status,
            "timestamp": self.deploy_time,
            "reason": "Initial deployment"
        }]
    
    def update_status(self, status: str, reason: str = "") -> None:
        """
        Update deployment status.
        
        Args:
            status: New status
            reason: Reason for status change
        """
        self.status = status
        self.last_update_time = datetime.now().isoformat()
        
        # Add to history
        self.status_history.append({
            "status": status,
            "timestamp": self.last_update_time,
            "reason": reason
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            "strategy_id": self.strategy_id,
            "deployment_id": self.deployment_id,
            "status": self.status,
            "config": self.config,
            "risk_params": self.risk_params,
            "deploy_time": self.deploy_time,
            "last_update_time": self.last_update_time,
            "metadata": self.metadata,
            "performance": self.performance,
            "status_history": self.status_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyDeployment':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary data
        
        Returns:
            StrategyDeployment instance
        """
        deployment = cls(
            strategy_id=data.get("strategy_id", ""),
            deployment_id=data.get("deployment_id", ""),
            status=data.get("status", DeploymentStatus.PENDING),
            config=data.get("config", {})
        )
        
        # Restore fields
        deployment.risk_params = data.get("risk_params", deployment.risk_params)
        deployment.deploy_time = data.get("deploy_time", deployment.deploy_time)
        deployment.last_update_time = data.get("last_update_time", deployment.last_update_time)
        deployment.metadata = data.get("metadata", deployment.metadata)
        deployment.performance = data.get("performance", deployment.performance)
        deployment.status_history = data.get("status_history", deployment.status_history)
        
        return deployment


class StrategyDeploymentPipeline:
    """
    Standardized workflow for deploying strategies from the autonomous engine
    to actual execution with proper risk controls.
    """
    
    def __init__(self, 
                 event_bus: Optional[EventBus] = None,
                 data_dir: Optional[str] = None):
        """
        Initialize the deployment pipeline.
        
        Args:
            event_bus: Event bus for communication
            data_dir: Directory for deployment data
        """
        self.event_bus = event_bus or EventBus()
        
        # Connect to risk manager
        self.risk_manager = get_autonomous_risk_manager()
        
        # Internal storage
        self.deployments = {}  # deployment_id -> StrategyDeployment
        self.strategy_to_deployment = {}  # strategy_id -> deployment_id
        
        # Configure data directory
        self.data_dir = data_dir or os.path.join(
            os.path.expanduser("~"),
            "trading_data",
            "strategy_deployments"
        )
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Register event handlers
        self._register_event_handlers()
        
        # Load existing deployments
        self._load_deployments()
        
        logger.info("Strategy Deployment Pipeline initialized")
    
    def _register_event_handlers(self) -> None:
        """Register event handlers."""
        # Strategy events
        self.event_bus.register(EventType.STRATEGY_OPTIMISED, self._handle_strategy_optimised)
        self.event_bus.register(EventType.STRATEGY_DEPLOYED_WITH_RISK, self._handle_strategy_deployed)
        self.event_bus.register(EventType.STRATEGY_PAUSED, self._handle_strategy_paused)
        self.event_bus.register(EventType.STRATEGY_RESUMED, self._handle_strategy_resumed)
        
        # Trade events
        self.event_bus.register(EventType.TRADE_EXECUTED, self._handle_trade_executed)
        self.event_bus.register(EventType.POSITION_CLOSED, self._handle_position_closed)
        
        # Risk events
        self.event_bus.register(EventType.CIRCUIT_BREAKER_TRIGGERED, self._handle_circuit_breaker)
        
        logger.info("Registered deployment pipeline event handlers")
    
    def deploy_strategy(self, 
                        strategy_id: str,
                        allocation_percentage: float = 5.0,
                        risk_level: Optional[RiskLevel] = None,
                        stop_loss_type: Optional[StopLossType] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        Deploy a strategy with the standardized workflow.
        
        Args:
            strategy_id: Strategy ID to deploy
            allocation_percentage: Percentage of capital to allocate
            risk_level: Risk level for the strategy
            stop_loss_type: Stop loss type for the strategy
            metadata: Additional metadata for the deployment
            
        Returns:
            (success, deployment_id or error message)
        """
        logger.info(f"Deploying strategy {strategy_id}")
        
        # Check if strategy already deployed
        if strategy_id in self.strategy_to_deployment:
            existing_deployment_id = self.strategy_to_deployment[strategy_id]
            existing_deployment = self.deployments.get(existing_deployment_id)
            
            if existing_deployment and existing_deployment.status in [
                DeploymentStatus.ACTIVE, DeploymentStatus.DEPLOYING, DeploymentStatus.PENDING
            ]:
                return False, f"Strategy {strategy_id} already deployed with ID {existing_deployment_id}"
        
        # Generate deployment ID
        deployment_id = f"deploy_{strategy_id}_{uuid.uuid4().hex[:8]}"
        
        # Create deployment config
        config = {
            "allocation_percentage": allocation_percentage,
            "risk_level": risk_level or RiskLevel.MEDIUM,
            "stop_loss_type": stop_loss_type or StopLossType.VOLATILITY,
            "metadata": metadata or {}
        }
        
        # Create deployment record
        deployment = StrategyDeployment(
            strategy_id=strategy_id,
            deployment_id=deployment_id,
            status=DeploymentStatus.DEPLOYING,
            config=config
        )
        
        # Store deployment
        self.deployments[deployment_id] = deployment
        self.strategy_to_deployment[strategy_id] = deployment_id
        
        try:
            # Deploy through risk manager
            success = self.risk_manager.deploy_strategy(
                strategy_id=strategy_id,
                allocation_percentage=allocation_percentage,
                risk_level=config["risk_level"],
                stop_loss_type=config["stop_loss_type"]
            )
            
            if success:
                # Update deployment status
                deployment.update_status(
                    DeploymentStatus.ACTIVE,
                    "Successfully deployed with risk controls"
                )
                
                # Emit deployment event
                self._emit_deployment_event(
                    event_type=EventType.STRATEGY_DEPLOYMENT_COMPLETED,
                    deployment=deployment,
                    details={"success": True}
                )
                
                # Save deployments
                self._save_deployments()
                
                logger.info(f"Strategy {strategy_id} deployed with ID {deployment_id}")
                return True, deployment_id
            else:
                # Update deployment status
                deployment.update_status(
                    DeploymentStatus.FAILED,
                    "Failed to deploy through risk manager"
                )
                
                # Emit failure event
                self._emit_deployment_event(
                    event_type=EventType.STRATEGY_DEPLOYMENT_FAILED,
                    deployment=deployment,
                    details={"reason": "Risk manager deployment failed"}
                )
                
                # Save deployments
                self._save_deployments()
                
                logger.error(f"Failed to deploy strategy {strategy_id}")
                return False, "Risk manager deployment failed"
                
        except Exception as e:
            # Update deployment status
            deployment.update_status(
                DeploymentStatus.FAILED,
                f"Deployment error: {str(e)}"
            )
            
            # Emit failure event
            self._emit_deployment_event(
                event_type=EventType.STRATEGY_DEPLOYMENT_FAILED,
                deployment=deployment,
                details={"error": str(e)}
            )
            
            # Save deployments
            self._save_deployments()
            
            logger.error(f"Error deploying strategy {strategy_id}: {e}")
            return False, f"Deployment error: {str(e)}"
    
    def pause_deployment(self, deployment_id: str, reason: str = "") -> bool:
        """
        Pause a deployed strategy.
        
        Args:
            deployment_id: Deployment ID
            reason: Reason for pausing
            
        Returns:
            bool: Success flag
        """
        if deployment_id not in self.deployments:
            logger.error(f"Deployment {deployment_id} not found")
            return False
        
        deployment = self.deployments[deployment_id]
        strategy_id = deployment.strategy_id
        
        try:
            # Pause through risk manager
            success = self.risk_manager.pause_strategy(
                strategy_id=strategy_id,
                reason=reason or "Manual pause"
            )
            
            if success:
                # Update deployment status
                deployment.update_status(
                    DeploymentStatus.PAUSED,
                    reason or "Manual pause"
                )
                
                # Emit event
                self._emit_deployment_event(
                    event_type=EventType.STRATEGY_DEPLOYMENT_PAUSED,
                    deployment=deployment,
                    details={"reason": reason}
                )
                
                # Save deployments
                self._save_deployments()
                
                logger.info(f"Deployment {deployment_id} paused: {reason}")
                return True
            else:
                logger.error(f"Failed to pause deployment {deployment_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error pausing deployment {deployment_id}: {e}")
            return False
    
    def resume_deployment(self, deployment_id: str) -> bool:
        """
        Resume a paused deployment.
        
        Args:
            deployment_id: Deployment ID
            
        Returns:
            bool: Success flag
        """
        if deployment_id not in self.deployments:
            logger.error(f"Deployment {deployment_id} not found")
            return False
        
        deployment = self.deployments[deployment_id]
        strategy_id = deployment.strategy_id
        
        if deployment.status != DeploymentStatus.PAUSED:
            logger.error(f"Deployment {deployment_id} is not paused")
            return False
        
        try:
            # Resume through risk manager
            success = self.risk_manager.resume_strategy(strategy_id=strategy_id)
            
            if success:
                # Update deployment status
                deployment.update_status(
                    DeploymentStatus.ACTIVE,
                    "Resumed deployment"
                )
                
                # Emit event
                self._emit_deployment_event(
                    event_type=EventType.STRATEGY_DEPLOYMENT_RESUMED,
                    deployment=deployment,
                    details={}
                )
                
                # Save deployments
                self._save_deployments()
                
                logger.info(f"Deployment {deployment_id} resumed")
                return True
            else:
                logger.error(f"Failed to resume deployment {deployment_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error resuming deployment {deployment_id}: {e}")
            return False
    
    def stop_deployment(self, deployment_id: str, reason: str = "") -> bool:
        """
        Stop a deployment permanently.
        
        Args:
            deployment_id: Deployment ID
            reason: Reason for stopping
            
        Returns:
            bool: Success flag
        """
        if deployment_id not in self.deployments:
            logger.error(f"Deployment {deployment_id} not found")
            return False
        
        deployment = self.deployments[deployment_id]
        strategy_id = deployment.strategy_id
        
        try:
            # First pause the strategy
            if deployment.status == DeploymentStatus.ACTIVE:
                self.risk_manager.pause_strategy(
                    strategy_id=strategy_id,
                    reason=reason or "Stopping deployment"
                )
            
            # Update deployment status
            deployment.update_status(
                DeploymentStatus.STOPPED,
                reason or "Manually stopped"
            )
            
            # Emit event
            self._emit_deployment_event(
                event_type=EventType.STRATEGY_DEPLOYMENT_STOPPED,
                deployment=deployment,
                details={"reason": reason}
            )
            
            # Save deployments
            self._save_deployments()
            
            logger.info(f"Deployment {deployment_id} stopped: {reason}")
            return True
                
        except Exception as e:
            logger.error(f"Error stopping deployment {deployment_id}: {e}")
            return False
    
    def get_deployments(self, 
                        status: Optional[str] = None,
                        strategy_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get deployments with optional filtering.
        
        Args:
            status: Filter by status
            strategy_id: Filter by strategy ID
            
        Returns:
            List of deployment dictionaries
        """
        filtered = []
        
        for deployment in self.deployments.values():
            # Apply filters
            if status and deployment.status != status:
                continue
                
            if strategy_id and deployment.strategy_id != strategy_id:
                continue
                
            filtered.append(deployment.to_dict())
        
        return filtered
    
    def get_deployment(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific deployment.
        
        Args:
            deployment_id: Deployment ID
            
        Returns:
            Deployment dictionary or None
        """
        deployment = self.deployments.get(deployment_id)
        return deployment.to_dict() if deployment else None
    
    def get_deployment_by_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Get deployment for a strategy.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Deployment dictionary or None
        """
        deployment_id = self.strategy_to_deployment.get(strategy_id)
        if not deployment_id:
            return None
            
        return self.get_deployment(deployment_id)
    
    def get_deployment_summary(self) -> Dict[str, Any]:
        """
        Get summary of all deployments.
        
        Returns:
            Summary dictionary
        """
        # Count by status
        status_counts = {}
        for deployment in self.deployments.values():
            status = deployment.status
            if status not in status_counts:
                status_counts[status] = 0
            status_counts[status] += 1
        
        # Get active deployments
        active_deployments = [
            d.to_dict() for d in self.deployments.values()
            if d.status == DeploymentStatus.ACTIVE
        ]
        
        # Calculate total allocation
        total_allocation = sum(
            d.risk_params.get("allocation_percentage", 0)
            for d in self.deployments.values()
            if d.status == DeploymentStatus.ACTIVE
        )
        
        # Calculate total P&L
        total_pnl = sum(
            d.performance.get("profit_loss", 0)
            for d in self.deployments.values()
        )
        
        return {
            "total_deployments": len(self.deployments),
            "status_counts": status_counts,
            "active_deployments": len(active_deployments),
            "total_allocation": total_allocation,
            "total_profit_loss": total_pnl,
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_strategy_optimised(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle strategy optimized event."""
        strategy_id = data.get("strategy_id")
        if not strategy_id:
            return
            
        logger.info(f"Received optimization event for strategy {strategy_id}")
    
    def _handle_strategy_deployed(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle strategy deployed event."""
        strategy_id = data.get("strategy_id")
        if not strategy_id:
            return
        
        logger.info(f"Strategy {strategy_id} deployed with risk controls")
        
        # If we have a deployment record for this strategy, update it
        if strategy_id in self.strategy_to_deployment:
            deployment_id = self.strategy_to_deployment[strategy_id]
            if deployment_id in self.deployments:
                deployment = self.deployments[deployment_id]
                deployment.update_status(
                    DeploymentStatus.ACTIVE,
                    "Deployment confirmed by risk manager"
                )
                self._save_deployments()
    
    def _handle_strategy_paused(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle strategy paused event."""
        strategy_id = data.get("strategy_id")
        reason = data.get("reason", "")
        
        if not strategy_id:
            return
            
        logger.info(f"Strategy {strategy_id} paused: {reason}")
        
        # If we have a deployment record for this strategy, update it
        if strategy_id in self.strategy_to_deployment:
            deployment_id = self.strategy_to_deployment[strategy_id]
            if deployment_id in self.deployments:
                deployment = self.deployments[deployment_id]
                deployment.update_status(
                    DeploymentStatus.PAUSED,
                    reason
                )
                self._save_deployments()
    
    def _handle_strategy_resumed(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle strategy resumed event."""
        strategy_id = data.get("strategy_id")
        
        if not strategy_id:
            return
            
        logger.info(f"Strategy {strategy_id} resumed")
        
        # If we have a deployment record for this strategy, update it
        if strategy_id in self.strategy_to_deployment:
            deployment_id = self.strategy_to_deployment[strategy_id]
            if deployment_id in self.deployments:
                deployment = self.deployments[deployment_id]
                deployment.update_status(
                    DeploymentStatus.ACTIVE,
                    "Deployment resumed"
                )
                self._save_deployments()
    
    def _handle_trade_executed(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle trade executed event."""
        strategy_id = data.get("strategy_id")
        
        if not strategy_id or strategy_id not in self.strategy_to_deployment:
            return
            
        deployment_id = self.strategy_to_deployment[strategy_id]
        deployment = self.deployments.get(deployment_id)
        
        if not deployment:
            return
            
        # Update trade count
        deployment.performance["trades"] += 1
        deployment.performance["last_updated"] = datetime.now().isoformat()
        
        # Save deployments
        self._save_deployments()
    
    def _handle_position_closed(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle position closed event."""
        strategy_id = data.get("strategy_id")
        profit_loss = data.get("profit_loss", 0)
        
        if not strategy_id or strategy_id not in self.strategy_to_deployment:
            return
            
        deployment_id = self.strategy_to_deployment[strategy_id]
        deployment = self.deployments.get(deployment_id)
        
        if not deployment:
            return
            
        # Update performance metrics
        current_pnl = deployment.performance.get("profit_loss", 0)
        new_pnl = current_pnl + profit_loss
        deployment.performance["profit_loss"] = new_pnl
        
        # Update win rate
        is_win = profit_loss > 0
        win_count = deployment.performance.get("win_count", 0)
        trade_count = deployment.performance.get("trades", 0)
        
        if is_win:
            win_count += 1
            deployment.performance["win_count"] = win_count
        
        if trade_count > 0:
            deployment.performance["win_rate"] = (win_count / trade_count) * 100
        
        deployment.performance["last_updated"] = datetime.now().isoformat()
        
        # Save deployments
        self._save_deployments()
    
    def _handle_circuit_breaker(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle circuit breaker event."""
        reasons = data.get("reasons", [])
        
        logger.warning(f"Circuit breaker triggered: {reasons}")
        
        # This might affect multiple deployments, so we'll emit a global notification
        self._emit_event(
            event_type=EventType.DEPLOYMENT_CIRCUIT_BREAKER,
            data={
                "reasons": reasons,
                "affected_deployments": len(self.get_deployments(status=DeploymentStatus.ACTIVE)),
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def _emit_deployment_event(self, 
                              event_type: str, 
                              deployment: StrategyDeployment,
                              details: Dict[str, Any]) -> None:
        """
        Emit deployment-specific event.
        
        Args:
            event_type: Event type
            deployment: Deployment object
            details: Event details
        """
        if not self.event_bus:
            return
            
        try:
            data = {
                "deployment_id": deployment.deployment_id,
                "strategy_id": deployment.strategy_id,
                "status": deployment.status,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add details
            data.update(details)
            
            # Create and emit event
            event = Event(
                event_type=event_type,
                source="DeploymentPipeline",
                data=data,
                timestamp=datetime.now()
            )
            self.event_bus.publish(event)
            
        except Exception as e:
            logger.error(f"Error emitting deployment event: {e}")
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Emit general event.
        
        Args:
            event_type: Event type
            data: Event data
        """
        if not self.event_bus:
            return
            
        try:
            event = Event(
                event_type=event_type,
                source="DeploymentPipeline",
                data=data,
                timestamp=datetime.now()
            )
            self.event_bus.publish(event)
            
        except Exception as e:
            logger.error(f"Error emitting event: {e}")
    
    def _save_deployments(self) -> None:
        """Save deployments to disk."""
        try:
            data = {
                deployment_id: deployment.to_dict()
                for deployment_id, deployment in self.deployments.items()
            }
            
            file_path = os.path.join(self.data_dir, "deployments.json")
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug("Saved deployment data")
            
        except Exception as e:
            logger.error(f"Error saving deployments: {e}")
    
    def _load_deployments(self) -> None:
        """Load deployments from disk."""
        file_path = os.path.join(self.data_dir, "deployments.json")
        
        if not os.path.exists(file_path):
            return
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            for deployment_id, deployment_data in data.items():
                deployment = StrategyDeployment.from_dict(deployment_data)
                self.deployments[deployment_id] = deployment
                self.strategy_to_deployment[deployment.strategy_id] = deployment_id
                
            logger.info(f"Loaded {len(self.deployments)} deployments from disk")
            
        except Exception as e:
            logger.error(f"Error loading deployments: {e}")


# Singleton instance for global access
_deployment_pipeline = None

def get_deployment_pipeline(event_bus: Optional[EventBus] = None,
                           data_dir: Optional[str] = None) -> StrategyDeploymentPipeline:
    """
    Get the singleton deployment pipeline instance.
    
    Args:
        event_bus: Event bus for communication
        data_dir: Directory for deployment data
        
    Returns:
        StrategyDeploymentPipeline instance
    """
    global _deployment_pipeline
    
    if _deployment_pipeline is None:
        _deployment_pipeline = StrategyDeploymentPipeline(event_bus, data_dir)
        
    return _deployment_pipeline
