#!/usr/bin/env python3
"""
Strategy Lifecycle Integration

This module provides the integration between the Strategy Lifecycle Manager
and other core components of the autonomous trading system:
- AutonomousEngine
- RiskManager
- DeploymentPipeline

It serves as the primary connection point to ensure full system autonomy.
"""

import logging
import os
import json
import threading
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

# Import core components
from trading_bot.autonomous.autonomous_engine import AutonomousEngine, StrategyCandidate
from trading_bot.autonomous.risk_integration import get_autonomous_risk_manager
from trading_bot.autonomous.strategy_deployment_pipeline import get_deployment_pipeline, DeploymentStatus
from trading_bot.strategies.components.component_registry import ComponentRegistry

# Import lifecycle components
from trading_bot.autonomous.strategy_lifecycle_manager import (
    get_strategy_lifecycle_manager, StrategyVersion, VersionStatus, VersionSource
)
from trading_bot.autonomous.strategy_lifecycle_extensions import get_strategy_lifecycle_extension
from trading_bot.autonomous.strategy_lifecycle_event_handlers import get_strategy_lifecycle_event_tracker, register_lifecycle_event_types

# Import event system
from trading_bot.event_system import EventBus, Event, EventType

logger = logging.getLogger(__name__)

class StrategyLifecycleIntegration:
    """
    Provides integration between the Strategy Lifecycle Manager and other
    core components of the autonomous trading system.
    """
    
    def __init__(self, 
                 autonomous_engine: Optional[AutonomousEngine] = None,
                 event_bus: Optional[EventBus] = None):
        """
        Initialize the lifecycle integration.
        
        Args:
            autonomous_engine: AutonomousEngine instance
            event_bus: Event bus for communication
        """
        self.event_bus = event_bus or EventBus()
        
        # Register lifecycle event types
        register_lifecycle_event_types(self.event_bus)
        
        # Get core components
        self.autonomous_engine = autonomous_engine
        self.risk_manager = get_autonomous_risk_manager()
        self.deployment_pipeline = get_deployment_pipeline()
        self.component_registry = ComponentRegistry()
        
        # Get lifecycle components
        self.lifecycle_manager = get_strategy_lifecycle_manager(self.event_bus)
        self.lifecycle_extension = get_strategy_lifecycle_extension(self.event_bus)
        self.event_tracker = get_strategy_lifecycle_event_tracker(self.event_bus)
        
        # Configuration
        self.config = {
            "auto_track_candidates": True,
            "auto_promote_successful_candidates": True,
            "sync_candidates_on_startup": True,
            "integration_check_interval_seconds": 300,  # 5 minutes
        }
        
        # Internal state
        self.is_running = False
        self.integration_thread = None
        self._lock = threading.RLock()
        
        # Register for events
        self._register_for_events()
        
        logger.info("Strategy Lifecycle Integration initialized")
        
        # Perform initial sync if configured
        if self.config["sync_candidates_on_startup"]:
            self.sync_engine_candidates()
    
    def _register_for_events(self) -> None:
        """Register for relevant events."""
        # Autonomous engine events
        self.event_bus.register(EventType.STRATEGY_CANDIDATE_GENERATED, self._handle_candidate_generated)
        self.event_bus.register(EventType.STRATEGY_CANDIDATE_EVALUATED, self._handle_candidate_evaluated)
        self.event_bus.register(EventType.OPTIMIZATION_COMPLETE, self._handle_optimization_complete)
        
        # Track deployment events for syncing
        self.event_bus.register(EventType.STRATEGY_DEPLOYED_WITH_RISK, self._handle_strategy_deployed)
        
        logger.info("Registered for events")
    
    def start_integration(self) -> None:
        """Start integration monitoring thread."""
        if self.is_running:
            logger.warning("Integration already running")
            return
        
        # Start lifecycle extension monitoring
        self.lifecycle_extension.start_monitoring()
        
        # Start integration thread
        self.is_running = True
        self.integration_thread = threading.Thread(target=self._integration_loop, daemon=True)
        self.integration_thread.start()
        
        logger.info("Started lifecycle integration")
    
    def stop_integration(self) -> None:
        """Stop integration monitoring thread."""
        self.is_running = False
        
        # Stop lifecycle extension monitoring
        self.lifecycle_extension.stop_monitoring()
        
        if self.integration_thread:
            self.integration_thread.join(timeout=2.0)
            self.integration_thread = None
        
        logger.info("Stopped lifecycle integration")
    
    def _integration_loop(self) -> None:
        """Main integration monitoring loop."""
        while self.is_running:
            try:
                # Synchronize with autonomous engine
                self.sync_engine_candidates()
                
                # Check deployments vs lifecycle state
                self.sync_active_deployments()
                
            except Exception as e:
                logger.error(f"Error in integration loop: {e}")
            
            # Sleep for the configured interval
            import time
            time.sleep(self.config["integration_check_interval_seconds"])
    
    def sync_engine_candidates(self) -> None:
        """Synchronize autonomous engine candidates with lifecycle manager."""
        if not self.autonomous_engine:
            logger.warning("Cannot sync candidates: AutonomousEngine not available")
            return
        
        try:
            # Get candidates from engine
            all_candidates = self.autonomous_engine.get_all_candidates()
            
            # Track candidates in lifecycle manager
            for candidate in all_candidates:
                strategy_id = candidate.strategy_id
                
                # Check if we have this strategy in lifecycle manager
                versions = self.lifecycle_manager.get_strategy_versions(strategy_id)
                
                if not versions or self.config["auto_track_candidates"]:
                    # Create a new version for this candidate
                    metrics = {
                        "sharpe_ratio": candidate.performance.get("sharpe_ratio", 0),
                        "total_return_pct": candidate.performance.get("total_return", 0),
                        "win_rate": candidate.performance.get("win_rate", 0),
                        "max_drawdown_pct": candidate.performance.get("max_drawdown", 0),
                        "total_trades": candidate.performance.get("trades", 0),
                        "backtest_days": candidate.performance.get("backtest_days", 30)
                    }
                    
                    # Create version ID from candidate ID
                    version_id = f"cand-{candidate.candidate_id}" if candidate.candidate_id else None
                    
                    # Track in lifecycle manager
                    self.lifecycle_manager.track_strategy_version(
                        strategy_id=strategy_id,
                        parameters=candidate.parameters,
                        metrics=metrics,
                        source=VersionSource.INITIAL,
                        version_id=version_id,
                        metadata={
                            "universe": candidate.universe,
                            "symbols": candidate.symbols,
                            "candidate_id": candidate.candidate_id,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    
                    logger.info(f"Tracked engine candidate for strategy {strategy_id}")
            
            logger.info(f"Synchronized {len(all_candidates)} candidates from engine")
            
        except Exception as e:
            logger.error(f"Error synchronizing engine candidates: {e}")
    
    def sync_active_deployments(self) -> None:
        """Synchronize active deployments with lifecycle manager."""
        try:
            # Get active deployments
            active_deployments = self.deployment_pipeline.get_deployments(
                status=DeploymentStatus.ACTIVE
            )
            
            # Check each deployment against lifecycle manager
            for deployment in active_deployments:
                strategy_id = deployment.get("strategy_id")
                deployment_id = deployment.get("deployment_id")
                
                if not strategy_id or not deployment_id:
                    continue
                
                # Get metadata
                metadata = deployment.get("metadata", {})
                version_id = metadata.get("version_id")
                
                # Check if version is tracked
                if version_id:
                    version = self.lifecycle_manager.get_version(strategy_id, version_id)
                    
                    if not version:
                        # Track this version
                        parameters = deployment.get("parameters", {})
                        self.lifecycle_manager.track_strategy_version(
                            strategy_id=strategy_id,
                            parameters=parameters,
                            source=VersionSource.MANUAL,
                            version_id=version_id,
                            metadata={"deployment_id": deployment_id}
                        )
                        
                        logger.info(f"Tracked missing version {version_id} for strategy {strategy_id}")
                    
                    # Ensure version is marked as deployed
                    if version and version.status != VersionStatus.DEPLOYED:
                        self.lifecycle_manager.set_version_status(
                            strategy_id=strategy_id,
                            version_id=version_id,
                            status=VersionStatus.DEPLOYED,
                            reason=f"Synchronized with deployment {deployment_id}"
                        )
                else:
                    # No version ID in metadata, check if we have an active version
                    active_version = self.lifecycle_manager.get_active_version(strategy_id)
                    
                    if not active_version:
                        # Create a new version for this deployment
                        parameters = deployment.get("parameters", {})
                        new_version_id = self.lifecycle_manager.track_strategy_version(
                            strategy_id=strategy_id,
                            parameters=parameters,
                            source=VersionSource.MANUAL,
                            metadata={"deployment_id": deployment_id}
                        )
                        
                        # Mark as deployed
                        self.lifecycle_manager.set_version_status(
                            strategy_id=strategy_id,
                            version_id=new_version_id,
                            status=VersionStatus.DEPLOYED,
                            reason=f"Synchronized with deployment {deployment_id}"
                        )
                        
                        logger.info(f"Created and deployed version for strategy {strategy_id}")
            
            logger.info(f"Synchronized {len(active_deployments)} active deployments")
            
        except Exception as e:
            logger.error(f"Error synchronizing active deployments: {e}")
    
    def deploy_version(self, 
                      strategy_id: str, 
                      version_id: str,
                      risk_params: Optional[Dict[str, Any]] = None,
                      reason: str = "Manual deployment") -> Dict[str, Any]:
        """
        Deploy a specific version of a strategy.
        
        Args:
            strategy_id: Strategy identifier
            version_id: Version identifier
            risk_params: Risk parameters
            reason: Reason for deployment
            
        Returns:
            Deployment result
        """
        # Get the version
        version = self.lifecycle_manager.get_version(strategy_id, version_id)
        
        if not version:
            raise ValueError(f"Unknown version {version_id} for strategy {strategy_id}")
        
        # Default risk parameters
        if not risk_params:
            risk_params = {
                "allocation_percentage": 5.0,
                "stop_loss_pct": 10.0,
                "risk_level": "MEDIUM"
            }
        
        # Deploy with pipeline
        deployment_result = self.deployment_pipeline.deploy_strategy(
            strategy_id=strategy_id,
            parameters=version.parameters,
            risk_params=risk_params,
            metadata={
                "version_id": version_id,
                "reason": reason
            }
        )
        
        # Update version status
        self.lifecycle_manager.set_version_status(
            strategy_id=strategy_id,
            version_id=version_id,
            status=VersionStatus.DEPLOYED,
            reason=f"Deployed with ID {deployment_result.get('deployment_id')}: {reason}"
        )
        
        logger.info(f"Deployed strategy {strategy_id} version {version_id}")
        
        return deployment_result
    
    def promote_engine_candidate(self, 
                               candidate_id: str,
                               auto_deploy: bool = False,
                               risk_params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Promote an engine candidate to a lifecycle version.
        
        Args:
            candidate_id: Candidate ID
            auto_deploy: Whether to automatically deploy
            risk_params: Risk parameters for deployment
            
        Returns:
            Version ID if successful
        """
        if not self.autonomous_engine:
            logger.warning("Cannot promote candidate: AutonomousEngine not available")
            return None
        
        # Find the candidate
        candidate = self.autonomous_engine.get_candidate_by_id(candidate_id)
        
        if not candidate:
            logger.warning(f"Cannot find candidate with ID {candidate_id}")
            return None
        
        strategy_id = candidate.strategy_id
        
        # Create version ID
        version_id = f"cand-{candidate_id}"
        
        # Check if already tracked
        if self.lifecycle_manager.get_version(strategy_id, version_id):
            logger.info(f"Candidate {candidate_id} already tracked as version {version_id}")
        else:
            # Track the candidate
            metrics = {
                "sharpe_ratio": candidate.performance.get("sharpe_ratio", 0),
                "total_return_pct": candidate.performance.get("total_return", 0),
                "win_rate": candidate.performance.get("win_rate", 0),
                "max_drawdown_pct": candidate.performance.get("max_drawdown", 0),
                "total_trades": candidate.performance.get("trades", 0),
                "backtest_days": candidate.performance.get("backtest_days", 30)
            }
            
            # Track in lifecycle manager
            self.lifecycle_manager.track_strategy_version(
                strategy_id=strategy_id,
                parameters=candidate.parameters,
                metrics=metrics,
                source=VersionSource.INITIAL,
                version_id=version_id,
                metadata={
                    "universe": candidate.universe,
                    "symbols": candidate.symbols,
                    "candidate_id": candidate_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Promote to candidate
        self.lifecycle_extension.promote_strategy_to_candidate(
            strategy_id=strategy_id,
            version_id=version_id,
            reason=f"Manual promotion of engine candidate {candidate_id}"
        )
        
        # Deploy if requested
        if auto_deploy:
            self.deploy_version(
                strategy_id=strategy_id,
                version_id=version_id,
                risk_params=risk_params,
                reason=f"Auto-deploy of promoted candidate {candidate_id}"
            )
        
        return version_id
    
    def optimize_and_track_strategy(self, 
                                  strategy_id: str,
                                  version_id: Optional[str] = None,
                                  optimization_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize a strategy and track the result.
        
        Args:
            strategy_id: Strategy identifier
            version_id: Optional version to optimize, otherwise uses active
            optimization_params: Optimization parameters
            
        Returns:
            Optimization result
        """
        if not self.autonomous_engine:
            raise ValueError("AutonomousEngine not available")
        
        # Get the version to optimize
        if version_id:
            version = self.lifecycle_manager.get_version(strategy_id, version_id)
            if not version:
                raise ValueError(f"Unknown version {version_id} for strategy {strategy_id}")
        else:
            # Use active version
            version = self.lifecycle_manager.get_active_version(strategy_id)
            if not version:
                raise ValueError(f"No active version for strategy {strategy_id}")
            version_id = version.version_id
        
        # Default optimization parameters
        if not optimization_params:
            optimization_params = {
                "method": "bayesian",
                "iterations": 20,
                "target_metric": "sharpe_ratio"
            }
        
        # Create a candidate for the autonomous engine
        candidate = StrategyCandidate(
            strategy_id=strategy_id,
            strategy_type=strategy_id.split('_')[0] if '_' in strategy_id else strategy_id,
            symbols=version.metadata.get("symbols", []),
            universe=version.metadata.get("universe", "US"),
            parameters=version.parameters
        )
        
        # Set candidate ID
        candidate.candidate_id = f"opt-{version_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Set performance from version metrics
        backtest = version.metrics.get("backtest", {})
        candidate.performance = {
            "sharpe_ratio": backtest.get("sharpe_ratio", 0),
            "total_return": backtest.get("total_return_pct", 0),
            "win_rate": backtest.get("win_rate", 0),
            "max_drawdown": backtest.get("max_drawdown_pct", 0),
            "trades": backtest.get("total_trades", 0)
        }
        
        # Perform optimization
        result = self.autonomous_engine.optimize_strategy(
            candidate=candidate,
            optimization_params=optimization_params
        )
        
        logger.info(f"Optimized strategy {strategy_id} version {version_id}")
        
        return result
    
    def _handle_candidate_generated(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle candidate generated event.
        
        Args:
            event_type: Event type
            data: Event data
        """
        candidate_id = data.get("candidate_id")
        
        if not candidate_id or not self.autonomous_engine:
            return
        
        # Only track if configured to auto-track
        if not self.config["auto_track_candidates"]:
            return
        
        # Get the candidate
        candidate = self.autonomous_engine.get_candidate_by_id(candidate_id)
        
        if not candidate:
            return
        
        strategy_id = candidate.strategy_id
        
        # Create version ID
        version_id = f"cand-{candidate_id}"
        
        # Check if already tracked
        if self.lifecycle_manager.get_version(strategy_id, version_id):
            return
        
        # Track the candidate
        metrics = {
            "sharpe_ratio": candidate.performance.get("sharpe_ratio", 0),
            "total_return_pct": candidate.performance.get("total_return", 0),
            "win_rate": candidate.performance.get("win_rate", 0),
            "max_drawdown_pct": candidate.performance.get("max_drawdown", 0),
            "total_trades": candidate.performance.get("trades", 0),
            "backtest_days": candidate.performance.get("backtest_days", 30)
        }
        
        # Track in lifecycle manager
        self.lifecycle_manager.track_strategy_version(
            strategy_id=strategy_id,
            parameters=candidate.parameters,
            metrics=metrics,
            source=VersionSource.INITIAL,
            version_id=version_id,
            metadata={
                "universe": candidate.universe,
                "symbols": candidate.symbols,
                "candidate_id": candidate_id,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(f"Tracked engine candidate {candidate_id} for strategy {strategy_id}")
    
    def _handle_candidate_evaluated(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle candidate evaluated event.
        
        Args:
            event_type: Event type
            data: Event data
        """
        candidate_id = data.get("candidate_id")
        status = data.get("status")
        
        if not candidate_id or not status or not self.autonomous_engine:
            return
        
        # Only promote if configured and status is successful
        if not self.config["auto_promote_successful_candidates"] or status != "SUCCESSFUL":
            return
        
        # Get the candidate
        candidate = self.autonomous_engine.get_candidate_by_id(candidate_id)
        
        if not candidate:
            return
        
        strategy_id = candidate.strategy_id
        
        # Create version ID
        version_id = f"cand-{candidate_id}"
        
        # Check if already tracked
        version = self.lifecycle_manager.get_version(strategy_id, version_id)
        
        if not version:
            # Track the candidate first
            self._handle_candidate_generated(event_type, data)
            version = self.lifecycle_manager.get_version(strategy_id, version_id)
        
        if not version:
            logger.warning(f"Failed to track candidate {candidate_id}")
            return
        
        # Promote to candidate
        self.lifecycle_extension.promote_strategy_to_candidate(
            strategy_id=strategy_id,
            version_id=version_id,
            reason=f"Auto-promotion of successful engine candidate {candidate_id}"
        )
        
        logger.info(f"Auto-promoted successful candidate {candidate_id} to lifecycle candidate")
    
    def _handle_optimization_complete(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle optimization complete event.
        
        Args:
            event_type: Event type
            data: Event data
        """
        # This is already handled by the strategy_optimised event
        # which is processed by the lifecycle manager
        pass
    
    def _handle_strategy_deployed(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle strategy deployed event.
        
        Args:
            event_type: Event type
            data: Event data
        """
        # This is already handled by the lifecycle event tracker
        pass


# Singleton instance for global access
_strategy_lifecycle_integration = None

def get_strategy_lifecycle_integration(autonomous_engine: Optional[AutonomousEngine] = None,
                                     event_bus: Optional[EventBus] = None) -> StrategyLifecycleIntegration:
    """
    Get singleton instance of strategy lifecycle integration.
    
    Args:
        autonomous_engine: AutonomousEngine instance
        event_bus: Event bus for communication
        
    Returns:
        StrategyLifecycleIntegration instance
    """
    global _strategy_lifecycle_integration
    
    if _strategy_lifecycle_integration is None:
        _strategy_lifecycle_integration = StrategyLifecycleIntegration(autonomous_engine, event_bus)
        
    return _strategy_lifecycle_integration
