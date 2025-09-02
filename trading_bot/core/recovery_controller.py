#!/usr/bin/env python3
"""
Recovery Controller for Trading Bot

This module implements a central recovery controller that integrates state 
persistence, system monitoring, and recovery mechanisms to ensure the trading
bot can recover from crashes or disconnects without human intervention.
"""

import os
import time
import logging
import threading
import traceback
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta

from trading_bot.core.state_manager import StateManager, StatePersistenceFormat
from trading_bot.core.constants import EventType
from trading_bot.core.event_bus import get_global_event_bus, Event
from trading_bot.monitoring.system_monitor import SystemMonitor

logger = logging.getLogger(__name__)

class HealthStatus:
    """Health status constants for system components."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"

class RecoveryController:
    """
    Coordinates crash recovery, state persistence, and health monitoring.
    
    Features:
    - Real-time crash detection and recovery
    - Integration with state manager for state persistence
    - Integration with system monitor for health checks
    - Component-level recovery actions
    - Heartbeat monitoring for all critical components
    - Transaction log validation to ensure idempotent operations
    """
    
    def __init__(
        self,
        state_dir: str = "data/state",
        snapshot_interval_seconds: int = 60,
        heartbeat_interval_seconds: int = 10,
        recovery_hooks: Optional[Dict[str, Callable]] = None,
        startup_recovery: bool = True,
        enable_auto_restart: bool = True
    ):
        """
        Initialize the recovery controller.
        
        Args:
            state_dir: Directory for state persistence
            snapshot_interval_seconds: How often to snapshot state
            heartbeat_interval_seconds: Interval for heartbeat checks
            recovery_hooks: Custom recovery functions by component name
            startup_recovery: Whether to attempt recovery on startup
            enable_auto_restart: Whether to auto-restart failed components
        """
        self.state_dir = state_dir
        self.snapshot_interval = snapshot_interval_seconds
        self.heartbeat_interval = heartbeat_interval_seconds
        self.recovery_hooks = recovery_hooks or {}
        self.startup_recovery = startup_recovery
        self.enable_auto_restart = enable_auto_restart
        
        # Create state directory
        os.makedirs(state_dir, exist_ok=True)
        
        # Initialize state manager
        self.state_manager = StateManager(
            state_dir=state_dir,
            snapshot_interval_seconds=snapshot_interval_seconds,
            format=StatePersistenceFormat.JSON,
            max_snapshots=5
        )
        
        # Initialize system monitor - pass log_dir, not state_dir
        monitor_log_dir = os.path.join(os.path.dirname(state_dir), "logs/system_monitor")
        self.system_monitor = SystemMonitor(
            log_dir=monitor_log_dir,
            check_interval=heartbeat_interval_seconds * 3,  # Less frequent checks
            enable_auto_recovery=enable_auto_restart
        )
        
        # Component registry for recovery
        self.components = {}
        
        # Heartbeat tracking
        self.component_heartbeats = {}
        self.last_heartbeat_check = time.time()
        
        # Recovery state 
        self.is_recovering = False
        self.recovery_history = []
        self.restart_attempts = {}
        
        # Control threads
        self._running = False
        self._heartbeat_thread = None
        self._recovery_lock = threading.RLock()
        
        # Event bus for notifications
        self.event_bus = get_global_event_bus()
    
    def register_component(
        self, 
        component_id: str, 
        component: Any, 
        is_critical: bool = True,
        recovery_method: Optional[str] = None,
        health_check_method: Optional[str] = None,
        max_restart_attempts: int = 3,
        restart_cooldown_seconds: int = 300
    ) -> None:
        """
        Register a component for crash recovery monitoring.
        
        Args:
            component_id: Unique identifier for the component
            component: The component instance
            is_critical: Whether this component is critical for operation
            recovery_method: Name of method to call for recovery
            health_check_method: Name of method to check component health
            max_restart_attempts: Maximum number of restart attempts before giving up
            restart_cooldown_seconds: Time to wait between restart attempts
        """
        self.components[component_id] = {
            "instance": component,
            "is_critical": is_critical,
            "recovery_method": recovery_method,
            "health_check_method": health_check_method,
            "max_restart_attempts": max_restart_attempts,
            "restart_cooldown_seconds": restart_cooldown_seconds,
            "last_restart_time": 0,
            "health_status": HealthStatus.UNKNOWN,
            "registration_time": time.time()
        }
        
        # Initialize heartbeat for this component
        self.component_heartbeats[component_id] = time.time()
        
        # Register with state manager if it has the required methods
        if hasattr(component, "get_state") and hasattr(component, "restore_state"):
            self.state_manager.register_component(component_id, component)
        
        # Register custom health check with system monitor
        if health_check_method and hasattr(component, health_check_method):
            self.system_monitor.register_custom_health_check(
                f"{component_id}_health",
                lambda: self._run_component_health_check(component_id),
                "warning"
            )
        
        # Register recovery action with system monitor
        self.system_monitor.register_recovery_action(
            f"{component_id}_failure", 
            lambda alert: self._handle_component_failure(component_id, alert)
        )
        
        logger.info(f"Registered component {component_id} for crash recovery")
    
    def unregister_component(self, component_id: str) -> bool:
        """
        Unregister a component.
        
        Args:
            component_id: Component ID to unregister
            
        Returns:
            True if component was unregistered, False otherwise
        """
        if component_id in self.components:
            # Remove from state manager
            self.state_manager.unregister_component(component_id)
            
            # Remove from components registry
            del self.components[component_id]
            
            # Remove from heartbeat tracking
            if component_id in self.component_heartbeats:
                del self.component_heartbeats[component_id]
            
            logger.info(f"Unregistered component {component_id}")
            return True
        
        return False
    
    def start(self) -> bool:
        """
        Start the recovery controller.
        
        Returns:
            True if started successfully, False otherwise
        """
        with self._recovery_lock:
            if self._running:
                logger.warning("Recovery controller is already running")
                return False
            
            self._running = True
            
            # Start state manager
            self.state_manager.start_auto_snapshot()
            
            # Start system monitor
            self.system_monitor.start()
            
            # Start heartbeat thread
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True,
                name="RecoveryControllerHeartbeat"
            )
            self._heartbeat_thread.start()
            
            # Perform startup recovery if enabled
            if self.startup_recovery:
                logger.info("Performing startup recovery...")
                self._perform_startup_recovery()
            
            logger.info("Recovery controller started")
            return True
    
    def stop(self) -> None:
        """Stop the recovery controller."""
        with self._recovery_lock:
            if not self._running:
                logger.warning("Recovery controller is not running")
                return
            
            self._running = False
            
            # Stop state manager
            self.state_manager.stop_auto_snapshot()
            
            # Stop system monitor
            self.system_monitor.stop()
            
            # Take a final snapshot
            self.state_manager.create_snapshot()
            
            logger.info("Recovery controller stopped")
    
    def record_heartbeat(self, component_id: str) -> None:
        """
        Record a heartbeat for a specific component.
        
        Args:
            component_id: Component ID to record heartbeat for
        """
        if component_id in self.component_heartbeats:
            self.component_heartbeats[component_id] = time.time()
    
    def verify_transaction(self, transaction_type: str, transaction_id: str) -> bool:
        """
        Verify if a transaction has already been processed.
        Used to prevent duplicate operations after a crash.
        
        Args:
            transaction_type: Type of transaction (e.g., "order", "position")
            transaction_id: Unique ID for the transaction
            
        Returns:
            True if this is a new transaction, False if already processed
        """
        return self.state_manager.check_transaction(transaction_type, transaction_id) is None
    
    def log_transaction(self, transaction_type: str, transaction_id: str, 
                       data: Dict[str, Any], expiry_seconds: Optional[int] = None) -> None:
        """
        Log a completed transaction to prevent duplication on restart.
        
        Args:
            transaction_type: Type of transaction (e.g., "order", "position")
            transaction_id: Unique ID for the transaction
            data: Transaction data
            expiry_seconds: Optional time after which this transaction can be forgotten
        """
        self.state_manager.log_transaction(
            transaction_type, transaction_id, data, expiry_seconds
        )
    
    def generate_transaction_id(self, data: Dict[str, Any]) -> str:
        """
        Generate a deterministic transaction ID from data.
        
        Args:
            data: Transaction data
            
        Returns:
            Transaction ID string
        """
        return self.state_manager.generate_transaction_id(data)
    
    def manual_recovery(self, component_id: Optional[str] = None) -> bool:
        """
        Manually trigger recovery for a component or all components.
        
        Args:
            component_id: Optional specific component to recover
            
        Returns:
            True if recovery was successful, False otherwise
        """
        with self._recovery_lock:
            try:
                self.is_recovering = True
                
                if component_id:
                    # Recover specific component
                    if component_id not in self.components:
                        logger.error(f"Component {component_id} not found")
                        return False
                    
                    result = self._recover_component(component_id)
                    
                    if result:
                        logger.info(f"Successfully recovered component {component_id}")
                    else:
                        logger.error(f"Failed to recover component {component_id}")
                    
                    return result
                else:
                    # Recover all components
                    return self._recover_all_components()
                
            finally:
                self.is_recovering = False
    
    def save_crash_report(self, error: Exception, context: Dict[str, Any]) -> None:
        """
        Save a crash report for later analysis.
        
        Args:
            error: The exception that caused the crash
            context: Additional context about the crash
        """
        try:
            # Create crash report
            crash_time = datetime.now()
            crash_report = {
                "timestamp": crash_time.isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "context": context
            }
            
            # Save to crash reports directory
            crash_dir = os.path.join(self.state_dir, "crash_reports")
            os.makedirs(crash_dir, exist_ok=True)
            
            # Create filename with timestamp
            crash_file = os.path.join(
                crash_dir, 
                f"crash_{crash_time.strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(crash_file, 'w') as f:
                import json
                json.dump(crash_report, f, indent=2)
            
            logger.info(f"Saved crash report to {crash_file}")
            
            # Add to recovery history
            self.recovery_history.append({
                "timestamp": crash_time,
                "error_type": type(error).__name__,
                "recovered": False
            })
            
            # Notify via event bus
            self.event_bus.publish(
                Event(
                    event_type=EventType.SYSTEM_ERROR,
                    data={
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "crash_report": crash_file
                    }
                )
            )
            
        except Exception as e:
            logger.error(f"Error saving crash report: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of all monitored components.
        
        Returns:
            Dictionary with health status data
        """
        status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": HealthStatus.HEALTHY,
            "components": {},
            "recovery_history": self.recovery_history[-10:],  # Last 10 entries
            "last_state_snapshot": self._get_last_snapshot_time(),
            "is_recovering": self.is_recovering
        }
        
        # Check each component
        critical_unhealthy = False
        
        for component_id, component_info in self.components.items():
            component_status = component_info["health_status"]
            heartbeat_age = time.time() - self.component_heartbeats.get(component_id, 0)
            
            # Check if component has missed heartbeats
            if heartbeat_age > self.heartbeat_interval * 3:
                if component_status != HealthStatus.CRITICAL:
                    component_status = HealthStatus.WARNING
                
                if heartbeat_age > self.heartbeat_interval * 10:
                    component_status = HealthStatus.CRITICAL
                    
                    if component_info["is_critical"]:
                        critical_unhealthy = True
            
            # Add to status report
            status["components"][component_id] = {
                "status": component_status,
                "last_heartbeat": heartbeat_age,
                "is_critical": component_info["is_critical"],
                "restart_attempts": self.restart_attempts.get(component_id, 0)
            }
        
        # Set overall status
        if critical_unhealthy:
            status["overall_status"] = HealthStatus.CRITICAL
        elif self.is_recovering:
            status["overall_status"] = HealthStatus.RECOVERING
        elif any(comp["status"] == HealthStatus.WARNING for comp in status["components"].values()):
            status["overall_status"] = HealthStatus.WARNING
        
        return status
    
    def _perform_startup_recovery(self) -> bool:
        """
        Perform recovery on system startup by restoring state.
        
        Returns:
            True if recovery was successful, False otherwise
        """
        try:
            logger.info("Starting system recovery on startup")
            self.is_recovering = True
            
            # Restore from latest snapshot
            restore_success = self.state_manager.restore_latest_snapshot()
            
            if restore_success:
                logger.info("Successfully restored state from snapshot")
                
                # Add to recovery history
                self.recovery_history.append({
                    "timestamp": datetime.now(),
                    "type": "startup_recovery",
                    "recovered": True
                })
                
                # Trigger recovered event
                self.event_bus.publish(
                    Event(
                        event_type=EventType.SYSTEM_RECOVERED,
                        data={"recovery_type": "startup", "success": True}
                    )
                )
                
                return True
            else:
                logger.warning("No state snapshot found or restore failed")
                return False
                
        except Exception as e:
            logger.error(f"Error during startup recovery: {str(e)}")
            return False
        finally:
            self.is_recovering = False
    
    def _heartbeat_loop(self) -> None:
        """Main heartbeat monitoring loop."""
        logger.info("Heartbeat monitoring started")
        
        while self._running:
            try:
                # Sleep for a short time
                time.sleep(1.0)
                
                # Check if it's time for heartbeat check
                current_time = time.time()
                if (current_time - self.last_heartbeat_check) >= self.heartbeat_interval:
                    self._check_component_heartbeats()
                    self.last_heartbeat_check = current_time
                    
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {str(e)}")
                time.sleep(self.heartbeat_interval)  # Sleep longer on error
        
        logger.info("Heartbeat monitoring stopped")
    
    def _check_component_heartbeats(self) -> None:
        """Check heartbeats for all components and trigger recovery if needed."""
        current_time = time.time()
        
        for component_id, last_heartbeat in self.component_heartbeats.items():
            if component_id not in self.components:
                continue  # Component was unregistered
                
            component_info = self.components[component_id]
            heartbeat_age = current_time - last_heartbeat
            
            # Check if component has missed too many heartbeats
            if heartbeat_age > self.heartbeat_interval * 5:
                # Critical failure threshold
                logger.warning(f"Component {component_id} has missed heartbeats for {heartbeat_age:.1f}s")
                
                # Set health status to warning
                component_info["health_status"] = HealthStatus.WARNING
                
                if heartbeat_age > self.heartbeat_interval * 10:
                    # Critical threshold exceeded
                    logger.error(f"Component {component_id} heartbeat critical: {heartbeat_age:.1f}s")
                    component_info["health_status"] = HealthStatus.CRITICAL
                    
                    # Attempt recovery for critical components
                    if component_info["is_critical"] and self.enable_auto_restart:
                        self._handle_component_failure(
                            component_id, 
                            {"reason": "heartbeat_timeout", "threshold": self.heartbeat_interval * 10}
                        )
    
    def _recover_component(self, component_id: str) -> bool:
        """
        Recover a specific component.
        
        Args:
            component_id: ID of component to recover
            
        Returns:
            True if recovery was successful, False otherwise
        """
        if component_id not in self.components:
            logger.error(f"Component {component_id} not found")
            return False
        
        component_info = self.components[component_id]
        component = component_info["instance"]
        
        # Check if we've exceeded max restart attempts
        restart_attempts = self.restart_attempts.get(component_id, 0)
        if restart_attempts >= component_info["max_restart_attempts"]:
            cooldown_time = component_info["restart_cooldown_seconds"]
            time_since_last = time.time() - component_info["last_restart_time"]
            
            if time_since_last < cooldown_time:
                logger.warning(
                    f"Component {component_id} exceeded max restart attempts. " +
                    f"In cooldown for {cooldown_time - time_since_last:.1f}s"
                )
                return False
            else:
                # Reset restart counter after cooldown
                self.restart_attempts[component_id] = 0
        
        # Attempt recovery
        try:
            recovery_success = False
            
            # Try using recovery method if specified
            if component_info["recovery_method"] and hasattr(component, component_info["recovery_method"]):
                recovery_method = getattr(component, component_info["recovery_method"])
                recovery_success = recovery_method()
            
            # Use custom recovery hook if available
            elif component_id in self.recovery_hooks:
                recovery_success = self.recovery_hooks[component_id](component)
            
            # Update component state
            if recovery_success:
                # Reset heartbeat
                self.component_heartbeats[component_id] = time.time()
                
                # Update health status
                component_info["health_status"] = HealthStatus.HEALTHY
                
                # Reset restart attempts after successful recovery
                if component_id in self.restart_attempts:
                    del self.restart_attempts[component_id]
                
                # Add to recovery history
                self.recovery_history.append({
                    "timestamp": datetime.now(),
                    "component_id": component_id,
                    "recovered": True
                })
                
                logger.info(f"Successfully recovered component {component_id}")
                return True
            else:
                # Increment restart attempts
                self.restart_attempts[component_id] = restart_attempts + 1
                component_info["last_restart_time"] = time.time()
                
                # Update health status
                component_info["health_status"] = HealthStatus.CRITICAL
                
                # Add to recovery history
                self.recovery_history.append({
                    "timestamp": datetime.now(),
                    "component_id": component_id,
                    "recovered": False
                })
                
                logger.error(f"Failed to recover component {component_id} " +
                             f"(attempt {self.restart_attempts[component_id]})")
                return False
                
        except Exception as e:
            logger.error(f"Error recovering component {component_id}: {str(e)}")
            return False
    
    def _recover_all_components(self) -> bool:
        """
        Recover all registered components.
        
        Returns:
            True if all critical components recovered, False otherwise
        """
        logger.info("Recovering all components...")
        
        # First restore state from snapshot
        state_restored = self.state_manager.restore_latest_snapshot()
        if state_restored:
            logger.info("Successfully restored state from snapshot")
        else:
            logger.warning("Failed to restore state from snapshot")
        
        # Track critical component recovery
        all_critical_recovered = True
        recovered_components = []
        
        # Recover each component
        for component_id in list(self.components.keys()):
            component_info = self.components[component_id]
            
            try:
                if self._recover_component(component_id):
                    recovered_components.append(component_id)
                elif component_info["is_critical"]:
                    all_critical_recovered = False
            except Exception as e:
                logger.error(f"Error recovering component {component_id}: {str(e)}")
                if component_info["is_critical"]:
                    all_critical_recovered = False
        
        # Log results
        logger.info(f"Recovery completed. Recovered {len(recovered_components)} components")
        
        # Publish recovery event
        self.event_bus.publish(
            Event(
                event_type=EventType.SYSTEM_RECOVERED,
                data={
                    "success": all_critical_recovered,
                    "recovered_components": recovered_components
                }
            )
        )
        
        return all_critical_recovered
    
    def _run_component_health_check(self, component_id: str) -> Dict[str, Any]:
        """
        Run health check for a specific component.
        
        Args:
            component_id: ID of component to check
            
        Returns:
            Dictionary with health check results
        """
        if component_id not in self.components:
            return {
                "status": "error",
                "message": f"Component {component_id} not found"
            }
        
        component_info = self.components[component_id]
        component = component_info["instance"]
        health_check_method = component_info["health_check_method"]
        
        # Skip if no health check method
        if not health_check_method or not hasattr(component, health_check_method):
            return {
                "status": "unknown",
                "message": f"No health check method for {component_id}"
            }
        
        try:
            # Call component's health check method
            health_check = getattr(component, health_check_method)
            result = health_check()
            
            # Update component health status
            if result.get("status") == "healthy":
                component_info["health_status"] = HealthStatus.HEALTHY
            elif result.get("status") == "warning":
                component_info["health_status"] = HealthStatus.WARNING
            elif result.get("status") == "error":
                component_info["health_status"] = HealthStatus.CRITICAL
            
            return result
            
        except Exception as e:
            logger.error(f"Error running health check for {component_id}: {str(e)}")
            component_info["health_status"] = HealthStatus.CRITICAL
            
            return {
                "status": "error",
                "message": f"Health check error: {str(e)}"
            }
    
    def _handle_component_failure(self, component_id: str, alert: Dict[str, Any]) -> None:
        """
        Handle component failure alert from system monitor.
        
        Args:
            component_id: ID of failed component
            alert: Alert details from system monitor
        """
        if component_id not in self.components:
            logger.warning(f"Received failure alert for unknown component {component_id}")
            return
        
        logger.warning(f"Handling failure of component {component_id}: {alert.get('reason', 'unknown')}")
        
        # Attempt recovery if auto-restart is enabled
        if self.enable_auto_restart:
            with self._recovery_lock:
                if not self.is_recovering:
                    self.is_recovering = True
                    try:
                        recovery_success = self._recover_component(component_id)
                        
                        # Take a state snapshot if recovery was successful
                        if recovery_success:
                            self.state_manager.create_snapshot()
                            
                    finally:
                        self.is_recovering = False
    
    def _get_last_snapshot_time(self) -> Optional[str]:
        """Get the timestamp of the last state snapshot."""
        try:
            snapshots = self.state_manager._get_snapshot_files()
            if snapshots:
                # Extract timestamp from filename
                filename = os.path.basename(snapshots[-1])
                if filename.startswith("state_snapshot_"):
                    try:
                        timestamp_str = filename.split("_")[2].split(".")[0]
                        timestamp = datetime.fromtimestamp(int(timestamp_str))
                        return timestamp.isoformat()
                    except:
                        pass
                        
            return None
        except:
            return None
