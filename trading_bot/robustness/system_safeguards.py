"""
System Safeguards for BensBot Trading

This module provides system-wide safeguards, health checks, and error recovery mechanisms
to enhance the robustness of the entire trading system. It monitors the health of all
critical components and provides automatic recovery procedures.

Features:
- Component health monitoring
- Error detection and recovery
- Automated sanity checks
- Circuit breakers
- Performance monitoring and throttling
- State validation and auto-correction
"""

import logging
import threading
import time
import traceback
import json
import os
from typing import Dict, List, Any, Optional, Callable, Union, Set
from datetime import datetime, timedelta

# Safeguard states
class SafeguardState:
    """States for the safeguard system."""
    OPERATIONAL = "operational"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"

class ComponentType:
    """Types of monitored components."""
    POSITION_MANAGER = "position_manager"
    TRADE_ACCOUNTING = "trade_accounting"
    EXIT_MANAGER = "exit_manager"
    CAPITAL_ALLOCATOR = "capital_allocator"
    BROKER_MANAGER = "broker_manager"
    EVENT_SYSTEM = "event_system"
    MARKET_DATA = "market_data"
    DATABASE = "database"

logger = logging.getLogger(__name__)

class SystemSafeguards:
    """
    System-wide safeguards manager for the trading system.
    
    This class monitors all critical components, detects issues,
    and provides automatic recovery procedures to maintain 
    system integrity and prevent financial losses.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the safeguard system.
        
        Args:
            config: Configuration parameters
        """
        # Initialize state
        self.state = SafeguardState.OPERATIONAL
        self.config = config or {}
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Component health status
        self.component_health: Dict[str, Dict[str, Any]] = {}
        
        # Error counters
        self.error_counts: Dict[str, int] = {}
        self.error_timestamps: Dict[str, List[datetime]] = {}
        
        # Performance metrics
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Recovery metrics
        self.recovery_attempts: Dict[str, int] = {}
        self.last_recovery: Dict[str, datetime] = {}
        
        # Registered components
        self.registered_components: Dict[str, Any] = {}
        
        # Recovery functions
        self.recovery_functions: Dict[str, Callable] = {}
        
        # Validation functions
        self.validation_functions: Dict[str, Callable] = {}
        
        # Circuit breaker state
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Critical errors requiring manual intervention
        self.critical_errors: Set[str] = set()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load configuration
        self._load_configuration()
        
        logger.info("System safeguards initialized")
    
    def _load_configuration(self) -> None:
        """Load configuration parameters and set defaults."""
        # Default thresholds for circuit breakers
        self.error_threshold = self.config.get('error_threshold', 5)  # Errors before recovery
        self.recovery_attempt_limit = self.config.get('recovery_attempt_limit', 3)  # Max recovery attempts
        self.recovery_cooldown = self.config.get('recovery_cooldown_seconds', 300)  # 5 min cooldown
        
        # Default performance thresholds
        self.memory_threshold_mb = self.config.get('memory_threshold_mb', 1000)  # 1GB
        self.cpu_threshold_pct = self.config.get('cpu_threshold_pct', 80)  # 80% CPU
        
        # Monitoring interval
        self.monitoring_interval_seconds = self.config.get('monitoring_interval_seconds', 30)  # 30s
        
        # Critical errors requiring manual intervention
        critical_errors = self.config.get('critical_errors', [
            'DATABASE_CORRUPTION',
            'ORDER_EXECUTION_FAILURE',
            'POSITION_RECONCILIATION_FAILURE',
            'ACCOUNT_BALANCE_MISMATCH'
        ])
        self.critical_errors = set(critical_errors)
    
    def register_component(self, component_type: str, component: Any, 
                          validation_function: Optional[Callable] = None,
                          recovery_function: Optional[Callable] = None) -> None:
        """
        Register a component for monitoring.
        
        Args:
            component_type: Type of component
            component: The component instance
            validation_function: Function to validate component state
            recovery_function: Function to recover from errors
        """
        with self._lock:
            self.registered_components[component_type] = component
            
            # Initialize health status
            self.component_health[component_type] = {
                'status': SafeguardState.OPERATIONAL,
                'last_checked': datetime.now().isoformat(),
                'errors': [],
                'warnings': []
            }
            
            # Register recovery function if provided
            if recovery_function:
                self.recovery_functions[component_type] = recovery_function
            
            # Register validation function if provided
            if validation_function:
                self.validation_functions[component_type] = validation_function
            
            # Initialize error tracking
            self.error_counts[component_type] = 0
            self.error_timestamps[component_type] = []
            self.recovery_attempts[component_type] = 0
            
            # Initialize circuit breaker
            self.circuit_breakers[component_type] = {
                'tripped': False,
                'trip_time': None,
                'recovery_attempts': 0,
                'error_count': 0
            }
            
            logger.info(f"Registered component for monitoring: {component_type}")
    
    def start_monitoring(self) -> bool:
        """
        Start the safeguard monitoring system.
        
        Returns:
            bool: Success status
        """
        if self.monitoring_active:
            logger.warning("Safeguard monitoring already active")
            return True
            
        try:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="SystemSafeguards",
                daemon=True
            )
            self.monitoring_thread.start()
            logger.info("Started safeguard monitoring")
            return True
        except Exception as e:
            self.monitoring_active = False
            logger.error(f"Error starting safeguard monitoring: {str(e)}")
            return False
            
    def stop_monitoring(self) -> bool:
        """
        Stop the safeguard monitoring system.
        
        Returns:
            bool: Success status
        """
        if not self.monitoring_active:
            logger.warning("Safeguard monitoring not active")
            return True
            
        try:
            self.monitoring_active = False
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            logger.info("Stopped safeguard monitoring")
            return True
        except Exception as e:
            logger.error(f"Error stopping safeguard monitoring: {str(e)}")
            return False
            
    def _monitoring_loop(self) -> None:
        """Main health monitoring loop."""
        logger.info("Safeguard monitoring loop started")
        
        while self.monitoring_active:
            try:
                # Check system resources
                self._check_system_resources()
                
                # Check all registered components
                for component_type, component in self.registered_components.items():
                    try:
                        self._check_component(component_type, component)
                    except Exception as e:
                        logger.error(f"Error checking {component_type} health: {str(e)}")
                        # Record error but continue with other components
                        self._record_error(component_type, f"Health check error: {str(e)}")
                
                # Update overall system state
                self._update_system_state()
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(max(self.monitoring_interval_seconds, 60))  # Longer sleep on error
        
        logger.info("Safeguard monitoring loop terminated")
        
    def _check_system_resources(self) -> None:
        """Check system resources like CPU, memory and disk space."""
        try:
            # In a real implementation, we'd use psutil to check:
            # - CPU usage
            # - Memory usage
            # - Disk space
            # For now, we'll just simulate this check
            
            # Record system metrics
            self.performance_metrics['system'] = {
                'timestamp': datetime.now().isoformat(),
                'memory_usage_mb': 500,  # Example value
                'cpu_percent': 25,       # Example value 
                'disk_free_gb': 100      # Example value
            }
        except Exception as e:
            logger.error(f"Error checking system resources: {str(e)}")
            
    def _check_component(self, component_type: str, component: Any) -> None:
        """Check the health of a specific component."""
        # This would contain specific health checks for each component type
        # For now, we'll implement a simple is_alive check
        
        try:
            # Basic check: is component None?
            if component is None:
                self._record_error(component_type, "Component is None")
                return
                
            # Check if component has a specific health check method
            if hasattr(component, 'is_healthy') and callable(component.is_healthy):
                is_healthy, messages = component.is_healthy()
                if not is_healthy:
                    for msg in messages:
                        self._record_error(component_type, msg)
                    return
                    
            # If we got here, component is healthy
            self.component_health[component_type]['status'] = SafeguardState.OPERATIONAL
            
        except Exception as e:
            self._record_error(component_type, f"Health check exception: {str(e)}")
            
    def _record_error(self, component_type: str, error_message: str) -> None:
        """Record an error for a component."""
        with self._lock:
            # Add error to health record
            errors = self.component_health.get(component_type, {}).get('errors', [])
            errors.append({
                'message': error_message,
                'timestamp': datetime.now().isoformat()
            })
            
            # Limit error history
            max_errors = self.config.get('max_error_history', 20)
            if len(errors) > max_errors:
                errors = errors[-max_errors:]
                
            self.component_health[component_type]['errors'] = errors
            
            # Update status to warning or critical
            self.component_health[component_type]['status'] = SafeguardState.WARNING
            
            # Increment error count
            self.error_counts[component_type] = self.error_counts.get(component_type, 0) + 1
            
            # Add timestamp
            self.error_timestamps[component_type] = self.error_timestamps.get(component_type, [])
            self.error_timestamps[component_type].append(datetime.now())
            
            # Check if error threshold is exceeded
            if self.error_counts[component_type] >= self.error_threshold:
                self.component_health[component_type]['status'] = SafeguardState.CRITICAL
                self._attempt_recovery(component_type)
                
    def _attempt_recovery(self, component_type: str) -> bool:
        """Attempt to recover a faulty component."""
        with self._lock:
            # Check if we've hit the recovery attempt limit
            attempts = self.recovery_attempts.get(component_type, 0)
            if attempts >= self.recovery_attempt_limit:
                logger.warning(f"Recovery attempt limit reached for {component_type}")
                self._trip_circuit_breaker(component_type)
                return False
                
            # Check if we're in recovery cooldown period
            last_recovery = self.last_recovery.get(component_type)
            if last_recovery:
                cooldown_time = last_recovery + timedelta(seconds=self.recovery_cooldown)
                if datetime.now() < cooldown_time:
                    logger.info(f"In recovery cooldown for {component_type}, skipping")
                    return False
                    
            # Increment recovery attempts
            self.recovery_attempts[component_type] = attempts + 1
            self.last_recovery[component_type] = datetime.now()
            
            # Log recovery attempt
            logger.warning(f"Attempting recovery for {component_type}, attempt {attempts+1}")
            
            # Call recovery function if available
            if component_type in self.recovery_functions:
                try:
                    success = self.recovery_functions[component_type]()
                    if success:
                        logger.info(f"Recovery successful for {component_type}")
                        # Reset error count on successful recovery
                        self.error_counts[component_type] = 0
                        self.component_health[component_type]['status'] = SafeguardState.OPERATIONAL
                        return True
                    else:
                        logger.warning(f"Recovery failed for {component_type}")
                        return False
                except Exception as e:
                    logger.error(f"Error during recovery of {component_type}: {str(e)}")
                    return False
            else:
                logger.warning(f"No recovery function for {component_type}")
                return False
                
    def _trip_circuit_breaker(self, component_type: str) -> None:
        """Trip a circuit breaker to prevent further operations on a faulty component."""
        with self._lock:
            logger.critical(f"CIRCUIT BREAKER TRIPPED for {component_type}")
            
            # Set circuit breaker state
            self.circuit_breakers[component_type] = {
                'tripped': True,
                'trip_time': datetime.now().isoformat(),
                'recovery_attempts': self.recovery_attempts.get(component_type, 0),
                'error_count': self.error_counts.get(component_type, 0)
            }
            
            # Set component status to critical
            self.component_health[component_type]['status'] = SafeguardState.CRITICAL
            
    def _update_system_state(self) -> None:
        """Update the overall system state based on component health."""
        with self._lock:
            # Check if any component is in critical state
            critical_components = [
                comp_type for comp_type, health in self.component_health.items()
                if health.get('status') == SafeguardState.CRITICAL
            ]
            
            if critical_components:
                self.state = SafeguardState.CRITICAL
                logger.critical(f"System in CRITICAL state due to components: {', '.join(critical_components)}")
                return
                
            # Check if any component is in warning state
            warning_components = [
                comp_type for comp_type, health in self.component_health.items()
                if health.get('status') == SafeguardState.WARNING
            ]
            
            if warning_components:
                self.state = SafeguardState.WARNING
                logger.warning(f"System in WARNING state due to components: {', '.join(warning_components)}")
                return
                
            # All components operational
            self.state = SafeguardState.OPERATIONAL
            
    def get_system_health(self) -> Dict[str, Any]:
        """Get a comprehensive system health report."""
        with self._lock:
            return {
                'system_state': self.state,
                'components': self.component_health,
                'circuit_breakers': self.circuit_breakers,
                'performance': self.performance_metrics,
                'timestamp': datetime.now().isoformat()
            }
    
    def reset_circuit_breaker(self, component_type: str) -> bool:
        """
        Reset a tripped circuit breaker (manual operation).
        
        Args:
            component_type: Type of component
            
        Returns:
            bool: Success status
        """
        with self._lock:
            if component_type not in self.circuit_breakers:
                logger.warning(f"No circuit breaker for {component_type}")
                return False
                
            if not self.circuit_breakers[component_type].get('tripped', False):
                logger.info(f"Circuit breaker for {component_type} not tripped")
                return True
                
            # Reset circuit breaker
            self.circuit_breakers[component_type]['tripped'] = False
            self.circuit_breakers[component_type]['reset_time'] = datetime.now().isoformat()
            
            # Reset error and recovery counts
            self.error_counts[component_type] = 0
            self.recovery_attempts[component_type] = 0
            
            # Reset component status
            self.component_health[component_type]['status'] = SafeguardState.OPERATIONAL
            
            logger.info(f"Reset circuit breaker for {component_type}")
            return True
