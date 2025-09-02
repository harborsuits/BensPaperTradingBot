#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Watchdog and fault tolerance system for BensBot
Monitors critical services and automatically recovers from failures
"""

import logging
import threading
import time
from typing import Dict, Any, Callable, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import traceback

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Possible statuses for monitored services"""
    UNKNOWN = "UNKNOWN"
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    RECOVERING = "RECOVERING"
    FAILED = "FAILED"

class RecoveryStrategy(Enum):
    """Recovery strategies for services"""
    RESTART = "RESTART"          # Restart the service
    RECONNECT = "RECONNECT"      # Reconnect to external service
    RELOAD_STATE = "RELOAD_STATE"  # Reload from saved state
    FAILOVER = "FAILOVER"        # Switch to backup system
    CUSTOM = "CUSTOM"            # Use custom recovery function

class ServiceMonitor:
    """Monitor for a single service"""
    
    def __init__(self, name: str, 
                health_check: Callable[[], bool], 
                recovery_action: Callable[[], bool],
                recovery_strategy: RecoveryStrategy = RecoveryStrategy.RESTART,
                check_interval_seconds: int = 30,
                max_failures: int = 3,
                cooldown_seconds: int = 300,
                dependencies: List[str] = None):
        """
        Initialize a service monitor
        
        Args:
            name: Service name
            health_check: Function that returns True if service is healthy
            recovery_action: Function to recover service, returns True if successful
            recovery_strategy: Type of recovery strategy
            check_interval_seconds: How often to check health (seconds)
            max_failures: Number of failures before recovery is triggered
            cooldown_seconds: Cooldown period after recovery
            dependencies: List of dependent service names
        """
        self.name = name
        self.health_check = health_check
        self.recovery_action = recovery_action
        self.recovery_strategy = recovery_strategy
        self.check_interval = check_interval_seconds
        self.max_failures = max_failures
        self.cooldown_seconds = cooldown_seconds
        self.dependencies = dependencies or []
        
        # Internal state
        self.failures = 0
        self.recovery_attempts = 0
        self.last_failure = None
        self.last_recovery = None
        self.status = ServiceStatus.UNKNOWN
        self.history = []  # Service status history
    
    def check_health(self) -> bool:
        """
        Run health check and update internal state
        
        Returns:
            bool: True if service is healthy
        """
        try:
            is_healthy = self.health_check()
            
            if is_healthy:
                old_status = self.status
                self.status = ServiceStatus.HEALTHY
                self.failures = 0
                
                # Log recovery if status improved
                if old_status in [ServiceStatus.UNHEALTHY, ServiceStatus.DEGRADED, 
                                ServiceStatus.RECOVERING, ServiceStatus.FAILED]:
                    logger.info(f"Service {self.name} recovered to HEALTHY state")
            else:
                self.failures += 1
                self.last_failure = datetime.now()
                
                # Update status based on failure count
                if self.failures >= self.max_failures:
                    self.status = ServiceStatus.UNHEALTHY
                else:
                    self.status = ServiceStatus.DEGRADED
                    
                logger.warning(f"Service {self.name} health check failed "
                             f"({self.failures}/{self.max_failures}, status: {self.status.value})")
                
            # Track history (limit to 100 entries)
            self.history.append({
                'timestamp': datetime.now(),
                'status': self.status.value,
                'failures': self.failures
            })
            
            if len(self.history) > 100:
                self.history.pop(0)
                
            return is_healthy
        except Exception as e:
            logger.error(f"Error checking health of service {self.name}: {str(e)}")
            self.failures += 1
            self.last_failure = datetime.now()
            self.status = ServiceStatus.FAILED
            
            # Track history
            self.history.append({
                'timestamp': datetime.now(),
                'status': self.status.value,
                'failures': self.failures,
                'error': str(e)
            })
            
            if len(self.history) > 100:
                self.history.pop(0)
                
            return False
    
    def attempt_recovery(self) -> bool:
        """
        Attempt to recover the service
        
        Returns:
            bool: True if recovery successful
        """
        # Don't attempt recovery if in cooldown period
        if (self.last_recovery and 
            (datetime.now() - self.last_recovery).total_seconds() < self.cooldown_seconds):
            logger.info(f"Service {self.name} in cooldown period, skipping recovery")
            return False
            
        # Don't attempt recovery if not needed
        if self.status == ServiceStatus.HEALTHY:
            return True
            
        # Don't attempt recovery if not unhealthy enough
        if self.failures < self.max_failures:
            return False
            
        # Update status and attempt recovery
        old_status = self.status
        self.status = ServiceStatus.RECOVERING
        self.recovery_attempts += 1
        self.last_recovery = datetime.now()
        
        logger.warning(f"Attempting recovery of service {self.name} "
                     f"(attempt #{self.recovery_attempts}, strategy: {self.recovery_strategy.value})")
        
        try:
            # Execute recovery action
            recovery_successful = self.recovery_action()
            
            if recovery_successful:
                self.failures = 0
                self.status = ServiceStatus.HEALTHY
                logger.info(f"Recovery of service {self.name} successful")
                
                # Track history
                self.history.append({
                    'timestamp': datetime.now(),
                    'status': self.status.value,
                    'recovery_attempt': self.recovery_attempts,
                    'success': True
                })
                
                if len(self.history) > 100:
                    self.history.pop(0)
                    
                return True
            else:
                self.status = ServiceStatus.FAILED
                logger.error(f"Recovery of service {self.name} failed")
                
                # Track history
                self.history.append({
                    'timestamp': datetime.now(),
                    'status': self.status.value,
                    'recovery_attempt': self.recovery_attempts,
                    'success': False
                })
                
                if len(self.history) > 100:
                    self.history.pop(0)
                    
                return False
        except Exception as e:
            self.status = ServiceStatus.FAILED
            logger.error(f"Error during recovery of service {self.name}: {str(e)}")
            
            # Track history
            self.history.append({
                'timestamp': datetime.now(),
                'status': self.status.value,
                'recovery_attempt': self.recovery_attempts,
                'success': False,
                'error': str(e)
            })
            
            if len(self.history) > 100:
                self.history.pop(0)
                
            return False
    
    def get_status_report(self) -> Dict[str, Any]:
        """
        Get a detailed status report
        
        Returns:
            dict: Status report with details
        """
        return {
            'name': self.name,
            'status': self.status.value,
            'failures': self.failures,
            'recovery_attempts': self.recovery_attempts,
            'recovery_strategy': self.recovery_strategy.value,
            'last_failure': self.last_failure.isoformat() if self.last_failure else None,
            'last_recovery': self.last_recovery.isoformat() if self.last_recovery else None,
            'dependencies': self.dependencies,
            'history': self.history[-5:] if self.history else []  # Last 5 entries
        }

class ServiceWatchdog:
    """
    Watchdog system that monitors and recovers critical services
    """
    
    def __init__(self, check_interval_seconds: int = 30, 
                persistence_manager=None):
        """
        Initialize the watchdog
        
        Args:
            check_interval_seconds: Default interval for health checks
            persistence_manager: Optional persistence manager for state storage
        """
        self.services = {}
        self.check_interval = check_interval_seconds
        self.running = False
        self.watchdog_thread = None
        self.persistence_manager = persistence_manager
        self.last_health_check = None
        self.start_time = None
        
    def register_service(self, name: str, 
                        health_check: Callable[[], bool], 
                        recovery_action: Callable[[], bool], 
                        recovery_strategy: RecoveryStrategy = RecoveryStrategy.RESTART,
                        check_interval_seconds: Optional[int] = None,
                        max_failures: int = 3,
                        cooldown_seconds: int = 300,
                        dependencies: List[str] = None) -> None:
        """
        Register a service to be monitored
        
        Args:
            name: Service name
            health_check: Function that returns True if service is healthy
            recovery_action: Function to recover service, returns True if successful
            recovery_strategy: Type of recovery strategy
            check_interval_seconds: How often to check health (seconds)
            max_failures: Number of failures before recovery is triggered
            cooldown_seconds: Cooldown period after recovery
            dependencies: List of dependent service names
        """
        # Use default check interval if not specified
        interval = check_interval_seconds if check_interval_seconds is not None else self.check_interval
        
        monitor = ServiceMonitor(
            name=name,
            health_check=health_check,
            recovery_action=recovery_action,
            recovery_strategy=recovery_strategy,
            check_interval_seconds=interval,
            max_failures=max_failures,
            cooldown_seconds=cooldown_seconds,
            dependencies=dependencies
        )
        
        self.services[name] = monitor
        logger.info(f"Registered service {name} with watchdog (strategy: {recovery_strategy.value})")
        
    def start(self) -> None:
        """Start the watchdog monitoring thread"""
        if self.running:
            logger.warning("Watchdog already running")
            return
            
        self.running = True
        self.start_time = datetime.now()
        self.watchdog_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="ServiceWatchdog"
        )
        self.watchdog_thread.start()
        logger.info("Watchdog monitoring started")
        
        # Log to persistence if available
        if self.persistence_manager:
            self.persistence_manager.log_system_event(
                level="INFO",
                message="Watchdog monitoring started",
                component="ServiceWatchdog"
            )
        
    def stop(self) -> None:
        """Stop the watchdog monitoring thread"""
        if not self.running:
            logger.warning("Watchdog not running")
            return
            
        self.running = False
        if self.watchdog_thread:
            self.watchdog_thread.join(timeout=5.0)
            logger.info("Watchdog stopped")
            
            # Log to persistence if available
            if self.persistence_manager:
                self.persistence_manager.log_system_event(
                    level="INFO",
                    message="Watchdog monitoring stopped",
                    component="ServiceWatchdog"
                )
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop that checks services periodically"""
        next_check_times = {}
        
        while self.running:
            try:
                current_time = datetime.now()
                self.last_health_check = current_time
                
                # Check services that are due
                for name, service in self.services.items():
                    # Skip if not time to check yet
                    if (name in next_check_times and 
                        current_time < next_check_times[name]):
                        continue
                        
                    # Set next check time
                    next_check_times[name] = current_time + timedelta(seconds=service.check_interval)
                    
                    # Check dependencies first
                    dependencies_healthy = True
                    for dep_name in service.dependencies:
                        if dep_name not in self.services:
                            logger.warning(f"Service {name} depends on unknown service {dep_name}")
                            continue
                            
                        dep_service = self.services[dep_name]
                        if dep_service.status != ServiceStatus.HEALTHY:
                            logger.warning(f"Service {name} depends on unhealthy service {dep_name}")
                            dependencies_healthy = False
                            
                    # Skip health check if dependencies are unhealthy
                    if not dependencies_healthy:
                        logger.warning(f"Skipping health check for {name} due to unhealthy dependencies")
                        continue
                    
                    # Run health check
                    is_healthy = service.check_health()
                    
                    # If unhealthy and reached failure threshold, attempt recovery
                    if (not is_healthy and 
                        service.failures >= service.max_failures):
                        logger.warning(f"Service {name} reached failure threshold, attempting recovery")
                        service.attempt_recovery()
                        
                        # Log to persistence if available
                        if self.persistence_manager:
                            self.persistence_manager.log_system_event(
                                level="WARNING",
                                message=f"Recovery attempted for service {name}",
                                component="ServiceWatchdog",
                                additional_data={
                                    "service": name,
                                    "failures": service.failures,
                                    "recovery_attempts": service.recovery_attempts
                                }
                            )
                
                # Calculate sleep time until next check
                if next_check_times:
                    # Find the next service due for a check
                    next_check = min(next_check_times.values())
                    sleep_time = (next_check - datetime.now()).total_seconds()
                    sleep_time = max(0.1, min(sleep_time, self.check_interval))
                else:
                    sleep_time = self.check_interval
                    
                # Sleep until next check
                time.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Error in watchdog monitoring loop: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Log to persistence if available
                if self.persistence_manager:
                    self.persistence_manager.log_system_event(
                        level="ERROR",
                        message=f"Watchdog monitoring error: {str(e)}",
                        component="ServiceWatchdog",
                        additional_data={
                            "traceback": traceback.format_exc()
                        }
                    )
                    
                # Sleep briefly before retrying
                time.sleep(5)
    
    def get_service_status(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific service
        
        Args:
            name: Service name
            
        Returns:
            dict or None: Service status report or None if not found
        """
        if name not in self.services:
            return None
            
        return self.services[name].get_status_report()
    
    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all services
        
        Returns:
            dict: Dictionary mapping service names to status reports
        """
        return {name: service.get_status_report() 
                for name, service in self.services.items()}
    
    def force_recovery(self, name: str) -> bool:
        """
        Force recovery of a service regardless of status
        
        Args:
            name: Service name
            
        Returns:
            bool: True if recovery successful
        """
        if name not in self.services:
            logger.warning(f"Cannot force recovery: Service {name} not found")
            return False
            
        service = self.services[name]
        
        # Set failures to threshold to force recovery
        service.failures = service.max_failures
        
        # Attempt recovery
        logger.info(f"Forcing recovery of service {name}")
        return service.attempt_recovery()
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """
        Get summary of overall system health
        
        Returns:
            dict: Health summary with counts and details
        """
        status_counts = {status.value: 0 for status in ServiceStatus}
        
        for service in self.services.values():
            status_counts[service.status.value] += 1
            
        # Determine overall system status
        overall_status = ServiceStatus.HEALTHY
        
        if status_counts[ServiceStatus.FAILED.value] > 0:
            overall_status = ServiceStatus.DEGRADED
            
        if (status_counts[ServiceStatus.HEALTHY.value] == 0 and 
            len(self.services) > 0):
            overall_status = ServiceStatus.UNHEALTHY
            
        return {
            'overall_status': overall_status.value,
            'service_count': len(self.services),
            'status_counts': status_counts,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'last_check': self.last_health_check.isoformat() if self.last_health_check else None
        }


# Utility functions for common health checks

def check_thread_alive(thread: threading.Thread) -> bool:
    """Check if a thread is alive"""
    return thread is not None and thread.is_alive()

def check_api_connection(api_client, test_method_name: str, *args, **kwargs) -> bool:
    """
    Check if an API connection is working
    
    Args:
        api_client: API client object
        test_method_name: Name of method to call to test connection
        *args, **kwargs: Arguments to pass to test method
        
    Returns:
        bool: True if connection working
    """
    try:
        test_method = getattr(api_client, test_method_name)
        test_method(*args, **kwargs)
        return True
    except Exception as e:
        logger.error(f"API connection check failed: {str(e)}")
        return False

def check_file_writeable(filepath: str) -> bool:
    """Check if a file is writeable"""
    try:
        with open(filepath, 'a') as f:
            return True
    except Exception:
        return False

def check_memory_usage(threshold_mb: float = 1000.0) -> bool:
    """
    Check if memory usage is below threshold
    
    Args:
        threshold_mb: Memory threshold in MB
        
    Returns:
        bool: True if memory usage below threshold
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        return memory_mb < threshold_mb
    except ImportError:
        logger.warning("psutil not installed, memory check not possible")
        return True
    except Exception as e:
        logger.error(f"Memory check failed: {str(e)}")
        return False

def check_cpu_usage(threshold_percent: float = 90.0) -> bool:
    """
    Check if CPU usage is below threshold
    
    Args:
        threshold_percent: CPU threshold percentage
        
    Returns:
        bool: True if CPU usage below threshold
    """
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return cpu_percent < threshold_percent
    except ImportError:
        logger.warning("psutil not installed, CPU check not possible")
        return True
    except Exception as e:
        logger.error(f"CPU check failed: {str(e)}")
        return False
