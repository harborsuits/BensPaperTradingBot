"""
Process Manager

This module provides a robust process management system for the trading bot,
ensuring 24/7 operation with automatic recovery, monitoring, and status logging.
It handles process lifecycle, crash recovery, and system health checks.
"""

import logging
import os
import signal
import subprocess
import sys
import time
import threading
import traceback
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
import psutil

# Import our components
try:
    from trading_bot.orchestration.scheduler_service import SchedulerService, create_scheduler
    from trading_bot.orchestration.main_orchestrator import MainOrchestrator
    from trading_bot.core.service_registry import ServiceRegistry
except ImportError as e:
    logging.error(f"Error importing required components: {e}")
    raise

logger = logging.getLogger(__name__)

class ProcessManager:
    """
    Process Manager for ensuring 24/7 operation of the trading bot
    
    Handles:
    - Process lifecycle management
    - Automatic crash recovery
    - Health monitoring
    - Status reporting
    - Scheduled restarts
    """
    
    def __init__(self, config=None):
        """
        Initialize the process manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Process management settings
        self.process_config = self.config.get("process_manager", {})
        self.max_restarts = self.process_config.get("max_restarts", 5)
        self.restart_cooldown = self.process_config.get("restart_cooldown", 300)  # seconds
        self.health_check_interval = self.process_config.get("health_check_interval", 60)  # seconds
        self.status_log_interval = self.process_config.get("status_log_interval", 3600)  # seconds
        self.enable_watchdog = self.process_config.get("enable_watchdog", True)
        
        # Component references
        self.scheduler = None
        self.orchestrator = None
        
        # Process state
        self.running = False
        self.start_time = None
        self.restart_count = 0
        self.last_restart_time = None
        self.health_status = "initializing"
        self.last_health_check = None
        self.last_status_log = None
        
        # Control threads
        self.stop_event = threading.Event()
        self.health_check_thread = None
        self.watchdog_thread = None
        
        # Status tracking
        self.component_status = {}
        self.system_metrics = {}
        
        # Initialize logging directory
        self.log_dir = self.process_config.get("log_directory", "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        logger.info("Process Manager initialized")
    
    def start(self):
        """Start the process manager and all managed components"""
        if self.running:
            logger.warning("Process Manager is already running")
            return
        
        try:
            logger.info("Starting Process Manager")
            self.running = True
            self.start_time = datetime.now()
            self.stop_event.clear()
            
            # Start components
            self._start_components()
            
            # Start monitoring threads
            self._start_monitoring()
            
            # Log initial status
            self._log_status()
            
            logger.info("Process Manager started successfully")
            
        except Exception as e:
            logger.error(f"Error starting Process Manager: {e}")
            logger.error(traceback.format_exc())
            self.running = False
    
    def _start_components(self):
        """Start all managed components"""
        try:
            # Start orchestrator if not already running
            if not self.orchestrator:
                logger.info("Initializing orchestrator")
                config_path = self.config.get("config_path")
                self.orchestrator = MainOrchestrator(config_path=config_path)
            
            if not getattr(self.orchestrator, 'running', False):
                logger.info("Starting orchestrator")
                self.orchestrator.start()
            
            # Start scheduler if not already running
            if not self.scheduler:
                logger.info("Initializing scheduler")
                self.scheduler = create_scheduler(
                    config=self.config
                )
            
            if not getattr(self.scheduler, 'running', False):
                logger.info("Starting scheduler")
                self.scheduler.start()
            
            # Update component status
            self.component_status["orchestrator"] = {
                "running": getattr(self.orchestrator, 'running', False),
                "start_time": self.start_time,
                "status": "running" if getattr(self.orchestrator, 'running', False) else "stopped"
            }
            
            self.component_status["scheduler"] = {
                "running": getattr(self.scheduler, 'running', False),
                "start_time": self.start_time,
                "status": "running" if getattr(self.scheduler, 'running', False) else "stopped"
            }
            
        except Exception as e:
            logger.error(f"Error starting components: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _start_monitoring(self):
        """Start monitoring threads"""
        # Start health check thread
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop, 
            daemon=True
        )
        self.health_check_thread.start()
        
        # Start watchdog thread if enabled
        if self.enable_watchdog:
            self.watchdog_thread = threading.Thread(
                target=self._watchdog_loop, 
                daemon=True
            )
            self.watchdog_thread.start()
    
    def stop(self):
        """Stop the process manager and all managed components"""
        if not self.running:
            logger.warning("Process Manager is already stopped")
            return
        
        try:
            logger.info("Stopping Process Manager")
            
            # Signal threads to stop
            self.running = False
            self.stop_event.set()
            
            # Stop scheduler
            if self.scheduler and getattr(self.scheduler, 'running', False):
                logger.info("Stopping scheduler")
                self.scheduler.stop()
            
            # Stop orchestrator
            if self.orchestrator and getattr(self.orchestrator, 'running', False):
                logger.info("Stopping orchestrator")
                self.orchestrator.stop()
            
            # Wait for threads to stop
            if self.health_check_thread and self.health_check_thread.is_alive():
                self.health_check_thread.join(timeout=5.0)
            
            if self.watchdog_thread and self.watchdog_thread.is_alive():
                self.watchdog_thread.join(timeout=5.0)
            
            # Final status log
            self._log_status()
            
            logger.info("Process Manager stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping Process Manager: {e}")
            logger.error(traceback.format_exc())
    
    def _health_check_loop(self):
        """Health check monitoring loop"""
        logger.info("Health check monitoring started")
        
        while not self.stop_event.is_set():
            try:
                # Perform health check
                self._perform_health_check()
                
                # Log status at regular intervals
                now = datetime.now()
                if (not self.last_status_log or 
                    (now - self.last_status_log).total_seconds() >= self.status_log_interval):
                    self._log_status()
                    self.last_status_log = now
                
                # Sleep until next check
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                logger.error(traceback.format_exc())
                time.sleep(self.health_check_interval)
        
        logger.info("Health check monitoring stopped")
    
    def _perform_health_check(self):
        """Perform health check on all components"""
        now = datetime.now()
        self.last_health_check = now
        health_issues = []
        
        # Check orchestrator
        if self.orchestrator:
            try:
                orchestrator_running = getattr(self.orchestrator, 'running', False)
                self.component_status["orchestrator"]["running"] = orchestrator_running
                self.component_status["orchestrator"]["last_check"] = now
                
                if not orchestrator_running:
                    health_issues.append("Orchestrator not running")
            except Exception as e:
                logger.error(f"Error checking orchestrator health: {e}")
                health_issues.append(f"Orchestrator check error: {str(e)}")
        else:
            health_issues.append("Orchestrator not initialized")
        
        # Check scheduler
        if self.scheduler:
            try:
                scheduler_running = getattr(self.scheduler, 'running', False)
                self.component_status["scheduler"]["running"] = scheduler_running
                self.component_status["scheduler"]["last_check"] = now
                
                # Get scheduler status
                if hasattr(self.scheduler, 'get_schedule_status'):
                    scheduler_status = self.scheduler.get_schedule_status()
                    self.component_status["scheduler"]["status_details"] = scheduler_status
                
                if not scheduler_running:
                    health_issues.append("Scheduler not running")
            except Exception as e:
                logger.error(f"Error checking scheduler health: {e}")
                health_issues.append(f"Scheduler check error: {str(e)}")
        else:
            health_issues.append("Scheduler not initialized")
        
        # Collect system metrics
        try:
            self._collect_system_metrics()
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            health_issues.append(f"System metrics error: {str(e)}")
        
        # Check for memory issues
        if self.system_metrics.get("memory_percent", 0) > 90:
            health_issues.append(f"High memory usage: {self.system_metrics.get('memory_percent')}%")
        
        # Check for CPU issues
        if self.system_metrics.get("cpu_percent", 0) > 90:
            health_issues.append(f"High CPU usage: {self.system_metrics.get('cpu_percent')}%")
        
        # Update overall health status
        if not health_issues:
            self.health_status = "healthy"
        else:
            self.health_status = f"issues detected: {', '.join(health_issues)}"
            logger.warning(f"Health check issues: {', '.join(health_issues)}")
    
    def _collect_system_metrics(self):
        """Collect system metrics for monitoring"""
        process = psutil.Process(os.getpid())
        
        # Memory usage
        memory_info = process.memory_info()
        self.system_metrics["memory_rss"] = memory_info.rss / (1024 * 1024)  # MB
        self.system_metrics["memory_percent"] = process.memory_percent()
        
        # CPU usage
        self.system_metrics["cpu_percent"] = process.cpu_percent(interval=1.0)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.system_metrics["disk_percent"] = disk.percent
        
        # Thread count
        self.system_metrics["thread_count"] = threading.active_count()
        
        # Uptime
        self.system_metrics["uptime_seconds"] = (
            datetime.now() - self.start_time).total_seconds()
    
    def _watchdog_loop(self):
        """Watchdog monitoring loop for automatic recovery"""
        logger.info("Watchdog monitoring started")
        
        while not self.stop_event.is_set():
            try:
                # Check if we need to restart components
                if self.running:
                    self._check_and_restart_components()
                
                # Sleep until next check
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in watchdog loop: {e}")
                logger.error(traceback.format_exc())
                time.sleep(self.health_check_interval)
        
        logger.info("Watchdog monitoring stopped")
    
    def _check_and_restart_components(self):
        """Check components and restart if necessary"""
        now = datetime.now()
        
        # Check if we're in restart cooldown period
        if (self.last_restart_time and 
            (now - self.last_restart_time).total_seconds() < self.restart_cooldown):
            return
        
        # Check if we've exceeded max restarts
        if self.restart_count >= self.max_restarts:
            logger.warning(f"Max restarts ({self.max_restarts}) exceeded, not attempting further restarts")
            return
        
        # Check orchestrator
        if self.orchestrator and not getattr(self.orchestrator, 'running', False):
            logger.warning("Detected orchestrator not running, attempting restart")
            try:
                self.orchestrator.start()
                logger.info("Successfully restarted orchestrator")
                self.restart_count += 1
                self.last_restart_time = now
            except Exception as e:
                logger.error(f"Failed to restart orchestrator: {e}")
        
        # Check scheduler
        if self.scheduler and not getattr(self.scheduler, 'running', False):
            logger.warning("Detected scheduler not running, attempting restart")
            try:
                self.scheduler.start()
                logger.info("Successfully restarted scheduler")
                self.restart_count += 1
                self.last_restart_time = now
            except Exception as e:
                logger.error(f"Failed to restart scheduler: {e}")
    
    def _log_status(self):
        """Log current status to file"""
        try:
            now = datetime.now()
            self.last_status_log = now
            
            # Prepare status data
            status_data = {
                "timestamp": now.isoformat(),
                "uptime_seconds": (now - self.start_time).total_seconds() if self.start_time else 0,
                "health_status": self.health_status,
                "restart_count": self.restart_count,
                "last_restart": self.last_restart_time.isoformat() if self.last_restart_time else None,
                "components": self.component_status,
                "system_metrics": self.system_metrics
            }
            
            # Create log filename
            log_file = os.path.join(self.log_dir, f"status_{now.strftime('%Y%m%d')}.json")
            
            # Write to log file
            with open(log_file, 'a') as f:
                f.write(json.dumps(status_data) + "\n")
            
            logger.info(f"Status logged to {log_file}")
            
            # Also log summary to application log
            logger.info(f"Status: {self.health_status}, uptime: {status_data['uptime_seconds']/3600:.1f} hours, " +
                       f"restarts: {self.restart_count}")
            
        except Exception as e:
            logger.error(f"Error logging status: {e}")
            logger.error(traceback.format_exc())
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current process manager status
        
        Returns:
            Dict with current status information
        """
        now = datetime.now()
        
        # Perform a fresh health check
        self._perform_health_check()
        
        return {
            "running": self.running,
            "health_status": self.health_status,
            "uptime": str(now - self.start_time) if self.start_time else "Not started",
            "restart_count": self.restart_count,
            "last_restart": self.last_restart_time,
            "components": self.component_status,
            "system_metrics": self.system_metrics,
            "last_health_check": self.last_health_check,
            "max_restarts": self.max_restarts,
            "restarts_remaining": max(0, self.max_restarts - self.restart_count)
        }
    
    def restart_component(self, component_name: str) -> bool:
        """
        Manually restart a specific component
        
        Args:
            component_name: Name of component to restart ('orchestrator' or 'scheduler')
            
        Returns:
            bool: Success status
        """
        try:
            if component_name == "orchestrator" and self.orchestrator:
                logger.info("Manually restarting orchestrator")
                if getattr(self.orchestrator, 'running', False):
                    self.orchestrator.stop()
                time.sleep(2)  # Brief pause
                self.orchestrator.start()
                return True
                
            elif component_name == "scheduler" and self.scheduler:
                logger.info("Manually restarting scheduler")
                if getattr(self.scheduler, 'running', False):
                    self.scheduler.stop()
                time.sleep(2)  # Brief pause
                self.scheduler.start()
                return True
                
            else:
                logger.warning(f"Unknown component: {component_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error restarting {component_name}: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def reset_restart_count(self):
        """Reset the restart counter"""
        self.restart_count = 0
        logger.info("Restart count reset to 0")


# Main entry point
def main():
    """Main entry point for running the process manager"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Trading Bot Process Manager")
    parser.add_argument("--config", dest="config_path", help="Path to configuration file")
    parser.add_argument("--log-level", dest="log_level", default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="Set the logging level")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("process_manager.log")
        ]
    )
    
    # Create config dictionary
    config = {"config_path": args.config_path} if args.config_path else {}
    
    # Create and start process manager
    process_manager = ProcessManager(config=config)
    
    # Register signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down")
        process_manager.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the process manager
    process_manager.start()
    
    # Main thread just waits
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
        process_manager.stop()
    except Exception as e:
        logger.error(f"Error in main thread: {e}")
        logger.error(traceback.format_exc())
        process_manager.stop()


if __name__ == "__main__":
    main()
