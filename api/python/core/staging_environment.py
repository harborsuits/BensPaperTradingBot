"""
Staging Environment for strategy testing before live deployment.

This module serves as the entry point for activating and configuring the 
complete staging environment, integrating all the staging-related components.
"""
import os
import logging
import threading
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import json

from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.core.event_bus import EventBus
from trading_bot.core.constants import EventType
from trading_bot.core.staging_mode_manager import StagingModeManager
from trading_bot.core.system_health_monitor import SystemHealthMonitor
from trading_bot.core.risk_violation_detector import RiskViolationDetector
from trading_bot.core.staging_report_generator import StagingReport

logger = logging.getLogger(__name__)

class StagingEnvironment:
    """
    Main staging environment that integrates all staging-related components.
    
    Provides a comprehensive testing environment for strategies before
    they're promoted to live trading, with enhanced monitoring, reporting,
    and validation capabilities.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the staging environment.
        
        Args:
            config_path: Path to staging configuration file
        """
        self.service_registry = ServiceRegistry.get_instance()
        self.event_bus = self.service_registry.get_service("event_bus")
        
        if not self.event_bus:
            self.event_bus = EventBus()
            self.service_registry.register_service("event_bus", self.event_bus)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize subcomponents
        self.mode_manager = None
        self.health_monitor = None
        self.risk_detector = None
        self.report_generator = None
        
        # Reporting thread
        self.reporting_thread = None
        self.running = False
        self.report_frequency_hours = self.config.get("reporting_frequency_hours", 24)
        
        # Last report time
        self.last_report_time = datetime.now()
        
        # Register service
        self.service_registry.register_service("staging_environment", self)
        
        logger.info("Staging environment initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load staging configuration from file or use defaults."""
        default_config = {
            "test_duration_days": 14,
            "min_trades_required": 30,
            "reporting_frequency_hours": 24,
            "memory_alert_threshold_mb": 500,
            "cpu_alert_threshold_pct": 70,
            "max_acceptable_error_rate": 0.01,
            "risk_tolerance_multiplier": 0.8,
            "enable_stress_testing": True,
            "reports_directory": "./reports/staging",
            "validation_checkpoints": {
                "min_sharpe_ratio": 0.8,
                "max_drawdown_pct": -10.0,
                "min_win_rate": 0.4,
                "max_daily_loss_pct": -3.0,
                "resource_utilization_threshold": 80,
                "min_profit_factor": 1.2
            }
        }
        
        if not config_path:
            logger.info("Using default staging configuration")
            return default_config
            
        try:
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                
            # Merge with defaults to ensure all keys exist
            merged_config = {**default_config, **custom_config}
            
            # If custom validation checkpoints provided, merge those too
            if "validation_checkpoints" in custom_config:
                merged_config["validation_checkpoints"] = {
                    **default_config["validation_checkpoints"],
                    **custom_config["validation_checkpoints"]
                }
                
            logger.info(f"Loaded custom staging configuration from {config_path}")
            return merged_config
        except Exception as e:
            logger.error(f"Error loading staging configuration: {str(e)}")
            return default_config
    
    def activate(self) -> bool:
        """
        Activate the staging environment.
        
        This initializes and starts all the staging components and
        forces all strategies to use paper trading.
        
        Returns:
            bool: Success status
        """
        try:
            logger.info("Activating staging environment")
            
            # Initialize mode manager
            self.mode_manager = StagingModeManager(self.config)
            
            # Initialize health monitor
            self.health_monitor = SystemHealthMonitor(self.config)
            
            # Initialize risk detector
            self.risk_detector = RiskViolationDetector(self.config)
            
            # Initialize report generator
            self.report_generator = StagingReport(self.config)
            
            # Register health monitor alert callback
            self.health_monitor.register_alert_callback(self._on_health_alert)
            
            # Register risk detector alert callback
            self.risk_detector.register_alert_callback(self._on_risk_alert)
            
            # Enable staging mode
            self.mode_manager.enable_staging_mode()
            
            # Start health monitoring
            self.health_monitor.start_monitoring()
            
            # Start reporting thread
            self._start_reporting_thread()
            
            logger.info("Staging environment activated successfully")
            return True
        except Exception as e:
            logger.error(f"Error activating staging environment: {str(e)}")
            self.deactivate()
            return False
    
    def deactivate(self) -> None:
        """Deactivate the staging environment and restore normal operation."""
        logger.info("Deactivating staging environment")
        
        # Stop reporting thread
        self._stop_reporting_thread()
        
        # Stop health monitoring
        if self.health_monitor:
            self.health_monitor.stop_monitoring()
        
        # Generate final report
        if self.report_generator:
            try:
                logger.info("Generating final comprehensive report")
                self.report_generator.generate_comprehensive_report()
            except Exception as e:
                logger.error(f"Error generating final report: {str(e)}")
        
        # Disable staging mode
        if self.mode_manager:
            self.mode_manager.disable_staging_mode()
        
        logger.info("Staging environment deactivated")
    
    def _start_reporting_thread(self) -> None:
        """Start the periodic reporting thread."""
        if self.running:
            return
            
        self.running = True
        self.reporting_thread = threading.Thread(
            target=self._reporting_loop,
            daemon=True
        )
        self.reporting_thread.start()
        logger.info("Periodic reporting thread started")
    
    def _stop_reporting_thread(self) -> None:
        """Stop the periodic reporting thread."""
        self.running = False
        if self.reporting_thread:
            self.reporting_thread.join(timeout=5)
            self.reporting_thread = None
    
    def _reporting_loop(self) -> None:
        """Main reporting loop that runs in a separate thread."""
        while self.running:
            try:
                # Check if it's time for a new report
                now = datetime.now()
                hours_since_last = (now - self.last_report_time).total_seconds() / 3600
                
                if hours_since_last >= self.report_frequency_hours:
                    logger.info("Generating periodic staging report")
                    
                    # Generate reports
                    self.report_generator.generate_daily_report()
                    
                    # Update last report time
                    self.last_report_time = now
                    
                    # If mode manager indicates we've met minimum duration,
                    # also generate a comprehensive report
                    if self.mode_manager.has_met_minimum_duration():
                        logger.info("Minimum duration met, generating comprehensive report")
                        self.report_generator.generate_comprehensive_report()
                
                # Sleep for a bit - check every 15 minutes
                time.sleep(15 * 60)
            except Exception as e:
                logger.error(f"Error in reporting loop: {str(e)}")
                time.sleep(15 * 60)
    
    def _on_health_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        """
        Handle health monitoring alerts.
        
        Args:
            alert_type: Type of alert
            data: Alert data
        """
        logger.warning(f"Health alert: {alert_type}")
        
        # Publish event to event bus
        if self.event_bus:
            self.event_bus.publish(
                EventType.SYSTEM_ALERT,
                {
                    "alert_type": f"health_{alert_type}",
                    "severity": "warning",
                    "timestamp": datetime.now().isoformat(),
                    "data": data
                }
            )
    
    def _on_risk_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        """
        Handle risk violation alerts.
        
        Args:
            alert_type: Type of alert
            data: Alert data
        """
        logger.warning(f"Risk violation alert: {alert_type}")
        
        # Publish event to event bus
        if self.event_bus:
            self.event_bus.publish(
                EventType.RISK_ALERT,
                {
                    "alert_type": f"risk_{alert_type}",
                    "severity": "warning",
                    "timestamp": datetime.now().isoformat(),
                    "data": data
                }
            )
    
    def get_staging_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the staging environment.
        
        Returns:
            Dict: Staging status
        """
        if not self.mode_manager:
            return {"active": False}
        
        # Get staging duration
        staging_duration = self.mode_manager.get_staging_duration()
        
        # Get mode manager status
        has_met_duration = self.mode_manager.has_met_minimum_duration()
        
        # Get health status
        health_status = {}
        if self.health_monitor:
            health_status = self.health_monitor.get_latest_metrics()
        
        # Get risk status
        risk_status = {}
        if self.risk_detector:
            risk_status = {
                "active_violations": self.risk_detector.get_active_violations_count()
            }
        
        return {
            "active": self.mode_manager.is_in_staging_mode(),
            "duration": {
                "days": staging_duration.days,
                "hours": staging_duration.seconds // 3600,
                "total_seconds": staging_duration.total_seconds()
            },
            "has_met_minimum_duration": has_met_duration,
            "health_status": health_status,
            "risk_status": risk_status,
            "last_report_time": self.last_report_time.isoformat(),
            "next_report_in_hours": max(0, self.report_frequency_hours - 
                                       (datetime.now() - self.last_report_time).total_seconds() / 3600)
        }
    
    def run_stress_test(self, test_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a stress test on the system.
        
        Args:
            test_type: Type of stress test
            params: Test parameters
            
        Returns:
            Dict: Test results
        """
        if not self.config.get("enable_stress_testing", True):
            return {"error": "Stress testing is disabled in configuration"}
        
        logger.info(f"Running stress test: {test_type}")
        
        if test_type == "market_volatility":
            return self._run_volatility_stress_test(params)
        elif test_type == "high_frequency":
            return self._run_frequency_stress_test(params)
        elif test_type == "concurrent_signals":
            return self._run_concurrent_signals_test(params)
        elif test_type == "network_disruption":
            return self._run_network_disruption_test(params)
        else:
            return {"error": f"Unknown stress test type: {test_type}"}
    
    def _run_volatility_stress_test(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a market volatility stress test.
        
        This would involve simulating rapidly changing market conditions.
        
        Args:
            params: Test parameters
            
        Returns:
            Dict: Test results
        """
        # This would be implemented based on your specific market data simulation capabilities
        # For now, just return a placeholder
        return {
            "test_type": "market_volatility",
            "status": "not_implemented",
            "message": "Market volatility stress test not implemented yet"
        }
    
    def _run_frequency_stress_test(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a high frequency trading stress test.
        
        This would involve simulating rapid order execution and updates.
        
        Args:
            params: Test parameters
            
        Returns:
            Dict: Test results
        """
        # This would be implemented based on your specific order simulation capabilities
        return {
            "test_type": "high_frequency",
            "status": "not_implemented",
            "message": "High frequency trading stress test not implemented yet"
        }
    
    def _run_concurrent_signals_test(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a concurrent signals stress test.
        
        This would involve simulating multiple strategies generating signals simultaneously.
        
        Args:
            params: Test parameters
            
        Returns:
            Dict: Test results
        """
        # This would be implemented based on your specific signal generation capabilities
        return {
            "test_type": "concurrent_signals",
            "status": "not_implemented",
            "message": "Concurrent signals stress test not implemented yet"
        }
    
    def _run_network_disruption_test(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a network disruption stress test.
        
        This would involve simulating connectivity issues.
        
        Args:
            params: Test parameters
            
        Returns:
            Dict: Test results
        """
        # This would be implemented based on your specific network simulation capabilities
        return {
            "test_type": "network_disruption",
            "status": "not_implemented",
            "message": "Network disruption stress test not implemented yet"
        }


def create_staging_environment(config_path: Optional[str] = None) -> StagingEnvironment:
    """
    Factory function to create and activate a staging environment.
    
    Args:
        config_path: Path to staging configuration file
        
    Returns:
        StagingEnvironment: Activated staging environment
    """
    env = StagingEnvironment(config_path)
    env.activate()
    return env
