#!/usr/bin/env python3
"""
Trading Bot Monitoring Service

This service monitors the trading bot system and ensures it's running properly.
It performs regular health checks and can restart components if necessary.
"""

import os
import sys
import time
import signal
import logging
import argparse
import threading
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("monitoring_service")

class TradingBotMonitor:
    """
    Monitors the trading bot system components and ensures they are running properly.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        check_interval: int = 60
    ):
        """
        Initialize the monitor.
        
        Args:
            config: Configuration dictionary
            check_interval: Interval between health checks in seconds
        """
        self.config = config
        self.check_interval = check_interval
        self.running = False
        self.monitor_thread = None
        
        # Initialize metrics
        self.last_check_time = {}
        self.component_status = {}
        self.failure_count = {}
        
        # Maximum consecutive failures before taking action
        self.max_failures = config.get('max_failures', 3)
        
        # Endpoints to monitor
        self.endpoints = config.get('endpoints', [])
        
        # Notification settings
        self.notify_url = config.get('notification_webhook')
        
        logger.info(f"Initialized monitoring for {len(self.endpoints)} endpoints")
    
    def start(self):
        """Start the monitoring service."""
        if self.running:
            logger.warning("Monitoring service already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="monitor-thread"
        )
        self.monitor_thread.start()
        logger.info("Started monitoring service")
    
    def stop(self):
        """Stop the monitoring service."""
        if not self.running:
            logger.warning("Monitoring service not running")
            return
        
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=30)
            logger.info("Stopped monitoring service")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._check_all_components()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}", exc_info=True)
            
            # Sleep until next check
            time.sleep(self.check_interval)
    
    def _check_all_components(self):
        """Check all components and take action if necessary."""
        for endpoint in self.endpoints:
            name = endpoint.get('name', endpoint.get('url', 'unknown'))
            url = endpoint.get('url')
            
            if not url:
                logger.warning(f"No URL specified for endpoint {name}, skipping")
                continue
            
            try:
                status, message = self._check_endpoint(endpoint)
                self.last_check_time[name] = datetime.now()
                
                if status:
                    # Reset failure count on success
                    self.failure_count[name] = 0
                    self.component_status[name] = 'healthy'
                    logger.info(f"Component {name} is healthy")
                else:
                    # Increment failure count
                    self.failure_count[name] = self.failure_count.get(name, 0) + 1
                    self.component_status[name] = 'unhealthy'
                    
                    failures = self.failure_count[name]
                    logger.warning(f"Component {name} is unhealthy: {message} (failure {failures}/{self.max_failures})")
                    
                    # Take action if too many failures
                    if failures >= self.max_failures:
                        self._handle_component_failure(name, endpoint, message)
                        
            except Exception as e:
                logger.error(f"Error checking component {name}: {str(e)}", exc_info=True)
    
    def _check_endpoint(self, endpoint: Dict[str, Any]) -> tuple[bool, str]:
        """
        Check a single endpoint.
        
        Args:
            endpoint: Endpoint configuration
            
        Returns:
            Tuple of (is_healthy, message)
        """
        name = endpoint.get('name', endpoint.get('url', 'unknown'))
        url = endpoint.get('url')
        method = endpoint.get('method', 'GET')
        timeout = endpoint.get('timeout', 10)
        headers = endpoint.get('headers', {})
        
        # Expected status code
        expected_status = endpoint.get('expected_status', 200)
        
        try:
            logger.debug(f"Checking endpoint {name} at {url}")
            
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                timeout=timeout
            )
            
            if response.status_code != expected_status:
                return False, f"Unexpected status code: {response.status_code}"
            
            # Check for specific content
            if 'expected_content' in endpoint:
                expected_content = endpoint['expected_content']
                if expected_content not in response.text:
                    return False, f"Expected content not found: {expected_content}"
            
            return True, "Healthy"
            
        except requests.RequestException as e:
            return False, f"Request failed: {str(e)}"
    
    def _handle_component_failure(self, name: str, endpoint: Dict[str, Any], message: str):
        """
        Handle a component failure.
        
        Args:
            name: Component name
            endpoint: Endpoint configuration
            message: Error message
        """
        logger.error(f"Component {name} failed too many times: {message}")
        
        # Send notification
        self._send_notification(
            title=f"Trading Bot Component Failure: {name}",
            message=f"Component {name} has failed health checks {self.failure_count[name]} times. Last error: {message}",
            severity="critical"
        )
        
        # Take action based on configuration
        action = endpoint.get('failure_action', 'notify')
        
        if action == 'restart':
            service_name = endpoint.get('service_name')
            if service_name:
                self._restart_service(service_name)
            else:
                logger.error(f"Cannot restart service for {name}: no service_name specified")
        
        # Reset failure count after taking action
        self.failure_count[name] = 0
    
    def _restart_service(self, service_name: str):
        """
        Restart a Kubernetes service.
        
        Args:
            service_name: Name of the service to restart
        """
        logger.info(f"Attempting to restart service {service_name}")
        
        try:
            import kubernetes.client
            from kubernetes.client.rest import ApiException
            
            # Load kube config
            kubernetes.config.load_incluster_config()
            
            # Create API client
            api = kubernetes.client.AppsV1Api()
            
            # Get deployment
            namespace = os.environ.get('NAMESPACE', 'default')
            deployment_name = f"trading-bot-{service_name}"
            
            logger.info(f"Restarting deployment {deployment_name} in namespace {namespace}")
            
            # Patch the deployment to trigger a rollout restart
            # This is done by adding/updating an annotation with the current time
            now = datetime.now().isoformat()
            patch = {
                "spec": {
                    "template": {
                        "metadata": {
                            "annotations": {
                                "kubectl.kubernetes.io/restartedAt": now
                            }
                        }
                    }
                }
            }
            
            api.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=patch
            )
            
            logger.info(f"Successfully triggered restart of {deployment_name}")
            
            # Send notification
            self._send_notification(
                title=f"Service Restart: {service_name}",
                message=f"The service {service_name} has been restarted due to health check failures.",
                severity="warning"
            )
            
        except ImportError:
            logger.error("Kubernetes client not installed - cannot restart service")
        except ApiException as e:
            logger.error(f"Kubernetes API error: {str(e)}")
        except Exception as e:
            logger.error(f"Error restarting service {service_name}: {str(e)}", exc_info=True)
    
    def _send_notification(self, title: str, message: str, severity: str = "info"):
        """
        Send a notification about component status.
        
        Args:
            title: Notification title
            message: Notification message
            severity: Severity level (info, warning, critical)
        """
        if not self.notify_url:
            logger.debug("No notification URL configured, skipping notification")
            return
        
        try:
            # Basic payload for webhook
            payload = {
                "title": title,
                "message": message,
                "severity": severity,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send to webhook
            response = requests.post(
                url=self.notify_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code >= 400:
                logger.error(f"Failed to send notification: {response.status_code} {response.text}")
            else:
                logger.info(f"Notification sent: {title}")
                
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}", exc_info=True)
    
    def get_status_report(self) -> Dict[str, Any]:
        """
        Get a status report of all monitored components.
        
        Returns:
            Dictionary with component status information
        """
        return {
            "components": {
                name: {
                    "status": self.component_status.get(name, "unknown"),
                    "failure_count": self.failure_count.get(name, 0),
                    "last_check": self.last_check_time.get(name, None)
                }
                for name in set(self.component_status.keys()) | set(self.failure_count.keys())
            },
            "timestamp": datetime.now().isoformat()
        }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Trading Bot Monitoring Service")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--interval", "-i", type=int, default=60, help="Check interval in seconds")
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config:
            import json
            with open(args.config, 'r') as f:
                config = json.load(f)
        else:
            # Default configuration
            config = {
                "endpoints": [
                    {
                        "name": "Trading System",
                        "url": "http://trading-bot-trading-system:8000/health",
                        "service_name": "trading-system",
                        "failure_action": "restart"
                    },
                    {
                        "name": "Data Collector",
                        "url": "http://trading-bot-data-collector:8000/health",
                        "service_name": "data-collector",
                        "failure_action": "restart"
                    },
                    {
                        "name": "API Server",
                        "url": "http://trading-bot-api-server:8000/health",
                        "service_name": "api-server",
                        "failure_action": "restart"
                    }
                ],
                "max_failures": 3,
                "notification_webhook": os.environ.get("NOTIFICATION_WEBHOOK")
            }
        
        # Create and start the monitor
        monitor = TradingBotMonitor(config, check_interval=args.interval)
        monitor.start()
        
        # Set up signal handlers for graceful shutdown
        def handle_signal(sig, frame):
            logger.info(f"Received signal {sig}, shutting down...")
            monitor.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Monitoring service interrupted")
    except Exception as e:
        logger.error(f"Error in monitoring service: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 