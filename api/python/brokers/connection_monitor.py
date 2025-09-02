"""
Broker Connection Monitoring

This module provides functionality for monitoring brokerage API connections,
detecting issues, and providing alerts when connection problems arise.
"""

import logging
import threading
import time
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import json
import os

from .brokerage_client import BrokerageClient, BrokerConnectionStatus

# Configure logging
logger = logging.getLogger(__name__)

class ConnectionAlert:
    """
    Class representing a broker connection alert.
    """
    def __init__(self, 
                broker_name: str, 
                status: BrokerConnectionStatus, 
                message: str,
                timestamp: Optional[datetime] = None):
        """
        Initialize a connection alert.
        
        Args:
            broker_name: Name of the broker
            status: Connection status that triggered the alert
            message: Alert message
            timestamp: Alert timestamp (defaults to now)
        """
        self.broker_name = broker_name
        self.status = status
        self.message = message
        self.timestamp = timestamp or datetime.now()
        self.resolved = False
        self.resolved_at = None
        self.resolution_message = None
    
    def resolve(self, message: Optional[str] = None) -> None:
        """
        Mark the alert as resolved.
        
        Args:
            message: Resolution message
        """
        self.resolved = True
        self.resolved_at = datetime.now()
        self.resolution_message = message or f"Connection to {self.broker_name} restored"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert alert to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Alert information
        """
        return {
            'broker_name': self.broker_name,
            'status': self.status.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolution_message': self.resolution_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConnectionAlert':
        """
        Create alert from dictionary.
        
        Args:
            data: Alert data
            
        Returns:
            ConnectionAlert: The alert
        """
        alert = cls(
            broker_name=data['broker_name'],
            status=BrokerConnectionStatus(data['status']),
            message=data['message'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )
        
        if data['resolved']:
            alert.resolved = True
            if data.get('resolved_at'):
                alert.resolved_at = datetime.fromisoformat(data['resolved_at'])
            alert.resolution_message = data.get('resolution_message')
        
        return alert

class ConnectionMonitor:
    """
    Class for monitoring broker connections and providing alerts.
    """
    def __init__(self, alert_callback: Optional[Callable[[ConnectionAlert], None]] = None):
        """
        Initialize the connection monitor.
        
        Args:
            alert_callback: Optional callback function for alerts
        """
        self.brokers: Dict[str, BrokerageClient] = {}
        self.broker_statuses: Dict[str, BrokerConnectionStatus] = {}
        self.alerts: List[ConnectionAlert] = []
        self.active_alerts: Dict[str, ConnectionAlert] = {}  # Keyed by broker name
        
        self.alert_callback = alert_callback
        self.alert_history_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            'connection_alerts.json'
        )
        
        # Monitoring settings
        self.monitor_thread = None
        self.monitor_running = False
        self.check_interval = 60  # seconds
        self.degraded_threshold = 2  # consecutive degraded statuses before alerting
        self.broker_degraded_count: Dict[str, int] = {}
        
        # Load previous alerts if available
        self._load_alerts()
    
    def register_broker(self, name: str, broker: BrokerageClient) -> None:
        """
        Register a broker for monitoring.
        
        Args:
            name: Broker name
            broker: BrokerageClient instance
        """
        self.brokers[name] = broker
        self.broker_statuses[name] = broker.connection_status
        self.broker_degraded_count[name] = 0
        logger.info(f"Registered broker {name} for connection monitoring")
    
    def unregister_broker(self, name: str) -> None:
        """
        Unregister a broker from monitoring.
        
        Args:
            name: Broker name
        """
        if name in self.brokers:
            del self.brokers[name]
            del self.broker_statuses[name]
            del self.broker_degraded_count[name]
            logger.info(f"Unregistered broker {name} from connection monitoring")
    
    def start_monitoring(self) -> None:
        """Start the connection monitoring thread."""
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            logger.warning("Connection monitoring already started")
            return
        
        self.monitor_running = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="broker-connection-monitor"
        )
        self.monitor_thread.start()
        logger.info("Started broker connection monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop the connection monitoring thread."""
        self.monitor_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            logger.info("Stopped broker connection monitoring")
    
    def check_connections(self) -> Dict[str, BrokerConnectionStatus]:
        """
        Check the connection status of all registered brokers.
        
        Returns:
            Dict[str, BrokerConnectionStatus]: Broker connection statuses
        """
        statuses = {}
        
        for name, broker in self.brokers.items():
            try:
                # Check if the connection status has already been updated
                # by the broker's internal monitoring
                current_status = broker.connection_status
                
                # If the broker is already disconnected or in error state,
                # we don't need to check again
                if current_status in [
                    BrokerConnectionStatus.DISCONNECTED,
                    BrokerConnectionStatus.RECONNECTING
                ]:
                    statuses[name] = current_status
                    continue
                
                # Otherwise, proactively check the connection
                status = broker.check_connection()
                statuses[name] = status
                
                # Update our tracking
                self._handle_status_change(name, status)
                
            except Exception as e:
                logger.error(f"Error checking connection for {name}: {str(e)}")
                statuses[name] = BrokerConnectionStatus.DEGRADED
                self._handle_status_change(name, BrokerConnectionStatus.DEGRADED)
        
        return statuses
    
    def get_alerts(self, include_resolved: bool = False) -> List[ConnectionAlert]:
        """
        Get connection alerts.
        
        Args:
            include_resolved: Whether to include resolved alerts
            
        Returns:
            List[ConnectionAlert]: Connection alerts
        """
        if include_resolved:
            return self.alerts
        else:
            return [alert for alert in self.alerts if not alert.resolved]
    
    def _monitoring_loop(self) -> None:
        """Connection monitoring thread main loop."""
        while self.monitor_running:
            try:
                self.check_connections()
            except Exception as e:
                logger.error(f"Error in connection monitoring: {str(e)}")
            
            # Sleep until next check
            time.sleep(self.check_interval)
    
    def _handle_status_change(self, broker_name: str, status: BrokerConnectionStatus) -> None:
        """
        Handle a broker connection status change.
        
        Args:
            broker_name: Broker name
            status: New connection status
        """
        previous_status = self.broker_statuses.get(broker_name)
        self.broker_statuses[broker_name] = status
        
        # Handle degraded status with threshold
        if status == BrokerConnectionStatus.DEGRADED:
            self.broker_degraded_count[broker_name] += 1
            
            # Only alert if degraded for multiple consecutive checks
            if self.broker_degraded_count[broker_name] >= self.degraded_threshold:
                # This is a persistent issue, create an alert
                if broker_name not in self.active_alerts:
                    self._create_alert(broker_name, status, 
                                     f"Connection to {broker_name} is degraded")
        else:
            # Reset degraded count if not degraded
            self.broker_degraded_count[broker_name] = 0
        
        # Handle disconnection
        if status == BrokerConnectionStatus.DISCONNECTED:
            if broker_name not in self.active_alerts:
                self._create_alert(broker_name, status, 
                                 f"Connection to {broker_name} has been lost")
        
        # Handle reconnection
        if status == BrokerConnectionStatus.CONNECTED and previous_status in [
            BrokerConnectionStatus.DISCONNECTED,
            BrokerConnectionStatus.DEGRADED,
            BrokerConnectionStatus.RECONNECTING
        ]:
            # Reconnected after a problem
            if broker_name in self.active_alerts:
                self._resolve_alert(broker_name)
    
    def _create_alert(self, broker_name: str, status: BrokerConnectionStatus, message: str) -> None:
        """
        Create a new connection alert.
        
        Args:
            broker_name: Broker name
            status: Connection status
            message: Alert message
        """
        alert = ConnectionAlert(broker_name, status, message)
        self.alerts.append(alert)
        self.active_alerts[broker_name] = alert
        
        logger.warning(f"Connection alert: {message}")
        
        # Call the alert callback if provided
        if self.alert_callback:
            self.alert_callback(alert)
        
        # Save alerts
        self._save_alerts()
    
    def _resolve_alert(self, broker_name: str) -> None:
        """
        Resolve an active alert.
        
        Args:
            broker_name: Broker name
        """
        if broker_name in self.active_alerts:
            alert = self.active_alerts[broker_name]
            alert.resolve()
            del self.active_alerts[broker_name]
            
            logger.info(f"Connection alert resolved: {alert.resolution_message}")
            
            # Call the alert callback if provided
            if self.alert_callback:
                self.alert_callback(alert)
            
            # Save alerts
            self._save_alerts()
    
    def _save_alerts(self) -> None:
        """Save alerts to a file for persistence."""
        try:
            with open(self.alert_history_file, 'w') as f:
                data = [alert.to_dict() for alert in self.alerts]
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving connection alerts: {str(e)}")
    
    def _load_alerts(self) -> None:
        """Load alerts from a file if available."""
        if not os.path.exists(self.alert_history_file):
            return
        
        try:
            with open(self.alert_history_file, 'r') as f:
                data = json.load(f)
                
                self.alerts = [ConnectionAlert.from_dict(alert_data) for alert_data in data]
                
                # Restore active alerts
                for alert in self.alerts:
                    if not alert.resolved:
                        self.active_alerts[alert.broker_name] = alert
                        
            logger.info(f"Loaded {len(self.alerts)} connection alerts from history")
            
        except Exception as e:
            logger.error(f"Error loading connection alerts: {str(e)}")
    
    def clear_resolved_alerts(self, older_than: Optional[timedelta] = None) -> int:
        """
        Clear resolved alerts from history.
        
        Args:
            older_than: Optional time threshold for clearing alerts
            
        Returns:
            int: Number of alerts cleared
        """
        if older_than is None:
            # Default to 30 days
            older_than = timedelta(days=30)
        
        threshold = datetime.now() - older_than
        
        # Count alerts before filtering
        original_count = len(self.alerts)
        
        # Filter alerts
        self.alerts = [
            alert for alert in self.alerts 
            if not alert.resolved or 
            alert.resolved_at is None or 
            alert.resolved_at > threshold
        ]
        
        # Save after filtering
        self._save_alerts()
        
        return original_count - len(self.alerts) 