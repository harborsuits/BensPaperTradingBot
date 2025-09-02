import os
import json
import logging
import time
import psutil
import threading
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
import smtplib
from email.message import EmailMessage
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemMonitor:
    """
    System health monitoring with multi-channel alerting for trading bots.
    
    Features:
    - Real-time system resource monitoring (CPU, memory, disk)
    - Process health checking
    - API and connectivity testing
    - Error rate monitoring
    - Multi-channel alerting (email, SMS, Telegram)
    - Custom health checks
    - Automatic recovery actions
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        log_dir: str = 'logs/system_monitor',
        check_interval: int = 60,  # seconds
        enable_auto_recovery: bool = True,
        notification_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the system health monitor.
        
        Args:
            config_path: Path to configuration file (optional)
            log_dir: Directory for logs
            check_interval: Interval between health checks in seconds
            enable_auto_recovery: Whether to enable automatic recovery actions
            notification_config: Configuration for notification channels
        """
        self.check_interval = check_interval
        self.log_dir = Path(log_dir)
        self.enable_auto_recovery = enable_auto_recovery
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.config = {
            'cpu_threshold': 80,  # 80% CPU usage threshold
            'memory_threshold': 80,  # 80% memory usage threshold
            'disk_threshold': 90,  # 90% disk usage threshold
            'max_error_rate': 0.05,  # 5% error rate threshold
            'api_endpoints': [],  # List of API endpoints to check
            'processes_to_monitor': [],  # List of processes to monitor
            'check_external_connectivity': True,  # Check internet connectivity
            'heartbeat_interval': 300,  # 5 minutes heartbeat interval
            'connectivity_test_urls': [
                'https://www.google.com',
                'https://api.binance.com/api/v3/time',
                'https://api-pub.bitfinex.com/v2/platform/status'
            ]
        }
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        
        # Notification configuration
        self.notification_config = notification_config or {
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'sender': '',
                'recipients': []
            },
            'sms': {
                'enabled': False,
                'service': 'twilio',
                'account_sid': '',
                'auth_token': '',
                'from_number': '',
                'to_numbers': []
            },
            'telegram': {
                'enabled': False,
                'bot_token': '',
                'chat_ids': []
            }
        }
        
        # Initialize state
        self.is_running = False
        self.monitor_thread = None
        self.alert_thread = None
        self.alert_queue = queue.Queue()
        self.last_check_time = datetime.now()
        self.error_counts = {
            'cpu': 0,
            'memory': 0,
            'disk': 0,
            'connectivity': 0,
            'api': 0,
            'process': 0,
            'custom': 0
        }
        
        # Custom health checks
        self.custom_health_checks = []
        
        # Recovery actions
        self.recovery_actions = {}
        
        # System status
        self.system_status = {
            'overall_health': 'unknown',
            'last_check': None,
            'metrics': {},
            'alerts': [],
            'errors': []
        }
        
        logger.info(f"System Monitor initialized with check interval: {check_interval} seconds")
    
    def _load_config(self, config_path: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Update configuration
            for key, value in loaded_config.items():
                self.config[key] = value
            
            logger.info(f"Loaded monitor configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def start(self) -> None:
        """
        Start the system monitoring.
        """
        if self.is_running:
            logger.warning("System monitor is already running")
            return
        
        self.is_running = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Start alert processing thread
        self.alert_thread = threading.Thread(target=self._alert_processing_loop)
        self.alert_thread.daemon = True
        self.alert_thread.start()
        
        logger.info("System monitoring started")
        
        # Send initial heartbeat
        self._send_heartbeat()
    
    def stop(self) -> None:
        """
        Stop the system monitoring.
        """
        if not self.is_running:
            logger.warning("System monitor is already stopped")
            return
        
        self.is_running = False
        
        # Wait for threads to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        if self.alert_thread:
            self.alert_thread.join(timeout=5.0)
        
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """
        Main monitoring loop that runs health checks at regular intervals.
        """
        while self.is_running:
            try:
                # Run health checks
                self._check_system_health()
                
                # Send heartbeat if needed
                current_time = datetime.now()
                seconds_since_heartbeat = (current_time - self.last_check_time).total_seconds()
                
                if seconds_since_heartbeat >= self.config['heartbeat_interval']:
                    self._send_heartbeat()
                
                # Update last check time
                self.last_check_time = current_time
                
                # Sleep until next check
                for _ in range(int(self.check_interval / 0.1)):
                    if not self.is_running:
                        break
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                
                # Add error to system status
                self.system_status['errors'].append({
                    'timestamp': datetime.now(),
                    'error': str(e),
                    'source': 'monitoring_loop'
                })
                
                time.sleep(5)  # Sleep briefly before retrying
    
    def _alert_processing_loop(self) -> None:
        """
        Process alerts from the alert queue.
        """
        while self.is_running:
            try:
                # Get alert from queue with timeout
                try:
                    alert = self.alert_queue.get(timeout=1.0)
                    self._process_alert(alert)
                    self.alert_queue.task_done()
                except queue.Empty:
                    pass  # No alerts in queue
                
            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
                time.sleep(1)  # Sleep briefly before retrying
    
    def _process_alert(self, alert: Dict[str, Any]) -> None:
        """
        Process a single alert.
        
        Args:
            alert: Alert dictionary
        """
        try:
            # Extract alert details
            alert_type = alert.get('type', 'unknown')
            message = alert.get('message', '')
            severity = alert.get('severity', 'info')
            source = alert.get('source', 'system')
            
            # Log alert
            log_message = f"ALERT [{severity.upper()}] {source}: {message}"
            
            if severity == 'critical':
                logger.critical(log_message)
            elif severity == 'error':
                logger.error(log_message)
            elif severity == 'warning':
                logger.warning(log_message)
            else:
                logger.info(log_message)
            
            # Add to alerts history
            self.system_status['alerts'].append(alert)
            
            # Trim alerts history (keep last 100)
            if len(self.system_status['alerts']) > 100:
                self.system_status['alerts'] = self.system_status['alerts'][-100:]
            
            # Send notifications based on severity
            if severity in ['critical', 'error']:
                self._send_notification(
                    f"[{severity.upper()}] {source} Alert",
                    message,
                    priority='high'
                )
            elif severity == 'warning':
                self._send_notification(
                    f"[WARNING] {source} Alert",
                    message,
                    priority='normal'
                )
            
            # Trigger recovery action if available and enabled
            if self.enable_auto_recovery and alert_type in self.recovery_actions:
                recovery_action = self.recovery_actions[alert_type]
                
                logger.info(f"Triggering recovery action for {alert_type}")
                
                try:
                    recovery_action(alert)
                    
                    logger.info(f"Recovery action for {alert_type} completed successfully")
                    
                    # Add recovery entry to alerts
                    recovery_alert = {
                        'timestamp': datetime.now(),
                        'type': 'recovery',
                        'source': source,
                        'severity': 'info',
                        'message': f"Recovery action for {alert_type} alert completed successfully",
                        'related_alert': alert
                    }
                    
                    self.system_status['alerts'].append(recovery_alert)
                    
                except Exception as e:
                    logger.error(f"Error in recovery action for {alert_type}: {e}")
            
        except Exception as e:
            logger.error(f"Error processing alert: {e}")
    
    def _check_system_health(self) -> None:
        """
        Run all health checks and update system status.
        """
        # Reset metrics for this check
        metrics = {}
        
        # Check CPU usage
        cpu_metrics = self._check_cpu_usage()
        metrics['cpu'] = cpu_metrics
        
        # Check memory usage
        memory_metrics = self._check_memory_usage()
        metrics['memory'] = memory_metrics
        
        # Check disk usage
        disk_metrics = self._check_disk_usage()
        metrics['disk'] = disk_metrics
        
        # Check processes
        process_metrics = self._check_processes()
        metrics['processes'] = process_metrics
        
        # Check connectivity
        connectivity_metrics = self._check_connectivity()
        metrics['connectivity'] = connectivity_metrics
        
        # Check API endpoints
        api_metrics = self._check_api_endpoints()
        metrics['api'] = api_metrics
        
        # Run custom health checks
        custom_metrics = self._run_custom_health_checks()
        metrics['custom'] = custom_metrics
        
        # Calculate overall health status
        overall_health = self._calculate_overall_health(metrics)
        
        # Update system status
        self.system_status['overall_health'] = overall_health
        self.system_status['last_check'] = datetime.now()
        self.system_status['metrics'] = metrics
        
        # Log overall health
        logger.info(f"System health check completed: {overall_health}")
        
        # Save status to file
        self._save_status_log()
    
    def _check_cpu_usage(self) -> Dict[str, Any]:
        """
        Check CPU usage.
        
        Returns:
            Dictionary with CPU metrics
        """
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_threshold = self.config['cpu_threshold']
        
        metrics = {
            'usage_percent': cpu_percent,
            'threshold': cpu_threshold,
            'status': 'ok'
        }
        
        # Check if CPU usage exceeds threshold
        if cpu_percent > cpu_threshold:
            metrics['status'] = 'warning'
            
            # Increment error count
            self.error_counts['cpu'] += 1
            
            # Add alert
            alert = {
                'timestamp': datetime.now(),
                'type': 'cpu_usage',
                'source': 'system',
                'severity': 'warning',
                'message': f"High CPU usage: {cpu_percent:.1f}% (threshold: {cpu_threshold}%)"
            }
            
            self.alert_queue.put(alert)
        else:
            # Reset error count if CPU usage is normal
            self.error_counts['cpu'] = 0
        
        # Check if CPU usage is extremely high
        if cpu_percent > 95:
            metrics['status'] = 'critical'
            
            # Add critical alert
            alert = {
                'timestamp': datetime.now(),
                'type': 'cpu_usage',
                'source': 'system',
                'severity': 'critical',
                'message': f"Critical CPU usage: {cpu_percent:.1f}%"
            }
            
            self.alert_queue.put(alert)
        
        return metrics
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """
        Check memory usage.
        
        Returns:
            Dictionary with memory metrics
        """
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_threshold = self.config['memory_threshold']
        
        metrics = {
            'usage_percent': memory_percent,
            'total_gb': memory.total / (1024 ** 3),
            'available_gb': memory.available / (1024 ** 3),
            'threshold': memory_threshold,
            'status': 'ok'
        }
        
        # Check if memory usage exceeds threshold
        if memory_percent > memory_threshold:
            metrics['status'] = 'warning'
            
            # Increment error count
            self.error_counts['memory'] += 1
            
            # Add alert
            alert = {
                'timestamp': datetime.now(),
                'type': 'memory_usage',
                'source': 'system',
                'severity': 'warning',
                'message': f"High memory usage: {memory_percent:.1f}% (threshold: {memory_threshold}%)"
            }
            
            self.alert_queue.put(alert)
        else:
            # Reset error count if memory usage is normal
            self.error_counts['memory'] = 0
        
        # Check if memory usage is extremely high
        if memory_percent > 95:
            metrics['status'] = 'critical'
            
            # Add critical alert
            alert = {
                'timestamp': datetime.now(),
                'type': 'memory_usage',
                'source': 'system',
                'severity': 'critical',
                'message': f"Critical memory usage: {memory_percent:.1f}%"
            }
            
            self.alert_queue.put(alert)
        
        return metrics
    
    def _check_disk_usage(self) -> Dict[str, Any]:
        """
        Check disk usage.
        
        Returns:
            Dictionary with disk metrics
        """
        disk_threshold = self.config['disk_threshold']
        disk_metrics = {}
        
        # Check all mounted partitions
        for partition in psutil.disk_partitions():
            try:
                partition_usage = psutil.disk_usage(partition.mountpoint)
                partition_percent = partition_usage.percent
                
                partition_metrics = {
                    'usage_percent': partition_percent,
                    'total_gb': partition_usage.total / (1024 ** 3),
                    'free_gb': partition_usage.free / (1024 ** 3),
                    'threshold': disk_threshold,
                    'status': 'ok'
                }
                
                # Check if disk usage exceeds threshold
                if partition_percent > disk_threshold:
                    partition_metrics['status'] = 'warning'
                    
                    # Increment error count
                    self.error_counts['disk'] += 1
                    
                    # Add alert
                    alert = {
                        'timestamp': datetime.now(),
                        'type': 'disk_usage',
                        'source': 'system',
                        'severity': 'warning',
                        'message': f"High disk usage on {partition.mountpoint}: {partition_percent:.1f}% (threshold: {disk_threshold}%)"
                    }
                    
                    self.alert_queue.put(alert)
                
                # Check if disk usage is extremely high
                if partition_percent > 98:
                    partition_metrics['status'] = 'critical'
                    
                    # Add critical alert
                    alert = {
                        'timestamp': datetime.now(),
                        'type': 'disk_usage',
                        'source': 'system',
                        'severity': 'critical',
                        'message': f"Critical disk usage on {partition.mountpoint}: {partition_percent:.1f}%"
                    }
                    
                    self.alert_queue.put(alert)
                
                # Add partition metrics
                disk_metrics[partition.mountpoint] = partition_metrics
            except PermissionError:
                # Skip partitions we don't have access to
                pass
        
        # If no disk errors, reset error count
        if all(m['status'] == 'ok' for m in disk_metrics.values()):
            self.error_counts['disk'] = 0
        
        return disk_metrics
    
    def _check_processes(self) -> Dict[str, Any]:
        """
        Check monitored processes.
        
        Returns:
            Dictionary with process metrics
        """
        process_metrics = {}
        processes_to_monitor = self.config['processes_to_monitor']
        
        if not processes_to_monitor:
            return {'status': 'not_configured'}
        
        # Get all running processes
        all_processes = {p.info['name']: p for p in psutil.process_iter(['name', 'pid', 'cpu_percent', 'memory_percent'])}
        
        # Check each monitored process
        for process_name in processes_to_monitor:
            if process_name in all_processes:
                process = all_processes[process_name]
                
                # Get process metrics
                try:
                    with process.oneshot():
                        pid = process.pid
                        cpu_percent = process.cpu_percent()
                        memory_percent = process.memory_percent()
                        cpu_times = process.cpu_times()
                        create_time = datetime.fromtimestamp(process.create_time())
                        running_time = datetime.now() - create_time
                        
                        process_metrics[process_name] = {
                            'status': 'running',
                            'pid': pid,
                            'cpu_percent': cpu_percent,
                            'memory_percent': memory_percent,
                            'user_time': cpu_times.user,
                            'system_time': cpu_times.system,
                            'create_time': create_time,
                            'running_time': running_time.total_seconds() / 3600  # hours
                        }
                except psutil.NoSuchProcess:
                    # Process ended between process_iter and here
                    process_metrics[process_name] = {
                        'status': 'stopped',
                        'message': 'Process ended unexpectedly'
                    }
                    
                    # Increment error count
                    self.error_counts['process'] += 1
                    
                    # Add alert
                    alert = {
                        'timestamp': datetime.now(),
                        'type': 'process_monitoring',
                        'source': 'system',
                        'severity': 'error',
                        'message': f"Process {process_name} ended unexpectedly"
                    }
                    
                    self.alert_queue.put(alert)
            else:
                process_metrics[process_name] = {
                    'status': 'stopped',
                    'message': 'Process not running'
                }
                
                # Increment error count
                self.error_counts['process'] += 1
                
                # Add alert
                alert = {
                    'timestamp': datetime.now(),
                    'type': 'process_monitoring',
                    'source': 'system',
                    'severity': 'error',
                    'message': f"Process {process_name} is not running"
                }
                
                self.alert_queue.put(alert)
        
        # If all processes are running, reset error count
        if all(m['status'] == 'running' for m in process_metrics.values()):
            self.error_counts['process'] = 0
        
        return process_metrics
    
    def _check_connectivity(self) -> Dict[str, Any]:
        """
        Check internet connectivity.
        
        Returns:
            Dictionary with connectivity metrics
        """
        if not self.config['check_external_connectivity']:
            return {'status': 'disabled'}
        
        connectivity_metrics = {
            'status': 'ok',
            'endpoints': {}
        }
        
        # Test connectivity to configured URLs
        test_urls = self.config['connectivity_test_urls']
        
        if not test_urls:
            return {'status': 'not_configured'}
        
        all_successful = True
        
        for url in test_urls:
            try:
                start_time = time.time()
                response = requests.get(url, timeout=5)
                elapsed_time = time.time() - start_time
                
                status_code = response.status_code
                
                connectivity_metrics['endpoints'][url] = {
                    'status': 'ok' if status_code < 400 else 'error',
                    'response_time': elapsed_time,
                    'status_code': status_code
                }
                
                if status_code >= 400:
                    all_successful = False
            except requests.RequestException as e:
                connectivity_metrics['endpoints'][url] = {
                    'status': 'error',
                    'error': str(e)
                }
                
                all_successful = False
        
        # Update overall status
        if not all_successful:
            connectivity_metrics['status'] = 'warning'
            
            # Increment error count
            self.error_counts['connectivity'] += 1
            
            # Add alert
            alert = {
                'timestamp': datetime.now(),
                'type': 'connectivity',
                'source': 'network',
                'severity': 'warning',
                'message': f"Connectivity issues detected with one or more endpoints"
            }
            
            self.alert_queue.put(alert)
        else:
            # Reset error count if all connections are successful
            self.error_counts['connectivity'] = 0
        
        return connectivity_metrics
    
    def _check_api_endpoints(self) -> Dict[str, Any]:
        """
        Check API endpoints.
        
        Returns:
            Dictionary with API metrics
        """
        api_endpoints = self.config['api_endpoints']
        
        if not api_endpoints:
            return {'status': 'not_configured'}
        
        api_metrics = {
            'status': 'ok',
            'endpoints': {}
        }
        
        all_successful = True
        
        for endpoint in api_endpoints:
            url = endpoint.get('url', '')
            method = endpoint.get('method', 'GET')
            headers = endpoint.get('headers', {})
            data = endpoint.get('data', None)
            expected_status = endpoint.get('expected_status', 200)
            
            if not url:
                continue
            
            try:
                start_time = time.time()
                
                if method.upper() == 'GET':
                    response = requests.get(url, headers=headers, timeout=10)
                elif method.upper() == 'POST':
                    response = requests.post(url, headers=headers, json=data, timeout=10)
                else:
                    # Skip unsupported methods
                    continue
                
                elapsed_time = time.time() - start_time
                status_code = response.status_code
                
                api_metrics['endpoints'][url] = {
                    'status': 'ok' if status_code == expected_status else 'error',
                    'response_time': elapsed_time,
                    'status_code': status_code,
                    'expected_status': expected_status
                }
                
                if status_code != expected_status:
                    all_successful = False
            except requests.RequestException as e:
                api_metrics['endpoints'][url] = {
                    'status': 'error',
                    'error': str(e)
                }
                
                all_successful = False
        
        # Update overall status
        if not all_successful:
            api_metrics['status'] = 'warning'
            
            # Increment error count
            self.error_counts['api'] += 1
            
            # Add alert
            alert = {
                'timestamp': datetime.now(),
                'type': 'api_endpoint',
                'source': 'api',
                'severity': 'warning',
                'message': f"API endpoint issues detected"
            }
            
            self.alert_queue.put(alert)
        else:
            # Reset error count if all API checks are successful
            self.error_counts['api'] = 0
        
        return api_metrics
    
    def _run_custom_health_checks(self) -> Dict[str, Any]:
        """
        Run custom health checks.
        
        Returns:
            Dictionary with custom health check results
        """
        if not self.custom_health_checks:
            return {'status': 'not_configured'}
        
        custom_metrics = {
            'status': 'ok',
            'checks': {}
        }
        
        all_successful = True
        
        for check in self.custom_health_checks:
            check_name = check.get('name', 'unnamed_check')
            check_func = check.get('function')
            severity = check.get('severity', 'warning')
            
            if not check_func:
                continue
            
            try:
                result = check_func()
                
                is_successful = result.get('status') == 'ok'
                message = result.get('message', '')
                
                custom_metrics['checks'][check_name] = result
                
                if not is_successful:
                    all_successful = False
                    
                    # Add alert
                    alert = {
                        'timestamp': datetime.now(),
                        'type': 'custom_check',
                        'source': check_name,
                        'severity': severity,
                        'message': message
                    }
                    
                    self.alert_queue.put(alert)
            except Exception as e:
                custom_metrics['checks'][check_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                
                all_successful = False
                
                # Add alert
                alert = {
                    'timestamp': datetime.now(),
                    'type': 'custom_check',
                    'source': check_name,
                    'severity': 'error',
                    'message': f"Custom health check failed with error: {e}"
                }
                
                self.alert_queue.put(alert)
        
        # Update overall status
        if not all_successful:
            custom_metrics['status'] = 'warning'
            
            # Increment error count
            self.error_counts['custom'] += 1
        else:
            # Reset error count if all custom checks are successful
            self.error_counts['custom'] = 0
        
        return custom_metrics
    
    def _calculate_overall_health(self, metrics: Dict[str, Any]) -> str:
        """
        Calculate overall system health status.
        
        Args:
            metrics: Dictionary with all health metrics
            
        Returns:
            Overall health status
        """
        # Check for critical issues
        if (
            metrics.get('cpu', {}).get('status') == 'critical' or
            metrics.get('memory', {}).get('status') == 'critical'
        ):
            return 'critical'
        
        # Check for errors in processes
        if (
            metrics.get('processes', {}).get('status') not in ['ok', 'not_configured', 'disabled'] and
            any(p.get('status') == 'stopped' for p in metrics.get('processes', {}).values() if isinstance(p, dict))
        ):
            return 'error'
        
        # Check for warnings
        if (
            metrics.get('cpu', {}).get('status') == 'warning' or
            metrics.get('memory', {}).get('status') == 'warning' or
            metrics.get('connectivity', {}).get('status') == 'warning' or
            metrics.get('api', {}).get('status') == 'warning' or
            metrics.get('custom', {}).get('status') == 'warning' or
            any(d.get('status') == 'warning' for d in metrics.get('disk', {}).values() if isinstance(d, dict))
        ):
            return 'warning'
        
        # All systems normal
        return 'ok'
    
    def _send_heartbeat(self) -> None:
        """
        Send a heartbeat signal to indicate the monitoring system is active.
        """
        # Update heartbeat timestamp
        self.system_status['last_heartbeat'] = datetime.now()
        
        # Log heartbeat
        logger.info("Monitoring heartbeat sent")
        
        # Send heartbeat notification (low priority)
        if self.notification_config.get('heartbeat_notifications', False):
            self._send_notification(
                "Trading Bot Heartbeat",
                f"System is running normally. Overall health: {self.system_status['overall_health']}",
                priority='low'
            )
        
        # Save status to file
        self._save_status_log()
    
    def _save_status_log(self) -> None:
        """
        Save current system status to log file.
        """
        # Create filename with date
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"system_status_{date_str}.json"
        filepath = self.log_dir / filename
        
        # Prepare data for serialization
        status_data = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': self.system_status['overall_health'],
            'metrics': self.system_status['metrics'],
            'alerts': [
                {
                    'timestamp': a['timestamp'].isoformat(),
                    'type': a['type'],
                    'source': a['source'],
                    'severity': a['severity'],
                    'message': a['message']
                }
                for a in self.system_status['alerts'][-10:]  # last 10 alerts
            ],
            'errors': [
                {
                    'timestamp': e['timestamp'].isoformat(),
                    'error': e['error'],
                    'source': e['source']
                }
                for e in self.system_status['errors'][-10:]  # last 10 errors
            ]
        }
        
        # Save to file
        try:
            with open(filepath, 'w') as f:
                json.dump(status_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving status log: {e}")
    
    def register_custom_health_check(
        self,
        name: str,
        check_function: Callable[[], Dict[str, Any]],
        severity: str = 'warning'
    ) -> None:
        """
        Register a custom health check.
        
        Args:
            name: Name of the health check
            check_function: Function that performs the check and returns a dict with 'status' and 'message'
            severity: Alert severity when check fails
        """
        self.custom_health_checks.append({
            'name': name,
            'function': check_function,
            'severity': severity
        })
        
        logger.info(f"Registered custom health check: {name}")
    
    def register_recovery_action(
        self,
        alert_type: str,
        action_function: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Register a recovery action for a specific alert type.
        
        Args:
            alert_type: Type of alert to trigger the action
            action_function: Function to call for recovery
        """
        self.recovery_actions[alert_type] = action_function
        
        logger.info(f"Registered recovery action for alert type: {alert_type}")
    
    def _send_notification(
        self,
        subject: str,
        message: str,
        priority: str = 'normal'
    ) -> None:
        """
        Send notification through configured channels.
        
        Args:
            subject: Notification subject
            message: Notification message body
            priority: Priority level (low, normal, high)
        """
        # Create a formatted message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {subject}\n\n{message}"
        
        # Log notification
        if priority == 'high':
            logger.error(formatted_message)
        elif priority == 'normal':
            logger.warning(formatted_message)
        else:
            logger.info(formatted_message)
        
        # Send email notification
        if self.notification_config['email']['enabled'] and priority != 'low':
            self._send_email_notification(subject, formatted_message)
        
        # Send SMS notification (only for high priority)
        if priority == 'high' and self.notification_config['sms']['enabled']:
            self._send_sms_notification(f"{subject}: {message[:100]}")
        
        # Send Telegram notification
        if self.notification_config['telegram']['enabled'] and priority != 'low':
            self._send_telegram_notification(formatted_message)
    
    def _send_email_notification(self, subject: str, message: str) -> None:
        """
        Send email notification.
        
        Args:
            subject: Email subject
            message: Email body
        """
        try:
            # Get email config
            config = self.notification_config['email']
            
            # Create email message
            email = EmailMessage()
            email['Subject'] = f"Trading Bot Monitor: {subject}"
            email['From'] = config['sender']
            email['To'] = ', '.join(config['recipients'])
            email.set_content(message)
            
            # Send email
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls()
                server.login(config['username'], config['password'])
                server.send_message(email)
            
            logger.info(f"Email notification sent: {subject}")
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    def _send_sms_notification(self, message: str) -> None:
        """
        Send SMS notification.
        
        Args:
            message: SMS message body
        """
        try:
            # Get SMS config
            config = self.notification_config['sms']
            
            if config['service'] == 'twilio':
                try:
                    from twilio.rest import Client
                    
                    client = Client(config['account_sid'], config['auth_token'])
                    
                    for number in config['to_numbers']:
                        client.messages.create(
                            body=message,
                            from_=config['from_number'],
                            to=number
                        )
                    
                    logger.info(f"SMS notification sent to {len(config['to_numbers'])} recipients")
                except ImportError:
                    logger.error("Twilio package not installed. Install with 'pip install twilio'")
            else:
                logger.error(f"Unsupported SMS service: {config['service']}")
        except Exception as e:
            logger.error(f"Error sending SMS notification: {e}")
    
    def _send_telegram_notification(self, message: str) -> None:
        """
        Send Telegram notification.
        
        Args:
            message: Telegram message body
        """
        try:
            # Get Telegram config
            config = self.notification_config['telegram']
            
            # Use threading to avoid blocking for API calls
            def send_telegram():
                try:
                    import requests
                    
                    bot_token = config['bot_token']
                    
                    for chat_id in config['chat_ids']:
                        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                        payload = {
                            'chat_id': chat_id,
                            'text': message,
                            'parse_mode': 'Markdown'
                        }
                        
                        response = requests.post(url, json=payload)
                        
                        if response.status_code != 200:
                            logger.error(f"Failed to send Telegram message: {response.text}")
                    
                    logger.info(f"Telegram notification sent to {len(config['chat_ids'])} chats")
                except ImportError:
                    logger.error("Requests package not installed. Install with 'pip install requests'")
                except Exception as e:
                    logger.error(f"Error in Telegram thread: {e}")
            
            # Start thread for sending Telegram message
            thread = threading.Thread(target=send_telegram)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            logger.error(f"Error setting up Telegram notification: {e}")
    
    def get_status_report(self) -> Dict[str, Any]:
        """
        Get a detailed status report.
        
        Returns:
            Dictionary with system status report
        """
        # Generate status report
        report = {
            'timestamp': datetime.now(),
            'overall_health': self.system_status['overall_health'],
            'uptime': self._get_system_uptime(),
            'metrics': self.system_status['metrics'],
            'error_counts': self.error_counts,
            'recent_alerts': self.system_status['alerts'][-10:] if self.system_status['alerts'] else [],
            'recent_errors': self.system_status['errors'][-10:] if self.system_status['errors'] else [],
            'monitoring_status': {
                'is_running': self.is_running,
                'last_check': self.last_check_time,
                'check_interval': self.check_interval,
                'auto_recovery': self.enable_auto_recovery
            }
        }
        
        return report
    
    def _get_system_uptime(self) -> float:
        """
        Get system uptime in hours.
        
        Returns:
            System uptime in hours
        """
        try:
            return psutil.boot_time() / 3600
        except:
            return 0.0 