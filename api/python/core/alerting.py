#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Alerting System for BensBot

This module provides robust alerting capabilities for critical events:
1. Component failures (watchdog-detected issues)
2. Strategy underperformance (strategy manager-detected issues)
3. Execution quality problems (execution model-detected issues)
4. System-level issues (resource constraints, connectivity)

Alerts can be delivered via multiple channels:
- Email
- SMS (via Twilio)
- Slack
- Webhook integrations
- Console logging

Usage:
    from trading_bot.core.alerting import AlertingSystem, AlertLevel, AlertChannel
    
    # Initialize the alerting system
    alerting = AlertingSystem(config={
        'email': {'recipients': ['admin@example.com']},
        'slack': {'webhook_url': 'https://hooks.slack.com/...'},
        'enabled_channels': ['email', 'slack', 'console']
    })
    
    # Send an alert
    alerting.send_alert(
        level=AlertLevel.CRITICAL,
        source="ServiceWatchdog",
        message="Data feed connection lost",
        details={"service": "data_feed", "attempts": 3}
    )
"""

import os
import sys
import time
import logging
import json
import smtplib
import requests
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3

class AlertChannel(Enum):
    """Available alert delivery channels"""
    CONSOLE = "console"
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"

class AlertThrottling:
    """Alert throttling to prevent alert storms"""
    
    def __init__(self, interval_seconds: int = 300):
        """
        Initialize alert throttling
        
        Args:
            interval_seconds: Minimum seconds between repeated alerts
        """
        self.interval_seconds = interval_seconds
        self.last_alerts: Dict[str, datetime] = {}
        self.lock = threading.RLock()
    
    def should_throttle(self, alert_key: str) -> bool:
        """
        Check if an alert should be throttled
        
        Args:
            alert_key: Unique key for the alert (source + message)
            
        Returns:
            bool: True if alert should be throttled, False otherwise
        """
        with self.lock:
            now = datetime.now()
            if alert_key in self.last_alerts:
                time_since_last = (now - self.last_alerts[alert_key]).total_seconds()
                if time_since_last < self.interval_seconds:
                    return True
            
            self.last_alerts[alert_key] = now
            return False
    
    def cleanup_old_entries(self) -> None:
        """Remove old entries from the throttling cache"""
        with self.lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.interval_seconds * 2)
            keys_to_remove = [
                key for key, timestamp in self.last_alerts.items()
                if timestamp < cutoff
            ]
            
            for key in keys_to_remove:
                del self.last_alerts[key]

class AlertingSystem:
    """Advanced alerting system for BensBot"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the alerting system
        
        Args:
            config: Configuration for alerting channels and options
        """
        self.config = config
        self.enabled_channels: Set[AlertChannel] = set()
        self.min_level = AlertLevel.INFO
        self.throttling = AlertThrottling(
            interval_seconds=config.get('throttling_interval_seconds', 300)
        )
        
        # Parse enabled channels
        channel_names = config.get('enabled_channels', ['console'])
        for channel_name in channel_names:
            try:
                self.enabled_channels.add(AlertChannel(channel_name))
            except ValueError:
                logger.warning(f"Unknown alert channel: {channel_name}")
        
        # Parse minimum alert level
        min_level_name = config.get('min_level', 'INFO')
        try:
            self.min_level = AlertLevel[min_level_name]
        except KeyError:
            logger.warning(f"Unknown alert level: {min_level_name}, defaulting to INFO")
            self.min_level = AlertLevel.INFO
        
        logger.info(f"Alerting system initialized with channels: {[c.value for c in self.enabled_channels]}")
    
    def send_alert(self, level: AlertLevel, source: str, message: str, 
                 details: Optional[Dict[str, Any]] = None,
                 channels: Optional[List[AlertChannel]] = None) -> bool:
        """
        Send an alert through configured channels
        
        Args:
            level: Alert severity level
            source: Component that generated the alert
            message: Alert message
            details: Additional alert details
            channels: Specific channels to use (overrides defaults)
            
        Returns:
            bool: True if alert was sent, False otherwise
        """
        # Check minimum level
        if level.value < self.min_level.value:
            return False
        
        # Prepare alert data
        alert_data = {
            'level': level.name,
            'source': source,
            'message': message,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Check for throttling
        alert_key = f"{source}:{message}"
        if self.throttling.should_throttle(alert_key):
            logger.debug(f"Throttling alert: {alert_key}")
            return False
        
        # Determine channels to use
        use_channels = channels if channels is not None else self.enabled_channels
        
        # Send to each channel
        success = False
        for channel in use_channels:
            if channel not in self.enabled_channels:
                logger.warning(f"Channel {channel.value} not enabled, skipping")
                continue
            
            try:
                if channel == AlertChannel.CONSOLE:
                    success |= self._send_console_alert(alert_data)
                elif channel == AlertChannel.EMAIL:
                    success |= self._send_email_alert(alert_data)
                elif channel == AlertChannel.SMS:
                    success |= self._send_sms_alert(alert_data)
                elif channel == AlertChannel.SLACK:
                    success |= self._send_slack_alert(alert_data)
                elif channel == AlertChannel.WEBHOOK:
                    success |= self._send_webhook_alert(alert_data)
            except Exception as e:
                logger.error(f"Error sending alert via {channel.value}: {str(e)}")
        
        # Clean up throttling cache periodically
        if random.random() < 0.1:  # 10% chance to clean up on each alert
            self.throttling.cleanup_old_entries()
        
        return success
    
    def _send_console_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send alert to console"""
        level_name = alert_data['level']
        source = alert_data['source']
        message = alert_data['message']
        timestamp = alert_data['timestamp']
        
        if level_name == 'INFO':
            logger.info(f"[{source}] {message}")
        elif level_name == 'WARNING':
            logger.warning(f"[{source}] {message}")
        elif level_name == 'ERROR':
            logger.error(f"[{source}] {message}")
        elif level_name == 'CRITICAL':
            logger.critical(f"[{source}] {message}")
        
        return True
    
    def _send_email_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send alert via email"""
        email_config = self.config.get('email', {})
        recipients = email_config.get('recipients', [])
        
        if not recipients:
            logger.warning("No email recipients configured")
            return False
        
        smtp_host = email_config.get('smtp_host', 'localhost')
        smtp_port = email_config.get('smtp_port', 25)
        smtp_user = email_config.get('smtp_user')
        smtp_password = email_config.get('smtp_password')
        from_address = email_config.get('from_address', 'bensbot@example.com')
        
        level_name = alert_data['level']
        source = alert_data['source']
        message = alert_data['message']
        details = alert_data['details']
        timestamp = alert_data['timestamp']
        
        subject = f"BensBot Alert: [{level_name}] {source} - {message}"
        
        # Create email message
        email_message = MIMEMultipart('alternative')
        email_message['Subject'] = subject
        email_message['From'] = from_address
        email_message['To'] = ", ".join(recipients)
        
        # Plain text version
        text_content = f"""
        BensBot Alert
        -------------
        Level: {level_name}
        Source: {source}
        Time: {timestamp}
        
        Message: 
        {message}
        
        Details:
        {json.dumps(details, indent=2)}
        """
        
        # HTML version
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .alert-info {{ background-color: #d1ecf1; padding: 15px; border-radius: 5px; }}
                .alert-warning {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; }}
                .alert-error {{ background-color: #f8d7da; padding: 15px; border-radius: 5px; }}
                .alert-critical {{ background-color: #dc3545; color: white; padding: 15px; border-radius: 5px; }}
                pre {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>BensBot Alert</h1>
            <div class="alert-{level_name.lower()}">
                <h2>{level_name}: {source}</h2>
                <p><strong>Time:</strong> {timestamp}</p>
                <p><strong>Message:</strong> {message}</p>
                <h3>Details:</h3>
                <pre>{json.dumps(details, indent=2)}</pre>
            </div>
        </body>
        </html>
        """
        
        # Attach parts
        part1 = MIMEText(text_content, 'plain')
        part2 = MIMEText(html_content, 'html')
        email_message.attach(part1)
        email_message.attach(part2)
        
        try:
            # Connect to SMTP server
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                if smtp_user and smtp_password:
                    server.starttls()
                    server.login(smtp_user, smtp_password)
                
                server.send_message(email_message)
            return True
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
            return False
    
    def _send_sms_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send alert via SMS (using Twilio)"""
        import random  # Needed for the throttling clean-up
        
        sms_config = self.config.get('sms', {})
        twilio_account_sid = sms_config.get('twilio_account_sid')
        twilio_auth_token = sms_config.get('twilio_auth_token')
        twilio_from_number = sms_config.get('twilio_from_number')
        recipient_numbers = sms_config.get('recipient_numbers', [])
        
        if not (twilio_account_sid and twilio_auth_token and twilio_from_number and recipient_numbers):
            logger.warning("SMS configuration incomplete")
            return False
        
        try:
            # Optional import - only needed if SMS alerting is used
            from twilio.rest import Client
            
            client = Client(twilio_account_sid, twilio_auth_token)
            
            level_name = alert_data['level']
            source = alert_data['source']
            message = alert_data['message']
            
            # Create SMS message (keep it short)
            sms_body = f"BensBot {level_name}: [{source}] {message}"
            
            # Send to all recipients
            for to_number in recipient_numbers:
                client.messages.create(
                    body=sms_body,
                    from_=twilio_from_number,
                    to=to_number
                )
            
            return True
        except ImportError:
            logger.error("Twilio package not installed, cannot send SMS alerts")
            return False
        except Exception as e:
            logger.error(f"Failed to send SMS alert: {str(e)}")
            return False
    
    def _send_slack_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send alert to Slack"""
        slack_config = self.config.get('slack', {})
        webhook_url = slack_config.get('webhook_url')
        
        if not webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False
        
        level_name = alert_data['level']
        source = alert_data['source']
        message = alert_data['message']
        details = alert_data['details']
        timestamp = alert_data['timestamp']
        
        # Set color based on alert level
        if level_name == 'INFO':
            color = "#17a2b8"  # info blue
        elif level_name == 'WARNING':
            color = "#ffc107"  # warning yellow
        elif level_name == 'ERROR':
            color = "#dc3545"  # error red
        elif level_name == 'CRITICAL':
            color = "#721c24"  # dark red
        else:
            color = "#6c757d"  # gray
        
        # Create Slack message
        slack_message = {
            "attachments": [
                {
                    "color": color,
                    "title": f"BensBot Alert: {level_name}",
                    "text": message,
                    "fields": [
                        {
                            "title": "Source",
                            "value": source,
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": timestamp,
                            "short": True
                        }
                    ],
                    "footer": "BensBot Alerting System"
                }
            ]
        }
        
        # Add details if present
        if details:
            slack_message["attachments"][0]["fields"].append({
                "title": "Details",
                "value": f"```{json.dumps(details, indent=2)}```",
                "short": False
            })
        
        try:
            response = requests.post(
                webhook_url,
                data=json.dumps(slack_message),
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")
            return False
    
    def _send_webhook_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send alert to a generic webhook"""
        webhook_config = self.config.get('webhook', {})
        webhook_url = webhook_config.get('url')
        webhook_method = webhook_config.get('method', 'POST')
        webhook_headers = webhook_config.get('headers', {'Content-Type': 'application/json'})
        
        if not webhook_url:
            logger.warning("Webhook URL not configured")
            return False
        
        try:
            if webhook_method.upper() == 'POST':
                response = requests.post(
                    webhook_url,
                    json=alert_data,
                    headers=webhook_headers
                )
            elif webhook_method.upper() == 'PUT':
                response = requests.put(
                    webhook_url,
                    json=alert_data,
                    headers=webhook_headers
                )
            else:
                logger.error(f"Unsupported webhook method: {webhook_method}")
                return False
            
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {str(e)}")
            return False

# Convenience functions
def send_info_alert(alerting_system: AlertingSystem, source: str, message: str, 
                  details: Optional[Dict[str, Any]] = None) -> bool:
    """Send an INFO level alert"""
    return alerting_system.send_alert(
        level=AlertLevel.INFO,
        source=source,
        message=message,
        details=details
    )

def send_warning_alert(alerting_system: AlertingSystem, source: str, message: str, 
                     details: Optional[Dict[str, Any]] = None) -> bool:
    """Send a WARNING level alert"""
    return alerting_system.send_alert(
        level=AlertLevel.WARNING,
        source=source,
        message=message,
        details=details
    )

def send_error_alert(alerting_system: AlertingSystem, source: str, message: str, 
                   details: Optional[Dict[str, Any]] = None) -> bool:
    """Send an ERROR level alert"""
    return alerting_system.send_alert(
        level=AlertLevel.ERROR,
        source=source,
        message=message,
        details=details
    )

def send_critical_alert(alerting_system: AlertingSystem, source: str, message: str, 
                      details: Optional[Dict[str, Any]] = None) -> bool:
    """Send a CRITICAL level alert"""
    return alerting_system.send_alert(
        level=AlertLevel.CRITICAL,
        source=source,
        message=message,
        details=details
    )

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Example configuration
    example_config = {
        'enabled_channels': ['console'],
        'min_level': 'INFO',
        'throttling_interval_seconds': 300,
        'email': {
            'recipients': ['admin@example.com'],
            'smtp_host': 'smtp.example.com',
            'smtp_port': 587,
            'smtp_user': 'user',
            'smtp_password': 'password',
            'from_address': 'bensbot@example.com'
        },
        'slack': {
            'webhook_url': 'https://hooks.slack.com/services/TXXXXXXXX/BXXXXXXXX/XXXXXXXXXXXXXXXXXXXXXXXX'
        },
        'sms': {
            'twilio_account_sid': 'ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
            'twilio_auth_token': 'your_auth_token',
            'twilio_from_number': '+1234567890',
            'recipient_numbers': ['+1234567890']
        },
        'webhook': {
            'url': 'https://example.com/webhook',
            'method': 'POST',
            'headers': {'Content-Type': 'application/json', 'X-API-Key': 'your_api_key'}
        }
    }
    
    # Create alerting system
    alerting = AlertingSystem(config=example_config)
    
    # Send test alerts
    send_info_alert(alerting, "TestComponent", "This is an info message")
    send_warning_alert(alerting, "TestComponent", "This is a warning message")
    send_error_alert(alerting, "TestComponent", "This is an error message")
    send_critical_alert(alerting, "TestComponent", "This is a critical message")
