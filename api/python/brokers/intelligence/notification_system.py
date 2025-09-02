#!/usr/bin/env python3
"""
Broker Intelligence Notification System

Handles alerting and notifications for broker intelligence events,
such as circuit breaker trips, health status changes, and failover
recommendations.

Supports multiple notification channels (email, SMS, Telegram)
with a pluggable provider architecture.
"""

import os
import json
import logging
import smtplib
import requests
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from trading_bot.event_system.event_bus import EventBus
from trading_bot.event_system.event_types import EventType
from trading_bot.core.credential_manager import CredentialManager


logger = logging.getLogger(__name__)


class NotificationProvider:
    """Base class for notification providers"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the notification provider
        
        Args:
            config: Provider-specific configuration
        """
        self.config = config
        self.enabled = config.get("enabled", True)
    
    def send_notification(self, subject: str, message: str, priority: str = "normal") -> bool:
        """
        Send a notification through this provider
        
        Args:
            subject: Notification subject/title
            message: Message body
            priority: Priority level (low, normal, high, critical)
            
        Returns:
            bool: True if notification was sent successfully
        """
        if not self.enabled:
            logger.info(f"Notification provider disabled: {self.__class__.__name__}")
            return False
        
        try:
            return self._send(subject, message, priority)
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")
            return False
    
    def _send(self, subject: str, message: str, priority: str) -> bool:
        """
        Provider-specific implementation of notification sending
        
        Args:
            subject: Notification subject/title
            message: Message body
            priority: Priority level
            
        Returns:
            bool: True if notification was sent successfully
        """
        raise NotImplementedError("Notification providers must implement _send method")


class EmailNotificationProvider(NotificationProvider):
    """Email notification provider using SMTP"""
    
    def __init__(self, config: Dict[str, Any], credentials: Optional[Dict[str, Any]] = None):
        """
        Initialize the email notification provider
        
        Args:
            config: Email provider configuration
            credentials: Optional credentials, if not provided in config
        """
        super().__init__(config)
        self.smtp_host = config.get("smtp_host", "smtp.gmail.com")
        self.smtp_port = config.get("smtp_port", 587)
        self.use_tls = config.get("use_tls", True)
        
        # Use provided credentials or get from config
        creds = credentials or {}
        self.username = creds.get("username") or config.get("username")
        self.password = creds.get("password") or config.get("password")
        
        # Email settings
        self.from_email = config.get("from_email", self.username)
        self.to_emails = config.get("to_emails", [])
        
        # Validate required fields
        if not all([self.smtp_host, self.smtp_port, self.username, self.password, self.to_emails]):
            logger.warning("Email provider missing required configuration")
            self.enabled = False
    
    def _send(self, subject: str, message: str, priority: str) -> bool:
        """Send email notification"""
        if not self.to_emails:
            logger.warning("No recipients configured for email notification")
            return False
        
        # Create multipart email
        email = MIMEMultipart()
        email["From"] = self.from_email
        email["To"] = ", ".join(self.to_emails)
        email["Subject"] = subject
        
        # Add priority header if needed
        if priority == "high" or priority == "critical":
            email["X-Priority"] = "1"
            email["X-MSMail-Priority"] = "High"
            email["Importance"] = "High"
        
        # Attach message body
        email.attach(MIMEText(message, "plain"))
        
        try:
            # Connect to SMTP server
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.ehlo()
            
            if self.use_tls:
                server.starttls()
                server.ehlo()
            
            # Login and send
            server.login(self.username, self.password)
            server.sendmail(self.from_email, self.to_emails, email.as_string())
            server.quit()
            
            logger.info(f"Email notification sent to {len(self.to_emails)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {str(e)}")
            return False


class TelegramNotificationProvider(NotificationProvider):
    """Telegram notification provider"""
    
    def __init__(self, config: Dict[str, Any], credentials: Optional[Dict[str, Any]] = None):
        """
        Initialize the Telegram notification provider
        
        Args:
            config: Telegram provider configuration
            credentials: Optional credentials, if not provided in config
        """
        super().__init__(config)
        
        # Use provided credentials or get from config
        creds = credentials or {}
        self.bot_token = creds.get("bot_token") or config.get("bot_token")
        
        # Telegram settings
        self.chat_ids = config.get("chat_ids", [])
        self.api_url = "https://api.telegram.org/bot{}/sendMessage"
        
        # Validate required fields
        if not all([self.bot_token, self.chat_ids]):
            logger.warning("Telegram provider missing required configuration")
            self.enabled = False
    
    def _send(self, subject: str, message: str, priority: str) -> bool:
        """Send Telegram notification"""
        if not self.chat_ids:
            logger.warning("No chat IDs configured for Telegram notification")
            return False
        
        # Format message with subject as header
        full_message = f"*{subject}*\n\n{message}"
        
        # Add priority tag if needed
        if priority == "high" or priority == "critical":
            priority_tag = "🔴 URGENT" if priority == "critical" else "🟠 HIGH PRIORITY"
            full_message = f"{priority_tag}\n{full_message}"
        
        # Send to all configured chat IDs
        success = True
        api_url = self.api_url.format(self.bot_token)
        
        for chat_id in self.chat_ids:
            payload = {
                "chat_id": chat_id,
                "text": full_message,
                "parse_mode": "Markdown"
            }
            
            try:
                response = requests.post(api_url, json=payload)
                if not response.json().get("ok", False):
                    logger.error(f"Telegram API error: {response.text}")
                    success = False
            except Exception as e:
                logger.error(f"Failed to send Telegram notification: {str(e)}")
                success = False
        
        if success:
            logger.info(f"Telegram notification sent to {len(self.chat_ids)} chats")
        
        return success


class SMSNotificationProvider(NotificationProvider):
    """SMS notification provider (using Twilio)"""
    
    def __init__(self, config: Dict[str, Any], credentials: Optional[Dict[str, Any]] = None):
        """
        Initialize the SMS notification provider
        
        Args:
            config: SMS provider configuration
            credentials: Optional credentials, if not provided in config
        """
        super().__init__(config)
        
        # Use provided credentials or get from config
        creds = credentials or {}
        self.account_sid = creds.get("account_sid") or config.get("account_sid")
        self.auth_token = creds.get("auth_token") or config.get("auth_token")
        
        # SMS settings
        self.from_number = config.get("from_number")
        self.to_numbers = config.get("to_numbers", [])
        
        # Only attempt to import Twilio if this provider is configured
        if all([self.account_sid, self.auth_token, self.from_number, self.to_numbers]):
            try:
                # Lazy import to avoid dependency if not used
                from twilio.rest import Client
                self._twilio_client = Client(self.account_sid, self.auth_token)
                self.twilio_available = True
            except ImportError:
                logger.warning("Twilio package not installed, SMS notifications disabled")
                self.enabled = False
                self.twilio_available = False
        else:
            logger.warning("SMS provider missing required configuration")
            self.enabled = False
            self.twilio_available = False
    
    def _send(self, subject: str, message: str, priority: str) -> bool:
        """Send SMS notification"""
        if not self.twilio_available or not self.to_numbers:
            return False
        
        # Format message with subject as header, keep it short
        full_message = f"{subject}: {message}"
        
        # Truncate if too long for SMS
        if len(full_message) > 1000:
            full_message = full_message[:997] + "..."
        
        # Only send SMSes for high priority notifications unless configured otherwise
        if priority not in ["high", "critical"] and not self.config.get("send_all_priorities", False):
            logger.info(f"Skipping SMS for {priority} priority notification")
            return False
        
        # Send to all configured numbers
        success = True
        for to_number in self.to_numbers:
            try:
                message = self._twilio_client.messages.create(
                    body=full_message,
                    from_=self.from_number,
                    to=to_number
                )
                logger.debug(f"SMS sent with SID: {message.sid}")
            except Exception as e:
                logger.error(f"Failed to send SMS notification: {str(e)}")
                success = False
        
        if success:
            logger.info(f"SMS notification sent to {len(self.to_numbers)} numbers")
        
        return success


class BrokerIntelligenceNotificationSystem:
    """
    Notification system for broker intelligence events
    
    Listens for broker intelligence events and sends notifications
    through configured providers based on event type and severity.
    """
    
    def __init__(
        self, 
        event_bus: EventBus,
        config_path: str,
        credential_manager: Optional[CredentialManager] = None
    ):
        """
        Initialize the broker intelligence notification system
        
        Args:
            event_bus: Event bus to subscribe to
            config_path: Path to notification config file
            credential_manager: Optional credential manager for secure credential retrieval
        """
        self.event_bus = event_bus
        self.credential_manager = credential_manager
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize notification providers
        self.providers = self._setup_providers()
        
        # Subscribe to relevant events
        self._subscribe_to_events()
        
        logger.info(f"Broker Intelligence Notification System initialized with {len(self.providers)} providers")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load notification system config from file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load notification config from {config_path}: {str(e)}")
            # Return default config
            return {
                "enabled": True,
                "notification_levels": {
                    "circuit_breaker_tripped": "high",
                    "circuit_breaker_reset": "normal",
                    "broker_health_update": "normal",
                    "broker_failover_recommended": "high"
                },
                "providers": {}
            }
    
    def _setup_providers(self) -> List[NotificationProvider]:
        """Set up notification providers from config"""
        providers = []
        
        # Skip if notification system is disabled
        if not self.config.get("enabled", True):
            logger.info("Notification system is disabled in config")
            return providers
        
        provider_configs = self.config.get("providers", {})
        
        # Set up email provider if configured
        if "email" in provider_configs:
            email_config = provider_configs["email"]
            if email_config.get("enabled", True):
                # Get credentials if credential manager is available
                credentials = None
                if self.credential_manager and email_config.get("credential_key"):
                    credentials = self.credential_manager.get_credentials(
                        email_config["credential_key"]
                    )
                
                providers.append(EmailNotificationProvider(email_config, credentials))
        
        # Set up Telegram provider if configured
        if "telegram" in provider_configs:
            telegram_config = provider_configs["telegram"]
            if telegram_config.get("enabled", True):
                # Get credentials if credential manager is available
                credentials = None
                if self.credential_manager and telegram_config.get("credential_key"):
                    credentials = self.credential_manager.get_credentials(
                        telegram_config["credential_key"]
                    )
                
                providers.append(TelegramNotificationProvider(telegram_config, credentials))
        
        # Set up SMS provider if configured
        if "sms" in provider_configs:
            sms_config = provider_configs["sms"]
            if sms_config.get("enabled", True):
                # Get credentials if credential manager is available
                credentials = None
                if self.credential_manager and sms_config.get("credential_key"):
                    credentials = self.credential_manager.get_credentials(
                        sms_config["credential_key"]
                    )
                
                providers.append(SMSNotificationProvider(sms_config, credentials))
        
        return providers
    
    def _subscribe_to_events(self):
        """Subscribe to relevant broker intelligence events"""
        # Subscribe to broker intelligence events
        self.event_bus.subscribe(
            event_type=EventType.BROKER_INTELLIGENCE,
            handler=self._handle_broker_intelligence_event
        )
        
        # Subscribe to orchestrator advisory events
        self.event_bus.subscribe(
            event_type=EventType.ORCHESTRATOR_ADVISORY,
            handler=self._handle_orchestrator_advisory_event
        )
    
    def _handle_broker_intelligence_event(self, event: Dict[str, Any]):
        """Handle broker intelligence events"""
        event_subtype = event.get("event_subtype")
        
        if event_subtype == "circuit_breaker_tripped":
            self._notify_circuit_breaker_tripped(event)
        
        elif event_subtype == "circuit_breaker_reset":
            self._notify_circuit_breaker_reset(event)
        
        elif event_subtype == "broker_health_update":
            self._notify_broker_health_update(event)
    
    def _handle_orchestrator_advisory_event(self, event: Dict[str, Any]):
        """Handle orchestrator advisory events"""
        event_subtype = event.get("event_subtype")
        
        if event_subtype == "broker_selection_advice":
            self._notify_broker_selection_advice(event)
    
    def _get_notification_priority(self, event_type: str) -> str:
        """Get notification priority for an event type"""
        notification_levels = self.config.get("notification_levels", {})
        return notification_levels.get(event_type, "normal")
    
    def _notify_circuit_breaker_tripped(self, event: Dict[str, Any]):
        """Send notification for circuit breaker tripped event"""
        broker_id = event.get("broker_id", "unknown")
        reason = event.get("reason", "unknown")
        reset_time = event.get("reset_time", "unknown")
        
        subject = f"ALERT: Circuit Breaker Tripped for {broker_id}"
        
        message = (
            f"A circuit breaker has been activated for broker {broker_id}\n\n"
            f"Reason: {reason}\n"
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Auto-reset time: {reset_time}\n\n"
            f"Trading operations through this broker will be suspended until the circuit breaker resets."
        )
        
        priority = self._get_notification_priority("circuit_breaker_tripped")
        self._send_notification(subject, message, priority)
    
    def _notify_circuit_breaker_reset(self, event: Dict[str, Any]):
        """Send notification for circuit breaker reset event"""
        broker_id = event.get("broker_id", "unknown")
        
        subject = f"Circuit Breaker Reset for {broker_id}"
        
        message = (
            f"The circuit breaker for broker {broker_id} has been reset.\n\n"
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Trading operations through this broker can now resume."
        )
        
        priority = self._get_notification_priority("circuit_breaker_reset")
        self._send_notification(subject, message, priority)
    
    def _notify_broker_health_update(self, event: Dict[str, Any]):
        """Send notification for broker health update event"""
        broker_id = event.get("broker_id", "unknown")
        health_status = event.get("health_status", "unknown")
        previous_status = event.get("previous_status")
        
        # Only notify on status changes or if it's CRITICAL
        if previous_status and health_status == previous_status and health_status != "CRITICAL":
            return
        
        # Determine severity
        priority = "normal"
        if health_status == "CRITICAL":
            priority = "high"
        elif health_status == "CAUTION":
            priority = "normal"
        
        subject = f"Broker Health Update: {broker_id} - {health_status}"
        
        message = (
            f"Broker {broker_id} health status has changed to {health_status}\n\n"
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        
        if previous_status:
            message += f"Previous status: {previous_status}\n"
        
        if "metrics" in event:
            message += "\nCurrent metrics:\n"
            for metric_name, metric_value in event["metrics"].items():
                message += f"- {metric_name}: {metric_value}\n"
        
        priority = self._get_notification_priority("broker_health_update")
        self._send_notification(subject, message, priority)
    
    def _notify_broker_selection_advice(self, event: Dict[str, Any]):
        """Send notification for broker selection advice event"""
        advice_data = event.get("advice", {})
        asset_class = advice_data.get("asset_class", "unknown")
        operation_type = advice_data.get("operation_type", "unknown")
        primary_broker = advice_data.get("primary_broker_id", "unknown")
        is_failover = advice_data.get("is_failover_recommended", False)
        
        # Only notify if failover is recommended
        if not is_failover:
            return
        
        subject = f"Broker Failover Recommended: {asset_class}/{operation_type}"
        
        message = (
            f"A broker failover is recommended for {asset_class}/{operation_type}\n\n"
            f"Recommended primary broker: {primary_broker}\n"
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        
        if "backup_broker_ids" in advice_data:
            backup_brokers = advice_data["backup_broker_ids"]
            message += f"Backup brokers: {', '.join(backup_brokers)}\n"
        
        if "blacklisted_broker_ids" in advice_data:
            blacklisted = advice_data["blacklisted_broker_ids"]
            if blacklisted:
                message += f"Blacklisted brokers: {', '.join(blacklisted)}\n"
        
        if "advisory_notes" in advice_data:
            notes = advice_data["advisory_notes"]
            if notes:
                message += "\nAdvisory notes:\n"
                for note in notes:
                    message += f"- {note}\n"
        
        priority = self._get_notification_priority("broker_failover_recommended")
        self._send_notification(subject, message, priority)
    
    def _send_notification(self, subject: str, message: str, priority: str = "normal"):
        """Send notification through all configured providers"""
        if not self.providers:
            logger.warning("No notification providers configured")
            return
        
        for provider in self.providers:
            provider.send_notification(subject, message, priority)
    
    def send_test_notification(self):
        """Send a test notification through all providers"""
        subject = "Broker Intelligence Test Notification"
        message = (
            f"This is a test notification from the Broker Intelligence System.\n\n"
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"If you received this message, notifications are working correctly."
        )
        
        logger.info("Sending test notification")
        self._send_notification(subject, message, "normal")
