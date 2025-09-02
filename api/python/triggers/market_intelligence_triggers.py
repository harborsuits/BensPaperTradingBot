"""
Market Intelligence Triggers
This module provides various triggers for the Market Intelligence system,
including timers, webhooks, and event-based triggers.
"""

import os
import sys
import json
import time
import logging
import asyncio
import threading
import datetime
from typing import Dict, List, Any, Optional, Union, Callable

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import our components
from trading_bot.market_intelligence_controller import get_market_intelligence_controller
from trading_bot.triggers.notification_connector import get_notification_connector

class TimerTrigger:
    """
    Timer-based trigger that runs the Market Intelligence update at specified intervals.
    """
    
    def __init__(self, config=None):
        """
        Initialize the timer trigger.
        
        Args:
            config: Configuration dictionary or None to use defaults
        """
        self._config = config or {}
        
        # Default intervals in seconds
        self.intervals = self._config.get("intervals", {
            "market_data": 900,  # 15 minutes
            "symbol_data": 300,  # 5 minutes
            "full_update": 3600,  # 1 hour
        })
        
        # Set up logging
        self.logger = logging.getLogger("TimerTrigger")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Controller instance
        self.controller = get_market_intelligence_controller()
        
        # Thread for timer loop
        self._timer_thread = None
        self._running = False
        self._stop_event = threading.Event()
        
        # Last update times
        self.last_updates = {
            "market_data": 0,
            "symbol_data": 0,
            "full_update": 0
        }
        
        # Market hours configuration
        self.market_hours = self._config.get("market_hours", {
            "enabled": True,
            "open_time": "09:30",
            "close_time": "16:00",
            "timezone": "America/New_York",
            "weekend_updates": False
        })
        
        # Initialize notification connector
        telegram_token = self._config.get("telegram", {}).get("token")
        telegram_chat_id = self._config.get("telegram", {}).get("chat_id")
        self.notification_connector = get_notification_connector(telegram_token, telegram_chat_id)
        
        self.logger.info("TimerTrigger initialized")
    
    def start(self):
        """Start the timer trigger."""
        if self._running:
            self.logger.info("Timer trigger already running")
            return
        
        self._running = True
        self._stop_event.clear()
        self._timer_thread = threading.Thread(target=self._run_timer_loop)
        self._timer_thread.daemon = True
        self._timer_thread.start()
        
        self.logger.info("Timer trigger started")
    
    def stop(self):
        """Stop the timer trigger."""
        if not self._running:
            return
        
        self._stop_event.set()
        self._running = False
        
        if self._timer_thread:
            self._timer_thread.join(timeout=5)
        
        self.logger.info("Timer trigger stopped")
    
    def _run_timer_loop(self):
        """Run the timer loop."""
        self.logger.info("Timer loop started")
        
        while not self._stop_event.is_set():
            try:
                # Check if we should run updates based on market hours
                if self._is_market_hours():
                    current_time = time.time()
                    
                    # Check if updates are due
                    for update_type, interval in self.intervals.items():
                        if current_time - self.last_updates.get(update_type, 0) >= interval:
                            self.logger.info(f"{update_type} update due")
                            
                            if update_type == "market_data":
                                # Update market data only
                                result = self.controller.market_context.update_market_data()
                                
                                # Send notification about the update
                                self.notification_connector.notify_market_update(
                                    "market_data",
                                    {
                                        "timestamp": datetime.datetime.now().isoformat(),
                                        "indices": result.get("indices", {}),
                                        "sentiment": result.get("sentiment", "")
                                    }
                                )
                                self.last_updates[update_type] = current_time
                                
                            elif update_type == "symbol_data":
                                # Update symbol data only
                                symbols = self.controller.market_context.get_top_symbols(20)
                                result = self.controller.market_context.update_symbol_data(symbols)
                                
                                # Send notification about the update
                                self.notification_connector.notify_market_update(
                                    "symbol_data",
                                    {
                                        "timestamp": datetime.datetime.now().isoformat(),
                                        "symbols": symbols,
                                        "noteworthy": result.get("noteworthy", [])
                                    }
                                )
                                self.last_updates[update_type] = current_time
                                
                            elif update_type == "full_update":
                                # Run full update
                                update_result = self.controller.update()
                                
                                # Get market context for notification
                                context = self.controller.market_context.get_market_context()
                                
                                # Send notification about the full update
                                self.notification_connector.notify_market_update(
                                    "full_update",
                                    {
                                        "timestamp": datetime.datetime.now().isoformat(),
                                        "market_regime": context.get("market", {}).get("regime", "unknown"),
                                        "top_strategies": context.get("strategies", {}).get("ranked", [])[:5]
                                    }
                                )
                                # Update all timestamps since full update covers everything
                                for key in self.last_updates.keys():
                                    self.last_updates[key] = current_time
                else:
                    self.logger.info("Outside market hours, skipping updates")
            
            except Exception as e:
                self.logger.error(f"Error in timer loop: {str(e)}")
            
            # Sleep for a bit to avoid busy-waiting
            time.sleep(60)  # Check every minute
    
    def _is_market_hours(self):
        """
        Check if we're currently in market hours.
        
        Returns:
            Boolean indicating if we're in market hours
        """
        # If market hours checking is disabled, always return True
        if not self.market_hours.get("enabled", True):
            return True
        
        # Get current time in configured timezone
        # This is a simplified version - in a real implementation, you'd use pytz
        now = datetime.datetime.now()
        current_time = now.strftime("%H:%M")
        
        # Get configured open/close times
        open_time = self.market_hours.get("open_time", "09:30")
        close_time = self.market_hours.get("close_time", "16:00")
        
        # Check if it's a weekend
        is_weekend = now.weekday() >= 5  # 5 = Saturday, 6 = Sunday
        
        # If it's a weekend and weekend updates are disabled, return False
        if is_weekend and not self.market_hours.get("weekend_updates", False):
            return False
        
        # Check if we're within market hours
        return open_time <= current_time <= close_time


class EventTrigger:
    """
    Event-based trigger that runs the Market Intelligence update when specific events occur.
    """
    
    def __init__(self, config=None):
        """
        Initialize the event trigger.
        
        Args:
            config: Configuration dictionary or None to use defaults
        """
        self._config = config or {}
        
        # Event thresholds
        self.thresholds = self._config.get("thresholds", {
            "vix_change": 1.0,  # VIX change percentage to trigger update
            "price_change": 3.0,  # Major price change percentage to trigger update
            "news_sentiment": 0.3  # News sentiment change to trigger update
        })
        
        # Set up logging
        self.logger = logging.getLogger("EventTrigger")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Controller instance
        self.controller = get_market_intelligence_controller()
        
        # Initialize notification connector
        telegram_token = self._config.get("telegram", {}).get("token")
        telegram_chat_id = self._config.get("telegram", {}).get("chat_id")
        self.notification_connector = get_notification_connector(telegram_token, telegram_chat_id)
        
        # Thread for event monitoring
        self._monitor_thread = None
        self._running = False
        self._stop_event = threading.Event()
        
        # Last event state
        self.last_state = {}
        
        # Event callbacks
        self.callbacks = {}
        
        self.logger.info("EventTrigger initialized")
    
    def start(self):
        """Start the event trigger."""
        if self._running:
            self.logger.info("Event trigger already running")
            return
        
        self._running = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._run_event_monitor)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
        self.logger.info("Event trigger started")
    
    def stop(self):
        """Stop the event trigger."""
        if not self._running:
            return
        
        self._stop_event.set()
        self._running = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        self.logger.info("Event trigger stopped")
    
    def register_callback(self, event_type, callback):
        """
        Register a callback for a specific event type.
        
        Args:
            event_type: Type of event to register for
            callback: Function to call when event occurs
        """
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        
        self.callbacks[event_type].append(callback)
        
        self.logger.info(f"Registered callback for {event_type}")
    
    def _run_event_monitor(self):
        """Run the event monitor loop."""
        self.logger.info("Event monitor started")
        
        # Initialize last state
        context = self.controller.market_context.get_market_context()
        self._update_last_state(context)
        
        while not self._stop_event.is_set():
            try:
                # Get latest context
                context = self.controller.market_context.get_market_context()
                
                # Check for triggering events
                events = self._check_for_events(context)
                
                # If events were detected, trigger callbacks
                for event in events:
                    event_type = event.get("type")
                    
                    # Call registered callbacks
                    for callback in self.callbacks.get(event_type, []):
                        try:
                            callback(event)
                        except Exception as e:
                            self.logger.error(f"Error in callback for {event_type}: {str(e)}")
                    
                    # Send notification for this event
                    try:
                        notification_result = self.notification_connector.notify_market_event(
                            event_type,
                            {
                                "description": event.get("description", f"{event_type} detected"),
                                "timestamp": datetime.datetime.now().isoformat(),
                                "severity": event.get("severity", "MEDIUM"),
                                "impact": event.get("impact", ""),
                                "recommendation": event.get("recommendation", ""),
                                "data": event.get("data", {})
                            }
                        )
                        self.logger.info(f"Notification sent for {event_type}: {notification_result}")
                    except Exception as e:
                        self.logger.error(f"Error sending notification for {event_type}: {str(e)}")
                    
                    # Also trigger update based on event type
                    if event_type == "vix_spike":
                        self.logger.info("VIX spike detected, triggering full update")
                        self.controller.update(force=True)
                    
                    elif event_type == "major_price_change":
                        symbol = event.get("symbol")
                        self.logger.info(f"Major price change for {symbol}, updating symbol data")
                        self.controller.market_context.update_symbol_data([symbol])
                    
                    elif event_type == "news_impact":
                        self.logger.info("Significant news impact detected, updating market data")
                        self.controller.market_context.update_market_data()
                
                # Update last state for next comparison
                self._update_last_state(context)
            
            except Exception as e:
                self.logger.error(f"Error in event monitor: {str(e)}")
            
            # Sleep for a bit to avoid busy-waiting
            time.sleep(60)  # Check every minute
    
    def _update_last_state(self, context):
        """
        Update the last state for event detection.
        
        Args:
            context: Current market context
        """
        # Extract relevant state information
        vix = context.get("market", {}).get("indicators", {}).get("vix")
        
        if vix:
            self.last_state["vix"] = vix
        
        # Extract symbol prices
        symbols = {}
        for symbol, data in context.get("symbols", {}).items():
            if "price" in data:
                symbols[symbol] = data["price"].get("current")
        
        self.last_state["symbols"] = symbols
    
    def _check_for_events(self, context):
        """
        Check for triggering events by comparing current context with last state.
        
        Args:
            context: Current market context
        
        Returns:
            List of detected events
        """
        events = []
        
        # Check VIX change
        current_vix = context.get("market", {}).get("indicators", {}).get("vix")
        if current_vix and "vix" in self.last_state:
            vix_change_pct = abs((current_vix - self.last_state["vix"]) / self.last_state["vix"] * 100)
            
            if vix_change_pct >= self.thresholds["vix_change"]:
                events.append({
                    "type": "vix_spike",
                    "old_value": self.last_state["vix"],
                    "new_value": current_vix,
                    "change_pct": vix_change_pct,
                    "timestamp": datetime.datetime.now().isoformat()
                })
        
        # Check price changes
        for symbol, data in context.get("symbols", {}).items():
            if "price" in data and symbol in self.last_state.get("symbols", {}):
                current_price = data["price"].get("current")
                last_price = self.last_state["symbols"][symbol]
                
                if current_price and last_price:
                    price_change_pct = abs((current_price - last_price) / last_price * 100)
                    
                    if price_change_pct >= self.thresholds["price_change"]:
                        events.append({
                            "type": "major_price_change",
                            "symbol": symbol,
                            "old_price": last_price,
                            "new_price": current_price,
                            "change_pct": price_change_pct,
                            "timestamp": datetime.datetime.now().isoformat()
                        })
        
        return events


class WebhookTrigger:
    """
    Webhook-based trigger that allows external systems to trigger Market Intelligence updates.
    This would typically be implemented as part of a web server, but for simplicity,
    we'll implement it as a standalone class that can be integrated into a web server.
    """
    
    def __init__(self, config=None):
        """
        Initialize the webhook trigger.
        
        Args:
            config: Configuration dictionary or None to use defaults
        """
        self._config = config or {}
        
        # Security settings
        self.security = self._config.get("security", {
            "api_key_required": True,
            "ip_whitelist": [],
            "rate_limit": 10  # Requests per minute
        })
        
        # Set up logging
        self.logger = logging.getLogger("WebhookTrigger")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Controller instance
        self.controller = get_market_intelligence_controller()
        
        # Initialize notification connector
        telegram_token = self._config.get("telegram", {}).get("token")
        telegram_chat_id = self._config.get("telegram", {}).get("chat_id")
        self.notification_connector = get_notification_connector(telegram_token, telegram_chat_id)
        
        # Request history for rate limiting
        self.request_history = []
        
        self.logger.info("WebhookTrigger initialized")
    
    def handle_webhook(self, request_data, request_headers=None, client_ip=None):
        """
        Handle a webhook request.
        
        Args:
            request_data: Dictionary containing request data
            request_headers: Dictionary containing request headers
            client_ip: IP address of the client
        
        Returns:
            Dictionary with response data
        """
        self.logger.info(f"Received webhook request: {request_data}")
        
        try:
            # Security checks
            if not self._check_security(request_headers, client_ip):
                return {"status": "error", "message": "Unauthorized"}
            
            # Rate limiting
            if not self._check_rate_limit(client_ip):
                return {"status": "error", "message": "Rate limit exceeded"}
            
            # Process the request
            action = request_data.get("action", "")
            
            if action == "update_market_data":
                self.controller.market_context.update_market_data()
                return {"status": "success", "message": "Market data update triggered"}
            
            elif action == "update_symbol_data":
                symbols = request_data.get("symbols", [])
                if not symbols:
                    return {"status": "error", "message": "No symbols specified"}
                
                self.controller.market_context.update_symbol_data(symbols)
                return {"status": "success", "message": f"Symbol data update triggered for {len(symbols)} symbols"}
            
            elif action == "full_update":
                force = request_data.get("force", False)
                update_result = self.controller.update(force=force)
                
                # Send notification about the manual update
                self.notification_connector.notify_market_update(
                    "webhook_update",
                    {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "source": request_data.get("source", "webhook"),
                        "result": {
                            "status": update_result.get("status", "unknown"),
                            "timestamp": update_result.get("timestamp", "")
                        }
                    }
                )
                
                return {"status": "success", "message": "Update triggered", "result": update_result}
            
            elif action == "backtest_pair":
                symbol = request_data.get("symbol")
                strategy = request_data.get("strategy")
                
                if not symbol or not strategy:
                    return {"status": "error", "message": "Symbol and strategy are required"}
                
                # Import BacktestExecutor
                from trading_bot.ml_pipeline.backtest_feedback_loop import get_backtest_executor
                executor = get_backtest_executor()
                
                # Run backtest
                result = executor.backtest_pair(symbol, strategy)
                
                # If the backtest was significant, send a notification
                if result.get("performance", {}).get("win_rate", 0) > 0.6 or \
                   abs(result.get("performance", {}).get("pnl_percent", 0)) > 5:
                    self.notification_connector.notify_market_update(
                        "backtest_result",
                        {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "symbol": symbol,
                            "strategy": strategy,
                            "performance": result.get("performance", {}),
                            "recommendation": "Consider implementing this strategy based on strong backtest results"
                            if result.get("performance", {}).get("pnl_percent", 0) > 5 else ""
                        }
                    )
                
                return {"status": "success", "message": "Backtest triggered", "result": result}
            
            else:
                return {"status": "error", "message": f"Unknown action: {action}"}
        
        except Exception as e:
            self.logger.error(f"Error handling webhook: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _check_security(self, headers, client_ip):
        """
        Check security requirements for the webhook.
        
        Args:
            headers: Request headers
            client_ip: Client IP address
        
        Returns:
            Boolean indicating if the request passes security checks
        """
        # API key check
        if self.security.get("api_key_required", True):
            if not headers or "X-API-Key" not in headers:
                self.logger.warning("API key missing")
                return False
            
            api_key = headers["X-API-Key"]
            valid_keys = self._config.get("api_keys", ["test_key"])
            
            if api_key not in valid_keys:
                self.logger.warning(f"Invalid API key: {api_key}")
                return False
        
        # IP whitelist check
        ip_whitelist = self.security.get("ip_whitelist", [])
        if ip_whitelist and client_ip not in ip_whitelist:
            self.logger.warning(f"IP not in whitelist: {client_ip}")
            return False
        
        return True
    
    def _check_rate_limit(self, client_ip):
        """
        Check rate limiting for the webhook.
        
        Args:
            client_ip: Client IP address
        
        Returns:
            Boolean indicating if the request passes rate limiting
        """
        # Clean up old requests
        current_time = time.time()
        self.request_history = [
            r for r in self.request_history
            if current_time - r["timestamp"] < 60  # Within the last minute
        ]
        
        # Count requests for this IP
        ip_requests = sum(1 for r in self.request_history if r["ip"] == client_ip)
        
        # Check if rate limit is exceeded
        rate_limit = self.security.get("rate_limit", 10)
        if ip_requests >= rate_limit:
            self.logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return False
        
        # Add this request to history
        self.request_history.append({
            "ip": client_ip,
            "timestamp": current_time
        })
        
        return True


# Create singleton instances
_timer_trigger = None
_event_trigger = None
_webhook_trigger = None

def get_timer_trigger(config=None):
    """
    Get the singleton TimerTrigger instance.
    
    Args:
        config: Optional configuration for the trigger
    
    Returns:
        TimerTrigger instance
    """
    global _timer_trigger
    if _timer_trigger is None:
        _timer_trigger = TimerTrigger(config)
    return _timer_trigger

def get_event_trigger(config=None):
    """
    Get the singleton EventTrigger instance.
    
    Args:
        config: Optional configuration for the trigger
    
    Returns:
        EventTrigger instance
    """
    global _event_trigger
    if _event_trigger is None:
        _event_trigger = EventTrigger(config)
    return _event_trigger

def get_webhook_trigger(config=None):
    """
    Get the singleton WebhookTrigger instance.
    
    Args:
        config: Optional configuration for the trigger
    
    Returns:
        WebhookTrigger instance
    """
    global _webhook_trigger
    if _webhook_trigger is None:
        _webhook_trigger = WebhookTrigger(config)
    return _webhook_trigger
