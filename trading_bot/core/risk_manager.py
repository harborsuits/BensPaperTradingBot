#!/usr/bin/env python3
"""
Risk Manager for BensBot

This module implements the risk management system for BensBot, including:
- Margin and leverage monitoring
- Portfolio-level circuit breakers for drawdown protection
- Volatility-based circuit breakers
- Automatic de-leveraging during margin calls
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import numpy as np
from collections import defaultdict

from trading_bot.core.events import (
    OrderFilled, OrderPartialFill, PortfolioEquityUpdate,
    VolatilityUpdate, TradingPaused, TradingResumed,
    MarginCall, MarginCallWarning, ForcedExitOrder
)
from trading_bot.event_system.event_bus import EventBus


# Event classes are now defined in events.py


class RiskManager:
    """
    Risk Manager for BensBot
    
    Responsible for monitoring margin usage, portfolio drawdowns,
    and triggering circuit breakers or forced exits when thresholds
    are exceeded.
    """
    
    def __init__(self, trading_system, config, event_bus: EventBus, alert_system):
        """
        Initialize the risk manager
        
        Args:
            trading_system: Trading system instance
            config: Configuration dictionary
            event_bus: EventBus for event subscription/emission
            alert_system: Alert system for notifications
        """
        self.system = trading_system
        self.config = config
        self.bus = event_bus
        self.alert = alert_system
        self.logger = logging.getLogger(__name__)
        
        # Extract margin configuration
        risk_config = config.get("risk", {})
        self.margin_threshold = risk_config.get("max_margin_ratio", 0.5)
        self.margin_buffer = risk_config.get("margin_call_buffer", 0.05)
        self.margin_check_interval = risk_config.get("margin_check_interval", 60)
        
        # Track margin status by broker
        self.margin_status = {}
        
        # Track if we're currently in a margin call
        self.active_margin_calls = set()
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Shutdown flag
        self.shutdown = False
        
        # Subscribe to relevant events
        self._subscribe_to_events()
        
        # Track last notification times to avoid spam
        self.last_warning_time = {}
        self.last_call_time = {}
        
        # Broker-specific configurations
        self.broker_specific = risk_config.get("broker_specific", {})
        
        # Forced deleveraging settings
        self.deleveraging_config = risk_config.get("forced_deleveraging", {
            "max_positions_per_step": 5,
            "priority_order": "largest_notional",
            "pause_seconds_between_steps": 30
        })
        
        # Start periodic margin check
        self._start_periodic_check()
        
        self.logger.info("Risk Manager initialized")
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events"""
        self.bus.on(OrderFilled, self._on_fill)
        self.bus.on(OrderPartialFill, self._on_fill)
    
    def _on_fill(self, event):
        """
        Handle order fill events
        
        Args:
            event: Order fill event
        """
        # Immediately check margin after any fill
        self._check_margin()
    
    def _start_periodic_check(self):
        """Start periodic margin check thread"""
        def loop():
            while not self.shutdown:
                try:
                    self._check_margin()
                except Exception as e:
                    self.logger.error(f"Error in margin check: {str(e)}")
                
                # Sleep until next check
                for _ in range(int(self.margin_check_interval)):
                    if self.shutdown:
                        break
                    time.sleep(1)
        
        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        
        self.logger.debug(f"Started periodic margin check every {self.margin_check_interval}s")
    
    def _check_margin(self):
        """Check margin status for all brokers"""
        try:
            # Get margin status for all brokers
            margin_statuses = self.get_margin_status()
            
            if not margin_statuses:
                self.logger.warning("No margin status available for any broker")
                return
            
            with self.lock:
                # Check each broker's margin status
                for broker_key, status in margin_statuses.items():
                    try:
                        if not status:
                            self.logger.warning(f"No margin status returned by broker {broker_key}")
                            continue
                        
                        # Store status
                        self.margin_status[broker_key] = status
                        
                        # Get broker-specific thresholds if available
                        margin_threshold = self.broker_specific.get(broker_key, {}).get(
                            "max_margin_ratio", self.margin_threshold)
                        margin_buffer = self.broker_specific.get(broker_key, {}).get(
                            "margin_call_buffer", self.margin_buffer)
                        
                        # Calculate margin ratio
                        if status.get("maintenance_requirement", 0) > 0:
                            ratio = status.get("margin_used", 0) / status.get("maintenance_requirement", 0)
                        else:
                            ratio = 0
                        
                        # Check thresholds
                        if ratio >= margin_threshold:
                            # Margin call
                            self._trigger_margin_call(broker_key, ratio, status, margin_threshold)
                        elif ratio >= (margin_threshold - margin_buffer):
                            # Approaching margin call
                            self._trigger_margin_buffer_warning(broker_key, ratio, status, 
                                                              margin_threshold, margin_buffer)
                        
                    except Exception as e:
                        self.logger.error(f"Error checking margin for {broker_key}: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error in margin check: {str(e)}")
    
    def _trigger_margin_buffer_warning(self, broker_key: str, ratio: float, status: Dict[str, Any], 
                                    threshold: float, buffer: float):
        """
        Trigger a margin buffer warning
        
        Args:
            broker_key: Broker identifier
            ratio: Current margin ratio
            status: Full margin status
            threshold: Margin call threshold
            buffer: Margin buffer
        """
        # Check if we should throttle this warning
        now = datetime.now()
        last_time = self.last_warning_time.get(broker_key, datetime.min)
        
        # Throttle warnings to at most once per hour per broker
        if (now - last_time).total_seconds() < 3600:
            self.logger.debug(f"Suppressing repeat margin warning for {broker_key}")
            return
            
        self.logger.warning(f"Margin buffer warning for {broker_key}: {ratio:.2%}")
        self.last_warning_time[broker_key] = now
        
        # Get notification settings
        notification_config = self.config.get("risk", {}).get("notifications", {}).get("margin_buffer_warning", {})
        
        # Calculate remaining buffer
        remaining_buffer = (threshold - ratio) / buffer if buffer > 0 else 0
        
        # Emit event
        self.bus.emit(MarginCallWarning(
            broker=broker_key, 
            ratio=ratio, 
            threshold=threshold, 
            account_id=status.get("account_id", "unknown"),
            remaining_buffer=remaining_buffer
        ))
        
        # Send alert
        if self.alert:
            channels = notification_config.get("channels", ["dashboard", "email"])
            level = notification_config.get("level", "warning")
            
            self.alert.send_alert(
                level=level,
                message=f"Margin buffer warning for {broker_key}: {ratio:.2%}",
                channels=channels
            )
    
    def _trigger_margin_call(self, broker_key: str, ratio: float, status: Dict[str, Any], threshold: float):
        """
        Trigger a margin call
        
        Args:
            broker_key: Broker identifier
            ratio: Current margin ratio
            status: Full margin status
            threshold: Margin call threshold
        """
        # Check if we should throttle this warning
        now = datetime.now()
        last_time = self.last_call_time.get(broker_key, datetime.min)
        
        # Check if this is a new margin call or if we should re-handle it due to time passing
        is_new_margin_call = broker_key not in self.active_margin_calls
        should_rehandle = (now - last_time).total_seconds() > 900  # Re-handle every 15 min
        
        # Add to active margin calls set
        self.active_margin_calls.add(broker_key)
        
        # Get notification settings
        notification_config = self.config.get("risk", {}).get("notifications", {}).get("margin_call", {})
        
        # Determine severity based on how far above threshold
        if ratio >= threshold * 1.25:
            severity = "critical"
        elif ratio >= threshold * 1.1:
            severity = "high"
        else:
            severity = "medium"
        
        # If this is a new margin call or we should re-handle, log and notify
        if is_new_margin_call or should_rehandle:
            self.logger.critical(f"Margin call triggered for {broker_key}: {ratio:.2%}")
            self.last_call_time[broker_key] = now
            
            # Emit event
            self.bus.emit(MarginCall(
                broker=broker_key, 
                ratio=ratio, 
                threshold=threshold,
                account_id=status.get("account_id", "unknown"),
                severity=severity
            ))
            
            # Send alert
            if self.alert:
                channels = notification_config.get("channels", ["dashboard", "email", "sms"])
                level = notification_config.get("level", "critical")
                
                self.alert.send_alert(
                    level=level,
                    message=f"Margin call triggered for {broker_key}: {ratio:.2%}",
                    channels=channels
                )
            
            # Handle the margin call
            self._handle_margin_call(broker_key, status)
        
        # Otherwise, just log the ongoing margin call
        else:
            self.logger.warning(f"Ongoing margin call for {broker_key}: {ratio:.2%}")
    
    def _handle_margin_call(self, broker_key: str, status: Dict[str, Any]):
        """
        Handle a margin call by de-leveraging positions
        
        Args:
            broker_key: Broker identifier
            status: Margin status
        """
        self.logger.info(f"Handling margin call for {broker_key}")
        
        try:
            # Get broker manager and position manager
            registry = self.system.service_registry
            broker_manager = registry.get("broker_manager")
            position_manager = registry.get("position_manager")
            
            if not broker_manager or not position_manager:
                self.logger.error("Missing required components to handle margin call")
                return
            
            # Get positions for this broker
            positions = position_manager.get_positions_by_broker(broker_key)
            
            if not positions:
                self.logger.warning(f"No positions found for broker {broker_key} to handle margin call")
                return
            
            # Get forced deleveraging config
            max_positions = self.deleveraging_config.get("max_positions_per_step", 5)
            priority = self.deleveraging_config.get("priority_order", "largest_notional")
            
            # Sort positions by the specified priority
            if priority == "largest_notional":
                positions.sort(key=lambda p: abs(p.quantity * p.avg_price), reverse=True)
            elif priority == "newest_first":
                positions.sort(key=lambda p: p.entry_time, reverse=True)
            elif priority == "oldest_first":
                positions.sort(key=lambda p: p.entry_time)
            elif priority == "largest_pnl":
                positions.sort(key=lambda p: p.unrealized_pnl, reverse=True)
            elif priority == "largest_loss":
                positions.sort(key=lambda p: p.unrealized_pnl)
            
            # Exit positions based on priority
            for position in positions[:max_positions]:
                self.logger.warning(f"Forced exit of {position.symbol} due to margin call")
                
                # Emit forced exit event
                self.bus.emit(ForcedExitOrder(
                    symbol=position.symbol,
                    qty=position.quantity,
                    reason=f"margin_call_{broker_key}",
                    broker=broker_key,
                    order_type="market"
                ))
                
            # If we still have more positions to exit, log that we'll continue in the next cycle
            if len(positions) > max_positions:
                self.logger.info(f"Will continue deleveraging {len(positions) - max_positions} more positions")
        
        except Exception as e:
            self.logger.error(f"Error handling margin call: {str(e)}")
    
    def get_margin_status(self):
        """
        Get current margin status for all brokers
        
        Returns:
            Dictionary of broker margin status
        """
        try:
            # Get service registry and broker manager
            registry = self.system.service_registry
            broker_manager = registry.get("broker_manager")
            
            if not broker_manager:
                self.logger.warning("No broker manager available for margin status check")
                return {}
            
            # Get all brokers
            brokers = broker_manager.get_all_brokers()
            
            # Collect margin status for each broker
            margin_statuses = {}
            for broker_key, broker in brokers.items():
                try:
                    # Get margin status
                    status = broker.get_margin_status()
                    margin_statuses[broker_key] = status
                except Exception as e:
                    self.logger.error(f"Error getting margin status for {broker_key}: {str(e)}")
            
            # Return all statuses
            return margin_statuses
            
        except Exception as e:
            self.logger.error(f"Error getting margin status: {str(e)}")
            return {}
    
    def is_in_margin_call(self, broker_key: Optional[str] = None) -> bool:
        """
        Check if a broker is currently in margin call
        
        Args:
            broker_key: Broker identifier, or None to check any broker
            
        Returns:
            True if in margin call
        """
        with self.lock:
            if broker_key:
                return broker_key in self.active_margin_calls
            else:
                return len(self.active_margin_calls) > 0
    
    def shutdown(self):
        """Shutdown the risk manager"""
        self.shutdown = True
        self.logger.info("Risk Manager shutdown")


class PortfolioCircuitBreaker:
    """
    Portfolio Circuit Breaker
    
    Monitors portfolio equity and volatility, triggering circuit breakers
    when drawdown or volatility thresholds are exceeded.
    """
    
    def __init__(self, trading_system, config, event_bus: EventBus, alert_system):
        """
        Initialize the portfolio circuit breaker
        
        Args:
            trading_system: Trading system instance
            config: Configuration dictionary
            event_bus: EventBus for event subscription/emission
            alert_system: Alert system for notifications
        """
        self.system = trading_system
        self.config = config
        self.bus = event_bus
        self.alert = alert_system
        self.logger = logging.getLogger(__name__)
        
        # Extract circuit breaker configuration
        risk_config = config.get("risk", {})
        breaker_config = risk_config.get("breakers", {})
        self.intraday_drawdown_threshold = breaker_config.get("intraday_drawdown_threshold", 0.05)
        self.overall_drawdown_threshold = breaker_config.get("overall_drawdown_threshold", 0.10)
        self.volatility_threshold = breaker_config.get("volatility_threshold", 0.025)
        
        # Track peak equity values
        self.peak_equity = self.system.position_manager.current_equity()
        self.daily_peak_equity = self.peak_equity
        self.last_reset_date = datetime.now().date()
        
        # Track circuit breaker status
        self.is_breaker_triggered = False
        self.active_breakers = set()
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Subscribe to relevant events
        self._subscribe_to_events()
        
        self.logger.info("Portfolio Circuit Breaker initialized")
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events"""
        self.bus.on(PortfolioEquityUpdate, self._on_equity_change)
        self.bus.on(VolatilityUpdate, self._on_volatility_update)
    
    def _on_equity_change(self, event):
        """
        Handle portfolio equity update events
        
        Args:
            event: Portfolio equity update event
        """
        equity = event.equity
        current_date = datetime.now().date()
        
        # Reset daily peak if date changed
        if current_date != self.last_reset_date:
            with self.lock:
                self.daily_peak_equity = equity
                self.last_reset_date = current_date
        
        # Update peak values
        with self.lock:
            # Update overall peak equity
            self.peak_equity = max(self.peak_equity, equity)
            
            # Update daily peak equity
            self.daily_peak_equity = max(self.daily_peak_equity, equity)
            
            # Calculate drawdowns
            if self.peak_equity <= 0:
                overall_drawdown = 0
            else:
                overall_drawdown = (self.peak_equity - equity) / self.peak_equity
            
            if self.daily_peak_equity <= 0:
                intraday_drawdown = 0
            else:
                intraday_drawdown = (self.daily_peak_equity - equity) / self.daily_peak_equity
        
        # Check intraday breaker
        if intraday_drawdown >= self.intraday_drawdown_threshold:
            self._trigger("intraday", intraday_drawdown)
        
        # Check overall breaker
        if overall_drawdown >= self.overall_drawdown_threshold:
            self._trigger("overall", overall_drawdown)
    
    def _on_volatility_update(self, event):
        """
        Handle volatility update events
        
        Args:
            event: Volatility update event
        """
        volatility = event.volatility
        
        # Check volatility breaker
        if volatility >= self.volatility_threshold:
            self._trigger("volatility", volatility)
    
    def _trigger(self, breaker_type: str, value: float):
        """
        Trigger a circuit breaker
        
        Args:
            breaker_type: Type of circuit breaker
            value: Current value that triggered the breaker
        """
        # Check if this type of breaker is already active
        is_new_trigger = False
        with self.lock:
            if breaker_type not in self.active_breakers:
                self.active_breakers.add(breaker_type)
                is_new_trigger = True
                self.is_breaker_triggered = True
        
        if is_new_trigger:
            if breaker_type == "intraday":
                message = f"Intraday drawdown of {value:.1%} exceeded threshold of {self.intraday_drawdown_threshold:.1%}"
            elif breaker_type == "overall":
                message = f"Overall drawdown of {value:.1%} exceeded threshold of {self.overall_drawdown_threshold:.1%}"
            elif breaker_type == "volatility":
                message = f"Portfolio volatility of {value:.1%} exceeded threshold of {self.volatility_threshold:.1%}"
            else:
                message = f"Circuit breaker triggered: {breaker_type}, value: {value:.1%}"
            
            self.logger.critical(message + ". Pausing trading.")
            
            # Pause trading in the orchestrator
            self.system.orchestrator.pause_trading(reason=f"{breaker_type}_breaker")
            
            # Emit event
            self.bus.emit(TradingPaused(reason=f"{breaker_type}_breaker"))
            
            # Send alert
            self.alert.send_alert(
                level="critical",
                message=message + " Trading has been paused."
            )
    
    def reset_breaker(self, breaker_type: Optional[str] = None):
        """
        Reset a circuit breaker
        
        Args:
            breaker_type: Type of circuit breaker to reset, or None for all
        """
        with self.lock:
            if breaker_type:
                if breaker_type in self.active_breakers:
                    self.active_breakers.remove(breaker_type)
                    self.logger.info(f"Reset {breaker_type} circuit breaker")
            else:
                # Reset all breakers
                self.active_breakers.clear()
                self.logger.info("Reset all circuit breakers")
            
            # Update overall triggered state
            self.is_breaker_triggered = len(self.active_breakers) > 0
        
        # If no more active breakers, resume trading
        if not self.is_breaker_triggered:
            self.system.orchestrator.resume_trading()
            self.bus.emit(TradingResumed())
            
            # Send alert
            self.alert.send_alert(
                level="info",
                message="All circuit breakers cleared. Trading resumed."
            )
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get circuit breaker status
        
        Returns:
            Dictionary of status information
        """
        with self.lock:
            current_equity = self.system.position_manager.current_equity()
            
            # Calculate drawdowns
            if self.peak_equity <= 0:
                overall_drawdown = 0
            else:
                overall_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            
            if self.daily_peak_equity <= 0:
                intraday_drawdown = 0
            else:
                intraday_drawdown = (self.daily_peak_equity - current_equity) / self.daily_peak_equity
            
            return {
                "is_triggered": self.is_breaker_triggered,
                "active_breakers": list(self.active_breakers),
                "current_equity": current_equity,
                "peak_equity": self.peak_equity,
                "daily_peak_equity": self.daily_peak_equity,
                "overall_drawdown": overall_drawdown,
                "intraday_drawdown": intraday_drawdown,
                "overall_threshold": self.overall_drawdown_threshold,
                "intraday_threshold": self.intraday_drawdown_threshold,
                "volatility_threshold": self.volatility_threshold
            }
    
    def is_triggered(self) -> bool:
        """
        Check if any circuit breaker is triggered
        
        Returns:
            True if triggered
        """
        with self.lock:
            return self.is_breaker_triggered
    
    def is_breaker_active(self, breaker_type: str) -> bool:
        """
        Check if a specific circuit breaker is active
        
        Args:
            breaker_type: Type of circuit breaker
            
        Returns:
            True if active
        """
        with self.lock:
            return breaker_type in self.active_breakers
