"""
Emergency Stop Procedures

This module implements comprehensive emergency stop mechanisms to protect
the trading system during adverse market conditions or system failures.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Set, Callable
from datetime import datetime, timedelta
import threading
import requests
import json

logger = logging.getLogger(__name__)

class EmergencyBrake:
    """
    Emergency stop procedures for protecting capital during adverse conditions.
    
    The EmergencyBrake implements multiple protective mechanisms to automatically
    pause or stop trading activities when predefined risk thresholds are exceeded
    or when system anomalies are detected.
    
    Features:
    1. Strategy-specific risk thresholds (drawdown, consecutive losses)
    2. Global risk thresholds for portfolio protection
    3. System health monitoring via heartbeat checks
    4. Manual kill switch capability for immediate trading suspension
    5. Execution quality monitoring (slippage detection)
    6. Notification system for emergency events
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the EmergencyBrake system with configuration parameters.
        
        Args:
            config: Dictionary of configuration parameters
        """
        self.config = config or {}
        
        # Risk thresholds
        self.max_consecutive_losses = self.config.get('max_consecutive_losses', 3)
        self.strategy_drawdown_limit_pct = self.config.get('strategy_drawdown_limit_pct', 0.10)
        self.global_drawdown_limit_pct = self.config.get('global_drawdown_limit_pct', 0.05)
        self.max_slippage_pct = self.config.get('max_slippage_pct', 0.01)
        
        # System health monitoring
        self.heartbeat_interval_secs = self.config.get('heartbeat_interval_secs', 30)
        self.heartbeat_timeout_secs = self.config.get('heartbeat_timeout_secs', 90)
        
        # Kill switch config
        self.kill_switch_endpoint = self.config.get('kill_switch_endpoint', '/api/system/kill_switch')
        self.kill_switch_persist = self.config.get('kill_switch_persist', True)
        
        # State tracking
        self.consecutive_losses = {}  # strategy_id -> count
        self.strategy_high_water_marks = {}  # strategy_id -> max_equity
        self.global_high_water_mark = 0.0
        self.killed_strategies = set()  # Set of paused strategy IDs
        self.global_kill_switch_active = False
        self.last_heartbeats = {}  # module_name -> timestamp
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Callbacks
        self.notification_callback = None
        self.strategy_pause_callback = None
        self.global_pause_callback = None
        
        logger.info(f"Initialized EmergencyBrake with max_consecutive_losses: {self.max_consecutive_losses}, "
                    f"global_drawdown_limit: {self.global_drawdown_limit_pct:.1%}, "
                    f"strategy_drawdown_limit: {self.strategy_drawdown_limit_pct:.1%}")
                    
    def register_callbacks(self, 
                          notify_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
                          strategy_pause_callback: Optional[Callable[[str, str, Dict[str, Any]], bool]] = None,
                          global_pause_callback: Optional[Callable[[str, Dict[str, Any]], bool]] = None):
        """
        Register callbacks for emergency actions.
        
        Args:
            notify_callback: Function to call for notifications (severity, message, data)
            strategy_pause_callback: Function to call to pause a strategy (strategy_id, reason, data)
            global_pause_callback: Function to call to pause all trading (reason, data)
        """
        self.notification_callback = notify_callback
        self.strategy_pause_callback = strategy_pause_callback
        self.global_pause_callback = global_pause_callback
        
    def start_monitoring(self):
        """Start the background monitoring thread for system health checks."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_system_health,
            daemon=True  # Thread will exit when main program exits
        )
        self.monitor_thread.start()
        logger.info("Started emergency brake monitoring thread")
        
    def stop_monitoring(self):
        """Stop the background monitoring thread."""
        self.monitoring_active = False
        if self.monitor_thread:
            # Let the thread exit naturally on next loop
            self.monitor_thread.join(timeout=self.heartbeat_interval_secs + 5)
            self.monitor_thread = None
        logger.info("Stopped emergency brake monitoring thread")
        
    def record_heartbeat(self, module_name: str):
        """
        Record a heartbeat from a system module.
        
        Args:
            module_name: Name of the module reporting heartbeat
        """
        self.last_heartbeats[module_name] = datetime.now()
        
    def record_trade_result(self, 
                           strategy_id: str, 
                           trade_profit: float, 
                           slippage_pct: Optional[float] = None):
        """
        Record a completed trade result for risk monitoring.
        
        Args:
            strategy_id: Identifier for the strategy that executed the trade
            trade_profit: Profit/loss amount (positive for profit, negative for loss)
            slippage_pct: Optional slippage percentage observed during execution
        
        Returns:
            bool: False if this trade triggered an emergency stop, True otherwise
        """
        # Initialize tracking for new strategies
        if strategy_id not in self.consecutive_losses:
            self.consecutive_losses[strategy_id] = 0
            
        # Check for loss
        if trade_profit < 0:
            self.consecutive_losses[strategy_id] += 1
            logger.debug(f"Strategy {strategy_id} recorded loss #{self.consecutive_losses[strategy_id]}")
            
            # Check consecutive loss threshold
            if self.consecutive_losses[strategy_id] >= self.max_consecutive_losses:
                self._trigger_strategy_pause(
                    strategy_id, 
                    f"Max consecutive losses ({self.max_consecutive_losses}) reached",
                    {
                        "consecutive_losses": self.consecutive_losses[strategy_id],
                        "threshold": self.max_consecutive_losses,
                        "last_loss": trade_profit
                    }
                )
                return False
        else:
            # Reset counter on profit
            if strategy_id in self.consecutive_losses:
                self.consecutive_losses[strategy_id] = 0
                
        # Check slippage if provided
        if slippage_pct is not None and slippage_pct > self.max_slippage_pct:
            self._trigger_strategy_pause(
                strategy_id,
                f"Excessive slippage detected: {slippage_pct:.2%}",
                {
                    "slippage": slippage_pct,
                    "threshold": self.max_slippage_pct,
                    "trade_profit": trade_profit
                }
            )
            return False
            
        return True
        
    def update_strategy_equity(self, strategy_id: str, current_equity: float):
        """
        Update equity tracking for drawdown monitoring on strategy level.
        
        Args:
            strategy_id: Strategy identifier
            current_equity: Current equity allocated to this strategy
            
        Returns:
            bool: False if this update triggered an emergency stop, True otherwise
        """
        # Initialize or update high water mark
        if strategy_id not in self.strategy_high_water_marks:
            self.strategy_high_water_marks[strategy_id] = current_equity
        elif current_equity > self.strategy_high_water_marks[strategy_id]:
            self.strategy_high_water_marks[strategy_id] = current_equity
            
        # Calculate drawdown
        high_water = self.strategy_high_water_marks[strategy_id]
        if high_water <= 0:  # Avoid division by zero
            return True
            
        drawdown = (high_water - current_equity) / high_water
        
        # Check drawdown threshold
        if drawdown > self.strategy_drawdown_limit_pct:
            self._trigger_strategy_pause(
                strategy_id,
                f"Strategy drawdown limit exceeded: {drawdown:.2%}",
                {
                    "drawdown": drawdown,
                    "threshold": self.strategy_drawdown_limit_pct,
                    "high_water_mark": high_water,
                    "current_equity": current_equity
                }
            )
            return False
            
        return True
        
    def update_portfolio_equity(self, total_equity: float):
        """
        Update equity tracking for drawdown monitoring on portfolio level.
        
        Args:
            total_equity: Current total portfolio equity
            
        Returns:
            bool: False if this update triggered a global emergency stop, True otherwise
        """
        # Initialize or update high water mark
        if self.global_high_water_mark <= 0:
            self.global_high_water_mark = total_equity
        elif total_equity > self.global_high_water_mark:
            self.global_high_water_mark = total_equity
            
        # Calculate drawdown
        if self.global_high_water_mark <= 0:  # Avoid division by zero
            return True
            
        drawdown = (self.global_high_water_mark - total_equity) / self.global_high_water_mark
        
        # Check drawdown threshold
        if drawdown > self.global_drawdown_limit_pct:
            self._trigger_global_pause(
                f"Global drawdown limit exceeded: {drawdown:.2%}",
                {
                    "drawdown": drawdown,
                    "threshold": self.global_drawdown_limit_pct,
                    "high_water_mark": self.global_high_water_mark,
                    "current_equity": total_equity
                }
            )
            return False
            
        return True
        
    def activate_kill_switch(self, reason: str = "Manual activation"):
        """
        Activate global kill switch to pause all trading.
        
        Args:
            reason: Reason for activating the kill switch
            
        Returns:
            bool: True if successfully activated, False otherwise
        """
        if self.global_kill_switch_active:
            logger.warning(f"Kill switch already active, ignoring activation request: {reason}")
            return True
            
        # Activate kill switch
        self.global_kill_switch_active = True
        
        # Trigger global pause callback
        success = self._trigger_global_pause(
            f"Kill switch activated: {reason}",
            {"reason": reason, "timestamp": datetime.now().isoformat()}
        )
        
        # Make API call if endpoint is configured
        endpoint = self.kill_switch_endpoint
        if endpoint and endpoint.startswith('/'):
            try:
                response = requests.post(
                    endpoint,
                    json={
                        "activated": True,
                        "reason": reason,
                        "persist": self.kill_switch_persist,
                        "timestamp": datetime.now().isoformat()
                    },
                    timeout=5
                )
                if response.status_code != 200:
                    logger.error(f"Failed to call kill switch API: {response.status_code}, {response.text}")
            except Exception as e:
                logger.error(f"Error calling kill switch API: {str(e)}")
                
        return success
        
    def deactivate_kill_switch(self, reason: str = "Manual deactivation"):
        """
        Deactivate global kill switch to resume trading.
        
        Args:
            reason: Reason for deactivating the kill switch
            
        Returns:
            bool: True if successfully deactivated, False otherwise
        """
        if not self.global_kill_switch_active:
            logger.warning(f"Kill switch not active, ignoring deactivation request: {reason}")
            return True
            
        # Deactivate kill switch
        self.global_kill_switch_active = False
        
        # Make notification
        if self.notification_callback:
            self.notification_callback(
                "info",
                f"Kill switch deactivated: {reason}",
                {"reason": reason, "timestamp": datetime.now().isoformat()}
            )
            
        # Make API call if endpoint is configured
        endpoint = self.kill_switch_endpoint
        if endpoint and endpoint.startswith('/'):
            try:
                response = requests.post(
                    endpoint,
                    json={
                        "activated": False,
                        "reason": reason,
                        "timestamp": datetime.now().isoformat()
                    },
                    timeout=5
                )
                if response.status_code != 200:
                    logger.error(f"Failed to call kill switch API: {response.status_code}, {response.text}")
                    return False
            except Exception as e:
                logger.error(f"Error calling kill switch API: {str(e)}")
                return False
                
        return True
        
    def reset_strategy_counters(self, strategy_id: str):
        """
        Reset risk counters for a specific strategy.
        
        Args:
            strategy_id: Strategy identifier to reset
        """
        if strategy_id in self.consecutive_losses:
            self.consecutive_losses[strategy_id] = 0
            
        # Remove from killed strategies set if present
        self.killed_strategies.discard(strategy_id)
        
        logger.info(f"Reset risk counters for strategy: {strategy_id}")
        
    def is_strategy_active(self, strategy_id: str) -> bool:
        """
        Check if a strategy is currently allowed to trade.
        
        Args:
            strategy_id: Strategy identifier to check
            
        Returns:
            bool: True if strategy is active, False if paused
        """
        return not (self.global_kill_switch_active or strategy_id in self.killed_strategies)
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the emergency brake system.
        
        Returns:
            Dictionary with current status information
        """
        now = datetime.now()
        
        # Check heartbeat status
        modules_status = {}
        for module, last_beat in self.last_heartbeats.items():
            time_since_beat = (now - last_beat).total_seconds()
            modules_status[module] = {
                "last_heartbeat": last_beat.isoformat(),
                "seconds_ago": time_since_beat,
                "status": "active" if time_since_beat < self.heartbeat_timeout_secs else "timeout"
            }
            
        return {
            "global_kill_switch": self.global_kill_switch_active,
            "paused_strategies": list(self.killed_strategies),
            "consecutive_losses": self.consecutive_losses,
            "monitoring_active": self.monitoring_active,
            "modules_status": modules_status,
            "risk_thresholds": {
                "max_consecutive_losses": self.max_consecutive_losses,
                "strategy_drawdown_limit": self.strategy_drawdown_limit_pct,
                "global_drawdown_limit": self.global_drawdown_limit_pct,
                "max_slippage_pct": self.max_slippage_pct
            }
        }
        
    def _monitor_system_health(self):
        """Background thread function for monitoring system health."""
        logger.info("System health monitoring thread started")
        
        while self.monitoring_active:
            now = datetime.now()
            
            # Check for heartbeat timeouts
            for module, last_beat in list(self.last_heartbeats.items()):
                time_since_beat = (now - last_beat).total_seconds()
                
                # Trigger alert if timeout exceeded
                if time_since_beat > self.heartbeat_timeout_secs:
                    # Log the event
                    logger.warning(f"Module heartbeat timeout: {module}, "
                                 f"Last seen {time_since_beat:.1f}s ago (limit: {self.heartbeat_timeout_secs}s)")
                    
                    # Make notification
                    if self.notification_callback:
                        self.notification_callback(
                            "warning",
                            f"Module heartbeat timeout: {module}",
                            {
                                "module": module,
                                "seconds_since_heartbeat": time_since_beat,
                                "threshold": self.heartbeat_timeout_secs
                            }
                        )
                    
                    # Critical modules should trigger global pause
                    if module in ['order_manager', 'execution_engine', 'risk_manager']:
                        self._trigger_global_pause(
                            f"Critical module timeout: {module}",
                            {
                                "module": module,
                                "seconds_since_heartbeat": time_since_beat,
                                "threshold": self.heartbeat_timeout_secs
                            }
                        )
            
            # Sleep until next check
            time.sleep(self.heartbeat_interval_secs)
            
        logger.info("System health monitoring thread stopped")
        
    def _trigger_strategy_pause(self, strategy_id: str, reason: str, data: Dict[str, Any]) -> bool:
        """
        Internal method to pause a strategy and trigger notifications.
        
        Args:
            strategy_id: Strategy to pause
            reason: Reason for pausing
            data: Additional data about the event
            
        Returns:
            bool: True if successfully paused, False otherwise
        """
        # Add to killed strategies set
        self.killed_strategies.add(strategy_id)
        
        # Log the event
        logger.warning(f"Emergency stop for strategy {strategy_id}: {reason}")
        
        # Make notification
        if self.notification_callback:
            self.notification_callback(
                "warning",
                f"Strategy paused: {strategy_id} - {reason}",
                {**data, "strategy_id": strategy_id}
            )
        
        # Call strategy pause callback if registered
        if self.strategy_pause_callback:
            return self.strategy_pause_callback(strategy_id, reason, data)
            
        return True
        
    def _trigger_global_pause(self, reason: str, data: Dict[str, Any]) -> bool:
        """
        Internal method to pause all trading and trigger notifications.
        
        Args:
            reason: Reason for global pause
            data: Additional data about the event
            
        Returns:
            bool: True if successfully paused, False otherwise
        """
        # Set kill switch active
        self.global_kill_switch_active = True
        
        # Log the event
        logger.error(f"GLOBAL EMERGENCY STOP: {reason}")
        
        # Make notification
        if self.notification_callback:
            self.notification_callback(
                "critical",
                f"ALL TRADING PAUSED: {reason}",
                data
            )
        
        # Call global pause callback if registered
        if self.global_pause_callback:
            return self.global_pause_callback(reason, data)
            
        return True
