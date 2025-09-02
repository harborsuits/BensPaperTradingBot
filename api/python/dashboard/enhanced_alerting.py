"""
Enhanced Alerting System for Adaptive Trading

This module extends the Telegram alerting system with:
1. Strategy-specific alerts based on regime detector signals
2. Drawdown threshold monitoring and alerts
3. Regime change detection with confidence metrics
4. Weight evolution notifications

It integrates with both the dashboard and the Telegram alert system.
"""

import logging
import os
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# Import trading system components
from trading_bot.alerts.telegram_alerts import (
    send_risk_alert, send_strategy_rotation_alert, 
    send_trade_alert, send_system_alert,
    get_telegram_alert_manager
)
from trading_bot.risk.adaptive_strategy_controller import AdaptiveStrategyController
from trading_bot.analytics.market_regime_detector import MarketRegimeDetector

logger = logging.getLogger(__name__)

class EnhancedAlertMonitor:
    """
    Enhanced alert monitoring system that integrates with the adaptive strategy controller
    to provide real-time alerting for significant trading events.
    """
    
    def __init__(self, controller: Optional[AdaptiveStrategyController] = None):
        """Initialize the enhanced alert monitor"""
        self.controller = controller
        self.telegram = get_telegram_alert_manager()
        self.enabled = True
        self.monitor_thread = None
        self.stop_event = threading.Event()
        
        # Alert thresholds
        self.thresholds = {
            'strategy_drawdown': 0.05,   # 5% strategy drawdown
            'global_drawdown': 0.03,     # 3% global drawdown
            'regime_confidence': 0.75,   # 75% confidence for regime change alert
            'weight_change': 0.1,        # 10% weight change threshold
            'risk_exposure': 0.80        # 80% max risk exposure threshold
        }
        
        # State tracking
        self.last_regime = None
        self.last_weights = {}
        self.max_equity = 0.0
        self.strategy_max_values = {}
        
        # Create output directory
        os.makedirs('./logs/alerts', exist_ok=True)
    
    def set_controller(self, controller: AdaptiveStrategyController):
        """Set the adaptive strategy controller reference"""
        self.controller = controller
        logger.info(f"EnhancedAlertMonitor now monitoring AdaptiveStrategyController")
    
    def set_thresholds(self, thresholds: Dict[str, float]):
        """Update alert thresholds"""
        self.thresholds.update(thresholds)
        logger.info(f"Updated alert thresholds: {self.thresholds}")
    
    def start_monitoring(self, interval_seconds: float = 5.0):
        """
        Start the alert monitoring thread
        
        Args:
            interval_seconds: How often to check for alert conditions
        """
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning("Alert monitoring already running")
            return False
        
        if not self.controller:
            logger.error("No controller set, cannot start monitoring")
            return False
        
        # Reset stop event
        self.stop_event.clear()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info(f"Started enhanced alert monitoring (interval: {interval_seconds}s)")
        
        # Send startup alert
        send_system_alert(
            component="Enhanced Alerts",
            status="online",
            message="Enhanced alerting system activated",
            severity="info"
        )
        
        return True
    
    def stop_monitoring(self):
        """Stop the alert monitoring thread"""
        if not self.monitor_thread or not self.monitor_thread.is_alive():
            return
        
        # Signal thread to stop
        self.stop_event.set()
        
        # Wait for thread to exit (with timeout)
        self.monitor_thread.join(timeout=5.0)
        
        logger.info("Stopped enhanced alert monitoring")
        
        # Send shutdown alert
        send_system_alert(
            component="Enhanced Alerts",
            status="offline",
            message="Enhanced alerting system deactivated",
            severity="info"
        )
    
    def check_drawdown_thresholds(self) -> List[Dict[str, Any]]:
        """
        Check for drawdown threshold violations
        
        Returns:
            List of drawdown alerts that were triggered
        """
        if not self.controller:
            return []
        
        alerts = []
        
        # Get performance tracker from controller
        perf_tracker = getattr(self.controller, 'performance_tracker', None)
        if not perf_tracker:
            return []
        
        # Check global portfolio drawdown
        metrics = perf_tracker.get_metrics()
        current_equity = metrics.get('equity', 0)
        
        # Update max equity if needed
        if current_equity > self.max_equity:
            self.max_equity = current_equity
        
        # Calculate current drawdown
        if self.max_equity > 0:
            current_drawdown = (self.max_equity - current_equity) / self.max_equity
            global_threshold = self.thresholds['global_drawdown']
            
            # Alert if drawdown exceeds threshold
            if current_drawdown > global_threshold:
                alert = {
                    'type': 'global_drawdown',
                    'drawdown': current_drawdown,
                    'threshold': global_threshold,
                    'max_equity': self.max_equity,
                    'current_equity': current_equity,
                    'time': datetime.now()
                }
                alerts.append(alert)
                
                # Send Telegram alert
                send_risk_alert(
                    risk_type="global_drawdown",
                    risk_level="high" if current_drawdown > global_threshold * 1.5 else "medium",
                    message=f"Global drawdown threshold exceeded: {current_drawdown:.2%}",
                    details={
                        "Current Drawdown": f"{current_drawdown:.2%}",
                        "Threshold": f"{global_threshold:.2%}",
                        "Max Equity": f"${self.max_equity:,.2f}",
                        "Current Equity": f"${current_equity:,.2f}"
                    }
                )
        
        # Check individual strategy drawdowns
        strategy_perf = perf_tracker.get_strategy_metrics()
        for strategy_id, strategy_metrics in strategy_perf.items():
            current_value = strategy_metrics.get('equity', 0)
            
            # Initialize max value if needed
            if strategy_id not in self.strategy_max_values:
                self.strategy_max_values[strategy_id] = current_value
            elif current_value > self.strategy_max_values[strategy_id]:
                self.strategy_max_values[strategy_id] = current_value
            
            # Calculate strategy drawdown
            max_value = self.strategy_max_values.get(strategy_id, 0)
            if max_value > 0:
                strategy_drawdown = (max_value - current_value) / max_value
                strategy_threshold = self.thresholds['strategy_drawdown']
                
                # Alert if strategy drawdown exceeds threshold
                if strategy_drawdown > strategy_threshold:
                    alert = {
                        'type': 'strategy_drawdown',
                        'strategy_id': strategy_id,
                        'drawdown': strategy_drawdown,
                        'threshold': strategy_threshold,
                        'max_value': max_value,
                        'current_value': current_value,
                        'time': datetime.now()
                    }
                    alerts.append(alert)
                    
                    # Send Telegram alert
                    send_risk_alert(
                        risk_type="strategy_drawdown",
                        risk_level="high" if strategy_drawdown > strategy_threshold * 1.5 else "medium",
                        message=f"Strategy '{strategy_id}' drawdown threshold exceeded: {strategy_drawdown:.2%}",
                        details={
                            "Strategy": strategy_id,
                            "Current Drawdown": f"{strategy_drawdown:.2%}",
                            "Threshold": f"{strategy_threshold:.2%}",
                            "Max Value": f"${max_value:,.2f}",
                            "Current Value": f"${current_value:,.2f}"
                        }
                    )
        
        return alerts
    
    def check_regime_changes(self) -> List[Dict[str, Any]]:
        """
        Check for market regime changes with high confidence
        
        Returns:
            List of regime change alerts that were triggered
        """
        if not self.controller:
            return []
        
        alerts = []
        
        # Get regime detector from controller
        regime_detector = getattr(self.controller, 'market_regime_detector', None)
        if not regime_detector:
            return []
        
        # Get current regime information
        current_regime = regime_detector.get_current_regime()
        if not current_regime:
            return []
        
        regime_type = current_regime.get('regime_type')
        confidence = current_regime.get('confidence', 0)
        confidence_threshold = self.thresholds['regime_confidence']
        
        # Check if regime has changed with high confidence
        if (regime_type and confidence >= confidence_threshold and
            regime_type != self.last_regime):
            
            alert = {
                'type': 'regime_change',
                'previous_regime': self.last_regime,
                'new_regime': regime_type,
                'confidence': confidence,
                'threshold': confidence_threshold,
                'indicators': current_regime.get('indicators', {}),
                'time': datetime.now()
            }
            alerts.append(alert)
            
            # Send Telegram alert
            send_system_alert(
                component="Market Regime",
                status="warning",
                message=f"New market regime detected: {regime_type} (confidence: {confidence:.2%})",
                severity="high" if confidence > 0.9 else "medium"
            )
            
            # Update last regime
            self.last_regime = regime_type
        
        return alerts
    
    def check_weight_changes(self) -> List[Dict[str, Any]]:
        """
        Check for significant changes in strategy weights
        
        Returns:
            List of weight change alerts that were triggered
        """
        if not self.controller:
            return []
        
        alerts = []
        
        # Get allocator from controller
        allocator = getattr(self.controller, 'snowball_allocator', None)
        if not allocator:
            return []
        
        # Get current allocations
        current_weights = allocator.get_current_allocation()
        if not current_weights:
            return []
        
        weight_change_threshold = self.thresholds['weight_change']
        significant_changes = {}
        
        # Check for significant weight changes
        for strategy_id, weight in current_weights.items():
            previous_weight = self.last_weights.get(strategy_id, 0)
            weight_change = abs(weight - previous_weight)
            
            if weight_change >= weight_change_threshold:
                significant_changes[strategy_id] = {
                    'previous': previous_weight,
                    'current': weight,
                    'change': weight_change,
                    'direction': 'increase' if weight > previous_weight else 'decrease'
                }
        
        # Create alert if any significant changes
        if significant_changes:
            alert = {
                'type': 'weight_change',
                'changes': significant_changes,
                'threshold': weight_change_threshold,
                'time': datetime.now()
            }
            alerts.append(alert)
            
            # Send Telegram alert with changes
            message = "Significant strategy weight changes detected:\n"
            details = {}
            
            for strategy_id, change in significant_changes.items():
                direction = '📈' if change['direction'] == 'increase' else '📉'
                message += f"\n{strategy_id}: {change['previous']:.2%} → {change['current']:.2%} {direction}"
                details[strategy_id] = f"{change['previous']:.2%} → {change['current']:.2%} ({change['direction']})"
            
            send_strategy_rotation_alert(
                trigger="Weight Evolution",
                old_strategies=[],  # Not a full rotation
                new_strategies=[],  # Not a full rotation
                reason=message
            )
            
            # Update last weights
            self.last_weights = current_weights.copy()
        
        return alerts
    
    def _monitoring_loop(self, interval_seconds: float):
        """Main monitoring loop that periodically checks alert conditions"""
        while not self.stop_event.is_set():
            try:
                if not self.controller:
                    time.sleep(interval_seconds)
                    continue
                
                all_alerts = []
                
                # Check drawdown thresholds
                drawdown_alerts = self.check_drawdown_thresholds()
                all_alerts.extend(drawdown_alerts)
                
                # Check regime changes
                regime_alerts = self.check_regime_changes()
                all_alerts.extend(regime_alerts)
                
                # Check weight changes
                weight_alerts = self.check_weight_changes()
                all_alerts.extend(weight_alerts)
                
                # Log alerts
                if all_alerts:
                    self._log_alerts(all_alerts)
                
            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {str(e)}")
            
            # Sleep until next check
            time.sleep(interval_seconds)
    
    def _log_alerts(self, alerts: List[Dict[str, Any]]):
        """Log alerts to file"""
        try:
            # Ensure log directory exists
            log_dir = './logs/alerts'
            os.makedirs(log_dir, exist_ok=True)
            
            # Create log file name with today's date
            today = datetime.now().strftime('%Y%m%d')
            log_file = os.path.join(log_dir, f'alerts_{today}.json')
            
            # Load existing alerts if the file exists
            existing_alerts = []
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        existing_alerts = json.load(f)
                except:
                    pass
            
            # Convert datetime objects to strings for JSON serialization
            serializable_alerts = []
            for alert in alerts:
                serializable_alert = alert.copy()
                if 'time' in serializable_alert and isinstance(serializable_alert['time'], datetime):
                    serializable_alert['time'] = serializable_alert['time'].isoformat()
                serializable_alerts.append(serializable_alert)
            
            # Append new alerts
            all_alerts = existing_alerts + serializable_alerts
            
            # Save to file
            with open(log_file, 'w') as f:
                json.dump(all_alerts, f, indent=2)
            
            logger.info(f"Logged {len(alerts)} alerts to {log_file}")
            
        except Exception as e:
            logger.error(f"Error logging alerts: {str(e)}")

# Singleton instance
_alert_monitor_instance = None

def get_alert_monitor() -> EnhancedAlertMonitor:
    """Get the global alert monitor instance"""
    global _alert_monitor_instance
    if _alert_monitor_instance is None:
        _alert_monitor_instance = EnhancedAlertMonitor()
    return _alert_monitor_instance

# Usage example (if run as script)
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create monitor instance
    monitor = EnhancedAlertMonitor()
    
    # Set custom thresholds
    monitor.set_thresholds({
        'strategy_drawdown': 0.03,  # 3% strategy drawdown
        'global_drawdown': 0.02,    # 2% global drawdown
    })
    
    print("Alert monitor configured. Would need a controller to start monitoring.")
    print(f"Current thresholds: {monitor.thresholds}")
