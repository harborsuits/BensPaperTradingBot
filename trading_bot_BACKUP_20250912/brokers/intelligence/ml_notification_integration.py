#!/usr/bin/env python3
"""
Integration between Machine Learning Prediction and Notification System

This module connects the broker performance ML predictions with the
notification system to generate alerts for anomalies and predicted failures.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import threading
import time

from trading_bot.brokers.intelligence.ml_prediction import BrokerPerformancePredictor
from trading_bot.brokers.intelligence.notification_system import BrokerIntelligenceNotifier
from trading_bot.brokers.intelligence.historical_tracker import BrokerPerformanceTracker

logger = logging.getLogger(__name__)


class MLAlertingSystem:
    """
    System for generating alerts based on ML predictions
    
    Monitors broker performance predictions and sends notifications
    when anomalies or failures are predicted.
    """
    
    def __init__(
        self,
        ml_predictor: BrokerPerformancePredictor,
        notifier: BrokerIntelligenceNotifier,
        check_interval: int = 3600,  # Check hourly by default
        anomaly_threshold: float = 0.15,  # Alert when 15% of recent data points are anomalies
        failure_threshold: float = 0.5,  # Alert when failure probability exceeds 50%
        alert_cooldown: int = 7200  # Seconds before sending another alert for the same broker
    ):
        """
        Initialize the ML alerting system
        
        Args:
            ml_predictor: ML predictor for broker performance
            notifier: Notifier for broker intelligence alerts
            check_interval: Interval in seconds between checks
            anomaly_threshold: Threshold for anomaly percentage to trigger alert
            failure_threshold: Threshold for failure probability to trigger alert
            alert_cooldown: Seconds before sending another alert for the same broker
        """
        self.predictor = ml_predictor
        self.notifier = notifier
        self.check_interval = check_interval
        self.anomaly_threshold = anomaly_threshold
        self.failure_threshold = failure_threshold
        self.alert_cooldown = alert_cooldown
        
        # Last alert times by broker
        self.last_alerts = {}
        
        # Flag for running background thread
        self.running = False
        self.thread = None
    
    def _should_send_alert(self, broker_id: str, alert_type: str) -> bool:
        """
        Check if we should send an alert based on cooldown
        
        Args:
            broker_id: Broker ID
            alert_type: Type of alert ('anomaly' or 'failure')
            
        Returns:
            True if alert should be sent
        """
        now = datetime.now()
        key = f"{broker_id}_{alert_type}"
        
        if key not in self.last_alerts:
            return True
        
        last_time = self.last_alerts[key]
        elapsed = (now - last_time).total_seconds()
        
        return elapsed >= self.alert_cooldown
    
    def _update_alert_time(self, broker_id: str, alert_type: str):
        """
        Update the last alert time for a broker
        
        Args:
            broker_id: Broker ID
            alert_type: Type of alert ('anomaly' or 'failure')
        """
        self.last_alerts[f"{broker_id}_{alert_type}"] = datetime.now()
    
    def check_broker(self, broker_id: str) -> List[Dict[str, Any]]:
        """
        Check a broker for anomalies and failure predictions
        
        Args:
            broker_id: Broker ID
            
        Returns:
            List of alerts that were sent
        """
        sent_alerts = []
        
        # Check for anomalies
        if broker_id in self.predictor.anomaly_models:
            _, anomaly_pct = self.predictor.detect_anomalies(broker_id)
            
            if anomaly_pct > self.anomaly_threshold and self._should_send_alert(broker_id, 'anomaly'):
                # Calculate severity based on threshold
                if anomaly_pct > self.anomaly_threshold * 2:
                    severity = 'critical'
                else:
                    severity = 'warning'
                
                # Create alert message
                alert_data = {
                    'broker_id': broker_id,
                    'alert_type': 'anomaly_detection',
                    'anomaly_percentage': round(anomaly_pct * 100, 1),
                    'severity': severity,
                    'timestamp': datetime.now().isoformat(),
                    'description': f"Detected unusual performance patterns for broker {broker_id}. {round(anomaly_pct * 100, 1)}% of recent data points are anomalous."
                }
                
                # Set priority based on severity
                priority = 'high' if severity == 'critical' else 'medium'
                
                # Send notification
                self.notifier.send_notification(
                    title=f"BROKER ANOMALY ALERT: {broker_id}",
                    message=f"Unusual performance detected for {broker_id}: {round(anomaly_pct * 100, 1)}% anomalous data points",
                    details=json.dumps(alert_data, indent=2),
                    priority=priority
                )
                
                # Update alert time
                self._update_alert_time(broker_id, 'anomaly')
                sent_alerts.append(alert_data)
                
                logger.info(f"Sent anomaly alert for broker {broker_id} with {round(anomaly_pct * 100, 1)}% anomalies")
        
        # Check for failure predictions
        if broker_id in self.predictor.failure_models:
            _, failure_prob = self.predictor.predict_failure_probability(broker_id)
            
            if failure_prob > self.failure_threshold and self._should_send_alert(broker_id, 'failure'):
                # Calculate severity based on threshold
                if failure_prob > 0.8:
                    severity = 'critical'
                else:
                    severity = 'warning'
                
                # Get prediction window in hours
                pred_window = self.predictor.prediction_window
                
                # Create alert message
                alert_data = {
                    'broker_id': broker_id,
                    'alert_type': 'failure_prediction',
                    'failure_probability': round(failure_prob * 100, 1),
                    'prediction_window_hours': pred_window,
                    'severity': severity,
                    'timestamp': datetime.now().isoformat(),
                    'description': f"Predicted potential failure for broker {broker_id} within the next {pred_window} hours. Failure probability: {round(failure_prob * 100, 1)}%"
                }
                
                # Set priority based on severity
                priority = 'high' if severity == 'critical' else 'medium'
                
                # Send notification
                self.notifier.send_notification(
                    title=f"BROKER FAILURE PREDICTION: {broker_id}",
                    message=f"Potential failure predicted for {broker_id} in the next {pred_window} hours. Probability: {round(failure_prob * 100, 1)}%",
                    details=json.dumps(alert_data, indent=2),
                    priority=priority
                )
                
                # Update alert time
                self._update_alert_time(broker_id, 'failure')
                sent_alerts.append(alert_data)
                
                logger.info(f"Sent failure prediction alert for broker {broker_id} with {round(failure_prob * 100, 1)}% probability")
        
        return sent_alerts
    
    def check_all_brokers(self) -> List[Dict[str, Any]]:
        """
        Check all brokers with ML models for anomalies and failures
        
        Returns:
            List of alerts that were sent
        """
        all_alerts = []
        
        # Get all brokers with models
        broker_ids = list(set(list(self.predictor.anomaly_models.keys()) + list(self.predictor.failure_models.keys())))
        
        for broker_id in broker_ids:
            alerts = self.check_broker(broker_id)
            all_alerts.extend(alerts)
        
        return all_alerts
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                logger.debug("Checking brokers for ML-based alerts")
                self.check_all_brokers()
            except Exception as e:
                logger.error(f"Error in ML alerting system monitoring loop: {str(e)}")
            
            # Sleep until next check
            time.sleep(self.check_interval)
    
    def start_monitoring(self):
        """Start the background monitoring thread"""
        if self.running:
            logger.warning("ML alerting system is already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        logger.info(f"ML alerting system started with {self.check_interval} second check interval")
    
    def stop_monitoring(self):
        """Stop the background monitoring thread"""
        if not self.running:
            logger.warning("ML alerting system is not running")
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)
        logger.info("ML alerting system stopped")


def setup_ml_alerting_system(
    tracker: BrokerPerformanceTracker,
    notifier: BrokerIntelligenceNotifier,
    model_dir: str = 'data/broker_ml_models',
    check_interval: int = 3600,
    anomaly_threshold: float = 0.15,
    failure_threshold: float = 0.5,
    alert_cooldown: int = 7200,
    start_monitoring: bool = True
) -> MLAlertingSystem:
    """
    Set up the ML alerting system
    
    Args:
        tracker: Broker performance tracker
        notifier: Broker intelligence notifier
        model_dir: Directory for ML models
        check_interval: Interval in seconds between checks
        anomaly_threshold: Threshold for anomaly percentage to trigger alert
        failure_threshold: Threshold for failure probability to trigger alert
        alert_cooldown: Seconds before sending another alert for the same broker
        start_monitoring: Whether to start monitoring immediately
        
    Returns:
        Initialized MLAlertingSystem
    """
    # Create ML predictor
    predictor = BrokerPerformancePredictor(
        performance_tracker=tracker,
        model_dir=model_dir
    )
    
    # Create alerting system
    alerting_system = MLAlertingSystem(
        ml_predictor=predictor,
        notifier=notifier,
        check_interval=check_interval,
        anomaly_threshold=anomaly_threshold,
        failure_threshold=failure_threshold,
        alert_cooldown=alert_cooldown
    )
    
    # Start monitoring if requested
    if start_monitoring:
        alerting_system.start_monitoring()
    
    return alerting_system
