#!/usr/bin/env python3
"""
Setup Script for Complete Broker Intelligence System

This script sets up the entire broker intelligence system:
1. Historical performance tracking
2. Machine learning predictions
3. Alerting and notification integration

Run this script to initialize and start the complete broker intelligence system.
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from trading_bot.event_system.event_bus import EventBus
from trading_bot.brokers.metrics.manager import BrokerMetricsManager
from trading_bot.brokers.intelligence.broker_advisor import BrokerAdvisor
from trading_bot.brokers.intelligence.notification_system import BrokerIntelligenceNotifier
from trading_bot.brokers.intelligence.historical_tracker import BrokerPerformanceTracker
from trading_bot.brokers.intelligence.historical_tracker_continued import BrokerPerformanceAnalyzer
from trading_bot.brokers.intelligence.ml_prediction import BrokerPerformancePredictor
from trading_bot.brokers.intelligence.ml_notification_integration import setup_ml_alerting_system


def setup_broker_intelligence_system(
    # Historical tracking settings
    storage_type='sqlite',
    storage_path='data/broker_performance',
    sampling_interval=300,
    retention_days=90,
    
    # ML prediction settings
    model_dir='data/broker_ml_models',
    min_training_days=7,
    prediction_window=24,
    
    # Notification settings
    notification_config_path='config/broker_intelligence_notifications.json',
    
    # ML alerting settings
    check_interval=3600,
    anomaly_threshold=0.15,
    failure_threshold=0.5,
    alert_cooldown=7200,
    
    # Operation flags
    start_tracking=True,
    start_alerting=True
):
    """
    Set up and initialize the complete broker intelligence system
    
    Args:
        storage_type: Storage type for historical data ('sqlite' or 'csv')
        storage_path: Path to store historical data
        sampling_interval: Interval in seconds between recordings
        retention_days: Days to retain historical data
        
        model_dir: Directory to store ML models
        min_training_days: Minimum days of data required for training
        prediction_window: Hours to predict ahead
        
        notification_config_path: Path to notification configuration file
        
        check_interval: Interval in seconds between ML checks
        anomaly_threshold: Threshold for anomaly percentage to trigger alert
        failure_threshold: Threshold for failure probability to trigger alert
        alert_cooldown: Seconds before sending another alert for the same broker
        
        start_tracking: Whether to start historical tracking
        start_alerting: Whether to start ML alerting
    
    Returns:
        Dict with initialized system components
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("broker_intelligence_setup")
    
    # Create event bus
    event_bus = EventBus()
    logger.info("Event Bus initialized")
    
    # Create metrics manager
    metrics_manager = BrokerMetricsManager(event_bus)
    logger.info("Broker Metrics Manager initialized")
    
    # Create performance tracker
    logger.info(f"Initializing Broker Performance Tracker with {storage_type} storage at {storage_path}")
    
    # Create storage directory if it doesn't exist
    os.makedirs(storage_path, exist_ok=True)
    
    tracker = BrokerPerformanceTracker(
        event_bus=event_bus,
        storage_type=storage_type,
        storage_path=storage_path,
        sampling_interval=sampling_interval,
        retention_days=retention_days
    )
    
    # Start recording if requested
    if start_tracking:
        tracker.start_recording()
        logger.info(f"Historical performance tracking started with {sampling_interval}s interval")
    
    # Create broker advisor
    advisor = BrokerAdvisor(event_bus)
    logger.info("Broker Advisor initialized")
    
    # Create notifier
    notifier = BrokerIntelligenceNotifier(
        config_file=notification_config_path,
        event_bus=event_bus
    )
    logger.info("Broker Intelligence Notifier initialized")
    
    # Create ML predictor
    predictor = BrokerPerformancePredictor(
        performance_tracker=tracker,
        model_dir=model_dir,
        min_training_days=min_training_days,
        prediction_window=prediction_window
    )
    logger.info(f"Broker Performance Predictor initialized with {prediction_window}h window")
    
    # Create ML alerting system
    alerting_system = setup_ml_alerting_system(
        tracker=tracker,
        notifier=notifier,
        model_dir=model_dir,
        check_interval=check_interval,
        anomaly_threshold=anomaly_threshold,
        failure_threshold=failure_threshold,
        alert_cooldown=alert_cooldown,
        start_monitoring=start_alerting
    )
    
    if start_alerting:
        logger.info(f"ML alerting system started with {check_interval}s check interval")
    
    # Return all components
    return {
        'event_bus': event_bus,
        'metrics_manager': metrics_manager,
        'performance_tracker': tracker,
        'broker_advisor': advisor,
        'notifier': notifier,
        'predictor': predictor,
        'alerting_system': alerting_system
    }


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Set up complete broker intelligence system')
    
    # Historical tracking options
    tracking_group = parser.add_argument_group('Historical Tracking')
    tracking_group.add_argument(
        '--storage-type',
        choices=['sqlite', 'csv'],
        default='sqlite',
        help='Storage type for historical data'
    )
    tracking_group.add_argument(
        '--storage-path',
        default='data/broker_performance',
        help='Path to storage location'
    )
    tracking_group.add_argument(
        '--sampling-interval',
        type=int,
        default=300,
        help='Interval in seconds between recordings'
    )
    tracking_group.add_argument(
        '--retention-days',
        type=int,
        default=90,
        help='Days to retain historical data'
    )
    
    # ML prediction options
    ml_group = parser.add_argument_group('ML Prediction')
    ml_group.add_argument(
        '--model-dir',
        default='data/broker_ml_models',
        help='Directory to store ML models'
    )
    ml_group.add_argument(
        '--min-training-days',
        type=int,
        default=7,
        help='Minimum days of data required for training'
    )
    ml_group.add_argument(
        '--prediction-window',
        type=int,
        default=24,
        help='Hours to predict ahead'
    )
    
    # Notification options
    notification_group = parser.add_argument_group('Notification')
    notification_group.add_argument(
        '--notification-config',
        default='config/broker_intelligence_notifications.json',
        help='Path to notification configuration file'
    )
    
    # ML alerting options
    alerting_group = parser.add_argument_group('ML Alerting')
    alerting_group.add_argument(
        '--check-interval',
        type=int,
        default=3600,
        help='Interval in seconds between ML checks'
    )
    alerting_group.add_argument(
        '--anomaly-threshold',
        type=float,
        default=0.15,
        help='Threshold for anomaly percentage to trigger alert'
    )
    alerting_group.add_argument(
        '--failure-threshold',
        type=float,
        default=0.5,
        help='Threshold for failure probability to trigger alert'
    )
    alerting_group.add_argument(
        '--alert-cooldown',
        type=int,
        default=7200,
        help='Seconds before sending another alert for the same broker'
    )
    
    # Operation flags
    operation_group = parser.add_argument_group('Operation')
    operation_group.add_argument(
        '--no-tracking',
        action='store_true',
        help='Do not start historical tracking'
    )
    operation_group.add_argument(
        '--no-alerting',
        action='store_true',
        help='Do not start ML alerting'
    )
    operation_group.add_argument(
        '--show-config',
        action='store_true',
        help='Print loaded configuration and exit'
    )
    
    args = parser.parse_args()
    
    # If showing config, just print and exit
    if args.show_config:
        config = {
            'storage_type': args.storage_type,
            'storage_path': args.storage_path,
            'sampling_interval': args.sampling_interval,
            'retention_days': args.retention_days,
            'model_dir': args.model_dir,
            'min_training_days': args.min_training_days,
            'prediction_window': args.prediction_window,
            'notification_config_path': args.notification_config,
            'check_interval': args.check_interval,
            'anomaly_threshold': args.anomaly_threshold,
            'failure_threshold': args.failure_threshold,
            'alert_cooldown': args.alert_cooldown,
            'start_tracking': not args.no_tracking,
            'start_alerting': not args.no_alerting
        }
        print(json.dumps(config, indent=2))
        return
    
    # Set up system
    system = setup_broker_intelligence_system(
        storage_type=args.storage_type,
        storage_path=args.storage_path,
        sampling_interval=args.sampling_interval,
        retention_days=args.retention_days,
        model_dir=args.model_dir,
        min_training_days=args.min_training_days,
        prediction_window=args.prediction_window,
        notification_config_path=args.notification_config,
        check_interval=args.check_interval,
        anomaly_threshold=args.anomaly_threshold,
        failure_threshold=args.failure_threshold,
        alert_cooldown=args.alert_cooldown,
        start_tracking=not args.no_tracking,
        start_alerting=not args.no_alerting
    )
    
    print(f"""
╔══════════════════════════════════════════════════════╗
║                                                      ║
║           Broker Intelligence System Active          ║
║                                                      ║
╚══════════════════════════════════════════════════════╝

  Historical Tracking: {'Active' if not args.no_tracking else 'Inactive'}
  ML Prediction & Alerting: {'Active' if not args.no_alerting else 'Inactive'}
  
  Storage: {args.storage_type.upper()} at {args.storage_path}
  Sampling Interval: {args.sampling_interval} seconds
  ML Check Interval: {args.check_interval} seconds
  
  Press Ctrl+C to exit
""")
    
    # Keep the process running
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down Broker Intelligence System...")
        
        # Stop tracking and alerting
        if not args.no_tracking:
            system['performance_tracker'].stop_recording()
            system['performance_tracker'].close()
        
        if not args.no_alerting:
            system['alerting_system'].stop_monitoring()
        
        print("Shutdown complete")


if __name__ == "__main__":
    main()
