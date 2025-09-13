#!/usr/bin/env python3
"""
Setup Script for Broker Performance Historical Tracker

Initializes the broker performance historical tracking system,
sets up database storage, and starts the periodic recording process.

Run this script to set up the historical tracking system before
using the analytics and visualization components.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from trading_bot.event_system.event_bus import EventBus
from trading_bot.brokers.metrics.manager import BrokerMetricsManager
from trading_bot.brokers.intelligence.broker_advisor import BrokerAdvisor
from trading_bot.brokers.intelligence.historical_tracker import BrokerPerformanceTracker
from trading_bot.brokers.intelligence.historical_tracker_continued import BrokerPerformanceAnalyzer


def setup_broker_performance_tracker(
    storage_type='sqlite',
    storage_path='data/broker_performance',
    sampling_interval=300,
    retention_days=90,
    start_recording=True
):
    """
    Set up and initialize the broker performance tracking system
    
    Args:
        storage_type: Storage type ('sqlite' or 'csv')
        storage_path: Path to storage location
        sampling_interval: Interval in seconds between recordings
        retention_days: Days to retain historical data
        start_recording: Whether to start recording immediately
        
    Returns:
        Initialized BrokerPerformanceTracker
    """
    # Create event bus
    event_bus = EventBus()
    
    # Create metrics manager
    metrics_manager = BrokerMetricsManager(event_bus)
    
    # Create performance tracker
    tracker = BrokerPerformanceTracker(
        event_bus=event_bus,
        storage_type=storage_type,
        storage_path=storage_path,
        sampling_interval=sampling_interval,
        retention_days=retention_days
    )
    
    # Start recording if requested
    if start_recording:
        tracker.start_recording()
    
    return tracker


def main():
    """Main entry point"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Set up broker performance tracking')
    
    parser.add_argument(
        '--storage-type',
        choices=['sqlite', 'csv'],
        default='sqlite',
        help='Storage type for historical data'
    )
    
    parser.add_argument(
        '--storage-path',
        default='data/broker_performance',
        help='Path to storage location'
    )
    
    parser.add_argument(
        '--sampling-interval',
        type=int,
        default=300,
        help='Interval in seconds between recordings'
    )
    
    parser.add_argument(
        '--retention-days',
        type=int,
        default=90,
        help='Days to retain historical data'
    )
    
    parser.add_argument(
        '--no-recording',
        action='store_true',
        help='Do not start recording immediately'
    )
    
    args = parser.parse_args()
    
    # Set up tracker
    tracker = setup_broker_performance_tracker(
        storage_type=args.storage_type,
        storage_path=args.storage_path,
        sampling_interval=args.sampling_interval,
        retention_days=args.retention_days,
        start_recording=not args.no_recording
    )
    
    logging.info(f"Broker performance tracker initialized with {args.storage_type} storage at {args.storage_path}")
    
    if not args.no_recording:
        logging.info(f"Recording started with {args.sampling_interval} second interval and {args.retention_days} day retention")
    
    # Keep the process running if recording
    if not args.no_recording:
        try:
            logging.info("Press Ctrl+C to stop recording and exit")
            # Keep the main thread alive
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Stopping recording and exiting")
            tracker.stop_recording()
            tracker.close()


if __name__ == "__main__":
    main()
