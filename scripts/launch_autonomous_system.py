#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BensBot Autonomous Trading System Launcher

This script launches the fully autonomous BensBot trading system with all components:
- Daily Trading Loop (premarket to postmarket orchestration)
- Watchdog (system monitoring and self-healing)
- Feedback Loop (performance-based learning)
- Performance Database (trade history and strategy stats)

Run this script to start the completely autonomous trading system.
"""

import os
import sys
import logging
import time
import signal
import threading
import subprocess
from datetime import datetime, timedelta
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("autonomous_system.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("autonomous_system")

# Make sure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import components
from trading_bot.core.main_loop import DailyTradingLoop
from trading_bot.core.watchdog import ServiceWatchdog, ServiceStatus
from trading_bot.core.feedback_loop import StrategyFeedbackLoop
from trading_bot.core.performance_db import connect_performance_db
from trading_bot.core.event_bus import get_global_event_bus
from trading_bot.core.decision_scoring import DecisionScorer
from trading_bot.core.strategy_intelligence_recorder import StrategyIntelligenceRecorder

class AutonomousSystem:
    """
    Main launcher for the fully autonomous trading system.
    Integrates all components and provides centralized control.
    """
    
    def __init__(self, config_path="./config/trading_config.json", telegram_alerts=True):
        """
        Initialize the autonomous system.
        
        Args:
            config_path: Path to configuration file
            telegram_alerts: Whether to enable Telegram alerts
        """
        self.config_path = config_path
        self.telegram_alerts = telegram_alerts
        self.running = False
        self.components = {}
        
        # Signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info(f"Autonomous System initialized with config: {config_path}")
        
    def start(self):
        """Start the autonomous trading system with all components."""
        if self.running:
            logger.warning("Autonomous system is already running")
            return
            
        logger.info("Starting Autonomous Trading System...")
        self.running = True
        
        try:
            # 1. Initialize the event bus (central communication)
            self.components['event_bus'] = get_global_event_bus()
            logger.info("Event bus initialized")
            
            # 2. Initialize the performance database
            self.components['performance_db'] = connect_performance_db()
            logger.info("Performance database initialized")
            
            # 3. Initialize decision scorer
            self.components['decision_scorer'] = DecisionScorer()
            logger.info("Decision scorer initialized")
            
            # 4. Initialize strategy intelligence recorder
            self.components['strategy_intelligence'] = StrategyIntelligenceRecorder()
            logger.info("Strategy intelligence recorder initialized")
            
            # 5. Initialize the feedback loop
            self.components['feedback_loop'] = StrategyFeedbackLoop(
                event_bus=self.components['event_bus'],
                decision_scorer=self.components['decision_scorer'],
                strategy_intelligence=self.components['strategy_intelligence']
            )
            logger.info("Feedback loop initialized")
            
            # 6. Initialize and start the daily trading loop
            self.components['trading_loop'] = DailyTradingLoop(config_path=self.config_path)
            self.components['trading_loop'].start()
            logger.info("Daily trading loop started")
            
            # 7. Initialize and start the watchdog
            self.setup_watchdog()
            logger.info("All autonomous system components started successfully")
            
            # 8. Send initialization confirmation
            if self.telegram_alerts:
                self.send_telegram_notification(
                    "🤖 AUTONOMOUS SYSTEM STARTED",
                    {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "components": list(self.components.keys()),
                        "config": self.config_path
                    }
                )
            
        except Exception as e:
            logger.error(f"Failed to start autonomous system: {str(e)}")
            self.running = False
            
    def setup_watchdog(self):
        """Set up the watchdog service to monitor all critical components."""
        # Create the watchdog
        watchdog = ServiceWatchdog()
        
        # Register event bus
        watchdog.register_service(
            name="event_bus",
            health_check=lambda: self.components['event_bus'] is not None,
            recovery_action=lambda: self.restart_component('event_bus')
        )
        
        # Register trading loop
        watchdog.register_service(
            name="trading_loop",
            health_check=lambda: self.components.get('trading_loop') is not None and 
                               hasattr(self.components['trading_loop'], 'running') and
                               self.components['trading_loop'].running,
            recovery_action=lambda: self.restart_component('trading_loop')
        )
        
        # Register performance database
        watchdog.register_service(
            name="performance_db",
            health_check=lambda: self.test_database_connection(),
            recovery_action=lambda: self.restart_component('performance_db')
        )
        
        # Register feedback loop
        watchdog.register_service(
            name="feedback_loop",
            health_check=lambda: self.components.get('feedback_loop') is not None,
            recovery_action=lambda: self.restart_component('feedback_loop')
        )
        
        # Start the watchdog
        watchdog.start()
        
        # Store the watchdog in components
        self.components['watchdog'] = watchdog
        logger.info("Watchdog service started")
    
    def test_database_connection(self):
        """Test the performance database connection."""
        try:
            if 'performance_db' not in self.components or not self.components['performance_db']:
                return False
                
            # Try to execute a simple query
            db = self.components['performance_db']
            return db.conn is not None and hasattr(db, 'cursor') and db.cursor is not None
        except Exception:
            return False
    
    def restart_component(self, component_name):
        """
        Restart a specific component.
        
        Args:
            component_name: Name of the component to restart
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Attempting to restart component: {component_name}")
        
        try:
            if component_name == 'event_bus':
                self.components['event_bus'] = get_global_event_bus(reset=True)
                return self.components['event_bus'] is not None
                
            elif component_name == 'performance_db':
                if self.components.get('performance_db'):
                    self.components['performance_db'].close()
                self.components['performance_db'] = connect_performance_db()
                return self.test_database_connection()
                
            elif component_name == 'trading_loop':
                if self.components.get('trading_loop'):
                    self.components['trading_loop'].stop()
                self.components['trading_loop'] = DailyTradingLoop(config_path=self.config_path)
                self.components['trading_loop'].start()
                return (self.components['trading_loop'] is not None and 
                        self.components['trading_loop'].running)
                
            elif component_name == 'feedback_loop':
                self.components['feedback_loop'] = StrategyFeedbackLoop(
                    event_bus=self.components['event_bus'],
                    decision_scorer=self.components['decision_scorer'],
                    strategy_intelligence=self.components['strategy_intelligence']
                )
                return self.components['feedback_loop'] is not None
                
            else:
                logger.warning(f"Unknown component: {component_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to restart {component_name}: {str(e)}")
            return False
    
    def send_telegram_notification(self, title, data):
        """
        Send a notification to Telegram.
        
        Args:
            title: Title of the notification
            data: Dictionary of data to include
        """
        try:
            # Format message
            message = f"📊 *{title}*\n\n"
            
            for key, value in data.items():
                formatted_key = key.replace("_", " ").title()
                
                if isinstance(value, list):
                    message += f"*{formatted_key}*:\n"
                    for item in value:
                        message += f"  • {item}\n"
                else:
                    message += f"*{formatted_key}*: {value}\n"
            
            # Normally we'd use the Telegram API here, but for now just log
            logger.info(f"Telegram notification: {title} - {data}")
            
            # Call the actual Telegram client in a production system
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {str(e)}")
            return False
    
    def stop(self):
        """Stop all components and shutdown the autonomous system."""
        if not self.running:
            logger.warning("Autonomous system is not running")
            return
            
        logger.info("Stopping Autonomous Trading System...")
        
        try:
            # Stop components in reverse order
            if 'watchdog' in self.components:
                self.components['watchdog'].stop()
                logger.info("Watchdog stopped")
            
            if 'trading_loop' in self.components:
                self.components['trading_loop'].stop()
                logger.info("Trading loop stopped")
            
            if 'performance_db' in self.components:
                self.components['performance_db'].close()
                logger.info("Performance database connection closed")
            
            # Send shutdown notification
            if self.telegram_alerts:
                self.send_telegram_notification(
                    "🛑 AUTONOMOUS SYSTEM STOPPED",
                    {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "reason": "Manual shutdown"
                    }
                )
            
            self.running = False
            logger.info("Autonomous system stopped")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
    
    def signal_handler(self, sig, frame):
        """Handle termination signals for clean shutdown."""
        logger.info(f"Received signal {sig}, shutting down...")
        self.stop()
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BensBot Autonomous Trading System")
    parser.add_argument("--config", "-c", type=str, default="./config/trading_config.json",
                      help="Path to configuration file")
    parser.add_argument("--no-telegram", action="store_true",
                      help="Disable Telegram notifications")
    args = parser.parse_args()
    
    # Start the autonomous system
    system = AutonomousSystem(
        config_path=args.config,
        telegram_alerts=not args.no_telegram
    )
    system.start()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        system.stop()
