#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BensBot Daily Trading Routine Orchestrator

This module implements the daily trading routine, orchestrating the entire
workflow from premarket analysis to postmarket reporting and feedback.
"""

import logging
import time
import sys
import os
import json
import threading
import subprocess
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pytz
import signal

# Internal imports
from trading_bot.core.event_bus import get_global_event_bus, Event
from trading_bot.core.enhanced_strategy_manager_impl import EnhancedStrategyManager
from trading_bot.ai_scoring.regime_aware_strategy_prioritizer import RegimeAwareStrategyPrioritizer
from trading_bot.core.recovery_controller import RecoveryController
from trading_bot.core.state_manager import StateManager
from trading_bot.core.watchdog import ServiceWatchdog, check_api_connection
from trading_bot.monitoring.system_monitor import SystemMonitor
from trading_bot.utils.market_hours import is_market_open, get_next_market_open, get_next_market_close
from trading_bot.news.news_fetcher import NewsFetcher
from trading_bot.performance.performance_analyzer import PerformanceAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot_main_loop.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("trading_bot.main_loop")

class TradingPhase:
    """Trading phases throughout the day"""
    PREMARKET = "PREMARKET"
    MARKET_OPEN = "MARKET_OPEN"
    MARKET_ACTIVE = "MARKET_ACTIVE"
    MARKET_CLOSE = "MARKET_CLOSE"
    POSTMARKET = "POSTMARKET"
    OVERNIGHT = "OVERNIGHT"

class DailyTradingLoop:
    """
    Main orchestrator for the daily trading routine.
    Manages the entire lifecycle of trading from premarket to postmarket.
    """
    
    def __init__(self, config_path: str = "./config/trading_config.json"):
        """
        Initialize the daily trading loop orchestrator.
        
        Args:
            config_path: Path to the trading configuration file
        """
        self.config_path = config_path
        self.load_config()
        
        # State management
        self.current_phase = TradingPhase.OVERNIGHT
        self.running = False
        self.stop_event = threading.Event()
        self.main_thread = None
        self.last_phase_change = datetime.now()
        
        # Initialize core components
        self.event_bus = get_global_event_bus()
        self.strategy_prioritizer = None
        self.recovery_controller = None
        self.state_manager = None
        self.watchdog = None
        self.system_monitor = None
        self.current_strategies = []
        
        # Helper components
        self.news_fetcher = None 
        self.performance_analyzer = None
        
        # Performance database connection (will be initialized later)
        self.performance_db = None
        
        logger.info("Daily trading loop initialized with config: %s", config_path)
    
    def load_config(self):
        """Load configuration from the JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            # Extract timezone configuration or use default (Eastern Time for US markets)
            self.timezone = pytz.timezone(self.config.get('timezone', 'US/Eastern'))
            
            # Trading hours
            self.premarket_start_hour = self.config.get('premarket_start_hour', 8)  # 8:00 AM ET
            self.market_open_hour = self.config.get('market_open_hour', 9)  # 9:30 AM ET
            self.market_open_minute = self.config.get('market_open_minute', 30)
            self.market_close_hour = self.config.get('market_close_hour', 16)  # 4:00 PM ET
            self.market_close_minute = self.config.get('market_close_minute', 0)
            self.postmarket_end_hour = self.config.get('postmarket_end_hour', 18)  # 6:00 PM ET
            
            logger.info("Configuration loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            return False
            
    def start(self):
        """Start the daily trading loop in a separate thread."""
        if self.running:
            logger.warning("Daily trading loop is already running")
            return
            
        # Initialize components
        self.initialize_components()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Start the main loop in a separate thread
        self.running = True
        self.main_thread = threading.Thread(target=self.main_loop, name="DailyTradingLoop")
        self.main_thread.daemon = True
        self.main_thread.start()
        
        logger.info("Daily trading loop started")
        
    def initialize_components(self):
        """Initialize all trading components."""
        try:
            # Initialize state management
            self.state_manager = StateManager()
            
            # Initialize recovery controller
            self.recovery_controller = RecoveryController(self.state_manager)
            
            # Initialize system monitor
            self.system_monitor = SystemMonitor()
            
            # Initialize strategy prioritizer
            self.strategy_prioritizer = RegimeAwareStrategyPrioritizer()
            
            # Initialize news fetcher
            self.news_fetcher = NewsFetcher()
            
            # Initialize performance analyzer
            self.performance_analyzer = PerformanceAnalyzer()
            
            # Initialize watchdog for monitoring critical services
            self.setup_watchdog()
            
            # Initialize performance database
            self.initialize_performance_db()
            
            logger.info("All components initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            return False
    
    def setup_watchdog(self):
        """Initialize and configure the watchdog service."""
        self.watchdog = ServiceWatchdog()
        
        # Register critical services to monitor
        self.watchdog.register_service(
            name="event_bus",
            health_check=lambda: self.event_bus is not None,
            recovery_action=lambda: self.restart_component("event_bus")
        )
        
        self.watchdog.register_service(
            name="market_data_api",
            health_check=lambda: self.check_market_data_connection(),
            recovery_action=lambda: self.reconnect_market_data()
        )
        
        # Register additional critical services
        
        # Start the watchdog service
        self.watchdog.start()
        logger.info("Watchdog service started and monitoring critical components")
    
    def initialize_performance_db(self):
        """Initialize the performance database connection."""
        try:
            # For now, we'll just import and initialize the performance db
            # In a real implementation, set up proper connection
            from trading_bot.core.performance_db import connect_performance_db
            self.performance_db = connect_performance_db()
            logger.info("Performance database initialized")
            return True
        except ImportError:
            logger.warning("Performance database module not found, tracking may be limited")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize performance database: {str(e)}")
            return False
    
    def check_market_data_connection(self) -> bool:
        """Check if market data connection is working."""
        # This is a placeholder - replace with actual implementation
        # Example: Try to fetch a test quote from the market data provider
        try:
            # Dummy implementation
            return True
        except Exception:
            return False
    
    def reconnect_market_data(self) -> bool:
        """Attempt to reconnect to the market data provider."""
        # This is a placeholder - replace with actual implementation
        try:
            # Dummy implementation for reconnection
            return True
        except Exception:
            return False
    
    def restart_component(self, component_name: str) -> bool:
        """Restart a specific component."""
        logger.info(f"Attempting to restart component: {component_name}")
        try:
            if component_name == "event_bus":
                self.event_bus = get_global_event_bus(reset=True)
                return self.event_bus is not None
            # Add other component restart logic as needed
            return False
        except Exception as e:
            logger.error(f"Failed to restart {component_name}: {str(e)}")
            return False
    
    def main_loop(self):
        """Main trading loop that runs indefinitely."""
        while self.running and not self.stop_event.is_set():
            try:
                # Get current time in trading timezone
                now = datetime.now(self.timezone)
                
                # Determine the current trading phase
                next_phase = self.determine_trading_phase(now)
                
                # If trading phase has changed, execute phase-specific actions
                if next_phase != self.current_phase:
                    logger.info(f"Trading phase changed from {self.current_phase} to {next_phase}")
                    self.current_phase = next_phase
                    self.last_phase_change = now
                    
                    # Execute phase-specific actions
                    self.execute_phase_actions(next_phase)
                
                # Always execute regular checks regardless of phase
                self.execute_regular_checks()
                
                # Calculate time to next check based on the phase
                if self.current_phase in [TradingPhase.MARKET_ACTIVE, TradingPhase.MARKET_OPEN]:
                    # More frequent checks during active trading
                    check_interval = 60  # 1 minute
                elif self.current_phase in [TradingPhase.PREMARKET, TradingPhase.MARKET_CLOSE]:
                    # Medium frequency checks during transitions
                    check_interval = 300  # 5 minutes
                else:
                    # Longer intervals during overnight
                    check_interval = 900  # 15 minutes
                
                # Sleep until next check
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in main trading loop: {str(e)}")
                time.sleep(30)  # Sleep before retry to avoid tight error loops
    
    def determine_trading_phase(self, now: datetime) -> str:
        """
        Determine the current trading phase based on time.
        
        Args:
            now: Current datetime in the trading timezone
            
        Returns:
            Current trading phase as a string
        """
        # Extract time components
        hour = now.hour
        minute = now.minute
        weekday = now.weekday()  # 0-6, Monday is 0
        
        # Check if it's a weekend (Saturday=5, Sunday=6)
        if weekday >= 5:
            return TradingPhase.OVERNIGHT
        
        # Check if it's premarket
        if hour >= self.premarket_start_hour and (hour < self.market_open_hour or 
                                               (hour == self.market_open_hour and 
                                                minute < self.market_open_minute)):
            return TradingPhase.PREMARKET
            
        # Check if it's market open transition
        if hour == self.market_open_hour and minute >= self.market_open_minute and minute < self.market_open_minute + 30:
            return TradingPhase.MARKET_OPEN
            
        # Check if it's active market hours
        if ((hour == self.market_open_hour and minute >= self.market_open_minute + 30) or 
            (hour > self.market_open_hour and hour < self.market_close_hour) or
            (hour == self.market_close_hour and minute < self.market_close_minute - 30)):
            return TradingPhase.MARKET_ACTIVE
            
        # Check if it's market close transition
        if hour == self.market_close_hour and minute >= self.market_close_minute - 30 and minute < self.market_close_minute + 30:
            return TradingPhase.MARKET_CLOSE
            
        # Check if it's postmarket
        if (hour == self.market_close_hour and minute >= self.market_close_minute + 30) or (hour > self.market_close_hour and hour < self.postmarket_end_hour):
            return TradingPhase.POSTMARKET
            
        # Otherwise it's overnight
        return TradingPhase.OVERNIGHT
    
    def execute_phase_actions(self, phase: str):
        """
        Execute actions specific to a trading phase.
        
        Args:
            phase: Current trading phase
        """
        logger.info(f"Executing actions for phase: {phase}")
        
        if phase == TradingPhase.PREMARKET:
            self.execute_premarket_actions()
        elif phase == TradingPhase.MARKET_OPEN:
            self.execute_market_open_actions()
        elif phase == TradingPhase.MARKET_ACTIVE:
            self.execute_market_active_actions()
        elif phase == TradingPhase.MARKET_CLOSE:
            self.execute_market_close_actions()
        elif phase == TradingPhase.POSTMARKET:
            self.execute_postmarket_actions()
        elif phase == TradingPhase.OVERNIGHT:
            self.execute_overnight_actions()
    
    def execute_premarket_actions(self):
        """Execute premarket actions: analysis, news, strategy selection."""
        logger.info("Executing premarket actions")
        
        try:
            # 1. Fetch macro and news context
            news_data = self.news_fetcher.fetch_important_news()
            logger.info(f"Fetched {len(news_data)} important news items")
            
            # 2. Analyze market regime
            market_regime = self.strategy_prioritizer.analyze_market_regime()
            logger.info(f"Current market regime: {market_regime}")
            
            # 3. Run strategy scorer via CLI
            self.run_strategy_scorer()
            
            # 4. Select top strategy
            self.current_strategies = self.strategy_prioritizer.get_prioritized_strategies()
            logger.info(f"Selected {len(self.current_strategies)} top strategies")
            
            # 5. Prepare trading systems
            self.prepare_for_market_open()
            
            # 6. Send premarket summary to Telegram
            self.send_telegram_notification("PREMARKET SUMMARY", {
                "market_regime": market_regime,
                "top_strategies": [s["name"] for s in self.current_strategies[:3]],
                "key_news": [n["headline"] for n in news_data[:3]]
            })
            
        except Exception as e:
            logger.error(f"Error in premarket actions: {str(e)}")
            self.send_telegram_notification("ERROR", {
                "phase": "Premarket",
                "error": str(e)
            })
    
    def run_strategy_scorer(self):
        """Run the strategy scorer CLI tool."""
        try:
            script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                      "approval_workflow_cli.py")
                                      
            # Run the script as a subprocess
            result = subprocess.run(
                [sys.executable, script_path, "--mode", "score", "--output", "json"],
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Strategy scorer completed with output: {result.stdout[:100]}...")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Strategy scorer failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error running strategy scorer: {str(e)}")
            return False
    
    def prepare_for_market_open(self):
        """Prepare all systems for market open."""
        # This would involve checking system readiness, initializing order managers, etc.
        logger.info("Preparing systems for market open")
        # Placeholder for actual implementation
    
    def execute_market_open_actions(self):
        """Execute market open actions: start trading engine."""
        logger.info("Executing market open actions")
        
        try:
            # 1. Start the trading engine for selected strategies
            self.start_trading_engine()
            
            # 2. Set up monitoring for initial trades
            # Placeholder for implementation
            
            # 3. Send market open notification to Telegram
            self.send_telegram_notification("MARKET OPEN", {
                "strategies_active": len(self.current_strategies),
                "status": "Trading engine started"
            })
            
        except Exception as e:
            logger.error(f"Error in market open actions: {str(e)}")
            self.send_telegram_notification("ERROR", {
                "phase": "Market Open",
                "error": str(e)
            })
    
    def start_trading_engine(self):
        """Start the automated trading workflow engine."""
        try:
            script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                      "automated_trading_workflow.py")
                                      
            # Start the trading engine as a subprocess
            # We use Popen instead of run() because we want it to run asynchronously
            self.trading_engine_process = subprocess.Popen(
                [sys.executable, script_path, "--strategies", ",".join([s["id"] for s in self.current_strategies])],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info(f"Trading engine started with PID {self.trading_engine_process.pid}")
            return True
        except Exception as e:
            logger.error(f"Error starting trading engine: {str(e)}")
            return False
    
    def execute_market_active_actions(self):
        """Execute actions during active market hours: monitor trades, adjust strategies."""
        logger.info("Executing market active actions")
        
        try:
            # 1. Monitor active trades and PnL
            # This would typically be handled by the trading engine
            # But we would periodically check status
            
            # 2. Check for any strategy adjustments needed
            if hasattr(self, 'last_strategy_check'):
                time_since_check = datetime.now() - self.last_strategy_check
                # Check every 30 minutes
                if time_since_check.total_seconds() >= 1800:
                    self.check_strategy_adjustments()
                    self.last_strategy_check = datetime.now()
            else:
                self.last_strategy_check = datetime.now()
                
        except Exception as e:
            logger.error(f"Error in market active actions: {str(e)}")
    
    def check_strategy_adjustments(self):
        """Check if strategy adjustments are needed based on performance."""
        logger.info("Checking for strategy adjustments")
        # Placeholder for actual implementation
    
    def execute_market_close_actions(self):
        """Execute market close actions: finalize trades, prepare for postmarket."""
        logger.info("Executing market close actions")
        
        try:
            # 1. Check for any open positions that need to be closed
            # Placeholder for implementation
            
            # 2. Prepare for orderly shutdown of trading engine
            # Placeholder for implementation
            
            # 3. Send market close notification to Telegram
            self.send_telegram_notification("MARKET CLOSE", {
                "status": "Preparing to close trading day"
            })
            
        except Exception as e:
            logger.error(f"Error in market close actions: {str(e)}")
    
    def execute_postmarket_actions(self):
        """Execute postmarket actions: collect results, analyze, send reports."""
        logger.info("Executing postmarket actions")
        
        try:
            # 1. Stop the trading engine if it's still running
            self.stop_trading_engine()
            
            # 2. Collect trade results
            trade_results = self.collect_trade_results()
            
            # 3. Run performance analysis
            performance_data = self.performance_analyzer.analyze_daily_performance()
            
            # 4. Store results in performance database
            self.store_performance_data(performance_data)
            
            # 5. Update strategy scoring based on performance
            self.update_strategy_scoring(performance_data)
            
            # 6. Send daily summary to Telegram
            self.send_telegram_notification("DAILY TRADING SUMMARY", {
                "total_trades": trade_results.get("total_trades", 0),
                "profitable_trades": trade_results.get("profitable_trades", 0),
                "pnl": trade_results.get("total_pnl", 0),
                "best_strategy": performance_data.get("best_strategy", "None"),
                "worst_strategy": performance_data.get("worst_strategy", "None")
            })
            
        except Exception as e:
            logger.error(f"Error in postmarket actions: {str(e)}")
            self.send_telegram_notification("ERROR", {
                "phase": "Postmarket",
                "error": str(e)
            })
    
    def stop_trading_engine(self):
        """Stop the trading engine process if it's running."""
        if hasattr(self, 'trading_engine_process') and self.trading_engine_process:
            logger.info(f"Stopping trading engine (PID {self.trading_engine_process.pid})")
            try:
                # First try to terminate gracefully
                self.trading_engine_process.terminate()
                
                # Wait up to 30 seconds for process to exit
                for _ in range(30):
                    if self.trading_engine_process.poll() is not None:
                        break
                    time.sleep(1)
                
                # If still running, force kill
                if self.trading_engine_process.poll() is None:
                    logger.warning("Trading engine not responding to terminate, forcing kill")
                    self.trading_engine_process.kill()
                
                logger.info("Trading engine stopped")
                return True
            except Exception as e:
                logger.error(f"Error stopping trading engine: {str(e)}")
                return False
    
    def collect_trade_results(self) -> Dict[str, Any]:
        """Collect trading results for the day."""
        logger.info("Collecting trade results")
        # Placeholder for implementation
        return {
            "total_trades": 0,
            "profitable_trades": 0,
            "total_pnl": 0.0
        }
    
    def store_performance_data(self, performance_data: Dict[str, Any]):
        """Store performance data in the database."""
        if not self.performance_db:
            logger.warning("Performance database not available, cannot store data")
            return False
            
        try:
            # This is a placeholder - would need actual DB implementation
            logger.info(f"Storing performance data: {performance_data}")
            return True
        except Exception as e:
            logger.error(f"Error storing performance data: {str(e)}")
            return False
    
    def update_strategy_scoring(self, performance_data: Dict[str, Any]):
        """Update strategy scoring based on today's performance."""
        logger.info("Updating strategy scoring based on performance")
        # Placeholder for implementation
    
    def execute_overnight_actions(self):
        """Execute overnight actions: system maintenance, updates, etc."""
        logger.info("Executing overnight actions")
        
        try:
            # 1. Check for any system updates
            # Placeholder for implementation
            
            # 2. Run database maintenance
            # Placeholder for implementation
            
            # 3. Prepare for next trading day
            # Placeholder for implementation
            
        except Exception as e:
            logger.error(f"Error in overnight actions: {str(e)}")
    
    def execute_regular_checks(self):
        """Execute regular checks that happen regardless of trading phase."""
        # 1. Check system health via watchdog
        if self.watchdog:
            health_summary = self.watchdog.get_system_health_summary()
            if health_summary["overall_status"] != "HEALTHY":
                logger.warning(f"System health issues detected: {health_summary}")
        
        # 2. Check for any emergency messages or alerts
        # Placeholder for implementation
    
    def send_telegram_notification(self, title: str, data: Dict[str, Any]):
        """
        Send a notification to Telegram.
        
        Args:
            title: Notification title
            data: Dictionary of data to include in the notification
        """
        try:
            # Format the message
            message = f"📊 *{title}*\n\n"
            
            for key, value in data.items():
                # Format the key nicely
                formatted_key = key.replace("_", " ").title()
                
                # Format arrays/lists nicely
                if isinstance(value, list):
                    message += f"*{formatted_key}*:\n"
                    for item in value:
                        message += f"  • {item}\n"
                else:
                    message += f"*{formatted_key}*: {value}\n"
            
            logger.info(f"Sending Telegram notification: {title}")
            # Actual implementation would use the Telegram API or client
            
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {str(e)}")
            return False
    
    def stop(self):
        """Stop the daily trading loop."""
        if not self.running:
            logger.warning("Daily trading loop is not running")
            return
        
        logger.info("Stopping daily trading loop")
        
        # Signal the main loop to stop
        self.stop_event.set()
        self.running = False
        
        # Stop the trading engine if it's running
        self.stop_trading_engine()
        
        # Stop the watchdog service
        if self.watchdog:
            self.watchdog.stop()
        
        # Wait for the main thread to finish
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=30)
        
        logger.info("Daily trading loop stopped")
    
    def signal_handler(self, sig, frame):
        """Handle signals for graceful shutdown."""
        logger.info(f"Received signal {sig}, shutting down...")
        self.stop()
        sys.exit(0)

# For command-line usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BensBot Daily Trading Loop")
    parser.add_argument("--config", "-c", type=str, default="./config/trading_config.json",
                        help="Path to trading configuration file")
    args = parser.parse_args()
    
    # Create and start the daily trading loop
    trading_loop = DailyTradingLoop(config_path=args.config)
    trading_loop.start()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        trading_loop.stop()
