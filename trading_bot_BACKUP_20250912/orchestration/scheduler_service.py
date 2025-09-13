"""
Scheduler Service

This module provides a scheduling service for the trading bot,
controlling when trading cycles are executed based on configured timeframes
and market hours. It handles both periodic scheduling and market-aware timing.
"""

import logging
import time
import threading
import schedule
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
import pytz

# Market hours utilities
try:
    from trading_bot.brokers.broker_interface import get_market_hours_for_symbol, check_market_open
    MARKET_HOURS_AVAILABLE = True
except ImportError:
    MARKET_HOURS_AVAILABLE = False
    logging.warning("Market hours utilities not available, running on schedule only")

# Import the orchestrator
from trading_bot.orchestration.main_orchestrator import MainOrchestrator

logger = logging.getLogger(__name__)

class SchedulerService:
    """
    Scheduler service for the trading bot
    
    Handles scheduling of trading cycles based on configurable intervals
    and market hours. Supports both time-based and event-based scheduling.
    """
    
    def __init__(self, orchestrator=None, config=None):
        """
        Initialize the scheduler service
        
        Args:
            orchestrator: MainOrchestrator instance or None to create new one
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize orchestrator
        if orchestrator and isinstance(orchestrator, MainOrchestrator):
            self.orchestrator = orchestrator
            logger.info("Using provided orchestrator for scheduling")
        else:
            logger.info("Creating new orchestrator for scheduling")
            config_path = self.config.get("config_path", None)
            self.orchestrator = MainOrchestrator(config_path=config_path)
            
        # Threading control
        self.scheduler_thread = None
        self.running = False
        self.stop_event = threading.Event()
        
        # Schedule configuration
        self.schedule_config = self.config.get("schedule", {})
        self.default_interval = self.schedule_config.get("default_interval", 60)  # Minutes
        self.market_hours_aware = self.schedule_config.get("market_hours_aware", True)
        self.timezone = pytz.timezone(self.schedule_config.get("timezone", "America/New_York"))
        
        # Strategy-specific schedules
        self.strategy_schedules = self.schedule_config.get("strategies", {})
        
        # Schedule state tracking
        self.last_run = {}
        self.next_run = {}
        
        # Initialize schedule
        self._initialize_schedules()
        
        logger.info("Scheduler service initialized")
    
    def _initialize_schedules(self):
        """Initialize all schedules based on configuration"""
        # Clear any existing schedules
        schedule.clear()
        
        # Get active strategies from orchestrator
        active_strategies = self.orchestrator.active_strategies
        
        # Schedule jobs for active strategies based on their configuration
        for strategy_name in active_strategies:
            # Get strategy-specific schedule config, fallback to default
            strategy_config = self.strategy_schedules.get(strategy_name, {})
            
            # Get interval in minutes, fallback to default
            interval_minutes = strategy_config.get("interval", self.default_interval)
            
            # Create schedule for this strategy
            if interval_minutes > 0:
                # Convert to hours/minutes for readability
                hours = interval_minutes // 60
                minutes = interval_minutes % 60
                
                # Create schedule description
                if hours > 0 and minutes > 0:
                    interval_desc = f"{hours} hours and {minutes} minutes"
                elif hours > 0:
                    interval_desc = f"{hours} hour{'s' if hours > 1 else ''}"
                else:
                    interval_desc = f"{minutes} minute{'s' if minutes > 1 else ''}"
                
                # Add to schedule
                self._schedule_strategy(strategy_name, interval_minutes)
                logger.info(f"Scheduled {strategy_name} to run every {interval_desc}")
        
        # Add special schedules based on time of day, if configured
        
        # Market open
        if self.schedule_config.get("run_at_market_open", False) and MARKET_HOURS_AVAILABLE:
            schedule.every().day.at("09:30").do(self._market_open_jobs)
            logger.info("Scheduled market open jobs at 09:30 ET")
        
        # Market close
        if self.schedule_config.get("run_at_market_close", False) and MARKET_HOURS_AVAILABLE:
            schedule.every().day.at("15:45").do(self._market_close_jobs)
            logger.info("Scheduled market close jobs at 15:45 ET (15 min before close)")
        
        # End of day
        if self.schedule_config.get("run_at_end_of_day", False):
            eod_time = self.schedule_config.get("end_of_day_time", "17:00")
            schedule.every().day.at(eod_time).do(self._end_of_day_jobs)
            logger.info(f"Scheduled end of day jobs at {eod_time} ET")
        
        logger.info("All schedules initialized")
    
    def _schedule_strategy(self, strategy_name: str, interval_minutes: int):
        """
        Schedule a strategy to run at the specified interval
        
        Args:
            strategy_name: Name of the strategy to schedule
            interval_minutes: Interval in minutes
        """
        def run_strategy_job():
            self._run_strategy_if_appropriate(strategy_name)
        
        # Calculate hours and minutes parts
        hours = interval_minutes // 60
        minutes = interval_minutes % 60
        
        # Create appropriate schedule
        if hours > 0:
            if minutes > 0:
                # Mixed hours and minutes - run every X hours at specific minute
                job = schedule.every(hours).hours
                # Add minute offset
                current_minute = datetime.now().minute
                target_minute = (current_minute + minutes) % 60
                job = job.at(f":{target_minute:02d}")
            else:
                # Even hours
                job = schedule.every(hours).hours
        else:
            # Just minutes
            job = schedule.every(minutes).minutes
        
        # Register the job
        job.do(run_strategy_job).tag(strategy_name)
    
    def _run_strategy_if_appropriate(self, strategy_name: str):
        """
        Run a strategy if market conditions are appropriate
        
        Args:
            strategy_name: Name of the strategy to run
        """
        try:
            # Skip if not running
            if not self.running:
                return
            
            # Check if should respect market hours
            strategy_config = self.strategy_schedules.get(strategy_name, {})
            respect_market_hours = strategy_config.get("respect_market_hours", self.market_hours_aware)
            
            # If respecting market hours and market hours utilities available
            if respect_market_hours and MARKET_HOURS_AVAILABLE:
                # Get symbols for this strategy
                symbols = strategy_config.get("symbols", [])
                
                # If no symbols specified, get from orchestrator
                if not symbols:
                    strategy_config = self.orchestrator.config.get("strategy", {}).get(strategy_name, {})
                    symbols = strategy_config.get("symbols", [])
                
                # If still no symbols, use market indices
                if not symbols:
                    symbols = ["SPY"]  # Default to SPY for US markets
                
                # Check if market is open for at least one symbol
                market_open = False
                for symbol in symbols:
                    if check_market_open(symbol):
                        market_open = True
                        break
                
                if not market_open:
                    logger.info(f"Skipping {strategy_name} run - market closed for all symbols")
                    return
            
            # Run the strategy
            logger.info(f"Running scheduled strategy: {strategy_name}")
            self.orchestrator.run_pipeline(strategy_name)
            
            # Update last run time
            self.last_run[strategy_name] = datetime.now()
            
            # Calculate next run time
            interval_minutes = strategy_config.get("interval", self.default_interval)
            self.next_run[strategy_name] = datetime.now() + timedelta(minutes=interval_minutes)
            
        except Exception as e:
            logger.error(f"Error running scheduled strategy {strategy_name}: {e}")
            logger.error(traceback.format_exc())
    
    def _market_open_jobs(self):
        """Run jobs that should execute at market open"""
        try:
            if not self.running:
                return
                
            logger.info("Running market open jobs")
            
            # Get strategies to run at market open
            market_open_strategies = []
            for strategy_name, config in self.strategy_schedules.items():
                if config.get("run_at_market_open", False):
                    market_open_strategies.append(strategy_name)
            
            # Run strategies
            for strategy_name in market_open_strategies:
                if strategy_name in self.orchestrator.active_strategies:
                    logger.info(f"Running {strategy_name} at market open")
                    self.orchestrator.run_pipeline(strategy_name)
                    
                    # Update tracking
                    self.last_run[strategy_name] = datetime.now()
        
        except Exception as e:
            logger.error(f"Error in market open jobs: {e}")
            logger.error(traceback.format_exc())
    
    def _market_close_jobs(self):
        """Run jobs that should execute near market close"""
        try:
            if not self.running:
                return
                
            logger.info("Running market close jobs")
            
            # Get strategies to run at market close
            market_close_strategies = []
            for strategy_name, config in self.strategy_schedules.items():
                if config.get("run_at_market_close", False):
                    market_close_strategies.append(strategy_name)
            
            # Run strategies
            for strategy_name in market_close_strategies:
                if strategy_name in self.orchestrator.active_strategies:
                    logger.info(f"Running {strategy_name} at market close")
                    self.orchestrator.run_pipeline(strategy_name)
                    
                    # Update tracking
                    self.last_run[strategy_name] = datetime.now()
        
        except Exception as e:
            logger.error(f"Error in market close jobs: {e}")
            logger.error(traceback.format_exc())
    
    def _end_of_day_jobs(self):
        """Run jobs that should execute at end of day"""
        try:
            if not self.running:
                return
                
            logger.info("Running end of day jobs")
            
            # Get strategies to run at end of day
            eod_strategies = []
            for strategy_name, config in self.strategy_schedules.items():
                if config.get("run_at_end_of_day", False):
                    eod_strategies.append(strategy_name)
            
            # Run strategies
            for strategy_name in eod_strategies:
                if strategy_name in self.orchestrator.active_strategies:
                    logger.info(f"Running {strategy_name} at end of day")
                    self.orchestrator.run_pipeline(strategy_name)
                    
                    # Update tracking
                    self.last_run[strategy_name] = datetime.now()
        
        except Exception as e:
            logger.error(f"Error in end of day jobs: {e}")
            logger.error(traceback.format_exc())
    
    def start(self):
        """Start the scheduler service"""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        try:
            # Initialize and start the orchestrator
            if not self.orchestrator.running:
                self.orchestrator.start()
                
            # Set running flag
            self.running = True
            self.stop_event.clear()
            
            # Start scheduler thread
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()
            
            logger.info("Scheduler service started")
            
        except Exception as e:
            logger.error(f"Error starting scheduler service: {e}")
            logger.error(traceback.format_exc())
            self.running = False
    
    def stop(self):
        """Stop the scheduler service"""
        if not self.running:
            logger.warning("Scheduler is already stopped")
            return
        
        try:
            # Set stop flag
            self.running = False
            self.stop_event.set()
            
            # Wait for scheduler thread to stop
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=5.0)
                
            # Stop the orchestrator
            if self.orchestrator.running:
                self.orchestrator.stop()
                
            logger.info("Scheduler service stopped")
            
        except Exception as e:
            logger.error(f"Error stopping scheduler service: {e}")
            logger.error(traceback.format_exc())
    
    def _scheduler_loop(self):
        """Main scheduler loop - runs in separate thread"""
        logger.info("Scheduler loop started")
        
        while not self.stop_event.is_set():
            try:
                # Run pending scheduled jobs
                schedule.run_pending()
                
                # Sleep for 1 second
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                logger.error(traceback.format_exc())
                # Brief sleep to prevent tight error loop
                time.sleep(5)
        
        logger.info("Scheduler loop stopped")
    
    def get_schedule_status(self) -> Dict[str, Any]:
        """
        Get current schedule status
        
        Returns:
            Dict with schedule information for all strategies
        """
        status = {
            "running": self.running,
            "strategies": {},
            "next_jobs": []
        }
        
        # Get status for each strategy
        for strategy_name in self.orchestrator.active_strategies:
            status["strategies"][strategy_name] = {
                "last_run": self.last_run.get(strategy_name),
                "next_run": self.next_run.get(strategy_name),
                "interval": self.strategy_schedules.get(strategy_name, {}).get("interval", self.default_interval),
                "respect_market_hours": self.strategy_schedules.get(strategy_name, {}).get("respect_market_hours", self.market_hours_aware)
            }
        
        # Get upcoming jobs
        for job in schedule.get_jobs():
            status["next_jobs"].append({
                "name": str(job.tags),
                "next_run": job.next_run
            })
        
        return status
    
    def update_schedule(self, strategy_name: str, interval_minutes: int, respect_market_hours: Optional[bool] = None):
        """
        Update schedule for a specific strategy
        
        Args:
            strategy_name: Name of the strategy to update
            interval_minutes: New interval in minutes
            respect_market_hours: Whether to respect market hours
            
        Returns:
            bool: Success status
        """
        try:
            # Check if strategy is active
            if strategy_name not in self.orchestrator.active_strategies:
                logger.warning(f"Strategy {strategy_name} is not active, cannot update schedule")
                return False
            
            # Update strategy schedule config
            if strategy_name not in self.strategy_schedules:
                self.strategy_schedules[strategy_name] = {}
                
            # Update interval
            self.strategy_schedules[strategy_name]["interval"] = interval_minutes
            
            # Update market hours respect if provided
            if respect_market_hours is not None:
                self.strategy_schedules[strategy_name]["respect_market_hours"] = respect_market_hours
            
            # Remove existing schedule for this strategy
            schedule.clear(strategy_name)
            
            # Add new schedule
            self._schedule_strategy(strategy_name, interval_minutes)
            
            # Log the change
            logger.info(f"Updated schedule for {strategy_name}: every {interval_minutes} minutes, respect_market_hours={self.strategy_schedules[strategy_name].get('respect_market_hours', self.market_hours_aware)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating schedule for {strategy_name}: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def run_now(self, strategy_name: str = None) -> Dict[str, Any]:
        """
        Run a strategy immediately, outside of the schedule
        
        Args:
            strategy_name: Name of the strategy to run, or None for all active
            
        Returns:
            Dict with execution result
        """
        try:
            # Run the pipeline
            result = self.orchestrator.run_pipeline(strategy_name)
            
            # Update last run time
            if strategy_name:
                self.last_run[strategy_name] = datetime.now()
            else:
                # Update all active strategies
                for strategy in self.orchestrator.active_strategies:
                    self.last_run[strategy] = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"Error running strategy {strategy_name}: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "errors": [str(e)]
            }


# Convenience function to create scheduler
def create_scheduler(config_path: str = None, config: Dict = None) -> SchedulerService:
    """
    Create a scheduler service with the given configuration
    
    Args:
        config_path: Path to configuration file
        config: Configuration dictionary
        
    Returns:
        SchedulerService instance
    """
    # Create config object
    scheduler_config = config or {}
    
    # Add config_path if provided
    if config_path:
        scheduler_config["config_path"] = config_path
        
    # Create and return scheduler
    return SchedulerService(config=scheduler_config)
