import logging
import os
import time
import json
from datetime import datetime, timedelta
import threading
import schedule

from trading_bot.market_context.context_analyzer import MarketContextAnalyzer
from trading_bot.security.security_utils import redact_sensitive_data

logger = logging.getLogger(__name__)

class AdaptiveContextScheduler:
    """
    Dynamically schedules market context updates with different frequencies 
    during market hours vs non-market hours
    """
    
    def __init__(self, config):
        """
        Initialize the adaptive context scheduler
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.market_hours_start = config.get("MARKET_HOURS_START", "05:00")  # 5:00 AM
        self.market_hours_end = config.get("MARKET_HOURS_END", "16:00")      # 4:00 PM
        self.market_hours_interval = config.get("MARKET_HOURS_INTERVAL", 15) # Minutes
        self.after_hours_interval = config.get("AFTER_HOURS_INTERVAL", 60)   # Minutes
        
        self.output_dir = config.get("OUTPUT_DIR", "data/market_context")
        self.output_filename = config.get("OUTPUT_FILENAME", "current_market_context.json")
        self.daily_filename = config.get("DAILY_FILENAME", "daily_strategy_bias.json")
        
        # Initialize context analyzer
        self.context_analyzer = MarketContextAnalyzer(config)
        
        # Status tracking
        self.last_update_time = None
        self.is_running = False
        self.current_schedule_type = None
        self._scheduler_thread = None
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "history"), exist_ok=True)
        
        logger.info(f"Adaptive context scheduler initialized with market hours {self.market_hours_start}-{self.market_hours_end}")
        logger.info(f"Update frequency: {self.market_hours_interval}min during market, {self.after_hours_interval}min after hours")
    
    def is_market_hours(self):
        """
        Check if current time is within market hours
        
        Returns:
            Boolean indicating if current time is within market hours
        """
        now = datetime.now()
        now_time = now.strftime("%H:%M")
        
        return self.market_hours_start <= now_time < self.market_hours_end
    
    def update_market_context(self, is_daily_update=False):
        """
        Update market context and save to file
        
        Args:
            is_daily_update: Boolean indicating if this is the daily update (5 AM)
            
        Returns:
            Path to the generated context file
        """
        try:
            logger.info(f"Updating market context (daily={is_daily_update})")
            
            # Get current date and time
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M")
            
            # Get market context with fresh data
            context = self.context_analyzer.get_market_context(force_refresh=True)
            
            # Add metadata
            context['date'] = today
            context['time'] = time_str
            context['is_market_hours'] = self.is_market_hours()
            
            # Save to current file
            output_path = os.path.join(self.output_dir, self.output_filename)
            with open(output_path, 'w') as f:
                json.dump(context, f, indent=2)
            
            # If this is the daily update (5 AM), also save as daily strategy bias
            if is_daily_update:
                daily_path = os.path.join(self.output_dir, self.daily_filename)
                
                # Determine strategies to avoid based on bias
                bias = context.get("bias", "neutral")
                suggested_strategies = context.get("suggested_strategies", [])
                
                # Simple logic to determine strategies to avoid
                avoid_bias = "bearish" if bias == "bullish" else "bullish"
                
                # Get strategy list from config
                strategy_list = self.config.get("STRATEGY_LIST", [])
                
                # Generate avoid strategies (those not recommended and opposite of current bias)
                avoid_strategies = [s for s in strategy_list if s not in suggested_strategies]
                
                # Create daily context
                daily_context = {
                    "date": today,
                    "bias": bias,
                    "confidence": context.get("confidence", 0.5),
                    "recommended_strategies": suggested_strategies,
                    "avoid_strategies": avoid_strategies,
                    "summary": context.get("reasoning", ""),
                    "market_drivers": context.get("triggers", []),
                    "generated_at": datetime.now().isoformat()
                }
                
                # Save daily context
                with open(daily_path, 'w') as f:
                    json.dump(daily_context, f, indent=2)
                
                # Also save a dated version for historical records
                historical_filename = f"strategy_bias_{today}.json"
                historical_path = os.path.join(self.output_dir, "history", historical_filename)
                with open(historical_path, 'w') as f:
                    json.dump(daily_context, f, indent=2)
                
                logger.info(f"Daily context generated and saved to {daily_path}")
            
            # Also save an hourly version for historical records
            hourly_filename = f"context_{today}_{now.strftime('%H%M')}.json"
            hourly_path = os.path.join(self.output_dir, "history", hourly_filename)
            with open(hourly_path, 'w') as f:
                json.dump(context, f, indent=2)
            
            # Update last update time
            self.last_update_time = now
            
            # Log safely
            safe_output = redact_sensitive_data(context)
            logger.info(f"Market context updated: {json.dumps(safe_output)}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error updating market context: {str(e)}", exc_info=True)
            self.last_update_time = datetime.now()  # Still update time to prevent retries too quickly
            return None
    
    def setup_schedule(self):
        """
        Set up schedule based on current time (market hours or after hours)
        """
        # Clear existing schedule
        schedule.clear()
        
        # Check if current time is within market hours
        if self.is_market_hours():
            if self.current_schedule_type != "market_hours":
                logger.info(f"Setting up market hours schedule (every {self.market_hours_interval} minutes)")
                schedule.every(self.market_hours_interval).minutes.do(self.update_market_context)
                self.current_schedule_type = "market_hours"
        else:
            if self.current_schedule_type != "after_hours":
                logger.info(f"Setting up after hours schedule (every {self.after_hours_interval} minutes)")
                schedule.every(self.after_hours_interval).minutes.do(self.update_market_context)
                self.current_schedule_type = "after_hours"
        
        # Special schedule for daily update (5 AM)
        now = datetime.now()
        now_time = now.strftime("%H:%M")
        if self.market_hours_start == now_time:
            # This is the 5 AM run - special treatment as the daily update
            schedule.every().day.at(self.market_hours_start).do(lambda: self.update_market_context(is_daily_update=True))
    
    def _run_scheduler(self):
        """
        Run the scheduler continuously
        """
        # Initial context update
        if not os.path.exists(os.path.join(self.output_dir, self.output_filename)):
            logger.info("No existing context file found - generating now")
            self.update_market_context(is_daily_update=(datetime.now().strftime("%H:%M") == self.market_hours_start))
        
        logger.info("Starting adaptive context scheduler")
        
        while self.is_running:
            # Update schedule configuration every loop to adapt to time changes
            self.setup_schedule()
            
            # Run pending scheduled tasks
            schedule.run_pending()
            
            # Sleep for a short time
            time.sleep(30)  # Check every 30 seconds
    
    def start(self):
        """
        Start the scheduler in a background thread
        """
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        self._scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._scheduler_thread.start()
        logger.info("Adaptive context scheduler started")
    
    def stop(self):
        """
        Stop the scheduler
        """
        self.is_running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=2.0)
            logger.info("Adaptive context scheduler stopped")
    
    def get_status(self):
        """
        Get the current status of the scheduler
        
        Returns:
            Dictionary with scheduler status
        """
        return {
            "is_running": self.is_running,
            "last_update": self.last_update_time.isoformat() if self.last_update_time else None,
            "market_hours": self.is_market_hours(),
            "current_schedule_type": self.current_schedule_type,
            "market_hours_interval": self.market_hours_interval,
            "after_hours_interval": self.after_hours_interval
        } 