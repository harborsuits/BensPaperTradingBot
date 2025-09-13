#!/usr/bin/env python3
"""
Adaptive Scheduler for Trading Bot

This module implements an adaptive scheduling system that optimizes API usage
and event processing based on market hours, volatility, and resource constraints.
It dynamically adjusts processing frequency to ensure maximum coverage with efficient
resource utilization.

Key features:
- Tiered symbol processing based on priority and market conditions
- Time-based scheduling with different frequencies for market/off-hours
- Dynamic adjustment based on market volatility
- API rate limiting and budget management 
- Event-driven reactive processing alongside scheduled tasks
"""

import asyncio
import datetime
import logging
import threading
import time
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Callable, Set, Tuple, Union
from collections import deque, defaultdict

from trading_bot.core.constants import EventType, AssetType
from trading_bot.core.event_bus import Event, get_global_event_bus

logger = logging.getLogger(__name__)

class SymbolTier(Enum):
    """Priority tiers for symbols, determining refresh frequency."""
    TIER_1 = 1  # Highest priority - active trades, core indices
    TIER_2 = 2  # Medium priority - watchlist, high interest
    TIER_3 = 3  # Low priority - passive coverage
    
class MarketHours(Enum):
    """Market hour phases for scheduling purposes."""
    PRE_MARKET = auto()
    MARKET_HOURS = auto()
    POST_MARKET = auto()
    OVERNIGHT = auto()
    WEEKEND = auto()

class DataType(Enum):
    """Types of data to fetch."""
    PRICE = auto()
    QUOTE = auto()
    NEWS = auto()
    FUNDAMENTAL = auto()
    OPTION_CHAIN = auto()
    MARKET_DEPTH = auto()

class APIProvider(Enum):
    """API provider options."""
    ALPACA = auto()
    TRADIER = auto()
    MARKETAUX = auto()
    ETRADE = auto()
    TRADESTATION = auto()
    CUSTOM = auto()

class APILimitConfig:
    """Configuration for API rate limits."""
    def __init__(self, 
                 provider: APIProvider,
                 calls_per_second: float = 1.0,
                 calls_per_minute: int = 60,
                 calls_per_hour: Optional[int] = None,
                 calls_per_day: Optional[int] = None):
        self.provider = provider
        self.calls_per_second = calls_per_second
        self.calls_per_minute = calls_per_minute
        self.calls_per_hour = calls_per_hour or calls_per_minute * 60
        self.calls_per_day = calls_per_day or self.calls_per_hour * 24
        
        # Tracking usage
        self.second_counter = 0
        self.minute_counter = 0
        self.hour_counter = 0
        self.day_counter = 0
        
        # Last reset timestamps
        self.last_second_reset = time.time()
        self.last_minute_reset = time.time()
        self.last_hour_reset = time.time()
        self.last_day_reset = time.time()
    
    def can_make_call(self) -> bool:
        """Check if a call can be made within rate limits."""
        self._update_counters()
        
        return (self.second_counter < self.calls_per_second and
                self.minute_counter < self.calls_per_minute and
                self.hour_counter < self.calls_per_hour and
                self.day_counter < self.calls_per_day)
    
    def record_call(self) -> None:
        """Record that a call was made."""
        self._update_counters()
        
        self.second_counter += 1
        self.minute_counter += 1
        self.hour_counter += 1
        self.day_counter += 1
    
    def remaining_calls(self, time_window: str = 'minute') -> int:
        """Get remaining calls for a specific time window."""
        self._update_counters()
        
        if time_window == 'second':
            return max(0, self.calls_per_second - self.second_counter)
        elif time_window == 'minute':
            return max(0, self.calls_per_minute - self.minute_counter)
        elif time_window == 'hour':
            return max(0, self.calls_per_hour - self.hour_counter)
        elif time_window == 'day':
            return max(0, self.calls_per_day - self.day_counter)
        else:
            raise ValueError(f"Invalid time window: {time_window}")
    
    def _update_counters(self) -> None:
        """Update counters based on elapsed time."""
        current_time = time.time()
        
        # Reset second counter if a second has passed
        if current_time - self.last_second_reset >= 1.0:
            self.second_counter = 0
            self.last_second_reset = current_time
        
        # Reset minute counter if a minute has passed
        if current_time - self.last_minute_reset >= 60.0:
            self.minute_counter = 0
            self.last_minute_reset = current_time
        
        # Reset hour counter if an hour has passed
        if current_time - self.last_hour_reset >= 3600.0:
            self.hour_counter = 0
            self.last_hour_reset = current_time
        
        # Reset day counter if a day has passed
        if current_time - self.last_day_reset >= 86400.0:
            self.day_counter = 0
            self.last_day_reset = current_time

class ScheduledTask:
    """Represents a scheduled task with timing information."""
    def __init__(self, 
                 name: str,
                 callback: Callable,
                 interval_seconds: Dict[MarketHours, float],
                 data_type: DataType,
                 provider: APIProvider,
                 symbols: Optional[List[str]] = None,
                 args: Optional[List] = None,
                 kwargs: Optional[Dict] = None):
        self.name = name
        self.callback = callback
        self.interval_seconds = interval_seconds
        self.next_run_time = time.time()
        self.data_type = data_type
        self.provider = provider
        self.symbols = symbols or []
        self.args = args or []
        self.kwargs = kwargs or {}
        self.last_run_time = 0
        
    def should_run(self, current_time: float, market_hours: MarketHours) -> bool:
        """Check if the task should run now."""
        return current_time >= self.next_run_time
    
    def update_next_run(self, current_time: float, market_hours: MarketHours) -> None:
        """Update the next run time based on the current market hours."""
        interval = self.interval_seconds.get(market_hours, 60.0)  # Default 60s if not specified
        self.next_run_time = current_time + interval
        self.last_run_time = current_time
        
    def execute(self) -> Any:
        """Execute the scheduled task."""
        return self.callback(*self.args, **self.kwargs)

class SymbolRotator:
    """Manages symbol rotation for efficient API usage."""
    def __init__(self, batch_size: int = 25):
        self.batch_size = batch_size
        self.symbols_by_tier = {
            SymbolTier.TIER_1: deque(),
            SymbolTier.TIER_2: deque(),
            SymbolTier.TIER_3: deque(),
        }
        self.symbol_tiers = {}  # Mapping from symbol to tier
        
    def add_symbol(self, symbol: str, tier: SymbolTier) -> None:
        """Add a symbol to be tracked in the specified tier."""
        if symbol not in self.symbol_tiers:
            self.symbols_by_tier[tier].append(symbol)
            self.symbol_tiers[symbol] = tier
        elif self.symbol_tiers[symbol] != tier:
            # Remove from old tier
            old_tier = self.symbol_tiers[symbol]
            try:
                self.symbols_by_tier[old_tier].remove(symbol)
            except ValueError:
                pass
            # Add to new tier
            self.symbols_by_tier[tier].append(symbol)
            self.symbol_tiers[symbol] = tier
    
    def get_next_batch(self, tier: SymbolTier = None) -> List[str]:
        """Get the next batch of symbols to process."""
        if tier is not None:
            return self._get_batch_from_tier(tier)
        
        # Get symbols in priority order
        batch = []
        remaining = self.batch_size
        
        # Fill with Tier 1 first
        tier1_batch = self._get_batch_from_tier(SymbolTier.TIER_1, remaining)
        batch.extend(tier1_batch)
        remaining -= len(tier1_batch)
        
        if remaining > 0:
            # Then fill with Tier 2
            tier2_batch = self._get_batch_from_tier(SymbolTier.TIER_2, remaining)
            batch.extend(tier2_batch)
            remaining -= len(tier2_batch)
        
        if remaining > 0:
            # Finally fill with Tier 3
            tier3_batch = self._get_batch_from_tier(SymbolTier.TIER_3, remaining)
            batch.extend(tier3_batch)
        
        return batch
    
    def _get_batch_from_tier(self, tier: SymbolTier, limit: Optional[int] = None) -> List[str]:
        """Get a batch of symbols from a specific tier."""
        tier_queue = self.symbols_by_tier[tier]
        batch_limit = limit or self.batch_size
        
        if not tier_queue:
            return []
        
        # Take symbols from the front
        batch = []
        for _ in range(min(batch_limit, len(tier_queue))):
            symbol = tier_queue.popleft()
            batch.append(symbol)
            tier_queue.append(symbol)  # Add back to the end
            
        return batch
    
    def promote_symbol(self, symbol: str) -> None:
        """Promote a symbol to a higher tier."""
        if symbol not in self.symbol_tiers:
            return
        
        current_tier = self.symbol_tiers[symbol]
        if current_tier == SymbolTier.TIER_1:
            return  # Already at highest tier
        
        # Remove from current tier
        try:
            self.symbols_by_tier[current_tier].remove(symbol)
        except ValueError:
            pass
        
        # Add to higher tier
        new_tier = SymbolTier(current_tier.value - 1)
        self.symbols_by_tier[new_tier].append(symbol)
        self.symbol_tiers[symbol] = new_tier
        
    def demote_symbol(self, symbol: str) -> None:
        """Demote a symbol to a lower tier."""
        if symbol not in self.symbol_tiers:
            return
        
        current_tier = self.symbol_tiers[symbol]
        if current_tier == SymbolTier.TIER_3:
            return  # Already at lowest tier
        
        # Remove from current tier
        try:
            self.symbols_by_tier[current_tier].remove(symbol)
        except ValueError:
            pass
        
        # Add to lower tier
        new_tier = SymbolTier(current_tier.value + 1)
        self.symbols_by_tier[new_tier].append(symbol)
        self.symbol_tiers[symbol] = new_tier


class AdaptiveScheduler:
    """Main scheduler class that adapts to market conditions."""
    
    def __init__(self, strategy_manager=None, broker_manager=None):
        self.strategy_manager = strategy_manager
        self.broker_manager = broker_manager
        self.event_bus = get_global_event_bus()
        
        # API rate limiting
        self.api_limits = {
            APIProvider.ALPACA: APILimitConfig(APIProvider.ALPACA, calls_per_second=1.0, calls_per_minute=60),
            APIProvider.TRADIER: APILimitConfig(APIProvider.TRADIER, calls_per_second=0.5, calls_per_minute=30),
            APIProvider.MARKETAUX: APILimitConfig(APIProvider.MARKETAUX, calls_per_second=0.1, calls_per_minute=6, calls_per_day=1000),
            APIProvider.ETRADE: APILimitConfig(APIProvider.ETRADE, calls_per_second=0.5, calls_per_minute=30),
            APIProvider.TRADESTATION: APILimitConfig(APIProvider.TRADESTATION, calls_per_second=0.5, calls_per_minute=30)
        }
        
        # Symbol management
        self.symbol_rotator = SymbolRotator(batch_size=25)
        
        # Tasks management
        self.scheduled_tasks = []
        self.event_handlers = {}
        
        # Scheduling control
        self.running = False
        self.scheduler_thread = None
        self.scheduler_lock = threading.RLock()
        
        # Market hours config - customize for your market
        self.market_open_time = datetime.time(9, 30)  # 9:30 AM
        self.market_close_time = datetime.time(16, 0)  # 4:00 PM
        self.pre_market_start = datetime.time(4, 0)   # 4:00 AM
        self.post_market_end = datetime.time(20, 0)   # 8:00 PM
        
        # Dynamic scheduling
        self.volatility_factor = 1.0  # Multiplier for task frequency based on volatility
        
        # Hook up event handlers
        self._register_event_handlers()
    
    def _register_event_handlers(self):
        """Register handlers for various market events."""
        self.event_bus.subscribe(EventType.MARKET_DATA_UPDATE, self._on_market_data)
        self.event_bus.subscribe(EventType.QUOTE_UPDATE, self._on_quote_update)
        # Add more event handlers as needed
    
    def _on_market_data(self, event):
        """Handle market data updates."""
        if not self.running:
            return
            
        # Get event data
        data = event.data
        symbol = data.get('symbol')
        if not symbol:
            return
            
        # Process handlers for this event type
        handlers = self.event_handlers.get(EventType.MARKET_DATA_UPDATE, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Error in market data handler: {str(e)}")
    
    def _on_quote_update(self, event):
        """Handle quote updates."""
        if not self.running:
            return
            
        # Get event data
        data = event.data
        symbol = data.get('symbol')
        if not symbol:
            return
            
        # Check for significant price changes that might warrant promotion
        try:
            last_price = data.get('last')
            prev_close = data.get('prev_close')
            if last_price and prev_close and prev_close > 0:
                change_pct = abs((last_price - prev_close) / prev_close)
                
                # Promote symbols with significant moves
                if change_pct > 0.05:  # 5% move
                    self.symbol_rotator.promote_symbol(symbol)
                    logger.info(f"Promoted {symbol} to higher tier due to {change_pct:.2%} price change")
        except Exception as e:
            logger.error(f"Error processing quote for {symbol}: {str(e)}")
    
    def add_scheduled_task(self, name, callback, interval_seconds, data_type, provider, 
                           symbols=None, args=None, kwargs=None):
        """Add a new scheduled task."""
        # Convert interval to dict if it's a single value
        if isinstance(interval_seconds, (int, float)):
            interval_dict = {market_hour: interval_seconds for market_hour in MarketHours}
        else:
            interval_dict = interval_seconds
        
        task = ScheduledTask(
            name=name,
            callback=callback,
            interval_seconds=interval_dict,
            data_type=data_type,
            provider=provider,
            symbols=symbols,
            args=args,
            kwargs=kwargs
        )
        
        with self.scheduler_lock:
            self.scheduled_tasks.append(task)
    
    def register_event_handler(self, event_type, handler):
        """Register a handler for a specific event type."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
    
    def add_symbol(self, symbol, tier=SymbolTier.TIER_2):
        """Add a symbol to be tracked."""
        self.symbol_rotator.add_symbol(symbol, tier)
    
    def start(self):
        """Start the scheduler."""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="AdaptiveScheduler"
        )
        self.scheduler_thread.start()
        logger.info("Adaptive scheduler started")
    
    def stop(self):
        """Stop the scheduler."""
        if not self.running:
            logger.warning("Scheduler is not running")
            return
        
        self.running = False
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)
        
        logger.info("Adaptive scheduler stopped")
    
    def _get_current_market_hours(self):
        """Determine the current market hours phase."""
        now = datetime.datetime.now().time()
        today = datetime.datetime.now().date()
        weekday = today.weekday()  # 0=Monday, 6=Sunday
        
        # Weekend check
        if weekday >= 5:  # Saturday or Sunday
            return MarketHours.WEEKEND
        
        # Check market phases
        if self.market_open_time <= now < self.market_close_time:
            return MarketHours.MARKET_HOURS
        elif self.pre_market_start <= now < self.market_open_time:
            return MarketHours.PRE_MARKET
        elif self.market_close_time <= now < self.post_market_end:
            return MarketHours.POST_MARKET
        else:
            return MarketHours.OVERNIGHT
    
    def _update_volatility_factor(self):
        """Update the volatility factor based on market conditions."""
        # This is a placeholder for a more sophisticated implementation
        # In a real implementation, you would analyze market-wide volatility indicators
        # such as VIX, recent price movements, trading volumes, etc.
        
        # For now, just use a higher factor during market hours
        market_hours = self._get_current_market_hours()
        
        if market_hours == MarketHours.MARKET_HOURS:
            # Check for high volatility indicators
            # For now, just use a placeholder implementation
            self.volatility_factor = 1.5  # Higher frequency during market hours
        elif market_hours == MarketHours.PRE_MARKET:
            self.volatility_factor = 1.2  # Moderate frequency pre-market
        elif market_hours == MarketHours.POST_MARKET:
            self.volatility_factor = 1.0  # Normal frequency post-market
        else:
            self.volatility_factor = 0.5  # Lower frequency overnight/weekends
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        logger.info("Scheduler loop started")
        
        while self.running:
            try:
                # Update market hours and volatility factor
                market_hours = self._get_current_market_hours()
                self._update_volatility_factor()
                
                current_time = time.time()
                
                # Process scheduled tasks
                with self.scheduler_lock:
                    for task in self.scheduled_tasks:
                        if task.should_run(current_time, market_hours):
                            # Check API limits
                            if task.provider in self.api_limits:
                                api_limit = self.api_limits[task.provider]
                                if not api_limit.can_make_call():
                                    logger.warning(f"Skipping task {task.name} due to API rate limit for {task.provider}")
                                    continue
                                
                                # Record the API call
                                api_limit.record_call()
                            
                            # Execute the task
                            try:
                                task.execute()
                                task.update_next_run(current_time, market_hours)
                            except Exception as e:
                                logger.error(f"Error executing task {task.name}: {str(e)}")
                
                # Sleep for a short time before next check
                # Use shorter intervals during market hours for responsiveness
                if market_hours == MarketHours.MARKET_HOURS:
                    time.sleep(0.1)  # 100ms during market hours
                else:
                    time.sleep(0.5)  # 500ms during off hours
                    
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                time.sleep(1.0)  # Sleep longer on error
        
        logger.info("Scheduler loop stopped")
    
    def get_status(self):
        """Get the current status of the scheduler."""
        status = {
            "running": self.running,
            "market_hours": self._get_current_market_hours().name,
            "volatility_factor": self.volatility_factor,
            "symbols": {
                "tier_1": len(self.symbol_rotator.symbols_by_tier[SymbolTier.TIER_1]),
                "tier_2": len(self.symbol_rotator.symbols_by_tier[SymbolTier.TIER_2]),
                "tier_3": len(self.symbol_rotator.symbols_by_tier[SymbolTier.TIER_3]),
            },
            "api_limits": {
                provider.name: {
                    "remaining_minute": limit.remaining_calls("minute"),
                    "remaining_day": limit.remaining_calls("day")
                }
                for provider, limit in self.api_limits.items()
            },
            "tasks": len(self.scheduled_tasks)
        }
        
        return status
