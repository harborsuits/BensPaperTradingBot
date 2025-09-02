"""
Live Data Source for BensBot

This module implements a real-time data source that can connect to various
market data providers and publish market data events to the event bus.
It supports both direct API connections and websocket feeds.
"""
import logging
import threading
import time
import random
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from trading_bot.core.event_bus import EventBus, get_global_event_bus, Event
from trading_bot.core.constants import EventType
from trading_bot.data.persistence import PersistenceManager
from trading_bot.data.models import TimeFrame

logger = logging.getLogger(__name__)

class LiveDataSource:
    """
    Live data source that provides real-time market data.
    Can connect to various market data providers and simulate data for testing.
    """
    
    def __init__(self, 
               persistence_manager: Optional[PersistenceManager] = None,
               provider: str = "simulation",
               symbols: Optional[List[str]] = None,
               timeframes: Optional[List[str]] = None,
               api_key: Optional[str] = None,
               api_secret: Optional[str] = None,
               **kwargs):
        """
        Initialize the live data source.
        
        Args:
            persistence_manager: Optional persistence manager for storing data
            provider: Data provider (e.g., "tradier", "alpaca", "simulation")
            symbols: List of symbols to track
            timeframes: List of timeframes to track (e.g., "1m", "5m", "1h")
            api_key: API key for the data provider
            api_secret: API secret for the data provider
            **kwargs: Additional provider-specific parameters
        """
        self.persistence = persistence_manager
        self.provider = provider.lower()
        self.symbols = symbols or ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"]
        self.timeframes = timeframes or ["1m", "5m", "15m", "1h", "4h", "1d"]
        self.api_key = api_key
        self.api_secret = api_secret
        self.provider_config = kwargs
        
        self.event_bus = get_global_event_bus()
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Store market data for each symbol and timeframe
        self.data: Dict[str, Dict[str, pd.DataFrame]] = {}
        
        # Initialize data structures
        for symbol in self.symbols:
            self.data[symbol] = {}
            for timeframe in self.timeframes:
                self.data[symbol][timeframe] = pd.DataFrame(columns=[
                    "timestamp", "open", "high", "low", "close", "volume"
                ])
        
        # Track market regimes for symbols
        self.market_regimes: Dict[str, Dict[str, Any]] = {}
        for symbol in self.symbols:
            self.market_regimes[symbol] = {
                "current_regime": random.choice(["trending", "ranging", "volatile", "low_volatility"]),
                "confidence": random.uniform(0.65, 0.95),
                "duration_bars": random.randint(10, 100)
            }
        
        logger.info(f"Initialized live data source with provider: {provider}")
    
    def start(self):
        """Start the live data source."""
        if self.running:
            logger.warning("Live data source already running")
            return
        
        self.running = True
        
        if self.provider == "simulation":
            # Start simulation in a separate thread
            self.thread = threading.Thread(target=self._run_simulation)
            self.thread.daemon = True
            self.thread.start()
            logger.info("Started simulation data thread")
        else:
            # Connect to the actual provider
            self._connect_to_provider()
        
        # Publish data source started event
        self.event_bus.create_and_publish(
            event_type=EventType.SYSTEM_STARTED,
            data={"component": "live_data_source", "provider": self.provider},
            source="live_data_source"
        )
    
    def stop(self):
        """Stop the live data source."""
        if not self.running:
            logger.warning("Live data source not running")
            return
        
        self.running = False
        
        if self.thread and self.thread.is_alive():
            # Give the thread time to exit cleanly
            self.thread.join(timeout=2.0)
        
        # Publish data source stopped event
        self.event_bus.create_and_publish(
            event_type=EventType.SYSTEM_STOPPED,
            data={"component": "live_data_source"},
            source="live_data_source"
        )
        
        logger.info("Stopped live data source")
    
    def is_running(self) -> bool:
        """Check if the live data source is running."""
        return self.running
    
    def _connect_to_provider(self):
        """Connect to the actual data provider."""
        logger.info(f"Connecting to provider: {self.provider}")
        
        if self.provider == "tradier":
            # Implement Tradier connection logic
            pass
        elif self.provider == "alpaca":
            # Implement Alpaca connection logic
            pass
        elif self.provider == "fmp":
            # Implement Financial Modeling Prep connection logic
            pass
        elif self.provider == "yahoo":
            # Implement Yahoo Finance connection logic
            pass
        else:
            logger.warning(f"Unknown provider: {self.provider}, falling back to simulation")
            self.provider = "simulation"
            self.thread = threading.Thread(target=self._run_simulation)
            self.thread.daemon = True
            self.thread.start()
    
    def _run_simulation(self):
        """Run the market data simulation."""
        logger.info("Starting market data simulation")
        
        # Define simulation parameters
        update_interval = 1.0  # seconds between updates
        bar_close_intervals = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400
        }
        
        # Initialize counters for bar closings
        bar_counters = {tf: 0 for tf in self.timeframes}
        
        # Generate initial price data
        prices = {}
        for symbol in self.symbols:
            if "USD" in symbol:
                # Forex pairs around typical values
                base_price = 1.0 if symbol.startswith("EUR") else (
                    1.3 if symbol.startswith("GBP") else (
                        0.7 if symbol.startswith("AUD") else (
                            110.0 if symbol.endswith("JPY") else 1.2
                        )
                    )
                )
            else:
                # Stocks/crypto with random starting prices
                base_price = random.uniform(10, 500)
            
            prices[symbol] = base_price
        
        # Main simulation loop
        last_time = datetime.now()
        while self.running:
            current_time = datetime.now()
            elapsed = (current_time - last_time).total_seconds()
            
            if elapsed >= update_interval:
                # Generate new price ticks for all symbols
                for symbol in self.symbols:
                    # Update price based on current market regime
                    regime = self.market_regimes[symbol]["current_regime"]
                    
                    # Adjust volatility based on regime
                    vol_factor = 0.0005  # base volatility
                    if regime == "volatile":
                        vol_factor *= 3
                    elif regime == "trending":
                        vol_factor *= 1.5
                    elif regime == "ranging":
                        vol_factor *= 1.2
                    
                    # Generate price movement
                    if regime == "trending":
                        # Trending markets have directional movement
                        drift = random.choice([-1, 1]) * 0.0002  # direction
                        price_change = drift + random.normalvariate(0, vol_factor)
                    elif regime == "ranging":
                        # Ranging markets revert to mean
                        mean_reversion = (base_price - prices[symbol]) * 0.05
                        price_change = mean_reversion + random.normalvariate(0, vol_factor)
                    elif regime == "volatile":
                        # Volatile markets have larger random moves
                        price_change = random.normalvariate(0, vol_factor)
                    else:  # low_volatility
                        # Low volatility markets have small random moves
                        price_change = random.normalvariate(0, vol_factor * 0.5)
                    
                    # Apply price change
                    price_change_pct = price_change
                    prices[symbol] *= (1 + price_change_pct)
                    
                    # Publish tick data to event bus
                    tick_data = {
                        "symbol": symbol,
                        "price": prices[symbol],
                        "timestamp": current_time,
                        "volume": random.randint(1000, 10000)
                    }
                    
                    self.event_bus.create_and_publish(
                        event_type=EventType.TICK_RECEIVED,
                        data=tick_data,
                        source="live_data_source"
                    )
                
                # Check for bar closings
                for timeframe, interval in bar_close_intervals.items():
                    bar_counters[timeframe] += elapsed
                    
                    if bar_counters[timeframe] >= interval:
                        # Reset counter
                        bar_counters[timeframe] = 0
                        
                        # Close bars for this timeframe
                        self._close_bars(timeframe, current_time)
                        
                        # Occasionally change market regimes (every ~20-100 bars)
                        for symbol in self.symbols:
                            regime_data = self.market_regimes[symbol]
                            regime_data["duration_bars"] -= 1
                            
                            if regime_data["duration_bars"] <= 0:
                                self._switch_market_regime(symbol)
                
                last_time = current_time
            
            # Sleep to avoid consuming too much CPU
            time.sleep(0.1)
    
    def _close_bars(self, timeframe: str, timestamp: datetime):
        """Close bars for the specified timeframe."""
        for symbol in self.symbols:
            # Generate OHLC data
            current_price = prices = {}
            
            # Simple simulation of OHLC
            price = prices.get(symbol, 1.0)
            vol_factor = 0.002  # volatility factor
            
            # Adjust based on regime
            regime = self.market_regimes[symbol]["current_regime"]
            if regime == "volatile":
                vol_factor *= 3
            elif regime == "trending":
                vol_factor *= 1.5
            
            # Generate OHLC
            open_price = price * (1 + random.normalvariate(0, vol_factor * 0.2))
            high_price = open_price * (1 + abs(random.normalvariate(0, vol_factor)))
            low_price = open_price * (1 - abs(random.normalvariate(0, vol_factor)))
            close_price = price * (1 + random.normalvariate(0, vol_factor * 0.2))
            
            # Ensure high >= open, close and low <= open, close
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            volume = random.randint(10000, 100000)
            
            # Create bar data
            bar_data = {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": timestamp,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume
            }
            
            # Add to data storage
            with self.lock:
                new_row = pd.DataFrame([{
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume
                }])
                
                self.data[symbol][timeframe] = pd.concat([
                    self.data[symbol][timeframe], new_row
                ]).reset_index(drop=True)
                
                # Keep only the most recent 1000 bars
                if len(self.data[symbol][timeframe]) > 1000:
                    self.data[symbol][timeframe] = self.data[symbol][timeframe].iloc[-1000:]
            
            # Publish bar_closed event
            self.event_bus.create_and_publish(
                event_type=EventType.BAR_CLOSED,
                data=bar_data,
                source="live_data_source"
            )
            
            # Store in persistence if available
            if self.persistence:
                collection_name = f"ohlc_{symbol.replace('/', '_')}_{timeframe}"
                self.persistence.insert_document(collection_name, bar_data)
    
    def _switch_market_regime(self, symbol: str):
        """Switch market regime for a symbol."""
        current_regime = self.market_regimes[symbol]["current_regime"]
        
        # Choose a new regime, different from the current one
        regimes = ["trending", "ranging", "volatile", "low_volatility"]
        regimes.remove(current_regime)
        new_regime = random.choice(regimes)
        
        # Set new regime data
        self.market_regimes[symbol] = {
            "current_regime": new_regime,
            "confidence": random.uniform(0.65, 0.95),
            "duration_bars": random.randint(20, 100),
            "previous_regime": current_regime
        }
        
        # Publish market regime change event
        self.event_bus.create_and_publish(
            event_type=EventType.MARKET_REGIME_CHANGED,
            data={
                "symbol": symbol,
                "current_regime": new_regime,
                "confidence": self.market_regimes[symbol]["confidence"],
                "previous_regime": current_regime,
                "timestamp": datetime.now(),
                "trigger": random.choice([
                    "volatility_spike", "trend_reversal", 
                    "consolidation", "breakout"
                ])
            },
            source="live_data_source"
        )
        
        logger.info(f"Switched market regime for {symbol}: {current_regime} -> {new_regime}")
    
    def get_current_market_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Get the current market data for a symbol and timeframe.
        
        Args:
            symbol: Symbol to get data for
            timeframe: Timeframe to get data for
            
        Returns:
            DataFrame containing market data or None if not available
        """
        with self.lock:
            if symbol in self.data and timeframe in self.data[symbol]:
                return self.data[symbol][timeframe].copy()
        return None
    
    def publish_data_event(self, event_type: EventType, data: Dict[str, Any]):
        """
        Publish a data event to the event bus.
        
        Args:
            event_type: Type of event to publish
            data: Event data
        """
        if not self.event_bus:
            logger.warning("Cannot publish event: Event bus not available")
            return
        
        self.event_bus.create_and_publish(
            event_type=event_type,
            data=data,
            source="live_data_source"
        )
    
    def get_current_market_regime(self, symbol: str) -> Dict[str, Any]:
        """
        Get the current market regime for a symbol.
        
        Args:
            symbol: Symbol to get regime for
            
        Returns:
            Dictionary containing regime information
        """
        if symbol in self.market_regimes:
            return self.market_regimes[symbol].copy()
        return {
            "current_regime": "unknown",
            "confidence": 0.0,
            "duration_bars": 0
        }
