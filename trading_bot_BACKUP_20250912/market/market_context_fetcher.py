#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MarketContextFetcher - Dedicated component to continuously collect, process, and
analyze market data to determine the current market regime (bullish, bearish,
volatile, or sideways).
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime, timedelta
import threading
import time
import queue

# Import shared components
from trading_bot.common.market_types import MarketRegime, MarketRegimeEvent, MarketData
from trading_bot.common.config_utils import (
    setup_directories, load_config, save_state, load_state
)
from trading_bot.common.data_utils import SyntheticDataGenerator

# Setup logging
logger = logging.getLogger("MarketContextFetcher")

class MarketContextFetcher:
    """
    Dedicated component designed to continuously collect, process, and analyze
    market data to determine the current market regime (bullish, bearish,
    volatile, or sideways).
    
    This component enables:
    1. Adaptive strategy selection based on current market conditions
    2. Dynamic risk management adjustments
    3. Performance optimization by deploying appropriate strategies
    
    Features:
    - Multi-source data collection (price, volume, indicators)
    - Real-time and historical data analysis
    - Multiple analytical methods for regime determination
    - Configurable thresholds and sensitivity
    - Event-based notification of regime changes
    - Integration with strategy rotator and risk management
    """
    
    def __init__(
        self,
        symbols: List[str] = None,
        data_provider: Any = None,
        config_path: str = None,
        update_interval: int = 60,  # seconds
        data_dir: str = None,
        event_listeners: List[Callable[[MarketRegimeEvent], None]] = None,
        debug_mode: bool = False
    ):
        """
        Initialize the market context fetcher.
        
        Args:
            symbols: List of symbols to monitor
            data_provider: Provider for market data
            config_path: Path to configuration file
            update_interval: How often to update market context (seconds)
            data_dir: Directory for data storage
            event_listeners: Callables that will be notified of regime changes
            debug_mode: Enable verbose logging and debug features
        """
        # Setup paths using shared utilities
        self.paths = setup_directories(
            data_dir=data_dir,
            component_name="market_context"
        )
        
        # Override config path if provided
        if config_path:
            self.paths["config_path"] = config_path
        
        # Load configuration
        self.config = load_config(
            self.paths["config_path"], 
            default_config_factory=self._get_default_config
        )
        
        # Set up symbols
        self.symbols = symbols or self.config.get("default_symbols", ["SPY"])
        
        # Set up data provider
        self.data_provider = data_provider
        
        # Set up update interval
        self.update_interval = update_interval
        
        # Set up event listeners
        self.event_listeners = event_listeners or []
        
        # Set up debug mode
        self.debug_mode = debug_mode
        
        # Initialize state
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_history = []
        self.last_update_time = None
        self.confidence_level = 0.0
        
        # Indicator and metric storage
        self.metrics = {}
        
        # Initialize windows for rolling calculations
        self.price_window = {}
        self.volume_window = {}
        self.volatility_window = {}
        
        # Threading and queue for async operation
        self.data_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        # Window sizes for different calculations
        self.short_window = self.config.get("short_window", 20)
        self.medium_window = self.config.get("medium_window", 50)
        self.long_window = self.config.get("long_window", 200)
        self.vol_window = self.config.get("volatility_window", 20)
        
        # Initialize synthetic data generator if needed
        if self.data_provider is None:
            self.data_generator = SyntheticDataGenerator(
                symbols=self.symbols
            )
            self.data_generator.set_default_regime_periods()
        else:
            self.data_generator = None
        
        # Initialize data windows for each symbol
        for symbol in self.symbols:
            self.price_window[symbol] = []
            self.volume_window[symbol] = []
            self.volatility_window[symbol] = []
        
        logger.info(f"MarketContextFetcher initialized with {len(self.symbols)} symbols")
        
        # Load state if available
        self.load_state()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "default_symbols": ["SPY", "QQQ", "IWM"],
            "update_interval": 60,  # seconds
            "short_window": 20,
            "medium_window": 50,
            "long_window": 200,
            "volatility_window": 20,
            "regime_thresholds": {
                "bull": {
                    "trend_strength": 0.05,
                    "volatility_ratio": 1.2
                },
                "bear": {
                    "trend_strength": -0.05,
                    "volatility_ratio": 1.2
                },
                "sideways": {
                    "trend_range": 0.03,
                    "volatility_ratio": 1.0
                },
                "high_vol": {
                    "volatility_ratio": 1.5
                },
                "low_vol": {
                    "volatility_ratio": 0.5
                },
                "crisis": {
                    "volatility_ratio": 2.0,
                    "drawdown": -0.1
                }
            },
            "indicator_weights": {
                "trend": 0.4,
                "volatility": 0.3,
                "momentum": 0.2,
                "volume": 0.1
            },
            "regime_change_threshold": 3,  # consecutive readings to confirm change
            "data_sources": ["price", "volume", "technical_indicators"],
            "logging": {
                "level": "INFO",
                "file": "market_context.log"
            }
        }
    
    def start(self) -> None:
        """Start the market context fetcher."""
        if self.running:
            logger.warning("MarketContextFetcher is already running")
            return
        
        self.running = True
        
        # Start worker thread for processing
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        logger.info("MarketContextFetcher started")
    
    def stop(self) -> None:
        """Stop the market context fetcher."""
        if not self.running:
            logger.warning("MarketContextFetcher is not running")
            return
        
        self.running = False
        
        # Wait for worker thread to terminate
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
            self.worker_thread = None
        
        # Save state before stopping
        self.save_state()
        
        logger.info("MarketContextFetcher stopped")
    
    def _worker_loop(self) -> None:
        """Main worker loop for periodic updates."""
        last_fetch_time = datetime.now() - timedelta(seconds=self.update_interval)
        
        while self.running:
            current_time = datetime.now()
            
            # Check if it's time to update
            if (current_time - last_fetch_time).total_seconds() >= self.update_interval:
                try:
                    # Fetch and process new data
                    self._fetch_and_process_data()
                    
                    # Update last fetch time
                    last_fetch_time = current_time
                except Exception as e:
                    logger.error(f"Error in worker loop: {str(e)}")
            
            # Process data from queue
            self._process_queue_data()
            
            # Sleep for a short time to prevent CPU overuse
            time.sleep(1.0)
    
    def _fetch_and_process_data(self) -> None:
        """Fetch new market data and process it."""
        if self.data_provider is None:
            # Generate sample data using the generator
            self._generate_data_from_generator()
            return
            
        # Fetch data from provider for each symbol
        for symbol in self.symbols:
            try:
                # Get latest data (implementation depends on provider)
                data = self.data_provider.get_latest_data(symbol)
                
                # Add to processing queue
                self.data_queue.put((symbol, data))
                
                if self.debug_mode:
                    logger.debug(f"Fetched data for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
    
    def _generate_data_from_generator(self) -> None:
        """Generate data using the synthetic data generator."""
        for symbol in self.symbols:
            try:
                # Generate market data
                if isinstance(self.data_generator, SyntheticDataGenerator):
                    market_data = self.data_generator.generate_market_data(symbol)
                    
                    # Convert to dictionary format expected by processing
                    data = market_data.to_dict()
                    
                    # Add to processing queue
                    self.data_queue.put((symbol, data))
                    
                    if self.debug_mode:
                        logger.debug(f"Generated data for {symbol}: {data}")
            except Exception as e:
                logger.error(f"Error generating data for {symbol}: {str(e)}")
    
    def _process_queue_data(self) -> None:
        """Process data from the queue."""
        processed_count = 0
        
        while not self.data_queue.empty() and processed_count < 100:  # Limit batch size
            try:
                # Get data from queue
                symbol, data = self.data_queue.get(block=False)
                
                # Process the data
                self._process_symbol_data(symbol, data)
                
                # Mark as done
                self.data_queue.task_done()
                processed_count += 1
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing queue data: {str(e)}")
        
        # Update market regime if data was processed
        if processed_count > 0:
            self._update_market_regime()
    
    def _process_symbol_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Process data for a single symbol."""
        # Extract data fields (adjust based on your data format)
        price = data.get("price", None)
        volume = data.get("volume", None)
        
        if price is not None:
            # Add to price window
            self.price_window[symbol].append(price)
            
            # Limit window size
            if len(self.price_window[symbol]) > self.long_window:
                self.price_window[symbol] = self.price_window[symbol][-self.long_window:]
        
        if volume is not None:
            # Add to volume window
            self.volume_window[symbol].append(volume)
            
            # Limit window size
            if len(self.volume_window[symbol]) > self.long_window:
                self.volume_window[symbol] = self.volume_window[symbol][-self.long_window:]
        
        # Update volatility if we have enough price data
        if len(self.price_window[symbol]) >= 2:
            # Calculate return
            returns = np.diff(self.price_window[symbol]) / self.price_window[symbol][:-1]
            
            # Add latest return to volatility window
            if returns.size > 0:
                self.volatility_window[symbol].append(returns[-1])
                
                # Limit window size
                if len(self.volatility_window[symbol]) > self.long_window:
                    self.volatility_window[symbol] = self.volatility_window[symbol][-self.long_window:]
    
    def _update_market_regime(self) -> None:
        """Update the current market regime based on processed data."""
        # Check if we have enough data for all symbols
        if not all(len(self.price_window[symbol]) > self.medium_window for symbol in self.symbols):
            logger.info("Not enough data to determine market regime yet")
            return
        
        # Calculate metrics for regime detection
        metrics = self._calculate_regime_metrics()
        
        # Determine regime based on metrics
        new_regime, confidence = self._determine_regime(metrics)
        
        # Log regime metrics in debug mode
        if self.debug_mode:
            logger.debug(f"Regime metrics: {metrics}")
            logger.debug(f"Determined regime: {new_regime.name} (confidence: {confidence:.2f})")
        
        # Check if regime changed
        if new_regime != self.current_regime:
            # Create regime change event
            event = MarketRegimeEvent(
                timestamp=datetime.now(),
                previous_regime=self.current_regime,
                new_regime=new_regime,
                confidence=confidence,
                metrics=metrics
            )
            
            # Update current regime
            self.current_regime = new_regime
            self.confidence_level = confidence
            self.last_update_time = datetime.now()
            
            # Add to history
            self.regime_history.append(event.to_dict())
            
            # Notify listeners
            self._notify_listeners(event)
            
            # Log regime change
            logger.info(str(event))
        else:
            # Update confidence if regime didn't change
            self.confidence_level = confidence
            self.last_update_time = datetime.now()
    
    def _calculate_regime_metrics(self) -> Dict[str, float]:
        """Calculate metrics used for regime detection."""
        metrics = {}
        
        # Aggregate data across symbols
        prices_by_symbol = {}
        returns_by_symbol = {}
        volatility_by_symbol = {}
        volume_by_symbol = {}
        
        for symbol in self.symbols:
            # Get price data
            prices = self.price_window[symbol]
            if len(prices) < self.medium_window:
                continue
                
            prices_by_symbol[symbol] = prices
            
            # Calculate returns
            returns = np.diff(prices) / prices[:-1]
            returns_by_symbol[symbol] = returns
            
            # Calculate volatility (std dev of returns)
            if len(returns) >= self.vol_window:
                recent_volatility = np.std(returns[-self.vol_window:]) * np.sqrt(252)  # Annualized
                historical_volatility = np.std(returns) * np.sqrt(252)  # Annualized
                volatility_by_symbol[symbol] = {
                    "recent": recent_volatility,
                    "historical": historical_volatility,
                    "ratio": recent_volatility / historical_volatility if historical_volatility > 0 else 1.0
                }
            
            # Get volume data
            if len(self.volume_window[symbol]) > self.medium_window:
                volumes = self.volume_window[symbol]
                recent_volume = np.mean(volumes[-self.short_window:])
                historical_volume = np.mean(volumes)
                volume_by_symbol[symbol] = {
                    "recent": recent_volume,
                    "historical": historical_volume,
                    "ratio": recent_volume / historical_volume if historical_volume > 0 else 1.0
                }
        
        # Only proceed if we have data for at least one symbol
        if not prices_by_symbol:
            return metrics
        
        # Calculate aggregate trend metrics
        short_returns = []
        medium_returns = []
        
        for symbol, prices in prices_by_symbol.items():
            # Calculate short-term return (e.g., 20-day)
            if len(prices) >= self.short_window:
                short_return = (prices[-1] / prices[-self.short_window] - 1)
                short_returns.append(short_return)
            
            # Calculate medium-term return (e.g., 50-day)
            if len(prices) >= self.medium_window:
                medium_return = (prices[-1] / prices[-self.medium_window] - 1)
                medium_returns.append(medium_return)
        
        # Average the returns across symbols
        metrics["short_term_return"] = np.mean(short_returns) if short_returns else 0.0
        metrics["medium_term_return"] = np.mean(medium_returns) if medium_returns else 0.0
        
        # Calculate trend strength (comparing short vs medium term)
        if short_returns and medium_returns:
            metrics["trend_strength"] = metrics["short_term_return"] - metrics["medium_term_return"]
        
        # Calculate aggregate volatility metrics
        recent_vols = [v["recent"] for v in volatility_by_symbol.values()]
        hist_vols = [v["historical"] for v in volatility_by_symbol.values()]
        vol_ratios = [v["ratio"] for v in volatility_by_symbol.values()]
        
        metrics["recent_volatility"] = np.mean(recent_vols) if recent_vols else 0.0
        metrics["historical_volatility"] = np.mean(hist_vols) if hist_vols else 0.0
        metrics["volatility_ratio"] = np.mean(vol_ratios) if vol_ratios else 1.0
        
        # Calculate maximum drawdown over relevant period
        for symbol, prices in prices_by_symbol.items():
            if len(prices) >= self.medium_window:
                rolling_max = np.maximum.accumulate(prices[-self.medium_window:])
                drawdown = (prices[-self.medium_window:] / rolling_max) - 1.0
                metrics[f"{symbol}_max_drawdown"] = np.min(drawdown)
        
        # Average drawdown across symbols
        symbol_drawdowns = [metrics[f"{symbol}_max_drawdown"] for symbol in prices_by_symbol.keys() 
                           if f"{symbol}_max_drawdown" in metrics]
        metrics["max_drawdown"] = np.mean(symbol_drawdowns) if symbol_drawdowns else 0.0
        
        # Calculate aggregate volume metrics
        if volume_by_symbol:
            vol_ratios = [v["ratio"] for v in volume_by_symbol.values()]
            metrics["volume_ratio"] = np.mean(vol_ratios)
        
        # Calculate momentum indicators (e.g., RSI)
        for symbol, returns in returns_by_symbol.items():
            if len(returns) >= self.short_window:
                recent_returns = returns[-self.short_window:]
                gains = np.sum(recent_returns[recent_returns > 0])
                losses = np.abs(np.sum(recent_returns[recent_returns < 0]))
                
                if losses > 0:
                    rs = gains / losses
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100.0
                
                metrics[f"{symbol}_rsi"] = rsi
        
        # Average RSI across symbols
        symbol_rsis = [metrics[f"{symbol}_rsi"] for symbol in returns_by_symbol.keys() 
                      if f"{symbol}_rsi" in metrics]
        metrics["avg_rsi"] = np.mean(symbol_rsis) if symbol_rsis else 50.0
        
        return metrics
    
    def _determine_regime(self, metrics: Dict[str, float]) -> Tuple[MarketRegime, float]:
        """Determine the market regime based on calculated metrics."""
        # Get threshold values from config
        thresholds = self.config.get("regime_thresholds", {})
        
        # Default values if config values are missing
        bull_trend_threshold = thresholds.get("bull", {}).get("trend_strength", 0.05)
        bear_trend_threshold = thresholds.get("bear", {}).get("trend_strength", -0.05)
        sideways_range = thresholds.get("sideways", {}).get("trend_range", 0.03)
        high_vol_threshold = thresholds.get("high_vol", {}).get("volatility_ratio", 1.5)
        low_vol_threshold = thresholds.get("low_vol", {}).get("volatility_ratio", 0.5)
        crisis_vol_threshold = thresholds.get("crisis", {}).get("volatility_ratio", 2.0)
        crisis_drawdown = thresholds.get("crisis", {}).get("drawdown", -0.1)
        
        # Initialize confidence scores for each regime
        regime_scores = {
            MarketRegime.BULL: 0.0,
            MarketRegime.BEAR: 0.0,
            MarketRegime.SIDEWAYS: 0.0,
            MarketRegime.HIGH_VOL: 0.0,
            MarketRegime.LOW_VOL: 0.0,
            MarketRegime.CRISIS: 0.0,
            MarketRegime.UNKNOWN: 0.0
        }
        
        # Extract key metrics
        trend_strength = metrics.get("trend_strength", 0.0)
        volatility_ratio = metrics.get("volatility_ratio", 1.0)
        max_drawdown = metrics.get("max_drawdown", 0.0)
        avg_rsi = metrics.get("avg_rsi", 50.0)
        
        # Check for crisis conditions first (highest priority)
        if volatility_ratio > crisis_vol_threshold and max_drawdown < crisis_drawdown:
            regime_scores[MarketRegime.CRISIS] = min(1.0, 
                (volatility_ratio / crisis_vol_threshold) * (abs(max_drawdown) / abs(crisis_drawdown)))
        
        # Check for high/low volatility regimes
        if volatility_ratio > high_vol_threshold:
            regime_scores[MarketRegime.HIGH_VOL] = min(1.0, volatility_ratio / high_vol_threshold)
        
        if volatility_ratio < low_vol_threshold:
            regime_scores[MarketRegime.LOW_VOL] = min(1.0, low_vol_threshold / volatility_ratio)
        
        # Check trend direction for normal volatility
        if trend_strength > bull_trend_threshold:
            # Bullish trend
            bull_score = min(1.0, trend_strength / bull_trend_threshold)
            
            # Increase score if RSI confirms trend
            if avg_rsi > 60:
                bull_score *= 1.2
                
            regime_scores[MarketRegime.BULL] = min(1.0, bull_score)
            
        elif trend_strength < bear_trend_threshold:
            # Bearish trend
            bear_score = min(1.0, abs(trend_strength) / abs(bear_trend_threshold))
            
            # Increase score if RSI confirms trend
            if avg_rsi < 40:
                bear_score *= 1.2
                
            regime_scores[MarketRegime.BEAR] = min(1.0, bear_score)
            
        elif abs(trend_strength) < sideways_range:
            # Sideways market
            regime_scores[MarketRegime.SIDEWAYS] = min(1.0, sideways_range / (abs(trend_strength) + 0.001))
        
        # If no strong signals, set unknown
        if max(regime_scores.values()) < 0.3:
            regime_scores[MarketRegime.UNKNOWN] = 0.5
        
        # Find regime with highest score
        top_regime = max(regime_scores.items(), key=lambda x: x[1])
        
        return top_regime[0], top_regime[1]
    
    def _notify_listeners(self, event: MarketRegimeEvent) -> None:
        """Notify all event listeners of a regime change."""
        for listener in self.event_listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Error notifying listener: {str(e)}")
    
    def add_event_listener(self, listener: Callable[[MarketRegimeEvent], None]) -> None:
        """Add an event listener to be notified of regime changes."""
        if listener not in self.event_listeners:
            self.event_listeners.append(listener)
    
    def remove_event_listener(self, listener: Callable[[MarketRegimeEvent], None]) -> None:
        """Remove an event listener."""
        if listener in self.event_listeners:
            self.event_listeners.remove(listener)
    
    def get_current_regime(self) -> Tuple[MarketRegime, float]:
        """
        Get the current market regime and confidence level.
        
        Returns:
            Tuple of (regime, confidence)
        """
        return self.current_regime, self.confidence_level
    
    def get_regime_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of regime changes.
        
        Returns:
            List of regime change events as dictionaries
        """
        return self.regime_history.copy()
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """
        Get the latest metrics used for regime detection.
        
        Returns:
            Dictionary of metrics
        """
        # Calculate fresh metrics if needed
        if not self.metrics or (datetime.now() - self.last_update_time).total_seconds() > 60:
            self.metrics = self._calculate_regime_metrics()
        
        return self.metrics.copy()
    
    def save_state(self) -> None:
        """Save current state to file."""
        state = {
            "current_regime": self.current_regime.name,
            "confidence_level": self.confidence_level,
            "last_update_time": self.last_update_time.isoformat() if self.last_update_time else None,
            "metrics": self.metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        # Use shared utility for saving state
        save_state(self.paths["state_path"], state)
        
        # Save regime history separately
        try:
            with open(self.paths["history_path"], 'w') as f:
                json.dump(self.regime_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving regime history: {str(e)}")
    
    def load_state(self) -> bool:
        """
        Load state from file.
        
        Returns:
            True if state was loaded successfully, False otherwise
        """
        # Use shared utility for loading state
        state = load_state(self.paths["state_path"])
        
        if state is None:
            return False
        
        try:
            self.current_regime = MarketRegime[state.get("current_regime", "UNKNOWN")]
            self.confidence_level = state.get("confidence_level", 0.0)
            
            last_update_time = state.get("last_update_time")
            if last_update_time:
                self.last_update_time = datetime.fromisoformat(last_update_time)
            
            self.metrics = state.get("metrics", {})
            
            # Load regime history
            history_path = self.paths["history_path"]
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.regime_history = json.load(f)
            
            return True
        except Exception as e:
            logger.error(f"Error processing loaded state: {str(e)}")
            return False
    
    def reset(self) -> None:
        """Reset the market context fetcher to initial state."""
        # Stop if running
        was_running = self.running
        if was_running:
            self.stop()
        
        # Reset state
        self.current_regime = MarketRegime.UNKNOWN
        self.confidence_level = 0.0
        self.last_update_time = None
        self.metrics = {}
        self.regime_history = []
        
        # Reset data windows
        for symbol in self.symbols:
            self.price_window[symbol] = []
            self.volume_window[symbol] = []
            self.volatility_window[symbol] = []
        
        # Clear queue
        while not self.data_queue.empty():
            try:
                self.data_queue.get(block=False)
                self.data_queue.task_done()
            except queue.Empty:
                break
        
        logger.info("Market context fetcher reset to initial state")
        
        # Restart if it was running
        if was_running:
            self.start()
    
    def __repr__(self) -> str:
        """String representation"""
        return (f"MarketContextFetcher(symbols={self.symbols}, "
                f"regime={self.current_regime.name}, "
                f"confidence={self.confidence_level:.2f})")


# Simple example usage
if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up a simple example
    def print_regime_change(event):
        print(f"\n*** {event} ***\n")
    
    # Create fetcher instance
    fetcher = MarketContextFetcher(
        symbols=["SPY", "QQQ", "IWM"],
        update_interval=5,  # Update every 5 seconds for demo
        debug_mode=True
    )
    
    # Add listener for regime changes
    fetcher.add_event_listener(print_regime_change)
    
    # Start fetcher
    fetcher.start()
    
    try:
        # Run for a while
        print("Market Context Fetcher running. Press Ctrl+C to stop.")
        while True:
            regime, confidence = fetcher.get_current_regime()
            print(f"Current regime: {regime.name} (Confidence: {confidence:.2f})", end="\r")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        
    # Stop fetcher
    fetcher.stop() 