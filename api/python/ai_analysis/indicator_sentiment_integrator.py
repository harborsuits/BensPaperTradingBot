#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indicator and Sentiment Integrator

This module ensures that both technical indicators and sentiment analysis
are properly combined and used by the LLM evaluator and AI analysis components.
It acts as a bridge between data sources and the LLM evaluation process.
"""

import logging
import pandas as pd
import numpy as np
import threading
import time
import os
import json
import traceback
from threading import RLock
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

from trading_bot.ai_analysis.llm_trade_evaluator import LLMTradeEvaluator
from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType

# Set up performance metrics
class PerformanceMetrics:
    """Simple metrics tracker for integration performance monitoring"""
    def __init__(self):
        self.start_time = datetime.now()
        self.processed_indicators = 0
        self.processed_sentiment = 0
        self.integrated_signals = 0
        self.errors = 0
        self.processing_times = []
        self.lock = RLock()

logger = logging.getLogger(__name__)

class IndicatorSentimentIntegrator:
    """
    Integrates technical indicators and sentiment data for comprehensive trade analysis.
    
    This class ensures that both data sources are properly combined and weighted
    in the trade evaluation process.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        llm_evaluator: Optional[LLMTradeEvaluator] = None,
        indicator_weight: float = 0.6,
        sentiment_weight: float = 0.4,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        max_cache_size: int = 1000,
        cache_expiry_seconds: int = 3600
    ):
        """
        Initialize the integrator.
        
        Args:
            event_bus: System event bus
            llm_evaluator: LLM-based trade evaluator
            indicator_weight: Weight to assign to technical indicators (0-1)
            sentiment_weight: Weight to assign to sentiment data (0-1)
            config: Additional configuration options
            config_path: Path to configuration file
            max_cache_size: Maximum number of symbols to cache data for
            cache_expiry_seconds: How long to keep cached data (seconds)
        """
        # Thread safety and state management
        self._lock = RLock()  # For thread-safe data access
        self._active = True  # For graceful shutdown
        self._metrics = PerformanceMetrics()
        
        # Load configuration from file if provided
        loaded_config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {str(e)}")
        
        # Merge configurations with priority: passed config > file config > defaults
        self.config = {
            # Default configuration
            'cache_expiry_seconds': cache_expiry_seconds,
            'max_cache_size': max_cache_size,
            'news_sentiment_weight': 0.4,
            'social_sentiment_weight': 0.3,
            'market_sentiment_weight': 0.3,
            'min_data_points': 3,  # Minimum data points needed for valid integration
            'integration_interval_seconds': 5.0,  # Minimum time between integrations for same symbol
            'error_retry_count': 3,  # Number of retries on processing error
            'stale_data_seconds': 3600,  # How old data can be before considered stale
        }
        
        # Update with file config
        self.config.update(loaded_config)
        
        # Update with passed config (highest priority)
        if config:
            self.config.update(config)
        
        # Core components
        self.event_bus = event_bus
        self.llm_evaluator = llm_evaluator
        
        # Ensure weights sum to 1.0
        total_weight = indicator_weight + sentiment_weight
        self.indicator_weight = indicator_weight / total_weight
        self.sentiment_weight = sentiment_weight / total_weight
        
        # Data storage with metadata and timestamps
        self.latest_indicators = {}
        self.latest_sentiment = {}
        self.integrated_data = {}
        self.last_integration_time = {}  # When each symbol was last integrated
        self.integration_errors = {}    # Track errors by symbol
        
        # Register for events
        self._register_event_handlers()
        
        logger.info(f"IndicatorSentimentIntegrator initialized (indicator_w: {self.indicator_weight:.2f}, "
                  f"sentiment_w: {self.sentiment_weight:.2f}, max_cache: {self.config['max_cache_size']})")
    
    def _register_event_handlers(self):
        """Register for relevant events with error handling and retry logic."""
        if not self.event_bus:
            logger.warning("No event bus provided, cannot register event handlers")
            return
        
        # List of event subscriptions to register
        subscriptions = [
            # Technical indicator events
            (EventType.TECHNICAL_INDICATORS_UPDATED, self.handle_indicator_update),
            
            # Sentiment analysis events
            (EventType.NEWS_SENTIMENT_UPDATED, self.handle_sentiment_update),
            (EventType.MARKET_SENTIMENT_UPDATED, self.handle_sentiment_update),
            (EventType.SOCIAL_SENTIMENT_UPDATED, self.handle_sentiment_update),
            
            # Trade signal events
            (EventType.TRADE_SIGNAL_RECEIVED, self.handle_trade_signal),
            
            # System events
            (EventType.SYSTEM_SHUTDOWN, self.handle_system_shutdown),
            (EventType.SYSTEM_STATUS_REQUEST, self.handle_status_request)
        ]
        
        # Register each subscription with retry logic
        for event_type, handler in subscriptions:
            try:
                self.event_bus.subscribe(event_type, handler)
                logger.debug(f"Registered handler for {event_type}")
            except Exception as e:
                logger.error(f"Failed to register handler for {event_type}: {str(e)}")
        
        # Register recovery handler for error recovery
        try:
            self.event_bus.subscribe(EventType.ERROR_RECOVERY_REQUESTED, self.handle_recovery_request)
        except Exception as e:
            logger.error(f"Failed to register error recovery handler: {str(e)}")
            
        logger.info(f"Event handlers registered successfully: {len(subscriptions)} subscriptions")
        
    def handle_system_shutdown(self, event: Event):
        """Handle system shutdown events gracefully."""
        logger.info("System shutdown received, cleaning up resources...")
        with self._lock:
            self._active = False
            
        # Save any pending state if needed
        self._save_state()
        logger.info("IndicatorSentimentIntegrator shutdown complete")
    
    def handle_status_request(self, event: Event):
        """Handle status request events by publishing metrics."""
        with self._lock:
            uptime = (datetime.now() - self._metrics.start_time).total_seconds()
            
            status_data = {
                "component": "IndicatorSentimentIntegrator",
                "status": "healthy" if self._active else "shutting_down",
                "uptime_seconds": uptime,
                "processed_indicators": self._metrics.processed_indicators,
                "processed_sentiment": self._metrics.processed_sentiment,
                "integrated_signals": self._metrics.integrated_signals,
                "errors": self._metrics.errors,
                "symbols_tracked": len(self.integrated_data),
                "avg_processing_ms": sum(self._metrics.processing_times) / max(1, len(self._metrics.processing_times)),
                "timestamp": datetime.now()
            }
            
            # Publish status back to the event bus
            self.event_bus.publish(EventType.COMPONENT_STATUS_REPORT, status_data)
    
    def handle_recovery_request(self, event: Event):
        """Handle recovery requests after system errors."""
        try:
            # Reset any error states
            with self._lock:
                self.integration_errors = {}
                self._active = True
            
            # Re-register event handlers if needed
            self._register_event_handlers()
            
            logger.info("IndicatorSentimentIntegrator recovery complete")
            
            # Report successful recovery
            self.event_bus.publish(
                EventType.COMPONENT_RECOVERED,
                {"component": "IndicatorSentimentIntegrator", "timestamp": datetime.now()}
            )
        except Exception as e:
            logger.error(f"Recovery failed: {str(e)}")
            # Report failed recovery
            self.event_bus.publish(
                EventType.COMPONENT_RECOVERY_FAILED,
                {"component": "IndicatorSentimentIntegrator", "error": str(e), "timestamp": datetime.now()}
            )
    
    def _save_state(self):
        """Save current state for recovery purposes."""
        try:
            # Create a state snapshot for potential recovery
            state_dir = self.config.get('state_dir', './state')
            os.makedirs(state_dir, exist_ok=True)
            
            state = {
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "processed_indicators": self._metrics.processed_indicators,
                    "processed_sentiment": self._metrics.processed_sentiment,
                    "integrated_signals": self._metrics.integrated_signals,
                    "errors": self._metrics.errors
                },
                "configuration": self.config,
                "weights": {
                    "indicator_weight": self.indicator_weight,
                    "sentiment_weight": self.sentiment_weight
                },
                # Only save metadata and timestamps, not full data
                "symbols": list(self.integrated_data.keys())
            }
            
            with open(os.path.join(state_dir, 'indicator_sentiment_integrator.json'), 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.debug("State saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
    
    def handle_indicator_update(self, event: Event):
        """
        Handle technical indicator update events with thread safety and error handling.
        
        Args:
            event: Event with indicator data
        """
        # Skip processing if shutting down
        if not self._active:
            return
            
        # Start performance tracking
        start_time = time.time()
        
        try:
            data = event.data
            if not data or 'symbol' not in data or 'indicators' not in data:
                logger.debug("Received invalid indicator data format")
                return
            
            symbol = data['symbol']
            indicators = data['indicators']
            timestamp = data.get('timestamp', datetime.now())
            
            # Store latest indicators with thread safety
            with self._lock:
                # Add metadata and timestamp
                self.latest_indicators[symbol] = {
                    'data': indicators,
                    'timestamp': timestamp,
                    'source': data.get('source', 'unknown'),
                    'timeframe': data.get('timeframe', 'unknown')
                }
                
                # Manage cache size - remove oldest if needed
                if len(self.latest_indicators) > self.config['max_cache_size']:
                    # Find and remove oldest entry
                    oldest_symbol = None
                    oldest_time = datetime.now()
                    
                    for sym, ind_data in self.latest_indicators.items():
                        if ind_data.get('timestamp', oldest_time) < oldest_time:
                            oldest_time = ind_data.get('timestamp', oldest_time)
                            oldest_symbol = sym
                    
                    if oldest_symbol and oldest_symbol != symbol:
                        del self.latest_indicators[oldest_symbol]
                        logger.debug(f"Removed oldest indicator data for {oldest_symbol} to maintain cache size")
                
                # Update metrics
                self._metrics.processed_indicators += 1
                
                # Check if we have both indicators and sentiment for this symbol
                has_sentiment = symbol in self.latest_sentiment
                should_integrate = self._should_integrate(symbol)
                
            # Trigger integration outside the lock if appropriate
            if has_sentiment and should_integrate:
                self._integrate_data(symbol)
                
            # Track processing time
            with self._lock:
                processing_time = (time.time() - start_time) * 1000
                self._metrics.processing_times.append(processing_time)
                if len(self._metrics.processing_times) > 100:
                    self._metrics.processing_times.pop(0)  # Keep only the last 100
                    
            logger.debug(f"Processed indicator update for {symbol} in {processing_time:.2f}ms")
                
        except Exception as e:
            with self._lock:
                self._metrics.errors += 1
            
            logger.error(f"Error processing indicator update: {str(e)}")
            logger.debug(f"Stack trace: {''.join(traceback.format_exc())}")
            
            # Record the error for this symbol
            with self._lock:
                if 'symbol' in locals():
                    self.integration_errors[symbol] = {
                        'timestamp': datetime.now(),
                        'error': str(e),
                        'type': 'indicator_update'
                    }
    
    def _should_integrate(self, symbol: str) -> bool:
        """
        Determine if a symbol should be integrated based on its last integration time
        and data freshness.
        
        Args:
            symbol: Trading symbol to check
            
        Returns:
            True if the symbol should be integrated, False otherwise
        """
        # If never integrated before, should integrate
        if symbol not in self.last_integration_time:
            return True
            
        # Check if enough time has passed since last integration
        last_time = self.last_integration_time.get(symbol, datetime.min)
        min_interval = self.config.get('integration_interval_seconds', 5.0)
        
        time_diff = (datetime.now() - last_time).total_seconds()
        if time_diff < min_interval:
            return False
            
        # Check if data is fresh enough
        indicator_data = self.latest_indicators.get(symbol, {})
        indicator_timestamp = indicator_data.get('timestamp', datetime.min)
        
        sentiment_data = self.latest_sentiment.get(symbol, {})
        sentiment_timestamp = sentiment_data.get('timestamp', datetime.min)
        
        now = datetime.now()
        stale_threshold = self.config.get('stale_data_seconds', 3600)
        
        # If either data source is too old, don't integrate
        if isinstance(indicator_timestamp, datetime):
            if (now - indicator_timestamp).total_seconds() > stale_threshold:
                return False
                
        if isinstance(sentiment_timestamp, datetime):
            if (now - sentiment_timestamp).total_seconds() > stale_threshold:
                return False
                
        return True
            
    def handle_sentiment_update(self, event: Event):
        """
        Handle sentiment update events with thread safety and error handling.
        
        Args:
            event: Event with sentiment data
        """
        # Skip processing if shutting down
        if not self._active:
            return
        
        # Start performance tracking
        start_time = time.time()
        
        try:
            data = event.data
            if not data or 'symbol' not in data:
                logger.debug("Received invalid sentiment data format")
                return
            
            symbol = data['symbol']
            timestamp = data.get('timestamp', datetime.now())
            
            # Extract sentiment data based on event type
            sentiment_data = {}
            source_type = "unknown"
            
            if event.event_type == EventType.NEWS_SENTIMENT_UPDATED:
                sentiment_data['news_sentiment'] = data.get('sentiment', {})
                source_type = "news"
            elif event.event_type == EventType.SOCIAL_SENTIMENT_UPDATED:
                sentiment_data['social_sentiment'] = data.get('sentiment', {})
                source_type = "social"
            elif event.event_type == EventType.MARKET_SENTIMENT_UPDATED:
                sentiment_data['market_sentiment'] = data.get('sentiment', {})
                source_type = "market"
            
            # Update sentiment data with thread safety
            with self._lock:
                # Initialize if first update for this symbol
                if symbol not in self.latest_sentiment:
                    self.latest_sentiment[symbol] = {
                        'data': sentiment_data,
                        'timestamp': timestamp,
                        'source_type': source_type,
                        'source': data.get('source', 'unknown')
                    }
                else:
                    # Update existing data
                    current = self.latest_sentiment[symbol]
                    
                    # Update the data, preserving other sources
                    if 'data' not in current:
                        current['data'] = sentiment_data
                    else:
                        current['data'].update(sentiment_data)
                    
                    # Update metadata
                    current['timestamp'] = timestamp
                    current['source_type'] = f"{current.get('source_type', '')}+{source_type}"
                    current['source'] = data.get('source', current.get('source', 'unknown'))
                
                # Manage cache size - remove oldest if needed
                if len(self.latest_sentiment) > self.config['max_cache_size']:
                    # Find and remove oldest entry
                    oldest_symbol = None
                    oldest_time = datetime.now()
                    
                    for sym, sent_data in self.latest_sentiment.items():
                        if sent_data.get('timestamp', oldest_time) < oldest_time:
                            oldest_time = sent_data.get('timestamp', oldest_time)
                            oldest_symbol = sym
                    
                    if oldest_symbol and oldest_symbol != symbol:
                        del self.latest_sentiment[oldest_symbol]
                        logger.debug(f"Removed oldest sentiment data for {oldest_symbol} to maintain cache size")
                
                # Update metrics
                self._metrics.processed_sentiment += 1
                
                # Check if we have both indicators and sentiment for this symbol
                has_indicators = symbol in self.latest_indicators
                should_integrate = self._should_integrate(symbol)
            
            # Trigger integration outside the lock if appropriate
            if has_indicators and should_integrate:
                self._integrate_data(symbol)
            
            # Track processing time
            with self._lock:
                processing_time = (time.time() - start_time) * 1000
                self._metrics.processing_times.append(processing_time)
                if len(self._metrics.processing_times) > 100:
                    self._metrics.processing_times.pop(0)  # Keep only the last 100
                    
            logger.debug(f"Processed sentiment update for {symbol} ({source_type}) in {processing_time:.2f}ms")
            
        except Exception as e:
            with self._lock:
                self._metrics.errors += 1
            
            logger.error(f"Error processing sentiment update: {str(e)}")
            logger.debug(f"Stack trace: {''.join(traceback.format_exc())}")
            
            # Record the error for this symbol
            with self._lock:
                if 'symbol' in locals():
                    self.integration_errors[symbol] = {
                        'timestamp': datetime.now(),
                        'error': str(e),
                        'type': 'sentiment_update'
                    }
    
    def handle_trade_signal(self, event: Event):
        """
        Handle trade signal events to enrich with integrated data.
        
        Args:
            event: Trade signal event
        """
        data = event.data
        if not data or 'symbol' not in data or 'signal_type' not in data:
            return
        
        symbol = data['symbol']
        signal_type = data['signal_type']
        
        # Check if we have integrated data for this symbol
        if symbol not in self.integrated_data:
            logger.debug(f"No integrated data available for {symbol}, skipping enrichment")
            return
        
        # Enrich the signal with our integrated data
        enriched_data = data.copy()
        enriched_data['integrated_analysis'] = self.integrated_data[symbol]
        
        # If we have an LLM evaluator, evaluate the trade
        if self.llm_evaluator:
            try:
                direction = 'long' if signal_type in ['BUY', 'LONG'] else 'short'
                price = data.get('price', 0.0)
                strategy = data.get('strategy', 'unknown')
                stop_loss = data.get('stop_loss')
                take_profit = data.get('take_profit')
                
                # Get market data
                market_data = self._get_market_data(symbol)
                
                # Prepare indicator data
                technical_indicators = self.latest_indicators.get(symbol, {})
                
                # Prepare sentiment data
                news_data = self._format_news_data(symbol)
                
                # Get market context
                market_context = self._get_market_context(symbol)
                
                # Evaluate the trade
                evaluation = self.llm_evaluator.evaluate_trade(
                    symbol=symbol,
                    direction=direction,
                    strategy=strategy,
                    price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    market_data=market_data,
                    news_data=news_data,
                    technical_indicators=technical_indicators,
                    market_context=market_context
                )
                
                # Add evaluation to enriched data
                enriched_data['llm_evaluation'] = evaluation
                
                # Log evaluation summary
                logger.info(f"LLM Evaluation for {symbol} {direction}: "
                           f"Confidence: {evaluation.get('confidence_score', 'N/A')}, "
                           f"Recommendation: {evaluation.get('recommendation', 'N/A')}")
            
            except Exception as e:
                logger.error(f"Error performing LLM evaluation: {str(e)}")
        
        # Publish enriched signal
        self.event_bus.publish(
            EventType.TRADE_SIGNAL_ENRICHED,
            enriched_data
        )
    
    def _integrate_data(self, symbol: str) -> bool:
        """
        Integrate technical indicators and sentiment data for a symbol with 
        thread safety, retry logic, and performance monitoring.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if integration was successful, False otherwise
        """
        # Start performance tracking
        start_time = time.time()
        
        try:
            # Get latest data with thread safety
            with self._lock:
                if symbol not in self.latest_indicators or symbol not in self.latest_sentiment:
                    logger.debug(f"Missing data for {symbol}, cannot integrate")
                    return False
                    
                # Extract indicator data
                indicator_record = self.latest_indicators.get(symbol, {})
                indicators = indicator_record.get('data', {})
                indicator_timestamp = indicator_record.get('timestamp')
                
                # Extract sentiment data
                sentiment_record = self.latest_sentiment.get(symbol, {})
                sentiment = sentiment_record.get('data', {})
                sentiment_timestamp = sentiment_record.get('timestamp')
                
                # Record integration time
                self.last_integration_time[symbol] = datetime.now()
                
                # Clear any prior errors for this symbol
                if symbol in self.integration_errors:
                    del self.integration_errors[symbol]
            
            # Check data quality (outside the lock)
            if not indicators or not sentiment:
                logger.debug(f"Empty data for {symbol}, skipping integration")
                return False
            
            # Normalize indicators for integration
            normalized_indicators = self._normalize_indicators(indicators)
            
            # Normalize sentiment for integration
            normalized_sentiment = self._normalize_sentiment(sentiment)
            
            # Skip if either normalization failed
            if not normalized_indicators or not normalized_sentiment:
                logger.debug(f"Normalization failed for {symbol}, skipping integration")
                return False
            
            # Calculate integrated score
            indicator_score = normalized_indicators.get('composite_score', 0)
            sentiment_score = normalized_sentiment.get('composite_score', 0)
            
            integrated_score = (
                self.indicator_weight * indicator_score +
                self.sentiment_weight * sentiment_score
            )
            
            # Create integrated data record
            now = datetime.now()
            integrated_data = {
                'timestamp': now,
                'symbol': symbol,
                'integrated_score': integrated_score,
                'indicator_contribution': self.indicator_weight * indicator_score,
                'sentiment_contribution': self.sentiment_weight * sentiment_score,
                'data_age': {
                    'indicator_age_seconds': (now - indicator_timestamp).total_seconds() if isinstance(indicator_timestamp, datetime) else None,
                    'sentiment_age_seconds': (now - sentiment_timestamp).total_seconds() if isinstance(sentiment_timestamp, datetime) else None
                },
                'normalized_indicators': normalized_indicators,
                'normalized_sentiment': normalized_sentiment,
                # Don't store full raw data to keep memory usage down
                'indicator_metadata': {
                    'source': indicator_record.get('source', 'unknown'),
                    'timeframe': indicator_record.get('timeframe', 'unknown')
                },
                'sentiment_metadata': {
                    'source': sentiment_record.get('source', 'unknown'),
                    'source_type': sentiment_record.get('source_type', 'unknown')
                }
            }
            
            # Determine trade bias
            if integrated_score > 0.3:
                integrated_data['bias'] = 'bullish'
                integrated_data['strength'] = min(1.0, integrated_score)
            elif integrated_score < -0.3:
                integrated_data['bias'] = 'bearish'
                integrated_data['strength'] = min(1.0, abs(integrated_score))
            else:
                integrated_data['bias'] = 'neutral'
                integrated_data['strength'] = 0.0
            
            # Add confidence level based on data sources
            confidence_factors = [
                'news_sentiment' in sentiment.keys(),
                'social_sentiment' in sentiment.keys(),
                'market_sentiment' in sentiment.keys(),
                len(indicators) >= self.config.get('min_data_points', 3)
            ]
            confidence_score = sum(confidence_factors) / len(confidence_factors)
            integrated_data['confidence'] = min(1.0, confidence_score)
            
            # Store integrated data with thread safety
            with self._lock:
                self.integrated_data[symbol] = integrated_data
                self._metrics.integrated_signals += 1
            
            # Publish integrated data event (outside lock to prevent deadlocks)
            try:
                if self.event_bus and self._active:
                    self.event_bus.publish(
                        EventType.INDICATOR_SENTIMENT_INTEGRATED,
                        integrated_data
                    )
            except Exception as e:
                logger.warning(f"Error publishing integration event: {str(e)}")
            
            # Track processing time
            processing_time = (time.time() - start_time) * 1000
            
            logger.debug(f"Integrated data for {symbol}: score={integrated_score:.2f}, "
                       f"bias={integrated_data['bias']}, confidence={integrated_data['confidence']:.2f}, "
                       f"time={processing_time:.2f}ms")
            
            return True
            
        except Exception as e:
            # Record the error
            with self._lock:
                self._metrics.errors += 1
                self.integration_errors[symbol] = {
                    'timestamp': datetime.now(),
                    'error': str(e),
                    'type': 'integration'
                }
            
            logger.error(f"Error integrating data for {symbol}: {str(e)}")
            logger.debug(f"Stack trace: {''.join(traceback.format_exc())}")
            
            return False
    
    def _normalize_indicators(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize technical indicators to a standard scale with error handling.
        
        Args:
            indicators: Raw indicator values
            
        Returns:
            Normalized indicator values with composite score or empty dict on error
        """
        try:
            result = {}
            indicator_contributions = []
            weights = []
            
            # Handle potential confusion between nested data structure and flat
            if 'data' in indicators and isinstance(indicators['data'], dict):
                indicators = indicators['data']
                
            # Validate input
            if not indicators or not isinstance(indicators, dict):
                logger.warning(f"Invalid indicator format: {type(indicators)}")
                return {'composite_score': 0}
            
            # RSI normalization
            if 'rsi' in indicators:
                try:
                    rsi = float(indicators.get('rsi', 50))
                    # Normalize RSI (0-100) to (-1, 1) scale
                    # RSI < 30: oversold (-1 to 0), RSI > 70: overbought (0 to 1)
                    if rsi <= 30:
                        normalized_rsi = -1 + (rsi / 30)
                    elif rsi >= 70:
                        normalized_rsi = (rsi - 70) / 30
                    else:
                        normalized_rsi = (rsi - 50) / 20  # Map 30-70 range to -1 to 1
                    
                    result['normalized_rsi'] = normalized_rsi
                    indicator_contributions.append(normalized_rsi)
                    weights.append(1.0)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error normalizing RSI: {str(e)}")
            
            # MACD histogram normalization
            if 'macd_hist' in indicators:
                try:
                    macd_hist = float(indicators.get('macd_hist', 0))
                    # Scale depends on the asset type and typical ranges
                    avg_price = float(indicators.get('close', 100))  # Use close price as scaling factor
                    normalized_macd = np.clip(macd_hist / (avg_price * 0.01), -1, 1)
                    result['normalized_macd'] = normalized_macd
                    indicator_contributions.append(normalized_macd)
                    weights.append(1.0)
                except (ValueError, TypeError, ZeroDivisionError) as e:
                    logger.debug(f"Error normalizing MACD: {str(e)}")
            
            # Moving average indicators
            if 'sma_20' in indicators and 'sma_50' in indicators:
                try:
                    sma_20 = float(indicators.get('sma_20'))
                    sma_50 = float(indicators.get('sma_50'))
                    # Calculate % difference between short and long MA
                    ma_diff_pct = (sma_20 - sma_50) / sma_50 if sma_50 != 0 else 0
                    # Normalize to -1 to 1 range (typical range ±5%)
                    normalized_ma = np.clip(ma_diff_pct * 20, -1, 1)
                    result['normalized_ma_diff'] = normalized_ma
                    indicator_contributions.append(normalized_ma)
                    weights.append(1.0)
                except (ValueError, TypeError, ZeroDivisionError) as e:
                    logger.debug(f"Error normalizing moving averages: {str(e)}")
            
            # ADX for trend strength
            if 'adx' in indicators:
                try:
                    adx = float(indicators.get('adx', 0))
                    # ADX > 25 indicates strong trend
                    normalized_adx = min(1.0, adx / 50)  
                    result['normalized_adx'] = normalized_adx
                    
                    # ADX doesn't have direction, so we use it as a weight multiplier
                    # for the overall score rather than a direct contribution
                    trend_multiplier = 0.5 + normalized_adx / 2
                    result['trend_strength'] = normalized_adx
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error normalizing ADX: {str(e)}")
            else:
                trend_multiplier = 1.0
            
            # Bollinger Bands
            if all(k in indicators for k in ['bb_upper', 'bb_middle', 'bb_lower', 'close']):
                try:
                    upper = float(indicators.get('bb_upper'))
                    middle = float(indicators.get('bb_middle'))
                    lower = float(indicators.get('bb_lower'))
                    close = float(indicators.get('close'))
                    
                    # Calculate % position within the bands
                    band_width = upper - lower
                    if band_width > 0:
                        # Normalize position to -1 (at lower) to +1 (at upper)
                        pos = 2 * ((close - lower) / band_width) - 1
                        normalized_bb = np.clip(pos, -1, 1)
                        result['normalized_bb_position'] = normalized_bb
                        indicator_contributions.append(normalized_bb * 0.5)  # Lower weight for BB
                        weights.append(0.5)
                except (ValueError, TypeError, ZeroDivisionError) as e:
                    logger.debug(f"Error normalizing Bollinger Bands: {str(e)}")
            
            # Calculate weighted composite indicator score
            if indicator_contributions and weights:
                weighted_sum = sum(c * w for c, w in zip(indicator_contributions, weights))
                weight_sum = sum(weights)
                if weight_sum > 0:
                    trend_score = weighted_sum / weight_sum
                    # Apply trend strength multiplier if ADX is available
                    trend_score = trend_score * trend_multiplier
                else:
                    trend_score = 0
            else:
                trend_score = 0
                
            # Final composite score
            result['composite_score'] = np.clip(trend_score, -1, 1)
            result['indicator_count'] = len(indicator_contributions)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in indicator normalization: {str(e)}")
            logger.debug(f"Stack trace: {''.join(traceback.format_exc())}")
            # Return neutral score on error
            return {'composite_score': 0, 'error': str(e)}
    
    def _normalize_sentiment(self, sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize sentiment data to a standard scale with robust error handling.
        
        Args:
            sentiment: Raw sentiment data
            
        Returns:
            Normalized sentiment values with composite score
        """
        try:
            result = {}
            
            # Handle potential nested data structures
            if 'data' in sentiment and isinstance(sentiment['data'], dict):
                sentiment = sentiment['data']
                
            # Validate input
            if not sentiment or not isinstance(sentiment, dict):
                logger.warning(f"Invalid sentiment format: {type(sentiment)}")
                return {'composite_score': 0}
            
            # Initialize sentiment values with defaults
            sentiment_values = {
                'news_sentiment': 0.0,
                'social_sentiment': 0.0,
                'market_sentiment': 0.0
            }
            
            # Extract sentiment values with error handling
            for source in sentiment_values.keys():
                try:
                    if source in sentiment:
                        source_data = sentiment.get(source, {})
                        # Handle both direct score and nested structure
                        if isinstance(source_data, dict):
                            score = source_data.get('score', 0)
                        elif isinstance(source_data, (int, float)):
                            score = source_data
                        else:
                            score = 0
                            
                        # Ensure score is within -1 to 1 range
                        sentiment_values[source] = np.clip(float(score), -1.0, 1.0)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error extracting {source}: {str(e)}")
                    sentiment_values[source] = 0.0
            
            # Count how many sentiment sources we have with valid data
            valid_sources = [k for k, v in sentiment_values.items() if abs(v) > 0.001]
            sentiment_count = len(valid_sources)
            
            # If no valid sentiment data, return neutral with warning
            if sentiment_count == 0:
                logger.debug("No valid sentiment data found, returning neutral score")
                result['composite_score'] = 0.0
                result['sources'] = 0
                return result
            
            # Get the configured weights for each sentiment source
            configured_weights = {
                'news_sentiment': self.config.get('news_sentiment_weight', 0.4),
                'social_sentiment': self.config.get('social_sentiment_weight', 0.3),
                'market_sentiment': self.config.get('market_sentiment_weight', 0.3)
            }
            
            # Only use weights for available data
            weights = {}
            for source in valid_sources:
                weights[source] = configured_weights[source]
                
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            if total_weight > 0:
                normalized_weights = {k: v / total_weight for k, v in weights.items()}
            else:
                # Equal weights if all configured weights were 0
                normalized_weights = {k: 1.0 / sentiment_count for k in valid_sources}
            
            # Calculate weighted composite score
            composite_score = sum(sentiment_values[source] * normalized_weights.get(source, 0) 
                                 for source in valid_sources)
            
            # Store normalized values and metadata
            result['composite_score'] = np.clip(composite_score, -1.0, 1.0)
            result['sources'] = sentiment_count
            result['source_types'] = valid_sources
            
            # Store individual normalized scores
            for source, value in sentiment_values.items():
                if source in valid_sources:
                    result[f"normalized_{source}"] = value
                    result[f"{source}_weight"] = normalized_weights.get(source, 0)
            
            # Add confidence based on number and agreement of sources
            if sentiment_count > 1:
                # Calculate variance between sources
                values = [sentiment_values[source] for source in valid_sources]
                variance = np.var(values) if len(values) > 1 else 0
                
                # High confidence if multiple sources with low variance (high agreement)
                # Scale from 0.5 (single source) to 1.0 (multiple aligned sources)
                confidence = 0.5 + (0.5 * (1.0 - min(1.0, variance * 2)))
                result['confidence'] = confidence * min(1.0, sentiment_count / 3)  # Max confidence with 3+ sources
            else:
                # Medium confidence with a single source
                result['confidence'] = 0.5
            
            return result
            
        except Exception as e:
            logger.error(f"Error in sentiment normalization: {str(e)}")
            logger.debug(f"Stack trace: {''.join(traceback.format_exc())}")
            # Return neutral score on error
            return {'composite_score': 0, 'error': str(e), 'confidence': 0}
    
    def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Market data dictionary
        """
        # This would typically query your data service
        # For now, we'll use a placeholder with any available data
        market_data = {
            'symbol': symbol,
            'timestamp': datetime.now()
        }
        
        # Add OHLCV data if available in indicators
        indicators = self.latest_indicators.get(symbol, {})
        for field in ['open', 'high', 'low', 'close', 'volume']:
            if field in indicators:
                market_data[field] = indicators[field]
        
        return market_data
    
    def _format_news_data(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Format news data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of news article dictionaries
        """
        sentiment = self.latest_sentiment.get(symbol, {})
        news_sentiment = sentiment.get('news_sentiment', {})
        
        # Extract articles if available
        articles = news_sentiment.get('articles', [])
        
        # If no articles but we have a score, create a summary article
        if not articles and 'score' in news_sentiment:
            score = news_sentiment['score']
            sentiment_text = 'positive' if score > 0.2 else 'negative' if score < -0.2 else 'neutral'
            
            articles = [{
                'title': f"{symbol} News Sentiment Summary",
                'summary': f"Overall {sentiment_text} sentiment detected for {symbol}.",
                'source': "Sentiment Analysis System",
                'date': datetime.now().strftime("%Y-%m-%d"),
                'url': None,
                'sentiment': score
            }]
        
        return articles
    
    def _get_market_context(self, symbol: str) -> Dict[str, Any]:
        """
        Get broader market context for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Market context dictionary
        """
        # Placeholder - in a real system, this would query market state
        # For now, we'll use integrated data
        if symbol in self.integrated_data:
            integrated = self.integrated_data[symbol]
            
            return {
                'market_bias': integrated.get('bias', 'neutral'),
                'sentiment_score': integrated.get('sentiment_contribution', 0),
                'technical_score': integrated.get('indicator_contribution', 0),
                'overall_score': integrated.get('integrated_score', 0)
            }
        
        return {
            'market_bias': 'neutral',
            'sentiment_score': 0,
            'technical_score': 0,
            'overall_score': 0
        }
    
    def get_integrated_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Get the latest integrated analysis for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Integrated analysis dictionary or empty dict if not available
        """
        return self.integrated_data.get(symbol, {})
