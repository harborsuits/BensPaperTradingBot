import json
import time
import logging
import os
from typing import Dict, List, Any, Optional, Tuple, Callable
from .interfaces import MarketContext, MarketRegimeClassifier
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

class DynamicMarketContext(MarketContext):
    """
    Implementation of MarketContext that manages market state data
    with timestamps and supports complex queries
    """
    
    def __init__(self, regime_classifier: MarketRegimeClassifier = None):
        self._data = {}  # Main data store
        self._timestamps = {}  # When values were last updated
        self._regime_classifier = regime_classifier
        self._cached_regime = "unknown"
        self._cached_regime_timestamp = 0
        self._regime_cache_ttl = 300  # 5 minutes
        
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a value from the context by key"""
        return self._data.get(key, default)
        
    def set_value(self, key: str, value: Any) -> None:
        """Set a value in the context by key"""
        self._data[key] = value
        self._timestamps[key] = time.time()
        
        # Reset regime cache if market data changes
        if key.startswith('market_'):
            self._cached_regime_timestamp = 0
        
    def get_all_values(self) -> Dict[str, Any]:
        """Get all key-value pairs in the context"""
        return self._data.copy()
        
    def get_last_updated(self, key: str) -> Optional[float]:
        """Get the timestamp when a value was last updated"""
        return self._timestamps.get(key)
        
    def get_regime(self) -> str:
        """
        Get the current market regime, with caching for performance
        """
        current_time = time.time()
        
        # If we have a cached regime that's still valid, return it
        if (current_time - self._cached_regime_timestamp) < self._regime_cache_ttl:
            return self._cached_regime
            
        # Need to recalculate regime
        if self._regime_classifier:
            try:
                market_data = self.get_market_state()
                self._cached_regime = self._regime_classifier.classify_regime(market_data)
                self._cached_regime_timestamp = current_time
                return self._cached_regime
            except Exception as e:
                logger.error(f"Error classifying market regime: {str(e)}")
                # Fall back to the last known regime
                return self._cached_regime
        else:
            logger.warning("No regime classifier set, returning unknown regime")
            return "unknown"
        
    def get_market_state(self) -> Dict[str, Any]:
        """
        Get comprehensive market state data, focused on values
        needed for regime classification and strategy selection
        """
        result = {}
        
        # Get all market data
        for key, value in self._data.items():
            if key.startswith('market_') or key.startswith('indicator_'):
                result[key] = value
                
        # Add metadata
        result['_context_time'] = time.time()
        result['_last_regime'] = self._cached_regime
        
        return result
        
    def export_data(self) -> Dict[str, Any]:
        """Export all context data for serialization"""
        export_data = {
            'data': {k: v for k, v in self._data.items() if isinstance(v, (str, int, float, bool, list, dict))},
            'timestamps': self._timestamps,
            'cached_regime': self._cached_regime,
            'cached_regime_timestamp': self._cached_regime_timestamp,
            'export_time': time.time()
        }
        return export_data
        
    def import_data(self, data: Dict[str, Any]) -> bool:
        """Import data from a serialized format"""
        try:
            self._data.update(data.get('data', {}))
            self._timestamps.update(data.get('timestamps', {}))
            self._cached_regime = data.get('cached_regime', 'unknown')
            self._cached_regime_timestamp = data.get('cached_regime_timestamp', 0)
            return True
        except Exception as e:
            logger.error(f"Error importing context data: {str(e)}")
            return False
            
    def save_to_file(self, filepath: str) -> bool:
        """Save context to a file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(self.export_data(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving context to file: {str(e)}")
            return False
            
    def load_from_file(self, filepath: str) -> bool:
        """Load context from a file"""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Context file not found: {filepath}")
                return False
                
            with open(filepath, 'r') as f:
                data = json.load(f)
            return self.import_data(data)
        except Exception as e:
            logger.error(f"Error loading context from file: {str(e)}")
            return False
            
    def set_regime_classifier(self, classifier: MarketRegimeClassifier) -> None:
        """Set the regime classifier instance"""
        self._regime_classifier = classifier
        # Reset cache to force recalculation with new classifier
        self._cached_regime_timestamp = 0
        
    def flush_cache(self) -> None:
        """Force recalculation of derived values"""
        self._cached_regime_timestamp = 0
        
    def get_historical_value(self, key: str, lookback_days: int = 30) -> List[Tuple[float, Any]]:
        """
        Get historical values for a key if available
        Returns list of (timestamp, value) tuples
        """
        historical_key = f"history_{key}"
        history = self.get_value(historical_key, [])
        
        # Filter by lookback period if needed
        if lookback_days > 0:
            cutoff_time = time.time() - (lookback_days * 86400)
            history = [(ts, val) for ts, val in history if ts >= cutoff_time]
            
        return history 

class ContextDecisionIntegration(MarketContextProvider):
    """
    Provides market context data for strategy decisions by integrating
    multiple data sources including technical indicators, sentiment analysis,
    news events, and market regime information.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the context integration system
        
        Args:
            config_path: Path to configuration file
        """
        # Data sources and their weights
        self.data_sources = {}
        self.data_weights = {}
        
        # Cache for context data
        self.context_cache = {}
        self.cache_expiry = {}
        self.default_ttl = timedelta(minutes=15)
        
        # Historical context data
        self.context_history = []
        self.max_history_items = 100
        
        # Market event handlers
        self.event_handlers = {}
        
        # Load configuration if provided
        if config_path:
            self._load_config(config_path)
            
    def get_market_context(self, include_raw_data: bool = False) -> Dict[str, Any]:
        """
        Get the current integrated market context
        
        Args:
            include_raw_data: Whether to include raw data from sources
            
        Returns:
            context: Integrated market context data
        """
        # Start with empty context
        context = {
            "timestamp": datetime.now().isoformat(),
            "technical": {},
            "sentiment": {},
            "macro": {},
            "volatility": 0.0,
            "trend": 0.0,
            "regime_indicators": {},
        }
        
        raw_data = {}
        
        # Collect data from all registered sources
        for source_id, source_info in self.data_sources.items():
            # Skip disabled sources
            if not source_info.get("enabled", True):
                continue
                
            # Check if we have cached data
            if (source_id in self.context_cache and 
                datetime.now() < self.cache_expiry.get(source_id, datetime.min)):
                source_data = self.context_cache[source_id]
                logger.debug(f"Using cached data for {source_id}")
            else:
                # Fetch fresh data from source
                try:
                    fetch_function = source_info.get("fetch_function")
                    if fetch_function and callable(fetch_function):
                        source_data = fetch_function()
                        
                        # Update cache
                        self.context_cache[source_id] = source_data
                        ttl = source_info.get("ttl", self.default_ttl)
                        self.cache_expiry[source_id] = datetime.now() + ttl
                        
                        logger.debug(f"Fetched fresh data for {source_id}")
                    else:
                        logger.warning(f"No fetch function for source {source_id}")
                        continue
                except Exception as e:
                    logger.error(f"Error fetching data from {source_id}: {str(e)}")
                    # Use cached data if available, otherwise skip
                    if source_id in self.context_cache:
                        source_data = self.context_cache[source_id]
                        logger.warning(f"Using stale cached data for {source_id}")
                    else:
                        continue
            
            # Store raw data if requested
            if include_raw_data:
                raw_data[source_id] = source_data
                
            # Integrate data into context based on source type
            source_type = source_info.get("type", "unknown")
            self._integrate_source_data(context, source_id, source_type, source_data)
            
        # Calculate aggregate metrics
        self._calculate_aggregate_metrics(context)
        
        # Add raw data if requested
        if include_raw_data:
            context["raw_data"] = raw_data
            
        # Add to history
        self._add_to_history(context)
        
        return context
        
    def get_historical_context(self, 
                              lookback_periods: int = 10, 
                              interval: str = "hour") -> List[Dict[str, Any]]:
        """
        Get historical market context data
        
        Args:
            lookback_periods: Number of periods to look back
            interval: Time interval ('minute', 'hour', 'day')
            
        Returns:
            history: List of historical context data points
        """
        # If we have enough history items, use them
        if interval == "hour" and len(self.context_history) >= lookback_periods:
            return self.context_history[-lookback_periods:]
            
        # Otherwise, we need to generate historical context
        # This would typically come from a database or time-series storage
        # For now, return the available history
        return self.context_history
        
    def register_data_source(self, 
                           source_id: str,
                           source_type: str,
                           fetch_function: Callable[[], Dict[str, Any]],
                           weight: float = 1.0,
                           ttl: Optional[timedelta] = None) -> bool:
        """
        Register a new data source
        
        Args:
            source_id: Unique identifier for the source
            source_type: Type of data ('technical', 'sentiment', 'news', 'macro')
            fetch_function: Function to fetch data from the source
            weight: Weight of this source in aggregation (0.0-1.0)
            ttl: Time-to-live for cached data
            
        Returns:
            success: Whether registration was successful
        """
        if source_id in self.data_sources:
            logger.warning(f"Data source {source_id} already registered. Updating.")
            
        # Validate source type
        valid_types = ["technical", "sentiment", "news", "macro", "volatility", "trend"]
        if source_type not in valid_types:
            logger.error(f"Invalid source type: {source_type}. Must be one of {valid_types}")
            return False
            
        # Validate weight
        if not 0.0 <= weight <= 1.0:
            logger.warning(f"Invalid weight {weight} for source {source_id}. Using 1.0")
            weight = 1.0
            
        # Register the source
        self.data_sources[source_id] = {
            "type": source_type,
            "fetch_function": fetch_function,
            "enabled": True,
            "ttl": ttl or self.default_ttl
        }
        
        # Set weight
        self.data_weights[source_id] = weight
        
        logger.info(f"Registered data source {source_id} of type {source_type}")
        return True
        
    def unregister_data_source(self, source_id: str) -> bool:
        """
        Unregister a data source
        
        Args:
            source_id: Identifier of the source to unregister
            
        Returns:
            success: Whether unregistration was successful
        """
        if source_id not in self.data_sources:
            logger.warning(f"Data source {source_id} not registered")
            return False
            
        # Remove the source
        del self.data_sources[source_id]
        
        # Remove weight
        if source_id in self.data_weights:
            del self.data_weights[source_id]
            
        # Clear cache
        if source_id in self.context_cache:
            del self.context_cache[source_id]
            
        if source_id in self.cache_expiry:
            del self.cache_expiry[source_id]
            
        logger.info(f"Unregistered data source {source_id}")
        return True
        
    def register_event_handler(self, 
                             event_type: str, 
                             handler_function: Callable[[Dict[str, Any]], None]) -> str:
        """
        Register a handler for market events
        
        Args:
            event_type: Type of event to handle
            handler_function: Function to call when event occurs
            
        Returns:
            handler_id: Identifier for the registered handler
        """
        handler_id = f"{event_type}_{len(self.event_handlers) + 1}"
        
        self.event_handlers[handler_id] = {
            "event_type": event_type,
            "handler": handler_function,
            "enabled": True
        }
        
        logger.info(f"Registered event handler {handler_id} for event type {event_type}")
        return handler_id
        
    def unregister_event_handler(self, handler_id: str) -> bool:
        """
        Unregister an event handler
        
        Args:
            handler_id: Identifier of the handler to unregister
            
        Returns:
            success: Whether unregistration was successful
        """
        if handler_id not in self.event_handlers:
            logger.warning(f"Event handler {handler_id} not registered")
            return False
            
        del self.event_handlers[handler_id]
        logger.info(f"Unregistered event handler {handler_id}")
        return True
        
    def trigger_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Trigger a market event
        
        Args:
            event_type: Type of event
            event_data: Data associated with the event
        """
        event_count = 0
        
        # Find handlers for this event type
        for handler_id, handler_info in self.event_handlers.items():
            if handler_info["event_type"] == event_type and handler_info["enabled"]:
                try:
                    handler_info["handler"](event_data)
                    event_count += 1
                except Exception as e:
                    logger.error(f"Error in event handler {handler_id}: {str(e)}")
                    
        logger.debug(f"Triggered {event_type} event, handled by {event_count} handlers")
        
    def save_state(self, file_path: str) -> bool:
        """
        Save current state to a file
        
        Args:
            file_path: Path to save the state
            
        Returns:
            success: Whether the save was successful
        """
        try:
            # Cannot serialize functions, so we need to save a simplified state
            serializable_sources = {}
            for source_id, source_info in self.data_sources.items():
                serializable_sources[source_id] = {
                    "type": source_info["type"],
                    "enabled": source_info["enabled"],
                    "ttl": source_info["ttl"].total_seconds()
                }
                
            # Create state dictionary
            state = {
                "data_sources": serializable_sources,
                "data_weights": self.data_weights,
                "context_history": self.context_history,
                "default_ttl": self.default_ttl.total_seconds()
            }
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Context integration state saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save context integration state: {str(e)}")
            return False
        
    def _integrate_source_data(self, 
                             context: Dict[str, Any], 
                             source_id: str,
                             source_type: str, 
                             source_data: Dict[str, Any]) -> None:
        """
        Integrate data from a source into the context
        
        Args:
            context: Context to update
            source_id: Identifier of the source
            source_type: Type of the source
            source_data: Data from the source
        """
        # Weight for this source
        weight = self.data_weights.get(source_id, 1.0)
        
        if source_type == "technical":
            # Technical indicators go into the technical section
            for key, value in source_data.items():
                if key in context["technical"]:
                    # Weighted average for existing keys
                    existing_value = context["technical"][key]
                    context["technical"][key] = (existing_value + value * weight) / (1 + weight)
                else:
                    context["technical"][key] = value
                    
            # Technical data can contribute to trend and volatility
            if "trend" in source_data:
                context["trend"] += source_data["trend"] * weight
                
            if "volatility" in source_data:
                context["volatility"] += source_data["volatility"] * weight
                
        elif source_type == "sentiment":
            # Sentiment data goes into the sentiment section
            for key, value in source_data.items():
                if key in context["sentiment"]:
                    # Weighted average for existing keys
                    existing_value = context["sentiment"][key]
                    context["sentiment"][key] = (existing_value + value * weight) / (1 + weight)
                else:
                    context["sentiment"][key] = value
                    
        elif source_type == "news":
            # News events can create regime indicators
            if "events" in source_data:
                for event in source_data["events"]:
                    event_type = event.get("type")
                    event_impact = event.get("impact", 0.0)
                    
                    if event_type and event_impact:
                        context["regime_indicators"][event_type] = event_impact
                        
        elif source_type == "macro":
            # Macro indicators go into the macro section
            for key, value in source_data.items():
                if key in context["macro"]:
                    # Weighted average for existing keys
                    existing_value = context["macro"][key]
                    context["macro"][key] = (existing_value + value * weight) / (1 + weight)
                else:
                    context["macro"][key] = value
                    
        elif source_type == "volatility":
            # Direct volatility measure
            context["volatility"] += source_data.get("value", 0.0) * weight
            
        elif source_type == "trend":
            # Direct trend measure
            context["trend"] += source_data.get("value", 0.0) * weight
            
    def _calculate_aggregate_metrics(self, context: Dict[str, Any]) -> None:
        """
        Calculate aggregate metrics from integrated data
        
        Args:
            context: Context to update with aggregate metrics
        """
        # Normalize trend and volatility
        # These are accumulated with weights from various sources
        sources_with_trend = sum(1 for s in self.data_sources.values() 
                               if s.get("type") in ["technical", "trend"] and s.get("enabled", True))
        
        sources_with_volatility = sum(1 for s in self.data_sources.values() 
                                    if s.get("type") in ["technical", "volatility"] and s.get("enabled", True))
        
        if sources_with_trend > 0:
            context["trend"] /= sources_with_trend
            
        if sources_with_volatility > 0:
            context["volatility"] /= sources_with_volatility
            
        # Calculate regime probabilities based on indicators
        if context["regime_indicators"]:
            regime_scores = {
                "bullish": 0.0,
                "bearish": 0.0,
                "neutral": 0.0,
                "volatile": 0.0,
                "trending": 0.0
            }
            
            # Use technical and sentiment data to score regimes
            tech_trend = context["technical"].get("trend_strength", 0.0)
            tech_vol = context["technical"].get("volatility", 0.0)
            
            sentiment_score = context["sentiment"].get("market_sentiment", 0.0)
            
            # Score regimes
            if tech_trend > 0.5:
                regime_scores["trending"] += tech_trend
                if sentiment_score > 0.2:
                    regime_scores["bullish"] += sentiment_score
                    
            if tech_trend < -0.3:
                if sentiment_score < -0.2:
                    regime_scores["bearish"] += abs(sentiment_score)
                    
            if abs(tech_trend) < 0.3:
                regime_scores["neutral"] += 0.5
                
            if tech_vol > 0.6:
                regime_scores["volatile"] += tech_vol
                
            # Add regime scores to context
            context["regime_scores"] = regime_scores
            
        # Add additional derived metrics
        context["market_health"] = self._calculate_market_health(context)
        context["risk_level"] = self._calculate_risk_level(context)
        
    def _calculate_market_health(self, context: Dict[str, Any]) -> float:
        """
        Calculate market health score based on context data
        
        Args:
            context: Current market context
            
        Returns:
            health_score: Market health score (0.0-1.0)
        """
        # Start with neutral health
        health = 0.5
        
        # Technical indicators contribution
        tech = context["technical"]
        if tech:
            # Positive factors
            if "trend_strength" in tech and tech["trend_strength"] > 0:
                health += tech["trend_strength"] * 0.1
                
            if "market_breadth" in tech:
                health += (tech["market_breadth"] - 0.5) * 0.1
                
            # Negative factors
            if "volatility" in tech:
                health -= (tech["volatility"] - 0.5) * 0.1
                
        # Sentiment contribution
        sentiment = context["sentiment"]
        if sentiment:
            if "market_sentiment" in sentiment:
                health += sentiment["market_sentiment"] * 0.1
                
            if "fear_greed_index" in sentiment:
                # Normalize to -0.1 to 0.1 range
                normalized = (sentiment["fear_greed_index"] - 50) / 500
                health += normalized
                
        # Macro factors
        macro = context["macro"]
        if macro:
            if "economic_surprise" in macro:
                health += macro["economic_surprise"] * 0.05
                
            if "liquidity" in macro:
                health += (macro["liquidity"] - 0.5) * 0.05
                
        # Clip to 0-1 range
        return max(0.0, min(1.0, health))
        
    def _calculate_risk_level(self, context: Dict[str, Any]) -> float:
        """
        Calculate current market risk level
        
        Args:
            context: Current market context
            
        Returns:
            risk_level: Risk level (0.0-1.0)
        """
        # Start with moderate risk
        risk = 0.5
        
        # Volatility is a primary risk factor
        risk += context["volatility"] * 0.2
        
        # Negative trend increases risk
        if context["trend"] < 0:
            risk += abs(context["trend"]) * 0.1
            
        # Regime indicators
        regime_scores = context.get("regime_scores", {})
        if regime_scores:
            # Volatile and bearish regimes increase risk
            risk += regime_scores.get("volatile", 0.0) * 0.1
            risk += regime_scores.get("bearish", 0.0) * 0.1
            
            # Bullish and trending regimes decrease risk
            risk -= regime_scores.get("bullish", 0.0) * 0.05
            
        # Sentiment factors
        sentiment = context["sentiment"]
        if sentiment:
            # Extreme sentiment (either direction) increases risk
            market_sentiment = sentiment.get("market_sentiment", 0.0)
            risk += abs(market_sentiment) * 0.05
            
        # Clip to 0-1 range
        return max(0.0, min(1.0, risk))
        
    def _add_to_history(self, context: Dict[str, Any]) -> None:
        """
        Add current context to history
        
        Args:
            context: Current market context
        """
        # Create a simplified version for history
        history_item = {
            "timestamp": context["timestamp"],
            "trend": context["trend"],
            "volatility": context["volatility"],
            "market_health": context.get("market_health", 0.5),
            "risk_level": context.get("risk_level", 0.5),
        }
        
        # Add regime scores if available
        if "regime_scores" in context:
            history_item["regime_scores"] = context["regime_scores"]
            
        # Add to history
        self.context_history.append(history_item)
        
        # Trim history if needed
        if len(self.context_history) > self.max_history_items:
            self.context_history = self.context_history[-self.max_history_items:]
            
    def _load_config(self, config_path: str) -> None:
        """
        Load configuration from a file
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Load default TTL
            if "default_ttl_minutes" in config:
                minutes = config["default_ttl_minutes"]
                self.default_ttl = timedelta(minutes=minutes)
                
            # Load max history items
            if "max_history_items" in config:
                self.max_history_items = config["max_history_items"]
                
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}") 