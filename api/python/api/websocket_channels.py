"""
WebSocket Channels Configuration for BenBot Trading API

This module defines and configures the WebSocket channels used for real-time updates
between the BenBot trading engine and the React frontend.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Set
import logging
import json
from pydantic import BaseModel

logger = logging.getLogger("websocket_channels")

class ChannelType(str, Enum):
    """Enum for WebSocket channel types"""
    CONTEXT = "context"
    STRATEGY = "strategy"
    TRADING = "trading"
    PORTFOLIO = "portfolio"
    LOGGING = "logging"
    EVOTESTER = "evotester"
    SYSTEM = "system"

class MessageType(str, Enum):
    """Enum for WebSocket message types"""
    # Context messages
    MARKET_REGIME_UPDATE = "market_regime_update"
    SENTIMENT_UPDATE = "sentiment_update"
    FEATURE_UPDATE = "feature_update"
    ANOMALY_DETECTED = "anomaly_detected"
    NEWS_UPDATE = "news_update"
    PREDICTION_UPDATE = "prediction_update"
    
    # Strategy messages
    STRATEGY_UPDATE = "strategy_update"
    STRATEGY_RANKING_UPDATE = "strategy_ranking_update"
    ACTIVE_STRATEGY_UPDATE = "active_strategy_update"
    STRATEGY_SIGNAL = "strategy_signal"
    STRATEGY_INSIGHT = "strategy_insight"
    
    # Trading messages
    TRADE_EXECUTED = "trade_executed"
    ORDER_UPDATE = "order_update"
    DECISION_MADE = "decision_made"
    
    # Portfolio messages
    POSITION_UPDATE = "position_update"
    PORTFOLIO_UPDATE = "portfolio_update"
    PERFORMANCE_UPDATE = "performance_update"
    
    # Logging messages
    LOG_ENTRY = "log_entry"
    ALERT = "alert"
    SYSTEM_STATUS_UPDATE = "system_status_update"
    CRITICAL_EVENT = "critical_event"
    
    # EvoTester messages
    EVOTESTER_PROGRESS = "evotester_progress"
    EVOTESTER_GENERATION = "evotester_generation"
    EVOTESTER_COMPLETE = "evotester_complete"
    
    # System messages
    CONNECTION_ESTABLISHED = "connection_established"
    SUBSCRIPTION_UPDATE = "subscription_update"
    SUBSCRIPTION_ERROR = "subscription_error"
    ERROR = "error"

# Map channel types to their respective message types
CHANNEL_MESSAGE_TYPES: Dict[ChannelType, List[MessageType]] = {
    ChannelType.CONTEXT: [
        MessageType.MARKET_REGIME_UPDATE,
        MessageType.SENTIMENT_UPDATE,
        MessageType.FEATURE_UPDATE,
        MessageType.ANOMALY_DETECTED,
        MessageType.NEWS_UPDATE,
        MessageType.PREDICTION_UPDATE,
    ],
    ChannelType.STRATEGY: [
        MessageType.STRATEGY_UPDATE,
        MessageType.STRATEGY_RANKING_UPDATE,
        MessageType.ACTIVE_STRATEGY_UPDATE,
        MessageType.STRATEGY_SIGNAL,
        MessageType.STRATEGY_INSIGHT,
    ],
    ChannelType.TRADING: [
        MessageType.TRADE_EXECUTED,
        MessageType.ORDER_UPDATE,
        MessageType.DECISION_MADE,
    ],
    ChannelType.PORTFOLIO: [
        MessageType.POSITION_UPDATE,
        MessageType.PORTFOLIO_UPDATE,
        MessageType.PERFORMANCE_UPDATE,
    ],
    ChannelType.LOGGING: [
        MessageType.LOG_ENTRY,
        MessageType.ALERT,
        MessageType.SYSTEM_STATUS_UPDATE,
        MessageType.CRITICAL_EVENT,
    ],
    ChannelType.EVOTESTER: [
        MessageType.EVOTESTER_PROGRESS,
        MessageType.EVOTESTER_GENERATION,
        MessageType.EVOTESTER_COMPLETE,
    ],
    ChannelType.SYSTEM: [
        MessageType.CONNECTION_ESTABLISHED,
        MessageType.SUBSCRIPTION_UPDATE,
        MessageType.SUBSCRIPTION_ERROR,
        MessageType.ERROR,
    ],
}

# Topic definitions for each channel type (used for more granular subscriptions)
class ContextTopics(str, Enum):
    MARKET_REGIME = "market_regime"
    SENTIMENT = "sentiment"
    FEATURES = "features"
    ANOMALIES = "anomalies"
    NEWS = "news"
    PREDICTIONS = "predictions"
    ALL = "all_context"

class StrategyTopics(str, Enum):
    STRATEGY_UPDATES = "strategy_updates"
    ACTIVE_STRATEGIES = "active_strategies"
    STRATEGY_RANKINGS = "strategy_rankings"
    STRATEGY_INSIGHTS = "strategy_insights"
    ALL = "all_strategy"

class TradingTopics(str, Enum):
    TRADES = "trades"
    ORDERS = "orders"
    DECISIONS = "decisions"
    ALL = "all_trading"

class PortfolioTopics(str, Enum):
    POSITIONS = "positions"
    PERFORMANCE = "performance"
    ALL = "all_portfolio"

class LoggingTopics(str, Enum):
    LOGS = "logs"
    ALERTS = "alerts"
    SYSTEM_STATUS = "system_status"
    CRITICAL_EVENTS = "critical_events"
    ALL = "all_logging"

class EvoTesterTopics(str, Enum):
    PROGRESS = "evotester_progress"
    RESULTS = "evotester_results"
    QUEUE = "evotester_queue"
    ALL = "all_evotester"

# Map channel types to their topics
CHANNEL_TOPICS = {
    ChannelType.CONTEXT: {topic.value for topic in ContextTopics},
    ChannelType.STRATEGY: {topic.value for topic in StrategyTopics},
    ChannelType.TRADING: {topic.value for topic in TradingTopics},
    ChannelType.PORTFOLIO: {topic.value for topic in PortfolioTopics},
    ChannelType.LOGGING: {topic.value for topic in LoggingTopics},
    ChannelType.EVOTESTER: {topic.value for topic in EvoTesterTopics},
    ChannelType.SYSTEM: set(),  # System channel doesn't have topics
}

# Helper functions

def get_channel_for_message(message_type: str) -> Optional[ChannelType]:
    """Get the channel type for a given message type"""
    message_enum = None
    try:
        message_enum = MessageType(message_type)
    except ValueError:
        return None
    
    for channel, message_types in CHANNEL_MESSAGE_TYPES.items():
        if message_enum in message_types:
            return channel
    
    return None

def get_topic_for_message(message_type: str) -> Optional[str]:
    """Get the most specific topic for a given message type"""
    # Map specific message types to topics
    topic_mapping = {
        MessageType.MARKET_REGIME_UPDATE: ContextTopics.MARKET_REGIME.value,
        MessageType.SENTIMENT_UPDATE: ContextTopics.SENTIMENT.value,
        MessageType.FEATURE_UPDATE: ContextTopics.FEATURES.value,
        MessageType.ANOMALY_DETECTED: ContextTopics.ANOMALIES.value,
        MessageType.NEWS_UPDATE: ContextTopics.NEWS.value,
        MessageType.PREDICTION_UPDATE: ContextTopics.PREDICTIONS.value,
        
        MessageType.STRATEGY_UPDATE: StrategyTopics.STRATEGY_UPDATES.value,
        MessageType.ACTIVE_STRATEGY_UPDATE: StrategyTopics.ACTIVE_STRATEGIES.value,
        MessageType.STRATEGY_RANKING_UPDATE: StrategyTopics.STRATEGY_RANKINGS.value,
        MessageType.STRATEGY_INSIGHT: StrategyTopics.STRATEGY_INSIGHTS.value,
        
        MessageType.TRADE_EXECUTED: TradingTopics.TRADES.value,
        MessageType.ORDER_UPDATE: TradingTopics.ORDERS.value,
        MessageType.DECISION_MADE: TradingTopics.DECISIONS.value,
        
        MessageType.POSITION_UPDATE: PortfolioTopics.POSITIONS.value,
        MessageType.PORTFOLIO_UPDATE: PortfolioTopics.POSITIONS.value,
        MessageType.PERFORMANCE_UPDATE: PortfolioTopics.PERFORMANCE.value,
        
        MessageType.LOG_ENTRY: LoggingTopics.LOGS.value,
        MessageType.ALERT: LoggingTopics.ALERTS.value,
        MessageType.SYSTEM_STATUS_UPDATE: LoggingTopics.SYSTEM_STATUS.value,
        MessageType.CRITICAL_EVENT: LoggingTopics.CRITICAL_EVENTS.value,
        
        MessageType.EVOTESTER_PROGRESS: EvoTesterTopics.PROGRESS.value,
        MessageType.EVOTESTER_COMPLETE: EvoTesterTopics.RESULTS.value,
    }
    
    try:
        message_enum = MessageType(message_type)
        return topic_mapping.get(message_enum)
    except ValueError:
        return None

def should_broadcast_to_topic(message_type: str, subscribed_topic: str) -> bool:
    """
    Determine if a message should be broadcast to a specific topic subscription.
    
    Args:
        message_type: The type of message being broadcast
        subscribed_topic: The topic the client is subscribed to
        
    Returns:
        True if the message should be sent to this subscription, False otherwise
    """
    specific_topic = get_topic_for_message(message_type)
    if not specific_topic:
        return False
    
    # If subscribed to the specific topic or the ALL topic for the channel
    channel = get_channel_for_message(message_type)
    if not channel:
        return False
    
    all_topic_value = None
    if channel == ChannelType.CONTEXT:
        all_topic_value = ContextTopics.ALL.value
    elif channel == ChannelType.STRATEGY:
        all_topic_value = StrategyTopics.ALL.value
    elif channel == ChannelType.TRADING:
        all_topic_value = TradingTopics.ALL.value
    elif channel == ChannelType.PORTFOLIO:
        all_topic_value = PortfolioTopics.ALL.value
    elif channel == ChannelType.LOGGING:
        all_topic_value = LoggingTopics.ALL.value
    elif channel == ChannelType.EVOTESTER:
        all_topic_value = EvoTesterTopics.ALL.value
        
    return subscribed_topic == specific_topic or subscribed_topic == all_topic_value
