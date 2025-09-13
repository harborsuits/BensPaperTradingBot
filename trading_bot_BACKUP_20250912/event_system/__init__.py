#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event System - Event-driven architecture for trading system components.

This module provides a robust event-driven architecture for the trading system,
enabling decoupled components, parallel processing, and real-time responsiveness.

Key components:
1. Event Types & Base Events - Core event definitions
2. Event Bus - Central event routing with prioritization
3. Event Manager - High-level coordination of event flow
4. Message Queue - Enhanced queuing with multiple backends
5. Channel System - Real-time pub/sub channels for streaming data
"""

# Core event system
from trading_bot.event_system.event_types import (
    EventType, Event, MarketDataEvent, SignalEvent,
    OrderEvent, RiskEvent, AnalysisEvent
)
from trading_bot.event_system.event_bus import EventBus, EventHandler
from trading_bot.event_system.event_manager import EventManager

# Enhanced messaging system
from trading_bot.event_system.message_queue import (
    MessageQueue, Message, QueueType, QueueBackend
)
from trading_bot.event_system.channel_system import (
    Channel, ChannelManager
)

__all__ = [
    # Core event system
    'EventType', 'Event', 'MarketDataEvent', 'SignalEvent',
    'OrderEvent', 'RiskEvent', 'AnalysisEvent',
    'EventBus', 'EventHandler', 'EventManager',
    
    # Enhanced messaging system
    'MessageQueue', 'Message', 'QueueType', 'QueueBackend',
    'Channel', 'ChannelManager'
]
