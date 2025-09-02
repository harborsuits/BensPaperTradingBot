#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Event system API endpoints for the trading dashboard.

This module provides REST API endpoints that expose the event system state,
including events, message queues, and channels for frontend visualization.
"""

from flask import Blueprint, jsonify, request
from flask_cors import CORS
import datetime
import uuid
import json

# Import the event system components
from trading_bot.event_system import (
    EventBus, EventManager, EventType, Event,
    MessageQueue, ChannelManager
)

# Create a blueprint for event system endpoints
event_api = Blueprint('event_api', __name__, url_prefix='/api/events')
CORS(event_api)  # Enable CORS for all event endpoints

# Global references to event system components
event_bus = EventBus()
event_manager = EventManager(event_bus)
channel_manager = ChannelManager()

# In-memory storage for recent events for the API
recent_events = []
MAX_STORED_EVENTS = 1000

# Register to receive all events for API exposure
@event_bus.subscribe(event_type=None)  # None = all events
def event_logger(event):
    """Log all events and make them available via API."""
    # Store in recent events list with timestamp
    event_dict = {
        'id': str(uuid.uuid4()),
        'type': event.event_type.name if hasattr(event, 'event_type') else str(event.__class__.__name__),
        'timestamp': datetime.datetime.now().isoformat(),
        'data': event.data if hasattr(event, 'data') else {},
        'source': event.source if hasattr(event, 'source') else 'unknown',
        'priority': event.priority if hasattr(event, 'priority') else 0,
        'processed': True
    }
    
    global recent_events
    recent_events.insert(0, event_dict)
    
    # Keep the list at a reasonable size
    if len(recent_events) > MAX_STORED_EVENTS:
        recent_events = recent_events[:MAX_STORED_EVENTS]
    
    return True  # Continue event processing

# API Endpoints

@event_api.route('/status', methods=['GET'])
def get_event_bus_status():
    """Get the current status of the event bus."""
    # Get event counts by type
    event_types = {}
    for event in recent_events:
        event_type = event['type']
        if event_type in event_types:
            event_types[event_type] += 1
        else:
            event_types[event_type] = 1
    
    # Calculate active, processed and pending events
    active_count = len(recent_events)
    processed_count = sum(1 for e in recent_events if e['processed'])
    pending_count = active_count - processed_count
    
    return jsonify({
        'activeEvents': active_count,
        'processedEvents': processed_count,
        'pendingEvents': pending_count,
        'eventTypes': event_types
    })

@event_api.route('/recent', methods=['GET'])
def get_recent_events():
    """Get the most recent events, optionally filtered by type."""
    event_type = request.args.get('type')
    limit = int(request.args.get('limit', 100))
    
    if event_type and event_type != 'all':
        filtered_events = [e for e in recent_events if e['type'] == event_type]
        return jsonify(filtered_events[:limit])
    else:
        return jsonify(recent_events[:limit])

@event_api.route('/queues', methods=['GET'])
def get_queue_status():
    """Get status information about all message queues."""
    # For demo, create some example queues if none exist
    queue_list = []
    
    # Try to get actual queue information from MessageQueue instances
    try:
        # This would be replaced with actual logic to get queue information
        # from the MessageQueue component
        for name, queue in MessageQueue._instances.items():
            queue_info = {
                'name': name,
                'size': queue.size(),
                'consumerCount': queue.consumer_count(),
                'messageRate': queue.message_rate(),
                'status': queue.status()
            }
            queue_list.append(queue_info)
    except (AttributeError, Exception) as e:
        # If no actual queues are found or accessible, provide sample data
        sample_queues = [
            {
                'name': 'market_data_queue',
                'size': 24,
                'consumerCount': 3,
                'messageRate': 8.5,
                'status': 'active'
            },
            {
                'name': 'signal_queue',
                'size': 7,
                'consumerCount': 2,
                'messageRate': 2.1,
                'status': 'active'
            },
            {
                'name': 'order_queue',
                'size': 3,
                'consumerCount': 1,
                'messageRate': 0.8,
                'status': 'active'
            }
        ]
        queue_list = sample_queues
    
    return jsonify(queue_list)

@event_api.route('/channels', methods=['GET'])
def get_channel_status():
    """Get status information about all event channels."""
    # Try to get actual channel information
    channel_list = []
    
    try:
        # This would be replaced with actual logic to get channel information
        for name, channel in channel_manager.channels.items():
            channel_info = {
                'name': name,
                'subscribers': len(channel.subscribers),
                'messageRate': getattr(channel, 'message_rate', 0),
                'lastMessage': getattr(channel, 'last_message', None)
            }
            channel_list.append(channel_info)
    except (AttributeError, Exception) as e:
        # If no actual channels are found or accessible, provide sample data
        sample_channels = [
            {
                'name': 'market_data_stream',
                'subscribers': 5,
                'messageRate': 12.3,
                'lastMessage': {'symbol': 'AAPL', 'price': 182.63, 'timestamp': '2025-05-07T21:59:12'}
            },
            {
                'name': 'trade_signals',
                'subscribers': 3,
                'messageRate': 0.5,
                'lastMessage': {'strategy': 'mean_reversion', 'symbol': 'QQQ', 'action': 'BUY', 'confidence': 0.87}
            },
            {
                'name': 'order_stream',
                'subscribers': 2,
                'messageRate': 0.2,
                'lastMessage': {'orderId': 'ord-12345', 'symbol': 'SPY', 'quantity': 10, 'side': 'buy', 'status': 'filled'}
            }
        ]
        channel_list = sample_channels
    
    return jsonify(channel_list)

@event_api.route('/publish', methods=['POST'])
def publish_test_event():
    """Publish a test event to the event bus (for testing purposes)."""
    try:
        event_data = request.json
        event_type_name = event_data.get('type', 'TestEvent')
        
        # Create an appropriate event object based on type
        if hasattr(EventType, event_type_name):
            event_type = getattr(EventType, event_type_name)
            test_event = Event(
                event_type=event_type,
                data=event_data.get('data', {}),
                source=event_data.get('source', 'API'),
                priority=event_data.get('priority', 5)
            )
        else:
            # Generic event if type not found
            test_event = Event(
                event_type=EventType.CUSTOM,
                data=event_data.get('data', {}),
                source=event_data.get('source', 'API'),
                priority=event_data.get('priority', 5)
            )
        
        # Publish to event bus
        event_bus.publish(test_event)
        
        return jsonify({
            'success': True,
            'message': f'Published test event of type {event_type_name}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# Method to generate sample events for testing
def generate_sample_events():
    """Generate sample events when the real event system is not producing events."""
    event_types = ['MarketDataEvent', 'SignalEvent', 'OrderEvent', 'RiskEvent', 'AnalysisEvent']
    sources = ['market_data_handler', 'signal_generator', 'order_manager', 'risk_manager', 'analysis_engine']
    
    sample_events = []
    
    for i in range(20):
        event_type = event_types[i % len(event_types)]
        source = sources[i % len(sources)]
        
        # Create sample data based on event type
        if event_type == 'MarketDataEvent':
            data = {
                'symbol': f'{"AAPL" if i % 2 == 0 else "MSFT"}',
                'price': 150 + (i * 0.5),
                'volume': 10000 + (i * 100),
                'timestamp': (datetime.datetime.now() - datetime.timedelta(seconds=i*10)).isoformat()
            }
        elif event_type == 'SignalEvent':
            data = {
                'strategy': 'momentum_strategy',
                'symbol': f'{"SPY" if i % 2 == 0 else "QQQ"}',
                'action': 'BUY' if i % 3 == 0 else 'SELL',
                'confidence': round(0.7 + (i * 0.01), 2),
                'reason': 'price_breakout' if i % 2 == 0 else 'trend_following'
            }
        elif event_type == 'OrderEvent':
            data = {
                'orderId': f'ord-{1000 + i}',
                'symbol': f'{"AMZN" if i % 2 == 0 else "GOOGL"}',
                'quantity': 10 * (i + 1),
                'price': 200 + (i * 2),
                'side': 'buy' if i % 2 == 0 else 'sell',
                'status': 'new' if i % 5 == 0 else 'filled'
            }
        elif event_type == 'RiskEvent':
            data = {
                'type': 'position_limit' if i % 2 == 0 else 'exposure_warning',
                'symbol': f'{"TSLA" if i % 2 == 0 else "NVDA"}',
                'severity': 'high' if i % 3 == 0 else 'medium',
                'message': f'Risk threshold exceeded for {"TSLA" if i % 2 == 0 else "NVDA"}'
            }
        else:  # AnalysisEvent
            data = {
                'metricType': 'performance',
                'value': round(0.8 + (i * 0.02), 2),
                'threshold': 0.9,
                'status': 'normal' if i % 2 == 0 else 'warning'
            }
        
        # Create the sample event
        sample_event = {
            'id': str(uuid.uuid4()),
            'type': event_type,
            'timestamp': (datetime.datetime.now() - datetime.timedelta(seconds=i*30)).isoformat(),
            'data': data,
            'source': source,
            'priority': i % 10,
            'processed': i % 4 != 0  # Some events are still processing
        }
        
        sample_events.append(sample_event)
    
    return sample_events

# Initialize with sample data if needed
if not recent_events:
    recent_events = generate_sample_events()
