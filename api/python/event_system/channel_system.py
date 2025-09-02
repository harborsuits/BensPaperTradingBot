#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Channel System - Implements a publish-subscribe channel pattern for
real-time data streaming between components.

Based on best practices from OctoBot and enterprise messaging systems.
"""

import logging
import threading
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable, Set, TypeVar, Generic
import uuid
from datetime import datetime
import traceback
import json
import weakref

from trading_bot.event_system.message_queue import Message, MessageQueue, QueueType, QueueBackend

# Set up logging
logger = logging.getLogger("ChannelSystem")

# Generic type for channel data
T = TypeVar('T')

class Channel(Generic[T]):
    """
    Implements a publish-subscribe channel for streaming data
    between components with filtering capabilities.
    """
    
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        buffer_size: int = 100,
        allow_replay: bool = False
    ):
        """
        Initialize a channel
        
        Args:
            name: Channel name
            description: Channel description
            filters: Default filters for this channel
            buffer_size: Maximum number of messages to keep for replay
            allow_replay: Whether to support message replay
        """
        self.name = name
        self.description = description or f"Channel for {name} data"
        self.filters = filters or {}
        self.buffer_size = buffer_size
        self.allow_replay = allow_replay
        
        # Subscribers - using weakref to avoid memory leaks
        self.subscribers: Dict[str, List[Callable[[T, Dict[str, Any]], None]]] = {}
        self.subscriber_filters: Dict[str, Dict[str, Any]] = {}
        self.subscriber_stats: Dict[str, Dict[str, Any]] = {}
        
        # Message buffer for replay
        self.message_buffer: List[Tuple[T, Dict[str, Any]]] = []
        
        # Channel statistics
        self.stats = {
            "messages_published": 0,
            "messages_delivered": 0,
            "created": datetime.now(),
            "last_message": None,
            "subscriber_count": 0
        }
        
        logger.info(f"Channel '{name}' initialized")
    
    def publish(self, data: T, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Publish data to the channel
        
        Args:
            data: Data to publish
            metadata: Additional metadata
            
        Returns:
            Number of subscribers that received the message
        """
        metadata = metadata or {}
        metadata["channel"] = self.name
        metadata["timestamp"] = datetime.now().isoformat()
        
        # Add to buffer if replay is enabled
        if self.allow_replay:
            self.message_buffer.append((data, metadata))
            # Trim buffer if needed
            if len(self.message_buffer) > self.buffer_size:
                self.message_buffer = self.message_buffer[-self.buffer_size:]
        
        # Track statistics
        self.stats["messages_published"] += 1
        self.stats["last_message"] = datetime.now()
        
        # Deliver to subscribers
        return self._deliver_to_subscribers(data, metadata)
    
    def subscribe(
        self, 
        callback: Callable[[T, Dict[str, Any]], None], 
        subscriber_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        replay_last: int = 0
    ) -> str:
        """
        Subscribe to channel
        
        Args:
            callback: Function to call with published data
            subscriber_id: Unique subscriber ID (generated if not provided)
            filters: Filters to apply to messages
            replay_last: Number of past messages to replay (0 = none)
            
        Returns:
            Subscriber ID
        """
        subscriber_id = subscriber_id or f"sub_{len(self.subscribers)}"
        
        if subscriber_id not in self.subscribers:
            self.subscribers[subscriber_id] = []
            self.subscriber_filters[subscriber_id] = filters or {}
            self.subscriber_stats[subscriber_id] = {
                "messages_received": 0,
                "last_message": None,
                "subscribed_at": datetime.now()
            }
            
        self.subscribers[subscriber_id].append(callback)
        
        # Update stats
        self.stats["subscriber_count"] = len(self.subscribers)
        
        # Replay past messages if requested
        if replay_last > 0 and self.allow_replay:
            replay_count = min(replay_last, len(self.message_buffer))
            for data, meta in self.message_buffer[-replay_count:]:
                if self._matches_filters(meta, self.subscriber_filters[subscriber_id]):
                    try:
                        callback(data, meta)
                    except Exception as e:
                        logger.error(f"Error in replay callback for subscriber {subscriber_id}: {e}")
        
        logger.debug(f"Added subscriber {subscriber_id} to channel '{self.name}'")
        return subscriber_id
    
    def unsubscribe(self, subscriber_id: str) -> bool:
        """
        Unsubscribe from channel
        
        Args:
            subscriber_id: Subscriber ID
            
        Returns:
            True if unsubscribed successfully
        """
        if subscriber_id in self.subscribers:
            del self.subscribers[subscriber_id]
            del self.subscriber_filters[subscriber_id]
            del self.subscriber_stats[subscriber_id]
            
            # Update stats
            self.stats["subscriber_count"] = len(self.subscribers)
            
            logger.debug(f"Removed subscriber {subscriber_id} from channel '{self.name}'")
            return True
            
        return False
    
    def _deliver_to_subscribers(self, data: T, metadata: Dict[str, Any]) -> int:
        """
        Deliver data to matching subscribers
        
        Args:
            data: Data to deliver
            metadata: Metadata
            
        Returns:
            Number of subscribers that received the message
        """
        delivered = 0
        
        for subscriber_id, callbacks in list(self.subscribers.items()):
            # Check if message matches subscriber filters
            if not self._matches_filters(metadata, self.subscriber_filters[subscriber_id]):
                continue
                
            # Deliver to all callbacks for this subscriber
            for callback in callbacks:
                try:
                    callback(data, metadata)
                    delivered += 1
                    
                    # Update stats
                    self.subscriber_stats[subscriber_id]["messages_received"] += 1
                    self.subscriber_stats[subscriber_id]["last_message"] = datetime.now()
                    
                except Exception as e:
                    logger.error(f"Error in callback for subscriber {subscriber_id}: {e}")
        
        # Update delivery stats
        self.stats["messages_delivered"] += delivered
        
        return delivered
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Check if metadata matches filters
        
        Args:
            metadata: Message metadata
            filters: Subscriber filters
            
        Returns:
            True if matches, False otherwise
        """
        if not filters:
            return True
            
        for key, value in filters.items():
            if key not in metadata:
                return False
                
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
                
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get channel statistics"""
        stats = self.stats.copy()
        
        # Add detailed subscriber stats
        stats["subscribers"] = {}
        for sub_id, sub_stats in self.subscriber_stats.items():
            stats["subscribers"][sub_id] = sub_stats.copy()
            stats["subscribers"][sub_id]["filters"] = self.subscriber_filters[sub_id]
        
        # Add buffer info
        stats["buffer_size"] = self.buffer_size
        stats["buffer_used"] = len(self.message_buffer)
        
        # Add uptime
        stats["uptime_seconds"] = (datetime.now() - stats["created"]).total_seconds()
        
        return stats


class ChannelManager:
    """
    Manages multiple channels and provides a central point for
    channel creation, publishing, and subscription.
    """
    
    def __init__(self):
        """Initialize the channel manager"""
        self.channels: Dict[str, Channel] = {}
        self.queue_channels: Dict[str, MessageQueue] = {}
        self.created_at = datetime.now()
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("ChannelManager initialized")
    
    def create_channel(
        self,
        name: str,
        description: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        buffer_size: int = 100,
        allow_replay: bool = False
    ) -> Channel:
        """
        Create a new channel
        
        Args:
            name: Channel name
            description: Channel description
            filters: Default filters
            buffer_size: Message buffer size
            allow_replay: Allow message replay
            
        Returns:
            Created channel
        """
        with self.lock:
            if name in self.channels:
                logger.warning(f"Channel '{name}' already exists, returning existing channel")
                return self.channels[name]
                
            channel = Channel(
                name=name,
                description=description,
                filters=filters,
                buffer_size=buffer_size,
                allow_replay=allow_replay
            )
            
            self.channels[name] = channel
            
        return channel
    
    def create_queue_channel(
        self,
        name: str,
        queue_type: QueueType = QueueType.TOPIC,
        backend: QueueBackend = QueueBackend.MEMORY,
        max_size: int = 1000,
        worker_threads: int = 2,
        backend_config: Optional[Dict[str, Any]] = None
    ) -> MessageQueue:
        """
        Create a channel backed by a message queue
        
        Args:
            name: Channel name
            queue_type: Queue type
            backend: Queue backend
            max_size: Maximum queue size
            worker_threads: Worker thread count
            backend_config: Backend-specific config
            
        Returns:
            Created queue
        """
        with self.lock:
            if name in self.queue_channels:
                logger.warning(f"Queue channel '{name}' already exists, returning existing queue")
                return self.queue_channels[name]
                
            queue = MessageQueue(
                name=name,
                queue_type=queue_type,
                backend=backend,
                max_size=max_size,
                worker_threads=worker_threads,
                backend_config=backend_config
            )
            
            # Start the queue
            queue.start()
            
            self.queue_channels[name] = queue
            
        return queue
    
    def get_channel(self, name: str) -> Optional[Channel]:
        """
        Get channel by name
        
        Args:
            name: Channel name
            
        Returns:
            Channel or None if not found
        """
        return self.channels.get(name)
    
    def get_queue_channel(self, name: str) -> Optional[MessageQueue]:
        """
        Get queue channel by name
        
        Args:
            name: Queue channel name
            
        Returns:
            Queue channel or None if not found
        """
        return self.queue_channels.get(name)
    
    def publish(
        self, 
        channel_name: str, 
        data: Any, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Publish to channel
        
        Args:
            channel_name: Channel name
            data: Data to publish
            metadata: Additional metadata
            
        Returns:
            Number of subscribers that received the message, or -1 if channel not found
        """
        channel = self.get_channel(channel_name)
        
        if channel:
            return channel.publish(data, metadata)
        
        # Try queue channel
        queue = self.get_queue_channel(channel_name)
        
        if queue:
            message = Message(
                payload=data,
                source="channel_manager",
                destination=channel_name,
                metadata=metadata
            )
            if queue.publish(message):
                return 1  # Assume at least one subscriber
            else:
                return 0
                
        logger.warning(f"Channel '{channel_name}' not found")
        return -1
    
    def subscribe(
        self,
        channel_name: str,
        callback: Callable,
        subscriber_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        replay_last: int = 0,
        create_if_missing: bool = False
    ) -> Optional[str]:
        """
        Subscribe to channel
        
        Args:
            channel_name: Channel name
            callback: Callback function
            subscriber_id: Subscriber ID
            filters: Message filters
            replay_last: Messages to replay
            create_if_missing: Create channel if it doesn't exist
            
        Returns:
            Subscriber ID or None if failed
        """
        channel = self.get_channel(channel_name)
        
        if not channel and create_if_missing:
            channel = self.create_channel(channel_name)
            
        if channel:
            return channel.subscribe(callback, subscriber_id, filters, replay_last)
        
        # Try queue channel
        queue = self.get_queue_channel(channel_name)
        
        if queue:
            return queue.subscribe(callback, subscriber_id)
            
        logger.warning(f"Channel '{channel_name}' not found and create_if_missing=False")
        return None
    
    def unsubscribe(self, channel_name: str, subscriber_id: str) -> bool:
        """
        Unsubscribe from channel
        
        Args:
            channel_name: Channel name
            subscriber_id: Subscriber ID
            
        Returns:
            True if unsubscribed successfully
        """
        channel = self.get_channel(channel_name)
        
        if channel:
            return channel.unsubscribe(subscriber_id)
        
        # Try queue channel
        queue = self.get_queue_channel(channel_name)
        
        if queue:
            return queue.unsubscribe(subscriber_id)
            
        logger.warning(f"Channel '{channel_name}' not found")
        return False
    
    def shutdown(self) -> None:
        """Shutdown all channels and queues"""
        with self.lock:
            # Stop all queue channels
            for name, queue in self.queue_channels.items():
                logger.info(f"Stopping queue channel '{name}'")
                queue.stop()
                
            self.queue_channels.clear()
            self.channels.clear()
            
        logger.info("ChannelManager shutdown complete")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        stats = {
            "channels": len(self.channels),
            "queue_channels": len(self.queue_channels),
            "uptime_seconds": (datetime.now() - self.created_at).total_seconds(),
            "channel_stats": {},
            "queue_stats": {}
        }
        
        # Add channel stats
        for name, channel in self.channels.items():
            stats["channel_stats"][name] = channel.get_stats()
            
        # Add queue stats
        for name, queue in self.queue_channels.items():
            stats["queue_stats"][name] = queue.get_stats()
            
        return stats
