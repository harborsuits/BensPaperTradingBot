#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MessageQueue - Component to handle back-pressure between real-time data ingestion
and processing, providing buffering capabilities.
"""

import asyncio
import logging
from collections import deque
from typing import Any, Callable, Optional, Deque, Dict
from datetime import datetime

# Optional Redis support
try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Setup logging
logger = logging.getLogger("MessageQueue")

class InMemoryQueue:
    """
    In-memory message queue using a deque for buffering market data.
    """
    
    def __init__(self, maxlen: Optional[int] = None):
        """
        Initialize an in-memory queue.
        
        Args:
            maxlen: Maximum length of queue (None for unlimited)
        """
        self.queue: Deque[Any] = deque(maxlen=maxlen)
        self.size_warning_threshold = 1000
        self.last_warning_time = None
        self.warning_interval = 60  # seconds
        
    def put(self, item: Any) -> bool:
        """
        Add an item to the queue.
        
        Args:
            item: Item to add
            
        Returns:
            bool: True if added, False if queue is full (only happens with maxlen)
        """
        try:
            self.queue.append(item)
            
            # Check size and emit warning if queue is getting large
            current_size = len(self.queue)
            if (current_size > self.size_warning_threshold and 
                    (self.last_warning_time is None or 
                     (datetime.now() - self.last_warning_time).total_seconds() > self.warning_interval)):
                logger.warning(f"Queue size is large: {current_size} items")
                self.last_warning_time = datetime.now()
                
            return True
        except IndexError:
            # This happens if the deque is full (maxlen reached)
            logger.warning("Queue is full, item dropped")
            return False
    
    def get(self) -> Optional[Any]:
        """
        Get an item from the queue.
        
        Returns:
            Item if available, None if queue is empty
        """
        try:
            return self.queue.popleft()
        except IndexError:
            return None
    
    def get_batch(self, max_items: int) -> list:
        """
        Get multiple items from the queue.
        
        Args:
            max_items: Maximum number of items to get
            
        Returns:
            list: List of items (may be empty if queue is empty)
        """
        batch = []
        for _ in range(min(max_items, len(self.queue))):
            batch.append(self.queue.popleft())
        return batch
    
    def size(self) -> int:
        """
        Get current queue size.
        
        Returns:
            int: Number of items in queue
        """
        return len(self.queue)
    
    def clear(self) -> None:
        """Clear all items from the queue."""
        self.queue.clear()


class RedisQueue:
    """
    Redis-backed message queue for distributed systems.
    """
    
    def __init__(
        self, 
        redis_url: str,
        queue_name: str = "market_data_queue",
        max_queue_size: int = 10000
    ):
        """
        Initialize a Redis-backed queue.
        
        Args:
            redis_url: Redis connection URL
            queue_name: Redis key for the queue
            max_queue_size: Maximum size of the queue
        """
        if not REDIS_AVAILABLE:
            raise ImportError("aioredis package is required for RedisQueue")
            
        self.redis_url = redis_url
        self.queue_name = queue_name
        self.max_queue_size = max_queue_size
        self.redis = None
        self.connected = False
    
    async def connect(self):
        """Connect to Redis."""
        try:
            self.redis = await aioredis.create_redis_pool(self.redis_url)
            self.connected = True
            logger.info(f"Connected to Redis queue: {self.queue_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.connected = False
    
    async def put(self, item: Any) -> bool:
        """
        Add an item to the Redis queue.
        
        Args:
            item: Item to add (will be JSON serialized)
            
        Returns:
            bool: True if added, False if queue is full or error
        """
        if not self.connected or not self.redis:
            await self.connect()
            if not self.connected:
                return False
        
        try:
            # Check if we're at max capacity
            queue_size = await self.redis.llen(self.queue_name)
            if queue_size >= self.max_queue_size:
                logger.warning(f"Redis queue {self.queue_name} is full ({queue_size})")
                return False
            
            # Add to queue
            import json
            serialized = json.dumps(item)
            await self.redis.lpush(self.queue_name, serialized)
            return True
        except Exception as e:
            logger.error(f"Error adding to Redis queue: {e}")
            return False
    
    async def get(self) -> Optional[Any]:
        """
        Get an item from the Redis queue.
        
        Returns:
            Item if available, None if queue is empty or error
        """
        if not self.connected or not self.redis:
            await self.connect()
            if not self.connected:
                return None
        
        try:
            result = await self.redis.rpop(self.queue_name)
            if result:
                import json
                return json.loads(result)
            return None
        except Exception as e:
            logger.error(f"Error getting from Redis queue: {e}")
            return None
    
    async def get_batch(self, max_items: int) -> list:
        """
        Get multiple items from the Redis queue.
        
        Args:
            max_items: Maximum number of items to get
            
        Returns:
            list: List of items (may be empty if queue is empty)
        """
        if not self.connected or not self.redis:
            await self.connect()
            if not self.connected:
                return []
        
        batch = []
        try:
            # This isn't atomic, but it's a reasonable approach
            for _ in range(max_items):
                item = await self.get()
                if item is None:
                    break
                batch.append(item)
            return batch
        except Exception as e:
            logger.error(f"Error getting batch from Redis queue: {e}")
            return batch  # Return whatever we got before the error
    
    async def size(self) -> int:
        """
        Get current queue size.
        
        Returns:
            int: Number of items in queue
        """
        if not self.connected or not self.redis:
            await self.connect()
            if not self.connected:
                return 0
        
        try:
            return await self.redis.llen(self.queue_name)
        except Exception as e:
            logger.error(f"Error getting Redis queue size: {e}")
            return 0
    
    async def clear(self) -> None:
        """Clear all items from the Redis queue."""
        if not self.connected or not self.redis:
            await self.connect()
            if not self.connected:
                return
        
        try:
            await self.redis.delete(self.queue_name)
        except Exception as e:
            logger.error(f"Error clearing Redis queue: {e}")
    
    async def close(self) -> None:
        """Close the Redis connection."""
        if self.connected and self.redis:
            self.redis.close()
            await self.redis.wait_closed()
            self.connected = False


class MessageBroker:
    """
    Message broker to handle message routing and back-pressure.
    """
    
    def __init__(
        self, 
        queue_type: str = "memory", 
        redis_url: Optional[str] = None,
        queue_name: str = "market_data_queue",
        max_queue_size: int = 10000
    ):
        """
        Initialize the message broker.
        
        Args:
            queue_type: Type of queue to use ("memory" or "redis")
            redis_url: Redis connection URL (required for "redis" queue)
            queue_name: Queue name for Redis
            max_queue_size: Maximum queue size
        """
        self.queue_type = queue_type
        
        if queue_type == "memory":
            self.queue = InMemoryQueue(maxlen=max_queue_size)
        elif queue_type == "redis":
            if not redis_url:
                raise ValueError("redis_url is required for Redis queue")
            self.queue = RedisQueue(redis_url, queue_name, max_queue_size)
        else:
            raise ValueError(f"Unsupported queue type: {queue_type}")
            
        self.handlers = []
        self.processing = False
        self.stats = {
            "messages_received": 0,
            "messages_processed": 0,
            "messages_dropped": 0,
            "last_message_time": None
        }
    
    async def connect(self):
        """Connect to the message broker (only needed for Redis)."""
        if self.queue_type == "redis":
            await self.queue.connect()
    
    async def publish(self, message: Any) -> bool:
        """
        Publish a message to the queue.
        
        Args:
            message: Message to publish
            
        Returns:
            bool: True if published, False if queue is full or error
        """
        self.stats["messages_received"] += 1
        self.stats["last_message_time"] = datetime.now()
        
        if self.queue_type == "memory":
            result = self.queue.put(message)
        else:  # redis
            result = await self.queue.put(message)
            
        if not result:
            self.stats["messages_dropped"] += 1
            
        return result
    
    def add_handler(self, handler: Callable[[Any], None]) -> None:
        """
        Add a message handler function.
        
        Args:
            handler: Function to call with each message
        """
        self.handlers.append(handler)
    
    async def start_processing(self, interval: float = 0.01, batch_size: int = 10) -> None:
        """
        Start processing messages in the background.
        
        Args:
            interval: Time to wait between processing batches (seconds)
            batch_size: Maximum number of messages to process per batch
        """
        if self.processing:
            return
            
        self.processing = True
        asyncio.create_task(self._process_loop(interval, batch_size))
    
    async def _process_loop(self, interval: float, batch_size: int) -> None:
        """
        Background loop to process messages.
        
        Args:
            interval: Sleep interval between batches
            batch_size: Maximum messages per batch
        """
        logger.info(f"Starting message processing loop (interval={interval}s, batch_size={batch_size})")
        
        while self.processing:
            try:
                # Get batch of messages
                if self.queue_type == "memory":
                    messages = self.queue.get_batch(batch_size)
                else:  # redis
                    messages = await self.queue.get_batch(batch_size)
                
                # Process messages
                if messages:
                    for message in messages:
                        for handler in self.handlers:
                            await self._call_handler(handler, message)
                        self.stats["messages_processed"] += 1
                
                # Sleep to prevent CPU spinning
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in message processing loop: {e}")
                await asyncio.sleep(interval * 10)  # Longer sleep on error
    
    async def _call_handler(self, handler, message):
        """
        Call a handler with error handling.
        
        Args:
            handler: Handler function
            message: Message to process
        """
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(message)
            else:
                handler(message)
        except Exception as e:
            logger.error(f"Error in message handler: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get broker statistics.
        
        Returns:
            Dict with statistics
        """
        stats = self.stats.copy()
        
        # Add queue size
        if self.queue_type == "memory":
            stats["queue_size"] = self.queue.size()
        else:  # redis
            stats["queue_size"] = await self.queue.size()
            
        return stats
    
    def stop(self) -> None:
        """Stop message processing."""
        self.processing = False
        logger.info("Message processing stopped") 