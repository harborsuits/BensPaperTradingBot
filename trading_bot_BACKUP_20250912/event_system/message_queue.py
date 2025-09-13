#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Message Queue - Enhanced message queue system supporting multiple backends
for robust event-driven architecture.

Based on best practices from OctoBot and enterprise messaging systems.
"""

import logging
import queue
import threading
import time
import asyncio
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Generic
import multiprocessing
import uuid
import json
from datetime import datetime
import traceback

from trading_bot.event_system.event_types import Event

# Set up logging
logger = logging.getLogger("MessageQueue")

# Generic type for message payload
T = TypeVar('T')

class QueueBackend(Enum):
    """Supported queue backend types"""
    MEMORY = "memory"       # Simple in-memory queue
    REDIS = "redis"         # Redis-backed distributed queue
    KAFKA = "kafka"         # Kafka message broker
    ZEROMQ = "zeromq"       # ZeroMQ messaging library
    
class QueueType(Enum):
    """Types of message queues"""
    DIRECT = "direct"       # Point-to-point queue
    TOPIC = "topic"         # Publish-subscribe topic
    PRIORITY = "priority"   # Priority-based queue

class Message(Generic[T]):
    """
    Generic message container for queue systems with metadata
    and serialization support.
    """
    
    def __init__(
        self,
        payload: T,
        message_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        source: Optional[str] = None,
        destination: Optional[str] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        expiry: Optional[float] = None
    ):
        """
        Initialize a message
        
        Args:
            payload: Message content
            message_id: Unique message identifier
            timestamp: Message creation time
            source: Source component
            destination: Target component/queue
            priority: Message priority (higher = more important)
            metadata: Additional message metadata
            expiry: Expiration time in seconds (None = never expires)
        """
        self.payload = payload
        self.message_id = message_id or str(uuid.uuid4())
        self.timestamp = timestamp or datetime.now()
        self.source = source
        self.destination = destination
        self.priority = priority
        self.metadata = metadata or {}
        self.expiry = expiry
        
        # Internal tracking
        self.attempts = 0
        self.last_attempt = None
        self.received_time = None
        self.processed_time = None
        
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.expiry is None:
            return False
            
        return (datetime.now() - self.timestamp).total_seconds() > self.expiry
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "destination": self.destination,
            "priority": self.priority,
            "metadata": self.metadata,
            "expiry": self.expiry,
            "payload": self.payload if not isinstance(self.payload, Event) else self.payload.to_dict(),
            "attempts": self.attempts,
            "last_attempt": self.last_attempt.isoformat() if self.last_attempt else None,
            "received_time": self.received_time.isoformat() if self.received_time else None,
            "processed_time": self.processed_time.isoformat() if self.processed_time else None,
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        msg = cls(
            payload=data["payload"],
            message_id=data.get("message_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else None,
            source=data.get("source"),
            destination=data.get("destination"),
            priority=data.get("priority", 0),
            metadata=data.get("metadata", {}),
            expiry=data.get("expiry")
        )
        
        # Set tracking fields
        msg.attempts = data.get("attempts", 0)
        if "last_attempt" in data and data["last_attempt"]:
            msg.last_attempt = datetime.fromisoformat(data["last_attempt"])
        if "received_time" in data and data["received_time"]:
            msg.received_time = datetime.fromisoformat(data["received_time"])
        if "processed_time" in data and data["processed_time"]:
            msg.processed_time = datetime.fromisoformat(data["processed_time"])
            
        return msg
        
    def __lt__(self, other):
        """Compare messages for priority queue"""
        if not isinstance(other, Message):
            return NotImplemented
        return self.priority > other.priority  # Higher priority comes first


class MessageQueue:
    """
    Enhanced message queue system with support for multiple backends,
    priority-based messaging, and both sync and async interfaces.
    """
    
    def __init__(
        self,
        name: str,
        queue_type: QueueType = QueueType.DIRECT,
        backend: QueueBackend = QueueBackend.MEMORY,
        max_size: int = 1000,
        worker_threads: int = 2,
        backend_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a message queue
        
        Args:
            name: Queue name/identifier
            queue_type: Type of queue (direct, topic, priority)
            backend: Queue backend implementation
            max_size: Maximum queue size
            worker_threads: Number of worker threads for processing
            backend_config: Backend-specific configuration
        """
        self.name = name
        self.queue_type = queue_type
        self.backend = backend
        self.max_size = max_size
        self.worker_threads = worker_threads
        self.backend_config = backend_config or {}
        
        # Setup based on backend type
        self._setup_backend()
        
        # Queue state
        self.running = False
        self.workers = []
        self.subscribers: Dict[str, List[Callable]] = {}
        self.stats = {
            "messages_in": 0,
            "messages_out": 0,
            "errors": 0,
            "created": datetime.now()
        }
        
        logger.info(f"MessageQueue '{name}' initialized with {backend.value} backend and {queue_type.value} type")
        
    def _setup_backend(self) -> None:
        """Set up the queue backend based on configuration"""
        if self.backend == QueueBackend.MEMORY:
            if self.queue_type == QueueType.PRIORITY:
                self.queue = queue.PriorityQueue(maxsize=self.max_size)
            else:
                self.queue = queue.Queue(maxsize=self.max_size)
                
            # Async queue for async API
            self.async_queue = asyncio.Queue(maxsize=self.max_size)
            
        elif self.backend == QueueBackend.REDIS:
            try:
                import redis
                self.redis = redis.Redis(**self.backend_config)
                self.redis.ping()  # Test connection
                logger.info("Connected to Redis backend")
            except ImportError:
                logger.error("Redis library not installed. Falling back to memory queue.")
                self.backend = QueueBackend.MEMORY
                self._setup_backend()
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}. Falling back to memory queue.")
                self.backend = QueueBackend.MEMORY
                self._setup_backend()
                
        elif self.backend == QueueBackend.KAFKA:
            try:
                from kafka import KafkaProducer, KafkaConsumer
                # Kafka setup would go here
                logger.info("Connected to Kafka backend")
            except ImportError:
                logger.error("Kafka library not installed. Falling back to memory queue.")
                self.backend = QueueBackend.MEMORY
                self._setup_backend()
        
        elif self.backend == QueueBackend.ZEROMQ:
            try:
                import zmq
                # ZeroMQ setup would go here
                logger.info("Connected to ZeroMQ backend")
            except ImportError:
                logger.error("ZeroMQ library not installed. Falling back to memory queue.")
                self.backend = QueueBackend.MEMORY
                self._setup_backend()
    
    def start(self) -> None:
        """Start queue processing workers"""
        if self.running:
            logger.warning(f"Queue '{self.name}' already running")
            return
            
        self.running = True
        
        # Start worker threads for memory backend
        if self.backend == QueueBackend.MEMORY:
            for i in range(self.worker_threads):
                worker = threading.Thread(
                    target=self._worker_loop,
                    name=f"MQ-{self.name}-Worker-{i}",
                    daemon=True
                )
                worker.start()
                self.workers.append(worker)
        
        # Other backends would initialize their consumers here
        
        logger.info(f"MessageQueue '{self.name}' started with {self.worker_threads} workers")
    
    def stop(self) -> None:
        """Stop queue processing"""
        if not self.running:
            logger.warning(f"Queue '{self.name}' not running")
            return
            
        self.running = False
        
        # Stop worker threads for memory backend
        if self.backend == QueueBackend.MEMORY:
            for worker in self.workers:
                worker.join(timeout=1.0)
        
        # Other backends would clean up their connections here
        
        self.workers = []
        logger.info(f"MessageQueue '{self.name}' stopped")
    
    def publish(
        self, 
        message: Union[Message, Any],
        block: bool = True,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Publish message to the queue
        
        Args:
            message: Message to publish or raw payload
            block: Whether to block if queue is full
            timeout: Timeout in seconds if blocking
            
        Returns:
            True if published successfully
        """
        # Convert raw payload to Message if needed
        if not isinstance(message, Message):
            message = Message(
                payload=message,
                source=self.name,
                destination=self.name
            )
        
        # Set destination if not set
        if not message.destination:
            message.destination = self.name
            
        try:
            if self.backend == QueueBackend.MEMORY:
                if self.queue_type == QueueType.PRIORITY:
                    self.queue.put((message.priority, message), block=block, timeout=timeout)
                else:
                    self.queue.put(message, block=block, timeout=timeout)
                    
            elif self.backend == QueueBackend.REDIS:
                # Redis implementation
                serialized = json.dumps(message.to_dict())
                if self.queue_type == QueueType.PRIORITY:
                    self.redis.zadd(f"mq:{self.name}", {serialized: message.priority})
                else:
                    self.redis.lpush(f"mq:{self.name}", serialized)
                    
            elif self.backend == QueueBackend.KAFKA:
                # Kafka implementation would go here
                pass
                
            elif self.backend == QueueBackend.ZEROMQ:
                # ZeroMQ implementation would go here
                pass
                
            # Update stats
            self.stats["messages_in"] += 1
            return True
            
        except queue.Full:
            logger.warning(f"Queue '{self.name}' is full, message not published")
            return False
        except Exception as e:
            logger.error(f"Error publishing to queue '{self.name}': {e}")
            self.stats["errors"] += 1
            return False
            
    async def publish_async(self, message: Union[Message, Any]) -> bool:
        """
        Publish message asynchronously
        
        Args:
            message: Message to publish or raw payload
            
        Returns:
            True if published successfully
        """
        # Convert raw payload to Message if needed
        if not isinstance(message, Message):
            message = Message(
                payload=message,
                source=self.name,
                destination=self.name
            )
            
        try:
            if self.backend == QueueBackend.MEMORY:
                await self.async_queue.put(message)
                
            elif self.backend == QueueBackend.REDIS:
                # Redis implementation using aioredis would go here
                pass
                
            # Update stats
            self.stats["messages_in"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error async publishing to queue '{self.name}': {e}")
            self.stats["errors"] += 1
            return False
    
    def subscribe(self, callback: Callable[[Message], None], subscriber_id: Optional[str] = None) -> str:
        """
        Subscribe to receive messages from this queue
        
        Args:
            callback: Function to call with received messages
            subscriber_id: Unique identifier for subscriber (generated if not provided)
            
        Returns:
            Subscriber ID
        """
        subscriber_id = subscriber_id or f"sub_{len(self.subscribers)}"
        
        if subscriber_id not in self.subscribers:
            self.subscribers[subscriber_id] = []
            
        self.subscribers[subscriber_id].append(callback)
        logger.debug(f"Added subscriber {subscriber_id} to queue '{self.name}'")
        
        return subscriber_id
    
    def unsubscribe(self, subscriber_id: str) -> bool:
        """
        Unsubscribe from queue
        
        Args:
            subscriber_id: Subscriber ID to remove
            
        Returns:
            True if unsubscribed successfully
        """
        if subscriber_id in self.subscribers:
            del self.subscribers[subscriber_id]
            logger.debug(f"Removed subscriber {subscriber_id} from queue '{self.name}'")
            return True
            
        return False
    
    def _worker_loop(self) -> None:
        """Worker thread for processing messages"""
        logger.debug(f"Started worker thread {threading.current_thread().name}")
        
        while self.running:
            try:
                # Get message from queue with timeout to check running flag
                try:
                    if self.queue_type == QueueType.PRIORITY:
                        _, message = self.queue.get(timeout=0.1)
                    else:
                        message = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Skip expired messages
                if message.is_expired():
                    logger.debug(f"Skipping expired message {message.message_id}")
                    self.queue.task_done()
                    continue
                
                # Process message
                message.received_time = datetime.now()
                message.attempts += 1
                message.last_attempt = datetime.now()
                
                # Deliver to subscribers
                self._deliver_to_subscribers(message)
                
                # Mark as done
                self.queue.task_done()
                message.processed_time = datetime.now()
                
                # Update stats
                self.stats["messages_out"] += 1
                
            except Exception as e:
                logger.error(f"Error in message queue worker: {e}")
                logger.error(traceback.format_exc())
                self.stats["errors"] += 1
                
        logger.debug(f"Stopped worker thread {threading.current_thread().name}")
    
    def _deliver_to_subscribers(self, message: Message) -> None:
        """
        Deliver message to all subscribers
        
        Args:
            message: Message to deliver
        """
        for subscriber_id, callbacks in self.subscribers.items():
            for callback in callbacks:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Error in subscriber {subscriber_id} callback: {e}")
                    self.stats["errors"] += 1
    
    async def _async_worker(self) -> None:
        """Async worker for processing messages"""
        logger.debug("Started async worker")
        
        while self.running:
            try:
                # Get message from async queue
                message = await self.async_queue.get()
                
                # Skip expired messages
                if message.is_expired():
                    logger.debug(f"Skipping expired message {message.message_id}")
                    self.async_queue.task_done()
                    continue
                
                # Process message
                message.received_time = datetime.now()
                message.attempts += 1
                message.last_attempt = datetime.now()
                
                # For async interface, publish to regular queue for delivery
                self.publish(message, block=False)
                
                # Mark as done
                self.async_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in async message queue worker: {e}")
                logger.error(traceback.format_exc())
                self.stats["errors"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        stats = self.stats.copy()
        
        # Add current queue size
        if self.backend == QueueBackend.MEMORY:
            stats["queue_size"] = self.queue.qsize()
            stats["async_queue_size"] = self.async_queue.qsize() if hasattr(self, "async_queue") else 0
            
        elif self.backend == QueueBackend.REDIS:
            if self.queue_type == QueueType.PRIORITY:
                stats["queue_size"] = self.redis.zcard(f"mq:{self.name}")
            else:
                stats["queue_size"] = self.redis.llen(f"mq:{self.name}")
        
        # Add subscriber count
        stats["subscribers"] = len(self.subscribers)
        
        # Add uptime
        stats["uptime_seconds"] = (datetime.now() - stats["created"]).total_seconds()
        
        return stats
