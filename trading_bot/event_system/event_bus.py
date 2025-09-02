#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event Bus - Core component of the event-driven architecture that handles
dispatching events to registered handlers.

Based on best practices from OctoBot and EA31337.
"""

import logging
import asyncio
import threading
import queue
from typing import Dict, List, Any, Callable, Set, Optional, Union, Tuple
from datetime import datetime
import time
import traceback
import multiprocessing
import concurrent.futures

from trading_bot.event_system.event_types import Event, EventType

# Set up logging
logger = logging.getLogger("EventBus")

class EventHandler:
    """
    Handler for a specific event type with filtering capabilities.
    Each handler registers a callback function and optional filters.
    """
    
    def __init__(
        self,
        callback: Callable[[Event], None],
        event_type: Optional[Union[EventType, List[EventType]]] = None,
        source_filter: Optional[Union[str, List[str]]] = None,
        symbol_filter: Optional[Union[str, List[str]]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        async_handler: bool = False,
        name: Optional[str] = None
    ):
        """
        Initialize event handler
        
        Args:
            callback: Function to call when event matches
            event_type: Event type(s) to handle (all if None)
            source_filter: Source(s) to handle (all if None)
            symbol_filter: Symbol(s) to handle (all if None)
            metadata_filter: Metadata filters (all if None)
            priority: Handler priority (higher runs first)
            async_handler: Whether handler is asynchronous
            name: Handler name for identification
        """
        self.callback = callback
        self.name = name or callback.__name__
        
        # Convert single values to lists for consistent handling
        if event_type is not None and not isinstance(event_type, list):
            self.event_types = [event_type]
        else:
            self.event_types = event_type
            
        if source_filter is not None and not isinstance(source_filter, list):
            self.source_filters = [source_filter]
        else:
            self.source_filters = source_filter
            
        if symbol_filter is not None and not isinstance(symbol_filter, list):
            self.symbol_filters = [symbol_filter]
        else:
            self.symbol_filters = symbol_filter
        
        self.metadata_filter = metadata_filter
        self.priority = priority
        self.async_handler = async_handler
        
        # Statistics
        self.events_processed = 0
        self.last_event_time = None
        self.total_processing_time = 0
        
    def matches(self, event: Event) -> bool:
        """
        Check if event matches this handler's filters
        
        Args:
            event: Event to check
            
        Returns:
            True if event matches, False otherwise
        """
        # Check event type
        if self.event_types is not None and event.event_type not in self.event_types:
            return False
        
        # Check source
        if self.source_filters is not None and event.source not in self.source_filters:
            return False
        
        # Check symbol (if event has symbol data)
        if self.symbol_filters is not None:
            symbol = event.data.get("symbol")
            if not symbol or symbol not in self.symbol_filters:
                return False
        
        # Check metadata filters (all keys must match)
        if self.metadata_filter is not None:
            for key, value in self.metadata_filter.items():
                if key not in event.metadata or event.metadata[key] != value:
                    return False
        
        return True
    
    def handle(self, event: Event) -> None:
        """
        Handle event by calling the callback function
        
        Args:
            event: Event to handle
        """
        start_time = time.time()
        
        try:
            self.callback(event)
            
            # Update statistics
            self.events_processed += 1
            self.last_event_time = datetime.now()
            self.total_processing_time += (time.time() - start_time)
            
            # Mark as processed
            if self.name not in event.processed_by:
                event.processed_by.append(self.name)
                
        except Exception as e:
            logger.error(f"Error in event handler {self.name}: {e}")
            logger.error(traceback.format_exc())
            
    async def handle_async(self, event: Event) -> None:
        """
        Handle event asynchronously
        
        Args:
            event: Event to handle
        """
        if not self.async_handler:
            # Run synchronous handler in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self.handle(event))
        else:
            # Run directly if handler is already asynchronous
            start_time = time.time()
            
            try:
                await self.callback(event)
                
                # Update statistics
                self.events_processed += 1
                self.last_event_time = datetime.now()
                self.total_processing_time += (time.time() - start_time)
                
                # Mark as processed
                if self.name not in event.processed_by:
                    event.processed_by.append(self.name)
                    
            except Exception as e:
                logger.error(f"Error in async event handler {self.name}: {e}")
                logger.error(traceback.format_exc())


class EventBus:
    """
    Central event dispatch system that supports both synchronous and
    asynchronous event handling, with priority-based dispatch and
    filtering capabilities.
    """
    
    def __init__(self, max_queue_size: int = 1000, worker_threads: int = 4):
        """
        Initialize the event bus
        
        Args:
            max_queue_size: Maximum number of events in queue before blocking
            worker_threads: Number of worker threads for event processing
        """
        self.handlers: List[EventHandler] = []
        self.event_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.worker_threads = worker_threads
        self.running = False
        self.workers: List[threading.Thread] = []
        
        # Async event loop
        self.async_mode = False
        self.async_queue = asyncio.Queue(maxsize=max_queue_size)
        self.async_task = None
        
        # Thread pool for concurrent dispatch
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_threads, 
            thread_name_prefix="EventBus"
        )
        
        # Metrics
        self.events_processed = 0
        self.events_published = 0
        self.start_time = datetime.now()
        
        logger.info(f"EventBus initialized with {worker_threads} workers")
        
    def register_handler(self, handler: EventHandler) -> None:
        """
        Register a new event handler
        
        Args:
            handler: Handler to register
        """
        self.handlers.append(handler)
        # Sort handlers by priority (highest first)
        self.handlers.sort(key=lambda h: h.priority, reverse=True)
        logger.debug(f"Registered handler: {handler.name} with priority {handler.priority}")
        
    def unregister_handler(self, handler_name: str) -> bool:
        """
        Unregister an event handler by name
        
        Args:
            handler_name: Name of handler to unregister
            
        Returns:
            True if handler was unregistered, False if not found
        """
        for i, handler in enumerate(self.handlers):
            if handler.name == handler_name:
                self.handlers.pop(i)
                logger.debug(f"Unregistered handler: {handler_name}")
                return True
        
        logger.warning(f"Handler not found: {handler_name}")
        return False
    
    def publish(self, event: Event, block: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Publish event to the event bus
        
        Args:
            event: Event to publish
            block: Whether to block if queue is full
            timeout: Timeout in seconds if blocking
            
        Returns:
            True if event was published, False if timeout or queue full
        """
        # Add event to queue
        try:
            # Use negative priority as lower number means higher priority in queue
            priority = -event.priority  
            self.event_queue.put((priority, event), block=block, timeout=timeout)
            self.events_published += 1
            return True
        except queue.Full:
            logger.warning(f"Event queue full, could not publish event: {event}")
            return False
    
    def start(self) -> None:
        """Start event processing workers"""
        if self.running:
            logger.warning("EventBus already running")
            return
            
        self.running = True
        
        # Create and start worker threads
        for i in range(self.worker_threads):
            worker = threading.Thread(
                target=self._event_worker,
                name=f"EventBus-Worker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            
        logger.info(f"EventBus started with {self.worker_threads} workers")
    
    def stop(self) -> None:
        """Stop event processing workers"""
        if not self.running:
            logger.warning("EventBus not running")
            return
            
        self.running = False
        
        # Wait for worker threads to finish
        for worker in self.workers:
            worker.join(timeout=1.0)
            
        self.workers = []
        logger.info("EventBus stopped")
    
    def _event_worker(self) -> None:
        """Worker thread for processing events"""
        logger.debug(f"Event worker started: {threading.current_thread().name}")
        
        while self.running:
            try:
                # Get event from queue (with timeout to check running flag)
                try:
                    _, event = self.event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                    
                # Process event
                self._process_event(event)
                
                # Mark as done
                self.event_queue.task_done()
                self.events_processed += 1
                
            except Exception as e:
                logger.error(f"Error in event worker: {e}")
                logger.error(traceback.format_exc())
                
        logger.debug(f"Event worker stopped: {threading.current_thread().name}")
    
    def _process_event(self, event: Event) -> None:
        """
        Process event by dispatching to matching handlers
        
        Args:
            event: Event to process
        """
        matching_handlers = [h for h in self.handlers if h.matches(event)]
        
        if not matching_handlers:
            logger.debug(f"No handlers for event: {event}")
            return
            
        # Process with matching handlers
        futures = []
        for handler in matching_handlers:
            future = self.thread_pool.submit(handler.handle, event)
            futures.append(future)
            
        # Wait for all handlers to complete
        concurrent.futures.wait(futures)
        
    # Async API for EventBus
    
    async def publish_async(self, event: Event) -> bool:
        """
        Publish event to the async event queue
        
        Args:
            event: Event to publish
            
        Returns:
            True if event was published
        """
        if not self.async_mode:
            self.async_mode = True
            self._start_async_task()
            
        await self.async_queue.put(event)
        self.events_published += 1
        return True
    
    def _start_async_task(self) -> None:
        """Start the async event processing task"""
        if self.async_task is None:
            loop = asyncio.get_event_loop()
            self.async_task = loop.create_task(self._async_event_processor())
            logger.info("Started async event processor")
    
    async def _async_event_processor(self) -> None:
        """Async task for processing events"""
        logger.debug("Async event processor started")
        
        while True:
            try:
                # Get event from queue
                event = await self.async_queue.get()
                
                # Process event
                matching_handlers = [h for h in self.handlers if h.matches(event)]
                
                if matching_handlers:
                    # Process with matching handlers
                    tasks = []
                    for handler in matching_handlers:
                        task = asyncio.create_task(handler.handle_async(event))
                        tasks.append(task)
                        
                    # Wait for all handlers to complete
                    await asyncio.gather(*tasks)
                
                # Mark as done
                self.async_queue.task_done()
                self.events_processed += 1
                
            except Exception as e:
                logger.error(f"Error in async event processor: {e}")
                logger.error(traceback.format_exc())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus performance metrics"""
        uptime = datetime.now() - self.start_time
        uptime_seconds = uptime.total_seconds()
        
        metrics = {
            "events_published": self.events_published,
            "events_processed": self.events_processed,
            "events_per_second": self.events_processed / uptime_seconds if uptime_seconds > 0 else 0,
            "queue_size": self.event_queue.qsize(),
            "handler_count": len(self.handlers),
            "uptime_seconds": uptime_seconds,
            "handlers": []
        }
        
        # Add handler metrics
        for handler in self.handlers:
            avg_time = handler.total_processing_time / handler.events_processed if handler.events_processed > 0 else 0
            
            metrics["handlers"].append({
                "name": handler.name,
                "events_processed": handler.events_processed,
                "avg_processing_time_ms": avg_time * 1000,
                "priority": handler.priority
            })
            
        return metrics
