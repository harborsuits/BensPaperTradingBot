#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event Manager - High-level component that connects the event system
to trading strategies, modes, and risk management.

Based on best practices from OctoBot, FreqTrade, and EA31337.
"""

import logging
import threading
import asyncio
from typing import Dict, List, Any, Callable, Optional, Union, Set
from datetime import datetime
import time
import traceback
import queue
import uuid

from trading_bot.event_system.event_bus import EventBus, EventHandler
from trading_bot.event_system.event_types import (
    Event, EventType, MarketDataEvent, SignalEvent, 
    OrderEvent, RiskEvent, AnalysisEvent
)
from trading_bot.strategies.base.base_strategy import Strategy, SignalType
from trading_bot.trading_modes.base_trading_mode import BaseTradingMode, Order
from trading_bot.risk.risk_manager import RiskManager

# Set up logging
logger = logging.getLogger("EventManager")

class EventManager:
    """
    Central coordinator for the event-driven architecture, connecting
    strategies, trading modes, risk management, and data sources through
    the event bus.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the event manager
        
        Args:
            config: Optional configuration
        """
        self.config = config or {}
        
        # Initialize event bus
        worker_threads = self.config.get("event_worker_threads", 4)
        max_queue_size = self.config.get("event_queue_size", 1000)
        self.event_bus = EventBus(
            max_queue_size=max_queue_size,
            worker_threads=worker_threads
        )
        
        # Components
        self.strategies: Dict[str, Strategy] = {}
        self.trading_mode: Optional[BaseTradingMode] = None
        self.risk_manager: Optional[RiskManager] = None
        
        # Processing status
        self.running = False
        self.processing_thread = None
        self.last_process_time = None
        
        # Scheduled tasks
        self.scheduled_tasks: Dict[str, Dict[str, Any]] = {}
        self.last_schedule_check = time.time()
        
        # ID for this instance
        self.instance_id = str(uuid.uuid4())[:8]
        
        logger.info(f"EventManager initialized with {worker_threads} workers")
    
    def register_strategy(self, name: str, strategy: Strategy) -> None:
        """
        Register a trading strategy
        
        Args:
            name: Strategy name
            strategy: Strategy instance
        """
        self.strategies[name] = strategy
        logger.info(f"Registered strategy: {name}")
        
        # Register event handlers for this strategy
        self._register_strategy_handlers(name, strategy)
    
    def set_trading_mode(self, trading_mode: BaseTradingMode) -> None:
        """
        Set the active trading mode
        
        Args:
            trading_mode: Trading mode instance
        """
        self.trading_mode = trading_mode
        logger.info(f"Set trading mode: {trading_mode.name}")
        
        # Register event handlers for trading mode
        self._register_trading_mode_handlers(trading_mode)
    
    def set_risk_manager(self, risk_manager: RiskManager) -> None:
        """
        Set the risk manager
        
        Args:
            risk_manager: Risk manager instance
        """
        self.risk_manager = risk_manager
        logger.info("Set risk manager")
        
        # Register event handlers for risk manager
        self._register_risk_manager_handlers(risk_manager)
    
    def start(self) -> None:
        """Start the event manager and all components"""
        if self.running:
            logger.warning("EventManager already running")
            return
            
        # Start event bus
        self.event_bus.start()
        
        # Set running flag
        self.running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            name="EventManager-Processor",
            daemon=True
        )
        self.processing_thread.start()
        
        # Publish system start event
        self.publish_event(
            Event(
                event_type=EventType.SYSTEM_START,
                data={"timestamp": datetime.now().isoformat()},
                source="event_manager",
                metadata={"instance_id": self.instance_id}
            )
        )
        
        logger.info("EventManager started")
    
    def stop(self) -> None:
        """Stop the event manager and all components"""
        if not self.running:
            logger.warning("EventManager not running")
            return
            
        # Set running flag
        self.running = False
        
        # Publish system stop event
        self.publish_event(
            Event(
                event_type=EventType.SYSTEM_STOP,
                data={"timestamp": datetime.now().isoformat()},
                source="event_manager",
                metadata={"instance_id": self.instance_id}
            )
        )
        
        # Wait for processing thread to finish
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
            
        # Stop event bus
        self.event_bus.stop()
        
        logger.info("EventManager stopped")
    
    def publish_event(self, event: Event, block: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Publish an event to the event bus
        
        Args:
            event: Event to publish
            block: Whether to block if queue is full
            timeout: Timeout in seconds if blocking
            
        Returns:
            True if event was published, False if timeout or queue full
        """
        return self.event_bus.publish(event, block=block, timeout=timeout)
    
    async def publish_event_async(self, event: Event) -> bool:
        """
        Publish an event to the event bus asynchronously
        
        Args:
            event: Event to publish
            
        Returns:
            True if event was published
        """
        return await self.event_bus.publish_async(event)
    
    def publish_market_data(
        self,
        symbol: str,
        data: Dict[str, Any],
        source: str = "data_feed",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Publish market data event
        
        Args:
            symbol: Market symbol
            data: Market data
            source: Data source
            metadata: Additional metadata
            
        Returns:
            True if event was published
        """
        event = MarketDataEvent(
            symbol=symbol,
            data=data,
            source=source,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        return self.publish_event(event)
    
    def publish_signal(
        self,
        symbol: str,
        signal_type: str,
        direction: int,
        strength: float,
        strategy: str,
        source: str = "strategy",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Publish signal event
        
        Args:
            symbol: Market symbol
            signal_type: Type of signal
            direction: Signal direction
            strength: Signal strength
            strategy: Strategy name
            source: Signal source
            metadata: Additional metadata
            
        Returns:
            True if event was published
        """
        event = SignalEvent(
            symbol=symbol,
            signal_type=signal_type,
            direction=direction,
            strength=strength,
            strategy=strategy,
            source=source,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        return self.publish_event(event)
    
    def publish_order(
        self,
        order_id: str,
        symbol: str,
        order_type: str,
        side: int,
        quantity: float,
        price: Optional[float],
        status: str,
        source: str = "trading_mode",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Publish order event
        
        Args:
            order_id: Order ID
            symbol: Market symbol
            order_type: Order type
            side: Order side
            quantity: Order quantity
            price: Order price
            status: Order status
            source: Order source
            metadata: Additional metadata
            
        Returns:
            True if event was published
        """
        event = OrderEvent(
            order_id=order_id,
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            price=price,
            status=status,
            source=source,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        return self.publish_event(event)
    
    def _processing_loop(self) -> None:
        """Main processing loop for periodic tasks"""
        logger.debug("Processing loop started")
        
        while self.running:
            try:
                # Record process time
                self.last_process_time = time.time()
                
                # Run scheduled tasks
                self._check_scheduled_tasks()
                
                # Sleep for a short time
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                logger.error(traceback.format_exc())
                
        logger.debug("Processing loop stopped")
    
    def _check_scheduled_tasks(self) -> None:
        """Check and run scheduled tasks"""
        current_time = time.time()
        
        # Only check every second
        if current_time - self.last_schedule_check < 1.0:
            return
            
        self.last_schedule_check = current_time
        
        # Check each task
        for task_id, task in list(self.scheduled_tasks.items()):
            next_run = task.get("next_run", 0)
            
            if current_time >= next_run:
                # Task is due to run
                try:
                    # Run task function
                    task["function"](*task.get("args", []), **task.get("kwargs", {}))
                    
                    # Update next run time
                    interval = task.get("interval", 60)
                    task["next_run"] = current_time + interval
                    
                    # Remove one-time tasks
                    if task.get("one_time", False):
                        self.scheduled_tasks.pop(task_id)
                        
                except Exception as e:
                    logger.error(f"Error running scheduled task {task_id}: {e}")
                    logger.error(traceback.format_exc())
    
    def schedule_task(
        self,
        function: Callable,
        interval: float,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        one_time: bool = False
    ) -> str:
        """
        Schedule a task to run periodically
        
        Args:
            function: Function to run
            interval: Interval in seconds
            args: Function arguments
            kwargs: Function keyword arguments
            name: Task name
            one_time: Whether task runs only once
            
        Returns:
            Task ID
        """
        task_id = name or f"task_{len(self.scheduled_tasks)}"
        
        self.scheduled_tasks[task_id] = {
            "function": function,
            "interval": interval,
            "args": args or [],
            "kwargs": kwargs or {},
            "next_run": time.time() + interval,
            "one_time": one_time
        }
        
        logger.debug(f"Scheduled task: {task_id}, interval: {interval}s")
        return task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a scheduled task
        
        Args:
            task_id: Task ID
            
        Returns:
            True if task was canceled, False if not found
        """
        if task_id in self.scheduled_tasks:
            self.scheduled_tasks.pop(task_id)
            logger.debug(f"Canceled task: {task_id}")
            return True
            
        logger.warning(f"Task not found: {task_id}")
        return False
    
    def _register_strategy_handlers(self, name: str, strategy: Strategy) -> None:
        """
        Register event handlers for a strategy
        
        Args:
            name: Strategy name
            strategy: Strategy instance
        """
        # Handler for market data
        def handle_market_data(event: Event) -> None:
            """Process market data for strategy"""
            if event.event_type != EventType.MARKET_DATA:
                return
                
            symbol = event.data.get("symbol")
            if not symbol:
                return
                
            # Create data dictionary in required format
            data = {symbol: event.data}
            
            # Generate signals
            signals = strategy.generate_signals(data, datetime.now())
            
            # Publish signals
            for sym, signal in signals.items():
                if signal != SignalType.FLAT:  # Only publish non-flat signals
                    self.publish_signal(
                        symbol=sym,
                        signal_type="entry" if signal in [SignalType.LONG, SignalType.SHORT] else "exit",
                        direction=signal.value,
                        strength=1.0,  # Set default strength
                        strategy=name,
                        source=name,
                        metadata={"strategy_type": strategy.__class__.__name__}
                    )
        
        # Register handler
        self.event_bus.register_handler(
            EventHandler(
                callback=handle_market_data,
                event_type=EventType.MARKET_DATA,
                name=f"strategy_{name}_market_data",
                priority=10  # High priority for strategy signal generation
            )
        )
        
        logger.debug(f"Registered event handlers for strategy: {name}")
    
    def _register_trading_mode_handlers(self, trading_mode: BaseTradingMode) -> None:
        """
        Register event handlers for trading mode
        
        Args:
            trading_mode: Trading mode instance
        """
        # Handler for signal events
        def handle_signal(event: Event) -> None:
            """Process signals for trading mode"""
            if event.event_type != EventType.SIGNAL_GENERATED:
                return
                
            # Extract signal details
            symbol = event.data.get("symbol")
            strategy = event.data.get("strategy")
            direction = event.data.get("direction")
            
            if not all([symbol, strategy, direction is not None]):
                logger.warning(f"Invalid signal event: {event}")
                return
                
            # Map signals by strategy
            strategy_signals = {
                strategy: {
                    symbol: SignalType(direction)
                }
            }
            
            # Process signals
            # NOTE: In real implementation, would need market data and account balance
            # This is a simplified example
            market_data = {symbol: {"price": 100.0}}  # Mock data
            account_balance = 10000.0  # Mock balance
            
            orders = trading_mode.process_signals(
                strategy_signals=strategy_signals,
                market_data=market_data,
                current_time=datetime.now(),
                account_balance=account_balance
            )
            
            # Publish order events
            for order in orders:
                self.publish_order(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    order_type=order.order_type.value,
                    side=order.side.value,
                    quantity=order.quantity,
                    price=order.price,
                    status="created",
                    source=trading_mode.name,
                    metadata={
                        "trading_mode": trading_mode.name,
                        "order": order.to_dict()
                    }
                )
        
        # Register signal handler
        self.event_bus.register_handler(
            EventHandler(
                callback=handle_signal,
                event_type=EventType.SIGNAL_GENERATED,
                name=f"trading_mode_{trading_mode.name}_signal",
                priority=5  # Medium priority
            )
        )
        
        # Handler for order updates
        def handle_order_update(event: Event) -> None:
            """Process order updates for trading mode"""
            if event.event_type not in [EventType.ORDER_FILLED, EventType.ORDER_CANCELED, EventType.ORDER_UPDATED]:
                return
                
            # Extract order details
            order_id = event.data.get("order_id")
            status = event.data.get("status")
            
            if not all([order_id, status]):
                logger.warning(f"Invalid order update event: {event}")
                return
                
            # Update trading mode
            trading_mode.handle_order_update(event.data)
        
        # Register order update handler
        self.event_bus.register_handler(
            EventHandler(
                callback=handle_order_update,
                event_type=[EventType.ORDER_FILLED, EventType.ORDER_CANCELED, EventType.ORDER_UPDATED],
                name=f"trading_mode_{trading_mode.name}_order_update",
                priority=8  # Higher priority for order updates
            )
        )
        
        logger.debug(f"Registered event handlers for trading mode: {trading_mode.name}")
    
    def _register_risk_manager_handlers(self, risk_manager: RiskManager) -> None:
        """
        Register event handlers for risk manager
        
        Args:
            risk_manager: Risk manager instance
        """
        # Handler for order events
        def handle_order_created(event: Event) -> None:
            """Process order creation for risk checks"""
            if event.event_type != EventType.ORDER_CREATED:
                return
                
            # Extract order details
            order_id = event.data.get("order_id")
            symbol = event.data.get("symbol")
            side = event.data.get("side")
            quantity = event.data.get("quantity")
            price = event.data.get("price")
            
            if not all([order_id, symbol, side is not None, quantity, price]):
                logger.warning(f"Invalid order created event: {event}")
                return
                
            # Check risk limits
            should_reduce, reasons = risk_manager.check_risk_limits()
            
            if should_reduce:
                # Publish risk limit breach event
                self.publish_event(
                    RiskEvent(
                        risk_type="limit_breach",
                        level=risk_manager.risk_level.value,
                        details={
                            "reasons": reasons,
                            "order_id": order_id,
                            "symbol": symbol,
                            "side": side,
                            "quantity": quantity,
                            "price": price
                        },
                        source="risk_manager",
                        timestamp=datetime.now(),
                        metadata={"priority": 10}  # High priority for risk events
                    )
                )
        
        # Register order created handler
        self.event_bus.register_handler(
            EventHandler(
                callback=handle_order_created,
                event_type=EventType.ORDER_CREATED,
                name="risk_manager_order_created",
                priority=9  # High priority for risk checks
            )
        )
        
        # Handler for market data to update stop losses
        def handle_market_data_for_stops(event: Event) -> None:
            """Update trailing stops based on market data"""
            if event.event_type != EventType.MARKET_DATA:
                return
                
            symbol = event.data.get("symbol")
            price = event.data.get("price")
            
            if not all([symbol, price]):
                return
                
            # Update market data for stop loss checks
            market_data = {symbol: {"price": price}}
            
            # Update trailing stops
            risk_manager.update_trailing_stops(market_data)
            
            # Check stop losses
            triggered_stops = risk_manager.check_stop_losses(market_data)
            
            # Publish stop loss events for triggered stops
            for stop in triggered_stops:
                self.publish_event(
                    RiskEvent(
                        risk_type="stop_loss",
                        level=risk_manager.risk_level.value,
                        details=stop,
                        source="risk_manager",
                        timestamp=datetime.now(),
                        metadata={"priority": 10}  # High priority for risk events
                    )
                )
        
        # Register market data handler for stop loss updates
        self.event_bus.register_handler(
            EventHandler(
                callback=handle_market_data_for_stops,
                event_type=EventType.MARKET_DATA,
                name="risk_manager_market_data",
                priority=7  # Medium-high priority
            )
        )
        
        logger.debug("Registered event handlers for risk manager")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for event system"""
        
        # Get event bus metrics
        bus_metrics = self.event_bus.get_metrics()
        
        # Build overall metrics
        metrics = {
            "event_bus": bus_metrics,
            "components": {
                "strategies": len(self.strategies),
                "trading_mode": self.trading_mode.name if self.trading_mode else None,
                "risk_manager": "active" if self.risk_manager else "none"
            },
            "scheduled_tasks": len(self.scheduled_tasks),
            "running": self.running,
            "instance_id": self.instance_id
        }
        
        return metrics
