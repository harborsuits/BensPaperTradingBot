#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Bot Orchestrator

This script serves as the main entry point for starting the trading bot system.
It initializes all components, connects them to the event system, and manages
their lifecycle. The orchestrator ensures all components work together cohesively.
"""

import os
import sys
import time
import logging
import threading
import signal
import argparse
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading_bot.log")
    ]
)
logger = logging.getLogger("Orchestrator")

# Import event system
from trading_bot.event_system import (
    EventBus, EventManager, EventType, Event,
    MessageQueue, ChannelManager
)

# Extend EventType with news and backtesting types
EventType.NEWS = "NEWS"
EventType.BACKTEST = "BACKTEST"

# Import trading components
from trading_bot.trading_modes import (
    BaseTradingMode, StandardTradingMode, 
    Order, OrderType, OrderStatus
)
from trading_bot.core import StrategyBase

# Import news sentiment and backtesting components
import random
from datetime import datetime, timedelta
from trading_bot.strategies_new.news_sentiment import NewsSentimentStrategy
from trading_bot.backtesting.backtester import Backtester
from trading_bot.strategies.news_sentiment_strategy import NewsSentimentStrategy as DeployableNewsSentimentStrategy
from trading_bot.autonomous.strategy_deployment_pipeline import get_deployment_pipeline, StrategyDeployment, DeploymentStatus
from trading_bot.risk.risk_manager import RiskLevel, StopLossType

# Import API functionality (will be started in a separate process)
import uvicorn
from multiprocessing import Process
from trading_bot.config.typed_settings import load_config, APISettings

class TradingBotOrchestrator:
    """Main orchestrator for the trading bot system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the orchestrator.
        
        Args:
            config_path: Optional path to config file. If None, default config is used.
        """
        logger.info("Initializing Trading Bot Orchestrator")
        
        # Load configuration
        try:
            if config_path:
                os.environ["CONFIG_PATH"] = config_path
            self.config = load_config()
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.config = None
            # Use default settings
        
        # Initialize components
        self.event_bus = EventBus()
        self.event_manager = EventManager(self.event_bus)
        self.channel_manager = ChannelManager()
        
        # Trading modes (strategies)
        self.trading_modes: Dict[str, BaseTradingMode] = {}
        
        # API server process
        self.api_server_process = None
        
        # Component threads
        self.threads = {}
        
        # Shutdown flag
        self.shutdown_requested = threading.Event()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("Orchestrator initialized")
    
    def initialize_event_system(self):
        """Initialize and connect the event system components."""
        logger.info("Initializing event system")
        
        # Create standard channels for different event types
        self.channel_manager.create_channel("market_data")
        self.channel_manager.create_channel("signals")
        self.channel_manager.create_channel("orders")
        self.channel_manager.create_channel("execution")
        self.channel_manager.create_channel("risk")
        self.channel_manager.create_channel("news")
        self.channel_manager.create_channel("backtest_results")
        
        # Create message queues for different processing pipelines
        MessageQueue.create("market_data_queue")
        MessageQueue.create("signal_queue")
        MessageQueue.create("order_queue")
        MessageQueue.create("news_queue")
        MessageQueue.create("backtest_queue")
        
        # Connect event handlers
        self.event_bus.subscribe(EventType.MARKET_DATA, self.handle_market_data_event)
        self.event_bus.subscribe(EventType.SIGNAL, self.handle_signal_event)
        self.event_bus.subscribe(EventType.ORDER, self.handle_order_event)
        self.event_bus.subscribe(EventType.RISK, self.handle_risk_event)
        self.event_bus.subscribe(EventType.NEWS, self.handle_news_event)
        self.event_bus.subscribe(EventType.BACKTEST, self.handle_backtest_event)
        
        logger.info("Event system initialized")
    
    def initialize_trading_modes(self):
        """Initialize trading modes based on configuration."""
        logger.info("Initializing trading modes")
        
        # Initialize backtester
        self.backtester = Backtester(self.event_bus, self.channel_manager)
        
        # Initialize news sentiment strategy
        self.news_sentiment_strategy = NewsSentimentStrategy(self.event_bus)
        
        # Create standard trading mode
        standard_mode = StandardTradingMode(
            name="standard",
            event_bus=self.event_bus,
            config=getattr(self.config, "trading_modes", {}).get("standard", {})
        )
        self.trading_modes["standard"] = standard_mode
        
        # TODO: Initialize other trading modes based on configuration
        
        logger.info(f"Trading modes initialized: {list(self.trading_modes.keys())}")
    
    def start_api_server(self):
        """Start the API server in a separate process."""
        logger.info("Starting API server")
        
        api_settings = getattr(self.config, "api", APISettings())
        
        # Define API server function to run in separate process
        def run_api_server():
            import uvicorn
            from trading_bot.api.app import app, initialize_api
            
            # Initialize API with references to components
            initialize_api(self.event_manager, None)  # No strategy rotator for now
            
            # Start uvicorn server
            uvicorn.run(
                app,
                host=api_settings.host,
                port=api_settings.port,
                log_level="info"
            )
        
        # Start API server in separate process
        self.api_server_process = Process(target=run_api_server)
        self.api_server_process.start()
        
        logger.info(f"API server started at http://{api_settings.host}:{api_settings.port}")
    
    def start_component_threads(self):
        """Start threads for various components."""
        logger.info("Starting component threads")
        
        # Market data processor thread
        market_data_thread = threading.Thread(
            target=self.market_data_processor,
            name="market_data_processor"
        )
        market_data_thread.daemon = True
        market_data_thread.start()
        self.threads["market_data"] = market_data_thread
        
        # Signal processor thread
        signal_thread = threading.Thread(
            target=self.signal_processor,
            name="signal_processor"
        )
        signal_thread.daemon = True
        signal_thread.start()
        self.threads["signal"] = signal_thread
        
        # Order processor thread
        order_thread = threading.Thread(
            target=self.order_processor,
            name="order_processor"
        )
        order_thread.daemon = True
        order_thread.start()
        self.threads["order"] = order_thread
        
        # News sentiment processor thread
        news_thread = threading.Thread(
            target=self.news_processor,
            name="news_processor"
        )
        news_thread.daemon = True
        news_thread.start()
        self.threads["news"] = news_thread
        
        # Backtesting processor thread
        backtest_thread = threading.Thread(
            target=self.backtest_processor,
            name="backtest_processor"
        )
        backtest_thread.daemon = True
        backtest_thread.start()
        self.threads["backtest"] = backtest_thread
        
        logger.info("Component threads started")
        
        for mode_name, mode in self.trading_modes.items():
            self.threads[f"mode_{mode_name}"] = threading.Thread(
                target=mode.run,
                name=f"TradingMode_{mode_name}"
            )
            self.threads[f"mode_{mode_name}"].daemon = True
            self.threads[f"mode_{mode_name}"].start()
    
    def market_data_processor(self):
        """Process market data events."""
        logger.info("Market data processor started")
        market_queue = MessageQueue.get("market_data_queue")
        
        while not self.shutdown_requested.is_set():
            try:
                # Process market data messages
                message = market_queue.get(timeout=1.0)
                if message:
                    # Process market data
                    logger.debug(f"Processing market data: {message.data}")
                    
                    # Publish processed data to market_data channel
                    self.channel_manager.get_channel("market_data").publish(message.data)
            except Exception as e:
                if not self.shutdown_requested.is_set():
                    logger.error(f"Error in market data processor: {e}")
            
            # Simulated work - generate sample market data if queue is empty
            if not self.shutdown_requested.is_set() and market_queue.is_empty():
                self.generate_sample_market_data()
                time.sleep(2)  # Generate sample data every 2 seconds
        
        logger.info("Market data processor stopped")
    
    def signal_processor(self):
        """Process signal events."""
        logger.info("Signal processor started")
        signal_queue = MessageQueue.get("signal_queue")
        
        while not self.shutdown_requested.is_set():
            try:
                # Process signal messages
                message = signal_queue.get(timeout=1.0)
                if message:
                    # Process signal
                    logger.debug(f"Processing signal: {message.data}")
                    
                    # Publish processed signal to signals channel
                    self.channel_manager.get_channel("signals").publish(message.data)
                    
                    # Convert signals to orders
                    self.convert_signal_to_order(message.data)
            except Exception as e:
                if not self.shutdown_requested.is_set():
                    logger.error(f"Error in signal processor: {e}")
            
            time.sleep(0.1)  # Small sleep to prevent CPU thrashing
        
        logger.info("Signal processor stopped")
    
    def order_processor(self):
        """Process order events."""
        logger.info("Order processor started")
        order_queue = MessageQueue.get("order_queue")
        
        while not self.shutdown_requested.is_set():
            try:
                # Process order messages
                message = order_queue.get(timeout=1.0)
                if message:
                    # Process order
                    logger.debug(f"Processing order: {message.data}")
                    
                    # Publish processed order to orders channel
                    self.channel_manager.get_channel("orders").publish(message.data)
                    
                    # Simulate order execution
                    self.simulate_order_execution(message.data)
            except Exception as e:
                if not self.shutdown_requested.is_set():
                    logger.error(f"Error in order processor: {e}")
            
            time.sleep(0.1)  # Small sleep to prevent CPU thrashing
        
        logger.info("Order processor stopped")
    
    def handle_market_data_event(self, event):
        """Handle market data events."""
        logger.debug(f"Received market data event: {event.data}")
        # Put market data in queue for processing
        MessageQueue.get("market_data_queue").put(event)
        return True
    
    def handle_signal_event(self, event):
        """Handle signal events."""
        logger.debug(f"Received signal event: {event.data}")
        # Put signal in queue for processing
        MessageQueue.get("signal_queue").put(event)
        return True
    
    def handle_order_event(self, event):
        """Handle order events."""
        logger.debug(f"Received order event: {event.data}")
        # Put order in queue for processing
        MessageQueue.get("order_queue").put(event)
        return True
    
    def handle_risk_event(self, event):
        """Handle risk events."""
        logger.debug(f"Handling risk event: {event.event_type}")
        # Process risk event
        pass
    
    def _process_news_sentiment(self, data, event_bus):
        """Process news sentiment data for either live or backtest mode."""
        logger.info(f"Processing news sentiment data: {data}")
        
        # Extract sentiment and impact
        sentiment = data.get('sentiment', 0)
        impact = data.get('impact', 0.5)
        symbol = data.get('symbol')
        timestamp = data.get('timestamp')
        headline = data.get('headline', '')
        category = data.get('category', 'general')
        source = data.get('source', 'unknown')
        
        # Categorize sentiment
        sentiment_category = 'neutral'
        if sentiment > 0.3:
            sentiment_category = 'positive'
        elif sentiment < -0.3:
            sentiment_category = 'negative'
        
        # Generate a trading signal based on news sentiment
        signal_value = sentiment * impact
        
        # Prepare the news event for backtesting data collection
        news_event = {
            'headline': headline,
            'date': timestamp,
            'sentiment': sentiment_category,
            'impact': signal_value,
            'symbol': symbol,
            'category': category,
            'source': source
        }
        
        # Send the news event to the backtester event bus
        event_bus.emit('news_event', news_event)
        
        # Only generate trading signals for significant news
        if abs(signal_value) > self.news_threshold:
            signal_data = {
                'source': 'news_sentiment',
                'symbol': symbol,
                'value': signal_value,
                'timestamp': timestamp,
                'metadata': {
                    'headline': headline,
                    'sentiment': sentiment,
                    'impact': impact,
                    'category': category
                }
            }
            
            # Emit the signal event
            event_bus.emit('signal', signal_data)
    
    def handle_news_event(self, event):
        """Handle news events."""
        logger.debug(f"Handling news event: {event.event_type}")
        
        # Extract news data
        news_data = event.data
        
        # If in backtest mode, pass to backtester
        if hasattr(self, 'backtester') and self.backtester.is_active:
            self.backtester.process_news_data(news_data)
        
        # Feed to news sentiment strategy
        if hasattr(self, 'news_sentiment_strategy'):
            self._process_news_sentiment(news_data, self.event_bus)
    
    def handle_backtest_event(self, event):
        """Handle backtest events."""
        logger.debug(f"Handling backtest event: {event.event_type}")
        
        # Process backtest event
        if event.data.get('status') == 'completed':
            # Notify API of backtest completion
            self.channel_manager.get_channel("backtest_results").publish(event.data)
    
    def generate_sample_market_data(self):
        """Generate sample market data for testing."""
        import random
        
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        for symbol in symbols:
            price = round(random.uniform(100, 500), 2)
            volume = random.randint(100, 10000)
            
            # Create market data event
            market_event = Event(
                event_type=EventType.MARKET_DATA,
                data={
                    "symbol": symbol,
                    "price": price,
                    "volume": volume,
                    "timestamp": time.time()
                },
                source="market_data_simulator",
                priority=3
            )
            
            # Publish to event bus
            self.event_bus.publish(market_event)
    
    def convert_signal_to_order(self, signal_data):
        """Convert trading signals to orders."""
        # Create order from signal
        if random.random() > 0.7:  # Only convert some signals to orders
            order_data = {
                "id": f"ord-{int(time.time() * 1000)}",
                "symbol": signal_data.get("symbol", "UNKNOWN"),
                "quantity": random.randint(1, 10) * 10,
                "price": signal_data.get("price", 0),
                "side": signal_data.get("action", "BUY"),
                "type": "MARKET",
                "status": "NEW",
                "timestamp": time.time()
            }
            
            # Create order event
            order_event = Event(
                event_type=EventType.ORDER,
                data=order_data,
                source="signal_processor",
                priority=5  # Orders have higher priority
            )
            
            # Publish to event bus
            self.event_bus.publish(order_event)
    
    def simulate_order_execution(self, order_data):
        """Simulate order execution."""
        # Wait a bit to simulate processing time
        time.sleep(random.uniform(0.5, 2.0))
        
        # Update order status
        order_data["status"] = "FILLED"
        order_data["filled_at"] = time.time()
        
        # Small slippage
        slippage = random.uniform(-0.01, 0.01)
        order_data["executed_price"] = order_data["price"] * (1 + slippage)
        
        # Create execution event
        execution_event = Event(
            event_type=EventType.EXECUTION,
            data=order_data,
            source="order_processor",
            priority=5
        )
        
        # Publish to event bus
        self.event_bus.publish(execution_event)
        
        # Publish to execution channel
        self.channel_manager.get_channel("execution").publish(order_data)
    
    def signal_handler(self, sig, frame):
        """Handle shutdown signals."""
        logger.info(f"Received shutdown signal {sig}")
        self.shutdown()
    
    def shutdown(self):
        """Shutdown all components gracefully."""
        logger.info("Shutting down trading bot...")
        
        # Set shutdown flag
        self.shutdown_requested.set()
        
        # Terminate API server process
        if self.api_server_process and self.api_server_process.is_alive():
            logger.info("Terminating API server")
            self.api_server_process.terminate()
            self.api_server_process.join(timeout=5)
        
        # Wait for all threads to finish
        for name, thread in self.threads.items():
            logger.info(f"Waiting for {name} thread to finish")
            if thread.is_alive():
                thread.join(timeout=5)
        
        # Shutdown event system
        logger.info("Shutting down event system")
        # Any additional cleanup for event system
        
        logger.info("Trading bot shutdown complete")
    
    def news_processor(self):
        """Process news events from news queue."""
        logger.info("Starting news processor thread")
        
        news_queue = MessageQueue.get("news_queue")
        
        while not self.shutdown_requested.is_set():
            try:
                # Generate synthetic news data for testing
                if random.random() < 0.2:  # 20% chance of news event
                    self.generate_sample_news_data()
                
                # Process any news items in the queue
                if not news_queue.empty():
                    news_data = news_queue.get()
                    logger.debug(f"Processing news: {news_data.get('headline', 'Unknown')}")
                    
                    # Pass to news sentiment strategy
                    if hasattr(self, 'news_sentiment_strategy'):
                        self.news_sentiment_strategy.analyze(news_data)
                
                time.sleep(random.uniform(5, 10))  # News comes less frequently
                
            except Exception as e:
                logger.error(f"Error in news processor: {e}")
                time.sleep(1)  # To avoid tight error loop
        
        logger.info("News processor thread stopped")
    
    def backtest_processor(self):
        """Process backtesting events and manage backtest runs."""
        logger.info("Starting backtest processor thread")
        
        backtest_queue = MessageQueue.get("backtest_queue")
        
        while not self.shutdown_requested.is_set():
            try:
                # Process any backtest requests in the queue
                if not backtest_queue.empty():
                    backtest_config = backtest_queue.get()
                    logger.info(f"Starting backtest: {backtest_config.get('id')}")
                    
                    # Run backtest with news sentiment integration
                    self.run_news_sentiment_backtest(backtest_config)
                
                time.sleep(1)  # Check for new backtest requests
                
            except Exception as e:
                logger.error(f"Error in backtest processor: {e}")
                time.sleep(1)  # To avoid tight error loop
        
        logger.info("Backtest processor thread stopped")
    
    def generate_sample_news_data(self):
        """Generate synthetic news data for testing."""
        symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA"]
        sentiment_values = ["positive", "negative", "neutral"]
        impact_levels = ["high", "medium", "low"]
        
        # Create news event
        symbol = random.choice(symbols)
        sentiment = random.choice(sentiment_values)
        impact = random.choice(impact_levels)
        sentiment_score = random.uniform(-1.0, 1.0)
        
        news_data = {
            "id": f"news-{int(time.time() * 1000)}",
            "timestamp": time.time(),
            "headline": f"Sample news for {symbol}",
            "summary": f"This is a synthetic {sentiment} news item for testing.",
            "source": "Trading Bot Test",
            "symbols": [symbol],
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "impact": impact
        }
        
        # Create news event
        news_event = Event(
            event_type=EventType.NEWS,
            data=news_data,
            source="news_generator",
            priority=2
        )
        
        # Publish to event bus
        self.event_bus.publish(news_event)
        
        # Publish to news channel
        self.channel_manager.get_channel("news").publish(news_data)
        
        logger.debug(f"Generated news: {news_data['headline']} ({sentiment})")
    
    def run_news_sentiment_backtest(self, config):
        """Run a backtest focused on news sentiment analysis."""
        logger.info(f"Running news sentiment backtest: {config.get('id')}")
        
        if not hasattr(self, 'backtester') or not hasattr(self, 'news_sentiment_strategy'):
            logger.error("Backtester or news sentiment strategy not initialized")
            return
        
        try:
            # Configure backtester
            start_date = config.get('start_date', '2020-01-01')
            end_date = config.get('end_date', '2024-12-31')
            symbols = config.get('symbols', ['SPY', 'QQQ', 'IWM'])
            
            # Activate backtester
            self.backtester.reset()
            self.backtester.is_active = True
            
            # Set time period
            self.backtester.set_period(start_date, end_date)
            
            # Add symbols
            for symbol in symbols:
                self.backtester.add_symbol(symbol)
            
            # Connect strategy
            self.backtester.add_strategy(self.news_sentiment_strategy)
            
            # Create progress update event
            self.update_backtest_progress(config['id'], 0, 'Initializing backtest')
            
            # Run backtester
            results = self.backtester.run()
            
            # Add additional metadata to results
            results['config'] = config
            results['strategy_type'] = 'news_sentiment'
            results['sentiment_threshold'] = config.get('sentiment_threshold', 0.3)
            results['impact_threshold'] = config.get('impact_threshold', 0.5)
            
            # Process results
            self.update_backtest_progress(config['id'], 100, 'Completed', results)
            
            logger.info(f"Backtest completed: {config.get('id')}")
            
        except Exception as e:
            logger.error(f"Error in backtest run: {e}")
            self.update_backtest_progress(config['id'], -1, f'Error: {str(e)}')
        finally:
            # Deactivate backtester
            self.backtester.is_active = False
    
    def update_backtest_progress(self, backtest_id, progress, stage, results=None):
        """Update backtest progress and publish events."""
        update_data = {
            'id': backtest_id,
            'progress': progress,
            'executionStage': stage,
            'timestamp': time.time(),
            'status': 'running' if progress < 100 else 'completed'
        }
        
        if results:
            update_data['results'] = results
            
            # If completed and successful, check if it should be deployed
            if progress == 100 and self.config and self.config.get('auto_deploy_successful_strategies', False):
                self.evaluate_strategy_for_deployment(backtest_id, results)
        
        # Create event
        progress_event = Event(
            event_type=EventType.BACKTEST,
            data=update_data,
            source="backtest_processor",
            priority=1
        )
        
        # Publish event
        self.event_bus.publish(progress_event)
        
        # Update channel
        self.channel_manager.get_channel("backtest_results").publish(update_data)
    
    def run(self):
        """Run the trading bot system."""
        logger.info("Starting trading bot")
        
        try:
            # Initialize all components
            self.initialize_event_system()
            self.initialize_trading_modes()
            
            # Initialize the deployment pipeline
            self.deployment_pipeline = get_deployment_pipeline(self.event_bus)
            
            # Start API server
            self.start_api_server()
            
            # Start component threads
            self.start_component_threads()
            
            logger.info("Trading bot started successfully")
            
            # Schedule a news sentiment backtest to run immediately
            self.schedule_news_sentiment_backtest()
            
            # Keep main thread alive until shutdown
            while not self.shutdown_requested.is_set():
                time.sleep(1)
        
        except Exception as e:
            logger.error(f"Error in trading bot: {e}")
            self.shutdown()
        
        logger.info("Trading bot stopped")
    
    def schedule_news_sentiment_backtest(self, custom_config=None):
        """Schedule a news sentiment backtest to run."""
        backtest_config = custom_config or {
            'id': f"bt-{int(time.time())}",
            'symbols': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA'],
            'start_date': '2020-01-01',
            'end_date': '2024-12-31',
            'strategy': 'news_sentiment',
            'sentiment_threshold': 0.3,
            'impact_threshold': 0.5,
            'news_decay_hours': 24
        }
        
        # Add to backtest queue
        backtest_queue = MessageQueue.get("backtest_queue")
        backtest_queue.put(backtest_config)
        
        logger.info(f"Scheduled news sentiment backtest: {backtest_config['id']}")
    
    def evaluate_strategy_for_deployment(self, backtest_id, results):
        """Evaluate backtest results and deploy successful strategies to paper trading."""
        try:
            logger.info(f"Evaluating strategy results for potential deployment: {backtest_id}")
            
            # Performance thresholds for automatic deployment
            min_profit_factor = 1.5
            min_win_rate = 0.55
            min_trades = 10
            max_drawdown = -0.15  # -15% max drawdown
            
            # Extract performance metrics
            profit_factor = results.get('profit_factor', 0)
            win_rate = results.get('win_rate', 0)
            trade_count = results.get('trade_count', 0)
            max_drawdown_pct = results.get('max_drawdown_pct', -1)
            total_return = results.get('total_return', 0)
            
            # Run an assessment
            strategy_config = results.get('config', {})
            symbols = strategy_config.get('symbols', [])
            sentiment_threshold = results.get('sentiment_threshold', 0.3)
            impact_threshold = results.get('impact_threshold', 0.5)
            
            # Decide if strategy should be deployed
            should_deploy = (
                profit_factor >= min_profit_factor and 
                win_rate >= min_win_rate and
                trade_count >= min_trades and
                max_drawdown_pct >= max_drawdown and
                total_return > 0
            )
            
            if should_deploy:
                logger.info(f"Strategy {backtest_id} meets deployment criteria - deploying to paper trading")
                
                # Create deployment ID
                deployment_id = f"auto_deploy_{backtest_id}_{int(time.time())}"
                
                # Create strategy configuration
                strategy_config = {
                    "sentiment_threshold": sentiment_threshold,
                    "impact_threshold": impact_threshold,
                    "position_sizing_factor": 1.0,  # Start conservative
                    "news_decay_hours": strategy_config.get('news_decay_hours', 24),
                    "symbols": symbols,
                    "news_categories": ["earnings", "analyst", "regulatory", "product", "general"]
                }
                
                # Create strategy ID
                strategy_id = f"news_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Create the strategy instance
                strategy = DeployableNewsSentimentStrategy(
                    name="NewsSentiment_Auto",
                    config=strategy_config,
                    event_bus=self.event_bus
                )
                
                # Determine risk level based on performance
                if profit_factor > 2.0 and win_rate > 0.65:
                    risk_level = RiskLevel.MEDIUM
                    allocation_percentage = 5.0
                else:
                    risk_level = RiskLevel.LOW
                    allocation_percentage = 2.0
                
                # Set stop loss type
                stop_loss_type = StopLossType.VOLATILITY
                
                # Create deployment metadata
                metadata = {
                    "creator": "auto_deployment_pipeline",
                    "symbols": symbols,
                    "sentiment_threshold": sentiment_threshold,
                    "impact_threshold": impact_threshold,
                    "backtest_id": backtest_id,
                    "backtest_results": {
                        "profit_factor": profit_factor,
                        "win_rate": win_rate,
                        "trade_count": trade_count,
                        "max_drawdown_pct": max_drawdown_pct,
                        "total_return": total_return
                    }
                }
                
                # Deploy the strategy
                try:
                    logger.info(f"Deploying News Sentiment Strategy with {len(symbols)} symbols")
                    deployment_id = self.deployment_pipeline.deploy_strategy(
                        strategy_id=strategy_id,
                        strategy=strategy,
                        allocation_percentage=allocation_percentage,
                        risk_level=risk_level,
                        stop_loss_type=stop_loss_type,
                        metadata=metadata
                    )
                    
                    logger.info(f"Strategy automatically deployed with ID: {deployment_id}")
                    
                    # Create event to notify UI
                    deployment_event = Event(
                        event_type="STRATEGY_DEPLOYMENT",
                        data={
                            "id": deployment_id,
                            "strategy_id": strategy_id,
                            "status": "deployed",
                            "timestamp": time.time(),
                            "metadata": metadata
                        },
                        source="auto_deployment_pipeline",
                        priority=1
                    )
                    
                    # Publish event
                    self.event_bus.publish(deployment_event)
                    
                    return deployment_id
                    
                except Exception as e:
                    logger.error(f"Error in auto-deployment: {e}")
                    return None
            else:
                logger.info(f"Strategy {backtest_id} does not meet deployment criteria - not deploying")
                reasons = []
                if profit_factor < min_profit_factor:
                    reasons.append(f"Profit factor too low: {profit_factor:.2f} < {min_profit_factor}")
                if win_rate < min_win_rate:
                    reasons.append(f"Win rate too low: {win_rate:.2f} < {min_win_rate}")
                if trade_count < min_trades:
                    reasons.append(f"Not enough trades: {trade_count} < {min_trades}")
                if max_drawdown_pct < max_drawdown:
                    reasons.append(f"Drawdown too high: {max_drawdown_pct:.2f} < {max_drawdown}")
                if total_return <= 0:
                    reasons.append(f"Negative or zero return: {total_return:.2f}")
                
                logger.info(f"Deployment prevented due to: {', '.join(reasons)}")
                return None
                
        except Exception as e:
            logger.error(f"Error evaluating strategy for deployment: {e}")
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Bot Orchestrator")
    parser.add_argument("--config", help="Path to configuration file")
    args = parser.parse_args()
    
    orchestrator = TradingBotOrchestrator(config_path=args.config)
    orchestrator.run()
