#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deploy News Sentiment Strategy

This script automates the deployment of the news sentiment trading strategy
to paper trading accounts with proper risk parameters.
"""

import os
import sys
import logging
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core components
from trading_bot.event_system import EventBus, Event, EventType
from trading_bot.strategies.news_sentiment_strategy import NewsSentimentStrategy
from trading_bot.autonomous.strategy_deployment_pipeline import (
    get_deployment_pipeline, 
    StrategyDeployment, 
    DeploymentStatus
)
from trading_bot.risk.risk_manager import RiskLevel, StopLossType
from trading_bot.brokers.paper_broker import PaperBroker
from trading_bot.brokers.broker_factory import create_broker, get_broker_types

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("deployment")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Deploy News Sentiment Strategy to Paper Trading")
    
    parser.add_argument(
        "--allocation", 
        type=float, 
        default=5.0,
        help="Percentage of capital to allocate (default: 5.0)"
    )
    
    parser.add_argument(
        "--risk-level", 
        choices=['low', 'medium', 'high'],
        default='medium',
        help="Risk level for the strategy (default: medium)"
    )
    
    parser.add_argument(
        "--stop-loss", 
        choices=['fixed', 'volatility', 'trailing'], 
        default='volatility',
        help="Stop loss type to use (default: volatility)"
    )
    
    parser.add_argument(
        "--sentiment-threshold", 
        type=float, 
        default=0.3,
        help="Minimum sentiment value to generate signals (default: 0.3)"
    )
    
    parser.add_argument(
        "--impact-threshold", 
        type=float, 
        default=0.5,
        help="Minimum impact value to generate signals (default: 0.5)"
    )
    
    parser.add_argument(
        "--symbols", 
        type=str,
        default="AAPL,MSFT,AMZN,GOOGL,META,TSLA",
        help="Comma-separated list of symbols to watch (default: AAPL,MSFT,AMZN,GOOGL,META,TSLA)"
    )
    
    parser.add_argument(
        "--news-decay-hours", 
        type=int, 
        default=24,
        help="Hours before news impact decays (default: 24)"
    )
    
    parser.add_argument(
        "--broker", 
        type=str, 
        default="paper",
        help=f"Broker to use for trading (default: paper)"
    )
    
    parser.add_argument(
        "--account", 
        type=str, 
        default="default",
        help="Broker account to use"
    )
    
    parser.add_argument(
        "--monitor", 
        action="store_true",
        help="Monitor strategy after deployment"
    )
    
    return parser.parse_args()

def deploy_strategy(args):
    """Deploy the news sentiment strategy with the specified parameters"""
    # Initialize the event bus
    event_bus = EventBus()
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(",")]
    
    # Convert arguments to enums
    risk_level_map = {
        "low": RiskLevel.LOW,
        "medium": RiskLevel.MEDIUM,
        "high": RiskLevel.HIGH
    }
    
    stop_loss_map = {
        "fixed": StopLossType.FIXED,
        "volatility": StopLossType.VOLATILITY,
        "trailing": StopLossType.TRAILING
    }
    
    risk_level = risk_level_map[args.risk_level]
    stop_loss_type = stop_loss_map[args.stop_loss]
    
    # Create strategy configuration
    strategy_config = {
        "sentiment_threshold": args.sentiment_threshold,
        "impact_threshold": args.impact_threshold,
        "position_sizing_factor": args.allocation / 5.0,  # Scale sizing
        "news_decay_hours": args.news_decay_hours,
        "symbols": symbols,
        "news_categories": ["earnings", "analyst", "regulatory", "product", "general"]
    }
    
    # Create the strategy
    strategy = NewsSentimentStrategy(
        name="NewsSentiment",
        config=strategy_config,
        event_bus=event_bus
    )
    
    # Create strategy ID
    strategy_id = f"news_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Save strategy details
    strategy_details = {
        "id": strategy_id,
        "name": "News Sentiment Strategy",
        "description": f"Trading strategy based on news sentiment analysis for {len(symbols)} symbols",
        "type": "news_sentiment",
        "config": strategy_config,
        "created_at": datetime.now().isoformat()
    }
    
    # Get deployment pipeline
    deployment_pipeline = get_deployment_pipeline(event_bus)
    
    # Create deployment config
    deployment_config = {
        "broker": args.broker,
        "account": args.account,
        "allocation_percentage": args.allocation,
        "risk_level": risk_level,
        "stop_loss_type": stop_loss_type,
        "metadata": {
            "creator": "news_sentiment_deployer",
            "symbols": symbols,
            "sentiment_threshold": args.sentiment_threshold
        }
    }
    
    # Deploy the strategy
    try:
        logger.info(f"Deploying News Sentiment Strategy with {len(symbols)} symbols")
        deployment_id = deployment_pipeline.deploy_strategy(
            strategy_id=strategy_id,
            allocation_percentage=args.allocation,
            risk_level=risk_level,
            stop_loss_type=stop_loss_type,
            metadata=deployment_config["metadata"]
        )
        
        logger.info(f"Strategy deployed successfully with ID: {deployment_id}")
        return deployment_id, strategy
        
    except Exception as e:
        logger.error(f"Error deploying strategy: {e}")
        return None, strategy

def monitor_strategy(deployment_id, strategy, event_bus):
    """Monitor the deployed strategy and display real-time information"""
    logger.info("Monitoring strategy. Press Ctrl+C to exit.")
    
    # Subscribe to relevant events
    def on_signal_event(event):
        data = event.data
        logger.info(f"SIGNAL: {data.get('direction', '?')} {data.get('symbol', '?')} - "
                   f"Size: {data.get('size', '?')}, Confidence: {data.get('confidence', '?')}")
    
    def on_order_event(event):
        data = event.data
        logger.info(f"ORDER: {data.get('status', '?')} {data.get('side', '?')} "
                   f"{data.get('symbol', '?')} - {data.get('quantity', '?')} @ {data.get('price', '?')}")
    
    def on_news_event(event):
        data = event.data
        logger.info(f"NEWS: {data.get('headline', '?')} - {data.get('symbol', '?')} - "
                   f"Sentiment: {data.get('sentiment', '?')}, Impact: {data.get('impact', '?')}")
    
    # Register event handlers
    event_bus.subscribe(EventType.SIGNAL, on_signal_event)
    event_bus.subscribe(EventType.ORDER, on_order_event)
    event_bus.subscribe(EventType.NEWS, on_news_event)
    event_bus.subscribe("news_event", on_news_event)
    
    # Keep running until keyboard interrupt
    try:
        from time import sleep
        while True:
            sleep(1)
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")

def main():
    """Main entry point"""
    args = parse_arguments()
    logger.info("Starting News Sentiment Strategy Deployment")
    
    # Deploy the strategy
    deployment_id, strategy = deploy_strategy(args)
    
    if deployment_id:
        logger.info(f"News Sentiment Strategy deployed with ID: {deployment_id}")
        
        # Monitor if requested
        if args.monitor:
            monitor_strategy(deployment_id, strategy, strategy.event_bus)
    else:
        logger.error("Failed to deploy News Sentiment Strategy")
        sys.exit(1)

if __name__ == "__main__":
    main()
