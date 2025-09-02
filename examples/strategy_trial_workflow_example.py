#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Trial Workflow Example

This example demonstrates the complete strategy trial workflow:
1. New strategies start in paper trading mode automatically
2. Performance is tracked and evaluated against criteria
3. Strategies that perform well can be promoted to live trading
4. The workflow manages status changes and notifications
"""

import os
import sys
import time
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from trading_bot.brokers.multi_broker_manager import MultiBrokerManager
from trading_bot.brokers.paper_broker import PaperBroker
from trading_bot.brokers.paper_broker_factory import create_paper_broker
from trading_bot.core.strategy_broker_router import StrategyBrokerRouter
from trading_bot.core.performance_tracker import PerformanceTracker
from trading_bot.core.strategy_trial_workflow import (
    StrategyTrialWorkflow, StrategyStatus, TrialPhase, PromotionCriteria
)
from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_event_listeners(event_bus):
    """Set up event listeners for workflow events."""
    
    def on_workflow_event(event):
        """Handler for workflow events."""
        data = event.data
        event_type = data.get("workflow_event", "unknown")
        strategy_id = data.get("strategy_id", "unknown")
        details = data.get("details", {})
        
        logger.info(f"Workflow event: {event_type} for strategy {strategy_id}")
        logger.info(f"Details: {json.dumps(details, indent=2)}")
    
    # Register event handlers
    event_bus.subscribe(EventType.WORKFLOW_EVENT, on_workflow_event)


def setup_test_environment():
    """Set up the testing environment with required components."""
    # Create directories
    os.makedirs("data/performance", exist_ok=True)
    os.makedirs("data/workflow", exist_ok=True)
    
    # Create event bus and register with service registry
    event_bus = EventBus()
    service_registry = ServiceRegistry.get_instance()
    service_registry.register_service(EventBus, event_bus)
    
    # Set up event listeners
    setup_event_listeners(event_bus)
    
    # Create broker manager
    broker_manager = MultiBrokerManager()
    
    # Create a dummy broker for data
    class DummyDataSource:
        def get_quote(self, symbol):
            import random
            price = round(random.uniform(100, 500), 2)
            return {
                "symbol": symbol,
                "bid": price - 0.01,
                "ask": price + 0.01,
                "last": price,
                "volume": int(random.uniform(1000, 10000)),
                "timestamp": datetime.now().isoformat()
            }
    
    # Create paper broker configuration
    paper_config = {
        "name": "TestPaperBroker",
        "id": "paper",
        "initial_balance": 100000.0,
        "slippage_model": {"type": "fixed", "basis_points": 3},
        "commission_model": {"type": "per_share", "per_share": 0.005, "minimum": 1.0}
    }
    
    # Create live broker configuration
    live_config = {
        "name": "TestLiveBroker",
        "id": "live",
        "initial_balance": 100000.0
    }
    
    # Create brokers and add to manager
    data_source = DummyDataSource()
    paper_broker = create_paper_broker(paper_config, data_source)
    # Live broker would be a real implementation in production
    live_broker = create_paper_broker(live_config, data_source)
    
    broker_manager.add_broker("paper", paper_broker, None, False)
    broker_manager.add_broker("live", live_broker, None, True)
    
    # Create strategy broker router
    router = StrategyBrokerRouter(
        broker_manager=broker_manager,
        default_paper_broker_id="paper",
        default_live_broker_id="live"
    )
    
    # Create performance tracker
    performance_tracker = PerformanceTracker(
        data_dir="data/performance",
        lookback_days=90,
        save_interval_minutes=5
    )
    
    # Create promotion criteria
    criteria = PromotionCriteria(
        min_trading_days=5,  # Reduced for example
        min_trades=10,       # Reduced for example
        min_profit_factor=1.2,
        min_win_rate=0.55,
        min_sharpe_ratio=0.8,
        max_drawdown_pct=-10.0
    )
    
    # Create strategy trial workflow
    workflow = StrategyTrialWorkflow(
        broker_router=router,
        performance_tracker=performance_tracker,
        promotion_criteria=criteria,
        workflow_config_path="data/workflow/workflow_config.json"
    )
    
    return {
        "broker_manager": broker_manager,
        "router": router,
        "performance_tracker": performance_tracker,
        "workflow": workflow,
        "paper_broker": paper_broker,
        "live_broker": live_broker,
        "event_bus": event_bus
    }


class TestStrategy:
    """Simple test strategy for the example."""
    
    def __init__(self, strategy_id, symbols=None, timeframe="1h"):
        self.strategy_id = strategy_id
        self.symbols = symbols or ["AAPL", "MSFT", "GOOG"]
        self.timeframe = timeframe
        self._tags = []
    
    def get_id(self):
        """Get strategy ID."""
        return self.strategy_id
    
    def get_symbols(self):
        """Get symbols traded by this strategy."""
        return self.symbols
    
    def add_tag(self, tag):
        """Add a tag to the strategy."""
        if tag not in self._tags:
            self._tags.append(tag)
    
    def get_tags(self):
        """Get all tags."""
        return self._tags


def create_test_strategies():
    """Create test strategies for the example."""
    strategies = {
        "moving_avg_cross": TestStrategy(
            strategy_id="moving_avg_cross",
            symbols=["AAPL", "MSFT", "AMZN"],
            timeframe="1h"
        ),
        "rsi_strategy": TestStrategy(
            strategy_id="rsi_strategy",
            symbols=["TSLA", "NFLX", "META"],
            timeframe="1d"
        ),
        "bollingerband_strategy": TestStrategy(
            strategy_id="bollingerband_strategy",
            symbols=["SPY", "QQQ", "DIA"],
            timeframe="4h"
        )
    }
    
    return strategies


def create_strategy_configs():
    """Create strategy configurations."""
    configs = {
        "moving_avg_cross": {
            "name": "Moving Average Crossover Strategy",
            "id": "moving_avg_cross",
            "description": "Simple MA crossover strategy using 10 and 30 period MAs",
            "symbols": ["AAPL", "MSFT", "AMZN"],
            "timeframe": "1h",
            "parameters": {
                "fast_ma": 10,
                "slow_ma": 30,
                "volume_factor": 1.5
            },
            "risk": {
                "max_position_size": 0.05,
                "stop_loss_pct": 0.02
            }
        },
        "rsi_strategy": {
            "name": "RSI Mean Reversion Strategy",
            "id": "rsi_strategy",
            "description": "RSI mean reversion strategy with 30/70 thresholds",
            "symbols": ["TSLA", "NFLX", "META"],
            "timeframe": "1d",
            "parameters": {
                "rsi_period": 14,
                "overbought": 70,
                "oversold": 30
            },
            "risk": {
                "max_position_size": 0.04,
                "stop_loss_pct": 0.025
            }
        },
        "bollingerband_strategy": {
            "name": "Bollinger Band Strategy",
            "id": "bollingerband_strategy",
            "description": "Bollinger Band breakout/reversion strategy",
            "symbols": ["SPY", "QQQ", "DIA"],
            "timeframe": "4h",
            "parameters": {
                "bb_period": 20,
                "bb_std": 2.0,
                "mean_reversion": True
            },
            "risk": {
                "max_position_size": 0.03,
                "stop_loss_pct": 0.015
            }
        }
    }
    
    return configs


def simulate_strategy_trades(performance_tracker, strategy_id, num_trades=20, win_rate=0.6):
    """Simulate trades for a strategy to generate performance data."""
    logger.info(f"Simulating {num_trades} trades for strategy {strategy_id} with {win_rate*100:.0f}% win rate")
    
    import random
    
    # Get the strategy's symbols
    test_strategies = create_test_strategies()
    strategy = test_strategies.get(strategy_id)
    if not strategy:
        logger.error(f"Strategy {strategy_id} not found")
        return
    
    symbols = strategy.get_symbols()
    
    # Simulate trades over the past 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Generate trade dates (random but ordered)
    trade_dates = []
    current_date = start_date
    while current_date < end_date:
        if random.random() < 0.3:  # 30% chance of a trade on any day
            trade_dates.append(current_date)
        current_date += timedelta(days=1)
    
    # Ensure we have enough trade dates
    while len(trade_dates) < num_trades:
        # Add random time during trading hours
        random_day = random.randint(0, 29)
        random_hour = random.randint(9, 16)
        random_minute = random.randint(0, 59)
        
        trade_date = start_date + timedelta(days=random_day, hours=random_hour, minutes=random_minute)
        if trade_date not in trade_dates:
            trade_dates.append(trade_date)
    
    # Sort dates
    trade_dates.sort()
    
    # Generate trades
    for i, trade_date in enumerate(trade_dates[:num_trades]):
        # Pick a random symbol
        symbol = random.choice(symbols)
        
        # Determine if this is a winning trade
        is_win = random.random() < win_rate
        
        # Generate realistic prices
        base_price = random.uniform(100, 500)
        
        if is_win:
            # Buy low, sell high
            buy_price = base_price
            sell_price = base_price * (1 + random.uniform(0.01, 0.05))  # 1-5% gain
            
            # Record buy trade
            performance_tracker.record_trade(
                strategy_id=strategy_id,
                symbol=symbol,
                side="buy",
                quantity=100,
                price=buy_price,
                timestamp=trade_date,
                order_id=f"paper-buy-{i}",
                tags=["PAPER", strategy_id]
            )
            
            # Record sell trade a day later
            performance_tracker.record_trade(
                strategy_id=strategy_id,
                symbol=symbol,
                side="sell",
                quantity=100,
                price=sell_price,
                timestamp=trade_date + timedelta(days=1),
                order_id=f"paper-sell-{i}",
                tags=["PAPER", strategy_id]
            )
        else:
            # Buy high, sell low (loss)
            buy_price = base_price
            sell_price = base_price * (1 - random.uniform(0.01, 0.03))  # 1-3% loss
            
            # Record buy trade
            performance_tracker.record_trade(
                strategy_id=strategy_id,
                symbol=symbol,
                side="buy",
                quantity=100,
                price=buy_price,
                timestamp=trade_date,
                order_id=f"paper-buy-{i}",
                tags=["PAPER", strategy_id]
            )
            
            # Record sell trade a day later
            performance_tracker.record_trade(
                strategy_id=strategy_id,
                symbol=symbol,
                side="sell",
                quantity=100,
                price=sell_price,
                timestamp=trade_date + timedelta(days=1),
                order_id=f"paper-sell-{i}",
                tags=["PAPER", strategy_id]
            )
    
    logger.info(f"Completed simulating trades for strategy {strategy_id}")


def run_strategy_trial_example():
    """Run the complete strategy trial workflow example."""
    logger.info("Starting Strategy Trial Workflow Example")
    
    # Setup test environment
    env = setup_test_environment()
    workflow = env["workflow"]
    performance_tracker = env["performance_tracker"]
    
    # Create test strategies and configs
    strategies = create_test_strategies()
    configs = create_strategy_configs()
    
    # Register strategies with workflow
    for strategy_id, strategy in strategies.items():
        config = configs[strategy_id]
        workflow.register_new_strategy(strategy, config)
        logger.info(f"Registered strategy: {strategy_id}")
    
    # Update statuses to paper testing
    for strategy_id in strategies:
        workflow.update_strategy_status(
            strategy_id=strategy_id,
            status=StrategyStatus.PAPER_TESTING,
            remarks="Initial paper testing phase"
        )
    
    # Simulate trades for each strategy with different performance
    # Moving average: good performance (promote candidate)
    simulate_strategy_trades(
        performance_tracker=performance_tracker,
        strategy_id="moving_avg_cross",
        num_trades=15,
        win_rate=0.75  # 75% win rate (good)
    )
    
    # RSI: medium performance (continue testing)
    simulate_strategy_trades(
        performance_tracker=performance_tracker,
        strategy_id="rsi_strategy",
        num_trades=12,
        win_rate=0.5  # 50% win rate (marginal)
    )
    
    # Bollinger: poor performance (reject candidate)
    simulate_strategy_trades(
        performance_tracker=performance_tracker,
        strategy_id="bollingerband_strategy",
        num_trades=10,
        win_rate=0.3  # 30% win rate (poor)
    )
    
    # Save performance data
    performance_tracker.save_performance_data()
    
    # Wait for data to be processed
    logger.info("Waiting for performance data to be processed...")
    time.sleep(2)
    
    # Evaluate performance for each strategy
    logger.info("\n===== STRATEGY EVALUATIONS =====")
    for strategy_id in strategies:
        evaluation = workflow.evaluate_performance(strategy_id)
        
        logger.info(f"\nStrategy: {strategy_id}")
        logger.info(f"Status: {evaluation.get('status')}")
        logger.info(f"Metrics: {json.dumps({k: v for k, v in evaluation.get('metrics', {}).items() if k not in ['error']}, indent=2)}")
        logger.info(f"Meets criteria: {evaluation.get('meets_all_criteria')}")
        logger.info(f"Recommendation: {evaluation.get('recommendation')}")
    
    # Process workflow changes based on evaluations
    # 1. MA Crossover - Approve and promote to live
    ma_eval = workflow.evaluate_performance("moving_avg_cross")
    if ma_eval.get("meets_all_criteria", False):
        logger.info("\n===== PROMOTING SUCCESSFUL STRATEGY =====")
        # First approve
        workflow.update_strategy_status(
            strategy_id="moving_avg_cross",
            status=StrategyStatus.APPROVED,
            remarks="Strategy met all performance criteria"
        )
        
        # Then promote to live with position limits
        workflow.promote_to_live(
            strategy_id="moving_avg_cross",
            live_broker_id="live",
            position_limit_pct=0.02,  # Start with 2% position limit
            approved_by="example_user",
            remarks="Promoted to live with 2% position limit"
        )
    
    # 2. RSI - Continue paper testing
    rsi_eval = workflow.evaluate_performance("rsi_strategy")
    if not rsi_eval.get("meets_all_criteria", False):
        logger.info("\n===== CONTINUING PAPER TESTING =====")
        workflow.update_strategy_phase(
            strategy_id="rsi_strategy",
            phase=TrialPhase.PAPER_TRADE,
            remarks="Continue paper testing, performance is marginal"
        )
    
    # 3. Bollinger - Reject
    bb_eval = workflow.evaluate_performance("bollingerband_strategy")
    if not bb_eval.get("meets_all_criteria", False):
        logger.info("\n===== REJECTING POOR PERFORMER =====")
        workflow.update_strategy_status(
            strategy_id="bollingerband_strategy",
            status=StrategyStatus.REJECTED,
            remarks="Strategy performance is below acceptable thresholds"
        )
    
    # Save workflow state
    workflow.save_workflow_state("data/workflow/workflow_state.json")
    
    # Display final status
    logger.info("\n===== FINAL WORKFLOW STATE =====")
    for strategy_id in strategies:
        metadata = workflow.get_strategy_metadata(strategy_id)
        logger.info(f"\nStrategy: {metadata.get('name')}")
        logger.info(f"Status: {metadata.get('status')}")
        logger.info(f"Phase: {metadata.get('phase')}")
        if metadata.get('status') == StrategyStatus.LIVE.value:
            logger.info(f"Live with broker: {metadata.get('config', {}).get('broker')}")
            logger.info(f"Position limit: {metadata.get('config', {}).get('risk', {}).get('position_limit_pct')}%")
    
    logger.info("\nStrategy Trial Workflow Example completed successfully")


if __name__ == "__main__":
    run_strategy_trial_example()
