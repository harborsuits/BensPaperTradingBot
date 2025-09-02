#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Approval and Go-Live Example

Demonstrates the complete workflow for approving a strategy to transition from
paper trading to live trading, including:
1. Requesting approval for a paper-traded strategy
2. Handling open positions during transition
3. Approving the strategy for live trading
4. Initializing the strategy with proper live trading parameters
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
from trading_bot.core.strategy_approval_manager import (
    StrategyApprovalManager, PositionTransitionMode, ApprovalResult
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
    """Set up event listeners for approval events."""
    
    def on_approval_event(event):
        """Handler for approval events."""
        data = event.data
        event_type = data.get("approval_event", "unknown")
        strategy_id = data.get("strategy_id", "unknown")
        details = data.get("details", {})
        
        logger.info(f"Approval event: {event_type} for strategy {strategy_id}")
        logger.info(f"Details: {json.dumps(details, indent=2)}")
    
    # Register event handlers
    event_bus.subscribe(EventType.APPROVAL_EVENT, on_approval_event)


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
    
    # Create strategy approval manager
    approval_manager = StrategyApprovalManager(
        workflow=workflow,
        router=router,
        performance_tracker=performance_tracker,
        default_transition_mode=PositionTransitionMode.CLOSE_PAPER_START_FLAT,
        approval_required=True,
        default_position_limit_pct=0.02  # 2% position limit for newly approved strategies
    )
    
    return {
        "broker_manager": broker_manager,
        "router": router,
        "performance_tracker": performance_tracker,
        "workflow": workflow,
        "approval_manager": approval_manager,
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
        "rsi_pullback": TestStrategy(
            strategy_id="rsi_pullback",
            symbols=["AAPL", "MSFT", "AMZN"],
            timeframe="1h"
        ),
        "breakout_strategy": TestStrategy(
            strategy_id="breakout_strategy", 
            symbols=["TSLA", "NFLX", "META"],
            timeframe="1d"
        ),
        "mean_reversion": TestStrategy(
            strategy_id="mean_reversion",
            symbols=["SPY", "QQQ", "DIA"],
            timeframe="4h"
        )
    }
    
    return strategies


def create_strategy_configs():
    """Create strategy configurations."""
    configs = {
        "rsi_pullback": {
            "name": "RSI Pullback Strategy",
            "id": "rsi_pullback",
            "description": "Buys oversold pullbacks in uptrending securities",
            "symbols": ["AAPL", "MSFT", "AMZN"],
            "timeframe": "1h",
            "parameters": {
                "rsi_period": 14,
                "oversold_threshold": 30,
                "ma_period": 50
            },
            "risk": {
                "max_position_size": 0.05,
                "stop_loss_pct": 0.02
            }
        },
        "breakout_strategy": {
            "name": "Volatility Breakout Strategy",
            "id": "breakout_strategy",
            "description": "Trades breakouts from price consolidation",
            "symbols": ["TSLA", "NFLX", "META"],
            "timeframe": "1d",
            "parameters": {
                "atr_period": 14,
                "breakout_multiple": 1.5,
                "consolidation_days": 7
            },
            "risk": {
                "max_position_size": 0.04,
                "stop_loss_pct": 0.025
            }
        },
        "mean_reversion": {
            "name": "Mean Reversion Strategy",
            "id": "mean_reversion",
            "description": "Fades extreme moves expecting reversion to the mean",
            "symbols": ["SPY", "QQQ", "DIA"],
            "timeframe": "4h",
            "parameters": {
                "bollinger_period": 20,
                "bollinger_std": 2.0,
                "min_volume_multiple": 1.5
            },
            "risk": {
                "max_position_size": 0.03,
                "stop_loss_pct": 0.015
            }
        }
    }
    
    return configs


def simulate_strategy_trades(performance_tracker, strategy_id, num_trades=20, win_rate=0.6, create_open_positions=False):
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
    for i, trade_date in enumerate(trade_dates[:num_trades-1]):  # Leave one slot for potential open position
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
    
    # Optionally create an open position
    if create_open_positions:
        # Pick a random symbol
        symbol = random.choice(symbols)
        base_price = random.uniform(100, 500)
        
        # Record buy trade with no matching sell (open position)
        performance_tracker.record_trade(
            strategy_id=strategy_id,
            symbol=symbol,
            side="buy",
            quantity=100,
            price=base_price,
            timestamp=trade_dates[-1],  # Use the last date
            order_id=f"paper-buy-open",
            tags=["PAPER", strategy_id]
        )
        
        # Update position
        performance_tracker.update_position(
            strategy_id=strategy_id,
            symbol=symbol,
            quantity=100,
            average_price=base_price,
            current_price=base_price * 1.01,  # Slight profit
            unrealized_pl=100 * base_price * 0.01  # 1% unrealized gain
        )
        
        logger.info(f"Created open position for {strategy_id}: 100 shares of {symbol} @ {base_price}")
    
    logger.info(f"Completed simulating trades for strategy {strategy_id}")


def simulate_approval_workflow():
    """Simulate the approval workflow for strategies."""
    logger.info("Starting Strategy Approval Example")
    
    # Setup test environment
    env = setup_test_environment()
    workflow = env["workflow"]
    performance_tracker = env["performance_tracker"]
    approval_manager = env["approval_manager"]
    
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
    
    # Simulate trades for each strategy with different performance characteristics
    # RSI Pullback - Good performer with open positions
    simulate_strategy_trades(
        performance_tracker=performance_tracker,
        strategy_id="rsi_pullback",
        num_trades=15,
        win_rate=0.75,  # 75% win rate (good)
        create_open_positions=True  # Has open positions
    )
    
    # Breakout - Good performer with no open positions
    simulate_strategy_trades(
        performance_tracker=performance_tracker,
        strategy_id="breakout_strategy",
        num_trades=12,
        win_rate=0.67,  # 67% win rate (good)
        create_open_positions=False  # No open positions
    )
    
    # Mean Reversion - Poor performer
    simulate_strategy_trades(
        performance_tracker=performance_tracker,
        strategy_id="mean_reversion",
        num_trades=10,
        win_rate=0.40,  # 40% win rate (poor)
        create_open_positions=False
    )
    
    # Save performance data
    performance_tracker.save_performance_data()
    
    # Wait for data to be processed
    logger.info("Waiting for performance data to be processed...")
    time.sleep(2)
    
    # Display strategies eligible for approval
    logger.info("\n===== STRATEGIES ELIGIBLE FOR APPROVAL =====")
    eligible_strategies = approval_manager.get_strategies_eligible_for_approval()
    for strategy in eligible_strategies:
        logger.info(f"Strategy: {strategy.get('name')} (ID: {strategy.get('id')})")
    
    # Approval Request Examples
    logger.info("\n===== APPROVAL REQUEST EXAMPLES =====")
    
    # Example 1: Request approval for RSI Pullback (has open positions)
    logger.info("\n1. Request approval for RSI Pullback (has open positions):")
    rsi_result = approval_manager.request_approval(
        strategy_id="rsi_pullback",
        requested_by="trader@example.com",
        notes="Strategy has shown consistent performance over 30 days"
    )
    logger.info(f"Approval request result: {rsi_result.message}")
    
    # Example 2: Approve RSI Pullback with CLOSE_PAPER_START_FLAT mode
    logger.info("\n2. Approve RSI Pullback with CLOSE_PAPER_START_FLAT transition mode:")
    approval_result = approval_manager.approve_strategy(
        strategy_id="rsi_pullback",
        approved_by="risk_manager@example.com",
        transition_mode=PositionTransitionMode.CLOSE_PAPER_START_FLAT,
        position_limit_pct=0.02,
        notes="Approved with initial 2% position limit. All paper positions will be closed."
    )
    logger.info(f"Approval result: {approval_result.message}")
    logger.info(f"Details: {json.dumps(approval_result.details, indent=2)}")
    
    # Example 3: Request approval for Breakout Strategy (no open positions)
    logger.info("\n3. Request approval for Breakout Strategy (no open positions):")
    breakout_result = approval_manager.request_approval(
        strategy_id="breakout_strategy",
        requested_by="trader@example.com",
        notes="Strategy ready for live trading, performance exceeds benchmarks"
    )
    logger.info(f"Approval request result: {breakout_result.message}")
    
    # Example 4: Approve Breakout Strategy with WAIT_FOR_FLAT mode
    logger.info("\n4. Approve Breakout Strategy with WAIT_FOR_FLAT transition mode:")
    approval_result = approval_manager.approve_strategy(
        strategy_id="breakout_strategy",
        approved_by="risk_manager@example.com",
        transition_mode=PositionTransitionMode.WAIT_FOR_FLAT,
        position_limit_pct=0.03,  # 3% position limit
        notes="Approved with 3% position limit. No open positions to handle."
    )
    logger.info(f"Approval result: {approval_result.message}")
    logger.info(f"Details: {json.dumps(approval_result.details, indent=2)}")
    
    # Example 5: Request approval for Mean Reversion (poor performer)
    logger.info("\n5. Request approval for Mean Reversion (poor performer):")
    mean_rev_result = approval_manager.request_approval(
        strategy_id="mean_reversion",
        requested_by="trader@example.com",
        notes="Strategy ready for live trading despite mixed results"
    )
    logger.info(f"Approval request result: {mean_rev_result.message}")
    
    # Example 6: Reject Mean Reversion approval
    logger.info("\n6. Reject Mean Reversion approval:")
    rejection_result = approval_manager.reject_approval(
        strategy_id="mean_reversion",
        rejected_by="risk_manager@example.com",
        reason="Performance does not meet required threshold. Win rate below 50%."
    )
    logger.info(f"Rejection result: {rejection_result.message}")
    
    # Display approval history
    logger.info("\n===== APPROVAL HISTORY =====")
    approval_history = approval_manager.get_approval_history()
    for entry in approval_history:
        status = entry.get("status", "unknown")
        strategy_id = entry.get("strategy_id", "unknown")
        requested_by = entry.get("requested_by", "unknown")
        
        if status == "approved":
            approved_by = entry.get("approved_by", "unknown")
            approved_at = entry.get("approved_at", "unknown")
            logger.info(f"Strategy {strategy_id}: APPROVED by {approved_by} at {approved_at}")
        elif status == "rejected":
            rejected_by = entry.get("rejected_by", "unknown")
            rejected_at = entry.get("rejected_at", "unknown")
            reason = entry.get("rejection_reason", "No reason provided")
            logger.info(f"Strategy {strategy_id}: REJECTED by {rejected_by} at {rejected_at}")
            logger.info(f"  Reason: {reason}")
    
    # Check final status of strategies
    logger.info("\n===== FINAL STRATEGY STATUS =====")
    for strategy_id in strategies:
        metadata = workflow.get_strategy_metadata(strategy_id)
        logger.info(f"Strategy: {metadata.get('name')} (ID: {strategy_id})")
        logger.info(f"  Status: {metadata.get('status')}")
        logger.info(f"  Phase: {metadata.get('phase')}")
        
        # For live strategies, show broker and position limits
        if metadata.get('status') == StrategyStatus.LIVE.value:
            broker_id = metadata.get('config', {}).get('broker', 'unknown')
            position_limit = metadata.get('config', {}).get('risk', {}).get('position_limit_pct', 'unknown')
            logger.info(f"  Live with broker: {broker_id}")
            logger.info(f"  Position limit: {position_limit}")
    
    logger.info("\nStrategy Approval Example completed successfully")


if __name__ == "__main__":
    simulate_approval_workflow()
