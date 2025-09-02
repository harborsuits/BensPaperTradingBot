"""
Example script demonstrating the use of the Staging Environment for testing strategies.

This example shows how to:
1. Configure and activate the staging environment
2. Register test strategies
3. Run them through the complete validation cycle
4. Generate reports and evaluate results
"""
import os
import time
import logging
import json
from datetime import datetime, timedelta
import sys
import pandas as pd
import numpy as np

# Add the project root to the Python path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.core.event_bus import EventBus
from trading_bot.core.constants import EventType, StrategyStatus, StrategyPhase
from trading_bot.core.staging_environment import create_staging_environment
from trading_bot.core.strategy_trial_workflow import StrategyTrialWorkflow
from trading_bot.core.performance_tracker import PerformanceTracker
from trading_bot.brokers.paper_broker_factory import create_paper_broker
from trading_bot.core.strategy_base import Strategy
from trading_bot.strategies.simple.moving_average_crossover import MovingAverageCrossoverStrategy
from trading_bot.strategies.forex.base.pip_based_position_sizing import PipBasedPositionSizing

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("staging_example")

class ExampleTestRunner:
    """Example class for running a complete staging test cycle."""
    
    def __init__(self):
        """Initialize the test runner."""
        # Initialize services
        self.service_registry = ServiceRegistry.get_instance()
        self.event_bus = EventBus()
        self.service_registry.register_service("event_bus", self.event_bus)
        
        # Create staging configuration with shortened duration for example
        self.staging_config = {
            "test_duration_days": 0.01,  # Set very short for example (less than 15 minutes)
            "min_trades_required": 10,   # Lower threshold for example
            "reporting_frequency_hours": 0.01,  # Generate reports every ~30 seconds
            "reports_directory": "./example_reports/staging",
            "validation_checkpoints": {
                "min_sharpe_ratio": 0.5,  # Lower threshold for example
                "max_drawdown_pct": -15.0,  # More lenient for example
                "min_win_rate": 0.3,      # Lower threshold for example
                "min_profit_factor": 1.1   # Lower threshold for example
            }
        }
        
        # Save config to file
        os.makedirs("./example_reports", exist_ok=True)
        os.makedirs(self.staging_config["reports_directory"], exist_ok=True)
        
        with open("./example_reports/staging_config.json", "w") as f:
            json.dump(self.staging_config, f, indent=2)
    
    def setup_services(self):
        """Setup required services for the staging environment."""
        logger.info("Setting up required services...")
        
        # Create paper broker for testing
        create_paper_broker(
            broker_id="paper_main",
            name="Paper Broker for Testing",
            initial_balance=100000.0
        )
        
        # Setup performance tracker
        performance_tracker = PerformanceTracker()
        self.service_registry.register_service("performance_tracker", performance_tracker)
        
        # Setup strategy trial workflow
        workflow = StrategyTrialWorkflow(default_phase=StrategyPhase.PAPER_TRADE)
        self.service_registry.register_service("strategy_trial_workflow", workflow)
        
        logger.info("Services initialized")
        
    def register_test_strategies(self):
        """Register test strategies for the staging environment."""
        logger.info("Registering test strategies...")
        
        workflow = self.service_registry.get_service("strategy_trial_workflow")
        if not workflow:
            logger.error("Workflow service not available")
            return
        
        # Register a few test strategies with different characteristics
        
        # Strategy 1: Moving Average Crossover (should perform well)
        strategy1 = MovingAverageCrossoverStrategy(
            name="MA_Crossover_Test_1",
            symbols=["AAPL", "MSFT", "GOOGL"],
            short_window=10,
            long_window=30
        )
        workflow.register_strategy(strategy1, broker_id="paper_main")
        
        # Strategy 2: Moving Average Crossover with poor parameters (should perform poorly)
        strategy2 = MovingAverageCrossoverStrategy(
            name="MA_Crossover_Test_2",
            symbols=["AAPL", "MSFT", "GOOGL"],
            short_window=5,
            long_window=7  # Too close to short window
        )
        workflow.register_strategy(strategy2, broker_id="paper_main")
        
        # Strategy 3: Pip-based position sizing strategy (should be average)
        strategy3 = PipBasedPositionSizing(
            name="PipBased_Test_1",
            symbols=["EUR/USD", "GBP/USD"],
            risk_per_trade_pct=1.0,
            take_profit_pips=30,
            stop_loss_pips=20
        )
        workflow.register_strategy(strategy3, broker_id="paper_main")
        
        logger.info(f"Registered {len(workflow.get_all_strategies())} test strategies")
    
    def simulate_trading_activity(self):
        """Simulate trading activity for the test strategies."""
        logger.info("Simulating trading activity...")
        
        workflow = self.service_registry.get_service("strategy_trial_workflow")
        performance_tracker = self.service_registry.get_service("performance_tracker")
        
        if not workflow or not performance_tracker:
            logger.error("Required services not available")
            return
        
        strategies = workflow.get_all_strategies()
        
        # Generate synthetic trades for each strategy
        for strategy in strategies:
            strategy_id = strategy.get("id")
            strategy_name = strategy.get("name", "Unknown")
            
            logger.info(f"Generating synthetic trades for {strategy_name} ({strategy_id})")
            
            # Different performance profiles
            if "MA_Crossover_Test_1" in strategy_name:
                # Good performance
                win_rate = 0.65
                avg_profit = 1.5
                avg_loss = -1.0
                trade_count = 50
            elif "MA_Crossover_Test_2" in strategy_name:
                # Poor performance
                win_rate = 0.35
                avg_profit = 1.0
                avg_loss = -1.5
                trade_count = 50
            else:
                # Average performance
                win_rate = 0.50
                avg_profit = 1.2
                avg_loss = -1.0
                trade_count = 50
            
            # Generate trades with appropriate performance characteristics
            symbols = ["AAPL", "MSFT", "GOOGL", "EUR/USD", "GBP/USD"]
            
            for i in range(trade_count):
                # Determine if this is a winning trade
                is_win = np.random.random() < win_rate
                
                # Generate trade details
                symbol = np.random.choice(symbols)
                pnl = avg_profit if is_win else avg_loss
                entry_price = 100 + np.random.random() * 50
                exit_price = entry_price * (1 + pnl/100)
                quantity = 10 + np.random.randint(1, 10)
                
                # Record the trade with the performance tracker
                trade_data = {
                    "strategy_id": strategy_id,
                    "symbol": symbol,
                    "entry_time": datetime.now() - timedelta(minutes=np.random.randint(1, 60)),
                    "exit_time": datetime.now(),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "quantity": quantity,
                    "side": "buy" if is_win else "sell",
                    "pnl": pnl * quantity,
                    "commission": 1.0,
                    "tags": ["PAPER", "SIMULATED"]
                }
                
                performance_tracker.record_trade(trade_data)
            
            # Also record a few positions
            for j in range(3):
                symbol = np.random.choice(symbols)
                current_price = 100 + np.random.random() * 50
                quantity = 10 + np.random.randint(1, 10)
                
                position_data = {
                    "strategy_id": strategy_id,
                    "symbol": symbol,
                    "quantity": quantity,
                    "entry_price": current_price * 0.98,
                    "current_price": current_price,
                    "unrealized_pl": quantity * (current_price - current_price * 0.98),
                    "market_value": quantity * current_price,
                    "side": "long",
                    "tags": ["PAPER", "SIMULATED"]
                }
                
                performance_tracker.update_position(position_data)
        
        logger.info("Trading activity simulation completed")
    
    def run_full_test(self):
        """Run a full test cycle with the staging environment."""
        # Setup services
        self.setup_services()
        
        # Register test strategies
        self.register_test_strategies()
        
        # Create and activate staging environment
        logger.info("Creating staging environment...")
        staging_env = create_staging_environment("./example_reports/staging_config.json")
        
        # Simulate trading activity
        self.simulate_trading_activity()
        
        # Monitor the environment status
        logger.info("Monitoring staging environment...")
        
        # Run for a few minutes to allow reports to generate
        start_time = datetime.now()
        duration = timedelta(minutes=5)
        
        while datetime.now() - start_time < duration:
            # Get staging status
            status = staging_env.get_staging_status()
            logger.info(f"Staging status: {json.dumps(status, indent=2)}")
            
            # Simulate more trades every 30 seconds
            self.simulate_trading_activity()
            
            # Sleep for a bit
            time.sleep(30)
        
        # Generate final reports
        logger.info("Generating final reports...")
        report_generator = staging_env.report_generator
        if report_generator:
            final_report = report_generator.generate_comprehensive_report()
            logger.info(f"Final report generated with {len(final_report.get('strategy_reports', {}))} strategies")
        
        # Deactivate staging environment
        logger.info("Deactivating staging environment...")
        staging_env.deactivate()
        
        logger.info("Staging environment test completed")
        
        # Display report locations
        logger.info(f"Reports are available in: {os.path.abspath(self.staging_config['reports_directory'])}")


if __name__ == "__main__":
    runner = ExampleTestRunner()
    runner.run_full_test()
