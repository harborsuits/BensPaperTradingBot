#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Trading Bot with Market Context - Demonstrates how to integrate the MarketContextFetcher
with a trading bot for adaptive strategy rotation based on market regimes.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import time
import json

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our components
from trading_bot.market.market_context_fetcher import MarketContextFetcher, MarketRegime
from trading_bot.strategies.integrated_strategy_rotator import IntegratedStrategyRotator
from trading_bot.strategies.stocks.momentum import MomentumStrategy
from trading_bot.strategies.stocks.trend_following import TrendFollowingStrategy
from trading_bot.strategies.stocks.mean_reversion import MeanReversionStrategy

# If you have a backtesting module, import it as well
# from trading_bot.backtesting.order_book_simulator import OrderBookSimulator
# from trading_bot.backtesting.backtest_circuit_breaker_manager import BacktestCircuitBreakerManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingBot")

class TradingBot:
    """
    Trading bot that integrates market context, strategy rotation, and trade execution.
    """
    
    def __init__(
        self,
        symbols: list,
        data_dir: str,
        update_interval: int = 60,
        backtest_mode: bool = False,
        paper_trading: bool = True,
        risk_manager=None,
        data_provider=None
    ):
        """
        Initialize the trading bot.
        
        Args:
            symbols: List of symbols to trade
            data_dir: Directory for data storage
            update_interval: How often to update (seconds)
            backtest_mode: Whether to run in backtest mode
            paper_trading: Whether to use paper trading (vs live)
            risk_manager: Risk management component
            data_provider: Data provider for market data
        """
        self.symbols = symbols
        self.data_dir = data_dir
        self.update_interval = update_interval
        self.backtest_mode = backtest_mode
        self.paper_trading = paper_trading
        self.risk_manager = risk_manager
        self.data_provider = data_provider
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
        
        # Trading state
        self.is_trading = False
        self.positions = {}
        self.order_history = []
        self.performance_metrics = {}
        
        logger.info(f"Trading bot initialized with {len(symbols)} symbols")
    
    def _initialize_components(self):
        """Initialize the trading bot components."""
        # Initialize market context fetcher
        self.market_context = MarketContextFetcher(
            symbols=self.symbols,
            data_provider=self.data_provider,
            update_interval=self.update_interval,
            data_dir=os.path.join(self.data_dir, "market_context")
        )
        
        # Initialize strategies
        self.strategies = {
            "momentum": MomentumStrategy(),
            "trend_following": TrendFollowingStrategy(),
            "mean_reversion": MeanReversionStrategy()
        }
        
        # Initialize strategy rotator
        self.rotator = IntegratedStrategyRotator(
            strategies=list(self.strategies.keys()),
            data_dir=os.path.join(self.data_dir, "strategy_rotator")
        )
        
        # Add market context event listener
        self.market_context.add_event_listener(self.on_regime_change)
        
        logger.info("Trading bot components initialized")
    
    def on_regime_change(self, event):
        """
        Handler for market regime change events.
        
        Args:
            event: MarketRegimeEvent object
        """
        logger.info(f"Market regime change detected: {event}")
        
        # Log detailed metrics
        logger.info(f"Regime change metrics: {json.dumps(event.metrics, indent=2)}")
        
        # Rotate strategies based on new regime
        self._rotate_strategies_based_on_regime(event.new_regime)
        
        # Adjust risk parameters based on regime
        self._adjust_risk_based_on_regime(event.new_regime)
        
        # Notify any external systems if needed
        self._notify_regime_change(event)
    
    def _rotate_strategies_based_on_regime(self, regime):
        """
        Rotate strategies based on the current market regime.
        
        Args:
            regime: Current MarketRegime
        """
        logger.info(f"Rotating strategies for regime: {regime.name}")
        
        # Get the latest market data metrics
        metrics = self.market_context.get_latest_metrics()
        
        # Create a dataframe with relevant metrics
        market_data = pd.DataFrame({
            'close': [metrics.get('short_term_return', 0) + 1.0],
            'volatility': [metrics.get('volatility_ratio', 1.0)],
            'regime': [regime.name]
        })
        
        # Perform strategy rotation
        rotation_result = self.rotator.rotate_strategies(market_data, force_rotation=True)
        
        logger.info(f"New strategy allocations: {rotation_result['new_allocations']}")
        
        # Here you would update actual positions based on the new allocations
        if self.is_trading:
            self._rebalance_portfolio(rotation_result['new_allocations'])
    
    def _adjust_risk_based_on_regime(self, regime):
        """
        Adjust risk parameters based on the current market regime.
        
        Args:
            regime: Current MarketRegime
        """
        if self.risk_manager is None:
            return
        
        logger.info(f"Adjusting risk parameters for regime: {regime.name}")
        
        # Define risk adjustments for different regimes
        if regime == MarketRegime.CRISIS:
            # Highest risk - reduce position sizes and increase stops
            self.risk_manager.set_position_size_multiplier(0.3)
            self.risk_manager.set_stop_loss_multiplier(1.5)
            
        elif regime == MarketRegime.BEAR:
            # High risk - reduce position sizes
            self.risk_manager.set_position_size_multiplier(0.5)
            self.risk_manager.set_stop_loss_multiplier(1.2)
            
        elif regime == MarketRegime.HIGH_VOL:
            # Elevated risk - slightly reduce position sizes
            self.risk_manager.set_position_size_multiplier(0.7)
            self.risk_manager.set_stop_loss_multiplier(1.2)
            
        elif regime == MarketRegime.SIDEWAYS:
            # Moderate risk - normal position sizes
            self.risk_manager.set_position_size_multiplier(0.9)
            self.risk_manager.set_stop_loss_multiplier(1.0)
            
        elif regime == MarketRegime.BULL:
            # Lower risk - can use full position sizes
            self.risk_manager.set_position_size_multiplier(1.0)
            self.risk_manager.set_stop_loss_multiplier(1.0)
            
        elif regime == MarketRegime.LOW_VOL:
            # Lowest risk - can use slightly larger position sizes
            self.risk_manager.set_position_size_multiplier(1.1)
            self.risk_manager.set_stop_loss_multiplier(0.9)
            
        else:  # UNKNOWN regime
            # Default to moderate risk
            self.risk_manager.set_position_size_multiplier(0.8)
            self.risk_manager.set_stop_loss_multiplier(1.0)
    
    def _notify_regime_change(self, event):
        """
        Notify external systems of regime change.
        
        Args:
            event: MarketRegimeEvent object
        """
        # This could send notifications via email, slack, etc.
        pass
    
    def _rebalance_portfolio(self, allocations):
        """
        Rebalance portfolio based on new strategy allocations.
        
        Args:
            allocations: Dictionary of strategy allocations
        """
        logger.info("Rebalancing portfolio based on new allocations")
        
        # Here you would implement your portfolio rebalancing logic
        # For example:
        # 1. Calculate target positions based on allocations
        # 2. Compare with current positions
        # 3. Generate orders to bring current positions in line with targets
        # 4. Execute the orders
        
        # This is a placeholder for the actual implementation
        for strategy, allocation in allocations.items():
            logger.info(f"Setting {strategy} allocation to {allocation:.2f}%")
            
            # Get signals from this strategy
            if strategy in self.strategies:
                strategy_obj = self.strategies[strategy]
                signals = strategy_obj.get_current_signals()
                
                logger.info(f"{strategy} signals: {signals}")
                
                # Here you would use these signals with the allocation
                # to determine position sizes and execute trades
    
    def start_trading(self):
        """Start the trading bot."""
        if self.is_trading:
            logger.warning("Trading bot is already running")
            return
        
        logger.info("Starting trading bot")
        
        # Start market context fetcher
        self.market_context.start()
        
        # Set trading flag
        self.is_trading = True
        
        # Initial strategy rotation based on current regime
        current_regime, _ = self.market_context.get_current_regime()
        self._rotate_strategies_based_on_regime(current_regime)
        
        logger.info(f"Trading bot started in {'backtest' if self.backtest_mode else 'live'} mode")
        logger.info(f"Using {'paper trading' if self.paper_trading else 'real money'}")
    
    def stop_trading(self):
        """Stop the trading bot."""
        if not self.is_trading:
            logger.warning("Trading bot is not running")
            return
        
        logger.info("Stopping trading bot")
        
        # Stop market context fetcher
        self.market_context.stop()
        
        # Set trading flag
        self.is_trading = False
        
        # Save state
        self._save_state()
        
        logger.info("Trading bot stopped")
    
    def run_backtest(self, data, start_date, end_date):
        """
        Run the trading bot in backtest mode.
        
        Args:
            data: Historical data for backtesting
            start_date: Start date for backtest
            end_date: End date for backtest
        """
        if not self.backtest_mode:
            logger.error("Cannot run backtest in live mode")
            return
        
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Set up backtesting environment
        # For example, initialize the OrderBookSimulator if available
        
        # Simulate trading day by day
        current_date = start_date
        while current_date <= end_date:
            logger.info(f"Backtesting for date: {current_date}")
            
            # Get data for current date
            daily_data = data[data.index == current_date]
            
            if daily_data.empty:
                current_date += timedelta(days=1)
                continue
            
            # Update market context
            # In a real implementation, you would feed this data to the market context fetcher
            
            # Process trading signals
            # In a real implementation, you would execute trades based on signals
            
            # Move to next day
            current_date += timedelta(days=1)
        
        # Calculate and report backtest results
        self._calculate_performance()
        
        logger.info("Backtest completed")
    
    def _calculate_performance(self):
        """Calculate performance metrics."""
        # Calculate various performance metrics
        # - Total return
        # - Sharpe ratio
        # - Max drawdown
        # - Win rate
        # - etc.
        
        # This is a placeholder for the actual implementation
        self.performance_metrics = {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0
        }
        
        logger.info(f"Performance metrics: {self.performance_metrics}")
    
    def _save_state(self):
        """Save the current state of the trading bot."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "is_trading": self.is_trading,
            "positions": self.positions,
            "performance_metrics": self.performance_metrics
        }
        
        try:
            with open(os.path.join(self.data_dir, "trading_bot_state.json"), 'w') as f:
                json.dump(state, f, indent=2)
            logger.info("Trading bot state saved")
        except Exception as e:
            logger.error(f"Error saving trading bot state: {str(e)}")
    
    def _load_state(self):
        """Load the previous state of the trading bot."""
        state_file = os.path.join(self.data_dir, "trading_bot_state.json")
        
        if not os.path.exists(state_file):
            logger.info("No previous state found")
            return False
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            self.positions = state.get("positions", {})
            self.performance_metrics = state.get("performance_metrics", {})
            
            logger.info("Trading bot state loaded")
            return True
        except Exception as e:
            logger.error(f"Error loading trading bot state: {str(e)}")
            return False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run trading bot with market context")
    
    parser.add_argument("--backtest", action="store_true", help="Run in backtest mode")
    parser.add_argument("--paper-trading", action="store_true", help="Use paper trading (no real money)")
    parser.add_argument("--start-date", type=str, help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for backtest (YYYY-MM-DD)")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory for data storage")
    parser.add_argument("--symbols", type=str, nargs="+", default=["SPY", "QQQ", "IWM", "TLT"],
                        help="Symbols to trade")
    parser.add_argument("--interval", type=int, default=60, help="Update interval in seconds")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Create data directory
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Initialize trading bot
    bot = TradingBot(
        symbols=args.symbols,
        data_dir=args.data_dir,
        update_interval=args.interval,
        backtest_mode=args.backtest,
        paper_trading=args.paper_trading
    )
    
    if args.backtest:
        # Run backtest
        if not args.start_date or not args.end_date:
            logger.error("Start date and end date are required for backtest mode")
            return
        
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        
        # Load historical data for backtest
        # This is a placeholder - you would need to implement data loading
        data = pd.DataFrame()  # Replace with actual data loading
        
        bot.run_backtest(data, start_date, end_date)
    else:
        # Run live trading
        try:
            bot.start_trading()
            
            # Keep running until interrupted
            logger.info("Press Ctrl+C to stop trading")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Trading interrupted by user")
        finally:
            # Stop trading and clean up
            bot.stop_trading()

if __name__ == "__main__":
    main() 