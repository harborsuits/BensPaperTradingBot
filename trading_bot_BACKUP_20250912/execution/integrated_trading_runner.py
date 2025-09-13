#!/usr/bin/env python
"""
Integrated Trading Runner

This script integrates all components of the enhanced trading system:
1. Paper trading with the adaptive strategy controller
2. Real-time data feeds with latency monitoring
3. Enhanced monitoring and visualization dashboard
4. Telegram alerts for risk events and regime changes
"""

import os
import sys
import logging
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import trading system components
from trading_bot.risk.adaptive_strategy_controller import AdaptiveStrategyController
from trading_bot.execution.adaptive_paper_integration import AdaptivePaperTrading, get_paper_trading_instance
from trading_bot.execution.realtime_data_feed import RealtimeDataFeed, get_realtime_feed_instance
from trading_bot.dashboard.enhanced_alerting import EnhancedAlertMonitor, get_alert_monitor
from trading_bot.dashboard.strategy_signal_visualization import StrategySignalVisualizer, get_visualizer
from trading_bot.alerts.telegram_alerts import send_system_alert

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'integrated_trading.log'))
    ]
)

logger = logging.getLogger(__name__)

class IntegratedTradingSystem:
    """
    Integrated trading system that combines adaptive strategy control with
    paper trading, real-time data feeds, enhanced monitoring, and visualization.
    """
    
    def __init__(self):
        """Initialize the integrated trading system"""
        # Create output directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('dashboard', exist_ok=True)
        
        # System components
        self.controller = None
        self.paper_trading = None
        self.data_feed = None
        self.alert_monitor = None
        self.visualizer = None
        
        # System status
        self.running = False
        self.visualization_interval = 300  # 5 minutes
        self.last_visualization_time = 0
    
    def initialize(self, 
                 config_file: Optional[str] = None,
                 initial_cash: float = 100000.0,
                 use_real_comparison: bool = False,
                 data_source: str = 'alpaca'):
        """
        Initialize the integrated trading system
        
        Args:
            config_file: Path to configuration file (optional)
            initial_cash: Initial cash for paper trading
            use_real_comparison: Whether to use real broker APIs for comparison
            data_source: Real-time data source ('alpaca', 'tradier', etc.)
        """
        logger.info("Initializing integrated trading system...")
        
        # 1. Create the adaptive strategy controller
        logger.info("Creating adaptive strategy controller...")
        self.controller = AdaptiveStrategyController()
        
        # 2. Initialize paper trading
        logger.info("Initializing paper trading...")
        self.paper_trading = get_paper_trading_instance()
        self.paper_trading.initialize(
            strategy_controller=self.controller,
            initial_cash=initial_cash,
            use_real_comparison=use_real_comparison,
            sandbox_mode=True
        )
        
        # 3. Initialize real-time data feed
        logger.info("Initializing real-time data feed...")
        self.data_feed = get_realtime_feed_instance()
        
        # Get API configuration based on data source
        api_config = self._get_api_config(data_source)
        
        # Connect to data source
        if api_config:
            self.data_feed.connect_data_source(data_source, api_config)
        
        # 4. Initialize enhanced alerting
        logger.info("Initializing enhanced alerting...")
        self.alert_monitor = get_alert_monitor()
        self.alert_monitor.set_controller(self.controller)
        
        # 5. Initialize visualizer
        logger.info("Initializing strategy signal visualizer...")
        self.visualizer = get_visualizer()
        self.visualizer.set_controller(self.controller)
        
        # Send initialization alert
        send_system_alert(
            component="Trading System",
            status="online",
            message="Integrated trading system initialized",
            severity="info"
        )
        
        logger.info("Integrated trading system initialization complete!")
        return True
    
    def start(self, 
            trading_interval: float = 60.0,  # 1 minute
            update_interval: float = 5.0):   # 5 seconds
        """
        Start the integrated trading system
        
        Args:
            trading_interval: Seconds between trading cycles
            update_interval: Seconds between visualization updates
        """
        if self.running:
            logger.warning("Trading system already running")
            return False
        
        if not self.controller or not self.paper_trading:
            logger.error("Trading system not properly initialized")
            return False
        
        self.running = True
        logger.info(f"Starting integrated trading system (interval: {trading_interval}s)")
        
        # Start enhanced alerting
        self.alert_monitor.start_monitoring()
        
        # Subscribe to market data for watched symbols
        symbols = self.controller.get_watched_symbols()
        if symbols:
            logger.info(f"Subscribing to market data for {len(symbols)} symbols")
            self.data_feed.subscribe_to_symbols(symbols)
            
            # Start real-time data monitoring
            self.data_feed.start_monitoring(self._process_market_data)
        
        # Run initial trading cycle
        logger.info("Running initial trading cycle...")
        self.paper_trading.run_trading_cycle()
        
        # Record initial state for visualization
        self.visualizer.record_current_state()
        self.last_visualization_time = time.time()
        
        # Trading loop
        last_trading_time = time.time()
        
        try:
            while self.running:
                current_time = time.time()
                
                # Check if it's time for a trading cycle
                if current_time - last_trading_time >= trading_interval:
                    logger.info("Running trading cycle...")
                    self.paper_trading.run_trading_cycle()
                    last_trading_time = current_time
                
                # Check if it's time to update visualizations
                if current_time - self.last_visualization_time >= self.visualization_interval:
                    self._update_visualizations()
                    self.last_visualization_time = current_time
                
                # Sleep for a bit
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in trading loop: {str(e)}")
        finally:
            self.stop()
        
        return True
    
    def stop(self):
        """Stop the integrated trading system"""
        if not self.running:
            return
        
        self.running = False
        logger.info("Stopping integrated trading system...")
        
        # Stop data monitoring
        if self.data_feed:
            self.data_feed.stop_monitoring()
        
        # Stop alert monitoring
        if self.alert_monitor:
            self.alert_monitor.stop_monitoring()
        
        # Final visualization update
        self._update_visualizations()
        
        # Save all data
        if self.visualizer:
            self.visualizer.save_data()
        
        # Analyze fill performance
        if self.paper_trading:
            self.paper_trading.analyze_fill_performance()
        
        # Send shutdown alert
        send_system_alert(
            component="Trading System",
            status="offline",
            message="Integrated trading system shutdown complete",
            severity="info"
        )
        
        logger.info("Integrated trading system stopped")
    
    def run_overnight_simulation(self, 
                              hours: float = 12.0,
                              interval_minutes: float = 5.0,
                              speed_multiplier: float = 60.0):
        """
        Run an overnight simulation to test the strategy with
        accelerated time.
        
        Args:
            hours: Number of hours to simulate
            interval_minutes: Minutes between trading cycles
            speed_multiplier: How much faster to run than real-time
        """
        if not self.paper_trading:
            logger.error("Paper trading not initialized")
            return False
        
        # Calculate simulation parameters
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=hours)
        
        logger.info(f"Starting overnight simulation from {start_time} to {end_time}")
        logger.info(f"Speed multiplier: {speed_multiplier}x")
        
        # Start enhanced alerting
        self.alert_monitor.start_monitoring()
        
        # Run simulation
        simulation_result = self.paper_trading.run_overnight_simulation(
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
            speed_multiplier=speed_multiplier
        )
        
        # Generate visualizations
        self._update_visualizations()
        
        # Stop alert monitoring
        self.alert_monitor.stop_monitoring()
        
        # Save all data
        self.visualizer.save_data()
        
        return simulation_result
    
    def _process_market_data(self, data: Dict[str, Any]):
        """Process incoming market data from real-time feed"""
        # Forward data to controller
        if self.controller:
            symbol = data.get('symbol')
            if symbol:
                # Format data for controller
                market_data = {
                    symbol: {
                        'price': data.get('price'),
                        'timestamp': data.get('timestamp', datetime.now()),
                        'volume': data.get('volume', 0),
                        'source': data.get('source', 'realtime')
                    }
                }
                
                # Process data in controller
                self.controller.process_market_data(market_data)
    
    def _update_visualizations(self):
        """Update all visualizations"""
        if not self.visualizer:
            return
        
        logger.info("Updating visualizations...")
        
        # Record current state
        self.visualizer.record_current_state()
        
        # Generate visualizations for each symbol
        symbols = self.controller.get_watched_symbols()
        for symbol in symbols:
            self.visualizer.generate_position_vs_regime_chart(symbol)
        
        # Generate weight evolution chart
        self.visualizer.generate_weight_evolution_chart()
        
        # Generate performance dashboard
        self.visualizer.generate_performance_dashboard()
        
        logger.info("Visualization update complete")
    
    def _get_api_config(self, data_source: str) -> Dict[str, Any]:
        """Get API configuration for the specified data source"""
        # Try to import config
        try:
            from trading_bot.config import API_KEYS
            
            if data_source.lower() == 'alpaca':
                if 'alpaca' in API_KEYS:
                    return {
                        'api_key': API_KEYS['alpaca'].get('key', ''),
                        'api_secret': API_KEYS['alpaca'].get('secret', ''),
                        'endpoint': API_KEYS['alpaca'].get('endpoint', 'https://paper-api.alpaca.markets')
                    }
            
            elif data_source.lower() == 'tradier':
                if 'tradier' in API_KEYS:
                    return {
                        'api_key': API_KEYS['tradier'],
                        'paper': True
                    }
            
            elif data_source.lower() == 'finnhub':
                if 'finnhub' in API_KEYS:
                    return {
                        'api_key': API_KEYS['finnhub']
                    }
            
            logger.warning(f"No API configuration found for {data_source}")
            return {}
            
        except ImportError:
            logger.warning("Could not import API_KEYS from config.py")
            return {}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Integrated Trading System')
    
    parser.add_argument(
        '--mode',
        choices=['live', 'simulation', 'backtest'],
        default='live',
        help='Operation mode'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=100000.0,
        help='Initial capital'
    )
    
    parser.add_argument(
        '--data-source',
        choices=['alpaca', 'tradier', 'finnhub'],
        default='alpaca',
        help='Real-time data source'
    )
    
    parser.add_argument(
        '--interval',
        type=float,
        default=60.0,
        help='Trading interval in seconds'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=12.0,
        help='Simulation duration in hours'
    )
    
    parser.add_argument(
        '--speed',
        type=float,
        default=60.0,
        help='Simulation speed multiplier'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    
    return parser.parse_args()

def main():
    """Main function to run the integrated trading system"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create system instance
    system = IntegratedTradingSystem()
    
    # Initialize the system
    system.initialize(
        config_file=args.config,
        initial_cash=args.capital,
        data_source=args.data_source
    )
    
    # Run based on mode
    if args.mode == 'live':
        logger.info("Starting in LIVE mode")
        system.start(trading_interval=args.interval)
    
    elif args.mode == 'simulation':
        logger.info("Starting in SIMULATION mode")
        system.run_overnight_simulation(
            hours=args.duration,
            interval_minutes=args.interval / 60,
            speed_multiplier=args.speed
        )
    
    elif args.mode == 'backtest':
        logger.info("Starting in BACKTEST mode")
        # This would use the existing backtesting functionality
        from trading_bot.backtest.adaptive_backtest_runner import main as run_backtest
        run_backtest()
    
    else:
        logger.error(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
