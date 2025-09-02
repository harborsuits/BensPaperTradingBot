"""
Adaptive Strategy Paper Trading Integration

This module connects the AdaptiveStrategyController with broker adapters (Tradier/Alpaca)
in paper trading mode for testing and evaluation.
"""

import logging
import os
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import threading
import pandas as pd
import numpy as np

# Import trading system components
from trading_bot.risk.adaptive_strategy_controller import AdaptiveStrategyController
from trading_bot.analytics.performance_tracker import PerformanceTracker
from trading_bot.analytics.market_regime_detector import MarketRegimeDetector
from trading_bot.brokers.paper.adapter import PaperTradeAdapter, PaperTradeConfig
from trading_bot.brokers.tradier.adapter import TradierAdapter
from trading_bot.brokers.alpaca.adapter import AlpacaAdapter
from trading_bot.core.events import (
    OrderPlaced, OrderAcknowledged, OrderPartialFill, OrderFilled, 
    OrderCancelled, OrderRejected, SlippageMetric
)
from trading_bot.event_system.event_bus import EventBus
from trading_bot.alerts.telegram_alerts import (
    send_risk_alert, send_strategy_rotation_alert, 
    send_trade_alert, send_system_alert
)

logger = logging.getLogger(__name__)

class AdaptivePaperTrading:
    """
    Integrates the AdaptiveStrategyController with paper trading for testing
    and evaluation. Provides detailed logging of orders, fills, and performance
    metrics compared to expected results.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """Initialize the adaptive paper trading integration"""
        self.event_bus = event_bus or EventBus()
        self.controller = None
        self.paper_adapter = None
        self.real_adapter = None  # For comparison with paper results
        self.tracking_enabled = True
        self.last_run_time = None
        self.order_log = []
        self.fill_log = []
        self.performance_log = []
        self.expected_vs_actual = []
        
        # Create output directories
        os.makedirs('./results/paper_trading', exist_ok=True)
        os.makedirs('./logs/paper_trading', exist_ok=True)
        
        # Setup event handlers
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Setup handlers for relevant trading events"""
        self.event_bus.subscribe(OrderPlaced, self._on_order_placed)
        self.event_bus.subscribe(OrderAcknowledged, self._on_order_acknowledged)
        self.event_bus.subscribe(OrderFilled, self._on_order_filled)
        self.event_bus.subscribe(OrderPartialFill, self._on_order_partial_fill)
        self.event_bus.subscribe(OrderRejected, self._on_order_rejected)
        self.event_bus.subscribe(OrderCancelled, self._on_order_cancelled)
        self.event_bus.subscribe(SlippageMetric, self._on_slippage_metric)
    
    def initialize(self, 
                 strategy_controller: Optional[AdaptiveStrategyController] = None,
                 initial_cash: float = 100000.0,
                 use_real_comparison: bool = False,
                 real_broker: str = 'tradier',
                 sandbox_mode: bool = True):
        """
        Initialize the paper trading system
        
        Args:
            strategy_controller: AdaptiveStrategyController instance (or will create one)
            initial_cash: Initial cash balance for paper trading
            use_real_comparison: Whether to use a real broker adapter for comparison
            real_broker: Which real broker to use ('tradier' or 'alpaca')
            sandbox_mode: Whether to use sandbox APIs for the real broker
        """
        # Create controller if not provided
        if strategy_controller is None:
            from trading_bot.risk.adaptive_strategy_controller import AdaptiveStrategyController
            logger.info("Creating new AdaptiveStrategyController instance")
            self.controller = AdaptiveStrategyController(event_bus=self.event_bus)
        else:
            logger.info("Using provided AdaptiveStrategyController instance")
            self.controller = strategy_controller
        
        # Initialize paper trading adapter
        paper_config = PaperTradeConfig(
            initial_cash=initial_cash,
            slippage_model='random',
            slippage_range=(0.0001, 0.0010),  # 1-10 bps slippage
            fill_latency_range=(0.1, 1.0),    # 100ms to 1s latency
            partial_fills_probability=0.25,    # 25% chance of partial fills
            enable_shorting=True,
            margin_requirement=0.5,
            simulation_mode='realtime',
            state_file='./data/paper_trading_state.json'
        )
        
        self.paper_adapter = PaperTradeAdapter(event_bus=self.event_bus)
        self.paper_adapter.connect(paper_config)
        logger.info(f"Connected to paper trading adapter with ${initial_cash} initial capital")
        
        # If comparing with real broker, initialize that adapter too
        if use_real_comparison:
            if real_broker.lower() == 'tradier':
                self.real_adapter = TradierAdapter(event_bus=self.event_bus)
                self.real_adapter.connect({'paper': sandbox_mode})
                logger.info(f"Connected to Tradier {'sandbox' if sandbox_mode else 'live'} adapter for comparison")
            elif real_broker.lower() == 'alpaca':
                self.real_adapter = AlpacaAdapter(event_bus=self.event_bus)
                self.real_adapter.connect({'paper': sandbox_mode})
                logger.info(f"Connected to Alpaca {'paper' if sandbox_mode else 'live'} adapter for comparison")
        
        # Send initialization alert
        send_system_alert(
            component="Paper Trading",
            status="online",
            message=f"Paper trading initialized with ${initial_cash:,.2f} capital",
            severity="info"
        )
        
        return True
    
    def run_trading_cycle(self):
        """Run a single trading cycle using the adaptive controller"""
        if not self.paper_adapter or not self.paper_adapter.is_connected():
            logger.error("Paper trading adapter not connected")
            return False
        
        if not self.controller:
            logger.error("Strategy controller not initialized")
            return False
        
        # Get current account state
        account_info = self.paper_adapter.get_account()
        positions = self.paper_adapter.get_positions()
        
        # Update controller with current state
        self.controller.update_account_info(account_info)
        self.controller.update_positions(positions)
        
        # Get current market data
        symbols = self.controller.get_watched_symbols()
        market_data = {}
        
        for symbol in symbols:
            quote = self.paper_adapter.get_quote(symbol)
            if quote:
                market_data[symbol] = quote
        
        # Run controller cycle
        self.controller.process_market_data(market_data)
        self.controller.execute_strategy_cycle()
        
        # Record state for analysis
        self._record_performance()
        
        self.last_run_time = datetime.now()
        return True
    
    def run_overnight_simulation(self, 
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None,
                              interval_minutes: int = 15,
                              speed_multiplier: float = 60.0):
        """
        Run an overnight simulation to test the strategy with
        simulated market conditions.
        
        Args:
            start_time: Start time for simulation (None for now)
            end_time: End time for simulation (None for end of day)
            interval_minutes: Minutes between trading cycles
            speed_multiplier: How much faster to run than real-time
        """
        # Set default times if not provided
        now = datetime.now()
        if start_time is None:
            start_time = now
        if end_time is None:
            # Default to market close (4pm) if within trading day, 
            # otherwise +12 hours
            market_close = datetime(
                now.year, now.month, now.day,
                16, 0, 0  # 4:00 PM
            )
            if now.hour < 16:
                end_time = market_close
            else:
                end_time = now + timedelta(hours=12)
        
        # Calculate simulation parameters
        total_duration = (end_time - start_time).total_seconds()
        simulation_intervals = int(total_duration / (interval_minutes * 60))
        interval_time = (interval_minutes * 60) / speed_multiplier
        
        logger.info(f"Starting overnight simulation from {start_time} to {end_time}")
        logger.info(f"Running {simulation_intervals} intervals at {interval_minutes} minute spacing")
        logger.info(f"Speed multiplier: {speed_multiplier}x (interval: {interval_time:.2f}s)")
        
        # Send start alert
        send_system_alert(
            component="Overnight Simulation",
            status="starting",
            message=f"Beginning {simulation_intervals} cycle simulation at {speed_multiplier}x speed",
            severity="info"
        )
        
        # Setup the paper adapter for simulation
        current_config = self.paper_adapter.config
        current_config.simulation_mode = 'backtest'
        self.paper_adapter.connect(current_config)
        
        # Set the initial time
        simulation_time = start_time
        self.paper_adapter.current_time = simulation_time
        
        # Run simulation
        for i in range(simulation_intervals):
            try:
                # Set current simulation time
                self.paper_adapter.current_time = simulation_time
                
                # Run trading cycle
                self.run_trading_cycle()
                
                # Update time for next cycle
                simulation_time += timedelta(minutes=interval_minutes)
                
                # Sleep for interval time
                time.sleep(interval_time)
                
                # Log progress periodically
                if i % 10 == 0 or i == simulation_intervals - 1:
                    progress = (i + 1) / simulation_intervals * 100
                    logger.info(f"Simulation progress: {progress:.1f}% - Current time: {simulation_time}")
            except Exception as e:
                logger.error(f"Error in simulation cycle {i}: {str(e)}")
                send_system_alert(
                    component="Overnight Simulation",
                    status="error",
                    message=f"Error in simulation cycle {i}: {str(e)}",
                    severity="high"
                )
        
        # Reset to real-time mode
        current_config.simulation_mode = 'realtime'
        self.paper_adapter.connect(current_config)
        
        # Generate and save simulation report
        report = self._generate_simulation_report(start_time, end_time)
        
        # Send completion alert
        send_system_alert(
            component="Overnight Simulation",
            status="online",
            message=f"Completed {simulation_intervals} cycle simulation. Final P&L: ${report['total_pnl']:,.2f}",
            severity="info"
        )
        
        return report
    
    def analyze_fill_performance(self):
        """
        Analyze the difference between expected fills and actual fills,
        computing metrics like slippage and latency.
        """
        if not self.expected_vs_actual:
            logger.warning("No fill data available for analysis")
            return {}
        
        # Calculate slippage statistics
        slippage_bps = [record['slippage_bps'] for record in self.expected_vs_actual]
        slippage_amount = [record['slippage_amount'] for record in self.expected_vs_actual]
        fill_latency = [record['latency_seconds'] for record in self.expected_vs_actual]
        
        results = {
            'num_orders': len(self.expected_vs_actual),
            'avg_slippage_bps': np.mean(slippage_bps),
            'median_slippage_bps': np.median(slippage_bps),
            'max_slippage_bps': max(slippage_bps),
            'min_slippage_bps': min(slippage_bps),
            'total_slippage_amount': sum(slippage_amount),
            'avg_latency_seconds': np.mean(fill_latency),
            'median_latency_seconds': np.median(fill_latency),
            'max_latency_seconds': max(fill_latency),
            'min_latency_seconds': min(fill_latency),
            'records': self.expected_vs_actual
        }
        
        # Save detailed report
        report_path = os.path.join('./results/paper_trading', 'fill_analysis.json')
        with open(report_path, 'w') as f:
            # Create a serializable version of the results
            serializable_results = {k: v for k, v in results.items() if k != 'records'}
            serializable_results['records'] = []
            
            for record in self.expected_vs_actual:
                serializable_record = {
                    'order_id': record['order_id'],
                    'symbol': record['symbol'],
                    'expected_price': record['expected_price'],
                    'actual_price': record['actual_price'],
                    'quantity': record['quantity'],
                    'slippage_bps': record['slippage_bps'],
                    'slippage_amount': record['slippage_amount'],
                    'latency_seconds': record['latency_seconds'],
                    'timestamp': record['timestamp'].isoformat() if isinstance(record['timestamp'], datetime) else record['timestamp']
                }
                serializable_results['records'].append(serializable_record)
            
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Fill analysis report saved to {report_path}")
        return results
    
    def _on_order_placed(self, event):
        """Handle order placed event"""
        if not self.tracking_enabled:
            return
        
        # Record order details
        order_data = {
            'event_type': 'order_placed',
            'timestamp': datetime.now(),
            'order_id': event.order_id,
            'client_order_id': event.client_order_id,
            'symbol': event.symbol,
            'quantity': event.quantity,
            'price': event.price if hasattr(event, 'price') else None,
            'order_type': event.order_type,
            'side': event.side,
            'expected_price': event.expected_price if hasattr(event, 'expected_price') else None
        }
        
        self.order_log.append(order_data)
    
    def _on_order_acknowledged(self, event):
        """Handle order acknowledged event"""
        if not self.tracking_enabled:
            return
        
        # Record order acknowledgement
        ack_data = {
            'event_type': 'order_acknowledged',
            'timestamp': datetime.now(),
            'order_id': event.order_id,
            'client_order_id': event.client_order_id,
            'broker_order_id': event.broker_order_id if hasattr(event, 'broker_order_id') else None
        }
        
        self.order_log.append(ack_data)
    
    def _on_order_filled(self, event):
        """Handle order filled event"""
        if not self.tracking_enabled:
            return
        
        # Find the original order to compare with expected price
        expected_price = None
        order_time = None
        
        for order_event in reversed(self.order_log):
            if (order_event['order_id'] == event.order_id and 
                order_event['event_type'] == 'order_placed'):
                expected_price = order_event.get('expected_price')
                order_time = order_event.get('timestamp')
                break
        
        # Calculate statistics if we have an expected price
        if expected_price is not None and event.price is not None:
            # Calculate price difference in basis points (1bp = 0.01%)
            price_diff = event.price - expected_price
            slippage_bps = (price_diff / expected_price) * 10000  # Convert to basis points
            slippage_amount = price_diff * event.quantity
            
            # Calculate latency if we have order time
            latency_seconds = 0
            if order_time:
                latency_seconds = (datetime.now() - order_time).total_seconds()
            
            # Record comparison
            self.expected_vs_actual.append({
                'order_id': event.order_id,
                'symbol': event.symbol,
                'expected_price': expected_price,
                'actual_price': event.price,
                'quantity': event.quantity,
                'slippage_bps': slippage_bps,
                'slippage_amount': slippage_amount,
                'latency_seconds': latency_seconds,
                'timestamp': datetime.now()
            })
        
        # Record fill details
        fill_data = {
            'event_type': 'order_filled',
            'timestamp': datetime.now(),
            'order_id': event.order_id,
            'client_order_id': event.client_order_id,
            'symbol': event.symbol,
            'quantity': event.quantity,
            'price': event.price,
            'expected_price': expected_price,
            'slippage_bps': slippage_bps if expected_price is not None else None,
            'latency_seconds': latency_seconds if order_time is not None else None
        }
        
        self.fill_log.append(fill_data)
    
    def _on_order_partial_fill(self, event):
        """Handle order partial fill event"""
        if not self.tracking_enabled:
            return
        
        # Similar logic to full fill, but for partial fill
        expected_price = None
        order_time = None
        
        for order_event in reversed(self.order_log):
            if (order_event['order_id'] == event.order_id and 
                order_event['event_type'] == 'order_placed'):
                expected_price = order_event.get('expected_price')
                order_time = order_event.get('timestamp')
                break
        
        # Calculate statistics if we have an expected price
        slippage_bps = None
        slippage_amount = None
        latency_seconds = None
        
        if expected_price is not None and event.price is not None:
            # Calculate price difference in basis points (1bp = 0.01%)
            price_diff = event.price - expected_price
            slippage_bps = (price_diff / expected_price) * 10000  # Convert to basis points
            slippage_amount = price_diff * event.quantity
            
            # Calculate latency if we have order time
            if order_time:
                latency_seconds = (datetime.now() - order_time).total_seconds()
            
            # Record comparison
            self.expected_vs_actual.append({
                'order_id': event.order_id,
                'symbol': event.symbol,
                'expected_price': expected_price,
                'actual_price': event.price,
                'quantity': event.quantity,
                'slippage_bps': slippage_bps,
                'slippage_amount': slippage_amount,
                'latency_seconds': latency_seconds,
                'timestamp': datetime.now()
            })
        
        # Record partial fill details
        fill_data = {
            'event_type': 'order_partial_fill',
            'timestamp': datetime.now(),
            'order_id': event.order_id,
            'client_order_id': event.client_order_id,
            'symbol': event.symbol,
            'quantity': event.quantity,
            'price': event.price,
            'expected_price': expected_price,
            'slippage_bps': slippage_bps,
            'latency_seconds': latency_seconds
        }
        
        self.fill_log.append(fill_data)
    
    def _on_order_rejected(self, event):
        """Handle order rejected event"""
        if not self.tracking_enabled:
            return
        
        # Record rejection details
        reject_data = {
            'event_type': 'order_rejected',
            'timestamp': datetime.now(),
            'order_id': event.order_id,
            'client_order_id': event.client_order_id,
            'reason': event.reason if hasattr(event, 'reason') else 'unknown'
        }
        
        self.order_log.append(reject_data)
        
        # Send alert for rejected order
        send_system_alert(
            component="Paper Trading",
            status="warning",
            message=f"Order rejected: {event.reason if hasattr(event, 'reason') else 'unknown'}",
            severity="medium"
        )
    
    def _on_order_cancelled(self, event):
        """Handle order cancelled event"""
        if not self.tracking_enabled:
            return
        
        # Record cancellation details
        cancel_data = {
            'event_type': 'order_cancelled',
            'timestamp': datetime.now(),
            'order_id': event.order_id,
            'client_order_id': event.client_order_id
        }
        
        self.order_log.append(cancel_data)
    
    def _on_slippage_metric(self, event):
        """Handle slippage metric event"""
        if not self.tracking_enabled:
            return
        
        # Record slippage metric
        slippage_data = {
            'event_type': 'slippage_metric',
            'timestamp': datetime.now(),
            'order_id': event.order_id,
            'expected_price': event.expected_price,
            'actual_price': event.actual_price,
            'quantity': event.quantity,
            'slippage_bps': event.slippage_bps,
            'slippage_amount': event.slippage_amount
        }
        
        self.order_log.append(slippage_data)
    
    def _record_performance(self):
        """Record current portfolio performance"""
        # Get account information
        account = self.paper_adapter.get_account()
        
        # Get performance metrics from controller if available
        performance_metrics = {}
        if hasattr(self.controller, 'performance_tracker'):
            performance_metrics = self.controller.performance_tracker.get_metrics()
        
        # Get allocation data if available
        allocations = {}
        if hasattr(self.controller, 'snowball_allocator'):
            allocations = self.controller.snowball_allocator.get_current_allocation()
        
        # Get regime data if available
        regime_data = {}
        if hasattr(self.controller, 'market_regime_detector'):
            regime_data = self.controller.market_regime_detector.get_current_regime()
        
        # Record performance snapshot
        performance_data = {
            'timestamp': datetime.now(),
            'equity': account.get('equity', 0),
            'cash': account.get('cash', 0),
            'positions_value': account.get('positions_value', 0),
            'num_positions': len(self.paper_adapter.get_positions()),
            'performance_metrics': performance_metrics,
            'allocations': allocations,
            'regime_data': regime_data
        }
        
        self.performance_log.append(performance_data)
    
    def _generate_simulation_report(self, start_time, end_time):
        """Generate a report of the overnight simulation"""
        if not self.performance_log:
            return {
                'status': 'error',
                'message': 'No performance data available'
            }
        
        # Get start and end snapshots
        start_snapshot = self.performance_log[0]
        end_snapshot = self.performance_log[-1]
        
        # Calculate key metrics
        starting_equity = start_snapshot.get('equity', 0)
        ending_equity = end_snapshot.get('equity', 0)
        
        total_pnl = ending_equity - starting_equity
        percent_return = (total_pnl / starting_equity) * 100 if starting_equity > 0 else 0
        
        # Get fill statistics
        fill_stats = self.analyze_fill_performance()
        
        # Create report
        report = {
            'status': 'success',
            'simulation_start': start_time.isoformat(),
            'simulation_end': end_time.isoformat(),
            'trading_cycles': len(self.performance_log),
            'orders_placed': len([o for o in self.order_log if o['event_type'] == 'order_placed']),
            'orders_filled': len([f for f in self.fill_log if f['event_type'] == 'order_filled']),
            'starting_equity': starting_equity,
            'ending_equity': ending_equity,
            'total_pnl': total_pnl,
            'percent_return': percent_return,
            'fill_statistics': {
                'avg_slippage_bps': fill_stats.get('avg_slippage_bps', 0),
                'total_slippage_amount': fill_stats.get('total_slippage_amount', 0),
                'avg_latency_seconds': fill_stats.get('avg_latency_seconds', 0),
            }
        }
        
        # Save report to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join('./results/paper_trading', f'simulation_report_{timestamp}.json')
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Simulation report saved to {report_path}")
        return report

# Function to get a singleton instance
_paper_trading_instance = None

def get_paper_trading_instance() -> AdaptivePaperTrading:
    """Get the global paper trading instance"""
    global _paper_trading_instance
    if _paper_trading_instance is None:
        _paper_trading_instance = AdaptivePaperTrading()
    return _paper_trading_instance

# Usage example (if run as script)
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create instance and initialize
    paper_trading = AdaptivePaperTrading()
    paper_trading.initialize(initial_cash=100000.0)
    
    # Run a trading cycle
    paper_trading.run_trading_cycle()
    
    # Optionally run an overnight simulation
    paper_trading.run_overnight_simulation(
        interval_minutes=5,
        speed_multiplier=300  # 300x speed (5min → 1sec)
    )
    
    # Analyze results
    fill_analysis = paper_trading.analyze_fill_performance()
    print(f"Average slippage: {fill_analysis.get('avg_slippage_bps', 0):.2f} bps")
    print(f"Total slippage cost: ${fill_analysis.get('total_slippage_amount', 0):.2f}")
