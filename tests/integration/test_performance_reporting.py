#!/usr/bin/env python
"""
Performance Reporting Integration Test

This test validates the performance reporting and monitoring components of the trading pipeline,
ensuring proper performance tracking, reporting, and notification.
"""

import unittest
import logging
import time
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType
from trading_bot.data.persistence import PersistenceManager

# Import monitoring-related modules
try:
    from trading_bot.monitoring.recap_reporting import create_performance_report
except ImportError:
    create_performance_report = None

try:
    from trading_bot.monitoring.performance_tracker import PerformanceTracker
except ImportError:
    # Try alternate import paths
    try:
        from trading_bot.analytics.performance_tracker import PerformanceTracker
    except ImportError:
        try:
            from trading_bot.performance.tracker import PerformanceTracker
        except ImportError:
            PerformanceTracker = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceEventTracker:
    """Tracks performance-related events for test validation"""
    
    def __init__(self):
        self.events = []
        self.event_types_received = set()
        self.events_by_type = {}
        self.trades_completed = []
        self.portfolio_updates = []
        
    def handle_event(self, event: Event):
        """Process and record an event"""
        self.events.append(event)
        self.event_types_received.add(event.event_type)
        
        # Track by type
        if event.event_type not in self.events_by_type:
            self.events_by_type[event.event_type] = []
        self.events_by_type[event.event_type].append(event)
        
        # Track specific event types
        if event.event_type == EventType.TRADE_CLOSED:
            self.trades_completed.append(event.data)
        elif event.event_type == EventType.PORTFOLIO_EXPOSURE_UPDATED:
            self.portfolio_updates.append(event.data)


class PerformanceReportingTest(unittest.TestCase):
    """Tests the performance reporting and monitoring components"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test resources"""
        # Create test event bus
        cls.event_bus = EventBus()
        
        # Create memory-based persistence
        cls.persistence = PersistenceManager(uri="sqlite:///:memory:", db_name="test_reporting")
        
        # Create event tracker
        cls.tracker = PerformanceEventTracker()
        cls.event_bus.subscribe_all(cls.tracker.handle_event)
        
        # Create test directory for reports
        cls.test_report_dir = "./test_reports"
        os.makedirs(cls.test_report_dir, exist_ok=True)
        
        # Create a performance tracker if available
        cls.performance_tracker = None
        if PerformanceTracker is not None:
            cls.performance_tracker = PerformanceTracker(
                persistence_manager=cls.persistence,
                event_bus=cls.event_bus
            )
            logger.info("Initialized PerformanceTracker")
    
    def setUp(self):
        """Reset for each test"""
        self.tracker.events = []
        self.tracker.event_types_received = set()
        self.tracker.events_by_type = {}
        self.tracker.trades_completed = []
        self.tracker.portfolio_updates = []
    
    def _generate_test_trades(self, count=5):
        """Generate test trade events"""
        test_trades = []
        
        symbols = ["AAPL", "MSFT", "SPY", "QQQ", "AMZN"]
        strategies = ["momentum", "trend_following", "breakout", "mean_reversion"]
        
        # Generate trades with a mix of wins and losses
        for i in range(count):
            symbol = symbols[i % len(symbols)]
            strategy = strategies[i % len(strategies)]
            entry_price = 100 + i
            
            # 70% win rate for test data
            is_win = (i % 10) < 7
            exit_price = entry_price * (1.05 if is_win else 0.97)
            
            # Calculate P&L
            qty = 10
            pnl = (exit_price - entry_price) * qty
            
            trade = {
                'trade_id': f"test_trade_{i}",
                'symbol': symbol,
                'strategy': strategy,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': qty,
                'entry_time': datetime.now() - timedelta(hours=24),
                'exit_time': datetime.now() - timedelta(minutes=i*30),
                'pnl': pnl,
                'win': is_win,
                'hold_time': timedelta(hours=24-i/2).total_seconds() / 3600  # in hours
            }
            
            test_trades.append(trade)
            
            # Publish trade closed event
            self.event_bus.publish(Event(
                event_type=EventType.TRADE_CLOSED,
                data=trade
            ))
        
        return test_trades
    
    def _generate_portfolio_updates(self, count=10):
        """Generate portfolio update events"""
        initial_equity = 100000.0
        current_equity = initial_equity
        
        # Generate a series of portfolio updates with a slight uptrend
        for i in range(count):
            # Add some randomness to equity changes
            import random
            pct_change = random.uniform(-0.005, 0.01)  # -0.5% to 1% change
            
            current_equity *= (1 + pct_change)
            
            portfolio_data = {
                'timestamp': datetime.now() - timedelta(hours=count-i),
                'equity': current_equity,
                'cash': current_equity * 0.3,  # 30% cash
                'positions_value': current_equity * 0.7,  # 70% invested
                'margin_used': 0.0,
                'daily_pnl': current_equity - initial_equity if i == count-1 else 0
            }
            
            self.event_bus.publish(Event(
                event_type=EventType.PORTFOLIO_EXPOSURE_UPDATED,
                data=portfolio_data
            ))
        
        return current_equity
    
    def test_1_performance_tracking(self):
        """Test performance tracking of completed trades"""
        # Generate test trades
        test_trades = self._generate_test_trades(10)
        
        # Wait for event processing
        time.sleep(0.5)
        
        # Verify trade events were captured
        self.assertEqual(len(self.tracker.trades_completed), 10)
        
        # If we have a performance tracker, verify it processed the trades
        if self.performance_tracker is not None:
            # Check that trades were recorded
            tracked_trades = self.performance_tracker.get_recent_trades()
            self.assertGreaterEqual(len(tracked_trades), 5)
            
            # Check performance metrics
            metrics = self.performance_tracker.get_performance_metrics()
            self.assertIsNotNone(metrics)
            
            # Log metrics for inspection
            logger.info("Performance metrics calculated:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.warning("PerformanceTracker not available, skipping tracker-specific tests")
    
    def test_2_portfolio_tracking(self):
        """Test portfolio value tracking"""
        # Generate portfolio updates
        final_equity = self._generate_portfolio_updates(10)
        
        # Wait for event processing
        time.sleep(0.5)
        
        # Verify portfolio events were captured
        self.assertEqual(len(self.tracker.portfolio_updates), 10)
        
        # Check final equity value
        if self.tracker.portfolio_updates:
            last_update = self.tracker.portfolio_updates[-1]
            self.assertAlmostEqual(
                last_update['equity'], 
                final_equity, 
                places=2
            )
        
        # If we have a performance tracker, verify it tracked portfolio
        if self.performance_tracker is not None:
            equity_history = self.performance_tracker.get_equity_history()
            self.assertIsNotNone(equity_history)
            self.assertGreaterEqual(len(equity_history), 5)
            
            # Create a simple equity curve
            if isinstance(equity_history, list):
                equity_curve = pd.DataFrame([
                    {'timestamp': entry['timestamp'], 'equity': entry['equity']}
                    for entry in equity_history
                ])
                
                # Log equity curve stats
                if len(equity_curve) > 0:
                    logger.info(f"Equity curve stats: start={equity_curve['equity'].iloc[0]:.2f}, " +
                               f"end={equity_curve['equity'].iloc[-1]:.2f}")
    
    def test_3_performance_report_generation(self):
        """Test generation of performance reports"""
        if create_performance_report is None:
            self.skipTest("create_performance_report function not available")
            return
        
        # Create test data for the report
        today_results = {
            'date': datetime.now(),
            'daily_pnl': 1250.50,
            'daily_return': 0.0125,
            'ending_equity': 101250.50,
            'trades': 10,
            'win_rate': 0.70
        }
        
        benchmark_performance = {
            'SPY': {
                'return': 0.008,
                'correlation': 0.65
            },
            'QQQ': {
                'return': 0.012,
                'correlation': 0.72
            }
        }
        
        # Generate a report
        try:
            report_path = create_performance_report(
                today_results,
                benchmark_performance,
                alerts=[],
                suggestions=[],
                output_dir=self.test_report_dir
            )
            
            # Verify report exists
            self.assertTrue(os.path.exists(report_path))
            logger.info(f"Performance report generated at: {report_path}")
            
            # Verify report content
            with open(report_path, 'r') as f:
                report_content = f.read()
                
            self.assertIn("Daily PnL", report_content)
            self.assertIn("$1250.50", report_content.replace(",", ""))  # Account for possible formatting
            self.assertIn("70%", report_content)  # Win rate
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            self.fail(f"Report generation failed: {str(e)}")
    
    def test_4_strategy_performance_comparison(self):
        """Test comparison of strategy performance"""
        # Generate test trades for multiple strategies
        strategies = {
            "momentum": [],
            "trend_following": [],
            "breakout": [],
            "mean_reversion": []
        }
        
        # Create trades for each strategy
        for strategy in strategies.keys():
            for i in range(5):
                # Create win/loss patterns unique to each strategy
                is_win = False
                if strategy == "momentum" and i % 3 != 0:  # 4/5 win rate
                    is_win = True
                elif strategy == "trend_following" and i % 2 == 0:  # 3/5 win rate
                    is_win = True
                elif strategy == "breakout" and i % 5 < 3:  # 3/5 win rate
                    is_win = True
                elif strategy == "mean_reversion" and i % 4 != 0:  # 4/5 win rate
                    is_win = True
                
                # P&L magnitude varies by strategy
                pnl_multiplier = {
                    "momentum": 2.0,
                    "trend_following": 3.0,
                    "breakout": 2.5,
                    "mean_reversion": 1.5
                }.get(strategy, 1.0)
                
                # Base P&L
                base_pnl = 100 * (1.05 if is_win else 0.95)
                pnl = (base_pnl - 100) * pnl_multiplier
                
                trade = {
                    'trade_id': f"{strategy}_trade_{i}",
                    'strategy': strategy,
                    'symbol': f"SYMB_{i}",
                    'entry_price': 100,
                    'exit_price': 100 + pnl/10,  # Derive from PnL for simplicity
                    'quantity': 10,
                    'entry_time': datetime.now() - timedelta(hours=24),
                    'exit_time': datetime.now() - timedelta(minutes=i*30),
                    'pnl': pnl,
                    'win': is_win
                }
                
                strategies[strategy].append(trade)
                
                # Publish trade event
                self.event_bus.publish(Event(
                    event_type=EventType.TRADE_CLOSED,
                    data=trade
                ))
        
        # Wait for event processing
        time.sleep(0.5)
        
        # Calculate and compare strategy performance
        strategy_performance = {}
        
        for strategy, trades in strategies.items():
            total_pnl = sum(t['pnl'] for t in trades)
            win_rate = sum(1 for t in trades if t['win']) / len(trades)
            avg_win = np.mean([t['pnl'] for t in trades if t['win']]) if any(t['win'] for t in trades) else 0
            avg_loss = np.mean([t['pnl'] for t in trades if not t['win']]) if any(not t['win'] for t in trades) else 0
            
            strategy_performance[strategy] = {
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'trades': len(trades)
            }
        
        # Log strategy performance comparison
        logger.info("Strategy Performance Comparison:")
        for strategy, metrics in strategy_performance.items():
            logger.info(f"  {strategy.upper()}: PnL=${metrics['total_pnl']:.2f}, " +
                       f"Win={metrics['win_rate']:.1%}, " +
                       f"Avg Win=${metrics['avg_win']:.2f}, " +
                       f"Avg Loss=${metrics['avg_loss']:.2f}")
        
        # Save comparison to JSON file
        comparison_path = os.path.join(self.test_report_dir, "strategy_comparison.json")
        with open(comparison_path, 'w') as f:
            json.dump(strategy_performance, f, indent=2)
        
        logger.info(f"Saved strategy comparison to {comparison_path}")
        
        # Verify file was created
        self.assertTrue(os.path.exists(comparison_path))
    
    def test_5_alert_generation(self):
        """Test generation of monitoring alerts"""
        # Create test alert events
        alert_types = [
            ("WARNING", "Strategy performance deteriorating", "momentum"),
            ("INFO", "New high water mark reached", None),
            ("ERROR", "API connection failure", None),
            ("WARNING", "Position size approaching limit", "AAPL")
        ]
        
        alerts_sent = []
        
        for severity, message, source in alert_types:
            alert_data = {
                'timestamp': datetime.now(),
                'severity': severity,
                'message': message,
                'source': source
            }
            
            self.event_bus.publish(Event(
                event_type=EventType.ALERT_GENERATED,
                data=alert_data
            ))
            
            alerts_sent.append(alert_data)
        
        # Wait for event processing
        time.sleep(0.5)
        
        # Verify alerts were captured
        alerts_received = [e.data for e in self.tracker.events 
                          if e.event_type == EventType.ALERT_GENERATED]
        
        self.assertEqual(len(alerts_received), len(alert_types))
        
        # Log alerts
        logger.info("Alerts generated:")
        for alert in alerts_received:
            logger.info(f"  [{alert['severity']}] {alert['message']} " + 
                       f"(Source: {alert['source'] or 'system'})")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up resources"""
        # Clear event bus
        cls.event_bus.clear_subscribers()
        
        # Stop performance tracker if exists
        if cls.performance_tracker is not None and hasattr(cls.performance_tracker, 'stop'):
            cls.performance_tracker.stop()
        
        # Clean up test report directory
        if os.path.exists(cls.test_report_dir):
            import shutil
            shutil.rmtree(cls.test_report_dir)


if __name__ == '__main__':
    unittest.main()
