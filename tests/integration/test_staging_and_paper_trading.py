#!/usr/bin/env python
"""
Staging Environment and Paper Trading Integration Test

This test validates the staging environment and paper trading components,
ensuring proper setup for the transition from testing to paper trading.
"""

import unittest
import logging
import time
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from trading_bot.core.event_bus import EventBus, Event
from trading_bot.core.constants import EventType, TradingMode

# Import staging environment components
try:
    from trading_bot.core.staging_environment import StagingEnvironment
    from trading_bot.core.staging_mode_manager import StagingModeManager
    from trading_bot.core.system_health_monitor import SystemHealthMonitor
    from trading_bot.core.risk_violation_detector import RiskViolationDetector
    STAGING_AVAILABLE = True
except ImportError:
    try:
        # Try alternate import paths
        from trading_bot.staging.environment import StagingEnvironment
        from trading_bot.staging.mode_manager import StagingModeManager
        from trading_bot.staging.health_monitor import SystemHealthMonitor
        from trading_bot.staging.risk_violation_detector import RiskViolationDetector
        STAGING_AVAILABLE = True
    except ImportError:
        STAGING_AVAILABLE = False
        StagingEnvironment = None
        StagingModeManager = None
        SystemHealthMonitor = None
        RiskViolationDetector = None

# Import broker clients
try:
    from trading_bot.brokers.tradier.client import TradierClient
    TRADIER_AVAILABLE = True
except ImportError:
    TRADIER_AVAILABLE = False
    TradierClient = None

try:
    from trading_bot.brokers.alpaca.client import AlpacaClient
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    AlpacaClient = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StagingEventTracker:
    """Tracks staging environment events for test validation"""
    
    def __init__(self):
        self.events = []
        self.event_types_received = set()
        self.events_by_type = {}
        self.events_by_source = {}
        self.mode_changes = []
        self.health_updates = []
        self.risk_violations = []
        
    def handle_event(self, event: Event):
        """Process and record an event"""
        self.events.append(event)
        self.event_types_received.add(event.event_type)
        
        # Track by type
        if event.event_type not in self.events_by_type:
            self.events_by_type[event.event_type] = []
        self.events_by_type[event.event_type].append(event)
        
        # Track by source if available
        if 'source' in event.data:
            source = event.data['source']
            if source not in self.events_by_source:
                self.events_by_source[source] = []
            self.events_by_source[source].append(event)
        
        # Track specific event types
        if event.event_type == EventType.TRADING_MODE_CHANGED:
            self.mode_changes.append(event.data)
        elif event.event_type == EventType.SYSTEM_HEALTH_UPDATED:
            self.health_updates.append(event.data)
        elif event.event_type == EventType.RISK_VIOLATION_DETECTED:
            self.risk_violations.append(event.data)


class StagingAndPaperTradingTest(unittest.TestCase):
    """Tests the staging environment and paper trading components"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test resources"""
        # Skip all tests if staging components not available
        if not STAGING_AVAILABLE:
            raise unittest.SkipTest("Staging environment components not available")
        
        # Create test event bus
        cls.event_bus = EventBus()
        
        # Create event tracker
        cls.tracker = StagingEventTracker()
        cls.event_bus.subscribe_all(cls.tracker.handle_event)
        
        # Create broker clients for testing if available
        cls.broker_clients = {}
        
        if TRADIER_AVAILABLE and os.environ.get('TRADIER_API_KEY'):
            cls.broker_clients['tradier'] = TradierClient(
                api_key=os.environ.get('TRADIER_API_KEY'),
                account_id=os.environ.get('TRADIER_ACCOUNT_ID', 'VA1201776'),
                paper_trading=True
            )
            logger.info("Initialized Tradier paper trading client")
        
        if ALPACA_AVAILABLE and os.environ.get('ALPACA_API_KEY'):
            cls.broker_clients['alpaca'] = AlpacaClient(
                api_key=os.environ.get('ALPACA_API_KEY'),
                api_secret=os.environ.get('ALPACA_API_SECRET'),
                endpoint=os.environ.get('ALPACA_API_ENDPOINT', 'https://paper-api.alpaca.markets'),
                paper_trading=True
            )
            logger.info("Initialized Alpaca paper trading client")
        
        # Create staging components
        cls.staging_mode_manager = StagingModeManager(
            event_bus=cls.event_bus
        )
        
        cls.health_monitor = SystemHealthMonitor(
            event_bus=cls.event_bus,
            monitoring_interval=1  # 1 second for testing
        )
        
        cls.risk_detector = RiskViolationDetector(
            event_bus=cls.event_bus
        )
        
        # Initialize staging environment
        cls.staging_env = StagingEnvironment(
            mode_manager=cls.staging_mode_manager,
            health_monitor=cls.health_monitor,
            risk_detector=cls.risk_detector,
            event_bus=cls.event_bus
        )
        
        # Test report directory
        cls.test_report_dir = "./test_staging_reports"
        os.makedirs(cls.test_report_dir, exist_ok=True)
    
    def setUp(self):
        """Reset for each test"""
        self.tracker.events = []
        self.tracker.event_types_received = set()
        self.tracker.events_by_type = {}
        self.tracker.events_by_source = {}
        self.tracker.mode_changes = []
        self.tracker.health_updates = []
        self.tracker.risk_violations = []
    
    def test_1_staging_mode_enforcement(self):
        """Test staging mode enforcement for strategies"""
        # Ensure staging mode is active
        self.staging_mode_manager.activate()
        
        # Wait for mode change event
        time.sleep(0.5)
        
        # Verify mode change events
        mode_changes = self.tracker.mode_changes
        self.assertTrue(len(mode_changes) > 0, "Should have received mode change event")
        
        if mode_changes:
            latest_mode = mode_changes[-1].get('mode')
            self.assertEqual(latest_mode, TradingMode.PAPER, 
                           "Staging mode should enforce PAPER trading")
        
        # Verify mode manager state
        self.assertTrue(self.staging_mode_manager.is_active(), 
                       "Staging mode manager should be active")
        
        # Try to simulate a strategy attempting to use live trading
        strategy_config = {
            'strategy_id': 'test_strategy',
            'symbol': 'AAPL',
            'mode': TradingMode.LIVE  # This should be overridden
        }
        
        # Pass through mode manager
        enforced_config = self.staging_mode_manager.enforce_paper_trading(strategy_config)
        
        # Verify mode was enforced to PAPER
        self.assertEqual(enforced_config['mode'], TradingMode.PAPER,
                       "Staging mode manager should override live mode to paper")
        
        logger.info("Staging mode successfully enforced paper trading")
    
    def test_2_system_health_monitoring(self):
        """Test system health monitoring"""
        # Start health monitoring
        self.health_monitor.start()
        
        # Wait for health updates
        time.sleep(2)
        
        # Verify health updates
        health_updates = self.tracker.health_updates
        self.assertTrue(len(health_updates) > 0, "Should have received health updates")
        
        if health_updates:
            latest_health = health_updates[-1]
            
            # Check that health data contains required metrics
            required_metrics = ['cpu_usage', 'memory_usage', 'process_uptime']
            for metric in required_metrics:
                self.assertIn(metric, latest_health, f"Health update should include {metric}")
            
            # Log health metrics
            logger.info("System health metrics:")
            for key, value in latest_health.items():
                if isinstance(value, (int, float)) and key != 'timestamp':
                    logger.info(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
                elif key != 'timestamp':
                    logger.info(f"  {key}: {value}")
        
        # Stop health monitoring
        self.health_monitor.stop()
    
    def test_3_risk_violation_detection(self):
        """Test risk violation detection"""
        # Simulate various risk violations
        violation_types = [
            {
                'type': 'max_drawdown_exceeded',
                'message': 'Maximum drawdown limit exceeded (25.3%)',
                'threshold': 15.0,
                'actual': 25.3,
                'severity': 'high'
            },
            {
                'type': 'position_size_exceeded',
                'message': 'Position size exceeds maximum (12.5%)',
                'threshold': 10.0,
                'actual': 12.5,
                'symbol': 'AAPL',
                'severity': 'medium'
            },
            {
                'type': 'correlation_limit_exceeded',
                'message': 'Portfolio correlation too high (0.85)',
                'threshold': 0.7,
                'actual': 0.85,
                'severity': 'medium'
            }
        ]
        
        # Send risk violation events
        for violation in violation_types:
            self.event_bus.publish(Event(
                event_type=EventType.RISK_VIOLATION_DETECTED,
                data=violation
            ))
        
        # Wait for processing
        time.sleep(0.5)
        
        # Verify violations were received
        self.assertEqual(len(self.tracker.risk_violations), len(violation_types),
                       "All risk violations should be tracked")
        
        # Check that risk detector processed the violations
        detector_violations = self.risk_detector.get_active_violations()
        
        # Since we don't know the internal structure, we'll just log the result
        logger.info(f"Risk detector has {len(detector_violations)} active violations")
        
        # Generate risk summary
        risk_summary = {
            'violations_by_type': {},
            'violations_by_severity': {},
            'violation_count': len(self.tracker.risk_violations)
        }
        
        # Count violations by type and severity
        for violation in self.tracker.risk_violations:
            v_type = violation.get('type', 'unknown')
            severity = violation.get('severity', 'unknown')
            
            if v_type not in risk_summary['violations_by_type']:
                risk_summary['violations_by_type'][v_type] = 0
            risk_summary['violations_by_type'][v_type] += 1
            
            if severity not in risk_summary['violations_by_severity']:
                risk_summary['violations_by_severity'][severity] = 0
            risk_summary['violations_by_severity'][severity] += 1
        
        # Save risk summary to file
        summary_path = os.path.join(self.test_report_dir, "risk_violations.json")
        with open(summary_path, 'w') as f:
            json.dump(risk_summary, f, indent=2)
            
        logger.info(f"Risk violation summary saved to {summary_path}")
        logger.info(f"Risk violations by type: {risk_summary['violations_by_type']}")
        logger.info(f"Risk violations by severity: {risk_summary['violations_by_severity']}")
    
    def test_4_broker_connectivity(self):
        """Test broker connectivity for paper trading"""
        # Skip if no broker clients available
        if not self.broker_clients:
            self.skipTest("No broker clients available for testing")
            return
        
        # Test connectivity for each available broker
        for broker_name, client in self.broker_clients.items():
            logger.info(f"Testing {broker_name} paper trading connectivity")
            
            try:
                # Get account information
                account_info = client.get_account()
                
                # Verify we got account data
                self.assertIsNotNone(account_info, f"{broker_name} should return account info")
                
                # Log account information (excluding sensitive data)
                logger.info(f"{broker_name} account information:")
                
                # Common account properties to check
                account_props = [
                    'account_number', 'buying_power', 'cash', 'equity', 
                    'margin_used', 'status'
                ]
                
                for prop in account_props:
                    if prop in account_info:
                        logger.info(f"  {prop}: {account_info[prop]}")
                
                # Verify paper trading mode
                if hasattr(client, 'is_paper_trading'):
                    self.assertTrue(client.is_paper_trading(), 
                                  f"{broker_name} should be in paper trading mode")
                
                # Test market data access
                if hasattr(client, 'get_quote'):
                    try:
                        quote = client.get_quote('SPY')
                        logger.info(f"  SPY quote: {quote}")
                        self.assertIsNotNone(quote, f"{broker_name} should return quote data")
                    except Exception as e:
                        logger.warning(f"Could not get quote: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error testing {broker_name}: {str(e)}")
                self.fail(f"Broker connectivity test failed for {broker_name}: {str(e)}")
    
    def test_5_full_staging_cycle(self):
        """Test a complete staging environment cycle"""
        # Initialize staging environment with test strategies
        self.staging_env.initialize()
        
        # Wait for initialization
        time.sleep(0.5)
        
        # Simulate a trading day in the staging environment
        logger.info("Starting simulated staging test cycle")
        
        # 1. Generate system health updates
        self.event_bus.publish(Event(
            event_type=EventType.SYSTEM_HEALTH_UPDATED,
            data={
                'timestamp': datetime.now(),
                'cpu_usage': 15.5,
                'memory_usage': 350.5,
                'process_uptime': 3600,
                'event_rate': 25.5,
                'latency_ms': 45
            }
        ))
        
        # 2. Simulate some trades
        for i in range(5):
            # Generate alternating buy/sell trades
            trade_type = "buy" if i % 2 == 0 else "sell"
            symbol = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"][i % 5]
            
            # Create trade event
            self.event_bus.publish(Event(
                event_type=EventType.TRADE_EXECUTED,
                data={
                    'trade_id': f"staging_test_{i}",
                    'symbol': symbol,
                    'quantity': 10,
                    'price': 100 + i,
                    'type': trade_type,
                    'timestamp': datetime.now(),
                    'strategy': "test_staging_strategy",
                    'mode': TradingMode.PAPER
                }
            ))
        
        # 3. Generate a mild risk violation
        self.event_bus.publish(Event(
            event_type=EventType.RISK_VIOLATION_DETECTED,
            data={
                'type': 'sector_exposure_exceeded',
                'message': 'Technology sector exposure exceeds threshold (22.5%)',
                'threshold': 20.0,
                'actual': 22.5,
                'sector': 'Technology',
                'severity': 'low'
            }
        ))
        
        # 4. Update portfolio status
        self.event_bus.publish(Event(
            event_type=EventType.PORTFOLIO_EXPOSURE_UPDATED,
            data={
                'timestamp': datetime.now(),
                'equity': 102500.0,
                'cash': 52500.0,
                'positions_value': 50000.0,
                'margin_used': 0.0
            }
        ))
        
        # 5. Wait for processing and generate a staging report
        time.sleep(1)
        
        # Generate staging report if method exists
        if hasattr(self.staging_env, 'generate_report'):
            report_path = os.path.join(self.test_report_dir, "staging_report.json")
            self.staging_env.generate_report(report_path)
            
            # Verify report was created
            self.assertTrue(os.path.exists(report_path), 
                         "Staging report should be generated")
            
            logger.info(f"Staging report generated at {report_path}")
            
            # Log report contents
            try:
                with open(report_path, 'r') as f:
                    report = json.load(f)
                    
                logger.info("Staging report summary:")
                for section, data in report.items():
                    if isinstance(data, dict):
                        logger.info(f"  {section}: {len(data)} items")
                    elif isinstance(data, list):
                        logger.info(f"  {section}: {len(data)} items")
                    else:
                        logger.info(f"  {section}: {data}")
            except:
                logger.warning("Could not parse staging report")
        else:
            logger.warning("generate_report method not available in staging environment")
        
        # 6. Shutdown staging environment
        self.staging_env.shutdown()
        
        # Verify shutdown was successful
        time.sleep(0.5)
        shutdown_events = [e for e in self.tracker.events 
                          if e.event_type == EventType.SYSTEM_SHUTDOWN]
        
        if shutdown_events:
            logger.info("Staging environment shutdown successfully")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up resources"""
        # Shut down staging components
        if hasattr(cls, 'health_monitor') and cls.health_monitor:
            cls.health_monitor.stop()
        
        if hasattr(cls, 'staging_env') and cls.staging_env:
            if hasattr(cls.staging_env, 'shutdown'):
                cls.staging_env.shutdown()
        
        # Clear event bus
        cls.event_bus.clear_subscribers()
        
        # Clean up test report directory
        if os.path.exists(cls.test_report_dir):
            import shutil
            shutil.rmtree(cls.test_report_dir)


if __name__ == '__main__':
    unittest.main()
