#!/usr/bin/env python3
"""
Simple Smart Features Test

This script tests the basic functionality of the smart forex modules
without requiring additional dependencies.
"""

import os
import sys
import json
import logging
import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('simple_smart_test')

# Import modules (will handle missing imports gracefully)
try:
    from forex_smart_news import SmartNewsAnalyzer
    from forex_smart_compliance import SmartComplianceMonitor
    from forex_smart_benbot import SmartBenBotConnector
    
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Import error: {e}. Will run limited tests.")
    MODULES_AVAILABLE = False


def test_news_analysis():
    """Test the basic functionality of the news analyzer."""
    logger.info("Testing SmartNewsAnalyzer...")
    
    try:
        analyzer = SmartNewsAnalyzer()
        
        # Test with a sample news event
        test_event = {
            'title': 'US Non-Farm Payrolls',
            'importance': 'high',
            'country': 'US',
            'forecast': '200K',
            'previous': '180K'
        }
        
        impact = analyzer.predict_news_impact(test_event, 'EURUSD')
        logger.info(f"Predicted impact: {impact}")
        
        # Test position sizing
        adjusted_size = analyzer.calculate_news_position_size(0.1, [test_event], 'EURUSD', 2.5)
        logger.info(f"Adjusted position size: {adjusted_size:.2f} lots (normal: 0.10 lots)")
        
        # Test post-news opportunity detection
        opportunity = analyzer.is_post_news_entry_opportunity(test_event, 'EURUSD', 15)
        logger.info(f"Post-news opportunity: {opportunity}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing news analysis: {e}")
        return False


def test_compliance_monitor():
    """Test the basic functionality of the compliance monitor."""
    logger.info("Testing SmartComplianceMonitor...")
    
    try:
        monitor = SmartComplianceMonitor()
        
        # Test with different drawdown levels
        monitor.update_account_state(990.0, daily_pnl=-10.0)
        position_size = monitor.calculate_position_size(0.1)
        logger.info(f"Position size with 1% drawdown: {position_size:.2f} lots")
        
        monitor.update_account_state(960.0, daily_pnl=-15.0)
        position_size = monitor.calculate_position_size(0.1)
        logger.info(f"Position size with 4% drawdown: {position_size:.2f} lots")
        
        # Test trading allowed check
        is_allowed, reason = monitor.is_trading_allowed()
        logger.info(f"Trading allowed: {is_allowed} - {reason}")
        
        # Record a sample trade
        monitor.record_trade_result({
            'pair': 'EURUSD',
            'pnl': 15.0,
            'pnl_percent': 0.15,
            'session': 'London'
        })
        
        # Test risk projection
        risk = monitor.project_drawdown_risk()
        logger.info(f"Risk projection completed")
        
        return True
    except Exception as e:
        logger.error(f"Error testing compliance monitor: {e}")
        return False


def test_benbot_connector():
    """Test the basic functionality of the BenBot connector."""
    logger.info("Testing SmartBenBotConnector...")
    
    try:
        # Create connector with a mock endpoint
        connector = SmartBenBotConnector("http://mock-benbot-server")
        
        # Override the _make_benbot_request method to return mock data
        def mock_request(action, data):
            return {
                'decision': data.get('original_decision', True),
                'confidence': 0.8,
                'reasoning': "Mock BenBot response"
            }
        
        connector._make_benbot_request = mock_request
        
        # Test trade entry consultation
        entry_response = connector.consult_benbot_with_confidence(
            'trade_entry',
            {
                'pair': 'EURUSD',
                'decision': True,
                'signal_strength': 0.75,
                'session_optimal': True,
                'strategy_win_rate': 0.6
            }
        )
        
        logger.info(f"Trade entry consultation result: {entry_response.get('decision')}")
        logger.info(f"Source: {entry_response.get('source')}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing BenBot connector: {e}")
        return False


def main():
    """Main test function."""
    if not MODULES_AVAILABLE:
        logger.error("Required modules are not available. Tests cannot run.")
        return False
    
    logger.info("Starting simple smart feature tests...")
    
    results = {
        'news_analysis': test_news_analysis(),
        'compliance_monitor': test_compliance_monitor(),
        'benbot_connector': test_benbot_connector(),
    }
    
    # Print summary
    logger.info("\n--- Test Results Summary ---")
    for test, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test}: {status}")
    
    all_passed = all(results.values())
    logger.info(f"Overall result: {'PASSED' if all_passed else 'FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
