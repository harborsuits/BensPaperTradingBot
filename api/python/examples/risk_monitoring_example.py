#!/usr/bin/env python3
"""
Example script demonstrating risk management and system monitoring features.
"""

import os
import time
import logging
import threading
import argparse
from datetime import datetime
import json
import random
from pathlib import Path

from trading_bot.risk.risk_manager import RiskManager
from trading_bot.monitoring.system_monitor import SystemMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Risk management and system monitoring example')
parser.add_argument('--account_size', type=float, default=100000.0, help='Initial account size')
parser.add_argument('--daily_loss_limit', type=float, default=1.0, help='Daily loss limit percentage')
parser.add_argument('--max_drawdown', type=float, default=5.0, help='Maximum drawdown percentage')
parser.add_argument('--position_limit', type=float, default=5.0, help='Position concentration limit percentage')
parser.add_argument('--email', type=str, default='', help='Email address for notifications')
parser.add_argument('--simulate_errors', action='store_true', help='Simulate system errors and risk breaches')
parser.add_argument('--duration', type=int, default=300, help='Duration to run the simulation in seconds')
args = parser.parse_args()

# Example notification configuration
notification_config = {
    'email': {
        'enabled': bool(args.email),
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': os.environ.get('EMAIL_USERNAME', ''),
        'password': os.environ.get('EMAIL_PASSWORD', ''),
        'sender': os.environ.get('EMAIL_SENDER', args.email),
        'recipients': [args.email] if args.email else []
    },
    'telegram': {
        'enabled': False,
        'bot_token': os.environ.get('TELEGRAM_BOT_TOKEN', ''),
        'chat_ids': os.environ.get('TELEGRAM_CHAT_IDS', '').split(',')
    }
}

# Initialize directory to save results
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

def emergency_shutdown_callback(reason: str) -> None:
    """Example emergency shutdown callback."""
    logger.critical(f"EMERGENCY SHUTDOWN TRIGGERED: {reason}")
    logger.critical("Executing emergency procedures...")
    
    # In a real system, you would:
    # 1. Close all open positions (market orders)
    # 2. Cancel all pending orders
    # 3. Disable trading systems
    # 4. Notify administrators
    
    # For this example, we just log the event
    with open(results_dir / 'emergency_shutdown.log', 'a') as f:
        f.write(f"{datetime.now()}: EMERGENCY SHUTDOWN - {reason}\n")
    
    logger.critical("Emergency procedures completed.")

def initialize_risk_manager() -> RiskManager:
    """Initialize the risk management system."""
    logger.info("Initializing risk management system...")
    
    # Initialize risk manager
    risk_manager = RiskManager(
        account_size=args.account_size,
        daily_loss_limit_pct=args.daily_loss_limit,
        max_drawdown_pct=args.max_drawdown,
        position_limit_pct=args.position_limit,
        sector_limit_pct=20.0,
        asset_class_limits={
            'equity': 80.0,
            'options': 20.0,
            'futures': 30.0,
            'forex': 30.0,
            'crypto': 10.0
        },
        enable_emergency_shutdown=True,
        notification_config=notification_config
    )
    
    # Register emergency shutdown callback
    risk_manager.register_emergency_shutdown_callback(emergency_shutdown_callback)
    
    logger.info("Risk management system initialized successfully.")
    return risk_manager

def initialize_system_monitor() -> SystemMonitor:
    """Initialize the system monitoring."""
    logger.info("Initializing system monitoring...")
    
    # Configure system monitor
    monitor_config = {
        'cpu_threshold': 80,
        'memory_threshold': 80,
        'disk_threshold': 90,
        'check_interval': 10,  # seconds
        'processes_to_monitor': ['python'],
        'check_external_connectivity': True,
        'connectivity_test_urls': [
            'https://www.google.com',
            'https://api.binance.com/api/v3/time'
        ],
        'api_endpoints': [
            {
                'url': 'https://api.binance.com/api/v3/time',
                'method': 'GET',
                'expected_status': 200
            }
        ]
    }
    
    # Create monitor instance
    monitor = SystemMonitor(
        check_interval=10,  # seconds
        enable_auto_recovery=True,
        notification_config=notification_config
    )
    
    # Update configuration
    monitor.config.update(monitor_config)
    
    # Register custom health check
    monitor.register_custom_health_check(
        name="Database Connection",
        check_function=lambda: {
            'status': 'ok',
            'message': 'Database connection is healthy',
            'latency_ms': 15
        }
    )
    
    logger.info("System monitoring initialized successfully.")
    return monitor

def simulate_trading_activity(risk_manager: RiskManager) -> None:
    """Simulate trading activity to test risk management."""
    logger.info("Starting trading activity simulation...")
    
    # Example positions
    positions = {
        'AAPL': {
            'quantity': 100,
            'value': 17500.0,
            'asset_class': 'equity',
            'sector': 'technology'
        },
        'GOOGL': {
            'quantity': 50,
            'value': 12000.0,
            'asset_class': 'equity',
            'sector': 'technology'
        },
        'SPY': {
            'quantity': 200,
            'value': 10000.0,
            'asset_class': 'equity',
            'sector': 'index'
        },
        'XOM': {
            'quantity': 150,
            'value': 9000.0,
            'asset_class': 'equity',
            'sector': 'energy'
        }
    }
    
    # Update positions
    risk_manager.update_positions(positions)
    
    # Example trades
    trades = [
        {
            'symbol': 'MSFT',
            'trade_type': 'market',
            'direction': 'buy',
            'quantity': 75,
            'price': 200.0,
            'asset_class': 'equity',
            'sector': 'technology'
        },
        {
            'symbol': 'AMZN',
            'trade_type': 'limit',
            'direction': 'buy',
            'quantity': 20,
            'price': 150.0,
            'asset_class': 'equity',
            'sector': 'technology'
        },
        {
            'symbol': 'SPY',
            'trade_type': 'market',
            'direction': 'sell',
            'quantity': 50,
            'price': 420.0,
            'asset_class': 'equity',
            'sector': 'index'
        }
    ]
    
    # Check each trade
    for trade in trades:
        result = risk_manager.check_trade_risk(
            symbol=trade['symbol'],
            trade_type=trade['trade_type'],
            direction=trade['direction'],
            quantity=trade['quantity'],
            price=trade['price'],
            asset_class=trade['asset_class'],
            sector=trade['sector']
        )
        
        logger.info(f"Trade check for {trade['symbol']}: {result['approved']} - {result['reason']}")
    
    # Simulate account balance changes
    current_balance = args.account_size
    
    # If simulating errors, create a losing scenario
    if args.simulate_errors:
        # Simulate a significant daily loss
        daily_loss = -args.account_size * (args.daily_loss_limit * 1.1) / 100
        current_balance += daily_loss
        logger.warning(f"Simulating daily loss: ${daily_loss:.2f}")
    
    # Update account balance
    risk_manager.update_account_balance(current_balance)
    
    # Get risk status
    status = risk_manager.get_risk_status()
    
    # Log risk status summary
    logger.info(f"Account balance: ${status['account_size']:.2f}")
    logger.info(f"Daily P&L: ${status['daily_pnl']:.2f} ({status['daily_pnl_pct']:.2f}%)")
    logger.info(f"Trading enabled: {status['trading_enabled']}")
    logger.info(f"Risk reduction mode: {status['risk_reduction_mode']}")
    logger.info(f"Emergency shutdown: {status['emergency_shutdown']['triggered']}")
    
    # Save risk log
    log_path = risk_manager.save_risk_log()
    logger.info(f"Risk log saved to {log_path}")
    
    # Get recommendations
    recommendations = risk_manager.get_risk_recommendations()
    
    for rec in recommendations:
        logger.info(f"Risk recommendation [{rec['type']}]: {rec['message']}")
        logger.info(f"  Action: {rec['action']}")

def simulate_system_issues(monitor: SystemMonitor) -> None:
    """Simulate system issues to test monitoring alerts."""
    if not args.simulate_errors:
        return
    
    logger.info("Simulating system issues...")
    
    # Add a critical alert to the queue
    monitor.alert_queue.put({
        'timestamp': datetime.now(),
        'type': 'simulated_issue',
        'source': 'simulator',
        'severity': 'critical',
        'message': 'This is a simulated critical issue for testing purposes.'
    })
    
    # Simulate high CPU usage
    def cpu_stress():
        logger.info("Starting CPU stress test...")
        end_time = time.time() + 5
        while time.time() < end_time:
            # Busy loop to spike CPU
            for _ in range(10000000):
                pass
        logger.info("CPU stress test completed.")
    
    # Start CPU stress in a separate thread
    threading.Thread(target=cpu_stress).start()
    
    logger.info("System issues simulation completed.")

def main():
    """Main function."""
    logger.info("Starting risk management and system monitoring example")
    
    # Initialize risk manager
    risk_manager = initialize_risk_manager()
    
    # Initialize system monitor
    monitor = initialize_system_monitor()
    
    # Start system monitoring
    monitor.start()
    
    try:
        # Run for the specified duration
        end_time = time.time() + args.duration
        
        while time.time() < end_time:
            # Simulate trading activity
            simulate_trading_activity(risk_manager)
            
            # Simulate system issues if enabled
            simulate_system_issues(monitor)
            
            # Wait before next iteration
            time.sleep(30)
            
            # Update with gradually declining balance if simulating errors
            if args.simulate_errors:
                current_balance = risk_manager.account_size * 0.99  # 1% decline each time
                risk_manager.update_account_balance(current_balance)
                logger.info(f"Updated account balance: ${current_balance:.2f}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Stop system monitoring
        monitor.stop()
        
        # Save final status reports
        status_report = monitor.get_status_report()
        risk_status = risk_manager.get_risk_status()
        
        # Save results
        with open(results_dir / 'final_status_report.json', 'w') as f:
            json.dump(status_report, f, indent=2, default=str)
        
        with open(results_dir / 'final_risk_status.json', 'w') as f:
            json.dump(risk_status, f, indent=2, default=str)
        
        logger.info("Final reports saved to results directory")
        logger.info("Example completed")

if __name__ == "__main__":
    main() 