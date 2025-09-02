#!/usr/bin/env python3
"""
Test Dashboard System

This script demonstrates the complete authentication system and dashboard.
It sets up a test environment with sample brokers and events, then launches
the Streamlit dashboard for credential management and audit log visualization.
"""

import os
import sys
import json
import logging
import tempfile
import random
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from trading_bot.brokers.auth_manager import (
    initialize_auth_system, create_default_config, save_config
)
from trading_bot.core.event_bus import (
    get_global_event_bus, Event
)
from trading_bot.core.constants import EventType
from trading_bot.brokers.trade_audit_log import AuditEventType


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_dashboard")


def create_test_config():
    """Create a test configuration with sample brokers"""
    config = create_default_config()
    
    # Make sure directories exist
    os.makedirs("config", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Update config with sample brokers
    config['credential_store'] = {
        'type': 'encrypted',
        'path': 'data/credentials.enc',
        'master_password': os.environ.get('TRADING_BOT_MASTER_PASSWORD', 'test_master_password')
    }
    
    config['audit_log'] = {
        'enabled': True,
        'type': 'sqlite',
        'path': 'data/trading_audit.db'
    }
    
    config['brokers'] = {
        'tradier': {
            'enabled': True,
            'api_key': 'DEMO_TRADIER_API_KEY',
            'account_id': 'DEMO_TRADIER_ACCOUNT',
            'sandbox': True,
            'primary': True
        },
        'alpaca': {
            'enabled': True,
            'api_key': 'DEMO_ALPACA_API_KEY',
            'api_secret': 'DEMO_ALPACA_API_SECRET',
            'paper_trading': True
        },
        'etrade': {
            'enabled': False,
            'consumer_key': 'DEMO_ETRADE_CONSUMER_KEY',
            'consumer_secret': 'DEMO_ETRADE_CONSUMER_SECRET',
            'sandbox': True
        }
    }
    
    config['asset_routing'] = {
        'stock': 'tradier',
        'option': 'tradier',
        'forex': 'alpaca',
        'crypto': 'alpaca'
    }
    
    # Save config to file
    config_path = "config/test_broker_config.json"
    save_config(config, config_path)
    
    return config_path


def generate_sample_events(event_bus, count=50):
    """Generate sample events to populate the audit log"""
    logger.info(f"Generating {count} sample events for audit log...")
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'FB', 'NVDA', 'EURUSD', 'GBPUSD', 'BTCUSD']
    brokers = ['tradier', 'alpaca']
    strategies = ['momentum_1', 'trend_following', 'mean_reversion', 'breakout', 'volatility_etf']
    
    event_types = [
        EventType.ORDER_CREATED,
        EventType.ORDER_SUBMITTED,
        EventType.ORDER_FILLED,
        EventType.ORDER_CANCELLED,
        EventType.ORDER_REJECTED,
        EventType.TRADE_EXECUTED,
        EventType.TRADE_CLOSED,
        EventType.STRATEGY_STARTED,
        EventType.STRATEGY_STOPPED,
        EventType.SIGNAL_GENERATED,
        EventType.SYSTEM_STARTED,
        EventType.HEALTH_CHECK,
        EventType.RISK_LIMIT_REACHED,
        EventType.POSITION_SIZE_CALCULATED
    ]
    
    # Create a set of order IDs that we'll reuse for order lifecycle events
    order_ids = [f"order_{i}" for i in range(1, 21)]
    
    # Generate events over the past 7 days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    time_range = (end_time - start_time).total_seconds()
    
    for i in range(count):
        # Generate random timestamp within the past week
        random_seconds = random.randint(0, int(time_range))
        timestamp = start_time + timedelta(seconds=random_seconds)
        
        # Select random attributes
        event_type = random.choice(event_types)
        symbol = random.choice(symbols)
        broker_id = random.choice(brokers)
        strategy_id = random.choice(strategies)
        
        # Create event data
        data = {
            'symbol': symbol,
            'broker_id': broker_id,
            'strategy_id': strategy_id,
            'timestamp': timestamp.isoformat()
        }
        
        # Add order-specific data for order events
        if event_type in [EventType.ORDER_CREATED, EventType.ORDER_SUBMITTED, 
                         EventType.ORDER_FILLED, EventType.ORDER_CANCELLED, 
                         EventType.ORDER_REJECTED, EventType.TRADE_EXECUTED,
                         EventType.TRADE_CLOSED]:
            
            order_id = random.choice(order_ids)
            data['order_id'] = order_id
            data['quantity'] = random.randint(1, 100)
            data['price'] = round(random.uniform(50.0, 500.0), 2)
            data['order_type'] = random.choice(['market', 'limit', 'stop'])
            
        # Add strategy-specific data for strategy events
        if event_type in [EventType.STRATEGY_STARTED, EventType.STRATEGY_STOPPED, 
                         EventType.SIGNAL_GENERATED]:
            
            data['parameters'] = {
                'timeframe': random.choice(['1m', '5m', '15m', '1h', '4h', '1d']),
                'lookback': random.randint(10, 100),
                'threshold': round(random.uniform(0.5, 2.0), 2)
            }
        
        # Create and publish the event with the historical timestamp
        event = Event(
            event_type=event_type,
            data=data,
            source='test_script',
            timestamp=timestamp
        )
        
        # Publish event to event bus
        event_bus.publish(event)
        
        # Sleep a tiny bit to avoid overwhelming the system
        time.sleep(0.01)
    
    logger.info("Sample events generated successfully")


def launch_dashboard(config_path):
    """Launch the Streamlit dashboard"""
    try:
        import streamlit.web.cli as stcli
        
        logger.info(f"Launching Streamlit dashboard with config: {config_path}")
        
        sys.argv = [
            "streamlit", "run",
            "dashboard/secure_dashboard.py",
            "--server.port=8501",
            f"-- --config={config_path}"
        ]
        
        stcli.main()
    except ImportError:
        logger.error("Streamlit is not installed. Please install it with 'pip install streamlit'")
        print("\n⚠️ Streamlit is not installed. Please install it with:")
        print("pip install streamlit\n")


def main():
    """Main function to set up the test environment and launch the dashboard"""
    print("\n🔒 Secure Trading System Dashboard - Test Environment\n")
    
    # Check for master password in environment
    if 'TRADING_BOT_MASTER_PASSWORD' not in os.environ:
        password = "test_master_password"
        os.environ['TRADING_BOT_MASTER_PASSWORD'] = password
        print(f"❗ Setting test master password: {password}")
        print("❗ In production, set this securely via environment variable:")
        print("   export TRADING_BOT_MASTER_PASSWORD='your_secure_password'")
    else:
        print("✅ Using master password from environment variable")
    
    # Create test configuration
    config_path = create_test_config()
    print(f"✅ Created test configuration at: {config_path}")
    
    # Initialize the authentication system
    print("🔄 Initializing authentication system...")
    config = {}
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    credential_store, audit_log, audit_listener = initialize_auth_system(config)
    
    if credential_store and audit_log and audit_listener:
        print("✅ Authentication system initialized successfully")
        print(f"🔑 Credential store: {credential_store.__class__.__name__}")
        print(f"📝 Audit log: {audit_log.__class__.__name__}")
        print(f"👂 Audit listener: {audit_listener.__class__.__name__}")
    else:
        print("❌ Failed to initialize authentication system")
        return
    
    # Get event bus and generate sample events
    event_bus = get_global_event_bus()
    print("🔄 Generating sample events...")
    generate_sample_events(event_bus, count=100)
    print("✅ Sample events generated successfully")
    
    # Launch dashboard
    print("\n🚀 Launching Streamlit dashboard...")
    print("🌐 URL: http://localhost:8501")
    print("📊 The dashboard will open in your default browser")
    print("❗ Press Ctrl+C to stop the dashboard")
    print("\n")
    
    # Launch the dashboard
    launch_dashboard(config_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Exiting test dashboard")
    except Exception as e:
        logger.exception("Error running test dashboard")
        print(f"\n❌ Error: {str(e)}")
