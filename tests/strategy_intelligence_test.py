#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Strategy Intelligence Test Script

This script tests the Strategy Intelligence Recorder system by:
1. Initializing all enhanced components including the recorder
2. Starting the live data source to generate market data events
3. Verifying that events are being captured and stored in MongoDB
4. Displaying the recorded strategy intelligence data
"""
import os
import sys
import time
import logging
import argparse
from pprint import pprint
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required components
from trading_bot.core.enhanced_integration import EnhancedComponents, create_demo_intelligence_data
from trading_bot.core.strategy_intelligence_recorder import StrategyIntelligenceRecorder
from trading_bot.core.event_bus import EventBus, get_global_event_bus, Event
from trading_bot.core.constants import EventType
from trading_bot.data.persistence import PersistenceManager

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Strategy Intelligence Test')
    parser.add_argument('--mongo-uri', dest='mongo_uri', default=os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/'),
                      help='MongoDB connection URI (default: %(default)s)')
    parser.add_argument('--db-name', dest='db_name', default=os.environ.get('MONGODB_DATABASE', 'bensbot'),
                      help='MongoDB database name (default: %(default)s)')
    parser.add_argument('--mock-data', dest='mock_data', action='store_true',
                      help='Use mock data instead of running live simulation')
    parser.add_argument('--duration', dest='duration', type=int, default=60,
                      help='Test duration in seconds (default: %(default)s)')
    
    return parser.parse_args()

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_collection_contents(persistence, collection_name, limit=5):
    """Print the contents of a MongoDB collection."""
    try:
        documents = persistence.find_documents(collection_name, {}, limit=limit)
        if documents:
            print(f"\nCollection: {collection_name} (showing {min(limit, len(documents))} of {len(documents)} documents)")
            for i, doc in enumerate(documents[:limit]):
                print(f"\nDocument {i+1}:")
                pprint(doc)
        else:
            print(f"\nCollection {collection_name} is empty")
    except Exception as e:
        print(f"Error reading collection {collection_name}: {str(e)}")

def test_with_mock_data(args):
    """Test with mock data without running the live simulation."""
    print_section("TESTING WITH MOCK DATA")
    print("Initializing components with mock data...")
    
    # Create components with mock data
    components = create_demo_intelligence_data()
    
    # Verify the data is available
    if not components.persistence or not components.persistence.is_connected():
        print("ERROR: Persistence layer not connected")
        return False
    
    # Display the mock data
    print("\nVerifying mock strategy intelligence data:")
    
    print_section("MARKET REGIME DATA")
    regime_data = components.persistence.load_strategy_state("market_regime_detector")
    print("Current Market Regime:")
    pprint(regime_data)
    
    print_section("ASSET SELECTION DATA")
    asset_data = components.persistence.load_strategy_state("market_analysis")
    print("Asset Selection:")
    if asset_data and "asset_classes" in asset_data:
        for asset in asset_data["asset_classes"]:
            print(f"- {asset['asset_class']}: Score {asset['opportunity_score']} - {asset['selection_reason']}")
    
    print_section("SYMBOL SELECTION DATA")
    symbol_data = components.persistence.load_strategy_state("symbol_selection")
    print("Symbol Rankings:")
    if symbol_data and "rankings" in symbol_data:
        df = pd.DataFrame(symbol_data["rankings"])
        print(df[["symbol", "total_score", "rank"]].to_string(index=False))
    
    print_section("STRATEGY COMPATIBILITY DATA")
    compat_data = components.persistence.load_strategy_state("strategy_compatibility")
    print("Strategy Compatibility Matrix:")
    if compat_data and "matrix" in compat_data:
        # Convert to pandas DataFrame
        strategies = list(compat_data["matrix"].keys())
        regimes = list(compat_data["matrix"][strategies[0]].keys())
        data = []
        for strategy in strategies:
            row = [strategy]
            for regime in regimes:
                row.append(compat_data["matrix"][strategy][regime])
            data.append(row)
        
        df = pd.DataFrame(data, columns=["Strategy"] + regimes)
        print(df.to_string(index=False))
    
    print_section("PERFORMANCE ATTRIBUTION DATA")
    attribution_data = components.persistence.load_strategy_state("performance_attribution")
    print("Performance Attribution:")
    if attribution_data and "factors" in attribution_data:
        df = pd.DataFrame(attribution_data["factors"])
        print(df.to_string(index=False))
    
    print_section("STRATEGY ADAPTATION DATA")
    adaptation_data = components.persistence.load_strategy_state("strategy_adaptation")
    print("Strategy Adaptation Events:")
    if adaptation_data and "events" in adaptation_data:
        for event in adaptation_data["events"][:5]:  # Show first 5 events
            print(f"- {event['timestamp']}: {event['strategy']} - {event['event_type']} - {event['description']}")
    
    print("\nMock data testing completed successfully!")
    return True

def test_with_live_simulation(args):
    """Test with live data simulation."""
    print_section("TESTING WITH LIVE SIMULATION")
    print(f"Initializing components for live simulation (duration: {args.duration} seconds)...")
    
    # Create configuration
    config = {
        'mongodb_uri': args.mongo_uri,
        'mongodb_database': args.db_name,
        'use_live_data': True,  # Enable live data source
        'live_data_config': {
            'provider': 'simulation',
            'symbols': ['EUR/USD', 'GBP/USD', 'USD/JPY'],
            'timeframes': ['1m', '5m', '1h']
        }
    }
    
    # Initialize components
    components = EnhancedComponents()
    if not components.initialize(config):
        print("ERROR: Failed to initialize components")
        return False
    
    # Subscribe to events to display them
    event_count = {'total': 0}
    
    def event_listener(event: Event):
        """Listen to all events and count them."""
        event_type = event.event_type
        if event_type not in event_count:
            event_count[event_type] = 0
        event_count[event_type] += 1
        event_count['total'] += 1
        
        # Print select events for visibility
        if event_type in [EventType.MARKET_REGIME_CHANGED, EventType.MARKET_REGIME_DETECTED,
                        EventType.ASSET_CLASS_SELECTED, EventType.STRATEGY_SELECTED]:
            print(f"Event: {event_type}, Data: {event.data}")
    
    # Subscribe to all events
    event_bus = get_global_event_bus()
    event_bus.subscribe_all(event_listener)
    
    # Start services
    print("Starting services...")
    components.start_services()
    
    # Run for the specified duration
    print(f"Running simulation for {args.duration} seconds...")
    start_time = time.time()
    try:
        while time.time() - start_time < args.duration:
            # Print event count every 5 seconds
            if int(time.time() - start_time) % 5 == 0:
                print(f"Events processed: {event_count['total']}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Test interrupted by user")
    finally:
        # Stop services
        print("Stopping services...")
        components.stop_services()
    
    # Display event counts
    print("\nEvent counts:")
    for event_type, count in event_count.items():
        if event_type != 'total':
            print(f"- {event_type}: {count}")
    print(f"Total events: {event_count['total']}")
    
    # Verify data in MongoDB
    print("\nVerifying data in MongoDB...")
    if components.persistence and components.persistence.is_connected():
        # Check strategy intelligence data
        states = [
            "market_regime_detector",
            "market_analysis",
            "symbol_selection",
            "strategy_compatibility",
            "performance_attribution",
            "strategy_adaptation"
        ]
        
        for state_name in states:
            state_data = components.persistence.load_strategy_state(state_name)
            print(f"\n{state_name}:")
            if state_data:
                print(f"- State found with {len(state_data)} keys")
            else:
                print("- No data found")
        
        # Check OHLC data
        print("\nOHLC Data Collections:")
        collections = components.persistence.list_collections()
        ohlc_collections = [coll for coll in collections if coll.startswith("ohlc_")]
        for coll in ohlc_collections:
            doc_count = components.persistence.count_documents(coll, {})
            print(f"- {coll}: {doc_count} documents")
    else:
        print("ERROR: Persistence layer not connected")
    
    print("\nLive simulation testing completed!")
    return True

def main():
    """Main function."""
    args = parse_args()
    
    print_section("STRATEGY INTELLIGENCE TEST")
    print(f"MongoDB URI: {args.mongo_uri}")
    print(f"Database: {args.db_name}")
    
    if args.mock_data:
        success = test_with_mock_data(args)
    else:
        success = test_with_live_simulation(args)
    
    if success:
        print("\nTest completed successfully!")
        return 0
    else:
        print("\nTest failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
