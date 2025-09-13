#!/usr/bin/env python
"""
Market Regime System CLI

Command-line interface for initializing, configuring, and monitoring
the market regime detection system.
"""

import argparse
import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import regime components
from trading_bot.analytics.market_regime.bootstrap import (
    DEFAULT_CONFIG, DEFAULT_PARAMETERS, DEFAULT_TIMEFRAME_MAPPINGS,
    DEFAULT_STRATEGY_CONFIGS, initialize_regime_system_with_defaults
)
from trading_bot.analytics.market_regime.detector import MarketRegimeType
from trading_bot.analytics.market_regime.integration import MarketRegimeManager
from trading_bot.analytics.market_regime.data_collector import create_regime_data_collector
from trading_bot.core.event_bus import EventBus
from trading_bot.brokers.multi_broker_manager import MultiBrokerManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("regime_cli")

def init_command(args):
    """Initialize the market regime system configuration."""
    output_path = args.output
    
    # Create config dictionary
    config = DEFAULT_CONFIG.copy()
    
    # Modify based on arguments
    if args.symbols:
        config["watched_symbols"] = args.symbols.split(',')
    
    if args.timeframes:
        config["default_timeframes"] = args.timeframes.split(',')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Generated configuration file at: {output_path}")
    print(f"Configured to monitor {len(config['watched_symbols'])} symbols")
    
    # If requested, also generate parameter files
    if args.generate_params:
        params_dir = os.path.join(os.path.dirname(output_path), "regime_parameters")
        os.makedirs(params_dir, exist_ok=True)
        
        for strategy_id, params in DEFAULT_PARAMETERS.items():
            # Convert MarketRegimeType to strings
            strategy_params = {}
            for regime_type, param_set in params.items():
                strategy_params[regime_type.value] = param_set
            
            # Write to file
            param_path = os.path.join(params_dir, f"strategy_{strategy_id}.json")
            with open(param_path, 'w') as f:
                json.dump(strategy_params, f, indent=2)
        
        print(f"Generated parameter files for {len(DEFAULT_PARAMETERS)} strategies in {params_dir}")
    
    return 0

def status_command(args):
    """Get status of the market regime system."""
    # Load config
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        return 1
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print("Market Regime System Status")
    print("--------------------------")
    print(f"Monitoring {len(config.get('watched_symbols', []))} symbols")
    print(f"Using timeframes: {', '.join(config.get('default_timeframes', []))}")
    print(f"Primary timeframe: {config.get('primary_timeframe', '1d')}")
    print(f"Auto-update: {config.get('detector', {}).get('auto_update', True)}")
    print(f"Detection interval: {config.get('detector', {}).get('detection_interval_seconds', 1800)} seconds")
    
    # If connected to a running system, we could get real-time status
    # but that would require more infrastructure
    
    return 0

def analyze_command(args):
    """Analyze historical data for regime detection."""
    symbol = args.symbol
    timeframe = args.timeframe
    data_file = args.data_file
    
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return 1
    
    # Load data
    try:
        if data_file.endswith('.csv'):
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        elif data_file.endswith('.json'):
            df = pd.read_json(data_file, orient='records')
        else:
            print(f"Unsupported file format: {data_file}")
            return 1
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return 1
    
    # Create temporary event bus (we won't actually use it)
    event_bus = EventBus()
    
    # Create detector with minimal config
    detector_config = {
        "auto_update": False,
        "min_data_points": 50,
        "emit_events": False
    }
    
    # Import here to avoid cyclic imports
    from trading_bot.analytics.market_regime.detector import MarketRegimeDetector
    
    detector = MarketRegimeDetector(event_bus, detector_config)
    
    # Analyze data
    print(f"Analyzing {len(df)} data points for {symbol} {timeframe}")
    
    # Add symbol to detector
    detector.add_symbol(symbol, [timeframe])
    
    # Mock the broker manager so we can insert our data
    detector.broker_manager = None
    detector._get_ohlcv_data = lambda s, t, count=None: df.copy()
    
    # Detect regimes
    regime_info = detector._detect_regime(symbol, timeframe)
    print(f"Current regime: {regime_info['regime']} (confidence: {regime_info['confidence']:.2f})")
    
    # Calculate regimes for historical data
    results = []
    window_size = 100  # Number of bars to use for detection
    
    for i in range(window_size, len(df), 20):  # Step by 20 for efficiency
        try:
            window = df.iloc[i-window_size:i]
            features = detector.features_calculator.calculate_features(window)
            regime, confidence = detector.classifier.classify_regime(features)
            
            results.append({
                'timestamp': df.index[i-1],
                'regime': regime,
                'confidence': confidence,
                'close': df.iloc[i-1]['close']
            })
        except Exception as e:
            print(f"Error analyzing window ending at index {i}: {str(e)}")
    
    if not results:
        print("No regime detection results were generated")
        return 1
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot results
    if args.plot:
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Plot price
            ax1.plot(results_df['timestamp'], results_df['close'])
            ax1.set_title(f"{symbol} {timeframe} Price")
            ax1.grid(True)
            
            # Plot regimes
            # Convert regime to numeric for plotting
            regime_map = {regime.value: i for i, regime in enumerate(MarketRegimeType)}
            regime_values = [regime_map.get(r.value, 0) for r in results_df['regime']]
            
            ax2.scatter(results_df['timestamp'], regime_values, c=results_df['confidence'], cmap='viridis', alpha=0.7)
            ax2.set_title("Market Regimes (color = confidence)")
            ax2.set_yticks(list(regime_map.values()))
            ax2.set_yticklabels(list(regime_map.keys()))
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting results: {str(e)}")
    
    # Print summary
    regime_counts = results_df['regime'].value_counts()
    print("\nRegime Distribution:")
    for regime, count in regime_counts.items():
        print(f"{regime.value}: {count} ({count/len(results_df)*100:.1f}%)")
    
    # If requested, save results
    if args.output:
        # Convert regime enum to string for JSON serialization
        results_df['regime'] = results_df['regime'].apply(lambda r: r.value)
        
        if args.output.endswith('.csv'):
            results_df.to_csv(args.output)
        elif args.output.endswith('.json'):
            results_df.to_json(args.output, orient='records', date_format='iso')
        else:
            results_df.to_csv(args.output + '.csv')
        
        print(f"Saved results to {args.output}")
    
    return 0

def optimize_command(args):
    """Run parameter optimization for a strategy and regime."""
    strategy_id = args.strategy
    regime = args.regime
    output_file = args.output
    
    # Try to convert regime string to enum
    try:
        regime_type = MarketRegimeType(regime)
    except ValueError:
        print(f"Invalid regime type: {regime}")
        print(f"Valid regimes: {[r.value for r in MarketRegimeType]}")
        return 1
    
    # Check if strategy exists in DEFAULT_PARAMETERS
    if strategy_id not in DEFAULT_PARAMETERS:
        print(f"Strategy not found: {strategy_id}")
        print(f"Available strategies: {list(DEFAULT_PARAMETERS.keys())}")
        return 1
    
    # Get default parameters for this strategy and regime
    default_params = DEFAULT_PARAMETERS[strategy_id][regime_type]
    
    print(f"Default parameters for {strategy_id} in {regime} regime:")
    for param, value in default_params.items():
        print(f"  {param}: {value}")
    
    # Load custom parameters if provided
    custom_params = {}
    if args.params_file:
        try:
            with open(args.params_file, 'r') as f:
                custom_params = json.load(f)
        except Exception as e:
            print(f"Error loading parameters file: {str(e)}")
            return 1
    
    # Merge default with custom
    merged_params = {**default_params, **custom_params}
    
    # Convert MarketRegimeType to string for output
    output_data = {
        regime_type.value: merged_params
    }
    
    # Save to file
    if output_file:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            existing_data = {}
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    existing_data = json.load(f)
            
            # Merge with existing data
            existing_data.update(output_data)
            
            with open(output_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
            
            print(f"Saved optimized parameters to {output_file}")
        except Exception as e:
            print(f"Error saving parameters: {str(e)}")
            return 1
    
    return 0

def dashboard_command(args):
    """Start the regime dashboard."""
    print("Starting market regime dashboard...")
    
    try:
        # Import dashboard components
        from trading_bot.dashboard.app import create_app
        from trading_bot.dashboard.components.regime_dashboard import setup_regime_routes
        import uvicorn
        
        # Create FastAPI app
        app = create_app()
        
        # Register regime routes
        setup_regime_routes(app)
        
        # Start server
        print(f"Dashboard starting on http://localhost:{args.port}")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
        
    except ImportError as e:
        print(f"Error importing dashboard components: {str(e)}")
        print("Make sure FastAPI and uvicorn are installed:")
        print("  pip install fastapi uvicorn")
        return 1
    except Exception as e:
        print(f"Error starting dashboard: {str(e)}")
        return 1
    
    return 0

def main():
    parser = argparse.ArgumentParser(description='Market Regime System CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize the market regime system configuration')
    init_parser.add_argument('--output', '-o', default='config/market_regime_config.json', help='Output file path')
    init_parser.add_argument('--symbols', '-s', help='Comma-separated list of symbols to monitor')
    init_parser.add_argument('--timeframes', '-t', help='Comma-separated list of timeframes to monitor')
    init_parser.add_argument('--generate-params', '-g', action='store_true', help='Generate parameter files')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Get status of the market regime system')
    status_parser.add_argument('--config', '-c', default='config/market_regime_config.json', help='Config file path')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze historical data for regime detection')
    analyze_parser.add_argument('--symbol', '-s', required=True, help='Symbol to analyze')
    analyze_parser.add_argument('--timeframe', '-t', default='1d', help='Timeframe to analyze')
    analyze_parser.add_argument('--data-file', '-d', required=True, help='CSV or JSON file with OHLCV data')
    analyze_parser.add_argument('--plot', '-p', action='store_true', help='Show plot of results')
    analyze_parser.add_argument('--output', '-o', help='Output file for results (CSV or JSON)')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Generate optimized parameters for a strategy and regime')
    optimize_parser.add_argument('--strategy', '-s', required=True, help='Strategy ID')
    optimize_parser.add_argument('--regime', '-r', required=True, help='Regime type')
    optimize_parser.add_argument('--params-file', '-p', help='JSON file with custom parameters')
    optimize_parser.add_argument('--output', '-o', default='data/regime_parameters/strategy_{strategy_id}.json', help='Output file for parameters')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Start the regime dashboard')
    dashboard_parser.add_argument('--port', '-p', type=int, default=8080, help='Port to run the dashboard on')
    
    args = parser.parse_args()
    
    if args.command == 'init':
        return init_command(args)
    elif args.command == 'status':
        return status_command(args)
    elif args.command == 'analyze':
        return analyze_command(args)
    elif args.command == 'optimize':
        return optimize_command(args)
    elif args.command == 'dashboard':
        return dashboard_command(args)
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main())
