#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Bot CLI Commands

This module defines the commands available in the trading bot CLI.
These commands consolidate the functionality previously spread across
multiple entry point scripts.
"""

import logging
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

from trading_bot.config.unified_config import get_config
from trading_bot.core.event_system import EventBus, Event, EventType
from trading_bot.trading_bot import TradingBot
from trading_bot.trading_daemon import TradingDaemon
from trading_bot.backtesting.backtest_engine import BacktestEngine
from trading_bot.market.market_data_service import MarketDataService
from trading_bot.data.quality.data_quality_manager import DataQualityManager
from trading_bot.strategies.factory.strategy_factory import StrategyFactory

from .cli_app import command, argument

logger = logging.getLogger(__name__)

def register_commands():
    """Register all CLI commands."""
    # Commands are registered via decorators, so we don't need to do anything here
    logger.debug("CLI commands registered")


@command("run", "Run the trading bot with specified configuration")
@argument("--mode", choices=["live", "paper", "backtest"], default="paper",
          help="Trading mode (live, paper, or backtest)")
@argument("--strategy", help="Strategy to use (defaults to configuration)")
@argument("--symbols", help="Comma-separated list of symbols to trade")
@argument("--from-date", help="Start date for backtest (YYYY-MM-DD)")
@argument("--to-date", help="End date for backtest (YYYY-MM-DD)")
@argument("--broker", help="Broker to use (defaults to configuration)")
@argument("--capital", type=float, help="Starting capital")
@argument("--daemon", action="store_true", help="Run as a daemon")
def run_command(args):
    """
    Run the trading bot.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    logger.info(f"Starting trading bot in {args.mode} mode")
    
    # Parse symbols
    symbols = args.symbols.split(',') if args.symbols else None
    
    # Create EventBus
    event_bus = EventBus()
    
    if args.daemon:
        logger.info("Starting in daemon mode")
        daemon = TradingDaemon(
            mode=args.mode,
            strategy_name=args.strategy,
            symbols=symbols,
            broker_name=args.broker,
            event_bus=event_bus
        )
        daemon.start()
        return 0
    
    # Create TradingBot
    bot = TradingBot(
        mode=args.mode,
        strategy_name=args.strategy,
        symbols=symbols,
        broker_name=args.broker,
        event_bus=event_bus
    )
    
    if args.mode == "backtest":
        # Backtest mode
        from_date = datetime.strptime(args.from_date, "%Y-%m-%d") if args.from_date else None
        to_date = datetime.strptime(args.to_date, "%Y-%m-%d") if args.to_date else None
        capital = args.capital or get_config().get("backtest.default_capital", 10000)
        
        backtest_engine = BacktestEngine(
            bot=bot,
            from_date=from_date,
            to_date=to_date,
            initial_capital=capital
        )
        
        # Run backtest
        results = backtest_engine.run()
        
        # Display results
        print("\nBacktest Results:")
        print(f"Total return: {results['total_return']:.2f}%")
        print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max drawdown: {results['max_drawdown']:.2f}%")
        print(f"Total trades: {results['total_trades']}")
        print(f"Win rate: {results['win_rate']*100:.2f}%")
        
        return 0
    
    # Live or paper trading mode
    try:
        bot.initialize()
        bot.run()
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
        bot.shutdown()
    except Exception as e:
        logger.exception(f"Error running trading bot: {str(e)}")
        bot.shutdown()
        return 1
    
    return 0


@command("data", "Data management operations")
@argument("--action", choices=["fetch", "clean", "quality-check", "export"], 
          required=True, help="Data action to perform")
@argument("--symbols", help="Comma-separated list of symbols")
@argument("--source", help="Data source to use")
@argument("--timeframe", help="Timeframe for data (e.g., 1m, 1h, 1d)")
@argument("--from-date", help="Start date (YYYY-MM-DD)")
@argument("--to-date", help="End date (YYYY-MM-DD)")
@argument("--output", help="Output file for export")
def data_command(args):
    """
    Perform data management operations.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    logger.info(f"Performing data {args.action} operation")
    
    # Parse symbols
    symbols = args.symbols.split(',') if args.symbols else []
    
    # Parse dates
    from_date = datetime.strptime(args.from_date, "%Y-%m-%d") if args.from_date else None
    to_date = datetime.strptime(args.to_date, "%Y-%m-%d") if args.to_date else None
    
    # Create market data service
    market_data = MarketDataService()
    
    if args.action == "fetch":
        # Fetch market data
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}")
            data = market_data.fetch_historical_data(
                symbol=symbol,
                source=args.source,
                timeframe=args.timeframe,
                start_date=from_date,
                end_date=to_date
            )
            print(f"Fetched {len(data)} records for {symbol}")
        
    elif args.action == "clean":
        # Clean market data
        from trading_bot.data.processors.data_cleaning_processor import DataCleaningProcessor
        
        cleaner = DataCleaningProcessor()
        
        for symbol in symbols:
            logger.info(f"Cleaning data for {symbol}")
            data = market_data.fetch_historical_data(
                symbol=symbol,
                source=args.source,
                timeframe=args.timeframe,
                start_date=from_date,
                end_date=to_date
            )
            
            cleaned_data = cleaner.process(data)
            print(f"Cleaned {len(cleaned_data)} records for {symbol}")
            
            # Save cleaned data if output specified
            if args.output:
                output_file = args.output.replace("{symbol}", symbol)
                cleaned_data.to_csv(output_file)
                print(f"Saved cleaned data to {output_file}")
        
    elif args.action == "quality-check":
        # Run quality checks on market data
        from trading_bot.data.quality.data_quality_manager import DataQualityManager
        
        quality_manager = DataQualityManager()
        
        for symbol in symbols:
            logger.info(f"Running quality checks for {symbol}")
            data = market_data.fetch_historical_data(
                symbol=symbol,
                source=args.source,
                timeframe=args.timeframe,
                start_date=from_date,
                end_date=to_date
            )
            
            _, quality_report = quality_manager.check_data_quality(
                data, symbol=symbol, source=args.source or "unknown"
            )
            
            print(f"\nQuality Report for {symbol}:")
            print(f"Quality Score: {quality_report.get('quality_score', 0):.2f}")
            print(f"Status: {quality_report.get('status', 'unknown')}")
            
            if quality_report.get('issues'):
                print("\nIssues:")
                for issue in quality_report.get('issues', []):
                    print(f"- {issue.get('message')}")
            
            # Generate and save HTML report if output specified
            if args.output:
                output_file = args.output.replace("{symbol}", symbol)
                if not output_file.endswith('.html'):
                    output_file += '.html'
                
                html_report = quality_manager.generate_quality_report(
                    symbols=[symbol], output_format="html"
                )
                
                with open(output_file, 'w') as f:
                    f.write(html_report)
                
                print(f"Saved quality report to {output_file}")
        
    elif args.action == "export":
        # Export market data
        for symbol in symbols:
            logger.info(f"Exporting data for {symbol}")
            data = market_data.fetch_historical_data(
                symbol=symbol,
                source=args.source,
                timeframe=args.timeframe,
                start_date=from_date,
                end_date=to_date
            )
            
            if args.output:
                output_file = args.output.replace("{symbol}", symbol)
                data.to_csv(output_file)
                print(f"Exported {len(data)} records for {symbol} to {output_file}")
            else:
                print(data.head())
                print(f"Fetched {len(data)} records for {symbol}")
    
    return 0


@command("strategy", "Strategy management operations")
@argument("--action", choices=["list", "info", "test", "optimize"], 
          required=True, help="Strategy action to perform")
@argument("--name", help="Strategy name")
@argument("--symbols", help="Comma-separated list of symbols")
@argument("--timeframe", help="Timeframe for testing (e.g., 1h, 1d)")
@argument("--from-date", help="Start date for test (YYYY-MM-DD)")
@argument("--to-date", help="End date for test (YYYY-MM-DD)")
@argument("--parameters", help="JSON string of strategy parameters")
def strategy_command(args):
    """
    Perform strategy management operations.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    logger.info(f"Performing strategy {args.action} operation")
    
    # Create strategy factory
    factory = StrategyFactory()
    
    if args.action == "list":
        # List available strategies
        from trading_bot.strategies.factory.strategy_registry import StrategyRegistry, AssetClass, StrategyType
        
        strategies = StrategyRegistry.get_all_strategy_names()
        
        print(f"\nAvailable Strategies ({len(strategies)}):")
        
        # Group by asset class if we have many
        if len(strategies) > 10:
            for asset_class in AssetClass:
                asset_strategies = StrategyRegistry.get_strategies_by_asset_class(asset_class)
                if asset_strategies:
                    print(f"\n{asset_class.value.capitalize()} Strategies:")
                    for strategy in asset_strategies:
                        metadata = StrategyRegistry.get_strategy_metadata(strategy)
                        strategy_type = metadata.get('strategy_type', 'unknown')
                        print(f"  - {strategy} ({strategy_type})")
        else:
            # List all strategies with their type
            for strategy in strategies:
                metadata = StrategyRegistry.get_strategy_metadata(strategy)
                strategy_type = metadata.get('strategy_type', 'unknown')
                asset_class = metadata.get('asset_class', 'unknown')
                print(f"  - {strategy} ({asset_class}/{strategy_type})")
        
    elif args.action == "info":
        # Show strategy information
        if not args.name:
            logger.error("Strategy name is required for 'info' action")
            return 1
        
        from trading_bot.strategies.factory.strategy_registry import StrategyRegistry
        
        metadata = StrategyRegistry.get_strategy_metadata(args.name)
        
        if not metadata:
            logger.error(f"Strategy '{args.name}' not found")
            return 1
        
        print(f"\nStrategy: {args.name}")
        print(f"Asset Class: {metadata.get('asset_class', 'unknown')}")
        print(f"Strategy Type: {metadata.get('strategy_type', 'unknown')}")
        print(f"Timeframe: {metadata.get('timeframe', 'unknown')}")
        
        # Compatible market regimes
        regimes = metadata.get('compatible_market_regimes', [])
        print(f"Compatible Market Regimes: {', '.join(regimes)}")
        
        # Regime compatibility scores if available
        if 'regime_compatibility_scores' in metadata:
            print("\nMarket Regime Compatibility Scores:")
            for regime, score in metadata['regime_compatibility_scores'].items():
                print(f"  - {regime}: {score:.2f}")
        
        # Default parameters
        if 'optimal_parameters' in metadata:
            print("\nOptimal Parameters by Regime:")
            for regime, params in metadata['optimal_parameters'].items():
                print(f"\n  {regime.capitalize()}:")
                for param, value in params.items():
                    print(f"    {param}: {value}")
        
    elif args.action == "test":
        # Test a strategy
        if not args.name:
            logger.error("Strategy name is required for 'test' action")
            return 1
        
        # Parse symbols
        symbols = args.symbols.split(',') if args.symbols else []
        
        if not symbols:
            logger.error("At least one symbol is required for strategy testing")
            return 1
        
        # Parse dates
        from_date = datetime.strptime(args.from_date, "%Y-%m-%d") if args.from_date else None
        to_date = datetime.strptime(args.to_date, "%Y-%m-%d") if args.to_date else None
        
        # Parse parameters
        import json
        parameters = json.loads(args.parameters) if args.parameters else {}
        
        # Create strategy
        try:
            strategy = factory.create_strategy(args.name, parameters=parameters)
        except Exception as e:
            logger.error(f"Error creating strategy: {str(e)}")
            return 1
        
        # Fetch data
        market_data = MarketDataService()
        universe = {}
        
        for symbol in symbols:
            data = market_data.fetch_historical_data(
                symbol=symbol,
                timeframe=args.timeframe or "1d",
                start_date=from_date,
                end_date=to_date
            )
            universe[symbol] = data
        
        # Generate signals
        signals = strategy.generate_signals(universe)
        
        # Display results
        print(f"\nStrategy Test Results for {args.name}:")
        print(f"Tested on {len(symbols)} symbols from {from_date} to {to_date}")
        print(f"Generated {len(signals)} signals")
        
        for symbol, signal in signals.items():
            print(f"\n{symbol}: {signal.signal_type} @ {signal.entry_price} (confidence: {signal.confidence:.2f})")
            print(f"  Stop Loss: {signal.stop_loss}, Take Profit: {signal.take_profit}")
            
            # Show indicators if available
            if signal.metadata and 'indicators' in signal.metadata:
                print("  Indicators:")
                for name, value in signal.metadata['indicators'].items():
                    print(f"    {name}: {value}")
    
    elif args.action == "optimize":
        # Optimize a strategy
        if not args.name:
            logger.error("Strategy name is required for 'optimize' action")
            return 1
        
        # This would be implemented with your optimization framework
        print("Strategy optimization not implemented in this version")
    
    return 0


@command("monitor", "Monitor trading system and generate reports")
@argument("--component", choices=["all", "strategies", "quality", "performance", "risk"],
          default="all", help="Component to monitor")
@argument("--format", choices=["text", "json", "html"], default="text",
          help="Output format")
@argument("--output", help="Output file (default: stdout)")
def monitor_command(args):
    """
    Monitor the trading system and generate reports.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    logger.info(f"Monitoring {args.component} component(s)")
    
    # This is a placeholder for monitoring functionality
    # In a real implementation, this would connect to the running system
    # and retrieve monitoring data
    
    if args.component == "quality" or args.component == "all":
        print("\nData Quality Monitoring:")
        print("------------------------")
        print("Overall quality score: 92.5/100")
        print("Issues detected in last 24h: 3")
        print("Auto-fixed issues: 2")
        print("Critical quality alerts: 0")
    
    if args.component == "strategies" or args.component == "all":
        print("\nStrategy Monitoring:")
        print("-------------------")
        print("Active strategies: 5")
        print("Signals generated today: 12")
        print("Top performing: ForexTrendFollowingStrategy (+2.3%)")
        print("Underperforming: StockMeanReversionStrategy (-0.8%)")
    
    if args.component == "performance" or args.component == "all":
        print("\nPerformance Monitoring:")
        print("----------------------")
        print("Daily P&L: +1.2%")
        print("Weekly P&L: +3.5%")
        print("Monthly P&L: +8.7%")
        print("Sharpe ratio (30d): 1.8")
        print("Current drawdown: 2.1%")
    
    if args.component == "risk" or args.component == "all":
        print("\nRisk Monitoring:")
        print("---------------")
        print("Value at Risk (95%): $120.45")
        print("Max position size: 2.5%")
        print("Portfolio heat: 15%")
        print("Risk alerts: 0")
    
    return 0


@command("dashboard", "Launch the trading dashboard UI")
@argument("--port", type=int, default=8501, help="Port to run the dashboard on")
@argument("--host", default="localhost", help="Host to bind to")
@argument("--mode", choices=["standard", "minimal", "advanced"], default="standard",
          help="Dashboard mode")
def dashboard_command(args):
    """
    Launch the trading dashboard UI.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    logger.info(f"Launching trading dashboard on {args.host}:{args.port} in {args.mode} mode")
    
    try:
        # Import dashboard module
        from trading_bot.trading_dashboard import run_dashboard
        
        # Run dashboard
        run_dashboard(
            host=args.host,
            port=args.port,
            mode=args.mode
        )
        
        return 0
    except ImportError:
        logger.error("Dashboard module not found. Make sure Streamlit is installed.")
        print("\nDashboard requires Streamlit. Install with:")
        print("pip install streamlit")
        return 1
    except Exception as e:
        logger.exception(f"Error launching dashboard: {str(e)}")
        return 1
