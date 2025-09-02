#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command Line Interface for the Trading Bot system.
"""

import argparse
import logging
import sys
from datetime import datetime

from trading_bot.strategy import (
    StrategyRotator,
    MomentumStrategy,
    TrendFollowingStrategy,
    MeanReversionStrategy
)
from trading_bot.common.market_types import MarketRegime

def setup_logging(log_level="INFO"):
    """
    Set up logging configuration.
    
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"trading_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )

def run_example(args):
    """
    Run the example strategy rotator.
    
    Args:
        args: Command line arguments
    """
    # Import here to avoid circular imports
    from trading_bot.strategy.examples.rotator_example import run_example as run_rotator_example
    
    setup_logging(args.log_level)
    logging.info("Running example strategy rotator")
    run_rotator_example()

def run_custom(args):
    """
    Run custom strategy configuration.
    
    Args:
        args: Command line arguments
    """
    setup_logging(args.log_level)
    logging.info("Running custom strategy configuration")
    
    # Create strategies based on arguments
    strategies = []
    
    if args.momentum:
        strategies.append(MomentumStrategy(
            "MomentumStrategy",
            {"fast_period": args.momentum_fast, "slow_period": args.momentum_slow}
        ))
        
    if args.trend_following:
        strategies.append(TrendFollowingStrategy(
            "TrendFollowingStrategy",
            {"short_ma_period": args.trend_short, "long_ma_period": args.trend_long}
        ))
        
    if args.mean_reversion:
        strategies.append(MeanReversionStrategy(
            "MeanReversionStrategy",
            {"period": args.mean_period, "std_dev_factor": args.mean_std_factor}
        ))
    
    if not strategies:
        logging.error("No strategies selected. Use --momentum, --trend-following, or --mean-reversion")
        return
    
    # Create rotator
    rotator = StrategyRotator(
        strategies=strategies,
        data_dir=args.data_dir,
        regime_adaptation=not args.no_regime_adaptation
    )
    
    # Set market regime if provided
    if args.regime:
        try:
            regime = MarketRegime[args.regime.upper()]
            rotator.update_market_regime(regime, confidence=args.regime_confidence)
        except KeyError:
            logging.error(f"Invalid market regime: {args.regime}")
            valid_regimes = [r.name for r in MarketRegime]
            logging.error(f"Valid regimes: {', '.join(valid_regimes)}")
    
    # Generate sample market data
    # In a real application, you would get market data from a data provider
    from trading_bot.strategy.examples.rotator_example import generate_sample_market_data
    
    market_data = generate_sample_market_data(
        days=30,
        initial_price=100.0,
        volatility=args.volatility,
        trend=args.trend
    )
    
    # Generate signals
    signals = rotator.generate_signals(market_data)
    combined_signal = rotator.get_combined_signal()
    
    # Log results
    logging.info("\nStrategy signals:")
    for name, signal in signals.items():
        logging.info(f"  {name}: {signal:.4f}")
    
    logging.info(f"\nCombined signal: {combined_signal:.4f}")
    
    # Log strategy weights
    logging.info("\nStrategy weights:")
    for name, weight in rotator.get_strategy_weights().items():
        logging.info(f"  {name}: {weight:.4f}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Trading Bot CLI")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Example command
    example_parser = subparsers.add_parser("example", help="Run the example strategy rotator")
    example_parser.set_defaults(func=run_example)
    
    # Custom command
    custom_parser = subparsers.add_parser("custom", help="Run custom strategy configuration")
    custom_parser.set_defaults(func=run_custom)
    
    # Strategy options
    custom_parser.add_argument("--momentum", action="store_true", help="Use momentum strategy")
    custom_parser.add_argument("--momentum-fast", type=int, default=5, help="Momentum fast period")
    custom_parser.add_argument("--momentum-slow", type=int, default=20, help="Momentum slow period")
    
    custom_parser.add_argument("--trend-following", action="store_true", help="Use trend following strategy")
    custom_parser.add_argument("--trend-short", type=int, default=10, help="Trend following short MA period")
    custom_parser.add_argument("--trend-long", type=int, default=30, help="Trend following long MA period")
    
    custom_parser.add_argument("--mean-reversion", action="store_true", help="Use mean reversion strategy")
    custom_parser.add_argument("--mean-period", type=int, default=20, help="Mean reversion period")
    custom_parser.add_argument("--mean-std-factor", type=float, default=2.0, help="Mean reversion std dev factor")
    
    # Market options
    custom_parser.add_argument("--volatility", type=float, default=0.01, help="Market volatility")
    custom_parser.add_argument("--trend", type=float, default=0.001, help="Market trend factor")
    
    # Regime options
    custom_parser.add_argument("--regime", help="Market regime (BULL, BEAR, SIDEWAYS, etc.)")
    custom_parser.add_argument("--regime-confidence", type=float, default=0.8, help="Confidence in regime detection")
    custom_parser.add_argument("--no-regime-adaptation", action="store_true", help="Disable regime adaptation")
    
    # Data options
    custom_parser.add_argument("--data-dir", help="Data directory for storing state")
    
    args = parser.parse_args()
    
    if args.command:
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 