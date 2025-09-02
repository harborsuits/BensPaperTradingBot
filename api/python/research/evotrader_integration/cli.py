"""
Command Line Interface for BensBot-EvoTrader Integration

This module provides a command-line interface for using the EvoTrader
evolutionary trading strategy research capabilities within BensBot.
"""

import os
import sys
import json
import logging
import argparse
import datetime
from typing import Dict, List, Any

from trading_bot.research.evotrader_integration.bridge import EvoTraderIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evotrader_cli.log')
    ]
)

logger = logging.getLogger("benbot.research.evotrader_integration.cli")

def setup_parser():
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="BensBot EvoTrader R&D CLI - Evolve and optimize trading strategies"
    )
    
    # Main command options
    parser.add_argument(
        "--command", "-c",
        choices=["evolve", "evaluate", "deploy", "list"],
        required=True,
        help="Command to execute"
    )
    
    # Asset class
    parser.add_argument(
        "--asset", "-a",
        choices=["forex", "crypto"],
        default="forex",
        help="Asset class to focus on"
    )
    
    # EvoTrader repository path
    parser.add_argument(
        "--repo-path",
        help="Path to EvoTrader repository"
    )
    
    # Evolution parameters
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Number of generations for evolution"
    )
    
    parser.add_argument(
        "--population",
        type=int,
        default=50,
        help="Population size for evolution"
    )
    
    # Symbols to test/trade
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to test/trade (e.g., EURUSD GBPUSD for forex, BTC/USD ETH/USD for crypto)"
    )
    
    # Timeframes
    parser.add_argument(
        "--timeframes",
        nargs="+",
        help="Timeframes to use (e.g., 1h 4h 1d)"
    )
    
    # Strategy ID for operations that need it
    parser.add_argument(
        "--strategy-id",
        help="Strategy ID for evaluate or deploy commands"
    )
    
    # Config file
    parser.add_argument(
        "--config",
        help="Path to configuration JSON file"
    )
    
    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser

def load_config(config_path):
    """Load configuration from file."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def command_evolve(args, config):
    """Run strategy evolution."""
    logger.info(f"Starting strategy evolution for {args.asset}")
    
    # Create integration
    integration = EvoTraderIntegration(args.repo_path)
    if not integration.available:
        logger.error("EvoTrader integration is not available. Please check the repository path.")
        return
    
    # Set up evolution parameters
    evolution_config = {
        "evolution": {
            "population_size": args.population,
            "generations": args.generations
        }
    }
    
    # Update with loaded config
    if config:
        evolution_config.update(config)
    
    # Run evolution
    logger.info(f"Running evolution with {args.population} strategies for {args.generations} generations...")
    start_time = datetime.datetime.now()
    
    results = integration.run_evolution(
        asset_class=args.asset,
        generations=args.generations,
        population_size=args.population,
        symbols=args.symbols,
        timeframes=args.timeframes
    )
    
    if not results:
        logger.error("Evolution failed or returned no results.")
        return
    
    elapsed = (datetime.datetime.now() - start_time).total_seconds()
    logger.info(f"Evolution completed in {elapsed:.2f} seconds")
    
    # Get best strategies
    best_strategies = integration.get_best_strategies(results, count=3)
    
    # Print results
    print("\n==== Evolution Results ====")
    print(f"Asset Class: {args.asset}")
    print(f"Generations: {args.generations}")
    print(f"Population Size: {args.population}")
    
    print("\nTop Strategies:")
    for i, strategy in enumerate(best_strategies, 1):
        strategy_id = strategy.get("strategy_id", f"Strategy {i}")
        fitness = strategy.get("fitness", 0)
        metrics = strategy.get("metrics", {})
        
        print(f"\n{i}. {strategy_id}")
        print(f"   Fitness: {fitness:.4f}")
        print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        print(f"   Return: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"   Win Rate: {metrics.get('win_rate_pct', 0):.2f}%")
        print(f"   Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    
    return results

def command_deploy(args, config):
    """Deploy a strategy to BensBot trading environment."""
    strategy_id = args.strategy_id
    if not strategy_id:
        logger.error("Strategy ID is required for deployment")
        return
    
    logger.info(f"Preparing to deploy strategy {strategy_id}")
    
    # Create integration
    integration = EvoTraderIntegration(args.repo_path)
    if not integration.available:
        logger.error("EvoTrader integration is not available. Please check the repository path.")
        return
    
    # Create bridge to access EvoTrader functionality
    bridge = integration.create_bridge(config)
    if not bridge:
        logger.error("Failed to create EvoTrader bridge.")
        return
    
    # Find the strategy by ID
    try:
        # This is a simplified placeholder - in reality we'd need to look up
        # the strategy from EvoTrader's storage
        strategy = {"strategy_id": strategy_id}  # Placeholder
        
        # Deploy the strategy to BensBot
        success = integration.deploy_strategy(strategy, target_env="paper")
        
        if success:
            print(f"Successfully deployed strategy {strategy_id} to paper trading.")
        else:
            print(f"Failed to deploy strategy {strategy_id}.")
            
        return success
    except Exception as e:
        logger.error(f"Error deploying strategy: {e}")
        return False

def command_list(args, config):
    """List available strategies and evolution results."""
    print("Listing available strategies...")
    
    # Create integration
    integration = EvoTraderIntegration(args.repo_path)
    if not integration.available:
        logger.error("EvoTrader integration is not available. Please check the repository path.")
        return
    
    # TODO: Implement listing of available strategies from EvoTrader storage
    print("This feature is coming soon!")

def main():
    """Main entry point for CLI."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        # Set all loggers to DEBUG level
        logging.getLogger("benbot.research.evotrader_integration").setLevel(logging.DEBUG)
    
    # Load config
    config = load_config(args.config)
    
    # Execute command
    if args.command == "evolve":
        command_evolve(args, config)
    elif args.command == "deploy":
        command_deploy(args, config)
    elif args.command == "list":
        command_list(args, config)
    elif args.command == "evaluate":
        # TODO: Implement evaluate command
        print("Evaluate command not yet implemented.")
    else:
        print(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()
