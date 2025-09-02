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
from typing import Dict, List, Any, Optional

from trading_bot.research.evotrader.bridge import EvoTraderBridge
from trading_bot.research.evotrader.strategy_templates import create_strategy_template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evotrader_cli.log')
    ]
)

logger = logging.getLogger("benbot.research.evotrader.cli")

def setup_parser():
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="BensBot EvoTrader R&D CLI - Evolve and optimize trading strategies"
    )
    
    # Main command options
    parser.add_argument(
        "--command", "-c",
        choices=["evolve", "evaluate", "deploy", "status", "list"],
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
    
    # EvoTrader path (optional)
    parser.add_argument(
        "--evotrader-path",
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
    
    # Strategy template
    parser.add_argument(
        "--template",
        default="default",
        help="Strategy template to use as starting point"
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
    
    # Output directory
    parser.add_argument(
        "--output-dir",
        default="evotrader_results",
        help="Directory for results"
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
    
    # Create bridge
    bridge = EvoTraderBridge(
        config=config,
        evotrader_path=args.evotrader_path
    )
    
    # Create template strategy
    template = create_strategy_template(
        asset_class=args.asset,
        strategy_type=args.template
    )
    
    # Get symbols and timeframes
    symbols = args.symbols
    timeframes = args.timeframes
    
    # Create config for evolution
    evolution_config = {
        "output_dir": args.output_dir,
        "evolution": {
            "population_size": args.population,
            "generations": args.generations
        }
    }
    
    # Add symbols and timeframes if provided
    if symbols:
        evolution_config[args.asset] = {"symbols": symbols}
    if timeframes:
        if args.asset not in evolution_config:
            evolution_config[args.asset] = {}
        evolution_config[args.asset]["timeframes"] = timeframes
    
    # Update bridge config
    for key, value in evolution_config.items():
        if isinstance(value, dict):
            if key not in bridge.config:
                bridge.config[key] = {}
            bridge.config[key].update(value)
        else:
            bridge.config[key] = value
    
    # Create initial population
    logger.info("Creating initial population...")
    population = bridge.evolution_manager.create_initial_population(template)
    
    # Run evolution
    logger.info(f"Running evolution with {len(population)} strategies...")
    start_time = datetime.datetime.now()
    
    results = bridge.evolve_strategies(
        strategies=population,
        generations=args.generations,
        asset_class=args.asset
    )
    
    end_time = datetime.datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    # Get best strategies
    best_strategies = bridge.get_best_strategies(results, count=3)
    
    logger.info(f"Evolution completed in {elapsed:.2f} seconds")
    
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
    
    print(f"\nResults saved to {args.output_dir}")
    
    return results

def command_evaluate(args, config):
    """Evaluate a specific strategy."""
    strategy_id = args.strategy_id
    if not strategy_id:
        logger.error("Strategy ID is required for evaluation")
        return
    
    logger.info(f"Evaluating strategy {strategy_id}")
    
    # Create bridge
    bridge = EvoTraderBridge(
        config=config,
        evotrader_path=args.evotrader_path
    )
    
    # Find the strategy
    strategy = None
    results_dir = args.output_dir
    
    # Search in all evolution result files
    for filename in os.listdir(results_dir):
        if filename.startswith("evolution_") and filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            
            with open(filepath, 'r') as f:
                evolution_results = json.load(f)
            
            # Check each generation
            for gen in evolution_results.get("generations", []):
                for strat in gen.get("strategies", []):
                    if strat.get("strategy_id") == strategy_id:
                        # Found the strategy
                        from trading_bot.research.evotrader.strategy_adapter import BensBotStrategyAdapter
                        
                        # Create strategy template
                        template = create_strategy_template(
                            asset_class=args.asset,
                            strategy_type=args.template
                        )
                        
                        # Create adapter with parameters
                        strategy = BensBotStrategyAdapter(
                            benbot_strategy=template,
                            strategy_id=strategy_id,
                            parameters=strat.get("parameters", {})
                        )
                        break
                
                if strategy:
                    break
            
            if strategy:
                break
    
    if not strategy:
        logger.error(f"Strategy {strategy_id} not found in evolution results")
        return
    
    # Evaluate the strategy
    logger.info(f"Running detailed evaluation for {strategy_id}")
    
    evaluation_results = bridge.evaluate_strategy(
        strategy=strategy,
        asset_class=args.asset,
        detailed=True
    )
    
    # Print results
    print("\n==== Strategy Evaluation ====")
    print(f"Strategy ID: {strategy_id}")
    print(f"Asset Class: {args.asset}")
    
    print("\nPerformance Metrics:")
    print(f"Sharpe Ratio: {evaluation_results.get('sharpe_ratio', 0):.4f}")
    print(f"Total Return: {evaluation_results.get('total_return_pct', 0):.2f}%")
    print(f"Annualized Return: {evaluation_results.get('annualized_return_pct', 0):.2f}%")
    print(f"Max Drawdown: {evaluation_results.get('max_drawdown_pct', 0):.2f}%")
    print(f"Win Rate: {evaluation_results.get('win_rate_pct', 0):.2f}%")
    print(f"Profit Factor: {evaluation_results.get('profit_factor', 0):.2f}")
    print(f"Total Trades: {evaluation_results.get('total_trades', 0)}")
    
    # Check if strategy meets promotion criteria
    meets_criteria = bridge.should_promote_strategy(evaluation_results)
    
    print(f"\nPromotion Criteria: {'MEETS' if meets_criteria else 'DOES NOT MEET'}")
    
    return evaluation_results

def command_deploy(args, config):
    """Deploy a strategy to BensBot trading environment."""
    strategy_id = args.strategy_id
    if not strategy_id:
        logger.error("Strategy ID is required for deployment")
        return
    
    logger.info(f"Preparing to deploy strategy {strategy_id}")
    
    # Create bridge
    bridge = EvoTraderBridge(
        config=config,
        evotrader_path=args.evotrader_path
    )
    
    # Find the strategy (same as in evaluate command)
    strategy = None
    results_dir = args.output_dir
    
    # Search in all evolution result files
    for filename in os.listdir(results_dir):
        if filename.startswith("evolution_") and filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            
            with open(filepath, 'r') as f:
                evolution_results = json.load(f)
            
            # Check each generation
            for gen in evolution_results.get("generations", []):
                for strat in gen.get("strategies", []):
                    if strat.get("strategy_id") == strategy_id:
                        # Found the strategy
                        from trading_bot.research.evotrader.strategy_adapter import BensBotStrategyAdapter
                        
                        # Create strategy template
                        template = create_strategy_template(
                            asset_class=args.asset,
                            strategy_type=args.template
                        )
                        
                        # Create adapter with parameters
                        strategy = BensBotStrategyAdapter(
                            benbot_strategy=template,
                            strategy_id=strategy_id,
                            parameters=strat.get("parameters", {})
                        )
                        break
                
                if strategy:
                    break
            
            if strategy:
                break
    
    if not strategy:
        logger.error(f"Strategy {strategy_id} not found in evolution results")
        return
    
    # Evaluate the strategy to make sure it meets criteria
    logger.info(f"Evaluating {strategy_id} before deployment")
    
    evaluation_results = bridge.evaluate_strategy(
        strategy=strategy,
        asset_class=args.asset
    )
    
    # Check if strategy meets promotion criteria
    meets_criteria = bridge.should_promote_strategy(evaluation_results)
    
    if not meets_criteria:
        print("\n==== Deployment Blocked ====")
        print(f"Strategy {strategy_id} does not meet promotion criteria.")
        print("Please use a different strategy or adjust promotion criteria.")
        
        print("\nPerformance Metrics:")
        print(f"Sharpe Ratio: {evaluation_results.get('sharpe_ratio', 0):.4f}")
        print(f"Total Return: {evaluation_results.get('total_return_pct', 0):.2f}%")
        print(f"Win Rate: {evaluation_results.get('win_rate_pct', 0):.2f}%")
        print(f"Max Drawdown: {evaluation_results.get('max_drawdown_pct', 0):.2f}%")
        
        return False
    
    # Deploy to paper trading environment
    logger.info(f"Deploying {strategy_id} to paper trading")
    
    success = bridge.deploy_strategy_to_benbot(
        strategy=strategy,
        target_env="paper"
    )
    
    if success:
        print("\n==== Deployment Successful ====")
        print(f"Strategy {strategy_id} has been deployed to paper trading.")
        print("You can monitor its performance in the BensBot dashboard.")
        
        print("\nStrategy Parameters:")
        for param, value in strategy.parameters.items():
            print(f"{param}: {value}")
    else:
        print("\n==== Deployment Failed ====")
        print(f"Failed to deploy strategy {strategy_id}.")
        print("Check the logs for more details.")
    
    return success

def command_list(args, config):
    """List available strategies and results."""
    results_dir = args.output_dir
    
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} not found.")
        return
    
    # Get all evolution result files
    evolution_files = [f for f in os.listdir(results_dir) 
                      if f.startswith("evolution_") and f.endswith(".json")]
    
    if not evolution_files:
        print("No evolution results found.")
        return
    
    # Sort by modification time (newest first)
    evolution_files.sort(
        key=lambda f: os.path.getmtime(os.path.join(results_dir, f)),
        reverse=True
    )
    
    print("\n==== Available Evolution Results ====")
    
    # Process each file
    for filename in evolution_files:
        filepath = os.path.join(results_dir, filename)
        
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        # Extract info
        asset_class = results.get("asset_class", "unknown")
        symbols = results.get("symbols", [])
        timeframes = results.get("timeframes", [])
        start_time = results.get("start_time", "unknown")
        generations = results.get("generations", [])
        
        # Format date
        try:
            start_date = datetime.datetime.fromisoformat(start_time).strftime("%Y-%m-%d %H:%M")
        except:
            start_date = start_time
        
        print(f"\nFile: {filename}")
        print(f"Date: {start_date}")
        print(f"Asset Class: {asset_class}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Timeframes: {', '.join(timeframes)}")
        print(f"Generations: {len(generations)}")
        
        # Show top strategies from last generation
        if generations:
            last_gen = generations[-1]
            strategies = last_gen.get("strategies", [])
            
            # Sort by fitness
            strategies.sort(key=lambda s: s.get("fitness", 0), reverse=True)
            
            print("\nTop Strategies:")
            for i, strategy in enumerate(strategies[:3], 1):
                strategy_id = strategy.get("strategy_id", f"Strategy {i}")
                fitness = strategy.get("fitness", 0)
                metrics = strategy.get("metrics", {})
                
                print(f"  {i}. {strategy_id}")
                print(f"     Fitness: {fitness:.4f}")
                print(f"     Return: {metrics.get('total_return_pct', 0):.2f}%")
                print(f"     Win Rate: {metrics.get('win_rate_pct', 0):.2f}%")
    
    # Also show deployed strategies
    deployed_dir = os.path.join(results_dir, "deployed_strategies", "paper")
    if os.path.exists(deployed_dir):
        deployed_files = [f for f in os.listdir(deployed_dir) if f.endswith(".json")]
        
        if deployed_files:
            print("\n==== Deployed Strategies ====")
            
            for filename in deployed_files:
                filepath = os.path.join(deployed_dir, filename)
                
                with open(filepath, 'r') as f:
                    strategy = json.load(f)
                
                strategy_id = strategy.get("strategy_id", "unknown")
                deployment_time = strategy.get("deployment_time", "unknown")
                
                # Format date
                try:
                    deploy_date = datetime.datetime.fromisoformat(deployment_time).strftime("%Y-%m-%d %H:%M")
                except:
                    deploy_date = deployment_time
                
                print(f"\nStrategy: {strategy_id}")
                print(f"Deployed: {deploy_date}")
                print(f"Environment: {strategy.get('target_env', 'paper')}")
                
                # Show key parameters
                params = strategy.get("parameters", {})
                if params:
                    print("Key Parameters:")
                    for param, value in list(params.items())[:5]:  # Show first 5 params only
                        print(f"  {param}: {value}")
                    
                    if len(params) > 5:
                        print(f"  ... and {len(params) - 5} more parameters")

def main():
    """Main entry point for CLI."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger("benbot.research.evotrader").setLevel(logging.DEBUG)
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Execute command
    if args.command == "evolve":
        command_evolve(args, config)
    elif args.command == "evaluate":
        command_evaluate(args, config)
    elif args.command == "deploy":
        command_deploy(args, config)
    elif args.command == "list":
        command_list(args, config)
    elif args.command == "status":
        # Status command to be implemented
        print("Status command not yet implemented")
    else:
        print(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()
