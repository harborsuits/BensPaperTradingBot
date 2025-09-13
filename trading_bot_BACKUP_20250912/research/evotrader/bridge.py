"""
EvoTrader Bridge module for BensBot

This module provides the core interface for connecting BensBot with EvoTrader's
evolutionary trading strategy research capabilities. It serves as the main
point of integration between the two systems.
"""

import os
import sys
import json
import logging
import datetime
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Configure logging
logger = logging.getLogger("benbot.research.evotrader")

class EvoTraderBridge:
    """
    Main bridge component connecting BensBot with EvoTrader's evolutionary capabilities.
    
    This class provides methods for:
    - Adapting BensBot strategies for use with EvoTrader
    - Running evolutionary optimization on strategies
    - Testing evolved strategies through simulation
    - Deploying evolved strategies back to BensBot
    - Tracking performance across generations
    """
    
    def __init__(self, config: Dict[str, Any] = None, evotrader_path: str = None):
        """
        Initialize the EvoTrader bridge.
        
        Args:
            config: Configuration dictionary for the bridge
            evotrader_path: Path to the EvoTrader repository (if not on PYTHONPATH)
        """
        self.config = config or self._get_default_config()
        
        # Add EvoTrader to Python path if provided
        if evotrader_path:
            if os.path.exists(evotrader_path):
                if evotrader_path not in sys.path:
                    sys.path.append(evotrader_path)
                    logger.info(f"Added EvoTrader path to PYTHONPATH: {evotrader_path}")
            else:
                logger.warning(f"EvoTrader path not found: {evotrader_path}")
        
        # Initialize EvoTrader components
        self._init_evotrader_components()
        
        # Create output directory
        self.output_dir = self.config.get("output_dir", "evotrader_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("EvoTrader bridge initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration settings."""
        return {
            "output_dir": "evotrader_results",
            "evolution": {
                "population_size": 50,
                "generations": 10,
                "selection_rate": 0.3,
                "mutation_rate": 0.2,
                "crossover_rate": 0.3,
                "elitism": 2
            },
            "forex": {
                "symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
                "timeframes": ["1h", "4h", "1d"],
                "indicator_period_ranges": {
                    "fast": [5, 20],
                    "medium": [20, 50],
                    "slow": [50, 200]
                }
            },
            "crypto": {
                "symbols": ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"],
                "timeframes": ["15m", "1h", "4h", "1d"],
                "indicator_period_ranges": {
                    "fast": [5, 20],
                    "medium": [20, 50],
                    "slow": [50, 200]
                }
            },
            "promotion_criteria": {
                "min_sharpe_ratio": 1.5,
                "min_profitable_trades_pct": 55,
                "max_drawdown_pct": 15,
                "min_profit_factor": 1.3
            }
        }
    
    def _init_evotrader_components(self):
        """Initialize EvoTrader components for evolution and testing."""
        try:
            # Import EvoTrader components
            # These imports are inside the method to avoid import errors
            # if EvoTrader is not available at module import time
            from evotrader.core.strategy import Strategy as EvoStrategy
            from evotrader.core.trading_bot import TradingBot
            from evotrader.core.challenge_manager import ChallengeManager
            
            self.evo_strategy_class = EvoStrategy
            self.trading_bot_class = TradingBot
            self.challenge_manager = ChallengeManager()
            
            # Import successful
            self.evotrader_available = True
            logger.info("EvoTrader components initialized successfully")
            
        except ImportError as e:
            # EvoTrader not available
            self.evotrader_available = False
            logger.warning(f"Could not import EvoTrader components: {e}")
            logger.warning("Please ensure EvoTrader is installed or provide path via evotrader_path")
    
    def adapt_strategy(self, benbot_strategy, strategy_id=None, parameters=None):
        """
        Adapt a BensBot strategy for use with EvoTrader.
        
        Args:
            benbot_strategy: The BensBot strategy to adapt
            strategy_id: Optional ID for the strategy (generated if None)
            parameters: Optional parameters dictionary to override defaults
            
        Returns:
            Adapted strategy ready for evolution
        """
        from trading_bot.research.evotrader.strategy_adapter import BensBotStrategyAdapter
        
        # Create adapter
        adapted_strategy = BensBotStrategyAdapter(
            benbot_strategy=benbot_strategy,
            strategy_id=strategy_id,
            parameters=parameters
        )
        
        logger.info(f"Adapted BensBot strategy: {adapted_strategy.strategy_id}")
        return adapted_strategy
    
    def evolve_strategies(self, strategies=None, generations=None, asset_class="forex"):
        """
        Evolve a population of strategies through genetic algorithms.
        
        Args:
            strategies: Initial strategies to evolve (creates defaults if None)
            generations: Number of generations to evolve (default from config)
            asset_class: Asset class to focus on ("forex" or "crypto")
            
        Returns:
            Dictionary with evolution results
        """
        if not self.evotrader_available:
            logger.error("EvoTrader components not available")
            return None
        
        # Import evolution manager
        from trading_bot.research.evotrader.evolution_manager import EvolutionManager
        
        # Get configuration for the specified asset class
        asset_config = self.config.get(asset_class, {})
        
        # Create evolution manager
        evolution_manager = EvolutionManager(
            asset_class=asset_class,
            symbols=asset_config.get("symbols", []),
            timeframes=asset_config.get("timeframes", []),
            config=self.config.get("evolution", {})
        )
        
        # Set number of generations
        if generations is not None:
            evolution_manager.config["generations"] = generations
        
        # If no strategies provided, create initial population
        if not strategies:
            strategies = evolution_manager.create_initial_population()
        
        # Run evolution
        start_time = time.time()
        
        logger.info(f"Starting evolution with {len(strategies)} strategies")
        evolution_results = evolution_manager.run_evolution(strategies)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Evolution completed in {elapsed_time:.2f} seconds")
        
        # Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f"evolution_{asset_class}_{timestamp}.json")
        
        with open(results_file, "w") as f:
            json.dump(evolution_results, f, indent=2)
        
        logger.info(f"Evolution results saved to {results_file}")
        
        return evolution_results
    
    def evaluate_strategy(self, strategy, asset_class="forex", detailed=False):
        """
        Evaluate a strategy through backtesting.
        
        Args:
            strategy: Strategy to evaluate
            asset_class: Asset class to test on ("forex" or "crypto")
            detailed: Whether to return detailed metrics or summary
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.evotrader_available:
            logger.error("EvoTrader components not available")
            return None
        
        # Import evaluator
        from trading_bot.research.evotrader.strategy_evaluator import StrategyEvaluator
        
        # Get configuration for the specified asset class
        asset_config = self.config.get(asset_class, {})
        
        # Create evaluator
        evaluator = StrategyEvaluator(
            symbols=asset_config.get("symbols", []),
            timeframes=asset_config.get("timeframes", []),
            output_dir=os.path.join(self.output_dir, "evaluations")
        )
        
        # Run evaluation
        logger.info(f"Evaluating strategy: {strategy.strategy_id}")
        evaluation_results = evaluator.evaluate(strategy, detailed=detailed)
        
        # If detailed results requested, return everything
        if detailed:
            return evaluation_results
        
        # Otherwise, return a summary
        summary = {
            "strategy_id": strategy.strategy_id,
            "sharpe_ratio": evaluation_results.get("sharpe_ratio", 0),
            "total_return_pct": evaluation_results.get("total_return_pct", 0),
            "max_drawdown_pct": evaluation_results.get("max_drawdown_pct", 0),
            "win_rate_pct": evaluation_results.get("win_rate_pct", 0),
            "profit_factor": evaluation_results.get("profit_factor", 0),
            "evaluation_time": datetime.datetime.now().isoformat()
        }
        
        return summary
    
    def get_best_strategies(self, results=None, count=3):
        """
        Get the best strategies from evolution results.
        
        Args:
            results: Evolution results (loads most recent if None)
            count: Number of strategies to return
            
        Returns:
            List of best strategies
        """
        if results is None:
            # Find most recent results file
            results_files = [f for f in os.listdir(self.output_dir) if f.startswith("evolution_")]
            if not results_files:
                logger.error("No evolution results found")
                return []
            
            # Sort by modification time (most recent first)
            results_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.output_dir, f)), reverse=True)
            
            # Load most recent results
            with open(os.path.join(self.output_dir, results_files[0]), "r") as f:
                results = json.load(f)
        
        # Get last generation
        last_gen = results.get("generations", [])[-1]
        
        # Get strategies sorted by fitness (descending)
        strategies = sorted(
            last_gen.get("strategies", []),
            key=lambda s: s.get("fitness", 0),
            reverse=True
        )
        
        # Return top strategies
        return strategies[:count]
    
    def should_promote_strategy(self, strategy_metrics):
        """
        Check if a strategy meets promotion criteria.
        
        Args:
            strategy_metrics: Dictionary with strategy evaluation metrics
            
        Returns:
            Boolean indicating whether strategy should be promoted
        """
        criteria = self.config.get("promotion_criteria", {})
        
        # Check criteria
        sharpe_ratio = strategy_metrics.get("sharpe_ratio", 0)
        win_rate = strategy_metrics.get("win_rate_pct", 0)
        max_drawdown = strategy_metrics.get("max_drawdown_pct", 100)
        profit_factor = strategy_metrics.get("profit_factor", 0)
        
        # Check all criteria
        meets_sharpe = sharpe_ratio >= criteria.get("min_sharpe_ratio", 1.5)
        meets_win_rate = win_rate >= criteria.get("min_profitable_trades_pct", 55)
        meets_drawdown = max_drawdown <= criteria.get("max_drawdown_pct", 15)
        meets_profit_factor = profit_factor >= criteria.get("min_profit_factor", 1.3)
        
        # Log results
        logger.info(f"Promotion check for strategy {strategy_metrics.get('strategy_id', 'unknown')}:")
        logger.info(f"  Sharpe Ratio: {sharpe_ratio:.2f} (≥ {criteria.get('min_sharpe_ratio', 1.5)}) - {'PASS' if meets_sharpe else 'FAIL'}")
        logger.info(f"  Win Rate: {win_rate:.2f}% (≥ {criteria.get('min_profitable_trades_pct', 55)}%) - {'PASS' if meets_win_rate else 'FAIL'}")
        logger.info(f"  Max Drawdown: {max_drawdown:.2f}% (≤ {criteria.get('max_drawdown_pct', 15)}%) - {'PASS' if meets_drawdown else 'FAIL'}")
        logger.info(f"  Profit Factor: {profit_factor:.2f} (≥ {criteria.get('min_profit_factor', 1.3)}) - {'PASS' if meets_profit_factor else 'FAIL'}")
        
        # Strategy must meet all criteria
        return meets_sharpe and meets_win_rate and meets_drawdown and meets_profit_factor
    
    def deploy_strategy_to_benbot(self, strategy, target_env="paper"):
        """
        Deploy an evolved strategy to BensBot for trading.
        
        Args:
            strategy: The strategy to deploy
            target_env: Target environment ("paper" or "live")
            
        Returns:
            Boolean indicating success
        """
        from trading_bot.research.evotrader.strategy_adapter import convert_to_benbot_strategy
        
        # Convert strategy back to BensBot format
        benbot_strategy = convert_to_benbot_strategy(strategy)
        
        if benbot_strategy is None:
            logger.error("Failed to convert strategy to BensBot format")
            return False
        
        # Deploy to target environment
        logger.info(f"Deploying strategy {strategy.strategy_id} to BensBot {target_env} environment")
        
        # In a real implementation, this would connect to BensBot's strategy
        # management system to register and activate the new strategy
        
        # For demo purposes, we'll just save the strategy to disk
        strategy_dir = os.path.join(self.output_dir, "deployed_strategies", target_env)
        os.makedirs(strategy_dir, exist_ok=True)
        
        strategy_file = os.path.join(strategy_dir, f"{strategy.strategy_id}.json")
        
        with open(strategy_file, "w") as f:
            json.dump({
                "strategy_id": strategy.strategy_id,
                "parameters": strategy.parameters,
                "metadata": strategy.metadata,
                "deployment_time": datetime.datetime.now().isoformat(),
                "target_env": target_env
            }, f, indent=2)
        
        logger.info(f"Strategy deployed successfully: {strategy_file}")
        return True
