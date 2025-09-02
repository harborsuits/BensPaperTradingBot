#!/usr/bin/env python3
"""
Test Funded Strategies - Run and analyze the performance of funded account strategies

This script tests the best strategies evolved for funded account criteria against
multiple market conditions and generates detailed performance reports.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union
import datetime
import logging
import argparse
from collections import defaultdict

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import our modules
from funded_account_evaluator import FundedAccountEvaluator
from strategy_registry import StrategyRegistry
from synthetic_market_generator import SyntheticMarketGenerator
from advanced_strategies import (
    MeanReversionStrategy, 
    MomentumStrategy, 
    VolumeProfileStrategy, 
    VolatilityBreakout
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/test_funded_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger('test_funded_strategies')


def create_strategy(strategy_type: str, parameters: Dict[str, Any]) -> Any:
    """
    Create a strategy instance from type and parameters.
    
    Args:
        strategy_type: Type of strategy
        parameters: Strategy parameters
        
    Returns:
        Strategy instance
    """
    if strategy_type == "MeanReversion":
        return MeanReversionStrategy(**parameters)
    elif strategy_type == "Momentum":
        return MomentumStrategy(**parameters)
    elif strategy_type == "VolumeProfile":
        return VolumeProfileStrategy(**parameters)
    elif strategy_type == "VolatilityBreakout":
        return VolatilityBreakoutStrategy(**parameters)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


def load_strategies_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load strategies from a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of strategy configurations
    """
    with open(file_path, "r") as f:
        strategies_data = json.load(f)
    
    # Convert to standard format
    strategies = []
    
    for data in strategies_data:
        if "strategy_type" in data and "parameters" in data:
            strategies.append({
                "strategy_type": data["strategy_type"],
                "parameters": data["parameters"]
            })
        elif "id" in data and "strategy_type" in data and "parameters" in data:
            # Format from robust_strategies.json
            strategies.append({
                "strategy_type": data["strategy_type"],
                "parameters": data["parameters"]
            })
    
    return strategies


def test_strategies_on_multiple_markets(
    strategies: List[Dict[str, Any]],
    evaluator: FundedAccountEvaluator,
    market_types: Optional[List[str]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Test strategies on multiple market types.
    
    Args:
        strategies: List of strategy configurations
        evaluator: Funded account evaluator
        market_types: List of market types to test on
        output_dir: Directory for output
        
    Returns:
        Dictionary of test results
    """
    if market_types is None:
        market_types = ["bull", "bear", "sideways", "volatile", "mixed"]
    
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"funded_test_results_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Test each strategy on each market type
    results = {}
    
    for market_type in market_types:
        logger.info(f"Testing strategies on {market_type} market")
        
        # Generate market data
        market_data = evaluator.generate_evaluation_period(21, market_type)
        
        # Save market data
        market_data.to_csv(os.path.join(output_dir, f"{market_type}_market.csv"))
        
        # Test strategies
        market_results = []
        
        for strategy_config in strategies:
            strategy_type = strategy_config["strategy_type"]
            parameters = strategy_config["parameters"]
            
            # Create strategy
            strategy = create_strategy(strategy_type, parameters)
            
            # Evaluate strategy
            evaluation = evaluator.evaluate_strategy(strategy, market_data)
            
            # Record result
            result = {
                "strategy_type": strategy_type,
                "parameters": parameters,
                "evaluation": evaluation
            }
            
            market_results.append(result)
        
        # Sort by score
        market_results.sort(key=lambda r: r["evaluation"]["score"], reverse=True)
        
        # Store results
        results[market_type] = market_results
    
    # Calculate strategy robustness
    strategy_robustness = calculate_strategy_robustness(results)
    
    # Save overall results
    with open(os.path.join(output_dir, "overall_results.json"), "w") as f:
        # Convert to serializable format
        serializable_results = {}
        
        for market_type, market_results in results.items():
            serializable_results[market_type] = []
            
            for result in market_results:
                # Convert daily_pnl keys to strings if needed
                evaluation = result["evaluation"].copy()
                
                if "daily_pnl" in evaluation:
                    evaluation["daily_pnl"] = {
                        str(date) if not isinstance(date, str) else date: value
                        for date, value in evaluation["daily_pnl"].items()
                    }
                
                serializable_results[market_type].append({
                    "strategy_type": result["strategy_type"],
                    "parameters": result["parameters"],
                    "evaluation": evaluation
                })
        
        json.dump(serializable_results, f, indent=2)
    
    # Save robustness results
    with open(os.path.join(output_dir, "strategy_robustness.json"), "w") as f:
        json.dump(strategy_robustness, f, indent=2)
    
    # Generate summary report
    generate_summary_report(results, strategy_robustness, output_dir)
    
    # Generate visualizations
    generate_visualizations(results, strategy_robustness, output_dir)
    
    return results


def calculate_strategy_robustness(results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Calculate the robustness of strategies across different market types.
    
    Args:
        results: Dictionary of test results by market type
        
    Returns:
        List of strategy robustness metrics
    """
    # Track performance across market types
    strategy_performance = defaultdict(dict)
    
    for market_type, market_results in results.items():
        for result in market_results:
            strategy_type = result["strategy_type"]
            parameters = result["parameters"]
            evaluation = result["evaluation"]
            
            # Create a strategy identifier
            strategy_id = f"{strategy_type}-{hash(str(parameters))}"
            
            # Record performance in this market type
            strategy_performance[strategy_id][market_type] = {
                "score": evaluation["score"],
                "passes_evaluation": evaluation["passes_evaluation"],
                "metrics": evaluation["metrics"]
            }
            
    # Calculate robustness metrics
    robustness = []
    
    for strategy_id, performance in strategy_performance.items():
        # Extract strategy type and parameters
        for market_type, market_results in results.items():
            for result in market_results:
                test_id = f"{result['strategy_type']}-{hash(str(result['parameters']))}"
                
                if test_id == strategy_id:
                    strategy_type = result["strategy_type"]
                    parameters = result["parameters"]
                    break
            else:
                continue
            break
        
        # Count market types where strategy passes
        passing_markets = []
        scores = []
        returns = []
        drawdowns = []
        daily_losses = []
        
        for market_type, metrics in performance.items():
            scores.append(metrics["score"])
            
            if metrics["passes_evaluation"]:
                passing_markets.append(market_type)
            
            returns.append(metrics["metrics"]["total_return_pct"])
            drawdowns.append(metrics["metrics"]["max_drawdown"])
            
            if "worst_daily_loss" in metrics["metrics"]:
                daily_losses.append(metrics["metrics"]["worst_daily_loss"])
        
        # Calculate variability
        return_variability = np.std(returns) if returns else 0
        drawdown_variability = np.std(drawdowns) if drawdowns else 0
        
        # Calculate robustness score (higher is better)
        avg_score = np.mean(scores) if scores else 0
        passing_rate = len(passing_markets) / len(performance) if performance else 0
        
        # This formula rewards high average score, high passing rate, and low variability
        robustness_score = (
            avg_score * 0.4 +  # 40% weight for score
            passing_rate * 50 +  # 50% weight for passing rate (scaled to 0-50)
            (1 - return_variability / 10) * 5 +  # 5% weight for return stability
            (1 - drawdown_variability / 5) * 5  # 5% weight for drawdown stability
        )
        
        # Add to robustness list
        robustness.append({
            "strategy_id": strategy_id,
            "strategy_type": strategy_type,
            "parameters": parameters,
            "robustness_score": robustness_score,
            "passing_markets": passing_markets,
            "passing_rate": passing_rate,
            "avg_score": avg_score,
            "avg_return": np.mean(returns) if returns else 0,
            "avg_drawdown": np.mean(drawdowns) if drawdowns else 0,
            "avg_daily_loss": np.mean(daily_losses) if daily_losses else 0,
            "return_variability": return_variability,
            "drawdown_variability": drawdown_variability
        })
    
    # Sort by robustness score
    robustness.sort(key=lambda r: r["robustness_score"], reverse=True)
    
    return robustness


def generate_summary_report(
    results: Dict[str, List[Dict[str, Any]]],
    robustness: List[Dict[str, Any]],
    output_dir: str
):
    """
    Generate a summary report of test results.
    
    Args:
        results: Dictionary of test results by market type
        robustness: List of strategy robustness metrics
        output_dir: Directory for output
    """
    report = []
    report.append("=======================================")
    report.append("FUNDED STRATEGY TEST SUMMARY REPORT")
    report.append("=======================================")
    report.append("")
    
    # Report on each market type
    for market_type, market_results in results.items():
        passing_count = sum(1 for r in market_results if r["evaluation"]["passes_evaluation"])
        
        report.append(f"{market_type.upper()} MARKET:")
        report.append(f"  Total Strategies: {len(market_results)}")
        report.append(f"  Passing Strategies: {passing_count} ({passing_count/len(market_results)*100:.1f}%)")
        
        if passing_count > 0:
            # Get top passing strategy
            top_passing = next((r for r in market_results if r["evaluation"]["passes_evaluation"]), market_results[0])
            
            report.append(f"  Top Strategy: {top_passing['strategy_type']}")
            report.append(f"  Score: {top_passing['evaluation']['score']:.2f}")
            report.append(f"  Return: {top_passing['evaluation']['metrics']['total_return_pct']:.2f}%")
            report.append(f"  Max Drawdown: {top_passing['evaluation']['metrics']['max_drawdown']:.2f}%")
            
            if "worst_daily_loss" in top_passing['evaluation']['metrics']:
                report.append(f"  Worst Daily Loss: {top_passing['evaluation']['metrics']['worst_daily_loss']:.2f}%")
        
        report.append("")
    
    # Report on robust strategies
    report.append("MOST ROBUST STRATEGIES:")
    
    for i, strategy in enumerate(robustness[:5]):  # Top 5 robust strategies
        report.append(f"{i+1}. {strategy['strategy_type']} (ID: {strategy['strategy_id'][-6:]})")
        report.append(f"   Robustness Score: {strategy['robustness_score']:.2f}")
        report.append(f"   Passing Rate: {strategy['passing_rate']:.2f} ({len(strategy['passing_markets'])} of {len(results)} markets)")
        report.append(f"   Passing Markets: {', '.join(strategy['passing_markets'])}")
        report.append(f"   Avg Score: {strategy['avg_score']:.2f}")
        report.append(f"   Avg Return: {strategy['avg_return']:.2f}%")
        report.append(f"   Avg Drawdown: {strategy['avg_drawdown']:.2f}%")
        report.append("")
    
    # Add strategy type analysis
    strategy_type_performance = defaultdict(list)
    
    for market_type, market_results in results.items():
        for result in market_results:
            strategy_type = result["strategy_type"]
            evaluation = result["evaluation"]
            
            strategy_type_performance[strategy_type].append({
                "market_type": market_type,
                "score": evaluation["score"],
                "passes_evaluation": evaluation["passes_evaluation"],
                "return": evaluation["metrics"]["total_return_pct"],
                "drawdown": evaluation["metrics"]["max_drawdown"]
            })
    
    # Calculate average performance by strategy type
    report.append("STRATEGY TYPE ANALYSIS:")
    
    for strategy_type, performances in strategy_type_performance.items():
        passing_count = sum(1 for p in performances if p["passes_evaluation"])
        
        avg_score = np.mean([p["score"] for p in performances])
        avg_return = np.mean([p["return"] for p in performances])
        avg_drawdown = np.mean([p["drawdown"] for p in performances])
        
        report.append(f"{strategy_type}:")
        report.append(f"  Passing Rate: {passing_count/len(performances)*100:.1f}%")
        report.append(f"  Avg Score: {avg_score:.2f}")
        report.append(f"  Avg Return: {avg_return:.2f}%")
        report.append(f"  Avg Drawdown: {avg_drawdown:.2f}%")
        report.append("")
    
    report.append("=======================================")
    
    # Save report
    with open(os.path.join(output_dir, "summary_report.txt"), "w") as f:
        f.write("\n".join(report))
    
    # Print report
    print("\n".join(report))


def generate_visualizations(
    results: Dict[str, List[Dict[str, Any]]],
    robustness: List[Dict[str, Any]],
    output_dir: str
):
    """
    Generate visualizations of test results.
    
    Args:
        results: Dictionary of test results by market type
        robustness: List of strategy robustness metrics
        output_dir: Directory for output
    """
    # Create visualizations directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Passing rate by market type
    passing_rates = {}
    
    for market_type, market_results in results.items():
        passing_count = sum(1 for r in market_results if r["evaluation"]["passes_evaluation"])
        passing_rates[market_type] = passing_count / len(market_results) if market_results else 0
    
    plt.figure(figsize=(10, 6))
    market_types = list(passing_rates.keys())
    rates = [passing_rates[m] for m in market_types]
    
    plt.bar(market_types, rates, color='skyblue')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.6)
    
    plt.title("Strategy Pass Rate by Market Type")
    plt.xlabel("Market Type")
    plt.ylabel("Pass Rate")
    plt.ylim(0, 1)
    
    for i, v in enumerate(rates):
        plt.text(i, v + 0.02, f"{v:.1%}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "pass_rate_by_market.png"))
    plt.close()
    
    # 2. Robustness scores
    plt.figure(figsize=(12, 6))
    
    top_n = min(10, len(robustness))
    top_strategies = robustness[:top_n]
    
    strategy_ids = [f"{s['strategy_type']}\n({s['strategy_id'][-6:]})" for s in top_strategies]
    scores = [s["robustness_score"] for s in top_strategies]
    
    plt.bar(strategy_ids, scores, color='lightgreen')
    
    plt.title("Top Strategy Robustness Scores")
    plt.xlabel("Strategy")
    plt.ylabel("Robustness Score")
    plt.xticks(rotation=45, ha="right")
    
    for i, v in enumerate(scores):
        plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "robustness_scores.png"))
    plt.close()
    
    # 3. Performance metrics by strategy type
    strategy_type_performance = defaultdict(list)
    
    for market_results in results.values():
        for result in market_results:
            strategy_type = result["strategy_type"]
            evaluation = result["evaluation"]
            
            strategy_type_performance[strategy_type].append({
                "score": evaluation["score"],
                "return": evaluation["metrics"]["total_return_pct"],
                "drawdown": evaluation["metrics"]["max_drawdown"],
                "passes": evaluation["passes_evaluation"]
            })
    
    # 3a. Average scores by strategy type
    plt.figure(figsize=(10, 6))
    
    strategy_types = list(strategy_type_performance.keys())
    avg_scores = [np.mean([p["score"] for p in strategy_type_performance[st]]) for st in strategy_types]
    
    plt.bar(strategy_types, avg_scores, color='lightblue')
    
    plt.title("Average Score by Strategy Type")
    plt.xlabel("Strategy Type")
    plt.ylabel("Average Score")
    
    for i, v in enumerate(avg_scores):
        plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "avg_score_by_type.png"))
    plt.close()
    
    # 3b. Pass rate by strategy type
    plt.figure(figsize=(10, 6))
    
    pass_rates = [
        sum(1 for p in strategy_type_performance[st] if p["passes"]) / len(strategy_type_performance[st])
        for st in strategy_types
    ]
    
    plt.bar(strategy_types, pass_rates, color='lightcoral')
    
    plt.title("Pass Rate by Strategy Type")
    plt.xlabel("Strategy Type")
    plt.ylabel("Pass Rate")
    plt.ylim(0, 1)
    
    for i, v in enumerate(pass_rates):
        plt.text(i, v + 0.02, f"{v:.1%}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "pass_rate_by_type.png"))
    plt.close()
    
    # 4. Return vs. drawdown scatter plot
    plt.figure(figsize=(12, 8))
    
    for market_type, market_results in results.items():
        returns = [r["evaluation"]["metrics"]["total_return_pct"] for r in market_results]
        drawdowns = [r["evaluation"]["metrics"]["max_drawdown"] for r in market_results]
        passing = [r["evaluation"]["passes_evaluation"] for r in market_results]
        
        # Use different markers for passing and failing strategies
        passing_returns = [r for r, p in zip(returns, passing) if p]
        passing_drawdowns = [d for d, p in zip(drawdowns, passing) if p]
        
        failing_returns = [r for r, p in zip(returns, passing) if not p]
        failing_drawdowns = [d for d, p in zip(drawdowns, passing) if not p]
        
        plt.scatter(
            passing_drawdowns, 
            passing_returns, 
            alpha=0.7, 
            marker="o", 
            label=f"{market_type} (Pass)",
            s=100
        )
        
        plt.scatter(
            failing_drawdowns, 
            failing_returns, 
            alpha=0.4, 
            marker="x", 
            label=f"{market_type} (Fail)",
            s=50
        )
    
    # Add threshold lines
    plt.axhline(y=8, color='g', linestyle='--', alpha=0.6, label="Min Return (8%)")
    plt.axvline(x=5, color='r', linestyle='--', alpha=0.6, label="Max Drawdown (5%)")
    
    plt.title("Return vs. Drawdown by Market Type")
    plt.xlabel("Maximum Drawdown (%)")
    plt.ylabel("Total Return (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "return_vs_drawdown.png"))
    plt.close()
    
    # 5. Strategy type distribution in top performers
    top_strategies = []
    
    for market_results in results.values():
        # Add top 3 from each market
        top_strategies.extend(market_results[:3])
    
    strategy_types_count = defaultdict(int)
    
    for strategy in top_strategies:
        strategy_types_count[strategy["strategy_type"]] += 1
    
    plt.figure(figsize=(10, 6))
    
    types = list(strategy_types_count.keys())
    counts = [strategy_types_count[t] for t in types]
    
    plt.pie(
        counts, 
        labels=types, 
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette("Set3", len(types))
    )
    
    plt.title("Strategy Type Distribution in Top Performers")
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "top_strategy_types.png"))
    plt.close()


def test_registry_strategies(
    registry_path: str = "./strategy_registry",
    min_generation: int = 10,
    max_strategies_per_type: int = 5,
    output_dir: Optional[str] = None
):
    """
    Test strategies from the registry.
    
    Args:
        registry_path: Path to strategy registry
        min_generation: Minimum generation to consider
        max_strategies_per_type: Maximum number of strategies per type
        output_dir: Directory for output
        
    Returns:
        Test results
    """
    # Initialize registry
    registry = StrategyRegistry(registry_path)
    
    # Get all strategies
    all_strategies = registry.get_all_strategies()
    
    # Filter by generation
    filtered_strategies = [
        s for s in all_strategies
        if s.get("generation", 0) >= min_generation
    ]
    
    logger.info(f"Found {len(filtered_strategies)} strategies with generation >= {min_generation}")
    
    # Group by strategy type
    strategies_by_type = defaultdict(list)
    
    for strategy in filtered_strategies:
        strategy_type = strategy.get("strategy_type")
        
        strategies_by_type[strategy_type].append({
            "strategy_type": strategy_type,
            "parameters": strategy.get("parameters", {}),
            "fitness": strategy.get("fitness", 0),
            "generation": strategy.get("generation", 0)
        })
    
    # Sort each group by fitness and take top N
    selected_strategies = []
    
    for strategy_type, strategies in strategies_by_type.items():
        # Sort by fitness (descending)
        sorted_strategies = sorted(
            strategies, 
            key=lambda s: s.get("fitness", 0), 
            reverse=True
        )
        
        # Take top N
        selected_strategies.extend(sorted_strategies[:max_strategies_per_type])
    
    logger.info(f"Selected {len(selected_strategies)} strategies for testing")
    
    # Initialize evaluator
    evaluator = FundedAccountEvaluator()
    
    # Test strategies
    results = test_strategies_on_multiple_markets(
        strategies=[
            {
                "strategy_type": s["strategy_type"],
                "parameters": s["parameters"]
            }
            for s in selected_strategies
        ],
        evaluator=evaluator,
        output_dir=output_dir
    )
    
    return results


def main():
    """Main function to parse args and run tests."""
    parser = argparse.ArgumentParser(description="Test funded account strategies")
    
    parser.add_argument(
        "--strategies", 
        type=str,
        help="Path to JSON file containing strategies to test"
    )
    
    parser.add_argument(
        "--registry", 
        type=str, 
        default="./strategy_registry",
        help="Path to strategy registry"
    )
    
    parser.add_argument(
        "--min-generation", 
        type=int, 
        default=10,
        help="Minimum generation to consider from registry"
    )
    
    parser.add_argument(
        "--max-per-type", 
        type=int, 
        default=5,
        help="Maximum number of strategies per type to test"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"funded_test_results_{timestamp}"
    else:
        output_dir = args.output
    
    # Load strategies
    if args.strategies:
        # Load from file
        strategies = load_strategies_from_file(args.strategies)
        
        if not strategies:
            logger.error(f"No strategies found in {args.strategies}")
            return
        
        logger.info(f"Loaded {len(strategies)} strategies from {args.strategies}")
        
        # Initialize evaluator
        evaluator = FundedAccountEvaluator()
        
        # Test strategies
        test_strategies_on_multiple_markets(
            strategies=strategies,
            evaluator=evaluator,
            output_dir=output_dir
        )
    else:
        # Test strategies from registry
        test_registry_strategies(
            registry_path=args.registry,
            min_generation=args.min_generation,
            max_strategies_per_type=args.max_per_type,
            output_dir=output_dir
        )


if __name__ == "__main__":
    main()
