#!/usr/bin/env python3
"""
Strategy Evaluator - Determines if strategies meet performance criteria and scores them

This module evaluates trading strategies against strict performance criteria
and generates standardized scores to guide the evolutionary process.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import datetime
import logging
from scipy.stats import linregress

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import our modules
from synthetic_market_generator import SyntheticMarketGenerator
from strategy_registry import StrategyRegistry


class StrategyEvaluator:
    """
    Evaluates trading strategies against performance thresholds and 
    generates standardized scores for comparison and selection.
    """
    
    # Default thresholds (ambitious targets based on user requirements)
    DEFAULT_THRESHOLDS = {
        "min_return_pct": 10.0,           # Minimum return percentage
        "max_drawdown_pct": 15.0,         # Maximum drawdown percentage
        "min_win_rate_pct": 70.0,         # Minimum win rate percentage
        "min_trade_count": 20,            # Minimum number of trades
        "min_profit_factor": 1.5,         # Minimum profit factor
        "min_sharpe_ratio": 1.0,          # Minimum Sharpe ratio
        "max_consistency_score": 0.7,     # Maximum equity curve consistency score (lower is better)
    }
    
    # Default scoring weights
    DEFAULT_WEIGHTS = {
        "return_pct": 0.30,               # Weight for return
        "risk_adjusted_return": 0.25,     # Weight for risk-adjusted return (return/drawdown)
        "win_rate": 0.15,                 # Weight for win rate 
        "consistency": 0.15,              # Weight for equity curve consistency
        "profit_factor": 0.15,            # Weight for profit factor
    }
    
    def __init__(
        self, 
        thresholds: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
        registry_path: str = "./strategy_registry"
    ):
        """
        Initialize the strategy evaluator.
        
        Args:
            thresholds: Custom performance thresholds
            weights: Custom scoring weights
            registry_path: Path to strategy registry
        """
        # Use default or custom thresholds
        self.thresholds = thresholds if thresholds else self.DEFAULT_THRESHOLDS.copy()
        
        # Use default or custom weights
        self.weights = weights if weights else self.DEFAULT_WEIGHTS.copy()
        
        # Initialize registry
        self.registry = StrategyRegistry(registry_path)
        
        # Initialize market generator
        self.market_generator = SyntheticMarketGenerator()
        
        # Logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"logs/strategy_evaluator_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            ]
        )
        
        self.logger = logging.getLogger('strategy_evaluator')
        
        # Results directory
        self.results_dir = f"evaluation_results/results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def evaluate_strategy(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a strategy against performance thresholds.
        
        Args:
            strategy_results: Dictionary with strategy backtest results
            
        Returns:
            Dictionary with evaluation results
        """
        # Extract key metrics
        return_pct = strategy_results.get("total_return_pct", 0)
        drawdown_pct = strategy_results.get("max_drawdown", 0)
        win_rate_pct = strategy_results.get("win_rate", 0)
        trade_count = strategy_results.get("trade_count", 0)
        profit_factor = strategy_results.get("profit_factor", 0)
        
        # Calculate additional metrics
        risk_adjusted_return = return_pct / drawdown_pct if drawdown_pct > 0 else 0
        
        # Calculate equity curve consistency
        equity_curve = strategy_results.get("equity_curve", [])
        consistency_score = self._calculate_consistency_score(equity_curve)
        
        # Check thresholds
        meets_return = return_pct >= self.thresholds["min_return_pct"]
        meets_drawdown = drawdown_pct <= self.thresholds["max_drawdown_pct"]
        meets_win_rate = win_rate_pct >= self.thresholds["min_win_rate_pct"]
        meets_trade_count = trade_count >= self.thresholds["min_trade_count"]
        meets_profit_factor = profit_factor >= self.thresholds["min_profit_factor"]
        meets_consistency = consistency_score <= self.thresholds["max_consistency_score"]
        
        # Overall threshold check
        meets_all_thresholds = all([
            meets_return,
            meets_drawdown,
            meets_win_rate,
            meets_trade_count,
            meets_profit_factor,
            meets_consistency
        ])
        
        # Calculate score (0-100)
        score = self._calculate_score(
            return_pct=return_pct,
            risk_adjusted_return=risk_adjusted_return,
            win_rate_pct=win_rate_pct,
            consistency_score=consistency_score,
            profit_factor=profit_factor
        )
        
        # Prepare evaluation result
        evaluation = {
            "meets_thresholds": meets_all_thresholds,
            "score": score,
            "metrics": {
                "return_pct": return_pct,
                "drawdown_pct": drawdown_pct,
                "win_rate_pct": win_rate_pct,
                "trade_count": trade_count,
                "profit_factor": profit_factor,
                "risk_adjusted_return": risk_adjusted_return,
                "consistency_score": consistency_score
            },
            "threshold_results": {
                "meets_return": meets_return,
                "meets_drawdown": meets_drawdown,
                "meets_win_rate": meets_win_rate,
                "meets_trade_count": meets_trade_count,
                "meets_profit_factor": meets_profit_factor,
                "meets_consistency": meets_consistency
            }
        }
        
        return evaluation
    
    def _calculate_consistency_score(self, equity_curve: List[float]) -> float:
        """
        Calculate consistency score for an equity curve.
        Lower scores are better (less volatility).
        
        Args:
            equity_curve: List of equity values
            
        Returns:
            Consistency score (0-1)
        """
        if len(equity_curve) < 2:
            return 1.0  # Worst score for insufficient data
        
        # Calculate daily returns
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Calculate volatility of returns
        volatility = np.std(returns)
        
        # Calculate drawdowns
        drawdowns = []
        peak = equity_curve[0]
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            
            drawdown = (peak - equity) / peak
            drawdowns.append(drawdown)
        
        # Calculate average drawdown
        avg_drawdown = np.mean(drawdowns)
        
        # Calculate linear regression R^2 (higher is more consistent)
        x = np.arange(len(equity_curve))
        slope, intercept, r_value, p_value, std_err = linregress(x, equity_curve)
        r_squared = r_value ** 2
        
        # Combine metrics for consistency score
        # Lower is better - scale between 0 and 1
        consistency_score = (
            (volatility * 10) +      # Weight volatility heavily
            (avg_drawdown * 5) +     # Weight drawdowns
            (1 - r_squared)          # Linear trend deviation
        ) / 16                       # Normalize to 0-1 range
        
        return min(1.0, max(0.0, consistency_score))
    
    def _calculate_score(
        self,
        return_pct: float,
        risk_adjusted_return: float,
        win_rate_pct: float,
        consistency_score: float,
        profit_factor: float
    ) -> float:
        """
        Calculate overall strategy score (0-100).
        
        Args:
            return_pct: Return percentage
            risk_adjusted_return: Risk-adjusted return
            win_rate_pct: Win rate percentage
            consistency_score: Consistency score (lower is better)
            profit_factor: Profit factor
            
        Returns:
            Strategy score (0-100)
        """
        # Normalize metrics to 0-100 scale
        normalized_return = min(100, return_pct * 5)  # Cap at 100 (20% return)
        
        normalized_risk_adj = min(100, risk_adjusted_return * 20)  # Cap at 100 (ratio of 5)
        
        normalized_win_rate = win_rate_pct
        
        # Consistency score is inverted (lower is better)
        normalized_consistency = 100 * (1 - consistency_score)
        
        normalized_profit_factor = min(100, profit_factor * 25)  # Cap at 100 (factor of 4)
        
        # Calculate weighted score
        score = (
            self.weights["return_pct"] * normalized_return +
            self.weights["risk_adjusted_return"] * normalized_risk_adj +
            self.weights["win_rate"] * normalized_win_rate +
            self.weights["consistency"] * normalized_consistency +
            self.weights["profit_factor"] * normalized_profit_factor
        )
        
        return round(score, 2)
    
    def evaluate_against_multiple_scenarios(
        self,
        strategy_obj: Any,
        scenarios: Optional[List[str]] = None,
        min_score: float = 70.0
    ) -> Dict[str, Any]:
        """
        Evaluate a strategy against multiple market scenarios.
        
        Args:
            strategy_obj: Strategy object
            scenarios: List of scenario names to test against
            min_score: Minimum score to be considered successful
            
        Returns:
            Dictionary with multi-scenario evaluation results
        """
        # Default scenarios if none provided
        if scenarios is None:
            scenarios = [
                "bull_market", "bear_market", "sideways_market",
                "volatile_market", "flash_crash", "sector_rotation"
            ]
        
        # Generate market data for each scenario
        scenario_data = {}
        for scenario in scenarios:
            if scenario == "bull_market":
                scenario_data[scenario] = self.market_generator.generate_bull_market()
            elif scenario == "bear_market":
                scenario_data[scenario] = self.market_generator.generate_bear_market()
            elif scenario == "sideways_market":
                scenario_data[scenario] = self.market_generator.generate_sideways_market()
            elif scenario == "volatile_market":
                scenario_data[scenario] = self.market_generator.generate_volatile_market()
            elif scenario == "flash_crash":
                scenario_data[scenario] = self.market_generator.generate_flash_crash()
            elif scenario == "sector_rotation":
                scenario_data[scenario] = self.market_generator.generate_sector_rotation()
        
        # Evaluate on each scenario
        scenario_results = {}
        overall_success = True
        total_score = 0
        
        for scenario, market_data in scenario_data.items():
            # Backtest strategy
            from advanced_strategies import backtest_strategy
            backtest_result = backtest_strategy(strategy_obj, market_data)
            
            # Evaluate results
            evaluation = self.evaluate_strategy(backtest_result)
            
            # Store results
            scenario_results[scenario] = evaluation
            
            # Update overall success
            if evaluation["score"] < min_score:
                overall_success = False
            
            # Accumulate score
            total_score += evaluation["score"]
        
        # Calculate average score
        avg_score = total_score / len(scenarios) if scenarios else 0
        
        # Prepare multi-scenario evaluation
        multi_scenario_eval = {
            "overall_success": overall_success,
            "average_score": avg_score,
            "scenario_results": scenario_results,
            "strategy_name": strategy_obj.strategy_name,
            "strategy_parameters": strategy_obj.parameters
        }
        
        return multi_scenario_eval
    
    def format_for_benbot_records(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format evaluation results for BenBot data records.
        
        Args:
            evaluation: Strategy evaluation results
            
        Returns:
            Formatted data for BenBot records
        """
        # Generate strategy fingerprint
        if "strategy_parameters" in evaluation:
            strategy_type = evaluation.get("strategy_name", "Unknown")
            parameters = evaluation.get("strategy_parameters", {})
            
            # Create simple fingerprint based on type and parameters
            parameter_string = "-".join([f"{k}_{v}" for k, v in sorted(parameters.items())])
            fingerprint = f"{strategy_type}-{hash(parameter_string) % 100000}"
        else:
            fingerprint = f"Strategy-{hash(str(evaluation)) % 100000}"
        
        # Format for BenBot records
        benbot_data = {
            "strategy_id": fingerprint,
            "timestamp": datetime.datetime.now().isoformat(),
            "overall_score": evaluation.get("average_score", evaluation.get("score", 0)),
            "meets_thresholds": evaluation.get("overall_success", evaluation.get("meets_thresholds", False)),
            "performance_metrics": {}
        }
        
        # Add metrics
        if "metrics" in evaluation:
            benbot_data["performance_metrics"] = evaluation["metrics"]
        
        # Add scenario data if available
        if "scenario_results" in evaluation:
            benbot_data["scenario_performance"] = {}
            
            for scenario, results in evaluation["scenario_results"].items():
                benbot_data["scenario_performance"][scenario] = {
                    "score": results.get("score", 0),
                    "meets_thresholds": results.get("meets_thresholds", False),
                }
                
                if "metrics" in results:
                    benbot_data["scenario_performance"][scenario]["metrics"] = results["metrics"]
        
        # Add best/worst scenarios
        if "scenario_results" in evaluation:
            scenarios = list(evaluation["scenario_results"].keys())
            
            if scenarios:
                # Find best scenario
                best_scenario = max(
                    scenarios,
                    key=lambda s: evaluation["scenario_results"][s]["score"]
                )
                
                # Find worst scenario
                worst_scenario = min(
                    scenarios,
                    key=lambda s: evaluation["scenario_results"][s]["score"]
                )
                
                benbot_data["best_scenario"] = best_scenario
                benbot_data["worst_scenario"] = worst_scenario
        
        return benbot_data
    
    def save_benbot_data(self, benbot_data: Dict[str, Any]) -> str:
        """
        Save BenBot data to file.
        
        Args:
            benbot_data: Formatted BenBot data
            
        Returns:
            Path to saved file
        """
        # Create BenBot data directory
        benbot_dir = os.path.join(self.results_dir, "benbot_data")
        os.makedirs(benbot_dir, exist_ok=True)
        
        # Generate filename
        strategy_id = benbot_data.get("strategy_id", "unknown")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{strategy_id}_{timestamp}.json"
        
        # Save to file
        file_path = os.path.join(benbot_dir, filename)
        
        with open(file_path, "w") as f:
            json.dump(benbot_data, f, indent=2)
        
        return file_path


def evaluate_registry_strategies(
    registry_path: str = "./strategy_registry",
    min_generation: int = 5,
    min_score: float = 70.0,
    output_dir: str = None
):
    """
    Evaluate strategies from the registry.
    
    Args:
        registry_path: Path to strategy registry
        min_generation: Minimum generation to consider
        min_score: Minimum score to be considered successful
        output_dir: Directory for output
    """
    # Initialize evaluator
    evaluator = StrategyEvaluator(registry_path=registry_path)
    
    # Get strategies from registry
    registry = StrategyRegistry(registry_path)
    all_strategies = registry.get_all_strategies()
    
    # Filter by generation
    evolved_strategies = [
        s for s in all_strategies 
        if s.get("generation", 0) >= min_generation
    ]
    
    print(f"Evaluating {len(evolved_strategies)} strategies from generation {min_generation} onwards")
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"evaluation_results/registry_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare for results
    all_evaluations = []
    passing_strategies = []
    top_strategies = []
    
    # Evaluate each strategy
    for i, strategy_data in enumerate(evolved_strategies):
        print(f"Evaluating strategy {i+1}/{len(evolved_strategies)}")
        
        # Load strategy object
        strategy_type = strategy_data.get("strategy_type")
        parameters = strategy_data.get("parameters", {})
        
        try:
            # Import dynamically based on strategy type
            if strategy_type in ["MeanReversion", "Momentum", "VolumeProfile", "VolatilityBreakout"]:
                from advanced_strategies import (
                    MeanReversionStrategy, 
                    MomentumStrategy, 
                    VolumeProfileStrategy, 
                    VolatilityBreakoutStrategy
                )
                
                if strategy_type == "MeanReversion":
                    strategy_obj = MeanReversionStrategy(**parameters)
                elif strategy_type == "Momentum":
                    strategy_obj = MomentumStrategy(**parameters)
                elif strategy_type == "VolumeProfile":
                    strategy_obj = VolumeProfileStrategy(**parameters)
                elif strategy_type == "VolatilityBreakout":
                    strategy_obj = VolatilityBreakoutStrategy(**parameters)
                
                # Evaluate against multiple scenarios
                evaluation = evaluator.evaluate_against_multiple_scenarios(
                    strategy_obj=strategy_obj,
                    min_score=min_score
                )
                
                # Format for BenBot
                benbot_data = evaluator.format_for_benbot_records(evaluation)
                
                # Save BenBot data
                evaluator.save_benbot_data(benbot_data)
                
                # Store evaluation
                all_evaluations.append(evaluation)
                
                # Check if strategy passes thresholds
                if evaluation["overall_success"]:
                    passing_strategies.append(evaluation)
                
                # Add to top strategies
                top_strategies.append(evaluation)
                
            else:
                print(f"Unsupported strategy type: {strategy_type}")
        
        except Exception as e:
            print(f"Error evaluating strategy: {e}")
    
    # Sort top strategies by average score
    top_strategies.sort(key=lambda e: e.get("average_score", 0), reverse=True)
    top_strategies = top_strategies[:10]  # Keep top 10
    
    # Generate summary report
    summary = {
        "total_strategies": len(evolved_strategies),
        "passing_strategies": len(passing_strategies),
        "pass_rate": len(passing_strategies) / len(evolved_strategies) if evolved_strategies else 0,
        "top_strategies": [
            {
                "strategy_name": s.get("strategy_name"),
                "parameters": s.get("strategy_parameters"),
                "average_score": s.get("average_score"),
                "best_scenario": s.get("best_scenario"),
                "worst_scenario": s.get("worst_scenario")
            }
            for s in top_strategies
        ]
    }
    
    # Save summary
    with open(os.path.join(output_dir, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Evaluation complete. {len(passing_strategies)} strategies passed thresholds.")
    print(f"Results saved to {output_dir}")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trading strategies")
    
    parser.add_argument(
        "--registry", 
        type=str, 
        default="./strategy_registry",
        help="Path to strategy registry"
    )
    
    parser.add_argument(
        "--min-generation", 
        type=int, 
        default=5,
        help="Minimum generation to consider"
    )
    
    parser.add_argument(
        "--min-score", 
        type=float, 
        default=70.0,
        help="Minimum score to be considered successful"
    )
    
    args = parser.parse_args()
    
    evaluate_registry_strategies(
        registry_path=args.registry,
        min_generation=args.min_generation,
        min_score=args.min_score
    )
