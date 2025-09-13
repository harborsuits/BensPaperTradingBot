"""
Backtest Scorer for Machine Learning Pipelines
This module scores backtest results against strategy-specific optimization goals.
"""

import os
import sys
import logging
import datetime
from typing import Dict, List, Any, Optional, Union
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import variant generator for strategy profiles
from trading_bot.ml_pipeline.variant_generator import get_variant_generator


class BacktestScorer:
    """
    Scores backtest results against strategy-specific optimization goals.
    """
    
    def __init__(self):
        """Initialize the backtest scorer."""
        self.logger = logging.getLogger("BacktestScorer")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Get variant generator for strategy profiles
        self.variant_generator = get_variant_generator()
        
        # Performance weights (can be customized)
        self.weight_sharpe = 0.4
        self.weight_return = 0.25
        self.weight_drawdown = 0.2
        self.weight_win_rate = 0.15
        
        self.logger.info("BacktestScorer initialized")
    
    def score_result(self, result: Dict, strategy_id: str, current_regime: str = None) -> Dict:
        """
        Score a backtest result against strategy-specific optimization goals.
        
        Args:
            result: Dictionary containing backtest results
            strategy_id: Strategy identifier
            current_regime: Optional current market regime
        
        Returns:
            Dictionary with score and breakdown
        """
        if not result:
            return {"score": 0, "components": {}, "passed": False, "reason": "No results provided"}
        
        # Get optimization goals for the strategy
        goals = self.variant_generator.get_optimization_goals(strategy_id)
        
        if not goals:
            self.logger.warning(f"No optimization goals found for strategy: {strategy_id}")
            goals = {
                "min_sharpe": 1.0,
                "target_sharpe": 1.5,
                "max_drawdown": -15.0,
                "min_return": 8.0,
                "min_win_rate": 0.5
            }
        
        # Extract metrics from result
        sharpe = result.get("sharpe_ratio", 0)
        total_return = result.get("total_return", 0)
        win_rate = result.get("win_rate", 0)
        max_drawdown = result.get("max_drawdown", -999)  # More negative is worse
        
        # For drawdown, make sure we have the sign we expect (negative)
        if max_drawdown > 0:
            max_drawdown = -max_drawdown
        
        # Calculate score components
        # Each component is normalized between 0-1
        
        # Sharpe Ratio score (0-1)
        min_sharpe = goals.get("min_sharpe", 1.0)
        target_sharpe = goals.get("target_sharpe", 1.5)
        sharpe_range = target_sharpe - min_sharpe
        
        if sharpe_range <= 0:
            sharpe_score = 1.0 if sharpe >= min_sharpe else 0.0
        else:
            sharpe_score = min(max((sharpe - min_sharpe) / sharpe_range, 0), 1)
        
        # Return score (0-1)
        min_return = goals.get("min_return", 8.0)
        target_return = min_return * 1.5  # Target 50% better than minimum
        return_range = target_return - min_return
        
        if return_range <= 0:
            return_score = 1.0 if total_return >= min_return else 0.0
        else:
            return_score = min(max((total_return - min_return) / return_range, 0), 1)
        
        # Drawdown score (0-1)
        max_dd_goal = goals.get("max_drawdown", -15.0)
        if max_dd_goal > 0:
            max_dd_goal = -max_dd_goal  # Make sure it's negative
        
        # Good is close to 0, bad is very negative
        target_dd = max_dd_goal / 2  # Target half as much drawdown
        
        if max_dd_goal >= 0:
            drawdown_score = 0.0  # Invalid goal
        else:
            # Normalize: 1 if drawdown is 0, 0 if drawdown is worse than max_dd_goal
            drawdown_score = min(max((max_drawdown - max_dd_goal) / (0 - max_dd_goal), 0), 1)
        
        # Win rate score (0-1)
        min_win_rate = goals.get("min_win_rate", 0.5)
        target_win_rate = min(min_win_rate + 0.2, 0.9)  # Target 20% better, capped at 90%
        win_rate_range = target_win_rate - min_win_rate
        
        if win_rate_range <= 0:
            win_rate_score = 1.0 if win_rate >= min_win_rate else 0.0
        else:
            win_rate_score = min(max((win_rate - min_win_rate) / win_rate_range, 0), 1)
        
        # Calculate weighted score
        weighted_score = (
            sharpe_score * self.weight_sharpe +
            return_score * self.weight_return +
            drawdown_score * self.weight_drawdown +
            win_rate_score * self.weight_win_rate
        )
        
        # Round to 2 decimal places
        weighted_score = round(weighted_score, 2)
        
        # Determine if the result passes minimum criteria
        # Each component must meet at least 70% of its minimum threshold
        passed = (
            sharpe >= min_sharpe * 0.7 and
            total_return >= min_return * 0.7 and
            max_drawdown >= max_dd_goal * 1.3 and  # 1.3x because less negative is better
            win_rate >= min_win_rate * 0.7
        )
        
        # Determine the reason for pass/fail
        reason = "All criteria met" if passed else "Failed criteria: "
        
        if not passed:
            if sharpe < min_sharpe * 0.7:
                reason += f"Sharpe ({sharpe:.2f} < {min_sharpe * 0.7:.2f}) "
            if total_return < min_return * 0.7:
                reason += f"Return ({total_return:.2f}% < {min_return * 0.7:.2f}%) "
            if max_drawdown < max_dd_goal * 1.3:  # More negative is worse
                reason += f"Drawdown ({max_drawdown:.2f}% < {max_dd_goal * 1.3:.2f}%) "
            if win_rate < min_win_rate * 0.7:
                reason += f"Win Rate ({win_rate:.2f} < {min_win_rate * 0.7:.2f}) "
        
        # Regime compatibility bonus (10% boost if optimal regime)
        preferred_regime = self.variant_generator.get_preferred_regime(strategy_id)
        regime_compatible = (preferred_regime == "all" or 
                             (current_regime and preferred_regime == current_regime))
        
        if regime_compatible and passed and preferred_regime != "all":
            weighted_score = min(weighted_score * 1.1, 1.0)  # 10% boost, capped at 1.0
            reason += f" (Regime bonus: {preferred_regime})"
        
        # Create score breakdown
        score_breakdown = {
            "sharpe": {
                "value": sharpe,
                "goal": min_sharpe,
                "score": round(sharpe_score, 2),
                "weight": self.weight_sharpe
            },
            "return": {
                "value": total_return,
                "goal": min_return,
                "score": round(return_score, 2),
                "weight": self.weight_return
            },
            "drawdown": {
                "value": max_drawdown,
                "goal": max_dd_goal,
                "score": round(drawdown_score, 2),
                "weight": self.weight_drawdown
            },
            "win_rate": {
                "value": win_rate,
                "goal": min_win_rate,
                "score": round(win_rate_score, 2),
                "weight": self.weight_win_rate
            },
            "regime_compatible": regime_compatible
        }
        
        # Return combined score and breakdown
        return {
            "score": weighted_score,
            "components": score_breakdown,
            "passed": passed,
            "reason": reason
        }
    
    def summarize_results(self, all_results: List[Dict], strategy_id: str) -> Dict:
        """
        Summarize a set of backtest results.
        
        Args:
            all_results: List of dictionaries with backtest results and scores
            strategy_id: Strategy identifier
        
        Returns:
            Dictionary with summary statistics
        """
        if not all_results:
            return {"count": 0, "pass_rate": 0, "avg_score": 0}
        
        # Extract scores
        scores = [r.get("score", {}).get("score", 0) for r in all_results]
        passed = [r.get("score", {}).get("passed", False) for r in all_results]
        
        # Calculate statistics
        count = len(all_results)
        pass_count = sum(passed)
        pass_rate = pass_count / count if count > 0 else 0
        avg_score = sum(scores) / count if count > 0 else 0
        max_score = max(scores) if scores else 0
        min_score = min(scores) if scores else 0
        
        # Find best result
        best_idx = scores.index(max_score) if scores else -1
        best_result = all_results[best_idx] if best_idx >= 0 else None
        
        return {
            "count": count,
            "pass_count": pass_count,
            "pass_rate": round(pass_rate, 2),
            "avg_score": round(avg_score, 2),
            "max_score": round(max_score, 2),
            "min_score": round(min_score, 2),
            "best_result": best_result
        }


# Create singleton instance
_backtest_scorer = None

def get_backtest_scorer():
    """
    Get the singleton BacktestScorer instance.
    
    Returns:
        BacktestScorer instance
    """
    global _backtest_scorer
    if _backtest_scorer is None:
        _backtest_scorer = BacktestScorer()
    return _backtest_scorer
