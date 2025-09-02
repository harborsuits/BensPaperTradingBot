#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EvoTrader Tournament Engine

This module implements the tournament system for evaluating and
comparing strategy candidates in competitive environments.
"""

import os
import sys
import random
import logging
import datetime
import uuid
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from collections import defaultdict

from trading_bot.evo.architecture import (
    StrategyCandidate, StrategyStage, FitnessMetric
)

logger = logging.getLogger(__name__)


class TournamentEngine:
    """
    Tournament engine for evaluating and comparing strategy candidates
    
    This engine runs parallel tournaments with different market conditions
    to identify strategies that perform consistently across environments.
    """
    
    def __init__(self, core, config: Dict[str, Any] = None):
        """
        Initialize the tournament engine
        
        Args:
            core: Reference to the EvoTrader core
            config: Configuration dictionary
        """
        self.core = core
        self.config = config or {}
        
        if not self.config and hasattr(core, 'config') and 'tournament' in core.config:
            self.config = core.config.get('tournament', {})
            
        self._set_default_config()
        
        # Track tournament statistics
        self.stats = {
            "tournaments_run": 0,
            "candidates_evaluated": 0,
            "promotions": 0,
            "eliminations": 0
        }
        
        # Initialize backtest engine reference
        self.backtest_engine = None
        
        logger.info("Tournament engine initialized")
    
    def _set_default_config(self):
        """Set default configuration parameters"""
        # Tournament parameters
        self.config.setdefault("parallel_tournaments", 5)
        self.config.setdefault("candidates_per_tournament", 20)
        self.config.setdefault("promotion_threshold", 0.2)  # Top 20% get promoted
        self.config.setdefault("survival_rate", 0.5)        # Bottom 50% get eliminated
        
        # Market scenarios
        self.config.setdefault("market_scenarios", [
            "bull_trend",       # Strong upward trend
            "bear_trend",       # Strong downward trend
            "sideways",         # Range-bound, low volatility
            "high_volatility",  # High volatility, no clear direction
            "sector_rotation",  # Sector rotation, mixed performance
            "market_crisis",    # Sudden market crash
            "recovery",         # Recovery after drawdown
            "normal"            # Normal market conditions
        ])
        
        # Evaluation parameters
        self.config.setdefault("evaluation", {
            "min_trades": 20,        # Minimum number of trades for valid evaluation
            "backtest_days": 120,    # Lookback period for backtests
            "required_metrics": [    # Metrics required for a complete evaluation
                "sharpe_ratio",
                "sortino_ratio",
                "profit_factor",
                "max_drawdown",
                "win_rate",
                "expectancy"
            ],
            "metric_weights": {      # Weights for composite scoring
                "sharpe_ratio": 0.3,
                "sortino_ratio": 0.2,
                "profit_factor": 0.2,
                "max_drawdown": -0.2,
                "win_rate": 0.1
            }
        })
        
        # Tournament history
        self.config.setdefault("history_dir", "data/evotrader/tournament_history")
    
    def run_cycle(self) -> Dict[str, Any]:
        """
        Run a full tournament cycle
        
        This method organizes candidates into tournaments, runs backtests,
        and promotes/eliminates candidates based on results.
        
        Returns:
            Results dictionary
        """
        logger.info("Starting tournament cycle")
        
        # Get candidates from tournament pool
        candidates = list(self.core.tournament_pool.values())
        
        if not candidates:
            logger.warning("No candidates in tournament pool")
            return {
                "error": "No candidates in tournament pool",
                "tournament_id": None,
                "candidates_evaluated": 0
            }
            
        logger.info(f"Tournament pool has {len(candidates)} candidates")
        
        # Generate a tournament ID
        tournament_id = f"tournament_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Split candidates into parallel tournaments
        tournaments = self._organize_tournaments(candidates)
        
        # Run each tournament
        tournament_results = {}
        all_rankings = []
        
        for i, (scenario, tournament_candidates) in enumerate(tournaments):
            logger.info(f"Running tournament {i+1}/{len(tournaments)}: "
                      f"Scenario '{scenario}' with {len(tournament_candidates)} candidates")
            
            # Run tournament
            results = self._run_tournament(tournament_id, i, scenario, tournament_candidates)
            
            # Store results
            tournament_results[f"tournament_{i+1}_{scenario}"] = results
            
            # Add to overall rankings
            for rank_data in results["rankings"]:
                all_rankings.append(rank_data)
        
        # Process results
        promotions, eliminations = self._process_results(all_rankings)
        
        # Save tournament history
        self._save_tournament_history(tournament_id, tournaments, tournament_results)
        
        # Update stats
        self.stats["tournaments_run"] += len(tournaments)
        self.stats["candidates_evaluated"] += len(candidates)
        self.stats["promotions"] += len(promotions)
        self.stats["eliminations"] += len(eliminations)
        
        logger.info(f"Tournament cycle completed: {len(promotions)} promotions, "
                  f"{len(eliminations)} eliminations")
        
        return {
            "tournament_id": tournament_id,
            "tournaments": len(tournaments),
            "candidates_evaluated": len(candidates),
            "promotions": len(promotions),
            "promotion_ids": promotions,
            "eliminations": len(eliminations),
            "elimination_ids": eliminations,
            "results": tournament_results
        }
    
    def _organize_tournaments(self, candidates: List[StrategyCandidate]) -> List[Tuple[str, List[StrategyCandidate]]]:
        """
        Organize candidates into parallel tournaments
        
        Args:
            candidates: List of all candidates
            
        Returns:
            List of (scenario, candidates) tuples for each tournament
        """
        # Determine how many tournaments to run
        num_tournaments = min(self.config["parallel_tournaments"], len(self.config["market_scenarios"]))
        
        # Choose scenarios
        scenarios = random.sample(self.config["market_scenarios"], num_tournaments)
        
        # Organize candidates
        tournaments = []
        
        # Shuffle candidates
        shuffled = random.sample(candidates, len(candidates))
        
        # Distribute candidates across tournaments
        candidates_per_tournament = min(
            self.config["candidates_per_tournament"],
            max(5, len(shuffled) // num_tournaments)  # At least 5 candidates per tournament
        )
        
        # Create tournaments
        for i in range(num_tournaments):
            start_idx = i * candidates_per_tournament
            end_idx = min((i + 1) * candidates_per_tournament, len(shuffled))
            
            # Handle last tournament - include any remaining candidates
            if i == num_tournaments - 1:
                end_idx = len(shuffled)
            
            tournament_candidates = shuffled[start_idx:end_idx]
            
            if tournament_candidates:
                tournaments.append((scenarios[i], tournament_candidates))
        
        return tournaments
    
    def _run_tournament(self, 
                       tournament_id: str, 
                       tournament_num: int, 
                       scenario: str, 
                       candidates: List[StrategyCandidate]) -> Dict[str, Any]:
        """
        Run a single tournament
        
        Args:
            tournament_id: Overall tournament ID
            tournament_num: Tournament number
            scenario: Market scenario to test
            candidates: List of candidates in this tournament
            
        Returns:
            Tournament results
        """
        # In a real implementation, this would use the backtest engine
        # to run backtests for each candidate with the specified scenario.
        # For now, we'll simulate results.
        
        tournament_results = {
            "tournament_id": f"{tournament_id}_t{tournament_num}_{scenario}",
            "scenario": scenario,
            "num_candidates": len(candidates),
            "candidate_ids": [c.id for c in candidates],
            "rankings": [],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Simulate backtest results for each candidate
        for candidate in candidates:
            # Get real backtest results if available
            if candidate.backtest_results and len(candidate.backtest_results) > 0:
                # Use most recent backtest result
                backtest_result = candidate.backtest_results[-1]
            else:
                # Simulate backtest results
                backtest_result = self._simulate_backtest(candidate, scenario)
                
                # Add to candidate's history
                candidate.backtest_results.append(backtest_result)
            
            # Calculate composite score
            composite_score = self._calculate_score(backtest_result["metrics"])
            
            # Add to rankings
            ranking = {
                "candidate_id": candidate.id,
                "candidate_name": candidate.name,
                "strategy_type": candidate.strategy_type,
                "composite_score": composite_score,
                "metrics": backtest_result["metrics"],
                "trades": backtest_result["num_trades"]
            }
            
            tournament_results["rankings"].append(ranking)
        
        # Sort rankings by composite score
        tournament_results["rankings"] = sorted(
            tournament_results["rankings"],
            key=lambda x: x["composite_score"],
            reverse=True  # Higher is better
        )
        
        return tournament_results
    
    def _simulate_backtest(self, candidate: StrategyCandidate, scenario: str) -> Dict[str, Any]:
        """
        Simulate backtest results for a candidate
        
        Args:
            candidate: Strategy candidate
            scenario: Market scenario
            
        Returns:
            Simulated backtest results
        """
        # Random baseline values
        baseline = {
            "sharpe_ratio": random.uniform(-0.5, 2.0),
            "sortino_ratio": random.uniform(-0.5, 2.5),
            "profit_factor": random.uniform(0.7, 1.5),
            "max_drawdown": random.uniform(-0.3, -0.05),
            "win_rate": random.uniform(0.3, 0.6),
            "expectancy": random.uniform(-0.1, 0.3),
            "profit": random.uniform(-5, 10),
            "roi": random.uniform(-0.1, 0.2)
        }
        
        # Adjust based on strategy type and scenario
        strategy_factor = 1.0
        scenario_factor = 1.0
        
        # Strategy type adjustments
        if candidate.strategy_type == "momentum":
            if scenario in ["bull_trend", "recovery"]:
                strategy_factor = 1.3
            elif scenario in ["bear_trend", "market_crisis"]:
                strategy_factor = 0.7
                
        elif candidate.strategy_type == "mean_reversion":
            if scenario in ["sideways", "high_volatility"]:
                strategy_factor = 1.3
            elif scenario in ["bull_trend", "bear_trend"]:
                strategy_factor = 0.8
                
        elif candidate.strategy_type == "trend_following":
            if scenario in ["bull_trend", "bear_trend"]:
                strategy_factor = 1.4
            elif scenario in ["sideways"]:
                strategy_factor = 0.6
                
        elif candidate.strategy_type == "breakout":
            if scenario in ["high_volatility", "recovery"]:
                strategy_factor = 1.3
            elif scenario in ["sideways"]:
                strategy_factor = 0.7
                
        elif candidate.strategy_type == "volatility":
            if scenario in ["high_volatility", "market_crisis"]:
                strategy_factor = 1.5
            elif scenario in ["sideways"]:
                strategy_factor = 0.8
                
        elif candidate.strategy_type == "ml_enhanced":
            # ML strategies adapt better across scenarios
            strategy_factor = 1.1
                
        # Scenario adjustments
        if scenario == "market_crisis":
            # Most strategies struggle in crisis
            scenario_factor = 0.6
        elif scenario == "high_volatility":
            # High vol can be challenging
            scenario_factor = 0.8
            
        # Generation adjustment - advanced generations perform better
        generation_factor = min(1.0 + candidate.generation * 0.01, 1.2)
        
        # Apply adjustments
        metrics = {}
        for key, value in baseline.items():
            # Apply factors
            adjusted = value * strategy_factor * scenario_factor * generation_factor
            
            # Add some randomness
            noise = random.uniform(0.9, 1.1)
            metrics[key] = adjusted * noise
            
            # Clamp to reasonable ranges
            if key == "max_drawdown":
                metrics[key] = max(min(metrics[key], -0.02), -0.4)
            elif key in ["sharpe_ratio", "sortino_ratio"]:
                metrics[key] = max(min(metrics[key], 4.0), -1.0)
            elif key == "profit_factor":
                metrics[key] = max(metrics[key], 0.1)
            elif key == "win_rate":
                metrics[key] = max(min(metrics[key], 0.8), 0.2)
        
        # Generate trade count
        num_trades = random.randint(20, 120)
        
        # Collect results
        backtest_result = {
            "id": f"backtest_{uuid.uuid4().hex[:8]}",
            "candidate_id": candidate.id,
            "scenario": scenario,
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": metrics,
            "num_trades": num_trades,
            "duration_days": self.config["evaluation"]["backtest_days"]
        }
        
        return backtest_result
    
    def _calculate_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate a composite score from backtest metrics
        
        Args:
            metrics: Dictionary of backtest metrics
            
        Returns:
            Composite score
        """
        weights = self.config["evaluation"]["metric_weights"]
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
                
        return score
    
    def _process_results(self, all_rankings: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        Process tournament results to promote and eliminate candidates
        
        Args:
            all_rankings: List of ranking results from all tournaments
            
        Returns:
            Tuple of (promoted_ids, eliminated_ids)
        """
        # Group rankings by candidate
        candidate_scores = defaultdict(list)
        
        for ranking in all_rankings:
            candidate_id = ranking["candidate_id"]
            candidate_scores[candidate_id].append(ranking["composite_score"])
        
        # Calculate average score for each candidate
        average_scores = {}
        for candidate_id, scores in candidate_scores.items():
            average_scores[candidate_id] = sum(scores) / len(scores)
        
        # Sort candidates by average score
        sorted_candidates = sorted(
            average_scores.items(),
            key=lambda x: x[1],
            reverse=True  # Higher is better
        )
        
        # Get candidates
        candidate_ids = [c[0] for c in sorted_candidates]
        
        # Determine how many to promote and eliminate
        num_candidates = len(candidate_ids)
        num_promote = max(1, int(num_candidates * self.config["promotion_threshold"]))
        num_eliminate = max(0, int(num_candidates * (1 - self.config["survival_rate"])))
        
        # Get promotion and elimination lists
        promotions = candidate_ids[:num_promote]
        eliminations = candidate_ids[-num_eliminate:] if num_eliminate > 0 else []
        
        # Update candidates
        for candidate_id in promotions:
            candidate = self.core.get_candidate_by_id(candidate_id)
            if candidate:
                # Update fitness scores with latest metrics
                latest_scores = {}
                for ranking in all_rankings:
                    if ranking["candidate_id"] == candidate_id:
                        metrics = ranking["metrics"]
                        for metric, value in metrics.items():
                            # Average if multiple tournament results
                            if metric in latest_scores:
                                latest_scores[metric] = (latest_scores[metric] + value) / 2
                            else:
                                latest_scores[metric] = value
                
                candidate.fitness_scores.update(latest_scores)
                
                # Store tournament results
                tournament_result = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "average_score": average_scores[candidate_id],
                    "promoted": True,
                    "metrics": latest_scores
                }
                candidate.tournament_results.append(tournament_result)
                
                # Promote to paper trading
                self.core.promote_to_paper_trading(candidate_id)
        
        # Handle eliminations
        for candidate_id in eliminations:
            candidate = self.core.get_candidate_by_id(candidate_id)
            if candidate:
                # Store tournament results
                tournament_result = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "average_score": average_scores[candidate_id],
                    "promoted": False,
                    "eliminated": True
                }
                candidate.tournament_results.append(tournament_result)
                
                # Retire the candidate
                self.core.retire_from_production(candidate_id)
        
        return promotions, eliminations
    
    def _save_tournament_history(self, 
                                tournament_id: str, 
                                tournaments: List[Tuple[str, List[StrategyCandidate]]], 
                                results: Dict[str, Any]) -> bool:
        """
        Save tournament history to disk
        
        Args:
            tournament_id: Tournament ID
            tournaments: List of tournament scenarios and candidates
            results: Tournament results
            
        Returns:
            Success flag
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.config["history_dir"], exist_ok=True)
            
            # Create history record
            history = {
                "tournament_id": tournament_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "scenarios": [t[0] for t in tournaments],
                "candidate_counts": [len(t[1]) for t in tournaments],
                "total_candidates": sum(len(t[1]) for t in tournaments),
                "results": results
            }
            
            # Save as JSON
            filepath = os.path.join(
                self.config["history_dir"], 
                f"{tournament_id}.json"
            )
            
            with open(filepath, 'w') as f:
                json.dump(history, f, indent=2)
            
            logger.info(f"Tournament history saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving tournament history: {e}")
            return False


# Factory function
def create_tournament_engine(core, config: Dict[str, Any] = None) -> TournamentEngine:
    """
    Create a tournament engine instance
    
    Args:
        core: Reference to EvoTrader core
        config: Optional configuration dictionary
        
    Returns:
        Initialized tournament engine
    """
    return TournamentEngine(core, config)
