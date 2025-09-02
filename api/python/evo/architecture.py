#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EvoTrader Architecture

This module defines the core architecture for the EvoTrader system,
which automates the full lifecycle of trading strategies using
genetic algorithms, reinforcement learning, and statistical validation.
"""

import os
import sys
import json
import logging
import datetime
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class StrategyStage(Enum):
    """Enumeration of strategy lifecycle stages"""
    CANDIDATE = "candidate"  # Initial candidate generation
    TOURNAMENT = "tournament"  # Competing in tournaments
    PAPER_TRADING = "paper_trading"  # Paper trading validation
    PRODUCTION = "production"  # Live trading
    RETIRED = "retired"  # Retired from production

class FitnessMetric(Enum):
    """Enumeration of fitness metrics for evaluating strategies"""
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"  
    CALMAR_RATIO = "calmar_ratio"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    MAX_DRAWDOWN = "max_drawdown"
    PROFIT = "profit"
    ROI = "roi"
    EXPECTANCY = "expectancy"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"

@dataclass
class StrategyCandidate:
    """Representation of a strategy candidate in the EvoTrader system"""
    # Identity
    id: str
    name: str
    version: str
    parent_ids: List[str] = field(default_factory=list)
    
    # Strategy type and configuration  
    strategy_type: str
    parameters: Dict[str, Any]
    
    # Evolutionary metadata
    generation: int = 0
    stage: StrategyStage = StrategyStage.CANDIDATE
    birth_date: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    # Performance metrics
    fitness_scores: Dict[str, float] = field(default_factory=dict)
    tournament_results: List[Dict[str, Any]] = field(default_factory=list)
    backtest_results: List[Dict[str, Any]] = field(default_factory=list)
    paper_trading_results: Dict[str, Any] = field(default_factory=dict)
    production_results: Dict[str, Any] = field(default_factory=dict)
    
    # Genetic metadata
    mutation_rate: float = 0.05
    crossover_points: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the candidate to a dictionary"""
        result = {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "parent_ids": self.parent_ids,
            "strategy_type": self.strategy_type,
            "parameters": self.parameters,
            "generation": self.generation,
            "stage": self.stage.value,
            "birth_date": self.birth_date.isoformat(),
            "fitness_scores": self.fitness_scores,
            "tournament_results": self.tournament_results,
            "backtest_results": self.backtest_results,
            "paper_trading_results": self.paper_trading_results,
            "production_results": self.production_results,
            "mutation_rate": self.mutation_rate,
            "crossover_points": self.crossover_points
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyCandidate':
        """Create a candidate from a dictionary"""
        # Handle datetime conversion
        if isinstance(data.get("birth_date"), str):
            data["birth_date"] = datetime.datetime.fromisoformat(data["birth_date"])
        
        # Handle enum conversion
        if isinstance(data.get("stage"), str):
            data["stage"] = StrategyStage(data["stage"])
            
        return cls(**data)
    
    def calculate_composite_fitness(self, weights: Dict[str, float] = None) -> float:
        """
        Calculate a composite fitness score based on weighted metrics
        
        Args:
            weights: Dictionary of metric names to weights
            
        Returns:
            Composite fitness score
        """
        if not weights:
            weights = {
                "sharpe_ratio": 0.3,
                "sortino_ratio": 0.2,
                "profit_factor": 0.2, 
                "max_drawdown": -0.2,
                "win_rate": 0.1
            }
            
        score = 0.0
        for metric, weight in weights.items():
            if metric in self.fitness_scores:
                score += self.fitness_scores[metric] * weight
                
        return score


class EvoTraderCore:
    """
    Core component of the EvoTrader system, responsible for managing
    the overall strategy lifecycle.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the EvoTrader core
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._set_default_config()
        
        # Strategy registry
        self.candidates: Dict[str, StrategyCandidate] = {}
        self.tournament_pool: Dict[str, StrategyCandidate] = {}
        self.paper_trading_pool: Dict[str, StrategyCandidate] = {}
        self.production_pool: Dict[str, StrategyCandidate] = {}
        self.retired_pool: Dict[str, StrategyCandidate] = {}
        
        # Component references
        self.genetic_engine = None
        self.tournament_engine = None
        self.statistical_validator = None
        self.strategy_factory = None
        
        logger.info("EvoTrader core initialized")
    
    def _set_default_config(self):
        """Set default configuration parameters"""
        # System paths
        self.config.setdefault("data_dir", "data/evotrader")
        self.config.setdefault("candidates_dir", "data/evotrader/candidates")
        self.config.setdefault("tournament_results_dir", "data/evotrader/tournament_results")
        
        # Genetic parameters
        self.config.setdefault("genetic", {
            "population_size": 100,
            "tournament_size": 20,
            "elite_count": 5,
            "mutation_rate": 0.05,
            "crossover_rate": 0.7,
            "generations": 50
        })
        
        # Tournament parameters
        self.config.setdefault("tournament", {
            "parallel_tournaments": 5,
            "candidates_per_tournament": 20,
            "promotion_threshold": 0.2,  # Top 20% get promoted
            "survival_rate": 0.5         # Bottom 50% get eliminated
        })
        
        # Statistical validation
        self.config.setdefault("validation", {
            "min_sharpe_ratio": 0.8,
            "min_profit_factor": 1.2, 
            "max_drawdown": -0.20,
            "min_trades": 30,
            "min_win_rate": 0.45,
            "p_value_threshold": 0.05,
            "out_of_sample_ratio": 0.3,
            "walk_forward_windows": 5
        })
        
        # Production promotion
        self.config.setdefault("promotion", {
            "paper_trading_days": 30,
            "min_paper_sharpe": 0.7,
            "max_paper_drawdown": -0.15,
            "consistency_threshold": 0.7  # Backtest vs paper trading consistency
        })
    
    def register_candidate(self, candidate: StrategyCandidate) -> str:
        """
        Register a new strategy candidate
        
        Args:
            candidate: Strategy candidate to register
            
        Returns:
            Candidate ID
        """
        # Add to candidates registry
        self.candidates[candidate.id] = candidate
        
        # Save candidate to filesystem
        self._save_candidate(candidate)
        
        logger.info(f"Registered candidate {candidate.id}: {candidate.name} (v{candidate.version})")
        return candidate.id
    
    def promote_to_tournament(self, candidate_id: str) -> bool:
        """
        Promote a candidate to the tournament stage
        
        Args:
            candidate_id: ID of candidate to promote
            
        Returns:
            Success flag
        """
        if candidate_id not in self.candidates:
            logger.error(f"Cannot promote unknown candidate {candidate_id}")
            return False
        
        candidate = self.candidates[candidate_id]
        candidate.stage = StrategyStage.TOURNAMENT
        self.tournament_pool[candidate_id] = candidate
        
        logger.info(f"Promoted candidate {candidate_id} to tournament stage")
        return True
    
    def promote_to_paper_trading(self, candidate_id: str) -> bool:
        """
        Promote a candidate to paper trading
        
        Args:
            candidate_id: ID of candidate to promote
            
        Returns:
            Success flag
        """
        if candidate_id not in self.tournament_pool:
            logger.error(f"Cannot promote unknown tournament candidate {candidate_id}")
            return False
        
        candidate = self.tournament_pool[candidate_id]
        candidate.stage = StrategyStage.PAPER_TRADING
        self.paper_trading_pool[candidate_id] = candidate
        
        logger.info(f"Promoted candidate {candidate_id} to paper trading stage")
        return True
    
    def promote_to_production(self, candidate_id: str) -> bool:
        """
        Promote a candidate to production
        
        Args:
            candidate_id: ID of candidate to promote
            
        Returns:
            Success flag
        """
        if candidate_id not in self.paper_trading_pool:
            logger.error(f"Cannot promote unknown paper trading candidate {candidate_id}")
            return False
        
        candidate = self.paper_trading_pool[candidate_id]
        candidate.stage = StrategyStage.PRODUCTION
        self.production_pool[candidate_id] = candidate
        
        logger.info(f"Promoted candidate {candidate_id} to production stage")
        return True
    
    def retire_from_production(self, candidate_id: str) -> bool:
        """
        Retire a candidate from production
        
        Args:
            candidate_id: ID of candidate to retire
            
        Returns:
            Success flag
        """
        if candidate_id not in self.production_pool:
            logger.error(f"Cannot retire unknown production candidate {candidate_id}")
            return False
        
        candidate = self.production_pool[candidate_id]
        candidate.stage = StrategyStage.RETIRED
        self.retired_pool[candidate_id] = candidate
        
        logger.info(f"Retired candidate {candidate_id} from production")
        return True
    
    def _save_candidate(self, candidate: StrategyCandidate) -> bool:
        """
        Save a candidate to the filesystem
        
        Args:
            candidate: Candidate to save
            
        Returns:
            Success flag
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.config["candidates_dir"], exist_ok=True)
            
            # Save as JSON
            filepath = os.path.join(
                self.config["candidates_dir"], 
                f"{candidate.id}_{candidate.version}.json"
            )
            
            with open(filepath, 'w') as f:
                json.dump(candidate.to_dict(), f, indent=2)
                
            return True
            
        except Exception as e:
            logger.error(f"Error saving candidate {candidate.id}: {e}")
            return False
    
    def load_candidate(self, candidate_id: str, version: str = None) -> Optional[StrategyCandidate]:
        """
        Load a candidate from the filesystem
        
        Args:
            candidate_id: Candidate ID
            version: Optional specific version to load
            
        Returns:
            Loaded candidate or None if not found
        """
        try:
            # Find candidate file
            candidate_dir = self.config["candidates_dir"]
            
            if version:
                # Load specific version
                filepath = os.path.join(candidate_dir, f"{candidate_id}_{version}.json")
                if not os.path.exists(filepath):
                    logger.error(f"Candidate {candidate_id} version {version} not found")
                    return None
            else:
                # Find latest version
                files = [f for f in os.listdir(candidate_dir) 
                        if f.startswith(f"{candidate_id}_") and f.endswith(".json")]
                
                if not files:
                    logger.error(f"No versions found for candidate {candidate_id}")
                    return None
                
                # Sort by version (assuming semantic versioning)
                files.sort(reverse=True)
                filepath = os.path.join(candidate_dir, files[0])
            
            # Load JSON
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Create candidate
            candidate = StrategyCandidate.from_dict(data)
            
            # Add to appropriate pool based on stage
            if candidate.stage == StrategyStage.CANDIDATE:
                self.candidates[candidate.id] = candidate
            elif candidate.stage == StrategyStage.TOURNAMENT:
                self.tournament_pool[candidate.id] = candidate
            elif candidate.stage == StrategyStage.PAPER_TRADING:
                self.paper_trading_pool[candidate.id] = candidate
            elif candidate.stage == StrategyStage.PRODUCTION:
                self.production_pool[candidate.id] = candidate
            elif candidate.stage == StrategyStage.RETIRED:
                self.retired_pool[candidate.id] = candidate
            
            return candidate
            
        except Exception as e:
            logger.error(f"Error loading candidate {candidate_id}: {e}")
            return None
    
    def get_all_candidates(self) -> Dict[str, List[StrategyCandidate]]:
        """
        Get all candidates organized by stage
        
        Returns:
            Dictionary of candidates by stage
        """
        return {
            "candidates": list(self.candidates.values()),
            "tournament": list(self.tournament_pool.values()),
            "paper_trading": list(self.paper_trading_pool.values()),
            "production": list(self.production_pool.values()),
            "retired": list(self.retired_pool.values())
        }
    
    def get_candidate_by_id(self, candidate_id: str) -> Optional[StrategyCandidate]:
        """
        Get a candidate by ID from any pool
        
        Args:
            candidate_id: Candidate ID
            
        Returns:
            Candidate or None if not found
        """
        # Check all pools
        if candidate_id in self.candidates:
            return self.candidates[candidate_id]
        elif candidate_id in self.tournament_pool:
            return self.tournament_pool[candidate_id]
        elif candidate_id in self.paper_trading_pool:
            return self.paper_trading_pool[candidate_id]
        elif candidate_id in self.production_pool:
            return self.production_pool[candidate_id]
        elif candidate_id in self.retired_pool:
            return self.retired_pool[candidate_id]
        else:
            return None
    
    def run_genetic_cycle(self) -> Dict[str, Any]:
        """
        Run a full genetic cycle to evolve strategies
        
        Returns:
            Results dictionary
        """
        if not self.genetic_engine:
            logger.error("Genetic engine not initialized")
            return {"error": "Genetic engine not initialized"}
        
        # Run genetic cycle
        return self.genetic_engine.run_cycle()
    
    def run_tournament_cycle(self) -> Dict[str, Any]:
        """
        Run a full tournament cycle to evaluate strategies
        
        Returns:
            Results dictionary
        """
        if not self.tournament_engine:
            logger.error("Tournament engine not initialized")
            return {"error": "Tournament engine not initialized"}
        
        # Run tournament cycle
        return self.tournament_engine.run_cycle()
    
    def run_validation_cycle(self) -> Dict[str, Any]:
        """
        Run a full validation cycle to validate strategies
        
        Returns:
            Results dictionary
        """
        if not self.statistical_validator:
            logger.error("Statistical validator not initialized")
            return {"error": "Statistical validator not initialized"}
        
        # Run validation cycle
        return self.statistical_validator.run_cycle()


# Factory function to create a strategy candidate
def create_strategy_candidate(
    name: str,
    strategy_type: str,
    parameters: Dict[str, Any],
    parent_ids: List[str] = None
) -> StrategyCandidate:
    """
    Create a new strategy candidate
    
    Args:
        name: Strategy name
        strategy_type: Type of strategy
        parameters: Strategy parameters
        parent_ids: Optional IDs of parent strategies
        
    Returns:
        New strategy candidate
    """
    # Generate a unique ID
    candidate_id = f"strat_{uuid.uuid4().hex[:8]}"
    
    # Create candidate
    candidate = StrategyCandidate(
        id=candidate_id,
        name=name,
        version="1.0.0",
        parent_ids=parent_ids or [],
        strategy_type=strategy_type,
        parameters=parameters,
        generation=0 if not parent_ids else 1
    )
    
    return candidate


# EvoTrader factory function
def create_evotrader_core(config: Dict[str, Any] = None) -> EvoTraderCore:
    """
    Create an EvoTrader core instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized EvoTrader core
    """
    return EvoTraderCore(config)
