#!/usr/bin/env python3
"""
Strategy DNA Mapping System

This module provides a framework for classifying trading strategies based on
their characteristics, creating a standardized representation that can be used
for strategy selection and rotation.
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional, Union, Set
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

class StrategyDNA:
    """
    Represents the DNA (characteristics) of a trading strategy.
    
    This class encapsulates the core attributes and behavior patterns of a
    trading strategy, creating a standardized representation that can be used
    for classification and comparison.
    """
    
    # Core DNA attributes that define a strategy
    DNA_ATTRIBUTES = [
        # Market conditions the strategy performs well in
        "market_regime_affinity",  # e.g., ['bullish', 'neutral', 'volatile']
        "volatility_preference",   # 'high', 'medium', 'low'
        "trend_requirement",       # 'strong', 'moderate', 'any'
        
        # Trading characteristics
        "time_frame",              # 'intraday', 'swing', 'position', 'multi_timeframe'
        "signal_frequency",        # 'high', 'medium', 'low'
        "direction",               # 'long', 'short', 'both'
        
        # Risk and performance characteristics
        "win_rate_potential",      # 'high', 'medium', 'low'
        "profit_factor_potential", # 'high', 'medium', 'low'
        "drawdown_potential",      # 'high', 'medium', 'low'
        
        # Behavioral characteristics
        "mean_reversion_tendency", # 0-10 scale
        "momentum_tendency",       # 0-10 scale
        "breakout_tendency",       # 0-10 scale
        "counter_trend_tendency",  # 0-10 scale
        
        # Implementation details
        "complexity",              # 'high', 'medium', 'low'
        "automation_level",        # 'fully_automated', 'semi_automated', 'discretionary'
        "capital_requirement",     # 'high', 'medium', 'low'
        
        # Asset class suitability
        "asset_class_suitability", # list of asset classes the strategy works well with
                                   # e.g., ['stocks', 'futures', 'forex', 'options']
    ]
    
    def __init__(self, strategy_id: str, name: str, description: str, dna_data: Dict[str, Any]):
        """
        Initialize a StrategyDNA instance.
        
        Args:
            strategy_id: Unique identifier for the strategy
            name: Human-readable name for the strategy
            description: Detailed description of the strategy
            dna_data: Dictionary containing the DNA attributes
        """
        self.strategy_id = strategy_id
        self.name = name
        self.description = description
        self.dna = {}
        
        # Process and validate DNA data
        self._process_dna_data(dna_data)
        
    def _process_dna_data(self, dna_data: Dict[str, Any]) -> None:
        """
        Process and validate DNA data.
        
        Args:
            dna_data: Dictionary containing the DNA attributes
        """
        # Copy valid attributes
        for attr in self.DNA_ATTRIBUTES:
            if attr in dna_data:
                self.dna[attr] = dna_data[attr]
            else:
                logger.warning(f"Missing DNA attribute '{attr}' for strategy '{self.name}'")
                # Use sensible defaults based on attribute type
                if attr.endswith('_tendency'):
                    self.dna[attr] = 5  # Middle of 0-10 scale
                elif attr == 'asset_class_suitability':
                    self.dna[attr] = ['stocks']  # Default to stocks
                elif attr in ['market_regime_affinity']:
                    self.dna[attr] = ['neutral']  # Default to neutral
                else:
                    self.dna[attr] = 'medium'  # Default to medium for most attributes
        
        # Store additional attributes that weren't in the standard list
        for key, value in dna_data.items():
            if key not in self.DNA_ATTRIBUTES:
                self.dna[key] = value
                
    def match_market_regime(self, regime: str) -> float:
        """
        Calculate how well this strategy matches a given market regime.
        
        Args:
            regime: Market regime to match against (e.g., 'bullish', 'bearish')
            
        Returns:
            Match score between 0.0 and 1.0
        """
        if regime in self.dna.get('market_regime_affinity', []):
            return 1.0
        return 0.0
        
    def match_conditions(self, conditions: Dict[str, Any]) -> float:
        """
        Calculate how well this strategy matches a set of market conditions.
        
        Args:
            conditions: Dictionary of market conditions
            
        Returns:
            Match score between 0.0 and 1.0
        """
        score = 0.0
        total_factors = 0
        
        # Match regime
        if 'regime' in conditions:
            regime_score = self.match_market_regime(conditions['regime'])
            score += regime_score
            total_factors += 1
            
        # Match volatility
        if 'volatility' in conditions:
            vol_match = 0.0
            if conditions['volatility'] == 'high' and self.dna.get('volatility_preference') == 'high':
                vol_match = 1.0
            elif conditions['volatility'] == 'medium' and self.dna.get('volatility_preference') in ['medium', 'high']:
                vol_match = 0.8 if self.dna.get('volatility_preference') == 'medium' else 0.5
            elif conditions['volatility'] == 'low' and self.dna.get('volatility_preference') in ['low', 'medium']:
                vol_match = 0.8 if self.dna.get('volatility_preference') == 'low' else 0.5
            score += vol_match
            total_factors += 1
            
        # Match trend strength
        if 'trend_strength' in conditions:
            trend_match = 0.0
            if conditions['trend_strength'] == 'strong' and self.dna.get('trend_requirement') in ['strong', 'moderate']:
                trend_match = 1.0 if self.dna.get('trend_requirement') == 'strong' else 0.7
            elif conditions['trend_strength'] == 'moderate' and self.dna.get('trend_requirement') in ['moderate', 'any']:
                trend_match = 0.8
            elif conditions['trend_strength'] == 'weak' and self.dna.get('trend_requirement') == 'any':
                trend_match = 0.6
            score += trend_match
            total_factors += 1
            
        # Calculate average score
        return score / total_factors if total_factors > 0 else 0.0
        
    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the StrategyDNA to a dictionary.
        
        Returns:
            Dictionary representation of the StrategyDNA
        """
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "description": self.description,
            "dna": self.dna
        }
        
    def __str__(self) -> str:
        """Return string representation of the StrategyDNA."""
        return f"StrategyDNA(id={self.strategy_id}, name={self.name})"


class StrategyLibrary:
    """
    Manages a collection of trading strategies and their DNA profiles.
    
    This class serves as a repository for strategy DNA profiles, providing
    methods for adding, retrieving, and matching strategies based on
    market conditions.
    """
    
    def __init__(self, library_path: Optional[str] = None):
        """
        Initialize the strategy library.
        
        Args:
            library_path: Path to the JSON file containing strategy definitions
        """
        self.strategies: Dict[str, StrategyDNA] = {}
        
        if library_path and os.path.exists(library_path):
            self.load_from_file(library_path)
            
    def add_strategy(self, strategy: StrategyDNA) -> None:
        """
        Add a strategy to the library.
        
        Args:
            strategy: StrategyDNA instance to add
        """
        self.strategies[strategy.strategy_id] = strategy
        logger.info(f"Added strategy '{strategy.name}' to library")
        
    def get_strategy(self, strategy_id: str) -> Optional[StrategyDNA]:
        """
        Get a strategy by ID.
        
        Args:
            strategy_id: ID of the strategy to retrieve
            
        Returns:
            StrategyDNA instance if found, None otherwise
        """
        return self.strategies.get(strategy_id)
        
    def get_all_strategies(self) -> List[StrategyDNA]:
        """
        Get all strategies in the library.
        
        Returns:
            List of all StrategyDNA instances
        """
        return list(self.strategies.values())
        
    def match_market_conditions(self, conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Match strategies against market conditions.
        
        Args:
            conditions: Dictionary of market conditions
            
        Returns:
            List of dictionaries with strategy and match score, sorted by score
        """
        matches = []
        
        for strategy_id, strategy in self.strategies.items():
            score = strategy.match_conditions(conditions)
            matches.append({
                "strategy_id": strategy_id,
                "name": strategy.name,
                "score": score
            })
            
        # Sort by score in descending order
        return sorted(matches, key=lambda x: x["score"], reverse=True)
        
    def save_to_file(self, file_path: str) -> None:
        """
        Save the strategy library to a JSON file.
        
        Args:
            file_path: Path to save the library to
        """
        data = {
            "strategies": [strategy.as_dict() for strategy in self.strategies.values()]
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved strategy library to {file_path}")
        
    def load_from_file(self, file_path: str) -> None:
        """
        Load strategies from a JSON file.
        
        Args:
            file_path: Path to the JSON file to load
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            for strategy_data in data.get("strategies", []):
                strategy = StrategyDNA(
                    strategy_id=strategy_data["strategy_id"],
                    name=strategy_data["name"],
                    description=strategy_data["description"],
                    dna_data=strategy_data["dna"]
                )
                self.add_strategy(strategy)
                
            logger.info(f"Loaded {len(self.strategies)} strategies from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading strategy library from {file_path}: {e}")
            

def create_default_library() -> StrategyLibrary:
    """
    Create a default strategy library with common trading strategies.
    
    Returns:
        StrategyLibrary with pre-defined strategies
    """
    library = StrategyLibrary()
    
    # Define common trading strategies
    
    # 1. Trend Following
    trend_following = StrategyDNA(
        strategy_id="trend_following",
        name="Trend Following",
        description="Follows established market trends using momentum indicators and moving averages.",
        dna_data={
            "market_regime_affinity": ["bullish", "bearish"],
            "volatility_preference": "medium",
            "trend_requirement": "strong",
            "time_frame": "position",
            "signal_frequency": "low",
            "direction": "both",
            "win_rate_potential": "medium",
            "profit_factor_potential": "high",
            "drawdown_potential": "medium",
            "mean_reversion_tendency": 2,
            "momentum_tendency": 9,
            "breakout_tendency": 6,
            "counter_trend_tendency": 1,
            "complexity": "low",
            "automation_level": "fully_automated",
            "capital_requirement": "medium",
            "asset_class_suitability": ["stocks", "futures", "forex", "etfs"]
        }
    )
    library.add_strategy(trend_following)
    
    # 2. Mean Reversion
    mean_reversion = StrategyDNA(
        strategy_id="mean_reversion",
        name="Mean Reversion",
        description="Capitalizes on price movements returning to an average or mean value.",
        dna_data={
            "market_regime_affinity": ["neutral", "volatile"],
            "volatility_preference": "high",
            "trend_requirement": "any",
            "time_frame": "swing",
            "signal_frequency": "medium",
            "direction": "both",
            "win_rate_potential": "high",
            "profit_factor_potential": "medium",
            "drawdown_potential": "medium",
            "mean_reversion_tendency": 9,
            "momentum_tendency": 2,
            "breakout_tendency": 3,
            "counter_trend_tendency": 8,
            "complexity": "medium",
            "automation_level": "fully_automated",
            "capital_requirement": "medium",
            "asset_class_suitability": ["stocks", "etfs", "options"]
        }
    )
    library.add_strategy(mean_reversion)
    
    # 3. Momentum
    momentum = StrategyDNA(
        strategy_id="momentum",
        name="Momentum",
        description="Trades assets showing strong price and volume momentum relative to the market.",
        dna_data={
            "market_regime_affinity": ["bullish", "bearish"],
            "volatility_preference": "medium",
            "trend_requirement": "moderate",
            "time_frame": "swing",
            "signal_frequency": "medium",
            "direction": "both",
            "win_rate_potential": "medium",
            "profit_factor_potential": "high",
            "drawdown_potential": "medium",
            "mean_reversion_tendency": 3,
            "momentum_tendency": 9,
            "breakout_tendency": 7,
            "counter_trend_tendency": 2,
            "complexity": "medium",
            "automation_level": "fully_automated",
            "capital_requirement": "medium",
            "asset_class_suitability": ["stocks", "etfs", "futures"]
        }
    )
    library.add_strategy(momentum)
    
    # 4. Breakout
    breakout_swing = StrategyDNA(
        strategy_id="breakout_swing",
        name="Breakout Swing",
        description="Identifies and trades breakouts from chart patterns or key levels.",
        dna_data={
            "market_regime_affinity": ["bullish", "bearish", "neutral"],
            "volatility_preference": "low",
            "trend_requirement": "any",
            "time_frame": "swing",
            "signal_frequency": "medium",
            "direction": "both",
            "win_rate_potential": "medium",
            "profit_factor_potential": "high",
            "drawdown_potential": "medium",
            "mean_reversion_tendency": 3,
            "momentum_tendency": 7,
            "breakout_tendency": 9,
            "counter_trend_tendency": 3,
            "complexity": "medium",
            "automation_level": "semi_automated",
            "capital_requirement": "medium",
            "asset_class_suitability": ["stocks", "futures", "forex"]
        }
    )
    library.add_strategy(breakout_swing)
    
    # 5. Volatility Breakout
    volatility_breakout = StrategyDNA(
        strategy_id="volatility_breakout",
        name="Volatility Breakout",
        description="Capitalizes on sudden increases in volatility and price movement.",
        dna_data={
            "market_regime_affinity": ["volatile"],
            "volatility_preference": "high",
            "trend_requirement": "any",
            "time_frame": "intraday",
            "signal_frequency": "high",
            "direction": "both",
            "win_rate_potential": "medium",
            "profit_factor_potential": "high",
            "drawdown_potential": "high",
            "mean_reversion_tendency": 4,
            "momentum_tendency": 6,
            "breakout_tendency": 9,
            "counter_trend_tendency": 5,
            "complexity": "high",
            "automation_level": "fully_automated",
            "capital_requirement": "medium",
            "asset_class_suitability": ["futures", "options", "forex"]
        }
    )
    library.add_strategy(volatility_breakout)
    
    # 6. Option Spreads
    option_spreads = StrategyDNA(
        strategy_id="option_spreads",
        name="Option Spreads",
        description="Trades various option spread strategies based on market outlook.",
        dna_data={
            "market_regime_affinity": ["bullish", "bearish", "neutral", "volatile"],
            "volatility_preference": "high",
            "trend_requirement": "any",
            "time_frame": "position",
            "signal_frequency": "low",
            "direction": "both",
            "win_rate_potential": "high",
            "profit_factor_potential": "medium",
            "drawdown_potential": "low",
            "mean_reversion_tendency": 6,
            "momentum_tendency": 5,
            "breakout_tendency": 4,
            "counter_trend_tendency": 6,
            "complexity": "high",
            "automation_level": "semi_automated",
            "capital_requirement": "high",
            "asset_class_suitability": ["options"]
        }
    )
    library.add_strategy(option_spreads)

    # 7. Gap Trading
    gap_trading = StrategyDNA(
        strategy_id="gap_trading",
        name="Gap Trading",
        description="Trades price gaps in the market direction or fades them depending on context.",
        dna_data={
            "market_regime_affinity": ["volatile", "bullish", "bearish"],
            "volatility_preference": "high",
            "trend_requirement": "any",
            "time_frame": "intraday",
            "signal_frequency": "high",
            "direction": "both",
            "win_rate_potential": "medium",
            "profit_factor_potential": "medium",
            "drawdown_potential": "medium",
            "mean_reversion_tendency": 7,
            "momentum_tendency": 6,
            "breakout_tendency": 7,
            "counter_trend_tendency": 6,
            "complexity": "medium",
            "automation_level": "fully_automated",
            "capital_requirement": "medium",
            "asset_class_suitability": ["stocks", "futures", "etfs"]
        }
    )
    library.add_strategy(gap_trading)
    
    # 8. Pairs Trading
    pairs_trading = StrategyDNA(
        strategy_id="pairs_trading",
        name="Pairs Trading",
        description="Market neutral strategy that trades correlated securities when their relative prices diverge.",
        dna_data={
            "market_regime_affinity": ["neutral", "volatile", "bearish"],
            "volatility_preference": "medium",
            "trend_requirement": "any",
            "time_frame": "swing",
            "signal_frequency": "low",
            "direction": "both",
            "win_rate_potential": "high",
            "profit_factor_potential": "medium",
            "drawdown_potential": "low",
            "mean_reversion_tendency": 9,
            "momentum_tendency": 2,
            "breakout_tendency": 2,
            "counter_trend_tendency": 7,
            "complexity": "high",
            "automation_level": "fully_automated",
            "capital_requirement": "high",
            "asset_class_suitability": ["stocks", "etfs"]
        }
    )
    library.add_strategy(pairs_trading)
    
    # 9. Swing Trading
    swing_trading = StrategyDNA(
        strategy_id="swing_trading",
        name="Swing Trading",
        description="Captures medium-term price moves using a combination of technical indicators.",
        dna_data={
            "market_regime_affinity": ["bullish", "bearish", "neutral"],
            "volatility_preference": "medium",
            "trend_requirement": "moderate",
            "time_frame": "swing",
            "signal_frequency": "medium",
            "direction": "both",
            "win_rate_potential": "medium",
            "profit_factor_potential": "medium",
            "drawdown_potential": "medium",
            "mean_reversion_tendency": 5,
            "momentum_tendency": 6,
            "breakout_tendency": 6,
            "counter_trend_tendency": 5,
            "complexity": "medium",
            "automation_level": "semi_automated",
            "capital_requirement": "medium",
            "asset_class_suitability": ["stocks", "etfs"]
        }
    )
    library.add_strategy(swing_trading)
    
    # 10. Scalping
    scalping = StrategyDNA(
        strategy_id="scalping",
        name="Scalping",
        description="Takes advantage of small price moves with quick entries and exits.",
        dna_data={
            "market_regime_affinity": ["neutral", "volatile"],
            "volatility_preference": "medium",
            "trend_requirement": "any",
            "time_frame": "intraday",
            "signal_frequency": "high",
            "direction": "both",
            "win_rate_potential": "high",
            "profit_factor_potential": "medium",
            "drawdown_potential": "low",
            "mean_reversion_tendency": 6,
            "momentum_tendency": 5,
            "breakout_tendency": 5,
            "counter_trend_tendency": 6,
            "complexity": "high",
            "automation_level": "fully_automated",
            "capital_requirement": "medium",
            "asset_class_suitability": ["stocks", "futures", "forex"]
        }
    )
    library.add_strategy(scalping)
    
    return library


def save_default_library(file_path: str = "data/strategy_library.json") -> None:
    """
    Create and save a default strategy library.
    
    Args:
        file_path: Path to save the library to
    """
    library = create_default_library()
    library.save_to_file(file_path)
    logger.info(f"Created and saved default strategy library to {file_path}")


# If run as a script, create and save the default strategy library
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    save_default_library()
    print("Created and saved default strategy library.") 