#!/usr/bin/env python3
"""
Strategy DNA Mapping System

This module provides a taxonomic framework that helps GPT understand each strategy's
fundamental characteristics, strengths, weaknesses, and optimal conditions, resulting 
in more intelligent strategy selection and rotation.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union

# Configure logging
logger = logging.getLogger(__name__)

class StrategyDNA:
    """
    Represents the fundamental characteristics of a trading strategy.
    
    The DNA profile includes traits, style factors, ideal market conditions,
    weaknesses, timeframe sensitivity, and correlation with other strategies.
    """
    
    def __init__(self, strategy_name: str):
        """
        Initialize a strategy DNA profile.
        
        Args:
            strategy_name: Name of the strategy
        """
        self.strategy_name = strategy_name
        self.traits: List[str] = []
        self.style_factors: Dict[str, float] = {}
        self.ideal_conditions: List[str] = []
        self.weaknesses: List[str] = []
        self.timeframe_sensitivity: Dict[str, float] = {}
        self.correlation_profile: Dict[str, float] = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert DNA profile to dictionary for serialization."""
        return {
            "name": self.strategy_name,
            "traits": self.traits,
            "style_factors": self.style_factors,
            "ideal_conditions": self.ideal_conditions,
            "weaknesses": self.weaknesses,
            "timeframe_sensitivity": self.timeframe_sensitivity,
            "correlation_profile": self.correlation_profile
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyDNA':
        """
        Create a DNA profile from a dictionary.
        
        Args:
            data: Dictionary containing DNA profile data
            
        Returns:
            StrategyDNA instance
        """
        dna = cls(data["name"])
        dna.traits = data.get("traits", [])
        dna.style_factors = data.get("style_factors", {})
        dna.ideal_conditions = data.get("ideal_conditions", [])
        dna.weaknesses = data.get("weaknesses", [])
        dna.timeframe_sensitivity = data.get("timeframe_sensitivity", {})
        dna.correlation_profile = data.get("correlation_profile", {})
        return dna


class StrategyLibrary:
    """
    Manages a collection of trading strategies with their DNA profiles.
    
    This class handles loading, saving, and accessing strategy definitions
    and their corresponding DNA profiles.
    """
    
    def __init__(self, config_path: str = 'data/strategy_library.json'):
        """
        Initialize the strategy library.
        
        Args:
            config_path: Path to the strategy library configuration file
        """
        self.config_path = config_path
        self.strategies: Dict[str, Dict[str, Any]] = {}
        self.load_strategies()
        
    def load_strategies(self) -> None:
        """Load strategies from JSON config file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self.strategies = json.load(f)
                logger.info(f"Loaded {len(self.strategies)} strategies from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading strategy library: {e}")
                # Initialize with sample strategies
                self._initialize_sample_strategies()
        else:
            logger.info(f"Strategy library file not found at {self.config_path}. Initializing with samples.")
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            # Initialize with sample strategies
            self._initialize_sample_strategies()
            # Save to file
            self.save_strategies()
    
    def _initialize_sample_strategies(self) -> None:
        """Initialize with sample strategies if no config exists."""
        self.strategies = {
            "MomentumEdge": {
                "name": "MomentumEdge",
                "description": "Captures momentum in trending markets",
                "active": True,
                "dna": {
                    "traits": ["momentum", "trend-following", "bull-biased"],
                    "style_factors": {
                        "value": -0.8,
                        "momentum": 0.9,
                        "volatility": 0.3,
                        "size": 0.2,
                        "quality": 0.1
                    },
                    "ideal_conditions": [
                        "bullish",
                        "low-volatility",
                        "strong-breadth",
                        "clear-sector-leadership"
                    ],
                    "weaknesses": [
                        "choppy-markets",
                        "high-vix-environment",
                        "trend-reversals",
                        "low-volume"
                    ],
                    "timeframe_sensitivity": {
                        "short_term": 0.8,
                        "medium_term": 0.6,
                        "long_term": 0.3
                    },
                    "correlation_profile": {
                        "MeanReversionPro": -0.4,
                        "VolatilityCrusher": 0.1,
                        "TrendFollower": 0.8
                    }
                },
                "parameters": {
                    "lookback_period": 20,
                    "momentum_threshold": 0.02,
                    "stop_loss": 0.05
                }
            },
            "MeanReversionPro": {
                "name": "MeanReversionPro",
                "description": "Capitalize on price overshoots and reversals",
                "active": True,
                "dna": {
                    "traits": ["mean-reversion", "contrarian", "oversold-biased"],
                    "style_factors": {
                        "value": 0.7,
                        "momentum": -0.6,
                        "volatility": 0.5,
                        "size": -0.1,
                        "quality": 0.3
                    },
                    "ideal_conditions": [
                        "range-bound",
                        "oversold",
                        "volatility-expansion",
                        "sector-rotation"
                    ],
                    "weaknesses": [
                        "strong-trends",
                        "fundamental-breakdowns",
                        "low-liquidity",
                        "news-driven-moves"
                    ],
                    "timeframe_sensitivity": {
                        "short_term": 0.9,
                        "medium_term": 0.4,
                        "long_term": 0.1
                    },
                    "correlation_profile": {
                        "MomentumEdge": -0.4,
                        "VolatilityCrusher": 0.3,
                        "TrendFollower": -0.5
                    }
                },
                "parameters": {
                    "lookback_period": 10,
                    "zscore_threshold": 2.0,
                    "take_profit": 0.03
                }
            },
            "BreakoutHunter": {
                "name": "BreakoutHunter",
                "description": "Identifies and capitalizes on price breakouts",
                "active": True,
                "dna": {
                    "traits": ["breakout", "momentum", "volatility-expansion"],
                    "style_factors": {
                        "value": -0.2,
                        "momentum": 0.7,
                        "volatility": 0.8,
                        "size": 0.0,
                        "quality": -0.1
                    },
                    "ideal_conditions": [
                        "pre-trend",
                        "consolidation-patterns",
                        "increasing-volume",
                        "sector-rotation-beginning"
                    ],
                    "weaknesses": [
                        "false-breakouts",
                        "late-stage-trends",
                        "high-noise-environments",
                        "low-volume-breakouts"
                    ],
                    "timeframe_sensitivity": {
                        "short_term": 0.7,
                        "medium_term": 0.7,
                        "long_term": 0.4
                    },
                    "correlation_profile": {
                        "MomentumEdge": 0.6,
                        "MeanReversionPro": -0.2,
                        "VolatilityCrusher": -0.1
                    }
                },
                "parameters": {
                    "lookback_period": 30,
                    "breakout_threshold": 0.03,
                    "volume_factor": 1.5
                }
            },
            "VolatilityBreakout": {
                "name": "VolatilityBreakout",
                "description": "Capitalizes on sudden market volatility and price breakouts",
                "active": True,
                "dna": {
                    "traits": ["volatility", "breakout", "mean-reversion"],
                    "style_factors": {
                        "value": 0.2,
                        "momentum": 0.4,
                        "volatility": 0.9,
                        "size": 0.3,
                        "quality": -0.3
                    },
                    "ideal_conditions": [
                        "volatile",
                        "high-vix-environment",
                        "earnings-season",
                        "macro-announcements"
                    ],
                    "weaknesses": [
                        "low-volatility",
                        "tight-ranges",
                        "steady-trends",
                        "illiquid-markets"
                    ],
                    "timeframe_sensitivity": {
                        "short_term": 0.9,
                        "medium_term": 0.5,
                        "long_term": 0.1
                    },
                    "correlation_profile": {
                        "MomentumEdge": 0.2,
                        "MeanReversionPro": 0.4,
                        "BreakoutHunter": 0.6
                    }
                },
                "parameters": {
                    "vix_threshold": 25,
                    "atr_multiple": 2.0,
                    "position_size_reduction": 0.25
                }
            },
            "TrendFollower": {
                "name": "TrendFollower",
                "description": "Follows established market trends with position scaling",
                "active": True,
                "dna": {
                    "traits": ["trend-following", "momentum", "position-scaling"],
                    "style_factors": {
                        "value": -0.7,
                        "momentum": 0.8,
                        "volatility": 0.1,
                        "size": 0.0,
                        "quality": 0.4
                    },
                    "ideal_conditions": [
                        "strong-trends",
                        "high-adx",
                        "sector-momentum",
                        "low-correlation-environment"
                    ],
                    "weaknesses": [
                        "sideways-markets",
                        "sudden-reversals",
                        "high-volatility",
                        "choppy-price-action"
                    ],
                    "timeframe_sensitivity": {
                        "short_term": 0.4,
                        "medium_term": 0.8,
                        "long_term": 0.7
                    },
                    "correlation_profile": {
                        "MomentumEdge": 0.7,
                        "MeanReversionPro": -0.5,
                        "BreakoutHunter": 0.4,
                        "VolatilityBreakout": 0.1
                    }
                },
                "parameters": {
                    "trend_strength_min": 25,
                    "moving_average_periods": [20, 50, 200],
                    "position_scale_increment": 0.25
                }
            }
        }
        logger.info("Initialized sample strategies")
            
    def save_strategies(self) -> None:
        """Save strategies to JSON config file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.strategies, f, indent=4)
            logger.info(f"Saved {len(self.strategies)} strategies to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving strategy library: {e}")
            
    def get_strategy(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific strategy by name.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Strategy data or None if not found
        """
        return self.strategies.get(strategy_name)
        
    def get_all_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active strategies.
        
        Returns:
            Dictionary of active strategies
        """
        return {name: strat for name, strat in self.strategies.items() 
                if strat.get("active", True)}
        
    def get_strategy_dna(self, strategy_name: str) -> Optional[StrategyDNA]:
        """
        Get the DNA profile for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            StrategyDNA instance or None if not found
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy or "dna" not in strategy:
            return None
            
        return StrategyDNA.from_dict({"name": strategy_name, **strategy["dna"]})
        
    def get_all_dna_profiles(self) -> Dict[str, StrategyDNA]:
        """
        Get DNA profiles for all active strategies.
        
        Returns:
            Dictionary mapping strategy names to DNA profiles
        """
        return {name: self.get_strategy_dna(name) 
                for name in self.get_all_strategies()}
        
    def update_strategy_dna(self, strategy_name: str, dna_update: Dict[str, Any]) -> bool:
        """
        Update a strategy's DNA profile.
        
        Args:
            strategy_name: Name of the strategy
            dna_update: Dictionary with DNA updates
            
        Returns:
            True if successful, False otherwise
        """
        if strategy_name not in self.strategies:
            logger.warning(f"Cannot update DNA for unknown strategy: {strategy_name}")
            return False
            
        # Update DNA
        current_dna = self.strategies[strategy_name].get("dna", {})
        for key, value in dna_update.items():
            if key in current_dna and isinstance(current_dna[key], dict) and isinstance(value, dict):
                # For nested dicts, update instead of replace
                current_dna[key].update(value)
            else:
                current_dna[key] = value
            
        self.strategies[strategy_name]["dna"] = current_dna
        
        # Save to file
        self.save_strategies()
        logger.info(f"Updated DNA for strategy: {strategy_name}")
        return True
        
    def add_strategy(self, strategy_data: Dict[str, Any]) -> bool:
        """
        Add a new strategy to the library.
        
        Args:
            strategy_data: Complete strategy data including DNA
            
        Returns:
            True if successful, False otherwise
        """
        if "name" not in strategy_data:
            logger.error("Cannot add strategy without a name")
            return False
            
        name = strategy_data["name"]
        if name in self.strategies:
            logger.warning(f"Strategy already exists: {name}. Use update_strategy instead.")
            return False
            
        self.strategies[name] = strategy_data
        self.save_strategies()
        logger.info(f"Added new strategy: {name}")
        return True
        
    def update_strategy(self, strategy_name: str, updates: Dict[str, Any]) -> bool:
        """
        Update a strategy's configuration.
        
        Args:
            strategy_name: Name of the strategy
            updates: Dictionary with updates
            
        Returns:
            True if successful, False otherwise
        """
        if strategy_name not in self.strategies:
            logger.warning(f"Cannot update unknown strategy: {strategy_name}")
            return False
            
        # Update strategy data
        for key, value in updates.items():
            if key == "dna":
                # Use special DNA update method for DNA updates
                self.update_strategy_dna(strategy_name, value)
            else:
                self.strategies[strategy_name][key] = value
                
        self.save_strategies()
        logger.info(f"Updated strategy: {strategy_name}")
        return True
        
    def disable_strategy(self, strategy_name: str) -> bool:
        """
        Disable a strategy (mark as inactive).
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            True if successful, False otherwise
        """
        if strategy_name not in self.strategies:
            logger.warning(f"Cannot disable unknown strategy: {strategy_name}")
            return False
            
        self.strategies[strategy_name]["active"] = False
        self.save_strategies()
        logger.info(f"Disabled strategy: {strategy_name}")
        return True
        
    def enable_strategy(self, strategy_name: str) -> bool:
        """
        Enable a strategy (mark as active).
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            True if successful, False otherwise
        """
        if strategy_name not in self.strategies:
            logger.warning(f"Cannot enable unknown strategy: {strategy_name}")
            return False
            
        self.strategies[strategy_name]["active"] = True
        self.save_strategies()
        logger.info(f"Enabled strategy: {strategy_name}")
        return True


class GPTStrategyScorer:
    """
    Uses GPT to score trading strategies based on market conditions and strategy DNA.
    """
    
    def __init__(self, gpt_client, strategy_library: StrategyLibrary):
        """
        Initialize the strategy scorer.
        
        Args:
            gpt_client: Client for accessing GPT API
            strategy_library: Strategy library instance
        """
        self.gpt_client = gpt_client
        self.strategy_library = strategy_library
        
    def construct_strategy_dna_prompt(self, strategies: List[str], market_context: Dict[str, Any]) -> str:
        """
        Construct a detailed prompt for GPT that incorporates strategy DNA
        and current market conditions.
        
        Args:
            strategies: List of strategy names to score
            market_context: Current market context
            
        Returns:
            Prompt string for GPT
        """
        # Get DNA profiles for all strategies
        dna_profiles = {name: self.strategy_library.get_strategy_dna(name) 
                        for name in strategies}
        
        # Format market context
        market_regime = market_context.get("regime", {}).get("primary_regime", "unknown")
        market_traits = market_context.get("regime", {}).get("traits", [])
        
        # Construct prompt with DNA information
        prompt = f"""
        You are analyzing the optimal trading strategies for the current market conditions.
        
        Current Market Context:
        - Primary Regime: {market_regime}
        - Market Traits: {', '.join(market_traits)}
        - VIX: {market_context.get('vix', 'N/A')}
        - Recent Market Performance: {market_context.get('recent_performance', 'N/A')}
        - Sector Leaders: {', '.join(market_context.get('sector_leaders', []))}
        - Sector Laggards: {', '.join(market_context.get('sector_laggards', []))}
        
        For each strategy below, analyze how well it is likely to perform in the current market conditions.
        Consider each strategy's traits, ideal conditions, and weaknesses.
        
        Score each strategy from 0-10, where:
        - 10: Optimal conditions for this strategy
        - 7-9: Favorable conditions
        - 4-6: Neutral conditions
        - 1-3: Unfavorable conditions
        - 0: Worst possible conditions, strategy likely to lose money
        
        Also provide a brief reasoning for each score.
        
        Strategies to evaluate:
        """
        
        # Add strategy DNA information
        for name, dna in dna_profiles.items():
            if not dna:
                continue
                
            traits_str = ', '.join(dna.traits) if dna.traits else "None"
            ideal_conditions_str = ', '.join(dna.ideal_conditions) if dna.ideal_conditions else "None"
            weaknesses_str = ', '.join(dna.weaknesses) if dna.weaknesses else "None"
            style_factors_str = ', '.join([f"{k}: {v}" for k, v in dna.style_factors.items()]) if dna.style_factors else "None"
            
            prompt += f"""
            
            {name}:
            - Traits: {traits_str}
            - Ideal Conditions: {ideal_conditions_str}
            - Weaknesses: {weaknesses_str}
            - Style Factors: {style_factors_str}
            """
        
        prompt += """
        
        Format your response as a JSON object with strategy names as keys, and objects containing 'score' and 'reasoning' as values.
        """
        
        return prompt
        
    def score_strategies(self, strategies: List[str], market_context: Dict[str, Any]) -> Dict[str, Dict[str, Union[float, str]]]:
        """
        Score strategies using GPT with DNA-enhanced prompting.
        
        Args:
            strategies: List of strategy names to score
            market_context: Current market context
            
        Returns:
            Dictionary mapping strategy names to scores and reasoning
        """
        prompt = self.construct_strategy_dna_prompt(strategies, market_context)
        logger.info(f"Generating GPT scores for {len(strategies)} strategies")
        
        try:
            # Call GPT API
            response = self.gpt_client.complete(prompt, response_format={"type": "json_object"})
            
            # Parse response
            content = response.choices[0].message.content
            scores = json.loads(content)
            
            # Validate and clean scores
            validated_scores = {}
            for name, data in scores.items():
                if name in strategies:
                    score = data.get("score", 5)
                    # Ensure score is within bounds
                    score = max(0, min(10, float(score)))
                    
                    validated_scores[name] = {
                        "score": score,
                        "reasoning": data.get("reasoning", "No reasoning provided")
                    }
            
            logger.info(f"Successfully scored {len(validated_scores)} strategies with GPT")
            return validated_scores
            
        except Exception as e:
            logger.error(f"Error scoring strategies with GPT: {e}")
            # Fallback to default scoring if GPT response parsing fails
            return {name: {"score": 5, "reasoning": "Default score due to GPT error"} for name in strategies} 