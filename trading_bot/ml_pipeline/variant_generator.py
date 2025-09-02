"""
Variant Generator for Backtest Parameter Combinations
This module generates all possible parameter combinations for strategy backtesting.
"""

import os
import sys
import yaml
import logging
from itertools import product
from typing import Dict, List, Any, Optional, Union

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class VariantGenerator:
    """
    Generates parameter combinations for backtesting strategies.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the variant generator.
        
        Args:
            config_path: Optional path to strategy profiles YAML file
        """
        self.logger = logging.getLogger("VariantGenerator")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Default config path
        if not config_path:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs",
                "strategy_profiles.yaml"
            )
        
        self.config_path = config_path
        self.profiles = self._load_profiles()
        
        self.logger.info(f"Loaded {len(self.profiles)} strategy profiles")
    
    def _load_profiles(self) -> Dict:
        """
        Load strategy profiles from YAML file.
        
        Returns:
            Dictionary of strategy profiles
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    profiles = yaml.safe_load(f)
                return profiles
            else:
                self.logger.warning(f"Strategy profiles file not found: {self.config_path}")
                return {}
        
        except Exception as e:
            self.logger.error(f"Error loading strategy profiles: {str(e)}")
            return {}
    
    def get_profile(self, strategy_id: str) -> Dict:
        """
        Get a specific strategy profile.
        
        Args:
            strategy_id: Strategy identifier
        
        Returns:
            Strategy profile dictionary or empty dict if not found
        """
        return self.profiles.get(strategy_id, {})
    
    def generate_variants(self, param_grid: Dict) -> List[Dict]:
        """
        Generate all possible parameter combinations from a parameter grid.
        
        Args:
            param_grid: Dictionary of parameters and their possible values
        
        Returns:
            List of parameter combinations as dictionaries
        """
        if not param_grid:
            return [{}]
        
        # Extract keys and values
        keys, values = zip(*param_grid.items())
        
        # Generate all combinations
        variants = [dict(zip(keys, v)) for v in product(*values)]
        
        return variants
    
    def get_strategy_variants(self, strategy_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all parameter variants for a specific strategy.
        
        Args:
            strategy_id: Strategy identifier
            limit: Optional maximum number of variants to return
        
        Returns:
            List of parameter combinations for the strategy
        """
        profile = self.get_profile(strategy_id)
        
        if not profile:
            self.logger.warning(f"Strategy profile not found: {strategy_id}")
            return []
        
        # Get parameter grid from profile
        param_grid = profile.get("parameters", {})
        
        # Generate all combinations
        variants = self.generate_variants(param_grid)
        
        self.logger.info(f"Generated {len(variants)} variants for {strategy_id}")
        
        # Limit if specified
        if limit is not None and limit > 0:
            variants = variants[:limit]
            self.logger.info(f"Limited to {len(variants)} variants")
        
        return variants
    
    def get_optimization_goals(self, strategy_id: str) -> Dict:
        """
        Get optimization goals for a specific strategy.
        
        Args:
            strategy_id: Strategy identifier
        
        Returns:
            Dictionary with optimization goals
        """
        profile = self.get_profile(strategy_id)
        
        if not profile:
            self.logger.warning(f"Strategy profile not found: {strategy_id}")
            return {}
        
        return profile.get("optimization", {})
    
    def get_preferred_regime(self, strategy_id: str) -> str:
        """
        Get preferred market regime for a specific strategy.
        
        Args:
            strategy_id: Strategy identifier
        
        Returns:
            Preferred market regime as string
        """
        profile = self.get_profile(strategy_id)
        
        if not profile:
            self.logger.warning(f"Strategy profile not found: {strategy_id}")
            return "all"
        
        return profile.get("regime", "all")
    
    def is_regime_compatible(self, strategy_id: str, current_regime: str) -> bool:
        """
        Check if a strategy is compatible with the current market regime.
        
        Args:
            strategy_id: Strategy identifier
            current_regime: Current market regime
        
        Returns:
            Boolean indicating if the strategy is compatible
        """
        preferred_regime = self.get_preferred_regime(strategy_id)
        
        # If preferred regime is "all", it's compatible with any regime
        if preferred_regime == "all":
            return True
        
        # Otherwise, check if current regime matches preferred regime
        return preferred_regime == current_regime


# Create singleton instance
_variant_generator = None

def get_variant_generator(config_path=None):
    """
    Get the singleton VariantGenerator instance.
    
    Args:
        config_path: Optional path to strategy profiles YAML file
    
    Returns:
        VariantGenerator instance
    """
    global _variant_generator
    if _variant_generator is None:
        _variant_generator = VariantGenerator(config_path)
    return _variant_generator


if __name__ == "__main__":
    # Example usage
    generator = get_variant_generator()
    
    strategy_id = "momentum_breakout"
    variants = generator.get_strategy_variants(strategy_id, limit=5)
    goals = generator.get_optimization_goals(strategy_id)
    
    print(f"Strategy: {strategy_id}")
    print(f"Optimization Goals: {goals}")
    print(f"Sample Variants:")
    for i, variant in enumerate(variants):
        print(f"{i+1}. {variant}")
