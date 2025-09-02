#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prioritizer Integration - Utilities to integrate the EnhancedStrategyPrioritizer 
with other components of the trading system.
"""

import os
import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta

from trading_bot.ai_scoring.enhanced_strategy_prioritizer import EnhancedStrategyPrioritizer
from trading_bot.utils.market_context_fetcher import MarketContextFetcher
from trading_bot.ai_scoring.strategy_rotator import StrategyRotator

# Configure logging
logger = logging.getLogger("PrioritizerIntegration")

class PrioritizerIntegration:
    """
    Integration utilities for the EnhancedStrategyPrioritizer.
    Provides methods to connect the prioritizer with other trading bot components.
    """
    
    def __init__(
        self,
        strategies: List[str],
        api_key: Optional[str] = None,
        use_mock: bool = False,
        data_dir: Optional[str] = None,
        strategy_rotator: Optional[StrategyRotator] = None
    ):
        """
        Initialize the integration utilities.
        
        Args:
            strategies: List of strategy names
            api_key: API key for language model service
            use_mock: Whether to use mock responses
            data_dir: Data directory for cache and state
            strategy_rotator: Optional strategy rotator instance
        """
        # Set up data paths
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)
        
        cache_dir = os.path.join(data_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize components
        self.prioritizer = EnhancedStrategyPrioritizer(
            strategies=strategies,
            api_key=api_key,
            use_mock=use_mock,
            cache_dir=cache_dir,
            memory_file=os.path.join(data_dir, "prioritizer_memory.json"),
            enable_sentiment_data=True,
            enable_macro_data=True
        )
        
        self.market_context_fetcher = MarketContextFetcher()
        self.strategy_rotator = strategy_rotator
        self.current_allocations = None
        self.allocation_history = []
        
        logger.info(f"PrioritizerIntegration initialized with {len(strategies)} strategies")
    
    def get_allocations(
        self, 
        force_refresh: bool = False,
        apply_to_rotator: bool = True
    ) -> Dict[str, float]:
        """
        Get strategy allocations from the prioritizer and optionally apply to rotator.
        
        Args:
            force_refresh: Whether to force refresh (bypass cache)
            apply_to_rotator: Whether to apply allocations to strategy rotator
            
        Returns:
            Dictionary mapping strategy names to allocation percentages
        """
        # Get allocation from prioritizer
        result = self.prioritizer.get_strategy_allocation(
            previous_allocations=self.current_allocations,
            force_refresh=force_refresh
        )
        
        # Extract allocations
        allocations = result.get("allocations", {})
        
        # Store current allocations
        self.current_allocations = allocations.copy()
        
        # Record in history
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "allocations": allocations,
            "explanation": result.get("explanation", ""),
            "reasoning": result.get("reasoning", []),
            "risk_level": result.get("risk_level", "normal"),
            "risk_warnings": result.get("risk_warnings", [])
        }
        self.allocation_history.append(history_entry)
        
        # Apply to strategy rotator if provided and requested
        if self.strategy_rotator is not None and apply_to_rotator:
            self._apply_to_rotator(allocations)
        
        # Return allocations
        return allocations
    
    def _apply_to_rotator(self, allocations: Dict[str, float]) -> None:
        """
        Apply allocations to the strategy rotator.
        
        Args:
            allocations: Strategy allocations (percentages)
        """
        try:
            # Convert percentage allocations to weights
            weights = {k: v / 100.0 for k, v in allocations.items()}
            
            # Apply to rotator
            # This depends on your StrategyRotator implementation
            if hasattr(self.strategy_rotator, 'update_strategy_weights'):
                # Direct method if available
                self.strategy_rotator.update_strategy_weights(weights)
            elif hasattr(self.strategy_rotator, 'set_strategy_weights'):
                # Alternative method name
                self.strategy_rotator.set_strategy_weights(weights)
            else:
                # Manual approach - access weights dictionary directly if exposed
                for strategy, weight in weights.items():
                    if hasattr(self.strategy_rotator, 'strategies_by_name') and \
                       strategy in self.strategy_rotator.strategies_by_name:
                        strategy_obj = self.strategy_rotator.strategies_by_name[strategy]
                        strategy_obj.current_weight = weight
                
                # Normalize weights in rotator if method available
                if hasattr(self.strategy_rotator, '_normalize_weights'):
                    self.strategy_rotator._normalize_weights()
            
            logger.info(f"Applied allocations to strategy rotator: {allocations}")
        except Exception as e:
            logger.error(f"Error applying allocations to rotator: {e}")
    
    def record_performance(self, performance_metrics: Dict[str, float]) -> None:
        """
        Record performance metrics for feedback loop.
        
        Args:
            performance_metrics: Performance metrics (returns, sharpe, etc.)
        """
        if not self.current_allocations:
            logger.warning("No allocations to associate with performance")
            return
        
        # Record feedback in prioritizer
        self.prioritizer.record_performance_feedback(
            self.current_allocations, 
            performance_metrics
        )
        
        logger.info("Recorded performance feedback")
    
    def get_allocation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get allocation history with explanations.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of allocation history entries
        """
        if limit is None or limit >= len(self.allocation_history):
            return self.allocation_history.copy()
        
        return self.allocation_history[-limit:]
    
    def export_allocations_to_csv(self, filepath: Optional[str] = None) -> str:
        """
        Export allocation history to CSV file.
        
        Args:
            filepath: Path to save the CSV file (default: prioritizer_allocations.csv)
            
        Returns:
            Path to the saved CSV file
        """
        if not self.allocation_history:
            logger.warning("No allocation history to export")
            return ""
        
        if filepath is None:
            filepath = "prioritizer_allocations.csv"
        
        # Prepare data for CSV
        rows = []
        for entry in self.allocation_history:
            row = {
                "timestamp": entry.get("timestamp", ""),
                "risk_level": entry.get("risk_level", "")
            }
            
            # Add allocations
            for strategy, allocation in entry.get("allocations", {}).items():
                row[f"{strategy}_allocation"] = allocation
            
            rows.append(row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Exported {len(rows)} allocation entries to {filepath}")
        return filepath
    
    def get_annotated_allocations(self) -> Dict[str, Any]:
        """
        Get current allocations with reasoning and risk information.
        
        Returns:
            Dictionary with allocations, reasoning, and risk info
        """
        if not self.allocation_history:
            return {
                "allocations": {},
                "reasoning": [],
                "risk_level": "normal",
                "risk_warnings": []
            }
        
        # Get most recent entry
        latest = self.allocation_history[-1]
        
        return {
            "allocations": latest.get("allocations", {}),
            "explanation": latest.get("explanation", ""),
            "reasoning": latest.get("reasoning", []),
            "risk_level": latest.get("risk_level", "normal"),
            "risk_warnings": latest.get("risk_warnings", []),
            "timestamp": latest.get("timestamp", datetime.now().isoformat())
        }
    
    def clear_cache(self) -> None:
        """Clear prioritizer cache."""
        self.prioritizer.clear_cache()
        logger.info("Cleared prioritizer cache")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Define available strategies
    strategies = [
        "momentum",
        "trend_following",
        "breakout_swing",
        "mean_reversion",
        "volatility_breakout", 
        "option_spreads"
    ]
    
    # Initialize integration
    integration = PrioritizerIntegration(
        strategies=strategies,
        use_mock=True
    )
    
    # Get allocations
    allocations = integration.get_allocations()
    
    # Print allocations
    print("\nStrategy Allocations:")
    for strategy, allocation in allocations.items():
        print(f"{strategy}: {allocation:.1f}%")
    
    # Get annotated allocations with explanations
    annotated = integration.get_annotated_allocations()
    
    # Print explanation
    if "explanation" in annotated:
        print("\nExplanation:")
        print(annotated["explanation"])
    
    # Print reasoning
    if "reasoning" in annotated:
        print("\nReasoning:")
        for point in annotated["reasoning"]:
            print(f"- {point}")
    
    # Print risk information
    print(f"\nRisk Level: {annotated['risk_level']}")
    
    if annotated.get("risk_warnings"):
        print("\nRisk Warnings:")
        for warning in annotated["risk_warnings"]:
            print(f"- {warning}") 