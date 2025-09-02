"""
Factory for creating and configuring the complete strategy management system.
"""

import os
import logging
import json
from typing import Dict, Any, Optional

from .interfaces import CoreContext, StrategyPrioritizer
from .dynamic_rotation import DynamicStrategyRotator
from .context_integration import ContextDecisionIntegration
from .learning import ContinuousLearningSystem

logger = logging.getLogger("strategy_factory")

def create_strategy_management_system(
    core_context: CoreContext,
    strategy_prioritizer: StrategyPrioritizer,
    unified_context_manager: Any,
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create the complete strategy management system with all components.
    
    Args:
        core_context: Core context instance
        strategy_prioritizer: Strategy prioritizer implementation
        unified_context_manager: Unified context manager instance
        config_path: Path to configuration file (optional)
    
    Returns:
        Dictionary containing all system components
    """
    try:
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Default configuration
            config = {
                "rotator": {
                    "rotation_frequency_days": 7,
                    "min_change_threshold": 5.0,
                    "force_on_regime_change": True,
                    "max_allocation_change": 15.0,
                    "drawdown_allocation_reduction": 0.5,
                    "max_drawdown_threshold": 10.0
                },
                "integration": {
                    "auto_rotation_enabled": True,
                    "auto_rotation_interval_days": 7,
                    "respond_to_unified_signals": True,
                    "unified_signal_threshold": 0.7,
                    "risk_triggered_rotation": True,
                    "risk_threshold": "elevated"
                },
                "learning": {
                    "learning_frequency_days": 14,
                    "min_data_points": 30,
                    "max_history_days": 365,
                    "learning_rate": 0.2,
                    "regime_weight": 0.7,
                    "data_dir": "data/learning"
                }
            }
        
        # Create the system components
        rotator = DynamicStrategyRotator(
            core_context=core_context,
            strategy_prioritizer=strategy_prioritizer,
            config=config.get("rotator", {})
        )
        
        integration = ContextDecisionIntegration(
            core_context=core_context,
            unified_context_manager=unified_context_manager,
            strategy_rotator=rotator,
            config=config.get("integration", {})
        )
        
        learning = ContinuousLearningSystem(
            core_context=core_context,
            config=config.get("learning", {})
        )
        
        logger.info("Created strategy management system")
        
        return {
            "rotator": rotator,
            "integration": integration,
            "learning": learning
        }
    except Exception as e:
        logger.error(f"Error creating strategy management system: {str(e)}")
        raise 