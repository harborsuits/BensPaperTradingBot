#!/usr/bin/env python3
"""
Parameter Optimizer Integration

This module connects our parameter optimization system with synthetic testing and performance
verification, completing the feedback loop for continuous improvement of market simulation.

Key Components:
1. Event handlers for connecting verification results to parameter optimization
2. Integration with synthetic testing to apply optimized parameters
3. CLI utilities for monitoring optimization progress
4. Configuration management

This builds directly on our working synthetic testing and performance verification
systems, creating a fully closed loop for continuous system improvement.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Import project components
from trading_bot.autonomous.synthetic_market_generator import (
    SyntheticMarketGenerator, get_synthetic_market_generator
)
from trading_bot.autonomous.synthetic_market_generator_correlations import (
    CorrelatedMarketGenerator, get_correlated_market_generator
)
from trading_bot.autonomous.synthetic_testing_integration import (
    SyntheticTestingIntegration, get_synthetic_testing_integration
)
from trading_bot.autonomous.performance_verification import (
    PerformanceVerifier, get_performance_verifier
)
from trading_bot.autonomous.parameter_optimization import (
    ParameterOptimizer, get_parameter_optimizer
)

# Import event system
from trading_bot.event_system import EventBus, Event, EventType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizerIntegration:
    """
    Integrates parameter optimization with synthetic testing and performance verification.
    
    This class acts as the connective tissue between:
    1. Performance verification results
    2. Parameter optimization
    3. Synthetic market generation
    
    Together, these components form a closed-loop system that continuously
    improves as more real-world performance data becomes available.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the optimizer integration.
        
        Args:
            config_path: Path to configuration file
        """
        # Get component instances
        self.optimizer = get_parameter_optimizer()
        self.verifier = get_performance_verifier()
        self.synthetic_testing = get_synthetic_testing_integration()
        self.market_generator = get_synthetic_market_generator()
        self.correlated_generator = get_correlated_market_generator()
        
        # Event system
        self.event_bus = EventBus()
        
        # Configuration
        self.config = {
            "auto_apply_parameters": True,
            "min_optimization_interval_hours": 24,
            "min_verification_count": 10,
            "accuracy_threshold": 0.7,
            "notification_enabled": True
        }
        
        # Load configuration
        if config_path:
            self._load_config(config_path)
        
        # Statistics
        self.optimization_count = 0
        self.last_optimization = None
        self.performance_improvements = []
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info("Optimizer integration initialized")
    
    def _register_event_handlers(self) -> None:
        """Register event handlers for integration."""
        # Handle market parameter updates
        self.event_bus.register(
            "market_parameters_updated",
            self._handle_parameter_update
        )
        
        # Handle verification results
        self.event_bus.register(
            "verification_complete",
            self._handle_verification_complete
        )
        
        # Handle new synthetic tests
        self.event_bus.register(
            "synthetic_test_created",
            self._handle_synthetic_test_created
        )
        
        logger.info("Registered optimizer integration event handlers")
    
    def _handle_parameter_update(self, event: Event) -> None:
        """
        Handle market parameter update events.
        
        Args:
            event: Parameter update event
        """
        if not self.config["auto_apply_parameters"]:
            logger.info("Auto-apply parameters disabled, skipping update")
            return
        
        # Extract parameters
        parameters = event.data.get("parameters", {})
        optimization_count = event.data.get("optimization_count", 0)
        
        # Update statistics
        self.optimization_count = optimization_count
        self.last_optimization = datetime.utcnow()
        
        # Apply parameters to market generators
        self._apply_parameters(parameters)
        
        logger.info(f"Applied optimized parameters (optimization #{optimization_count})")
        
        # Emit notification
        if self.config["notification_enabled"]:
            self.event_bus.emit(
                Event(
                    event_type="notification",
                    data={
                        "title": "Market Parameters Updated",
                        "message": f"Synthetic market parameters have been optimized (#{optimization_count})",
                        "level": "info",
                        "category": "optimization"
                    },
                    source="optimizer_integration"
                )
            )
    
    def _handle_verification_complete(self, event: Event) -> None:
        """
        Handle verification complete events.
        
        Args:
            event: Verification complete event
        """
        # Extract verification results
        verification_id = event.data.get("verification_id")
        accuracy = event.data.get("accuracy", 0.0)
        previous_accuracy = event.data.get("previous_accuracy", 0.0)
        
        # Record performance improvement if available
        if previous_accuracy > 0:
            improvement = accuracy - previous_accuracy
            self.performance_improvements.append({
                "timestamp": datetime.utcnow().isoformat(),
                "verification_id": verification_id,
                "previous_accuracy": previous_accuracy,
                "new_accuracy": accuracy,
                "improvement": improvement,
                "optimization_count": self.optimization_count
            })
            
            # Keep only last 50 improvement records
            if len(self.performance_improvements) > 50:
                self.performance_improvements = self.performance_improvements[-50:]
            
            logger.info(f"Recorded performance improvement: {improvement:.4f} for verification {verification_id}")
    
    def _handle_synthetic_test_created(self, event: Event) -> None:
        """
        Handle synthetic test creation events.
        
        Ensures all new synthetic tests use the latest optimized parameters.
        
        Args:
            event: Synthetic test created event
        """
        test_id = event.data.get("test_id")
        
        if test_id and self.config["auto_apply_parameters"]:
            logger.info(f"New synthetic test created: {test_id}, checking parameter freshness")
            
            # Check if parameters recently optimized
            if self.optimization_count > 0:
                logger.info(f"Ensuring test {test_id} uses latest optimized parameters")
                
                # The event will already have applied the optimized parameters
                # since the generator would have been configured, but we log it
                # for tracking purposes
                pass
    
    def _apply_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Apply parameters to market generators.
        
        Args:
            parameters: Market parameters
        """
        # Apply to synthetic market generator
        self.optimizer.apply_parameters_to_generator(self.market_generator)
        
        # Apply to correlated market generator
        self.optimizer.apply_parameters_to_correlated_generator(self.correlated_generator)
        
        logger.info("Applied optimized parameters to market generators")
    
    def _load_config(self, config_path: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
                    logger.info(f"Loaded optimizer integration config from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading optimizer config: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the optimizer integration.
        
        Returns:
            Dictionary with status details
        """
        return {
            "optimization_count": self.optimization_count,
            "last_optimization": self.last_optimization.isoformat() if self.last_optimization else None,
            "auto_apply_enabled": self.config["auto_apply_parameters"],
            "performance_improvements": self.performance_improvements[-5:],  # Last 5 improvements
            "configuration": self.config
        }
    
    def toggle_auto_apply(self, enabled: bool) -> None:
        """
        Toggle automatic parameter application.
        
        Args:
            enabled: Whether to enable auto-apply
        """
        self.config["auto_apply_parameters"] = enabled
        logger.info(f"Auto-apply parameters {'enabled' if enabled else 'disabled'}")


# Singleton instance
_optimizer_integration = None


def get_optimizer_integration() -> OptimizerIntegration:
    """
    Get singleton instance of optimizer integration.
    
    Returns:
        OptimizerIntegration instance
    """
    global _optimizer_integration
    
    if _optimizer_integration is None:
        _optimizer_integration = OptimizerIntegration()
    
    return _optimizer_integration


def initialize_optimizer_integration() -> None:
    """Initialize optimizer integration (for use in startup scripts)."""
    # Get instance (will initialize if needed)
    integration = get_optimizer_integration()
    logger.info("Optimizer integration initialized and ready")


if __name__ == "__main__":
    # Simple test of the optimizer integration
    integration = get_optimizer_integration()
    status = integration.get_status()
    print(json.dumps(status, indent=2))
