#!/usr/bin/env python3
"""
Parameter Optimization System

This module implements an automatic feedback mechanism that adjusts synthetic market
parameters based on verification results, creating a closed-loop system that
continuously improves prediction accuracy over time.

Key components:
1. Parameter adjustment based on verification metrics
2. Event-driven optimization on verification report generation
3. Automatic parameter tuning with configurable thresholds
4. Persistent storage of optimized parameters
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# Import synthetic market components
from trading_bot.autonomous.synthetic_market_generator import (
    SyntheticMarketGenerator, MarketRegimeType
)
from trading_bot.autonomous.synthetic_market_generator_correlations import (
    CorrelatedMarketGenerator, CorrelationStructure
)

# Import performance verification
from trading_bot.autonomous.performance_verification import (
    get_performance_verifier, VerificationMetrics
)

# Import event system
from trading_bot.event_system import EventBus, Event, EventType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketParameters:
    """
    Container for synthetic market generation parameters.
    
    These parameters control the behavior of synthetic market generation
    and can be automatically adjusted based on verification results.
    """
    
    def __init__(self):
        """Initialize with default parameters."""
        # Base parameters - used across all regimes
        self.base_volatility = 0.01
        self.base_drift = 0.0001
        self.mean_reversion_strength = 0.1
        self.correlation_strength = 0.3
        
        # Regime-specific parameters
        self.regime_parameters = {
            MarketRegimeType.BULLISH.value: {
                "volatility_multiplier": 1.5,
                "drift_multiplier": 8.0,      # ~20% annualized return
                "momentum_factor": 0.2,
                "volatility_clustering": 0.1,
                "panic_selling_probability": 0.05
            },
            MarketRegimeType.BEARISH.value: {
                "volatility_multiplier": 2.0,
                "drift_multiplier": -6.0,     # ~-15% annualized return
                "momentum_factor": 0.1,
                "volatility_clustering": 0.15,
                "panic_selling_probability": 0.3
            },
            MarketRegimeType.SIDEWAYS.value: {
                "volatility_multiplier": 0.8,
                "drift_multiplier": 1.0,      # ~2.5% annualized return
                "momentum_factor": 0.05,
                "volatility_clustering": 0.05,
                "panic_selling_probability": 0.1
            },
            MarketRegimeType.VOLATILE.value: {
                "volatility_multiplier": 3.0,
                "drift_multiplier": 1.0,      # High volatility, slight upward bias
                "momentum_factor": 0.15,
                "volatility_clustering": 0.3,
                "panic_selling_probability": 0.2
            }
        }
        
        # Sector correlation parameters
        self.intra_sector_correlation = 0.7
        self.inter_sector_correlation = 0.3
        
        # Optimization metadata
        self.last_updated = datetime.utcnow().isoformat()
        self.optimization_history = []
        self.version = "1.0.0"
    
    def get_effective_parameters(self, regime: str) -> Dict[str, float]:
        """
        Get effective parameters for a specific regime.
        
        Args:
            regime: Market regime
            
        Returns:
            Dictionary of parameter values
        """
        if regime not in self.regime_parameters:
            regime = MarketRegimeType.SIDEWAYS.value  # Default to sideways
        
        regime_params = self.regime_parameters[regime]
        
        return {
            "volatility": self.base_volatility * regime_params["volatility_multiplier"],
            "drift": self.base_drift * regime_params["drift_multiplier"],
            "momentum_factor": regime_params["momentum_factor"],
            "volatility_clustering": regime_params["volatility_clustering"],
            "panic_selling_probability": regime_params["panic_selling_probability"],
            "mean_reversion_strength": self.mean_reversion_strength,
            "correlation_strength": self.correlation_strength,
            "intra_sector_correlation": self.intra_sector_correlation,
            "inter_sector_correlation": self.inter_sector_correlation
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary for serialization."""
        return {
            "base_volatility": self.base_volatility,
            "base_drift": self.base_drift,
            "mean_reversion_strength": self.mean_reversion_strength,
            "correlation_strength": self.correlation_strength,
            "regime_parameters": self.regime_parameters,
            "intra_sector_correlation": self.intra_sector_correlation,
            "inter_sector_correlation": self.inter_sector_correlation,
            "last_updated": self.last_updated,
            "optimization_history": self.optimization_history,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketParameters':
        """Create parameters from dictionary."""
        params = cls()
        
        # Base parameters
        params.base_volatility = data.get("base_volatility", params.base_volatility)
        params.base_drift = data.get("base_drift", params.base_drift)
        params.mean_reversion_strength = data.get("mean_reversion_strength", params.mean_reversion_strength)
        params.correlation_strength = data.get("correlation_strength", params.correlation_strength)
        
        # Regime-specific parameters
        regime_params = data.get("regime_parameters", {})
        for regime, regime_data in regime_params.items():
            if regime in params.regime_parameters:
                params.regime_parameters[regime].update(regime_data)
        
        # Sector correlation parameters
        params.intra_sector_correlation = data.get("intra_sector_correlation", params.intra_sector_correlation)
        params.inter_sector_correlation = data.get("inter_sector_correlation", params.inter_sector_correlation)
        
        # Metadata
        params.last_updated = data.get("last_updated", params.last_updated)
        params.optimization_history = data.get("optimization_history", params.optimization_history)
        params.version = data.get("version", params.version)
        
        return params


class ParameterOptimizer:
    """
    Adjusts synthetic market parameters based on verification results.
    
    This creates a closed-loop system where:
    1. Synthetic markets generate test data
    2. A/B tests evaluate strategies
    3. Approval process promotes strategies
    4. Performance verification compares real vs. synthetic
    5. Parameter optimization improves synthetic parameters
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the parameter optimizer.
        
        Args:
            storage_path: Path to store optimized parameters
        """
        # Set default storage path
        if storage_path is None:
            home_dir = str(Path.home())
            self.storage_path = os.path.join(home_dir, ".trading_bot", "market_parameters.json")
        else:
            self.storage_path = storage_path
        
        # Core components
        self.event_bus = EventBus()
        self.verifier = get_performance_verifier()
        
        # Market parameters
        self.parameters = MarketParameters()
        
        # Optimization settings
        self.optimization_enabled = True
        self.learning_rate = 0.05  # How aggressively to adjust parameters
        self.min_data_points = 10  # Minimum verification data points needed
        self.max_adjustment = 0.2  # Maximum adjustment per optimization cycle
        
        # Optimization counter
        self.optimization_count = 0
        self.last_optimization = None
        
        # Load existing parameters if available
        self._load_from_disk()
        
        # Register for verification events
        self._register_event_handlers()
        
        logger.info(f"Parameter Optimizer initialized with storage at {self.storage_path}")
    
    def _register_event_handlers(self) -> None:
        """Register for relevant events."""
        # Listen for verification report generation
        self.event_bus.register(
            "verification_report_generated",
            self._handle_verification_report
        )
    
    def _handle_verification_report(self, event: Event) -> None:
        """
        Handle verification report events.
        
        Args:
            event: Verification report event
        """
        if not self.optimization_enabled:
            logger.info("Parameter optimization is disabled, skipping")
            return
        
        report = event.data.get("report")
        if not report:
            logger.warning("Verification report event without report data")
            return
        
        # Check if we have enough data for optimization
        metrics = report.get("overall_metrics", {})
        if metrics.get("total_predictions", 0) < self.min_data_points:
            logger.info(f"Not enough verification data for optimization ({metrics.get('total_predictions', 0)}/{self.min_data_points})")
            return
        
        # Optimize parameters based on report
        self._optimize_parameters(report)
        
        # Save updated parameters
        self._save_to_disk()
        
        # Emit event for parameter update
        self.event_bus.emit(
            Event(
                event_type="market_parameters_updated",
                data={
                    "optimization_count": self.optimization_count,
                    "parameters": self.parameters.to_dict()
                },
                source="parameter_optimizer"
            )
        )
    
    def _optimize_parameters(self, report: Dict[str, Any]) -> None:
        """
        Optimize parameters based on verification report.
        
        Args:
            report: Verification report
        """
        # Get overall metrics
        metrics = report.get("overall_metrics", {})
        prediction_accuracy = metrics.get("prediction_accuracy", 0.0)
        sharpe_correlation = metrics.get("sharpe_correlation", 0.0)
        drawdown_correlation = metrics.get("drawdown_correlation", 0.0)
        win_rate_correlation = metrics.get("win_rate_correlation", 0.0)
        
        # Get regime-specific accuracy
        regime_accuracy = report.get("regime_accuracy", {})
        
        # Record optimization attempt
        optimization_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_accuracy": prediction_accuracy,
            "adjustments": {}
        }
        
        # Adjust base parameters based on overall metrics
        adjustments = {}
        
        # If sharpe correlation is low, adjust volatility
        if sharpe_correlation < 0.7:
            adjustment_factor = (0.7 - sharpe_correlation) * self.learning_rate
            adjustment_factor = min(adjustment_factor, self.max_adjustment)
            
            # Determine direction based on accuracy trend
            if len(self.parameters.optimization_history) > 0:
                prev_accuracy = self.parameters.optimization_history[-1].get("overall_accuracy", 0.0)
                if prediction_accuracy > prev_accuracy:
                    # Keep going in same direction
                    direction = 1.0 if adjustments.get("base_volatility_direction", 0) >= 0 else -1.0
                else:
                    # Reverse direction
                    direction = -1.0 if adjustments.get("base_volatility_direction", 0) >= 0 else 1.0
            else:
                # Initial direction
                direction = 1.0
            
            # Apply adjustment
            adjustments["base_volatility"] = direction * adjustment_factor
            adjustments["base_volatility_direction"] = direction
            
            new_volatility = self.parameters.base_volatility * (1.0 + adjustments["base_volatility"])
            self.parameters.base_volatility = max(0.001, new_volatility)  # Ensure minimum volatility
            
            optimization_record["adjustments"]["base_volatility"] = {
                "before": self.parameters.base_volatility / (1.0 + adjustments["base_volatility"]),
                "after": self.parameters.base_volatility,
                "adjustment_factor": adjustment_factor,
                "direction": direction
            }
        
        # If drawdown correlation is low, adjust regime-specific parameters
        if drawdown_correlation < 0.7:
            # Adjust panic selling probability
            for regime, accuracy_data in regime_accuracy.items():
                if regime in self.parameters.regime_parameters:
                    regime_accuracy_value = accuracy_data.get("accuracy", 0.0)
                    if regime_accuracy_value < 0.7:
                        # Calculate adjustment
                        adjustment_factor = (0.7 - regime_accuracy_value) * self.learning_rate
                        adjustment_factor = min(adjustment_factor, self.max_adjustment)
                        
                        # Apply adjustment to panic selling probability
                        panic_key = "panic_selling_probability"
                        current_value = self.parameters.regime_parameters[regime][panic_key]
                        
                        # Adjust based on regime
                        if regime == MarketRegimeType.BEARISH.value:
                            # Increase for bearish markets
                            new_value = current_value * (1.0 + adjustment_factor)
                            new_value = min(new_value, 0.5)  # Cap at 50%
                        else:
                            # Decrease for other markets
                            new_value = current_value * (1.0 - adjustment_factor)
                            new_value = max(new_value, 0.01)  # Minimum 1%
                        
                        self.parameters.regime_parameters[regime][panic_key] = new_value
                        
                        # Record adjustment
                        if regime not in optimization_record["adjustments"]:
                            optimization_record["adjustments"][regime] = {}
                        
                        optimization_record["adjustments"][regime][panic_key] = {
                            "before": current_value,
                            "after": new_value,
                            "adjustment_factor": adjustment_factor
                        }
        
        # If win rate correlation is low, adjust mean reversion strength
        if win_rate_correlation < 0.7:
            adjustment_factor = (0.7 - win_rate_correlation) * self.learning_rate
            adjustment_factor = min(adjustment_factor, self.max_adjustment)
            
            # Determine direction
            if len(self.parameters.optimization_history) > 0:
                prev_accuracy = self.parameters.optimization_history[-1].get("overall_accuracy", 0.0)
                if prediction_accuracy > prev_accuracy:
                    # Keep going in same direction
                    direction = 1.0 if adjustments.get("mean_reversion_direction", 0) >= 0 else -1.0
                else:
                    # Reverse direction
                    direction = -1.0 if adjustments.get("mean_reversion_direction", 0) >= 0 else 1.0
            else:
                # Initial direction
                direction = 1.0
            
            # Apply adjustment
            adjustments["mean_reversion"] = direction * adjustment_factor
            adjustments["mean_reversion_direction"] = direction
            
            new_mean_reversion = self.parameters.mean_reversion_strength * (1.0 + adjustments["mean_reversion"])
            self.parameters.mean_reversion_strength = max(0.05, min(0.5, new_mean_reversion))
            
            optimization_record["adjustments"]["mean_reversion_strength"] = {
                "before": self.parameters.mean_reversion_strength / (1.0 + adjustments["mean_reversion"]),
                "after": self.parameters.mean_reversion_strength,
                "adjustment_factor": adjustment_factor,
                "direction": direction
            }
        
        # Apply regime-specific drift adjustments
        for regime, accuracy_data in regime_accuracy.items():
            if regime in self.parameters.regime_parameters:
                regime_accuracy_value = accuracy_data.get("accuracy", 0.0)
                if regime_accuracy_value < 0.7:
                    # Calculate adjustment
                    adjustment_factor = (0.7 - regime_accuracy_value) * self.learning_rate
                    adjustment_factor = min(adjustment_factor, self.max_adjustment)
                    
                    # Apply adjustment to drift multiplier
                    drift_key = "drift_multiplier"
                    current_value = self.parameters.regime_parameters[regime][drift_key]
                    
                    # Determine direction based on regime
                    if regime == MarketRegimeType.BULLISH.value and current_value < 10.0:
                        # Increase drift for bullish markets
                        new_value = current_value * (1.0 + adjustment_factor)
                    elif regime == MarketRegimeType.BEARISH.value and current_value > -10.0:
                        # Decrease drift for bearish markets
                        new_value = current_value * (1.0 + adjustment_factor)
                    elif abs(current_value) > 1.0:
                        # Move towards neutral for other regimes
                        new_value = current_value * (1.0 - adjustment_factor)
                    else:
                        new_value = current_value
                    
                    self.parameters.regime_parameters[regime][drift_key] = new_value
                    
                    # Record adjustment
                    if regime not in optimization_record["adjustments"]:
                        optimization_record["adjustments"][regime] = {}
                    
                    optimization_record["adjustments"][regime][drift_key] = {
                        "before": current_value,
                        "after": new_value,
                        "adjustment_factor": adjustment_factor
                    }
        
        # Update metadata
        self.parameters.last_updated = datetime.utcnow().isoformat()
        self.parameters.optimization_history.append(optimization_record)
        
        # Keep only last 50 optimization records
        if len(self.parameters.optimization_history) > 50:
            self.parameters.optimization_history = self.parameters.optimization_history[-50:]
        
        # Increment optimization counter
        self.optimization_count += 1
        self.last_optimization = datetime.utcnow()
        
        logger.info(f"Optimized market parameters based on verification data (accuracy: {prediction_accuracy:.2f})")
    
    def apply_parameters_to_generator(self, generator: SyntheticMarketGenerator) -> None:
        """
        Apply optimized parameters to a market generator.
        
        Args:
            generator: Synthetic market generator to configure
        """
        # Set base parameters
        generator.base_volatility = self.parameters.base_volatility
        generator.base_drift = self.parameters.base_drift
        generator.mean_reversion_strength = self.parameters.mean_reversion_strength
        
        # Set regime-specific parameters if generator has them
        if hasattr(generator, "regime_parameters"):
            for regime, params in self.parameters.regime_parameters.items():
                if regime in generator.regime_parameters:
                    generator.regime_parameters[regime].update(params)
        
        logger.info(f"Applied optimized parameters to market generator")
        return generator
    
    def apply_parameters_to_correlated_generator(
        self, 
        generator: CorrelatedMarketGenerator
    ) -> None:
        """
        Apply optimized parameters to a correlated market generator.
        
        Args:
            generator: Correlated market generator to configure
        """
        # Apply base parameters
        self.apply_parameters_to_generator(generator)
        
        # Set correlation parameters
        generator.default_intra_sector_correlation = self.parameters.intra_sector_correlation
        generator.default_inter_sector_correlation = self.parameters.inter_sector_correlation
        
        logger.info(f"Applied optimized parameters to correlated market generator")
        return generator
    
    def reset_to_defaults(self) -> None:
        """Reset parameters to defaults."""
        self.parameters = MarketParameters()
        self._save_to_disk()
        
        logger.info("Reset market parameters to defaults")
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """
        Generate a report on parameter optimization.
        
        Returns:
            Dictionary with optimization details
        """
        # Get current parameters
        current_params = self.parameters.to_dict()
        
        # Extract optimization history
        history = current_params.get("optimization_history", [])
        
        # Calculate accuracy over time
        timestamps = []
        accuracy_values = []
        
        for record in history:
            timestamps.append(record["timestamp"])
            accuracy_values.append(record["overall_accuracy"])
        
        # Calculate parameter changes over time
        parameter_changes = {}
        
        if len(history) > 0:
            # Get parameters that have been adjusted
            all_params = set()
            for record in history:
                for param, details in record.get("adjustments", {}).items():
                    all_params.add(param)
            
            # Track changes for each parameter
            for param in all_params:
                parameter_changes[param] = []
                
                for record in history:
                    if param in record.get("adjustments", {}):
                        adjustment = record["adjustments"][param]
                        parameter_changes[param].append({
                            "timestamp": record["timestamp"],
                            "value": adjustment.get("after", None)
                        })
        
        return {
            "current_parameters": current_params,
            "optimization_count": self.optimization_count,
            "last_optimization": self.last_optimization.isoformat() if self.last_optimization else None,
            "accuracy_history": {
                "timestamps": timestamps,
                "values": accuracy_values
            },
            "parameter_history": parameter_changes,
            "learning_rate": self.learning_rate,
            "optimization_enabled": self.optimization_enabled
        }
    
    def _save_to_disk(self) -> None:
        """Save parameters to disk."""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Prepare data for serialization
            data = self.parameters.to_dict()
            
            # Write to file
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved market parameters to {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Error saving market parameters: {str(e)}")
    
    def _load_from_disk(self) -> None:
        """Load parameters from disk."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                # Load parameters
                self.parameters = MarketParameters.from_dict(data)
                
                logger.info(f"Loaded market parameters from {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Error loading market parameters: {str(e)}")
            # Keep default parameters
            self.parameters = MarketParameters()


# Singleton instance
_parameter_optimizer = None


def get_parameter_optimizer() -> ParameterOptimizer:
    """
    Get the singleton instance of the parameter optimizer.
    
    Returns:
        ParameterOptimizer instance
    """
    global _parameter_optimizer
    
    if _parameter_optimizer is None:
        _parameter_optimizer = ParameterOptimizer()
    
    return _parameter_optimizer


if __name__ == "__main__":
    # Simple test of the parameter optimizer
    optimizer = get_parameter_optimizer()
    report = optimizer.generate_optimization_report()
    print(json.dumps(report, indent=2))
