#!/usr/bin/env python3
"""
Feedback Loop Enhancements

This module provides improved learning mechanisms for the feedback loop,
addressing the fluctuations in prediction accuracy by implementing:

1. Adaptive learning rates
2. Parameter-specific update rules
3. Momentum-based learning
4. Accuracy tracking and adaptive adjustments

These enhancements build directly on our existing multi_objective_simplified.py
implementation, focusing only on the learning mechanism improvements.
"""

class AdaptiveLearningRates:
    """
    Provides adaptive learning rates for feedback loop parameter updates.
    
    This class manages learning rates that automatically adjust based on
    the observed accuracy trends and convergence patterns.
    """
    
    def __init__(self, initial_rate=0.2, min_rate=0.05, max_rate=0.4):
        """
        Initialize adaptive learning rates.
        
        Args:
            initial_rate: Starting learning rate
            min_rate: Minimum learning rate
            max_rate: Maximum learning rate
        """
        self.base_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        
        # Per-regime and per-parameter learning rates
        self.regime_rates = {}
        self.parameter_rates = {
            'trend': initial_rate,
            'volatility': initial_rate,
            'mean_reversion': initial_rate
        }
        
        # Momentum terms
        self.momentum = 0.1
        self.previous_updates = {}
        
        # History tracking
        self.accuracy_history = []
        self.adjustment_history = []
    
    def get_rate(self, regime, parameter):
        """
        Get the appropriate learning rate for a specific regime and parameter.
        
        Args:
            regime: Market regime (e.g., 'bullish')
            parameter: Parameter name (e.g., 'trend')
            
        Returns:
            Current learning rate
        """
        # Initialize regime if not present
        if regime not in self.regime_rates:
            self.regime_rates[regime] = {}
            self.previous_updates[regime] = {}
            
            for param in self.parameter_rates:
                self.regime_rates[regime][param] = self.parameter_rates[param]
                self.previous_updates[regime][param] = 0.0
        
        # Return the specific learning rate
        return self.regime_rates[regime][parameter]
    
    def update_rates(self, accuracy_trend, param_distances):
        """
        Update learning rates based on accuracy trends and parameter distances.
        
        Args:
            accuracy_trend: List of recent accuracy values
            param_distances: Dict of distances between synthetic and real parameters
        """
        self.accuracy_history.append(sum(accuracy_trend) / len(accuracy_trend) if accuracy_trend else 0)
        
        # Skip if we don't have enough history
        if len(self.accuracy_history) < 2:
            return
        
        # Analyze accuracy trend
        accuracy_improving = self.accuracy_history[-1] > self.accuracy_history[-2]
        
        # Global adjustment based on accuracy trend
        adjustment = 0.05 if accuracy_improving else -0.05
        self.adjustment_history.append(adjustment)
        
        # Update each regime and parameter rate
        for regime, params in param_distances.items():
            for param, distance in params.items():
                # Current rate
                current_rate = self.regime_rates[regime][param]
                
                # Parameter-specific adjustments
                if param == 'trend':
                    # Trend parameters need finer adjustments
                    param_factor = 0.8
                elif param == 'volatility':
                    # Volatility can handle larger adjustments
                    param_factor = 1.2
                else:
                    param_factor = 1.0
                
                # Distance-based adjustment - larger distance = higher rate
                distance_factor = min(1.5, max(0.5, distance))
                
                # Calculate new rate
                new_rate = current_rate + (adjustment * param_factor * distance_factor)
                
                # Apply limits
                new_rate = max(self.min_rate, min(self.max_rate, new_rate))
                
                # Update the rate
                self.regime_rates[regime][param] = new_rate
    
    def get_update_with_momentum(self, regime, parameter, update):
        """
        Apply momentum to parameter updates for smoother learning.
        
        Args:
            regime: Market regime
            parameter: Parameter name
            update: Raw update value
            
        Returns:
            Update value with momentum applied
        """
        if regime not in self.previous_updates:
            self.previous_updates[regime] = {}
        
        if parameter not in self.previous_updates[regime]:
            self.previous_updates[regime][parameter] = 0.0
        
        # Apply momentum
        momentum_update = (1 - self.momentum) * update + self.momentum * self.previous_updates[regime][parameter]
        
        # Store for next iteration
        self.previous_updates[regime][parameter] = momentum_update
        
        return momentum_update


class EnhancedMarketModelUpdater:
    """
    Enhanced updater for market models with adaptive learning and momentum.
    
    This class provides an improved update mechanism for the feedback loop,
    replacing the fixed learning rate approach with an adaptive system.
    """
    
    def __init__(self):
        """Initialize the enhanced market model updater."""
        self.learning_rates = AdaptiveLearningRates()
        self.iteration = 0
        self.parameter_distances = {}
    
    def calculate_parameter_distances(self, synthetic_models, real_world_params=None):
        """
        Calculate distances between synthetic and real parameter values.
        
        In a real system, we don't know the "real" values, but we can estimate
        the distance based on prediction accuracy.
        
        Args:
            synthetic_models: Current synthetic market models
            real_world_params: Real-world parameters (for demo purposes)
            
        Returns:
            Dict of normalized distances by regime and parameter
        """
        distances = {}
        
        # For each regime
        for regime, syn_params in synthetic_models.items():
            distances[regime] = {}
            
            if real_world_params and regime in real_world_params:
                # We have real values (for demo only)
                real_params = real_world_params[regime]
                
                for param, syn_value in syn_params.items():
                    if param in real_params:
                        # Calculate normalized distance (0-1 range)
                        distances[regime][param] = abs(syn_value - real_params[param])
                        
                        # Normalize by param type
                        if param == 'trend':
                            # Trend params are typically small
                            distances[regime][param] /= 0.01
                        elif param == 'volatility':
                            # Volatility is typically 0.01-0.05
                            distances[regime][param] /= 0.05
                        else:
                            # Mean reversion is typically 0.1-0.9
                            distances[regime][param] /= 0.5
                        
                        # Clamp to 0-1 range
                        distances[regime][param] = min(1.0, max(0.0, distances[regime][param]))
            else:
                # Real system approach - use defaults
                for param in syn_params:
                    distances[regime][param] = 0.5  # Neutral starting point
        
        self.parameter_distances = distances
        return distances
    
    def update_market_models(self, market_models, expected_perf, actual_perf, 
                            accuracy_history, real_world_params=None):
        """
        Update market models with enhanced learning mechanisms.
        
        Args:
            market_models: Current synthetic market models to update
            expected_perf: Expected performance metrics from optimization
            actual_perf: Actual performance metrics from testing
            accuracy_history: History of prediction accuracies
            real_world_params: Real-world parameters (for demo purposes)
            
        Returns:
            Updated market models
        """
        self.iteration += 1
        
        # Create a copy of the input models
        updated_models = {}
        for regime, params in market_models.items():
            updated_models[regime] = params.copy()
        
        # Calculate parameter distances
        self.calculate_parameter_distances(market_models, real_world_params)
        
        # Update learning rates based on accuracy trend
        accuracy_trend = accuracy_history['overall'][-3:] if len(accuracy_history['overall']) >= 3 else accuracy_history['overall']
        self.learning_rates.update_rates(accuracy_trend, self.parameter_distances)
        
        # For each regime, update parameters
        for regime in updated_models:
            if regime not in expected_perf or regime not in actual_perf:
                continue
            
            # Calculate performance ratios
            return_ratio = self._calculate_ratio(
                actual_perf[regime]['return'], 
                expected_perf[regime]['return']
            )
            
            volatility_ratio = self._calculate_ratio(
                actual_perf[regime]['max_drawdown'], 
                expected_perf[regime]['max_drawdown']
            )
            
            # Update trend parameter
            trend_rate = self.learning_rates.get_rate(regime, 'trend')
            trend_update = self._calculate_trend_update(regime, return_ratio)
            trend_update = self.learning_rates.get_update_with_momentum(regime, 'trend', trend_update)
            updated_models[regime]['trend'] *= (1.0 + trend_update * trend_rate)
            
            # Update volatility parameter
            vol_rate = self.learning_rates.get_rate(regime, 'volatility')
            vol_update = self._calculate_volatility_update(regime, volatility_ratio)
            vol_update = self.learning_rates.get_update_with_momentum(regime, 'volatility', vol_update)
            updated_models[regime]['volatility'] *= (1.0 + vol_update * vol_rate)
            
            # Update mean reversion parameter
            mr_rate = self.learning_rates.get_rate(regime, 'mean_reversion')
            mr_update = self._calculate_mean_reversion_update(regime, return_ratio, volatility_ratio)
            mr_update = self.learning_rates.get_update_with_momentum(regime, 'mean_reversion', mr_update)
            updated_models[regime]['mean_reversion'] *= (1.0 + mr_update * mr_rate)
            
            # Apply parameter constraints
            self._apply_constraints(updated_models[regime])
        
        return updated_models
    
    def _calculate_ratio(self, actual, expected):
        """Calculate ratio between actual and expected values, handling edge cases."""
        if expected == 0:
            return 1.0 if actual == 0 else (2.0 if actual > 0 else 0.5)
        return actual / expected
    
    def _calculate_trend_update(self, regime, return_ratio):
        """Calculate trend parameter update based on return ratio."""
        # Different logic based on regime
        if regime == 'bullish':
            # In bullish regime, positive trend is expected
            if return_ratio > 1.1:
                return return_ratio - 1.0  # Increase trend if returns are higher
            elif return_ratio < 0.9:
                return return_ratio - 1.0  # Decrease trend if returns are lower
            
        elif regime == 'bearish':
            # In bearish regime, negative trend is expected
            if return_ratio > 1.1:
                return -(return_ratio - 1.0)  # Make trend more negative
            elif return_ratio < 0.9:
                return -(return_ratio - 1.0)  # Make trend less negative
        
        else:
            # Sideways and volatile regimes
            if abs(return_ratio - 1.0) > 0.1:
                return (return_ratio - 1.0) * 0.5  # Smaller adjustments
        
        return 0.0  # No update needed
    
    def _calculate_volatility_update(self, regime, volatility_ratio):
        """Calculate volatility parameter update based on drawdown ratio."""
        if volatility_ratio > 1.1:
            # Actual drawdowns higher than expected - increase volatility
            return volatility_ratio - 1.0
        elif volatility_ratio < 0.9:
            # Actual drawdowns lower than expected - decrease volatility
            return volatility_ratio - 1.0
        
        return 0.0  # No update needed
    
    def _calculate_mean_reversion_update(self, regime, return_ratio, volatility_ratio):
        """Calculate mean reversion update based on combined factors."""
        # Different logic by regime
        if regime in ['sideways', 'volatile']:
            # Mean reversion is more important in these regimes
            factor = 1.5
        else:
            factor = 1.0
            
        # Combined update logic
        if return_ratio < 0.9 and volatility_ratio > 1.1:
            # Lower returns and higher drawdowns suggest more mean reversion
            return 0.2 * factor
        elif return_ratio > 1.1 and volatility_ratio < 0.9:
            # Higher returns and lower drawdowns suggest less mean reversion
            return -0.2 * factor
            
        return 0.0  # No update needed
    
    def _apply_constraints(self, params):
        """Apply constraints to keep parameters in reasonable ranges."""
        # Trend constraints (tighter for sideways, wider for bullish/bearish)
        params['trend'] = max(-0.01, min(0.01, params['trend']))
        
        # Volatility constraints
        params['volatility'] = max(0.005, min(0.05, params['volatility']))
        
        # Mean reversion constraints
        params['mean_reversion'] = max(0.05, min(0.9, params['mean_reversion']))


def apply_enhanced_updates(feedback_loop, expected_perf, actual_perf):
    """
    Apply enhanced updates to a feedback loop instance.
    
    This function can be used to enhance an existing feedback loop with
    the improved learning mechanism.
    
    Args:
        feedback_loop: Instance of MultiObjectiveFeedbackLoop
        expected_perf: Expected performance data
        actual_perf: Actual performance data
        
    Returns:
        Updated market models
    """
    # Create updater if not already attached
    if not hasattr(feedback_loop, '_enhanced_updater'):
        feedback_loop._enhanced_updater = EnhancedMarketModelUpdater()
    
    # Apply updates
    updated_models = feedback_loop._enhanced_updater.update_market_models(
        feedback_loop.market_models,
        expected_perf,
        actual_perf,
        feedback_loop.accuracy_history,
        feedback_loop.real_world_params if hasattr(feedback_loop, 'real_world_params') else None
    )
    
    # Apply to feedback loop
    for regime, params in updated_models.items():
        feedback_loop.market_models[regime] = params
    
    return updated_models


# Example usage with the existing feedback loop
def patch_feedback_loop(feedback_loop_instance):
    """
    Patch an existing feedback loop with the enhanced update mechanism.
    
    Args:
        feedback_loop_instance: Instance of MultiObjectiveFeedbackLoop
        
    Returns:
        Updated feedback loop instance with the enhanced update mechanism
    """
    # Store original update method for reference
    feedback_loop_instance._original_update_market_models = feedback_loop_instance.update_market_models
    
    # Replace with enhanced version
    def enhanced_update_market_models(self):
        """Enhanced version of the update_market_models method."""
        if not self.expected_perf or not self.actual_perf:
            print("No performance data available for model updates.")
            return
        
        print("\nUpdating market models with enhanced learning mechanism...")
        
        # Store current parameters for tracking
        self.param_history.append({})
        for regime, params in self.market_models.items():
            self.param_history[-1][regime] = params.copy()
        
        # Get the most recent expected and actual performance
        expected = self.expected_perf[-1]
        actual = self.actual_perf[-1]
        
        # Apply enhanced updates
        apply_enhanced_updates(self, expected, actual)
        
        print("Updated market models:")
        for regime, params in self.market_models.items():
            print(f"  {regime}: {params}")
    
    # Attach new method to the instance
    feedback_loop_instance.update_market_models = enhanced_update_market_models.__get__(feedback_loop_instance)
    
    return feedback_loop_instance
