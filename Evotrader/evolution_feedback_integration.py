#!/usr/bin/env python3
"""
Evolution Feedback Integration

This module integrates live performance feedback into the evolution process,
creating a closed-loop system that learns from real-world trading results.
"""

import os
import json
import yaml
import logging
import random
import datetime
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/evolution_feedback.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('evolution_feedback')

class EvolutionFeedbackIntegrator:
    """
    Integrates live performance feedback into the evolutionary process.
    """
    
    def __init__(self, 
                 bias_config_path: str = "config/evolution_bias.json",
                 config_path: str = "forex_evotrader_config.yaml"):
        """
        Initialize the feedback integrator.
        
        Args:
            bias_config_path: Path to the evolution bias configuration
            config_path: Path to the EvoTrader configuration
        """
        self.bias_config_path = bias_config_path
        self.config_path = config_path
        
        # Load bias configuration if it exists
        self.bias_config = self._load_bias_config()
        
        # Load EvoTrader configuration
        self.config = self._load_config()
        
        # Extract evolution parameters
        self.evolution_params = self.config.get('evolution', {})
        
    def _load_bias_config(self) -> Optional[Dict[str, Any]]:
        """Load evolution bias configuration."""
        if os.path.exists(self.bias_config_path):
            try:
                with open(self.bias_config_path, 'r') as f:
                    bias_config = json.load(f)
                
                # Check if the bias config is recent (within 14 days)
                generated = datetime.datetime.fromisoformat(bias_config.get('generated', '2000-01-01'))
                age_days = (datetime.datetime.now() - generated).days
                
                if age_days > 14:
                    logger.warning(f"Evolution bias config is {age_days} days old, consider regenerating")
                
                return bias_config
            except Exception as e:
                logger.error(f"Error loading bias config: {e}")
        
        logger.warning(f"Bias config file {self.bias_config_path} not found")
        return None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load EvoTrader configuration."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        logger.warning(f"Config file {self.config_path} not found, using defaults")
        return {'evolution': {}}
        
    def modify_evolution_parameters(self) -> Dict[str, Any]:
        """
        Modify evolution parameters based on feedback from live performance.
        
        Returns:
            Modified evolution parameters
        """
        if not self.bias_config:
            logger.warning("No bias configuration available, using original parameters")
            return self.evolution_params
        
        # Create a copy of the original parameters
        modified_params = self.evolution_params.copy()
        
        # Get successful and unsuccessful patterns
        successful = self.bias_config.get('successful_patterns', {})
        unsuccessful = self.bias_config.get('unsuccessful_patterns', {})
        
        # 1. Modify indicator probabilities
        if 'indicator_weights' in modified_params:
            self._modify_indicator_weights(modified_params, successful, unsuccessful)
        
        # 2. Modify parameter ranges
        if 'parameter_ranges' in modified_params:
            self._modify_parameter_ranges(modified_params, successful)
        
        # 3. Modify timeframe preferences
        if 'timeframe_weights' in modified_params:
            self._modify_timeframe_weights(modified_params, successful)
        
        # 4. Adjust mutation rates based on success patterns
        if 'mutation_rates' in modified_params:
            self._adjust_mutation_rates(modified_params, successful, unsuccessful)
        
        # Save the modified parameters
        self._save_modified_parameters(modified_params)
        
        return modified_params
        
    def _modify_indicator_weights(self, 
                                  params: Dict[str, Any], 
                                  successful: Dict[str, Any], 
                                  unsuccessful: Dict[str, Any]):
        """
        Modify indicator selection weights based on successful patterns.
        
        Args:
            params: Evolution parameters to modify
            successful: Successful patterns
            unsuccessful: Unsuccessful patterns
        """
        indicator_weights = params['indicator_weights']
        successful_indicators = successful.get('indicators', {})
        unsuccessful_indicators = unsuccessful.get('indicators', {})
        
        # Adjust weights
        for indicator, weight in indicator_weights.items():
            # Boost weight for successful indicators
            if indicator in successful_indicators:
                success_ratio = successful_indicators[indicator]
                boost = 1.0 + min(0.5, success_ratio * 2)  # Cap boost at 50%
                indicator_weights[indicator] = weight * boost
                logger.debug(f"Boosted {indicator} weight from {weight:.2f} to {indicator_weights[indicator]:.2f}")
            
            # Reduce weight for unsuccessful indicators
            if indicator in unsuccessful_indicators:
                failure_ratio = unsuccessful_indicators[indicator]
                reduction = max(0.5, 1.0 - failure_ratio)  # Cap reduction at 50%
                indicator_weights[indicator] = weight * reduction
                logger.debug(f"Reduced {indicator} weight from {weight:.2f} to {indicator_weights[indicator]:.2f}")
        
        # Normalize weights
        total_weight = sum(indicator_weights.values())
        for indicator in indicator_weights:
            indicator_weights[indicator] /= total_weight
        
        params['indicator_weights'] = indicator_weights
        
    def _modify_parameter_ranges(self, 
                                params: Dict[str, Any], 
                                successful: Dict[str, Any]):
        """
        Modify parameter ranges based on successful patterns.
        
        Args:
            params: Evolution parameters to modify
            successful: Successful patterns
        """
        parameter_ranges = params['parameter_ranges']
        successful_params = successful.get('parameters', {})
        
        for param, range_info in parameter_ranges.items():
            if param in successful_params:
                param_stats = successful_params[param]
                
                # Don't modify if we don't have enough data
                if param_stats.get('count', 0) < 3:
                    continue
                
                # Apply a soft bias - narrow the range toward successful values
                # but don't completely override the original range
                current_min = range_info.get('min', 0)
                current_max = range_info.get('max', 100)
                
                successful_min = param_stats.get('min', current_min)
                successful_max = param_stats.get('max', current_max)
                successful_avg = param_stats.get('avg', (current_min + current_max) / 2)
                
                # Calculate new ranges with bias toward successful values
                # but don't narrow more than 25% from each side
                range_width = current_max - current_min
                max_narrowing = range_width * 0.25
                
                new_min = max(current_min, min(successful_min, current_min + max_narrowing))
                new_max = min(current_max, max(successful_max, current_max - max_narrowing))
                
                # Ensure the range remains reasonable
                if new_min >= new_max:
                    # Expand around the successful average
                    mid = successful_avg
                    new_min = mid - range_width / 4
                    new_max = mid + range_width / 4
                
                # Update the range
                range_info['min'] = new_min
                range_info['max'] = new_max
                
                logger.debug(f"Adjusted {param} range from [{current_min}, {current_max}] to [{new_min}, {new_max}]")
        
        params['parameter_ranges'] = parameter_ranges
        
    def _modify_timeframe_weights(self, 
                                 params: Dict[str, Any], 
                                 successful: Dict[str, Any]):
        """
        Modify timeframe selection weights based on successful patterns.
        
        Args:
            params: Evolution parameters to modify
            successful: Successful patterns
        """
        timeframe_weights = params.get('timeframe_weights', {})
        successful_timeframes = successful.get('timeframes', {})
        
        if not timeframe_weights or not successful_timeframes:
            return
        
        # Adjust weights
        for timeframe, weight in timeframe_weights.items():
            if timeframe in successful_timeframes:
                success_ratio = successful_timeframes[timeframe]
                boost = 1.0 + min(0.5, success_ratio * 2)  # Cap boost at 50%
                timeframe_weights[timeframe] = weight * boost
                logger.debug(f"Boosted {timeframe} weight from {weight:.2f} to {timeframe_weights[timeframe]:.2f}")
        
        # Normalize weights
        total_weight = sum(timeframe_weights.values())
        for timeframe in timeframe_weights:
            timeframe_weights[timeframe] /= total_weight
        
        params['timeframe_weights'] = timeframe_weights
        
    def _adjust_mutation_rates(self, 
                              params: Dict[str, Any], 
                              successful: Dict[str, Any], 
                              unsuccessful: Dict[str, Any]):
        """
        Adjust mutation rates based on successful patterns.
        
        Args:
            params: Evolution parameters to modify
            successful: Successful patterns
            unsuccessful: Unsuccessful patterns
        """
        mutation_rates = params['mutation_rates']
        
        # If we have a lot of successful strategies, reduce mutation rates
        # If we have mostly unsuccessful strategies, increase mutation rates
        successful_count = len(successful.get('indicators', {}))
        unsuccessful_count = len(unsuccessful.get('indicators', {}))
        
        if successful_count + unsuccessful_count == 0:
            return
        
        success_ratio = successful_count / (successful_count + unsuccessful_count)
        
        # Base mutation adjustment on success ratio
        # Higher success ratio = lower mutation rate (fine-tuning)
        # Lower success ratio = higher mutation rate (exploration)
        adjustment_factor = 1.0
        
        if success_ratio > 0.7:
            # Many successful strategies, fine-tune with lower mutation
            adjustment_factor = 0.8
        elif success_ratio < 0.3:
            # Few successful strategies, explore more with higher mutation
            adjustment_factor = 1.2
        
        # Apply adjustment factor
        for param, rate in mutation_rates.items():
            mutation_rates[param] = min(0.5, max(0.01, rate * adjustment_factor))
            
        logger.debug(f"Adjusted mutation rates by factor: {adjustment_factor:.2f}")
        
        params['mutation_rates'] = mutation_rates
        
    def _save_modified_parameters(self, modified_params: Dict[str, Any]):
        """
        Save the modified parameters to a file.
        
        Args:
            modified_params: Modified evolution parameters
        """
        # Create timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        os.makedirs("config/evolved_params", exist_ok=True)
        
        # Save to file
        output_path = f"config/evolved_params/evolution_params_{timestamp}.yaml"
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(modified_params, f, default_flow_style=False)
                
            logger.info(f"Saved modified parameters to {output_path}")
            
            # Also save as latest.yaml for easy access
            latest_path = "config/evolved_params/latest.yaml"
            with open(latest_path, 'w') as f:
                yaml.dump(modified_params, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Error saving modified parameters: {e}")
            
    def apply_bias_to_population(self, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply bias to a population of strategies.
        
        Args:
            population: List of strategy dictionaries
            
        Returns:
            Modified population with bias applied
        """
        if not self.bias_config:
            return population
        
        # Get successful and unsuccessful patterns
        successful = self.bias_config.get('successful_patterns', {})
        unsuccessful = self.bias_config.get('unsuccessful_patterns', {})
        
        for strategy in population:
            # Apply indicator bias
            self._apply_indicator_bias(strategy, successful, unsuccessful)
            
            # Apply parameter bias
            self._apply_parameter_bias(strategy, successful)
            
            # Apply timeframe bias
            self._apply_timeframe_bias(strategy, successful)
        
        return population
            
    def _apply_indicator_bias(self, 
                             strategy: Dict[str, Any], 
                             successful: Dict[str, Any], 
                             unsuccessful: Dict[str, Any]):
        """
        Apply indicator bias to a strategy.
        
        Args:
            strategy: Strategy to modify
            successful: Successful patterns
            unsuccessful: Unsuccessful patterns
        """
        indicators = strategy.get('indicators', [])
        successful_indicators = successful.get('indicators', {})
        unsuccessful_indicators = unsuccessful.get('indicators', {})
        
        # Add successful indicators (20% chance for each successful indicator)
        for indicator, ratio in successful_indicators.items():
            if indicator not in indicators and random.random() < ratio * 0.2:
                indicators.append(indicator)
                logger.debug(f"Added successful indicator {indicator} to strategy")
        
        # Remove unsuccessful indicators (15% chance for each unsuccessful indicator)
        for indicator, ratio in unsuccessful_indicators.items():
            if indicator in indicators and random.random() < ratio * 0.15:
                indicators.remove(indicator)
                logger.debug(f"Removed unsuccessful indicator {indicator} from strategy")
        
        strategy['indicators'] = indicators
            
    def _apply_parameter_bias(self, 
                             strategy: Dict[str, Any], 
                             successful: Dict[str, Any]):
        """
        Apply parameter bias to a strategy.
        
        Args:
            strategy: Strategy to modify
            successful: Successful patterns
        """
        parameters = strategy.get('parameters', {})
        successful_params = successful.get('parameters', {})
        
        # Adjust parameters toward successful values (10% chance for each parameter)
        for param, value in parameters.items():
            if param in successful_params and isinstance(value, (int, float)) and random.random() < 0.1:
                successful_avg = successful_params[param].get('avg', value)
                
                # Move value 30% of the way toward the successful average
                parameters[param] = value + (successful_avg - value) * 0.3
                logger.debug(f"Adjusted parameter {param} from {value} toward {successful_avg}")
        
        strategy['parameters'] = parameters
            
    def _apply_timeframe_bias(self, 
                             strategy: Dict[str, Any], 
                             successful: Dict[str, Any]):
        """
        Apply timeframe bias to a strategy.
        
        Args:
            strategy: Strategy to modify
            successful: Successful patterns
        """
        timeframe = strategy.get('timeframe')
        successful_timeframes = successful.get('timeframes', {})
        
        # 15% chance to switch to a successful timeframe
        if timeframe and random.random() < 0.15:
            # Select a timeframe based on success ratio
            if successful_timeframes:
                timeframes = list(successful_timeframes.keys())
                weights = list(successful_timeframes.values())
                
                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]
                    
                    # Select a timeframe based on weights
                    new_timeframe = random.choices(timeframes, weights=weights, k=1)[0]
                    
                    if new_timeframe != timeframe:
                        logger.debug(f"Changed timeframe from {timeframe} to {new_timeframe}")
                        strategy['timeframe'] = new_timeframe
    
    def apply_bias_to_mutation(self, 
                              strategy: Dict[str, Any], 
                              mutation_rate: float = 0.1) -> Dict[str, Any]:
        """
        Apply bias to strategy mutation.
        
        Args:
            strategy: Strategy to mutate
            mutation_rate: Base mutation rate
            
        Returns:
            Mutated strategy
        """
        if not self.bias_config:
            return strategy
        
        # Get successful patterns
        successful = self.bias_config.get('successful_patterns', {})
        successful_params = successful.get('parameters', {})
        
        parameters = strategy.get('parameters', {})
        
        # Bias mutation toward successful parameter ranges
        for param, value in parameters.items():
            if param in successful_params and isinstance(value, (int, float)) and random.random() < mutation_rate:
                param_stats = successful_params[param]
                
                # Get min and max from successful range
                min_val = param_stats.get('min')
                max_val = param_stats.get('max')
                
                if min_val is not None and max_val is not None:
                    # Mutate toward successful range
                    parameters[param] = random.uniform(min_val, max_val)
                    logger.debug(f"Mutated {param} to {parameters[param]:.4f} (within successful range)")
        
        strategy['parameters'] = parameters
        return strategy

def main():
    """Main function."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Initialize integrator
    integrator = EvolutionFeedbackIntegrator()
    
    # Modify evolution parameters
    modified_params = integrator.modify_evolution_parameters()
    
    # Print results
    print("Evolution parameters modified based on live performance feedback.")
    print(f"Modified parameters saved to: config/evolved_params/latest.yaml")
    
    # Show some key changes
    if 'indicator_weights' in modified_params:
        print("\nTop 5 indicator weights:")
        weights = modified_params['indicator_weights']
        top_indicators = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
        for indicator, weight in top_indicators:
            print(f"  {indicator}: {weight:.4f}")
    
    if 'mutation_rates' in modified_params:
        print("\nMutation rates:")
        for param, rate in modified_params['mutation_rates'].items():
            print(f"  {param}: {rate:.4f}")

if __name__ == "__main__":
    main()
