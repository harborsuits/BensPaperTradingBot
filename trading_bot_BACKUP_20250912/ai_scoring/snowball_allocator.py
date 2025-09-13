"""
Snowball Dynamic Capital Allocation System

This module implements a profit-based "snowball" capital allocation strategy
that lets winning strategies compound by reinvesting their profits.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)

class SnowballAllocator:
    """
    Implements the Snowball capital allocation strategy, which reinvests profits
    back into strategies that are performing well, allowing them to compound.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the Snowball allocation system with configuration parameters.
        
        Args:
            config: Dictionary of configuration parameters
        """
        config = config or {}
        
        # Configuration parameters with defaults
        self.rebalance_frequency = config.get('rebalance_frequency', 'daily')
        self.snowball_reinvestment_ratio = config.get('snowball_reinvestment_ratio', 0.5)
        self.min_weight = config.get('min_weight', 0.05)
        self.max_weight = config.get('max_weight', 0.90)
        self.normalization_method = config.get('normalization_method', 'simple')
        self.smoothing_factor = config.get('smoothing_factor', 0.2)
        
        # Track historical data
        self.allocation_history = {}
        self.equity_history = {}
        self.profit_history = {}
        self.last_rebalance_time = None
        
        # Manual overrides
        self.manual_allocation_weights = {}
        self.manual_weight_expiry = {}
        
        logger.info(f"Initialized SnowballAllocator with reinvestment ratio: {self.snowball_reinvestment_ratio}, "
                    f"min_weight: {self.min_weight}, max_weight: {self.max_weight}")
    
    def update_allocations(self, 
                           current_weights: Dict[str, float], 
                           profit_data: Dict[str, float],
                           total_equity: float,
                           timestamp: Optional[datetime] = None) -> Dict[str, float]:
        """
        Update strategy allocations based on their profits using the Snowball method.
        
        Args:
            current_weights: Dictionary of current strategy allocations (strategy -> weight)
            profit_data: Dictionary of strategy profits (strategy -> profit amount)
            total_equity: Total portfolio equity
            timestamp: Timestamp for this update (defaults to now)
            
        Returns:
            Dictionary of updated weights (strategy -> new weight)
        """
        timestamp = timestamp or datetime.now()
        
        # Determine if we should rebalance based on frequency
        if not self._should_rebalance(timestamp):
            return current_weights
        
        # Store data in history
        self._update_history(current_weights, profit_data, total_equity, timestamp)
        
        # Apply manual overrides if they exist and haven't expired
        weights_to_update = self._apply_manual_overrides(current_weights, timestamp)
        
        # Calculate profit ratios
        profit_ratios = {
            strategy: profit / total_equity if total_equity > 0 else 0
            for strategy, profit in profit_data.items()
        }
        
        # Update weights based on profit reinvestment
        updated_weights = {}
        for strategy, current_weight in weights_to_update.items():
            profit_ratio = profit_ratios.get(strategy, 0)
            
            # Apply the snowball formula
            new_weight = current_weight + (self.snowball_reinvestment_ratio * profit_ratio)
            
            # Clamp to min/max
            new_weight = max(self.min_weight, min(new_weight, self.max_weight))
            
            updated_weights[strategy] = new_weight
        
        # Normalize weights
        normalized_weights = self._normalize_weights(updated_weights)
        
        # Apply smoothing with previous weights if enabled
        if self.smoothing_factor > 0:
            for strategy in normalized_weights:
                old_weight = current_weights.get(strategy, 0.0)
                normalized_weights[strategy] = (
                    self.smoothing_factor * old_weight + 
                    (1 - self.smoothing_factor) * normalized_weights[strategy]
                )
            
            # Re-normalize after smoothing
            normalized_weights = self._normalize_weights(normalized_weights)
        
        # Update last rebalance time
        self.last_rebalance_time = timestamp
        
        return normalized_weights
    
    def _should_rebalance(self, timestamp: datetime) -> bool:
        """
        Determine if we should rebalance based on the frequency setting.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Boolean indicating whether rebalancing should occur
        """
        if self.last_rebalance_time is None:
            return True
            
        if self.rebalance_frequency == 'daily':
            return (timestamp.date() > self.last_rebalance_time.date())
        elif self.rebalance_frequency == 'weekly':
            days_diff = (timestamp - self.last_rebalance_time).days
            return days_diff >= 7
        elif self.rebalance_frequency == 'monthly':
            # Different month or year
            return (timestamp.month != self.last_rebalance_time.month or 
                    timestamp.year != self.last_rebalance_time.year)
        else:
            # Default to daily if invalid frequency
            logger.warning(f"Invalid rebalance_frequency: {self.rebalance_frequency}, defaulting to daily")
            return (timestamp.date() > self.last_rebalance_time.date())
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.0 using the specified method.
        
        Args:
            weights: Dictionary of unnormalized weights
            
        Returns:
            Dictionary of normalized weights
        """
        if not weights:
            return {}
            
        if self.normalization_method == 'softmax':
            # Temperature parameter could be configurable too
            temperature = 1.0
            values = np.array(list(weights.values())) / temperature
            exp_values = np.exp(values - np.max(values))  # For numerical stability
            softmax_values = exp_values / np.sum(exp_values)
            
            return {
                strategy: float(value)
                for strategy, value in zip(weights.keys(), softmax_values)
            }
        else:  # 'simple' (default)
            total = sum(weights.values())
            if total <= 0:
                # Equal weights if total is zero or negative
                equal_weight = 1.0 / len(weights)
                return {strategy: equal_weight for strategy in weights}
                
            return {
                strategy: weight / total
                for strategy, weight in weights.items()
            }
    
    def _update_history(self, 
                        weights: Dict[str, float], 
                        profits: Dict[str, float],
                        equity: float,
                        timestamp: datetime) -> None:
        """
        Update historical data for tracking purposes.
        
        Args:
            weights: Current weights
            profits: Current profits
            equity: Current equity
            timestamp: Timestamp
        """
        date_str = timestamp.strftime("%Y-%m-%d")
        
        # Store in history
        for strategy, weight in weights.items():
            if strategy not in self.allocation_history:
                self.allocation_history[strategy] = []
            
            self.allocation_history[strategy].append({
                'date': date_str,
                'weight': weight
            })
        
        for strategy, profit in profits.items():
            if strategy not in self.profit_history:
                self.profit_history[strategy] = []
            
            self.profit_history[strategy].append({
                'date': date_str,
                'profit': profit
            })
        
        # Store equity
        if 'total' not in self.equity_history:
            self.equity_history['total'] = []
        
        self.equity_history['total'].append({
            'date': date_str,
            'equity': equity
        })
    
    def set_manual_allocation(self, 
                              strategy: str, 
                              weight: float, 
                              duration_secs: int = 86400) -> None:
        """
        Set a manual allocation weight for a strategy that overrides automatic allocation.
        
        Args:
            strategy: Strategy name
            weight: Weight to assign (will be normalized with other strategies)
            duration_secs: How long the override should last, in seconds
        """
        expiry_time = datetime.now() + timedelta(seconds=duration_secs)
        self.manual_allocation_weights[strategy] = weight
        self.manual_weight_expiry[strategy] = expiry_time
        
        logger.info(f"Set manual allocation for {strategy} to {weight} until {expiry_time}")
    
    def clear_manual_allocation(self, strategy: str = None) -> None:
        """
        Clear manual allocation for a strategy or all strategies.
        
        Args:
            strategy: Strategy to clear, or None to clear all
        """
        if strategy is None:
            self.manual_allocation_weights = {}
            self.manual_weight_expiry = {}
            logger.info("Cleared all manual allocations")
        elif strategy in self.manual_allocation_weights:
            del self.manual_allocation_weights[strategy]
            if strategy in self.manual_weight_expiry:
                del self.manual_weight_expiry[strategy]
            logger.info(f"Cleared manual allocation for {strategy}")
    
    def _apply_manual_overrides(self, 
                               current_weights: Dict[str, float],
                               timestamp: datetime) -> Dict[str, float]:
        """
        Apply manual weight overrides if they haven't expired.
        
        Args:
            current_weights: Current weights dictionary
            timestamp: Current timestamp
            
        Returns:
            Dictionary with overrides applied where applicable
        """
        # Copy the weights to avoid modifying the input
        weights = current_weights.copy()
        
        # Apply manual overrides that haven't expired
        for strategy, expiry_time in list(self.manual_weight_expiry.items()):
            if timestamp >= expiry_time:
                # Override has expired, remove it
                self.clear_manual_allocation(strategy)
            elif strategy in self.manual_allocation_weights:
                # Apply the override
                weights[strategy] = self.manual_allocation_weights[strategy]
        
        return weights
    
    def get_allocation_history(self, 
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> Dict[str, List[Dict]]:
        """
        Get historical allocation data for analysis.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Dictionary of allocation history per strategy
        """
        if not start_date and not end_date:
            return self.allocation_history
            
        filtered_history = {}
        for strategy, history in self.allocation_history.items():
            filtered = []
            for entry in history:
                entry_date = datetime.strptime(entry['date'], "%Y-%m-%d")
                if start_date and entry_date < start_date:
                    continue
                if end_date and entry_date > end_date:
                    continue
                filtered.append(entry)
            
            if filtered:
                filtered_history[strategy] = filtered
                
        return filtered_history
