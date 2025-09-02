"""
Additional methods for DynamicCapitalAllocator class
"""

def _reallocate_capital(self):
    """
    Reallocate capital across strategies based on performance and market regimes
    """
    if not self.strategy_allocations:
        logger.debug("No strategies to allocate capital to")
        return
        
    # Record current allocations for comparison
    old_allocations = self.strategy_allocations.copy()
    
    # Base allocations on recent performance
    performance_allocations = self._calculate_performance_based_allocations()
    
    # Adjust for volatility
    risk_adjusted_allocations = self._adjust_allocations_for_volatility(performance_allocations)
    
    # Apply smoothing to prevent sudden allocation changes
    for strategy_id, allocation in risk_adjusted_allocations.items():
        if strategy_id in old_allocations:
            old_alloc = old_allocations[strategy_id]
            max_change = old_alloc * self.allocation_smoothing
            
            # Limit change to smoothing factor
            if allocation > old_alloc + max_change:
                allocation = old_alloc + max_change
            elif allocation < old_alloc - max_change:
                allocation = max(self.min_strategy_allocation, old_alloc - max_change)
        
        # Apply min/max constraints
        allocation = max(self.min_strategy_allocation, min(allocation, self.max_strategy_allocation))
        self.strategy_allocations[strategy_id] = allocation
    
    # Normalize allocations to ensure they sum to <= 1.0
    self._normalize_allocations()
    
    # Record allocation history
    self.allocation_history.append({
        'timestamp': datetime.now(),
        'allocations': self.strategy_allocations.copy(),
        'equity': self.current_capital
    })
    
    # Update last reallocation timestamp
    self.last_reallocation = datetime.now()
    
    # Log changes
    for strategy_id, allocation in self.strategy_allocations.items():
        if strategy_id in old_allocations and abs(allocation - old_allocations[strategy_id]) > 0.01:
            logger.info(f"Allocation for {strategy_id} changed from {old_allocations[strategy_id]:.2%} to {allocation:.2%}")
        elif strategy_id not in old_allocations:
            logger.info(f"New allocation for {strategy_id}: {allocation:.2%}")
            
    # Publish allocation update event
    self.event_bus.publish(Event(
        event_type=EventType.CAPITAL_ALLOCATION_UPDATED,
        data={
            'allocations': self.strategy_allocations.copy(),
            'timestamp': datetime.now()
        }
    ))

def _apply_snowball_strategy(self, profit: float):
    """
    Apply snowball strategy by reinvesting profits proportionally to strategy performance
    
    Args:
        profit: Current profit amount
    """
    if not self.strategy_allocations or not self.use_snowball:
        return
        
    # Calculate excess profit above threshold
    threshold_profit = self.initial_capital * self.snowball_threshold
    excess_profit = max(0, profit - threshold_profit)
    
    if excess_profit <= 0:
        return
        
    # Calculate amount to reallocate based on snowball factor
    snowball_amount = excess_profit * self.snowball_allocation_factor
    
    # Get performance-weighted allocations of the snowball amount
    weighted_allocations = {}
    total_weight = 0
    
    # Calculate weights based on strategy performance
    for strategy_id in self.strategy_allocations:
        # Default weight
        weight = 1.0
        
        if strategy_id in self.strategy_performance and self.strategy_performance[strategy_id]:
            # Use recent performance to determine weight
            recent_perf = self.strategy_performance[strategy_id][-1]['metrics']
            profit_factor = recent_perf.get('profit_factor', 1.0)
            sharpe = recent_perf.get('sharpe_ratio', 0.5)
            win_rate = recent_perf.get('win_rate', 0.5)
            
            # Calculate performance score
            perf_score = (profit_factor * 0.4) + (sharpe * 0.4) + (win_rate * 0.2)
            weight = max(0.5, perf_score)
        
        weighted_allocations[strategy_id] = weight
        total_weight += weight
    
    # Normalize weights and calculate additional allocation
    if total_weight > 0:
        for strategy_id, weight in weighted_allocations.items():
            weighted_allocations[strategy_id] = (weight / total_weight) * snowball_amount
            
        # Add snowball allocation to strategy allocations
        logger.info(f"Applying snowball strategy with {snowball_amount:.2f} additional capital")
        
        for strategy_id, additional in weighted_allocations.items():
            if additional > 0:
                logger.info(f"  {strategy_id}: +{additional:.2f}")
                
        # Trigger reallocation to incorporate snowball changes
        self._reallocate_capital()

def _update_allocations_for_regime(self, symbol: str, regime: str):
    """
    Update strategy allocations based on market regime changes
    
    Args:
        symbol: Symbol with regime change
        regime: New market regime
    """
    if not hasattr(self, 'strategy_types') or not self.strategy_types:
        return
        
    # Check if any strategies trade this symbol
    strategies_for_symbol = []
    for strategy_id in self.strategy_allocations:
        # This is a simplification - in real implementation we'd have a way to 
        # check if a strategy trades a particular symbol
        strategies_for_symbol.append(strategy_id)
    
    if not strategies_for_symbol:
        return
        
    # Adjust allocations based on regime and strategy type
    for strategy_id in strategies_for_symbol:
        strategy_type = self.strategy_types.get(strategy_id, 'default')
        
        # Get regime adjustment factor
        if strategy_type in self.regime_allocations and regime in self.regime_allocations[strategy_type]:
            regime_factor = self.regime_allocations[strategy_type][regime]
        else:
            # Use default adjustment
            regime_factor = self.regime_allocations['default'].get(regime, 1.0)
        
        # Apply adjustment to allocation
        if strategy_id in self.strategy_allocations:
            current_allocation = self.strategy_allocations[strategy_id]
            adjusted_allocation = current_allocation * regime_factor
            
            # Apply min/max constraints
            adjusted_allocation = max(self.min_strategy_allocation, 
                                    min(adjusted_allocation, self.max_strategy_allocation))
            
            # Update allocation if significant change
            if abs(adjusted_allocation - current_allocation) > 0.01:
                self.strategy_allocations[strategy_id] = adjusted_allocation
                logger.info(f"Adjusted allocation for {strategy_id} from {current_allocation:.2%} to "
                           f"{adjusted_allocation:.2%} due to {regime} regime for {symbol}")
    
    # Normalize allocations if needed
    self._normalize_allocations()

def _update_strategy_allocation(self, strategy_id: str, performance_metrics: Dict[str, Any]):
    """
    Update allocation for a specific strategy based on new performance metrics
    
    Args:
        strategy_id: Strategy identifier
        performance_metrics: Performance metrics dictionary
    """
    if not self.use_adaptive_allocation or strategy_id not in self.strategy_allocations:
        return
        
    current_allocation = self.strategy_allocations[strategy_id]
    adjustment_factor = 1.0
    
    # Apply performance-based adjustments
    profit_factor = performance_metrics.get('profit_factor', 1.0)
    win_rate = performance_metrics.get('win_rate', 0.5)
    sharpe = performance_metrics.get('sharpe_ratio', 0.0)
    drawdown = performance_metrics.get('max_drawdown', 0.0)
    
    # Profit factor adjustment
    if profit_factor >= 2.0:
        adjustment_factor *= 1.2  # Increase allocation for high profit factor
    elif profit_factor < 1.0:
        adjustment_factor *= 0.8  # Decrease allocation for losing strategies
    
    # Win rate adjustment
    if win_rate >= 0.6:
        adjustment_factor *= 1.1  # Boost for high win rate
    elif win_rate < 0.4:
        adjustment_factor *= 0.9  # Reduce for low win rate
    
    # Sharpe ratio adjustment
    if sharpe >= 1.5:
        adjustment_factor *= 1.15  # Boost for high Sharpe
    elif sharpe < 0.5:
        adjustment_factor *= 0.85  # Reduce for low Sharpe
    
    # Drawdown adjustment (more conservative with higher drawdowns)
    if drawdown > 0.15:  # 15% drawdown
        adjustment_factor *= 0.8  # Significant reduction
    elif drawdown > 0.1:  # 10% drawdown
        adjustment_factor *= 0.9  # Moderate reduction
    
    # Calculate adjusted allocation
    adjusted_allocation = current_allocation * adjustment_factor
    
    # Apply min/max constraints
    adjusted_allocation = max(self.min_strategy_allocation, 
                            min(adjusted_allocation, self.max_strategy_allocation))
    
    # Apply smoothing - limit change to max percentage
    max_change = current_allocation * self.allocation_smoothing
    if adjusted_allocation > current_allocation + max_change:
        adjusted_allocation = current_allocation + max_change
    elif adjusted_allocation < current_allocation - max_change:
        adjusted_allocation = max(self.min_strategy_allocation, current_allocation - max_change)
    
    # Update if significant change
    if abs(adjusted_allocation - current_allocation) > 0.01:
        self.strategy_allocations[strategy_id] = adjusted_allocation
        logger.info(f"Updated allocation for {strategy_id} from {current_allocation:.2%} to "
                   f"{adjusted_allocation:.2%} based on performance metrics")
        
        # Consider renormalizing all allocations
        self._normalize_allocations()

def _calculate_performance_based_allocations(self) -> Dict[str, float]:
    """
    Calculate allocations based on strategy performance
    
    Returns:
        Dictionary of strategy_id -> allocation percentage
    """
    performance_allocations = {}
    
    for strategy_id in self.strategy_allocations:
        # Start with default allocation
        allocation = self.default_allocation
        
        # Adjust based on performance if available
        if strategy_id in self.strategy_performance and self.strategy_performance[strategy_id]:
            # Get most recent performance metrics
            recent_metrics = self.strategy_performance[strategy_id][-1]['metrics']
            
            # Profitable strategy bonus
            profit_factor = recent_metrics.get('profit_factor', 1.0)
            if profit_factor > 1.5:
                allocation *= (1.0 + self.winning_bonus)  # e.g. 20% bonus
            elif profit_factor < 0.8:
                allocation *= (1.0 - self.losing_penalty)  # e.g. 50% penalty
            
            # Sharpe ratio adjustment
            sharpe = recent_metrics.get('sharpe_ratio', 0.0)
            if sharpe > 1.5:
                allocation *= 1.2  # 20% bonus for high Sharpe
            elif sharpe < 0.5 and sharpe > 0:
                allocation *= 0.9  # 10% penalty for low Sharpe
            
            # Drawdown penalty
            drawdown = recent_metrics.get('max_drawdown', 0.0)
            if drawdown > 0.2:  # 20% drawdown
                allocation *= 0.7  # 30% penalty
            elif drawdown > 0.15:  # 15% drawdown
                allocation *= 0.85  # 15% penalty
        
        # Store calculated allocation
        performance_allocations[strategy_id] = allocation
    
    return performance_allocations

def _adjust_allocations_for_volatility(self, allocations: Dict[str, float]) -> Dict[str, float]:
    """
    Adjust allocations based on strategy volatility to target consistent risk
    
    Args:
        allocations: Initial allocations dictionary
        
    Returns:
        Volatility-adjusted allocations
    """
    adjusted_allocations = allocations.copy()
    
    # Only adjust if we have volatility data
    if not self.strategy_volatility:
        return adjusted_allocations
    
    # Calculate average volatility
    volatilities = list(self.strategy_volatility.values())
    if not volatilities:
        return adjusted_allocations
        
    avg_volatility = sum(volatilities) / len(volatilities)
    if avg_volatility == 0:
        return adjusted_allocations
    
    # Adjust allocations inversely proportional to volatility
    for strategy_id, allocation in adjusted_allocations.items():
        if strategy_id in self.strategy_volatility and self.strategy_volatility[strategy_id] > 0:
            vol_ratio = avg_volatility / self.strategy_volatility[strategy_id]
            
            # Higher volatility -> lower allocation to maintain consistent risk
            # Lower volatility -> higher allocation
            adjusted_allocations[strategy_id] = allocation * vol_ratio
    
    return adjusted_allocations

def _normalize_allocations(self):
    """
    Normalize strategy allocations to ensure they sum to <= 1.0
    """
    if not self.strategy_allocations:
        return
        
    total_allocation = sum(self.strategy_allocations.values())
    
    if total_allocation > 1.0:
        # Rescale proportionally
        for strategy_id in self.strategy_allocations:
            self.strategy_allocations[strategy_id] /= total_allocation
    
    # Calculate current capital utilization
    self.capital_utilization = sum(self.strategy_allocations.values())

def get_strategy_allocation(self, strategy_id: str) -> float:
    """
    Get current allocation for a strategy
    
    Args:
        strategy_id: Strategy identifier
        
    Returns:
        Current allocation as percentage (0.0-1.0)
    """
    return self.strategy_allocations.get(strategy_id, 0.0)

def get_all_allocations(self) -> Dict[str, float]:
    """
    Get all current strategy allocations
    
    Returns:
        Dictionary of all strategy allocations
    """
    return self.strategy_allocations.copy()

def get_allocation_history(self) -> List[Dict[str, Any]]:
    """
    Get allocation history
    
    Returns:
        List of historical allocation records
    """
    return self.allocation_history.copy()
