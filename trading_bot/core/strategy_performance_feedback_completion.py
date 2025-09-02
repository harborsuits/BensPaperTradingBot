"""
Remaining Strategy Performance Feedback Loop Methods

This file contains the final methods needed to complete the
StrategyPerformanceFeedback class functionality.
"""

def _update_strategy_weights(self):
    """Update strategy weights based on performance history."""
    # Process each market regime
    for regime in self.strategy_weights.keys():
        # Get all strategies with performance data for this regime
        strategies_with_data = {}
        
        # Extract performance scores for each strategy in this regime
        for strategy_name, strategy_data in self.performance_history['strategies'].items():
            regime_record = strategy_data['by_regime'].get(regime)
            
            if regime_record and regime_record['trades'] >= self.parameters['min_trades_for_significance']:
                # Get performance score
                strategies_with_data[strategy_name] = regime_record['performance_score']
        
        # If no strategies have significant data, use default equal weights
        if not strategies_with_data:
            continue
            
        # Calculate total score
        total_score = sum(strategies_with_data.values())
        
        # Calculate weights based on relative performance
        if total_score > 0:
            # First pass - calculate raw weights
            raw_weights = {strategy: score / total_score for strategy, score in strategies_with_data.items()}
            
            # Second pass - apply min/max constraints
            min_weight = self.parameters['min_strategy_weight']
            max_weight = self.parameters['max_strategy_weight']
            
            # Initial scaling pass - ensure all weights meet minimum
            adjusted_weights = {}
            remaining_weight = 1.0
            
            for strategy, raw_weight in raw_weights.items():
                if raw_weight < min_weight:
                    adjusted_weights[strategy] = min_weight
                    remaining_weight -= min_weight
                else:
                    adjusted_weights[strategy] = raw_weight
            
            # Calculate sum of unconstrained weights
            unconstrained_total = sum(weight for strategy, weight in adjusted_weights.items() 
                                     if strategy not in adjusted_weights or adjusted_weights[strategy] > min_weight)
            
            # Scale unconstrained weights to use remaining weight
            if unconstrained_total > 0:
                for strategy in raw_weights:
                    if strategy not in adjusted_weights or adjusted_weights[strategy] > min_weight:
                        adjusted_weights[strategy] = adjusted_weights[strategy] * remaining_weight / unconstrained_total
            
            # Apply maximum constraint
            for strategy in adjusted_weights:
                if adjusted_weights[strategy] > max_weight:
                    adjusted_weights[strategy] = max_weight
            
            # Normalize weights to sum to 1.0
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                normalized_weights = {strategy: weight / total_weight for strategy, weight in adjusted_weights.items()}
            else:
                # Fallback to equal weights
                count = len(adjusted_weights)
                normalized_weights = {strategy: 1.0 / count for strategy in adjusted_weights}
                
            # Blend with existing weights for smoother transitions
            adaptation_speed = self.parameters['adaptation_speed']
            
            for strategy, new_weight in normalized_weights.items():
                # Get existing weight or default to 0
                existing_weight = self.strategy_weights[regime].get(strategy, 0.0)
                
                # Blend weights
                blended_weight = existing_weight * (1 - adaptation_speed) + new_weight * adaptation_speed
                
                # Update strategy weight
                self.strategy_weights[regime][strategy] = blended_weight
            
            # Log weight updates
            logger.info(f"Updated strategy weights for {regime} regime")
    
def _get_regime_performance_summary(self, regime: str) -> Dict[str, Any]:
    """
    Get performance summary for a market regime.
    
    Args:
        regime: Market regime
        
    Returns:
        Performance summary data
    """
    # Check if regime exists
    if regime not in self.performance_history['regimes']:
        return {
            'regime': regime,
            'strategies': [],
            'asset_classes': [],
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }
        
    # Get regime data
    regime_data = self.performance_history['regimes'][regime]
    
    # Get overall performance
    overall = regime_data['overall']
    
    # Get top performing strategies
    strategies = []
    for strategy_name, strategy_data in self.performance_history['strategies'].items():
        regime_record = strategy_data['by_regime'].get(regime)
        
        if regime_record and regime_record['trades'] >= self.parameters['min_trades_for_significance']:
            strategies.append({
                'name': strategy_name,
                'trades': regime_record['trades'],
                'win_rate': regime_record['win_rate'],
                'profit_factor': regime_record['profit_factor'],
                'performance_score': regime_record['performance_score'],
                'weight': self.strategy_weights[regime].get(strategy_name, 0.0)
            })
    
    # Sort strategies by performance score
    strategies.sort(key=lambda x: x['performance_score'], reverse=True)
    
    # Get asset class performance
    asset_classes = []
    for asset_class, asset_data in self.performance_history['asset_classes'].items():
        regime_record = asset_data['by_regime'].get(regime)
        
        if regime_record and regime_record['trades'] >= self.parameters['min_trades_for_significance']:
            asset_classes.append({
                'name': asset_class,
                'trades': regime_record['trades'],
                'win_rate': regime_record['win_rate'],
                'profit_factor': regime_record['profit_factor'],
                'performance_score': regime_record['performance_score']
            })
    
    # Sort asset classes by performance score
    asset_classes.sort(key=lambda x: x['performance_score'], reverse=True)
    
    # Return summary
    return {
        'regime': regime,
        'strategies': strategies,
        'asset_classes': asset_classes,
        'total_trades': overall['trades'],
        'win_rate': overall['win_rate'],
        'profit_factor': overall['profit_factor'],
        'performance_score': overall['performance_score']
    }

def _load_performance_data(self):
    """Load performance data from disk."""
    # Performance history file
    history_file = os.path.join(self.parameters['performance_data_path'], 'performance_history.json')
    
    # Strategy weights file
    weights_file = os.path.join(self.parameters['performance_data_path'], 'strategy_weights.json')
    
    # Load performance history
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                self.performance_history = json.load(f)
                logger.info(f"Loaded performance history with {len(self.performance_history.get('strategies', {}))} strategies")
        except Exception as e:
            logger.error(f"Error loading performance history: {e}")
    
    # Load strategy weights
    if os.path.exists(weights_file):
        try:
            with open(weights_file, 'r') as f:
                self.strategy_weights = json.load(f)
                logger.info(f"Loaded strategy weights for {len(self.strategy_weights)} regimes")
        except Exception as e:
            logger.error(f"Error loading strategy weights: {e}")

def _save_performance_data(self):
    """Save performance data to disk."""
    # Performance history file
    history_file = os.path.join(self.parameters['performance_data_path'], 'performance_history.json')
    
    # Strategy weights file
    weights_file = os.path.join(self.parameters['performance_data_path'], 'strategy_weights.json')
    
    # Save performance history
    try:
        with open(history_file, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving performance history: {e}")
    
    # Save strategy weights
    try:
        with open(weights_file, 'w') as f:
            json.dump(self.strategy_weights, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving strategy weights: {e}")

def get_strategy_performance(self, strategy_name: str) -> Dict[str, Any]:
    """
    Get performance data for a specific strategy.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Strategy performance data
    """
    # Check if strategy exists
    if strategy_name not in self.performance_history['strategies']:
        return {
            'strategy': strategy_name,
            'exists': False,
            'message': 'No performance data available for this strategy'
        }
    
    # Get strategy data
    strategy_data = self.performance_history['strategies'][strategy_name]
    overall = strategy_data['overall']
    
    # Get regime-specific performance
    regime_performance = {}
    for regime, record in strategy_data['by_regime'].items():
        if record['trades'] >= self.parameters['min_trades_for_significance']:
            regime_performance[regime] = {
                'trades': record['trades'],
                'win_rate': record['win_rate'],
                'profit_factor': record['profit_factor'],
                'performance_score': record['performance_score'],
                'weight': self.strategy_weights[regime].get(strategy_name, 0.0)
            }
    
    # Get asset class performance
    asset_class_performance = {}
    for asset_class, asset_data in strategy_data['by_asset_class'].items():
        record = asset_data['overall']
        
        if record['trades'] >= self.parameters['min_trades_for_significance']:
            asset_class_performance[asset_class] = {
                'trades': record['trades'],
                'win_rate': record['win_rate'],
                'profit_factor': record['profit_factor'],
                'performance_score': record['performance_score']
            }
    
    # Return performance data
    return {
        'strategy': strategy_name,
        'exists': True,
        'overall': {
            'trades': overall['trades'],
            'wins': overall['wins'],
            'losses': overall['losses'],
            'win_rate': overall['win_rate'],
            'profit_factor': overall['profit_factor'],
            'performance_score': overall['performance_score'],
            'trade_history': overall['trade_history'][-20:]  # Last 20 trades
        },
        'by_regime': regime_performance,
        'by_asset_class': asset_class_performance,
        'weights': {regime: weights.get(strategy_name, 0.0) for regime, weights in self.strategy_weights.items()}
    }

def get_top_strategies_by_regime(self) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get top performing strategies for each market regime.
    
    Returns:
        Dictionary mapping regimes to lists of top strategies
    """
    result = {}
    
    # Process each regime
    for regime in self.strategy_weights:
        # Get performance summary for regime
        summary = self._get_regime_performance_summary(regime)
        
        # Get top strategies
        result[regime] = summary['strategies'][:5]  # Top 5 strategies
    
    return result

def get_asset_class_performance(self) -> Dict[str, Dict[str, Any]]:
    """
    Get performance data for each asset class.
    
    Returns:
        Asset class performance data
    """
    result = {}
    
    # Process each asset class
    for asset_class, asset_data in self.performance_history['asset_classes'].items():
        overall = asset_data['overall']
        
        if overall['trades'] >= self.parameters['min_trades_for_significance']:
            # Get regime-specific performance
            regime_performance = {}
            for regime, record in asset_data['by_regime'].items():
                if record['trades'] >= self.parameters['min_trades_for_significance']:
                    regime_performance[regime] = {
                        'trades': record['trades'],
                        'win_rate': record['win_rate'],
                        'profit_factor': record['profit_factor'],
                        'performance_score': record['performance_score']
                    }
            
            # Add to result
            result[asset_class] = {
                'trades': overall['trades'],
                'win_rate': overall['win_rate'],
                'profit_factor': overall['profit_factor'],
                'performance_score': overall['performance_score'],
                'by_regime': regime_performance
            }
    
    return result

# Add custom event types
EventType.STRATEGY_PERFORMANCE_WEIGHTS = "STRATEGY_PERFORMANCE_WEIGHTS"
EventType.TRADE_OPENED = "TRADE_OPENED"
EventType.TRADE_CLOSED = "TRADE_CLOSED"
