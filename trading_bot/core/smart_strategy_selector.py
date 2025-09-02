"""
Smart Strategy Selector

This module implements an intelligent strategy selection system that uses:
1. Historical decision outcomes from the decision_scoring module
2. Current market regime data
3. Pattern recognition to select optimal strategies

It selects the most appropriate trading strategy based on current market conditions
and historical performance in similar conditions.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
import math
from collections import defaultdict

from trading_bot.core.decision_scoring import DecisionScorer
from trading_bot.data.persistence import PersistenceManager

logger = logging.getLogger(__name__)

class SmartStrategySelector:
    """
    Intelligent strategy selector that learns from past performance
    and adapts to current market conditions.
    
    Features:
    - Strategy fitness scoring by market regime and volatility cluster
    - Multi-armed bandit algorithm for exploration/exploitation balance
    - Weighted strategy selection based on historical performance
    - Pattern recognition for optimal strategy-condition matching
    - Real-time adaptation to changing market conditions
    """
    
    def __init__(self, 
                 persistence_manager: PersistenceManager,
                 decision_scorer: Optional[DecisionScorer] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the smart strategy selector.
        
        Args:
            persistence_manager: For loading strategy metadata and market state
            decision_scorer: For accessing decision history and performance data
            config: Configuration parameters
        """
        self.persistence = persistence_manager
        self.config = config or {}
        
        # Initialize decision scorer if not provided
        if decision_scorer:
            self.decision_scorer = decision_scorer
        else:
            self.decision_scorer = DecisionScorer(persistence_manager)
            
        # Load available strategies
        self.strategies = self._load_available_strategies()
        
        # Strategy selection parameters
        self.selection_params = {
            'recent_performance_weight': 0.4,    # Weight for recent performance
            'regime_compatibility_weight': 0.3,  # Weight for regime compatibility
            'pattern_success_weight': 0.2,       # Weight for pattern recognition
            'long_term_performance_weight': 0.1, # Weight for long-term performance
            'min_selection_threshold': 0.6,      # Minimum score to select a strategy
            'pattern_confidence_threshold': 0.7, # Minimum confidence for pattern matching
            'performance_lookback_days': 30,     # Days of performance history to consider
            
            # Bandit algorithm parameters
            'exploration_weight': 0.2,           # Higher values favor exploration (0-1)
            'initial_fitness_score': 0.5,        # Initial fitness score for new strategies
            'learning_rate': 0.1,                # Rate of fitness score updates (0-1)
            'min_trials': 5,                     # Minimum trials before reliable fitness score
        }
        
        # Strategy fitness tracking
        self.strategy_fitness = defaultdict(lambda: defaultdict(lambda: {
            'score': self.selection_params['initial_fitness_score'],
            'trials': 0,
            'wins': 0,
            'cumulative_pnl': 0.0,
            'last_updated': datetime.now()
        }))
        
        # Override with config if provided
        if 'selection_params' in self.config:
            self.selection_params.update(self.config['selection_params'])
        
        logger.info("Smart Strategy Selector initialized")
    
    def select_strategy(self, 
                       symbol: str, 
                       market_data: Dict[str, Any],
                       additional_context: Optional[Dict[str, Any]] = None,
                       use_bandit_selection: bool = True) -> Dict[str, Any]:
        """
        Select the best strategy for the current market conditions.
        
        Args:
            symbol: Trading symbol
            market_data: Current market data including technical indicators
            additional_context: Any additional context information
            
        Returns:
            Selected strategy information with confidence score
        """
        # Get current market regime
        market_regime = self._get_current_market_regime()
        
        # Build complete context
        context = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'market_regime': market_regime,
            'indicators': market_data.get('indicators', {}),
            'recent_price_action': market_data.get('price_action', {})
        }
        
        if additional_context:
            context.update(additional_context)
        
        # Calculate scores for each strategy
        strategy_scores = []
        
        for strategy in self.strategies:
            # Skip strategies not compatible with this symbol
            if not self._is_compatible_with_symbol(strategy, symbol):
                continue
                
            # Get regime and volatility for fitness lookup
            regime = context['market_regime']
            volatility = context.get('volatility', 'medium')
            
            # Calculate composite score or use bandit algorithm
            if use_bandit_selection:
                score = self._calculate_bandit_score(strategy, context, regime, volatility)
            else:
                score = self._calculate_strategy_score(strategy, context)
            
            strategy_scores.append({
                'strategy_id': strategy['strategy_id'],
                'strategy_name': strategy['strategy_name'],
                'score': score,
                'strategy_data': strategy,
                'fitness': self._get_strategy_fitness(strategy['strategy_id'], regime, volatility)
            })
        
        # Sort by score (descending)
        strategy_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Check if best strategy meets threshold
        if strategy_scores and strategy_scores[0]['score'] >= self.selection_params['min_selection_threshold']:
            selected = strategy_scores[0]
            logger.info(f"Selected strategy {selected['strategy_name']} with score {selected['score']:.2f}")
            
            # Add explanation
            selected['explanation'] = self._generate_selection_explanation(
                selected['strategy_data'], 
                context,
                strategy_scores[:3] if len(strategy_scores) >= 3 else strategy_scores
            )
            
            # Record selection in fitness tracking (will be updated when trade completes)
            regime = context['market_regime']
            volatility = context.get('volatility', 'medium')
            self._record_strategy_selection(selected['strategy_id'], regime, volatility)
            
            return selected
        else:
            # No strategy meets threshold
            logger.warning(f"No strategy meets selection threshold for {symbol} in {market_regime} regime")
            return {
                'strategy_id': None, 
                'strategy_name': None, 
                'score': 0,
                'explanation': "No strategy meets the minimum selection threshold for current conditions."
            }
    
    def _calculate_strategy_score(self, strategy: Dict[str, Any], context: Dict[str, Any]) -> float:
        """
        Calculate a composite score for a strategy based on multiple factors.
        
        Args:
            strategy: Strategy metadata
            context: Current market context
            
        Returns:
            Composite score (0-1)
        """
        strategy_id = strategy['strategy_id']
        market_regime = context['market_regime']
        symbol = context['symbol']
        
        # 1. Recent performance score
        recent_performance = self.decision_scorer.get_strategy_performance(
            strategy_id, 
            'week'  # Recent = last week
        )
        recent_score = min(1.0, (recent_performance['win_rate'] * 0.7) + 
                             (recent_performance['avg_score'] / 5.0 * 0.3))
        
        # 2. Regime compatibility score
        regime_score = self._calculate_regime_compatibility(strategy, market_regime)
        
        # 3. Pattern matching score
        pattern_score = self._calculate_pattern_match_score(strategy, context)
        
        # 4. Long-term performance
        long_term_performance = self.decision_scorer.get_strategy_performance(
            strategy_id, 
            'month'  # Long-term = last month
        )
        long_term_score = min(1.0, (long_term_performance['win_rate'] * 0.6) + 
                               (long_term_performance['avg_score'] / 5.0 * 0.4))
        
        # Combine scores using configured weights
        weights = self.selection_params
        composite_score = (
            recent_score * weights['recent_performance_weight'] +
            regime_score * weights['regime_compatibility_weight'] +
            pattern_score * weights['pattern_success_weight'] +
            long_term_score * weights['long_term_performance_weight']
        )
        
        # Log component scores for debugging
        logger.debug(f"Strategy {strategy['strategy_name']} scores: " +
                   f"recent={recent_score:.2f}, regime={regime_score:.2f}, " +
                   f"pattern={pattern_score:.2f}, long_term={long_term_score:.2f}, " +
                   f"composite={composite_score:.2f}")
        
        return composite_score
    
    def _calculate_regime_compatibility(self, strategy: Dict[str, Any], market_regime: str) -> float:
        """
        Calculate how compatible a strategy is with the current market regime.
        
        Args:
            strategy: Strategy metadata
            market_regime: Current market regime
            
        Returns:
            Compatibility score (0-1)
        """
        # Get compatible regimes from strategy metadata
        compatible_regimes = strategy.get('compatible_market_regimes', [])
        
        # If strategy claims compatibility with 'all_weather', it works in all regimes
        if 'all_weather' in compatible_regimes:
            return 0.8  # Good but not perfect
            
        # Direct match
        if market_regime in compatible_regimes:
            return 1.0
            
        # Partial matches (related regimes)
        regime_similarities = {
            'trending': ['bull_trend', 'bear_trend', 'momentum'],
            'ranging': ['consolidation', 'sideways', 'mean_reversion'],
            'volatile': ['breakout', 'reversal', 'high_volatility'],
            'bull_trend': ['trending', 'momentum'],
            'bear_trend': ['trending', 'momentum'],
            'consolidation': ['ranging', 'sideways', 'mean_reversion'],
        }
        
        # Check for related regimes
        if market_regime in regime_similarities:
            related_regimes = regime_similarities[market_regime]
            for related in related_regimes:
                if related in compatible_regimes:
                    return 0.8  # Good match with related regime
        
        # No match found
        return 0.3  # Base compatibility level
    
    def _calculate_pattern_match_score(self, strategy: Dict[str, Any], context: Dict[str, Any]) -> float:
        """
        Calculate how well current conditions match successful patterns for this strategy.
        
        Args:
            strategy: Strategy metadata
            context: Current market context
            
        Returns:
            Pattern match score (0-1)
        """
        strategy_id = strategy['strategy_id']
        market_regime = context['market_regime']
        
        # Get successful patterns for this strategy in this regime
        successful_patterns = self.decision_scorer.find_successful_patterns(
            regime=market_regime,
            min_occurrences=3,
            min_success_rate=self.selection_params['pattern_confidence_threshold']
        )
        
        # Filter to this strategy only
        strategy_patterns = [p for p in successful_patterns if p['strategy_id'] == strategy_id]
        
        if not strategy_patterns:
            return 0.5  # Neutral score if no patterns found
        
        # Extract current indicator values from context
        current_indicators = context.get('indicators', {})
        
        # Try to match patterns
        best_match_score = 0.0
        for pattern in strategy_patterns:
            pattern_elements = pattern['pattern'].split(',')
            match_score = self._match_pattern_elements(pattern_elements, current_indicators)
            
            # Weight by pattern success rate
            weighted_score = match_score * pattern['success_rate']
            
            # Keep track of best match
            if weighted_score > best_match_score:
                best_match_score = weighted_score
        
        return best_match_score
    
    def _match_pattern_elements(self, pattern_elements: List[str], indicators: Dict[str, Any]) -> float:
        """
        Calculate how well current indicators match a pattern.
        
        Args:
            pattern_elements: Elements of the pattern (e.g., 'rsi:overbought')
            indicators: Current indicator values
            
        Returns:
            Match score (0-1)
        """
        if not pattern_elements or not indicators:
            return 0.0
            
        matches = 0
        total_elements = len(pattern_elements)
        
        for element in pattern_elements:
            if ':' not in element:
                continue
                
            indicator, value = element.split(':', 1)
            
            if indicator in indicators:
                current_value = indicators[indicator]
                
                # Compare string values directly
                if isinstance(current_value, str) and current_value == value:
                    matches += 1
                    
                # For numeric values, use fuzzy matching
                elif isinstance(current_value, (int, float)) and isinstance(value, str):
                    try:
                        # Try to convert value to number if possible
                        numeric_value = float(value)
                        # Allow 10% tolerance
                        if abs(current_value - numeric_value) / max(1, numeric_value) <= 0.1:
                            matches += 1
                    except ValueError:
                        # Handle condition-based values like 'overbought', 'bullish', etc.
                        if self._match_condition_value(indicator, current_value, value):
                            matches += 1
        
        # Calculate match percentage
        return matches / total_elements if total_elements > 0 else 0.0
    
    def _match_condition_value(self, indicator: str, current_value: float, condition: str) -> bool:
        """
        Match a condition-based value like 'overbought' or 'bullish'.
        
        Args:
            indicator: Indicator name
            current_value: Current numeric value
            condition: Condition string
            
        Returns:
            True if matches, False otherwise
        """
        # RSI conditions
        if indicator == 'rsi':
            if condition == 'overbought' and current_value > 70:
                return True
            if condition == 'oversold' and current_value < 30:
                return True
            if condition == 'neutral' and 40 <= current_value <= 60:
                return True
                
        # MACD conditions
        elif indicator == 'macd':
            if condition == 'bullish' and current_value > 0:
                return True
            if condition == 'bearish' and current_value < 0:
                return True
                
        # Generic conditions
        if condition == 'high' and current_value > 0.8:
            return True
        if condition == 'low' and current_value < 0.2:
            return True
        if condition == 'medium' and 0.3 <= current_value <= 0.7:
            return True
            
        return False
    
    def _generate_selection_explanation(self, 
                                      strategy: Dict[str, Any], 
                                      context: Dict[str, Any],
                                      top_strategies: List[Dict[str, Any]]) -> str:
        """
        Generate a human-readable explanation for the strategy selection.
        
        Args:
            strategy: Selected strategy data
            context: Current market context
            top_strategies: Top-scoring strategies for comparison
            
        Returns:
            Explanation text
        """
        strategy_name = strategy['strategy_name']
        strategy_id = strategy['strategy_id']
        market_regime = context['market_regime']
        volatility = context.get('volatility', 'medium')
        symbol = context['symbol']
        
        # Basic explanation parts
        parts = [
            f"Selected {strategy_name} for {symbol} in {market_regime} market regime with {volatility} volatility."
        ]
        
        # Add fitness information
        fitness_data = self._get_strategy_fitness(strategy_id, market_regime, volatility)
        if fitness_data['trials'] > 0:
            win_rate = (fitness_data['wins'] / fitness_data['trials']) * 100
            avg_pnl = fitness_data['cumulative_pnl'] / fitness_data['trials']
            parts.append(f"Strategy fitness score: {fitness_data['score']:.2f} ({win_rate:.1f}% win rate over "
                       f"{fitness_data['trials']} trials, avg PnL: {avg_pnl:.2f})")
        
        # Add regime compatibility
        compatible_regimes = strategy.get('compatible_market_regimes', [])
        if market_regime in compatible_regimes:
            parts.append(f"{strategy_name} is specifically designed for {market_regime} markets.")
        elif 'all_weather' in compatible_regimes:
            parts.append(f"{strategy_name} is an all-weather strategy that adapts to different market conditions.")
        
        # Add performance information
        recent_performance = self.decision_scorer.get_strategy_performance(
            strategy_id, 
            'week'
        )
        if recent_performance['total_trades'] > 0:
            win_rate = recent_performance['win_rate'] * 100
            parts.append(f"Recent performance: {win_rate:.1f}% win rate over {recent_performance['total_trades']} trades.")
        
        # Add pattern match information if available
        successful_patterns = self.decision_scorer.find_successful_patterns(
            regime=market_regime,
            min_occurrences=3,
            min_success_rate=0.6
        )
        strategy_patterns = [p for p in successful_patterns if p['strategy_id'] == strategy_id]
        if strategy_patterns:
            best_pattern = max(strategy_patterns, key=lambda p: p['success_rate'])
            success_rate = best_pattern['success_rate'] * 100
            parts.append(f"Current market conditions match a pattern with {success_rate:.1f}% historical success rate.")
        
        # Compare to alternatives
        if len(top_strategies) > 1:
            alternatives = [s for s in top_strategies if s['strategy_id'] != strategy_id]
            if alternatives:
                runner_up = alternatives[0]
                score_diff = (strategy['score'] - runner_up['score']) * 100
                parts.append(f"Outscored {runner_up['strategy_name']} by {score_diff:.1f} points.")
                
                # Add exploration/exploitation insight if using bandit algorithm
                if 'fitness' in strategy and 'fitness' in runner_up:
                    if strategy['fitness']['trials'] < runner_up['fitness']['trials']:
                        parts.append(f"Selected for exploration value ({strategy['fitness']['trials']} vs "
                                  f"{runner_up['fitness']['trials']} trials for {runner_up['strategy_name']}).")
        
        return "\n".join(parts)
        
    def _is_compatible_with_symbol(self, strategy: Dict[str, Any], symbol: str) -> bool:
        """
        Check if a strategy is compatible with the symbol.
        
        Args:
            strategy: Strategy metadata
            symbol: Symbol to check
            
        Returns:
            True if compatible, False otherwise
        """
        # Extract asset class from symbol
        asset_class = self._get_asset_class_from_symbol(symbol)
        
        # Check supported asset classes
        supported_assets = strategy.get('supported_asset_classes', [])
        
        # Special case for 'all' - supports all asset classes
        if 'all' in supported_assets:
            return True
            
        return asset_class in supported_assets
    
    def _get_asset_class_from_symbol(self, symbol: str) -> str:
        """
        Extract asset class from symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Asset class
        """
        # Simple heuristic rules
        if '/' in symbol:
            return 'forex'
        if symbol.endswith('USD') or symbol.startswith('USD'):
            return 'forex'
        if symbol.lower().endswith('usdt') or symbol.lower().endswith('btc'):
            return 'crypto'
        if '.' in symbol:
            return 'stocks'
        
        # Default to stocks
        return 'stocks'
    
    def _get_current_market_regime(self) -> str:
        """
        Get current market regime from persistence.
        
        Returns:
            Market regime name
        """
        # Load the current market regime
        market_regime_data = self.persistence.load_strategy_state("market_regime_detector") or {}
        
        # Get regime name, default to 'unknown'
        return market_regime_data.get('current_regime', 'unknown')
    
    def _load_available_strategies(self) -> List[Dict[str, Any]]:
        """
        Load available strategies from persistence.
        
        Returns:
            List of strategy metadata
        """
        # First try to load from persistence
        strategies_data = self.persistence.load_strategy_state("available_strategies") or {}
        strategies_list = strategies_data.get('strategies', [])
        
        if strategies_list:
            return strategies_list
            
        
        if strategies_list:
            return strategies_list
            
        # If not available, return empty list
        logger.warning("No strategies found in persistence, returning empty list")
        return []

    def update_strategy_performance(self, decision_outcome: Dict[str, Any]) -> None:
        """
        Update strategy performance data after a trade is closed.
        
        Args:
            decision_outcome: Outcome data from decision scorer
        """
        # Extract key information
        strategy_id = decision_outcome.get('strategy_id')
        if not strategy_id:
            return
                
        # Get trade result details
        pnl = decision_outcome.get('pnl', 0)
        is_win = pnl > 0
        market_regime = decision_outcome.get('market_regime', 'unknown')
        volatility = decision_outcome.get('volatility', 'medium')
            
        # Update fitness scores
        self._update_strategy_fitness(strategy_id, market_regime, volatility, is_win, pnl)
                
        # Also forward to decision scorer to update its internal tracking
        logger.info(f"Updated performance for strategy {strategy_id}: {'win' if is_win else 'loss'} with PnL {pnl:.2f}")

    def _calculate_bandit_score(self, strategy: Dict[str, Any], context: Dict[str, Any], 
                              regime: str, volatility: str) -> float:
    """
    Calculate strategy score using Upper Confidence Bound (UCB) bandit algorithm.
    
    Args:
        strategy: Strategy metadata
        context: Current market context
        regime: Current market regime
        volatility: Current volatility state
            
    Returns:
        UCB score for exploration/exploitation balance
    """
    strategy_id = strategy['strategy_id']
    fitness_data = self._get_strategy_fitness(strategy_id, regime, volatility)
        
    # Get current fitness score and trials
    fitness_score = fitness_data['score']
    trials = fitness_data['trials']
        
    # Calculate base score from standard metrics
    base_score = self._calculate_strategy_score(strategy, context)
        
    # If we have enough trials, use UCB formula
    if trials > 0:
        # Total number of trials across all strategies in this regime
        total_selections = sum(self._get_strategy_fitness(s['strategy_id'], regime, volatility)['trials'] 
                             for s in self.strategies)
            
        # Add exploration bonus based on UCB formula
        # UCB = fitness + exploration_weight * sqrt(ln(total_trials) / trials)
        exploration_term = 0
        if total_selections > 0:
            exploration_term = math.sqrt(math.log(total_selections) / max(1, trials))
                
        exploration_bonus = self.selection_params['exploration_weight'] * exploration_term
        ucb_score = fitness_score + exploration_bonus
            
        # Blend UCB with base score
        final_score = (ucb_score * 0.7) + (base_score * 0.3)
            
        logger.debug(f"Strategy {strategy_id} UCB score: {ucb_score:.3f} = "
                   f"fitness {fitness_score:.3f} + exploration {exploration_bonus:.3f}, "
                   f"trials: {trials}, final score: {final_score:.3f}")
            
        return final_score
    else:
        # For new strategies, use high score to encourage exploration
        return base_score + self.selection_params['exploration_weight']

    def _get_strategy_fitness(self, strategy_id: str, regime: str, volatility: str) -> Dict[str, Any]:
    """
    Get fitness data for a strategy in specific market conditions.
    
    Args:
        strategy_id: Strategy identifier
        regime: Market regime
        volatility: Volatility state
            
    Returns:
        Fitness data dictionary
    """
    # Create cluster key for combined regime and volatility
    cluster_key = f"{regime}_{volatility}"
    return self.strategy_fitness[strategy_id][cluster_key]

    def _record_strategy_selection(self, strategy_id: str, regime: str, volatility: str) -> None:
    """
    Record that a strategy was selected for the current market conditions.
    
    Args:
        strategy_id: Strategy identifier
        regime: Market regime
        volatility: Volatility state
    """
    cluster_key = f"{regime}_{volatility}"
    self.strategy_fitness[strategy_id][cluster_key]['last_selected'] = datetime.now()

    def _update_strategy_fitness(self, strategy_id: str, regime: str, volatility: str, 
                              is_win: bool, pnl: float) -> None:
    """
    Update fitness score for a strategy based on trade outcome.
    
    Args:
        strategy_id: Strategy identifier
        regime: Market regime when trade was opened
        volatility: Volatility state when trade was opened
        is_win: Whether the trade was profitable
        pnl: Profit/loss amount
    """
    cluster_key = f"{regime}_{volatility}"
    fitness_data = self.strategy_fitness[strategy_id][cluster_key]
        
    # Update counts
    fitness_data['trials'] += 1
    if is_win:
        fitness_data['wins'] += 1
    fitness_data['cumulative_pnl'] += pnl
    fitness_data['last_updated'] = datetime.now()
        
    # Calculate new win rate
    win_rate = fitness_data['wins'] / fitness_data['trials']
        
    # Calculate PnL-based component (scaled to -1 to +1 range roughly)
    avg_pnl = fitness_data['cumulative_pnl'] / fitness_data['trials']
    pnl_factor = min(1.0, max(-1.0, avg_pnl / 100.0))  # Scale PnL
        
    # Calculate new fitness score (blend of win rate and PnL)
    # Weight win rate higher in early trials, then move toward PnL-based scoring
    if fitness_data['trials'] < self.selection_params['min_trials']:
        # During initial phase, mainly use win rate but account for luck/variance
        raw_fitness = win_rate * 0.8 + 0.2  # Start from 0.2 even with 0% win rate
    else:
        # With more data, use PnL-weighted approach
        pnl_weight = min(0.7, fitness_data['trials'] / 50.0)  # Max 70% weight to PnL
        win_rate_weight = 1.0 - pnl_weight
            
        raw_fitness = (win_rate * win_rate_weight) + \
                     ((pnl_factor + 1.0) / 2.0 * pnl_weight)  # Convert -1 to +1 range to 0 to 1
        
    # Apply learning rate for smooth updates
    learning_rate = self.selection_params['learning_rate']
    fitness_data['score'] = (1 - learning_rate) * fitness_data['score'] + learning_rate * raw_fitness
        
    logger.debug(f"Updated fitness for {strategy_id} in {cluster_key}: "
               f"score={fitness_data['score']:.3f}, wins={fitness_data['wins']}/{fitness_data['trials']}, "
               f"pnl={fitness_data['cumulative_pnl']:.2f}")
        
    # Save fitness data to persistence
    self._save_fitness_data()

    def _save_fitness_data(self) -> None:
    """
    Save fitness data to persistence.
    """
    if not self.persistence:
        return
            
    # Convert defaultdict to regular dict for serialization
    serializable_fitness = {}
    for strategy_id, clusters in self.strategy_fitness.items():
        serializable_fitness[strategy_id] = {}
        for cluster_key, data in clusters.items():
            # Convert datetime to string
            serialized_data = data.copy()
            if 'last_updated' in serialized_data:
                serialized_data['last_updated'] = serialized_data['last_updated'].isoformat()
            if 'last_selected' in serialized_data:
                serialized_data['last_selected'] = serialized_data['last_selected'].isoformat()
            serializable_fitness[strategy_id][cluster_key] = serialized_data
        
    self.persistence.save_strategy_state("strategy_fitness", serializable_fitness)

    def load_fitness_data(self) -> None:
    """
    Load fitness data from persistence.
    """
    if not self.persistence:
        return
            
    saved_data = self.persistence.load_strategy_state("strategy_fitness")
    if not saved_data:
        return
            
    # Convert to defaultdict and parse datetime strings
    for strategy_id, clusters in saved_data.items():
        for cluster_key, data in clusters.items():
            # Parse datetime strings
            if 'last_updated' in data and isinstance(data['last_updated'], str):
                try:
                    data['last_updated'] = datetime.fromisoformat(data['last_updated'])
                except ValueError:
                    data['last_updated'] = datetime.now()
                        
            if 'last_selected' in data and isinstance(data['last_selected'], str):
                try:
                    data['last_selected'] = datetime.fromisoformat(data['last_selected'])
                except ValueError:
                    data['last_selected'] = datetime.now()
                        
            # Update fitness data
            self.strategy_fitness[strategy_id][cluster_key].update(data)
        
    logger.info(f"Loaded fitness data for {len(saved_data)} strategies")
