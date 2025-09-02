"""
Decision Scoring System

This module implements a feedback loop system for evaluating trading decisions,
scoring their outcomes, and identifying successful patterns for future decisions.

Features:
- Decision quality scoring based on expected vs. actual outcomes
- Pattern recognition for successful decision factors
- Performance tracking by strategy and market regime
- Human-readable decision explanations
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class DecisionOutcome:
    """Classification of decision outcomes for scoring purposes."""
    EXCELLENT = "excellent"  # Significantly better than expected
    GOOD = "good"          # Better than expected
    ACCEPTABLE = "acceptable"  # Meets expectations
    POOR = "poor"          # Below expectations
    VERY_POOR = "very_poor"    # Significantly below expectations

class DecisionScorer:
    """
    Evaluates trading decisions and scores their outcomes.
    
    This class provides:
    1. A feedback loop between decisions and outcomes
    2. Pattern recognition for successful decision factors
    3. Strategy performance tracking across market regimes
    """
    
    def __init__(self, persistence_manager=None, config=None):
        """
        Initialize the decision scorer.
        
        Args:
            persistence_manager: Manager for persisting decision data
            config: Configuration dictionary
        """
        self.persistence = persistence_manager
        self.config = config or {}
        
        # Decision tracking and scoring
        self.decision_history = {}
        self.strategy_performance = defaultdict(list)  # Track strategy performance over time
        self.pattern_database = defaultdict(Counter)  # Market condition -> decision patterns
        self.decision_explanations = {}  # Track human-readable explanations for decisions
        
        # Configure scoring thresholds
        self.scoring_config = {
            'excellent_threshold': 1.5,    # 50% better than expected
            'good_threshold': 1.1,         # 10% better than expected
            'acceptable_threshold': 0.9,    # Within 10% of expected
            'poor_threshold': 0.7          # More than 30% worse than expected
        }
        # Override with any provided config
        if config and 'scoring_thresholds' in config:
            self.scoring_config.update(config['scoring_thresholds'])
            
        logger.info("Decision Scorer initialized")
    
    def record_decision(self, signal_id: str, decision_data: Dict[str, Any]) -> None:
        """
        Record a new decision for later evaluation.
        
        Args:
            signal_id: Unique ID for the trading signal
            decision_data: Decision details including strategy, symbol, direction, etc.
        """
        # Ensure we have required fields
        if not all(k in decision_data for k in ['strategy_id', 'symbol', 'direction']):
            logger.warning(f"Decision data missing required fields: {decision_data}")
            return
        
        # Add timestamp if not present
        if 'timestamp' not in decision_data:
            decision_data['timestamp'] = datetime.now()
            
        # Set initial status
        decision_data['status'] = 'pending'
        decision_data['trades'] = []
        decision_data['score'] = None
        decision_data['outcome'] = None
        
        # Store in memory for quick lookup
        self.decision_history[signal_id] = decision_data
        
        # Persist to database if available
        if self.persistence:
            self.persistence.save_strategy_decision(signal_id, decision_data)
        
        # Generate human-readable explanation
        explanation = self._generate_decision_explanation(decision_data)
        self.decision_explanations[signal_id] = explanation
        
        logger.info(f"Recorded decision {signal_id} from {decision_data.get('strategy_name', 'unknown')}")
        return explanation
    
    def update_decision_execution(self, signal_id: str, execution_data: Dict[str, Any]) -> None:
        """
        Update a decision with execution details.
        
        Args:
            signal_id: ID of the signal/decision
            execution_data: Execution details including price, size, etc.
        """
        if signal_id not in self.decision_history:
            if not self.persistence:
                logger.warning(f"Decision {signal_id} not found, cannot update execution")
                return
            # Try to load from persistence
            decision = self.persistence.load_strategy_decision(signal_id)
            if not decision:
                logger.warning(f"Decision {signal_id} not found in persistence, cannot update")
                return
            self.decision_history[signal_id] = decision
        
        # Get decision
        decision = self.decision_history[signal_id]
        
        # Update with execution details
        decision['status'] = 'executed'
        decision['execution_price'] = execution_data.get('price')
        decision['execution_time'] = execution_data.get('timestamp', datetime.now())
        decision['position_size'] = execution_data.get('quantity')
        
        # Add trade to list
        trade = {
            'order_id': execution_data.get('order_id'),
            'price': execution_data.get('price'),
            'quantity': execution_data.get('quantity'),
            'timestamp': execution_data.get('timestamp', datetime.now()),
            'status': 'open'
        }
        decision['trades'].append(trade)
        
        # Update in database if available
        if self.persistence:
            self.persistence.update_strategy_decision(signal_id, decision)
            
        logger.info(f"Updated decision {signal_id} with execution details")
    
    def close_and_score_decision(self, signal_id: str, outcome_data: Dict[str, Any]) -> Tuple[float, str]:
        """
        Close a decision and calculate its score based on outcome.
        
        Args:
            signal_id: ID of the signal/decision
            outcome_data: Outcome details including PnL, exit price, etc.
            
        Returns:
            Tuple of (score, outcome_category)
        """
        if signal_id not in self.decision_history:
            if not self.persistence:
                logger.warning(f"Decision {signal_id} not found, cannot score")
                return 0.0, DecisionOutcome.POOR
            # Try to load from persistence
            decision = self.persistence.load_strategy_decision(signal_id)
            if not decision:
                logger.warning(f"Decision {signal_id} not found in persistence, cannot score")
                return 0.0, DecisionOutcome.POOR
            self.decision_history[signal_id] = decision
        
        # Get decision
        decision = self.decision_history[signal_id]
        
        # Update trade status
        order_id = outcome_data.get('order_id')
        if order_id:
            for trade in decision['trades']:
                if trade.get('order_id') == order_id:
                    trade['status'] = 'closed'
                    trade['exit_price'] = outcome_data.get('exit_price')
                    trade['exit_time'] = outcome_data.get('timestamp', datetime.now())
                    trade['pnl'] = outcome_data.get('pnl')
                    trade['pnl_percent'] = outcome_data.get('pnl_percent')
        else:
            # If no order_id provided, update all trades
            for trade in decision['trades']:
                if trade['status'] != 'closed':
                    trade['status'] = 'closed'
                    trade['exit_price'] = outcome_data.get('exit_price')
                    trade['exit_time'] = outcome_data.get('timestamp', datetime.now())
                    trade['pnl'] = outcome_data.get('pnl', 0) / len(decision['trades'])  # Distribute PnL
                    trade['pnl_percent'] = outcome_data.get('pnl_percent', 0)
        
        # Check if all trades for this decision are closed
        all_closed = all(trade.get('status') == 'closed' for trade in decision['trades'])
        
        if all_closed:
            # Calculate final PnL
            total_pnl = sum(trade.get('pnl', 0) for trade in decision['trades'])
            avg_pnl_percent = sum(trade.get('pnl_percent', 0) for trade in decision['trades']) / len(decision['trades']) if decision['trades'] else 0
            
            # Update decision status
            decision['status'] = 'closed'
            decision['pnl'] = total_pnl
            decision['pnl_percent'] = avg_pnl_percent
            
            # Score the decision
            score, outcome = self._calculate_decision_score(decision)
            decision['score'] = score
            decision['outcome'] = outcome
            
            # Update pattern database
            self._update_pattern_database(decision)
            
            # Update strategy performance tracking
            strategy_id = decision.get('strategy_id')
            if strategy_id:
                self.strategy_performance[strategy_id].append({
                    'timestamp': datetime.now(),
                    'pnl': total_pnl,
                    'pnl_percent': avg_pnl_percent,
                    'score': score,
                    'outcome': outcome,
                    'signal_id': signal_id
                })
            
            # Update in database if available
            if self.persistence:
                self.persistence.update_strategy_decision(signal_id, decision)
                
            logger.info(f"Scored decision {signal_id} with outcome {outcome} and score {score}")
            return score, outcome
        else:
            logger.info(f"Decision {signal_id} has unclosed trades, not scoring yet")
            return 0.0, "pending"
    
    def _calculate_decision_score(self, decision: Dict[str, Any]) -> Tuple[float, str]:
        """
        Calculate a score for the decision based on outcomes versus expectations.
        
        Args:
            decision: The decision record with execution details
            
        Returns:
            Tuple of (score, outcome_category)
        """
        # Get expected and actual profit
        expected_profit = decision.get('expected_profit', 0)
        expected_risk = decision.get('expected_risk', 0)
        actual_profit = decision.get('pnl', 0)
        
        # Calculate risk-adjusted performance
        if expected_profit and expected_risk and expected_risk != 0:
            expected_reward_risk_ratio = expected_profit / expected_risk
            
            # Calculate actual reward/risk achievement
            if actual_profit > 0 and expected_reward_risk_ratio > 0:
                # Profitable trade
                performance_ratio = actual_profit / expected_profit
            elif actual_profit <= 0 and expected_reward_risk_ratio > 0:
                # Loss on expected profit
                performance_ratio = actual_profit / expected_risk
            else:
                # Default ratio
                performance_ratio = 0.0
        else:
            # Simple profit comparison if expected values are missing
            performance_ratio = 1.0 if actual_profit > 0 else 0.0
        
        # Score the performance against thresholds
        if performance_ratio >= self.scoring_config['excellent_threshold']:
            outcome = DecisionOutcome.EXCELLENT
            score = 5.0
        elif performance_ratio >= self.scoring_config['good_threshold']:
            outcome = DecisionOutcome.GOOD
            score = 4.0
        elif performance_ratio >= self.scoring_config['acceptable_threshold']:
            outcome = DecisionOutcome.ACCEPTABLE
            score = 3.0
        elif performance_ratio >= self.scoring_config['poor_threshold']:
            outcome = DecisionOutcome.POOR
            score = 2.0
        else:
            outcome = DecisionOutcome.VERY_POOR
            score = 1.0
            
        return score, outcome
    
    def _update_pattern_database(self, decision: Dict[str, Any]):
        """
        Update pattern database with this decision's outcome.
        
        Args:
            decision: Complete decision record with outcome
        """
        # Extract key pattern elements
        strategy_id = decision.get('strategy_id')
        market_context = decision.get('market_context', {})
        regime = market_context.get('regime', 'unknown')
        outcome = decision.get('outcome')
        rationale = decision.get('rationale', {})
        
        if not strategy_id or not outcome:
            return
        
        # Create pattern signature
        pattern_elements = []
        
        # Add key rationale elements to pattern
        for factor, details in rationale.items():
            if isinstance(details, dict) and 'value' in details:
                pattern_elements.append(f"{factor}:{details['value']}")
            elif isinstance(details, (int, float, str)):
                pattern_elements.append(f"{factor}:{details}")
        
        if pattern_elements:
            # Create a pattern key
            pattern_key = f"{strategy_id}:{','.join(sorted(pattern_elements))}"
            
            # Update pattern for this market regime
            self.pattern_database[regime][pattern_key] += 1 if outcome in [DecisionOutcome.EXCELLENT, DecisionOutcome.GOOD] else 0
            
            # Also count total occurrences to calculate success rate
            total_key = f"total:{pattern_key}"
            self.pattern_database[regime][total_key] += 1
    
    def get_strategy_performance(self, strategy_id: str, timeframe: str = 'all') -> Dict[str, Any]:
        """
        Get performance metrics for a specific strategy.
        
        Args:
            strategy_id: ID of the strategy
            timeframe: 'day', 'week', 'month', 'all'
            
        Returns:
            Performance metrics
        """
        if strategy_id not in self.strategy_performance:
            return {
                'win_rate': 0,
                'avg_score': 0,
                'avg_pnl': 0,
                'total_trades': 0,
                'timeframe': timeframe
            }
        
        # Filter by timeframe
        now = datetime.now()
        if timeframe == 'day':
            cutoff = now - timedelta(days=1)
        elif timeframe == 'week':
            cutoff = now - timedelta(days=7)
        elif timeframe == 'month':
            cutoff = now - timedelta(days=30)
        else:
            cutoff = datetime.min
        
        # Get relevant performance records
        records = [
            r for r in self.strategy_performance[strategy_id]
            if r['timestamp'] >= cutoff
        ]
        
        if not records:
            return {
                'win_rate': 0,
                'avg_score': 0,
                'avg_pnl': 0,
                'total_trades': 0,
                'timeframe': timeframe
            }
        
        # Calculate metrics
        total_trades = len(records)
        winning_trades = sum(1 for r in records if r['pnl'] > 0)
        win_rate = winning_trades / total_trades if total_trades else 0
        avg_score = sum(r['score'] for r in records) / total_trades if total_trades else 0
        avg_pnl = sum(r['pnl'] for r in records) / total_trades if total_trades else 0
        
        return {
            'win_rate': win_rate,
            'avg_score': avg_score,
            'avg_pnl': avg_pnl,
            'total_trades': total_trades,
            'timeframe': timeframe
        }
    
    def find_successful_patterns(self, regime: str = None, min_occurrences: int = 5, min_success_rate: float = 0.6) -> List[Dict[str, Any]]:
        """
        Find patterns that have been successful in the specified market regime.
        
        Args:
            regime: Market regime to look for patterns in (or None for all)
            min_occurrences: Minimum number of times pattern must have occurred
            min_success_rate: Minimum success rate for inclusion
            
        Returns:
            List of successful patterns with their success rates
        """
        successful_patterns = []
        
        # Determine which regimes to check
        regimes = [regime] if regime else self.pattern_database.keys()
        
        for reg in regimes:
            if reg not in self.pattern_database:
                continue
                
            counters = self.pattern_database[reg]
            
            # Find patterns that meet criteria
            for pattern_key, success_count in counters.items():
                if pattern_key.startswith('total:'):
                    continue
                    
                total_key = f"total:{pattern_key}"
                total_count = counters.get(total_key, 0)
                
                if total_count >= min_occurrences:
                    success_rate = success_count / total_count
                    
                    if success_rate >= min_success_rate:
                        # Parse the pattern key
                        strategy_id, pattern_elements = pattern_key.split(':', 1)
                        
                        successful_patterns.append({
                            'regime': reg,
                            'strategy_id': strategy_id,
                            'pattern': pattern_elements,
                            'success_rate': success_rate,
                            'occurrences': total_count,
                            'successful_occurrences': success_count
                        })
        
        # Sort by success rate (descending)
        successful_patterns.sort(key=lambda p: p['success_rate'], reverse=True)
        return successful_patterns
    
    def _generate_decision_explanation(self, decision: Dict[str, Any]) -> str:
        """
        Generate a human-readable explanation for a decision.
        
        Args:
            decision: Decision data
            
        Returns:
            Human-readable explanation string
        """
        # Extract key elements
        strategy = decision.get('strategy_name', 'Unknown strategy')
        symbol = decision.get('symbol', 'Unknown symbol')
        direction = decision.get('direction', 'unknown').upper()
        confidence = decision.get('confidence', 0) * 100  # Convert to percentage
        rationale = decision.get('rationale', {})
        market_context = decision.get('market_context', {})
        
        # Format explanation parts
        parts = [
            f"{strategy} generated a {direction} signal for {symbol} with {confidence:.1f}% confidence.",
        ]
        
        # Add market context
        if market_context:
            regime = market_context.get('regime', 'unknown')
            volatility = market_context.get('volatility', 'medium')
            trend = market_context.get('trend', 'neutral')
            parts.append(f"Market context: {regime} regime with {volatility} volatility and {trend} trend.")
        
        # Add rationale elements if available
        if rationale:
            parts.append("Rationale:")
            for factor, details in rationale.items():
                if isinstance(details, dict) and 'description' in details:
                    parts.append(f"  - {factor}: {details['description']}")
                else:
                    parts.append(f"  - {factor}: {details}")
        
        # Combine all parts
        return "\n".join(parts)
