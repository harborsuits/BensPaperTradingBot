#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Strategy Performance Manager for BensBot
Evaluates strategy performance and automatically promotes or retires strategies
based on their performance metrics
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class StrategyStatus(Enum):
    """Possible statuses for strategies"""
    ACTIVE = "ACTIVE"           # Normal active status
    PROMOTED = "PROMOTED"       # Strategy has been promoted due to good performance
    PROBATION = "PROBATION"     # Strategy is underperforming but still active
    RETIRED = "RETIRED"         # Strategy has been retired due to poor performance
    INACTIVE = "INACTIVE"       # Strategy manually disabled
    NEW = "NEW"                 # New strategy with insufficient data

class StrategyAction(Enum):
    """Possible actions for strategies"""
    MAINTAIN = "MAINTAIN"               # Keep current parameters
    PROMOTE = "PROMOTE"                 # Increase position sizing, etc.
    PROMOTE_PERCENTILE = "PROMOTE_PERCENTILE"  # Promoted based on percentile
    OPTIMIZE = "OPTIMIZE"               # Run optimization routine
    PROBATION = "PROBATION"             # Put on probation
    RETIRE = "RETIRE"                   # Deactivate strategy
    RETIRE_PERCENTILE = "RETIRE_PERCENTILE"  # Retired based on percentile

class StrategyPerformanceManager:
    """
    Manages strategy performance evaluation and lifecycle
    Automatically promotes or retires strategies based on performance metrics
    """
    
    def __init__(self, persistence_manager=None, evaluation_params=None):
        """
        Initialize the strategy performance manager
        
        Args:
            persistence_manager: Optional persistence manager for loading performance data
            evaluation_params: Optional evaluation parameters
        """
        # Default evaluation parameters
        default_params = {
            'min_trades': 20,           # Min trades before evaluation
            'evaluation_window': 30,     # Days to look back
            'evaluation_frequency': 7,   # Days between evaluations
            'sharpe_threshold': 0.8,     # Min Sharpe ratio for active strategies
            'win_rate_threshold': 0.45,  # Min win rate for active strategies
            'profit_factor_threshold': 1.1,  # Min profit factor
            'retirement_percentile': 25, # Bottom 25% get retired
            'promotion_percentile': 75,  # Top 25% get promoted
            'auto_adjust_params': True,  # Auto tune parameters for promoted
            'min_strategies': 3,         # Min strategies to keep active
            'performance_weight': {      # Weighting for performance metrics
                'sharpe_ratio': 0.35,
                'profit_factor': 0.25,
                'win_rate': 0.20,
                'expectancy': 0.15,
                'drawdown': 0.05         # Negative factor
            }
        }
        
        # Use provided parameters or defaults
        self.params = default_params.copy()
        if evaluation_params:
            self.params.update(evaluation_params)
            
        # Store persistence manager if provided
        self.persistence_manager = persistence_manager
        
        # Initialize tracking
        self.last_evaluation = {}
        self.next_evaluation = {}
        self.strategy_history = {}
        self.strategy_status = {}
        
        logger.info("Strategy Performance Manager initialized")
        logger.debug(f"Evaluation thresholds: Sharpe {self.params['sharpe_threshold']}, "
                   f"Win rate {self.params['win_rate_threshold']}, "
                   f"Profit factor {self.params['profit_factor_threshold']}")
    
    def register_strategy(self, strategy_id: str, strategy_obj: Any) -> None:
        """
        Register a strategy for performance tracking
        
        Args:
            strategy_id: Strategy identifier
            strategy_obj: Strategy object
        """
        # Set initial status to NEW
        if strategy_id not in self.strategy_status:
            self.strategy_status[strategy_id] = StrategyStatus.NEW
            self.next_evaluation[strategy_id] = datetime.now() + timedelta(days=self.params['evaluation_frequency'])
            
            logger.info(f"Registered strategy {strategy_id} for performance tracking")
            
    def evaluate_strategies(self, strategies_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all strategies and provide promotion/retirement recommendations
        
        Args:
            strategies_dict: Dictionary of {strategy_id: strategy_object}
            
        Returns:
            Dictionary of evaluation results and recommendations
        """
        results = {}
        current_time = datetime.now()
        
        # Track strategies to evaluate
        strategies_to_evaluate = []
        
        # Check which strategies are due for evaluation
        for strategy_id, strategy in strategies_dict.items():
            # Register strategy if not already registered
            if strategy_id not in self.strategy_status:
                self.register_strategy(strategy_id, strategy)
                
            # Check if due for evaluation
            if (strategy_id not in self.next_evaluation or 
                current_time >= self.next_evaluation[strategy_id]):
                strategies_to_evaluate.append(strategy_id)
                
        # Skip if no strategies need evaluation
        if not strategies_to_evaluate:
            logger.debug("No strategies due for evaluation")
            return {}
            
        logger.info(f"Evaluating {len(strategies_to_evaluate)} strategies: {strategies_to_evaluate}")
        
        # Get performance data for each strategy
        for strategy_id in strategies_to_evaluate:
            strategy = strategies_dict[strategy_id]
            
            # Get trades from persistence manager if available
            trades_df = pd.DataFrame()
            if self.persistence_manager:
                # Define lookback window
                window_date = current_time - timedelta(days=self.params['evaluation_window'])
                
                # Get trades for this strategy
                trades_df = self.persistence_manager.get_trades_history(
                    strategy_id=strategy_id,
                    start_date=window_date
                )
            
            # Alternatively, get performance metrics directly from strategy
            # if it keeps its own performance records
            metrics = None
            if hasattr(strategy, 'get_performance_metrics'):
                metrics = strategy.get_performance_metrics()
                
            # Check if we have enough data
            if len(trades_df) < self.params['min_trades'] and not metrics:
                # Not enough data
                results[strategy_id] = {
                    'status': 'INSUFFICIENT_DATA',
                    'action': StrategyAction.MAINTAIN.value,
                    'current_status': self.strategy_status[strategy_id].value,
                    'metrics': {
                        'trades': len(trades_df)
                    },
                    'next_evaluation': current_time + timedelta(days=self.params['evaluation_frequency'])
                }
                continue
                
            # Calculate performance metrics from trades if needed
            if metrics is None and not trades_df.empty:
                metrics = self._calculate_performance_metrics(trades_df)
            elif metrics is None:
                # No metrics available
                metrics = {
                    'win_rate': 0,
                    'profit_factor': 0,
                    'sharpe_ratio': 0,
                    'expectancy': 0,
                    'drawdown': 0,
                    'trades': 0
                }
                
            # Determine recommendation based on thresholds
            action = self._determine_action(metrics)
            
            # Store result
            results[strategy_id] = {
                'status': 'EVALUATED',
                'action': action.value,
                'current_status': self.strategy_status[strategy_id].value,
                'metrics': metrics,
                'next_evaluation': current_time + timedelta(days=self.params['evaluation_frequency'])
            }
            
            # Update evaluation tracking
            self.last_evaluation[strategy_id] = current_time
            self.next_evaluation[strategy_id] = current_time + timedelta(days=self.params['evaluation_frequency'])
            
        # Apply percentile-based retirement/promotion if we have enough strategies
        if len(results) >= max(3, self.params['min_strategies']):
            self._apply_percentile_adjustments(results)
            
        # Store evaluation results to history
        for strategy_id, result in results.items():
            if strategy_id not in self.strategy_history:
                self.strategy_history[strategy_id] = []
                
            # Store evaluation data
            self.strategy_history[strategy_id].append({
                'timestamp': current_time,
                'action': result['action'],
                'metrics': result['metrics']
            })
            
            # Limit history to last 10 evaluations
            if len(self.strategy_history[strategy_id]) > 10:
                self.strategy_history[strategy_id] = self.strategy_history[strategy_id][-10:]
                
        # Store results to persistence if available
        if self.persistence_manager:
            for strategy_id, result in results.items():
                self.persistence_manager.save_performance_metrics({
                    'type': 'strategy_evaluation',
                    'strategy_id': strategy_id,
                    'timestamp': current_time,
                    'result': result
                })
                
        return results
        
    def _calculate_performance_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate key performance indicators from trades
        
        Args:
            trades_df: DataFrame containing trade data
            
        Returns:
            dict: Dictionary of performance metrics
        """
        if trades_df.empty:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'expectancy': 0,
                'drawdown': 0,
                'trades': 0,
                'total_profit': 0
            }
            
        # Ensure we have profit column
        profit_col = 'profit_loss'
        if profit_col not in trades_df.columns and 'profit' in trades_df.columns:
            profit_col = 'profit'
            
        if profit_col not in trades_df.columns:
            logger.warning("No profit column found in trades dataframe")
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'expectancy': 0,
                'drawdown': 0,
                'trades': 0,
                'total_profit': 0
            }
            
        # Calculate metrics
        win_mask = trades_df[profit_col] > 0
        win_rate = win_mask.mean()
        
        winning_trades = trades_df[win_mask][profit_col].sum()
        losing_trades = abs(trades_df[~win_mask][profit_col].sum())
        
        profit_factor = winning_trades / losing_trades if losing_trades > 0 else float('inf')
        
        # Calculate expectancy
        avg_win = trades_df[win_mask][profit_col].mean() if win_mask.sum() > 0 else 0
        avg_loss = abs(trades_df[~win_mask][profit_col].mean()) if (~win_mask).sum() > 0 else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Daily returns for Sharpe
        if 'timestamp' in trades_df.columns:
            trades_df['date'] = pd.to_datetime(trades_df['timestamp']).dt.date
            daily_returns = trades_df.groupby('date')[profit_col].sum()
            sharpe = daily_returns.mean() / daily_returns.std() if len(daily_returns) > 1 else 0
        else:
            sharpe = 0
            
        # Drawdown calculation
        cumulative = trades_df[profit_col].cumsum()
        max_dd = 0
        peak = 0
        
        for value in cumulative:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
            
        return {
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'sharpe_ratio': float(sharpe),
            'expectancy': float(expectancy),
            'drawdown': float(max_dd),
            'trades': len(trades_df),
            'total_profit': float(trades_df[profit_col].sum())
        }
        
    def _determine_action(self, metrics: Dict[str, float]) -> StrategyAction:
        """
        Determine action based on performance thresholds
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            StrategyAction: Recommended action
        """
        # Check if strategy fails basic viability criteria
        if (metrics['win_rate'] < self.params['win_rate_threshold'] or
            metrics['profit_factor'] < self.params['profit_factor_threshold'] or
            metrics['sharpe_ratio'] < self.params['sharpe_threshold']):
            return StrategyAction.RETIRE
            
        # Check if strategy is performing well
        if (metrics['win_rate'] > 0.55 and 
            metrics['profit_factor'] > 1.5 and
            metrics['sharpe_ratio'] > 1.2):
            return StrategyAction.PROMOTE
            
        # Check if strategy needs optimization
        if (metrics['win_rate'] > self.params['win_rate_threshold'] and
            metrics['profit_factor'] > self.params['profit_factor_threshold'] and
            metrics['sharpe_ratio'] < self.params['sharpe_threshold']):
            return StrategyAction.OPTIMIZE
            
        # Check if strategy should be on probation
        if (metrics['win_rate'] <= self.params['win_rate_threshold'] + 0.05 or
            metrics['profit_factor'] <= self.params['profit_factor_threshold'] + 0.1):
            return StrategyAction.PROBATION
            
        # Default action
        return StrategyAction.MAINTAIN
        
    def _apply_percentile_adjustments(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        Apply percentile-based adjustments to identify best/worst strategies
        
        Args:
            results: Dictionary of evaluation results
        """
        # Extract strategies with sufficient data
        evaluated = {k: v for k, v in results.items() if v['status'] == 'EVALUATED'}
        if not evaluated:
            return
            
        # Calculate composite score for each strategy
        scores = {}
        
        for strategy_id, result in evaluated.items():
            m = result['metrics']
            
            # Create weighted performance score
            weights = self.params['performance_weight']
            score = (
                weights.get('sharpe_ratio', 0.3) * m.get('sharpe_ratio', 0) + 
                weights.get('profit_factor', 0.3) * min(m.get('profit_factor', 0), 5) + 
                weights.get('win_rate', 0.2) * m.get('win_rate', 0) + 
                weights.get('expectancy', 0.15) * min(m.get('expectancy', 0), 10) -
                weights.get('drawdown', 0.05) * m.get('drawdown', 0)
            )
            
            scores[strategy_id] = score
            
        # Sort strategies by score
        sorted_strategies = sorted(scores.items(), key=lambda x: x[1])
        
        # Identify bottom performers for retirement
        cutoff_retire = max(1, int(len(sorted_strategies) * self.params['retirement_percentile'] / 100))
        bottom_performers = sorted_strategies[:cutoff_retire]
        
        # Identify top performers for promotion
        cutoff_promote = max(1, int(len(sorted_strategies) * (100 - self.params['promotion_percentile']) / 100))
        top_performers = sorted_strategies[-cutoff_promote:]
        
        # Ensure we don't retire too many strategies
        active_count = sum(1 for k, v in results.items() 
                         if v['action'] not in [StrategyAction.RETIRE.value, StrategyAction.RETIRE_PERCENTILE.value])
                         
        max_retire = max(0, active_count - self.params['min_strategies'])
        
        bottom_idx = 0
        
        # Update recommendations
        for strategy_id, _ in bottom_performers:
            # Skip if already marked for retirement
            if results[strategy_id]['action'] == StrategyAction.RETIRE.value:
                continue
                
            # Limit number of strategies retired
            if bottom_idx >= max_retire:
                break
                
            # Mark for percentile-based retirement
            results[strategy_id]['action'] = StrategyAction.RETIRE_PERCENTILE.value
            bottom_idx += 1
                
        for strategy_id, _ in top_performers:
            # Skip if marked for retirement
            if results[strategy_id]['action'] in [StrategyAction.RETIRE.value, 
                                                StrategyAction.RETIRE_PERCENTILE.value]:
                continue
                
            # Mark for percentile-based promotion
            results[strategy_id]['action'] = StrategyAction.PROMOTE_PERCENTILE.value
                
    def apply_recommendations(self, strategies_dict: Dict[str, Any], 
                            results: Dict[str, Dict[str, Any]]) -> None:
        """
        Apply the recommended actions to strategies
        
        Args:
            strategies_dict: Dictionary of {strategy_id: strategy_object}
            results: Dictionary of evaluation results
        """
        updates = []
        
        for strategy_id, result in results.items():
            if strategy_id not in strategies_dict:
                continue
                
            strategy = strategies_dict[strategy_id]
            action = result['action']
            
            # Get previous status
            prev_status = self.strategy_status.get(strategy_id, StrategyStatus.NEW)
            
            # Apply action
            if action == StrategyAction.RETIRE.value or action == StrategyAction.RETIRE_PERCENTILE.value:
                logger.info(f"Retiring strategy {strategy_id} due to poor performance")
                
                # Deactivate strategy
                if hasattr(strategy, 'deactivate'):
                    strategy.deactivate()
                    
                # Update status
                self.strategy_status[strategy_id] = StrategyStatus.RETIRED
                
                updates.append({
                    'strategy_id': strategy_id,
                    'action': action,
                    'old_status': prev_status.value,
                    'new_status': StrategyStatus.RETIRED.value
                })
                
            elif action == StrategyAction.PROMOTE.value or action == StrategyAction.PROMOTE_PERCENTILE.value:
                logger.info(f"Promoting strategy {strategy_id} due to excellent performance")
                
                # Increase position sizing allowance
                if hasattr(strategy, 'risk_params'):
                    strategy.risk_params['position_size_factor'] = min(
                        2.0, strategy.risk_params.get('position_size_factor', 1.0) * 1.2
                    )
                
                # Run auto-optimization if available
                if self.params['auto_adjust_params'] and hasattr(strategy, 'auto_optimize'):
                    strategy.auto_optimize()
                    
                # Update status
                self.strategy_status[strategy_id] = StrategyStatus.PROMOTED
                
                updates.append({
                    'strategy_id': strategy_id,
                    'action': action,
                    'old_status': prev_status.value,
                    'new_status': StrategyStatus.PROMOTED.value
                })
                
            elif action == StrategyAction.PROBATION.value:
                logger.info(f"Putting strategy {strategy_id} on probation")
                
                # Reduce position sizing
                if hasattr(strategy, 'risk_params'):
                    strategy.risk_params['position_size_factor'] = max(
                        0.5, strategy.risk_params.get('position_size_factor', 1.0) * 0.7
                    )
                    
                # Update status
                self.strategy_status[strategy_id] = StrategyStatus.PROBATION
                
                updates.append({
                    'strategy_id': strategy_id,
                    'action': action,
                    'old_status': prev_status.value,
                    'new_status': StrategyStatus.PROBATION.value
                })
                
            elif action == StrategyAction.OPTIMIZE.value:
                logger.info(f"Optimizing strategy {strategy_id}")
                
                # Run optimization if available
                if hasattr(strategy, 'optimize'):
                    strategy.optimize()
                    
                # Keep status as active
                self.strategy_status[strategy_id] = StrategyStatus.ACTIVE
                
                updates.append({
                    'strategy_id': strategy_id,
                    'action': action,
                    'old_status': prev_status.value,
                    'new_status': StrategyStatus.ACTIVE.value
                })
                
            elif action == StrategyAction.MAINTAIN.value:
                # Just update status if needed
                if prev_status in [StrategyStatus.NEW, StrategyStatus.PROBATION]:
                    self.strategy_status[strategy_id] = StrategyStatus.ACTIVE
                    
                    updates.append({
                        'strategy_id': strategy_id,
                        'action': action,
                        'old_status': prev_status.value,
                        'new_status': StrategyStatus.ACTIVE.value
                    })
                    
        # Log summary of updates
        if updates:
            logger.info(f"Applied {len(updates)} strategy updates")
            
            # Store to persistence if available
            if self.persistence_manager:
                for update in updates:
                    self.persistence_manager.log_system_event(
                        level="INFO",
                        message=f"Strategy {update['strategy_id']} updated: {update['action']}",
                        component="StrategyManager",
                        additional_data=update
                    )
                    
    def get_active_strategies(self, strategies_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get only active strategies
        
        Args:
            strategies_dict: Dictionary of {strategy_id: strategy_object}
            
        Returns:
            dict: Dictionary of active strategies
        """
        return {
            strategy_id: strategy for strategy_id, strategy in strategies_dict.items()
            if (strategy_id in self.strategy_status and 
                self.strategy_status[strategy_id] in [StrategyStatus.ACTIVE, StrategyStatus.PROMOTED])
        }
    
    def get_strategy_status(self, strategy_id: str) -> Optional[str]:
        """
        Get status of a specific strategy
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            str or None: Strategy status or None if not found
        """
        if strategy_id not in self.strategy_status:
            return None
            
        return self.strategy_status[strategy_id].value
    
    def get_strategy_history(self, strategy_id: str) -> List[Dict[str, Any]]:
        """
        Get evaluation history for a strategy
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            list: List of historical evaluations
        """
        return self.strategy_history.get(strategy_id, [])
    
    def get_all_statuses(self) -> Dict[str, str]:
        """
        Get status of all strategies
        
        Returns:
            dict: Dictionary mapping strategy IDs to status values
        """
        return {strategy_id: status.value for strategy_id, status in self.strategy_status.items()}
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save current state to persistence manager
        
        Returns:
            dict: Serialized state
        """
        # Convert enums to strings for serialization
        status_dict = {k: v.value for k, v in self.strategy_status.items()}
        
        # Create serializable history
        history_dict = {}
        for strategy_id, history in self.strategy_history.items():
            history_dict[strategy_id] = [{
                'timestamp': h['timestamp'].isoformat(),
                'action': h['action'],
                'metrics': h['metrics']
            } for h in history]
            
        # Create serializable state
        state = {
            'strategy_status': status_dict,
            'last_evaluation': {k: v.isoformat() for k, v in self.last_evaluation.items()},
            'next_evaluation': {k: v.isoformat() for k, v in self.next_evaluation.items()},
            'strategy_history': history_dict
        }
        
        # Save to persistence manager if available
        if self.persistence_manager:
            self.persistence_manager.save_strategy_state('strategy_manager', state)
            
        return state
    
    def load_state(self, state: Dict[str, Any]) -> bool:
        """
        Load state from saved data
        
        Args:
            state: Serialized state dictionary
            
        Returns:
            bool: True if successful
        """
        try:
            # Load strategy status
            if 'strategy_status' in state:
                self.strategy_status = {
                    k: StrategyStatus(v) for k, v in state['strategy_status'].items()
                }
                
            # Load evaluation timestamps
            if 'last_evaluation' in state:
                self.last_evaluation = {
                    k: datetime.fromisoformat(v) for k, v in state['last_evaluation'].items()
                }
                
            if 'next_evaluation' in state:
                self.next_evaluation = {
                    k: datetime.fromisoformat(v) for k, v in state['next_evaluation'].items()
                }
                
            # Load history
            if 'strategy_history' in state:
                history_dict = state['strategy_history']
                for strategy_id, history in history_dict.items():
                    self.strategy_history[strategy_id] = [{
                        'timestamp': datetime.fromisoformat(h['timestamp']),
                        'action': h['action'],
                        'metrics': h['metrics']
                    } for h in history]
                    
            logger.info(f"Loaded Strategy Manager state: {len(self.strategy_status)} strategies")
            return True
        except Exception as e:
            logger.error(f"Error loading Strategy Manager state: {str(e)}")
            return False
