"""
Trade Journal and Feedback System

This module provides a comprehensive trade journaling system that logs all trades
with their rationale, outcome, and market context. It also implements a feedback
loop to improve strategy allocation and timing.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class TradeJournal:
    """
    Trade journal for logging and analyzing trading activity
    """
    
    def __init__(
        self,
        journal_dir: str = "trade_journal",
        config_path: Optional[str] = None
    ):
        """
        Initialize the trade journal.
        
        Args:
            journal_dir: Directory to store journal files
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Set journal directory
        self.journal_dir = journal_dir
        self.trades_file = os.path.join(journal_dir, "trades.json")
        self.feedback_file = os.path.join(journal_dir, "feedback.json")
        self.performance_file = os.path.join(journal_dir, "performance.json")
        
        # Create journal directory if it doesn't exist
        os.makedirs(journal_dir, exist_ok=True)
        
        # Initialize journal data structures
        self.trades = self._load_trades()
        self.feedback = self._load_feedback()
        self.performance = self._load_performance()
        
        logger.info(f"Trade journal initialized at {journal_dir}")
    
    def log_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        quantity: float,
        strategy: str,
        entry_time: Optional[str] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        market_context: Optional[Dict[str, Any]] = None,
        rationale: Optional[str] = None,
        evaluation: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        trade_id: Optional[str] = None
    ) -> str:
        """
        Log a new trade entry to the journal.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction ('long' or 'short')
            entry_price: Entry price
            quantity: Trade quantity
            strategy: Strategy name
            entry_time: Entry timestamp (default: current time)
            stop_loss: Stop loss price
            take_profit: Take profit target
            market_context: Market context data
            rationale: Explanation for entering the trade
            evaluation: LLM evaluation data
            tags: List of tags to categorize the trade
            trade_id: Custom trade ID (optional)
            
        Returns:
            Generated trade ID
        """
        # Generate entry time if not provided
        if not entry_time:
            entry_time = datetime.now().isoformat()
        
        # Generate trade ID if not provided
        if not trade_id:
            trade_id = f"{symbol}_{direction}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Calculate risk/reward if stop loss and take profit provided
        risk_reward = None
        if stop_loss and take_profit and entry_price:
            if direction == 'long':
                reward = take_profit - entry_price
                risk = entry_price - stop_loss
            else:  # short
                reward = entry_price - take_profit
                risk = stop_loss - entry_price
            
            if risk > 0:
                risk_reward = reward / risk
        
        # Create trade record
        trade = {
            'id': trade_id,
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'quantity': quantity,
            'strategy': strategy,
            'entry_time': entry_time,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': risk_reward,
            'market_context': market_context or {},
            'rationale': rationale or "No rationale provided",
            'evaluation': evaluation or {},
            'tags': tags or [],
            'status': 'open',
            'exit_price': None,
            'exit_time': None,
            'pnl': None,
            'pnl_percentage': None,
            'duration': None,
            'exit_reason': None,
            'notes': [],
            'modified_plan': False
        }
        
        # Add trade to journal
        self.trades.append(trade)
        self._save_trades()
        
        logger.info(f"Logged new {direction} trade for {symbol} at ${entry_price:.2f} with strategy '{strategy}'")
        return trade_id
    
    def log_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_time: Optional[str] = None,
        exit_reason: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Log the exit for an existing trade.
        
        Args:
            trade_id: Trade ID
            exit_price: Exit price
            exit_time: Exit timestamp (default: current time)
            exit_reason: Reason for exiting the trade
            notes: Additional notes
            
        Returns:
            Updated trade record
        """
        # Find the trade
        trade = self._get_trade(trade_id)
        if not trade:
            logger.warning(f"Trade with ID {trade_id} not found")
            return {}
        
        # Generate exit time if not provided
        if not exit_time:
            exit_time = datetime.now().isoformat()
        
        # Calculate P&L
        if trade['direction'] == 'long':
            pnl = (exit_price - trade['entry_price']) * trade['quantity']
            pnl_percentage = (exit_price / trade['entry_price'] - 1) * 100
        else:  # short
            pnl = (trade['entry_price'] - exit_price) * trade['quantity']
            pnl_percentage = (trade['entry_price'] / exit_price - 1) * 100
        
        # Calculate duration
        try:
            entry_dt = datetime.fromisoformat(trade['entry_time'])
            exit_dt = datetime.fromisoformat(exit_time)
            duration_seconds = (exit_dt - entry_dt).total_seconds()
            duration = duration_seconds / 3600  # Convert to hours
        except ValueError:
            duration = None
        
        # Update trade record
        trade['exit_price'] = exit_price
        trade['exit_time'] = exit_time
        trade['pnl'] = pnl
        trade['pnl_percentage'] = pnl_percentage
        trade['duration'] = duration
        trade['exit_reason'] = exit_reason or "Not specified"
        trade['status'] = 'closed'
        
        if notes:
            trade['notes'].append({
                'time': datetime.now().isoformat(),
                'content': notes
            })
        
        # Save trades and update performance metrics
        self._save_trades()
        self._update_performance_metrics()
        
        logger.info(f"Logged exit for trade {trade_id} at ${exit_price:.2f} with P&L: ${pnl:.2f} ({pnl_percentage:.2f}%)")
        
        # Generate feedback for completed trade
        self._generate_trade_feedback(trade)
        
        return trade
    
    def add_note(self, trade_id: str, note: str) -> Dict[str, Any]:
        """
        Add a note to an existing trade.
        
        Args:
            trade_id: Trade ID
            note: Note content
            
        Returns:
            Updated trade record
        """
        # Find the trade
        trade = self._get_trade(trade_id)
        if not trade:
            logger.warning(f"Trade with ID {trade_id} not found")
            return {}
        
        # Add note
        trade['notes'].append({
            'time': datetime.now().isoformat(),
            'content': note
        })
        
        # Save trades
        self._save_trades()
        
        logger.debug(f"Added note to trade {trade_id}")
        return trade
    
    def modify_plan(
        self,
        trade_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        note: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Modify the plan for an existing trade.
        
        Args:
            trade_id: Trade ID
            stop_loss: New stop loss price
            take_profit: New take profit target
            note: Note explaining the modification
            
        Returns:
            Updated trade record
        """
        # Find the trade
        trade = self._get_trade(trade_id)
        if not trade:
            logger.warning(f"Trade with ID {trade_id} not found")
            return {}
        
        if trade['status'] != 'open':
            logger.warning(f"Cannot modify closed trade {trade_id}")
            return trade
        
        # Save original plan
        if not trade.get('original_plan'):
            trade['original_plan'] = {
                'stop_loss': trade.get('stop_loss'),
                'take_profit': trade.get('take_profit')
            }
        
        # Update plan
        if stop_loss is not None:
            trade['stop_loss'] = stop_loss
        
        if take_profit is not None:
            trade['take_profit'] = take_profit
        
        # Recalculate risk/reward
        if trade['stop_loss'] and trade['take_profit'] and trade['entry_price']:
            if trade['direction'] == 'long':
                reward = trade['take_profit'] - trade['entry_price']
                risk = trade['entry_price'] - trade['stop_loss']
            else:  # short
                reward = trade['entry_price'] - trade['take_profit']
                risk = trade['stop_loss'] - trade['entry_price']
            
            if risk > 0:
                trade['risk_reward'] = reward / risk
        
        # Mark as modified and add note
        trade['modified_plan'] = True
        
        # Add modification note
        if note or stop_loss is not None or take_profit is not None:
            content = note or f"Modified plan: "
            if not note:
                if stop_loss is not None:
                    content += f"SL=${stop_loss:.2f} "
                if take_profit is not None:
                    content += f"TP=${take_profit:.2f}"
            
            trade['notes'].append({
                'time': datetime.now().isoformat(),
                'content': content
            })
        
        # Save trades
        self._save_trades()
        
        logger.info(f"Modified plan for trade {trade_id}")
        return trade
    
    def add_feedback(
        self,
        trade_id: str,
        execution_rating: int,
        strategy_fit_rating: int,
        timing_rating: int,
        management_rating: int,
        lessons: List[str],
        improvements: List[str]
    ) -> Dict[str, Any]:
        """
        Add feedback for a completed trade.
        
        Args:
            trade_id: Trade ID
            execution_rating: Rating for trade execution (1-5)
            strategy_fit_rating: Rating for strategy fit (1-5)
            timing_rating: Rating for entry/exit timing (1-5)
            management_rating: Rating for trade management (1-5)
            lessons: List of lessons learned
            improvements: List of improvements for next time
            
        Returns:
            Feedback record
        """
        # Find the trade
        trade = self._get_trade(trade_id)
        if not trade:
            logger.warning(f"Trade with ID {trade_id} not found")
            return {}
        
        # Create feedback record
        feedback = {
            'trade_id': trade_id,
            'symbol': trade['symbol'],
            'strategy': trade['strategy'],
            'direction': trade['direction'],
            'timestamp': datetime.now().isoformat(),
            'pnl': trade.get('pnl'),
            'pnl_percentage': trade.get('pnl_percentage'),
            'entry_price': trade['entry_price'],
            'exit_price': trade.get('exit_price'),
            'duration': trade.get('duration'),
            'ratings': {
                'execution': execution_rating,
                'strategy_fit': strategy_fit_rating,
                'timing': timing_rating,
                'management': management_rating,
                'overall': (execution_rating + strategy_fit_rating + timing_rating + management_rating) / 4
            },
            'lessons': lessons,
            'improvements': improvements
        }
        
        # Add feedback to journal
        self.feedback.append(feedback)
        self._save_feedback()
        
        logger.info(f"Added feedback for trade {trade_id}")
        return feedback
    
    def get_trades(
        self,
        status: Optional[str] = None,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        direction: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get trades matching specified filters.
        
        Args:
            status: Filter by trade status ('open', 'closed')
            symbol: Filter by symbol
            strategy: Filter by strategy
            direction: Filter by direction ('long', 'short')
            start_date: Filter by entry date (inclusive)
            end_date: Filter by entry date (inclusive)
            limit: Maximum number of trades to return
            
        Returns:
            List of filtered trade records
        """
        # Convert dates to datetime objects if provided
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        # Apply filters
        filtered_trades = []
        for trade in self.trades:
            # Skip if status doesn't match
            if status and trade['status'] != status:
                continue
            
            # Skip if symbol doesn't match
            if symbol and trade['symbol'] != symbol:
                continue
            
            # Skip if strategy doesn't match
            if strategy and trade['strategy'] != strategy:
                continue
            
            # Skip if direction doesn't match
            if direction and trade['direction'] != direction:
                continue
            
            # Skip if entry date is before start_date
            if start_dt:
                try:
                    entry_dt = datetime.fromisoformat(trade['entry_time'])
                    if entry_dt < start_dt:
                        continue
                except ValueError:
                    pass
            
            # Skip if entry date is after end_date
            if end_dt:
                try:
                    entry_dt = datetime.fromisoformat(trade['entry_time'])
                    if entry_dt > end_dt:
                        continue
                except ValueError:
                    pass
            
            filtered_trades.append(trade)
        
        # Sort by entry time (newest first) and limit results
        sorted_trades = sorted(
            filtered_trades,
            key=lambda x: x.get('entry_time', ''),
            reverse=True
        )
        
        return sorted_trades[:limit]
    
    def get_open_trades(self) -> List[Dict[str, Any]]:
        """
        Get all open trades.
        
        Returns:
            List of open trade records
        """
        return self.get_trades(status='open')
    
    def get_trade(self, trade_id: str) -> Dict[str, Any]:
        """
        Get a specific trade by ID.
        
        Args:
            trade_id: Trade ID
            
        Returns:
            Trade record or empty dict if not found
        """
        trade = self._get_trade(trade_id)
        return trade or {}
    
    def get_performance_metrics(
        self,
        strategy: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get performance metrics, optionally filtered by strategy, symbol, or date range.
        
        Args:
            strategy: Filter by strategy
            symbol: Filter by symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            Dictionary of performance metrics
        """
        # If no filters, return cached performance metrics
        if not strategy and not symbol and not start_date and not end_date:
            return self.performance
        
        # Filter trades
        filtered_trades = self.get_trades(
            status='closed',
            symbol=symbol,
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            limit=10000  # Large limit to ensure all trades are included
        )
        
        # Calculate metrics for filtered trades
        return self._calculate_metrics(filtered_trades)
    
    def get_strategy_insights(self) -> Dict[str, Any]:
        """
        Get insights on strategy performance.
        
        Returns:
            Dictionary of strategy insights
        """
        # Get all closed trades
        closed_trades = self.get_trades(status='closed', limit=10000)
        
        # Group trades by strategy
        strategies = {}
        
        for trade in closed_trades:
            strategy = trade.get('strategy')
            if not strategy:
                continue
            
            if strategy not in strategies:
                strategies[strategy] = []
            
            strategies[strategy].append(trade)
        
        # Calculate metrics for each strategy
        strategy_metrics = {}
        
        for strategy, trades in strategies.items():
            metrics = self._calculate_metrics(trades)
            
            # Calculate additional strategy-specific metrics
            win_rate = metrics.get('win_rate', 0)
            avg_profit = metrics.get('avg_profit', 0)
            avg_loss = metrics.get('avg_loss', 0)
            profit_factor = metrics.get('profit_factor', 0)
            
            # Calculate regime performance if market context available
            regime_performance = {}
            for trade in trades:
                if not trade.get('market_context'):
                    continue
                
                regime = trade.get('market_context', {}).get('market_regime')
                if not regime:
                    continue
                
                if regime not in regime_performance:
                    regime_performance[regime] = {
                        'count': 0,
                        'wins': 0,
                        'losses': 0,
                        'total_pnl': 0,
                    }
                
                regime_performance[regime]['count'] += 1
                pnl = trade.get('pnl', 0)
                regime_performance[regime]['total_pnl'] += pnl
                
                if pnl > 0:
                    regime_performance[regime]['wins'] += 1
                elif pnl < 0:
                    regime_performance[regime]['losses'] += 1
            
            # Calculate win rate and average PnL for each regime
            for regime, data in regime_performance.items():
                if data['count'] > 0:
                    data['win_rate'] = data['wins'] / data['count']
                    data['avg_pnl'] = data['total_pnl'] / data['count']
            
            # Store metrics
            strategy_metrics[strategy] = {
                'total_trades': len(trades),
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'avg_profit_loss_ratio': abs(avg_profit / avg_loss) if avg_loss else float('inf'),
                'total_pnl': metrics.get('total_pnl', 0),
                'regime_performance': regime_performance
            }
        
        return strategy_metrics
    
    def get_feedback_insights(self) -> Dict[str, Any]:
        """
        Get insights from trade feedback.
        
        Returns:
            Dictionary of feedback insights
        """
        if not self.feedback:
            return {}
        
        # Group feedback by strategy
        strategy_feedback = {}
        
        for feedback in self.feedback:
            strategy = feedback.get('strategy')
            if not strategy:
                continue
            
            if strategy not in strategy_feedback:
                strategy_feedback[strategy] = []
            
            strategy_feedback[strategy].append(feedback)
        
        # Calculate average ratings for each strategy
        strategy_ratings = {}
        
        for strategy, feedbacks in strategy_feedback.items():
            ratings = {
                'execution': [],
                'strategy_fit': [],
                'timing': [],
                'management': [],
                'overall': []
            }
            
            for feedback in feedbacks:
                for rating_type, rating in feedback.get('ratings', {}).items():
                    ratings[rating_type].append(rating)
            
            # Calculate averages
            avg_ratings = {}
            for rating_type, values in ratings.items():
                if values:
                    avg_ratings[rating_type] = sum(values) / len(values)
                else:
                    avg_ratings[rating_type] = 0
            
            strategy_ratings[strategy] = avg_ratings
        
        # Extract common lessons and improvements
        common_lessons = {}
        common_improvements = {}
        
        for feedback in self.feedback:
            # Process lessons
            for lesson in feedback.get('lessons', []):
                if lesson not in common_lessons:
                    common_lessons[lesson] = 0
                common_lessons[lesson] += 1
            
            # Process improvements
            for improvement in feedback.get('improvements', []):
                if improvement not in common_improvements:
                    common_improvements[improvement] = 0
                common_improvements[improvement] += 1
        
        # Sort by frequency
        sorted_lessons = sorted(
            common_lessons.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        sorted_improvements = sorted(
            common_improvements.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'strategy_ratings': strategy_ratings,
            'common_lessons': sorted_lessons[:10],  # Top 10 lessons
            'common_improvements': sorted_improvements[:10]  # Top 10 improvements
        }
    
    def get_strategy_allocation_recommendations(self) -> Dict[str, float]:
        """
        Get recommended strategy allocations based on historical performance.
        
        Returns:
            Dictionary mapping strategies to recommended allocation percentages
        """
        # Get strategy insights
        insights = self.get_strategy_insights()
        if not insights:
            return {}
        
        # Calculate baseline allocation scores
        allocation_scores = {}
        total_score = 0
        
        for strategy, metrics in insights.items():
            # Skip strategies with less than 5 trades
            if metrics.get('total_trades', 0) < 5:
                continue
            
            # Calculate score based on win rate and profit factor
            win_rate = metrics.get('win_rate', 0)
            profit_factor = metrics.get('profit_factor', 0)
            
            # Score formula can be adjusted as needed
            score = win_rate * 0.5 + min(profit_factor, 3) * 0.2 + metrics.get('avg_profit_loss_ratio', 0) * 0.3
            
            # Store score
            allocation_scores[strategy] = score
            total_score += score
        
        # Convert scores to percentages
        allocations = {}
        
        if total_score > 0:
            for strategy, score in allocation_scores.items():
                allocations[strategy] = (score / total_score) * 100
        
        return allocations
    
    def get_regime_strategy_recommendations(self) -> Dict[str, Dict[str, float]]:
        """
        Get recommended strategy allocations for different market regimes.
        
        Returns:
            Dictionary mapping regimes to strategy allocation dictionaries
        """
        # Get strategy insights
        insights = self.get_strategy_insights()
        if not insights:
            return {}
        
        # Initialize regime allocations
        regimes = [
            'bullish', 'moderately_bullish', 'neutral', 
            'moderately_bearish', 'bearish', 'volatile', 'sideways'
        ]
        
        regime_allocations = {regime: {} for regime in regimes}
        
        # Calculate scores for each strategy in each regime
        for strategy, metrics in insights.items():
            regime_perf = metrics.get('regime_performance', {})
            
            for regime, perf in regime_perf.items():
                # Skip if less than 3 trades in this regime
                if perf.get('count', 0) < 3:
                    continue
                
                # Calculate score based on win rate and average PnL
                win_rate = perf.get('win_rate', 0)
                avg_pnl = perf.get('avg_pnl', 0)
                
                # Skip if negative average PnL
                if avg_pnl <= 0:
                    continue
                
                # Score formula can be adjusted as needed
                score = win_rate * 0.7 + min(avg_pnl / 100, 1) * 0.3
                
                # Store score
                if regime not in regime_allocations:
                    regime_allocations[regime] = {}
                
                regime_allocations[regime][strategy] = score
        
        # Convert scores to percentages for each regime
        for regime, scores in regime_allocations.items():
            total_score = sum(scores.values())
            
            if total_score > 0:
                for strategy in scores:
                    scores[strategy] = (scores[strategy] / total_score) * 100
        
        return regime_allocations
    
    def export_to_csv(self, output_dir: str = None) -> Dict[str, str]:
        """
        Export journal data to CSV files.
        
        Args:
            output_dir: Directory to save CSV files (default: journal directory)
            
        Returns:
            Dictionary of file paths
        """
        # Set output directory
        if not output_dir:
            output_dir = self.journal_dir
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Export trades
        trades_df = pd.DataFrame(self.trades)
        trades_path = os.path.join(output_dir, "trades.csv")
        trades_df.to_csv(trades_path, index=False)
        
        # Export feedback
        feedback_df = pd.DataFrame(self.feedback)
        feedback_path = os.path.join(output_dir, "feedback.csv")
        feedback_df.to_csv(feedback_path, index=False)
        
        # Export performance
        performance_df = pd.DataFrame([self.performance])
        performance_path = os.path.join(output_dir, "performance.csv")
        performance_df.to_csv(performance_path, index=False)
        
        logger.info(f"Exported journal data to {output_dir}")
        
        return {
            'trades': trades_path,
            'feedback': feedback_path,
            'performance': performance_path
        }
    
    def _load_trades(self) -> List[Dict[str, Any]]:
        """Load trades from journal file."""
        if os.path.exists(self.trades_file):
            try:
                with open(self.trades_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading trades: {str(e)}")
        
        return []
    
    def _save_trades(self) -> None:
        """Save trades to journal file."""
        try:
            with open(self.trades_file, 'w') as f:
                json.dump(self.trades, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trades: {str(e)}")
    
    def _load_feedback(self) -> List[Dict[str, Any]]:
        """Load feedback from journal file."""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading feedback: {str(e)}")
        
        return []
    
    def _save_feedback(self) -> None:
        """Save feedback to journal file."""
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving feedback: {str(e)}")
    
    def _load_performance(self) -> Dict[str, Any]:
        """Load performance metrics from journal file."""
        if os.path.exists(self.performance_file):
            try:
                with open(self.performance_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading performance metrics: {str(e)}")
        
        return self._calculate_metrics([])
    
    def _save_performance(self) -> None:
        """Save performance metrics to journal file."""
        try:
            with open(self.performance_file, 'w') as f:
                json.dump(self.performance, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance metrics: {str(e)}")
    
    def _get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get a trade by ID."""
        for trade in self.trades:
            if trade['id'] == trade_id:
                return trade
        return None
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics based on current trades."""
        # Filter closed trades
        closed_trades = [t for t in self.trades if t['status'] == 'closed']
        
        # Calculate metrics
        self.performance = self._calculate_metrics(closed_trades)
        
        # Save performance metrics
        self._save_performance()
    
    def _calculate_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics for a list of trades."""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'avg_pnl_percentage': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'largest_profit': 0,
                'largest_loss': 0,
                'profit_factor': 0,
                'avg_duration': 0,
                'modified_plans': 0
            }
        
        # Initialize metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        avg_pnl_percentage = sum(t.get('pnl_percentage', 0) for t in trades) / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t.get('pnl', 0) for t in winning_trades)
        total_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
        
        avg_profit = total_profit / win_count if win_count > 0 else 0
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        
        largest_profit = max([t.get('pnl', 0) for t in trades]) if trades else 0
        largest_loss = min([t.get('pnl', 0) for t in trades]) if trades else 0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        durations = [t.get('duration', 0) for t in trades if t.get('duration') is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        modified_plans = len([t for t in trades if t.get('modified_plan', False)])
        
        return {
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_pnl_percentage': avg_pnl_percentage,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'largest_profit': largest_profit,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor,
            'avg_duration': avg_duration,
            'modified_plans': modified_plans
        }
    
    def _generate_trade_feedback(self, trade: Dict[str, Any]) -> None:
        """
        Automatically generate feedback for a completed trade.
        
        Args:
            trade: Completed trade record
        """
        if trade['status'] != 'closed':
            return
        
        # Calculate ratings
        execution_rating = 3  # Default neutral rating
        strategy_fit_rating = 3
        timing_rating = 3
        management_rating = 3
        
        lessons = []
        improvements = []
        
        # Analyze PnL
        pnl = trade.get('pnl', 0)
        
        # Analyze execution
        if pnl > 0:
            execution_rating = 4  # Good execution for profitable trade
        elif pnl < 0:
            execution_rating = 2  # Poor execution for losing trade
        
        # Analyze strategy fit
        if 'market_context' in trade and 'market_regime' in trade['market_context']:
            regime = trade['market_context']['market_regime']
            strategy = trade['strategy']
            
            # Adjust strategy fit rating based on known strategy-regime fits
            # This is a simple heuristic and should be improved with actual data
            if strategy == 'trend_following' and regime in ['bullish', 'moderately_bullish']:
                strategy_fit_rating = 4
            elif strategy == 'mean_reversion' and regime in ['sideways', 'neutral']:
                strategy_fit_rating = 4
            elif strategy == 'momentum' and regime in ['bullish', 'moderately_bullish']:
                strategy_fit_rating = 4
            elif strategy == 'volatility_breakout' and regime in ['volatile']:
                strategy_fit_rating = 4
        
        # Analyze trade management
        if trade.get('modified_plan', False):
            # Trade plan was modified
            original_plan = trade.get('original_plan', {})
            original_sl = original_plan.get('stop_loss')
            original_tp = original_plan.get('take_profit')
            
            final_sl = trade.get('stop_loss')
            final_tp = trade.get('take_profit')
            
            # Check if stop loss was loosened
            if original_sl and final_sl:
                if trade['direction'] == 'long' and final_sl < original_sl:
                    management_rating = 2  # Poor management for loosening stop loss
                    lessons.append("Avoid loosening stop loss during trade")
                elif trade['direction'] == 'short' and final_sl > original_sl:
                    management_rating = 2
                    lessons.append("Avoid loosening stop loss during trade")
            
            # Check if take profit was lowered
            if original_tp and final_tp:
                if trade['direction'] == 'long' and final_tp < original_tp:
                    # Taking profit early is not necessarily bad
                    if pnl > 0:
                        management_rating = 3  # Neutral for taking profit early
                    else:
                        management_rating = 2  # Poor if still resulted in a loss
                elif trade['direction'] == 'short' and final_tp > original_tp:
                    if pnl > 0:
                        management_rating = 3
                    else:
                        management_rating = 2
        
        # Analyze timing
        exit_reason = trade.get('exit_reason', '')
        if exit_reason == 'stop_loss':
            timing_rating = 2  # Poor timing if stop loss was hit
            lessons.append("Re-evaluate entry timing to avoid stop losses")
        elif exit_reason == 'take_profit':
            timing_rating = 4  # Good timing if take profit was hit
        elif 'trailing_stop' in exit_reason.lower():
            timing_rating = 4  # Good timing if trailing stop was hit after profit
        
        # Generate lessons and improvements
        if pnl < 0:
            lessons.append("Review entry criteria for this setup")
            improvements.append("Consider adding confirmation before entry")
        
        if trade.get('risk_reward', 0) < 1:
            lessons.append("Risk/reward ratio was less than 1:1")
            improvements.append("Only take trades with at least 2:1 risk/reward")
        
        # Add feedback
        self.add_feedback(
            trade_id=trade['id'],
            execution_rating=execution_rating,
            strategy_fit_rating=strategy_fit_rating,
            timing_rating=timing_rating,
            management_rating=management_rating,
            lessons=lessons,
            improvements=improvements
        )


class FeedbackLearningModule:
    """
    Module for learning from trade feedback and improving future trading decisions.
    """
    
    def __init__(self, journal: TradeJournal):
        """
        Initialize the feedback learning module.
        
        Args:
            journal: TradeJournal instance
        """
        self.journal = journal
        self.strategy_weights = {}
        self.regime_strategy_weights = {}
        
        # Load initial weights from journal
        self._update_weights()
    
    def _update_weights(self) -> None:
        """Update weights based on journal data."""
        # Get strategy allocation recommendations
        self.strategy_weights = self.journal.get_strategy_allocation_recommendations()
        
        # Get regime-strategy allocation recommendations
        self.regime_strategy_weights = self.journal.get_regime_strategy_recommendations()
    
    def get_strategy_allocations(self) -> Dict[str, float]:
        """
        Get recommended strategy allocations.
        
        Returns:
            Dictionary mapping strategies to allocation percentages
        """
        # Update weights if needed
        self._update_weights()
        return self.strategy_weights
    
    def get_regime_allocations(self, regime: str) -> Dict[str, float]:
        """
        Get recommended strategy allocations for a specific market regime.
        
        Args:
            regime: Market regime
            
        Returns:
            Dictionary mapping strategies to allocation percentages
        """
        # Update weights if needed
        self._update_weights()
        
        # Return regime-specific allocations if available
        if regime in self.regime_strategy_weights:
            return self.regime_strategy_weights[regime]
        
        # Fall back to general strategy allocations
        return self.strategy_weights
    
    def analyze_trade_feedback(self) -> Dict[str, Any]:
        """
        Analyze trade feedback to identify patterns and improvement areas.
        
        Returns:
            Dictionary of analysis results
        """
        # Get feedback insights
        insights = self.journal.get_feedback_insights()
        
        # Get strategy insights
        strategy_insights = self.journal.get_strategy_insights()
        
        # Combine insights
        analysis = {
            'strategy_performance': {},
            'improvement_areas': {},
            'recommendations': []
        }
        
        # Analyze strategy ratings
        strategy_ratings = insights.get('strategy_ratings', {})
        
        for strategy, ratings in strategy_ratings.items():
            performance = strategy_insights.get(strategy, {})
            
            analysis['strategy_performance'][strategy] = {
                'win_rate': performance.get('win_rate', 0),
                'avg_ratings': ratings,
                'total_trades': performance.get('total_trades', 0),
                'total_pnl': performance.get('total_pnl', 0)
            }
            
            # Identify improvement areas
            areas = {}
            
            for rating_type, value in ratings.items():
                if value < 3.0:
                    areas[rating_type] = value
            
            if areas:
                analysis['improvement_areas'][strategy] = areas
        
        # Generate recommendations
        common_lessons = insights.get('common_lessons', [])
        
        for lesson, count in common_lessons:
            if count > 1:  # Only include lessons mentioned multiple times
                analysis['recommendations'].append(lesson)
        
        return analysis 