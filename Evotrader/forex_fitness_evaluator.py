#!/usr/bin/env python3
"""
Forex Fitness Evaluator

This module extends the proprietary firm fitness evaluator with specialized
metrics and optimization criteria for Forex trading, including:
- Session-specific performance evaluation
- Pip-based returns and risk metrics
- Spread cost awareness
- News impact analysis
- Liquidity condition scoring
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime, time, timedelta
import yaml

# Import base evaluator
from prop_fitness_evaluator import PropFitnessEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('forex_fitness_evaluator')


# Define trading sessions
TRADING_SESSIONS = {
    'Asia': (time(21, 0), time(7, 0)),      # 21:00-07:00 UTC
    'London': (time(7, 0), time(16, 0)),    # 07:00-16:00 UTC
    'NewYork': (time(12, 0), time(21, 0)),  # 12:00-21:00 UTC
}


class ForexFitnessEvaluator(PropFitnessEvaluator):
    """
    Evaluates trading strategies against proprietary firm criteria,
    with specialized optimizations for Forex trading.
    """
    
    def __init__(self, 
                risk_profile_path: Optional[str] = None,
                news_calendar_path: Optional[str] = None):
        """
        Initialize Forex-optimized fitness evaluator.
        
        Args:
            risk_profile_path: Path to risk profile YAML
            news_calendar_path: Path to economic news calendar
        """
        # Initialize base evaluator
        super().__init__(risk_profile_path)
        
        # Forex-specific criteria
        self.forex_criteria = {
            'min_pip_factor': 1.5,          # Min ratio of pips won to pips lost
            'min_pips_per_trade': 5.0,      # Minimum average pips per trade
            'max_spread_to_gain': 0.3,      # Maximum ratio of spread cost to gain
            'min_session_win_rate': 0.45,   # Minimum win rate per session
            'news_avoidance_bonus': 5.0,    # Bonus points for avoiding high-impact news
            'gap_handling_bonus': 5.0,      # Bonus for safely handling weekend gaps
            'multi_session_bonus': 10.0,    # Bonus for performing in multiple sessions
        }
        
        # Load news calendar if provided
        self.news_calendar = None
        if news_calendar_path and os.path.exists(news_calendar_path):
            self._load_news_calendar(news_calendar_path)
        
        # Initialize session performance tracking
        self.session_metrics = {session: {} for session in TRADING_SESSIONS}
        
        logger.info("Forex fitness evaluator initialized")
    
    def _load_news_calendar(self, calendar_path: str):
        """
        Load economic news calendar from file.
        
        Args:
            calendar_path: Path to news calendar (CSV or JSON)
        """
        try:
            if calendar_path.endswith('.csv'):
                self.news_calendar = pd.read_csv(calendar_path, parse_dates=['datetime'])
            elif calendar_path.endswith('.json'):
                with open(calendar_path, 'r') as f:
                    self.news_calendar = pd.DataFrame(json.load(f))
                    
                    # Convert datetime strings to datetime objects
                    if 'datetime' in self.news_calendar.columns:
                        self.news_calendar['datetime'] = pd.to_datetime(self.news_calendar['datetime'])
            else:
                logger.warning(f"Unsupported news calendar format: {calendar_path}")
                return
            
            logger.info(f"Loaded {len(self.news_calendar)} news events from calendar")
            
        except Exception as e:
            logger.error(f"Failed to load news calendar: {e}")
    
    def _identify_session(self, timestamp) -> str:
        """
        Identify which trading session a timestamp belongs to.
        
        Args:
            timestamp: Datetime object
            
        Returns:
            Session name or 'Unknown'
        """
        if not timestamp:
            return 'Unknown'
            
        # Extract time component
        if isinstance(timestamp, str):
            try:
                timestamp = pd.to_datetime(timestamp)
            except:
                return 'Unknown'
                
        trade_time = timestamp.time()
        
        # Check each session
        for session, (start, end) in TRADING_SESSIONS.items():
            # Handle sessions that cross midnight
            if start > end:
                if trade_time >= start or trade_time < end:
                    return session
            else:
                if start <= trade_time < end:
                    return session
        
        return 'Unknown'
    
    def _calculate_pip_value(self, symbol: str, pip_amount: float) -> float:
        """
        Calculate monetary value of pips for a given symbol.
        
        Args:
            symbol: Currency pair
            pip_amount: Number of pips
            
        Returns:
            Monetary value
        """
        # Standard lot size (100,000 units)
        lot_size = 100000
        
        # Extract currency pairs
        if '/' in symbol:
            base, quote = symbol.split('/')
        else:
            # Format like 'EURUSD'
            base = symbol[:3]
            quote = symbol[3:]
        
        # For pairs with USD as quote currency
        if quote == 'USD':
            return pip_amount * 10  # 1 pip = 10 USD for standard lot
            
        # For other pairs, would need exchange rate
        # This is a simplified version
        return pip_amount * 10
    
    def _calculate_pip_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate pip-based performance metrics.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary of pip metrics
        """
        if not trades:
            return {
                'total_pips': 0,
                'avg_pips_per_trade': 0,
                'pip_factor': 0,
                'spread_to_gain': 0
            }
        
        # Extract pip values
        pip_values = []
        spread_costs = []
        
        for trade in trades:
            # Skip trades without pip data
            if 'pips' not in trade:
                continue
                
            pip_values.append(trade['pips'])
            
            # Get spread cost if available
            if 'spread_cost' in trade:
                spread_costs.append(trade['spread_cost'])
        
        if not pip_values:
            return {
                'total_pips': 0,
                'avg_pips_per_trade': 0,
                'pip_factor': 0,
                'spread_to_gain': 0
            }
        
        # Calculate metrics
        total_pips = sum(pip_values)
        avg_pips = total_pips / len(pip_values)
        
        # Calculate pip factor (ratio of winning pips to losing pips)
        win_pips = sum(p for p in pip_values if p > 0)
        loss_pips = abs(sum(p for p in pip_values if p < 0))
        
        pip_factor = win_pips / loss_pips if loss_pips > 0 else float('inf')
        
        # Calculate spread to gain ratio
        total_spread_cost = sum(spread_costs) if spread_costs else 0
        spread_to_gain = total_spread_cost / win_pips if win_pips > 0 else float('inf')
        
        return {
            'total_pips': total_pips,
            'avg_pips_per_trade': avg_pips,
            'pip_factor': pip_factor,
            'spread_to_gain': spread_to_gain
        }
    
    def _analyze_trades_by_session(self, trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze trade performance by trading session.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary of session metrics
        """
        # Initialize session data
        session_data = {
            session: {
                'trades': [],
                'win_count': 0,
                'loss_count': 0,
                'profit': 0,
                'loss': 0,
                'pips': 0
            }
            for session in TRADING_SESSIONS
        }
        session_data['Unknown'] = {
            'trades': [],
            'win_count': 0,
            'loss_count': 0,
            'profit': 0,
            'loss': 0,
            'pips': 0
        }
        
        # Process each trade
        for trade in trades:
            # Get entry time
            entry_time = trade.get('entry_time')
            
            # Identify session
            session = self._identify_session(entry_time)
            
            # Add trade to session
            session_data[session]['trades'].append(trade)
            
            # Update session stats
            pnl = trade.get('pnl', 0)
            pips = trade.get('pips', 0)
            
            if pnl > 0:
                session_data[session]['win_count'] += 1
                session_data[session]['profit'] += pnl
            else:
                session_data[session]['loss_count'] += 1
                session_data[session]['loss'] += abs(pnl)
            
            session_data[session]['pips'] += pips
        
        # Calculate win rates and other metrics for each session
        for session, data in session_data.items():
            total_trades = data['win_count'] + data['loss_count']
            
            if total_trades > 0:
                data['win_rate'] = data['win_count'] / total_trades
                data['avg_profit_per_win'] = data['profit'] / data['win_count'] if data['win_count'] > 0 else 0
                data['avg_loss_per_loss'] = data['loss'] / data['loss_count'] if data['loss_count'] > 0 else 0
                data['profit_factor'] = data['profit'] / data['loss'] if data['loss'] > 0 else float('inf')
                data['avg_pips_per_trade'] = data['pips'] / total_trades
            else:
                data['win_rate'] = 0
                data['avg_profit_per_win'] = 0
                data['avg_loss_per_loss'] = 0
                data['profit_factor'] = 0
                data['avg_pips_per_trade'] = 0
        
        return session_data
    
    def _check_news_avoidance(self, trades: List[Dict[str, Any]]) -> Tuple[bool, int]:
        """
        Check if strategy avoids trading around high-impact news events.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Tuple of (avoids_news: bool, news_trades: int)
        """
        if not self.news_calendar or not trades:
            return True, 0
        
        # News buffer (minutes before and after news release)
        buffer_minutes = 30
        
        # Count trades near high-impact news
        news_trades = 0
        
        for trade in trades:
            entry_time = trade.get('entry_time')
            
            if not entry_time:
                continue
                
            if isinstance(entry_time, str):
                try:
                    entry_time = pd.to_datetime(entry_time)
                except:
                    continue
            
            # Check if trade is near any high-impact news
            for _, news in self.news_calendar.iterrows():
                if 'impact' in news and 'datetime' in news:
                    # Only check high-impact news
                    if news['impact'].lower() != 'high':
                        continue
                        
                    news_time = news['datetime']
                    
                    # Calculate time difference
                    time_diff = abs((entry_time - news_time).total_seconds()) / 60
                    
                    # Check if trade is within buffer window
                    if time_diff <= buffer_minutes:
                        news_trades += 1
                        break
        
        # Determine if strategy avoids news
        avoids_news = news_trades <= len(trades) * 0.1  # Less than 10% of trades near news
        
        return avoids_news, news_trades
    
    def _check_weekend_gap_handling(self, trades: List[Dict[str, Any]]) -> bool:
        """
        Check if strategy safely handles weekend gaps.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            True if strategy safely handles gaps
        """
        # Count trades held over weekend
        weekend_trades = 0
        weekend_losses = 0
        
        for trade in trades:
            entry_time = trade.get('entry_time')
            exit_time = trade.get('exit_time')
            
            if not entry_time or not exit_time:
                continue
                
            if isinstance(entry_time, str):
                try:
                    entry_time = pd.to_datetime(entry_time)
                except:
                    continue
                    
            if isinstance(exit_time, str):
                try:
                    exit_time = pd.to_datetime(exit_time)
                except:
                    continue
            
            # Check if trade spans weekend (Friday to Monday)
            entry_day = entry_time.weekday()
            exit_day = exit_time.weekday()
            
            # Friday is 4, Monday is 0
            if entry_day <= 4 and exit_day >= 0 and (exit_day < entry_day or exit_time - entry_time > timedelta(days=2)):
                weekend_trades += 1
                
                # Check if trade was a loss
                if trade.get('pnl', 0) < 0:
                    weekend_losses += 1
        
        # Calculate weekend loss ratio
        weekend_loss_ratio = weekend_losses / weekend_trades if weekend_trades > 0 else 0
        
        # Strategy handles gaps well if less than 30% of weekend trades are losses
        handles_gaps = weekend_loss_ratio <= 0.3
        
        return handles_gaps
    
    def evaluate_forex_backtest(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate backtest results with Forex-specific criteria.
        
        Args:
            backtest_results: Dictionary with backtest results
            
        Returns:
            Enhanced evaluation results
        """
        # First get base evaluation
        base_evaluation = self.evaluate_backtest_results(backtest_results)
        
        # Extract trades
        trades = backtest_results.get('trades', [])
        
        # Calculate pip metrics
        pip_metrics = self._calculate_pip_metrics(trades)
        
        # Analyze trades by session
        session_analysis = self._analyze_trades_by_session(trades)
        
        # Check news avoidance
        avoids_news, news_trades = self._check_news_avoidance(trades)
        
        # Check weekend gap handling
        handles_gaps = self._check_weekend_gap_handling(trades)
        
        # Determine active sessions (sessions with at least 5 trades)
        active_sessions = [
            session for session, data in session_analysis.items()
            if len(data['trades']) >= 5 and session != 'Unknown'
        ]
        
        # Calculate session performance score
        session_scores = {}
        for session in active_sessions:
            data = session_analysis[session]
            
            # Calculate session score based on win rate, profit factor, and pips
            session_scores[session] = (
                min(100, data['win_rate'] * 100) * 0.3 +
                min(100, data['profit_factor'] * 20) * 0.4 +
                min(100, data['avg_pips_per_trade'] * 10) * 0.3
            )
        
        # Determine if strategy meets forex criteria
        meets_pip_factor = pip_metrics['pip_factor'] >= self.forex_criteria['min_pip_factor']
        meets_pips_per_trade = pip_metrics['avg_pips_per_trade'] >= self.forex_criteria['min_pips_per_trade']
        meets_spread_to_gain = pip_metrics['spread_to_gain'] <= self.forex_criteria['max_spread_to_gain']
        
        # Check session win rate for active sessions
        session_win_rates = {
            session: session_analysis[session]['win_rate']
            for session in active_sessions
        }
        
        meets_session_win_rate = all(
            win_rate >= self.forex_criteria['min_session_win_rate']
            for win_rate in session_win_rates.values()
        )
        
        # Generate forex-specific threshold results
        forex_thresholds = {
            'meets_pip_factor': meets_pip_factor,
            'meets_pips_per_trade': meets_pips_per_trade,
            'meets_spread_to_gain': meets_spread_to_gain,
            'meets_session_win_rate': meets_session_win_rate,
            'avoids_news': avoids_news,
            'handles_gaps': handles_gaps,
            'active_sessions': active_sessions
        }
        
        # Calculate bonus points for forex-specific factors
        bonus_points = 0
        
        # Bonus for news avoidance
        if avoids_news:
            bonus_points += self.forex_criteria['news_avoidance_bonus']
        
        # Bonus for weekend gap handling
        if handles_gaps:
            bonus_points += self.forex_criteria['gap_handling_bonus']
        
        # Bonus for multi-session performance
        if len(active_sessions) > 1:
            # Calculate the proportion of the bonus based on number of active sessions
            session_proportion = min(1.0, len(active_sessions) / 3)  # Max bonus for 3+ sessions
            bonus_points += self.forex_criteria['multi_session_bonus'] * session_proportion
        
        # Enhance base score with forex bonuses
        forex_score = base_evaluation['score'] + bonus_points
        
        # Combine results
        forex_evaluation = base_evaluation.copy()
        forex_evaluation['score'] = forex_score
        forex_evaluation['forex_thresholds'] = forex_thresholds
        forex_evaluation['pip_metrics'] = pip_metrics
        forex_evaluation['session_analysis'] = {
            session: {k: v for k, v in data.items() if k != 'trades'}
            for session, data in session_analysis.items()
        }
        forex_evaluation['session_scores'] = session_scores
        forex_evaluation['bonus_points'] = bonus_points
        
        # Determine overall forex compliance
        forex_compliant = (
            base_evaluation['passes_evaluation'] and
            meets_pip_factor and
            meets_pips_per_trade and
            meets_session_win_rate
        )
        
        forex_evaluation['forex_compliant'] = forex_compliant
        
        return forex_evaluation
    
    def evaluate_strategy(self, strategy, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate a Forex strategy on market data.
        
        Args:
            strategy: Strategy object with backtest method
            market_data: Market data as pandas DataFrame
            
        Returns:
            Forex-optimized evaluation results
        """
        try:
            # Run backtest if strategy has backtest method
            if hasattr(strategy, 'backtest'):
                backtest_results = strategy.backtest(market_data)
            else:
                # Try importing backtest function from advanced_strategies
                from advanced_strategies import backtest_strategy
                backtest_results = backtest_strategy(strategy, market_data)
                
            # Process trades to enhance with forex-specific data if needed
            trades = backtest_results.get('trades', [])
            
            # Add pip calculations if not present
            for trade in trades:
                if 'pips' not in trade and 'entry_price' in trade and 'exit_price' in trade:
                    # Simple pip calculation (simplified, normally depends on the pair)
                    price_diff = trade['exit_price'] - trade['entry_price']
                    direction = trade.get('direction', 1)  # 1 for long, -1 for short
                    
                    # For 4-digit pairs (most pairs except JPY)
                    trade['pips'] = direction * price_diff * 10000
                    
                    # For JPY pairs, would use:
                    # if 'JPY' in trade.get('symbol', ''):
                    #     trade['pips'] = direction * price_diff * 100
            
            # Update backtest results with enhanced trades
            backtest_results['trades'] = trades
                
            # Evaluate with forex-specific criteria
            return self.evaluate_forex_backtest(backtest_results)
            
        except Exception as e:
            logger.error(f"Error evaluating forex strategy: {e}")
            return {
                'passes_evaluation': False,
                'forex_compliant': False,
                'score': 0,
                'error': str(e)
            }


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate forex strategy against prop firm criteria")
    
    parser.add_argument(
        "--strategy", 
        type=str, 
        required=True,
        help="Path to strategy file or strategy name"
    )
    
    parser.add_argument(
        "--market-data", 
        type=str, 
        required=True,
        help="Path to market data CSV file"
    )
    
    parser.add_argument(
        "--risk-profile", 
        type=str, 
        default=None,
        help="Path to risk profile YAML file"
    )
    
    parser.add_argument(
        "--news-calendar", 
        type=str, 
        default=None,
        help="Path to economic news calendar"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output file path for evaluation results (JSON)"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ForexFitnessEvaluator(args.risk_profile, args.news_calendar)
    
    # Load market data
    try:
        market_data = pd.read_csv(args.market_data, parse_dates=['date'])
    except Exception as e:
        logger.error(f"Failed to load market data: {e}")
        import sys
        sys.exit(1)
    
    # Load strategy
    try:
        # This is a simplified example. Real implementation would need to handle
        # different ways to load strategies (from file, from registry, etc.)
        import importlib.util
        import sys
        
        module_name = args.strategy.split('/')[-1].replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, args.strategy)
        strategy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(strategy_module)
        
        # Assume the strategy is the main class in the module
        strategy_class = getattr(strategy_module, module_name)
        strategy = strategy_class()
        
    except Exception as e:
        logger.error(f"Failed to load strategy: {e}")
        sys.exit(1)
    
    # Evaluate strategy
    evaluation_results = evaluator.evaluate_strategy(strategy, market_data)
    
    # Output results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
    else:
        import json
        print(json.dumps(evaluation_results, indent=2))
