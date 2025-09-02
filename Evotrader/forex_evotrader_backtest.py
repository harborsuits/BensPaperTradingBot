#!/usr/bin/env python3
"""
Forex EvoTrader Backtest - Session-Aware Backtesting Module

This module handles backtesting of forex strategies with:
- Session awareness (London, New York, Asia)
- Pip-based performance metrics
- Spread cost tracking
- Prop firm compliance validation
- BenBot integration
"""

import os
import sys
import yaml
import json
import logging
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('forex_evotrader_backtest')


class ForexStrategyBacktest:
    """
    Handles backtesting of forex strategies with session awareness and pip-based metrics.
    """
    
    def __init__(self, forex_evotrader, config: Dict[str, Any] = None):
        """
        Initialize the forex strategy backtest module.
        
        Args:
            forex_evotrader: Parent ForexEvoTrader instance
            config: Backtest configuration (if None, uses parent config)
        """
        self.forex_evotrader = forex_evotrader
        
        # Use parent config if none provided
        if config is None and hasattr(forex_evotrader, 'config'):
            config = forex_evotrader.config
        
        self.config = config or {}
        self.backtest_config = self.config.get('backtest', {})
        
        # Access parent components
        self.pair_manager = getattr(forex_evotrader, 'pair_manager', None)
        self.news_guard = getattr(forex_evotrader, 'news_guard', None)
        self.session_manager = getattr(forex_evotrader, 'session_manager', None)
        self.pip_logger = getattr(forex_evotrader, 'pip_logger', None)
        self.performance_tracker = getattr(forex_evotrader, 'performance_tracker', None)
        
        # Load prop firm compliance rules
        self.prop_compliance_rules = self._load_prop_compliance_rules()
        
        logger.info("Forex Strategy Backtest module initialized")
    
    def _load_prop_compliance_rules(self) -> Dict[str, Any]:
        """Load prop firm compliance rules from configuration."""
        prop_config = self.config.get('prop_compliance', {})
        
        # Default rules
        default_rules = {
            'max_drawdown_percent': 5.0,
            'daily_loss_limit_percent': 3.0,
            'target_profit_percent': 8.0,
            'verify_all_trades': True,
            'reject_non_compliant': True
        }
        
        # Merge with config
        rules = {**default_rules, **prop_config}
        
        # Load from prop risk profile if available
        risk_profile_path = self.config.get('prop_risk_profile', 'prop_risk_profile.yaml')
        if os.path.exists(risk_profile_path):
            try:
                with open(risk_profile_path, 'r') as f:
                    risk_profile = yaml.safe_load(f)
                
                # Update rules from risk profile
                if 'compliance' in risk_profile:
                    for key, value in risk_profile['compliance'].items():
                        rules[key] = value
                        
                logger.info(f"Loaded compliance rules from {risk_profile_path}")
            except Exception as e:
                logger.error(f"Error loading prop risk profile: {e}")
        
        return rules
    
    def backtest_strategy(self,
                         strategy_id: Optional[str] = None,
                         strategy_params: Optional[Dict[str, Any]] = None,
                         pair: str = 'EURUSD',
                         timeframe: str = '1h',
                         strategy_type: Optional[str] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         data: Optional[pd.DataFrame] = None,
                         session_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Backtest a forex strategy with session awareness and pip-based metrics.
        
        Args:
            strategy_id: ID of existing strategy (optional if params provided)
            strategy_params: Strategy parameters (optional if ID provided)
            pair: Currency pair (e.g., 'EURUSD')
            timeframe: Timeframe for the backtest (e.g., '1h', '4h')
            strategy_type: Type of strategy (e.g., 'london_breakout')
            start_date: Start date for backtest (optional)
            end_date: End date for backtest (optional)
            data: OHLCV data for backtest (optional, will load if not provided)
            session_filter: Filter to specific session (optional)
            
        Returns:
            Dictionary with backtest results
        """
        # Load strategy by ID if provided
        if strategy_id and not strategy_params:
            strategy_params, strategy_type = self._load_strategy_by_id(strategy_id)
            if not strategy_params:
                logger.error(f"Strategy not found: {strategy_id}")
                return {'error': f"Strategy not found: {strategy_id}"}
        
        # Validate inputs
        if not strategy_params:
            logger.error("No strategy parameters provided")
            return {'error': "No strategy parameters provided"}
        
        # Check pair validity
        if self.pair_manager and not self.pair_manager.get_pair(pair):
            logger.error(f"Invalid pair: {pair}")
            return {'error': f"Invalid pair: {pair}"}
        
        # Load or verify data
        if data is None:
            data = self._load_market_data(pair, timeframe, start_date, end_date)
            if data is None or len(data) < 100:  # Minimum data requirement
                return {'error': f"Insufficient data for {pair} {timeframe}"}
        
        logger.info(f"Running backtest on {pair} {timeframe}, {len(data)} data points")
        
        # Label data with sessions if not already done
        if self.session_manager and 'session_London' not in data.columns:
            data = self.session_manager.label_dataframe_sessions(data)
            logger.info(f"Added session labels to data")
        
        # Apply session filter if specified
        if session_filter and self.session_manager:
            data_filtered = self.session_manager.filter_by_session(data, session_filter)
            if len(data_filtered) < 100:
                logger.warning(f"Session filter produced too few data points ({len(data_filtered)}), using all data")
            else:
                data = data_filtered
                logger.info(f"Applied session filter '{session_filter}', {len(data)} data points")
        
        # Run the backtest
        backtest_results = self._run_backtest(
            data=data,
            strategy_params=strategy_params,
            strategy_type=strategy_type,
            pair=pair,
            timeframe=timeframe
        )
        
        # Apply prop firm compliance checks
        compliance_results = self._check_prop_compliance(backtest_results)
        backtest_results['compliance'] = compliance_results
        
        # Process trades by session
        if self.session_manager:
            session_results = self._analyze_session_performance(backtest_results)
            backtest_results['session_performance'] = session_results
        
        # Log trades with pip logger if available
        if self.pip_logger and 'trades' in backtest_results:
            self._log_trades_to_pip_logger(backtest_results, pair, strategy_id)
        
        # Update performance tracker if strategy_id provided
        if strategy_id and self.performance_tracker and 'session_performance' in backtest_results:
            for session, metrics in backtest_results['session_performance'].items():
                self.performance_tracker.db.update_session_performance(
                    strategy_id=strategy_id,
                    strategy_name=backtest_results.get('strategy_name', f"Strategy {strategy_id}"),
                    session=session,
                    metrics=metrics
                )
            
            logger.info(f"Updated session performance for strategy {strategy_id}")
        
        # Consult BenBot if available
        if hasattr(self.forex_evotrader, 'benbot_available') and self.forex_evotrader.benbot_available:
            benbot_feedback = self.forex_evotrader._consult_benbot('backtest_results', {
                'strategy_id': strategy_id,
                'pair': pair,
                'timeframe': timeframe,
                'summary': {
                    'total_trades': backtest_results.get('total_trades', 0),
                    'win_rate': backtest_results.get('win_rate', 0),
                    'profit_factor': backtest_results.get('profit_factor', 0),
                    'total_pips': backtest_results.get('total_pips', 0),
                    'max_drawdown': backtest_results.get('max_drawdown', 0),
                    'compliant': compliance_results.get('compliant', False)
                }
            })
            
            backtest_results['benbot_feedback'] = benbot_feedback
        
        logger.info(f"Backtest completed: {backtest_results.get('total_trades', 0)} trades")
        return backtest_results
    
    def _load_strategy_by_id(self, strategy_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Load strategy parameters by ID.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Tuple of (parameters, strategy_type) or (None, None) if not found
        """
        # Check if strategy exists in results directory
        results_dir = self.config.get('results_dir', 'results')
        
        # Try to find matching files
        strategy_files = []
        if os.path.exists(results_dir):
            for filename in os.listdir(results_dir):
                if strategy_id in filename and filename.endswith('.json'):
                    strategy_files.append(os.path.join(results_dir, filename))
        
        if strategy_files:
            # Sort by modification time (newest first)
            strategy_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            try:
                with open(strategy_files[0], 'r') as f:
                    strategy_data = json.load(f)
                
                if 'best_strategy' in strategy_data and 'parameters' in strategy_data['best_strategy']:
                    logger.info(f"Loaded strategy {strategy_id} from {strategy_files[0]}")
                    return strategy_data['best_strategy']['parameters'], strategy_data.get('template', 'unknown')
                    
            except Exception as e:
                logger.error(f"Error loading strategy file: {e}")
        
        # If not found in files, check performance tracker
        if self.performance_tracker:
            metadata = self.performance_tracker.db.get_strategy_metadata(strategy_id)
            if metadata and 'params' in metadata:
                logger.info(f"Loaded strategy {strategy_id} from performance tracker")
                return metadata['params'], metadata.get('strategy_type', 'unknown')
        
        return None, None
    
    def _load_market_data(self, 
                        pair: str, 
                        timeframe: str,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load market data for the specified pair and timeframe.
        
        Args:
            pair: Currency pair
            timeframe: Timeframe
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            DataFrame with OHLCV data or None if not available
        """
        data_dir = self.config.get('data_dir', 'data')
        filepath = f"{data_dir}/{pair}_{timeframe}.csv"
        
        if os.path.exists(filepath):
            try:
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                
                # Apply date filters if provided
                if start_date:
                    data = data[data.index >= pd.to_datetime(start_date)]
                if end_date:
                    data = data[data.index <= pd.to_datetime(end_date)]
                    
                logger.info(f"Loaded {len(data)} rows of {pair} {timeframe} data from {filepath}")
                return data
            except Exception as e:
                logger.error(f"Error loading data: {e}")
        else:
            logger.warning(f"Data file not found: {filepath}")
            
        return None
    
    def _run_backtest(self,
                    data: pd.DataFrame,
                    strategy_params: Dict[str, Any],
                    strategy_type: Optional[str],
                    pair: str,
                    timeframe: str) -> Dict[str, Any]:
        """
        Run backtest with specified parameters.
        
        Args:
            data: OHLCV data
            strategy_params: Strategy parameters
            strategy_type: Strategy type
            pair: Currency pair
            timeframe: Timeframe
            
        Returns:
            Dictionary with backtest results
        """
        # In a real implementation, this would:
        # 1. Instantiate the appropriate strategy class based on strategy_type
        # 2. Run the backtest using the strategy parameters
        # 3. Collect trades, equity curve, and performance metrics
        # 4. Calculate pip-based metrics using pair_manager
        
        # For now, we'll return simulated backtest results
        
        # Simulate an equity curve
        equity_curve = [100.0]  # Start with $100
        for i in range(1, len(data)):
            # Random daily return between -1% and 2%
            daily_return = np.random.normal(0.001, 0.01)
            equity_curve.append(equity_curve[-1] * (1 + daily_return))
        
        # Calculate peak and drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown)
        
        # Generate simulated trades
        trades = []
        total_trades = np.random.randint(50, 200)
        
        for i in range(total_trades):
            # Random entry in first 80% of data
            entry_idx = np.random.randint(0, int(len(data) * 0.8))
            # Hold for 5-30 bars
            exit_idx = min(entry_idx + np.random.randint(5, 30), len(data) - 1)
            
            # Direction (long/short)
            direction = np.random.choice([1, -1])
            
            # Entry/exit prices
            entry_price = data.iloc[entry_idx]['close']
            exit_price = data.iloc[exit_idx]['close']
            
            # Calculate pip gain/loss
            pip_multiplier = 10000  # Default for 4-decimal pairs
            if 'JPY' in pair:
                pip_multiplier = 100  # For JPY pairs (2 decimals)
                
            if self.pair_manager:
                pair_obj = self.pair_manager.get_pair(pair)
                if pair_obj:
                    pip_multiplier = pair_obj.get_pip_multiplier()
            
            price_diff = exit_price - entry_price
            pip_diff = price_diff * pip_multiplier
            pips = pip_diff * direction  # Positive if profitable in given direction
            
            # Simulated spread costs
            spread_pips = np.random.uniform(0.5, 3.0)
            net_pips = pips - spread_pips
            
            # Default pip value
            pip_value = 10.0  # $10 per pip per standard lot
            lot_size = 0.1    # 0.1 standard lots
            
            if self.pair_manager:
                pair_obj = self.pair_manager.get_pair(pair)
                if pair_obj:
                    pip_value = pair_obj.calculate_pip_value(lot_size)
            
            profit_loss = net_pips * pip_value * lot_size
            
            # Get session labels if available
            entry_time = data.index[entry_idx]
            exit_time = data.index[exit_idx]
            
            session = None
            if self.session_manager:
                active_sessions = self.session_manager.get_active_sessions(entry_time)
                if active_sessions:
                    session = active_sessions[0]
            
            # Create trade record
            trade = {
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': direction,
                'pips': pips,
                'spread_pips': spread_pips,
                'net_pips': net_pips,
                'profit_loss': profit_loss,
                'lot_size': lot_size,
                'session': session
            }
            
            trades.append(trade)
        
        # Calculate overall metrics
        winning_trades = [t for t in trades if t['net_pips'] > 0]
        losing_trades = [t for t in trades if t['net_pips'] < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        total_pips = sum(t['net_pips'] for t in trades)
        
        # Profit factor
        gross_win_pips = sum(t['net_pips'] for t in winning_trades) if winning_trades else 0
        gross_loss_pips = abs(sum(t['net_pips'] for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_win_pips / gross_loss_pips if gross_loss_pips > 0 else float('inf')
        
        # Sharpe ratio (simulated)
        daily_returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        # Compile results
        results = {
            'strategy_type': strategy_type,
            'strategy_params': strategy_params,
            'pair': pair,
            'timeframe': timeframe,
            'data_points': len(data),
            'start_date': data.index[0].strftime('%Y-%m-%d'),
            'end_date': data.index[-1].strftime('%Y-%m-%d'),
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pips': total_pips,
            'gross_win_pips': gross_win_pips,
            'gross_loss_pips': gross_loss_pips,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_equity': equity_curve[-1],
            'total_return': (equity_curve[-1] / equity_curve[0] - 1) * 100,
            'trades': trades,
            'equity_curve': equity_curve
        }
        
        return results
    
    def _analyze_session_performance(self, backtest_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze performance by trading session.
        
        Args:
            backtest_results: Backtest results dictionary
            
        Returns:
            Dictionary with session-specific metrics
        """
        if 'trades' not in backtest_results:
            return {}
        
        # Group trades by session
        session_trades = {}
        for trade in backtest_results['trades']:
            session = trade.get('session', 'Unknown')
            if session not in session_trades:
                session_trades[session] = []
            session_trades[session].append(trade)
        
        # Calculate metrics for each session
        session_metrics = {}
        
        for session, trades in session_trades.items():
            if not trades:
                continue
                
            # Basic metrics
            total_trades = len(trades)
            winning_trades = [t for t in trades if t['net_pips'] > 0]
            losing_trades = [t for t in trades if t['net_pips'] < 0]
            
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            total_pips = sum(t['net_pips'] for t in trades)
            avg_pips_per_trade = total_pips / total_trades if total_trades > 0 else 0
            
            # Profit factor
            gross_win_pips = sum(t['net_pips'] for t in winning_trades) if winning_trades else 0
            gross_loss_pips = abs(sum(t['net_pips'] for t in losing_trades)) if losing_trades else 0
            profit_factor = gross_win_pips / gross_loss_pips if gross_loss_pips > 0 else float('inf')
            
            # Store metrics
            session_metrics[session] = {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'total_pips': total_pips,
                'avg_pips_per_trade': avg_pips_per_trade,
                'profit_factor': profit_factor,
                'gross_win_pips': gross_win_pips,
                'gross_loss_pips': gross_loss_pips
            }
        
        return session_metrics
    
    def _check_prop_compliance(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if backtest results comply with prop firm rules.
        
        Args:
            backtest_results: Backtest results dictionary
            
        Returns:
            Dictionary with compliance check results
        """
        max_drawdown_percent = self.prop_compliance_rules.get('max_drawdown_percent', 5.0)
        daily_loss_limit_percent = self.prop_compliance_rules.get('daily_loss_limit_percent', 3.0)
        target_profit_percent = self.prop_compliance_rules.get('target_profit_percent', 8.0)
        
        # Check max drawdown
        actual_drawdown_percent = backtest_results.get('max_drawdown', 0) * 100
        drawdown_compliant = actual_drawdown_percent <= max_drawdown_percent
        
        # Check daily loss limit
        daily_loss_exceeded = False
        if 'equity_curve' in backtest_results:
            equity_curve = backtest_results['equity_curve']
            daily_returns = np.diff(equity_curve) / equity_curve[:-1] * 100
            worst_daily_loss = abs(min(daily_returns)) if len(daily_returns) > 0 else 0
            daily_loss_exceeded = worst_daily_loss > daily_loss_limit_percent
        
        # Check profit target
        total_return = backtest_results.get('total_return', 0)
        profit_target_reached = total_return >= target_profit_percent
        
        # Overall compliance
        compliant = drawdown_compliant and not daily_loss_exceeded
        
        # Compile results
        compliance_results = {
            'compliant': compliant,
            'drawdown_compliant': drawdown_compliant,
            'max_drawdown_percent': actual_drawdown_percent,
            'max_drawdown_limit': max_drawdown_percent,
            'daily_loss_compliant': not daily_loss_exceeded,
            'worst_daily_loss': worst_daily_loss if 'worst_daily_loss' in locals() else None,
            'daily_loss_limit': daily_loss_limit_percent,
            'profit_target_reached': profit_target_reached,
            'total_return': total_return,
            'profit_target': target_profit_percent
        }
        
        return compliance_results
    
    def _log_trades_to_pip_logger(self, backtest_results: Dict[str, Any], pair: str, strategy_id: Optional[str] = None) -> None:
        """
        Log trades to pip logger for detailed analysis.
        
        Args:
            backtest_results: Backtest results dictionary
            pair: Currency pair
            strategy_id: Strategy ID (optional)
        """
        if not self.pip_logger or 'trades' not in backtest_results:
            return
        
        for trade in backtest_results['trades']:
            try:
                self.pip_logger.log_trade(
                    pair=pair,
                    entry_time=trade['entry_time'],
                    exit_time=trade['exit_time'],
                    direction=trade['direction'],
                    entry_price=trade['entry_price'],
                    exit_price=trade['exit_price'],
                    lot_size=trade.get('lot_size', 0.1),
                    entry_spread_pips=trade.get('spread_pips', 0) / 2,  # Split spread between entry/exit
                    exit_spread_pips=trade.get('spread_pips', 0) / 2,
                    session=trade.get('session'),
                    strategy_name=strategy_id if strategy_id else backtest_results.get('strategy_type', 'Unknown')
                )
            except Exception as e:
                logger.error(f"Error logging trade to pip logger: {e}")
                
        logger.info(f"Logged {len(backtest_results['trades'])} trades to pip logger")


# Test function
def test_backtest():
    """Test the backtest functionality with sample data."""
    # Create sample OHLCV data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1h')
    data = pd.DataFrame({
        'open': np.random.normal(1.1, 0.02, len(dates)),
        'high': np.random.normal(1.11, 0.02, len(dates)),
        'low': np.random.normal(1.09, 0.02, len(dates)),
        'close': np.random.normal(1.1, 0.02, len(dates)),
        'volume': np.random.randint(100, 1000, len(dates))
    }, index=dates)
    
    # Create minimal forex evotrader mock
    class MockEvoTrader:
        def __init__(self):
            self.config = {
                'backtest': {
                    'default_timeframes': ['1h', '4h']
                },
                'prop_compliance': {
                    'max_drawdown_percent': 5.0
                }
            }
            
        def _consult_benbot(self, *args, **kwargs):
            return {'status': 'success', 'proceed': True}
    
    mock_evotrader = MockEvoTrader()
    
    # Create backtest instance
    backtest = ForexStrategyBacktest(mock_evotrader)
    
    # Run backtest
    results = backtest._run_backtest(
        data=data,
        strategy_params={'param1': 10, 'param2': 20},
        strategy_type='test_strategy',
        pair='EURUSD',
        timeframe='1h'
    )
    
    # Check compliance
    compliance = backtest._check_prop_compliance(results)
    
    # Print results
    print(f"Total trades: {results['total_trades']}")
    print(f"Win rate: {results['win_rate']:.2%}")
    print(f"Total pips: {results['total_pips']:.1f}")
    print(f"Profit factor: {results['profit_factor']:.2f}")
    print(f"Max drawdown: {results['max_drawdown']:.2%}")
    print(f"Compliant: {compliance['compliant']}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Forex Strategy Backtest")
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a test backtest"
    )
    
    args = parser.parse_args()
    
    if args.test:
        test_backtest()
    else:
        print("Use --test to run a test backtest")
