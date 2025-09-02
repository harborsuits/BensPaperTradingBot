#!/usr/bin/env python3
"""
Proprietary Firm Compliance Filter

This module implements strict compliance filtering for strategies post-evolution
to ensure they strictly adhere to all proprietary trading firm requirements.
It acts as a final verification layer before strategies are approved for live testing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import datetime
import json
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('prop_compliance_filter')


class PropComplianceFilter:
    """
    Enforces compliance with proprietary firm rules.
    
    This filter performs more stringent checks than the fitness evaluator
    and is used as a final verification step before strategies are
    approved for forward testing or live deployment.
    """
    
    def __init__(self, risk_profile_path: Optional[str] = None):
        """
        Initialize with optional risk profile.
        
        Args:
            risk_profile_path: Path to YAML risk profile configuration
        """
        # Default compliance thresholds (stricter than fitness evaluation)
        self.thresholds = {
            'max_drawdown': 4.5,           # Maximum drawdown (slightly below 5% for safety)
            'daily_loss_limit': 2.5,       # Maximum daily loss (slightly below 3% for safety)
            'profit_target_min': 8.0,      # Minimum profit target
            'max_leverage': 10.0,          # Maximum leverage
            'max_position_size_pct': 5.0,  # Maximum position size as % of account
            'min_win_rate': 0.45,          # Minimum win rate
            'min_profit_factor': 1.2,      # Minimum profit factor
            'max_correlation': 0.8,        # Maximum correlation with other approved strategies
            'max_trade_frequency': 100,    # Maximum trades per day
            'min_trade_duration': 5,       # Minimum trade duration in minutes
            'min_trades_for_analysis': 20  # Minimum trades required for meaningful analysis
        }
        
        # Symbol whitelist/blacklist
        self.symbol_whitelist = None  # Allow all symbols by default
        self.symbol_blacklist = []    # No blacklisted symbols by default
        
        # Restricted trading hours (if any)
        self.restricted_hours = []
        
        # Maximum consecutive losses allowed
        self.max_consecutive_losses = 5
        
        # Load custom risk profile if provided
        if risk_profile_path:
            self._load_risk_profile(risk_profile_path)
    
    def _load_risk_profile(self, profile_path: str):
        """
        Load risk profile from YAML file.
        
        Args:
            profile_path: Path to risk profile configuration
        """
        try:
            import yaml
            with open(profile_path, 'r') as file:
                risk_profile = yaml.safe_load(file)
            
            # Update thresholds
            if 'compliance_thresholds' in risk_profile:
                for key, value in risk_profile['compliance_thresholds'].items():
                    if key in self.thresholds:
                        self.thresholds[key] = value
            
            # Update symbol lists
            if 'symbol_whitelist' in risk_profile:
                self.symbol_whitelist = risk_profile['symbol_whitelist']
            
            if 'symbol_blacklist' in risk_profile:
                self.symbol_blacklist = risk_profile['symbol_blacklist']
            
            # Update restricted hours
            if 'restricted_hours' in risk_profile:
                self.restricted_hours = risk_profile['restricted_hours']
            
            # Update max consecutive losses
            if 'max_consecutive_losses' in risk_profile:
                self.max_consecutive_losses = risk_profile['max_consecutive_losses']
            
            logger.info(f"Loaded compliance settings from {profile_path}")
            
        except Exception as e:
            logger.error(f"Failed to load risk profile: {e}")
    
    def check_symbol_compliance(self, symbols: List[str]) -> Tuple[bool, List[str]]:
        """
        Check if all symbols comply with whitelist/blacklist rules.
        
        Args:
            symbols: List of symbols used by the strategy
            
        Returns:
            Tuple of (compliant: bool, non_compliant_symbols: List[str])
        """
        non_compliant = []
        
        for symbol in symbols:
            # Check if symbol is blacklisted
            if symbol in self.symbol_blacklist:
                non_compliant.append(f"{symbol} (blacklisted)")
                continue
            
            # Check if symbol is not in whitelist (if whitelist is active)
            if self.symbol_whitelist is not None and symbol not in self.symbol_whitelist:
                non_compliant.append(f"{symbol} (not in whitelist)")
        
        return len(non_compliant) == 0, non_compliant
    
    def check_trading_hours_compliance(self, trade_timestamps: List[datetime.datetime]) -> Tuple[bool, List[str]]:
        """
        Check if all trades comply with trading hour restrictions.
        
        Args:
            trade_timestamps: List of trade timestamps
            
        Returns:
            Tuple of (compliant: bool, non_compliant_times: List[str])
        """
        if not self.restricted_hours:
            return True, []
        
        non_compliant = []
        
        for timestamp in trade_timestamps:
            # For each restricted period
            for period in self.restricted_hours:
                start_time = period.get('start')
                end_time = period.get('end')
                
                # Skip if start or end time not defined
                if not start_time or not end_time:
                    continue
                
                # Parse time strings into hours and minutes
                start_h, start_m = map(int, start_time.split(':'))
                end_h, end_m = map(int, end_time.split(':'))
                
                # Check if timestamp is within restricted period
                trade_time = timestamp.time()
                start = datetime.time(start_h, start_m)
                end = datetime.time(end_h, end_m)
                
                if start <= trade_time <= end:
                    non_compliant.append(f"{timestamp} (restricted: {start_time}-{end_time})")
                    break
        
        return len(non_compliant) == 0, non_compliant
    
    def check_consecutive_losses(self, trade_results: List[float]) -> Tuple[bool, int]:
        """
        Check if strategy exceeds max consecutive losses.
        
        Args:
            trade_results: List of trade P&L values
            
        Returns:
            Tuple of (compliant: bool, max_consecutive_losses: int)
        """
        max_streak = 0
        current_streak = 0
        
        for result in trade_results:
            if result < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak <= self.max_consecutive_losses, max_streak
    
    def check_leverage_compliance(self, max_leverage_used: float) -> Tuple[bool, float]:
        """
        Check if leverage used complies with maximum allowed.
        
        Args:
            max_leverage_used: Maximum leverage used by strategy
            
        Returns:
            Tuple of (compliant: bool, max_leverage_used: float)
        """
        return max_leverage_used <= self.thresholds['max_leverage'], max_leverage_used
    
    def check_position_size_compliance(self, position_sizes: List[float]) -> Tuple[bool, float]:
        """
        Check if position sizes comply with maximum allowed.
        
        Args:
            position_sizes: List of position sizes as percentage of account
            
        Returns:
            Tuple of (compliant: bool, max_position_size: float)
        """
        if not position_sizes:
            return True, 0.0
            
        max_size = max(position_sizes)
        return max_size <= self.thresholds['max_position_size_pct'], max_size
    
    def check_trade_frequency_compliance(self, trades_per_day: Dict[str, int]) -> Tuple[bool, int]:
        """
        Check if trade frequency complies with maximum allowed.
        
        Args:
            trades_per_day: Dictionary of {date: trade_count}
            
        Returns:
            Tuple of (compliant: bool, max_trades_per_day: int)
        """
        if not trades_per_day:
            return True, 0
            
        max_trades = max(trades_per_day.values()) if trades_per_day else 0
        return max_trades <= self.thresholds['max_trade_frequency'], max_trades
    
    def check_metric_compliance(self, metric_name: str, metric_value: float) -> bool:
        """
        Check if a metric complies with its threshold.
        
        Args:
            metric_name: Name of metric to check
            metric_value: Value of metric
            
        Returns:
            True if compliant, False otherwise
        """
        # Dictionary mapping metrics to compliance check functions
        compliance_checks = {
            'max_drawdown': lambda x: x <= self.thresholds['max_drawdown'],
            'worst_daily_loss': lambda x: x <= self.thresholds['daily_loss_limit'],
            'total_return_pct': lambda x: x >= self.thresholds['profit_target_min'],
            'win_rate': lambda x: x >= self.thresholds['min_win_rate'],
            'profit_factor': lambda x: x >= self.thresholds['min_profit_factor']
        }
        
        if metric_name not in compliance_checks:
            return True  # If no specific check defined, assume compliant
        
        return compliance_checks[metric_name](metric_value)
    
    def analyze_backtest_results(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive compliance analysis on backtest results.
        
        Args:
            backtest_results: Dictionary of backtest results
            
        Returns:
            Dictionary with compliance analysis results
        """
        # Extract metrics from backtest results
        metrics = backtest_results.get('metrics', {})
        trades = backtest_results.get('trades', [])
        
        # Number of trades check
        if len(trades) < self.thresholds['min_trades_for_analysis']:
            return {
                'compliant': False,
                'reason': f"Insufficient trades for analysis: {len(trades)} < {self.thresholds['min_trades_for_analysis']}",
                'checks': {'min_trades': False}
            }
        
        # Initialize compliance checks dictionary
        compliance_checks = {}
        non_compliant_reasons = []
        
        # Check core metrics compliance
        for metric_name in ['max_drawdown', 'worst_daily_loss', 'total_return_pct', 'win_rate', 'profit_factor']:
            if metric_name in metrics:
                compliant = self.check_metric_compliance(metric_name, metrics[metric_name])
                compliance_checks[metric_name] = compliant
                
                if not compliant:
                    threshold = self.thresholds.get(f'{"min" if "min" in metric_name else "max"}_{metric_name}', 
                                                  self.thresholds.get(metric_name))
                    non_compliant_reasons.append(
                        f"{metric_name}: {metrics[metric_name]} (threshold: {threshold})"
                    )
        
        # Extract trade data for additional compliance checks
        if trades:
            # Check for symbols compliance
            trade_symbols = list(set(trade.get('symbol', 'unknown') for trade in trades if 'symbol' in trade))
            symbols_compliant, non_compliant_symbols = self.check_symbol_compliance(trade_symbols)
            compliance_checks['symbols'] = symbols_compliant
            
            if not symbols_compliant:
                non_compliant_reasons.append(f"Non-compliant symbols: {', '.join(non_compliant_symbols)}")
            
            # Check for trading hours compliance
            trade_times = [trade.get('entry_time') for trade in trades if 'entry_time' in trade]
            hours_compliant, non_compliant_times = self.check_trading_hours_compliance(trade_times)
            compliance_checks['trading_hours'] = hours_compliant
            
            if not hours_compliant:
                non_compliant_reasons.append(f"Trades during restricted hours: {len(non_compliant_times)}")
            
            # Check for consecutive losses
            trade_results = [trade.get('pnl', 0) for trade in trades if 'pnl' in trade]
            losses_compliant, max_consecutive = self.check_consecutive_losses(trade_results)
            compliance_checks['consecutive_losses'] = losses_compliant
            
            if not losses_compliant:
                non_compliant_reasons.append(
                    f"Max consecutive losses: {max_consecutive} (threshold: {self.max_consecutive_losses})"
                )
            
            # Check for leverage compliance if available
            if 'max_leverage' in metrics:
                leverage_compliant, max_leverage = self.check_leverage_compliance(metrics['max_leverage'])
                compliance_checks['leverage'] = leverage_compliant
                
                if not leverage_compliant:
                    non_compliant_reasons.append(
                        f"Max leverage: {max_leverage} (threshold: {self.thresholds['max_leverage']})"
                    )
            
            # Check for position size compliance
            position_sizes = [trade.get('position_size_pct', 0) for trade in trades if 'position_size_pct' in trade]
            
            if position_sizes:
                position_compliant, max_position = self.check_position_size_compliance(position_sizes)
                compliance_checks['position_size'] = position_compliant
                
                if not position_compliant:
                    non_compliant_reasons.append(
                        f"Max position size: {max_position}% (threshold: {self.thresholds['max_position_size_pct']}%)"
                    )
            
            # Check for trade frequency compliance
            trades_per_day = {}
            for trade in trades:
                if 'entry_time' in trade:
                    entry_date = trade['entry_time'].date().isoformat()
                    trades_per_day[entry_date] = trades_per_day.get(entry_date, 0) + 1
            
            frequency_compliant, max_frequency = self.check_trade_frequency_compliance(trades_per_day)
            compliance_checks['trade_frequency'] = frequency_compliant
            
            if not frequency_compliant:
                non_compliant_reasons.append(
                    f"Max trades per day: {max_frequency} (threshold: {self.thresholds['max_trade_frequency']})"
                )
        
        # Determine overall compliance
        overall_compliant = all(compliance_checks.values())
        
        # Build compliance report
        compliance_report = {
            'compliant': overall_compliant,
            'checks': compliance_checks,
            'non_compliant_reasons': non_compliant_reasons if not overall_compliant else [],
            'thresholds_applied': self.thresholds.copy(),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        return compliance_report
    
    def evaluate_strategy(self, strategy, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run a full compliance evaluation on a strategy.
        
        Args:
            strategy: Strategy object with backtest method
            market_data: Market data as pandas DataFrame
            
        Returns:
            Compliance evaluation results
        """
        try:
            # Run backtest if strategy has backtest method
            if hasattr(strategy, 'backtest'):
                backtest_results = strategy.backtest(market_data)
            else:
                # Try importing backtest function from advanced_strategies
                from advanced_strategies import backtest_strategy
                backtest_results = backtest_strategy(strategy, market_data)
                
            # Analyze backtest results for compliance
            compliance_results = self.analyze_backtest_results(backtest_results)
            
            # Add basic strategy info
            compliance_results['strategy_info'] = {
                'name': strategy.__class__.__name__,
                'parameters': getattr(strategy, 'parameters', {}),
                'strategy_type': getattr(strategy, 'strategy_type', 'unknown')
            }
            
            return compliance_results
            
        except Exception as e:
            logger.error(f"Error evaluating strategy compliance: {e}")
            return {
                'compliant': False,
                'error': str(e)
            }
    
    def filter_compliant_strategies(self, strategies: List[Any], market_data: pd.DataFrame) -> Dict[str, List[Any]]:
        """
        Filter a list of strategies into compliant and non-compliant groups.
        
        Args:
            strategies: List of strategy objects
            market_data: Market data for backtesting
            
        Returns:
            Dictionary with 'compliant' and 'non_compliant' strategy lists
        """
        compliant_strategies = []
        non_compliant_strategies = []
        compliance_reports = {}
        
        for strategy in strategies:
            strategy_name = strategy.__class__.__name__
            logger.info(f"Evaluating compliance for strategy: {strategy_name}")
            
            compliance_result = self.evaluate_strategy(strategy, market_data)
            compliance_reports[strategy_name] = compliance_result
            
            if compliance_result.get('compliant', False):
                compliant_strategies.append(strategy)
            else:
                non_compliant_strategies.append(strategy)
        
        logger.info(f"Compliance filtering complete. {len(compliant_strategies)} compliant, "
                   f"{len(non_compliant_strategies)} non-compliant strategies.")
        
        return {
            'compliant': compliant_strategies,
            'non_compliant': non_compliant_strategies,
            'reports': compliance_reports
        }


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter strategies for compliance with prop firm rules")
    
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
        "--output", 
        type=str, 
        default=None,
        help="Output file path for compliance results (JSON)"
    )
    
    args = parser.parse_args()
    
    # Initialize filter
    compliance_filter = PropComplianceFilter(args.risk_profile)
    
    # Load market data
    try:
        market_data = pd.read_csv(args.market_data, parse_dates=['date'])
    except Exception as e:
        logger.error(f"Failed to load market data: {e}")
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
    
    # Evaluate strategy compliance
    compliance_results = compliance_filter.evaluate_strategy(strategy, market_data)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(compliance_results, f, indent=2)
    else:
        print(json.dumps(compliance_results, indent=2))
