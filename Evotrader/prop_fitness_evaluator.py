#!/usr/bin/env python3
"""
Proprietary Firm Fitness Evaluator

This module implements a specialized fitness function designed specifically
for proprietary trading firm evaluations, optimizing for:
- Sharpe Ratio (risk-adjusted returns)
- Maximum drawdown compliance
- Daily loss limits
- Performance stability
- Profit targets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('prop_fitness_evaluator')


class PropFitnessEvaluator:
    """
    Evaluates trading strategies against proprietary firm criteria.
    
    Designed to prioritize risk management, consistency, and compliance
    with funded account requirements.
    """
    
    def __init__(self, risk_profile_path: Optional[str] = None):
        """
        Initialize the evaluator with optional risk profile parameters.
        
        Args:
            risk_profile_path: Path to YAML risk profile config file
        """
        # Default prop firm criteria
        self.criteria = {
            'max_drawdown': 5.0,        # Maximum drawdown percentage
            'profit_target': 8.0,       # Minimum profit target percentage
            'daily_loss_limit': 3.0,    # Maximum daily loss percentage
            'min_trading_days': 10,     # Minimum number of active trading days
            'min_profitable_days': 0.6, # Minimum percentage of profitable days
            'min_trades': 15,           # Minimum number of trades
            'min_reward_risk': 1.5,     # Minimum reward-to-risk ratio
            'min_sharpe': 1.0,          # Minimum Sharpe ratio
        }
        
        # Load custom risk profile if provided
        if risk_profile_path:
            self._load_risk_profile(risk_profile_path)
    
    def _load_risk_profile(self, profile_path: str):
        """
        Load risk profile from YAML file.
        
        Args:
            profile_path: Path to risk profile YAML
        """
        try:
            import yaml
            with open(profile_path, 'r') as file:
                risk_profile = yaml.safe_load(file)
                
            # Update criteria with loaded values
            for key, value in risk_profile.get('criteria', {}).items():
                if key in self.criteria:
                    self.criteria[key] = value
                    
            logger.info(f"Loaded risk profile from {profile_path}")
            
        except Exception as e:
            logger.error(f"Failed to load risk profile: {e}")
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio based on daily returns.
        
        Args:
            returns: List of percentage returns (not decimal)
            risk_free_rate: Annual risk-free rate (default 0%)
            
        Returns:
            Sharpe ratio (annualized)
        """
        # Convert percentage returns to decimal
        decimal_returns = [r / 100 for r in returns]
        
        # Annualized Sharpe calculation
        daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = [r - daily_risk_free for r in decimal_returns]
        
        if not excess_returns or len(excess_returns) < 2:
            return 0.0
            
        mean_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns, ddof=1)
        
        if std_excess_return == 0:
            return 0.0
            
        daily_sharpe = mean_excess_return / std_excess_return
        
        # Annualize
        annual_sharpe = daily_sharpe * np.sqrt(252)
        
        return annual_sharpe
    
    def calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sortino ratio based on daily returns (only downside risk).
        
        Args:
            returns: List of percentage returns (not decimal)
            risk_free_rate: Annual risk-free rate (default 0%)
            
        Returns:
            Sortino ratio (annualized)
        """
        # Convert percentage returns to decimal
        decimal_returns = [r / 100 for r in returns]
        
        # Annualized Sortino calculation
        daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = [r - daily_risk_free for r in decimal_returns]
        
        if not excess_returns or len(excess_returns) < 2:
            return 0.0
            
        mean_excess_return = np.mean(excess_returns)
        
        # Calculate downside deviation (only negative returns)
        negative_returns = [r for r in excess_returns if r < 0]
        
        if not negative_returns:
            # No negative returns is excellent, but to avoid division by zero
            return 10.0  # Assign a high but reasonable value
            
        downside_deviation = np.sqrt(np.mean(np.square(negative_returns)))
        
        if downside_deviation == 0:
            return 0.0
            
        daily_sortino = mean_excess_return / downside_deviation
        
        # Annualize
        annual_sortino = daily_sortino * np.sqrt(252)
        
        return annual_sortino
    
    def calculate_calmar_ratio(self, returns: List[float], max_drawdown_pct: float) -> float:
        """
        Calculate Calmar ratio (return / maximum drawdown).
        
        Args:
            returns: List of percentage returns (not decimal)
            max_drawdown_pct: Maximum drawdown percentage
            
        Returns:
            Calmar ratio (annualized return / max drawdown)
        """
        if not returns:
            return 0.0
            
        if max_drawdown_pct <= 0:
            return 0.0  # Avoid division by zero
            
        # Convert percentage returns to decimal for calculation
        decimal_returns = [r / 100 for r in returns]
        
        # Calculate annualized return
        cumulative_return = np.prod([1 + r for r in decimal_returns]) - 1
        trading_days = len(returns)
        
        if trading_days < 5:  # Too few days for meaningful annualization
            return 0.0
            
        # Annualize the return
        annual_return = (1 + cumulative_return) ** (252 / trading_days) - 1
        
        # Calculate Calmar ratio
        calmar_ratio = annual_return / (max_drawdown_pct / 100)
        
        return calmar_ratio
    
    def calculate_stability_score(self, daily_returns: List[float]) -> float:
        """
        Calculate stability score based on daily returns consistency.
        
        Args:
            daily_returns: List of daily percentage returns
            
        Returns:
            Stability score (0-100)
        """
        if not daily_returns or len(daily_returns) < 5:
            return 0.0
            
        # Calculate volatility of returns
        volatility = np.std(daily_returns)
        
        # Calculate autocorrelation to detect patterns
        try:
            from statsmodels.tsa.stattools import acf
            autocorrelation = acf(daily_returns, nlags=1)[1]  # lag 1 autocorrelation
        except:
            # Fallback if statsmodels not available
            returns_shifted = daily_returns[:-1]
            returns = daily_returns[1:]
            
            if not returns or len(returns) < 2:
                autocorrelation = 0
            else:
                cov = np.cov(returns_shifted, returns)[0][1]
                var_shifted = np.var(returns_shifted)
                var = np.var(returns)
                
                if var_shifted == 0 or var == 0:
                    autocorrelation = 0
                else:
                    autocorrelation = cov / np.sqrt(var_shifted * var)
        
        # Calculate percentage of profitable days
        profitable_days = sum(1 for r in daily_returns if r > 0)
        profitable_pct = profitable_days / len(daily_returns) if daily_returns else 0
        
        # More consistent strategies have:
        # - Lower volatility (but not zero, as that might indicate lack of trading)
        # - Positive autocorrelation (momentum)
        # - Higher percentage of profitable days
        
        # Convert volatility to score (lower volatility = higher score)
        # but penalize very low volatility (< 0.1) as it might indicate lack of trading
        if volatility < 0.1:
            volatility_score = 50  # Penalize extremely low volatility
        else:
            volatility_score = 100 * np.exp(-volatility / 5)  # Scale to reasonable range
            
        # Convert autocorrelation to score
        autocorrelation_score = 50 + (autocorrelation * 50)  # Center around 50
        
        # Convert profitable days percentage to score
        profitable_days_score = profitable_pct * 100
        
        # Combine scores with weights
        stability_score = (
            volatility_score * 0.4 +
            autocorrelation_score * 0.2 +
            profitable_days_score * 0.4
        )
        
        return min(100, max(0, stability_score))
    
    def evaluate_backtest_results(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate backtest results against proprietary firm criteria.
        
        Args:
            backtest_results: Dictionary containing backtest metrics
            
        Returns:
            Dictionary with evaluation results and fitness score
        """
        # Extract key metrics from backtest results
        metrics = backtest_results.get('metrics', {})
        
        # Extract daily returns if available
        daily_returns = backtest_results.get('daily_returns', [])
        
        # Calculate additional metrics if not already present
        total_return = metrics.get('total_return_pct', 0.0)
        max_drawdown = metrics.get('max_drawdown', 0.0)
        worst_daily_loss = metrics.get('worst_daily_loss', 0.0)
        win_rate = metrics.get('win_rate', 0.0)
        profit_factor = metrics.get('profit_factor', 0.0)
        
        # Get number of trades
        num_trades = metrics.get('total_trades', 0)
        
        # Calculate profitable days percentage
        if daily_returns:
            profitable_days = sum(1 for r in daily_returns if r > 0)
            profitable_days_pct = profitable_days / len(daily_returns)
        else:
            profitable_days_pct = metrics.get('profitable_days_pct', 0.0)
        
        # Calculate Sharpe and Sortino ratios
        sharpe_ratio = metrics.get('sharpe_ratio', 
                                  self.calculate_sharpe_ratio(daily_returns) if daily_returns else 0.0)
        
        sortino_ratio = metrics.get('sortino_ratio',
                                   self.calculate_sortino_ratio(daily_returns) if daily_returns else 0.0)
        
        # Calculate Calmar ratio
        calmar_ratio = metrics.get('calmar_ratio',
                                  self.calculate_calmar_ratio(daily_returns, max_drawdown) if daily_returns else 0.0)
        
        # Calculate stability score
        stability_score = self.calculate_stability_score(daily_returns)
        
        # Check criteria thresholds
        meets_max_drawdown = max_drawdown <= self.criteria['max_drawdown']
        meets_profit_target = total_return >= self.criteria['profit_target']
        meets_daily_loss = worst_daily_loss <= self.criteria['daily_loss_limit']
        meets_min_trades = num_trades >= self.criteria['min_trades']
        meets_min_sharpe = sharpe_ratio >= self.criteria['min_sharpe']
        meets_min_reward_risk = profit_factor >= self.criteria['min_reward_risk']
        meets_min_profitable_days = profitable_days_pct >= self.criteria['min_profitable_days']
        
        # Overall evaluation result
        passes_evaluation = (
            meets_max_drawdown and
            meets_profit_target and
            meets_daily_loss and
            meets_min_trades and
            meets_min_profitable_days
        )
        
        # Calculate fitness score (0-100)
        # Base score from core requirements
        base_score = 0
        if meets_max_drawdown:
            base_score += 25
        if meets_profit_target:
            base_score += 25
        if meets_daily_loss:
            base_score += 15
        if meets_min_profitable_days:
            base_score += 10
        if meets_min_trades:
            base_score += 5
        
        # Additional bonus score from performance metrics
        # Scale Sharpe ratio from 0-15 points
        sharpe_score = min(15, max(0, sharpe_ratio * 5))
        
        # Scale Sortino ratio from 0-10 points
        sortino_score = min(10, max(0, sortino_ratio * 3))
        
        # Scale profit factor (reward/risk) from 0-10 points
        profit_factor_score = min(10, max(0, (profit_factor - 1) * 10))
        
        # Scale win rate from 0-10 points
        win_rate_score = min(10, max(0, win_rate * 10))
        
        # Scale stability from 0-15 points (normalized from stability_score 0-100)
        stability_points = stability_score * 0.15
        
        # Combine all scores
        fitness_score = (
            base_score + 
            sharpe_score + 
            sortino_score + 
            profit_factor_score + 
            win_rate_score +
            stability_points
        )
        
        # Compiled evaluation results
        evaluation_results = {
            'passes_evaluation': passes_evaluation,
            'score': fitness_score,
            'threshold_results': {
                'meets_max_drawdown': meets_max_drawdown,
                'meets_profit_target': meets_profit_target,
                'meets_daily_loss': meets_daily_loss,
                'meets_min_trades': meets_min_trades,
                'meets_min_profitable_days': meets_min_profitable_days,
                'meets_min_sharpe': meets_min_sharpe,
                'meets_min_reward_risk': meets_min_reward_risk
            },
            'enhanced_metrics': {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'stability_score': stability_score,
                'profitable_days_pct': profitable_days_pct
            },
            'raw_metrics': metrics
        }
        
        return evaluation_results
    
    def evaluate_strategy(self, strategy, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate a strategy on market data against proprietary firm criteria.
        
        Args:
            strategy: Strategy object with backtest method
            market_data: Market data as pandas DataFrame
            
        Returns:
            Evaluation results
        """
        try:
            # Run backtest if strategy has backtest method
            if hasattr(strategy, 'backtest'):
                backtest_results = strategy.backtest(market_data)
            else:
                # Try importing backtest function from advanced_strategies
                from advanced_strategies import backtest_strategy
                backtest_results = backtest_strategy(strategy, market_data)
                
            # Evaluate backtest results
            return self.evaluate_backtest_results(backtest_results)
            
        except Exception as e:
            logger.error(f"Error evaluating strategy: {e}")
            return {
                'passes_evaluation': False,
                'score': 0,
                'error': str(e)
            }


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate strategy against prop firm criteria")
    
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
        help="Output file path for evaluation results (JSON)"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = PropFitnessEvaluator(args.risk_profile)
    
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
