#!/usr/bin/env python3
"""
Funded Account Evaluator - Specialized evaluator for funded trading account requirements

This module evaluates strategies against strict funded account criteria:
- Maximum 5% drawdown
- 8-10% profit target
- Maximum 3% daily loss
- Consistency requirements
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
import datetime
import logging
from collections import defaultdict
import calendar

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import our modules
from strategy_registry import StrategyRegistry
from synthetic_market_generator import SyntheticMarketGenerator
from advanced_strategies import (
    MeanReversionStrategy, 
    MomentumStrategy, 
    VolumeProfileStrategy, 
    VolatilityBreakoutStrategy,
    backtest_strategy
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/funded_evaluator_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger('funded_account_evaluator')


class FundedAccountEvaluator:
    """
    Specialized evaluator for funded trading account requirements.
    
    Evaluates strategies against common funded account criteria:
    - Profit target (typically 8-10%)
    - Maximum drawdown (5%)
    - Daily loss limit (3%)
    - Consistency requirements
    - Minimum trading days
    """
    
    # Default funded account thresholds (based on user specifications)
    DEFAULT_FUNDED_THRESHOLDS = {
        "profit_target_pct": 8.0,        # 8% profit target (minimum)
        "stretch_profit_target_pct": 10.0, # 10% profit target (stretch goal)
        "max_drawdown_pct": 5.0,         # 5% maximum drawdown
        "max_daily_loss_pct": 3.0,       # 3% maximum daily loss
        "min_trading_days": 10,          # Minimum days with trades (out of typical 21-day period)
        "min_win_rate_pct": 55.0,        # Minimum win rate for consistency
        "max_position_size_pct": 2.0,    # Maximum 2% risk per trade
        "consistency_factor": 0.6        # 60% of days should be profitable
    }
    
    def __init__(
        self, 
        thresholds: Optional[Dict[str, float]] = None,
        registry_path: str = "./strategy_registry",
        account_size: float = 100000.0  # Default $100k account size (typical for funded accounts)
    ):
        """
        Initialize the funded account evaluator.
        
        Args:
            thresholds: Custom funded account thresholds
            registry_path: Path to strategy registry
            account_size: Account size for position sizing
        """
        # Use default or custom thresholds
        self.thresholds = thresholds if thresholds else self.DEFAULT_FUNDED_THRESHOLDS.copy()
        
        # Initialize registry
        self.registry = StrategyRegistry(registry_path)
        
        # Initialize market generator
        self.market_generator = SyntheticMarketGenerator()
        
        # Set account size
        self.account_size = account_size
        
        # Create output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"funded_evaluation_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_evaluation_period(
        self,
        period_days: int = 21,  # Typical evaluation period (1 month of trading)
        market_type: str = "mixed"  # Type of market to generate
    ) -> pd.DataFrame:
        """
        Generate a synthetic market for the evaluation period.
        
        Args:
            period_days: Number of trading days in period
            market_type: Type of market (bull, bear, mixed, volatile)
            
        Returns:
            DataFrame with OHLCV data for the evaluation period
        """
        # Generate base market data
        if market_type == "bull":
            market_data = self.market_generator.generate_bull_market(days=period_days)
        elif market_type == "bear":
            market_data = self.market_generator.generate_bear_market(days=period_days)
        elif market_type == "volatile":
            market_data = self.market_generator.generate_volatile_market(days=period_days)
        elif market_type == "sideways":
            market_data = self.market_generator.generate_sideways_market(days=period_days)
        elif market_type == "mixed":
            # Generate a mixed market with different segments
            segments = []
            
            # Start with a sideways segment
            days_remaining = period_days
            sideways_days = min(7, days_remaining)
            segments.append(self.market_generator.generate_sideways_market(days=sideways_days))
            days_remaining -= sideways_days
            
            # Add a bull or bear segment
            if days_remaining > 0:
                trend_days = min(7, days_remaining)
                if np.random.random() > 0.5:
                    segments.append(self.market_generator.generate_bull_market(days=trend_days))
                else:
                    segments.append(self.market_generator.generate_bear_market(days=trend_days))
                days_remaining -= trend_days
            
            # Add a volatile segment to test risk management
            if days_remaining > 0:
                volatile_days = min(5, days_remaining)
                segments.append(self.market_generator.generate_volatile_market(days=volatile_days))
                days_remaining -= volatile_days
            
            # Fill any remaining days with sideways market
            if days_remaining > 0:
                segments.append(self.market_generator.generate_sideways_market(days=days_remaining))
            
            # Combine segments
            market_data = pd.concat(segments)
            
            # Reset index to create continuous dates
            start_date = datetime.datetime.now() - datetime.timedelta(days=period_days * 2)
            dates = [start_date + datetime.timedelta(days=i) for i in range(len(market_data))]
            market_data.index = dates
        else:
            # Default to mixed
            return self.generate_evaluation_period(period_days, "mixed")
        
        # Save market data
        market_data.to_csv(os.path.join(self.output_dir, f"{market_type}_market_{period_days}days.csv"))
        
        return market_data
    
    def calculate_daily_pnl(
        self,
        equity_curve: List[float],
        dates: List[datetime.datetime]
    ) -> Dict[datetime.date, float]:
        """
        Calculate daily P&L from equity curve.
        
        Args:
            equity_curve: List of equity values
            dates: List of corresponding dates
            
        Returns:
            Dictionary of date to daily P&L percentage
        """
        daily_pnl = {}
        
        for i in range(1, len(equity_curve)):
            previous_equity = equity_curve[i-1]
            current_equity = equity_curve[i]
            
            # Calculate percentage change
            if previous_equity > 0:
                daily_change_pct = (current_equity - previous_equity) / previous_equity * 100
            else:
                daily_change_pct = 0
            
            # Store by date
            date = dates[i].date()
            daily_pnl[date] = daily_change_pct
        
        return daily_pnl
    
    def evaluate_for_funded_account(
        self,
        strategy_results: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluate strategy results against funded account criteria.
        
        Args:
            strategy_results: Backtest results
            market_data: Market data used for backtest
            
        Returns:
            Evaluation results
        """
        # Extract key metrics
        equity_curve = strategy_results.get("equity_curve", [])
        total_return_pct = strategy_results.get("total_return_pct", 0)
        max_drawdown = strategy_results.get("max_drawdown", 0)
        win_rate = strategy_results.get("win_rate", 0)
        trades = strategy_results.get("trades", [])
        
        # Calculate daily P&L
        daily_pnl = self.calculate_daily_pnl(equity_curve, market_data.index)
        
        # Find worst daily loss
        if daily_pnl:
            min_daily_pnl = min(daily_pnl.values())
        else:
            min_daily_pnl = 0
        
        # Count trading days
        trading_days = set()
        for trade in trades:
            if "entry_date" in trade:
                trading_days.add(trade["entry_date"].date())
            if "exit_date" in trade:
                trading_days.add(trade["exit_date"].date())
        
        trading_day_count = len(trading_days)
        
        # Count profitable days
        profitable_days = sum(1 for pnl in daily_pnl.values() if pnl > 0)
        total_days = len(daily_pnl)
        
        if total_days > 0:
            consistency = profitable_days / total_days
        else:
            consistency = 0
        
        # Evaluate against funded account criteria
        meets_profit_target = total_return_pct >= self.thresholds["profit_target_pct"]
        meets_max_drawdown = max_drawdown <= self.thresholds["max_drawdown_pct"]
        meets_daily_loss = abs(min_daily_pnl) <= self.thresholds["max_daily_loss_pct"]
        meets_min_trading_days = trading_day_count >= self.thresholds["min_trading_days"]
        meets_win_rate = win_rate >= self.thresholds["min_win_rate_pct"]
        meets_consistency = consistency >= self.thresholds["consistency_factor"]
        
        # Check if strategy passes all criteria
        passes_evaluation = all([
            meets_profit_target,
            meets_max_drawdown,
            meets_daily_loss,
            meets_min_trading_days
            # Win rate and consistency are desirable but not strict requirements
        ])
        
        # Additional score factors
        exceeds_profit_target = total_return_pct >= self.thresholds["stretch_profit_target_pct"]
        
        # Calculate funded account score (0-100)
        # This weights the most important factors for funded accounts
        score_components = {
            "profit": min(40, total_return_pct * 4),  # Up to 40 points for profit
            "drawdown": 30 * (1 - max_drawdown / self.thresholds["max_drawdown_pct"]),  # Up to 30 points for low drawdown
            "daily_loss": 15 * (1 - abs(min_daily_pnl) / self.thresholds["max_daily_loss_pct"]),  # Up to 15 points for limiting daily losses
            "consistency": 15 * consistency,  # Up to 15 points for consistency
        }
        
        score = sum(score_components.values())
        
        # Prepare evaluation results
        evaluation = {
            "passes_evaluation": passes_evaluation,
            "score": min(100, score),  # Cap at 100
            "score_components": score_components,
            "metrics": {
                "total_return_pct": total_return_pct,
                "max_drawdown": max_drawdown,
                "worst_daily_loss": min_daily_pnl,
                "win_rate": win_rate,
                "trading_days": trading_day_count,
                "profitable_days": profitable_days,
                "total_days": total_days,
                "consistency": consistency
            },
            "threshold_results": {
                "meets_profit_target": meets_profit_target,
                "exceeds_profit_target": exceeds_profit_target,
                "meets_max_drawdown": meets_max_drawdown,
                "meets_daily_loss": meets_daily_loss,
                "meets_min_trading_days": meets_min_trading_days,
                "meets_win_rate": meets_win_rate,
                "meets_consistency": meets_consistency
            },
            "daily_pnl": daily_pnl
        }
        
        return evaluation
    
    def generate_evaluation_report(self, evaluation: Dict[str, Any]) -> str:
        """
        Generate a human-readable evaluation report.
        
        Args:
            evaluation: Evaluation results
            
        Returns:
            Report text
        """
        report = []
        report.append("================================")
        report.append("FUNDED ACCOUNT EVALUATION REPORT")
        report.append("================================")
        report.append("")
        
        # Overall result
        if evaluation["passes_evaluation"]:
            report.append("OVERALL RESULT: PASS ✅")
        else:
            report.append("OVERALL RESULT: FAIL ❌")
        
        report.append(f"Score: {evaluation['score']:.1f}/100")
        report.append("")
        
        # Key metrics
        metrics = evaluation["metrics"]
        report.append("KEY METRICS:")
        report.append(f"- Total Return: {metrics['total_return_pct']:.2f}% (Target: {self.thresholds['profit_target_pct']:.1f}%)")
        report.append(f"- Max Drawdown: {metrics['max_drawdown']:.2f}% (Limit: {self.thresholds['max_drawdown_pct']:.1f}%)")
        report.append(f"- Worst Daily Loss: {metrics['worst_daily_loss']:.2f}% (Limit: {self.thresholds['max_daily_loss_pct']:.1f}%)")
        report.append(f"- Win Rate: {metrics['win_rate']:.1f}%")
        report.append(f"- Trading Days: {metrics['trading_days']} (Minimum: {self.thresholds['min_trading_days']})")
        report.append(f"- Consistency: {metrics['consistency']:.2f} ({metrics['profitable_days']} profitable days out of {metrics['total_days']})")
        report.append("")
        
        # Detailed results
        threshold_results = evaluation["threshold_results"]
        report.append("CRITERIA RESULTS:")
        
        for criterion, result in threshold_results.items():
            icon = "✅" if result else "❌"
            report.append(f"- {criterion.replace('_', ' ').title()}: {icon}")
        
        report.append("")
        
        # Score breakdown
        score_components = evaluation["score_components"]
        report.append("SCORE BREAKDOWN:")
        for component, value in score_components.items():
            report.append(f"- {component.title()}: {value:.1f}")
        
        report.append("")
        report.append("================================")
        
        return "\n".join(report)
    
    def evaluate_strategy(
        self,
        strategy,
        market_data: Optional[pd.DataFrame] = None,
        market_type: str = "mixed",
        period_days: int = 21
    ) -> Dict[str, Any]:
        """
        Evaluate a strategy against funded account criteria.
        
        Args:
            strategy: Strategy object to evaluate
            market_data: Market data to use (if None, will generate based on market_type)
            market_type: Type of market to generate (if market_data is None)
            period_days: Number of trading days in evaluation period
            
        Returns:
            Evaluation results
        """
        # Generate market data if not provided
        if market_data is None:
            market_data = self.generate_evaluation_period(period_days, market_type)
        
        # Backtest strategy
        backtest_results = backtest_strategy(strategy, market_data)
        
        # Evaluate against funded account criteria
        evaluation = self.evaluate_for_funded_account(backtest_results, market_data)
        
        # Generate report
        report = self.generate_evaluation_report(evaluation)
        
        # Save report
        strategy_name = getattr(strategy, "strategy_name", "Unknown")
        report_path = os.path.join(self.output_dir, f"{strategy_name}_report.txt")
        
        with open(report_path, "w") as f:
            f.write(report)
        
        # Print report
        print(report)
        
        # Save the evaluation
        evaluation_path = os.path.join(self.output_dir, f"{strategy_name}_evaluation.json")
        
        with open(evaluation_path, "w") as f:
            # Convert daily_pnl keys (dates) to strings for JSON serialization
            if "daily_pnl" in evaluation:
                evaluation["daily_pnl"] = {
                    str(date): value 
                    for date, value in evaluation["daily_pnl"].items()
                }
            
            json.dump(evaluation, f, indent=2)
        
        # Save equity curve
        self._plot_equity_curve(
            equity_curve=backtest_results["equity_curve"],
            dates=market_data.index,
            daily_pnl=evaluation["daily_pnl"],
            strategy_name=strategy_name
        )
        
        return evaluation
    
    def _plot_equity_curve(
        self,
        equity_curve: List[float],
        dates: List[datetime.datetime],
        daily_pnl: Dict[str, float],
        strategy_name: str
    ):
        """
        Plot equity curve and daily P&L.
        
        Args:
            equity_curve: List of equity values
            dates: List of dates
            daily_pnl: Dictionary of daily P&L values
            strategy_name: Name of strategy
        """
        # Convert date strings back to dates if needed
        if isinstance(next(iter(daily_pnl.keys()), None), str):
            daily_pnl = {
                datetime.datetime.strptime(date, "%Y-%m-%d").date(): value
                for date, value in daily_pnl.items()
            }
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot equity curve
        ax1.plot(dates, equity_curve)
        ax1.set_title(f"Equity Curve - {strategy_name}")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Equity")
        ax1.grid(True)
        
        # Add horizontal line for initial equity
        ax1.axhline(y=equity_curve[0], color='r', linestyle='--', alpha=0.3)
        
        # Add horizontal lines for profit targets
        profit_target = equity_curve[0] * (1 + self.thresholds["profit_target_pct"] / 100)
        stretch_target = equity_curve[0] * (1 + self.thresholds["stretch_profit_target_pct"] / 100)
        
        ax1.axhline(y=profit_target, color='g', linestyle='--', alpha=0.5, label=f"{self.thresholds['profit_target_pct']}% Target")
        ax1.axhline(y=stretch_target, color='g', linestyle='-', alpha=0.5, label=f"{self.thresholds['stretch_profit_target_pct']}% Target")
        
        # Add legend
        ax1.legend()
        
        # Plot daily P&L as bar chart
        daily_dates = sorted(daily_pnl.keys())
        daily_values = [daily_pnl[date] for date in daily_dates]
        
        # Convert dates to matplotlib format if needed
        if isinstance(daily_dates[0], datetime.date):
            daily_dates = [datetime.datetime.combine(date, datetime.time.min) for date in daily_dates]
        
        # Plot bars with green for positive, red for negative
        colors = ['green' if val >= 0 else 'red' for val in daily_values]
        ax2.bar(daily_dates, daily_values, color=colors, alpha=0.6)
        
        # Add horizontal line at zero
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add horizontal line for daily loss limit
        ax2.axhline(y=-self.thresholds["max_daily_loss_pct"], color='red', linestyle='--', alpha=0.5, label="Daily Loss Limit")
        
        ax2.set_title("Daily P&L")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Daily P&L (%)")
        ax2.grid(True)
        ax2.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, f"{strategy_name}_equity_curve.png"))
        plt.close()
    
    def evaluate_all_strategies(
        self,
        strategies: List[Dict[str, Any]],
        market_type: str = "mixed",
        period_days: int = 21
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple strategies against funded account criteria.
        
        Args:
            strategies: List of strategy configurations
            market_type: Type of market to use
            period_days: Number of trading days
            
        Returns:
            List of evaluation results
        """
        # Generate market data (use same data for all strategies for fair comparison)
        market_data = self.generate_evaluation_period(period_days, market_type)
        
        # Evaluate each strategy
        evaluations = []
        
        for strategy_config in strategies:
            strategy_type = strategy_config["strategy_type"]
            parameters = strategy_config["parameters"]
            
            # Create strategy
            if strategy_type == "MeanReversion":
                strategy = MeanReversionStrategy(**parameters)
            elif strategy_type == "Momentum":
                strategy = MomentumStrategy(**parameters)
            elif strategy_type == "VolumeProfile":
                strategy = VolumeProfileStrategy(**parameters)
            elif strategy_type == "VolatilityBreakout":
                strategy = VolatilityBreakoutStrategy(**parameters)
            else:
                logger.warning(f"Unknown strategy type: {strategy_type}")
                continue
            
            # Evaluate strategy
            evaluation = self.evaluate_strategy(strategy, market_data)
            
            # Add strategy information
            evaluation["strategy_type"] = strategy_type
            evaluation["parameters"] = parameters
            
            evaluations.append(evaluation)
        
        # Rank by score
        evaluations.sort(key=lambda e: e["score"], reverse=True)
        
        # Generate summary report
        self._generate_summary_report(evaluations, market_type)
        
        return evaluations
    
    def _generate_summary_report(
        self,
        evaluations: List[Dict[str, Any]],
        market_type: str
    ):
        """
        Generate summary report for multiple strategy evaluations.
        
        Args:
            evaluations: List of evaluation results
            market_type: Type of market used
        """
        if not evaluations:
            logger.warning("No evaluations to summarize")
            return
        
        # Count passing strategies
        passing = sum(1 for e in evaluations if e["passes_evaluation"])
        
        # Create summary report
        report = []
        report.append("==================================")
        report.append("FUNDED ACCOUNT EVALUATION SUMMARY")
        report.append("==================================")
        report.append("")
        report.append(f"Market Type: {market_type}")
        report.append(f"Total Strategies: {len(evaluations)}")
        report.append(f"Passing Strategies: {passing} ({passing/len(evaluations)*100:.1f}%)")
        report.append("")
        
        report.append("TOP 5 STRATEGIES:")
        for i, evaluation in enumerate(evaluations[:5]):
            strategy_type = evaluation["strategy_type"]
            score = evaluation["score"]
            total_return = evaluation["metrics"]["total_return_pct"]
            max_drawdown = evaluation["metrics"]["max_drawdown"]
            
            # Create a short parameter summary
            params = evaluation["parameters"]
            param_summary = ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" 
                                     for k, v in params.items()
                                     if k in list(params.keys())[:3])  # Show first 3 parameters
            
            status = "PASS ✅" if evaluation["passes_evaluation"] else "FAIL ❌"
            
            report.append(f"{i+1}. {strategy_type} ({param_summary}...)")
            report.append(f"   Score: {score:.1f}, Return: {total_return:.2f}%, Drawdown: {max_drawdown:.2f}%")
            report.append(f"   Status: {status}")
            report.append("")
        
        report.append("FAILURE ANALYSIS:")
        
        # Count reasons for failure
        failure_reasons = defaultdict(int)
        
        for evaluation in evaluations:
            if not evaluation["passes_evaluation"]:
                for criterion, result in evaluation["threshold_results"].items():
                    if not result:
                        failure_reasons[criterion] += 1
        
        # Sort by frequency
        for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
            percentage = count / (len(evaluations) - passing) * 100 if (len(evaluations) - passing) > 0 else 0
            report.append(f"- {reason.replace('_', ' ').title()}: {count} strategies ({percentage:.1f}%)")
        
        report.append("")
        report.append("==================================")
        
        # Save summary report
        summary_path = os.path.join(self.output_dir, f"summary_{market_type}.txt")
        
        with open(summary_path, "w") as f:
            f.write("\n".join(report))
        
        # Print summary
        print("\n".join(report))


def test_on_sample_strategies():
    """
    Test the funded account evaluator on sample strategies.
    """
    # Create evaluator
    evaluator = FundedAccountEvaluator()
    
    # Create sample strategies
    strategies = [
        {
            "strategy_type": "MeanReversion",
            "parameters": {
                "lookback_period": 20,
                "entry_std": 2.0,
                "exit_std": 0.5,
                "smoothing": 3
            }
        },
        {
            "strategy_type": "Momentum",
            "parameters": {
                "short_period": 14,
                "medium_period": 30,
                "long_period": 90,
                "threshold": 0.02,
                "smoothing": 3
            }
        },
        {
            "strategy_type": "VolumeProfile",
            "parameters": {
                "lookback_period": 20,
                "volume_threshold": 1.5,
                "price_levels": 20,
                "smoothing": 3
            }
        },
        {
            "strategy_type": "VolatilityBreakout",
            "parameters": {
                "atr_period": 14,
                "breakout_multiple": 1.5,
                "lookback_period": 5,
                "filter_threshold": 0.2
            }
        }
    ]
    
    # Evaluate strategies
    evaluations = evaluator.evaluate_all_strategies(strategies, "mixed", 21)
    
    return evaluations


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate strategies against funded account criteria")
    
    parser.add_argument(
        "--registry", 
        type=str, 
        default="./strategy_registry",
        help="Path to strategy registry"
    )
    
    parser.add_argument(
        "--market-type", 
        type=str, 
        default="mixed",
        choices=["bull", "bear", "sideways", "volatile", "mixed"],
        help="Type of market to evaluate against"
    )
    
    parser.add_argument(
        "--period-days", 
        type=int, 
        default=21,
        help="Number of trading days in evaluation period"
    )
    
    parser.add_argument(
        "--min-generation", 
        type=int, 
        default=5,
        help="Minimum generation to consider from registry"
    )
    
    parser.add_argument(
        "--test-samples", 
        action="store_true",
        help="Test on sample strategies"
    )
    
    args = parser.parse_args()
    
    if args.test_samples:
        test_on_sample_strategies()
    else:
        # Create evaluator
        evaluator = FundedAccountEvaluator(registry_path=args.registry)
        
        # Get strategies from registry
        registry = StrategyRegistry(args.registry)
        all_strategies = registry.get_all_strategies()
        
        # Filter by generation
        strategies = [
            {
                "strategy_type": s["strategy_type"],
                "parameters": s["parameters"]
            }
            for s in all_strategies
            if s.get("generation", 0) >= args.min_generation
        ]
        
        if not strategies:
            print(f"No strategies found with generation >= {args.min_generation}")
            sys.exit(1)
        
        print(f"Evaluating {len(strategies)} strategies against funded account criteria")
        
        # Evaluate strategies
        evaluator.evaluate_all_strategies(
            strategies=strategies,
            market_type=args.market_type,
            period_days=args.period_days
        )
