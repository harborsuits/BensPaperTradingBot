"""
Market Condition Testing Framework

Tests trading strategies under various simulated market conditions to evaluate robustness.
"""

import os
import json
import time
import logging
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketConditionSimulator:
    """Generates synthetic market data with specific conditions."""
    
    # Market condition presets
    MARKET_CONDITIONS = {
        "bull_trend": {
            "description": "Strong upward trend with small pullbacks",
            "base_trend": 0.001,  # 0.1% daily up trend
            "volatility": 0.015,  # 1.5% daily volatility
            "mean_reversion": 0.05,  # 5% mean reversion strength
            "cycle_amplitude": 0.02,  # 2% cycle amplitude
            "cycle_period": 20,  # 20-day cycles
            "gap_frequency": 0.05,  # 5% chance of gaps
            "gap_size": 0.02,  # 2% average gap size
        },
        "bear_trend": {
            "description": "Strong downward trend with relief rallies",
            "base_trend": -0.001,  # 0.1% daily down trend
            "volatility": 0.025,  # 2.5% daily volatility (bear markets more volatile)
            "mean_reversion": 0.1,  # 10% mean reversion strength
            "cycle_amplitude": 0.03,  # 3% cycle amplitude
            "cycle_period": 15,  # 15-day cycles (faster in bear markets)
            "gap_frequency": 0.1,  # 10% chance of gaps
            "gap_size": 0.03,  # 3% average gap size
        },
        "sideways_choppy": {
            "description": "Sideways market with high volatility and no clear trend",
            "base_trend": 0.0,  # No trend
            "volatility": 0.02,  # 2% daily volatility
            "mean_reversion": 0.2,  # 20% mean reversion strength
            "cycle_amplitude": 0.04,  # 4% cycle amplitude
            "cycle_period": 10,  # 10-day cycles (shorter cycles)
            "gap_frequency": 0.07,  # 7% chance of gaps
            "gap_size": 0.015,  # 1.5% average gap size
        },
        "low_volatility": {
            "description": "Slow market with minimal price movement",
            "base_trend": 0.0002,  # Very slight upward bias
            "volatility": 0.008,  # 0.8% daily volatility
            "mean_reversion": 0.3,  # 30% mean reversion strength
            "cycle_amplitude": 0.01,  # 1% cycle amplitude
            "cycle_period": 30,  # 30-day cycles (longer cycles)
            "gap_frequency": 0.02,  # 2% chance of gaps
            "gap_size": 0.01,  # 1% average gap size
        },
        "high_volatility": {
            "description": "Extremely volatile market with large price swings",
            "base_trend": 0.0,  # No trend
            "volatility": 0.035,  # 3.5% daily volatility
            "mean_reversion": 0.1,  # 10% mean reversion strength
            "cycle_amplitude": 0.06,  # 6% cycle amplitude
            "cycle_period": 12,  # 12-day cycles
            "gap_frequency": 0.15,  # 15% chance of gaps
            "gap_size": 0.035,  # 3.5% average gap size
        },
        "trend_reversal": {
            "description": "Market transitioning from downtrend to uptrend",
            "base_trend": -0.001,  # Start with downtrend
            "trend_change_point": 0.5,  # Halfway through, trend changes
            "new_trend": 0.001,  # End with uptrend
            "volatility": 0.025,  # 2.5% daily volatility
            "mean_reversion": 0.15,  # 15% mean reversion strength
            "cycle_amplitude": 0.035,  # 3.5% cycle amplitude
            "cycle_period": 18,  # 18-day cycles
            "gap_frequency": 0.08,  # 8% chance of gaps
            "gap_size": 0.025,  # 2.5% average gap size
        }
    }
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the simulator with optional random seed."""
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def generate_market_data(self, 
                            condition: str,
                            days: int = 120, 
                            base_price: float = 100.0,
                            symbol: str = "SYNTHETIC") -> pd.DataFrame:
        """
        Generate synthetic market data for a specific condition.
        
        Args:
            condition: Market condition key (must be in MARKET_CONDITIONS)
            days: Number of days of data to generate
            base_price: Starting price
            symbol: Symbol name for the synthetic data
            
        Returns:
            DataFrame with OHLCV data
        """
        if condition not in self.MARKET_CONDITIONS:
            raise ValueError(f"Unknown market condition: {condition}. Must be one of {list(self.MARKET_CONDITIONS.keys())}")
        
        params = self.MARKET_CONDITIONS[condition]
        logger.info(f"Generating {days} days of {condition} market data: {params['description']}")
        
        # Create date range
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Initialize price series with base price
        prices = np.zeros(days + 1)
        prices[0] = base_price
        
        # Apply trend and other factors
        for i in range(1, days + 1):
            # Determine trend based on day (handles trend reversal)
            if 'trend_change_point' in params and i >= days * params['trend_change_point']:
                trend = params['new_trend']
            else:
                trend = params['base_trend']
            
            # Calculate components
            trend_component = trend * prices[i-1]
            
            # Random walk (volatility)
            random_component = np.random.normal(0, params['volatility']) * prices[i-1]
            
            # Mean reversion
            if i > 1:
                mean_price = np.mean(prices[:i])
                reversion = params['mean_reversion'] * (mean_price - prices[i-1])
            else:
                reversion = 0
                
            # Cyclical component
            cycle = params['cycle_amplitude'] * prices[i-1] * np.sin(2 * np.pi * i / params['cycle_period'])
            
            # Gaps (overnight price jumps)
            gap = 0
            if random.random() < params['gap_frequency']:
                gap_direction = 1 if random.random() > 0.5 else -1
                gap = gap_direction * random.uniform(0, params['gap_size']) * prices[i-1]
            
            # Combine all components
            prices[i] = prices[i-1] + trend_component + random_component + reversion + cycle + gap
            
            # Ensure price doesn't go negative or too close to zero
            prices[i] = max(prices[i], prices[0] * 0.1)
        
        # Create OHLCV data
        data = []
        for i in range(len(date_range)):
            # Generate daily candle from close price
            close = prices[i+1]
            
            # Generate intraday volatility
            daily_vol = params['volatility'] * close
            
            # Generate open, high, low
            if i > 0 and random.random() < params['gap_frequency']:
                # Simulate gap
                gap_direction = 1 if random.random() > 0.5 else -1
                gap_size = random.uniform(0, params['gap_size']) * close
                open_price = prices[i] + (gap_direction * gap_size)
            else:
                open_price = prices[i] * (1 + np.random.normal(0, daily_vol * 0.1))
            
            # Ensure proper high/low
            intraday_range = random.uniform(1.0, 2.0) * daily_vol
            if close > open_price:
                high = close + random.uniform(0, intraday_range)
                low = open_price - random.uniform(0, intraday_range)
            else:
                high = open_price + random.uniform(0, intraday_range)
                low = close - random.uniform(0, intraday_range)
            
            # Ensure low is the minimum
            low = min(low, open_price, close)
            
            # Ensure high is the maximum
            high = max(high, open_price, close)
            
            # Generate volume (higher in volatile periods)
            price_change_pct = abs(close - open_price) / open_price
            base_volume = random.uniform(10000, 50000)
            volume = base_volume * (1 + 5 * price_change_pct)
            
            data.append({
                'date': date_range[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'symbol': symbol
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        logger.info(f"Generated market data with final price: {df['close'].iloc[-1]:.2f}")
        
        return df

def test_strategy_in_conditions(strategy, conditions=None, days=120, runs_per_condition=3):
    """
    Test a strategy across multiple market conditions.
    
    Args:
        strategy: Strategy object to test
        conditions: List of market conditions to test (default: all conditions)
        days: Number of days of data for each test
        runs_per_condition: Number of runs per condition (with different seeds)
        
    Returns:
        Dict with test results
    """
    simulator = MarketConditionSimulator()
    
    # Use all conditions if none specified
    if conditions is None:
        conditions = list(MarketConditionSimulator.MARKET_CONDITIONS.keys())
    
    logger.info(f"Testing strategy across {len(conditions)} market conditions, {runs_per_condition} runs each")
    
    results = {
        "strategy_id": getattr(strategy, "strategy_id", "unknown"),
        "strategy_type": strategy.__class__.__name__,
        "parameters": getattr(strategy, "get_parameters", lambda: {})(),
        "conditions": {}
    }
    
    for condition in conditions:
        condition_results = []
        
        for run in range(runs_per_condition):
            # Create a new simulator with different seed for each run
            seed = hash(f"{condition}_{run}") % 10000
            run_simulator = MarketConditionSimulator(seed=seed)
            
            # Generate market data
            market_data = run_simulator.generate_market_data(
                condition=condition,
                days=days,
                base_price=100.0,
                symbol=f"SYN_{condition.upper()}"
            )
            
            # Run backtest
            signals = []
            trades = []
            position = 0
            entry_price = 0
            capital = 10000.0
            equity = [capital]
            
            # Process each day
            for i in range(len(market_data) - 1):  # Skip last day to avoid lookahead
                day_data = market_data.iloc[:i+1]
                
                # Get signal
                if hasattr(strategy, "calculate_signal"):
                    signal = strategy.calculate_signal(day_data)
                elif hasattr(strategy, "generate_signals"):
                    signal = strategy.generate_signals(day_data)
                else:
                    signal = {"signal": "none", "confidence": 0}
                
                signals.append(signal)
                
                # Process signal
                next_day = market_data.iloc[i+1]
                if signal["signal"] == "buy" and position <= 0:
                    # Enter long position
                    position = 1
                    entry_price = next_day["open"]
                    trades.append({
                        "type": "buy",
                        "entry_date": next_day.name,
                        "entry_price": entry_price,
                        "size": position
                    })
                elif signal["signal"] == "sell" and position >= 0:
                    # Enter short position
                    position = -1
                    entry_price = next_day["open"]
                    trades.append({
                        "type": "sell",
                        "entry_date": next_day.name,
                        "entry_price": entry_price,
                        "size": position
                    })
                
                # Update equity
                if position != 0:
                    # Calculate P&L
                    price_change = (next_day["close"] - entry_price) / entry_price
                    if position < 0:
                        price_change = -price_change
                    
                    # Update equity
                    new_equity = equity[-1] * (1 + price_change)
                    equity.append(new_equity)
                else:
                    equity.append(equity[-1])
            
            # Calculate performance metrics
            final_equity = equity[-1]
            returns = np.diff(equity) / equity[:-1]
            
            # Avoid division by zero for all metrics
            if len(returns) < 2 or np.std(returns) == 0:
                sharpe = 0
            else:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            
            max_drawdown = 0
            peak = equity[0]
            for e in equity:
                if e > peak:
                    peak = e
                drawdown = (peak - e) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # Compile run results
            run_result = {
                "final_equity": final_equity,
                "total_return": (final_equity - capital) / capital,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_drawdown,
                "trades": len(trades),
                "seed": seed
            }
            
            condition_results.append(run_result)
        
        # Calculate average metrics across runs
        if condition_results:
            avg_return = sum(r["total_return"] for r in condition_results) / len(condition_results)
            avg_sharpe = sum(r["sharpe_ratio"] for r in condition_results) / len(condition_results)
            avg_drawdown = sum(r["max_drawdown"] for r in condition_results) / len(condition_results)
            avg_trades = sum(r["trades"] for r in condition_results) / len(condition_results)
            
            results["conditions"][condition] = {
                "description": MarketConditionSimulator.MARKET_CONDITIONS[condition]["description"],
                "avg_return": avg_return,
                "avg_sharpe": avg_sharpe,
                "avg_drawdown": avg_drawdown,
                "avg_trades": avg_trades,
                "runs": condition_results
            }
    
    # Calculate overall robustness score
    condition_scores = []
    for condition, metrics in results["conditions"].items():
        # A simple robustness score as combination of return and drawdown
        # Higher return is better, lower drawdown is better
        condition_score = metrics["avg_return"] - metrics["avg_drawdown"] * 2
        condition_scores.append(condition_score)
    
    if condition_scores:
        results["robustness_score"] = sum(condition_scores) / len(condition_scores)
        results["worst_condition"] = min(
            results["conditions"].keys(), 
            key=lambda c: results["conditions"][c]["avg_return"]
        )
        results["best_condition"] = max(
            results["conditions"].keys(), 
            key=lambda c: results["conditions"][c]["avg_return"]
        )
    
    return results

def generate_condition_report(results, output_dir=None):
    """
    Generate a markdown report of market condition test results.
    
    Args:
        results: Results dictionary from test_strategy_in_conditions
        output_dir: Directory to save report (default: current directory)
        
    Returns:
        Path to report file
    """
    if output_dir is None:
        output_dir = "."
    
    os.makedirs(output_dir, exist_ok=True)
    
    report_file = os.path.join(output_dir, "market_condition_report.md")
    
    with open(report_file, "w") as f:
        f.write("# Market Condition Robustness Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Strategy Information\n\n")
        f.write(f"- **Strategy Type:** {results['strategy_type']}\n")
        f.write(f"- **Strategy ID:** {results['strategy_id']}\n")
        
        # Include parameters if available
        if results.get("parameters"):
            f.write("- **Parameters:**\n")
            for key, value in results["parameters"].items():
                f.write(f"  - {key}: {value}\n")
        
        f.write("\n## Overall Results\n\n")
        
        if "robustness_score" in results:
            f.write(f"- **Robustness Score:** {results['robustness_score']:.4f}\n")
            f.write(f"- **Best Condition:** {results['best_condition']} ({results['conditions'][results['best_condition']]['description']})\n")
            f.write(f"- **Worst Condition:** {results['worst_condition']} ({results['conditions'][results['worst_condition']]['description']})\n\n")
        
        f.write("## Performance by Market Condition\n\n")
        f.write("| Condition | Description | Return | Sharpe | Max Drawdown | Trades |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        
        for condition, metrics in results["conditions"].items():
            f.write(f"| {condition} | {metrics['description']} | ")
            f.write(f"{metrics['avg_return']*100:.2f}% | ")
            f.write(f"{metrics['avg_sharpe']:.2f} | ")
            f.write(f"{metrics['avg_drawdown']*100:.2f}% | ")
            f.write(f"{metrics['avg_trades']:.1f} |\n")
        
        f.write("\n## Detailed Condition Results\n\n")
        
        for condition, metrics in results["conditions"].items():
            f.write(f"### {condition}: {metrics['description']}\n\n")
            
            f.write("| Run | Return | Sharpe | Max Drawdown | Trades |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            
            for i, run in enumerate(metrics["runs"]):
                f.write(f"| {i+1} | ")
                f.write(f"{run['total_return']*100:.2f}% | ")
                f.write(f"{run['sharpe_ratio']:.2f} | ")
                f.write(f"{run['max_drawdown']*100:.2f}% | ")
                f.write(f"{run['trades']} |\n")
            
            f.write("\n")
        
        f.write("## Analysis and Recommendations\n\n")
        
        # Generate simple recommendations based on results
        f.write("### Strategy Strengths\n\n")
        
        # Find the best performing conditions
        sorted_conditions = sorted(
            results["conditions"].keys(),
            key=lambda c: results["conditions"][c]["avg_return"],
            reverse=True
        )
        
        if sorted_conditions:
            best_conditions = sorted_conditions[:min(3, len(sorted_conditions))]
            f.write("This strategy performs well in the following conditions:\n\n")
            for condition in best_conditions:
                metrics = results["conditions"][condition]
                f.write(f"- **{condition}** ({metrics['description']}): {metrics['avg_return']*100:.2f}% return\n")
            
            f.write("\n### Strategy Weaknesses\n\n")
            
            # Find the worst performing conditions
            worst_conditions = sorted_conditions[-min(3, len(sorted_conditions)):]
            worst_conditions.reverse()
            f.write("This strategy struggles in the following conditions:\n\n")
            for condition in worst_conditions:
                metrics = results["conditions"][condition]
                f.write(f"- **{condition}** ({metrics['description']}): {metrics['avg_return']*100:.2f}% return\n")
        
        f.write("\n### Robustness Assessment\n\n")
        
        # Generate robustness assessment
        if "robustness_score" in results:
            robustness = results["robustness_score"]
            if robustness > 0.2:
                f.write("The strategy demonstrates **excellent robustness** across varied market conditions.")
                f.write(" It performs consistently well in most environments and can be deployed with high confidence.\n")
            elif robustness > 0.1:
                f.write("The strategy shows **good robustness** across different market conditions.")
                f.write(" It performs reasonably well in most environments but may need monitoring in specific conditions.\n")
            elif robustness > 0:
                f.write("The strategy exhibits **moderate robustness** across market conditions.")
                f.write(" It performs adequately in some environments but struggles in others.\n")
            else:
                f.write("The strategy demonstrates **poor robustness** across market conditions.")
                f.write(" It is highly sensitive to market environments and should be used with caution.\n")
        
        f.write("\n### Recommendation\n\n")
        
        # Generate recommendation
        if "robustness_score" in results:
            if robustness > 0.1:
                f.write("- Deploy the strategy in live trading with normal position sizing\n")
            elif robustness > 0:
                f.write("- Deploy the strategy with reduced position sizing\n")
                f.write("- Consider implementing additional filters for challenging market conditions\n")
            else:
                f.write("- Use the strategy only in favorable market conditions\n")
                f.write("- Further optimize the strategy parameters before deployment\n")
                f.write("- Consider combining with complementary strategies\n")
    
    logger.info(f"Market condition report saved to: {report_file}")
    return report_file

def main():
    """Run the market condition testing with command line arguments."""
    import argparse
    import importlib
    import sys
    
    parser = argparse.ArgumentParser(description="Test a strategy across different market conditions")
    parser.add_argument("--strategy", type=str, required=True, help="Path to strategy module and class (module.ClassName)")
    parser.add_argument("--conditions", type=str, nargs="+", help="Market conditions to test (default: all)")
    parser.add_argument("--days", type=int, default=120, help="Days of data for each test")
    parser.add_argument("--runs", type=int, default=3, help="Runs per condition")
    parser.add_argument("--output-dir", type=str, help="Output directory (default: timestamped directory)")
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        timestamp = int(time.time())
        args.output_dir = f"test_results/market_conditions_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load strategy
    try:
        module_path, class_name = args.strategy.rsplit(".", 1)
        
        # Add current directory to path if needed
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        
        module = importlib.import_module(module_path)
        strategy_class = getattr(module, class_name)
        strategy = strategy_class()
        
        logger.info(f"Loaded strategy: {strategy_class.__name__}")
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load strategy: {e}")
        sys.exit(1)
    
    # Run tests
    results = test_strategy_in_conditions(
        strategy=strategy,
        conditions=args.conditions,
        days=args.days,
        runs_per_condition=args.runs
    )
    
    # Save results
    results_file = os.path.join(args.output_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate report
    report_file = generate_condition_report(results, args.output_dir)
    
    print(f"\nMarket condition testing completed!")
    print(f"Results saved to: {results_file}")
    print(f"Report generated: {report_file}")
    
    # Print summary
    if "robustness_score" in results:
        print(f"\nRobustness Score: {results['robustness_score']:.4f}")
        print(f"Best Condition: {results['best_condition']}")
        print(f"Worst Condition: {results['worst_condition']}")

if __name__ == "__main__":
    main()
