#!/usr/bin/env python3
"""
Strategy Performance Analyzer - Visualize and analyze strategy performance across market conditions
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union
import argparse
import datetime
from collections import defaultdict

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import our modules
from strategy_registry import StrategyRegistry
from advanced_strategies import (
    MeanReversionStrategy, 
    MomentumStrategy, 
    VolumeProfileStrategy, 
    VolatilityBreakoutStrategy,
    backtest_strategy
)
from synthetic_market_generator import SyntheticMarketGenerator


class StrategyPerformanceAnalyzer:
    """Analyzes and visualizes trading strategy performance across market conditions."""
    
    def __init__(self, registry_path: str = "./strategy_registry"):
        """
        Initialize the analyzer.
        
        Args:
            registry_path: Path to strategy registry
        """
        self.registry = StrategyRegistry(registry_path)
        self.market_generator = SyntheticMarketGenerator()
        
        # Timestamp for output files
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        self.output_dir = f"analysis_results/performance_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_strategy(self, strategy_type: str, parameters: Dict[str, Any]) -> Any:
        """
        Load a strategy with the given parameters.
        
        Args:
            strategy_type: Type of strategy
            parameters: Strategy parameters
            
        Returns:
            Strategy instance
        """
        if strategy_type == "MeanReversion":
            return MeanReversionStrategy(**parameters)
        elif strategy_type == "Momentum":
            return MomentumStrategy(**parameters)
        elif strategy_type == "VolumeProfile":
            return VolumeProfileStrategy(**parameters)
        elif strategy_type == "VolatilityBreakout":
            return VolatilityBreakoutStrategy(**parameters)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    def generate_market_scenarios(self) -> Dict[str, pd.DataFrame]:
        """
        Generate various market scenarios for testing.
        
        Returns:
            Dictionary of scenario name to market data
        """
        scenarios = {}
        
        # Standard scenarios
        scenarios["bull_market"] = self.market_generator.generate_bull_market()
        scenarios["bear_market"] = self.market_generator.generate_bear_market()
        scenarios["sideways_market"] = self.market_generator.generate_sideways_market()
        scenarios["volatile_market"] = self.market_generator.generate_volatile_market()
        
        # Advanced scenarios
        scenarios["flash_crash"] = self.market_generator.generate_flash_crash()
        scenarios["sector_rotation"] = self.market_generator.generate_sector_rotation()
        
        # Save scenarios for reference
        for name, data in scenarios.items():
            data.to_csv(os.path.join(self.output_dir, f"{name}_data.csv"))
        
        return scenarios
    
    def analyze_strategies_across_scenarios(
        self, 
        strategies: List[Dict[str, Any]], 
        scenarios: Optional[Dict[str, pd.DataFrame]] = None
    ):
        """
        Analyze a list of strategies across different market scenarios.
        
        Args:
            strategies: List of strategy dictionaries with type and parameters
            scenarios: Dictionary of scenarios (name to market data)
        """
        if scenarios is None:
            print("Generating market scenarios...")
            scenarios = self.generate_market_scenarios()
        
        # Dictionary to hold results per scenario per strategy
        results = {scenario: {} for scenario in scenarios}
        
        # Run each strategy on each scenario
        for i, strategy_dict in enumerate(strategies):
            print(f"Processing strategy {i+1}/{len(strategies)}: {strategy_dict['strategy_type']}")
            
            strategy = self.load_strategy(
                strategy_dict["strategy_type"],
                strategy_dict["parameters"]
            )
            
            strategy_id = f"{strategy.strategy_name}-{hash(str(strategy_dict['parameters']))}"
            
            for scenario_name, market_data in scenarios.items():
                print(f"  Testing on {scenario_name}...")
                backtest_result = backtest_strategy(strategy, market_data)
                results[scenario_name][strategy_id] = {
                    "strategy_type": strategy.strategy_name,
                    "parameters": strategy_dict["parameters"],
                    "return": backtest_result["total_return_pct"],
                    "drawdown": backtest_result["max_drawdown"],
                    "win_rate": backtest_result["win_rate"],
                    "trade_count": backtest_result["trade_count"],
                    "profit_factor": backtest_result["profit_factor"],
                    "equity_curve": backtest_result["equity_curve"]
                }
        
        # Save results
        self.save_analysis_results(results)
        
        # Generate visualizations
        self.generate_performance_visualizations(results)
        
        return results
    
    def save_analysis_results(self, results: Dict[str, Dict[str, Dict[str, Any]]]):
        """
        Save analysis results to JSON.
        
        Args:
            results: Dictionary of results by scenario and strategy
        """
        # Convert to serializable format (remove numpy arrays and pandas objects)
        serializable_results = {}
        
        for scenario, strategies in results.items():
            serializable_results[scenario] = {}
            
            for strategy_id, metrics in strategies.items():
                serializable_results[scenario][strategy_id] = {
                    k: v for k, v in metrics.items() 
                    if k != "equity_curve"  # Exclude non-serializable objects
                }
        
        # Save to file
        with open(os.path.join(self.output_dir, "performance_results.json"), "w") as f:
            json.dump(serializable_results, f, indent=2)
    
    def generate_performance_visualizations(self, results: Dict[str, Dict[str, Dict[str, Any]]]):
        """
        Generate visualizations of performance results.
        
        Args:
            results: Dictionary of results by scenario and strategy
        """
        # 1. Returns heatmap
        self._create_metric_heatmap(results, "return", "Returns (%)")
        
        # 2. Drawdown heatmap
        self._create_metric_heatmap(results, "drawdown", "Max Drawdown (%)", cmap="YlOrRd_r")
        
        # 3. Win rate heatmap
        self._create_metric_heatmap(results, "win_rate", "Win Rate (%)")
        
        # 4. Profit factor heatmap
        self._create_metric_heatmap(results, "profit_factor", "Profit Factor")
        
        # 5. Strategy type performance comparison
        self._create_strategy_type_comparison(results)
        
        # 6. Equity curves by scenario
        self._create_equity_curves_by_scenario(results)
        
        # 7. Strategy robustness analysis
        self._analyze_strategy_robustness(results)
    
    def _create_metric_heatmap(
        self, 
        results: Dict[str, Dict[str, Dict[str, Any]]], 
        metric: str, 
        title: str,
        cmap: str = "YlGnBu"
    ):
        """
        Create a heatmap for a specific metric across scenarios and strategies.
        
        Args:
            results: Dictionary of results by scenario and strategy
            metric: Metric to visualize
            title: Chart title
            cmap: Colormap to use
        """
        # Extract all strategy IDs
        all_strategy_ids = set()
        for scenario_results in results.values():
            all_strategy_ids.update(scenario_results.keys())
        
        # Create data frame
        data = []
        
        for scenario, strategies in results.items():
            for strategy_id in all_strategy_ids:
                if strategy_id in strategies:
                    strategy_type = strategies[strategy_id]["strategy_type"]
                    value = strategies[strategy_id][metric]
                    
                    data.append({
                        "Scenario": scenario,
                        "Strategy": f"{strategy_type} ({strategy_id[-6:]})",
                        "Value": value
                    })
                else:
                    # Strategy not tested on this scenario
                    pass
        
        df = pd.DataFrame(data)
        
        # Create pivot table
        pivot = df.pivot(index="Scenario", columns="Strategy", values="Value")
        
        # Create figure
        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot, annot=True, cmap=cmap, fmt=".1f")
        plt.title(f"{title} by Strategy and Market Scenario")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, f"{metric}_heatmap.png"))
        plt.close()
    
    def _create_strategy_type_comparison(self, results: Dict[str, Dict[str, Dict[str, Any]]]):
        """
        Create a comparison of different strategy types across scenarios.
        
        Args:
            results: Dictionary of results by scenario and strategy
        """
        # Group results by strategy type
        data = []
        
        for scenario, strategies in results.items():
            # Group by strategy type
            strategy_type_results = defaultdict(list)
            
            for strategy_id, metrics in strategies.items():
                strategy_type = metrics["strategy_type"]
                strategy_type_results[strategy_type].append(metrics)
            
            # Calculate average metrics for each strategy type
            for strategy_type, type_metrics in strategy_type_results.items():
                avg_return = np.mean([m["return"] for m in type_metrics])
                avg_drawdown = np.mean([m["drawdown"] for m in type_metrics])
                avg_win_rate = np.mean([m["win_rate"] for m in type_metrics])
                avg_profit_factor = np.mean([m["profit_factor"] for m in type_metrics])
                
                data.append({
                    "Scenario": scenario,
                    "Strategy Type": strategy_type,
                    "Avg Return": avg_return,
                    "Avg Drawdown": avg_drawdown,
                    "Avg Win Rate": avg_win_rate,
                    "Avg Profit Factor": avg_profit_factor
                })
        
        df = pd.DataFrame(data)
        
        # 1. Average returns by strategy type and scenario
        plt.figure(figsize=(12, 8))
        chart = sns.barplot(x="Strategy Type", y="Avg Return", hue="Scenario", data=df)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45, ha="right")
        plt.title("Average Returns by Strategy Type and Market Scenario")
        plt.ylabel("Return (%)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "strategy_type_returns.png"))
        plt.close()
        
        # 2. Average drawdown by strategy type and scenario
        plt.figure(figsize=(12, 8))
        chart = sns.barplot(x="Strategy Type", y="Avg Drawdown", hue="Scenario", data=df)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45, ha="right")
        plt.title("Average Drawdown by Strategy Type and Market Scenario")
        plt.ylabel("Drawdown (%)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "strategy_type_drawdowns.png"))
        plt.close()
        
        # 3. Combined metrics radar chart
        self._create_strategy_radar_chart(df)
    
    def _create_strategy_radar_chart(self, df: pd.DataFrame):
        """
        Create a radar chart comparing strategy types across metrics.
        
        Args:
            df: DataFrame with strategy type metrics
        """
        # Get unique strategy types
        strategy_types = df["Strategy Type"].unique()
        
        # Average metrics across all scenarios
        avg_metrics = df.groupby("Strategy Type").mean().reset_index()
        
        # Metrics to plot
        metrics = ["Avg Return", "Avg Win Rate", "Avg Profit Factor"]
        
        # Scale metrics to 0-1 range for radar chart
        scaled_metrics = avg_metrics.copy()
        
        for metric in metrics:
            max_val = scaled_metrics[metric].max()
            min_val = scaled_metrics[metric].min()
            
            if max_val == min_val:
                scaled_metrics[metric] = 0.5  # All values are the same
            else:
                scaled_metrics[metric] = (scaled_metrics[metric] - min_val) / (max_val - min_val)
        
        # For drawdown, lower is better so invert the scale
        max_drawdown = scaled_metrics["Avg Drawdown"].max()
        min_drawdown = scaled_metrics["Avg Drawdown"].min()
        
        if max_drawdown == min_drawdown:
            scaled_metrics["Avg Drawdown"] = 0.5  # All values are the same
        else:
            scaled_metrics["Avg Drawdown"] = 1 - (scaled_metrics["Avg Drawdown"] - min_drawdown) / (max_drawdown - min_drawdown)
        
        # Add drawdown to metrics list
        metrics.append("Avg Drawdown")
        
        # Create radar chart
        labels = np.array(metrics)
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for _, row in scaled_metrics.iterrows():
            values = row[metrics].values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, label=row["Strategy Type"])
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        
        ax.set_ylim(0, 1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title("Strategy Types Performance Comparison", y=1.08)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "strategy_radar_chart.png"))
        plt.close()
        
        # Create a table with actual values (not scaled)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis('off')
        ax.axis('tight')
        
        # Create the table with actual values
        table_data = avg_metrics[["Strategy Type"] + metrics].copy()
        
        # Format the values
        table_data["Avg Return"] = table_data["Avg Return"].round(2)
        table_data["Avg Drawdown"] = table_data["Avg Drawdown"].round(2)
        table_data["Avg Win Rate"] = table_data["Avg Win Rate"].round(2)
        table_data["Avg Profit Factor"] = table_data["Avg Profit Factor"].round(2)
        
        table = ax.table(
            cellText=table_data.values, 
            colLabels=table_data.columns,
            loc='center',
            cellLoc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.title("Strategy Types Performance Metrics")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "strategy_metrics_table.png"))
        plt.close()
    
    def _create_equity_curves_by_scenario(self, results: Dict[str, Dict[str, Dict[str, Any]]]):
        """
        Create equity curve plots for each scenario.
        
        Args:
            results: Dictionary of results by scenario and strategy
        """
        for scenario, strategies in results.items():
            plt.figure(figsize=(12, 8))
            
            for strategy_id, metrics in strategies.items():
                # Get strategy type and shortened ID
                strategy_type = metrics["strategy_type"]
                short_id = strategy_id[-6:]
                
                # Plot equity curve
                equity_curve = metrics["equity_curve"]
                plt.plot(
                    range(len(equity_curve)), 
                    equity_curve, 
                    label=f"{strategy_type} ({short_id})"
                )
            
            plt.title(f"Equity Curves - {scenario.replace('_', ' ').title()}")
            plt.xlabel("Trading Days")
            plt.ylabel("Equity")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.output_dir, f"{scenario}_equity_curves.png"))
            plt.close()
    
    def _analyze_strategy_robustness(self, results: Dict[str, Dict[str, Dict[str, Any]]]):
        """
        Analyze the robustness of strategies across different scenarios.
        
        Args:
            results: Dictionary of results by scenario and strategy
        """
        # Get all unique strategy IDs across all scenarios
        all_strategy_ids = set()
        for scenario_results in results.values():
            all_strategy_ids.update(scenario_results.keys())
        
        # Calculate robustness metrics
        robustness_data = []
        
        for strategy_id in all_strategy_ids:
            # Collect metrics across scenarios
            returns = []
            drawdowns = []
            present_in_scenarios = []
            
            for scenario, strategies in results.items():
                if strategy_id in strategies:
                    returns.append(strategies[strategy_id]["return"])
                    drawdowns.append(strategies[strategy_id]["drawdown"])
                    present_in_scenarios.append(scenario)
            
            # Only analyze strategies tested in multiple scenarios
            if len(returns) > 1:
                strategy_type = None
                
                # Find strategy type (from any scenario where it's present)
                for scenario in present_in_scenarios:
                    strategy_type = results[scenario][strategy_id]["strategy_type"]
                    if strategy_type:
                        break
                
                # Calculate robustness metrics
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                min_return = np.min(returns)
                max_drawdown = np.max(drawdowns)
                
                # Coefficient of variation (lower is more consistent)
                if avg_return != 0:
                    coef_variation = std_return / abs(avg_return)
                else:
                    coef_variation = float('inf')
                
                # Calculate robustness score
                # Good strategies have: high returns, low variation, positive worst case
                robustness_score = (
                    avg_return / 10  # Scaled return contribution
                    - coef_variation * 5  # Penalize variation
                    + min_return / 5  # Reward positive worst case
                    - max_drawdown / 10  # Penalize large drawdowns
                )
                
                robustness_data.append({
                    "Strategy ID": strategy_id[-6:],
                    "Strategy Type": strategy_type,
                    "Avg Return": avg_return,
                    "Return Std Dev": std_return,
                    "Min Return": min_return,
                    "Max Drawdown": max_drawdown,
                    "Coefficient of Variation": coef_variation,
                    "Robustness Score": robustness_score,
                    "Scenarios": len(present_in_scenarios)
                })
        
        # Create DataFrame and sort by robustness score
        robustness_df = pd.DataFrame(robustness_data)
        robustness_df = robustness_df.sort_values("Robustness Score", ascending=False)
        
        # Save robustness data
        robustness_df.to_csv(os.path.join(self.output_dir, "strategy_robustness.csv"), index=False)
        
        # Create robustness score visualization
        plt.figure(figsize=(12, 8))
        
        # Color by strategy type
        strategy_types = robustness_df["Strategy Type"].unique()
        colors = sns.color_palette("husl", len(strategy_types))
        color_map = dict(zip(strategy_types, colors))
        
        # Create bar chart
        sns.barplot(
            x="Strategy ID", 
            y="Robustness Score", 
            hue="Strategy Type",
            palette=color_map,
            data=robustness_df.head(15)  # Show top 15 most robust strategies
        )
        
        plt.title("Strategy Robustness Across Market Scenarios")
        plt.xlabel("Strategy ID")
        plt.ylabel("Robustness Score")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, "strategy_robustness_scores.png"))
        plt.close()
        
        # Create scatter plot of return vs. variation
        plt.figure(figsize=(10, 8))
        
        for strategy_type in strategy_types:
            type_data = robustness_df[robustness_df["Strategy Type"] == strategy_type]
            
            plt.scatter(
                type_data["Coefficient of Variation"],
                type_data["Avg Return"],
                s=100,
                alpha=0.7,
                label=strategy_type,
                color=color_map[strategy_type]
            )
        
        plt.title("Return vs. Consistency Across Market Scenarios")
        plt.xlabel("Coefficient of Variation (lower is more consistent)")
        plt.ylabel("Average Return (%)")
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, "return_vs_consistency.png"))
        plt.close()
        
        # Print top robust strategies
        print("\nMost Robust Strategies:")
        for i, (_, row) in enumerate(robustness_df.head(5).iterrows()):
            print(f"{i+1}. {row['Strategy Type']} (ID: {row['Strategy ID']})")
            print(f"   Robustness Score: {row['Robustness Score']:.2f}")
            print(f"   Avg Return: {row['Avg Return']:.2f}%")
            print(f"   Return Consistency: {row['Coefficient of Variation']:.2f}")
            print(f"   Worst Case Return: {row['Min Return']:.2f}%")
            print(f"   Max Drawdown: {row['Max Drawdown']:.2f}%")
            print()
        
        return robustness_df

    def analyze_registry_strategies(self, min_generation: int = 5):
        """
        Analyze strategies from the registry and their performance.
        
        Args:
            min_generation: Minimum generation to consider (to focus on evolved strategies)
        """
        print("Analyzing registry strategies...")
        
        # Get all strategies from registry
        all_strategies = self.registry.get_all_strategies()
        
        # Filter to strategies from later generations
        evolved_strategies = [
            s for s in all_strategies 
            if s.get("generation", 0) >= min_generation
        ]
        
        if not evolved_strategies:
            print("No evolved strategies found in registry.")
            return
        
        print(f"Found {len(evolved_strategies)} evolved strategies.")
        
        # Group by strategy type
        strategy_types = {}
        
        for strategy in evolved_strategies:
            strategy_type = strategy.get("strategy_type", "Unknown")
            
            if strategy_type not in strategy_types:
                strategy_types[strategy_type] = []
            
            strategy_types[strategy_type].append(strategy)
        
        # Print statistics for each strategy type
        print("\nStrategy Type Distribution:")
        for strategy_type, strategies in strategy_types.items():
            print(f"{strategy_type}: {len(strategies)} strategies")
        
        # Generate market scenarios
        scenarios = self.generate_market_scenarios()
        
        # For each strategy type, select top performers by fitness
        top_strategies = []
        
        for strategy_type, strategies in strategy_types.items():
            # Sort by fitness (if available)
            if "fitness" in strategies[0]:
                sorted_strategies = sorted(
                    strategies, 
                    key=lambda s: s.get("fitness", 0), 
                    reverse=True
                )
            else:
                # If fitness not available, try to use performance metrics
                sorted_strategies = sorted(
                    strategies, 
                    key=lambda s: s.get("performance", {}).get("total_return_pct", 0), 
                    reverse=True
                )
            
            # Take top 5 of each type
            for strategy in sorted_strategies[:5]:
                top_strategies.append({
                    "strategy_type": strategy["strategy_type"],
                    "parameters": strategy["parameters"]
                })
        
        # Analyze top strategies across scenarios
        if top_strategies:
            print(f"\nAnalyzing top {len(top_strategies)} strategies across market scenarios...")
            self.analyze_strategies_across_scenarios(top_strategies, scenarios)
        else:
            print("No strategies to analyze.")


def test_with_default_strategies():
    """Test the analyzer with default strategy configurations."""
    analyzer = StrategyPerformanceAnalyzer()
    
    # Create test strategies (one of each type)
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
    
    # Run analysis
    analyzer.analyze_strategies_across_scenarios(strategies)
    
    print(f"Analysis complete. Results saved to {analyzer.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze trading strategy performance")
    
    parser.add_argument(
        "--registry", 
        type=str, 
        default="./strategy_registry",
        help="Path to strategy registry"
    )
    
    parser.add_argument(
        "--analyze-registry", 
        action="store_true",
        help="Analyze strategies from registry"
    )
    
    parser.add_argument(
        "--min-generation", 
        type=int, 
        default=5,
        help="Minimum generation for registry strategies"
    )
    
    parser.add_argument(
        "--test-defaults", 
        action="store_true",
        help="Test with default strategy configurations"
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = StrategyPerformanceAnalyzer(args.registry)
    
    if args.analyze_registry:
        analyzer.analyze_registry_strategies(args.min_generation)
    elif args.test_defaults:
        test_with_default_strategies()
    else:
        print("No action specified. Use --analyze-registry or --test-defaults")
