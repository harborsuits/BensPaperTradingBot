#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Straddle/Strangle Strategy Grid Search Optimizer

This script performs a comprehensive grid search to optimize the parameters
of the Straddle/Strangle strategy across different market regimes.
Results are stored in CSV files and visualized for easier analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from itertools import product
import json
import logging
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("straddle_grid_search.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import backtester and strategy components
try:
    from trading_bot.backtester.unified_backtester import UnifiedBacktester
    from trading_bot.strategies.options.volatility_spreads.straddle_strangle_strategy import StraddleStrangleStrategy
    from trading_bot.market.market_data import MarketData
    from trading_bot.market.option_chains import OptionChains
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    logger.error(f"Error importing components: {e}")
    IMPORTS_SUCCESSFUL = False


class StraddleGridSearch:
    """Grid search optimizer for Straddle/Strangle strategy."""
    
    def __init__(self, output_dir="reports/straddle_validation"):
        """Initialize the grid search optimizer."""
        logger.info("Initializing Straddle/Strangle Grid Search Optimizer")
        
        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define market regimes for testing
        self.market_regimes = {
            "high_volatility": {
                "start_date": "2020-03-01",
                "end_date": "2020-06-30",
                "description": "COVID-19 high volatility period"
            },
            "low_volatility": {
                "start_date": "2019-01-01",
                "end_date": "2019-06-30",
                "description": "2019 low volatility period"
            },
            "bullish": {
                "start_date": "2021-01-01",
                "end_date": "2021-06-30",
                "description": "2021 bullish period"
            },
            "bearish": {
                "start_date": "2022-01-01",
                "end_date": "2022-06-30",
                "description": "2022 bearish period"
            }
        }
        
        # Define parameter grid ranges
        self.param_grid = {
            "strategy_variant": ["straddle", "strangle"],
            "profit_target_pct": [30, 35, 40, 45, 50],
            "stop_loss_pct": [50, 60, 70, 80],
            "max_dte": [20, 30, 45],
            "exit_dte": [5, 7, 10],
            "exit_iv_drop_pct": [10, 15, 20, 25, 30],
            "max_drawdown_threshold": [8, 10, 12, 15]
        }
        
        # Define test symbols
        self.test_symbols = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA"]
        
        # Initialize backtester if imports successful
        if IMPORTS_SUCCESSFUL:
            self.backtester = UnifiedBacktester()
            self.market_data = MarketData()
            self.option_chains = OptionChains()
            logger.info("Components initialized successfully")
        else:
            logger.warning("Running in limited mode due to import errors")
    
    def generate_parameter_combinations(self):
        """Generate all parameter combinations for grid search."""
        logger.info("Generating parameter combinations")
        
        # Create all possible combinations of parameters
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = list(product(*values))
        
        # Convert to list of parameter dictionaries
        param_combinations = []
        for combo in combinations:
            param_dict = {keys[i]: combo[i] for i in range(len(keys))}
            param_combinations.append(param_dict)
        
        logger.info(f"Generated {len(param_combinations)} parameter combinations")
        return param_combinations
    
    def run_backtest(self, params, regime_name, symbol=None):
        """Run a backtest with specific parameters for a market regime."""
        if not IMPORTS_SUCCESSFUL:
            logger.error("Cannot run backtest - imports failed")
            return None
        
        # Get regime dates
        regime = self.market_regimes[regime_name]
        start_date = regime["start_date"]
        end_date = regime["end_date"]
        
        # Set up symbols to test
        symbols = [symbol] if symbol else self.test_symbols
        
        # Create strategy instance with parameters
        strategy = StraddleStrangleStrategy(
            strategy_id=f"grid_search_{regime_name}",
            name=f"Grid Search {regime_name}",
            parameters=params
        )
        
        # Run backtest
        try:
            results = self.backtester.run_backtest(
                strategy=strategy,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                initial_capital=100000,
                commission=1.0,  # $1 per contract
                slippage=0.01     # 1% slippage
            )
            
            # Extract key performance metrics
            metrics = {
                "regime": regime_name,
                "symbol": symbol if symbol else "multi",
                **params,  # Include all parameters
                "total_return_pct": results["total_return_pct"],
                "sharpe_ratio": results["sharpe_ratio"],
                "sortino_ratio": results["sortino_ratio"],
                "max_drawdown_pct": results["max_drawdown_pct"],
                "win_rate": results["win_rate"],
                "avg_profit_per_trade": results["avg_profit_per_trade"],
                "total_trades": results["total_trades"]
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return None
    
    def run_parallel_grid_search(self, regime_name, symbol=None):
        """Run grid search in parallel for a market regime."""
        logger.info(f"Running parallel grid search for regime: {regime_name}")
        
        # Generate parameter combinations
        param_combinations = self.generate_parameter_combinations()
        
        # Create args for parallel processing
        args = [(params, regime_name, symbol) for params in param_combinations]
        
        # Run backtests in parallel
        with Pool(processes=max(1, cpu_count() - 1)) as pool:
            results = list(tqdm(pool.starmap(self.run_backtest, args), total=len(args)))
        
        # Filter out None results
        results = [r for r in results if r is not None]
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        filename = f"{self.output_dir}/grid_search_{regime_name}"
        if symbol:
            filename += f"_{symbol}"
        filename += ".csv"
        
        results_df.to_csv(filename, index=False)
        logger.info(f"Saved results to {filename}")
        
        return results_df
    
    def run_full_grid_search(self):
        """Run grid search across all market regimes and symbols."""
        logger.info("Starting full grid search")
        all_results = []
        
        # Run for each regime
        for regime_name in self.market_regimes.keys():
            # First run with all symbols
            regime_results = self.run_parallel_grid_search(regime_name)
            all_results.append(regime_results)
            
            # Then run for each individual symbol
            for symbol in self.test_symbols:
                symbol_results = self.run_parallel_grid_search(regime_name, symbol)
                all_results.append(symbol_results)
        
        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(f"{self.output_dir}/all_grid_search_results.csv", index=False)
        
        # Generate summary reports
        self.generate_summary_reports(combined_results)
        
        logger.info("Full grid search completed")
        return combined_results
    
    def generate_summary_reports(self, results_df):
        """Generate summary reports from grid search results."""
        logger.info("Generating summary reports")
        
        # 1. Best parameters by market regime
        regime_best = results_df[results_df["symbol"] == "multi"].groupby("regime").apply(
            lambda x: x.loc[x["sharpe_ratio"].idxmax()]
        ).reset_index(drop=True)
        
        regime_best.to_csv(f"{self.output_dir}/best_params_by_regime.csv", index=False)
        
        # 2. Best parameters by symbol
        symbol_best = results_df.groupby(["symbol", "regime"]).apply(
            lambda x: x.loc[x["sharpe_ratio"].idxmax()]
        ).reset_index(drop=True)
        
        symbol_best.to_csv(f"{self.output_dir}/best_params_by_symbol_regime.csv", index=False)
        
        # 3. Parameter sensitivity analysis
        param_sensitivity = {}
        for param in self.param_grid.keys():
            # Skip strategy variant for the sensitivity analysis
            if param == "strategy_variant":
                continue
                
            # Calculate mean Sharpe ratio for each parameter value
            sensitivity = results_df.groupby([param, "regime"])["sharpe_ratio"].mean().reset_index()
            param_sensitivity[param] = sensitivity
            
            # Save parameter sensitivity
            sensitivity.to_csv(f"{self.output_dir}/sensitivity_{param}.csv", index=False)
        
        # 4. Generate consolidated recommendations
        self._generate_recommendations(regime_best, symbol_best, param_sensitivity)
        
        # 5. Create visualizations
        self._create_visualizations(results_df, regime_best, param_sensitivity)
    
    def _generate_recommendations(self, regime_best, symbol_best, param_sensitivity):
        """Generate consolidated parameter recommendations."""
        logger.info("Generating consolidated recommendations")
        
        # Create recommendations dictionary
        recommendations = {
            "generated_at": datetime.now().isoformat(),
            "regime_recommendations": {},
            "symbol_specific_recommendations": {},
            "overall_recommendations": {}
        }
        
        # Add regime-specific recommendations
        for _, row in regime_best.iterrows():
            regime = row["regime"]
            recommendations["regime_recommendations"][regime] = {
                param: row[param] for param in self.param_grid.keys()
            }
            recommendations["regime_recommendations"][regime]["sharpe_ratio"] = row["sharpe_ratio"]
            recommendations["regime_recommendations"][regime]["win_rate"] = row["win_rate"]
            recommendations["regime_recommendations"][regime]["total_return_pct"] = row["total_return_pct"]
        
        # Add symbol-specific recommendations
        for symbol in self.test_symbols:
            symbol_data = symbol_best[symbol_best["symbol"] == symbol]
            if not symbol_data.empty:
                recommendations["symbol_specific_recommendations"][symbol] = {}
                for _, row in symbol_data.iterrows():
                    regime = row["regime"]
                    recommendations["symbol_specific_recommendations"][symbol][regime] = {
                        param: row[param] for param in self.param_grid.keys()
                    }
                    recommendations["symbol_specific_recommendations"][symbol][regime]["sharpe_ratio"] = row["sharpe_ratio"]
        
        # Calculate overall best parameters (weighted by regime)
        # Weight recent regimes more heavily
        regime_weights = {
            "high_volatility": 0.3,
            "low_volatility": 0.3,
            "bullish": 0.2,
            "bearish": 0.2
        }
        
        # For each parameter, find the weighted best value
        for param in self.param_grid.keys():
            if param == "strategy_variant":
                # For strategy variant, count which one performs better in more regimes
                variant_counts = regime_best[param].value_counts()
                recommendations["overall_recommendations"][param] = variant_counts.idxmax()
            else:
                # For numeric parameters, calculate weighted average
                weighted_sum = 0
                total_weight = 0
                
                for regime, weight in regime_weights.items():
                    regime_row = regime_best[regime_best["regime"] == regime]
                    if not regime_row.empty:
                        weighted_sum += regime_row[param].values[0] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    # Round to nearest valid parameter value
                    avg_value = weighted_sum / total_weight
                    valid_values = sorted(self.param_grid[param])
                    closest_value = min(valid_values, key=lambda x: abs(x - avg_value))
                    recommendations["overall_recommendations"][param] = closest_value
        
        # Save recommendations to JSON
        with open(f"{self.output_dir}/parameter_recommendations.json", "w") as f:
            json.dump(recommendations, f, indent=4)
        
        logger.info(f"Saved parameter recommendations to {self.output_dir}/parameter_recommendations.json")
    
    def _create_visualizations(self, results_df, regime_best, param_sensitivity):
        """Create visualizations of grid search results."""
        logger.info("Creating visualizations")
        
        # Create visualizations directory
        viz_dir = f"{self.output_dir}/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Parameter sensitivity heatmap
        for param, sensitivity_df in param_sensitivity.items():
            plt.figure(figsize=(12, 8))
            pivot_table = sensitivity_df.pivot(index="regime", columns=param, values="sharpe_ratio")
            sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".2f")
            plt.title(f"Sharpe Ratio Sensitivity to {param}")
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/{param}_sensitivity.png")
            plt.close()
        
        # 2. Best parameters radar chart
        # Get the best parameters for each regime
        best_params = regime_best.set_index("regime")
        
        # Create radar chart
        params_to_plot = [p for p in self.param_grid.keys() if p != "strategy_variant"]
        n_params = len(params_to_plot)
        
        for regime in self.market_regimes.keys():
            if regime in best_params.index:
                # Create radar chart for each regime
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, polar=True)
                
                # Calculate angles
                angles = np.linspace(0, 2*np.pi, n_params, endpoint=False).tolist()
                angles += angles[:1]  # Close the loop
                
                # Get values
                values = [best_params.loc[regime, p] for p in params_to_plot]
                
                # Normalize values between 0 and 1
                normalized_values = []
                for i, val in enumerate(values):
                    param = params_to_plot[i]
                    min_val = min(self.param_grid[param])
                    max_val = max(self.param_grid[param])
                    normalized = (val - min_val) / (max_val - min_val)
                    normalized_values.append(normalized)
                
                # Close the loop
                normalized_values += normalized_values[:1]
                
                # Plot
                ax.plot(angles, normalized_values, linewidth=2, linestyle='solid')
                ax.fill(angles, normalized_values, alpha=0.1)
                
                # Add labels
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(params_to_plot)
                
                plt.title(f"Best Parameters for {regime} Regime")
                plt.tight_layout()
                plt.savefig(f"{viz_dir}/{regime}_radar.png")
                plt.close()
        
        # 3. Comparative performance bar chart
        plt.figure(figsize=(14, 8))
        performance_metrics = ["sharpe_ratio", "win_rate", "total_return_pct"]
        bar_data = regime_best[["regime"] + performance_metrics].melt(
            id_vars=["regime"], 
            value_vars=performance_metrics,
            var_name="Metric",
            value_name="Value"
        )
        
        g = sns.catplot(
            data=bar_data,
            kind="bar",
            x="regime",
            y="Value",
            hue="Metric",
            height=6,
            aspect=1.5
        )
        g.set_xticklabels(rotation=45)
        plt.title("Performance Metrics by Market Regime")
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/regime_performance.png")
        plt.close()
        
        logger.info(f"Saved visualizations to {viz_dir}")


def main():
    """Main entry point for grid search."""
    logger.info("----- STARTING STRADDLE/STRANGLE GRID SEARCH -----")
    
    # Initialize grid search
    grid_search = StraddleGridSearch()
    
    # Run grid search
    results = grid_search.run_full_grid_search()
    
    logger.info("----- GRID SEARCH COMPLETE -----")
    return results


if __name__ == "__main__":
    main()
