"""
Strategy Optimizer UI Module

Provides a professional user interface for strategy optimization,
parameter exploration, and performance analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime

from trading_bot.ml_pipeline.optimizer import BaseOptimizer, GeneticOptimizer, MultiTimeframeOptimizer
from trading_bot.strategies.strategy_factory import StrategyFactory

class OptimizerUI:
    """
    Streamlit UI for the strategy optimizer
    
    Provides an interface for:
    - Configuring optimization runs
    - Visualizing optimization results 
    - Comparing strategies across timeframes and regimes
    """
    
    def __init__(self, config=None):
        """
        Initialize the optimizer UI
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.data_folder = self.config.get('data_folder', 'data')
        self.results_folder = self.config.get('results_folder', 'optimization_results')
        
        # Ensure results folder exists
        os.makedirs(self.results_folder, exist_ok=True)
        
        # Default optimization parameters
        self.default_params = {
            'optimization_method': 'genetic',
            'metric': 'sharpe_ratio',
            'population_size': 50,
            'generations': 10,
            'crossover_rate': 0.7,
            'mutation_rate': 0.2,
            'n_trials': 100,
            'timeframes': ['1m', '5m', '15m', '1h', '4h', 'D']
        }
    
    def render(self):
        """
        Render the optimizer UI
        """
        st.title("Strategy Optimizer")
        
        tabs = st.tabs(["Configure Optimization", "Results Analysis", "Performance Comparison"])
        
        with tabs[0]:
            self._render_configuration_tab()
        
        with tabs[1]:
            self._render_results_tab()
        
        with tabs[2]:
            self._render_comparison_tab()
    
    def _render_configuration_tab(self):
        """
        Render the optimization configuration tab
        """
        st.header("Configure Optimization Run")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Strategy selection
            strategies = StrategyFactory.available_strategies()
            strategy_type = st.selectbox(
                "Strategy Type", 
                strategies,
                help="Select the strategy type to optimize"
            )
            
            # Optimization method
            optimization_method = st.selectbox(
                "Optimization Method", 
                ["grid", "random", "genetic"],
                index=2,
                help="Grid: exhaustive search, Random: random sampling, Genetic: evolutionary algorithm"
            )
            
            # Optimization metric
            metric = st.selectbox(
                "Optimization Metric", 
                ["total_profit", "sharpe_ratio", "sortino_ratio", "calmar_ratio", "expectancy"],
                index=1,
                help="Metric to optimize for. Sharpe/Sortino/Calmar are risk-adjusted returns"
            )
            
            # Symbol selection
            symbols = self._get_available_symbols()
            selected_symbols = st.multiselect(
                "Symbols to Include",
                symbols,
                default=[symbols[0]] if symbols else [],
                help="Select one or more symbols to include in optimization"
            )
        
        with col2:
            # Method-specific parameters
            st.subheader("Method Parameters")
            
            if optimization_method == "genetic":
                population_size = st.slider(
                    "Population Size", 
                    min_value=10, 
                    max_value=200,
                    value=50,
                    step=10,
                    help="Number of parameter sets in each generation"
                )
                
                generations = st.slider(
                    "Generations", 
                    min_value=5, 
                    max_value=50,
                    value=10,
                    step=5,
                    help="Number of evolutionary generations to run"
                )
                
                crossover_rate = st.slider(
                    "Crossover Rate", 
                    min_value=0.1, 
                    max_value=0.9,
                    value=0.7,
                    step=0.1,
                    help="Probability of parameter crossover between individuals"
                )
                
                mutation_rate = st.slider(
                    "Mutation Rate", 
                    min_value=0.05, 
                    max_value=0.5,
                    value=0.2,
                    step=0.05,
                    help="Probability of random parameter mutation"
                )
            else:
                n_trials = st.slider(
                    "Number of Trials", 
                    min_value=10, 
                    max_value=500,
                    value=100,
                    step=10,
                    help="Number of parameter combinations to test"
                )
            
            # Multi-timeframe options
            st.subheader("Timeframe Options")
            
            timeframes = st.multiselect(
                "Timeframes",
                ["1m", "5m", "15m", "30m", "1h", "4h", "D", "W"],
                default=["1h", "4h", "D"],
                help="Select timeframes to test strategy on"
            )
            
            use_multi_timeframe = st.checkbox(
                "Enable Multi-Timeframe Testing", 
                value=True,
                help="Test consistency across multiple timeframes"
            )
            
            include_regime_detection = st.checkbox(
                "Include Market Regime Analysis", 
                value=True,
                help="Test performance in different market regimes"
            )
        
        # Parameter space definition
        st.subheader("Parameter Space")
        st.markdown("Define the parameter ranges to search through")
        
        param_space = {}
        
        # Get default parameters for the selected strategy
        default_strategy_params = self._get_default_params(strategy_type)
        
        for param_name, param_info in default_strategy_params.items():
            param_type = param_info.get('type', 'float')
            default_value = param_info.get('default', 0.0)
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                include_param = st.checkbox(f"Optimize {param_name}", value=True)
            
            if include_param:
                with col2:
                    if param_type in ['float', 'int']:
                        min_value = st.number_input(
                            f"{param_name} Min",
                            value=float(default_value) * 0.5 if param_type == 'float' else int(default_value) - 5,
                            step=0.1 if param_type == 'float' else 1
                        )
                    else:
                        min_value = None
                
                with col3:
                    if param_type in ['float', 'int']:
                        max_value = st.number_input(
                            f"{param_name} Max",
                            value=float(default_value) * 1.5 if param_type == 'float' else int(default_value) + 5,
                            step=0.1 if param_type == 'float' else 1
                        )
                    else:
                        max_value = None
                
                # Create parameter ranges based on type
                if param_type == 'float':
                    step = (max_value - min_value) / 10
                    param_space[param_name] = np.round(np.arange(min_value, max_value + step, step), 2).tolist()
                elif param_type == 'int':
                    param_space[param_name] = list(range(int(min_value), int(max_value) + 1))
                elif param_type == 'bool':
                    param_space[param_name] = [True, False]
                elif param_type == 'category':
                    options = param_info.get('options', [])
                    param_space[param_name] = options
        
        # Run optimization button
        if st.button("Run Optimization", type="primary"):
            if not param_space:
                st.error("Please define at least one parameter to optimize")
            elif not selected_symbols:
                st.error("Please select at least one symbol")
            elif not timeframes:
                st.error("Please select at least one timeframe")
            else:
                # Create config for optimizer
                optimizer_config = {
                    'optimization_method': optimization_method,
                    'metric': metric,
                    'strategy_type': strategy_type,
                    'symbols': selected_symbols,
                    'timeframes': timeframes,
                    'param_space': param_space,
                    'use_multi_timeframe': use_multi_timeframe,
                    'include_regime_detection': include_regime_detection,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                }
                
                # Add method-specific parameters
                if optimization_method == "genetic":
                    optimizer_config.update({
                        'population_size': population_size,
                        'generations': generations,
                        'crossover_rate': crossover_rate,
                        'mutation_rate': mutation_rate
                    })
                else:
                    optimizer_config.update({
                        'n_trials': n_trials
                    })
                
                # Save configuration
                self._save_optimization_config(optimizer_config)
                
                # Display confirmation
                st.success(f"Optimization job created for {strategy_type} strategy")
                st.info("The optimization will run in the background. View results in the 'Results Analysis' tab once complete.")
                
                # In a real implementation, would trigger optimizer to run here
                # For now, just save the configuration
    
    def _render_results_tab(self):
        """
        Render the results analysis tab
        """
        st.header("Optimization Results Analysis")
        
        # Get available optimization results
        optimization_runs = self._get_optimization_runs()
        
        if not optimization_runs:
            st.info("No optimization results available. Run an optimization in the 'Configure Optimization' tab.")
            return
        
        # Select optimization run
        selected_run = st.selectbox(
            "Select Optimization Run",
            optimization_runs,
            format_func=lambda x: f"{x['strategy_type']} - {x['timestamp']} - {x['optimization_method']}"
        )
        
        if not selected_run:
            return
        
        # Display optimization summary
        st.subheader("Optimization Summary")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"**Strategy:** {selected_run['strategy_type']}")
            st.markdown(f"**Method:** {selected_run['optimization_method']}")
            st.markdown(f"**Metric:** {selected_run['metric']}")
            st.markdown(f"**Symbols:** {', '.join(selected_run['symbols'])}")
            st.markdown(f"**Timeframes:** {', '.join(selected_run['timeframes'])}")
        
        with col2:
            if selected_run['optimization_method'] == 'genetic':
                st.markdown(f"**Population Size:** {selected_run.get('population_size', 'N/A')}")
                st.markdown(f"**Generations:** {selected_run.get('generations', 'N/A')}")
            else:
                st.markdown(f"**Number of Trials:** {selected_run.get('n_trials', 'N/A')}")
            
            st.markdown(f"**Multi-Timeframe Testing:** {'Enabled' if selected_run.get('use_multi_timeframe', False) else 'Disabled'}")
            st.markdown(f"**Regime Analysis:** {'Enabled' if selected_run.get('include_regime_detection', False) else 'Disabled'}")
        
        # Show best parameters
        st.subheader("Best Parameters")
        
        if 'best_params' in selected_run:
            best_params_df = pd.DataFrame({
                'Parameter': list(selected_run['best_params'].keys()),
                'Value': list(selected_run['best_params'].values())
            })
            st.dataframe(best_params_df)
            
            # Option to apply these parameters
            if st.button("Apply These Parameters to Strategy"):
                st.success(f"Parameters applied to {selected_run['strategy_type']} strategy")
                # In a real implementation, would update strategy parameters here
        else:
            st.info("No best parameters available for this run")
        
        # Visualize optimization progress
        if selected_run['optimization_method'] == 'genetic' and 'progress' in selected_run:
            st.subheader("Optimization Progress")
            
            progress_df = pd.DataFrame(selected_run['progress'])
            
            fig = px.line(
                progress_df, 
                x='generation', 
                y='best_fitness',
                title=f"Best {selected_run['metric']} per Generation"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        if 'best_metrics' in selected_run:
            st.subheader("Performance Metrics")
            
            metrics = selected_run['best_metrics']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Profit", f"{metrics.get('total_profit', 0):.2%}")
                st.metric("Win Rate", f"{metrics.get('win_rate', 0):.2%}")
                st.metric("Expectancy", f"{metrics.get('expectancy', 0):.4f}")
            
            with col2:
                st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}")
                st.metric("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.2f}")
            
            with col3:
                st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
                st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
                st.metric("SQN", f"{metrics.get('sqn', 0):.2f}")
        
        # Multi-timeframe analysis
        if selected_run.get('use_multi_timeframe', False) and 'timeframe_metrics' in selected_run:
            st.subheader("Multi-Timeframe Analysis")
            
            tf_metrics = selected_run['timeframe_metrics']
            
            # Create dataframe from timeframe metrics
            tf_data = []
            for tf, metrics in tf_metrics.items():
                row = {'timeframe': tf}
                row.update(metrics)
                tf_data.append(row)
            
            tf_df = pd.DataFrame(tf_data)
            
            # Plot timeframe performance
            fig = px.bar(
                tf_df,
                x='timeframe',
                y=selected_run['metric'],
                title=f"{selected_run['metric']} by Timeframe"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show parameter consistency
            if 'param_stability' in selected_run:
                st.subheader("Parameter Stability")
                
                stability = selected_run['param_stability']
                
                st.markdown(f"**Stability Score:** {stability.get('stability_score', 0):.2f}")
                
                if 'parameter_variance' in stability:
                    param_var_df = pd.DataFrame({
                        'Parameter': list(stability['parameter_variance'].keys()),
                        'Variance': list(stability['parameter_variance'].values())
                    })
                    st.dataframe(param_var_df)
    
    def _render_comparison_tab(self):
        """
        Render the performance comparison tab
        """
        st.header("Strategy Performance Comparison")
        
        # Get available optimization results
        optimization_runs = self._get_optimization_runs()
        
        if len(optimization_runs) < 2:
            st.info("Need at least two optimization runs for comparison. Run more optimizations in the 'Configure Optimization' tab.")
            return
        
        # Select optimization runs to compare
        selected_runs = st.multiselect(
            "Select Optimization Runs to Compare",
            optimization_runs,
            format_func=lambda x: f"{x['strategy_type']} - {x['timestamp']} - {x['optimization_method']}",
            default=optimization_runs[:2] if len(optimization_runs) >= 2 else []
        )
        
        if len(selected_runs) < 2:
            st.info("Select at least two optimization runs to compare")
            return
        
        # Compare metrics
        st.subheader("Performance Metrics Comparison")
        
        # Create comparison dataframe
        comparison_data = []
        
        for run in selected_runs:
            if 'best_metrics' not in run:
                continue
                
            row = {
                'Strategy': run['strategy_type'],
                'Method': run['optimization_method'],
                'Timestamp': run['timestamp']
            }
            row.update(run['best_metrics'])
            comparison_data.append(row)
        
        if not comparison_data:
            st.info("No metrics available for comparison")
            return
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Select metrics for comparison
        metric_options = [col for col in comparison_df.columns if col not in ['Strategy', 'Method', 'Timestamp']]
        selected_metrics = st.multiselect(
            "Select Metrics to Compare",
            metric_options,
            default=['total_profit', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        )
        
        if not selected_metrics:
            st.info("Select at least one metric for comparison")
            return
        
        # Create comparison chart
        fig = go.Figure()
        
        for i, run in enumerate(comparison_df.iterrows()):
            run_data = run[1]
            
            name = f"{run_data['Strategy']} ({run_data['Timestamp']})"
            values = [run_data[metric] for metric in selected_metrics]
            
            fig.add_trace(go.Bar(
                x=selected_metrics,
                y=values,
                name=name
            ))
        
        fig.update_layout(
            title="Strategy Performance Comparison",
            xaxis_title="Metric",
            yaxis_title="Value",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed comparison table
        st.subheader("Detailed Comparison")
        
        # Prepare detailed table
        detail_cols = ['Strategy', 'Method', 'Timestamp'] + selected_metrics
        st.dataframe(comparison_df[detail_cols])
        
        # Correlation analysis
        if len(selected_runs) >= 3 and 'equity_curve' in selected_runs[0]:
            st.subheader("Strategy Correlation Analysis")
            
            # Create correlation matrix
            corr_data = {}
            
            for run in selected_runs:
                if 'equity_curve' not in run:
                    continue
                    
                name = f"{run['strategy_type']} ({run['timestamp']})"
                corr_data[name] = pd.Series(run['equity_curve'])
            
            if corr_data:
                # Align all series to common index
                corr_df = pd.DataFrame(corr_data)
                corr_matrix = corr_df.corr()
                
                # Plot correlation heatmap
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    title="Strategy Return Correlation"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Note:** Lower correlation between strategies indicates higher diversification potential.
                Strategies with correlation < 0.5 may be good candidates for portfolio inclusion.
                """)
    
    def _get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols with data
        
        Returns:
            List of symbol names
        """
        # This is a placeholder - in a real implementation would scan data folder
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "FB", "TSLA", "BTC-USD", "ETH-USD"]
    
    def _get_default_params(self, strategy_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Get default parameters for a strategy
        
        Args:
            strategy_type: Type of strategy
            
        Returns:
            Dict of parameter info
        """
        # This is a placeholder - in a real implementation would get from strategy
        if strategy_type == "hybrid":
            return {
                "tech_weight": {"type": "float", "default": 0.33},
                "ml_weight": {"type": "float", "default": 0.33},
                "custom_weight": {"type": "float", "default": 0.34},
                "min_confidence": {"type": "float", "default": 0.6},
                "use_regime_adaptation": {"type": "bool", "default": True}
            }
        elif strategy_type == "rsi":
            return {
                "rsi_period": {"type": "int", "default": 14},
                "overbought": {"type": "int", "default": 70},
                "oversold": {"type": "int", "default": 30},
                "use_confirmation": {"type": "bool", "default": True}
            }
        elif strategy_type == "moving_average":
            return {
                "fast_period": {"type": "int", "default": 12},
                "slow_period": {"type": "int", "default": 26},
                "signal_period": {"type": "int", "default": 9},
                "ma_type": {"type": "category", "default": "EMA", "options": ["SMA", "EMA", "WMA"]}
            }
        else:
            return {}
    
    def _get_optimization_runs(self) -> List[Dict[str, Any]]:
        """
        Get list of available optimization runs
        
        Returns:
            List of optimization run configs
        """
        # This is a placeholder - in a real implementation would load from disk
        return [
            {
                "strategy_type": "hybrid",
                "optimization_method": "genetic",
                "metric": "sharpe_ratio",
                "symbols": ["AAPL", "MSFT"],
                "timeframes": ["1h", "4h", "D"],
                "use_multi_timeframe": True,
                "include_regime_detection": True,
                "population_size": 50,
                "generations": 10,
                "timestamp": "20250424_101523",
                "best_params": {
                    "tech_weight": 0.25,
                    "ml_weight": 0.45,
                    "custom_weight": 0.3,
                    "min_confidence": 0.65,
                    "use_regime_adaptation": True
                },
                "best_metrics": {
                    "total_profit": 0.385,
                    "sharpe_ratio": 1.85,
                    "sortino_ratio": 2.31,
                    "calmar_ratio": 1.42,
                    "win_rate": 0.62,
                    "max_drawdown": -0.15,
                    "profit_factor": 1.75,
                    "expectancy": 0.021,
                    "sqn": 2.3
                },
                "progress": [
                    {"generation": 1, "best_fitness": 1.2},
                    {"generation": 2, "best_fitness": 1.35},
                    {"generation": 3, "best_fitness": 1.47},
                    {"generation": 4, "best_fitness": 1.52},
                    {"generation": 5, "best_fitness": 1.65},
                    {"generation": 6, "best_fitness": 1.71},
                    {"generation": 7, "best_fitness": 1.75},
                    {"generation": 8, "best_fitness": 1.81},
                    {"generation": 9, "best_fitness": 1.83},
                    {"generation": 10, "best_fitness": 1.85}
                ],
                "timeframe_metrics": {
                    "1h": {"total_profit": 0.32, "sharpe_ratio": 1.65},
                    "4h": {"total_profit": 0.38, "sharpe_ratio": 1.85},
                    "D": {"total_profit": 0.41, "sharpe_ratio": 1.92}
                },
                "param_stability": {
                    "stability_score": 0.87,
                    "parameter_variance": {
                        "tech_weight": 0.02,
                        "ml_weight": 0.03,
                        "custom_weight": 0.015,
                        "min_confidence": 0.05,
                        "use_regime_adaptation": 0
                    }
                }
            },
            {
                "strategy_type": "rsi",
                "optimization_method": "grid",
                "metric": "total_profit",
                "symbols": ["AMZN", "TSLA"],
                "timeframes": ["4h", "D"],
                "use_multi_timeframe": False,
                "include_regime_detection": False,
                "n_trials": 100,
                "timestamp": "20250424_092304",
                "best_params": {
                    "rsi_period": 12,
                    "overbought": 75,
                    "oversold": 25,
                    "use_confirmation": True
                },
                "best_metrics": {
                    "total_profit": 0.28,
                    "sharpe_ratio": 1.45,
                    "sortino_ratio": 1.92,
                    "calmar_ratio": 1.21,
                    "win_rate": 0.58,
                    "max_drawdown": -0.18,
                    "profit_factor": 1.62,
                    "expectancy": 0.018,
                    "sqn": 1.9
                }
            }
        ]
    
    def _save_optimization_config(self, config: Dict[str, Any]):
        """
        Save optimization configuration
        
        Args:
            config: Optimization configuration
        """
        # This is a placeholder - in a real implementation would save to disk
        filename = f"{config['strategy_type']}_{config['timestamp']}_config.json"
        filepath = os.path.join(self.results_folder, filename)
        
        # In a real implementation, would save to disk
        # with open(filepath, 'w') as f:
        #     json.dump(config, f, indent=2)
        
        # For now, just log
        print(f"Would save optimization config to {filepath}")
