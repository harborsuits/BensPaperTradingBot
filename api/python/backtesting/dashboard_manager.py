#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Manager for ML models and backtests visualization
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO

from trading_bot.backtesting.data_manager import BacktestDataManager
from trading_bot.backtesting.learner import BacktestLearner
from trading_bot.backtesting.automated_backtester import AutomatedBacktester

logger = logging.getLogger(__name__)

class LearningDashboardManager:
    """
    Manager for handling ML model data and visualization in the dashboard.
    """
    
    def __init__(
        self,
        data_manager: Optional[BacktestDataManager] = None,
        learner: Optional[BacktestLearner] = None,
        backtester: Optional[AutomatedBacktester] = None
    ):
        """
        Initialize the Learning Dashboard Manager.
        
        Args:
            data_manager: Data manager instance
            learner: ML model learner instance
            backtester: Automated backtester instance
        """
        self.data_manager = data_manager if data_manager else BacktestDataManager()
        self.learner = learner if learner else BacktestLearner(self.data_manager)
        self.backtester = backtester if backtester else AutomatedBacktester(self.data_manager, self.learner)
    
    def get_model_list(self) -> List[Dict[str, Any]]:
        """
        Get a list of all available ML models.
        
        Returns:
            List of model information dictionaries
        """
        return self.learner.list_models()
    
    def get_model_details(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model details
        """
        # Check if model exists
        if model_name not in self.learner.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Get model configuration
        config = self.learner.model_configs.get(model_name, {})
        
        # Get model history if available
        history = self.learner.model_histories.get(model_name, {})
        
        # Convert history to format suitable for plotting
        plot_history = {}
        for metric, values in history.items():
            if isinstance(values, list):
                plot_history[metric] = values
        
        # Get model performance metrics
        metrics = {}
        if model_name in self.learner.model_metrics:
            metrics = self.learner.model_metrics[model_name]
        
        return {
            "name": model_name,
            "type": config.get("model_type"),
            "created_at": config.get("created_at"),
            "last_trained": config.get("last_trained"),
            "feature_columns": config.get("feature_columns", []),
            "target_column": config.get("target_column"),
            "hyperparameters": config.get("hyperparameters", {}),
            "history": plot_history,
            "metrics": metrics
        }
    
    def generate_model_plot(self, model_name: str, plot_type: str = "history") -> Dict[str, Any]:
        """
        Generate visualization for a model.
        
        Args:
            model_name: Name of the model
            plot_type: Type of plot (history, feature_importance, confusion_matrix)
            
        Returns:
            Dictionary with plot data
        """
        # Check if model exists
        if model_name not in self.learner.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Get model details
        model_details = self.get_model_details(model_name)
        
        if plot_type == "history":
            # Plot training history
            return self._generate_history_plot(model_name, model_details)
        
        elif plot_type == "feature_importance":
            # Plot feature importance if available
            return self._generate_feature_importance_plot(model_name, model_details)
        
        elif plot_type == "confusion_matrix":
            # Plot confusion matrix if available
            return self._generate_confusion_matrix_plot(model_name, model_details)
        
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
    
    def _generate_history_plot(self, model_name: str, model_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a plot of model training history.
        
        Args:
            model_name: Name of the model
            model_details: Dictionary with model details
            
        Returns:
            Dictionary with plot data
        """
        history = model_details.get("history", {})
        
        if not history:
            return {"error": "No training history available for this model"}
        
        # Create plotly figure
        fig = make_subplots(
            rows=len(history),
            cols=1,
            subplot_titles=list(history.keys()),
            vertical_spacing=0.1
        )
        
        # Add each metric as a trace
        row = 1
        for metric, values in history.items():
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(values) + 1)),
                    y=values,
                    mode="lines+markers",
                    name=metric
                ),
                row=row,
                col=1
            )
            row += 1
        
        # Update layout
        fig.update_layout(
            title=f"Training History - {model_name}",
            height=300 * len(history),
            width=800,
            showlegend=False
        )
        
        # Return plot as JSON
        return {
            "plot_type": "history",
            "plotly_json": fig.to_json()
        }
    
    def _generate_feature_importance_plot(self, model_name: str, model_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a plot of feature importance.
        
        Args:
            model_name: Name of the model
            model_details: Dictionary with model details
            
        Returns:
            Dictionary with plot data
        """
        # Check if model type supports feature importance
        model_type = model_details.get("type")
        
        if model_type not in ["random_forest", "xgboost", "linear"]:
            return {"error": f"Feature importance not available for model type: {model_type}"}
        
        # Try to get feature importance
        try:
            model = self.learner.models[model_name]
            
            if model_type == "random_forest":
                importances = model.feature_importances_
                feature_names = model_details.get("feature_columns", [])
                
                if not feature_names:
                    return {"error": "Feature names not available"}
                
                # Create plotly figure
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=feature_names,
                        y=importances,
                        marker_color="royalblue"
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Feature Importance - {model_name}",
                    xaxis_title="Features",
                    yaxis_title="Importance",
                    height=600,
                    width=800
                )
                
                # Return plot as JSON
                return {
                    "plot_type": "feature_importance",
                    "plotly_json": fig.to_json()
                }
            
            elif model_type == "linear":
                if hasattr(model, "coef_"):
                    coefficients = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                    feature_names = model_details.get("feature_columns", [])
                    
                    if not feature_names:
                        return {"error": "Feature names not available"}
                    
                    # Create plotly figure
                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            x=feature_names,
                            y=coefficients,
                            marker_color="royalblue"
                        )
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Feature Coefficients - {model_name}",
                        xaxis_title="Features",
                        yaxis_title="Coefficient",
                        height=600,
                        width=800
                    )
                    
                    # Return plot as JSON
                    return {
                        "plot_type": "feature_importance",
                        "plotly_json": fig.to_json()
                    }
            
            # Default case if feature importance not available
            return {"error": "Feature importance not available for this model"}
        
        except Exception as e:
            logger.error(f"Error generating feature importance plot: {e}")
            return {"error": f"Error generating feature importance plot: {str(e)}"}
    
    def _generate_confusion_matrix_plot(self, model_name: str, model_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a confusion matrix plot.
        
        Args:
            model_name: Name of the model
            model_details: Dictionary with model details
            
        Returns:
            Dictionary with plot data
        """
        # Check if confusion matrix is available in metrics
        metrics = model_details.get("metrics", {})
        confusion_matrix = metrics.get("confusion_matrix")
        
        if not confusion_matrix:
            return {"error": "Confusion matrix not available for this model"}
        
        # Create plotly figure
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=["Predicted Negative", "Predicted Positive"],
            y=["Actual Negative", "Actual Positive"],
            colorscale="Blues",
            showscale=False
        ))
        
        # Add annotations
        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix[i])):
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=str(confusion_matrix[i][j]),
                    showarrow=False,
                    font=dict(color="white" if confusion_matrix[i][j] > 100 else "black")
                )
        
        # Update layout
        fig.update_layout(
            title=f"Confusion Matrix - {model_name}",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=500,
            width=500
        )
        
        # Return plot as JSON
        return {
            "plot_type": "confusion_matrix",
            "plotly_json": fig.to_json()
        }
    
    def get_backtest_list(self) -> List[Dict[str, Any]]:
        """
        Get a list of all available ML backtests.
        
        Returns:
            List of backtest information dictionaries
        """
        return self.backtester.list_backtest_results()
    
    def get_backtest_details(self, backtest_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific backtest.
        
        Args:
            backtest_id: ID of the backtest
            
        Returns:
            Dictionary with backtest details
        """
        # Load backtest result if not in memory
        if backtest_id not in self.backtester.backtest_results:
            try:
                self.backtester.load_backtest_result(backtest_id)
            except Exception as e:
                raise ValueError(f"Error loading backtest result {backtest_id}: {e}")
        
        # Get backtest result
        result = self.backtester.backtest_results[backtest_id]
        
        # Extract key information
        return {
            "backtest_id": backtest_id,
            "model_info": result.get("model_info", {}),
            "symbol": result.get("symbol"),
            "timeframe": result.get("timeframe"),
            "start_date": result.get("start_date"),
            "end_date": result.get("end_date"),
            "initial_capital": result.get("initial_capital"),
            "final_capital": result.get("final_capital"),
            "total_return": result.get("total_return"),
            "sharpe_ratio": result.get("sharpe_ratio"),
            "max_drawdown": result.get("max_drawdown"),
            "win_rate": result.get("win_rate"),
            "profit_factor": result.get("profit_factor"),
            "total_trades": result.get("total_trades")
        }
    
    def generate_backtest_plot(self, backtest_id: str, plot_type: str = "equity_curve") -> Dict[str, Any]:
        """
        Generate visualization for a backtest.
        
        Args:
            backtest_id: ID of the backtest
            plot_type: Type of plot (equity_curve, drawdown, returns, trades, predictions)
            
        Returns:
            Dictionary with plot data
        """
        # Load backtest result if not in memory
        if backtest_id not in self.backtester.backtest_results:
            try:
                self.backtester.load_backtest_result(backtest_id)
            except Exception as e:
                raise ValueError(f"Error loading backtest result {backtest_id}: {e}")
        
        # Get backtest result
        result = self.backtester.backtest_results[backtest_id]
        
        if plot_type == "equity_curve":
            # Plot equity curve
            return self._generate_equity_curve_plot(backtest_id, result)
        
        elif plot_type == "drawdown":
            # Plot drawdown curve
            return self._generate_drawdown_plot(backtest_id, result)
        
        elif plot_type == "returns":
            # Plot returns distribution
            return self._generate_returns_plot(backtest_id, result)
        
        elif plot_type == "trades":
            # Plot trades
            return self._generate_trades_plot(backtest_id, result)
        
        elif plot_type == "predictions":
            # Plot model predictions
            return self._generate_predictions_plot(backtest_id, result)
        
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
    
    def _generate_equity_curve_plot(self, backtest_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an equity curve plot.
        
        Args:
            backtest_id: ID of the backtest
            result: Dictionary with backtest results
            
        Returns:
            Dictionary with plot data
        """
        # Extract equity curve data
        equity_curve = result.get("equity_curve")
        
        if not equity_curve:
            return {"error": "Equity curve data not available for this backtest"}
        
        # Convert to pandas Series if it's a list
        if isinstance(equity_curve, list):
            equity_curve = pd.Series(equity_curve)
        
        # Create plotly figure
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=equity_curve,
                mode="lines",
                name="Equity",
                line=dict(color="royalblue")
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f"Equity Curve - {result.get('symbol')} - {result.get('model_info', {}).get('model_name')}",
            xaxis_title="Trade Number",
            yaxis_title="Equity",
            height=600,
            width=1000
        )
        
        # Return plot as JSON
        return {
            "plot_type": "equity_curve",
            "plotly_json": fig.to_json()
        }
    
    def _generate_drawdown_plot(self, backtest_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a drawdown plot.
        
        Args:
            backtest_id: ID of the backtest
            result: Dictionary with backtest results
            
        Returns:
            Dictionary with plot data
        """
        # Extract drawdown data
        drawdown = result.get("drawdown")
        
        if not drawdown:
            return {"error": "Drawdown data not available for this backtest"}
        
        # Convert to pandas Series if it's a list
        if isinstance(drawdown, list):
            drawdown = pd.Series(drawdown)
        
        # Create plotly figure
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=drawdown,
                mode="lines",
                name="Drawdown",
                line=dict(color="crimson")
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f"Drawdown - {result.get('symbol')} - {result.get('model_info', {}).get('model_name')}",
            xaxis_title="Trade Number",
            yaxis_title="Drawdown (%)",
            height=400,
            width=1000
        )
        
        # Return plot as JSON
        return {
            "plot_type": "drawdown",
            "plotly_json": fig.to_json()
        }
    
    def _generate_returns_plot(self, backtest_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a returns distribution plot.
        
        Args:
            backtest_id: ID of the backtest
            result: Dictionary with backtest results
            
        Returns:
            Dictionary with plot data
        """
        # Extract returns data
        returns = result.get("returns")
        
        if not returns:
            return {"error": "Returns data not available for this backtest"}
        
        # Convert to pandas Series if it's a list
        if isinstance(returns, list):
            returns = pd.Series(returns)
        
        # Create plotly figure
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=returns,
                name="Returns",
                marker_color="royalblue",
                opacity=0.7
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f"Returns Distribution - {result.get('symbol')} - {result.get('model_info', {}).get('model_name')}",
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            height=500,
            width=800
        )
        
        # Return plot as JSON
        return {
            "plot_type": "returns",
            "plotly_json": fig.to_json()
        }
    
    def _generate_trades_plot(self, backtest_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a trades plot.
        
        Args:
            backtest_id: ID of the backtest
            result: Dictionary with backtest results
            
        Returns:
            Dictionary with plot data
        """
        # Extract trades data
        trades = result.get("trades")
        
        if not trades:
            return {"error": "Trades data not available for this backtest"}
        
        # Convert to DataFrame if it's a list of dictionaries
        if isinstance(trades, list):
            trades = pd.DataFrame(trades)
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add winning trades
        winning_trades = trades[trades["pnl"] > 0]
        if not winning_trades.empty:
            fig.add_trace(
                go.Scatter(
                    x=winning_trades.index,
                    y=winning_trades["pnl"],
                    mode="markers",
                    name="Winning Trades",
                    marker=dict(
                        color="green",
                        size=10,
                        symbol="circle"
                    )
                )
            )
        
        # Add losing trades
        losing_trades = trades[trades["pnl"] < 0]
        if not losing_trades.empty:
            fig.add_trace(
                go.Scatter(
                    x=losing_trades.index,
                    y=losing_trades["pnl"],
                    mode="markers",
                    name="Losing Trades",
                    marker=dict(
                        color="red",
                        size=10,
                        symbol="circle"
                    )
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f"Trades - {result.get('symbol')} - {result.get('model_info', {}).get('model_name')}",
            xaxis_title="Trade Number",
            yaxis_title="Profit/Loss",
            height=500,
            width=1000
        )
        
        # Return plot as JSON
        return {
            "plot_type": "trades",
            "plotly_json": fig.to_json()
        }
    
    def _generate_predictions_plot(self, backtest_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a predictions plot.
        
        Args:
            backtest_id: ID of the backtest
            result: Dictionary with backtest results
            
        Returns:
            Dictionary with plot data
        """
        # Extract predictions data
        predictions = result.get("predictions")
        
        if not predictions:
            return {"error": "Predictions data not available for this backtest"}
        
        # Create plotly figure
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=predictions,
                mode="lines",
                name="Predictions",
                line=dict(color="purple")
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f"Model Predictions - {result.get('symbol')} - {result.get('model_info', {}).get('model_name')}",
            xaxis_title="Time",
            yaxis_title="Prediction Value",
            height=400,
            width=1000
        )
        
        # Return plot as JSON
        return {
            "plot_type": "predictions",
            "plotly_json": fig.to_json()
        }
    
    def compare_backtests(self, backtest_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple backtests.
        
        Args:
            backtest_ids: List of backtest IDs to compare
            
        Returns:
            Dictionary with comparison results and plots
        """
        # Check if all backtest IDs are valid
        for backtest_id in backtest_ids:
            if backtest_id not in self.backtester.backtest_results and not os.path.exists(os.path.join(self.backtester.results_dir, f"{backtest_id}.json")):
                raise ValueError(f"Backtest result {backtest_id} not found")
        
        # Get comparison data from backtester
        comparison = self.backtester.compare_backtests(backtest_ids)
        
        # Generate comparison plots
        comparison["plots"] = {
            "equity_curves": self._generate_comparison_equity_plot(backtest_ids),
            "metrics": self._generate_comparison_metrics_plot(comparison)
        }
        
        return comparison
    
    def _generate_comparison_equity_plot(self, backtest_ids: List[str]) -> Dict[str, Any]:
        """
        Generate a comparison plot of equity curves.
        
        Args:
            backtest_ids: List of backtest IDs to compare
            
        Returns:
            Dictionary with plot data
        """
        # Create plotly figure
        fig = go.Figure()
        
        # Add equity curve for each backtest
        for backtest_id in backtest_ids:
            result = self.backtester.backtest_results[backtest_id]
            
            # Extract equity curve data
            equity_curve = result.get("equity_curve")
            
            if not equity_curve:
                continue
            
            # Convert to pandas Series if it's a list
            if isinstance(equity_curve, list):
                equity_curve = pd.Series(equity_curve)
            
            # Normalize to percentage return for fair comparison
            normalized_equity = (equity_curve / equity_curve.iloc[0] - 1) * 100
            
            # Add trace
            fig.add_trace(
                go.Scatter(
                    y=normalized_equity,
                    mode="lines",
                    name=f"{result.get('model_info', {}).get('model_name')} - {result.get('symbol')}",
                )
            )
        
        # Update layout
        fig.update_layout(
            title="Equity Curve Comparison",
            xaxis_title="Trade Number",
            yaxis_title="Return (%)",
            height=600,
            width=1000
        )
        
        # Return plot as JSON
        return {
            "plot_type": "comparison_equity",
            "plotly_json": fig.to_json()
        }
    
    def _generate_comparison_metrics_plot(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comparison plot of key metrics.
        
        Args:
            comparison: Dictionary with comparison results
            
        Returns:
            Dictionary with plot data
        """
        # Extract metrics
        metrics = comparison.get("metrics", {})
        
        if not metrics:
            return {"error": "Metrics data not available for comparison"}
        
        # Select key metrics for comparison
        key_metrics = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]
        
        # Create data for radar chart
        categories = key_metrics
        backtest_ids = list(metrics.keys())
        
        # Create plotly figure
        fig = go.Figure()
        
        # Normalize metrics for radar chart
        normalized_metrics = {}
        
        for metric in key_metrics:
            values = [metrics[bid].get(metric, 0) for bid in backtest_ids]
            if metric == "max_drawdown":
                # Invert max_drawdown so lower is better
                min_val = min(values)
                max_val = max(values)
                normalized_metrics[metric] = [(max_val - val) / (max_val - min_val) if max_val != min_val else 0.5 for val in values]
            else:
                min_val = min(values)
                max_val = max(values)
                normalized_metrics[metric] = [(val - min_val) / (max_val - min_val) if max_val != min_val else 0.5 for val in values]
        
        # Add trace for each backtest
        for i, backtest_id in enumerate(backtest_ids):
            model_name = metrics[backtest_id].get("model_name", "Unknown")
            symbol = metrics[backtest_id].get("symbol", "Unknown")
            
            values = [normalized_metrics[metric][i] for metric in key_metrics]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=f"{model_name} - {symbol}"
            ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Metrics Comparison",
            height=600,
            width=600
        )
        
        # Return plot as JSON
        return {
            "plot_type": "comparison_metrics",
            "plotly_json": fig.to_json()
        }
    
    def generate_correlation_matrix(self, model_names: List[str]) -> Dict[str, Any]:
        """
        Generate a correlation matrix of model predictions.
        
        Args:
            model_names: List of model names to compare
            
        Returns:
            Dictionary with plot data
        """
        # Check if all models exist
        for model_name in model_names:
            if model_name not in self.learner.models:
                raise ValueError(f"Model {model_name} not found")
        
        # Get all backtests for these models
        backtest_results = []
        backtest_ids = []
        
        for result_id, result in self.backtester.backtest_results.items():
            model_info = result.get("model_info", {})
            model_name = model_info.get("model_name")
            
            if model_name in model_names and "predictions" in result:
                backtest_results.append(result)
                backtest_ids.append(result_id)
        
        if not backtest_results:
            return {"error": "No backtest results found for these models"}
        
        # Create correlation matrix
        predictions_dict = {}
        
        for result in backtest_results:
            model_info = result.get("model_info", {})
            model_name = model_info.get("model_name")
            symbol = result.get("symbol")
            predictions = result.get("predictions", [])
            
            key = f"{model_name} - {symbol}"
            predictions_dict[key] = predictions
        
        # Check if we have predictions of the same length
        lengths = [len(preds) for preds in predictions_dict.values()]
        
        if len(set(lengths)) > 1:
            # Take the minimum length
            min_length = min(lengths)
            for key in predictions_dict:
                predictions_dict[key] = predictions_dict[key][:min_length]
        
        # Create a DataFrame with predictions
        df = pd.DataFrame(predictions_dict)
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create plotly figure
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="Viridis",
            zmin=-1,
            zmax=1
        ))
        
        # Add annotations
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=f"{corr_matrix.iloc[i, j]:.2f}",
                    showarrow=False,
                    font=dict(color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
                )
        
        # Update layout
        fig.update_layout(
            title="Model Predictions Correlation Matrix",
            height=600,
            width=800
        )
        
        # Return plot as JSON
        return {
            "plot_type": "correlation_matrix",
            "plotly_json": fig.to_json()
        } 