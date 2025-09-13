#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Visualization Dashboard for Trading Strategies

This module provides interactive visualizations for model performance,
feature importance, and trade analysis to support model interpretability.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import pickle

# Try to import plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Try to import shap for model interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

class ModelDashboard:
    """
    Dashboard for visualizing model performance, feature importance,
    and trade analysis to support model interpretability.
    
    This class provides:
    1. Performance visualizations by regime and timeframe
    2. Feature importance visualizations
    3. Trade analysis visualizations
    4. SHAP visualizations for feature contributions
    5. Model comparison visualizations
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the model dashboard.
        
        Args:
            params: Configuration parameters
        """
        self.params = params
        self.trade_analyzer = None
        self.model_trainer = None
        self.feature_engineering = None
        
        # Setup output directory
        self.output_dir = self.params.get('output_dir', './output/visualizations')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style for static plots
        plt.style.use('seaborn-whitegrid')
        
    def connect_components(self, trade_analyzer=None, model_trainer=None, feature_engineering=None):
        """
        Connect the dashboard to analysis components.
        
        Args:
            trade_analyzer: TradeAnalyzer instance
            model_trainer: ModelTrainer instance
            feature_engineering: FeatureEngineering instance
        """
        self.trade_analyzer = trade_analyzer
        self.model_trainer = model_trainer
        self.feature_engineering = feature_engineering
    
    def plot_model_performance(self, save_path: str = None, 
                              interactive: bool = False) -> Optional[Union[plt.Figure, Any]]:
        """
        Plot model performance metrics.
        
        Args:
            save_path: Optional path to save the plot
            interactive: Whether to use interactive plotly visualization
            
        Returns:
            Figure object or None
        """
        if self.trade_analyzer is None:
            print("TradeAnalyzer not connected. Call connect_components() first.")
            return None
            
        # Get performance summaries
        performance = self.trade_analyzer.get_performance_summary()
        regimes_performance = self.trade_analyzer.compare_regimes()
        
        if 'error' in performance:
            print(f"Error: {performance['error']}")
            return None
            
        if interactive and PLOTLY_AVAILABLE:
            return self._plot_performance_interactive(performance, regimes_performance, save_path)
        else:
            return self._plot_performance_static(performance, regimes_performance, save_path)
    
    def _plot_performance_static(self, performance: Dict[str, Dict[str, Any]], 
                               regimes_performance: Dict[str, Dict[str, Any]],
                               save_path: str = None) -> plt.Figure:
        """Create static performance visualization."""
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # Plot 1: Performance by timeframe
        ax1 = fig.add_subplot(gs[0, 0])
        timeframes = list(performance.keys())
        timeframes.sort(key=lambda x: {'week': 0, 'month': 1, 'quarter': 2, 'year': 3, 'all': 4}.get(x, 5))
        
        accuracy = [performance[tf]['accuracy'] * 100 if performance[tf]['accuracy'] is not None else 0 
                  for tf in timeframes]
        win_rate = [performance[tf]['win_rate'] * 100 if performance[tf]['win_rate'] is not None else 0 
                  for tf in timeframes]
        
        ax1.bar(timeframes, accuracy, alpha=0.7, label='Accuracy (%)')
        ax1.bar(timeframes, win_rate, alpha=0.7, label='Win Rate (%)')
        ax1.set_title('Performance by Timeframe')
        ax1.set_ylabel('Percentage (%)')
        ax1.legend()
        
        # Plot 2: Performance by regime
        ax2 = fig.add_subplot(gs[0, 1])
        regimes = list(regimes_performance.keys())
        
        total_trades = [regimes_performance[r]['total_trades'] for r in regimes]
        profit_factor = [min(regimes_performance[r]['profit_factor'], 5) if regimes_performance[r]['profit_factor'] is not None else 0 
                       for r in regimes]
        
        ax2.bar(regimes, total_trades, alpha=0.7, label='Total Trades')
        
        ax3 = ax2.twinx()
        ax3.plot(regimes, profit_factor, 'ro-', label='Profit Factor (capped at 5)')
        
        ax2.set_title('Trades and Profit Factor by Regime')
        ax2.set_ylabel('Number of Trades')
        ax3.set_ylabel('Profit Factor')
        ax2.legend(loc='upper left')
        ax3.legend(loc='upper right')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 3: PnL Over Time
        ax4 = fig.add_subplot(gs[1, :])
        
        # Process trade history to create cumulative PnL
        if self.trade_analyzer.trades_history:
            trades_df = pd.DataFrame([{
                'timestamp': t['timestamp'],
                'pnl': t.get('pnl', 0),
                'regime': t['regime']
            } for t in self.trade_analyzer.trades_history if 'pnl' in t])
            
            if not trades_df.empty:
                trades_df = trades_df.sort_values('timestamp')
                trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
                
                # Plot cumulative PnL
                for regime in trades_df['regime'].unique():
                    regime_df = trades_df[trades_df['regime'] == regime]
                    ax4.plot(regime_df['timestamp'], regime_df['cumulative_pnl'], 
                            label=f"{regime} Regime", marker='o', markersize=5, alpha=0.7)
                
                # Plot overall cumulative PnL
                ax4.plot(trades_df['timestamp'], trades_df['cumulative_pnl'], 
                        label='Overall', linewidth=2, color='black')
                
                ax4.set_title('Cumulative P&L Over Time')
                ax4.set_ylabel('Cumulative P&L')
                ax4.set_xlabel('Date')
                ax4.legend()
                
                # Format x-axis dates
                ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def _plot_performance_interactive(self, performance: Dict[str, Dict[str, Any]], 
                                    regimes_performance: Dict[str, Dict[str, Any]],
                                    save_path: str = None) -> Any:
        """Create interactive performance visualization."""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Install with 'pip install plotly'.")
            return self._plot_performance_static(performance, regimes_performance, save_path)
            
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{}, {}], [{"colspan": 2}, None]],
            subplot_titles=('Performance by Timeframe', 'Performance by Regime', 'Cumulative P&L Over Time')
        )
        
        # Plot 1: Performance by timeframe
        timeframes = list(performance.keys())
        timeframes.sort(key=lambda x: {'week': 0, 'month': 1, 'quarter': 2, 'year': 3, 'all': 4}.get(x, 5))
        
        accuracy = [performance[tf]['accuracy'] * 100 if performance[tf]['accuracy'] is not None else 0 
                  for tf in timeframes]
        win_rate = [performance[tf]['win_rate'] * 100 if performance[tf]['win_rate'] is not None else 0 
                  for tf in timeframes]
        
        fig.add_trace(
            go.Bar(x=timeframes, y=accuracy, name='Accuracy (%)', marker_color='blue', opacity=0.7),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=timeframes, y=win_rate, name='Win Rate (%)', marker_color='green', opacity=0.7),
            row=1, col=1
        )
        
        # Plot 2: Performance by regime
        regimes = list(regimes_performance.keys())
        
        total_trades = [regimes_performance[r]['total_trades'] for r in regimes]
        profit_factor = [min(regimes_performance[r]['profit_factor'], 5) if regimes_performance[r]['profit_factor'] is not None else 0 
                       for r in regimes]
        
        fig.add_trace(
            go.Bar(x=regimes, y=total_trades, name='Total Trades', marker_color='purple', opacity=0.7),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=regimes, y=profit_factor, name='Profit Factor', mode='lines+markers',
                     marker=dict(color='red', size=8)),
            row=1, col=2
        )
        
        # Plot 3: PnL Over Time
        if self.trade_analyzer.trades_history:
            trades_df = pd.DataFrame([{
                'timestamp': t['timestamp'],
                'pnl': t.get('pnl', 0),
                'regime': t['regime']
            } for t in self.trade_analyzer.trades_history if 'pnl' in t])
            
            if not trades_df.empty:
                trades_df = trades_df.sort_values('timestamp')
                trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
                
                # Plot overall cumulative PnL
                fig.add_trace(
                    go.Scatter(x=trades_df['timestamp'], y=trades_df['cumulative_pnl'],
                             name='Overall P&L', mode='lines', line=dict(color='black', width=3)),
                    row=2, col=1
                )
                
                # Plot by regime
                for regime in trades_df['regime'].unique():
                    regime_df = trades_df[trades_df['regime'] == regime]
                    fig.add_trace(
                        go.Scatter(x=regime_df['timestamp'], y=regime_df['cumulative_pnl'],
                                 name=f"{regime} Regime", mode='lines+markers', opacity=0.7),
                        row=2, col=1
                    )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            title='Trading Model Performance Dashboard',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def plot_feature_importance(self, model_name: str = 'default', 
                               regime: str = None, save_path: str = None,
                               interactive: bool = False) -> Optional[Union[plt.Figure, Any]]:
        """
        Plot feature importance for a specific model.
        
        Args:
            model_name: Name of the model
            regime: Optional market regime
            save_path: Optional path to save the plot
            interactive: Whether to use interactive plotly visualization
            
        Returns:
            Figure object or None
        """
        if self.model_trainer is None:
            print("ModelTrainer not connected. Call connect_components() first.")
            return None
        
        # Get model key
        model_key = f"{model_name}_{regime}" if regime else model_name
        
        # Get feature importance
        try:
            feature_importance = self.model_trainer.get_top_features(model_name, regime, top_n=20)
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            return None
            
        if interactive and PLOTLY_AVAILABLE:
            return self._plot_feature_importance_interactive(feature_importance, model_key, save_path)
        else:
            return self._plot_feature_importance_static(feature_importance, model_key, save_path)
    
    def _plot_feature_importance_static(self, feature_importance: Dict[str, float], 
                                      model_key: str, save_path: str = None) -> plt.Figure:
        """Create static feature importance visualization."""
        # Sort by importance
        sorted_importance = {k: v for k, v in sorted(
            feature_importance.items(), key=lambda item: item[1], reverse=False
        )}
        
        features = list(sorted_importance.keys())
        importance = list(sorted_importance.values())
        
        fig, ax = plt.subplots(figsize=(10, 12))
        
        # Create horizontal bar chart
        bars = ax.barh(features, importance, color='skyblue')
        
        # Add values on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{width:.4f}', ha='left', va='center')
        
        ax.set_title(f'Feature Importance - {model_key}')
        ax.set_xlabel('Importance')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def _plot_feature_importance_interactive(self, feature_importance: Dict[str, float], 
                                           model_key: str, save_path: str = None) -> Any:
        """Create interactive feature importance visualization."""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Install with 'pip install plotly'.")
            return self._plot_feature_importance_static(feature_importance, model_key, save_path)
        
        # Sort by importance
        sorted_importance = {k: v for k, v in sorted(
            feature_importance.items(), key=lambda item: item[1], reverse=False
        )}
        
        features = list(sorted_importance.keys())
        importance = list(sorted_importance.values())
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            y=features,
            x=importance,
            orientation='h',
            marker=dict(
                color=importance,
                colorscale='Viridis'
            ),
            text=[f'{x:.4f}' for x in importance],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f'Feature Importance - {model_key}',
            xaxis_title='Importance',
            height=max(600, len(features) * 25),
            width=900
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def plot_feature_stability(self, save_path: str = None,
                              interactive: bool = False) -> Optional[Union[plt.Figure, Any]]:
        """
        Plot feature stability over time.
        
        Args:
            save_path: Optional path to save the plot
            interactive: Whether to use interactive plotly visualization
            
        Returns:
            Figure object or None
        """
        if self.trade_analyzer is None:
            print("TradeAnalyzer not connected. Call connect_components() first.")
            return None
        
        # Get feature stability data
        stability_data = self.trade_analyzer.analyze_feature_stability(top_n=15)
        
        if 'error' in stability_data:
            print(f"Error: {stability_data['error']}")
            return None
            
        feature_trends = self.trade_analyzer.get_feature_trends(top_n=10)
        
        if 'error' in feature_trends:
            print(f"Error: {feature_trends['error']}")
            return None
            
        if interactive and PLOTLY_AVAILABLE:
            return self._plot_feature_stability_interactive(stability_data, feature_trends, save_path)
        else:
            return self._plot_feature_stability_static(stability_data, feature_trends, save_path)
    
    def _plot_feature_stability_static(self, stability_data: Dict[str, Any],
                                     feature_trends: Dict[str, List[Dict[str, Any]]],
                                     save_path: str = None) -> plt.Figure:
        """Create static feature stability visualization."""
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(2, 2, figure=fig)
        
        # Plot 1: Top features by average importance
        ax1 = fig.add_subplot(gs[0, 0])
        
        features = list(stability_data['top_features'].keys())[:10]  # Top 10
        avg_importance = [stability_data['top_features'][f]['avg_importance'] for f in features]
        
        y_pos = np.arange(len(features))
        ax1.barh(y_pos, avg_importance, align='center', color='skyblue')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(features)
        ax1.invert_yaxis()  # Labels read top-to-bottom
        ax1.set_title('Top 10 Features by Importance')
        ax1.set_xlabel('Average Importance')
        
        # Plot 2: Feature stability
        ax2 = fig.add_subplot(gs[0, 1])
        
        stability_scores = [stability_data['stability_scores'][f] for f in features]
        importance_variance = [stability_data['top_features'][f]['importance_variance'] for f in features]
        
        # Normalize for better visualization
        max_stability = max(stability_scores) if stability_scores else 1
        normalized_stability = [s / max_stability * 100 for s in stability_scores]
        
        ax2.barh(y_pos, normalized_stability, align='center', color='lightgreen')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(features)
        ax2.invert_yaxis()  # Labels read top-to-bottom
        ax2.set_title('Feature Stability (Normalized)')
        ax2.set_xlabel('Stability Score (Normalized)')
        
        # Add variance as text
        for i, v in enumerate(importance_variance):
            ax2.text(normalized_stability[i] + 1, i, f'Var: {v:.4f}', va='center')
        
        # Plot 3: Feature importance over time for top 5 features
        ax3 = fig.add_subplot(gs[1, :])
        
        # Get top 5 features
        top5_features = features[:5]
        
        for feature in top5_features:
            if feature in feature_trends:
                trend_data = feature_trends[feature]
                timestamps = [entry['timestamp'] for entry in trend_data]
                importances = [entry['importance'] for entry in trend_data]
                
                if timestamps:  # Only plot if we have data
                    ax3.plot(timestamps, importances, marker='o', linewidth=2, 
                           alpha=0.7, label=feature)
        
        ax3.set_title('Top 5 Feature Importance Over Time')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Importance')
        ax3.legend()
        
        # Format x-axis dates
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def _plot_feature_stability_interactive(self, stability_data: Dict[str, Any],
                                          feature_trends: Dict[str, List[Dict[str, Any]]],
                                          save_path: str = None) -> Any:
        """Create interactive feature stability visualization."""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Install with 'pip install plotly'.")
            return self._plot_feature_stability_static(stability_data, feature_trends, save_path)
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{}, {}], [{"colspan": 2}, None]],
            subplot_titles=('Top Features by Importance', 'Feature Stability', 
                         'Feature Importance Over Time')
        )
        
        # Plot 1: Top features by average importance
        features = list(stability_data['top_features'].keys())[:10]  # Top 10
        avg_importance = [stability_data['top_features'][f]['avg_importance'] for f in features]
        
        fig.add_trace(
            go.Bar(y=features, x=avg_importance, orientation='h', 
                 marker_color='skyblue', name='Avg Importance'),
            row=1, col=1
        )
        
        # Plot 2: Feature stability
        stability_scores = [stability_data['stability_scores'][f] for f in features]
        importance_variance = [stability_data['top_features'][f]['importance_variance'] for f in features]
        
        # Normalize for better visualization
        max_stability = max(stability_scores) if stability_scores else 1
        normalized_stability = [s / max_stability * 100 for s in stability_scores]
        
        fig.add_trace(
            go.Bar(y=features, x=normalized_stability, orientation='h',
                 marker_color='lightgreen', name='Stability (Normalized)',
                 text=[f'Var: {v:.4f}' for v in importance_variance],
                 textposition='outside'),
            row=1, col=2
        )
        
        # Plot 3: Feature importance over time for top 5 features
        top5_features = features[:5]
        
        for feature in top5_features:
            if feature in feature_trends:
                trend_data = feature_trends[feature]
                timestamps = [entry['timestamp'] for entry in trend_data]
                importances = [entry['importance'] for entry in trend_data]
                
                if timestamps:  # Only plot if we have data
                    fig.add_trace(
                        go.Scatter(x=timestamps, y=importances, mode='lines+markers',
                                 name=feature),
                        row=2, col=1
                    )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            title='Feature Stability Analysis',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def plot_regime_feature_importance(self, model_name: str = 'default', 
                                     save_path: str = None,
                                     interactive: bool = False) -> Optional[Union[plt.Figure, Any]]:
        """
        Plot feature importance breakdown by market regime.
        
        Args:
            model_name: Name of the model
            save_path: Optional path to save the plot
            interactive: Whether to use interactive plotly visualization
            
        Returns:
            Figure object or None
        """
        if self.trade_analyzer is None or self.model_trainer is None:
            print("TradeAnalyzer and ModelTrainer must be connected. Call connect_components() first.")
            return None
        
        # Get feature usage by regime
        feature_usage = self.trade_analyzer.feature_usage_history
        
        if not feature_usage:
            print("No feature usage data available.")
            return None
            
        # Get regimes
        regimes = set()
        for feature, entries in feature_usage.items():
            regimes.update([entry['regime'] for entry in entries])
        
        # Remove "unknown" regime
        if "unknown" in regimes:
            regimes.remove("unknown")
            
        if not regimes:
            print("No regime data available.")
            return None
            
        # Get top 10 features from model
        try:
            if len(self.model_trainer.models) > 0:
                top_features = list(self.model_trainer.get_top_features(model_name, top_n=10).keys())
            else:
                # Fallback: use most frequent features from trade analyzer
                feature_counts = {f: len(entries) for f, entries in feature_usage.items()}
                top_features = sorted(feature_counts.keys(), key=lambda x: feature_counts[x], reverse=True)[:10]
        except Exception as e:
            print(f"Error getting top features: {str(e)}")
            # Fallback: use most frequent features from trade analyzer
            feature_counts = {f: len(entries) for f, entries in feature_usage.items()}
            top_features = sorted(feature_counts.keys(), key=lambda x: feature_counts[x], reverse=True)[:10]
            
        # Calculate average importance by regime for top features
        regime_importance = {regime: {feature: [] for feature in top_features} for regime in regimes}
        
        for feature in top_features:
            if feature in feature_usage:
                for entry in feature_usage[feature]:
                    regime = entry['regime']
                    if regime in regimes:
                        regime_importance[regime][feature].append(entry['importance'])
        
        # Calculate averages
        avg_importance = {regime: {feature: np.mean(values) if values else 0 
                                 for feature, values in features.items()}
                        for regime, features in regime_importance.items()}
        
        if interactive and PLOTLY_AVAILABLE:
            return self._plot_regime_importance_interactive(avg_importance, top_features, save_path)
        else:
            return self._plot_regime_importance_static(avg_importance, top_features, save_path)
    
    def _plot_regime_importance_static(self, avg_importance: Dict[str, Dict[str, float]],
                                     top_features: List[str],
                                     save_path: str = None) -> plt.Figure:
        """Create static regime feature importance visualization."""
        regimes = list(avg_importance.keys())
        n_regimes = len(regimes)
        
        # Create figure
        fig, axes = plt.subplots(n_regimes, 1, figsize=(12, 5 * n_regimes))
        
        # Handle case of single regime (axes not being a list)
        if n_regimes == 1:
            axes = [axes]
        
        # Plot importance for each regime
        for i, regime in enumerate(regimes):
            # Sort features by importance for this regime
            sorted_features = sorted(top_features, 
                                   key=lambda x: avg_importance[regime][x], 
                                   reverse=True)
            
            importance_values = [avg_importance[regime][f] for f in sorted_features]
            
            # Create horizontal bar chart
            bars = axes[i].barh(sorted_features, importance_values, 
                             color=plt.cm.tab10(i % 10))
            
            # Add values on bars
            for j, bar in enumerate(bars):
                width = bar.get_width()
                axes[i].text(width + 0.01, bar.get_y() + bar.get_height()/2,
                          f'{width:.4f}', ha='left', va='center')
            
            axes[i].set_title(f'Feature Importance - {regime} Regime')
            axes[i].set_xlabel('Importance')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def _plot_regime_importance_interactive(self, avg_importance: Dict[str, Dict[str, float]],
                                          top_features: List[str],
                                          save_path: str = None) -> Any:
        """Create interactive regime feature importance visualization."""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Install with 'pip install plotly'.")
            return self._plot_regime_importance_static(avg_importance, top_features, save_path)
        
        regimes = list(avg_importance.keys())
        n_regimes = len(regimes)
        
        # Create subplot figure
        fig = make_subplots(
            rows=n_regimes, cols=1,
            subplot_titles=[f'Feature Importance - {regime} Regime' for regime in regimes]
        )
        
        # Plot importance for each regime
        for i, regime in enumerate(regimes):
            # Sort features by importance for this regime
            sorted_features = sorted(top_features, 
                                   key=lambda x: avg_importance[regime][x], 
                                   reverse=True)
            
            importance_values = [avg_importance[regime][f] for f in sorted_features]
            
            fig.add_trace(
                go.Bar(y=sorted_features, x=importance_values, orientation='h',
                     marker_color=px.colors.qualitative.Plotly[i % 10],
                     name=regime),
                row=i+1, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=300 * n_regimes,
            width=900,
            title='Feature Importance by Market Regime',
            showlegend=False
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def plot_trade_explanations(self, n_trades: int = 5, save_path: str = None) -> Optional[plt.Figure]:
        """
        Plot feature contributions for sample trades.
        
        Args:
            n_trades: Number of recent trades to include
            save_path: Optional path to save the plot
            
        Returns:
            Figure object or None
        """
        if self.trade_analyzer is None:
            print("TradeAnalyzer not connected. Call connect_components() first.")
            return None
        
        # Get recent trades
        trades = self.trade_analyzer.trades_history[-n_trades:] if len(self.trade_analyzer.trades_history) >= n_trades else self.trade_analyzer.trades_history
        
        if not trades:
            print("No trade history available.")
            return None
            
        # Setup figure
        fig, axes = plt.subplots(len(trades), 1, figsize=(12, 5 * len(trades)))
        
        # Handle case of single trade
        if len(trades) == 1:
            axes = [axes]
        
        # Plot feature contributions for each trade
        for i, trade in enumerate(trades):
            if 'top_features' in trade:
                # Sort features by absolute importance
                sorted_features = sorted(trade['top_features'].items(), 
                                       key=lambda x: abs(x[1]), 
                                       reverse=True)
                
                features = [item[0] for item in sorted_features]
                importances = [item[1] for item in sorted_features]
                
                # Color bars based on direction of influence
                colors = ['green' if imp > 0 else 'red' for imp in importances]
                
                # Create horizontal bar chart
                axes[i].barh(features, importances, color=colors)
                
                # Add trade details as title
                trade_result = "Correct" if trade.get('correct', False) else "Incorrect"
                title = (f"Trade on {trade['timestamp']} - {trade_result}\n"
                       f"Predicted: {trade['prediction']}, Actual: {trade.get('actual_outcome', 'N/A')}, "
                       f"PnL: {trade.get('pnl', 'N/A')}")
                
                axes[i].set_title(title)
                axes[i].axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # Add feature values if available
                if 'feature_values' in trade:
                    for j, feature in enumerate(features):
                        if feature in trade['feature_values']:
                            value = trade['feature_values'][feature]
                            axes[i].text(0, j, f" = {value:.4f}" if isinstance(value, float) else f" = {value}", 
                                      va='center')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def plot_shap_summary(self, model_name: str = 'default', regime: str = None,
                        X_sample: pd.DataFrame = None, save_path: str = None) -> Optional[plt.Figure]:
        """
        Plot SHAP value summary for feature importance.
        
        Args:
            model_name: Name of the model
            regime: Optional market regime
            X_sample: Sample data to compute SHAP values on
            save_path: Optional path to save the plot
            
        Returns:
            Figure object or None
        """
        if self.model_trainer is None:
            print("ModelTrainer not connected. Call connect_components() first.")
            return None
            
        if not SHAP_AVAILABLE:
            print("SHAP not available. Install with 'pip install shap'.")
            return None
            
        # Get model key
        model_key = f"{model_name}_{regime}" if regime else model_name
        
        # Check if SHAP values are already calculated
        if model_key in self.model_trainer.shap_values:
            # If sample data not provided, use existing SHAP values
            shap_data = self.model_trainer.shap_values[model_key]
            
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Plot SHAP summary
            shap.summary_plot(shap_data['values'], feature_names=shap_data['feature_names'], show=False)
            
            plt.title(f'SHAP Feature Importance - {model_key}')
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path)
                
            return plt.gcf()
        else:
            # Need sample data and model
            if X_sample is None or model_key not in self.model_trainer.models:
                print(f"Need sample data and model to compute SHAP values for {model_key}.")
                return None
                
            # Get model
            model = self.model_trainer.models[model_key]
            
            # Create explainer
            if hasattr(model, 'feature_importances_'):  # Tree-based model
                explainer = shap.TreeExplainer(model)
            else:  # Fallback to kernel explainer
                explainer = shap.KernelExplainer(model.predict, X_sample.iloc[:50])
                
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Plot SHAP summary
            shap.summary_plot(shap_values, X_sample, show=False)
            
            plt.title(f'SHAP Feature Importance - {model_key}')
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path)
                
            return plt.gcf()
    
    def create_dashboard(self, output_dir: str = None, 
                       interactive: bool = True) -> Dict[str, str]:
        """
        Create a complete dashboard with all visualizations.
        
        Args:
            output_dir: Directory to save visualizations
            interactive: Whether to use interactive plotly visualizations
            
        Returns:
            Dictionary mapping visualization types to file paths
        """
        # Use default output dir if not specified
        if output_dir is None:
            output_dir = self.output_dir
            
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        # Model performance
        try:
            perf_path = os.path.join(output_dir, 'model_performance.html' if interactive else 'model_performance.png')
            self.plot_model_performance(save_path=perf_path, interactive=interactive)
            results['model_performance'] = perf_path
        except Exception as e:
            print(f"Error creating model performance visualization: {str(e)}")
        
        # Feature importance
        try:
            fi_path = os.path.join(output_dir, 'feature_importance.html' if interactive else 'feature_importance.png')
            if self.model_trainer and hasattr(self.model_trainer, 'models') and self.model_trainer.models:
                model_name = list(self.model_trainer.models.keys())[0].split('_')[0]
                self.plot_feature_importance(model_name=model_name, save_path=fi_path, interactive=interactive)
                results['feature_importance'] = fi_path
        except Exception as e:
            print(f"Error creating feature importance visualization: {str(e)}")
        
        # Feature stability
        try:
            fs_path = os.path.join(output_dir, 'feature_stability.html' if interactive else 'feature_stability.png')
            self.plot_feature_stability(save_path=fs_path, interactive=interactive)
            results['feature_stability'] = fs_path
        except Exception as e:
            print(f"Error creating feature stability visualization: {str(e)}")
        
        # Regime feature importance
        try:
            rfi_path = os.path.join(output_dir, 'regime_feature_importance.html' if interactive else 'regime_feature_importance.png')
            if self.model_trainer and hasattr(self.model_trainer, 'models') and self.model_trainer.models:
                model_name = list(self.model_trainer.models.keys())[0].split('_')[0]
                self.plot_regime_feature_importance(model_name=model_name, save_path=rfi_path, interactive=interactive)
                results['regime_feature_importance'] = rfi_path
        except Exception as e:
            print(f"Error creating regime feature importance visualization: {str(e)}")
        
        # Trade explanations
        try:
            te_path = os.path.join(output_dir, 'trade_explanations.png')
            self.plot_trade_explanations(n_trades=5, save_path=te_path)
            results['trade_explanations'] = te_path
        except Exception as e:
            print(f"Error creating trade explanations visualization: {str(e)}")
        
        # Create index.html to link all visualizations
        if interactive:
            try:
                index_path = os.path.join(output_dir, 'index.html')
                with open(index_path, 'w') as f:
                    f.write('<html><head><title>Trading Model Dashboard</title></head><body>\n')
                    f.write('<h1>Trading Model Dashboard</h1>\n')
                    
                    for viz_type, path in results.items():
                        if path.endswith('.html'):
                            filename = os.path.basename(path)
                            f.write(f'<h2>{viz_type.replace("_", " ").title()}</h2>\n')
                            f.write(f'<iframe src="{filename}" width="100%" height="600px"></iframe>\n')
                            f.write(f'<p><a href="{filename}" target="_blank">Open in new window</a></p>\n')
                    
                    f.write('</body></html>')
                    
                results['index'] = index_path
            except Exception as e:
                print(f"Error creating dashboard index: {str(e)}")
        
        return results
    
    def get_model_summary(self, model_name: str = 'default', 
                        regime: str = None) -> Dict[str, Any]:
        """
        Get a comprehensive summary of model performance and features.
        
        Args:
            model_name: Name of the model
            regime: Optional market regime
            
        Returns:
            Dictionary with model summary information
        """
        summary = {
            'model_name': model_name,
            'regime': regime,
            'timestamp': datetime.now().isoformat()
        }
        
        # Get model trainer info if available
        if self.model_trainer is not None:
            try:
                model_key = f"{model_name}_{regime}" if regime else model_name
                
                if model_key in self.model_trainer.models:
                    # Add model type
                    model = self.model_trainer.models[model_key]
                    summary['model_type'] = type(model).__name__
                    
                    # Add feature importance
                    if model_key in self.model_trainer.feature_importance:
                        top_features = self.model_trainer.get_top_features(model_name, regime, top_n=10)
                        summary['top_features'] = top_features
                    
                    # Add performance metrics if available
                    if model_name in self.model_trainer.performance_metrics:
                        metrics = self.model_trainer.performance_metrics[model_name]
                        summary['cv_performance'] = {
                            'mean_train_score': metrics.get('mean_train_score'),
                            'mean_test_score': metrics.get('mean_test_score'),
                            'std_test_score': metrics.get('std_test_score')
                        }
            except Exception as e:
                summary['model_trainer_error'] = str(e)
        
        # Get trade analyzer info if available
        if self.trade_analyzer is not None:
            try:
                # Add performance by timeframe
                performance = self.trade_analyzer.get_performance_summary()
                if 'error' not in performance:
                    summary['performance'] = performance
                
                # Add regime performance if applicable
                if regime:
                    regime_perf = self.trade_analyzer.analyze_model_performance(regime=regime)
                    summary['regime_performance'] = regime_perf
                
                # Add feature stability info
                stability = self.trade_analyzer.analyze_feature_stability(top_n=10)
                if 'error' not in stability:
                    summary['feature_stability'] = {
                        'stability_scores': stability['stability_scores']
                    }
            except Exception as e:
                summary['trade_analyzer_error'] = str(e)
                
        return summary 