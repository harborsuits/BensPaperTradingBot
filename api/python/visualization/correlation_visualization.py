#!/usr/bin/env python3
"""
Correlation Visualization

This module provides visualization tools for correlation analysis, including:
- Correlation heatmaps
- Time series views of correlation changes
- Regime classification visualization
- Allocation adjustment visualization

These visualizations can be used in the trading UI or exported as standalone charts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import io
import base64

# Import correlation components
from trading_bot.autonomous.correlation_monitor import get_correlation_monitor
from trading_bot.autonomous.correlation_regime_detector import get_correlation_regime_detector, RegimeType


class CorrelationVisualizer:
    """
    Provides visualization tools for correlation analysis data.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        # Get correlation components
        self.monitor = get_correlation_monitor()
        self.regime_detector = get_correlation_regime_detector()
        
        # Set default plot style
        plt.style.use('ggplot')
        self.figsize = (10, 8)
        self.color_map = 'RdBu_r'  # Red-Blue diverging colormap
        
        # Set color scheme for regimes
        self.regime_colors = {
            RegimeType.STABLE.value: 'green',
            RegimeType.VOLATILE.value: 'red',
            RegimeType.TRANSITIONAL.value: 'orange',
            RegimeType.UNKNOWN.value: 'gray'
        }
    
    def generate_correlation_heatmap(self, 
                                    title: Optional[str] = None,
                                    include_values: bool = True,
                                    as_base64: bool = False) -> Any:
        """
        Generate correlation heatmap visualization.
        
        Args:
            title: Custom title for the heatmap
            include_values: Whether to include correlation values in cells
            as_base64: If True, return base64 encoded image
            
        Returns:
            Figure, base64 string, or None if no data
        """
        # Get correlation data
        heatmap_data = self.monitor.get_correlation_heatmap_data()
        strategies = heatmap_data.get('strategies', [])
        
        if not strategies:
            return None
        
        # Create DataFrame from matrix
        matrix = np.array(heatmap_data.get('matrix', []))
        if matrix.size == 0:
            return None
            
        df = pd.DataFrame(matrix, index=strategies, columns=strategies)
        
        # Create figure
        plt.figure(figsize=self.figsize)
        
        # Create heatmap
        mask = np.zeros_like(df, dtype=bool)
        mask[np.triu_indices_from(mask)] = True  # Mask upper triangle
        
        # Plot heatmap
        if include_values:
            sns.heatmap(df, mask=mask, cmap=self.color_map, 
                        annot=True, fmt=".2f", center=0,
                        linewidths=0.5, cbar_kws={"shrink": 0.8})
        else:
            sns.heatmap(df, mask=mask, cmap=self.color_map, 
                        center=0, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        # Set title
        if title:
            plt.title(title)
        else:
            current_regime = self.regime_detector.get_current_regime_info()
            regime_name = current_regime.get('regime', 'unknown')
            plt.title(f"Strategy Correlation Matrix (Regime: {regime_name.capitalize()})")
        
        plt.tight_layout()
        
        if as_base64:
            # Convert to base64 for embedding in web pages
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            return img_str
        else:
            return plt.gcf()
    
    def generate_correlation_time_series(self,
                                         strategy_pairs: List[Tuple[str, str]],
                                         days: int = 30,
                                         as_base64: bool = False) -> Any:
        """
        Generate time series visualization of correlation changes.
        
        Args:
            strategy_pairs: List of (strategy1, strategy2) pairs to plot
            days: Number of days to show
            as_base64: If True, return base64 encoded image
            
        Returns:
            Figure, base64 string, or None if no data
        """
        if not strategy_pairs:
            return None
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Set colors for up to 10 pairs
        colors = plt.cm.tab10.colors
        
        # Plot each pair
        for i, (strategy1, strategy2) in enumerate(strategy_pairs):
            # Get rolling correlation
            corr_series = self.monitor.correlation_matrix.get_rolling_correlation(
                strategy1, strategy2, start_date, end_date
            )
            
            if corr_series.empty:
                continue
                
            # Plot correlation
            color = colors[i % len(colors)]
            plt.plot(corr_series.index, corr_series.values, 
                     label=f"{strategy1} - {strategy2}", color=color, linewidth=2)
        
        # Add horizontal lines at correlation thresholds
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axhline(y=self.monitor.config['high_correlation_threshold'], 
                    color='red', linestyle='--', alpha=0.5,
                    label=f"High Correlation Threshold ({self.monitor.config['high_correlation_threshold']})")
        plt.axhline(y=-self.monitor.config['high_correlation_threshold'], 
                    color='blue', linestyle='--', alpha=0.5,
                    label=f"High Negative Correlation (-{self.monitor.config['high_correlation_threshold']})")
        
        # Add regime changes if available
        regime_history = self.regime_detector.get_regime_history(limit=days)
        if regime_history:
            for regime_event in regime_history:
                timestamp = datetime.fromisoformat(regime_event['timestamp'])
                regime = regime_event['regime']
                
                # Only add vertical lines for regime changes
                if timestamp >= start_date:
                    plt.axvline(x=timestamp, color=self.regime_colors.get(regime, 'gray'), 
                                linestyle='-.', alpha=0.7)
        
        # Set labels and title
        plt.title(f"Strategy Pair Correlations (Last {days} Days)")
        plt.xlabel("Date")
        plt.ylabel("Correlation")
        plt.ylim(-1.05, 1.05)
        
        # Format x-axis to show dates nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days // 10)))
        plt.gcf().autofmt_xdate()
        
        # Add legend
        plt.legend(loc='best')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if as_base64:
            # Convert to base64 for embedding in web pages
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            return img_str
        else:
            return plt.gcf()
    
    def generate_regime_visualization(self, 
                                    days: int = 90,
                                    as_base64: bool = False) -> Any:
        """
        Generate visualization of market regime changes and eigenvalue trends.
        
        Args:
            days: Number of days to show
            as_base64: If True, return base64 encoded image
            
        Returns:
            Figure, base64 string, or None if no data
        """
        # Get regime history and eigenvalue trends
        regime_history = self.regime_detector.get_regime_history(limit=days)
        eigenvalue_trends = self.regime_detector.get_eigenvalue_trends()
        
        if not regime_history or not eigenvalue_trends['timestamps']:
            return None
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
        
        # Plot 1: Regime Timeline
        regimes = []
        timestamps = []
        colors = []
        
        for event in regime_history:
            regimes.append(event['regime'])
            timestamps.append(datetime.fromisoformat(event['timestamp']))
            colors.append(self.regime_colors.get(event['regime'], 'gray'))
        
        # Create regime timeline
        for i in range(len(timestamps)-1):
            ax1.axvspan(timestamps[i], timestamps[i+1], alpha=0.3, color=colors[i])
            
        # Add current regime
        if timestamps:
            ax1.axvspan(timestamps[-1], datetime.now(), alpha=0.3, color=colors[-1])
        
        # Add labels for regimes
        current_regime = None
        regime_start = None
        
        for i, (ts, regime) in enumerate(zip(timestamps, regimes)):
            if regime != current_regime:
                if current_regime is not None:
                    # Add label for the previous regime
                    midpoint = regime_start + (ts - regime_start) / 2
                    ax1.text(midpoint, 0.5, current_regime.capitalize(), 
                            ha='center', va='center', fontsize=10,
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
                
                current_regime = regime
                regime_start = ts
        
        # Add label for the last regime
        if current_regime is not None and regime_start is not None:
            midpoint = regime_start + (datetime.now() - regime_start) / 2
            ax1.text(midpoint, 0.5, current_regime.capitalize(), 
                    ha='center', va='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        ax1.set_title("Market Regime Timeline")
        ax1.set_ylim(0, 1)
        ax1.set_yticks([])
        
        # Add regime legend
        handles = [plt.Rectangle((0,0), 1, 1, color=self.regime_colors[regime], alpha=0.3) 
                  for regime in self.regime_colors]
        labels = [regime.capitalize() for regime in self.regime_colors]
        ax1.legend(handles, labels, loc='upper right', ncol=len(handles))
        
        # Plot 2: Eigenvalue Trends
        eigenvalue_times = [datetime.fromisoformat(ts) for ts in eigenvalue_trends['timestamps']]
        
        # Filter for the time range we want
        start_date = datetime.now() - timedelta(days=days)
        filtered_indices = [i for i, ts in enumerate(eigenvalue_times) if ts >= start_date]
        
        if filtered_indices:
            filtered_times = [eigenvalue_times[i] for i in filtered_indices]
            filtered_dispersion = [eigenvalue_trends['dispersion'][i] for i in filtered_indices]
            
            # Plot dispersion ratio
            ax2.plot(filtered_times, filtered_dispersion, 'b-', linewidth=2, label='Dispersion Ratio')
            
            # Plot top eigenvalues if available
            if eigenvalue_trends['eigenvalues']:
                top_eigenvalues = np.array([eigenvalue_trends['eigenvalues'][i] for i in filtered_indices])
                
                # Plot first eigenvalue
                if top_eigenvalues.shape[1] >= 1:
                    ax2.plot(filtered_times, [values[0] for values in top_eigenvalues], 
                            'r-', linewidth=1.5, label='1st Eigenvalue')
                
                # Plot second eigenvalue
                if top_eigenvalues.shape[1] >= 2:
                    ax2.plot(filtered_times, [values[1] for values in top_eigenvalues], 
                            'g-', linewidth=1.5, label='2nd Eigenvalue')
            
            # Add threshold line
            ax2.axhline(y=self.regime_detector.config['regime_change_threshold'], 
                      color='red', linestyle='--', alpha=0.5,
                      label=f"Regime Change Threshold ({self.regime_detector.config['regime_change_threshold']})")
        
        ax2.set_title("Correlation Structure Evolution")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Eigenvalue / Dispersion")
        ax2.legend(loc='best')
        
        # Format x-axis to show dates nicely
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days // 10)))
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        if as_base64:
            # Convert to base64 for embedding in web pages
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            return img_str
        else:
            return fig
    
    def generate_allocation_adjustment_visualization(self,
                                                   as_base64: bool = False) -> Any:
        """
        Generate visualization of allocation adjustments based on correlation.
        
        Args:
            as_base64: If True, return base64 encoded image
            
        Returns:
            Figure, base64 string, or None if no data
        """
        # Get allocation adjustments
        report = self.monitor.get_correlation_report()
        adjustments = report.get('allocation_adjustments', {})
        
        if not adjustments:
            return None
        
        # Create a DataFrame for plotting
        data = []
        for strategy_id, adj in adjustments.items():
            data.append({
                'strategy_id': strategy_id,
                'original': adj.get('original_allocation', 0),
                'adjusted': adj.get('new_allocation', 0),
                'factor': adj.get('adjustment_factor', 1.0),
                'timestamp': datetime.fromisoformat(adj.get('timestamp', datetime.now().isoformat()))
            })
        
        if not data:
            return None
            
        df = pd.DataFrame(data)
        
        # Sort by original allocation (descending)
        df = df.sort_values('original', ascending=False)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create bar chart
        x = np.arange(len(df))
        width = 0.35
        
        plt.bar(x - width/2, df['original'], width, label='Original Allocation', color='blue', alpha=0.7)
        plt.bar(x + width/2, df['adjusted'], width, label='Adjusted Allocation', color='red', alpha=0.7)
        
        # Add data labels
        for i, row in enumerate(df.itertuples()):
            plt.text(i - width/2, row.original + 0.5, f"{row.original:.1f}%", 
                    ha='center', va='bottom', fontsize=9)
            plt.text(i + width/2, row.adjusted + 0.5, f"{row.adjusted:.1f}%", 
                    ha='center', va='bottom', fontsize=9)
            
            # Add factor labels
            plt.text(i, 0.5, f"{row.factor:.2f}x", 
                    ha='center', va='bottom', fontsize=8, color='black')
        
        # Add strategy labels
        plt.xticks(x, df['strategy_id'], rotation=45, ha='right')
        
        # Add title and labels
        current_regime = self.regime_detector.get_current_regime_info()
        regime_name = current_regime.get('regime', 'unknown')
        
        plt.title(f"Correlation-Based Allocation Adjustments (Regime: {regime_name.capitalize()})")
        plt.ylabel("Allocation (%)")
        plt.legend(loc='best')
        
        plt.tight_layout()
        
        if as_base64:
            # Convert to base64 for embedding in web pages
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            return img_str
        else:
            return plt.gcf()


# Singleton instance for global access
_correlation_visualizer = None

def get_correlation_visualizer() -> CorrelationVisualizer:
    """
    Get singleton instance of correlation visualizer.
    
    Returns:
        CorrelationVisualizer instance
    """
    global _correlation_visualizer
    
    if _correlation_visualizer is None:
        _correlation_visualizer = CorrelationVisualizer()
        
    return _correlation_visualizer


def generate_correlation_dashboard(output_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Generate a complete set of correlation visualizations.
    
    Args:
        output_dir: Directory to save visualizations, if None just return base64
        
    Returns:
        Dict mapping visualization names to base64 strings or file paths
    """
    visualizer = get_correlation_visualizer()
    
    # Generate all visualizations
    results = {}
    
    # Correlation heatmap
    heatmap = visualizer.generate_correlation_heatmap(as_base64=True)
    if heatmap:
        results['correlation_heatmap'] = heatmap
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, 'correlation_heatmap.png')
            with open(file_path, 'wb') as f:
                f.write(base64.b64decode(heatmap))
            results['correlation_heatmap_file'] = file_path
    
    # Get top correlated pairs
    monitor = get_correlation_monitor()
    report = monitor.get_correlation_report()
    pairs = [(p['strategy1'], p['strategy2']) for p in report.get('highly_correlated_pairs', [])]
    
    # If no highly correlated pairs, get a few random ones
    if not pairs and 'tracked_strategies' in report:
        strategies = report.get('tracked_strategies', [])
        if len(strategies) >= 2:
            # Take a few random pairs
            import random
            for _ in range(min(3, len(strategies))):
                s1, s2 = random.sample(strategies, 2)
                pairs.append((s1, s2))
    
    # Correlation time series
    if pairs:
        time_series = visualizer.generate_correlation_time_series(pairs, as_base64=True)
        if time_series:
            results['correlation_time_series'] = time_series
            
            if output_dir:
                file_path = os.path.join(output_dir, 'correlation_time_series.png')
                with open(file_path, 'wb') as f:
                    f.write(base64.b64decode(time_series))
                results['correlation_time_series_file'] = file_path
    
    # Regime visualization
    regime_viz = visualizer.generate_regime_visualization(as_base64=True)
    if regime_viz:
        results['regime_visualization'] = regime_viz
        
        if output_dir:
            file_path = os.path.join(output_dir, 'regime_visualization.png')
            with open(file_path, 'wb') as f:
                f.write(base64.b64decode(regime_viz))
            results['regime_visualization_file'] = file_path
    
    # Allocation adjustment visualization
    allocation_viz = visualizer.generate_allocation_adjustment_visualization(as_base64=True)
    if allocation_viz:
        results['allocation_visualization'] = allocation_viz
        
        if output_dir:
            file_path = os.path.join(output_dir, 'allocation_visualization.png')
            with open(file_path, 'wb') as f:
                f.write(base64.b64decode(allocation_viz))
            results['allocation_visualization_file'] = file_path
    
    return results


if __name__ == '__main__':
    # When run directly, generate dashboard and save to default location
    output_dir = os.path.join(os.path.expanduser("~"), ".trading_bot", "visualizations")
    results = generate_correlation_dashboard(output_dir)
    
    print(f"Generated {len(results)} visualizations in {output_dir}")
    for name, path in results.items():
        if name.endswith('_file'):
            print(f"- {name}: {path}")
