#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Performance Visualization Dashboard

This module creates visualizations to track strategy performance across different
market regimes, with historical tracking to show performance improvements over time.
It helps users understand how strategies perform under different market conditions
and how the ML regime detection improves strategy selection.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from trading_bot.strategies.strategy_template import MarketRegime
from trading_bot.strategies.forex.strategy_selector import ForexStrategySelector
from trading_bot.backtesting.performance_integration import PerformanceIntegration

# Default output directory for visualizations
VISUALIZATIONS_DIR = os.path.join(project_root, 'visualizations')


class PerformanceDashboard:
    """Generates performance visualizations for forex strategies."""
    
    def __init__(self, 
                 output_dir: Optional[str] = None,
                 perf_integration: Optional[PerformanceIntegration] = None):
        """
        Initialize the performance dashboard.
        
        Args:
            output_dir: Directory to save visualizations
            perf_integration: PerformanceIntegration instance or None to create new
        """
        # Set output directory
        self.output_dir = output_dir or VISUALIZATIONS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize or use existing performance integration
        self.perf_integration = perf_integration or PerformanceIntegration()
        
        # Set plot style
        self._setup_plot_style()
    
    def _setup_plot_style(self) -> None:
        """Set up matplotlib/seaborn plot style."""
        sns.set(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
        # Create a custom palette for market regimes
        self.regime_colors = {
            'TRENDING_UP': '#1e88e5',       # Blue
            'TRENDING_DOWN': '#e53935',     # Red
            'RANGING': '#8e24aa',           # Purple
            'VOLATILE_BREAKOUT': '#ffb300', # Orange
            'VOLATILE_REVERSAL': '#43a047', # Green
            'CHOPPY': '#757575',            # Gray
            'UNKNOWN': '#212121'            # Dark Gray
        }
    
    def generate_strategy_heatmap(self, 
                                 snapshot_date: Optional[str] = None,
                                 save_path: Optional[str] = None) -> str:
        """
        Generate a heatmap showing strategy performance across market regimes.
        
        Args:
            snapshot_date: Date string for performance snapshot (YYYY-MM-DD)
            save_path: Path to save the visualization or None for default
            
        Returns:
            Path to the saved visualization
        """
        # Get performance matrix
        perf_matrix = self.perf_integration.get_performance_matrix()
        
        # Format for heatmap
        strategies = perf_matrix['strategy'].tolist()
        
        # Reorganize data for heatmap
        heatmap_data = perf_matrix.drop('strategy', axis=1).values
        
        # Set up regime names (columns)
        regime_names = [r.name for r in MarketRegime if r != MarketRegime.UNKNOWN]
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Generate heatmap
        ax = sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.2f',
            cmap='viridis',
            cbar_kws={'label': 'Performance Score (0-1)'},
            xticklabels=regime_names,
            yticklabels=strategies,
            linewidths=0.5
        )
        
        # Add title
        date_str = snapshot_date or datetime.now().strftime('%Y-%m-%d')
        plt.title(f'Strategy Performance by Market Regime (as of {date_str})', 
                 fontsize=16, pad=20)
        
        # Adjust labels
        plt.xlabel('Market Regime', fontsize=14, labelpad=10)
        plt.ylabel('Strategy', fontsize=14, labelpad=10)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = os.path.join(
                self.output_dir, 
                f'strategy_heatmap_{date_str.replace("-", "")}.png'
            )
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also export data as CSV for further analysis
        csv_path = save_path.replace('.png', '.csv')
        perf_matrix.to_csv(csv_path, index=False)
        
        return save_path
    
    def generate_regime_distribution_chart(self, 
                                          data_path: Optional[str] = None,
                                          days_back: int = 90,
                                          save_path: Optional[str] = None) -> str:
        """
        Generate a chart showing the distribution of market regimes over time.
        
        Args:
            data_path: Path to regime history data or None to use default
            days_back: Number of days to include in the chart
            save_path: Path to save the visualization or None for default
            
        Returns:
            Path to the saved visualization
        """
        # Load regime history data
        # In a real implementation, this would load from a database or file
        if data_path is None:
            # Create synthetic data for demonstration
            # In a real system, this would be loaded from historical logs
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Generate synthetic regime history
            np.random.seed(42)  # For reproducibility
            
            # Define regimes and their probabilities (skewed to make it more realistic)
            regimes = [r.name for r in MarketRegime if r != MarketRegime.UNKNOWN]
            probabilities = [0.25, 0.2, 0.3, 0.1, 0.1, 0.05]  # Probabilities sum to 1
            
            # Generate regime history
            regime_history = np.random.choice(regimes, size=len(date_range), p=probabilities)
            
            history_df = pd.DataFrame({
                'date': date_range,
                'regime': regime_history
            })
        else:
            # Load real data
            history_df = pd.read_csv(data_path)
            
            # Ensure date is in datetime format
            history_df['date'] = pd.to_datetime(history_df['date'])
            
            # Filter to requested time range
            cutoff_date = datetime.now() - timedelta(days=days_back)
            history_df = history_df[history_df['date'] >= cutoff_date]
        
        # Calculate regime distribution
        regime_counts = history_df['regime'].value_counts().to_dict()
        
        # Calculate percentages
        total = sum(regime_counts.values())
        regime_percentages = {regime: count / total * 100 for regime, count in regime_counts.items()}
        
        # Set up plot
        plt.figure(figsize=(12, 8))
        
        # Create bars with custom colors
        bars = plt.bar(
            list(regime_percentages.keys()), 
            list(regime_percentages.values()),
            color=[self.regime_colors.get(regime, '#212121') for regime in regime_percentages.keys()]
        )
        
        # Add percentage labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2., 
                height + 1,
                f'{height:.1f}%',
                ha='center', 
                va='bottom',
                fontsize=11
            )
        
        # Add title and labels
        plt.title(f'Market Regime Distribution (Last {days_back} Days)', fontsize=16, pad=20)
        plt.xlabel('Market Regime', fontsize=14, labelpad=10)
        plt.ylabel('Percentage (%)', fontsize=14, labelpad=10)
        
        # Add grid
        plt.grid(axis='y', alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Y-axis range
        plt.ylim(0, max(regime_percentages.values()) * 1.2)
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = os.path.join(
                self.output_dir, 
                f'regime_distribution_{datetime.now().strftime("%Y%m%d")}.png'
            )
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_strategy_performance_trend(self,
                                          strategy_name: str,
                                          data_points: Optional[List[Dict[str, Any]]] = None,
                                          save_path: Optional[str] = None) -> str:
        """
        Generate a time series chart showing strategy performance trend.
        
        Args:
            strategy_name: Name of the strategy to visualize
            data_points: List of performance data points or None to use synthetic
            save_path: Path to save the visualization or None for default
            
        Returns:
            Path to the saved visualization
        """
        # If no data provided, generate synthetic data for demonstration
        if data_points is None:
            # Create synthetic performance history
            # In a real system, this would come from stored performance logs
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)  # 6 months of data
            
            # Generate dates at 2-week intervals
            dates = pd.date_range(start=start_date, end=end_date, freq='2W')
            
            # Synthetic performance data with improvement trend
            # Start lower and gradually improve with some noise
            np.random.seed(42)  # For reproducibility
            
            # Base performance for each regime
            regime_base_perf = {
                'TRENDING_UP': 0.65,
                'TRENDING_DOWN': 0.60,
                'RANGING': 0.70,
                'VOLATILE_BREAKOUT': 0.55,
                'VOLATILE_REVERSAL': 0.50,
                'CHOPPY': 0.45
            }
            
            # Generate synthetic data for each regime
            data_points = []
            
            for regime, base_perf in regime_base_perf.items():
                # Start lower and improve over time with noise
                initial_perf = base_perf - 0.2
                performance_values = []
                
                for i, date in enumerate(dates):
                    # Linear improvement with noise
                    progress_factor = i / (len(dates) - 1)  # 0 to 1
                    performance = initial_perf + (0.3 * progress_factor) + (np.random.normal(0, 0.03))
                    performance = max(0.1, min(0.95, performance))  # Clip to reasonable range
                    
                    performance_values.append(performance)
                
                # Create data points for this regime
                for i, date in enumerate(dates):
                    data_points.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'strategy': strategy_name,
                        'regime': regime,
                        'performance': performance_values[i]
                    })
        
        # Convert to DataFrame for easier processing
        perf_df = pd.DataFrame(data_points)
        
        # Convert date to datetime if it's not already
        if perf_df['date'].dtype == 'object':
            perf_df['date'] = pd.to_datetime(perf_df['date'])
        
        # Filter to requested strategy
        perf_df = perf_df[perf_df['strategy'] == strategy_name]
        
        # Set up plot
        plt.figure(figsize=(14, 8))
        
        # Plot each regime as a separate line
        for regime in perf_df['regime'].unique():
            regime_data = perf_df[perf_df['regime'] == regime]
            
            plt.plot(
                regime_data['date'], 
                regime_data['performance'],
                marker='o',
                linewidth=2,
                label=regime,
                color=self.regime_colors.get(regime, '#212121')
            )
        
        # Add title and labels
        plt.title(f'Performance Trend: {strategy_name}', fontsize=16, pad=20)
        plt.xlabel('Date', fontsize=14, labelpad=10)
        plt.ylabel('Performance Score (0-1)', fontsize=14, labelpad=10)
        
        # Add legend
        plt.legend(title='Market Regime', title_fontsize=12)
        
        # Add grid
        plt.grid(alpha=0.3)
        
        # Format x-axis dates
        plt.gcf().autofmt_xdate()
        
        # Y-axis range
        plt.ylim(0, 1.0)
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = os.path.join(
                self.output_dir, 
                f'strategy_trend_{strategy_name}_{datetime.now().strftime("%Y%m%d")}.png'
            )
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_ml_vs_traditional_comparison(self,
                                            accuracy_data: Optional[Dict[str, List[float]]] = None,
                                            save_path: Optional[str] = None) -> str:
        """
        Generate a chart comparing ML vs traditional regime detection accuracy.
        
        Args:
            accuracy_data: Dict with accuracy values or None to use synthetic
            save_path: Path to save the visualization or None for default
            
        Returns:
            Path to the saved visualization
        """
        # If no data provided, use synthetic data
        if accuracy_data is None:
            # Synthetic data showing improvement with ML methods
            accuracy_data = {
                'RandomForest': [0.72, 0.68, 0.75, 0.82, 0.79, 0.74],
                'LSTM': [0.75, 0.70, 0.78, 0.85, 0.84, 0.78],
                'Traditional': [0.65, 0.60, 0.68, 0.72, 0.71, 0.64]
            }
        
        # Get regimes (assuming same for all methods)
        regimes = [r.name for r in MarketRegime if r != MarketRegime.UNKNOWN]
        
        # Ensure we have same number of regimes as data points
        regimes = regimes[:len(next(iter(accuracy_data.values())))]
        
        # Set up plot
        plt.figure(figsize=(14, 8))
        
        # Set bar width
        bar_width = 0.25
        
        # Set up positions for grouped bars
        positions = np.arange(len(regimes))
        
        # Custom colors for each method
        method_colors = {
            'RandomForest': '#1565c0',  # Darker blue
            'LSTM': '#6a1b9a',          # Darker purple
            'Traditional': '#616161'     # Gray
        }
        
        # Plot bars for each method
        i = 0
        for method, accuracies in accuracy_data.items():
            plt.bar(
                positions + (i * bar_width), 
                accuracies[:len(regimes)],
                bar_width,
                label=method,
                color=method_colors.get(method, '#212121'),
                alpha=0.85
            )
            i += 1
        
        # Add title and labels
        plt.title('Regime Detection Accuracy: ML vs Traditional Methods', fontsize=16, pad=20)
        plt.xlabel('Market Regime', fontsize=14, labelpad=10)
        plt.ylabel('Accuracy (%)', fontsize=14, labelpad=10)
        
        # Set X-axis ticks and labels
        plt.xticks(
            positions + bar_width, 
            regimes,
            rotation=45,
            ha='right'
        )
        
        # Format Y-axis as percentage
        plt.yticks(np.arange(0, 1.1, 0.1), [f'{int(x*100)}%' for x in np.arange(0, 1.1, 0.1)])
        
        # Add legend
        plt.legend(title='Detection Method', title_fontsize=12)
        
        # Add grid
        plt.grid(axis='y', alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = os.path.join(
                self.output_dir, 
                f'ml_comparison_{datetime.now().strftime("%Y%m%d")}.png'
            )
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_dashboard_summary(self, output_format: str = 'html') -> str:
        """
        Generate a complete dashboard summary with all visualizations.
        
        Args:
            output_format: Format to save dashboard ('html' or 'json')
            
        Returns:
            Path to the saved dashboard file
        """
        # Create timestamp for the dashboard
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        date_code = datetime.now().strftime('%Y%m%d')
        
        # Generate all visualizations
        heatmap_path = self.generate_strategy_heatmap()
        regime_dist_path = self.generate_regime_distribution_chart()
        
        # Get strategy list
        strategy_selector = ForexStrategySelector()
        strategies = list(strategy_selector.strategy_compatibility.keys())
        
        # Generate performance trends for each strategy
        trend_paths = {}
        for strategy in strategies:
            trend_path = self.generate_strategy_performance_trend(strategy)
            trend_paths[strategy] = os.path.basename(trend_path)
        
        # Generate ML comparison
        ml_comparison_path = self.generate_ml_vs_traditional_comparison()
        
        # Prepare dashboard content
        dashboard_data = {
            'timestamp': timestamp,
            'visualizations': {
                'strategy_heatmap': os.path.basename(heatmap_path),
                'regime_distribution': os.path.basename(regime_dist_path),
                'strategy_trends': trend_paths,
                'ml_comparison': os.path.basename(ml_comparison_path)
            },
            'strategy_count': len(strategies),
            'regime_count': len([r for r in MarketRegime if r != MarketRegime.UNKNOWN])
        }
        
        # Save dashboard data
        if output_format == 'json':
            # Save as JSON
            output_path = os.path.join(self.output_dir, f'dashboard_summary_{date_code}.json')
            
            with open(output_path, 'w') as f:
                json.dump(dashboard_data, f, indent=2)
        else:
            # Generate HTML
            output_path = os.path.join(self.output_dir, f'dashboard_summary_{date_code}.html')
            
            # Simple HTML template
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Forex Strategy Performance Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333; }}
                    .dashboard-section {{ margin-bottom: 40px; }}
                    .visualization {{ margin: 20px 0; }}
                    .visualization img {{ max-width: 100%; border: 1px solid #ddd; }}
                    .timestamp {{ color: #666; font-style: italic; }}
                </style>
            </head>
            <body>
                <h1>Forex Strategy Performance Dashboard</h1>
                <p class="timestamp">Generated on: {timestamp}</p>
                
                <div class="dashboard-section">
                    <h2>Strategy Performance by Market Regime</h2>
                    <div class="visualization">
                        <img src="{os.path.basename(heatmap_path)}" alt="Strategy Heatmap">
                    </div>
                </div>
                
                <div class="dashboard-section">
                    <h2>Market Regime Distribution</h2>
                    <div class="visualization">
                        <img src="{os.path.basename(regime_dist_path)}" alt="Regime Distribution">
                    </div>
                </div>
                
                <div class="dashboard-section">
                    <h2>ML vs Traditional Regime Detection</h2>
                    <div class="visualization">
                        <img src="{os.path.basename(ml_comparison_path)}" alt="ML Comparison">
                    </div>
                </div>
                
                <div class="dashboard-section">
                    <h2>Strategy Performance Trends</h2>
            """
            
            # Add trend charts for each strategy
            for strategy, path in trend_paths.items():
                html_content += f"""
                    <div class="visualization">
                        <h3>{strategy}</h3>
                        <img src="{path}" alt="{strategy} Performance Trend">
                    </div>
                """
            
            # Close HTML tags
            html_content += """
                </div>
            </body>
            </html>
            """
            
            # Write HTML to file
            with open(output_path, 'w') as f:
                f.write(html_content)
        
        return output_path


def main():
    """Command line interface for performance dashboard generation."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate forex strategy performance dashboard')
    parser.add_argument('--output', help='Output directory for visualizations')
    parser.add_argument('--format', choices=['html', 'json'], default='html',
                     help='Output format for dashboard summary')
    
    args = parser.parse_args()
    
    # Initialize dashboard
    dashboard = PerformanceDashboard(output_dir=args.output)
    
    # Generate dashboard summary
    output_path = dashboard.generate_dashboard_summary(output_format=args.format)
    
    print(f"Dashboard generated: {output_path}")
    print(f"All visualizations saved to: {dashboard.output_dir}")


if __name__ == "__main__":
    main()
