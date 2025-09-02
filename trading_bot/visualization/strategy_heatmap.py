"""
Strategy Effectiveness Heatmap

This module provides visualization tools for analyzing strategy performance
across different market regimes, allowing traders to identify optimal
strategy selection based on market conditions.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from trading_bot.market.regime_classifier import MarketRegimeClassifier
    from trading_bot.journal.trade_journal import TradeJournal
    from trading_bot.data.market_data_provider import MarketDataProvider
    from trading_bot.config.config_manager import ConfigManager
except ImportError as e:
    logger.error(f"Failed to import required modules: {str(e)}")
    
class StrategyHeatmapGenerator:
    """
    Generates interactive heatmaps and visualizations showing strategy
    performance across different market regimes.
    
    This class analyzes historical trade data, categorizes trades by market regime,
    and visualizes key performance metrics to help traders identify which strategies
    work best in specific market conditions.
    """
    
    def __init__(
        self,
        config_path: str,
        journal_dir: str,
        output_dir: Optional[str] = None,
        lookback_days: int = 90
    ):
        """
        Initialize the strategy heatmap generator.
        
        Args:
            config_path: Path to configuration file
            journal_dir: Path to trade journal directory
            output_dir: Directory to save visualization outputs
            lookback_days: Number of days to analyze
        """
        self.config_path = config_path
        self.journal_dir = journal_dir
        self.output_dir = output_dir or os.path.join(os.path.dirname(config_path), "visualizations")
        self.lookback_days = lookback_days
        
        # Load configuration
        try:
            self.config_manager = ConfigManager(config_path)
            self.config = self.config_manager.load_config()
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
        
        # Initialize trade journal
        self.trade_journal = TradeJournal(self.journal_dir)
        
        # Initialize market data provider
        self.market_data_provider = MarketDataProvider(
            api_key=self.config.get("market_data", {}).get("api_key"),
            sources=self.config.get("market_data", {}).get("sources", ["yahoo"])
        )
        
        # Initialize regime classifier
        self.regime_classifier = MarketRegimeClassifier(
            config_path=self.config.get("regime_classifier", {}).get("config_path"),
            data_provider=self.market_data_provider
        )
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Strategy Heatmap Generator initialized with {lookback_days} days lookback")
    
    def generate_strategy_regime_heatmap(
        self,
        metric: str = "win_rate",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_trades: int = 5,
        include_strategies: Optional[List[str]] = None,
        exclude_strategies: Optional[List[str]] = None,
        output_file: Optional[str] = None,
        show_plot: bool = True
    ) -> go.Figure:
        """
        Generate a heatmap showing strategy performance by market regime.
        
        Args:
            metric: Performance metric to visualize ('win_rate', 'avg_pnl', 'sharpe', 'trades')
            start_date: Start date in YYYY-MM-DD format (defaults to lookback_days ago)
            end_date: End date in YYYY-MM-DD format (defaults to today)
            min_trades: Minimum number of trades required for strategy/regime inclusion
            include_strategies: List of strategy IDs to include (None for all)
            exclude_strategies: List of strategy IDs to exclude
            output_file: Path to save the visualization (None to auto-generate)
            show_plot: Whether to display the plot
            
        Returns:
            Plotly figure object
        """
        logger.info(f"Generating strategy regime heatmap for metric: {metric}")
        
        # Set date range
        end_date_obj = datetime.now().date()
        if end_date:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
            
        start_date_obj = end_date_obj - timedelta(days=self.lookback_days)
        if start_date:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        
        # Convert to string format
        start_date_str = start_date_obj.strftime("%Y-%m-%d")
        end_date_str = end_date_obj.strftime("%Y-%m-%d")
        
        # Get trades for the period
        trades = self._get_trades_with_regimes(start_date_str, end_date_str)
        
        if not trades:
            logger.warning(f"No trades found for period {start_date_str} to {end_date_str}")
            return self._create_empty_heatmap(metric)
        
        # Filter strategies if needed
        if include_strategies:
            trades = [t for t in trades if t.get("strategy_id") in include_strategies]
        if exclude_strategies:
            trades = [t for t in trades if t.get("strategy_id") not in exclude_strategies]
        
        # Group trades by strategy and regime
        df = self._prepare_dataframe(trades, metric)
        
        # Filter for minimum trades
        if min_trades > 0:
            # Create a pivot table of trade counts
            trade_counts = pd.pivot_table(
                df, 
                values='count', 
                index='strategy_id', 
                columns='market_regime',
                fill_value=0
            )
            
            # Get strategies that have at least min_trades in at least one regime
            valid_strategies = trade_counts[trade_counts.max(axis=1) >= min_trades].index.tolist()
            
            # Filter the dataframe
            df = df[df['strategy_id'].isin(valid_strategies)]
        
        # Create pivot table for heatmap
        if metric == "trades":
            pivot_df = pd.pivot_table(
                df, 
                values='count', 
                index='strategy_id', 
                columns='market_regime',
                fill_value=0
            )
        else:
            pivot_df = pd.pivot_table(
                df, 
                values=metric, 
                index='strategy_id', 
                columns='market_regime',
                fill_value=np.nan
            )
        
        # Generate the heatmap
        fig = self._create_heatmap(pivot_df, metric, start_date_str, end_date_str)
        
        # Save if output file is specified
        if output_file:
            output_path = output_file
        else:
            output_path = os.path.join(
                self.output_dir, 
                f"strategy_regime_heatmap_{metric}_{datetime.now().strftime('%Y%m%d')}.html"
            )
            
        try:
            fig.write_html(output_path)
            logger.info(f"Saved heatmap to {output_path}")
        except Exception as e:
            logger.error(f"Error saving heatmap: {str(e)}")
        
        return fig
    
    def generate_strategy_dashboard(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_strategies: Optional[List[str]] = None,
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Generate a comprehensive dashboard of strategy performance.
        
        Args:
            start_date: Start date in YYYY-MM-DD format (defaults to lookback_days ago)
            end_date: End date in YYYY-MM-DD format (defaults to today)
            include_strategies: List of strategy IDs to include (None for all)
            output_file: Path to save the visualization (None to auto-generate)
            
        Returns:
            Plotly figure object with multiple subplots
        """
        logger.info("Generating strategy performance dashboard")
        
        # Set date range
        end_date_obj = datetime.now().date()
        if end_date:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
            
        start_date_obj = end_date_obj - timedelta(days=self.lookback_days)
        if start_date:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        
        # Convert to string format
        start_date_str = start_date_obj.strftime("%Y-%m-%d")
        end_date_str = end_date_obj.strftime("%Y-%m-%d")
        
        # Get trades for the period
        trades = self._get_trades_with_regimes(start_date_str, end_date_str)
        
        if not trades:
            logger.warning(f"No trades found for period {start_date_str} to {end_date_str}")
            # Create empty dashboard
            fig = make_subplots(rows=2, cols=2, 
                subplot_titles=["Strategy Win Rates by Regime", 
                                "Average P&L by Regime",
                                "Trade Frequency by Regime",
                                "Overall Strategy Performance"])
            fig.update_layout(title_text=f"Strategy Performance Dashboard (No Data Available)", 
                              height=900, width=1200)
            return fig
        
        # Filter strategies if needed
        if include_strategies:
            trades = [t for t in trades if t.get("strategy_id") in include_strategies]
        
        # Create dataframe with all metrics
        df = self._prepare_complete_dataframe(trades)
        
        # Create a subplot figure
        fig = make_subplots(
            rows=2, cols=2, 
            subplot_titles=["Strategy Win Rates by Regime", 
                           "Average P&L by Regime",
                           "Trade Frequency by Regime",
                           "Overall Strategy Performance"],
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
                   [{"type": "heatmap"}, {"type": "bar"}]]
        )
        
        # Add win rate heatmap
        win_rate_pivot = pd.pivot_table(
            df, values='win_rate', index='strategy_id', columns='market_regime', fill_value=np.nan
        )
        heatmap1 = go.Heatmap(
            z=win_rate_pivot.values,
            x=win_rate_pivot.columns,
            y=win_rate_pivot.index,
            colorscale="RdYlGn",
            zmid=0.5,  # Center colorscale at 50%
            zmin=0,
            zmax=1,
            colorbar=dict(title="Win Rate", x=0.46)
        )
        fig.add_trace(heatmap1, row=1, col=1)
        
        # Add avg PnL heatmap
        pnl_pivot = pd.pivot_table(
            df, values='avg_pnl', index='strategy_id', columns='market_regime', fill_value=np.nan
        )
        heatmap2 = go.Heatmap(
            z=pnl_pivot.values,
            x=pnl_pivot.columns,
            y=pnl_pivot.index,
            colorscale="RdYlGn",
            zmid=0,  # Center colorscale at 0
            colorbar=dict(title="Avg P&L", x=0.96)
        )
        fig.add_trace(heatmap2, row=1, col=2)
        
        # Add trade count heatmap
        count_pivot = pd.pivot_table(
            df, values='count', index='strategy_id', columns='market_regime', fill_value=0
        )
        heatmap3 = go.Heatmap(
            z=count_pivot.values,
            x=count_pivot.columns,
            y=count_pivot.index,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Trade Count", x=0.46, y=0.25)
        )
        fig.add_trace(heatmap3, row=2, col=1)
        
        # Add overall performance bar chart
        # Group by strategy_id to get overall metrics
        overall_perf = df.groupby('strategy_id').agg({
            'win_rate': 'mean',
            'avg_pnl': 'mean',
            'count': 'sum',
            'total_pnl': 'sum'
        }).reset_index()
        
        # Sort by total P&L
        overall_perf = overall_perf.sort_values('total_pnl', ascending=False)
        
        # Add bar chart
        bar = go.Bar(
            x=overall_perf['strategy_id'],
            y=overall_perf['total_pnl'],
            marker_color=overall_perf['total_pnl'],
            marker=dict(
                color=overall_perf['total_pnl'],
                colorscale='RdYlGn',
                cmin=overall_perf['total_pnl'].min(),
                cmax=overall_perf['total_pnl'].max(),
            ),
            text=overall_perf['count'].apply(lambda x: f"{x} trades"),
            hovertemplate="<b>%{x}</b><br>Total P&L: %{y:.2f}<br>%{text}<extra></extra>"
        )
        fig.add_trace(bar, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title_text=f"Strategy Performance Dashboard ({start_date_str} to {end_date_str})",
            height=900,
            width=1200,
            showlegend=False
        )
        
        # Save if output file is specified
        if output_file:
            output_path = output_file
        else:
            output_path = os.path.join(
                self.output_dir, 
                f"strategy_dashboard_{datetime.now().strftime('%Y%m%d')}.html"
            )
            
        try:
            fig.write_html(output_path)
            logger.info(f"Saved dashboard to {output_path}")
        except Exception as e:
            logger.error(f"Error saving dashboard: {str(e)}")
        
        return fig
    
    def generate_regime_alpha_zones(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_trades: int = 5,
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Generate a visualization highlighting the "alpha zones" - 
        areas where strategies outperform across different market regimes.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            min_trades: Minimum trade count for inclusion
            output_file: Path to save the visualization
            
        Returns:
            Plotly figure object
        """
        logger.info("Generating regime alpha zones visualization")
        
        # Set date range
        end_date_obj = datetime.now().date()
        if end_date:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
            
        start_date_obj = end_date_obj - timedelta(days=self.lookback_days)
        if start_date:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        
        # Convert to string format
        start_date_str = start_date_obj.strftime("%Y-%m-%d")
        end_date_str = end_date_obj.strftime("%Y-%m-%d")
        
        # Get trades for the period
        trades = self._get_trades_with_regimes(start_date_str, end_date_str)
        
        if not trades:
            logger.warning(f"No trades found for period {start_date_str} to {end_date_str}")
            # Create empty plot
            fig = go.Figure()
            fig.update_layout(
                title="Strategy Alpha Zones (No Data Available)",
                xaxis_title="Market Regime",
                yaxis_title="Strategy",
                height=800,
                width=1000
            )
            return fig
        
        # Create a complete dataframe with all metrics
        df = self._prepare_complete_dataframe(trades)
        
        # Filter for strategies with minimum trades
        strategy_counts = df.groupby('strategy_id')['count'].sum().reset_index()
        valid_strategies = strategy_counts[strategy_counts['count'] >= min_trades]['strategy_id'].tolist()
        df = df[df['strategy_id'].isin(valid_strategies)]
        
        # Create a normalized score for alpha
        # Alpha score = win_rate * avg_pnl (normalized to 0-1 range)
        regime_groups = df.groupby('market_regime')
        
        # Normalize avg_pnl within each regime
        for regime, group in regime_groups:
            if len(group) > 0:
                min_pnl = group['avg_pnl'].min()
                max_pnl = group['avg_pnl'].max()
                
                if max_pnl > min_pnl:
                    # Normalize pnl to 0-1 range
                    df.loc[df['market_regime'] == regime, 'norm_pnl'] = (df.loc[df['market_regime'] == regime, 'avg_pnl'] - min_pnl) / (max_pnl - min_pnl)
                else:
                    df.loc[df['market_regime'] == regime, 'norm_pnl'] = 0.5
        
        # Calculate alpha score
        df['alpha_score'] = df['win_rate'] * df['norm_pnl'] * (df['count'] / df['count'].max())**0.5
        
        # Create pivot table for bubble chart
        alpha_pivot = pd.pivot_table(
            df, 
            values=['alpha_score', 'count', 'win_rate', 'avg_pnl', 'total_pnl'],
            index='strategy_id', 
            columns='market_regime',
            fill_value=np.nan
        )
        
        # Create the figure
        fig = go.Figure()
        
        # Define regime positions on x-axis
        regimes = df['market_regime'].unique()
        regime_positions = {regime: i for i, regime in enumerate(sorted(regimes))}
        
        # Add a trace for each strategy
        for strategy in alpha_pivot.index:
            x_values = []
            y_values = []
            sizes = []
            colors = []
            hover_texts = []
            
            for regime in regimes:
                if not np.isnan(alpha_pivot['alpha_score'][strategy].get(regime, np.nan)):
                    x_values.append(regime_positions[regime])
                    y_values.append(strategy)
                    
                    # Size based on trade count
                    count = alpha_pivot['count'][strategy].get(regime, 0)
                    sizes.append(count * 10)  # Scale for visibility
                    
                    # Color based on alpha score
                    alpha = alpha_pivot['alpha_score'][strategy].get(regime, 0)
                    colors.append(alpha)
                    
                    # Hover text
                    win_rate = alpha_pivot['win_rate'][strategy].get(regime, 0)
                    avg_pnl = alpha_pivot['avg_pnl'][strategy].get(regime, 0)
                    total_pnl = alpha_pivot['total_pnl'][strategy].get(regime, 0)
                    
                    hover_text = (
                        f"Strategy: {strategy}<br>"
                        f"Regime: {regime}<br>"
                        f"Win Rate: {win_rate:.1%}<br>"
                        f"Avg P&L: {avg_pnl:.2f}<br>"
                        f"Total P&L: {total_pnl:.2f}<br>"
                        f"Trade Count: {count}"
                    )
                    hover_texts.append(hover_text)
            
            # Add bubble trace
            if x_values:
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='markers',
                    marker=dict(
                        size=sizes,
                        sizemode='area',
                        sizeref=2. * max(sizes) / (40.**2),
                        sizemin=4,
                        color=colors,
                        colorscale='RdYlGn',
                        colorbar=dict(title="Alpha Score"),
                        showscale=(strategy == alpha_pivot.index[-1])  # Only show colorscale for last strategy
                    ),
                    text=hover_texts,
                    hoverinfo='text',
                    name=strategy
                ))
        
        # Update layout
        fig.update_layout(
            title=f"Strategy Alpha Zones by Market Regime ({start_date_str} to {end_date_str})",
            xaxis=dict(
                tickmode='array',
                tickvals=list(regime_positions.values()),
                ticktext=list(regime_positions.keys()),
                title="Market Regime"
            ),
            yaxis=dict(
                title="Strategy"
            ),
            showlegend=False,
            height=800,
            width=1000
        )
        
        # Save if output file is specified
        if output_file:
            output_path = output_file
        else:
            output_path = os.path.join(
                self.output_dir, 
                f"alpha_zones_{datetime.now().strftime('%Y%m%d')}.html"
            )
            
        try:
            fig.write_html(output_path)
            logger.info(f"Saved alpha zones visualization to {output_path}")
        except Exception as e:
            logger.error(f"Error saving visualization: {str(e)}")
        
        return fig
    
    def generate_strategy_drawdown_analysis(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Generate a visualization showing drawdowns by strategy and market regime.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            output_file: Path to save the visualization
            
        Returns:
            Plotly figure object
        """
        logger.info("Generating strategy drawdown analysis")
        
        # Set date range
        end_date_obj = datetime.now().date()
        if end_date:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
            
        start_date_obj = end_date_obj - timedelta(days=self.lookback_days)
        if start_date:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        
        # Convert to string format
        start_date_str = start_date_obj.strftime("%Y-%m-%d")
        end_date_str = end_date_obj.strftime("%Y-%m-%d")
        
        # Get trades
        trades = self._get_trades_with_regimes(start_date_str, end_date_str)
        
        if not trades:
            logger.warning(f"No trades found for period {start_date_str} to {end_date_str}")
            # Create empty plot
            fig = go.Figure()
            fig.update_layout(
                title="Strategy Drawdown Analysis (No Data Available)",
                height=600,
                width=1000
            )
            return fig
        
        # Create a dataframe with trades
        df = pd.DataFrame(trades)
        
        # Convert columns to appropriate types
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        df['realized_pnl'] = pd.to_numeric(df['realized_pnl'], errors='coerce')
        
        # Sort by exit time
        df = df.sort_values('exit_time')
        
        # Calculate cumulative P&L by strategy
        strategy_pnl = {}
        max_drawdowns = {}
        
        for strategy in df['strategy_id'].unique():
            strategy_trades = df[df['strategy_id'] == strategy]
            
            if len(strategy_trades) == 0:
                continue
                
            # Calculate cumulative P&L
            cumulative_pnl = strategy_trades['realized_pnl'].cumsum()
            
            # Calculate drawdowns
            running_max = cumulative_pnl.cummax()
            drawdown = running_max - cumulative_pnl
            
            # Record maximum drawdown
            max_drawdown = drawdown.max()
            max_drawdown_pct = max_drawdown / running_max.max() if running_max.max() > 0 else 0
            
            # Calculate drawdown by regime
            regime_drawdowns = {}
            
            for regime in strategy_trades['market_regime'].unique():
                regime_trades = strategy_trades[strategy_trades['market_regime'] == regime]
                
                if len(regime_trades) < 2:
                    regime_drawdowns[regime] = 0
                    continue
                
                # Calculate cumulative P&L for this regime
                regime_cumulative_pnl = regime_trades['realized_pnl'].cumsum()
                
                # Calculate drawdowns
                regime_running_max = regime_cumulative_pnl.cummax()
                regime_drawdown = regime_running_max - regime_cumulative_pnl
                
                # Record maximum drawdown
                regime_max_drawdown = regime_drawdown.max()
                regime_drawdowns[regime] = regime_max_drawdown
            
            # Store results
            strategy_pnl[strategy] = {
                'cumulative_pnl': cumulative_pnl,
                'running_max': running_max,
                'drawdown': drawdown,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown_pct,
                'regime_drawdowns': regime_drawdowns,
                'exit_times': strategy_trades['exit_time'].values
            }
            
            max_drawdowns[strategy] = {
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown_pct,
                'regime_drawdowns': regime_drawdowns
            }
        
        # Create subplots: drawdown heatmap and drawdown curves
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Maximum Drawdown by Strategy and Regime", "Drawdown Timeline"],
            row_heights=[0.4, 0.6],
            specs=[[{"type": "heatmap"}], [{"type": "scatter"}]]
        )
        
        # Create drawdown heatmap data
        strategies = list(max_drawdowns.keys())
        regimes = list(set().union(*[d['regime_drawdowns'].keys() for d in max_drawdowns.values()]))
        
        heatmap_data = []
        for strategy in strategies:
            row = []
            for regime in regimes:
                if regime in max_drawdowns[strategy]['regime_drawdowns']:
                    row.append(max_drawdowns[strategy]['regime_drawdowns'][regime])
                else:
                    row.append(np.nan)
            heatmap_data.append(row)
        
        # Add drawdown heatmap
        heatmap = go.Heatmap(
            z=heatmap_data,
            x=regimes,
            y=strategies,
            colorscale="YlOrRd_r",  # Reversed so darker = worse
            colorbar=dict(title="Drawdown"),
            hovertemplate="<b>%{y}</b><br>Regime: %{x}<br>Drawdown: %{z:.2f}<extra></extra>"
        )
        fig.add_trace(heatmap, row=1, col=1)
        
        # Add drawdown curves for each strategy
        for i, strategy in enumerate(strategies):
            # Skip if no data
            if 'drawdown' not in strategy_pnl[strategy]:
                continue
                
            # Convert to list for plotly
            drawdown_values = strategy_pnl[strategy]['drawdown'].tolist()
            exit_times = strategy_pnl[strategy]['exit_times'].tolist()
            
            # Add drawdown curve
            fig.add_trace(
                go.Scatter(
                    x=exit_times,
                    y=drawdown_values,
                    mode='lines',
                    name=strategy,
                    line=dict(
                        width=2
                    ),
                    hovertemplate="<b>%{x}</b><br>Drawdown: %{y:.2f}<extra>" + strategy + "</extra>"
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"Strategy Drawdown Analysis ({start_date_str} to {end_date_str})",
            height=900,
            width=1000,
            yaxis2=dict(title="Drawdown"),
            xaxis2=dict(title="Date")
        )
        
        # Save if output file is specified
        if output_file:
            output_path = output_file
        else:
            output_path = os.path.join(
                self.output_dir, 
                f"drawdown_analysis_{datetime.now().strftime('%Y%m%d')}.html"
            )
            
        try:
            fig.write_html(output_path)
            logger.info(f"Saved drawdown analysis to {output_path}")
        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}")
        
        return fig
    
    def _get_trades_with_regimes(self, start_date: str, end_date: str) -> List[Dict]:
        """
        Get trades with market regime information for the specified period.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of trade dictionaries with market regime information
        """
        try:
            # Get trades from journal
            trades = self.trade_journal.get_trades(
                start_date=start_date,
                end_date=end_date,
                include_open=False  # Only include closed trades
            )
            
            if not trades:
                return []
            
            # Get historical regimes
            historical_regimes = self.regime_classifier.get_historical_regimes(
                start_date=start_date,
                end_date=end_date
            )
            
            # Add regime to each trade
            for trade in trades:
                # Convert entry time to date
                entry_time = trade.get('entry_time')
                if not entry_time:
                    continue
                    
                if isinstance(entry_time, str):
                    entry_date = entry_time.split()[0]
                else:
                    entry_date = entry_time.strftime("%Y-%m-%d")
                
                # Get regime for this date
                trade['market_regime'] = historical_regimes.get(entry_date, "unknown")
            
            return trades
            
        except Exception as e:
            logger.error(f"Error getting trades with regimes: {str(e)}")
            return []
    
    def _prepare_dataframe(self, trades: List[Dict], metric: str) -> pd.DataFrame:
        """
        Prepare dataframe for visualization from trade data.
        
        Args:
            trades: List of trade dictionaries
            metric: Performance metric to calculate
            
        Returns:
            Pandas DataFrame with market regime, strategy, and performance metrics
        """
        # Convert to dataframe
        df = pd.DataFrame(trades)
        
        # Group by strategy and regime
        grouped = df.groupby(['strategy_id', 'market_regime'])
        
        # Initialize results
        results = []
        
        for (strategy_id, regime), group in grouped:
            # Skip if no trades
            if len(group) == 0:
                continue
                
            # Calculate metrics
            trade_count = len(group)
            win_count = sum(group['realized_pnl'] > 0)
            win_rate = win_count / trade_count if trade_count > 0 else 0
            total_pnl = group['realized_pnl'].sum()
            avg_pnl = group['realized_pnl'].mean()
            
            # Calculate Sharpe ratio if possible
            if len(group) > 1:
                returns = group['realized_pnl'].values
                sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
            else:
                sharpe = 0
            
            # Add to results
            results.append({
                'strategy_id': strategy_id,
                'market_regime': regime,
                'count': trade_count,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'sharpe': sharpe
            })
        
        return pd.DataFrame(results)
    
    def _prepare_complete_dataframe(self, trades: List[Dict]) -> pd.DataFrame:
        """
        Prepare a complete dataframe with all metrics for analysis.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Pandas DataFrame with all metrics calculated
        """
        # Convert to dataframe
        df = pd.DataFrame(trades)
        
        # Group by strategy and regime
        grouped = df.groupby(['strategy_id', 'market_regime'])
        
        # Initialize results
        results = []
        
        for (strategy_id, regime), group in grouped:
            # Skip if no trades
            if len(group) == 0:
                continue
                
            # Calculate metrics
            trade_count = len(group)
            win_count = sum(group['realized_pnl'] > 0)
            win_rate = win_count / trade_count if trade_count > 0 else 0
            total_pnl = group['realized_pnl'].sum()
            avg_pnl = group['realized_pnl'].mean()
            
            # Calculate Sharpe ratio if possible
            if len(group) > 1:
                returns = group['realized_pnl'].values
                sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
                max_drawdown = self._calculate_max_drawdown(returns)
            else:
                sharpe = 0
                max_drawdown = 0
            
            # Add to results
            results.append({
                'strategy_id': strategy_id,
                'market_regime': regime,
                'count': trade_count,
                'win_count': win_count,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown
            })
        
        return pd.DataFrame(results)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown from a series of returns.
        
        Args:
            returns: Array of returns
            
        Returns:
            Maximum drawdown as a positive value
        """
        # Calculate cumulative returns
        cumulative = np.cumsum(returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)
        
        # Calculate drawdown
        drawdown = running_max - cumulative
        
        # Return maximum drawdown
        return drawdown.max() if len(drawdown) > 0 else 0
    
    def _create_heatmap(self, pivot_df: pd.DataFrame, metric: str, start_date: str, end_date: str) -> go.Figure:
        """
        Create a heatmap visualization from pivot table data.
        
        Args:
            pivot_df: Pivot table DataFrame
            metric: Performance metric being visualized
            start_date: Start date string
            end_date: End date string
            
        Returns:
            Plotly figure object
        """
        # Map metric to title and color settings
        metric_settings = {
            "win_rate": {
                "title": "Win Rate",
                "colorscale": "RdYlGn",
                "zmid": 0.5,
                "zmin": 0,
                "zmax": 1,
                "format": ".0%"
            },
            "avg_pnl": {
                "title": "Average P&L",
                "colorscale": "RdYlGn",
                "zmid": 0,
                "format": ".2f"
            },
            "sharpe": {
                "title": "Sharpe Ratio",
                "colorscale": "RdYlGn",
                "zmid": 0,
                "format": ".2f"
            },
            "trades": {
                "title": "Number of Trades",
                "colorscale": "Blues",
                "format": ".0f"
            }
        }
        
        settings = metric_settings.get(metric, {
            "title": metric.capitalize(),
            "colorscale": "Viridis",
            "format": ".2f"
        })
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale=settings.get("colorscale", "Viridis"),
            zmid=settings.get("zmid"),
            zmin=settings.get("zmin"),
            zmax=settings.get("zmax"),
            hovertemplate=f"<b>%{{y}}</b><br><b>%{{x}}</b><br>{settings['title']}: %{{z{settings.get('format', '.2f')}}}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Strategy {settings['title']} by Market Regime ({start_date} to {end_date})",
            xaxis_title="Market Regime",
            yaxis_title="Strategy",
            height=700,
            width=900
        )
        
        return fig
    
    def _create_empty_heatmap(self, metric: str) -> go.Figure:
        """
        Create an empty heatmap when no data is available.
        
        Args:
            metric: Performance metric name
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        fig.update_layout(
            title=f"Strategy {metric.capitalize()} by Market Regime (No Data Available)",
            xaxis_title="Market Regime",
            yaxis_title="Strategy",
            height=700,
            width=900,
            annotations=[
                dict(
                    text="No trade data available for the selected period",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )
            ]
        )
        
        return fig


def main():
    """Command line interface for strategy heatmap generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate strategy effectiveness heatmaps")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/trading_config.json",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--journal", 
        type=str, 
        default="data/journal",
        help="Path to trade journal directory"
    )
    
    parser.add_argument(
        "--output", 
        type=str,
        help="Directory to save visualization outputs"
    )
    
    parser.add_argument(
        "--metric", 
        type=str,
        default="win_rate",
        choices=["win_rate", "avg_pnl", "sharpe", "trades"],
        help="Performance metric to visualize"
    )
    
    parser.add_argument(
        "--start-date", 
        type=str,
        help="Start date in YYYY-MM-DD format"
    )
    
    parser.add_argument(
        "--end-date", 
        type=str,
        help="End date in YYYY-MM-DD format"
    )
    
    parser.add_argument(
        "--lookback", 
        type=int,
        default=90,
        help="Number of days to look back if no start date is provided"
    )
    
    parser.add_argument(
        "--min-trades", 
        type=int,
        default=5,
        help="Minimum number of trades required for inclusion"
    )
    
    parser.add_argument(
        "--dashboard", 
        action="store_true",
        help="Generate full strategy dashboard"
    )
    
    parser.add_argument(
        "--alpha-zones", 
        action="store_true",
        help="Generate alpha zones visualization"
    )
    
    parser.add_argument(
        "--drawdown", 
        action="store_true",
        help="Generate drawdown analysis"
    )
    
    args = parser.parse_args()
    
    # Create heatmap generator
    generator = StrategyHeatmapGenerator(
        config_path=args.config,
        journal_dir=args.journal,
        output_dir=args.output,
        lookback_days=args.lookback
    )
    
    # Generate visualizations
    if args.dashboard:
        generator.generate_strategy_dashboard(
            start_date=args.start_date,
            end_date=args.end_date
        )
    elif args.alpha_zones:
        generator.generate_regime_alpha_zones(
            start_date=args.start_date,
            end_date=args.end_date,
            min_trades=args.min_trades
        )
    elif args.drawdown:
        generator.generate_strategy_drawdown_analysis(
            start_date=args.start_date,
            end_date=args.end_date
        )
    else:
        generator.generate_strategy_regime_heatmap(
            metric=args.metric,
            start_date=args.start_date,
            end_date=args.end_date,
            min_trades=args.min_trades
        )


if __name__ == "__main__":
    main() 