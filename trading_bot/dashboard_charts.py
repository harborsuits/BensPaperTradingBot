"""
Dashboard Charts

Provides interactive chart generation for the trading dashboard.
Uses Plotly to create interactive visualizations of trading performance,
strategy metrics, and market data.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple

# Interactive chart libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class DashboardCharts:
    """
    Generates interactive charts for trading dashboard using Plotly.
    
    This class provides a collection of chart generators for visualizing
    trading performance, strategy metrics, and market data. Charts can be
    saved as HTML files for embedding in dashboards or viewing in browsers.
    """
    
    def __init__(self, output_dir: str = None, dark_mode: bool = True):
        """
        Initialize the chart generator.
        
        Args:
            output_dir: Directory to save generated charts
            dark_mode: Whether to use dark mode theme for charts
        """
        self.output_dir = output_dir or os.path.expanduser("~/.trading_bot/charts")
        self.dark_mode = dark_mode
        self.template = "plotly_dark" if dark_mode else "plotly_white"
        
        # Create output directory if it doesn't exist
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Check if Plotly is available
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for chart generation. Install with 'pip install plotly'.")

    def equity_curve_chart(self, 
                         trades: List[Dict[str, Any]], 
                         days: int = 30,
                         include_deposits: bool = True,
                         title: str = "Equity Curve",
                         filename: str = None) -> Optional[go.Figure]:
        """
        Generate an equity curve chart from trade history.
        
        Args:
            trades: List of trade dictionaries with trade details
            days: Number of days to include in the chart
            include_deposits: Whether to include deposits and withdrawals
            title: Chart title
            filename: Filename to save chart (without extension)
            
        Returns:
            Plotly figure object (or None if no data)
        """
        if not trades:
            return None
        
        # Convert trades to DataFrame
        df = pd.DataFrame(trades)
        
        # Ensure we have required columns
        required_cols = ["timestamp", "pnl_dollars"]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing required columns in trade data: {missing}")
        
        # Convert timestamp to datetime
        df["date"] = pd.to_datetime(df["timestamp"])
        
        # Filter to specified time range
        start_date = datetime.now() - timedelta(days=days)
        df = df[df["date"] >= start_date]
        
        if df.empty:
            return None
        
        # Sort by date
        df = df.sort_values("date")
        
        # Calculate cumulative P&L
        df["cumulative_pnl"] = df["pnl_dollars"].cumsum()
        
        # Add deposits/withdrawals if available and requested
        if include_deposits and "deposit_amount" in df.columns:
            df["cumulative_balance"] = df["cumulative_pnl"] + df["deposit_amount"].cumsum()
        else:
            df["cumulative_balance"] = df["cumulative_pnl"]
        
        # Create figure
        fig = go.Figure()
        
        # Add equity curve line
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["cumulative_balance"],
                mode="lines",
                name="Equity",
                line=dict(color="#2E7DF7", width=2)
            )
        )
        
        # Add trade markers
        wins = df[df["pnl_dollars"] > 0]
        losses = df[df["pnl_dollars"] <= 0]
        
        fig.add_trace(
            go.Scatter(
                x=wins["date"],
                y=wins["cumulative_balance"],
                mode="markers",
                name="Winning Trades",
                marker=dict(color="#00CC96", size=8, symbol="circle"),
                hovertemplate="<b>%{customdata[0]}</b><br>" +
                             "Date: %{x|%Y-%m-%d %H:%M}<br>" +
                             "P&L: $%{customdata[1]:.2f}<br>" +
                             "Equity: $%{y:.2f}",
                customdata=list(zip(wins["symbol"], wins["pnl_dollars"]))
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=losses["date"],
                y=losses["cumulative_balance"],
                mode="markers",
                name="Losing Trades",
                marker=dict(color="#EF553B", size=8, symbol="circle"),
                hovertemplate="<b>%{customdata[0]}</b><br>" +
                             "Date: %{x|%Y-%m-%d %H:%M}<br>" +
                             "P&L: $%{customdata[1]:.2f}<br>" +
                             "Equity: $%{y:.2f}",
                customdata=list(zip(losses["symbol"], losses["pnl_dollars"]))
            )
        )
        
        # Add drawdown shading if we have enough data
        if len(df) > 1:
            # Calculate running maximum
            df["max_equity"] = df["cumulative_balance"].cummax()
            df["drawdown"] = df["cumulative_balance"] - df["max_equity"]
            
            # Add drawdown shading
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["max_equity"],
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="none"
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["cumulative_balance"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(231,107,243,0.2)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="none"
                )
            )
        
        # Configure layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Account Equity ($)",
            template=self.template,
            hovermode="closest",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add range selector
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        # Save to file if filename is provided
        if filename and self.output_dir:
            output_path = os.path.join(self.output_dir, f"{filename}.html")
            fig.write_html(output_path, include_plotlyjs="cdn")
        
        return fig

    def win_loss_ratio_chart(self,
                           trades: List[Dict[str, Any]],
                           count: int = 50,
                           title: str = "Win/Loss Ratio",
                           filename: str = None) -> Optional[go.Figure]:
        """
        Generate a win/loss ratio chart.
        
        Args:
            trades: List of trade dictionaries with trade details
            count: Number of most recent trades to include
            title: Chart title
            filename: Filename to save chart (without extension)
            
        Returns:
            Plotly figure object (or None if no data)
        """
        if not trades:
            return None
        
        # Convert trades to DataFrame
        df = pd.DataFrame(trades)
        
        # Ensure we have required columns
        required_cols = ["timestamp", "pnl_dollars"]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing required columns in trade data: {missing}")
        
        # Convert timestamp to datetime
        df["date"] = pd.to_datetime(df["timestamp"])
        
        # Sort by date and take most recent trades
        df = df.sort_values("date", ascending=False).head(count)
        
        if df.empty:
            return None
        
        # Calculate win/loss counts
        win_count = (df["pnl_dollars"] > 0).sum()
        loss_count = (df["pnl_dollars"] <= 0).sum()
        total_count = len(df)
        
        win_pct = win_count / total_count * 100 if total_count > 0 else 0
        
        # Create figure
        fig = go.Figure()
        
        # Create a pie chart
        fig.add_trace(
            go.Pie(
                labels=["Wins", "Losses"],
                values=[win_count, loss_count],
                hole=0.5,
                marker_colors=["#00CC96", "#EF553B"],
                textinfo="percent+value",
                hoverinfo="label+percent+value",
                textfont_size=14
            )
        )
        
        # Configure layout
        fig.update_layout(
            title=f"{title} (Last {count} Trades)",
            annotations=[
                dict(
                    text=f"Win Rate<br>{win_pct:.1f}%",
                    x=0.5, y=0.5,
                    font_size=20,
                    showarrow=False
                )
            ],
            template=self.template
        )
        
        # Save to file if filename is provided
        if filename and self.output_dir:
            output_path = os.path.join(self.output_dir, f"{filename}.html")
            fig.write_html(output_path, include_plotlyjs="cdn")
        
        return fig

    def strategy_performance_chart(self,
                                 strategy_metrics: Dict[str, Dict[str, Any]],
                                 top_count: int = 5,
                                 metric: str = "win_rate",
                                 title: str = "Strategy Performance",
                                 filename: str = None) -> Optional[go.Figure]:
        """
        Generate a strategy performance comparison chart.
        
        Args:
            strategy_metrics: Dictionary mapping strategy names to metrics
            top_count: Number of top strategies to include
            metric: Metric to sort by (win_rate, profit_factor, etc.)
            title: Chart title
            filename: Filename to save chart (without extension)
            
        Returns:
            Plotly figure object (or None if no data)
        """
        if not strategy_metrics:
            return None
        
        # Convert to DataFrame
        data = []
        for strategy, metrics in strategy_metrics.items():
            # Convert metrics to float where possible
            metrics_dict = {k: float(v) if isinstance(v, (int, float)) else v 
                          for k, v in metrics.items()}
            metrics_dict["strategy"] = strategy
            data.append(metrics_dict)
        
        df = pd.DataFrame(data)
        
        # Ensure we have required columns
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in strategy data")
        
        # Sort by the specified metric and take top N
        df = df.sort_values(metric, ascending=False).head(top_count)
        
        if df.empty:
            return None
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Win Rate (%)", "Profit Factor"],
            vertical_spacing=0.2,
            row_heights=[0.6, 0.4]
        )
        
        # Add win rate bars
        if "win_rate" in df.columns:
            win_rates = df["win_rate"] * 100 if df["win_rate"].max() <= 1 else df["win_rate"]
            
            fig.add_trace(
                go.Bar(
                    x=df["strategy"],
                    y=win_rates,
                    name="Win Rate",
                    marker_color="#2E7DF7",
                    text=win_rates.round(1).astype(str) + "%",
                    textposition="auto"
                ),
                row=1, col=1
            )
        
        # Add profit factor bars
        if "profit_factor" in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df["strategy"],
                    y=df["profit_factor"],
                    name="Profit Factor",
                    marker_color="#00CC96",
                    text=df["profit_factor"].round(2).astype(str),
                    textposition="auto"
                ),
                row=2, col=1
            )
        
        # Configure layout
        fig.update_layout(
            title=title,
            showlegend=False,
            template=self.template,
            height=600
        )
        
        # Save to file if filename is provided
        if filename and self.output_dir:
            output_path = os.path.join(self.output_dir, f"{filename}.html")
            fig.write_html(output_path, include_plotlyjs="cdn")
        
        return fig

    def pnl_distribution_chart(self,
                             trades: List[Dict[str, Any]],
                             bins: int = 20,
                             title: str = "P&L Distribution",
                             filename: str = None) -> Optional[go.Figure]:
        """
        Generate a P&L distribution histogram.
        
        Args:
            trades: List of trade dictionaries with trade details
            bins: Number of bins for the histogram
            title: Chart title
            filename: Filename to save chart (without extension)
            
        Returns:
            Plotly figure object (or None if no data)
        """
        if not trades:
            return None
        
        # Convert trades to DataFrame
        df = pd.DataFrame(trades)
        
        # Ensure we have required columns
        if "pnl_dollars" not in df.columns:
            raise ValueError("Missing required column 'pnl_dollars' in trade data")
        
        if df.empty:
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Create histogram
        fig.add_trace(
            go.Histogram(
                x=df["pnl_dollars"],
                nbinsx=bins,
                marker_color="#2E7DF7",
                opacity=0.7,
                name="P&L Distribution"
            )
        )
        
        # Add mean and median lines
        mean_pnl = df["pnl_dollars"].mean()
        median_pnl = df["pnl_dollars"].median()
        
        fig.add_vline(
            x=0,
            line_width=2,
            line_dash="dot",
            line_color="white",
            annotation_text="Breakeven",
            annotation_position="top"
        )
        
        fig.add_vline(
            x=mean_pnl,
            line_width=2,
            line_color="#00CC96",
            annotation_text=f"Mean: ${mean_pnl:.2f}",
            annotation_position="top right"
        )
        
        fig.add_vline(
            x=median_pnl,
            line_width=2,
            line_color="#EF553B",
            annotation_text=f"Median: ${median_pnl:.2f}",
            annotation_position="top left"
        )
        
        # Configure layout
        fig.update_layout(
            title=title,
            xaxis_title="P&L ($)",
            yaxis_title="Number of Trades",
            template=self.template,
            bargap=0.1
        )
        
        # Save to file if filename is provided
        if filename and self.output_dir:
            output_path = os.path.join(self.output_dir, f"{filename}.html")
            fig.write_html(output_path, include_plotlyjs="cdn")
        
        return fig

    def drawdown_chart(self,
                     trades: List[Dict[str, Any]],
                     days: int = 90,
                     title: str = "Drawdown Analysis",
                     filename: str = None) -> Optional[go.Figure]:
        """
        Generate a drawdown analysis chart.
        
        Args:
            trades: List of trade dictionaries with trade details
            days: Number of days to include in the chart
            title: Chart title
            filename: Filename to save chart (without extension)
            
        Returns:
            Plotly figure object (or None if no data)
        """
        if not trades:
            return None
        
        # Convert trades to DataFrame
        df = pd.DataFrame(trades)
        
        # Ensure we have required columns
        required_cols = ["timestamp", "pnl_dollars"]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing required columns in trade data: {missing}")
        
        # Convert timestamp to datetime
        df["date"] = pd.to_datetime(df["timestamp"])
        
        # Filter to specified time range
        start_date = datetime.now() - timedelta(days=days)
        df = df[df["date"] >= start_date]
        
        if df.empty:
            return None
        
        # Sort by date
        df = df.sort_values("date")
        
        # Calculate cumulative P&L
        df["cumulative_pnl"] = df["pnl_dollars"].cumsum()
        
        # Calculate drawdown metrics
        df["max_equity"] = df["cumulative_pnl"].cummax()
        df["drawdown"] = df["cumulative_pnl"] - df["max_equity"]
        df["drawdown_pct"] = df["drawdown"] / df["max_equity"].replace(0, np.nan)
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Equity Curve with Drawdowns", "Drawdown Percentage"],
            vertical_spacing=0.1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3]
        )
        
        # Add equity curve to top subplot
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["cumulative_pnl"],
                mode="lines",
                name="Equity",
                line=dict(color="#2E7DF7", width=2)
            ),
            row=1, col=1
        )
        
        # Add max equity line
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["max_equity"],
                mode="lines",
                name="Max Equity",
                line=dict(color="#00CC96", width=1, dash="dot")
            ),
            row=1, col=1
        )
        
        # Add drawdown shading
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["max_equity"],
                mode="lines",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="none"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["cumulative_pnl"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(231,107,243,0.2)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Drawdown",
                hoverinfo="none"
            ),
            row=1, col=1
        )
        
        # Add drawdown percentage to bottom subplot
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["drawdown_pct"] * 100,
                mode="lines",
                name="Drawdown %",
                line=dict(color="#EF553B", width=2)
            ),
            row=2, col=1
        )
        
        # Calculate and annotate max drawdown
        max_dd_idx = df["drawdown"].idxmin()
        max_dd = df.loc[max_dd_idx, "drawdown"]
        max_dd_pct = df.loc[max_dd_idx, "drawdown_pct"] * 100
        max_dd_date = df.loc[max_dd_idx, "date"]
        
        fig.add_annotation(
            x=max_dd_date,
            y=df.loc[max_dd_idx, "cumulative_pnl"],
            text=f"Max DD: {max_dd_pct:.1f}%<br>${max_dd:.2f}",
            showarrow=True,
            arrowhead=1,
            row=1, col=1
        )
        
        # Configure layout
        fig.update_layout(
            title=title,
            template=self.template,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=700
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        # Add range selector
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            row=1, col=1
        )
        
        # Save to file if filename is provided
        if filename and self.output_dir:
            output_path = os.path.join(self.output_dir, f"{filename}.html")
            fig.write_html(output_path, include_plotlyjs="cdn")
        
        return fig

    def time_analysis_chart(self,
                          trades: List[Dict[str, Any]],
                          analysis_type: str = "day_of_week",
                          title: str = None,
                          filename: str = None) -> Optional[go.Figure]:
        """
        Generate a time-based analysis chart (day of week, hour of day, etc.).
        
        Args:
            trades: List of trade dictionaries with trade details
            analysis_type: Type of analysis ("day_of_week", "hour_of_day", "month")
            title: Chart title (generated based on analysis_type if None)
            filename: Filename to save chart (without extension)
            
        Returns:
            Plotly figure object (or None if no data)
        """
        if not trades:
            return None
        
        # Convert trades to DataFrame
        df = pd.DataFrame(trades)
        
        # Ensure we have required columns
        required_cols = ["timestamp", "pnl_dollars"]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing required columns in trade data: {missing}")
        
        # Convert timestamp to datetime
        df["date"] = pd.to_datetime(df["timestamp"])
        
        if df.empty:
            return None
        
        # Create date-related columns based on analysis type
        if analysis_type == "day_of_week":
            df["category"] = df["date"].dt.day_name()
            sort_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            title = title or "Performance by Day of Week"
            x_title = "Day of Week"
        elif analysis_type == "hour_of_day":
            df["category"] = df["date"].dt.hour
            sort_order = list(range(24))
            title = title or "Performance by Hour of Day"
            x_title = "Hour of Day"
        elif analysis_type == "month":
            df["category"] = df["date"].dt.month_name()
            sort_order = ["January", "February", "March", "April", "May", "June", 
                        "July", "August", "September", "October", "November", "December"]
            title = title or "Performance by Month"
            x_title = "Month"
        else:
            raise ValueError(f"Invalid analysis_type: {analysis_type}")
        
        # Aggregate by category
        agg_data = df.groupby("category").agg(
            win_count=pd.NamedAgg(column="pnl_dollars", aggfunc=lambda x: (x > 0).sum()),
            loss_count=pd.NamedAgg(column="pnl_dollars", aggfunc=lambda x: (x <= 0).sum()),
            total_pnl=pd.NamedAgg(column="pnl_dollars", aggfunc="sum"),
            avg_win=pd.NamedAgg(column="pnl_dollars", aggfunc=lambda x: x[x > 0].mean()),
            avg_loss=pd.NamedAgg(column="pnl_dollars", aggfunc=lambda x: x[x <= 0].mean()),
            trade_count=pd.NamedAgg(column="pnl_dollars", aggfunc="count")
        )
        
        # Calculate win rate
        agg_data["win_rate"] = agg_data["win_count"] / agg_data["trade_count"] * 100
        
        # Sort by the specified order
        agg_data = agg_data.reindex(sort_order)
        
        # Only keep categories with data
        agg_data = agg_data.dropna()
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Win Rate (%)", "Total P&L ($)"],
            vertical_spacing=0.15,
            row_heights=[0.5, 0.5]
        )
        
        # Add win rate bars
        fig.add_trace(
            go.Bar(
                x=agg_data.index,
                y=agg_data["win_rate"],
                name="Win Rate",
                marker_color="#2E7DF7",
                text=[f"{x:.1f}%" for x in agg_data["win_rate"]],
                textposition="auto"
            ),
            row=1, col=1
        )
        
        # Add average horizontal line for win rate
        avg_win_rate = agg_data["win_rate"].mean()
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=avg_win_rate,
            x1=len(agg_data) - 0.5,
            y1=avg_win_rate,
            line=dict(color="white", width=1, dash="dash"),
            row=1, col=1
        )
        
        fig.add_annotation(
            x=len(agg_data) - 1,
            y=avg_win_rate,
            text=f"Avg: {avg_win_rate:.1f}%",
            showarrow=False,
            xanchor="right",
            row=1, col=1
        )
        
        # Add total P&L bars
        colors = ["#00CC96" if x >= 0 else "#EF553B" for x in agg_data["total_pnl"]]
        fig.add_trace(
            go.Bar(
                x=agg_data.index,
                y=agg_data["total_pnl"],
                name="Total P&L",
                marker_color=colors,
                text=[f"${x:.2f}" for x in agg_data["total_pnl"]],
                textposition="auto"
            ),
            row=2, col=1
        )
        
        # Configure layout
        fig.update_layout(
            title=title,
            template=self.template,
            showlegend=False,
            height=600
        )
        
        fig.update_xaxes(title_text=x_title, row=2, col=1)
        fig.update_yaxes(title_text="Win Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Total P&L ($)", row=2, col=1)
        
        # Save to file if filename is provided
        if filename and self.output_dir:
            output_path = os.path.join(self.output_dir, f"{filename}.html")
            fig.write_html(output_path, include_plotlyjs="cdn")
        
        return fig

    def trade_duration_chart(self,
                           trades: List[Dict[str, Any]],
                           title: str = "Trade Duration Analysis",
                           filename: str = None) -> Optional[go.Figure]:
        """
        Generate a trade duration analysis chart.
        
        Args:
            trades: List of trade dictionaries with trade details
            title: Chart title
            filename: Filename to save chart (without extension)
            
        Returns:
            Plotly figure object (or None if no data)
        """
        if not trades:
            return None
        
        # Convert trades to DataFrame
        df = pd.DataFrame(trades)
        
        # Check for duration column
        if "duration" not in df.columns:
            if all(col in df.columns for col in ["entry_time", "exit_time"]):
                # Calculate duration from timestamps
                df["entry_dt"] = pd.to_datetime(df["entry_time"])
                df["exit_dt"] = pd.to_datetime(df["exit_time"])
                df["duration_hours"] = (df["exit_dt"] - df["entry_dt"]).dt.total_seconds() / 3600
            else:
                raise ValueError("Missing duration information in trade data")
        else:
            # Extract numeric duration from string (e.g., "2.5h" -> 2.5)
            if df["duration"].dtype == object:
                df["duration_hours"] = df["duration"].str.replace("h", "").astype(float)
            else:
                df["duration_hours"] = df["duration"]
        
        if df.empty:
            return None
        
        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Duration Distribution", "Duration vs. P&L"],
            horizontal_spacing=0.1,
            column_widths=[0.4, 0.6]
        )
        
        # Add duration histogram
        fig.add_trace(
            go.Histogram(
                x=df["duration_hours"],
                name="Duration",
                marker_color="#2E7DF7",
                opacity=0.7,
                nbinsx=20
            ),
            row=1, col=1
        )
        
        # Add duration vs. P&L scatter plot
        # Color by win/loss
        colors = ["#00CC96" if x > 0 else "#EF553B" for x in df["pnl_dollars"]]
        hover_text = [
            f"Symbol: {row.get('symbol', 'N/A')}<br>"
            f"Duration: {row['duration_hours']:.1f}h<br>"
            f"P&L: ${row['pnl_dollars']:.2f}<br>"
            f"Strategy: {row.get('strategy_name', 'N/A')}"
            for _, row in df.iterrows()
        ]
        
        fig.add_trace(
            go.Scatter(
                x=df["duration_hours"],
                y=df["pnl_dollars"],
                mode="markers",
                name="Trades",
                marker=dict(
                    color=colors,
                    size=10,
                    opacity=0.7
                ),
                text=hover_text,
                hoverinfo="text"
            ),
            row=1, col=2
        )
        
        # Add zero line
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=df["duration_hours"].max() * 1.1,
            y1=0,
            line=dict(color="white", width=1, dash="dash"),
            row=1, col=2
        )
        
        # Configure layout
        fig.update_layout(
            title=title,
            template=self.template,
            showlegend=False,
            height=500
        )
        
        fig.update_xaxes(title_text="Duration (hours)", row=1, col=1)
        fig.update_yaxes(title_text="Number of Trades", row=1, col=1)
        
        fig.update_xaxes(title_text="Duration (hours)", row=1, col=2)
        fig.update_yaxes(title_text="P&L ($)", row=1, col=2)
        
        # Save to file if filename is provided
        if filename and self.output_dir:
            output_path = os.path.join(self.output_dir, f"{filename}.html")
            fig.write_html(output_path, include_plotlyjs="cdn")
        
        return fig

    def generate_dashboard_charts(self,
                               trades: List[Dict[str, Any]],
                               metrics: Dict[str, Any],
                               output_prefix: str = "dashboard_",
                               save_files: bool = True) -> Dict[str, go.Figure]:
        """
        Generate a complete set of dashboard charts.
        
        Args:
            trades: List of trade dictionaries with trade details
            metrics: Dictionary with metrics data including strategy_performance
            output_prefix: Prefix for output filenames
            save_files: Whether to save charts to files
            
        Returns:
            Dictionary mapping chart names to Plotly figures
        """
        charts = {}
        
        try:
            # Equity curve chart
            charts["equity_curve"] = self.equity_curve_chart(
                trades,
                days=30,
                title="Equity Curve (30 Days)",
                filename=f"{output_prefix}equity_curve" if save_files else None
            )
            
            # Win/loss ratio chart
            charts["win_loss"] = self.win_loss_ratio_chart(
                trades,
                count=50,
                title="Win/Loss Ratio (Last 50 Trades)",
                filename=f"{output_prefix}win_loss" if save_files else None
            )
            
            # Strategy performance chart
            if "strategy_performance" in metrics:
                charts["strategy_performance"] = self.strategy_performance_chart(
                    metrics["strategy_performance"],
                    top_count=5,
                    title="Top 5 Strategies by Win Rate",
                    filename=f"{output_prefix}strategy_performance" if save_files else None
                )
            
            # P&L distribution chart
            charts["pnl_distribution"] = self.pnl_distribution_chart(
                trades,
                title="P&L Distribution",
                filename=f"{output_prefix}pnl_distribution" if save_files else None
            )
            
            # Drawdown chart
            charts["drawdown"] = self.drawdown_chart(
                trades,
                days=90,
                title="Drawdown Analysis (90 Days)",
                filename=f"{output_prefix}drawdown" if save_files else None
            )
            
            # Day of week analysis
            charts["day_of_week"] = self.time_analysis_chart(
                trades,
                analysis_type="day_of_week",
                title="Performance by Day of Week",
                filename=f"{output_prefix}day_of_week" if save_files else None
            )
            
            # Trade duration analysis
            charts["trade_duration"] = self.trade_duration_chart(
                trades,
                title="Trade Duration Analysis",
                filename=f"{output_prefix}trade_duration" if save_files else None
            )
            
        except Exception as e:
            import logging
            logging.getLogger("DashboardCharts").error(f"Error generating charts: {e}")
        
        return {k: v for k, v in charts.items() if v is not None}

# Example usage
if __name__ == "__main__":
    import random
    
    # Generate sample trade data
    sample_trades = []
    start_date = datetime.now() - timedelta(days=90)
    equity = 10000.0
    
    for i in range(100):
        date = start_date + timedelta(days=i * 0.5 + random.random())
        symbol = random.choice(["AAPL", "MSFT", "GOOGL", "AMZN", "FB"])
        strategy = random.choice(["trend_following", "mean_reversion", "breakout", "momentum"])
        
        # Generate random P&L with slight bias towards profits
        pnl = random.normalvariate(50, 200)
        equity += pnl
        
        # Generate random duration between 1 and 24 hours
        duration = random.uniform(1, 24)
        
        trade = {
            "trade_id": f"trade_{i}",
            "timestamp": date.isoformat(),
            "symbol": symbol,
            "strategy_name": strategy,
            "entry_price": random.uniform(100, 200),
            "exit_price": random.uniform(100, 200),
            "pnl_dollars": pnl,
            "is_win": pnl > 0,
            "duration": f"{duration:.1f}h",
            "entry_time": date.isoformat(),
            "exit_time": (date + timedelta(hours=duration)).isoformat()
        }
        
        sample_trades.append(trade)
    
    # Generate sample metrics
    sample_metrics = {
        "strategy_performance": {
            "trend_following": {
                "win_rate": 0.65,
                "profit_factor": 1.8,
                "expectancy": 0.45,
                "avg_win": 250,
                "avg_loss": -150
            },
            "mean_reversion": {
                "win_rate": 0.72,
                "profit_factor": 2.1,
                "expectancy": 0.6,
                "avg_win": 180,
                "avg_loss": -120
            },
            "breakout": {
                "win_rate": 0.45,
                "profit_factor": 1.4,
                "expectancy": 0.25,
                "avg_win": 400,
                "avg_loss": -200
            },
            "momentum": {
                "win_rate": 0.58,
                "profit_factor": 1.6,
                "expectancy": 0.35,
                "avg_win": 300,
                "avg_loss": -180
            }
        }
    }
    
    # Create chart generator
    charts = DashboardCharts(output_dir="charts", dark_mode=True)
    
    # Generate dashboard charts
    dashboard_charts = charts.generate_dashboard_charts(
        sample_trades,
        sample_metrics
    )
    
    print(f"Generated {len(dashboard_charts)} charts")
    for name, fig in dashboard_charts.items():
        print(f"- {name}")
    
    # Open one of the charts in browser
    if "equity_curve" in dashboard_charts:
        dashboard_charts["equity_curve"].show() 