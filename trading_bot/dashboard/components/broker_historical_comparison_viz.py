"""
Broker Historical Performance Panel - Comparison Visualization Component

This component provides comparison visualization for historical broker performance data,
including side-by-side metrics, performance rankings, and correlation analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from trading_bot.dashboard.components.broker_historical_panel_base import format_metric_name


def calculate_summary_statistics(
    df: pd.DataFrame,
    metric_name: str
) -> Dict[str, float]:
    """
    Calculate summary statistics for a metric
    
    Args:
        df: DataFrame with performance data
        metric_name: Name of metric to analyze
        
    Returns:
        Dict with summary statistics
    """
    if df.empty or metric_name not in df.columns:
        return {}
    
    # Calculate statistics
    mean = df[metric_name].mean()
    median = df[metric_name].median()
    std = df[metric_name].std()
    min_val = df[metric_name].min()
    max_val = df[metric_name].max()
    latest = df[metric_name].iloc[-1] if len(df) > 0 else None
    
    # Calculate percentiles
    p25 = df[metric_name].quantile(0.25)
    p75 = df[metric_name].quantile(0.75)
    p90 = df[metric_name].quantile(0.90)
    p99 = df[metric_name].quantile(0.99)
    
    # Calculate volatility (coefficient of variation)
    cv = std / mean if mean != 0 else 0
    
    return {
        "mean": mean,
        "median": median,
        "std": std,
        "min": min_val,
        "max": max_val,
        "latest": latest,
        "p25": p25,
        "p75": p75,
        "p90": p90,
        "p99": p99,
        "cv": cv
    }


def render_broker_comparison(
    data: Dict[str, pd.DataFrame],
    broker_names: Dict[str, str]
):
    """
    Render broker comparison visualization
    
    Args:
        data: Dict mapping broker_id to DataFrame of performance data
        broker_names: Dict mapping broker_id to display name
    """
    # Get common metrics across all brokers
    common_metrics = set()
    first = True
    
    for broker_id, df in data.items():
        numeric_cols = [col for col in df.columns if isinstance(df[col].dtype, (np.float64, np.int64)) or df[col].dtype in [float, int]]
        
        if first:
            common_metrics = set(numeric_cols)
            first = False
        else:
            common_metrics = common_metrics.intersection(numeric_cols)
    
    if not common_metrics:
        st.warning("No common metrics found across selected brokers")
        return
    
    # Metric selector
    selected_metrics = st.multiselect(
        "Select Metrics to Compare",
        options=sorted(common_metrics),
        default=sorted(common_metrics)[:min(3, len(common_metrics))],
        format_func=format_metric_name
    )
    
    if not selected_metrics:
        st.info("Please select at least one metric to compare")
        return
    
    # Calculate summary statistics for each broker
    comparison_data = {}
    
    for broker_id, df in data.items():
        broker_name = broker_names.get(broker_id, broker_id)
        comparison_data[broker_id] = {
            "name": broker_name,
            "metrics": {}
        }
        
        for metric in selected_metrics:
            if metric in df.columns:
                comparison_data[broker_id]["metrics"][metric] = calculate_summary_statistics(df, metric)
    
    # Visualization tabs
    tabs = st.tabs(["Side-by-Side Comparison", "Radar Chart", "Performance Ranking"])
    
    # Tab 1: Side-by-Side Comparison
    with tabs[0]:
        # Create comparison table
        for metric in selected_metrics:
            st.subheader(f"Comparison: {format_metric_name(metric)}")
            
            # Prepare data for table
            table_data = []
            for broker_id, broker_data in comparison_data.items():
                if metric in broker_data["metrics"]:
                    stats = broker_data["metrics"][metric]
                    
                    table_data.append({
                        "Broker": broker_data["name"],
                        "Latest": f"{stats['latest']:.2f}" if stats['latest'] is not None else "N/A",
                        "Mean": f"{stats['mean']:.2f}",
                        "Median": f"{stats['median']:.2f}",
                        "Min": f"{stats['min']:.2f}",
                        "Max": f"{stats['max']:.2f}",
                        "Std Dev": f"{stats['std']:.2f}",
                        "CV (%)": f"{stats['cv'] * 100:.1f}%"
                    })
            
            if table_data:
                st.table(pd.DataFrame(table_data))
            
            # Create box plot
            box_data = []
            for broker_id, df in data.items():
                if metric in df.columns:
                    broker_name = broker_names.get(broker_id, broker_id)
                    
                    for _, row in df.iterrows():
                        box_data.append({
                            "Broker": broker_name,
                            "Value": row[metric]
                        })
            
            if box_data:
                box_df = pd.DataFrame(box_data)
                
                fig = px.box(
                    box_df,
                    x="Broker",
                    y="Value",
                    title=f"Distribution of {format_metric_name(metric)} by Broker",
                    labels={"Value": format_metric_name(metric)},
                    color="Broker"
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Radar Chart
    with tabs[1]:
        if len(selected_metrics) >= 3:
            # Create radar chart data
            radar_data = []
            
            for broker_id, broker_data in comparison_data.items():
                radar_values = []
                
                for metric in selected_metrics:
                    if metric in broker_data["metrics"]:
                        # Normalize value for radar chart
                        # For some metrics (like latency, errors), lower is better
                        # For others (like availability, execution quality), higher is better
                        
                        # Default to mean for comparison
                        value = broker_data["metrics"][metric]["mean"]
                        
                        # Convert to 0-1 scale with appropriate direction
                        # This is a simplification - could be improved with metric-specific logic
                        if "latency" in metric or "error" in metric or "slippage" in metric or "commission" in metric:
                            # Lower is better, invert
                            value = 1.0 - (value / max(1.0, value * 2))  # Avoid division by zero
                        else:
                            # Higher is better, normalize
                            value = value / max(1.0, value * 1.5)  # Avoid division by zero
                        
                        radar_values.append(value)
                    else:
                        radar_values.append(0)
                
                radar_data.append({
                    "broker_id": broker_id,
                    "name": broker_data["name"],
                    "values": radar_values
                })
            
            # Create radar chart
            fig = go.Figure()
            
            # Add traces
            for item in radar_data:
                fig.add_trace(go.Scatterpolar(
                    r=item["values"],
                    theta=[format_metric_name(m) for m in selected_metrics],
                    fill='toself',
                    name=item["name"]
                ))
            
            # Update layout
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Broker Performance Comparison (Normalized)",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation
            st.info("""
            **Radar Chart Interpretation:**
            
            This chart shows normalized performance across multiple metrics where:
            - For metrics where lower is better (latency, error rate), values closer to the edge are better
            - For metrics where higher is better (availability), values closer to the edge are also better
            - The larger the area covered by a broker, the better its overall performance
            
            Note that normalization is relative to the displayed brokers.
            """)
        else:
            st.info("Select at least 3 metrics to generate a radar chart")
    
    # Tab 3: Performance Ranking
    with tabs[2]:
        # Create ranking data
        st.subheader("Broker Performance Ranking")
        
        # Create scorecards for each metric
        for metric in selected_metrics:
            st.write(f"### {format_metric_name(metric)}")
            
            # Collect data for ranking
            ranking_data = []
            for broker_id, broker_data in comparison_data.items():
                if metric in broker_data["metrics"]:
                    stats = broker_data["metrics"][metric]
                    
                    ranking_data.append({
                        "broker_id": broker_id,
                        "name": broker_data["name"],
                        "value": stats["mean"],
                        "latest": stats["latest"],
                        "cv": stats["cv"]
                    })
            
            if not ranking_data:
                continue
            
            # Sort by appropriate criterion - lower is better for some metrics
            reverse_order = not (
                "latency" in metric or 
                "error" in metric or 
                "slippage" in metric or 
                "commission" in metric
            )
            
            ranking_data.sort(key=lambda x: x["value"], reverse=reverse_order)
            
            # Create ranking table
            table_data = []
            for i, item in enumerate(ranking_data):
                # Determine ranking emoji
                if i == 0:
                    rank_emoji = "🥇"
                elif i == 1:
                    rank_emoji = "🥈"
                elif i == 2:
                    rank_emoji = "🥉"
                else:
                    rank_emoji = f"{i+1}."
                
                # Format comparative text
                if i > 0 and ranking_data[0]["value"] != 0:
                    if reverse_order:
                        # Higher is better
                        pct_diff = ((item["value"] / ranking_data[0]["value"]) - 1) * 100
                        if pct_diff < 0:
                            comparative = f"{abs(pct_diff):.1f}% worse than best"
                        else:
                            comparative = f"{pct_diff:.1f}% better than best"
                    else:
                        # Lower is better
                        pct_diff = ((item["value"] / ranking_data[0]["value"]) - 1) * 100
                        if pct_diff > 0:
                            comparative = f"{pct_diff:.1f}% worse than best"
                        else:
                            comparative = f"{abs(pct_diff):.1f}% better than best"
                else:
                    comparative = "Best performer"
                
                table_data.append({
                    "Rank": rank_emoji,
                    "Broker": item["name"],
                    "Average": f"{item['value']:.2f}",
                    "Latest": f"{item['latest']:.2f}" if item['latest'] is not None else "N/A",
                    "Consistency": f"{item['cv'] * 100:.1f}% CV",
                    "Comparison": comparative
                })
            
            st.table(pd.DataFrame(table_data))
        
        # Overall ranking
        st.write("### Overall Performance")
        st.info("This overall ranking assigns points based on performance across all selected metrics.")
        
        # Calculate overall points
        overall_points = {}
        for broker_id in comparison_data:
            overall_points[broker_id] = 0
        
        for metric in selected_metrics:
            # Collect values for this metric
            metric_values = []
            for broker_id, broker_data in comparison_data.items():
                if metric in broker_data["metrics"]:
                    metric_values.append({
                        "broker_id": broker_id,
                        "value": broker_data["metrics"][metric]["mean"]
                    })
            
            if not metric_values:
                continue
            
            # Sort by appropriate criterion - lower is better for some metrics
            reverse_order = not (
                "latency" in metric or 
                "error" in metric or 
                "slippage" in metric or 
                "commission" in metric
            )
            
            metric_values.sort(key=lambda x: x["value"], reverse=reverse_order)
            
            # Assign points (1st = n points, 2nd = n-1 points, etc.)
            points = len(metric_values)
            for item in metric_values:
                overall_points[item["broker_id"]] += points
                points -= 1
        
        # Create overall ranking table
        overall_data = []
        for broker_id, points in overall_points.items():
            overall_data.append({
                "broker_id": broker_id,
                "name": comparison_data[broker_id]["name"],
                "points": points
            })
        
        # Sort by points
        overall_data.sort(key=lambda x: x["points"], reverse=True)
        
        # Create table
        table_data = []
        for i, item in enumerate(overall_data):
            # Determine ranking emoji
            if i == 0:
                rank_emoji = "🏆"
            elif i == 1:
                rank_emoji = "🥈"
            elif i == 2:
                rank_emoji = "🥉"
            else:
                rank_emoji = f"{i+1}."
            
            table_data.append({
                "Rank": rank_emoji,
                "Broker": item["name"],
                "Points": item["points"],
                "Performance": "Best Overall" if i == 0 else ""
            })
        
        st.table(pd.DataFrame(table_data))


def render_broker_comparison_analysis(data_service=None):
    """
    Render broker comparison for historical performance
    
    Args:
        data_service: Optional DataService instance (not used here, 
                     as we're using data from session state)
    """
    # Check if data is available in session state
    if not hasattr(st.session_state, 'broker_historical_data') or not st.session_state.broker_historical_data:
        st.warning("Please load broker historical data first")
        return
    
    # Get data from session state
    data = st.session_state.broker_historical_data
    broker_names = st.session_state.broker_historical_names
    
    # Ensure multiple brokers are selected
    if len(data) < 2:
        st.warning("Please select at least two brokers for comparison")
        return
    
    # Render broker comparison
    render_broker_comparison(data, broker_names)


if __name__ == "__main__":
    # For local testing only
    render_broker_comparison_analysis()
