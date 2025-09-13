#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Optimization UI Component

This module provides UI components for displaying strategy optimization results
and tracking the optimization process in the dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import json

# Import optimization tracking
try:
    from trading_bot.event_system.strategy_optimization_handlers import get_optimization_tracker
    from trading_bot.event_system import EventBus
    optimization_tracking_available = True
except ImportError:
    optimization_tracking_available = False
    
logger = logging.getLogger(__name__)

class OptimizationUI:
    """UI component for displaying strategy optimization information"""
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """Initialize the optimization UI component"""
        self.event_bus = event_bus
        
        # Get optimization tracker if available
        if optimization_tracking_available:
            self.tracker = get_optimization_tracker(event_bus)
        else:
            self.tracker = None
            
        # Initialize state
        if "optimization_ui_initialized" not in st.session_state:
            st.session_state.optimization_ui_initialized = True
            st.session_state.selected_strategy_id = None
            st.session_state.optimization_view = "summary"  # or "details"
            
    def render(self):
        """Render the optimization UI"""
        st.markdown("## Strategy Optimization")
        
        # Check if optimization tracking is available
        if not optimization_tracking_available:
            st.warning("Strategy optimization tracking is not available. Some features may be limited.")
            self._render_mock_ui()
            return
            
        # Get optimization data
        optimization_data = self.tracker.get_optimization_data()
        summary = self.tracker.get_optimization_summary()
        
        # Render summary metrics
        self._render_summary_metrics(summary)
        
        # Tab selection for different views
        tab1, tab2 = st.tabs(["Optimization Progress", "Optimization Details"])
        
        with tab1:
            self._render_optimization_progress(optimization_data)
            
        with tab2:
            self._render_optimization_details(optimization_data)
    
    def _render_summary_metrics(self, summary: Dict[str, Any]):
        """Render summary metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Strategies", summary.get("total", 0))
            
        with col2:
            st.metric("Optimized", summary.get("optimized", 0))
            
        with col3:
            st.metric("Exhausted", summary.get("exhausted", 0))
            
        with col4:
            st.metric("Pending", summary.get("pending", 0))
    
    def _render_optimization_progress(self, optimization_data: Dict[str, Any]):
        """Render optimization progress charts and tables"""
        if not optimization_data:
            st.info("No optimization data available yet. Start the autonomous engine to generate strategies and optimization results.")
            return
            
        # Create dataframe for optimization progress
        progress_data = []
        for strategy_id, data in optimization_data.items():
            progress_data.append({
                "strategy_id": strategy_id,
                "status": data.get("status", "pending"),
                "strategy_type": self._extract_strategy_type(data),
                "last_updated": data.get("last_updated", ""),
                "performance_before": self._get_initial_performance(data),
                "performance_after": self._get_final_performance(data)
            })
            
        progress_df = pd.DataFrame(progress_data)
        
        # Plot optimization status distribution
        if not progress_df.empty:
            status_counts = progress_df["status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            
            fig = px.pie(
                status_counts, 
                values="Count", 
                names="Status",
                color="Status",
                color_discrete_map={
                    "optimized": "#28a745",
                    "exhausted": "#dc3545",
                    "pending": "#ffc107"
                },
                title="Optimization Status Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show performance improvement for optimized strategies
            optimized_df = progress_df[progress_df["status"] == "optimized"].copy()
            if not optimized_df.empty:
                st.subheader("Performance Improvement for Optimized Strategies")
                
                # Calculate improvement percentage
                optimized_df["improvement"] = optimized_df.apply(
                    lambda x: self._calculate_improvement(
                        x["performance_before"], 
                        x["performance_after"]
                    ), 
                    axis=1
                )
                
                # Show table of improvements
                st.dataframe(
                    optimized_df[["strategy_id", "strategy_type", "improvement", "last_updated"]],
                    use_container_width=True
                )
                
                # Plot performance improvement
                if len(optimized_df) > 0:
                    fig = go.Figure()
                    
                    for i, row in optimized_df.iterrows():
                        before = row["performance_before"]
                        after = row["performance_after"]
                        
                        metric_names = list(before.keys())
                        before_values = list(before.values())
                        after_values = list(after.values())
                        
                        fig.add_trace(go.Bar(
                            x=metric_names,
                            y=before_values,
                            name=f"Before ({row['strategy_id']})",
                            marker_color="#ff7f0e"
                        ))
                        
                        fig.add_trace(go.Bar(
                            x=metric_names,
                            y=after_values,
                            name=f"After ({row['strategy_id']})",
                            marker_color="#2ca02c"
                        ))
                    
                    fig.update_layout(
                        barmode="group",
                        title="Performance Before vs After Optimization",
                        xaxis_title="Metric",
                        yaxis_title="Value",
                        legend_title="Strategy"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No strategies have been optimized yet.")
    
    def _render_optimization_details(self, optimization_data: Dict[str, Any]):
        """Render detailed optimization information for a selected strategy"""
        if not optimization_data:
            st.info("No optimization data available yet.")
            return
            
        # Strategy selector
        strategy_ids = list(optimization_data.keys())
        if not strategy_ids:
            st.info("No strategy optimization data available yet.")
            return
            
        selected_strategy_id = st.selectbox(
            "Select Strategy",
            strategy_ids,
            key="optimization_details_strategy_selector"
        )
        
        if not selected_strategy_id:
            st.info("Select a strategy to view optimization details.")
            return
            
        strategy_data = optimization_data.get(selected_strategy_id, {})
        
        # Display strategy details
        st.subheader(f"Strategy {selected_strategy_id} Optimization Details")
        
        # Status and basic info
        status = strategy_data.get("status", "pending")
        status_color = {
            "optimized": "🟢 Optimized",
            "exhausted": "🔴 Exhausted",
            "pending": "🟡 Pending"
        }.get(status, status)
        
        st.markdown(f"**Status:** {status_color}")
        st.markdown(f"**Last Updated:** {strategy_data.get('last_updated', 'N/A')}")
        
        # Display parameters before and after optimization
        if status == "optimized":
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Parameters")
                original_params = strategy_data.get("original_parameters", {})
                if original_params:
                    st.json(original_params)
                else:
                    st.info("No original parameters data available.")
                    
            with col2:
                st.subheader("Optimized Parameters")
                optimized_params = strategy_data.get("optimized_parameters", {})
                if optimized_params:
                    st.json(optimized_params)
                else:
                    st.info("No optimized parameters data available.")
            
            # Show parameter changes
            st.subheader("Parameter Changes")
            changes = self._get_parameter_changes(
                strategy_data.get("original_parameters", {}),
                strategy_data.get("optimized_parameters", {})
            )
            
            if changes:
                changes_df = pd.DataFrame(changes)
                st.dataframe(changes_df, use_container_width=True)
            else:
                st.info("No parameter changes detected.")
                
            # Performance comparison
            st.subheader("Performance Comparison")
            perf_before = self._get_initial_performance(strategy_data)
            perf_after = self._get_final_performance(strategy_data)
            
            if perf_before and perf_after:
                perf_data = []
                
                for metric in perf_before:
                    if metric in perf_after:
                        perf_data.append({
                            "Metric": metric,
                            "Before": perf_before[metric],
                            "After": perf_after[metric],
                            "Change": perf_after[metric] - perf_before[metric],
                            "Percent Change": f"{(perf_after[metric] - perf_before[metric]) / max(abs(perf_before[metric]), 0.001) * 100:.2f}%"
                        })
                
                if perf_data:
                    perf_df = pd.DataFrame(perf_data)
                    st.dataframe(perf_df, use_container_width=True)
                else:
                    st.info("No performance comparison data available.")
            else:
                st.info("Performance data not available for comparison.")
                
        elif status == "exhausted":
            # Show parameters and thresholds
            st.subheader("Strategy Parameters")
            params = strategy_data.get("parameters", {})
            if params:
                st.json(params)
            else:
                st.info("No parameter data available.")
                
            st.subheader("Performance Thresholds")
            thresholds = strategy_data.get("thresholds", {})
            if thresholds:
                st.json(thresholds)
            else:
                st.info("No threshold data available.")
                
            # Show final performance vs thresholds
            st.subheader("Performance vs Thresholds")
            perf = strategy_data.get("performance", {})
            
            if perf and thresholds:
                gap_data = []
                
                for metric, threshold in thresholds.items():
                    perf_key = metric.replace("min_", "").replace("max_", "")
                    if perf_key in perf:
                        is_min = metric.startswith("min_")
                        is_max = metric.startswith("max_")
                        
                        actual = perf[perf_key]
                        gap = None
                        met = False
                        
                        if is_min:
                            gap = actual - threshold
                            met = actual >= threshold
                        elif is_max:
                            gap = threshold - actual
                            met = actual <= threshold
                        
                        if gap is not None:
                            gap_data.append({
                                "Metric": perf_key,
                                "Threshold": threshold,
                                "Actual": actual,
                                "Gap": gap,
                                "Threshold Met": "✓" if met else "✗"
                            })
                
                if gap_data:
                    gap_df = pd.DataFrame(gap_data)
                    st.dataframe(gap_df, use_container_width=True)
                    
                    # Plot the gaps
                    fig = go.Figure()
                    
                    for row in gap_data:
                        metric = row["Metric"]
                        threshold = row["Threshold"]
                        actual = row["Actual"]
                        
                        fig.add_trace(go.Bar(
                            x=[metric],
                            y=[threshold],
                            name=f"Threshold ({metric})",
                            marker_color="#ff7f0e"
                        ))
                        
                        fig.add_trace(go.Bar(
                            x=[metric],
                            y=[actual],
                            name=f"Actual ({metric})",
                            marker_color="#2ca02c"
                        ))
                    
                    fig.update_layout(
                        barmode="group",
                        title="Performance vs Thresholds",
                        xaxis_title="Metric",
                        yaxis_title="Value"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No comparable metrics available.")
            else:
                st.info("Performance or threshold data not available.")
        
        # Event history
        st.subheader("Optimization Event History")
        events = strategy_data.get("events", [])
        
        if events:
            events_df = pd.DataFrame([
                {
                    "Timestamp": e.get("timestamp", ""),
                    "Event Type": e.get("event_type", ""),
                    "Details": str(e.get("data", {}))
                }
                for e in events
            ])
            st.dataframe(events_df, use_container_width=True)
        else:
            st.info("No event history available.")
    
    def _render_mock_ui(self):
        """Render a mock UI when optimization tracking is not available"""
        # Mock summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Strategies", "5")
            
        with col2:
            st.metric("Optimized", "2")
            
        with col3:
            st.metric("Exhausted", "1")
            
        with col4:
            st.metric("Pending", "2")
            
        # Mock tabs
        tab1, tab2 = st.tabs(["Optimization Progress", "Optimization Details"])
        
        with tab1:
            st.info("This is a mock UI. Connect to real optimization events to see actual data.")
            
            # Mock status chart
            mock_data = pd.DataFrame({
                "Status": ["optimized", "exhausted", "pending"],
                "Count": [2, 1, 2]
            })
            
            fig = px.pie(
                mock_data, 
                values="Count", 
                names="Status",
                color="Status",
                color_discrete_map={
                    "optimized": "#28a745",
                    "exhausted": "#dc3545",
                    "pending": "#ffc107"
                },
                title="Optimization Status Distribution (Mock Data)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Mock performance table
            mock_perf = pd.DataFrame({
                "strategy_id": ["strategy_001", "strategy_002"],
                "strategy_type": ["MeanReversion", "Momentum"],
                "improvement": ["Sharpe: +0.35, Win Rate: +5.2%", "Sharpe: +0.22, Win Rate: +3.1%"],
                "last_updated": ["2025-04-25 10:15:23", "2025-04-25 10:32:41"]
            })
            
            st.subheader("Performance Improvement for Optimized Strategies (Mock Data)")
            st.dataframe(mock_perf, use_container_width=True)
            
        with tab2:
            st.info("Select a strategy to view optimization details (mock data).")
            st.selectbox(
                "Select Strategy",
                ["strategy_001", "strategy_002", "strategy_003", "strategy_004", "strategy_005"],
                key="mock_strategy_selector"
            )
            
            st.subheader("Strategy Optimization Details (Mock Data)")
            st.markdown("**Status:** 🟢 Optimized")
            st.markdown("**Last Updated:** 2025-04-25 10:15:23")
            
            # Mock parameter comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Parameters")
                st.json({
                    "lookback_period": 20,
                    "entry_threshold": 2.0,
                    "exit_threshold": 1.0,
                    "stop_loss_pct": 5.0
                })
                
            with col2:
                st.subheader("Optimized Parameters")
                st.json({
                    "lookback_period": 25,
                    "entry_threshold": 1.8,
                    "exit_threshold": 1.2,
                    "stop_loss_pct": 4.5
                })
    
    def _extract_strategy_type(self, data: Dict[str, Any]) -> str:
        """Extract the strategy type from optimization data"""
        # Try to get from events
        events = data.get("events", [])
        for event in events:
            event_data = event.get("data", {})
            if "strategy_type" in event_data:
                return event_data["strategy_type"]
        
        # Fallback
        return "Unknown"
    
    def _get_initial_performance(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Get the initial performance metrics before optimization"""
        events = data.get("events", [])
        
        # Look for the first event with performance data
        for event in events:
            event_data = event.get("data", {})
            if "performance" in event_data:
                return event_data["performance"]
                
        # Fallback to empty dict
        return {}
    
    def _get_final_performance(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Get the final performance metrics after optimization"""
        # Get directly from optimization data if available
        if "performance" in data:
            return data["performance"]
            
        # Otherwise look for the last event with performance data
        events = data.get("events", [])
        events.reverse()  # Start from the most recent
        
        for event in events:
            event_data = event.get("data", {})
            if "performance" in event_data:
                return event_data["performance"]
                
        # Fallback to empty dict
        return {}
    
    def _calculate_improvement(self, before: Dict[str, float], after: Dict[str, float]) -> str:
        """Calculate and format the improvement between before and after metrics"""
        improvements = []
        
        # Check key metrics
        if "sharpe_ratio" in before and "sharpe_ratio" in after:
            sharpe_diff = after["sharpe_ratio"] - before["sharpe_ratio"]
            improvements.append(f"Sharpe: {sharpe_diff:+.2f}")
            
        if "win_rate" in before and "win_rate" in after:
            win_diff = after["win_rate"] - before["win_rate"]
            improvements.append(f"Win Rate: {win_diff:+.1f}%")
            
        if "profit_factor" in before and "profit_factor" in after:
            pf_diff = after["profit_factor"] - before["profit_factor"]
            improvements.append(f"Profit Factor: {pf_diff:+.2f}")
        
        # Return formatted string
        return ", ".join(improvements) if improvements else "No data"
    
    def _get_parameter_changes(self, before: Dict[str, Any], after: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get a list of parameter changes between before and after"""
        changes = []
        
        # Check all parameters that exist in either dict
        all_params = set(before.keys()) | set(after.keys())
        
        for param in all_params:
            before_val = before.get(param, "N/A")
            after_val = after.get(param, "N/A")
            
            # Only include if there's a change
            if before_val != after_val:
                changes.append({
                    "Parameter": param,
                    "Before": before_val,
                    "After": after_val,
                })
                
        return changes
