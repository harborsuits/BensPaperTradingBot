"""
Historical Backtest Results Visualization

This module provides visualizations for historical backtest results from the risk management system,
displaying performance across different market regimes and stress scenarios.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from dashboard.theme import COLORS
from dashboard.components import (
    section_header, 
    styled_metric_card, 
    format_currency, 
    format_percent,
    format_number
)
from dashboard.api_utils import get_mongo_collection

def historical_scenario_results(backtest_data: pd.DataFrame = None):
    """
    Display historical scenario results from backtests.
    
    Args:
        backtest_data: DataFrame containing backtest results
    """
    # If no data provided, fetch from database
    if backtest_data is None or backtest_data.empty:
        # Get backtest results data
        backtest_collection = get_mongo_collection("risk_backtest_results")
        if backtest_collection:
            backtest_data = pd.DataFrame(list(backtest_collection))
        else:
            backtest_data = pd.DataFrame()
    
    if backtest_data.empty:
        st.info("No historical backtest data available yet.")
        return
    
    # Display scenario results
    st.subheader("Historical Risk Scenario Performance")
    
    # Check if data has the expected format
    required_columns = ['scenario', 'start_date', 'end_date', 'performance', 'max_drawdown']
    if all(col in backtest_data.columns for col in required_columns):
        # Convert dates if they're strings
        for date_col in ['start_date', 'end_date']:
            if backtest_data[date_col].dtype == 'object':
                backtest_data[date_col] = pd.to_datetime(backtest_data[date_col])
        
        # Format data for display
        display_df = backtest_data[required_columns].copy()
        display_df['duration'] = (display_df['end_date'] - display_df['start_date']).dt.days
        
        # Format dates and metrics
        display_df['start_date'] = display_df['start_date'].dt.strftime('%Y-%m-%d')
        display_df['end_date'] = display_df['end_date'].dt.strftime('%Y-%m-%d')
        display_df['performance'] = display_df['performance'] * 100  # Convert to percentage
        display_df['max_drawdown'] = display_df['max_drawdown'] * 100  # Convert to percentage
        
        # Create a bar chart comparing performance across scenarios
        fig = px.bar(
            display_df,
            x='scenario',
            y='performance',
            color='max_drawdown',
            color_continuous_scale='RdYlGn_r',  # Reversed so that higher drawdowns are red
            hover_data=['start_date', 'end_date', 'duration'],
            labels={
                'scenario': 'Market Stress Scenario',
                'performance': 'Total Return (%)',
                'max_drawdown': 'Maximum Drawdown (%)',
                'start_date': 'Start Date',
                'end_date': 'End Date',
                'duration': 'Duration (days)'
            },
            title="Performance Across Historical Stress Scenarios"
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Market Stress Scenario",
            yaxis_title="Total Return (%)",
            coloraxis_colorbar_title="Max Drawdown (%)",
            height=400,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display best and worst scenarios
        col1, col2 = st.columns(2)
        
        with col1:
            best_scenario = display_df.loc[display_df['performance'].idxmax()]
            st.subheader("Best Performance Scenario")
            st.markdown(f"**Scenario:** {best_scenario['scenario']}")
            st.markdown(f"**Performance:** {best_scenario['performance']:.2f}%")
            st.markdown(f"**Max Drawdown:** {best_scenario['max_drawdown']:.2f}%")
            st.markdown(f"**Period:** {best_scenario['start_date']} to {best_scenario['end_date']}")
        
        with col2:
            worst_scenario = display_df.loc[display_df['performance'].idxmin()]
            st.subheader("Worst Performance Scenario")
            st.markdown(f"**Scenario:** {worst_scenario['scenario']}")
            st.markdown(f"**Performance:** {worst_scenario['performance']:.2f}%")
            st.markdown(f"**Max Drawdown:** {worst_scenario['max_drawdown']:.2f}%")
            st.markdown(f"**Period:** {worst_scenario['start_date']} to {worst_scenario['end_date']}")
        
        # Display detailed results in a table
        with st.expander("Detailed Scenario Results", expanded=False):
            # Rename columns for display
            display_df = display_df.rename(columns={
                'scenario': 'Scenario',
                'start_date': 'Start Date',
                'end_date': 'End Date',
                'performance': 'Return (%)',
                'max_drawdown': 'Max Drawdown (%)',
                'duration': 'Duration (days)'
            })
            
            st.dataframe(display_df, use_container_width=True)
    else:
        st.warning("Backtest data is not in the expected format.")

def risk_reduction_effectiveness(backtest_data: pd.DataFrame = None):
    """
    Display risk reduction effectiveness metrics.
    
    Args:
        backtest_data: DataFrame containing backtest results with risk reduction metrics
    """
    # If no data provided, fetch from database
    if backtest_data is None or backtest_data.empty:
        # Get risk reduction data
        reduction_collection = get_mongo_collection("risk_reduction_effectiveness")
        if reduction_collection:
            backtest_data = pd.DataFrame(list(reduction_collection))
        else:
            backtest_data = pd.DataFrame()
    
    if backtest_data.empty:
        st.info("No risk reduction effectiveness data available yet.")
        return
    
    # Display risk reduction metrics
    st.subheader("Risk Reduction Effectiveness")
    
    # Check if data has expected columns
    if 'scenario' in backtest_data.columns and 'base_drawdown' in backtest_data.columns and 'reduced_drawdown' in backtest_data.columns:
        # Calculate effectiveness
        backtest_data['reduction_percent'] = (backtest_data['base_drawdown'] - backtest_data['reduced_drawdown']) / backtest_data['base_drawdown'] * 100
        
        # Create bar chart comparing base vs reduced drawdowns
        chart_data = pd.melt(
            backtest_data,
            id_vars=['scenario'],
            value_vars=['base_drawdown', 'reduced_drawdown'],
            var_name='strategy',
            value_name='drawdown'
        )
        
        # Convert to percentages
        chart_data['drawdown'] = chart_data['drawdown'] * 100
        
        # Rename for display
        chart_data['strategy'] = chart_data['strategy'].replace({
            'base_drawdown': 'Without Risk Management',
            'reduced_drawdown': 'With Risk Management'
        })
        
        # Create grouped bar chart
        fig = px.bar(
            chart_data,
            x='scenario',
            y='drawdown',
            color='strategy',
            barmode='group',
            labels={
                'scenario': 'Market Stress Scenario',
                'drawdown': 'Maximum Drawdown (%)',
                'strategy': 'Strategy'
            },
            title="Drawdown Reduction by Risk Management",
            color_discrete_sequence=['#ff6b6b', '#51cf66']  # Red for base, green for reduced
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display average reduction
        avg_reduction = backtest_data['reduction_percent'].mean()
        
        styled_metric_card(
            "Average Drawdown Reduction", 
            f"{avg_reduction:.1f}%", 
            is_percent=True
        )
        
        # Display detailed metrics in a table
        with st.expander("Detailed Risk Reduction Metrics", expanded=False):
            # Prepare data for display
            display_df = backtest_data[['scenario', 'base_drawdown', 'reduced_drawdown', 'reduction_percent']].copy()
            
            # Convert to percentages for display
            display_df['base_drawdown'] = display_df['base_drawdown'] * 100
            display_df['reduced_drawdown'] = display_df['reduced_drawdown'] * 100
            
            # Rename columns for display
            display_df = display_df.rename(columns={
                'scenario': 'Scenario',
                'base_drawdown': 'Base Drawdown (%)',
                'reduced_drawdown': 'Reduced Drawdown (%)',
                'reduction_percent': 'Reduction (%)'
            })
            
            st.dataframe(display_df, use_container_width=True)
    else:
        st.warning("Risk reduction data is not in the expected format.")

def strategy_performance_during_stress(backtest_data: pd.DataFrame = None):
    """
    Display strategy performance during stress periods.
    
    Args:
        backtest_data: DataFrame containing strategy performance during stress periods
    """
    # If no data provided, fetch from database
    if backtest_data is None or backtest_data.empty:
        # Get strategy performance data
        performance_collection = get_mongo_collection("strategy_stress_performance")
        if performance_collection:
            backtest_data = pd.DataFrame(list(performance_collection))
        else:
            backtest_data = pd.DataFrame()
    
    if backtest_data.empty:
        st.info("No strategy stress performance data available yet.")
        return
    
    # Display strategy performance
    st.subheader("Strategy Performance During Market Stress")
    
    # Check if data has expected columns
    if 'strategy' in backtest_data.columns and 'scenario' in backtest_data.columns and 'performance' in backtest_data.columns:
        # Convert performance to percentage if needed
        if backtest_data['performance'].max() < 10:  # Likely decimal
            backtest_data['performance'] = backtest_data['performance'] * 100
        
        # Create heatmap of strategy performance across scenarios
        pivot_data = backtest_data.pivot(index='strategy', columns='scenario', values='performance')
        
        # Create heatmap
        fig = px.imshow(
            pivot_data,
            labels=dict(x="Market Stress Scenario", y="Strategy", color="Performance (%)"),
            x=pivot_data.columns,
            y=pivot_data.index,
            color_continuous_scale='RdYlGn',  # Red to green
            aspect="auto",
            title="Strategy Performance Across Market Stress Scenarios"
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Market Stress Scenario",
            yaxis_title="Strategy",
            coloraxis_colorbar_title="Performance (%)",
            height=400,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Identify best strategies for each scenario
        best_strategies = backtest_data.loc[backtest_data.groupby('scenario')['performance'].idxmax()]
        
        # Display best strategy for each scenario
        st.subheader("Best Strategy by Scenario")
        
        # Prepare data for display
        display_df = best_strategies[['scenario', 'strategy', 'performance']].copy()
        
        # Rename columns for display
        display_df = display_df.rename(columns={
            'scenario': 'Scenario',
            'strategy': 'Best Strategy',
            'performance': 'Performance (%)'
        })
        
        st.dataframe(display_df, use_container_width=True)
    else:
        st.warning("Strategy performance data is not in the expected format.")

def backtest_dashboard():
    """Main historical backtest results dashboard."""
    section_header("Historical Backtest Results", icon="📊")
    
    # Description
    st.markdown("""
    This dashboard shows how the current portfolio and risk management settings would have performed 
    during historical market stress periods. It helps validate the effectiveness of risk management 
    strategies and identify potential weaknesses.
    """)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs([
        "Historical Scenarios", 
        "Risk Reduction", 
        "Strategy Performance"
    ])
    
    with tab1:
        historical_scenario_results()
    
    with tab2:
        risk_reduction_effectiveness()
    
    with tab3:
        strategy_performance_during_stress()

if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(
        page_title="Backtest Results - BensBot Dashboard",
        page_icon="📊",
        layout="wide",
    )
    
    backtest_dashboard()
