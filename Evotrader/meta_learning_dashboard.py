#!/usr/bin/env python3
"""
Meta-Learning Dashboard

A Streamlit dashboard to visualize insights from the meta-learning system.
"""

import os
import sys
import json
import sqlite3
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple, Union

# Add project root to path to import local modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from meta_learning_db import MetaLearningDB
    from market_regime_detector import MarketRegimeDetector
    from strategy_pattern_analyzer import StrategyPatternAnalyzer
    from prop_strategy_registry import PropStrategyRegistry
except ImportError:
    st.error("Failed to import required modules. Make sure you're running from the project root.")
    
# Set page configuration
st.set_page_config(
    page_title="EvoTrader Meta-Learning Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dashboard styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2196F3;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.3rem;
        color: #FF9800;
        margin-top: 1rem;
        margin-bottom: 0.7rem;
    }
    .performance-metric {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .stMetric label {
        font-size: 1rem !important;
    }
    .highlight {
        background-color: #f0f8ff;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Dashboard title
st.markdown("<h1 class='main-header'>EvoTrader Meta-Learning Dashboard</h1>", unsafe_allow_html=True)
st.markdown("**Monitor meta-learning insights, market regimes, and strategy performance patterns**")

# Sidebar for controls
st.sidebar.markdown("<p class='sidebar-header'>Dashboard Controls</p>", unsafe_allow_html=True)

# Initialize instance references
meta_db = None
registry = None
regime_detector = None
pattern_analyzer = None

# Function to load data
@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_meta_learning_data(db_path):
    """Load data from meta-learning database"""
    try:
        # Initialize meta-learning DB
        meta_db = MetaLearningDB(db_path=db_path)
        
        # Get aggregated insights
        strategy_insights = meta_db.get_strategy_type_insights()
        indicator_insights = meta_db.get_indicator_insights()
        parameter_insights = meta_db.get_parameter_insights()
        regime_insights = meta_db.get_all_regime_insights()
        
        # Get raw data for detailed analysis
        strategy_results = meta_db.get_recent_strategy_results(limit=100)
        evolution_metrics = meta_db.get_evolution_metrics()
        
        return {
            'strategy_insights': strategy_insights,
            'indicator_insights': indicator_insights,
            'parameter_insights': parameter_insights,
            'regime_insights': regime_insights,
            'strategy_results': strategy_results,
            'evolution_metrics': evolution_metrics,
            'meta_db': meta_db
        }
    except Exception as e:
        st.error(f"Error loading meta-learning data: {e}")
        return None

@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_registry_data(db_path):
    """Load data from strategy registry"""
    try:
        # Initialize registry
        registry = PropStrategyRegistry(db_path=db_path)
        
        # Get strategies
        active_strategies = registry.get_active_strategies()
        promoted_strategies = registry.get_promotion_candidates()
        demoted_strategies = registry.get_demotion_candidates()
        
        return {
            'active_strategies': active_strategies,
            'promoted_strategies': promoted_strategies,
            'demoted_strategies': demoted_strategies,
            'registry': registry
        }
    except Exception as e:
        st.error(f"Error loading registry data: {e}")
        return None

# Configuration inputs
with st.sidebar.expander("Data Sources", expanded=False):
    meta_db_path = st.text_input("Meta-Learning DB Path", value="./meta_learning/meta_db.sqlite")
    registry_db_path = st.text_input("Registry DB Path", value="./forex_prop_strategies.db")
    price_data_path = st.text_input("Price Data Path (CSV)", value="")

# Load button
if st.sidebar.button("Load Data"):
    with st.spinner("Loading data..."):
        # Load meta-learning data
        meta_data = load_meta_learning_data(meta_db_path)
        if meta_data:
            meta_db = meta_data['meta_db']
            st.sidebar.success("Meta-learning data loaded successfully")
        
        # Load registry data
        registry_data = load_registry_data(registry_db_path)
        if registry_data:
            registry = registry_data['registry']
            st.sidebar.success("Registry data loaded successfully")
        
        # Initialize other components
        regime_detector = MarketRegimeDetector()
        pattern_analyzer = StrategyPatternAnalyzer()

# Time period selector
with st.sidebar.expander("Time Period", expanded=True):
    time_period = st.radio(
        "Analysis Period",
        options=["Last Week", "Last Month", "Last 3 Months", "Last 6 Months", "All Time"],
        index=1
    )

# Main dashboard tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview & Performance", 
    "Market Regimes", 
    "Strategy Patterns", 
    "Evolution Insights"
])

# Check if data is loaded
if not meta_db:
    st.info("Please load data using the sidebar controls to view the dashboard.")

# Initialize demo data
def create_demo_data():
    """Create demo data for visualization"""
    
    # Strategy performance by regime
    regime_performance = pd.DataFrame({
        'Regime': ['bullish', 'bearish', 'volatile_bullish', 'volatile_bearish', 'ranging', 'choppy'],
        'Trend Following': [0.82, 0.35, 0.68, 0.52, 0.28, 0.22],
        'Mean Reversion': [0.42, 0.56, 0.38, 0.45, 0.78, 0.65],
        'Breakout': [0.68, 0.48, 0.85, 0.76, 0.32, 0.45],
        'Pattern Recognition': [0.61, 0.58, 0.55, 0.62, 0.57, 0.51],
        'Multi Timeframe': [0.75, 0.68, 0.71, 0.65, 0.63, 0.58]
    })
    
    # Indicator effectiveness
    indicator_effectiveness = pd.DataFrame({
        'Indicator': ['MACD', 'RSI', 'Bollinger Bands', 'EMA', 'ATR', 'ADX', 'Stochastic', 'Volume', 'Ichimoku'],
        'Effectiveness': [1.42, 1.85, 1.65, 1.38, 0.98, 1.22, 1.58, 1.05, 1.18],
        'Win Rate': [0.62, 0.68, 0.65, 0.59, 0.52, 0.56, 0.63, 0.54, 0.57]
    })
    
    # Parameter distribution
    param_values = {
        'rsi_period': np.random.choice([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], size=50),
        'macd_fast': np.random.choice([8, 9, 10, 11, 12, 13, 14, 15, 16], size=50),
        'macd_slow': np.random.choice([18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], size=50),
        'bollinger_period': np.random.choice([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], size=50),
        'ema_period': np.random.choice(range(5, 51), size=50)
    }
    
    # Live vs backtest performance
    performance_comparison = pd.DataFrame({
        'Strategy': [f'Strategy {i}' for i in range(1, 11)],
        'Backtest Return': np.random.normal(12, 4, 10),
        'Live Return': np.random.normal(9, 5, 10),
        'Backtest Sharpe': np.random.normal(1.8, 0.4, 10),
        'Live Sharpe': np.random.normal(1.5, 0.6, 10),
        'Consistency Score': np.random.uniform(0.5, 0.95, 10)
    })
    
    # Add regime history
    today = datetime.datetime.now()
    dates = [today - datetime.timedelta(days=i) for i in range(60)]
    
    regimes = []
    for _ in range(60):
        regime = np.random.choice(['bullish', 'bearish', 'volatile_bullish', 'volatile_bearish', 'ranging', 'choppy'], 
                                 p=[0.3, 0.2, 0.1, 0.1, 0.2, 0.1])
        regimes.append(regime)
    
    regime_history = pd.DataFrame({
        'date': dates,
        'regime': regimes,
        'confidence': np.random.uniform(0.6, 0.95, 60)
    })
    
    return {
        'regime_performance': regime_performance,
        'indicator_effectiveness': indicator_effectiveness,
        'param_values': param_values,
        'performance_comparison': performance_comparison,
        'regime_history': regime_history
    }

# Create demo data
demo_data = create_demo_data()

# Tab 1: Overview & Performance
with tab1:
    st.markdown("<h2 class='section-header'>Trading System Performance</h2>", unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Active Strategies", value="42", delta="↑ 5")
    with col2:
        st.metric(label="Avg. Live Confidence", value="0.73", delta="↑ 0.08")
    with col3:
        st.metric(label="Avg. Return Delta", value="-2.1%", delta="↓ 0.5%")
    with col4:
        st.metric(label="Avg. Sharpe Delta", value="-0.21", delta="↑ 0.05")
    
    # Live vs Backtest Performance comparison
    st.markdown("<h3 class='subsection-header'>Live vs Backtest Performance</h3>", unsafe_allow_html=True)
    
    performance_df = demo_data['performance_comparison']
    
    # Create the figure
    fig = go.Figure()
    
    # Add backtest returns
    fig.add_trace(go.Bar(
        x=performance_df['Strategy'],
        y=performance_df['Backtest Return'],
        name='Backtest Return',
        marker_color='blue',
        opacity=0.7
    ))
    
    # Add live returns
    fig.add_trace(go.Bar(
        x=performance_df['Strategy'],
        y=performance_df['Live Return'],
        name='Live Return',
        marker_color='green',
        opacity=0.7
    ))
    
    # Add consistency score as a line
    fig.add_trace(go.Scatter(
        x=performance_df['Strategy'],
        y=performance_df['Consistency Score'],
        name='Consistency Score',
        mode='lines+markers',
        marker=dict(color='red'),
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title='Strategy Performance: Backtest vs Live',
        xaxis_title='Strategy',
        yaxis_title='Return (%)',
        yaxis2=dict(
            title='Consistency Score',
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            anchor='x',
            overlaying='y',
            side='right',
            range=[0, 1]
        ),
        barmode='group',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Promotion/Demotion Candidates
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 class='subsection-header'>Promotion Candidates</h3>", unsafe_allow_html=True)
        st.dataframe({
            'Strategy ID': [f'strat_{i:03d}' for i in range(1, 6)],
            'Type': ['Trend Following', 'Mean Reversion', 'Breakout', 'Trend Following', 'Multi Timeframe'],
            'Confidence': [0.92, 0.87, 0.85, 0.83, 0.81]
        })
    
    with col2:
        st.markdown("<h3 class='subsection-header'>Demotion Candidates</h3>", unsafe_allow_html=True)
        st.dataframe({
            'Strategy ID': [f'strat_{i:03d}' for i in range(6, 11)],
            'Type': ['Mean Reversion', 'Breakout', 'Pattern Recognition', 'Trend Following', 'Mean Reversion'],
            'Confidence': [0.32, 0.36, 0.38, 0.41, 0.42]
        })

# Tab 2: Market Regimes
with tab2:
    st.markdown("<h2 class='section-header'>Market Regime Analysis</h2>", unsafe_allow_html=True)
    
    # Current regime
    current_regime = "bullish"
    current_conf = 0.78
    
    st.markdown(
        f"<div class='highlight'><h3>Current Regime: {current_regime.title()}</h3>"
        f"<p>Confidence: {current_conf:.2f}</p></div>",
        unsafe_allow_html=True
    )
    
    # Regime history
    st.markdown("<h3 class='subsection-header'>Regime History</h3>", unsafe_allow_html=True)
    
    # Create the line chart
    regime_history = demo_data['regime_history']
    
    # Map regimes to numeric values for coloring
    regime_map = {
        'bullish': 5, 
        'volatile_bullish': 4, 
        'ranging': 3, 
        'choppy': 2, 
        'volatile_bearish': 1, 
        'bearish': 0
    }
    
    regime_history['regime_value'] = regime_history['regime'].map(regime_map)
    
    fig = px.scatter(
        regime_history, 
        x='date', 
        y='regime',
        color='regime_value',
        size='confidence',
        color_continuous_scale='Viridis',
        hover_data=['confidence'],
        labels={'date': 'Date', 'regime': 'Market Regime', 'confidence': 'Confidence'},
        title='Market Regime History'
    )
    
    fig.update_layout(
        yaxis=dict(
            categoryorder='array',
            categoryarray=['bearish', 'volatile_bearish', 'choppy', 'ranging', 'volatile_bullish', 'bullish']
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Strategy performance by regime
    st.markdown("<h3 class='subsection-header'>Strategy Performance by Regime</h3>", unsafe_allow_html=True)
    
    # Create a heatmap
    regime_performance = demo_data['regime_performance']
    
    # Melt the dataframe for heatmap
    regime_perf_melted = pd.melt(
        regime_performance, 
        id_vars=['Regime'], 
        var_name='Strategy Type', 
        value_name='Confidence Score'
    )
    
    fig = px.density_heatmap(
        regime_perf_melted,
        x='Regime',
        y='Strategy Type',
        z='Confidence Score',
        color_continuous_scale='Viridis',
        title='Strategy Type Performance by Market Regime'
    )
    
    fig.update_layout(
        xaxis=dict(
            categoryorder='array',
            categoryarray=['bearish', 'volatile_bearish', 'choppy', 'ranging', 'volatile_bullish', 'bullish']
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Strategy Patterns
with tab3:
    st.markdown("<h2 class='section-header'>Strategy Pattern Analysis</h2>", unsafe_allow_html=True)
    
    # Indicator effectiveness
    st.markdown("<h3 class='subsection-header'>Indicator Effectiveness</h3>", unsafe_allow_html=True)
    
    # Create the chart
    indicator_effectiveness = demo_data['indicator_effectiveness']
    
    fig = go.Figure()
    
    # Add bar chart for effectiveness
    fig.add_trace(go.Bar(
        x=indicator_effectiveness['Indicator'],
        y=indicator_effectiveness['Effectiveness'],
        name='Effectiveness',
        marker_color='blue',
        opacity=0.7
    ))
    
    # Add scatter plot for win rate
    fig.add_trace(go.Scatter(
        x=indicator_effectiveness['Indicator'],
        y=indicator_effectiveness['Win Rate'],
        name='Win Rate',
        mode='lines+markers',
        marker=dict(color='red'),
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title='Indicator Effectiveness & Win Rate',
        xaxis_title='Indicator',
        yaxis_title='Effectiveness Ratio',
        yaxis2=dict(
            title='Win Rate',
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            anchor='x',
            overlaying='y',
            side='right',
            range=[0, 1]
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Parameter value distributions
    st.markdown("<h3 class='subsection-header'>Parameter Value Distributions</h3>", unsafe_allow_html=True)
    
    param_select = st.selectbox(
        "Select Parameter",
        options=list(demo_data['param_values'].keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    # Create histogram
    fig = px.histogram(
        x=demo_data['param_values'][param_select],
        nbins=15,
        title=f'{param_select.replace("_", " ").title()} Distribution in Successful Strategies',
        labels={'x': param_select.replace('_', ' ').title(), 'y': 'Count'},
        color_discrete_sequence=['green']
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Indicator combinations
    st.markdown("<h3 class='subsection-header'>Effective Indicator Combinations</h3>", unsafe_allow_html=True)
    
    # Create a chord diagram for indicator combinations (simplified as a dataframe)
    st.dataframe({
        'Combination': ['MACD + RSI', 'RSI + Bollinger Bands', 'EMA + MACD', 'Bollinger Bands + ATR', 'RSI + Stochastic'],
        'Count': [28, 24, 21, 18, 17],
        'Avg. Confidence': [0.82, 0.79, 0.76, 0.72, 0.70]
    })

# Tab 4: Evolution Insights
with tab4:
    st.markdown("<h2 class='section-header'>Evolution Meta-Learning</h2>", unsafe_allow_html=True)
    
    # Evolution success metrics
    st.markdown("<h3 class='subsection-header'>Evolution Success Rates</h3>", unsafe_allow_html=True)
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Avg. Generations to Optimal", value="12.3", delta="↓ 2.1")
    with col2:
        st.metric(label="Top Strategy Confidence", value="0.88", delta="↑ 0.05")
    with col3:
        st.metric(label="Meta-Learning Impact", value="+15.2%", delta="↑ 3.1%")
    with col4:
        st.metric(label="Promotion Rate", value="28.5%", delta="↑ 4.2%")
    
    # Evolution learning curve
    st.markdown("<h3 class='subsection-header'>Evolution Learning Curve</h3>", unsafe_allow_html=True)
    
    # Create sample data
    generations = list(range(1, 21))
    with_meta = [0.42, 0.53, 0.61, 0.67, 0.72, 0.76, 0.79, 0.82, 0.84, 0.86, 
                0.87, 0.88, 0.89, 0.9, 0.9, 0.91, 0.91, 0.92, 0.92, 0.92]
    without_meta = [0.38, 0.45, 0.51, 0.56, 0.60, 0.63, 0.66, 0.68, 0.70, 0.72,
                   0.73, 0.74, 0.76, 0.77, 0.78, 0.78, 0.79, 0.79, 0.80, 0.80]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=generations,
        y=with_meta,
        name='With Meta-Learning',
        mode='lines+markers',
        marker=dict(color='green')
    ))
    
    fig.add_trace(go.Scatter(
        x=generations,
        y=without_meta,
        name='Without Meta-Learning',
        mode='lines+markers',
        marker=dict(color='gray')
    ))
    
    fig.update_layout(
        title='Evolution Fitness by Generation',
        xaxis_title='Generation',
        yaxis_title='Max Strategy Fitness',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Adaptive parameter ranges
    st.markdown("<h3 class='subsection-header'>Adaptive Parameter Ranges</h3>", unsafe_allow_html=True)
    
    # Create data for parameter range evolution
    parameters = ['RSI Period', 'MACD Fast', 'MACD Slow', 'Bollinger Period', 'EMA Period']
    
    initial_min = [7, 8, 18, 10, 5]
    initial_max = [21, 16, 30, 30, 50]
    
    adapted_min = [12, 10, 20, 14, 8]
    adapted_max = [17, 14, 25, 22, 21]
    
    fig = go.Figure()
    
    for i, param in enumerate(parameters):
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[initial_min[i], adapted_min[i]],
            mode='lines+markers',
            name=f'{param} Min',
            line=dict(color='rgba(31, 119, 180, 0.7)')
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[initial_max[i], adapted_max[i]],
            mode='lines+markers',
            name=f'{param} Max',
            line=dict(color='rgba(255, 127, 14, 0.7)')
        ))
    
    fig.update_layout(
        title='Parameter Range Adaptation',
        xaxis=dict(
            tickmode='array',
            tickvals=[0, 1],
            ticktext=['Initial Range', 'Meta-Learned Range']
        ),
        yaxis_title='Parameter Value',
        showlegend=False
    )
    
    # Add parameter labels
    for i, param in enumerate(parameters):
        fig.add_annotation(
            x=0.5,
            y=(adapted_min[i] + adapted_max[i]) / 2,
            text=param,
            showarrow=False,
            font=dict(size=10)
        )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>EvoTrader Meta-Learning Dashboard | Last updated: "
    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    unsafe_allow_html=True
)

# Note about demo data
st.sidebar.markdown("---")
st.sidebar.info(
    "This dashboard is currently showing demo data. "
    "Connect to real data sources using the controls above."
)
