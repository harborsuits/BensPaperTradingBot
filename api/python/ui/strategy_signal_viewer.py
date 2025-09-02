"""
Strategy Signal Viewer - Streamlit UI Component
This module provides Streamlit UI components for visualizing strategy signals,
including detailed reasoning, scoring heatmaps, and performance metrics.
"""

import os
import sys
import json
import time
import logging
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import our components
from trading_bot.market_intelligence_controller import get_market_intelligence_controller


def render_signal_viewer():
    """
    Render the Strategy Signal Viewer panel in the Streamlit UI.
    """
    st.header("📊 Strategy Signal Viewer")
    
    with st.expander("About Strategy Signal Viewer", expanded=False):
        st.markdown("""
        The Strategy Signal Viewer shows the top strategy-symbol pairs identified by the Market Intelligence system.
        Each pair includes:
        
        - **Score**: Overall match quality (0-1)
        - **Confidence**: Consistency across scoring factors (0-1)
        - **Reasoning**: Detailed explanation of why this pair was selected
        - **Factor Heatmap**: Visual breakdown of scoring components
        - **Performance**: Historical signal performance (if available)
        
        Use this tool to understand *why* specific strategies are recommended for each symbol
        and how confident the system is in each recommendation.
        """)
    
    # Initialize controller
    controller = get_market_intelligence_controller()
    
    # Get top pairs
    top_pairs = controller.get_top_symbol_strategy_pairs(limit=5)
    
    if not top_pairs:
        st.info("No strategy-symbol pairs available. Try updating the market data.")
        
        if st.button("Initialize Market Intelligence"):
            with st.spinner("Initializing Market Intelligence..."):
                controller.initialize()
                st.experimental_rerun()
        
        return
    
    # Display top pairs in cards
    st.subheader("Top Strategy-Symbol Pairs")
    
    # Create columns for the top 3 pairs
    top_cols = st.columns(min(3, len(top_pairs)))
    
    for i, pair in enumerate(top_pairs[:3]):
        symbol = pair.get("symbol", "")
        strategy = pair.get("strategy", "").replace('_', ' ').title()
        score = pair.get("score", 0)
        confidence = pair.get("confidence", 0)
        reasoning = pair.get("reasoning", [])
        
        with top_cols[i]:
            # Card header with score gauge
            st.markdown(f"""
            <div style="border:1px solid {'#2ecc71' if score > 0.7 else '#f39c12' if score > 0.5 else '#e74c3c'}; 
                        border-radius:5px; padding:15px; background-color:#1e2130;">
                <h3 style="color:#3498db; margin-top:0;">{symbol} + {strategy}</h3>
                <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                    <div>
                        <span style="color:#7f8c8d; font-size:14px;">Score</span>
                        <div style="font-size:20px; font-weight:bold;">{score:.2f}</div>
                    </div>
                    <div>
                        <span style="color:#7f8c8d; font-size:14px;">Confidence</span>
                        <div style="font-size:20px; font-weight:bold;">{confidence:.2f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Reasoning toggle
            with st.expander("View Reasoning", expanded=False):
                for reason in reasoning:
                    st.markdown(f"- {reason}")
            
            # Plot factor heatmap
            if "details" in pair:
                plot_factor_heatmap(pair["details"], f"{symbol}_{strategy}")
    
    # Display additional pairs
    if len(top_pairs) > 3:
        st.subheader("Additional Recommended Pairs")
        
        for pair in top_pairs[3:]:
            symbol = pair.get("symbol", "")
            strategy = pair.get("strategy", "").replace('_', ' ').title()
            score = pair.get("score", 0)
            confidence = pair.get("confidence", 0)
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{symbol}** with *{strategy}* strategy")
            
            with col2:
                st.markdown(f"Score: {score:.2f}")
            
            with col3:
                st.markdown(f"Confidence: {confidence:.2f}")
            
            # Reasoning toggle
            with st.expander("View Details", expanded=False):
                reasoning = pair.get("reasoning", [])
                for reason in reasoning:
                    st.markdown(f"- {reason}")
                
                # Plot factor heatmap
                if "details" in pair:
                    plot_factor_heatmap(pair["details"], f"{symbol}_{strategy}")
    
    # Performance tracking section
    st.subheader("Signal Performance Tracking")
    
    try:
        performance_data = load_signal_performance()
        
        if performance_data:
            # Create a dataframe for visualization
            perf_df = pd.DataFrame(performance_data)
            
            # Calculate success rate
            success_rate = len(perf_df[perf_df["successful"]]) / len(perf_df) if len(perf_df) > 0 else 0
            
            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Signals Tracked", len(perf_df))
            
            with col2:
                st.metric("Success Rate", f"{success_rate:.1%}")
            
            with col3:
                if len(perf_df) > 0:
                    avg_return = perf_df["return_pct"].mean()
                    st.metric("Avg. Return", f"{avg_return:.2f}%")
                else:
                    st.metric("Avg. Return", "N/A")
            
            # Display recent performance
            if len(perf_df) > 0:
                st.markdown("### Recent Signal Performance")
                
                # Sort by timestamp descending
                perf_df["timestamp"] = pd.to_datetime(perf_df["timestamp"])
                perf_df = perf_df.sort_values("timestamp", ascending=False)
                
                # Display the 5 most recent signals
                st.dataframe(
                    perf_df[["timestamp", "symbol", "strategy", "return_pct", "successful"]].head(5),
                    use_container_width=True
                )
                
                # Plot performance over time
                if len(perf_df) > 1:
                    st.markdown("### Performance Over Time")
                    
                    # Group by date
                    perf_df["date"] = perf_df["timestamp"].dt.date
                    daily_perf = perf_df.groupby("date")["return_pct"].mean().reset_index()
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(daily_perf["date"], daily_perf["return_pct"])
                    ax.axhline(y=0, color="r", linestyle="--", alpha=0.3)
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Average Return (%)")
                    ax.set_title("Daily Average Signal Return")
                    fig.autofmt_xdate()
                    
                    st.pyplot(fig)
        else:
            st.info("No performance data available yet. Performance tracking will populate as signals are evaluated.")
    
    except Exception as e:
        st.error(f"Error loading performance data: {str(e)}")
        st.info("Performance tracking will be available once signals have been evaluated.")


def plot_factor_heatmap(details, title):
    """
    Plot a heatmap showing the contribution of each factor to the overall score.
    
    Args:
        details: Dictionary containing factor details
        title: Title for the heatmap
    """
    # Extract scores
    scores = {
        "Technical": details.get("technical", {}).get("score", 0),
        "Sentiment": details.get("sentiment", {}).get("score", 0),
        "Fundamental": details.get("fundamental", {}).get("score", 0),
        "Regime": details.get("regime_alignment", {}).get("score", 0)
    }
    
    # Create a dataframe
    df = pd.DataFrame(
        [list(scores.values())],
        columns=list(scores.keys())
    )
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 1.5))
    sns.heatmap(
        df, 
        cmap="RdYlGn", 
        vmin=0, 
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar=False,
        ax=ax
    )
    ax.set_xticklabels(scores.keys(), rotation=0)
    ax.set_yticklabels(["Score"], rotation=0)
    st.pyplot(fig)


def load_signal_performance():
    """
    Load signal performance data from the database.
    
    Returns:
        List of performance records
    """
    # In a real implementation, this would load from a database
    # For now, we'll create some mock data
    
    # Check if we have mock data in session state
    if "mock_performance_data" not in st.session_state:
        # Create some mock data
        mock_data = []
        
        # Generate data for the past 30 days
        for i in range(30):
            date = datetime.now() - timedelta(days=i)
            
            # Create 0-3 signals per day
            for _ in range(np.random.randint(0, 4)):
                symbol = np.random.choice(["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"])
                strategy = np.random.choice(["momentum_etf", "value_dividend", "mean_reversion"])
                
                # Randomize returns, slightly positive bias
                return_pct = np.random.normal(0.5, 2.0)
                successful = return_pct > 0
                
                mock_data.append({
                    "timestamp": date.isoformat(),
                    "symbol": symbol,
                    "strategy": strategy,
                    "return_pct": return_pct,
                    "successful": successful,
                    "sharpe": np.random.uniform(0.5, 2.5) if successful else np.random.uniform(-1.0, 0.5),
                    "max_drawdown": np.random.uniform(0.5, 5.0)
                })
        
        st.session_state.mock_performance_data = mock_data
    
    return st.session_state.mock_performance_data


def add_performance_record(symbol, strategy, return_pct, sharpe=None, max_drawdown=None):
    """
    Add a new performance record for a signal.
    
    Args:
        symbol: Stock symbol
        strategy: Strategy ID
        return_pct: Return percentage
        sharpe: Sharpe ratio
        max_drawdown: Maximum drawdown
        
    Returns:
        Boolean indicating success
    """
    try:
        # In a real implementation, this would store to a database
        # For now, we'll add to the mock data in session state
        if "mock_performance_data" not in st.session_state:
            st.session_state.mock_performance_data = []
        
        # Create record
        record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "strategy": strategy,
            "return_pct": return_pct,
            "successful": return_pct > 0,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown
        }
        
        st.session_state.mock_performance_data.append(record)
        
        return True
        
    except Exception as e:
        print(f"Error adding performance record: {str(e)}")
        return False
