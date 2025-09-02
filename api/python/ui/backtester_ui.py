"""
Backtester UI - Modern Visualization for ML Testing and Learning
This module provides clean, visual components for the backtester interface.
"""

import os
import sys
import json
import time
import logging
import datetime
from typing import Dict, List, Any, Optional, Union
import random
import streamlit as st

# Handle optional dependencies gracefully
TRY_IMPORTS = True

# Basic data processing libraries
try:
    import pandas as pd
    import numpy as np
except ImportError:
    if TRY_IMPORTS:
        st.error("Missing required dependencies: pandas and/or numpy. Please install with 'pip install pandas numpy'")
    TRY_IMPORTS = False

# Visualization libraries - these are optional but recommended
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    if TRY_IMPORTS:
        st.warning("Plotly not available. Install with 'pip install plotly' for enhanced visualizations.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    if TRY_IMPORTS:
        st.warning("Matplotlib/Seaborn not available. Install with 'pip install matplotlib seaborn' for additional charts.")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from our components
from trading_bot.market_context.market_context import get_market_context
from trading_bot.ml_pipeline.backtest_feedback_loop import get_backtest_feedback_system, get_backtest_executor

class BacktesterUI:
    """
    Class to render modern, clean UI components for the backtester tab.
    """
    
    def __init__(self):
        """Initialize the backtester UI components."""
        self.feedback_system = get_backtest_feedback_system()
        self.backtest_executor = get_backtest_executor()
        
        # Cache for visualizations
        self._cache = {}
    
    def render_backtest_overview(self, backtest_id=None, strategy=None, symbol=None):
        """
        Render the backtest overview card.
        
        Args:
            backtest_id: Optional backtest ID to display
            strategy: Optional strategy name to display
            symbol: Optional symbol to display
        """
        # Get market context for regime info
        market_context = get_market_context().get_market_context()
        market_regime = market_context.get("market", {}).get("regime", "Unknown")
        regime_confidence = market_context.get("market", {}).get("regime_confidence", 0)
        vix = market_context.get("market", {}).get("indicators", {}).get("vix", "--")
        
        # Generate or use provided backtest ID
        if not backtest_id:
            backtest_id = f"BTX-{datetime.datetime.now().strftime('%Y%m%d')}-{random.randint(1, 999):03d}"
        
        # Use provided or default values
        strategy = strategy or "momentum_breakout_v4"
        symbol = symbol or "AAPL"
        
        # Get time window (default to last 90 days)
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=90)
        
        # Format dates
        start_str = start_date.strftime("%b %d")
        end_str = end_date.strftime("%b %d")
        
        # Create card with modern styling
        st.markdown(f"""
        <div style="background-color:#1E1E1E; border-radius:10px; padding:15px; margin-bottom:20px; border-left:5px solid #4CAF50;">
            <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                <div>
                    <span style="color:#BBBBBB; font-size:14px;">Backtest Session ID</span>
                    <h3 style="margin:0; color:white;">{backtest_id}</h3>
                </div>
                <div style="text-align:right;">
                    <span style="background-color:#2C3E50; color:white; padding:5px 10px; border-radius:4px; font-size:12px;">
                        {market_regime} Regime • {regime_confidence:.0%} Confidence
                    </span>
                </div>
            </div>
            <div style="display:flex; flex-wrap:wrap; gap:20px; margin-top:15px;">
                <div>
                    <span style="color:#BBBBBB; font-size:12px;">STRATEGY</span>
                    <p style="margin:0; color:white; font-weight:bold;">{strategy}</p>
                </div>
                <div>
                    <span style="color:#BBBBBB; font-size:12px;">INSTRUMENT</span>
                    <p style="margin:0; color:white; font-weight:bold;">{symbol}</p>
                </div>
                <div>
                    <span style="color:#BBBBBB; font-size:12px;">TIME WINDOW</span>
                    <p style="margin:0; color:white;">{start_str} – {end_str}</p>
                </div>
                <div>
                    <span style="color:#BBBBBB; font-size:12px;">VIX</span>
                    <p style="margin:0; color:white;">{vix}</p>
                </div>
                <div>
                    <span style="color:#BBBBBB; font-size:12px;">STATUS</span>
                    <p style="margin:0; color:#4CAF50; font-weight:bold;">Active</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_metrics_grid(self, backtest_results=None):
        """
        Render the metrics grid component.
        
        Args:
            backtest_results: Optional backtest results to display
        """
        # Use provided results or default to mock data
        if not backtest_results:
            # Mock data for demonstration
            backtest_results = {
                "sharpe_ratio": 1.87,
                "baseline_sharpe": 1.45,
                "max_drawdown": -6.3,
                "baseline_max_drawdown": -8.4,
                "total_return": 12.8,
                "baseline_return": 8.4,
                "win_rate": 61,
                "baseline_win_rate": 58,
                "model_confidence": 88,
                "tested_variants": 9
            }
        
        # Create the metrics grid with columns
        cols = st.columns(6)
        
        # Using st.metric for clean delta displays
        with cols[0]:
            st.metric(
                label="⚖️ Sharpe Ratio",
                value=f"{backtest_results['sharpe_ratio']:.2f}",
                delta=f"{backtest_results['sharpe_ratio'] - backtest_results['baseline_sharpe']:.2f}",
                delta_color="normal"
            )
        
        with cols[1]:
            st.metric(
                label="📉 Max Drawdown",
                value=f"{backtest_results['max_drawdown']:.1f}%",
                delta=f"{backtest_results['max_drawdown'] - backtest_results['baseline_max_drawdown']:.1f}%",
                delta_color="inverse"  # For drawdown, lower is better
            )
        
        with cols[2]:
            st.metric(
                label="💰 Net Return",
                value=f"{backtest_results['total_return']:.1f}%",
                delta=f"{backtest_results['total_return'] - backtest_results['baseline_return']:.1f}%",
                delta_color="normal"
            )
        
        with cols[3]:
            st.metric(
                label="🟩 Win Rate",
                value=f"{backtest_results['win_rate']}%",
                delta=f"{backtest_results['win_rate'] - backtest_results['baseline_win_rate']}%",
                delta_color="normal"
            )
        
        with cols[4]:
            st.metric(
                label="🧠 Model Confidence",
                value=f"{backtest_results['model_confidence']}%",
                delta=None
            )
        
        with cols[5]:
            st.metric(
                label="🧪 Tested Variants",
                value=backtest_results['tested_variants'],
                delta=None
            )
    
    def render_ml_progress_tracker(self, progress_items=None):
        """
        Render the ML progress tracker component.
        
        Args:
            progress_items: Optional list of progress items to display
        """
        if not progress_items:
            # Mock data for demonstration
            progress_items = [
                {
                    "status": "complete",
                    "test_num": 1,
                    "config": "MA(10/50), RSI(14)",
                    "metrics": {"sharpe": 1.34},
                    "notes": "DQ triggered on volatility"
                },
                {
                    "status": "complete",
                    "test_num": 2,
                    "config": "MA(20/100), RSI(9)",
                    "metrics": {"sharpe": 1.66},
                    "notes": "strong trend match"
                },
                {
                    "status": "complete",
                    "test_num": 3,
                    "config": "BB(20,2.0), MACD(12,26,9)",
                    "metrics": {"sharpe": 1.72},
                    "notes": "strong news alignment"
                },
                {
                    "status": "failed",
                    "test_num": 4,
                    "config": "Momentum(3d ROC) + EMA cross",
                    "metrics": {"sharpe": 0.82},
                    "notes": "poor results, discarded"
                },
                {
                    "status": "selected",
                    "test_num": 5,
                    "config": "MA(20/100) + RSI(9)",
                    "metrics": {"sharpe": 1.66},
                    "notes": "Selected as final config"
                }
            ]
        
        # Create a card for the ML progress tracker
        st.markdown("### 🧪 ML Progress Tracker")
        
        # Create a container for the timeline
        progress_container = st.container()
        
        with progress_container:
            # Iterate through progress items
            for item in progress_items:
                # Determine status icon and color
                if item["status"] == "complete":
                    icon = "✅"
                    color = "#4CAF50"
                elif item["status"] == "failed":
                    icon = "⚠️"
                    color = "#F44336"
                elif item["status"] == "selected":
                    icon = "🌟"
                    color = "#2196F3"
                elif item["status"] == "pending":
                    icon = "⏳"
                    color = "#FFC107"
                else:
                    icon = "🔄"
                    color = "#9E9E9E"
                
                # Create the progress item
                st.markdown(f"""
                <div style="display:flex; margin-bottom:10px; align-items:start;">
                    <div style="font-size:20px; margin-right:10px;">{icon}</div>
                    <div style="flex-grow:1;">
                        <div style="font-weight:bold; color:{color};">
                            Test #{item['test_num']}: {item['config']}
                        </div>
                        <div style="color:#BBBBBB; font-size:14px;">
                            Sharpe: {item['metrics'].get('sharpe', 'N/A')} | {item['notes']}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    def render_equity_curve(self, equity_data=None):
        """
        Render an equity curve visualization.
        
        Args:
            equity_data: Optional dataframe with equity curve data
        """
        # Check if plotly is available
        if not 'PLOTLY_AVAILABLE' in globals() or not PLOTLY_AVAILABLE:
            st.warning("📈 Equity curve visualization requires Plotly. Install with 'pip install plotly'.")
            # Show a text representation instead
            st.info("Equity curve would show strategy performance over time compared to baseline.")
            return
            
        try:
            if equity_data is None:
                # Generate mock equity curve data
                dates = pd.date_range(start='2025-01-15', periods=90)
                baseline = 100000 + np.cumsum(np.random.normal(50, 100, 90))
                strategy = 100000 + np.cumsum(np.random.normal(80, 120, 90))
                
                equity_data = pd.DataFrame({
                    'Date': dates,
                    'Strategy': strategy,
                    'Baseline': baseline
                })
            
            # Create a plotly figure
            fig = go.Figure()
            
            # Add equity curves
            fig.add_trace(go.Scatter(
                x=equity_data['Date'],
                y=equity_data['Strategy'],
                mode='lines',
                name='Strategy',
                line=dict(color='#4CAF50', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=equity_data['Date'],
                y=equity_data['Baseline'],
                mode='lines',
                name='Baseline',
                line=dict(color='#9E9E9E', width=1.5, dash='dash')
            ))
            
            # Customize layout
            fig.update_layout(
                title='Equity Curve',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=20, r=20, t=50, b=20),
                height=300,
                template="plotly_dark"
            )
            
            # Display the figure
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering equity curve: {e}")
            st.info("Equity curve would show strategy performance over time compared to baseline.")
    
    def render_win_loss_distribution(self, trades=None):
        """
        Render a win/loss distribution visualization.
        
        Args:
            trades: Optional dataframe with trade data
        """
        # Check if plotly is available
        if not 'PLOTLY_AVAILABLE' in globals() or not PLOTLY_AVAILABLE:
            st.warning("🔹 Win/loss distribution visualization requires Plotly. Install with 'pip install plotly'.")
            
            # Show a text summary instead
            wins = 60  # Mock data
            losses = 40  # Mock data
            win_rate = wins / (wins + losses) * 100
            
            st.markdown(f"**Win Rate: {win_rate:.1f}%** ({wins} wins, {losses} losses)")
            return
            
        try:
            if trades is None:
                # Generate mock trade data
                # Profits/losses with slightly more wins than losses
                returns = np.concatenate([
                    np.random.normal(2, 1, 60),  # 60 winning trades
                    np.random.normal(-1.5, 0.8, 40)  # 40 losing trades
                ])
                
                trades = pd.DataFrame({
                    'return_pct': returns
                })
            
            # Create histogram with Plotly
            fig = px.histogram(
                trades,
                x='return_pct',
                nbins=20,
                color_discrete_sequence=['#4CAF50'],
                title='Trade Return Distribution'
            )
            
            # Add a vertical line at 0
            fig.add_vline(
                x=0,
                line_dash="dash",
                line_color="red",
                annotation_text="Breakeven",
                annotation_position="top right"
            )
            
            # Customize layout
            fig.update_layout(
                xaxis_title='Return (%)',
                yaxis_title='Number of Trades',
                bargap=0.1,
                margin=dict(l=20, r=20, t=50, b=20),
                height=250,
                template="plotly_dark"
            )
            
            # Display the figure
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering win/loss distribution: {e}")
            st.info("Win/loss distribution would show the spread of trade returns.")
    
    def render_parameter_heatmap(self, param_results=None):
        """
        Render a parameter optimization heatmap.
        
        Args:
            param_results: Optional parameter optimization results
        """
        # Check if plotly is available
        if not 'PLOTLY_AVAILABLE' in globals() or not PLOTLY_AVAILABLE:
            st.warning("📏 Parameter heatmap visualization requires Plotly. Install with 'pip install plotly'.")
            
            # Show text summary instead
            st.markdown("**Parameter Optimization Results:**")
            st.markdown("- Best combination: MA Short=15, MA Long=100, Sharpe=1.7")
            st.markdown("- Second best: MA Short=10, MA Long=100, Sharpe=1.65")
            return
            
        try:
            if param_results is None:
                # Generate mock parameter optimization results
                # Create a grid of parameters and their resulting Sharpe ratios
                ma_short = [5, 10, 15, 20, 25]
                ma_long = [50, 100, 150, 200]
                
                grid = []
                for short in ma_short:
                    for long in ma_long:
                        # Higher Sharpe for certain parameter combinations
                        base_sharpe = 1.0
                        
                        # Peak at (10, 100) and (15, 100)
                        if (short == 10 and long == 100) or (short == 15 and long == 100):
                            sharpe = base_sharpe + 0.7 + np.random.normal(0, 0.1)
                        # Good at combinations around the peak
                        elif (short in [5, 20] and long == 100) or (short in [10, 15] and long in [50, 150]):
                            sharpe = base_sharpe + 0.4 + np.random.normal(0, 0.1)
                        # Weaker at other combinations
                        else:
                            sharpe = base_sharpe + np.random.normal(0, 0.2)
                        
                        grid.append({
                            'MA Short': short,
                            'MA Long': long,
                            'Sharpe': sharpe
                        })
                
                param_results = pd.DataFrame(grid)
            
            # Pivot to create a matrix format for the heatmap
            heatmap_data = param_results.pivot(
                index='MA Short',
                columns='MA Long',
                values='Sharpe'
            )
            
            # Create heatmap with Plotly
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="MA Long", y="MA Short", color="Sharpe Ratio"),
                x=heatmap_data.columns,
                y=heatmap_data.index,
                color_continuous_scale='Viridis',
                title='Parameter Optimization Heatmap'
            )
            
            # Customize layout
            fig.update_layout(
                xaxis_title='MA Long Period',
                yaxis_title='MA Short Period',
                coloraxis_colorbar=dict(title='Sharpe Ratio'),
                margin=dict(l=20, r=20, t=50, b=20),
                height=250,
                template="plotly_dark"
            )
            
            # Display the figure
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering parameter heatmap: {e}")
            st.info("Parameter heatmap would show optimal settings for the strategy.")
    
    def render_backtest_insight_card(self, insight_text=None):
        """
        Render a natural language insight card for the backtest.
        
        Args:
            insight_text: Optional insight text to display
        """
        if not insight_text:
            insight_text = """
            The AI tested 9 variations of momentum breakout logic under current regime.
            The most successful was MA(20/100) with RSI(9), which yielded a Sharpe of 1.66 and 12.8% net return.
            Drawdown remained within risk tolerance at -6.3%. Strategy was accepted and stored for forward testing.
            
            Backtest results show 61% win rate across 82 trades, with an average holding period of 3.2 days.
            The strategy performed particularly well during high volatility periods, suggesting it could 
            be valuable in the current market environment.
            """
        
        # Create card with 'AI insight' styling
        st.markdown(f"""
        <div style="background-color:#2C3E50; border-radius:10px; padding:15px; margin:20px 0;">
            <div style="display:flex; align-items:center; margin-bottom:10px;">
                <div style="background-color:#3498DB; border-radius:50%; width:36px; height:36px; display:flex; align-items:center; justify-content:center; margin-right:10px;">
                    <span style="color:white; font-size:20px;">🧠</span>
                </div>
                <h3 style="margin:0; color:white;">AI Strategy Insight</h3>
            </div>
            <div style="color:#ECF0F1; font-size:15px; line-height:1.5;">
                {insight_text.replace('\n', '<br>')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_action_buttons(self):
        """Render action buttons for the backtester."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            push_btn = st.button(
                "🟢 Push to Paper Trading",
                key="push_to_paper",
                help="Push this strategy to paper trading for live testing"
            )
        
        with col2:
            refine_btn = st.button(
                "🔄 Refine Parameters",
                key="refine_params",
                help="Run additional parameter optimizations"
            )
        
        with col3:
            compare_btn = st.button(
                "🔎 Compare to Previous",
                key="compare_previous",
                help="Compare with previous backtest runs"
            )
        
        with col4:
            new_backtest_btn = st.button(
                "➕ New Backtest",
                key="new_backtest",
                help="Start a new backtest with different parameters"
            )
    
    def run_backtest(self, symbol, strategy, params=None):
        """
        Run a backtest using the backtest executor.
        
        Args:
            symbol: Stock symbol to test
            strategy: Strategy ID to test
            params: Optional parameters for the backtest
        
        Returns:
            Dict containing backtest results
        """
        # Run the backtest
        results = self.backtest_executor.backtest_pair(
            symbol=symbol,
            strategy=strategy,
            params=params
        )
        
        return results
    
    def get_top_strategies(self, limit=5):
        """
        Get the top performing strategies from the feedback system.
        
        Args:
            limit: Maximum number of strategies to return
        
        Returns:
            List of top performing strategies
        """
        return self.feedback_system.get_top_performing_pairs(limit=limit)
    
    def render_full_backtest_interface(self, backtest_session=None):
        """
        Render the complete backtest interface.
        
        Args:
            backtest_session: Optional backtest session data
        """
        # Overview
        self.render_backtest_overview(
            backtest_id=backtest_session.get("id") if backtest_session else None,
            strategy=backtest_session.get("strategy") if backtest_session else None,
            symbol=backtest_session.get("symbol") if backtest_session else None
        )
        
        # Metrics Grid
        self.render_metrics_grid(
            backtest_results=backtest_session.get("results") if backtest_session else None
        )
        
        # Visual Row
        visual_col1, visual_col2 = st.columns([3, 2])
        
        with visual_col1:
            self.render_equity_curve()
        
        with visual_col2:
            self.render_win_loss_distribution()
        
        # Expand/Collapse sections
        with st.expander("📊 Parameter Optimization Details", expanded=False):
            self.render_parameter_heatmap()
        
        # ML Progress Tracker
        self.render_ml_progress_tracker(
            progress_items=backtest_session.get("progress") if backtest_session else None
        )
        
        # Insight Card
        self.render_backtest_insight_card(
            insight_text=backtest_session.get("insight") if backtest_session else None
        )
        
        # Action Buttons
        self.render_action_buttons()


# Create a singleton instance
_backtester_ui = None

def get_backtester_ui():
    """
    Get the singleton BacktesterUI instance.
    
    Returns:
        BacktesterUI instance
    """
    global _backtester_ui
    if _backtester_ui is None:
        _backtester_ui = BacktesterUI()
    return _backtester_ui


def render_new_backtest_form():
    """Render a form for creating a new backtest."""
    st.markdown("### 🧪 New Backtest Configuration")
    
    with st.form("new_backtest_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.text_input("Symbol", value="AAPL")
            strategy = st.selectbox(
                "Strategy",
                [
                    "momentum_breakout",
                    "mean_reversion",
                    "trend_following",
                    "value_dividend",
                    "volatility_etf",
                    "ai_sentiment"
                ]
            )
        
        with col2:
            start_date = st.date_input("Start Date", value=datetime.datetime.now() - datetime.timedelta(days=90))
            end_date = st.date_input("End Date", value=datetime.datetime.now())
        
        st.markdown("#### Strategy Parameters")
        param_col1, param_col2, param_col3 = st.columns(3)
        
        with param_col1:
            ma_short = st.number_input("MA Short", min_value=5, max_value=50, value=20)
            rsi_period = st.number_input("RSI Period", min_value=2, max_value=30, value=14)
        
        with param_col2:
            ma_long = st.number_input("MA Long", min_value=20, max_value=200, value=50)
            bb_std = st.number_input("Bollinger Std", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
        
        with param_col3:
            optimization = st.checkbox("Run Parameter Optimization", value=True)
            variants = st.number_input("Variants to Test", min_value=1, max_value=20, value=5)
        
        submitted = st.form_submit_button("Run Backtest", type="primary", use_container_width=True)
        
        if submitted:
            st.session_state.new_backtest = {
                "symbol": symbol,
                "strategy": strategy,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "params": {
                    "ma_short": ma_short,
                    "ma_long": ma_long,
                    "rsi_period": rsi_period,
                    "bb_std": bb_std,
                    "optimization": optimization,
                    "variants": variants
                }
            }
            return True
    
    return False
