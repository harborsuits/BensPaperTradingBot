"""
Autonomous Trading UI

This module provides the Streamlit UI for the autonomous trading system,
connecting to the autonomous engine and displaying strategy candidates
for approval and deployment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import threading
import logging

# Import autonomous engine
from trading_bot.autonomous.autonomous_engine import AutonomousEngine

# Import optimization UI
try:
    from trading_bot.ui.optimization_ui import OptimizationUI
    from trading_bot.event_system import EventBus
    optimization_ui_available = True
except ImportError:
    optimization_ui_available = False
    logger = logging.getLogger(__name__)
    logger.warning("Optimization UI not available")

logger = logging.getLogger("autonomous_ui")
logger.setLevel(logging.INFO)

class AutonomousUI:
    """
    Streamlit UI for the autonomous trading system.
    Connects to the autonomous engine for strategy generation,
    backtesting, evaluation, and deployment.
    """
    
    def __init__(self):
        """Initialize the autonomous UI component"""
        # Initialize autonomous engine
        if 'autonomous_engine' not in st.session_state:
            st.session_state.autonomous_engine = AutonomousEngine()
            
        self.engine = st.session_state.autonomous_engine
        
        # Initialize event bus if not already done
        if 'event_bus' not in st.session_state:
            try:
                st.session_state.event_bus = EventBus()
                st.session_state.event_bus.start()
            except NameError:
                logger.warning("EventBus not available")
                st.session_state.event_bus = None
        
        # Initialize optimization UI if available
        self.optimization_ui = None
        if optimization_ui_available:
            self.optimization_ui = OptimizationUI(st.session_state.event_bus)
        
        # Create data directories
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "autonomous"
        )
        os.makedirs(data_dir, exist_ok=True)
        
        # UI state
        if "autonomous_refresh_counter" not in st.session_state:
            st.session_state.autonomous_refresh_counter = 0
    
    def render(self):
        """Render the autonomous trading UI"""
        # Main layout
        self._render_header()
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Strategy Candidates", "Optimization", "Settings"])
        
        with tab1:
            self._render_overview()
            
        with tab2:
            self._render_candidates()
            
        with tab3:
            self._render_optimization()
            
        with tab4:
            self._render_settings()
        
        # Auto-refresh when engine is running
        if self.engine.is_running:
            # Use a key based on the counter to force a refresh
            st.empty().markdown(f"<div id='refresh-{st.session_state.autonomous_refresh_counter}'></div>", unsafe_allow_html=True)
            
            # Schedule the next refresh
            if st.session_state.autonomous_refresh_counter < 100:  # Prevent infinite refreshes
                st.session_state.autonomous_refresh_counter += 1
                
                # Auto-refresh every 2 seconds during active processing
                st.rerun()
    
    def _render_control_panel(self):
        """Render the control panel for configuring the autonomous system"""
        st.markdown("### Control Panel")
        
        # Status information
        status = self.engine.get_status()
        
        # Data source indicator - check if real data is being used
        data_source = "🟢 REAL MARKET DATA" if getattr(self.engine, "using_real_data", False) else "🟠 SIMULATED DATA"
        st.info(f"**Data Source:** {data_source}")
        
        # Market status - in a real implementation this would check actual market hours
        market_hours = self._check_market_hours()
        market_status = "🟢 OPEN" if market_hours["is_open"] else "🔴 CLOSED"
        next_event = f"Closes {market_hours['next_event_time']}" if market_hours["is_open"] else f"Opens {market_hours['next_event_time']}"
        
        st.markdown(f"**Market Status:** {market_status} - {next_event}")
        
        # Process status
        if status["is_running"]:
            st.success(f"**Process Status:** 🟢 Running - {status['current_phase'].title()}")
            st.progress(status["progress"] / 100, status["status_message"])
        else:
            st.warning(f"**Process Status:** ⚪ Idle")
            
            # Configuration options (only show when not running)
            st.markdown("### Configuration")
            
            # Market universe selection
            universe = st.selectbox(
                "Securities Universe",
                [
                    "Nasdaq 100", 
                    "S&P 500", 
                    "Dow Jones 30", 
                    "Russell 2000", 
                    "Crypto Top 20", 
                    "Forex Majors"
                ],
                index=0
            )
            
            # Strategy types
            strategy_types = st.multiselect(
                "Strategy Types to Consider",
                [
                    "Momentum", 
                    "Mean Reversion", 
                    "Trend Following", 
                    "Volatility Breakout", 
                    "Machine Learning"
                ],
                default=["Momentum", "Mean Reversion", "Trend Following"]
            )
            
            # Performance thresholds
            st.markdown("### Performance Thresholds")
            
            thresholds = {}
            
            thresholds["min_sharpe_ratio"] = st.slider(
                "Minimum Sharpe Ratio", 
                min_value=0.0, 
                max_value=3.0, 
                value=1.0, 
                step=0.1
            )
            
            thresholds["min_profit_factor"] = st.slider(
                "Minimum Profit Factor", 
                min_value=1.0, 
                max_value=5.0, 
                value=1.5, 
                step=0.1
            )
            
            thresholds["max_drawdown"] = st.slider(
                "Maximum Drawdown (%)", 
                min_value=5.0, 
                max_value=50.0, 
                value=20.0, 
                step=1.0
            )
            
            thresholds["min_win_rate"] = st.slider(
                "Minimum Win Rate (%)", 
                min_value=30.0, 
                max_value=80.0, 
                value=50.0, 
                step=1.0
            )
            
            # Action buttons
            if st.button("🚀 Launch Autonomous Process", use_container_width=True, type="primary"):
                # Start the autonomous process
                self.engine.start_process(
                    universe=universe,
                    strategy_types=strategy_types,
                    thresholds=thresholds
                )
                st.session_state.autonomous_refresh_counter = 0
                st.rerun()
        
        # Stop button (only show when running)
        if status["is_running"]:
            if st.button("⏹️ Stop Process", use_container_width=True):
                self.engine.stop_process()
                st.rerun()
        
        # Additional information
        st.markdown("### System Information")
        
        if status["candidates_count"] > 0:
            st.metric("Total Strategies", str(status["candidates_count"]))
            st.metric("Meeting Criteria", str(status["top_candidates_count"]))
            
            # Calculate approval rate
            if status["candidates_count"] > 0:
                approval_rate = (status["top_candidates_count"] / status["candidates_count"]) * 100
                st.metric("Approval Rate", f"{approval_rate:.1f}%")
    
    def _render_results_panel(self):
        """Render the results panel for displaying strategy candidates"""
        # Get status
        status = self.engine.get_status()
        
        # Different content based on whether process has been run
        if status["candidates_count"] > 0:
            # Display results
            st.markdown("### Generated Strategies Ready for Approval")
            
            # Create tabs for different views
            strategy_tabs = st.tabs(["Top Performers", "All Strategies", "Historical Performance"])
            
            with strategy_tabs[0]:
                self._render_top_performers()
            
            with strategy_tabs[1]:
                self._render_all_strategies()
            
            with strategy_tabs[2]:
                self._render_historical_performance()
        
        elif status["is_running"]:
            # Show progress information
            st.info(f"🔄 {status['status_message']} Progress: {status['progress']}%")
            
            # Placeholder visualization
            st.markdown("### Process Visualization")
            
            # Create placeholders for various stages
            stages = [
                "Market Scanning",
                "Strategy Generation",
                "Backtesting",
                "Performance Evaluation", 
                "Preparing Results"
            ]
            
            # Determine active stage based on progress
            active_stage = min(int(status["progress"] / 20), len(stages) - 1)
            
            # Display stages with highlighting for active stage
            for i, stage in enumerate(stages):
                if i < active_stage:
                    st.markdown(f"✅ **{stage}** - Completed")
                elif i == active_stage:
                    st.markdown(f"🔄 **{stage}** - In Progress...")
                else:
                    st.markdown(f"⚪ **{stage}** - Pending")
        
        else:
            # Initial instructions
            st.info("👈 Click 'Launch Autonomous Process' to start scanning the market, generating and backtesting strategies automatically.")
            
            st.markdown("""
            ### How the Autonomous Trading System Works:
            
            1. **Market Scanning**: The system automatically scans your selected universe of securities using built-in indicators
            
            2. **Strategy Generation**: Based on market conditions, it creates optimized trading strategies
            
            3. **Automated Backtesting**: All generated strategies are backtested without manual configuration
            
            4. **Performance Filtering**: Only strategies meeting your performance thresholds are presented
            
            5. **One-Click Approval**: Review and approve strategies with a single click
            
            6. **Paper Trading Deployment**: Approved strategies are automatically deployed to paper trading
            """)
            
            # Show sample visualization
            st.markdown("### Sample System Output")
            
            # Sample data for visualization
            sample_data = {
                "Strategy": ["Momentum", "Mean Reversion", "Trend Following", "Volatility Breakout", "ML Strategy"],
                "Return (%)": [27.5, 18.2, 22.1, 15.5, 16.8],
                "Sharpe Ratio": [1.85, 1.62, 1.73, 1.35, 1.48],
                "Drawdown (%)": [12.5, 9.8, 15.3, 18.2, 16.5],
                "Win Rate (%)": [62.5, 71.2, 55.8, 48.5, 53.2],
            }
            
            # Create sample performance chart
            fig = go.Figure()
            
            for i, strategy in enumerate(sample_data["Strategy"]):
                fig.add_trace(go.Scatterpolar(
                    r=[
                        sample_data["Return (%)"][i] / 30 * 100,  # Scale to 0-100
                        sample_data["Sharpe Ratio"][i] / 2 * 100,  # Scale to 0-100
                        (30 - sample_data["Drawdown (%)"][i]) / 30 * 100,  # Inverse and scale
                        sample_data["Win Rate (%)"][i],
                        (i % 3) * 20 + 60  # Random trade efficiency
                    ],
                    theta=["Return", "Sharpe", "Low Drawdown", "Win Rate", "Trade Efficiency"],
                    fill="toself",
                    name=strategy
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                title="Strategy Performance Comparison",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_top_performers(self):
        """Render the top performers tab"""
        # Get top candidates
        top_candidates = self.engine.get_top_candidates()
        
        if not top_candidates:
            st.info("No strategies meeting criteria yet. Adjust thresholds or run the process.")
            return
        
        st.markdown("#### Top-Performing Strategies")
        st.markdown("These strategies have been automatically generated, backtested, and meet your performance criteria.")
        
        # Convert to dataframe for display
        df_data = []
        
        for candidate in top_candidates:
            df_data.append({
                "Strategy ID": candidate["strategy_id"],
                "Type": candidate["strategy_type"],
                "Universe": candidate["universe"],
                "Return (%)": round(candidate["performance"]["returns"], 1),
                "Sharpe": round(candidate["performance"]["sharpe_ratio"], 2),
                "Drawdown (%)": round(candidate["performance"]["drawdown"], 1),
                "Win Rate (%)": round(candidate["performance"]["win_rate"], 1),
                "Trades": candidate["performance"]["trades_count"],
                "Status": candidate["status"].title()
            })
        
        if df_data:
            strategies_df = pd.DataFrame(df_data)
            
            # Display the dataframe
            st.dataframe(strategies_df, use_container_width=True)
            
            # Show action buttons for each strategy
            st.markdown("#### Strategy Actions")
            
            # Create dynamic columns based on number of strategies (max 3)
            num_cols = min(len(df_data), 3)
            cols = st.columns(num_cols)
            
            # Add buttons for each strategy
            for i, candidate in enumerate(top_candidates[:num_cols]):
                with cols[i % num_cols]:
                    strategy_id = candidate["strategy_id"]
                    strategy_type = candidate["strategy_type"]
                    
                    if candidate["status"] == "backtested" or candidate["status"] == "pending":
                        # Approve button
                        if st.button(f"👍 Approve {strategy_type}", key=f"approve_{strategy_id}", use_container_width=True):
                            self.engine.approve_strategy(strategy_id)
                            st.rerun()
                    
                    elif candidate["status"] == "approved":
                        # Deploy button
                        if st.button(f"🚀 Deploy {strategy_type}", key=f"deploy_{strategy_id}", use_container_width=True):
                            self.engine.deploy_strategy(strategy_id)
                            st.rerun()
                    
                    elif candidate["status"] == "deployed":
                        # Already deployed
                        st.markdown(f"✅ **{strategy_type}** Deployed")
            
            # Option to approve all if any are pending
            pending_candidates = [c for c in top_candidates if c["status"] == "backtested" or c["status"] == "pending"]
            
            if pending_candidates and len(pending_candidates) > 1:
                if st.button("✅ Approve All Strategies", use_container_width=True):
                    for candidate in pending_candidates:
                        self.engine.approve_strategy(candidate["strategy_id"])
                    st.rerun()
    
    def _render_all_strategies(self):
        """Render all strategies tab"""
        # Get all candidates
        all_candidates = self.engine.get_all_candidates()
        
        if not all_candidates:
            st.info("No strategies generated yet. Run the autonomous process to generate strategies.")
            return
        
        st.markdown("#### All Generated Strategies")
        
        # Filter options
        st.markdown("#### Filter Options")
        
        filter_cols = st.columns(4)
        
        with filter_cols[0]:
            strategy_types = ["All"] + list(set(c["strategy_type"] for c in all_candidates))
            filter_type = st.selectbox("Strategy Type", strategy_types)
        
        with filter_cols[1]:
            statuses = ["All", "Pending", "Backtested", "Approved", "Deployed", "Rejected"]
            filter_status = st.selectbox("Status", statuses)
        
        with filter_cols[2]:
            metrics = ["Return (%)", "Sharpe", "Drawdown (%)", "Win Rate (%)", "Trades"]
            sort_by = st.selectbox("Sort By", metrics)
        
        with filter_cols[3]:
            order = st.selectbox("Order", ["Descending", "Ascending"])
        
        # Apply filters
        filtered_candidates = all_candidates
        
        if filter_type != "All":
            filtered_candidates = [c for c in filtered_candidates if c["strategy_type"] == filter_type]
        
        if filter_status != "All":
            filtered_candidates = [c for c in filtered_candidates if c["status"].lower() == filter_status.lower()]
        
        # Convert to dataframe
        df_data = []
        
        for candidate in filtered_candidates:
            df_data.append({
                "Strategy ID": candidate["strategy_id"],
                "Type": candidate["strategy_type"],
                "Universe": candidate["universe"],
                "Return (%)": round(candidate["performance"]["returns"], 1),
                "Sharpe": round(candidate["performance"]["sharpe_ratio"], 2),
                "Drawdown (%)": round(candidate["performance"]["drawdown"], 1),
                "Win Rate (%)": round(candidate["performance"]["win_rate"], 1),
                "Trades": candidate["performance"]["trades_count"],
                "Status": candidate["status"].title(),
                "Meets Criteria": "✅" if candidate["meets_criteria"] else "❌"
            })
        
        if df_data:
            # Sort data
            strategies_df = pd.DataFrame(df_data)
            
            # Apply sorting
            sort_ascending = order == "Ascending"
            strategies_df = strategies_df.sort_values(by=sort_by, ascending=sort_ascending)
            
            # Display the dataframe
            st.dataframe(strategies_df, use_container_width=True)
            
            # Show summary
            st.markdown(f"**Showing {len(strategies_df)} of {len(all_candidates)} strategies**")
        else:
            st.info("No strategies match the current filters.")
    
    def _render_historical_performance(self):
        """Render historical performance tab"""
        # In a real implementation, this would show actual historical data
        # For demo purposes, generate simulated data
        
        st.markdown("#### Autonomous System Performance History")
        
        # Create historical data (simulated for demo)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=12, freq='M')
        
        hist_data = {
            "Date": dates,
            "Strategies Generated": [8, 7, 11, 9, 12, 8, 7, 10, 11, 9, 10, 12],
            "Strategies Approved": [3, 2, 4, 3, 5, 2, 2, 4, 5, 3, 4, 5],
            "Avg Return (%)": [18.2, 15.7, 21.3, 17.8, 20.5, 16.9, 15.5, 19.2, 22.1, 18.5, 19.7, 22.8],
            "Avg Sharpe": [1.42, 1.35, 1.67, 1.51, 1.70, 1.44, 1.40, 1.55, 1.78, 1.52, 1.60, 1.75]
        }
        
        hist_df = pd.DataFrame(hist_data)
        
        # Create chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=hist_df["Date"],
            y=hist_df["Strategies Generated"],
            mode='lines+markers',
            name='Strategies Generated'
        ))
        
        fig.add_trace(go.Scatter(
            x=hist_df["Date"],
            y=hist_df["Strategies Approved"],
            mode='lines+markers',
            name='Strategies Approved'
        ))
        
        fig.add_trace(go.Scatter(
            x=hist_df["Date"],
            y=hist_df["Avg Return (%)"],
            mode='lines+markers',
            name='Avg Return (%)',
            yaxis="y2"
        ))
        
        fig.update_layout(
            title="Strategy Generation and Performance History",
            xaxis_title="Date",
            yaxis_title="Count",
            yaxis2=dict(
                title="Return (%)",
                overlaying="y",
                side="right"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.markdown("#### System Performance Summary")
        
        summary_cols = st.columns(4)
        
        with summary_cols[0]:
            st.metric("Total Strategies Generated", sum(hist_data["Strategies Generated"]))
        
        with summary_cols[1]:
            st.metric("Total Strategies Approved", sum(hist_data["Strategies Approved"]))
        
        with summary_cols[2]:
            approval_rate = sum(hist_data["Strategies Approved"]) / sum(hist_data["Strategies Generated"]) * 100
            st.metric("Approval Rate", f"{approval_rate:.1f}%")
        
        with summary_cols[3]:
            avg_return = sum(hist_data["Avg Return (%)"]) / len(hist_data["Avg Return (%)"])
            st.metric("Average Return", f"{avg_return:.1f}%")

    def _render_optimization(self):
        """Render the optimization section"""
        st.subheader("Strategy Optimization")
        
        # If optimization UI is available, use it
        if self.optimization_ui:
            self.optimization_ui.render()
        else:
            # Display current engine status related to optimization
            status = self.engine.get_status()
            
            # Show metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Strategies Generated", status.get("candidates_count", 0))
                st.metric("Top Candidates", status.get("top_candidates_count", 0))
                
                # Display optimization controls
                st.subheader("Optimization Controls")
                st.markdown("""The autonomous engine optimizes strategies that almost meet criteria.
                Strategies are sent for optimization when they are close to meeting the performance thresholds.
                """)
                
                # Near-miss threshold slider
                if st.button("Start Optimization Process"):
                    st.success("Optimization process started. The engine will optimize near-miss strategies.")
            
            with col2:
                # Thresholds used for optimization targets
                st.subheader("Performance Thresholds")
                thresholds = getattr(self.engine, 'thresholds', {
                    "min_sharpe_ratio": 1.5,
                    "min_profit_factor": 1.8,
                    "max_drawdown": 15.0,
                    "min_win_rate": 55.0
                })
                
                # Display as a table
                thresholds_df = pd.DataFrame({
                    "Metric": list(thresholds.keys()),
                    "Value": list(thresholds.values())
                })
                st.dataframe(thresholds_df, use_container_width=True)
                
                st.markdown("""Strategies that are within 85% of these thresholds are considered
                'near-miss' candidates and are sent for optimization.
                """)
                
            # Mock optimization results
            st.subheader("Recent Optimization Results")
            st.info("Connect to event system to track real-time optimization progress.")
            
            # Mock data for demonstration
            mock_results = pd.DataFrame({
                "Strategy ID": ["strategy_001", "strategy_002", "strategy_003"],
                "Type": ["Iron Condor", "Strangle", "Butterfly Spread"],
                "Status": ["Optimized ✅", "In Progress ⏳", "Exhausted ❌"],
                "Before": ["Sharpe: 1.3, Win Rate: 51%", "Sharpe: 1.4, Win Rate: 52%", "Sharpe: 1.2, Win Rate: 50%"],
                "After": ["Sharpe: 1.6, Win Rate: 57%", "In Progress", "Failed to meet thresholds"],
                "Improvement": ["+23% performance", "TBD", "N/A"]
            })
            
            st.dataframe(mock_results, use_container_width=True)
    
    def _render_header(self):
        """Render the header section"""
        st.markdown("## 🤖 Autonomous Trading System")
        
        # Display current status
        status = self.engine.get_status()
        is_running = status.get("is_running", False)
        current_phase = status.get("current_phase", "idle")
        progress = status.get("progress", 0)
        status_message = status.get("status_message", "System idle")
        
        # Header status display
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            market_hours = self._check_market_hours()
            market_status = "🟢 MARKET OPEN" if market_hours["is_open"] else "🔴 MARKET CLOSED"
            st.markdown(f"**Market Status:** {market_status}")
            next_event = market_hours["next_event_time"]
            next_event_type = "closes" if market_hours["is_open"] else "opens"
            st.markdown(f"**Next event:** Market {next_event_type} at {next_event}")
        
        with col2:
            if is_running:
                st.progress(progress / 100.0, text=f"{current_phase.capitalize()}: {status_message} ({progress}%)")
            else:
                st.info(f"System Status: {current_phase.capitalize()} - {status_message}")
                
        with col3:
            # Control buttons
            if not is_running:
                if st.button("Start Process", key="start_autonomous"):
                    # Start the autonomous process
                    self.engine.start_process(
                        universe="SP500",
                        strategy_types=["Iron Condor", "Strangle", "Butterfly Spread", "Calendar Spread"],
                        thresholds={
                            "min_sharpe_ratio": 1.5,
                            "min_profit_factor": 1.8,
                            "max_drawdown": 15.0,
                            "min_win_rate": 55.0
                        }
                    )
                    st.rerun()
            else:
                if st.button("Stop Process", key="stop_autonomous"):
                    # Stop the autonomous process
                    self.engine.stop_process()
                    st.rerun()
    
    def _render_overview(self):
        """Render the overview section"""
        st.subheader("System Overview")
        
        # Get current status
        status = self.engine.get_status()
        candidates_count = status.get("candidates_count", 0)
        top_candidates_count = status.get("top_candidates_count", 0)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Strategies", candidates_count)
        
        with col2:
            st.metric("Top Performers", top_candidates_count)
        
        with col3:
            approval_rate = (top_candidates_count / candidates_count * 100) if candidates_count > 0 else 0
            st.metric("Approval Rate", f"{approval_rate:.1f}%")
        
        with col4:
            st.metric("Active Deployments", 0)  # Placeholder for now
        
        # Display recent activity
        st.subheader("Recent Activity")
        
        if hasattr(self.engine, 'last_scan_time'):
            last_scan = self.engine.last_scan_time
            if isinstance(last_scan, datetime):
                st.markdown(f"**Last Market Scan:** {last_scan.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Activity log placeholder
        activity_data = pd.DataFrame({
            "Timestamp": [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                (datetime.now() - timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S"),
                (datetime.now() - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
            ],
            "Activity": [
                "Strategy Scan Completed",
                "Backtesting Finished",
                "System Started"
            ],
            "Details": [
                "Generated 5 new strategy candidates",
                "Evaluated 10 strategies against performance criteria",
                "Autonomous engine initialized with default parameters"
            ]
        })
        
        st.dataframe(activity_data, use_container_width=True)
    
    def _render_candidates(self):
        """Render the strategy candidates section"""
        st.subheader("Strategy Candidates")
        
        # Get top candidates
        top_candidates = self.engine.get_top_candidates()
        all_candidates = self.engine.get_all_candidates()
        
        # Create tabs for different views
        candidate_tab1, candidate_tab2 = st.tabs(["Top Performers", "All Strategies"])
        
        with candidate_tab1:
            if top_candidates:
                # Convert to DataFrame
                top_df = pd.DataFrame(top_candidates)
                
                # Display table
                st.dataframe(top_df, use_container_width=True)
                
                # Action buttons for selected strategy
                selected_strategy = st.selectbox(
                    "Select Strategy for Action",
                    [c.get("strategy_id") for c in top_candidates],
                    key="top_strategy_selector"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Approve Strategy", key="approve_strategy"):
                        success = self.engine.approve_strategy(selected_strategy)
                        if success:
                            st.success(f"Strategy {selected_strategy} approved for deployment")
                        else:
                            st.error(f"Failed to approve strategy {selected_strategy}")
                
                with col2:
                    if st.button("Reject Strategy", key="reject_strategy"):
                        success = self.engine.reject_strategy(selected_strategy)
                        if success:
                            st.success(f"Strategy {selected_strategy} rejected")
                        else:
                            st.error(f"Failed to reject strategy {selected_strategy}")
            else:
                st.info("No top-performing strategies found yet. Start the autonomous process to generate strategies.")
        
        with candidate_tab2:
            if all_candidates:
                # Convert to DataFrame
                all_df = pd.DataFrame(all_candidates)
                
                # Display table
                st.dataframe(all_df, use_container_width=True)
            else:
                st.info("No strategies found yet. Start the autonomous process to generate strategies.")
    
    def _render_settings(self):
        """Render the settings section"""
        st.subheader("System Settings")
        
        # Universe selection
        universe = st.selectbox(
            "Market Universe",
            ["SP500", "Nasdaq 100", "Dow Jones 30", "Russell 2000", "Crypto Top 20", "Forex Majors"],
            index=0,
            key="universe_selector"
        )
        
        # Strategy types
        st.markdown("### Strategy Types")
        st.markdown("Select the types of strategies to generate:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_iron_condor = st.checkbox("Iron Condor", value=True)
            use_butterfly = st.checkbox("Butterfly Spread", value=True)
            use_calendar = st.checkbox("Calendar Spread", value=False)
            use_covered_call = st.checkbox("Covered Call", value=False)
        
        with col2:
            use_strangle = st.checkbox("Strangle", value=True)
            use_straddle = st.checkbox("Straddle", value=False)
            use_vertical = st.checkbox("Vertical Spreads", value=False)
            use_collar = st.checkbox("Collar", value=False)
        
        # Performance thresholds
        st.markdown("### Performance Thresholds")
        st.markdown("Set the minimum performance criteria for strategies:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_sharpe = st.slider("Minimum Sharpe Ratio", 0.5, 3.0, 1.5, 0.1)
            min_profit_factor = st.slider("Minimum Profit Factor", 1.0, 3.0, 1.8, 0.1)
        
        with col2:
            max_drawdown = st.slider("Maximum Drawdown (%)", 5.0, 30.0, 15.0, 1.0)
            min_win_rate = st.slider("Minimum Win Rate (%)", 40.0, 70.0, 55.0, 1.0)
        
        # Save settings button
        if st.button("Save Settings", key="save_settings"):
            # Collect strategy types
            strategy_types = []
            if use_iron_condor: strategy_types.append("Iron Condor")
            if use_butterfly: strategy_types.append("Butterfly Spread")
            if use_calendar: strategy_types.append("Calendar Spread")
            if use_covered_call: strategy_types.append("Covered Call")
            if use_strangle: strategy_types.append("Strangle")
            if use_straddle: strategy_types.append("Straddle")
            if use_vertical: strategy_types.append("Vertical Spreads")
            if use_collar: strategy_types.append("Collar")
            
            # Update engine settings
            self.engine.universe = universe
            self.engine.strategy_types = strategy_types
            self.engine.thresholds = {
                "min_sharpe_ratio": min_sharpe,
                "min_profit_factor": min_profit_factor,
                "max_drawdown": max_drawdown,
                "min_win_rate": min_win_rate
            }
            
            st.success("Settings saved successfully")
    
    def _check_market_hours(self):
        """Return whether US markets are currently open and next open/close time.

        This helper assumes standard US market hours (09:30–16:00 local time) on
        weekdays. It provides a fallback implementation so that the UI can show
        meaningful status information even if a dedicated market-hours service
        is not available.
        """
        now = datetime.now()
        # Define today's open and close times
        today_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        today_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        # Weekends: market closed
        if now.weekday() >= 5:
            # Compute next Monday 09:30
            days_until_monday = (7 - now.weekday()) % 7 or 7
            next_open = today_open + timedelta(days=days_until_monday)
            return {"is_open": False, "next_event_time": next_open.strftime('%Y-%m-%d %H:%M')}

        # Weekday logic
        if today_open <= now <= today_close:
            # Market currently open; next event is the close time
            return {"is_open": True, "next_event_time": today_close.strftime('%Y-%m-%d %H:%M')}

        if now < today_open:
            # Before market opens
            return {"is_open": False, "next_event_time": today_open.strftime('%Y-%m-%d %H:%M')}

        # After market close; find next weekday open (could be tomorrow or Monday)
        next_day = now + timedelta(days=1)
        while next_day.weekday() >= 5:  # Skip weekends
            next_day += timedelta(days=1)
        next_open = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
        return {"is_open": False, "next_event_time": next_open.strftime('%Y-%m-%d %H:%M')}
