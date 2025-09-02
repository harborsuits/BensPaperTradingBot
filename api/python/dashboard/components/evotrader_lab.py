"""
EvoTrader Strategy Lab Component for BensBot Dashboard

This component provides a dashboard interface for interacting with the
EvoTrader evolutionary strategy research platform, allowing users to:
1. Launch strategy evolution processes
2. View candidate strategies from evolution runs
3. Evaluate strategies through backtesting
4. Promote evolved strategies to live trading
5. Monitor performance of evolved strategies
"""

import os
import sys
import json
import time
import pandas as pd
import streamlit as st
import altair as alt
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Import EvoTrader integration
try:
    from trading_bot.research.evotrader_integration.bridge import EvoTraderIntegration
    EVOTRADER_AVAILABLE = True
except ImportError:
    EVOTRADER_AVAILABLE = False


def render_evotrader_lab(data_service):
    """
    Render the EvoTrader Strategy Lab component.
    
    Args:
        data_service: DataService instance for accessing backend data
    """
    st.header("EvoTrader Strategy Lab 🧬", divider="blue")
    
    if not EVOTRADER_AVAILABLE:
        st.error("EvoTrader integration is not available. Please ensure the EvoTrader repository is properly installed.")
        st.info("To install EvoTrader, clone the repository: `git clone https://github.com/TheClitCommander/Evotrader.git`")
        return
    
    # Initialize EvoTrader integration
    integration = EvoTraderIntegration()
    
    if not integration.available:
        st.error("EvoTrader bridge could not be initialized. Please check that the repository is properly installed.")
        repo_path = st.text_input("EvoTrader Repository Path:", value="/Users/bendickinson/Desktop/Trading:BenBot/Evotrader")
        if st.button("Try Again"):
            integration = EvoTraderIntegration(repo_path)
            if integration.available:
                st.success("EvoTrader bridge initialized successfully!")
                st.rerun()
        return
    
    # Create tabs for different sections
    tabs = st.tabs(["Evolution Control", "Candidate Strategies", "Strategy Deployment"])
    
    # Tab 1: Evolution Control
    with tabs[0]:
        render_evolution_control(integration, data_service)
    
    # Tab 2: Candidate Strategies
    with tabs[1]:
        render_candidate_strategies(integration, data_service)
    
    # Tab 3: Strategy Deployment
    with tabs[2]:
        render_strategy_deployment(integration, data_service)


def render_evolution_control(integration, data_service):
    """
    Render evolution control panel.
    
    Args:
        integration: EvoTraderIntegration instance
        data_service: DataService instance for accessing backend data
    """
    st.subheader("Strategy Evolution Control")
    
    # Session state for evolution status
    if 'evolution_running' not in st.session_state:
        st.session_state.evolution_running = False
    if 'evolution_progress' not in st.session_state:
        st.session_state.evolution_progress = 0
    if 'evolution_results' not in st.session_state:
        st.session_state.evolution_results = None
    
    # Create form for evolution parameters
    with st.form("evolution_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            asset_class = st.selectbox(
                "Asset Class:",
                options=["forex", "crypto"],
                index=0
            )
            
            generations = st.slider(
                "Generations:",
                min_value=5,
                max_value=50,
                value=10,
                step=5
            )
            
            population_size = st.slider(
                "Population Size:",
                min_value=20,
                max_value=200,
                value=50,
                step=10
            )
        
        with col2:
            if asset_class == "forex":
                default_symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
                default_timeframes = ["1h", "4h", "1d"]
            else:  # crypto
                default_symbols = ["BTC/USD", "ETH/USD"]
                default_timeframes = ["15m", "1h", "4h", "1d"]
            
            symbols = st.multiselect(
                "Symbols:",
                options=default_symbols + ["USDCAD", "EURGBP", "AUDJPY", "ETH/USD", "SOL/USD", "XRP/USD"],
                default=default_symbols
            )
            
            timeframes = st.multiselect(
                "Timeframes:",
                options=["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
                default=default_timeframes
            )
        
        submit_button = st.form_submit_button("Start Evolution Run")
    
    # Handle form submission
    if submit_button:
        if st.session_state.evolution_running:
            st.warning("Evolution process is already running. Please wait for it to complete.")
        else:
            st.session_state.evolution_running = True
            st.session_state.evolution_progress = 0
            
            # Launch evolution in a separate thread (this is just simulated for the demo)
            st.info(f"Starting evolution run with {symbols} on {timeframes} for {generations} generations")
            
            # In a real implementation, we would launch a background process
            # Here we'll simulate progress updates
            st.session_state.evolution_results = None
    
    # Show progress if evolution is running
    if st.session_state.evolution_running:
        progress_bar = st.progress(st.session_state.evolution_progress)
        status_text = st.empty()
        
        # Update progress (simulation)
        if st.session_state.evolution_progress < 1.0:
            # Increment progress
            st.session_state.evolution_progress += 0.1
            status_text.text(f"Generation {int(st.session_state.evolution_progress * generations)}/{generations}...")
            
            # Rerun to update progress
            time.sleep(0.1)  # Simulate processing time
            st.rerun()
        else:
            # Evolution complete
            status_text.text("Evolution complete!")
            st.session_state.evolution_running = False
            
            # Generate dummy results for demo
            st.session_state.evolution_results = {
                "generations": [{"generation_number": i+1} for i in range(generations)],
                "asset_class": asset_class,
                "symbols": symbols,
                "timeframes": timeframes,
                "start_time": datetime.now().isoformat(),
                "end_time": (datetime.now() + timedelta(minutes=5)).isoformat()
            }
            
            st.success("Evolution completed successfully!")
    
    # Display previous evolution results if available
    if st.session_state.evolution_results:
        st.subheader("Recent Evolution Results")
        
        results = st.session_state.evolution_results
        
        # Show results summary
        st.write(f"Asset Class: {results['asset_class']}")
        st.write(f"Symbols: {', '.join(results['symbols'])}")
        st.write(f"Timeframes: {', '.join(results['timeframes'])}")
        st.write(f"Generations: {len(results['generations'])}")
        
        # Plot dummy fitness improvement chart for demo
        generations = list(range(1, len(results['generations'])+1))
        fitness_values = [50 + i*5 + (10 - i*0.5) * (i/5) for i in range(len(generations))]
        
        fitness_df = pd.DataFrame({
            'Generation': generations,
            'Fitness': fitness_values
        })
        
        chart = alt.Chart(fitness_df).mark_line(point=True).encode(
            x='Generation',
            y='Fitness',
            tooltip=['Generation', 'Fitness']
        ).properties(
            title='Evolution Fitness Progression'
        )
        
        st.altair_chart(chart, use_container_width=True)


def render_candidate_strategies(integration, data_service):
    """
    Render candidate strategies panel.
    
    Args:
        integration: EvoTraderIntegration instance
        data_service: DataService instance for accessing backend data
    """
    st.subheader("Candidate Strategies")
    
    # Create dummy candidate strategies for demo
    candidates = [
        {
            "strategy_id": "gen10_strat3_ab12cd34",
            "fitness": 87.5,
            "metrics": {
                "sharpe_ratio": 2.4,
                "total_return_pct": 26.7,
                "win_rate_pct": 62.3,
                "profit_factor": 1.8,
                "max_drawdown_pct": 12.1
            },
            "parameters": {
                "fast_period": 12,
                "slow_period": 26,
                "signal_threshold": 0.001,
                "stop_loss_pct": 1.5,
                "take_profit_pct": 3.0
            }
        },
        {
            "strategy_id": "gen10_strat7_ef56gh78",
            "fitness": 82.1,
            "metrics": {
                "sharpe_ratio": 2.1,
                "total_return_pct": 22.3,
                "win_rate_pct": 58.7,
                "profit_factor": 1.7,
                "max_drawdown_pct": 10.5
            },
            "parameters": {
                "fast_period": 8,
                "slow_period": 21,
                "signal_threshold": 0.0015,
                "stop_loss_pct": 2.0,
                "take_profit_pct": 4.0
            }
        },
        {
            "strategy_id": "gen10_strat1_ij90kl12",
            "fitness": 79.8,
            "metrics": {
                "sharpe_ratio": 1.9,
                "total_return_pct": 19.5,
                "win_rate_pct": 60.1,
                "profit_factor": 1.6,
                "max_drawdown_pct": 11.2
            },
            "parameters": {
                "fast_period": 10,
                "slow_period": 30,
                "signal_threshold": 0.002,
                "stop_loss_pct": 1.8,
                "take_profit_pct": 3.5
            }
        }
    ]
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        min_sharpe = st.slider("Min Sharpe Ratio:", 0.0, 3.0, 1.5, 0.1)
    with col2:
        min_win_rate = st.slider("Min Win Rate (%):", 50.0, 70.0, 55.0, 1.0)
    
    # Filter strategies
    filtered_candidates = [
        c for c in candidates 
        if c["metrics"]["sharpe_ratio"] >= min_sharpe and 
           c["metrics"]["win_rate_pct"] >= min_win_rate
    ]
    
    # Display candidates
    if not filtered_candidates:
        st.warning("No candidate strategies match the current filters.")
    else:
        for i, candidate in enumerate(filtered_candidates):
            with st.expander(f"Strategy {i+1}: {candidate['strategy_id']} (Fitness: {candidate['fitness']:.1f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Performance Metrics")
                    metrics = candidate["metrics"]
                    metrics_df = pd.DataFrame({
                        'Metric': list(metrics.keys()),
                        'Value': list(metrics.values())
                    })
                    st.dataframe(metrics_df, hide_index=True)
                
                with col2:
                    st.subheader("Strategy Parameters")
                    params = candidate["parameters"]
                    params_df = pd.DataFrame({
                        'Parameter': list(params.keys()),
                        'Value': list(params.values())
                    })
                    st.dataframe(params_df, hide_index=True)
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Run Backtest", key=f"backtest_{i}"):
                        st.session_state[f"backtest_running_{i}"] = True
                
                with col2:
                    if st.button("Run Forward Test", key=f"forward_{i}"):
                        st.session_state[f"forward_running_{i}"] = True
                
                with col3:
                    if st.button("Promote to Trading", key=f"promote_{i}"):
                        st.session_state[f"promoted_{i}"] = True
                
                # Show backtest results if available
                if st.session_state.get(f"backtest_running_{i}", False):
                    st.subheader("Backtest Results")
                    
                    # Create dummy backtest data for visualization
                    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
                    equity = [10000]
                    
                    for j in range(1, 100):
                        # Simulate equity curve with some randomness
                        daily_return = 0.001 * (1 + 0.5 * (j % 10)) + 0.002 * (j % 3 - 1)
                        equity.append(equity[-1] * (1 + daily_return))
                    
                    backtest_df = pd.DataFrame({
                        'Date': dates,
                        'Equity': equity
                    })
                    
                    # Plot equity curve
                    chart = alt.Chart(backtest_df).mark_line().encode(
                        x='Date',
                        y='Equity',
                        tooltip=['Date', 'Equity']
                    ).properties(
                        title='Backtest Equity Curve'
                    )
                    
                    st.altair_chart(chart, use_container_width=True)
                    
                    # Show key metrics
                    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                    metrics_col1.metric("Total Return", f"{26.7}%")
                    metrics_col2.metric("Sharpe Ratio", f"{2.4}")
                    metrics_col3.metric("Win Rate", f"{62.3}%")
                    metrics_col4.metric("Max Drawdown", f"{12.1}%")
                
                # Show promotion status if promoted
                if st.session_state.get(f"promoted_{i}", False):
                    st.success(f"Strategy {candidate['strategy_id']} has been promoted to paper trading!")


def render_strategy_deployment(integration, data_service):
    """
    Render strategy deployment panel.
    
    Args:
        integration: EvoTraderIntegration instance
        data_service: DataService instance for accessing backend data
    """
    st.subheader("Strategy Deployment")
    
    # Create dummy deployed strategies for demo
    deployed = [
        {
            "strategy_id": "gen8_strat2_mn34op56",
            "deployed_date": "2023-04-15",
            "environment": "paper",
            "status": "active",
            "performance": {
                "total_return_pct": 12.5,
                "sharpe_ratio": 1.8,
                "win_rate_pct": 59.2,
                "profit_factor": 1.6
            }
        },
        {
            "strategy_id": "gen6_strat5_qr78st90",
            "deployed_date": "2023-03-22",
            "environment": "paper",
            "status": "paused",
            "performance": {
                "total_return_pct": 8.3,
                "sharpe_ratio": 1.5,
                "win_rate_pct": 56.8,
                "profit_factor": 1.4
            }
        }
    ]
    
    # Display deployed strategies
    if not deployed:
        st.info("No strategies have been deployed yet. Promote candidate strategies to see them here.")
    else:
        for i, strategy in enumerate(deployed):
            status_color = "green" if strategy["status"] == "active" else "orange"
            
            with st.container():
                st.markdown(f"### {strategy['strategy_id']} "
                           f"<span style='color:{status_color};'>[{strategy['status'].upper()}]</span>", 
                           unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Deployed Date:** {strategy['deployed_date']}")
                    st.write(f"**Environment:** {strategy['environment']}")
                
                with col2:
                    st.write(f"**Total Return:** {strategy['performance']['total_return_pct']}%")
                    st.write(f"**Sharpe Ratio:** {strategy['performance']['sharpe_ratio']}")
                
                with col3:
                    st.write(f"**Win Rate:** {strategy['performance']['win_rate_pct']}%")
                    st.write(f"**Profit Factor:** {strategy['performance']['profit_factor']}")
                
                # Action buttons
                button_col1, button_col2, button_col3, button_col4 = st.columns(4)
                
                with button_col1:
                    if strategy["status"] == "active":
                        if st.button("Pause", key=f"pause_{i}"):
                            st.info(f"Strategy {strategy['strategy_id']} paused.")
                    else:
                        if st.button("Resume", key=f"resume_{i}"):
                            st.success(f"Strategy {strategy['strategy_id']} resumed.")
                
                with button_col2:
                    if st.button("Stop", key=f"stop_{i}"):
                        st.warning(f"Strategy {strategy['strategy_id']} stopped.")
                
                with button_col3:
                    if st.button("Promote to Live", key=f"live_{i}"):
                        st.success(f"Strategy {strategy['strategy_id']} promoted to live trading!")
                
                with button_col4:
                    if st.button("View Trades", key=f"trades_{i}"):
                        st.session_state[f"show_trades_{i}"] = True
                
                st.markdown("---")
                
                # Show trades if requested
                if st.session_state.get(f"show_trades_{i}", False):
                    st.subheader("Recent Trades")
                    
                    # Create dummy trade data
                    trades = [
                        {"date": "2023-04-20", "symbol": "EURUSD", "direction": "BUY", "profit_pips": 35, "profit_usd": 350},
                        {"date": "2023-04-18", "symbol": "GBPUSD", "direction": "SELL", "profit_pips": -12, "profit_usd": -150},
                        {"date": "2023-04-15", "symbol": "EURUSD", "direction": "BUY", "profit_pips": 22, "profit_usd": 220},
                        {"date": "2023-04-12", "symbol": "USDJPY", "direction": "SELL", "profit_pips": 45, "profit_usd": 375}
                    ]
                    
                    trades_df = pd.DataFrame(trades)
                    st.dataframe(trades_df, hide_index=True)
                    
                    # Hide trades button
                    if st.button("Hide Trades", key=f"hide_trades_{i}"):
                        st.session_state[f"show_trades_{i}"] = False
                        st.rerun()
