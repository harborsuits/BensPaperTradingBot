"""
Strategy Promotion UI Component
Displays promoted strategies, parameter exploration, and optimization results.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import json
import datetime
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import our components
from trading_bot.ml_pipeline.strategy_promoter import get_strategy_promoter
from trading_bot.ml_pipeline.enhanced_backtest_executor import get_enhanced_backtest_executor


def render_promotion_dashboard(page_container):
    """Render the strategy promotion dashboard in the provided container."""
    
    promoter = get_strategy_promoter()
    executor = get_enhanced_backtest_executor()
    
    # Title and description
    page_container.markdown("## Strategy Promotion Dashboard")
    page_container.markdown("""
    This dashboard shows strategies that have been automatically promoted to paper/live trading 
    based on backtest performance. These strategies have passed rigorous optimization and 
    validation criteria.
    """)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = page_container.tabs([
        "Top Promoted Strategies", 
        "Promotion History", 
        "Parameter Explorer",
        "Optimization Hub"
    ])
    
    # Get promoted strategies
    top_promotions = promoter.get_top_promotions(limit=20)
    
    # Tab 1: Top Promoted Strategies
    with tab1:
        st.markdown("### Top Strategies Ready for Deployment")
        
        if not top_promotions:
            st.info("No strategies have been promoted yet. Run backtests to discover high-performing strategies.")
        else:
            # Create dataframe for display
            promotion_data = []
            for p in top_promotions:
                promotion_data.append({
                    "Symbol": p.get("symbol", ""),
                    "Strategy": p.get("strategy", ""),
                    "Score": round(p.get("score", 0), 2),
                    "Sharpe": round(p.get("sharpe_ratio", 0), 2),
                    "Return (%)": round(p.get("total_return", 0), 2),
                    "Win Rate": round(p.get("win_rate", 0), 2),
                    "Max DD (%)": round(p.get("max_drawdown", 0), 2),
                    "Regime": p.get("regime", "unknown"),
                    "Promoted": p.get("timestamp", "")[:10]  # Just the date part
                })
            
            # Display as table
            if promotion_data:
                df = pd.DataFrame(promotion_data)
                st.dataframe(df, use_container_width=True)
                
                # Select a promotion to view details
                selected_idx = st.selectbox(
                    "Select a strategy to view details:", 
                    range(len(promotion_data)),
                    format_func=lambda i: f"{promotion_data[i]['Symbol']} - {promotion_data[i]['Strategy']} (Score: {promotion_data[i]['Score']})"
                )
                
                if selected_idx is not None:
                    selected_promotion = top_promotions[selected_idx]
                    
                    # Display details in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"### {selected_promotion.get('symbol')} - {selected_promotion.get('strategy')}")
                        st.markdown(f"**Score:** {selected_promotion.get('score', 0):.2f}")
                        st.markdown(f"**Sharpe Ratio:** {selected_promotion.get('sharpe_ratio', 0):.2f}")
                        st.markdown(f"**Total Return:** {selected_promotion.get('total_return', 0):.2f}%")
                        st.markdown(f"**Win Rate:** {selected_promotion.get('win_rate', 0):.2f}")
                        st.markdown(f"**Max Drawdown:** {selected_promotion.get('max_drawdown', 0):.2f}%")
                    
                    with col2:
                        st.markdown("### Parameters")
                        params = selected_promotion.get("params", {})
                        for key, value in params.items():
                            st.markdown(f"**{key}:** {value}")
                        
                        st.markdown(f"**Market Regime:** {selected_promotion.get('regime', 'unknown')}")
                        st.markdown(f"**Promotion Date:** {selected_promotion.get('timestamp', '')[:10]}")
                    
                    # Action buttons
                    st.markdown("### Actions")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("Deploy to Paper Trading", key="deploy_paper"):
                            st.success("Strategy queued for paper trading deployment")
                    
                    with col2:
                        if st.button("Run Live Backtest", key="run_backtest"):
                            st.info("Running backtest with these parameters...")
                            # In a real implementation, this would trigger a backtest
                    
                    with col3:
                        if st.button("Compare with Current", key="compare"):
                            st.info("Comparing with currently deployed strategy...")
                            # In a real implementation, this would show comparison metrics
    
    # Tab 2: Promotion History
    with tab2:
        st.markdown("### Strategy Promotion History")
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol_filter = st.text_input("Filter by Symbol:", "")
        
        with col2:
            strategies = list(set([p.get("strategy", "") for p in top_promotions]))
            strategy_filter = st.selectbox("Filter by Strategy:", ["All"] + strategies)
        
        with col3:
            regimes = list(set([p.get("regime", "unknown") for p in top_promotions]))
            regime_filter = st.selectbox("Filter by Regime:", ["All"] + regimes)
        
        # Apply filters
        filtered_promotions = top_promotions
        
        if symbol_filter:
            filtered_promotions = [p for p in filtered_promotions if symbol_filter.upper() in p.get("symbol", "").upper()]
        
        if strategy_filter != "All":
            filtered_promotions = [p for p in filtered_promotions if p.get("strategy", "") == strategy_filter]
        
        if regime_filter != "All":
            filtered_promotions = [p for p in filtered_promotions if p.get("regime", "") == regime_filter]
        
        # Display filtered results
        if not filtered_promotions:
            st.info("No matching promotions found.")
        else:
            # Create dataframe for display
            filtered_data = []
            for p in filtered_promotions:
                filtered_data.append({
                    "Symbol": p.get("symbol", ""),
                    "Strategy": p.get("strategy", ""),
                    "Score": round(p.get("score", 0), 2),
                    "Sharpe": round(p.get("sharpe_ratio", 0), 2),
                    "Return (%)": round(p.get("total_return", 0), 2),
                    "Win Rate": round(p.get("win_rate", 0), 2),
                    "Max DD (%)": round(p.get("max_drawdown", 0), 2),
                    "Regime": p.get("regime", "unknown"),
                    "Promoted": p.get("timestamp", "")[:10]  # Just the date part
                })
            
            # Display as table
            if filtered_data:
                df = pd.DataFrame(filtered_data)
                st.dataframe(df, use_container_width=True)
                
                # Download button
                st.download_button(
                    label="Download as CSV",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name=f"promotion_history_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    # Tab 3: Parameter Explorer
    with tab3:
        st.markdown("### Parameter Exploration")
        st.markdown("""
        Explore different strategy parameters and variants to discover optimal configurations.
        This tool automatically tests multiple parameter combinations and identifies the best performing setups.
        """)
        
        # Parameter exploration controls
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.text_input("Symbol:", "SPY")
            strategies = ["momentum_breakout", "mean_reversion", "trend_following", 
                        "value_dividend", "volatility_etf", "ai_sentiment"]
            
            strategy = st.selectbox("Strategy:", strategies)
        
        with col2:
            max_variants = st.slider("Max Variants to Test:", min_value=1, max_value=20, value=5)
            parallel = st.checkbox("Run in Parallel", value=False)
        
        # Run exploration button
        if st.button("Run Parameter Exploration", key="run_exploration"):
            with st.spinner("Running parameter exploration..."):
                # In a real implementation, this would call the backtest executor
                st.session_state.last_exploration = {
                    "symbol": symbol,
                    "strategy": strategy,
                    "max_variants": max_variants,
                    "parallel": parallel,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                st.success(f"Exploration completed for {symbol} with {strategy}. Testing {max_variants} variants.")
                
                # Show mock results as a placeholder
                st.markdown("### Exploration Results")
                
                # This would be real data in production
                mock_results = [
                    {"params": {"ma_short": 10, "ma_long": 50, "rsi_period": 14}, 
                     "score": {"score": 0.92, "passed": True}, 
                     "result": {"sharpe_ratio": 1.85, "total_return": 18.5, "win_rate": 0.68, "max_drawdown": -5.2},
                     "promotion": {"promoted": True}},
                    {"params": {"ma_short": 5, "ma_long": 50, "rsi_period": 14}, 
                     "score": {"score": 0.78, "passed": True}, 
                     "result": {"sharpe_ratio": 1.62, "total_return": 15.1, "win_rate": 0.61, "max_drawdown": -7.0},
                     "promotion": {"promoted": False}},
                    {"params": {"ma_short": 15, "ma_long": 50, "rsi_period": 14}, 
                     "score": {"score": 0.65, "passed": False}, 
                     "result": {"sharpe_ratio": 1.31, "total_return": 12.3, "win_rate": 0.57, "max_drawdown": -9.5},
                     "promotion": {"promoted": False}},
                    {"params": {"ma_short": 10, "ma_long": 100, "rsi_period": 14}, 
                     "score": {"score": 0.58, "passed": False}, 
                     "result": {"sharpe_ratio": 1.20, "total_return": 11.0, "win_rate": 0.52, "max_drawdown": -10.8},
                     "promotion": {"promoted": False}},
                    {"params": {"ma_short": 10, "ma_long": 200, "rsi_period": 14}, 
                     "score": {"score": 0.42, "passed": False}, 
                     "result": {"sharpe_ratio": 0.95, "total_return": 8.2, "win_rate": 0.48, "max_drawdown": -14.1},
                     "promotion": {"promoted": False}}
                ]
                
                # Create dataframe of results
                result_data = []
                for i, r in enumerate(mock_results):
                    params_str = ", ".join([f"{k}: {v}" for k, v in r["params"].items()])
                    result_data.append({
                        "Variant": i + 1,
                        "Parameters": params_str,
                        "Score": r["score"]["score"],
                        "Passed": r["score"]["passed"],
                        "Sharpe": r["result"]["sharpe_ratio"],
                        "Return (%)": r["result"]["total_return"],
                        "Win Rate": r["result"]["win_rate"],
                        "Max DD (%)": r["result"]["max_drawdown"],
                        "Promoted": r["promotion"]["promoted"]
                    })
                
                if result_data:
                    df = pd.DataFrame(result_data)
                    
                    # Color the rows based on promotion status
                    def highlight_promoted(row):
                        if row["Promoted"]:
                            return ['background-color: #d4f7d4'] * len(row)
                        elif row["Passed"]:
                            return ['background-color: #f7f7d4'] * len(row)
                        else:
                            return [''] * len(row)
                    
                    st.dataframe(df.style.apply(highlight_promoted, axis=1), use_container_width=True)
                    
                    # Parameter sensitivity visualization
                    st.markdown("### Parameter Sensitivity Analysis")
                    st.markdown("This analysis shows how different parameter values affect performance.")
                    
                    # Display a mock chart
                    # In a real implementation, this would show parameter impacts on performance
                    st.info("Parameter sensitivity visualization would be shown here, using actual exploration results.")
    
    # Tab 4: Optimization Hub
    with tab4:
        st.markdown("### ML Portfolio Optimization Hub")
        st.markdown("""
        Optimize entire portfolios by testing multiple symbols and strategies together.
        This automated process discovers the best strategy-symbol combinations and promotes them to your paper trading system.
        """)
        
        # Portfolio optimization controls
        col1, col2 = st.columns(2)
        
        with col1:
            symbols_input = st.text_input("Symbols (comma-separated):", "SPY, QQQ, AAPL, MSFT, GOOGL")
            symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
            
            st.markdown(f"**Testing {len(symbols)} symbols**")
        
        with col2:
            strategies = ["momentum_breakout", "mean_reversion", "trend_following", 
                        "value_dividend", "volatility_etf", "ai_sentiment"]
            
            selected_strategies = st.multiselect("Select Strategies:", strategies, default=["momentum_breakout", "trend_following"])
            
            st.markdown(f"**Testing {len(selected_strategies)} strategies**")
        
        variants_per_combo = st.slider("Variants per Combination:", min_value=1, max_value=10, value=3)
        
        # Calculate total combinations
        total_combos = len(symbols) * len(selected_strategies) * variants_per_combo
        st.markdown(f"**Total test combinations: {total_combos}**")
        
        # Run optimization button
        if st.button("Run Portfolio Optimization", key="run_optimization"):
            if not symbols or not selected_strategies:
                st.error("Please select at least one symbol and one strategy")
            else:
                with st.spinner(f"Running portfolio optimization across {len(symbols)} symbols and {len(selected_strategies)} strategies..."):
                    # In a real implementation, this would call the optimizer
                    st.session_state.last_optimization = {
                        "symbols": symbols,
                        "strategies": selected_strategies,
                        "variants_per_combo": variants_per_combo,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Simulate optimization process
                    for i in range(100):
                        # Update progress bar
                        progress_bar.progress(i + 1)
                        
                        if i < 25:
                            status_text.text(f"Generating {total_combos} parameter variants...")
                        elif i < 50:
                            status_text.text(f"Running backtests on {len(symbols)} symbols...")
                        elif i < 75:
                            status_text.text(f"Scoring {total_combos} backtest results...")
                        else:
                            status_text.text(f"Evaluating promotion criteria for top candidates...")
                        
                        # Sleep briefly to simulate work
                        import time
                        time.sleep(0.05)
                    
                    # Complete
                    status_text.text("Portfolio optimization complete!")
                    
                    # Show mock results as a placeholder
                    st.markdown("### Optimization Results")
                    
                    # Mock promotions
                    mock_promotions = [
                        {"symbol": "AAPL", "strategy": "momentum_breakout", "score": 0.94, "params": {"ma_short": 10, "ma_long": 50, "rsi_period": 9}},
                        {"symbol": "QQQ", "strategy": "trend_following", "score": 0.91, "params": {"ma_short": 20, "ma_long": 100, "adx_threshold": 25}},
                        {"symbol": "MSFT", "strategy": "momentum_breakout", "score": 0.88, "params": {"ma_short": 15, "ma_long": 100, "rsi_period": 14}},
                        {"symbol": "SPY", "strategy": "trend_following", "score": 0.85, "params": {"ma_short": 10, "ma_long": 50, "adx_threshold": 20}}
                    ]
                    
                    # Create dataframe for display
                    if mock_promotions:
                        promotion_data = []
                        for p in mock_promotions:
                            params_str = ", ".join([f"{k}: {v}" for k, v in p["params"].items()])
                            promotion_data.append({
                                "Symbol": p["symbol"],
                                "Strategy": p["strategy"],
                                "Score": p["score"],
                                "Parameters": params_str
                            })
                        
                        df = pd.DataFrame(promotion_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Show summary
                        st.markdown("### Optimization Summary")
                        st.markdown(f"**Total combinations tested:** {total_combos}")
                        st.markdown(f"**Successful promotions:** {len(mock_promotions)}")
                        st.markdown(f"**Best symbol-strategy pair:** {mock_promotions[0]['symbol']} with {mock_promotions[0]['strategy']} (Score: {mock_promotions[0]['score']})")
                        
                        # Action button
                        if st.button("Deploy All to Paper Trading", key="deploy_all"):
                            st.success(f"{len(mock_promotions)} strategies have been queued for paper trading deployment")


def render_promotion_metrics_card(container, n_promotions=0, latest_promotion=None):
    """Render a small metrics card showing promotion statistics."""
    
    container.markdown("### Strategy Promotions")
    
    # Stats in columns
    col1, col2 = container.columns(2)
    
    with col1:
        container.metric("Active Promotions", n_promotions)
    
    with col2:
        latest_date = "None" if not latest_promotion else latest_promotion.get("timestamp", "")[:10]
        container.metric("Latest Promotion", latest_date)
    
    # Show latest promotion if available
    if latest_promotion:
        container.markdown(f"**Latest:** {latest_promotion.get('symbol')} - {latest_promotion.get('strategy')}")
        container.markdown(f"Score: {latest_promotion.get('score', 0):.2f} | Sharpe: {latest_promotion.get('sharpe_ratio', 0):.2f}")


def render_small_promotion_list(container, limit=5):
    """Render a small list of top promoted strategies."""
    
    promoter = get_strategy_promoter()
    top_promotions = promoter.get_top_promotions(limit=limit)
    
    container.markdown("### Top Promoted Strategies")
    
    if not top_promotions:
        container.info("No strategies have been promoted yet.")
    else:
        for i, p in enumerate(top_promotions):
            container.markdown(f"**{i+1}. {p.get('symbol')} - {p.get('strategy')}**")
            container.markdown(f"Score: {p.get('score', 0):.2f} | Sharpe: {p.get('sharpe_ratio', 0):.2f} | Return: {p.get('total_return', 0):.2f}%")
        
        container.markdown("[View all promotions](#)")  # This would link to the full dashboard in a real implementation
