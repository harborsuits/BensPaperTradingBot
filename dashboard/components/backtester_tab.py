"""
Backtester Tab Component

This module renders the Backtester tab of the trading platform, providing interfaces for
strategy experimentation, testing, analysis, and result tracking.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import random
from plotly.subplots import make_subplots

def hex_to_rgb(hex_color):
    """Convert hex color code to RGB values for use in rgba() CSS function"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16) 
    b = int(hex_color[4:6], 16)
    return f"{r}, {g}, {b}"

# Import UI styles
from dashboard.ui_styles import (
    ThemeMode, UIColors, UIEffects, UITypography, UISpacing,
    create_card, create_metric_card, format_currency, format_percentage,
    theme_plotly_chart
)

def create_experimental_strategy_section(strategy_factory=None):
    """Creates the Experimental Strategy section showing automated strategy generation"""
    
    st.markdown("<h2>Automated Strategy Generation</h2>", unsafe_allow_html=True)
    
    # Generate sample active strategies or get real ones from the factory
    if strategy_factory and hasattr(strategy_factory, 'get_all_strategies'):
        try:
            strategies = strategy_factory.get_all_strategies()
        except Exception as e:
            st.error(f"Error retrieving strategies: {str(e)}")
            strategies = get_sample_strategies()
    else:
        strategies = get_sample_strategies()
    
    # Display active strategy pipeline
    st.markdown("<h3>Strategy Generation Pipeline</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Show strategy generation process
        pipeline_html = f"""
        <div class="card" style="margin-bottom: 16px; padding: 16px;">
            <h4>Autonomous Strategy Creation Process</h4>
            <div style="margin: 10px 0 20px 0;">
                <div style="display: flex; margin-bottom: 20px;">
                    <div style="background-color: {UIColors.Dark.ACCENT_PRIMARY}; color: white; padding: 10px; border-radius: 4px; width: 180px; text-align: center;">
                        <strong>Data Analysis</strong>
                    </div>
                    <div style="margin: 10px 8px;">➡️</div>
                    <div style="background-color: {UIColors.Dark.ACCENT_SECONDARY}; color: white; padding: 10px; border-radius: 4px; width: 180px; text-align: center;">
                        <strong>Pattern Discovery</strong>
                    </div>
                    <div style="margin: 10px 8px;">➡️</div>
                    <div style="background-color: {UIColors.Dark.SUCCESS}; color: white; padding: 10px; border-radius: 4px; width: 180px; text-align: center;">
                        <strong>Strategy Generation</strong>
                    </div>
                </div>
                <div style="display: flex;">
                    <div style="background-color: {UIColors.Dark.WARNING}; color: white; padding: 10px; border-radius: 4px; width: 180px; text-align: center;">
                        <strong>Parameter Tuning</strong>
                    </div>
                    <div style="margin: 10px 8px;">⬅️</div>
                    <div style="background-color: {UIColors.Dark.ERROR}; color: white; padding: 10px; border-radius: 4px; width: 180px; text-align: center;">
                        <strong>Verification</strong>
                    </div>
                    <div style="margin: 10px 8px;">⬅️</div>
                    <div style="background-color: {UIColors.Dark.INFO}; color: white; padding: 10px; border-radius: 4px; width: 180px; text-align: center;">
                        <strong>Initial Backtest</strong>
                    </div>
                </div>
            </div>
            
            <h5>Current Processing Statistics</h5>
            <div style="display: flex; justify-content: space-between; margin-top: 12px;">
                <div>
                    <div style="font-size: 24px; font-weight: bold;">{len(strategies)}</div>
                    <div style="font-size: 14px; color: {UIColors.Dark.TEXT_TERTIARY};">Active Strategies</div>
                </div>
                <div>
                    <div style="font-size: 24px; font-weight: bold;">24</div>
                    <div style="font-size: 14px; color: {UIColors.Dark.TEXT_TERTIARY};">Strategies in Development</div>
                </div>
                <div>
                    <div style="font-size: 24px; font-weight: bold;">142</div>
                    <div style="font-size: 14px; color: {UIColors.Dark.TEXT_TERTIARY};">Patterns Identified</div>
                </div>
                <div>
                    <div style="font-size: 24px; font-weight: bold;">3.2M</div>
                    <div style="font-size: 14px; color: {UIColors.Dark.TEXT_TERTIARY};">Parameter Combinations</div>
                </div>
            </div>
        </div>
        """
        st.markdown(pipeline_html, unsafe_allow_html=True)
        
        # Recent discovery activity feed
        st.markdown("<h5>Recent Strategy Discovery Activity</h5>", unsafe_allow_html=True)
        
        activity_items = [
            {"time": "13:42:05", "event": "New pattern discovered in AAPL daily chart", "details": "Volume-price divergence pattern with 72% historical reliability"},
            {"time": "13:38:12", "event": "Parameter optimization complete", "details": "Optimized RSI parameters for Crypto Volatility strategy"},
            {"time": "13:30:22", "event": "Strategy verification passed", "details": "Gap Trading strategy passed all validation tests"},
            {"time": "13:25:45", "event": "Correlation analysis complete", "details": "Found 3 new low-correlation pairs for arbitrage strategies"},
            {"time": "13:15:33", "event": "New strategy generated", "details": "Created new Mean Reversion strategy based on recent market patterns"}
        ]
        
        for item in activity_items:
            activity_html = f"""
            <div style="display: flex; margin-bottom: 10px; padding: 8px; background-color: {UIColors.Dark.BG_TERTIARY}; border-radius: 4px;">
                <div style="width: 80px; color: {UIColors.Dark.TEXT_TERTIARY};">{item['time']}</div>
                <div>
                    <div style="font-weight: bold;">{item['event']}</div>
                    <div style="font-size: 12px; color: {UIColors.Dark.TEXT_SECONDARY};">{item['details']}</div>
                </div>
            </div>
            """
            st.markdown(activity_html, unsafe_allow_html=True)
    
    with col2:
        # Performance metrics of strategy generation
        metrics_html = f"""
        <div class="card" style="padding: 16px; height: 100%;">
            <h4>Generation Performance</h4>
            
            <div style="margin: 16px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <strong>AI Model Load:</strong>
                    <span>78%</span>
                </div>
                <div style="height: 8px; background-color: {UIColors.Dark.BG_SECONDARY}; border-radius: 4px;">
                    <div style="width: 78%; height: 100%; background-color: {UIColors.Dark.SUCCESS}; border-radius: 4px;"></div>
                </div>
            </div>
            
            <div style="margin: 16px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <strong>Pattern Recognition:</strong>
                    <span>89%</span>
                </div>
                <div style="height: 8px; background-color: {UIColors.Dark.BG_SECONDARY}; border-radius: 4px;">
                    <div style="width: 89%; height: 100%; background-color: {UIColors.Dark.SUCCESS}; border-radius: 4px;"></div>
                </div>
            </div>
            
            <div style="margin: 16px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <strong>Backtest Queue:</strong>
                    <span>45%</span>
                </div>
                <div style="height: 8px; background-color: {UIColors.Dark.BG_SECONDARY}; border-radius: 4px;">
                    <div style="width: 45%; height: 100%; background-color: {UIColors.Dark.WARNING}; border-radius: 4px;"></div>
                </div>
            </div>
            
            <div style="margin: 16px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <strong>Strategy Quality:</strong>
                    <span>92%</span>
                </div>
                <div style="height: 8px; background-color: {UIColors.Dark.BG_SECONDARY}; border-radius: 4px;">
                    <div style="width: 92%; height: 100%; background-color: {UIColors.Dark.SUCCESS}; border-radius: 4px;"></div>
                </div>
            </div>
            
            <div style="margin-top: 24px;">
                <h5>System Status</h5>
                <div style="display: flex; align-items: center; margin-top: 8px;">
                    <span class="status-dot success"></span>
                    <span>All Systems Operational</span>
                </div>
                <div style="font-size: 12px; color: {UIColors.Dark.TEXT_TERTIARY}; margin-top: 4px;">
                    Next scheduled maintenance: 2025-05-05 02:00 UTC
                </div>
            </div>
        </div>
        """
        st.markdown(metrics_html, unsafe_allow_html=True)

def get_sample_strategies():
    """Get sample strategies for demonstration"""
    return [
        {"name": "Gap Trading", "type": "Equity", "status": "Active", "win_rate": 58.3, "sharpe": 1.35},
        {"name": "Trend Following", "type": "Equity", "status": "Active", "win_rate": 52.1, "sharpe": 1.28},
        {"name": "Mean Reversion", "type": "Equity", "status": "Testing", "win_rate": 56.7, "sharpe": 1.42},
        {"name": "Sector Rotation", "type": "Equity", "status": "Active", "win_rate": 61.2, "sharpe": 1.56},
        {"name": "Options Iron Condor", "type": "Options", "status": "Active", "win_rate": 75.4, "sharpe": 1.12},
        {"name": "Forex Trend", "type": "Forex", "status": "Testing", "win_rate": 54.8, "sharpe": 1.31},
        {"name": "Crypto Volatility", "type": "Crypto", "status": "Active", "win_rate": 49.2, "sharpe": 1.65}
    ]

def create_current_test_section(strategy_manager=None):
    """Creates the Current Test/Pending section"""
    
    st.markdown("<h2>Current Test/Pending</h2>", unsafe_allow_html=True)
    
    # Sample test data
    current_test = {
        "name": "Modified Gap Trading Strategy",
        "status": "Running",
        "progress": 68,
        "start_time": "2025-05-04 12:15:32",
        "elapsed": "01:30:15",
        "estimated_completion": "2025-05-04 14:30:00",
        "symbols": ["AAPL", "MSFT", "AMZN", "GOOGL", "META"],
        "period": "2020-01-01 to 2025-05-01",
        "initial_results": {
            "win_rate": 56.4,
            "profit_factor": 1.82,
            "sharpe": 1.45,
            "max_drawdown": 12.3
        }
    }
    
    # Create progress indicator
    progress_html = f"""
    <div class="card" style="margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <h3 style="margin: 0;">{current_test['name']}</h3>
            <span style="background-color: {UIColors.Dark.INFO}; color: white; padding: 4px 10px; border-radius: 4px; font-size: 14px;">
                {current_test['status']}
            </span>
        </div>
        
        <div style="margin-bottom: 12px;">
            <div style="height: 10px; background-color: {UIColors.Dark.BG_SECONDARY}; border-radius: 5px; margin: 8px 0;">
                <div style="width: {current_test['progress']}%; height: 100%; background-color: {UIColors.Dark.ACCENT_PRIMARY}; border-radius: 5px;"></div>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <small>{current_test['progress']}% Complete</small>
                <small>ETA: {current_test['estimated_completion']}</small>
            </div>
        </div>
        
        <div style="display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 16px;">
            <div>
                <strong>Started:</strong> {current_test['start_time']}
            </div>
            <div>
                <strong>Elapsed:</strong> {current_test['elapsed']}
            </div>
            <div>
                <strong>Test Period:</strong> {current_test['period']}
            </div>
        </div>
        
        <div style="margin-bottom: 16px;">
            <strong>Symbols:</strong> {', '.join(current_test['symbols'])}
        </div>
        
        <h4>Initial Results</h4>
        <div style="display: flex; gap: 24px; flex-wrap: wrap;">
            <div>
                <div style="font-size: 24px; font-weight: bold; color: {UIColors.Dark.PROFIT};">
                    {current_test['initial_results']['win_rate']}%
                </div>
                <div style="font-size: 12px; color: {UIColors.Dark.TEXT_TERTIARY};">
                    Win Rate
                </div>
            </div>
            <div>
                <div style="font-size: 24px; font-weight: bold; color: {UIColors.Dark.PROFIT};">
                    {current_test['initial_results']['profit_factor']}
                </div>
                <div style="font-size: 12px; color: {UIColors.Dark.TEXT_TERTIARY};">
                    Profit Factor
                </div>
            </div>
            <div>
                <div style="font-size: 24px; font-weight: bold; color: {UIColors.Dark.PROFIT};">
                    {current_test['initial_results']['sharpe']}
                </div>
                <div style="font-size: 12px; color: {UIColors.Dark.TEXT_TERTIARY};">
                    Sharpe Ratio
                </div>
            </div>
            <div>
                <div style="font-size: 24px; font-weight: bold; color: {UIColors.Dark.LOSS};">
                    {current_test['initial_results']['max_drawdown']}%
                </div>
                <div style="font-size: 12px; color: {UIColors.Dark.TEXT_TERTIARY};">
                    Max Drawdown
                </div>
            </div>
        </div>
        
        <div style="display: flex; justify-content: flex-end; gap: 12px; margin-top: 16px;">
            <button style="background-color: {UIColors.Dark.ERROR}; color: white; border: none; padding: 6px 12px; border-radius: 4px;">
                Cancel Test
            </button>
            <button style="background-color: {UIColors.Dark.WARNING}; color: white; border: none; padding: 6px 12px; border-radius: 4px;">
                Pause Test
            </button>
        </div>
    </div>
    """
    
    st.markdown(progress_html, unsafe_allow_html=True)

def get_active_runs(strategy_manager=None):
    """Get active backtest runs from the strategy manager or return sample data"""
    if strategy_manager and hasattr(strategy_manager, 'get_active_runs'):
        try:
            return strategy_manager.get_active_runs()
        except Exception as e:
            # Fall back to sample data on error
            pass
    
    # Sample backtest run data
    return [
        {
            "name": "Gap Trading Strategy", 
            "status": "Running", 
            "asset": "Equities",
            "timeframe": "Daily",
            "capital": 100000,
            "progress": 82,
            "start_time": "14:05:32"
        },
        {
            "name": "Momentum Crypto", 
            "status": "Running", 
            "asset": "Crypto",
            "timeframe": "4-Hour",
            "capital": 50000,
            "progress": 45,
            "start_time": "14:12:18"
        },
        {
            "name": "Options Iron Condor", 
            "status": "Initializing", 
            "asset": "Options",
            "timeframe": "Daily",
            "capital": 250000,
            "progress": 12,
            "start_time": "14:18:45"
        }
    ]

def create_bottom_sections(strategy_manager=None):
    """Creates the bottom three sections: Queue, Analysis, and Winning Strategies"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<h3>Queue for Backtest</h3>", unsafe_allow_html=True)
        
        # Sample queue data
        queue_items = [
            {"name": "Modified RSI Strategy", "priority": "High", "estimated_time": "02:15:00"},
            {"name": "Volume Breakout v2", "priority": "Medium", "estimated_time": "01:45:00"},
            {"name": "News Sentiment + MA", "priority": "Low", "estimated_time": "03:30:00"}
        ]
        
        for i, item in enumerate(queue_items):
            priority_color = {
                "High": UIColors.Dark.ERROR,
                "Medium": UIColors.Dark.WARNING,
                "Low": UIColors.Dark.INFO
            }.get(item["priority"])
            
            queue_html = f"""
            <div class="card" style="margin-bottom: 8px; padding: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="font-size: 14px;">{i+1}. {item['name']}</strong>
                    <span style="background-color: {priority_color}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 12px;">
                        {item['priority']}
                    </span>
                </div>
                <div style="font-size: 12px; color: {UIColors.Dark.TEXT_TERTIARY}; margin-top: 4px;">
                    Est. Runtime: {item['estimated_time']}
                </div>
            </div>
            """
            st.markdown(queue_html, unsafe_allow_html=True)
        
        queue_actions = f"""
        </div>
        """
        st.markdown(queue_html, unsafe_allow_html=True)
        
        # Display active runs
        st.markdown("<h4>Active Runs</h4>", unsafe_allow_html=True)
        
        # Get active runs
        active_runs = get_active_runs(strategy_manager)
        
        for run in active_runs:
            progress = run.get("progress", 0)
            status_color = UIColors.Dark.SUCCESS if progress > 80 else UIColors.Dark.WARNING if progress > 30 else UIColors.Dark.ERROR
            
            run_html = f"""
            <div style="padding: 12px; margin-bottom: 10px; background-color: {UIColors.Dark.BG_TERTIARY}; border-radius: 4px;">
                <div style="display: flex; justify-content: space-between;">
                    <strong>{run['name']}</strong>
                    <span style="color: {status_color};">{run['status']}</span>
                </div>
                <div style="margin: 8px 0; font-size: 12px;">
                    <span>Asset: {run['asset']}</span> • 
                    <span>Period: {run['timeframe']}</span> • 
                    <span>Capital: ${run['capital']:,}</span>
                </div>
                <div style="height: 6px; background-color: {UIColors.Dark.BG_SECONDARY}; border-radius: 4px; margin: 8px 0;">
                    <div style="width: {progress}%; height: 100%; background-color: {status_color}; border-radius: 4px;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 12px;">
                    <span>Started: {run['start_time']}</span>
                    <span>{progress}% complete</span>
                </div>
            </div>
            """
            st.markdown(run_html, unsafe_allow_html=True)
    
    with col2:
        # Show detailed performance of current top backtests
        st.markdown("<h4>Real-Time Performance Metrics</h4>", unsafe_allow_html=True)
        
        # Performance metrics tabs
        tab1, tab2 = st.tabs(["Overall Performance", "Strategy Comparison"])
        
        with tab1:
            # Show metrics for top strategy
            col_metrics = st.columns(4)
            with col_metrics[0]:
                st.metric(label="Avg Win Rate", value="58.3%", delta="+2.1%")
            with col_metrics[1]:
                st.metric(label="Avg Sharpe", value="1.42", delta="+0.12")
            with col_metrics[2]:
                st.metric(label="Avg Drawdown", value="-6.8%", delta="+0.7%", delta_color="inverse")
            with col_metrics[3]:
                st.metric(label="Avg Return", value="+26.4%", delta="+4.2%")
            
            # Chart showing equity curve of top strategy being tested
            fig = go.Figure()
            
            # Sample data for equity curve
            dates = pd.date_range(start="2023-01-01", end="2023-09-30", freq="D")
            np.random.seed(42)
            
            # Initial value
            initial_value = 100000
            
            # Generate random returns (slightly positive drift)
            daily_returns = np.random.normal(0.0005, 0.01, len(dates))
            
            # Cumulative returns
            equity_curve = initial_value * (1 + np.cumsum(daily_returns))
            
            # Add some drawdowns for realism
            equity_curve[50:70] = equity_curve[50:70] * 0.95
            equity_curve[150:170] = equity_curve[150:170] * 0.93
            equity_curve[220:250] = equity_curve[220:250] * 0.90
            
            # Create a DataFrame
            df = pd.DataFrame({
                'date': dates,
                'equity': equity_curve
            })
            
            # Plot the equity curve
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['equity'],
                mode='lines',
                name='Gap Trading Strategy',
                line=dict(color=UIColors.Dark.ACCENT_PRIMARY, width=2),
                fill='tozeroy',
                fillcolor=f'rgba({hex_to_rgb(UIColors.Dark.ACCENT_PRIMARY)}, 0.1)'
            ))
            
            # Plot the benchmark (S&P 500)
            benchmark = initial_value * (1 + np.cumsum(np.random.normal(0.0003, 0.008, len(dates))))
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=benchmark,
                mode='lines',
                name='Benchmark (S&P 500)',
                line=dict(color=UIColors.Dark.NEUTRAL, width=1.5, dash='dash')
            ))
            
            # Add second strategy with different color for comparison
            strategy2 = initial_value * (1 + np.cumsum(np.random.normal(0.0006, 0.012, len(dates))))
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=strategy2,
                mode='lines',
                name='Mean Reversion Strategy',
                line=dict(color=UIColors.Dark.ACCENT_SECONDARY, width=2)
            ))
            
            theme_plotly_chart(fig)
            
            # Performance metrics details
            metrics_html = f"""
            <div style="margin-top: 16px; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px;">
                <div style="background-color: {UIColors.Dark.BG_TERTIARY}; padding: 10px; border-radius: 4px;">
                    <div style="font-size: 14px; font-weight: bold;">Monthly Returns</div>
                    <div style="font-size: 20px; margin-top: 4px;">+2.4%</div>
                    <div style="font-size: 12px; color: {UIColors.Dark.TEXT_TERTIARY};">Best: +6.8% / Worst: -3.2%</div>
        </div>
        """
        st.markdown(analysis_html, unsafe_allow_html=True)
    
    with col3:
        st.markdown("<h3>Winning Strategies</h3>", unsafe_allow_html=True)
        
        # Sample winning strategies
        winning_strategies = [
            {
                "name": "MACD Crossover", 
                "win_rate": 68.5, 
                "profit_factor": 2.34,
                "description": "MACD crossover with volume confirmation for trend following."
            },
            {
                "name": "Bollinger Breakout", 
                "win_rate": 62.1, 
                "profit_factor": 1.95,
                "description": "Trades breakouts from Bollinger Bands with momentum confirmation."
            },
            {
                "name": "RSI Reversal", 
                "win_rate": 59.8, 
                "profit_factor": 1.78,
                "description": "Trades reversals at extreme RSI levels with price action confirmation."
            }
        ]
        
        for strategy in winning_strategies:
            strategy_html = f"""
            <div class="card" style="margin-bottom: 10px; background-color: {UIColors.Dark.BG_TERTIARY};">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <h4 style="margin: 0;">{strategy['name']}</h4>
                    <div style="display: flex; flex-direction: column; align-items: flex-end;">
                        <span style="font-size: 16px; font-weight: bold; color: {UIColors.Dark.PROFIT};">
                            {strategy['win_rate']}% WR
                        </span>
                        <span style="font-size: 13px; color: {UIColors.Dark.PROFIT};">
                            {strategy['profit_factor']} PF
                        </span>
                    </div>
                </div>
                <p style="margin: 8px 0 4px 0; font-size: 13px;">{strategy['description']}</p>
                <div style="display: flex; gap: 6px; margin-top: 8px;">
                    <button style="flex: 1; background-color: {UIColors.Dark.ACCENT_PRIMARY}; color: white; border: none; padding: 4px 8px; border-radius: 4px; font-size: 12px;">
                        View Details
                    </button>
                    <button style="flex: 1; background-color: {UIColors.Dark.SUCCESS}; color: white; border: none; padding: 4px 8px; border-radius: 4px; font-size: 12px;">
                        Deploy Live
                    </button>
                </div>
            </div>
            """
            st.markdown(strategy_html, unsafe_allow_html=True)

def render_backtester_tab(strategy_manager=None, strategy_factory=None):
    """Renders the complete Backtester tab"""
    
    st.title("Strategy Backtester")
    
    # Experimental Strategy section
    create_experimental_strategy_section(strategy_factory)
    
    # Current Test/Pending section
    create_current_test_section(strategy_manager)
    
    # Bottom three sections: Queue, Analysis, and Winning Strategies
    create_bottom_sections(strategy_manager)
